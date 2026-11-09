"""Public-report assimilation and reciprocal RadioMap interference forecasts."""
from collections import deque
from dataclasses import asdict, is_dataclass
import hashlib
import time

import numpy as np

from ..continual_memory import ContinualResponseMemory, DifferentialPair
from ..closed_loop.runtime import SinglePredictor
from ..frontier.fast import FastMap
from ..native.world import HORIZONS
from .probes import managed_overlap


class ResponsePredictor:
    """Receives public geometry/reports, never a World or private probe runtime.

    The prior uses completed downlink RSSI under an explicit reciprocal-channel
    assumption. Local assimilation supplies no calibrated packet-risk bound.
    """
    def __init__(self, geometry, spec, mailbox, estimator='none', *, receiver_domains,
                 max_age_seconds=30.):
        if estimator not in ('none', 'prior', 'ignore', 'memory', 'bayesian'):
            raise ValueError('Unknown public response estimator')
        self.geometry, self.spec, self.mailbox = geometry, spec, mailbox
        domains = np.asarray(receiver_domains)
        if (domains.shape != (len(geometry.sites),) or domains.dtype.kind not in 'iu'
                or (domains < 0).any() or len(geometry.aps) != len(geometry.sites)):
            raise ValueError('Provide public nonnegative integer contention-domain IDs per receiver')
        self.receiver_domains = domains.copy()
        self.receiver_domains.setflags(write=False)
        self.receiver_channels = np.array([a.channel for a in geometry.aps])
        self.receiver_channels.setflags(write=False)
        self.base = SinglePredictor(geometry, 'radiomap')
        self.estimator = estimator
        self.max_age = float(max_age_seconds)
        if not np.isfinite(self.max_age) or self.max_age <= 0:
            raise ValueError('Use a finite positive evidence lifetime')
        self.xscale, self.yscale = float(np.max(spec.power_mw)), 1e-6
        self.memories = [ContinualResponseMemory(2, max_states=1,
                            max_pairs_per_state=24, max_age_seconds=self.max_age,
                            max_delivery_seconds=5., exposure_scale_mw=self.xscale,
                            response_scale_mw=self.yscale)
                         for _ in geometry.sites]
        self.rows = [deque(maxlen=12) for _ in geometry.sites]
        self.seen = deque(maxlen=256)
        self.records = []
        self.channel_mask = self.receiver_channels == spec.channel

    @staticmethod
    def report_values(report):
        return asdict(report) if is_dataclass(report) else dict(report)

    def consume(self, slot):
        received = self.mailbox.drain(slot)
        used = 0
        for report in received:
            p = self.report_values(report)
            identity = p['report_id']
            if identity in self.seen:
                continue
            self.seen.append(identity)
            measured_slots = tuple(p['measured_slots'])
            complete, delivered = int(p['completed_slot']), int(p['delivered_slot'])
            if (len(measured_slots) != 3 or tuple(sorted(measured_slots)) != measured_slots
                    or not 0 <= min(measured_slots) <= max(measured_slots) <= complete <= delivered <= slot
                    or p['channel'] != self.spec.channel
                    or tuple(p['transmitter_ids']) != ('managed-0', 'managed-1')):
                raise ValueError('Noncausal or mismatched probe report')
            receivers = tuple(p['receiver_sites'])
            if (len(set(receivers)) != len(receivers)
                    or any(type(s) is not int or not 0 <= s < len(self.rows) for s in receivers)):
                raise ValueError('Invalid public receiver identities')
            powers = np.asarray(p['received_mw'], dtype=float)
            exposures = np.asarray(p['exposure_mw'], dtype=float)
            valid = np.asarray(p['observed'])
            sigma = np.broadcast_to(np.asarray(p['noise_std_mw'], dtype=float), (len(receivers),))
            if (powers.shape != (3, len(receivers)) or exposures.shape != (3, 2)
                    or valid.shape != powers.shape or valid.dtype != bool
                    or not np.isfinite(exposures).all() or not np.isfinite(powers[valid]).all()
                    or not np.isfinite(sigma).all() or (sigma <= 0).any()):
                raise ValueError('Invalid public power/exposure report')
            if self.estimator not in ('memory', 'bayesian'):
                continue
            for column, receiver in enumerate(receivers):
                good = np.flatnonzero(valid[1:, column] & valid[0, column]) + 1
                if not len(good):
                    continue
                x = (exposures[good] - exposures[0]) / self.xscale
                y = (powers[good, column] - powers[0, column]) / self.yscale
                measured = min(measured_slots) / 100.
                if slot / 100. - measured > self.max_age:
                    continue
                for capture in good:
                    self.memories[receiver].ingest(('ap-' + str(receiver), 'managed-5g', 0),
                        DifferentialPair(f'{identity}-{int(capture)}',
                            exposures[capture] - exposures[0],
                            float(powers[capture, column] - powers[0, column]),
                            measured, delivered / 100., True,
                            'Registered managed-source capture ' + str(identity)), now=slot / 100.)
                self.rows[receiver].append((measured, x.copy(), y.copy(),
                                           float(sigma[column] / self.yscale)))
                used += len(good)
        return len(received), used

    def _bayesian(self, receiver, prior, prior_sd_db, now):
        """Gaussian linear update; paired differences share baseline noise."""
        mean = prior * self.xscale / self.yscale
        relative = np.clip(prior_sd_db * np.log(10.) / 10., .25, 3.)
        variance = np.maximum((mean * relative) ** 2, 1e-16)
        precision = np.diag(1 / variance)
        rhs = precision @ mean
        for measured, x, y, sigma in self.rows[receiver]:
            if now - measured > self.max_age:
                continue
            covariance = sigma ** 2 * (np.eye(len(y)) + np.ones((len(y), len(y))))
            precision += x.T @ np.linalg.solve(covariance, x)
            rhs += x.T @ np.linalg.solve(covariance, y)
        result = np.linalg.solve(precision, rhs) * self.yscale / self.xscale
        return np.maximum(result, 0.)

    def step(self, past, plan):
        mean, sd, kind, base_elapsed = self.base.step(past, plan)
        decision = len(past.value) - 60
        slot = decision * 100
        started = time.perf_counter()
        reports, observations = self.consume(slot)
        if self.estimator == 'none':
            self.records.append({'decision': decision, 'reports': reports,
                                 'used_observations': 0, 'local_compute_seconds': time.perf_counter()-started})
            return mean, sd, kind, base_elapsed
        positions = np.asarray(self.spec.positions_m, dtype=float)
        prior_rssi, prior_sd = FastMap(self.base.rmap).predict(positions[:, 0], positions[:, 1])
        prior = 10. ** ((prior_rssi[:, :, 1].T - self.geometry.tx[1]) / 10.)
        prior_sigma = prior_sd[:, :, 1].T
        output = mean.copy()
        identified = fallback = 0
        coefficients = prior.copy()
        for receiver in np.flatnonzero(self.channel_mask):
            if self.estimator == 'bayesian':
                coefficients[receiver] = self._bayesian(receiver, prior[receiver], prior_sigma[receiver], decision)
            elif self.estimator == 'memory':
                # CS/RF splitting requires individual source gains. An identified
                # synchronized sum does not identify which transmitter is sensed.
                for source, power in enumerate(self.spec.power_mw):
                    basis = np.zeros(2); basis[source] = power
                    estimate = self.memories[receiver].query(
                        ('ap-' + str(receiver), 'managed-5g', 0), basis,
                        now=decision, query_observed_at=decision, controlled_only=True)
                    if estimate.identified and estimate.predicted_delta_mw is not None:
                        coefficients[receiver, source] = max(0., estimate.predicted_delta_mw / power)
                        identified += 1
                    else:
                        fallback += 1
        predicted = []; predicted_busy = []
        # Only public activity and estimated/prior gains determine the split.
        # Carrier-sensed airtime is modeled by the kernel, not charged again as RF.
        for horizon, h in enumerate(HORIZONS):
            rf, busy = managed_overlap(coefficients, self.spec.power_mw,
                self.spec.activity(slot + int(h * 100)), self.receiver_domains,
                self.receiver_channels, self.spec.channel)
            if not np.isfinite(rf).all() or (rf < 0).any():
                raise ValueError('Invalid estimated residual interference power')
            penalty = 10 * np.log10(1 + rf / 10. ** (self.geometry.noise[1] / 10.))
            output[:, horizon, :, 1] -= penalty
            predicted.append(rf); predicted_busy.append(busy)
        elapsed = time.perf_counter() - started
        self.records.append({'decision': decision, 'reports': reports,
            'used_observations': observations, 'identified_queries': identified,
            'fallback_queries': fallback, 'local_compute_seconds': elapsed,
            'prediction_sha256': hashlib.sha256(np.asarray(predicted, dtype='f8').tobytes()).hexdigest(),
            'estimated_busy_sha256': hashlib.sha256(np.asarray(predicted_busy, dtype='f8').tobytes()).hexdigest(),
            'memory_query_unit': 'individual_source_gain',
            'uncertainty_calibrated': False})
        return output, sd, kind, base_elapsed + elapsed
