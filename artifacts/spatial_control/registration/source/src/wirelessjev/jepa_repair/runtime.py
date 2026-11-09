"""New completion-safe queue and exact outage admission; predecessors frozen."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from types import MappingProxyType
import numpy as np

from ..closed_loop.control import OutageUtility
from ..closed_loop.runtime import LiveForecasts
from ..fleetwm.data import Observations
from ..native.world import HORIZONS

KINDS = ('cold', 'warm', 'refit')
MAX_SLOT = int(np.iinfo(np.int64).max)


def _seconds(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and nonnegative')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be finite and nonnegative') from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _index(value, name, bound):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < bound:
        raise ValueError(f'{name} must be an integer within registered support')
    return int(value)


def _consume(queue, slot):
    if (isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer))
            or not 0 <= slot <= MAX_SLOT or slot < queue.last_take_slot):
        raise ValueError('Forecast consumption requires nonnegative integer slots in time order')
    slot = int(slot)
    index = int(np.searchsorted(queue.ready, slot, side='right')) - 1
    if index <= queue.last:
        queue.last_take_slot = slot
        return None
    if (not queue.last < index < len(queue.records) or slot < queue.ready[index]
            or queue.records[index]['decision'] != index
            or queue.records[index]['ready_slot'] != queue.ready[index]):
        raise RuntimeError('Forecast consumption precedes verified completion')
    age = slot / 100 - index
    record = {'slot': slot, 'forecast_decision': index, 'ready_slot': queue.ready[index], 'age_seconds': age}
    queue.consumption_records.append(record)
    queue.last_take_slot, queue.last, queue.age = slot, index, age
    return index


class CompletionSafeForecasts(LiveForecasts):
    """Do not release a forecast before its measured issue path completes.

    Charge max(sealed allowance, predictor-reported compute, measured history
    snapshot through output storage/hash time), then add the shared delay.
    Profile exceedances remain visible diagnostics; undercharged availability
    is a qualification failure. Work below the sealed allowance retains
    deterministic release slots despite different wall times.

    This is a synchronous simulation queue, not a hard real-time guarantee.
    A deployed asynchronous worker must release from its completion callback.
    """

    def __init__(self, predictor, history, duration, profile, extra):
        if not isinstance(profile, dict) or set(profile) != set(KINDS):
            raise ValueError('Require cold, warm and refit timing profiles')
        profile = {kind: _seconds(profile[kind], kind + ' profile') for kind in KINDS}
        extra = _seconds(extra, 'Shared extra delay')
        if (isinstance(duration, (bool, np.bool_)) or not isinstance(duration, (int, np.integer))
                or duration < 1):
            raise ValueError('Duration must be a positive integer')
        if (isinstance(history.n, (bool, np.bool_)) or not isinstance(history.n, (int, np.integer))
                or history.n < 1):
            raise ValueError('History must contain a positive integer fleet size')
        sites = np.asarray(predictor.geometry.sites)
        if sites.ndim != 2 or sites.shape[1] != 2 or not len(sites) or not np.isfinite(sites).all():
            raise ValueError('Require finite public site geometry')
        super().__init__(predictor, history, int(duration), profile, extra)
        self.failed = False
        self.offline_only = False
        self.accounting_mode = 'actual_cost'
        self.last_take_slot = -1
        self.consumption_records = []

    def issue(self, decision, plan):
        if self.failed:
            raise RuntimeError('Forecast queue failed; construct a fresh runtime')
        decision = _index(decision, 'Decision', len(self.mean))
        if decision != len(self.records):
            raise ValueError('Forecast jobs must be issued in strict time order')
        started = time.perf_counter()
        try:
            plan = np.asarray(plan)
            if plan.shape != (self.history.n, len(HORIZONS), 2) or not np.isfinite(plan).all():
                raise ValueError('Require a finite public N×horizon×XY plan')
            past = self.history.before(decision)
            if not isinstance(past, Observations) or len(past.value) != 60 + decision:
                raise ValueError('Use the strictly past observable history prefix')
            expected_history = (60 + decision, self.history.n, len(self.predictor.geometry.sites) * 3)
            if (any(np.shape(getattr(past, field)) != expected_history for field in ('value', 'heard', 'bound'))
                    or np.shape(past.positions) != expected_history[:2] + (2,)
                    or np.shape(past.scanned) != expected_history[:2]
                    or np.asarray(past.heard).dtype != np.bool_ or np.asarray(past.scanned).dtype != np.bool_):
                raise ValueError('Observable history axes/masks must match the public fleet and sites')
            mean, std, kind, elapsed = self.predictor.step(past, plan)
            elapsed = _seconds(elapsed, 'Actual predictor time')
            if kind not in self.profile:
                raise ValueError('Predictor returned an unknown timing regime')
            mean, std = np.asarray(mean), np.asarray(std)
            expected = self.mean.shape[1:]
            if (mean.shape != expected or std.shape != expected or mean.dtype.kind not in 'fi'
                    or std.dtype.kind not in 'fi' or not np.isfinite(mean).all()
                    or not np.isfinite(std).all() or (std < 0).any()):
                raise ValueError('Predictor returned a malformed finite mean/std forecast')
            mean, std = mean.astype('f4', copy=False), std.astype('f4', copy=False)
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError('Forecast cannot be represented as finite float32')
            self.mean[decision], self.std[decision] = mean, std
            forecast_sha = hashlib.sha256(mean.tobytes()).hexdigest()
            std_sha = hashlib.sha256(std.tobytes()).hexdigest()
            heard_entries = int(past.heard[-1].sum())
            scans_last_bin = int(past.scanned[-1].sum())
            measured_issue = _seconds(time.perf_counter() - started, 'Actual issue time')
            actual_compute = max(elapsed, measured_issue)
            charged_compute = max(self.profile[kind], actual_compute)
            service = _seconds(charged_compute + self.extra, 'Charged service time')
            ready = max(float(decision), self.previous_ready) + service
            if not math.isfinite(ready) or not math.isfinite(ready * 100):
                raise ValueError('Forecast ready time exceeds finite slot support')
            ready_slot = math.ceil(ready * 100 - 1e-10)
            if ready_slot > MAX_SLOT:
                raise ValueError('Forecast ready time exceeds integer slot support')
            record = {'decision': decision, 'history_cutoff_exclusive': 60 + decision,
                'ready_slot': ready_slot, 'kind': kind,
                'profile_seconds': self.profile[kind], 'charged_seconds': service,
                'charged_compute_seconds': charged_compute, 'shared_extra_seconds': self.extra,
                'actual_prediction_seconds': elapsed, 'actual_issue_seconds': measured_issue,
                'actual_compute_seconds': actual_compute,
                'predictor_runtime_overrun': elapsed > self.profile[kind],
                'runtime_overrun': actual_compute > self.profile[kind],
                'completion_uses_actual_elapsed': True,
                'accounting_mode': self.accounting_mode, 'offline_recorded_schedule': False,
                'forecast_shape': list(mean.shape), 'forecast_dtype': str(mean.dtype),
                'forecast_sha256': forecast_sha, 'forecast_std_sha256': std_sha,
                'heard_entries': heard_entries, 'scans_last_bin': scans_last_bin}
        except Exception:
            self.failed = True
            raise
        self.previous_ready = ready
        self.ready.append(ready_slot)
        self.records.append(record)
        return mean, std

    def take(self, slot):
        return _consume(self, slot)


class RecordedScheduleReplay(LiveForecasts):
    """Offline reproducibility under a previously measured completion ledger.

    Predictions are recomputed using CompletionSafeForecasts. Availability
    deliberately follows the verified ORIGINAL ledger. Fresh compute is
    logged separately and never claimed covered by that original schedule.
    A fresh live-cost replication is a different experiment.
    """

    def __init__(self, predictor, history, duration, profile, extra, *, recorded_jobs):
        delegate = CompletionSafeForecasts(predictor, history, duration, profile, extra)
        if not isinstance(recorded_jobs, (list, tuple)) or len(recorded_jobs) != duration:
            raise ValueError('Recorded schedule requires the complete registered decision membership')
        previous = 0.
        verified, finishes = [], []
        expected_shape = tuple(delegate.mean.shape[1:])
        for decision, source in enumerate(recorded_jobs):
            if not isinstance(source, dict):
                raise ValueError('Recorded schedule jobs must be dictionaries')
            job = copy.deepcopy(source)
            try:
                if (_index(job['decision'], 'Recorded decision', duration) != decision
                        or isinstance(job['history_cutoff_exclusive'], (bool, np.bool_))
                        or not isinstance(job['history_cutoff_exclusive'], (int, np.integer))
                        or job['history_cutoff_exclusive'] != 60 + decision):
                    raise ValueError('Recorded schedule decision/cutoff membership differs')
                kind = job['kind']
                if kind not in delegate.profile:
                    raise ValueError('Recorded schedule timing regime differs')
                prediction = _seconds(job['actual_prediction_seconds'], 'Recorded predictor time')
                issue = _seconds(job['actual_issue_seconds'], 'Recorded issue time')
                actual = _seconds(job['actual_compute_seconds'], 'Recorded actual compute')
                charged = _seconds(job['charged_compute_seconds'], 'Recorded charged compute')
                service = _seconds(job['charged_seconds'], 'Recorded service')
                equal = lambda a, b: math.isclose(_seconds(a, 'Recorded timing'), b, abs_tol=1e-12, rel_tol=0)
                if (not equal(job['profile_seconds'], delegate.profile[kind])
                        or not equal(job['shared_extra_seconds'], delegate.extra)
                        or not equal(actual, max(prediction, issue))
                        or not equal(charged, max(delegate.profile[kind], actual))
                        or not equal(service, charged + delegate.extra)
                        or job['predictor_runtime_overrun'] is not (prediction > delegate.profile[kind])
                        or job['runtime_overrun'] is not (actual > delegate.profile[kind])
                        or job['completion_uses_actual_elapsed'] is not True):
                    raise ValueError('Recorded schedule was not charged from its measured completion')
                previous = max(float(decision), previous) + service
                if not math.isfinite(previous * 100):
                    raise ValueError('Recorded schedule exceeds finite slot support')
                ready = math.ceil(previous * 100 - 1e-10)
                if (_index(job['ready_slot'], 'Recorded ready slot', MAX_SLOT + 1) != ready):
                    raise ValueError('Recorded schedule violates FIFO completion')
                shape = job['forecast_shape']
                if (not isinstance(shape, (list, tuple)) or tuple(shape) != expected_shape
                        or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                               for value in shape) or job['forecast_dtype'] != 'float32'):
                    raise ValueError('Recorded schedule forecast shapes differ from public geometry')
                for field in ('forecast_sha256', 'forecast_std_sha256'):
                    value = job[field]
                    if (not isinstance(value, str) or len(value) != 64
                            or any(character not in '0123456789abcdef' for character in value)):
                        raise ValueError('Recorded schedule requires complete forecast hashes')
                _index(job['heard_entries'], 'Recorded heard count', history.n * len(predictor.geometry.sites) * 3 + 1)
                _index(job['scans_last_bin'], 'Recorded scan count', history.n + 1)
            except (KeyError, TypeError, OverflowError) as error:
                raise ValueError('Malformed recorded schedule ledger') from error
            job['forecast_shape'] = tuple(shape)
            verified.append(MappingProxyType(job)); finishes.append(previous)
        self.delegate = delegate
        self.predictor, self.history = predictor, history
        self.profile, self.extra = delegate.profile, delegate.extra
        self.mean, self.std = delegate.mean, delegate.std
        self.recorded_jobs, self.recorded_ready = tuple(verified), tuple(job['ready_slot'] for job in verified)
        self._recorded_finishes = tuple(finishes)
        self.ready, self.records, self.consumption_records = [], [], []
        self.last, self.age, self.previous_ready, self.last_take_slot = -1, 0., 0., -1
        self.failed, self.offline_only = False, True
        self.accounting_mode = 'recorded_schedule_offline'

    def issue(self, decision, plan):
        if self.failed:
            raise RuntimeError('Recorded replay failed; construct a fresh runtime')
        decision = _index(decision, 'Replay decision', len(self.recorded_jobs))
        if decision != len(self.records):
            raise ValueError('Recorded replay jobs must be issued in strict time order')
        try:
            mean, std = self.delegate.issue(decision, plan)
            fresh = self.delegate.records[-1]
            original = self.recorded_jobs[decision]
            if fresh['kind'] != original['kind']:
                raise ValueError('Fresh replay changed the recorded timing regime')
            record = dict(original)
            record.update(accounting_mode=self.accounting_mode, offline_recorded_schedule=True,
                completion_uses_actual_elapsed=False, recorded_completion_used_actual_elapsed=True,
                recorded_forecast_sha256=original['forecast_sha256'],
                recorded_forecast_std_sha256=original['forecast_std_sha256'],
                forecast_sha256=fresh['forecast_sha256'], forecast_std_sha256=fresh['forecast_std_sha256'],
                fresh_actual_prediction_seconds=fresh['actual_prediction_seconds'],
                fresh_actual_issue_seconds=fresh['actual_issue_seconds'],
                fresh_actual_compute_seconds=fresh['actual_compute_seconds'],
                fresh_runtime_overrun=fresh['runtime_overrun'], fresh_ready_slot=fresh['ready_slot'],
                exceeds_original_recorded_compute=fresh['actual_compute_seconds'] > original['charged_compute_seconds'],
                fresh_exceeds_original_actual_compute=fresh['actual_compute_seconds'] > original['actual_compute_seconds'],
                matches_recorded_forecast=(fresh['forecast_sha256'] == original['forecast_sha256']
                    and fresh['forecast_std_sha256'] == original['forecast_std_sha256']))
        except Exception:
            self.failed = True
            raise
        self.ready.append(original['ready_slot'])
        self.records.append(record)
        self.previous_ready = self._recorded_finishes[decision]
        return mean, std

    def take(self, slot):
        return _consume(self, slot)


class ExactOutageAdmission(OutageUtility):
    """Use exact fractional outage integration for either handover cost.

    Wire as ``roam_gate``, with ``handover_admission=None``. The unchanged
    native kernel retains its RSSI recovery rule after a planned-move veto.
    ``scores`` ranks candidates with the same surrogate loss objective.
    """

    def __init__(self, forecasts, queue, cost, horizon=30.):
        cost, horizon = _seconds(cost, 'Outage cost'), _seconds(horizon, 'Outage horizon')
        if not cost <= horizon <= HORIZONS[-1]:
            raise ValueError('Outage/horizon exceeds forecast support')
        forecasts = np.asarray(forecasts)
        if (forecasts.ndim != 5 or not all(forecasts.shape) or forecasts.shape[2] != len(HORIZONS)
                or forecasts.shape[-1] != 3 or forecasts.dtype.kind not in 'fi'
                or not np.isfinite(forecasts).all()):
            raise ValueError('Require finite decision×robot×horizon×AP×band forecasts')
        if not callable(getattr(queue, 'advance', None)):
            raise TypeError('Require an age-aware forecast queue')
        super().__init__(forecasts, queue, cost, horizon)

    def loss(self, forecast):
        forecast = np.asarray(forecast)
        if (forecast.shape != self.forecasts.shape[1:] or forecast.dtype.kind not in 'fi'
                or not np.isfinite(forecast).all()):
            raise ValueError('Require a finite complete robot×horizon×AP×band forecast')
        return super().loss(forecast)

    def _sites(self, sites):
        sites = np.asarray(sites)
        if (sites.shape != (self.forecasts.shape[1],) or sites.dtype.kind not in 'iu'
                or (sites < 0).any() or (sites >= self.forecasts.shape[3]).any()):
            raise ValueError('AP indices must match the forecast robot/AP axes')
        return sites

    def scores(self, forecast, current):
        return super().scores(forecast, self._sites(current))

    def __call__(self, decision):
        _index(decision.decision, 'Admission decision', len(self.forecasts))
        self._sites(decision.current); self._sites(decision.candidate)
        eligible = np.asarray(decision.eligible)
        if eligible.dtype != np.bool_ or eligible.shape != (self.forecasts.shape[1],):
            raise ValueError('Admission requires one boolean eligibility per robot')
        return super().__call__(decision)
