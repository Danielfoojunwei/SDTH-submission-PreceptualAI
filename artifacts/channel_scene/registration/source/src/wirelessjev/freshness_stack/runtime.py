"""Actual-cost FIFO or no-wait forecast admission; old studies stay frozen.

Busy ``latest`` arrivals are refused before history copying or model work.
This synchronous slot-costed experiment is not a deployed real-time worker.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from types import MappingProxyType
import numpy as np

from ..fleetwm.data import Observations
from ..jepa_repair.runtime import CompletionSafeForecasts, MAX_SLOT, _index, _seconds
from ..native.world import HORIZONS

MODES = ('fifo', 'latest')


def _clock(slot, last=-1):
    if (isinstance(slot, (bool, np.bool_)) or not isinstance(slot, (int, np.integer))
            or not 0 <= slot <= MAX_SLOT or slot < last):
        raise ValueError('Use nonnegative integer slots in time order')
    return int(slot)


def _ready_slot(seconds):
    if not math.isfinite(seconds * 100):
        raise ValueError('Forecast ready time exceeds finite slot support')
    slot = math.ceil(seconds * 100 - 1e-10)
    if not 0 <= slot <= MAX_SLOT:
        raise ValueError('Forecast ready time exceeds integer slot support')
    return slot


def _arrival(decision, accepted, index, previous, mode):
    return {'decision': decision, 'arrival_slot': decision * 100,
        'accepted': accepted, 'job_index': index if accepted else None,
        'reason': 'accepted' if accepted else 'busy_previous_job',
        'previous_ready_seconds': previous, 'previous_ready_slot': _ready_slot(previous),
        'mode': mode}


class FreshForecasts(CompletionSafeForecasts):
    """Storage/consumption use source timestamps, independent of job ordinal."""

    def __init__(self, predictor, history, duration, profile, extra, *, mode='fifo'):
        if mode not in MODES:
            raise ValueError('Unknown forecast admission mode')
        super().__init__(predictor, history, duration, profile, extra)
        self.mode, self.arrivals, self.last_job = mode, [], -1

    def _measure_job(self, decision, plan, *, previous, job_index):
        """Measure a forward; publish ready records only after all checks."""
        started = time.perf_counter()
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
                or np.asarray(past.heard).dtype != np.bool_
                or np.asarray(past.scanned).dtype != np.bool_):
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
        mean_hash = hashlib.sha256(mean.tobytes()).hexdigest()
        std_hash = hashlib.sha256(std.tobytes()).hexdigest()
        heard_entries, scans_last_bin = int(past.heard[-1].sum()), int(past.scanned[-1].sum())
        issue = _seconds(time.perf_counter() - started, 'Actual issue time')
        actual = max(elapsed, issue)
        compute = max(self.profile[kind], actual)
        service = _seconds(compute + self.extra, 'Charged service time')
        start = max(float(decision), previous)
        finish = _seconds(start + service, 'Forecast completion')
        job = {'decision': decision, 'source_decision': decision, 'job_index': job_index,
            'issue_seconds': float(decision), 'start_seconds': start, 'mode': self.mode,
            'history_cutoff_exclusive': 60 + decision, 'ready_slot': _ready_slot(finish), 'kind': kind,
            'profile_seconds': self.profile[kind], 'charged_seconds': service,
            'charged_compute_seconds': compute, 'shared_extra_seconds': self.extra,
            'actual_prediction_seconds': elapsed, 'actual_issue_seconds': issue,
            'actual_compute_seconds': actual,
            'predictor_runtime_overrun': elapsed > self.profile[kind],
            'runtime_overrun': actual > self.profile[kind],
            'completion_uses_actual_elapsed': True,
            'accounting_mode': 'actual_cost', 'offline_recorded_schedule': False,
            'forecast_shape': list(mean.shape), 'forecast_dtype': str(mean.dtype),
            'forecast_sha256': mean_hash, 'forecast_std_sha256': std_hash,
            'heard_entries': heard_entries, 'scans_last_bin': scans_last_bin}
        return mean, std, job, finish

    def issue(self, decision, plan):
        if self.failed:
            raise RuntimeError('Forecast queue failed; construct a fresh runtime')
        decision = _index(decision, 'Decision', len(self.mean))
        if decision != len(self.arrivals):
            raise ValueError('Forecast arrivals must be issued in strict time order')
        accepted = self.mode == 'fifo' or float(decision) >= self.previous_ready
        arrival = _arrival(decision, accepted, len(self.records), self.previous_ready, self.mode)
        if not accepted:
            self.arrivals.append(arrival)
            return None
        try:
            mean, std, job, finish = self._measure_job(decision, plan,
                previous=self.previous_ready, job_index=len(self.records))
        except Exception:
            self.failed = True
            raise
        self.previous_ready = finish
        self.ready.append(job['ready_slot'])
        self.records.append(job)
        self.arrivals.append(arrival)
        return mean, std

    def take(self, slot):
        slot = _clock(slot, self.last_take_slot)
        index = int(np.searchsorted(self.ready, slot, side='right')) - 1
        if index <= self.last_job:
            self.last_take_slot = slot
            return None
        if not self.last_job < index < len(self.records):
            raise RuntimeError('Forecast consumption lacks a completed accepted job')
        job = self.records[index]
        source = job['decision']
        if (slot < self.ready[index] or job['job_index'] != index
                or job['source_decision'] != source or self.ready[index] != job['ready_slot']):
            raise RuntimeError('Forecast consumption precedes verified completion')
        age = slot / 100 - source
        self.consumption_records.append({'slot': slot, 'forecast_decision': source,
            'ready_slot': job['ready_slot'], 'age_seconds': age})
        self.last_take_slot, self.last_job, self.last, self.age = slot, index, source, age
        return source

    def current_age(self, slot):
        slot = _clock(slot, self.last_take_slot)
        return None if self.last == -1 else slot / 100 - self.last

    def forecast_for_slot(self, slot):
        """Read current age without consuming or changing the completion ledger."""
        age = self.current_age(slot)
        if age is None:
            return None
        # Retain the native clipped horizon interpolation, not extrapolation.
        from ..frontier.timing import ForecastQueue
        view = ForecastQueue([0.])
        view.age = age
        return view.advance(self.mean[self.last]), view.advance(self.std[self.last]), self.last, age


def _verified_schedule(queue, jobs, arrivals):
    """Check the entire original paid/admission schedule before any forward."""
    duration = len(queue.mean)
    if (not isinstance(jobs, (tuple, list)) or not isinstance(arrivals, (tuple, list))
            or len(arrivals) != duration or not 1 <= len(jobs) <= duration):
        raise ValueError('Recorded schedule requires complete arrivals and accepted jobs')
    previous, ordinal, verified_jobs, verified_arrivals, finishes = 0., 0, [], [], []
    for decision in range(duration):
        accepted = queue.mode == 'fifo' or float(decision) >= previous
        expected = _arrival(decision, accepted, ordinal, previous, queue.mode)
        original_arrival = arrivals[decision]
        if (not isinstance(original_arrival, dict) or set(original_arrival) != set(expected)
                or any(type(original_arrival[key]) is not type(value) or original_arrival[key] != value
                       for key, value in expected.items())):
            raise ValueError('Recorded arrival/admission schedule differs from actual paid service')
        verified_arrivals.append(MappingProxyType(copy.deepcopy(original_arrival)))
        if not accepted:
            continue
        if ordinal >= len(jobs) or not isinstance(jobs[ordinal], dict):
            raise ValueError('Recorded accepted-job membership differs')
        job = copy.deepcopy(jobs[ordinal])
        try:
            if (type(job['decision']) is not int or job['decision'] != decision
                    or type(job['source_decision']) is not int or job['source_decision'] != decision
                    or type(job['job_index']) is not int or job['job_index'] != ordinal
                    or type(job['history_cutoff_exclusive']) is not int
                    or job['history_cutoff_exclusive'] != 60 + decision or job['mode'] != queue.mode):
                raise ValueError('Recorded source/job ordinal/history membership differs')
            kind = job['kind']
            if kind not in queue.profile:
                raise ValueError('Recorded timing regime differs')
            prediction = _seconds(job['actual_prediction_seconds'], 'Recorded predictor time')
            issue = _seconds(job['actual_issue_seconds'], 'Recorded issue time')
            actual = max(prediction, issue)
            compute = max(queue.profile[kind], actual)
            service = _seconds(compute + queue.extra, 'Recorded charged service')
            start = max(float(decision), previous)
            equal = lambda value, target: math.isclose(_seconds(value, 'Recorded timing'), target,
                                                      abs_tol=1e-12, rel_tol=0)
            if (not equal(job['issue_seconds'], float(decision))
                    or not equal(job['start_seconds'], start)
                    or not equal(job['actual_compute_seconds'], actual)
                    or not equal(job['profile_seconds'], queue.profile[kind])
                    or not equal(job['charged_compute_seconds'], compute)
                    or not equal(job['shared_extra_seconds'], queue.extra)
                    or not equal(job['charged_seconds'], service)
                    or job['predictor_runtime_overrun'] is not (prediction > queue.profile[kind])
                    or job['runtime_overrun'] is not (actual > queue.profile[kind])
                    or job['completion_uses_actual_elapsed'] is not True
                    or job['accounting_mode'] != 'actual_cost'
                    or job['offline_recorded_schedule'] is not False):
                raise ValueError('Recorded schedule was not charged from measured completion')
            previous = start + service
            ready = _ready_slot(previous)
            if type(job['ready_slot']) is not int or job['ready_slot'] != ready:
                raise ValueError('Recorded paid completion differs')
            shape = job['forecast_shape']
            if (not isinstance(shape, (tuple, list)) or tuple(shape) != queue.mean.shape[1:]
                    or any(type(value) is not int for value in shape)
                    or job['forecast_dtype'] != 'float32'):
                raise ValueError('Recorded forecast geometry/dtype differs')
            for key in ('forecast_sha256', 'forecast_std_sha256'):
                value = job[key]
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in '0123456789abcdef' for char in value)):
                    raise ValueError('Recorded forecast hashes are incomplete')
            _index(job['heard_entries'], 'Recorded heard count', queue.history.n * len(queue.predictor.geometry.sites) * 3 + 1)
            _index(job['scans_last_bin'], 'Recorded scan count', queue.history.n + 1)
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('Malformed recorded schedule') from error
        job['forecast_shape'] = tuple(shape)
        verified_jobs.append(MappingProxyType(job))
        finishes.append(previous)
        ordinal += 1
    if ordinal != len(jobs):
        raise ValueError('Recorded schedule contains extra accepted jobs')
    return tuple(verified_jobs), tuple(verified_arrivals), tuple(finishes)


class RecordedScheduleReplay(FreshForecasts):
    """Offline original admission/completion; fresh cost is separate evidence.

    Fresh forwards do not decide skips or replace original availability.
    Equality proves replay, never fresh measured deadline coverage.
    """

    def __init__(self, predictor, history, duration, profile, extra, *, mode,
                 recorded_jobs, recorded_arrivals):
        super().__init__(predictor, history, duration, profile, extra, mode=mode)
        jobs, arrivals, finishes = _verified_schedule(self, recorded_jobs, recorded_arrivals)
        self.recorded_jobs, self.recorded_arrivals = jobs, arrivals
        self.recorded_ready = tuple(job['ready_slot'] for job in jobs)
        self._recorded_finishes = finishes
        self._fresh_previous_ready = 0.
        self.offline_only, self.accounting_mode = True, 'recorded_schedule_offline'

    def issue(self, decision, plan):
        if self.failed:
            raise RuntimeError('Recorded replay failed; construct a fresh runtime')
        decision = _index(decision, 'Replay decision', len(self.mean))
        if decision != len(self.arrivals):
            raise ValueError('Recorded replay arrivals must be issued in strict time order')
        arrival = self.recorded_arrivals[decision]
        if not arrival['accepted']:
            self.arrivals.append(dict(arrival))
            return None
        index = arrival['job_index']
        original = self.recorded_jobs[index]
        try:
            mean, std, fresh, finish = self._measure_job(decision, plan,
                previous=self._fresh_previous_ready, job_index=index)
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
                fresh_start_seconds=fresh['start_seconds'],
                fresh_charged_compute_seconds=fresh['charged_compute_seconds'],
                fresh_charged_seconds=fresh['charged_seconds'],
                exceeds_original_recorded_compute=fresh['actual_compute_seconds'] > original['charged_compute_seconds'],
                fresh_exceeds_original_actual_compute=fresh['actual_compute_seconds'] > original['actual_compute_seconds'],
                matches_recorded_forecast=(fresh['forecast_sha256'] == original['forecast_sha256']
                    and fresh['forecast_std_sha256'] == original['forecast_std_sha256']))
        except Exception:
            self.failed = True
            raise
        self._fresh_previous_ready = finish
        self.previous_ready = self._recorded_finishes[index]
        self.ready.append(original['ready_slot'])
        self.records.append(record)
        self.arrivals.append(dict(arrival))
        return mean, std
