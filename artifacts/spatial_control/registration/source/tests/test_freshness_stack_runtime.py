"""Software fixtures for scheduling, never scientific model/site evidence."""
from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
import pytest

from wirelessjev.fleetwm.data import Observations
from wirelessjev.freshness_stack.runtime import FreshForecasts, RecordedScheduleReplay
from wirelessjev.jepa_repair.runtime import CompletionSafeForecasts
from wirelessjev.native.world import HORIZONS

PLAN = np.zeros((2, len(HORIZONS), 2), dtype='f4')


class History:
    n = 2

    def __init__(self, *, delay=0., broken=False):
        self.calls, self.delay, self.broken = [], delay, broken

    def before(self, decision):
        self.calls.append(decision)
        if self.delay:
            time.sleep(self.delay)
        t = 60 + decision
        heard = np.zeros((t, 2, 6), dtype=bool)
        heard[-1, :, 0] = True
        if self.broken:
            heard = heard[:0]
        return Observations(np.zeros((t, 2, 2), dtype='f4'),
            np.zeros((t, 2, 6), dtype='f4'), heard,
            np.full((t, 2, 6), -90., dtype='f4'), np.zeros((t, 2), dtype=bool))


class Predictor:
    def __init__(self, elapsed=(.01,) * 60, bad=None):
        self.geometry = SimpleNamespace(sites=np.zeros((2, 2)))
        self.elapsed, self.bad, self.calls = list(elapsed), bad, []

    def step(self, past, plan):
        assert isinstance(past, Observations)
        decision = len(past.value) - 60
        self.calls.append(decision)
        mean = np.broadcast_to(np.asarray(HORIZONS)[None, :, None, None], (2, 9, 2, 3)).astype('f4').copy()
        mean += decision * 100
        std = np.ones_like(mean)
        kind = 'cold' if len(self.calls) == 1 else 'warm'
        if self.bad == 'nan': mean[0, 0, 0, 0] = np.nan
        if self.bad == 'negative_std': std[0, 0, 0, 0] = -1
        if self.bad == 'shape': mean = mean[:, :-1]
        if self.bad == 'kind': kind = 'invalid'
        return mean, std, kind, self.elapsed[len(self.calls) - 1]


def queue(mode='latest', duration=6, *, elapsed=(.01,) * 60, allowance=.85,
          extra=.5, source=None, bad=None):
    return FreshForecasts(Predictor(elapsed, bad), History() if source is None else source,
        duration, dict(cold=allowance, warm=allowance, refit=allowance), extra, mode=mode)


def issue_all(live):
    for decision in range(len(live.mean)):
        live.issue(decision, PLAN)
    return live


def replay(original, *, elapsed=(.01,) * 60):
    return RecordedScheduleReplay(Predictor(elapsed), History(), len(original.mean),
        original.profile, original.extra, mode=original.mode,
        recorded_jobs=original.records, recorded_arrivals=original.arrivals)


def test_latest_refuses_work_before_reading_plan_or_copying_history():
    class UnreadablePlan:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Skipped plan must not be read')
    live = queue()
    live.issue(0, PLAN)
    assert live.issue(1, UnreadablePlan()) is None
    assert live.predictor.calls == live.history.calls == [0]
    assert live.arrivals[1] == {'decision': 1, 'arrival_slot': 100, 'accepted': False,
        'job_index': None, 'reason': 'busy_previous_job', 'previous_ready_seconds': 1.35,
        'previous_ready_slot': 135, 'mode': 'latest'}
    assert not live.mean[1].any() and not live.std[1].any()


def test_sparse_source_time_is_not_accepted_job_ordinal():
    live = issue_all(queue())
    assert live.predictor.calls == live.history.calls == [0, 2, 4]
    assert [row['decision'] for row in live.records] == [0, 2, 4]
    assert [row['source_decision'] for row in live.records] == [0, 2, 4]
    assert [row['job_index'] for row in live.records] == [0, 1, 2]
    assert live.ready == [135, 335, 535]
    assert [row['start_seconds'] for row in live.records] == [0., 2., 4.]
    assert live.take(134) is None
    assert live.take(135) == 0
    assert live.take(335) == 2
    assert live.take(535) == 4
    assert live.last_job == 2 and live.last == 4
    assert all(row['age_seconds'] == pytest.approx(1.35) for row in live.consumption_records)
    np.testing.assert_allclose(live.advance(live.mean[4])[:, 0], 401.35, atol=2e-5)


def test_fifo_preserves_predecessor_schedules_outputs_and_consumption():
    live = queue('fifo')
    old = CompletionSafeForecasts(Predictor(), History(), 6, live.profile, live.extra)
    for decision in range(6):
        live.issue(decision, PLAN); old.issue(decision, PLAN)
    assert live.ready == old.ready == [135, 270, 405, 540, 675, 810]
    np.testing.assert_array_equal(live.mean, old.mean)
    np.testing.assert_array_equal(live.std, old.std)
    for a, b in zip(live.records, old.records):
        for key in b:
            if key.startswith('actual_'):
                continue
            assert a[key] == b[key], key
    for slot in range(900):
        assert live.take(slot) == old.take(slot)
    assert live.consumption_records == old.consumption_records


def test_current_age_and_readonly_view_update_held_age_without_new_consumption():
    live = issue_all(queue())
    assert live.current_age(0) is None and live.forecast_for_slot(0) is None
    live.take(135)
    saved = deepcopy(live.consumption_records)
    assert live.current_age(334) == pytest.approx(3.34)
    mean, std, source, age = live.forecast_for_slot(334)
    assert source == 0 and age == pytest.approx(3.34)
    np.testing.assert_allclose(mean[:, 0], 3.34, atol=1e-6)
    np.testing.assert_array_equal(std, 1.)
    assert live.age == pytest.approx(1.35)
    assert live.consumption_records == saved and live.last == 0


def test_available_but_unconsumed_job_cannot_leak_into_readonly_view():
    live = issue_all(queue())
    live.take(135)
    mean, _, source, age = live.forecast_for_slot(400)
    assert source == 0 and age == 4.
    np.testing.assert_allclose(mean[:, 0], 4.)
    assert live.take(400) == 2
    assert live.forecast_for_slot(400)[2] == 2


def test_actual_overrun_extends_service_and_the_next_admission_uses_paid_finish():
    live = queue(elapsed=(2.21, .01, .01))
    issue_all(live)
    assert live.predictor.calls == [0, 3, 5]
    assert live.records[0]['charged_seconds'] == pytest.approx(2.71)
    assert live.ready == [271, 435, 635]
    assert live.records[0]['runtime_overrun']
    assert live.take(270) is None
    assert live.take(271) == 0


def test_history_copy_is_measured_and_cannot_be_hidden_by_model_timer():
    live = queue(allowance=.001, extra=.02, elapsed=(0.,), source=History(delay=.025))
    live.issue(0, PLAN)
    job = live.records[0]
    assert job['actual_issue_seconds'] >= .02
    assert job['actual_prediction_seconds'] == 0.
    assert job['charged_seconds'] >= job['actual_issue_seconds'] + .02
    assert job['runtime_overrun'] and not job['predictor_runtime_overrun']
    assert live.take(job['ready_slot'] - 1) is None
    assert live.take(job['ready_slot']) == 0


def test_completion_exactly_at_arrival_is_admitted():
    live = issue_all(queue(allowance=.5, extra=.5))
    assert [row['decision'] for row in live.records] == list(range(6))
    assert live.ready == [100, 200, 300, 400, 500, 600]


def test_zero_duration_service_can_complete_at_its_source_arrival(monkeypatch):
    import wirelessjev.freshness_stack.runtime as runtime
    monkeypatch.setattr(runtime.time, 'perf_counter', lambda: 0.)
    live = issue_all(queue(allowance=0., extra=0., elapsed=(0.,) * 60))
    assert live.ready == [0, 100, 200, 300, 400, 500]
    assert live.take(0) == 0


def test_consumption_can_skip_multiple_ready_jobs_but_returns_latest_source():
    live = issue_all(queue())
    assert live.take(550) == 4
    assert live.consumption_records == [dict(slot=550, forecast_decision=4,
        ready_slot=535, age_seconds=1.5)]
    assert live.take(551) is None


@pytest.mark.parametrize('decision', [-1, True, .5, np.nan, 1, 6])
def test_bad_arrival_order_or_type_cannot_touch_predictor_or_mark_failed(decision):
    live = queue()
    with pytest.raises(ValueError): live.issue(decision, PLAN)
    assert not live.predictor.calls and not live.history.calls and not live.arrivals
    assert not live.failed


@pytest.mark.parametrize('slot', [-1, True, .5, np.inf, np.nan, 10**40])
def test_invalid_clock_never_consumes(slot):
    live = queue(); live.issue(0, PLAN)
    with pytest.raises(ValueError): live.take(slot)
    assert not live.consumption_records and live.last == -1


def test_readonly_and_consumption_clocks_cannot_rewind():
    live = queue(); live.issue(0, PLAN); live.take(135)
    for method in (live.take, live.current_age, live.forecast_for_slot):
        with pytest.raises(ValueError, match='time order'): method(134)


@pytest.mark.parametrize('bad,elapsed', [('nan', .01), ('negative_std', .01), ('shape', .01),
    ('kind', .01), (None, -1.), (None, np.inf), (None, True)])
def test_prediction_failure_cannot_publish_any_arrival_or_ready_job(bad, elapsed):
    live = queue(bad=bad, elapsed=(elapsed,))
    with pytest.raises(ValueError): live.issue(0, PLAN)
    assert live.failed and not live.arrivals and not live.ready and not live.records
    assert live.take(10000) is None
    with pytest.raises(RuntimeError): live.issue(0, PLAN)


def test_malformed_history_fails_before_predictor():
    live = queue(source=History(broken=True))
    with pytest.raises(ValueError, match='history axes'): live.issue(0, PLAN)
    assert live.failed and not live.predictor.calls and not live.records


@pytest.mark.parametrize('mode', [None, True, '', 'coalesce', 'LATEST'])
def test_unknown_admission_mode_is_rejected(mode):
    with pytest.raises(ValueError, match='mode'): queue(mode)


def test_offline_replay_preserves_original_sparse_admission_despite_fresh_overrun():
    original = issue_all(queue())
    recorded = replay(original, elapsed=(3.2, .01, .01))
    issue_all(recorded)
    assert recorded.offline_only and recorded.accounting_mode == 'recorded_schedule_offline'
    assert recorded.predictor.calls == [0, 2, 4]
    assert recorded.arrivals == original.arrivals
    assert recorded.ready == original.ready == [135, 335, 535]
    np.testing.assert_array_equal(recorded.mean, original.mean)
    np.testing.assert_array_equal(recorded.std, original.std)
    assert recorded.records[0]['fresh_ready_slot'] == 370
    assert recorded.records[1]['fresh_start_seconds'] == pytest.approx(3.7)
    assert recorded.records[1]['fresh_ready_slot'] == 505
    assert recorded.records[0]['fresh_actual_prediction_seconds'] == 3.2
    assert recorded.records[0]['actual_prediction_seconds'] == .01
    assert recorded.records[0]['exceeds_original_recorded_compute']
    assert recorded.records[0]['fresh_runtime_overrun']
    assert all(job['matches_recorded_forecast'] for job in recorded.records)
    assert all(job['offline_recorded_schedule'] and not job['completion_uses_actual_elapsed']
               and job['recorded_completion_used_actual_elapsed'] for job in recorded.records)
    for slot in range(600):
        assert recorded.take(slot) == original.take(slot)
    assert recorded.consumption_records == original.consumption_records


def test_offline_skips_do_not_read_plan_or_snapshot_history():
    original = issue_all(queue())
    recorded = replay(original)
    recorded.issue(0, PLAN)
    assert recorded.issue(1, object()) is None
    assert recorded.history.calls == recorded.predictor.calls == [0]


def test_offline_fifo_replay_can_have_different_live_costs_without_retiming_original():
    original = issue_all(queue('fifo', duration=4))
    recorded = replay(original, elapsed=(3.2, .01, .01, .01))
    issue_all(recorded)
    assert recorded.ready == original.ready
    assert recorded.arrivals == original.arrivals
    assert recorded.records[0]['fresh_ready_slot'] > original.ready[0]
    assert recorded.take(original.ready[0]) == 0


def test_replay_detaches_complete_ledger_before_caller_mutation():
    original = issue_all(queue())
    recorded = replay(original)
    original.arrivals[1]['accepted'] = True
    original.records[0]['ready_slot'] = 9999
    original.records[0]['forecast_shape'][0] = 999
    recorded.issue(0, PLAN)
    assert recorded.ready == [135]
    assert recorded.records[0]['forecast_shape'] == (2, 9, 2, 3)
    assert recorded.issue(1, object()) is None
    with pytest.raises(TypeError): recorded.recorded_jobs[0]['ready_slot'] = 1
    with pytest.raises(TypeError): recorded.recorded_arrivals[1]['accepted'] = True


@pytest.mark.parametrize('damage', ['arrivals_length', 'jobs_length', 'extra_job',
    'arrival_source', 'arrival_accept', 'arrival_ordinal', 'arrival_busy', 'arrival_mode',
    'arrival_bool', 'arrival_extra', 'job_source', 'job_ordinal', 'job_cutoff', 'job_ready',
    'job_start', 'job_issue', 'job_undercharge', 'job_actual', 'job_kind', 'job_shape',
    'job_dtype', 'job_hash', 'job_std_hash', 'job_overrun', 'job_premade_offline',
    'job_heard_bool', 'job_missing'])
def test_recorded_schedule_tamper_fails_before_any_model_or_history_access(damage):
    original = issue_all(queue())
    jobs, arrivals = deepcopy(original.records), deepcopy(original.arrivals)
    if damage == 'arrivals_length': arrivals.pop()
    elif damage == 'jobs_length': jobs.pop()
    elif damage == 'extra_job': jobs.append(deepcopy(jobs[-1]))
    elif damage == 'arrival_source': arrivals[1]['decision'] = 0
    elif damage == 'arrival_accept': arrivals[1]['accepted'] = True
    elif damage == 'arrival_ordinal': arrivals[2]['job_index'] = 2
    elif damage == 'arrival_busy': arrivals[1]['previous_ready_seconds'] -= .1
    elif damage == 'arrival_mode': arrivals[1]['mode'] = 'fifo'
    elif damage == 'arrival_bool': arrivals[1]['accepted'] = 0
    elif damage == 'arrival_extra': arrivals[1]['invented'] = 1
    elif damage == 'job_source': jobs[1]['source_decision'] = 1
    elif damage == 'job_ordinal': jobs[1]['job_index'] = 2
    elif damage == 'job_cutoff': jobs[1]['history_cutoff_exclusive'] = True
    elif damage == 'job_ready': jobs[1]['ready_slot'] -= 1
    elif damage == 'job_start': jobs[1]['start_seconds'] = 0.
    elif damage == 'job_issue': jobs[1]['issue_seconds'] = 0.
    elif damage == 'job_undercharge': jobs[1]['charged_seconds'] = .01
    elif damage == 'job_actual': jobs[1]['actual_compute_seconds'] = -1.
    elif damage == 'job_kind': jobs[1]['kind'] = 'unknown'
    elif damage == 'job_shape': jobs[1]['forecast_shape'][0] = 99
    elif damage == 'job_dtype': jobs[1]['forecast_dtype'] = 'float64'
    elif damage == 'job_hash': jobs[1]['forecast_sha256'] = 'z' * 64
    elif damage == 'job_std_hash': jobs[1]['forecast_std_sha256'] = ''
    elif damage == 'job_overrun': jobs[1]['runtime_overrun'] = True
    elif damage == 'job_premade_offline': jobs[1]['offline_recorded_schedule'] = True
    elif damage == 'job_heard_bool': jobs[1]['heard_entries'] = True
    elif damage == 'job_missing': jobs[1].pop('kind')
    predictor, source = Predictor(), History()
    with pytest.raises(ValueError):
        RecordedScheduleReplay(predictor, source, 6, original.profile, original.extra,
            mode='latest', recorded_jobs=jobs, recorded_arrivals=arrivals)
    assert predictor.calls == source.calls == []


def test_replay_rejects_changed_timing_regime_and_cannot_publish_job():
    original = issue_all(queue())
    recorded = replay(original)
    original_step = recorded.predictor.step
    def wrong_kind(past, plan):
        m, s, _, elapsed = original_step(past, plan)
        return m, s, 'warm', elapsed
    recorded.predictor.step = wrong_kind
    with pytest.raises(ValueError, match='regime'): recorded.issue(0, PLAN)
    assert recorded.failed and not recorded.arrivals and not recorded.records
    with pytest.raises(RuntimeError): recorded.issue(0, PLAN)
