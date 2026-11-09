"""Accounting and controller mathematics; no scientific sites or model fitting."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.fleetwm.data import Geometry, Observations
from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.jepa_repair.runtime import CompletionSafeForecasts, ExactOutageAdmission, RecordedScheduleReplay
from wirelessjev.native.gate import RoamDecision
from wirelessjev.native.world import HORIZONS


def history(duration=4, n=2):
    warm = Observations(np.zeros((60, n, 2), dtype='f4'),
        np.zeros((60, n, 6), dtype='f4'), np.zeros((60, n, 6), dtype=bool),
        np.full((60, n, 6), -90., dtype='f4'), np.zeros((60, n), dtype=bool))
    return DeliveredScans(warm, duration, 745)


class TimedPublicPredictor:
    """A deterministic timing-contract fixture, never scientific evidence."""
    def __init__(self, elapsed, *, delay=0., bad=None):
        self.geometry = Geometry(np.array([[0., 0.], [4., 0.]]), np.zeros(3),
                                 np.full(3, -95.), ())
        self.elapsed, self.delay, self.bad = list(elapsed), delay, bad
        self.calls = 0

    def step(self, past, plan):
        assert isinstance(past, Observations)
        assert len(past.value) == 60 + self.calls
        if self.delay:
            time.sleep(self.delay)
        mean = np.broadcast_to(np.asarray(HORIZONS)[None, :, None, None], (len(plan), 9, 2, 3)).astype('f4').copy()
        std = np.ones_like(mean)
        elapsed = self.elapsed[self.calls]
        kind = 'cold' if self.calls == 0 else 'warm'
        self.calls += 1
        if self.bad == 'nan':
            mean[0, 0, 0, 0] = np.nan
        elif self.bad == 'negative_std':
            std[0, 0, 0, 0] = -1
        elif self.bad == 'shape':
            mean = mean[:, :-1]
        elif self.bad == 'kind':
            kind = 'unknown'
        return mean, std, kind, elapsed


PLAN = np.zeros((2, 9, 2), dtype='f4')


def queue(elapsed=(.01, .01, .01), *, profile=.5, extra=.5, delay=0., bad=None):
    return CompletionSafeForecasts(TimedPublicPredictor(elapsed, delay=delay, bad=bad), history(), 4,
                                  {kind: profile for kind in ('cold', 'warm', 'refit')}, extra)


def test_profile_overrun_extends_fifo_and_remains_a_visible_diagnostic():
    live = queue((1.21, .04, .06), profile=.12)
    for decision in range(3):
        live.issue(decision, PLAN)
    assert live.ready == [171, 233, 295]
    assert [record['runtime_overrun'] for record in live.records] == [True, False, False]
    assert live.records[0]['charged_seconds'] == pytest.approx(1.71)
    assert live.records[0]['profile_seconds'] == .12
    assert live.records[0]['predictor_runtime_overrun'] is True
    assert live.take(170) is None
    assert live.take(171) == 0 and live.age == pytest.approx(1.71)
    assert live.take(232) is None
    assert live.take(233) == 1 and live.age == pytest.approx(1.33)
    advanced = live.advance(live.mean[1])
    np.testing.assert_allclose(advanced[:, 0], 1.33, atol=1e-6)
    assert live.consumption_records == [
        {'slot': 171, 'forecast_decision': 0, 'ready_slot': 171, 'age_seconds': pytest.approx(1.71)},
        {'slot': 233, 'forecast_decision': 1, 'ready_slot': 233, 'age_seconds': pytest.approx(1.33)}]


def test_measured_real_work_cannot_be_hidden_by_an_underreported_predictor_time():
    live = queue((0.,), profile=.001, extra=.02, delay=.025)
    live.issue(0, PLAN)
    record = live.records[0]
    assert record['actual_issue_seconds'] >= .02
    assert record['actual_prediction_seconds'] == 0
    assert record['predictor_runtime_overrun'] is False
    assert record['runtime_overrun'] is True
    assert record['charged_seconds'] >= record['actual_issue_seconds'] + .02
    assert live.ready[0] / 100 + 1e-12 >= record['actual_issue_seconds'] + .02
    assert live.take(live.ready[0] - 1) is None
    assert live.take(live.ready[0]) == 0


def test_history_snapshot_time_is_measured_outside_the_predictor():
    source = history()
    class SlowHistory:
        n = source.n
        def before(self, decision):
            time.sleep(.025)
            return source.before(decision)
    live = CompletionSafeForecasts(TimedPublicPredictor([0.]), SlowHistory(), 4,
                                  dict(cold=.001, warm=.001, refit=.001), .02)
    live.issue(0, PLAN)
    record = live.records[0]
    assert record['actual_prediction_seconds'] == 0
    assert record['actual_issue_seconds'] >= .02 and record['runtime_overrun']
    assert live.ready[0] / 100 + 1e-12 >= record['actual_issue_seconds'] + .02


def test_malformed_history_cannot_commit_a_job_before_metadata_failure():
    source = history()
    class BrokenHistory:
        n = source.n
        def before(self, decision):
            past = source.before(decision)
            return Observations(past.positions, past.value, past.heard[:0], past.bound, past.scanned)
    live = CompletionSafeForecasts(TimedPublicPredictor([0.]), BrokenHistory(), 4,
                                  dict(cold=.1, warm=.1, refit=.1), .5)
    with pytest.raises(ValueError, match='history axes'):
        live.issue(0, PLAN)
    assert live.failed and not live.ready and not live.records and live.previous_ready == 0
    assert live.predictor.calls == 0 and live.take(10000) is None


def test_different_actual_times_within_profile_preserve_exact_release_and_forecast_hashes():
    a, b = queue((.01, .04, .03)), queue((.08, .02, .05))
    for decision in range(3):
        a.issue(decision, PLAN); b.issue(decision, PLAN)
    assert a.ready == b.ready == [100, 200, 300]
    assert all(not row['runtime_overrun'] for row in a.records + b.records)
    assert [row['forecast_sha256'] for row in a.records] == [row['forecast_sha256'] for row in b.records]
    assert [row['forecast_std_sha256'] for row in a.records] == [row['forecast_std_sha256'] for row in b.records]
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.std, b.std)


@pytest.mark.parametrize('decision', [-1, True, .5, np.nan, 1, 4])
def test_issue_order_rejected_before_predictor_mutation(decision):
    live = queue()
    with pytest.raises(ValueError):
        live.issue(decision, PLAN)
    assert live.predictor.calls == 0 and not live.records and not live.failed
    live.issue(0, PLAN)
    with pytest.raises(ValueError):
        live.issue(0, PLAN)
    assert live.predictor.calls == 1


@pytest.mark.parametrize('value', [np.nan, np.inf, -1., True])
def test_invalid_budget_or_shared_delay_is_rejected(value):
    with pytest.raises(ValueError, match='finite and nonnegative'):
        queue(profile=value)
    with pytest.raises(ValueError, match='finite and nonnegative'):
        queue(extra=value)


@pytest.mark.parametrize('bad,elapsed', [('nan', .01), ('negative_std', .01), ('shape', .01),
                                       ('kind', .01), (None, -1.), (None, np.inf), (None, np.nan)])
def test_failed_prediction_never_creates_a_releasable_job(bad, elapsed):
    live = queue((elapsed,), bad=bad)
    with pytest.raises(ValueError):
        live.issue(0, PLAN)
    assert live.failed and live.ready == [] and live.records == []
    assert live.take(10000) is None
    with pytest.raises(RuntimeError, match='fresh runtime'):
        live.issue(0, PLAN)


def test_profile_keys_and_duration_are_checked_before_allocating():
    for profile in ({'warm': .2}, {'cold': .2, 'warm': .2, 'refit': .2, 'other': .2}):
        with pytest.raises(ValueError, match='cold, warm and refit'):
            CompletionSafeForecasts(TimedPublicPredictor([.01]), history(), 4, profile, .5)
    for duration in (True, 0, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            CompletionSafeForecasts(TimedPublicPredictor([.01]), history(), duration,
                                    dict(cold=.2, warm=.2, refit=.2), .5)


@pytest.mark.parametrize('slot', [np.nan, np.inf, -1, True, .5, 10 ** 40])
def test_invalid_consumption_time_cannot_release_a_future_job(slot):
    live = queue((.01,)); live.issue(0, PLAN)
    with pytest.raises(ValueError, match='integer slots'):
        live.take(slot)
    assert live.last == -1 and live.last_take_slot == -1
    assert live.take(99) is None
    assert live.take(100) == 0
    with pytest.raises(ValueError, match='time order'):
        live.take(99)


def original_schedule():
    original = queue((.01, .02, .03, .04))
    for decision in range(4):
        original.issue(decision, PLAN)
    return original


def test_offline_replay_retains_original_schedule_but_never_claims_new_runtime_coverage():
    original = original_schedule()
    replay = RecordedScheduleReplay(TimedPublicPredictor((1.2, .02, .03, .04)), history(), 4,
                                    original.profile, original.extra, recorded_jobs=original.records)
    assert replay.mean is replay.delegate.mean and replay.std is replay.delegate.std
    assert replay.offline_only and replay.accounting_mode == 'recorded_schedule_offline'
    assert replay.ready == []  # Future unissued jobs cannot be consumed.
    assert replay.recorded_ready == (100, 200, 300, 400)
    for decision in range(4):
        replay.issue(decision, PLAN)
    assert replay.ready == original.ready
    assert replay.delegate.ready[0] == 170
    first = replay.records[0]
    assert first['offline_recorded_schedule'] is True
    assert first['completion_uses_actual_elapsed'] is False
    assert first['recorded_completion_used_actual_elapsed'] is True
    assert first['actual_prediction_seconds'] == original.records[0]['actual_prediction_seconds']
    assert first['fresh_actual_prediction_seconds'] == 1.2
    assert first['fresh_runtime_overrun'] and first['exceeds_original_recorded_compute']
    assert first['fresh_exceeds_original_actual_compute']
    assert all(row['matches_recorded_forecast'] for row in replay.records)
    np.testing.assert_array_equal(replay.mean, original.mean)
    np.testing.assert_array_equal(replay.std, original.std)
    assert replay.take(99) is None
    assert replay.take(100) == 0
    assert replay.consumption_records == [
        {'slot': 100, 'forecast_decision': 0, 'ready_slot': 100, 'age_seconds': 1.}]
    np.testing.assert_allclose(replay.advance(replay.mean[0])[:, 0], 1.)


def test_offline_schedule_is_detached_from_mutation_of_input_records():
    original = original_schedule()
    replay = RecordedScheduleReplay(TimedPublicPredictor((.01,)), history(), 4,
                                    original.profile, original.extra, recorded_jobs=original.records)
    original.records[0]['ready_slot'] = 9999
    original.records[0]['forecast_shape'][0] = 999
    replay.issue(0, PLAN)
    assert replay.ready == [100]
    assert replay.records[0]['forecast_shape'] == (2, 9, 2, 3)
    with pytest.raises(TypeError):
        replay.recorded_jobs[0]['ready_slot'] = 99


@pytest.mark.parametrize('damage', ['membership', 'cutoff', 'early', 'undercharged', 'prediction',
                                  'issue', 'shape', 'dtype', 'hash', 'overrun', 'missing', 'length'])
def test_malformed_offline_ledger_fails_before_any_prediction(damage):
    import copy
    original = original_schedule(); jobs = copy.deepcopy(original.records)
    if damage == 'membership': jobs[1]['decision'] = 0
    elif damage == 'cutoff': jobs[1]['history_cutoff_exclusive'] = True
    elif damage == 'early': jobs[1]['ready_slot'] -= 1
    elif damage == 'undercharged': jobs[1]['charged_seconds'] = .01
    elif damage == 'prediction': jobs[1]['actual_prediction_seconds'] = np.nan
    elif damage == 'issue': jobs[1]['actual_issue_seconds'] = np.inf
    elif damage == 'shape': jobs[1]['forecast_shape'] = [2, 9, 3, 3]
    elif damage == 'dtype': jobs[1]['forecast_dtype'] = 'float64'
    elif damage == 'hash': jobs[1]['forecast_sha256'] = 'not a hash'
    elif damage == 'overrun': jobs[1]['runtime_overrun'] = True
    elif damage == 'missing': del jobs[1]['charged_compute_seconds']
    elif damage == 'length': jobs.pop()
    predictor = TimedPublicPredictor((.01,))
    with pytest.raises(ValueError):
        RecordedScheduleReplay(predictor, history(), 4, original.profile, original.extra, recorded_jobs=jobs)
    assert predictor.calls == 0


@pytest.mark.parametrize('slot', [np.nan, np.inf, -1, True, .5, 10 ** 40])
def test_offline_consumption_times_are_also_validated(slot):
    original = original_schedule()
    replay = RecordedScheduleReplay(TimedPublicPredictor((.01,)), history(), 4,
                                    original.profile, original.extra, recorded_jobs=original.records)
    replay.issue(0, PLAN)
    with pytest.raises(ValueError, match='integer slots'):
        replay.take(slot)
    assert replay.last == -1 and not replay.consumption_records
    assert replay.take(100) == 0
    with pytest.raises(ValueError, match='time order'):
        replay.take(99)


def test_too_large_finite_service_does_not_publish_an_unrepresentable_slot():
    live = queue((.01,), profile=1e20)
    with pytest.raises(ValueError, match='integer slot support'):
        live.issue(0, PLAN)
    assert live.failed and not live.ready and not live.records


@pytest.mark.parametrize('cost', [.03, 4.])
@pytest.mark.parametrize('snr', [-12., 0., 15.])
def test_constant_failure_has_exact_outage_cost_without_free_first_second(cost, snr):
    forecasts = np.full((1, 2, 9, 2, 3), snr, dtype='f4')
    gate = ExactOutageAdmission(forecasts, ForecastQueue([0.]), cost, 30.)
    stay, move = gate.loss(forecasts[0])
    probability = stay / 30.
    np.testing.assert_allclose(move - stay, cost * (1 - probability), atol=1e-12, rtol=1e-10)
    current = np.array([0, 1]); candidate = np.array([1, 0])
    decision = RoamDecision.capture(0, current, candidate, np.ones(2), np.ones(2, dtype=bool), np.zeros((2, 2)))
    np.testing.assert_array_equal(gate(decision), [False, False])
    score = gate.scores(forecasts[0], current)
    # At failure probability one all APs can tie; none may beat staying.
    assert (score[np.arange(2), current] >= score.max(1) - 1e-12).all()


def test_actual_phy_surrogate_demonstrates_removed_point97_second_credit():
    from wirelessjev.hardware_budget import OutageBudgetGate
    forecasts = np.full((1, 2, 9, 2, 3), -12., dtype='f4')
    old = OutageBudgetGate(forecasts, ForecastQueue([0.]), .03)
    new = ExactOutageAdmission(forecasts, ForecastQueue([0.]), .03)
    decision = RoamDecision.capture(0, np.array([0, 1]), np.array([1, 0]), np.ones(2),
                                    np.ones(2, dtype=bool), np.zeros((2, 2)))
    assert old(decision).all()
    assert not new(decision).any()


@pytest.mark.parametrize('cost', [.03, 4.])
def test_exact_admission_accepts_useful_moves_only_when_eligible(cost):
    forecasts = np.full((1, 3, 9, 2, 3), -20., dtype='f4'); forecasts[:, :, :, 1] = 30.
    gate = ExactOutageAdmission(forecasts, ForecastQueue([0.]), cost)
    assert gate.forecasts is forecasts  # Completed live requests fill this shared array.
    decision = RoamDecision.capture(0, np.zeros(3, dtype=int), np.ones(3, dtype=int), np.ones(3),
                                    np.array([True, False, True]), np.zeros((3, 2)))
    np.testing.assert_array_equal(gate(decision), [True, False, True])
    assert gate.scores(forecasts[0], decision.current).argmax(1).tolist() == [1, 1, 1]


@pytest.mark.parametrize('cost,horizon', [(np.nan, 30), (-1, 30), (True, 30), (4, 3), (4, 31)])
def test_exact_admission_rejects_invalid_integration_support(cost, horizon):
    with pytest.raises(ValueError):
        ExactOutageAdmission(np.zeros((1, 2, 9, 2, 3)), ForecastQueue([0]), cost, horizon)


def test_exact_admission_rejects_malformed_forecasts_and_indices():
    forecasts = np.zeros((1, 2, 9, 2, 3), dtype='f4')
    gate = ExactOutageAdmission(forecasts, ForecastQueue([0]), .03)
    with pytest.raises(ValueError, match='finite complete'):
        gate.loss(np.full((2, 9, 2, 3), np.nan))
    for current in (np.array([0., 1.]), np.array([0, 2]), np.array([True, False])):
        with pytest.raises(ValueError, match='AP indices'):
            gate.scores(forecasts[0], current)
    decision = SimpleNamespace(decision=0, current=np.array([0, 1]), candidate=np.array([1, 0]), eligible=np.array([1, 1]))
    with pytest.raises(ValueError, match='boolean eligibility'):
        gate(decision)
