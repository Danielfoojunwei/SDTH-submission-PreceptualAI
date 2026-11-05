"""Physical/cadence counterexamples for cached-prediction local feedback."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest

from wirelessjev.native.world import build_world, HORIZONS
from wirelessjev.native.sim import Policy
from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.fleetwm.data import Observations
from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
from wirelessjev.freshness_stack.sim import simulate as previous_simulate
from wirelessjev.pilot_fastloop.sim import simulate, fast_control_evidence
from wirelessjev.pilot_fastloop.telemetry import PilotDeliveredScans


def evidence(*, source=0, slot=100, current=0, values=None, measured=(15., 10., 5.),
             band=1, available=True, **metadata):
    if values is None:
        values = np.full((6, 1, 9, 2, 3), 20., dtype='f4')
    queue = SimpleNamespace(last=source, mean=values,
        records=[{'decision': d, 'ready_slot': max(0, d * 100),
                  'forecast_sha256': hashlib.sha256(values[d].tobytes()).hexdigest()}
                 for d in range(len(values))])
    defaults = dict(pilot_site=current, pilot_capture_slot=max(0, slot - 10),
        pilot_refresh_slot=slot, association_epoch_start=0, last_primary_unavailable=-1)
    defaults.update(metadata)
    return fast_control_evidence(queue, slot, 0, current, measured,
        control_band=band, primary_available=available, **defaults)


def test_current_pilot_is_origin_and_constant_calibration_cancels():
    values = np.full((6, 1, 9, 2, 3), -100., dtype='f4')
    row = evidence(values=values)
    assert row['mode'] == 'predictive' and row['proposed_control_band'] == 0
    assert row['worst_future_snr_db'] == [15., 10., 5.]
    assert row['native_delta_db_per_second'] == [0., 0., 0.]
    np.testing.assert_array_equal(row['route_innovation_db'], 0.)


def test_internal_original_knot_is_not_missed_after_fractional_age():
    values = np.full((6, 1, 9, 2, 3), 20., dtype='f4')
    values[0, 0, 2, 0, 0] = 0.  # source-time h2 is between new native query times
    row = evidence(values=values, slot=50, measured=(15., 2., -10.))
    assert row['mode'] == 'predictive'
    assert 1.5 in row['query_seconds_from_now']
    assert row['worst_future_snr_db'][0] == -5.
    assert row['proposed_control_band'] == 1
    # Six shifted integer points alone miss the dip and choose band0.
    integer_values = np.interp(.5 + np.arange(6), HORIZONS, values[0, 0, :, 0, 0])
    assert 15. + integer_values.min() - integer_values[0] > 2.


def test_MCS_corrects_capture_to_now_even_when_interval_crosses_a_corner():
    values = np.full((6, 1, 9, 2, 3), 20., dtype='f4')
    values[0, 0, :, 0, 0] = [0., 10., 0., 0., 0., 0., 0., 0., 0.]
    row = evidence(values=values, slot=105)
    # Capture .95 and current 1.05 have identical values. Future slope is
    # negative, but estimating current SNR from this pilot requires no change.
    assert row['native_delta_db_per_second'][0] == pytest.approx(0., abs=1e-12)
    assert row['forecast_at_capture_snr_db'][0] == pytest.approx(9.5)
    assert row['forecast_at_now_snr_db'][0] == pytest.approx(9.5)


def test_MCS_uses_capture_interval_rather_than_future_interval_before_slope_change():
    values = np.full((6, 1, 9, 2, 3), 20., dtype='f4')
    values[0, 0, :, 0, 0] = [0., 10., 0., 0., 0., 0., 0., 0., 0.]
    row = evidence(values=values, slot=95)
    assert row['native_mcs_correction_db'][0] == pytest.approx(1.)
    assert row['native_delta_db_per_second'][0] == pytest.approx(10.)
    # Using f(now+.1)-f(now) would incorrectly cancel this rising history.
    assert row['forecast_at_capture_snr_db'][0] == pytest.approx(8.5)
    assert row['forecast_at_now_snr_db'][0] == pytest.approx(9.5)


def test_irregular_capture_age_applies_full_correction_through_original100ms_multiplier():
    values = np.full((6, 1, 9, 2, 3), 20., dtype='f4')
    values[0, 0, :, 0, 0] = [0., 10., 0., 0., 0., 0., 0., 0., 0.]
    row = evidence(values=values, slot=105, pilot_capture_slot=90,
                   pilot_refresh_slot=100)
    assert row['forecast_capture_age_seconds'] == pytest.approx(.9)
    assert row['native_mcs_correction_db'][0] == pytest.approx(.5)
    assert row['native_delta_db_per_second'][0] * .1 == pytest.approx(.5)
    # First future point is current, corrected from the actual capture.
    assert row['route_innovation_db'][0][0] == pytest.approx(.5)
    assert row['forecast_at_now_snr_db'][0] == pytest.approx(9.5)


def test_capture_before_source_support_uses_myopic_pilot_without_extrapolation():
    row = evidence(source=1, slot=105, pilot_capture_slot=95,
                   pilot_refresh_slot=100)
    assert row['mode'] == 'myopic'
    assert row['reason'] == 'stale_or_unsupported_forecast'
    assert row['forecast_capture_age_seconds'] < 0.
    assert row['native_mcs_correction_db'] == [0., 0., 0.]


def test_sparse_source_timestamp_determines_age_not_job_ordinal():
    values = np.full((6, 1, 9, 2, 3), 20., dtype='f4')
    queue = SimpleNamespace(last=2, mean=values,
        records=[{'decision': 0, 'ready_slot': 150, 'forecast_sha256': 'a' * 64},
                 {'decision': 2, 'ready_slot': 350, 'forecast_sha256': 'b' * 64}])
    row = fast_control_evidence(queue, 400, 0, 0, [15., 10., 5.], control_band=1,
        primary_available=True, pilot_site=0, pilot_capture_slot=390,
        pilot_refresh_slot=400, association_epoch_start=0, last_primary_unavailable=-1)
    assert row['mode'] == 'predictive'
    assert row['source_decision'] == 2 and row['forecast_age_seconds'] == 2.


@pytest.mark.parametrize('metadata', [
    {'pilot_site': 1}, {'pilot_capture_slot': 80}, {'pilot_refresh_slot': 90},
    {'pilot_capture_slot': 101}, {'pilot_refresh_slot': 101},
    {'association_epoch_start': 91}, {'last_primary_unavailable': 90},
    {'association_epoch_start': False}, {'association_epoch_start': -1},
    {'last_primary_unavailable': -2}, {'last_primary_unavailable': 101},
    {'pilot_capture_slot': 98, 'pilot_refresh_slot': 95},
    {'pilot_capture_slot': -1}, {'pilot_refresh_slot': -1},
])
def test_old_association_unavailable_or_future_capture_holds_band(metadata):
    row = evidence(band=2, **metadata)
    assert row['mode'] == 'hold' and row['proposed_control_band'] == 2
    assert row['native_delta_db_per_second'] == [0., 0., 0.]


def test_offtick_handover_and_pre_outage_capture_are_not_current_evidence():
    wrong_ap = evidence(current=1, slot=405, pilot_site=0,
        pilot_capture_slot=390, pilot_refresh_slot=400,
        association_epoch_start=5, last_primary_unavailable=404)
    assert wrong_ap['mode'] == 'hold'
    # First post-outage refresh still reports a lagged time inside the outage.
    past_outage = evidence(current=1, slot=410, pilot_capture_slot=400,
        pilot_refresh_slot=410, association_epoch_start=5, last_primary_unavailable=404)
    assert past_outage['mode'] == 'hold'
    observed_after = evidence(current=1, slot=420, pilot_capture_slot=410,
        pilot_refresh_slot=420, association_epoch_start=5, last_primary_unavailable=404)
    assert observed_after['mode'] == 'predictive'


def test_valid_pilot_missing_or_stale_model_uses_myopic_fallback():
    assert evidence(source=-1)['mode'] == 'myopic'
    assert evidence(slot=501)['mode'] == 'myopic'
    row = evidence(available=False)
    assert row['mode'] == 'hold' and row['proposed_control_band'] == 1


@pytest.mark.parametrize('bad', [['a', 'b', 'c'], [1., np.nan, 3.], [[1.], [2.], [3.]]])
def test_malformed_pilot_cannot_change_band(bad):
    row = evidence(measured=bad, band=2)
    assert row['mode'] == 'hold' and row['proposed_control_band'] == 2


@pytest.mark.parametrize('bad', [np.nan, np.inf, 'x'])
def test_malformed_forecast_uses_valid_pilot_only(bad):
    dtype = object if isinstance(bad, str) else 'f4'
    values = np.full((6, 1, 9, 2, 3), bad, dtype=dtype)
    row = evidence(values=values)
    assert row['mode'] == 'myopic' and row['proposed_control_band'] == 0


class KnownQueue(ForecastQueue):
    def __init__(self, mean, *, sources=None, ready=None):
        self.mean = mean
        self.sources = list(range(len(mean))) if sources is None else sources
        self.ready = np.asarray([d * 100 + 20 for d in self.sources] if ready is None else ready)
        self.last_job, self.last, self.age = -1, -1, 0.
        self.calls, self.consumed = 0, []
        self.records = [{'decision': d, 'ready_slot': int(r),
                         'forecast_sha256': hashlib.sha256(mean[d].tobytes()).hexdigest()}
                        for d, r in zip(self.sources, self.ready)]

    def take(self, slot):
        self.calls += 1
        ix = int(np.searchsorted(self.ready, slot, side='right')) - 1
        if ix <= self.last_job:
            return None
        self.last_job, self.last = ix, self.sources[ix]
        self.age = slot / 100 - self.last
        self.consumed.append((slot, self.last))
        return self.last


def world_policy(duration=3, n=2):
    world = build_world(12349, duration_s=duration, n_robots=n, sigma_map=2.)
    policy = Policy('fixture', 'native', duplication=False, roam_cost_s=4.,
        scan_period_s=1000., roam_gain_db=1e9)
    return world, policy


def test_fast_ticks_add_no_AP_plans_or_forecast_consumption():
    world, policy = world_policy()
    plans = []
    def scores(fc, current):
        plans.append(current.copy())
        return np.zeros((len(current), len(world.sites)))
    queue = KnownQueue(world.forecast)
    result = simulate(deepcopy(world), policy, forecast_queue=queue,
        roam_scorer=scores, fast_local=True)
    audit = result['fast_local_execution']
    assert len(plans) == len(queue.consumed) == 3
    assert queue.calls == 300  # one original consumption check per native slot
    assert len(audit['fast_control_updates']) > len(plans) * world.n
    assert any(not row['forecast_arrival'] and row['mode'] == 'predictive'
               for row in audit['fast_control_updates'])
    assert result['fast_local_timing']['calls'] == len(audit['fast_control_updates'])
    assert not any('elapsed' in key for row in audit['fast_control_updates'] for key in row)
    slices = audit['forecast_slices']
    assert len(slices) <= len(queue.consumed) * world.n
    assert len(slices) < len(audit['fast_control_updates'])
    for row in audit['fast_control_updates']:
        assert 'original_forecast_snr_db' not in row
        assert 'forecast_at_queries_snr_db' not in row
        assert 'route_innovation_db' not in row
        if row['mode'] == 'predictive':
            source = slices[row['forecast_slice_id']]
            assert source['source_decision'] == row['source_decision']
            assert source['robot'] == row['robot'] and source['current_site'] == row['current_site']
            np.testing.assert_array_equal(source['original_forecast_snr_db'],
                queue.mean[row['source_decision'], row['robot'], :, row['current_site']])


def test_sparse_cached_sources_never_read_uncomputed_rows():
    world, policy = world_policy(duration=4, n=1)
    world.forecast[1] = np.nan
    world.forecast[3] = np.nan
    queue = KnownQueue(world.forecast, sources=[0, 2], ready=[150, 350])
    result = simulate(world, policy, forecast_queue=queue, fast_local=True)
    sources = {row['source_decision'] for row in result['fast_local_execution']['fast_control_updates']
               if row['mode'] == 'predictive'}
    assert sources == {0, 2}
    assert queue.consumed == [(150, 0), (350, 2)]


def warm_observer(world, duration, observer_type=TimedDeliveredScans, **kwargs):
    value = np.broadcast_to(world.rssi_dl[0].reshape(1, world.n, -1),
        (60, world.n, len(world.sites) * 3)).copy()
    heard = value >= -90.
    warm = Observations(np.broadcast_to(world.pos[0], (60, world.n, 2)).copy(),
        np.where(heard, value, 0.).astype('f4'), heard,
        np.full(value.shape, -90., dtype='f4'), np.ones((60, world.n), dtype=bool))
    return observer_type(warm, duration, 12349, **kwargs)


@pytest.mark.parametrize('kind', ['native', 'legacy', 'mlo_default'])
def test_disabled_feature_preserves_all_predecessor_results(kind):
    world, base = world_policy(duration=2, n=3)
    policy = Policy('parity', kind, roam_cost_s=4.)
    old = previous_simulate(deepcopy(world), policy)
    new = simulate(deepcopy(world), policy)
    audit, timing = new.pop('fast_local_execution'), new.pop('fast_local_timing')
    assert not audit['fast_local_enabled'] and not audit['fast_control_updates']
    assert timing == {'calls': 0, 'total_seconds': 0., 'max_seconds': 0.}
    assert new == old


def test_disabled_feature_preserves_existing_final_scan_and_emergency_controller():
    world, _ = world_policy(duration=3, n=2)
    policy = Policy('finalparity', 'native', duplication=False, cadence_scan=True,
                    roam_cost_s=4., scan_period_s=5.)
    a, b = KnownQueue(world.forecast), KnownQueue(world.forecast)
    kwargs = dict(emergency_recovery=True, risk_gated_scans=True,
                  scan_duration_s=.09, blockage_rate=0.)
    old = previous_simulate(deepcopy(world), policy, forecast_queue=a,
        scan_observer=warm_observer(world, 3), **kwargs)
    new = simulate(deepcopy(world), policy, forecast_queue=b,
        scan_observer=warm_observer(world, 3), **kwargs)
    new.pop('fast_local_execution');new.pop('fast_local_timing')
    assert new == old


def test_offtick_kernel_handover_rejects_capture_inside_outage_after_refresh():
    world, _ = world_policy(duration=5, n=1)
    world.snr[:] = -100.
    world.snr[:, :, 0] = [50., 20., 10.]
    world.snr[:, :, 1] = [-30., -20., -10.]
    world.rssi_dl[:] = world.snr + world.noise_db + world.tx_dl - 15.
    policy = Policy('offtick_fixture', 'native', duplication=False,
        cadence_scan=True, scan_period_s=1000., roam_cost_s=4.03)
    plans = []
    def scores(fc, current):
        plans.append(current.copy())
        result = np.zeros((1, len(world.sites)))
        result[:, 1] = 10.
        return result
    queue = KnownQueue(world.forecast, sources=[0], ready=[20])
    result = simulate(world, policy, forecast_queue=queue,
        scan_observer=warm_observer(world, 5), roam_scorer=scores,
        fast_local=True, blockage_rate=0.)
    rows = {r['slot']: r for r in result['fast_local_execution']['fast_control_updates']}
    assert len(plans) == len(queue.consumed) == 1
    assert rows[20]['current_site'] == 1
    assert rows[20]['association_epoch_start'] == 20
    assert rows[20]['mode'] == 'hold' and rows[20]['pilot_site'] == 0
    # The outage ends off tick at 423. The 430 refresh carries a 420 capture,
    # which predates the last unavailable slot (422) and cannot authorize RF.
    first = rows[430]
    assert first['pilot_site'] == 1 and first['pilot_capture_slot'] == 420
    assert first['last_primary_unavailable'] == 422
    assert first['mode'] == 'hold'
    assert first['actual_control_band'] == first['control_band_before'] == 0
    assert first['native_delta_db_per_second'] == [0., 0., 0.]
    second = rows[440]
    assert second['pilot_capture_slot'] == 430 and second['mode'] == 'predictive'
    assert second['source_decision'] == 0


def test_observer_snapshots_existing_pilots_and_prepares_before_control_draw(monkeypatch):
    world, _ = world_policy(duration=2, n=2)
    policy = Policy('ordering_fixture', 'native', duplication=False,
        cadence_scan=True, scan_period_s=1000., roam_gain_db=1e9)
    class Hooks(TimedDeliveredScans):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prepared_slot = -1
            self.captures = []
            self.deliveries = []

        def update_pilots(self, slot, **kwargs):
            self.captures.append((slot, deepcopy(kwargs)))

        def prepare_frames(self, slot, **kwargs):
            assert slot == self.prepared_slot + 1
            assert kwargs['attempted'].shape == (world.n,)
            self.prepared_slot = slot

        def deliver(self, slot, delivered):
            assert self.prepared_slot == slot
            self.deliveries.append(slot)
            super().deliver(slot, delivered)

    observer = warm_observer(world, 2, Hooks)
    real_default_rng = np.random.default_rng
    control_seed = 1_000_003 * (world.seed + 1)
    class DrawOrder:
        def __init__(self, delegate):
            self.delegate, self.slot = delegate, 0

        def random(self, size):
            assert observer.prepared_slot == self.slot
            assert len(observer.deliveries) == self.slot
            self.slot += 1
            return self.delegate.random(size)

    draws = DrawOrder(real_default_rng(control_seed))
    monkeypatch.setattr(np.random, 'default_rng',
        lambda seed=None: draws if seed == control_seed else real_default_rng(seed))
    simulate(world, policy, forecast_queue=KnownQueue(world.forecast),
        scan_observer=observer, fast_local=True)
    assert observer.deliveries == list(range(200)) and draws.slot == 200
    assert [slot for slot, _ in observer.captures] == list(range(0, 200, 10))
    for slot, record in observer.captures:
        capture = max(0, slot - 10)
        assert record['capture_slot'] == capture
        np.testing.assert_array_equal(record['positions'],
            world.pos[min(capture // 20, len(world.pos) - 1)])
        assert record['refreshed'].dtype == np.bool_


def test_existing100ms_reflex_and_MCS_work_before_any_forecast_completion():
    world, _ = world_policy(duration=2, n=1)
    world.snr[:] = -100.
    world.snr[:, :, 0] = [40., -40., -40.]
    world.snr[2:, :, 0, 0] = 0.
    world.rssi_dl[:] = world.snr + world.noise_db + world.tx_dl - 15.
    policy = Policy('existing_reflex_fixture', 'native', duplication=False,
        cadence_scan=True, scan_period_s=1000., roam_gain_db=1e9)
    queue = KnownQueue(world.forecast, sources=[0], ready=[1000])
    result = simulate(world, policy, forecast_queue=queue,
        scan_observer=warm_observer(world, 2), blockage_rate=0.)
    assert queue.consumed == []
    actions = result['packet_detail']['action_trace']
    assert actions[0]['control_band'] == [0]
    assert len({r['mcs'][0][0] for r in actions}) > 1
    assert not result['fast_local_execution']['fast_control_updates']


def test_disabled_new_observer_and_fastloop_preserve_frozen_final_results():
    world, _ = world_policy(duration=3, n=2)
    policy = Policy('final_observer_parity', 'native', duplication=False,
        cadence_scan=True, roam_cost_s=4., scan_period_s=5.)
    old_observer = warm_observer(world, 3)
    new_observer = warm_observer(world, 3, PilotDeliveredScans, pilot_enabled=False)
    kwargs = dict(emergency_recovery=True, risk_gated_scans=True,
                  scan_duration_s=.09, blockage_rate=0.)
    old_queue, new_queue = KnownQueue(world.forecast), KnownQueue(world.forecast)
    old = previous_simulate(deepcopy(world), policy, forecast_queue=old_queue,
        scan_observer=old_observer, **kwargs)
    new = simulate(deepcopy(world), policy, forecast_queue=new_queue,
        scan_observer=new_observer, **kwargs)
    new.pop('fast_local_execution');new.pop('fast_local_timing')
    assert new == old and new_queue.consumed == old_queue.consumed
    assert new_observer.summary() == old_observer.summary()
    np.testing.assert_array_equal(new_observer.history.value, old_observer.history.value)
    np.testing.assert_array_equal(new_observer.history.heard, old_observer.history.heard)
    assert new_observer.pilot_summary()['enabled'] is False


@pytest.mark.parametrize('policy', [
    Policy('legacy', 'legacy', duplication=False),
    Policy('dup', 'native', duplication=True),
    Policy('dual', 'native', duplication=False, dual_radio=True),
])
def test_fast_mode_rejects_unregistered_radio_modes(policy):
    world, _ = world_policy(duration=1, n=1)
    with pytest.raises(ValueError, match='Fast local'):
        simulate(world, policy, fast_local=True)
