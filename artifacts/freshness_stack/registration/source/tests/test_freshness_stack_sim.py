"""Physical unit counterexamples for fresh evidence and real whole-scan costs."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
from wirelessjev.native.world import build_world, CLIENT_TX_DBM, HORIZONS
from wirelessjev.native.sim import Policy
from wirelessjev.fleetwm.data import Observations
from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
from wirelessjev.costaware_recovery.sim import simulate as old_simulate
from wirelessjev.freshness_stack.sim import simulate, scan_suppression_evidence


def evidence(*, source=0, slot=100, forecast=None, pilot=-60., current=0, **kwargs):
    if forecast is None:
        forecast = np.full((1, 1, 9, 2, 3), 40., dtype='f4')
    q = SimpleNamespace(last=source, mean=forecast,
        records=[{'decision': 0, 'ready_slot': 1, 'forecast_sha256': 'a' * 64}])
    noise = np.full(3, -90.)
    tx = np.full(3, 20.)
    snr = np.full(3, pilot) - noise - tx + CLIENT_TX_DBM
    meta = dict(pilot_site=current, pilot_capture_slot=max(0, slot - 10), pilot_refresh_slot=slot)
    meta.update(kwargs)
    return scan_suppression_evidence(q, slot, 0, current, snr, noise, tx, 0., True, **meta)


def test_suppression_requires_genuine_current_pilot_and_consumed_completed_forecast():
    row = evidence()
    assert row['suppress_scan'] and row['reason'] == 'strong_current_and_predicted_link'
    assert row['forecast_age_seconds'] == 1. and row['forecast_ready_slot'] == 1
    assert np.shape(row['original_forecast_snr_db']) == (9, 3)
    np.testing.assert_array_equal(row['route_innovation_db'], np.zeros((6, 3)))
    assert not evidence(source=-1)['suppress_scan']
    assert not evidence(slot=0)['suppress_scan']  # job has not completed
    assert not evidence(slot=501)['suppress_scan']
    assert not evidence(pilot=-68.)['suppress_scan']


def test_age_shift_occurs_before_local_pilot_anchoring():
    forecast = np.full((1, 1, 9, 2, 3), 40., dtype='f4')
    forecast[:, :, 6:] = 10.  # h10..30; a0..5s unshifted query misses this drop
    row = evidence(slot=400, forecast=forecast)
    assert not row['suppress_scan'] and row['reason'] == 'predicted_link_risk'
    shifted = np.interp(np.arange(6) + 4., HORIZONS, forecast[0, 0, :, 0, 0])
    np.testing.assert_allclose(np.array(row['forecast_at_age_plus_samples_snr_db'])[:, 0], shifted)
    assert row['predicted_best_worst_rssi_dbm'] == pytest.approx(-60. + shifted.min() - shifted[0])


@pytest.mark.parametrize('kwargs', [
    {'pilot_site': 1}, {'pilot_capture_slot': 80}, {'pilot_refresh_slot': 90},
    {'pilot_capture_slot': 101}, {'pilot_refresh_slot': 101},
])
def test_stale_wrong_association_and_future_pilots_never_suppress(kwargs):
    row = evidence(**kwargs)
    assert not row['suppress_scan'] and row['reason'] == 'stale_or_wrong_association_pilot'


def test_off_tick_association_change_cannot_reuse_old_strong_pilot():
    # A4s outage can end between100ms measurement ticks. Old-site pilot
    # remains strong, but it cannot describe the newly associated AP.
    old = evidence(slot=105, current=1, pilot_site=0, pilot_refresh_slot=100, pilot_capture_slot=90)
    assert not old['suppress_scan'] and old['reason'] == 'stale_or_wrong_association_pilot'
    refreshed = evidence(slot=110, current=1, pilot_site=1, pilot_refresh_slot=110, pilot_capture_slot=100)
    assert refreshed['suppress_scan']


@pytest.mark.parametrize('kind', ['native', 'legacy', 'mlo_default'])
def test_disabled_scan_option_preserves_every_predecessor_field(kind):
    world = build_world(12349, duration_s=2, n_robots=3, sigma_map=2.)
    policy = Policy('parity', kind, roam_cost_s=4.)
    old = old_simulate(deepcopy(world), policy)
    new = simulate(deepcopy(world), policy)
    extra = new.pop('freshness_execution')
    assert not extra['risk_gated_scans_enabled'] and not extra['scan_assessments']
    assert new == old


class KnownQueue(ForecastQueue):
    def __init__(self, mean):
        super().__init__([0.] * len(mean))
        self.mean = mean
        self.records = [{'decision': i, 'ready_slot': i * 100,
            'forecast_sha256': hashlib.sha256(mean[i].tobytes()).hexdigest()} for i in range(len(mean))]


def fixture(*, fade_after=None, risk=True):
    world = build_world(12347, duration_s=8, n_robots=1, sigma_map=0.)
    world.snr[:] = -100.
    world.snr[:, :, 0] = 40.
    world.snr[:, :, 1] = 20.
    if fade_after is not None:
        world.snr[int(np.ceil(fade_after / .2)):, :, 0] = -100.
    world.rssi_dl[:] = world.snr + world.noise_db + world.tx_dl - CLIENT_TX_DBM
    world.forecast[:] = 40.
    world.bulk_mbps[:] = world.video_mbps[:] = 0.
    value = np.broadcast_to(world.rssi_dl[0].reshape(1, 1, -1), (60, 1, len(world.sites) * 3)).copy()
    heard = value >= -90.
    history = Observations(np.broadcast_to(world.pos[0], (60, 1, 2)).copy(),
        np.where(heard, value, 0.).astype('f4'), heard,
        np.full(value.shape, -90., dtype='f4'), np.ones((60, 1), dtype=bool))
    scans = TimedDeliveredScans(history, 8, 12348)  # fixed non-missing receiver fixture
    policy = Policy('riskfixture', 'native', planned_roam=False, roam_cost_s=4.,
        cadence_scan=True, scan_period_s=5., duplication=False)
    result = simulate(world, policy, scan_observer=scans, forecast_queue=KnownQueue(world.forecast),
        emergency_recovery=True, risk_gated_scans=risk, scan_duration_s=.09, blockage_rate=0.)
    return result, scans


def test_strong_known_link_postpones_real_scans_without_forging_observations():
    periodic, original_scans = fixture(risk=False)
    adaptive, scans = fixture(risk=True)
    audit = adaptive['freshness_execution']
    suppressed = [e for e in audit['scan_assessments'] if e['suppress_scan']]
    assert suppressed and not audit['scan_starts'] and not scans.local_completion_events
    assert original_scans.local_completion_events and periodic['freshness_execution']['scan_starts']
    assert not scans.events  # no fake central reports from skipped captures
    assert adaptive['control_misses'] <= periodic['control_misses']


def test_due_postponement_does_not_block_emergency_scan_during_control_silence():
    healthy, _ = fixture()
    first = next(e for e in healthy['freshness_execution']['scan_assessments'] if e['suppress_scan'])
    fade_after = (first['slot'] + 20) / 100
    bad, scans = fixture(fade_after=fade_after)
    audit = bad['freshness_execution']
    emergencies = [e for e in audit['scan_starts'] if e['emergency_silence']]
    assert emergencies and emergencies[0]['slot'] < first['slot'] + 500
    assert emergencies[0]['complete_slot'] - emergencies[0]['slot'] == 10
    assert any(e['measurement_slot'] == emergencies[0]['slot'] for e in scans.local_completion_events)
    assert any(e['reason'] == 'emergency_silence_bypass' and not e['suppress_scan'] for e in audit['scan_assessments'])


@pytest.mark.parametrize('bad', [['a', 'b', 'c'], [object(), object(), object()], [[1.], [2.], [3.]], [1., np.nan, 3.]])
def test_malformed_public_pilot_permits_scan_instead_of_crashing(bad):
    row = scan_suppression_evidence(None, 100, 0, 0, bad, [-90.] * 3, [20.] * 3,
        0., True, pilot_site=0, pilot_capture_slot=90, pilot_refresh_slot=100)
    assert not row['suppress_scan'] and row['reason'] == 'invalid_pilot'


@pytest.mark.parametrize('dtype', ['U2', 'O'])
def test_malformed_forecast_permits_original_scan(dtype):
    forecast = np.full((1, 1, 9, 2, 3), 'x', dtype=dtype)
    assert evidence(forecast=forecast)['reason'] == 'invalid_forecast_values'


def test_one_band_must_remain_strong_for_the_entire_sampled_window():
    values = np.full((1, 1, 9, 1, 3), 40., dtype='f4')
    for horizon, band in [(2, 0), (3, 1), (4, 2)]:
        values[0, 0, horizon, 0, band] = 20.
    row = evidence(forecast=values)
    future = np.array(row['current_rssi_dbm'])[None] + np.array(row['route_innovation_db'])
    assert future.max(1).min() == -60.
    assert future.min(0).max() == -80.
    assert not row['suppress_scan'] and row['predicted_best_worst_rssi_dbm'] == -80.


def test_fifo_emergency_kernel_retains_every_existing_recovery_field():
    world = build_world(12349, duration_s=2, n_robots=3, sigma_map=2.)
    value = np.broadcast_to(world.rssi_dl[0].reshape(1, 3, -1), (60, 3, len(world.sites) * 3)).copy()
    heard = value >= -90.
    warm = Observations(np.broadcast_to(world.pos[0], (60, 3, 2)).copy(),
        np.where(heard, value, 0.).astype('f4'), heard,
        np.full(value.shape, -90., dtype='f4'), np.ones((60, 3), dtype=bool))
    policy = Policy('parity', 'native', cadence_scan=True, roam_cost_s=4.)
    kwargs = dict(emergency_recovery=True, scan_duration_s=.09, blockage_rate=0.)
    old = old_simulate(deepcopy(world), policy,
        scan_observer=TimedDeliveredScans(warm, 2, 12349), **kwargs)
    new = simulate(deepcopy(world), policy,
        scan_observer=TimedDeliveredScans(warm, 2, 12349), **kwargs)
    new.pop('freshness_execution')
    assert new == old
