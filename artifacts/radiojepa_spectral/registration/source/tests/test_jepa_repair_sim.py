"""Native-PHY regression cases, not claims about warehouse performance."""
from dataclasses import replace
import numpy as np
import pytest
from wirelessjev.native.sim import Policy
from wirelessjev.native.world import CLIENT_TX_DBM, build_world
from wirelessjev.fleetwm.data import Observations
from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.semantic_jepa.sim import simulate as frozen_simulate
from wirelessjev.jepa_repair.sim import simulate


@pytest.mark.parametrize('observer', [None, object()])
def test_emergency_recovery_requires_observable_local_scans_before_world_access(observer):
    with pytest.raises(TypeError, match='observable local scans'):
        simulate(object(), Policy('no_oracle', 'native', cadence_scan=True),
                 scan_observer=observer, emergency_recovery=True)


@pytest.mark.parametrize('kind', ['native', 'legacy', 'mlo_default'])
def test_opt_out_preserves_every_predecessor_field(kind):
    world = build_world(12349, duration_s=2, n_robots=3, sigma_map=2.)
    policy = Policy('parity', kind, roam_cost_s=.03)
    previous = frozen_simulate(world, policy)
    actual = simulate(world, policy)
    audit = actual.pop('repair_execution')
    assert not audit['emergency_recovery_enabled']
    assert not audit['emergency_handovers'] and not audit['emergency_scans']
    assert previous == actual


def failed_current_link_world():
    world = build_world(12347, duration_s=10, n_robots=1, sigma_map=0.)
    world.snr[:] = -200.
    world.snr[:, :, 0] = 40.
    world.snr[:, :, 1] = 20.
    world.snr[4:, :, 0] = -100.
    world.rssi_dl[:] = world.snr + world.noise_db + world.tx_dl - CLIENT_TX_DBM
    world.forecast[:] = -200.
    world.forecast[:, :, :, 0] = 25.
    world.forecast[:, :, :, 1] = 35.
    world.bulk_mbps[:] = 0.
    world.video_mbps[:] = 0.
    return world


def history(world, know_neighbor):
    value = np.broadcast_to(world.rssi_dl[0].reshape(1, 1, -1), (60, 1, world.snr.shape[2] * 3)).copy()
    heard = value >= -90.
    if not know_neighbor:
        heard[:, :, 3:6] = False
    value = np.where(heard, value, 0.).astype('f4')
    return Observations(np.broadcast_to(world.pos[0], (60, 1, 2)).copy(), value,
                        heard, np.full(value.shape, -90., dtype='f4'), np.ones((60, 1), dtype=bool))


@pytest.mark.parametrize('cost', [.03, 4.])
def test_local_recovery_runs_while_all_cloud_forecasts_remain_pending(cost):
    from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
    world = failed_current_link_world()
    policy = Policy('cloud_stall', 'native', gate=True, roam_cost_s=cost,
                    cadence_scan=True, scan_period_s=5., cadence_roam=True,
                    duplication=False)
    def run(repair):
        scans = TimedDeliveredScans(history(world, True), 10, world.seed)
        queue = ForecastQueue([30.] * 10)
        return simulate(world, policy, scan_observer=scans, forecast_queue=queue,
                        emergency_recovery=repair, scan_duration_s=.09, blockage_rate=0.)
    old, repair = run(False), run(True)
    assert old['control_sent'] == repair['control_sent'] == 200
    assert old['roams'] == 0 and repair['roams'] >= 1
    assert repair['control_misses'] < old['control_misses']
    event = repair['repair_execution']['emergency_handovers'][0]
    assert event['silence_slots'] >= 20 and event['from_site'] == 0 and event['to_site'] == 1
    assert event['outage_until_slot'] - event['slot'] == round(cost * 100)
    assert event['neighbor_rssi_dbm'] >= -90.
    assert event['neighbor_heard'] is True
    assert event['selected_band'] in (0, 1, 2)
    assert event['neighbor_capture_age_slots'] == event['slot'] - event['neighbor_capture_slot']
    assert 0 <= event['neighbor_capture_age_slots'] <= event['neighbor_max_age_slots'] == 500
    assert event['pending_cadence_target_after'] == -1
    for check in repair['repair_execution']['watchdog_checks']:
        if check['best_other_site'] is None:
            assert check['selected_band'] is None and check['neighbor_heard'] is False
            assert check['neighbor_capture_slot'] is None and check['neighbor_capture_age_slots'] is None
        else:
            assert check['best_other_site'] != check['current_site']
            assert check['neighbor_heard'] is True and check['selected_band'] in (0, 1, 2)
            assert check['neighbor_capture_age_slots'] == check['slot'] - check['neighbor_capture_slot']
            assert 0 <= check['neighbor_capture_age_slots'] <= check['neighbor_max_age_slots']


def test_due_local_scan_can_break_delivery_dependency_without_invented_neighbors():
    from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
    world = failed_current_link_world()
    policy = Policy('unknown_neighbor', 'native', gate=True, roam_cost_s=.03,
                    cadence_scan=True, scan_period_s=5., cadence_roam=True,
                    duplication=False)
    scans = TimedDeliveredScans(history(world, False), 10, world.seed)
    result = simulate(world, policy, scan_observer=scans,
        forecast_queue=ForecastQueue([30.] * 10), emergency_recovery=True,
        scan_duration_s=.09, blockage_rate=0.)
    audit = result['repair_execution']
    assert audit['emergency_scans'] and audit['emergency_handovers']
    scan, move = audit['emergency_scans'][0], audit['emergency_handovers'][0]
    assert scan['silence_slots'] >= 20
    assert scan['scan_complete_slot'] - scan['slot'] == 10  # inclusive slot + actual90ms
    assert move['slot'] >= scan['scan_complete_slot']
    assert move['neighbor_rssi_dbm'] >= -90.
    assert move['neighbor_heard'] is True and move['neighbor_capture_slot'] >= scan['slot']
    assert move['neighbor_capture_age_slots'] == move['slot'] - move['neighbor_capture_slot']
    assert move['pending_cadence_target_after'] == -1
    assert any(row['delivery_slot'] >= scan['scan_complete_slot'] for row in scans.events)
