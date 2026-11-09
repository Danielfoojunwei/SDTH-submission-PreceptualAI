"""Native-PHY software counterexamples; no prospective site is opened."""
import math
import numpy as np
import pytest
from wirelessjev.native.sim import Policy
from wirelessjev.native.world import CLIENT_TX_DBM, build_world
from wirelessjev.fleetwm.data import Observations
from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.jepa_repair.sim import simulate as parent_simulate
from wirelessjev.costaware_recovery.sim import simulate
from wirelessjev.costaware_recovery.telemetry import TimedDeliveredScans


@pytest.mark.parametrize('kind', ['native', 'legacy', 'mlo_default'])
def test_disabled_kernel_preserves_parent_fields_exactly(kind):
    world = build_world(12349, duration_s=2, n_robots=3, sigma_map=2.)
    policy = Policy('parity', kind, roam_cost_s=4.)
    assert simulate(world, policy) == parent_simulate(world, policy)


@pytest.mark.parametrize('observer', [None, object()])
def test_emergency_refuses_missing_observable_interface_before_world(observer):
    with pytest.raises(TypeError, match='observable local scans'):
        simulate(object(), Policy('missing', 'native', cadence_scan=True),
                 scan_observer=observer, emergency_recovery=True)


def world_and_history(fade_seconds=None, know_neighbor=False):
    world = build_world(12347, duration_s=12, n_robots=1, sigma_map=0.)
    world.snr[:] = -200.
    world.snr[:, :, 0] = 40.
    world.snr[:, :, 1] = 20.
    end = len(world.snr) if fade_seconds is None else 4 + math.ceil(fade_seconds / .2)
    world.snr[4:end, :, 0] = -100.
    world.rssi_dl[:] = world.snr + world.noise_db + world.tx_dl - CLIENT_TX_DBM
    world.forecast[:] = -200.
    world.forecast[:, :, :, 0] = 25.
    world.forecast[:, :, :, 1] = 35.
    world.bulk_mbps[:] = world.video_mbps[:] = 0.
    return world, warm_history(world, know_neighbor)


def warm_history(world, know_neighbor):
    value = np.broadcast_to(world.rssi_dl[0].reshape(1, 1, -1),
                            (60, 1, world.snr.shape[2] * 3)).copy()
    heard = value >= -90.
    if not know_neighbor:
        heard[:, :, 3:6] = False
    value = np.where(heard, value, 0.).astype('f4')
    history = Observations(np.broadcast_to(world.pos[0], (60, 1, 2)).copy(), value,
        heard, np.full(value.shape, -90., dtype='f4'), np.ones((60, 1), dtype=bool))
    return history


def run(fade_seconds=None, repair=True):
    world, history = world_and_history(fade_seconds)
    scans = TimedDeliveredScans(history, 12, world.seed)
    policy = Policy('cloud_stall', 'native', gate=True, roam_cost_s=4.,
                    cadence_scan=True, scan_period_s=5., cadence_roam=True,
                    duplication=False)
    result = simulate(world, policy, scan_observer=scans,
        forecast_queue=ForecastQueue([30.] * 12), emergency_recovery=repair,
        scan_duration_s=.09, blockage_rate=0.)
    return result, scans


def test_sustained_dead_link_recovers_only_after_cost_derived_wait_and_pays_outage():
    old, _ = run(repair=False)
    actual, scans = run()
    audit = actual['repair_execution']
    assert actual['control_sent'] == old['control_sent'] == 240
    assert old['roams'] == 0 and actual['control_misses'] < old['control_misses']
    assert audit['watchdog_threshold_slots'] == 20
    assert audit['handover_threshold_slots'] == 420
    assert audit['emergency_scans'] and audit['emergency_handovers']
    for scan in audit['emergency_scans']:
        assert scan['silence_slots'] >= 20
        assert scan['scan_complete_slot'] - scan['slot'] == 10
    event = audit['emergency_handovers'][0]
    assert event['silence_slots'] >= event['available_silence_slots'] >= event['handover_threshold_slots'] == 420
    assert event['available_silence_slots_after'] == 0
    assert event['outage_until_slot'] - event['slot'] == 400
    assert event['pending_cadence_target_after'] == -1
    assert event['neighbor_heard'] is True and event['neighbor_rssi_dbm'] >= -90.
    assert event['neighbor_capture_age_slots'] == event['slot'] - event['neighbor_capture_slot']
    assert 0 <= event['neighbor_capture_age_slots'] <= 500
    assert any(e['measurement_slot'] == event['neighbor_capture_slot']
               and e['complete_slot'] <= event['slot'] for e in scans.local_completion_events)
    # Observation is permitted before the expensive move becomes eligible.
    assert any(not c['recovery_triggered'] and c['available_silence_slots'] < 420
               for c in audit['watchdog_checks'])
    during = [row for row in actual['packet_detail']['cumulative']
              if event['slot'] < row['slot_end'] <= event['outage_until_slot']]
    assert during and all(row['available_silence_slots'] == [0] for row in during)


@pytest.mark.parametrize('fade_seconds', [.5, 2.])
def test_short_fades_do_not_trigger_four_second_watchdog_handover(fade_seconds):
    actual, _ = run(fade_seconds=fade_seconds)
    audit = actual['repair_execution']
    assert actual['control_sent'] == 240
    assert actual['roams'] == 0 and not audit['emergency_handovers']
    assert audit['handover_threshold_slots'] == 420
    assert audit['watchdog_checks']
    assert all(not c['recovery_triggered'] for c in audit['watchdog_checks'])
    assert max(actual['packet_detail']['max_gap']) < 420


def test_runtime_and_telemetry_aliases_are_the_frozen_classes():
    from wirelessjev.costaware_recovery import runtime, telemetry
    from wirelessjev.jepa_repair import runtime as previous, telemetry as old_telemetry
    assert runtime.CompletionSafeForecasts is previous.CompletionSafeForecasts
    assert runtime.RecordedScheduleReplay is previous.RecordedScheduleReplay
    assert telemetry.TimedDeliveredScans is old_telemetry.TimedDeliveredScans


def test_own_handover_outage_cannot_immediately_chain_another_expensive_move():
    world, _ = world_and_history()
    # AP1 initially looks best, then becomes truly unusable while our4s
    # handover is underway. AP2 stays reachable. This makes a second move
    # tempting after completion if the first outage is miscounted as evidence.
    world.snr[:, :, 1] = 30.
    world.snr[30:, :, 1] = -100.  #6s, after first move but before completion
    world.snr[:, :, 2] = 20.
    world.rssi_dl[:] = world.snr + world.noise_db + world.tx_dl - CLIENT_TX_DBM
    scans = TimedDeliveredScans(warm_history(world, True), 12, world.seed)
    policy = Policy('no_chain', 'native', gate=True, roam_cost_s=4.,
                    cadence_scan=True, scan_period_s=5., cadence_roam=True,
                    duplication=False)
    actual = simulate(world, policy, scan_observer=scans,
        forecast_queue=ForecastQueue([30.] * 12), emergency_recovery=True,
        scan_duration_s=.09, blockage_rate=0.)
    events = actual['repair_execution']['emergency_handovers']
    assert len(events) == actual['roams'] == 1
    first = events[0]
    assert first['to_site'] == 1 and first['available_silence_slots_after'] == 0
    post = [c for c in actual['repair_execution']['watchdog_checks']
            if first['outage_until_slot'] <= c['slot'] < first['outage_until_slot'] +420]
    assert post and any(c['best_other_site'] == 2 for c in post)
    assert all(c['available_silence_slots'] <420 and not c['recovery_triggered'] for c in post)
    assert post[0]['silence_slots'] >=400
    assert post[0]['available_silence_slots'] ==0
