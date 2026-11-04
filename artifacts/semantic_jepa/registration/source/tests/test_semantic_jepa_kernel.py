"""Actual native simulations verify audit exports without changing service."""
from dataclasses import replace

import numpy as np
import pytest

from wirelessjev.closed_loop.sim import simulate as predecessor
from wirelessjev.contracted.data import runtime_world
from wirelessjev.fleetwm.data import episode
from wirelessjev.frontier.study import mask_scans
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.closed_loop.runtime import LiveForecasts, SinglePredictor
from wirelessjev.gate_repair.gate import reactive_rescue
from wirelessjev.native.sim import policies
from wirelessjev.native.world import build_world
from wirelessjev.semantic_jepa.sim import simulate


AUDIT_FIELDS = {'packet_detail', 'packet_accounting'}


def without_audit(result):
    return {key: value for key, value in result.items() if key not in AUDIT_FIELDS}


@pytest.mark.parametrize('name,gate', [
    ('native', False), ('native', True), ('legacy', False),
    ('mlo_default', False), ('mlo_split', False),
])
def test_every_predecessor_field_is_exactly_preserved(name, gate):
    world = build_world(12345, duration_s=3, n_robots=4, sigma_map=4.)
    policy = replace({p.name: p for p in policies()}[name], gate=gate)
    expected = predecessor(world, policy)
    actual = simulate(world, policy)
    assert set(actual) - set(expected) == AUDIT_FIELDS
    assert without_audit(actual) == expected


def test_integer_cumulative_counts_and_right_censored_fractional_endpoint():
    world = build_world(12345, duration_s=2, n_robots=10, sigma_map=4.)
    # An unreachable real-PHY fixture leaves pending commands. They must not
    # be flushed or counted as factual misses merely to complete accounting.
    world.snr[:] = -200.
    world.forecast[:] = -200.
    policy = replace({p.name: p for p in policies()}['native'],
                     route_forecast=False, planned_roam=False, reflex=False)
    result = simulate(world, policy, duration_s=1.037)
    assert without_audit(result) == predecessor(world, policy, duration_s=1.037)
    detail, account = result['packet_detail'], result['packet_accounting']
    assert account['audited_slots'] == 104
    assert [row['slot_end'] for row in detail['cumulative']] == [100, 104]
    for name in ('misses', 'sent', 'max_gap', 'pending_control_packet_age_slots'):
        assert all(type(value) is int for value in detail[name])
        assert detail['cumulative'][-1][name] == detail[name]
    assert sum(detail['misses']) == result['control_misses']
    assert sum(detail['sent']) == result['control_sent']
    pending = np.asarray(detail['pending_control_packet_age_slots']) >= 0
    assert pending.any()
    assert detail['pending_control_packets'] == int(pending.sum())
    assert detail['pending_control_right_censored'] is True
    assert account['pending_control_right_censored'] is True
    assert np.all(np.asarray(detail['misses']) + pending <= detail['sent'])
    assert np.all(np.asarray(detail['pending_control_packet_age_slots']) < 5)
    for name in ('misses', 'sent', 'max_gap'):
        assert np.all(np.diff([row[name] for row in detail['cumulative']], axis=0) >= 0)


def test_bulk_and_video_audit_use_real_queue_flows_and_original_demands():
    world = build_world(12345, duration_s=21, n_robots=2, sigma_map=4.)
    world.snr[:] = -200.
    world.forecast[:] = -200.
    # Float32 arrivals expose an incorrect duration * Mbps recomputation.
    world.bulk_mbps = np.array([.1234567, .7654321], dtype=np.float32)
    world.video_mbps[:] = 2.
    policy = replace({p.name: p for p in policies()}['native'],
                     route_forecast=False, planned_roam=False, reflex=False,
                     adaptive_video=True)
    result = simulate(world, policy)
    assert without_audit(result) == predecessor(world, policy)
    account = result['packet_accounting']
    offered = np.asarray(world.bulk_mbps * 1e6 * .01, dtype=np.float64) * 2100
    np.testing.assert_array_equal(account['per_robot_bulk_generated_bits'], offered)
    assert account['bulk_generated_bits'] == float(offered.sum())
    assert account['bulk_overflow_drop_bits'] > 0
    assert account['bulk_delivered_bits'] == 0
    assert account['bulk_overflow_inferred_from_mass_balance'] is True
    np.testing.assert_allclose(
        account['bulk_generated_bits'],
        account['bulk_delivered_bits'] + account['final_bulk_backlog_bits']
        + account['bulk_overflow_drop_bits'], rtol=1e-12, atol=1e-6)
    np.testing.assert_equal(account['per_robot_final_bulk_backlog_bits'],
                            result['packet_detail']['per_robot_final_bulk_backlog_bits'])
    assert account['video_requested_bits'] >= account['video_generated_bits'] >= account['video_delivered_bits']
    assert account['video_requested_bits'] > account['video_generated_bits']
    assert account['video_expiry_measured_from_queue'] is True
    np.testing.assert_allclose(
        account['video_generated_bits'],
        account['video_delivered_bits'] + account['video_pending_bits']
        + account['video_expired_bits'], rtol=1e-11, atol=1e-6)
    assert abs(account['video_flow_conservation_error_bits']) < 1e-5


def test_sampled_actions_are_actual_end_of_slot_state_and_replay_exactly():
    world = build_world(12345, duration_s=2, n_robots=4, sigma_map=4.)
    policy = {p.name: p for p in policies()}['native']
    result = simulate(world, policy, duration_s=1.037)
    assert result == simulate(world, policy, duration_s=1.037)
    detail = result['packet_detail']
    trace = detail['action_trace']
    assert [row['slot_end'] for row in trace] == [*range(10, 101, 10), 104]
    assert detail['action_trace_phase'] == 'end_of_slot_state_not_complete_event_log'
    assert detail['action_trace_period_slots'] == 10
    for row in trace:
        assert len(row['primary_site']) == len(row['control_band']) == world.n
        assert np.all((np.asarray(row['primary_site']) >= 0) &
                      (np.asarray(row['primary_site']) < len(world.sites)))
        assert np.all((np.asarray(row['control_band']) >= 0) &
                      (np.asarray(row['control_band']) < 3))
        assert np.shape(row['mcs']) == (world.n, 3)


def test_decision_hook_arguments_and_native_actions_remain_unchanged():
    world = build_world(12345, duration_s=3, n_robots=4, sigma_map=4.)
    policy = replace({p.name: p for p in policies()}['native'], gate=True)
    logs = [[], []]

    def recorder(destination):
        def record(decision, confidence, after_site):
            destination.append({
                'decision': decision.decision, 'current': decision.current.tolist(),
                'candidate': decision.candidate.tolist(), 'gain': decision.gain.tolist(),
                'eligible': decision.eligible.tolist(), 'confidence': confidence.tolist(),
                'after_site': after_site.tolist(),
            })
        return record

    old = predecessor(world, policy, decision_trace=recorder(logs[0]),
                      roam_gate=reactive_rescue)
    new = simulate(world, policy, decision_trace=recorder(logs[1]),
                   roam_gate=reactive_rescue)
    assert logs[0] == logs[1] and len(logs[0]) == 3
    assert without_audit(new) == old


def test_real_radiomap_delivered_forecasts_preserve_all_predecessor_results():
    import torch
    torch.set_num_threads(1)
    full, _, geometry, observations = episode(12346, 4, 3)
    observations = mask_scans(observations, 5)
    policy = replace({p.name: p for p in policies()}['native'],
                     cadence_scan=True, scan_period_s=1., gate=True)

    def run(kernel):
        history = DeliveredScans(observations.before(60), 3, 12346)
        forecasts = LiveForecasts(SinglePredictor(geometry, 'radiomap'), history, 3,
                                 {'cold': .2, 'warm': .2, 'refit': .2}, 0.)
        world = runtime_world(full, 3, forecasts.mean, forecasts.std[:, :, 0])
        result = kernel(world, policy, scan_observer=history,
                        forecast_source=forecasts, forecast_queue=forecasts,
                        roam_gate=reactive_rescue, scan_duration_s=.09)
        return result, history.summary(), forecasts

    old, old_history, old_forecasts = run(predecessor)
    new, new_history, new_forecasts = run(simulate)
    assert without_audit(new) == old
    assert new_history == old_history
    np.testing.assert_array_equal(new_forecasts.mean, old_forecasts.mean)
    np.testing.assert_array_equal(new_forecasts.std, old_forecasts.std)
    assert [row['ready_slot'] for row in new_forecasts.records] == [20, 120, 220]
