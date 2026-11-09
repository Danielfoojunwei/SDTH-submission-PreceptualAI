"""A native-PHY counterexample for the rejected-plan recovery boundary.

This constructed software fixture is not a warehouse benchmark. Both policies
receive the same stale forecast, real link curves, channel draws and traffic.
"""
import pytest

from wirelessjev.frontier.timing import ForecastQueue
from wirelessjev.gate_repair.gate import reactive_rescue
from wirelessjev.jepa_repair.runtime import ExactOutageAdmission
from wirelessjev.native.sim import Policy
from wirelessjev.native.world import CLIENT_TX_DBM, build_world
from wirelessjev.semantic_jepa.sim import simulate


@pytest.mark.parametrize('cost', [.03, 4.])
def test_rejected_plan_allows_observable_native_recovery(cost):
    world = build_world(12347, duration_s=7, n_robots=1, sigma_map=0.)
    world.snr[:] = -200.
    world.snr[:, :, 0] = 40.
    world.snr[:, :, 1] = 20.
    # The initially preferred AP becomes unreachable after 800 ms. Neighbor
    # measurements show the second AP remains usable; the stale map does not.
    world.snr[4:, :, 0] = -100.
    world.rssi_dl[:] = (world.snr + world.noise_db
                        + world.tx_dl - CLIENT_TX_DBM)
    world.forecast[:] = -200.
    world.forecast[:, :, :, 0] = 25.
    world.forecast[:, :, :, 1] = 35.
    world.bulk_mbps[:] = 0.
    world.video_mbps[:] = 0.
    policy = Policy('recovery_fixture', 'native', gate=True,
                    roam_cost_s=cost, cadence_roam=False, duplication=False)
    traces = [[], []]

    def run(repaired):
        admission = ExactOutageAdmission(world.forecast, ForecastQueue([0.] * 7), cost, 30.)
        def record(ctx, confident, after):
            traces[int(repaired)].append((ctx.decision, ctx.eligible.copy(),
                                          confident.copy(), after.copy()))
        if repaired:
            return simulate(world, policy, roam_gate=admission,
                            decision_trace=record, blockage_rate=0.)
        return simulate(world, policy, roam_gate=reactive_rescue,
                        handover_admission=admission, decision_trace=record,
                        blockage_rate=0.)

    old, repaired = run(False), run(True)
    assert old['control_sent'] == repaired['control_sent'] == 140
    assert old['roams'] == 0 and repaired['roams'] == 1
    assert repaired['control_misses'] < old['control_misses']
    assert any(eligible[0] and confidence[0] and after[0] == 0
               for _, eligible, confidence, after in traces[0])
    assert any(eligible[0] and not confidence[0] and after[0] == 1
               for _, eligible, confidence, after in traces[1])
    actions = repaired['packet_detail']['action_trace']
    first = next(row for row in actions if row['primary_site'][0] == 1)
    # Recovery pays the actual configured outage; no instantaneous switch or
    # changed packet accounting is used to make the repair appear beneficial.
    assert (first['primary_unavailable_until_slot'][0]
            - (first['slot_end'] - 10)) == round(cost * 100)
