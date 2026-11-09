from dataclasses import replace
import numpy as np
import pytest
from wirelessjev.packet_control import ShadowPacketAdmission,require_confirmed_packet_profile
from wirelessjev.native.gate import RoamDecision
from wirelessjev.native.world import build_world
from wirelessjev.native.sim import policies
from wirelessjev.closed_loop.sim import simulate
from wirelessjev.gate_repair.gate import reactive_rescue


def test_uncertain_rejecting_and_failed_critics_retain_the_baseline_action():
    c=RoamDecision.capture(1,np.array([0,1]),np.array([1,0]),np.array([6.,7.]),
                           np.array([True,True]),np.zeros((2,2)))
    def failure(context):raise ValueError('unavailable estimate')
    for critic in [lambda c:None,lambda c:np.zeros(2,dtype=bool),failure]:
        gate=ShadowPacketAdmission(lambda c:np.array([True,False]),critic)
        try:
            np.testing.assert_equal(gate(c),[True,False])
            assert gate.observations[-1]['active_override'] is False
        finally:gate.close()


def test_shadow_critic_preserves_native_packet_outcomes_exactly():
    w=build_world(299047,duration_s=6,n_robots=16,
                  site_params={'rack_scale':1.6,'pl_coef':38.,'shadow_db':7.})
    p=replace({p.name:p for p in policies()}['native'],gate=True)
    baseline=simulate(w,p,roam_gate=reactive_rescue)
    shadow=ShadowPacketAdmission(reactive_rescue,lambda c:np.zeros(w.n,dtype=bool))
    try:
        assert simulate(w,p,roam_gate=shadow)==baseline
        assert shadow.observations
    finally:shadow.close()


def test_busy_shadow_worker_does_not_wait_or_accumulate_admission_jobs():
    import threading
    entered=threading.Event();release=threading.Event()
    c=RoamDecision.capture(1,np.array([0]),np.array([1]),np.array([6.]),np.array([True]),np.zeros((1,2)))
    def waiting(context):
        entered.set();release.wait(2);return np.zeros(1,dtype=bool)
    gate=ShadowPacketAdmission(lambda c:np.ones(1,dtype=bool),waiting)
    try:
        np.testing.assert_equal(gate(c),[True]);assert entered.wait(1)
        np.testing.assert_equal(gate(c),[True])
        assert not release.is_set() and gate.observations[-1]['critic_status']=='busy_skipped'
    finally:release.set();gate.close()


def test_rejected_packet_profile_cannot_be_activated_as_confirmed():
    with pytest.raises(RuntimeError,match='unqualified|no completed'):
        require_confirmed_packet_profile()
