"""Admission/queue contracts only, not warehouse performance evidence."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wirelessjev.packet_value.control import PacketController


class PreferredCandidate(torch.nn.Module):
    config = {'family': 'test-action-values'}

    def __init__(self, value=-5.):
        super().__init__(); self.value=value

    def forward_actions(self, context, query):
        result=torch.zeros(query.shape[:2])
        result[:,1]=self.value
        return {'delta_misses': result}


def geometry():
    return SimpleNamespace(sites=np.array([[0.,0.],[10.,0.]]),tx=np.array([21.,23.,23.]),
                           noise=np.array([-95.,-95.,-95.]))


def observation(slot=0, candidate=-45.):
    n=2;pos=np.array([[1.,1.],[2.,2.]])
    return {'slot':slot,'current':np.zeros(n,dtype=int),'pilot_snr':np.full((n,3),20.),
            'local_rssi':np.array([[[-45.,-45.,-45.],[candidate]*3]]*n),
            'positions':pos,'plan':np.broadcast_to(pos[:,None,:],(n,9,2)).copy(),
            'load':np.array([2,0]),'misses':np.zeros(n),'sent':np.full(n,slot/5),
            'gap':np.zeros(n),'outage_remaining':np.zeros(n),'scan_remaining':np.zeros(n),
            'roams':np.zeros(n),'forecast':np.full((n,9,2,3),20.)}


def test_head_result_is_unavailable_before_fifo_completion():
    controller=PacketController(geometry(),.03,PreferredCandidate(),head_seconds=.15)
    controller.observe(observation())
    assert controller.take(14) is None
    p=controller.take(15)
    assert p['eligible'].tolist()==[True,True] and p['target'].tolist()==[1,1]
    assert p['valid_until']==100
    assert controller.take(15) is None
    corrected=PacketController(geometry(),.03,PreferredCandidate(),correction=4.)
    corrected.observe(observation())
    assert not corrected.take(1)['eligible'].any()


def test_immediate_bad_candidate_is_rejected_despite_optimistic_value():
    controller=PacketController(geometry(),4.,PreferredCandidate(-100.))
    controller.observe(observation(candidate=-89.))
    proposal=controller.take(1)
    assert not proposal['eligible'].any() and proposal['target'].tolist()==[0,0]
    for slot in range(100,1001,100):controller.observe(observation(slot,candidate=-89.))
    assert len(controller.decisions)==1
    controller.observe(observation(1100,candidate=-89.))
    assert len(controller.decisions)==2


def test_execution_resets_dwell_and_books_the_chosen_reservation_once():
    controller=PacketController(geometry(),4.,PreferredCandidate(-5.),correction=1.)
    controller.observe(observation())
    assert controller.take(0) is None
    controller.take(1)
    controller.executed(5,np.array([0]),np.array([1]))
    controller.executed(5,np.array([0]),np.array([1]))
    assert controller.next_decision[0]==1005 and controller.history.last_switch[0]==5
    summary=controller.summary()
    assert len(summary['executions'])==1
    assert summary['known_outage_seconds']==4. and summary['nominal_command_equivalent_exposure']==80.
    assert summary['booked_predicted_delta_misses']==-4. and summary['not_factual_guarantee']
    assert controller.take(1) is None
    assert controller.summary()['booked_predicted_delta_misses']==-4.


def test_missing_forecast_and_busy_radios_do_not_issue_jobs():
    controller=PacketController(geometry(),.03,PreferredCandidate())
    obs=observation();obs['forecast']=None
    controller.observe(obs)
    assert len(controller.observations)==1 and not controller.decisions
    obs=observation(100);obs['scan_remaining'][:]=9
    controller.observe(obs)
    assert len(controller.observations)==2 and not controller.decisions
    controller.observe(observation(200))
    assert len(controller.decisions)==1 and controller.decisions[0]['ready_slot']==201


def test_stale_intervention_execution_has_no_invented_modeled_credit():
    controller=PacketController(geometry(),.03,PreferredCandidate())
    controller.observe(observation())
    controller.executed(0,np.array([0]),np.array([1]))
    assert controller.summary()['known_outage_seconds']==.03
    assert controller.summary()['booked_predicted_delta_misses']==0.
    assert controller.summary()['executions'][0]['booked_predicted_delta_misses'] is None
    with pytest.raises(ValueError):PacketController(geometry(),4.,correction=-1.)


def test_overloaded_head_is_one_fifo_server_for_different_due_groups():
    controller=PacketController(geometry(),.03,PreferredCandidate(),head_seconds=2.)
    first=observation();first['scan_remaining'][1]=9
    controller.observe(first)
    controller.observe(observation(100))
    assert [r['ready_slot'] for r in controller.decisions]==[200,400]
    assert controller.take(199) is None
    first=controller.take(200)
    assert not first['eligible'].any() and first['valid_until']==100
    assert controller.take(399) is None
    second=controller.take(400)
    assert not second['eligible'].any() and second['valid_until']==200
    assert controller.summary()['booked_predicted_delta_misses']==0.


def test_absolute_one_second_ttl_expires_at_the_boundary_without_resetting_hold():
    controller=PacketController(geometry(),.03,PreferredCandidate())
    controller.observe(observation())
    deadline=controller.next_decision.copy()
    expired=controller.take(100)
    assert expired['valid_until']==100 and not expired['eligible'].any()
    np.testing.assert_array_equal(controller.next_decision,deadline)
    assert controller.summary()['decisions'][0]['expired_at_publication']
    controller.executed(101,np.array([0]),np.array([1]))
    assert controller.summary()['executions'][0]['booked_predicted_delta_misses'] is None
    assert controller.summary()['booked_predicted_delta_misses']==0.
    assert controller.summary()['known_outage_seconds']==.03
    published=PacketController(geometry(),.03,PreferredCandidate())
    published.observe(observation())
    assert published.take(1)['eligible'].all()
    published.executed(100,np.array([0]),np.array([1]))
    # Timely publication does not extend the original absolute execution TTL.
    assert published.summary()['executions'][0]['booked_predicted_delta_misses'] is None
