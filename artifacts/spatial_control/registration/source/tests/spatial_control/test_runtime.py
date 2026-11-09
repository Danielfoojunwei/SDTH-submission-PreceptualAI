"""Local adapter conformance against the actual deterministic simulator sensor.

All input arrays are explicit software fixtures, not warehouse/hardware evidence.
"""
from dataclasses import replace
import copy
import numpy as np
import pytest

from wirelessjev.fleetwm.data import Geometry, Observations
from wirelessjev.spatial_control.runtime import ClockedScans, LocalFleetRuntime, GAUGE


def fixture_runtime(n=2, sites=3, seed=1, seconds=.03):
    links=sites*3
    warm=Observations(np.zeros((60,n,2)),np.zeros((60,n,links),dtype='f4'),
        np.zeros((60,n,links),dtype=bool),np.full((60,n,links),-90.,dtype='f4'),
        np.zeros((60,n),dtype=bool))
    geom=Geometry(np.column_stack((np.arange(sites),np.zeros(sites))),
        np.full(3,20.),np.full(3,-95.),())
    sensor=ClockedScans(warm,20,seed)
    return LocalFleetRuntime(geom,sensor,seconds),sensor


def tick(runtime,sensor,slot=20,**changes):
    n=sensor.n
    args=dict(slot=slot,current=np.zeros(n,dtype=int),positions=np.zeros((n,2)),
        radio_available=np.ones(n,dtype=bool),pilot_snr=np.full((n,3),-15.),
        pilot_slot=np.full(n,slot,dtype=int),pilot_site=np.zeros(n,dtype=int),
        last_delivery_slot=np.zeros(n,dtype=int),sensor=sensor)
    args.update(changes)
    return runtime.tick(**args)


def actual_scan(runtime,sensor,robot=0,start=0,target=1):
    values=np.full((1,sensor.links//3,3),-100.)
    values[:,target,:]=-60.
    runtime.scan_started(robot,start,start+9)
    sensor.scan(start,np.array([robot]),np.zeros((1,2)),values,np.array([start+9]))
    sensor.complete(start+9)
    assert sensor.local_completed_payloads, 'Fixture uses a deterministic actual successful sensor draw'


def test_local_completed_scan_works_without_any_server_delivery():
    r,s=fixture_runtime(); actual_scan(r,s)
    assert not s.events  # No server delivery has occurred.
    proposal=tick(r,s)
    assert proposal['moves'] and proposal['moves'][0][:2]==(0,1)
    robot,target,aid=proposal['moves'][0]
    assert not r.validate(robot,target,20,aid)  # Paid local compute must finish first.
    assert r.validate(robot,target,21,aid)
    r.executed(robot,target,21,aid)
    assert r.controllers[0].charged_handover_slots==3
    assert r.summary()['warm_local_reports_used'] is False
    assert r.summary()['qualified_packet_risk'] is False


def test_own_report_never_leaks_into_another_client_controller():
    r,s=fixture_runtime(); actual_scan(r,s,robot=0)
    proposals=tick(r,s)
    assert r.controllers[0].reports
    assert not r.controllers[1].reports
    assert 1 in proposals['scan_robots']
    assert all(robot==0 for robot,_,_ in proposals['moves'])


def test_warm_local_values_do_not_receive_fabricated_exact_capture_times():
    r,s=fixture_runtime()
    s.local[:]=-50.  # Even excellent legacy warm cache values supply no exact clocks here.
    proposal=tick(r,s)
    assert not proposal['moves']
    assert all(not c.reports for c in r.controllers)


def test_runtime_ignores_remote_positions_and_sensor_values_until_completed():
    r,s=fixture_runtime(); actual_scan(r,s)
    proposal=tick(r,s,positions=np.full((s.n,2),np.nan))
    assert proposal['moves']
    r2,s2=fixture_runtime()
    s2.scan(0,np.array([0]),np.zeros((1,2)),np.full((1,3,3),-50.),np.array([30]))
    assert not s2.local_completed_payloads
    assert not tick(r2,s2)['moves']


def test_between_tick_completion_can_invalidate_a_pending_move():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    original=s.local_completed_payloads[0]
    later=copy.deepcopy(original)
    later.update(measurement_slot=12,complete_slot=21,processed_slot=21,scan_id='later')
    later['heard']=[False]*s.links
    later['value_dbm']=[0.]*s.links
    later['upper_dbm']=-90.
    # A deliberately constructed payload fixture tests delayed validation,
    # not the native sensor's 5-second cadence (which the kernel enforces).
    s.local_completed_payloads.append(later)
    assert not r.validate(robot,target,21,aid)
    assert r.controllers[0].reports[(1,5.2e9)].kind=='censored'
    assert r.controllers[0].reports[(1,5.2e9)].rssi_dbm is None


def test_request_is_rejected_if_newer_tick_observes_successful_delivery():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    tick(r,s,21,last_delivery_slot=np.array([21,0]))
    assert not r.validate(robot,target,22,aid)


def test_capabilities_are_checked_again_at_delayed_execution():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    c=r.controllers[robot]
    c.capability=replace(c.capability,local_authority=False)
    assert not r.validate(robot,target,21,aid)
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)


def test_scan_request_charges_only_after_actual_start_and_shares_background_cadence():
    r,s=fixture_runtime()
    p=tick(r,s)
    aid=p['scan_action_ids'][0]
    assert r.controllers[0].charged_scan_slots==0
    assert r.validate(0,None,21,aid)
    r.scan_started(0,21,30,aid)
    assert r.controllers[0].charged_scan_slots==9
    with pytest.raises(ValueError): r.scan_started(0,22,31)
    with pytest.raises(ValueError): r.scan_started(1,22,30)
    assert r.scan_started_slots[1] is None
    assert r.summary()['realized_scan_costs_equal_across_arms'] is False


def test_background_scan_blocks_a_previously_queued_local_scan():
    r,s=fixture_runtime(); aid=tick(r,s)['scan_action_ids'][0]
    r.scan_started(0,21,30)
    assert not r.validate(0,None,22,aid)
    r.rejected(aid,'background_scan_won',22)
    assert r.controllers[0]._pending is None
    assert r.controllers[0].charged_scan_slots==0


def test_rejected_requests_do_not_hang_or_spend_physical_time():
    r,s=fixture_runtime(); aid=tick(r,s)['scan_action_ids'][0]
    r.rejected(aid,'compute_budget',21); r.rejected(aid,'repeated',21)
    assert r.controllers[0]._pending is None
    assert not r.validate(0,None,22,aid)
    assert r.controllers[0].charged_scan_slots==0


def test_action_completion_waits_for_actual_receiver_availability():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    assert r.validate(robot,target,21,aid)
    r.executed(robot,target,21,aid)
    tick(r,s,24,current=np.array([1,0]),pilot_site=np.array([1,0]),
         radio_available=np.array([False,True]))
    assert r.controllers[0]._active is not None
    tick(r,s,30,current=np.array([1,0]),pilot_site=np.array([1,0]),
         last_delivery_slot=np.array([30,30]))
    assert r.controllers[0]._active is None
    assert r.controllers[0].events[-1]['observed_ap']==1


@pytest.mark.parametrize('changes',[
    {'radio_available':np.array([1,1])}, {'current':np.array([0.,0.])},
    {'current':np.array([0,3])}, {'pilot_snr':np.full((2,3),np.nan)},
    {'pilot_slot':np.array([21,20])}, {'last_delivery_slot':np.array([21,0])},
    {'pilot_site':np.array([4,0])}, {'pilot_slot':np.array([True,False])},
])
def test_invalid_local_tick_contract_is_rejected(changes):
    r,s=fixture_runtime()
    with pytest.raises(ValueError): tick(r,s,**changes)


def test_runtime_monotonic_clocks_and_sensor_identity_are_required():
    r,s=fixture_runtime(); tick(r,s)
    with pytest.raises(ValueError): tick(r,s,19)
    _,other=fixture_runtime()
    with pytest.raises(ValueError): tick(r,other)
    tick(r,s,21,last_delivery_slot=np.array([10,0]))
    with pytest.raises(ValueError): tick(r,s,22,last_delivery_slot=np.array([9,0]))


@pytest.mark.parametrize('change',[
    {'measurement_slot':-1}, {'complete_slot':21}, {'processed_slot':21},
    {'reference_id':'invented_gauge'}, {'robot':2}, {'heard':[1]*9},
    {'upper_dbm':float('nan')}, {'value_dbm':[float('nan')]*9},
])
def test_corrupt_or_future_completed_payloads_are_rejected(change):
    r,s=fixture_runtime(); actual_scan(r,s)
    s.local_completed_payloads[0].update(change)
    with pytest.raises(ValueError): tick(r,s)
    assert not r.controllers[0].reports


@pytest.mark.parametrize('cost',[0.,-1.,float('nan'),float('inf'),True])
def test_invalid_handover_cost_fails_closed(cost):
    with pytest.raises(ValueError): fixture_runtime(seconds=cost)


def test_healthy_tick_does_not_construct_unneeded_local_pilots():
    r,s=fixture_runtime(n=100)
    tick(r,s,0,last_delivery_slot=np.zeros(100,dtype=int))
    # First tick establishes all associations. Thereafter healthy clients need
    # no watchdog controller calls; unchanged clocks prove the fast path.
    tick(r,s,10,last_delivery_slot=np.full(100,10,dtype=int))
    assert all(c._clock==0 for c in r.controllers)


def test_execution_requires_prior_validation_and_success_is_single_use():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)
    assert r.controllers[robot].charged_handover_slots==0
    assert r.validate(robot,target,21,aid)
    r.executed(robot,target,21,aid)
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)
    assert r.controllers[robot].charged_handover_slots==3


@pytest.mark.parametrize('change',[{'robot':1},{'target':2},{'slot':22},{'action_id':'different'}])
def test_commit_cannot_change_the_validated_slot_client_target_or_id(change):
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    assert r.validate(robot,target,21,aid)
    params=dict(robot=robot,target=target,slot=21,action_id=aid);params.update(change)
    with pytest.raises(ValueError): r.executed(**params)
    assert r.controllers[robot].charged_handover_slots==0


def test_wrong_target_commit_consumes_authorization_until_explicit_revalidation():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    assert r.validate(robot,target,21,aid)
    with pytest.raises(ValueError): r.executed(robot,2,21,aid)
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)
    assert r.validate(robot,target,21,aid)
    r.executed(robot,target,21,aid)


def test_compute_overrun_rejection_invalidates_authorization_and_pending_request():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    assert r.validate(robot,target,21,aid)
    r.rejected(aid,'local_validation_compute_budget',21)
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)
    assert r.controllers[robot]._pending is None
    assert r.controllers[robot].charged_handover_slots==0


def test_unrelated_client_validation_does_not_destroy_a_batch_authorization():
    r,s=fixture_runtime(); actual_scan(r,s)
    proposed=tick(r,s)
    robot,target,aid=proposed['moves'][0]
    scan_id=proposed['scan_action_ids'][1]
    assert r.validate(robot,target,21,aid)
    assert r.validate(1,None,21,scan_id)
    r.executed(robot,target,21,aid)
    r.scan_started(1,21,30,scan_id)
    assert r.controllers[0].charged_handover_slots==3
    assert r.controllers[1].charged_scan_slots==9


def test_new_tick_invalidates_previously_validated_state():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    assert r.validate(robot,target,21,aid)
    tick(r,s,21,last_delivery_slot=np.array([21,0]))
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)


def test_new_unprocessed_completion_requires_measured_revalidation_not_commit_ingestion():
    r,s=fixture_runtime(); actual_scan(r,s)
    robot,target,aid=tick(r,s)['moves'][0]
    assert r.validate(robot,target,21,aid)
    later=copy.deepcopy(s.local_completed_payloads[0])
    later.update(measurement_slot=12,complete_slot=21,processed_slot=21,scan_id='later')
    later['heard']=[False]*s.links;later['value_dbm']=[0.]*s.links;later['upper_dbm']=-90.
    s.local_completed_payloads.append(later)
    before=r.completed_index
    with pytest.raises(ValueError): r.executed(robot,target,21,aid)
    assert r.completed_index==before  # Commit path performs no unpaid ingestion.
    assert not r.validate(robot,target,21,aid)
    assert r.completed_index==before+1


def test_background_scan_invalidates_authorized_autonomous_scan_for_same_client():
    r,s=fixture_runtime(); aid=tick(r,s)['scan_action_ids'][0]
    assert r.validate(0,None,21,aid)
    r.scan_started(0,21,30)
    with pytest.raises(ValueError): r.scan_started(0,21,30,aid)
    assert r.controllers[0].charged_scan_slots==0


def test_scan_start_requires_single_use_authorization_before_actual_callback():
    r,s=fixture_runtime(); aid=tick(r,s)['scan_action_ids'][0]
    with pytest.raises(ValueError): r.scan_started(0,21,30,aid)
    assert r.validate(0,None,21,aid)
    r.scan_started(0,21,30,aid)
    with pytest.raises(ValueError): r.scan_started(0,21,30,aid)
    assert r.controllers[0].charged_scan_slots==9
