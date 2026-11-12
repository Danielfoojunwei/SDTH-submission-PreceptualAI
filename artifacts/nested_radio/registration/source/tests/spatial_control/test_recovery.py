"""Software conformance only: these fixtures are not fleet/hardware evidence."""
from dataclasses import replace
import pytest

from wirelessjev.spatial_control.recovery import (
    RecoveryCapability, RecoveryLimits, LocalScanReport, LocalPilot,
    ClientLocalRecovery, LocalRecoveryAction, LocalScanRequest,
    CentralCommand, CommandDelivery, central_execution_allowed,
)


def capability(**changes):
    return replace(RecoveryCapability(7, (0,1,2), (2.4e9,5.2e9), 3,
        preinstalled=True, local_authority=True), **changes)


def controller(**changes):
    return ClientLocalRecovery(capability(**changes), RecoveryLimits())


def report(**changes):
    return replace(LocalScanReport(7,1,5.2e9,0,9,0,'scan-0','heard',-65.), **changes)


def decision(c, now=20, **changes):
    kwargs=dict(now=now,current_ap=0,last_delivery_slot=0,radio_available=True)
    kwargs.update(changes)
    return c.decide(**kwargs)


def test_silence_and_own_completed_scan_produce_request_without_executing():
    c=controller(); c.observe_scan(report(),9)
    assert decision(c,19) is None
    a=decision(c)
    assert isinstance(a,LocalRecoveryAction) and a.target_ap==1
    assert a.measurement_slot==0 and a.current_rssi_dbm is None
    assert c.charged_handover_slots==0
    assert decision(c,21)==a  # Deferred compute is allowed; capture is not refreshed.
    c.started(a,21)
    assert c.charged_handover_slots==3
    assert decision(c,23,current_ap=1,radio_available=False) is None
    with pytest.raises(ValueError): c.finish(23,1)
    c.finish(24,1)
    assert decision(c,25,current_ap=1) is None


@pytest.mark.parametrize('flag',['preinstalled','local_authority'])
def test_missing_installed_authority_never_actuates(flag):
    c=controller(**{flag:False,'autonomous_scan':True}); c.observe_scan(report(),9)
    assert decision(c) is None
    assert c.events==[]


def test_installation_time_is_not_inferred():
    c=controller(installed_at=50); c.observe_scan(report(),9)
    assert decision(c,20) is None
    assert isinstance(decision(c,50),LocalRecoveryAction)


@pytest.mark.parametrize('field,value',[('robot_id',8),('ap_id',4),('carrier_hz',6.5e9),('reference_id','other')])
def test_foreign_client_link_or_reference_is_rejected(field,value):
    c=controller()
    with pytest.raises(ValueError): c.observe_scan(report(**{field:value}),9)
    assert not c.reports


@pytest.mark.parametrize('kind,point,bound',[('censored',None,-80.),('missing',None,None)])
def test_new_censored_or_missing_scan_invalidates_old_heard(kind,point,bound):
    c=controller(); c.observe_scan(report(),9)
    a=decision(c)
    c.observe_scan(report(measurement_slot=20,complete_slot=29,scan_started_slot=20,
        scan_id='scan-1',kind=kind,rssi_dbm=point,upper_dbm=bound),29)
    assert decision(c,30) is None
    assert c.events[-1]['event']=='invalidated'
    with pytest.raises(ValueError): c.started(a,30)


@pytest.mark.parametrize('changes',[{'complete_slot':10}, {'scan_started_slot':1}, {'measurement_slot':10},
    {'kind':'censored'}, {'kind':'missing'}, {'rssi_dbm':float('nan')}, {'kind':'heard','upper_dbm':-80.}])
def test_malformed_or_not_yet_complete_report_is_rejected(changes):
    c=controller()
    with pytest.raises(ValueError): c.observe_scan(report(**changes),9)
    assert not c.reports


def test_report_needs_paid_scan_interruption_and_stable_identity():
    c=controller()
    with pytest.raises(ValueError): c.observe_scan(report(complete_slot=8),9)
    c.observe_scan(report(),9); c.observe_scan(report(),9)
    with pytest.raises(ValueError): c.observe_scan(report(rssi_dbm=-55.),9)
    assert c.reports[1,5.2e9].rssi_dbm==-65.


def test_stale_reading_cannot_be_refreshed_by_late_delivery_or_clock():
    c=controller(); c.observe_scan(report(),500)
    assert isinstance(decision(c,500),LocalRecoveryAction)
    assert decision(c,501) is None
    assert c.reports[1,5.2e9].measurement_slot==0
    with pytest.raises(ValueError): decision(c,500)


def test_recent_stronger_current_pilot_prevents_optimistic_switch():
    c=controller(); c.observe_scan(report(),9)
    p=LocalPilot(7,0,5.2e9,20,-66.)
    assert decision(c,current_pilots=(p,)) is None
    # A stale pilot provides no comparison, it never becomes a new sample.
    assert isinstance(decision(c,31,current_pilots=(p,)),LocalRecoveryAction)


@pytest.mark.parametrize('changes',[{'robot_id':8},{'ap_id':1},{'measurement_slot':21},{'reference_id':'other'}])
def test_bad_pilot_is_not_current_association_evidence(changes):
    c=controller(); c.observe_scan(report(),9)
    p=replace(LocalPilot(7,0,5.2e9,20,-80.),**changes)
    with pytest.raises(ValueError): decision(c,current_pilots=(p,))


def test_current_scan_comparison_requires_margin_and_absolute_target_threshold():
    c=controller(); c.observe_scan(report(ap_id=0,rssi_dbm=-70.),9)
    c.observe_scan(report(),9)
    assert decision(c) is None  # Only 5 dB gain.
    d=controller(); d.observe_scan(report(rssi_dbm=-80.),9)
    assert decision(d) is None  # Unknown current cannot waive absolute threshold.


def test_unknown_outcome_requires_reconciliation_and_retry_backoff():
    c=controller(); c.observe_scan(report(),9); a=decision(c); c.started(a,21)
    c.finish(24,None)
    assert decision(c,30) is None
    c.reconcile(31,0)
    assert decision(c,32) is None
    with pytest.raises(ValueError): c.reconcile(33,0)
    assert c.events[-1]['event']=='reconciled'


def test_known_outcome_still_enforces_dwell_backoff():
    c=controller(); c.observe_scan(report(),9); a=decision(c); c.started(a,21); c.finish(24,1)
    c.observe_scan(report(ap_id=2,measurement_slot=400,complete_slot=409,scan_started_slot=400,scan_id='new'),409)
    assert decision(c,523,current_ap=1) is None
    assert isinstance(decision(c,524,current_ap=1),LocalRecoveryAction)


def test_outage_unavailable_receiver_never_starts_scan_or_move():
    c=controller(autonomous_scan=True); c.observe_scan(report(),9)
    assert decision(c,radio_available=False) is None
    assert c.events==[]


def test_scan_request_has_explicit_authority_shared_cadence_and_actual_charge():
    c=controller(autonomous_scan=True)
    assert decision(c,last_scan_started_slot=0) is None
    request=decision(c,500,last_scan_started_slot=0)
    assert isinstance(request,LocalScanRequest)
    assert request.valid_until_slot>=502
    assert c.charged_scan_slots==0
    c.started(request,501)
    assert c.charged_scan_slots==9
    c.finish(510,0)
    assert decision(c,1000,last_scan_started_slot=501) is None
    assert isinstance(decision(c,1001,last_scan_started_slot=501),LocalScanRequest)


def test_scan_disabled_and_recent_background_scan_block_autonomy():
    assert decision(controller()) is None
    c=controller(autonomous_scan=True)
    request=decision(c)
    assert isinstance(request,LocalScanRequest)
    # An actual background scan has priority and invalidates this unstarted request.
    assert decision(c,21,last_scan_started_slot=21) is None
    with pytest.raises(ValueError): c.started(request,21)


def test_cancelling_a_rejected_request_spends_no_physical_time():
    c=controller(autonomous_scan=True); a=decision(c)
    c.cancel(a,21,'local_compute_deadline')
    assert c.charged_scan_slots==c.charged_handover_slots==0
    with pytest.raises(ValueError): c.started(a,21)
    with pytest.raises(ValueError): c.cancel(a,21,'again')
    with pytest.raises(ValueError): c.started(None,21)


def test_last_delivery_and_scan_clocks_are_causal():
    c=controller()
    with pytest.raises(ValueError): decision(c,last_delivery_slot=21)
    with pytest.raises(ValueError): decision(c,last_scan_started_slot=21)
    decision(c,last_delivery_slot=19)
    with pytest.raises(ValueError): decision(c,21,last_delivery_slot=18)


@pytest.mark.parametrize('changes',[{'robot_id':True},{'ap_ids':(0,0)}, {'carriers_hz':(0.,)},
 {'handover_slots':0},{'local_authority':1},{'reference_id':''},{'installed_at':-1}])
def test_capabilities_fail_closed_on_bad_types(changes):
    with pytest.raises(ValueError): capability(**changes)


@pytest.mark.parametrize('changes',[{'silence_slots':True},{'max_scan_age_slots':0},
 {'scan_slots':501},{'margin_db':-1},{'minimum_rssi_dbm':float('inf')}])
def test_limit_validation(changes):
    with pytest.raises(ValueError): replace(RecoveryLimits(),**changes)


def test_central_command_requires_exact_created_after_delivery_and_expiry():
    command=CentralCommand('cmd-1',7,100,200,0,1)
    delivery=CommandDelivery('cmd-1',7,100,101,0)
    assert central_execution_allowed(command,delivery,101,0,True)
    assert not central_execution_allowed(command,None,101,0,True)
    assert not central_execution_allowed(command,delivery,100,0,True)
    assert not central_execution_allowed(command,delivery,200,0,True)
    assert not central_execution_allowed(command,delivery,102,1,True)
    assert not central_execution_allowed(command,delivery,102,0,False)
    for changes in ({'command_id':'old'},{'created_slot':99},{'robot_id':8},{'from_ap':1}):
        assert not central_execution_allowed(command,replace(delivery,**changes),102,0,True)
    with pytest.raises(ValueError): CommandDelivery('cmd-1',7,100,100,0)
    with pytest.raises(ValueError): CentralCommand('stay',7,100,200,0,0)


def test_cancel_pending_is_idempotent_and_does_not_clear_a_newer_request():
    c=controller(autonomous_scan=True); old=decision(c)
    assert c.cancel_pending(old.action_id,21,'budget')
    assert not c.cancel_pending(old.action_id,21,'repeated')
    new=decision(c,22)
    assert not c.cancel_pending(old.action_id,22,'late_rejection')
    assert decision(c,23)==new
    c.started(new,23)
    with pytest.raises(ValueError): c.cancel_pending(new.action_id,24,'too_late')


def test_started_independently_refuses_superseded_target_evidence():
    c=controller(); c.observe_scan(report(),9); a=decision(c)
    c.observe_scan(report(scan_id='later',scan_started_slot=20,measurement_slot=20,
        complete_slot=29,kind='censored',rssi_dbm=None,upper_dbm=-89.),29)
    with pytest.raises(ValueError): c.started(a,29)
    assert c.charged_handover_slots==0


def test_one_scan_identity_has_one_set_of_event_clocks_across_all_links():
    c=controller(); c.observe_scan(report(),9)
    with pytest.raises(ValueError):
        c.observe_scan(report(ap_id=2,scan_started_slot=1,measurement_slot=1,complete_slot=10),10)
    assert (2,5.2e9) not in c.reports
