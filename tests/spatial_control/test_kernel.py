"""Analytic software fixtures for dispatch clocks; no warehouse accuracy claims."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from wirelessjev.fleetwm.data import Observations
from wirelessjev.jepa_repair.telemetry import TimedDeliveredScans
from wirelessjev.native.world import World, HORIZONS, CLIENT_TX_DBM
from wirelessjev.service_world.sim import simulate as predecessor
from wirelessjev.spatial_control import kernel


def fixture_world(duration=3, robots=2, *, failed_current=False):
    """Explicit two-AP constant fields, not a registered research episode."""
    positions = int(duration/.2)+1
    snr = np.full((positions, robots, 2, 3), 35., dtype='f4')
    if failed_current: snr[:, :, 0] = -200.
    rssi = np.full_like(snr, -45.)
    rssi[:, :, 0] = -30.  # Explicit initial AP0 regardless of PHY failure fixture.
    forecast = np.zeros((duration, robots, len(HORIZONS), 2, 3), dtype='f4')
    forecast[:, :, :, 1] = 40.
    return World(seed=81721, n=robots, sites=np.array(((0., 0.), (10., 0.))),
        snr=snr, rssi_dl=rssi, forecast=forecast,
        domain=np.zeros((2, 3), dtype=int), avail=np.ones(1),
        video_mbps=np.zeros(robots), bulk_mbps=np.zeros(robots),
        duration_s=duration, sigma_map=0., blind=False,
        sigma_belief=np.ones((duration, robots, 2, 3), dtype='f4'),
        noise_db=np.array((-94., -88., -85.)), channel=np.ones((2, 3), dtype=int),
        plan=np.ones((duration, robots, len(HORIZONS), 2), dtype='f4'),
        pos=np.ones((positions, robots, 2), dtype='f4'), tx_dl=np.array((21., 23., 23.)))


def sensor(world):
    shape=(60, world.n, 6)
    heard=np.zeros(shape, bool); heard[0]=True
    values=np.where(heard, -45., 0.).astype('f4')
    scanned=np.zeros(shape[:2], bool); scanned[0]=True
    warm=Observations(np.ones(shape[:2]+(2,), dtype='f4'), values, heard,
                      np.full(shape, -90., dtype='f4'), scanned)
    return TimedDeliveredScans(warm, int(world.duration_s), world.seed)


def policy(**changes):
    return replace(kernel.Policy('unit_dispatch', 'native', cadence_scan=True,
        cadence_roam=True, scan_period_s=100., gate=False, stagger=False,
        roam_cost_s=.03, duplication=False, predictive_mcs=False), **changes)


def run_new(world, runtime=None, **kwargs):
    return kernel.simulate(world, policy(**kwargs), scan_observer=sensor(world),
        scan_duration_s=.09, causal_central=True, spatial_runtime=runtime,
        blockage_rate=0., detailed=True)


class RecorderRuntime:
    supports_client_local_actions=True
    supports_autonomous_scans=True

    def __init__(self, requests=None, valid=True):
        self.requests=requests or {}; self.calls=[]; self.actions=[]; self.rejects=[]; self.scans=[]
        self.valid=valid

    def tick(self, **fields):
        self.calls.append(fields)
        return self.requests.get(fields['slot'], {})

    def validate(self, robot, target, slot, action_id): return self.valid
    def executed(self, robot, target, slot, action_id): self.actions.append((robot,target,slot,action_id))
    def rejected(self, action_id, reason, slot): self.rejects.append((action_id,reason,slot))
    def scan_started(self, robot, slot, complete, action_id=None): self.scans.append((robot,slot,complete,action_id))


@pytest.mark.parametrize('kind', ['native', 'legacy', 'mlo_default'])
@pytest.mark.parametrize('detailed', [False, True])
def test_disabled_extensions_preserve_every_original_packet_output(kind, detailed):
    w=fixture_world(duration=1)
    p=replace(policy(), kind=kind)
    old=predecessor(copy.deepcopy(w), p, detailed=detailed)
    actual=kernel.simulate(copy.deepcopy(w), p, detailed=detailed)
    assert actual==old and 'spatial_execution' not in actual


@pytest.mark.parametrize('handover', [.03, 4.])
def test_both_handover_costs_wait_for_newly_created_delivered_packet(handover):
    result=run_new(fixture_world(), roam_cost_s=handover)
    ledger=result['spatial_execution']
    queued=[r for r in ledger['central_events'] if r['event']=='queued']
    executed=[r for r in ledger['central_events'] if r['event']=='executed']
    assert queued and executed
    for event in executed:
        assert event['created_slot']==event['proposed_slot']+10
        assert event['created_slot']<=event['packet_created_slot']<=event['delivery_slot']
        assert event['delivery_slot']>event['created_slot']
        assert event['execution_slot']==event['delivery_slot']<event['expires_slot']
        assert event['outage_until_slot']==event['execution_slot']+1+int(np.ceil(handover/.01))
    assert result['control_sent']==120


def test_same_pending_target_does_not_renew_one_second_lifetime():
    result=run_new(fixture_world(failed_current=True, robots=1))
    ledger=result['spatial_execution']
    queued=[r for r in ledger['central_events'] if r['event']=='queued']
    assert [(r['proposed_slot'],r['created_slot'],r['expires_slot']) for r in queued]==[(0,10,110),(200,210,310)]
    assert any(r['event']=='not_renewed' and r['slot']==100 and r['expires_slot']==110 for r in ledger['dispatch_events'])
    assert any(r['event']=='cancelled' and r['slot']==110 and r['reason']=='expired' for r in ledger['central_events'])
    assert not any(r['event']=='executed' for r in ledger['central_events'])


def test_local_move_needs_no_central_delivery_but_pays_compute_and_handover():
    w=fixture_world(duration=1, failed_current=True, robots=1)
    w.forecast[:]=0.;w.forecast[:,:,:,0]=40. # No central roam proposal.
    runtime=RecorderRuntime({0:{'moves':[(0,1,'recover-0')]}})
    result=run_new(w, runtime)
    assert runtime.actions==[(0,1,1,'recover-0')]
    event=next(r for r in result['spatial_execution']['local_events'] if r['event']=='executed')
    assert event['slot']==event['request_slot']+1
    assert event['last_delivery_slot']==-1
    assert event['outage_until_slot']==event['slot']+1+3
    assert not result['spatial_execution']['central_events']
    assert set(runtime.calls[0])=={'slot','current','positions','radio_available','pilot_snr','pilot_slot','pilot_site','last_delivery_slot','sensor'}
    assert not {'world','snr','fade','blocked','forecast'} & set(runtime.calls[0])


def test_latest_local_evidence_is_revalidated_after_paid_delay():
    w=fixture_world(failed_current=True, robots=1)
    runtime=RecorderRuntime({0:{'moves':[(0,1,'stale')] }}, valid=False)
    result=run_new(w, runtime)
    assert not runtime.actions
    assert ('stale','local_evidence_no_longer_valid',1) in runtime.rejects
    assert not [r for r in result['spatial_execution']['local_events'] if r['event']=='executed']


def test_autonomous_scans_share_budget_and_charge_actual_completion():
    w=fixture_world(duration=6, failed_current=True, robots=1)
    requests={slot:{'scan_robots':[0], 'scan_action_ids':{0:f'scan-{slot}'}} for slot in (0,100,500)}
    runtime=RecorderRuntime(requests)
    result=run_new(w, runtime)
    scans=[r for r in result['spatial_execution']['scan_events'] if r['action_id'] is not None]
    assert [r['slot'] for r in scans]==[1,501]
    assert all(r['complete_slot']==r['slot']+10 and r['interruption_slots']==9 for r in scans)
    assert all(r['interruption_start_slot']==r['slot']+1 for r in scans)
    assert ('scan-100','shared_scan_budget',101) in runtime.rejects
    assert runtime.scans==[(0,1,11,'scan-0'),(0,501,511,'scan-500')]
    completions=result['spatial_execution']['local_completion_events']
    assert all(r['measurement_slot']<r['complete_slot']<=r['processed_slot'] for r in completions)
    assert len(completions)==2


def test_intervening_control_delivery_cancels_pending_local_recovery():
    w=fixture_world(duration=1, robots=1)
    # Find an ordinary unit-fixture seed whose release phase is slot0. A
    # strong current link then delivers after tick0, before delayed action1.
    for seed in range(81721,81740):
        w.seed=seed
        _,_,rng=kernel.hidden_draws(w,100,0.)
        if int(rng.integers(0,10000,size=1)[0]) % 5==0: break
    else: raise AssertionError('No release-phase fixture')
    runtime=RecorderRuntime({0:{'moves':[(0,1,'resolved')]}})
    result=run_new(w,runtime)
    assert ('resolved','silence_resolved_after_request',1) in runtime.rejects
    assert not runtime.actions


@pytest.mark.parametrize('flag,payload',[
    ('supports_client_local_actions', {'moves':[(0,1,'move')]}),
    ('supports_autonomous_scans', {'scan_robots':[0], 'scan_action_ids':{0:'scan'}}),
])
def test_local_actions_require_explicit_capabilities(flag,payload):
    runtime=RecorderRuntime({0:payload});setattr(runtime,flag,False)
    with pytest.raises(ValueError,match='declared preinstalled|explicit local capability'):
        run_new(fixture_world(robots=1),runtime)


@pytest.mark.parametrize('payload',[
    {'moves':[(True,1,'a')]}, {'moves':[(0,-1,'a')]}, {'moves':[(0,1,'')]},
    {'scan_robots':[0]}, {'moves':[(0,1,'a')], 'scan_robots':[0], 'scan_action_ids':{0:'b'}},
    {'moves':[(0,1,'a'),(1,1,'a')]}, {'hidden_scan':True},
])
def test_malformed_or_duplicate_requests_fail_closed(payload):
    with pytest.raises(ValueError): run_new(fixture_world(),RecorderRuntime({0:payload}))


def test_local_cpu_overrun_discards_requests_before_any_action(monkeypatch):
    clock=iter(np.arange(1000)*.006)
    monkeypatch.setattr(kernel.time,'perf_counter',lambda:float(next(clock)))
    runtime=RecorderRuntime({0:{'moves':[(0,1,'slow')]}})
    result=run_new(fixture_world(robots=1,failed_current=True),runtime)
    assert not runtime.actions and ('slow','local_compute_budget',0) in runtime.rejects
    assert result['spatial_execution']['local_decisions'][0]['overrun']


def test_central_cpu_overrun_discards_all_new_proposals(monkeypatch):
    clock=iter(np.arange(1000)*.2)
    monkeypatch.setattr(kernel.time,'perf_counter',lambda:float(next(clock)))
    result=run_new(fixture_world(robots=1))
    ledger=result['spatial_execution']
    assert ledger['central_decisions'] and all(r['overrun'] for r in ledger['central_decisions'])
    assert not ledger['central_events']
    assert all(r['reason']=='central_compute_budget' for r in ledger['dispatch_events'])
    assert result['roams']==0


@pytest.mark.parametrize('changes',[
    {'kind':'legacy'}, {'dual_radio':True}, {'cadence_scan':False},
    {'cadence_roam':False}, {'planned_roam':False}, {'online_map':True},
])
def test_unsupported_spatial_profiles_are_refused_before_world_access(changes):
    with pytest.raises(ValueError,match='complete native single-radio causal episodes'):
        kernel.simulate(object(),policy(**changes),causal_central=True,scan_duration_s=.09)


@pytest.mark.parametrize('options',[
    {'external_primary':object()}, {'checkpoint_at':0}, {'force_action':(0,1)},
    {'stop_slot':10}, {'resume':object()}, {'focal_hold_slots':1},
])
def test_forks_and_external_authority_are_outside_the_new_kernel_profile(options):
    with pytest.raises(ValueError,match='complete native single-radio causal episodes'):
        kernel.simulate(object(),policy(),causal_central=True,scan_duration_s=.09,**options)


def test_causal_creation_relative_ttl_is_not_shortened_by_legacy_option():
    w=fixture_world(failed_current=True, robots=1)
    result=kernel.simulate(w, policy(), scan_observer=sensor(w),
        scan_duration_s=.09, causal_central=True, native_proposal_ttl_slots=100,
        blockage_rate=0., detailed=True)
    queued=[e for e in result['spatial_execution']['central_events'] if e['event']=='queued']
    assert queued and all(e['expires_slot']-e['created_slot']==100 for e in queued)


def test_slow_local_tick_blocks_following_ticks_until_actual_compute_finishes(monkeypatch):
    clock=iter(np.arange(1000)*.126)
    monkeypatch.setattr(kernel.time,'perf_counter',lambda:float(next(clock)))
    runtime=RecorderRuntime()
    result=run_new(fixture_world(duration=1,robots=1),runtime)
    assert [row['slot'] for row in runtime.calls]==[0,20,40,60,80]
    assert [row['slot'] for row in result['spatial_execution']['local_compute_events']]==[10,30,50,70,90]
    assert result['spatial_execution']['local_decisions'][0]['ready_slot']==13


def test_validation_cpu_overrun_discards_request_before_changing_radio(monkeypatch):
    times=iter([0.,.001,.002,.003,.004,.010]+[.011+i*.001 for i in range(1000)])
    monkeypatch.setattr(kernel.time,'perf_counter',lambda:next(times))
    runtime=RecorderRuntime({0:{'moves':[(0,1,'validate-slow')]}})
    result=run_new(fixture_world(duration=1,robots=1,failed_current=True),runtime)
    assert not runtime.actions
    assert ('validate-slow','local_validation_compute_budget',1) in runtime.rejects
    validation=result['spatial_execution']['local_validation_batches'][0]
    assert validation['overrun'] and validation['elapsed_seconds']>.005
    assert not [row for row in result['spatial_execution']['local_events'] if row['event']=='executed']


def test_shared_validation_batch_budget_prevents_hidden_per_robot_cpu_sum(monkeypatch):
    times=iter([0.,.001,.002,.003,.004,.007,.008,.011]+[.012+i*.001 for i in range(1000)])
    monkeypatch.setattr(kernel.time,'perf_counter',lambda:next(times))
    runtime=RecorderRuntime({0:{'moves':[(0,1,'first'),(1,1,'second')]}})
    result=run_new(fixture_world(duration=1,robots=2,failed_current=True),runtime)
    assert not runtime.actions
    assert ('first','local_validation_compute_budget',1) in runtime.rejects
    assert ('second','local_validation_compute_budget',1) in runtime.rejects
    batch=result['spatial_execution']['local_validation_batches'][0]
    assert batch['overrun'] and all(row['elapsed_seconds']<.005 for row in batch['checks'])
    assert batch['elapsed_seconds']>.005


def test_new_ap_cannot_receive_a_pilot_measured_before_its_association():
    w=fixture_world(duration=1,robots=1,failed_current=True)
    w.forecast[:]=0.;w.forecast[:,:,:,0]=40.
    runtime=RecorderRuntime({0:{'moves':[(0,1,'new-ap')]}})
    run_new(w,runtime)
    snapshots={r['slot']:r for r in runtime.calls}
    assert snapshots[10]['current'][0]==1
    assert snapshots[10]['pilot_site'][0]==0 and snapshots[10]['pilot_slot'][0]==0
    assert snapshots[20]['pilot_site'][0]==1 and snapshots[20]['pilot_slot'][0]==10


def test_pilot_capture_during_scan_stays_unavailable_after_scan_finishes():
    runtime=RecorderRuntime({0:{'scan_robots':[0],'scan_action_ids':{0:'capture-gap'}}})
    run_new(fixture_world(duration=1,robots=1,failed_current=True),runtime)
    snapshots={r['slot']:r for r in runtime.calls}
    assert not snapshots[10]['radio_available'][0]
    assert snapshots[20]['radio_available'][0]
    assert snapshots[20]['pilot_slot'][0]==0  # Lagged capture at10 lay inside scan1..10.
    assert snapshots[30]['pilot_slot'][0]==20


def test_central_overrun_blocks_next_consumed_forecast_until_compute_finishes(monkeypatch):
    clock=iter(np.arange(1000)*1.26)
    monkeypatch.setattr(kernel.time,'perf_counter',lambda:float(next(clock)))
    result=run_new(fixture_world(duration=3,robots=1))
    ledger=result['spatial_execution']
    assert [r['slot'] for r in ledger['central_decisions']]==[0,200]
    assert ledger['central_decisions'][0]['compute_ready_slot']==126
    assert ledger['central_compute_events']==[{'event':'forecast_discarded_while_busy',
        'slot':100,'forecast_decision':1,'ready_slot':126}]
    assert result['roams']==0


@pytest.mark.parametrize('unsafe', ['power','interference'])
def test_spatial_capture_contract_refuses_dynamic_unrecorded_power_interference(unsafe):
    p=policy(power_control=unsafe=='power')
    with pytest.raises(ValueError,match='complete native single-radio causal episodes'):
        kernel.simulate(object(),p,causal_central=True,scan_duration_s=.09,interference=unsafe=='interference')


def test_central_and_mcs_estimate_cannot_read_invalid_new_ap_capture():
    w=fixture_world(duration=1,robots=1,failed_current=True)
    w.forecast[:]=0.;w.forecast[:,:,:,0]=40.
    runtime=RecorderRuntime({0:{'moves':[(0,1,'new-ap-mcs')]}})
    observed={}
    def capture(public):
        observed[public['slot']]=np.array(public['mcs'],copy=True)
    kernel.simulate(w,policy(),scan_observer=sensor(w),scan_duration_s=.09,
        causal_central=True,spatial_runtime=runtime,blockage_rate=0.,
        observation_period_slots=10,observation_hook=capture)
    assert np.all(observed[10]==0) # No valid AP1 capture existed at lagged slot0.
    assert np.any(observed[20]>0)  # The legitimate AP1 slot10 capture is now available.
