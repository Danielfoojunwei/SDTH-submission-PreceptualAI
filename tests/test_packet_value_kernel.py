"""Exact native state forks, not approximate counterfactual loss labels."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch

from wirelessjev.closed_loop.sim import simulate as sealed_simulate
from wirelessjev.packet_value.sim import simulate
from wirelessjev.closed_loop.runtime import SinglePredictor, LiveForecasts
from wirelessjev.closed_loop.telemetry import DeliveredScans
from wirelessjev.frontier.study import mask_scans, runtime_world
from wirelessjev.native.world import build_world
from wirelessjev.native.sim import policies
from wirelessjev.fleetwm.data import episode
from wirelessjev.gate_repair.gate import reactive_rescue


def native():
    return {p.name:p for p in policies()}['native']


def test_no_fork_options_preserve_sealed_native_counters():
    torch.set_num_threads(1)
    w=build_world(299040,duration_s=4,n_robots=7)
    for gate in (False,True):
        p=replace(native(),gate=gate,cadence_scan=True,scan_period_s=1.)
        assert simulate(copy.deepcopy(w),p)==sealed_simulate(copy.deepcopy(w),p)


def test_checkpoint_replays_factual_suffix_and_prefix_counters_exactly():
    torch.set_num_threads(1)
    w=build_world(299041,duration_s=6,n_robots=9)
    p=replace(native(),gate=True,cadence_scan=True,scan_period_s=1.,roam_cost_s=4.)
    saved=[]
    whole=simulate(copy.deepcopy(w),p,roam_gate=reactive_rescue,checkpoint_at={200,350},checkpoint_sink=saved.append,detailed=True)
    for checkpoint in saved:
        replay=simulate(w,p,resume=checkpoint,detailed=True)
        replay['packet_detail']['resume_slot']=0
        assert np.all(np.asarray(replay['packet_detail']['window_max_gap'])<=whole['packet_detail']['max_gap'])
        replay['packet_detail']['window_max_gap']=whole['packet_detail']['window_max_gap']
        assert replay==whole
        assert checkpoint.observable()['slot']==checkpoint.slot
    # A resumed branch can stop early without changing hidden-channel RNG draws.
    early=simulate(copy.deepcopy(w),p,roam_gate=reactive_rescue,stop_slot=400,detailed=True)
    resumed=simulate(w,p,resume=saved[0],stop_slot=400,detailed=True)
    resumed['packet_detail']['resume_slot']=0
    resumed['packet_detail']['window_max_gap']=early['packet_detail']['window_max_gap']
    assert resumed==early
    # The checkpoint's physical scenario is authoritative. A caller cannot
    # accidentally mix new noise/channel constants into the restored state.
    changed=copy.deepcopy(w);changed.noise_db+=15.;changed.channel[:]=1
    replay=simulate(changed,p,resume=saved[0],detailed=True)
    replay['packet_detail']['resume_slot']=0
    replay['packet_detail']['window_max_gap']=whole['packet_detail']['window_max_gap']
    assert replay==whole


def test_live_forecast_and_delivered_scan_aliases_survive_exact_state_fork():
    torch.set_num_threads(1)
    full,_,g,o=episode(299042,5,5);o=mask_scans(o,5)
    history=DeliveredScans(o.before(60),5,299042)
    live=LiveForecasts(SinglePredictor(g,'distilled'),history,5,{'cold':.2,'warm':.2,'refit':.2},0.)
    w=runtime_world(full,5,live.mean,live.std[:,:,0])
    p=replace(native(),gate=True,cadence_scan=True,scan_period_s=1.)
    saved=[]
    whole=simulate(w,p,roam_gate=reactive_rescue,scan_observer=history,forecast_source=live,forecast_queue=live,
                   checkpoint_at={200},checkpoint_sink=saved.append,detailed=True)
    replay=simulate(w,p,resume=saved[0],detailed=True)
    replay['packet_detail']['resume_slot']=0
    replay['packet_detail']['window_max_gap']=whole['packet_detail']['window_max_gap']
    assert replay==whole
    assert [x['slot'] for x in saved[0].observation_history]==[0,100,200]
    payload=saved[0].observable()
    assert set(('current','pilot_snr','local_rssi','positions','plan','misses','sent','gap'))<=payload.keys()
    assert 'snr' not in payload and 'blocked' not in payload and 'fade_db' not in payload
    assert payload['forecast_std'].shape==(5,len(g.sites),3)
    payload['current'][:]=-1
    assert (saved[0].observable()['current']>=0).all()


def test_forced_stay_or_switch_share_factual_state_and_charge_actual_outage():
    torch.set_num_threads(1)
    w=build_world(299043,duration_s=7,n_robots=6)
    # Constant generous field isolates the known physical interruption while
    # still executing actual native packet queues/retries/control deadlines.
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=35.
    p=replace(native(),gate=True,cadence_scan=True,scan_period_s=100.,roam_cost_s=4.)
    saved=[]
    simulate(w,p,roam_gate=reactive_rescue,checkpoint_at={100},checkpoint_sink=saved.append)
    cp=saved[0];current=int(cp.observable()['current'][0]);other=(current+1)%len(w.sites)
    stay=simulate(w,p,resume=cp,force_action=(0,current),stop_slot=600,detailed=True)
    switch=simulate(w,p,resume=cp,force_action=(0,other),stop_slot=600,detailed=True)
    assert switch['packet_detail']['misses'][0]-stay['packet_detail']['misses'][0] in (79,80)
    assert switch['roams']==stay['roams']+1
    assert np.array_equal(stay['packet_detail']['misses'][1:],switch['packet_detail']['misses'][1:])
    # Reusing the exact checkpoint is deterministic and cannot mutate it.
    assert switch==simulate(w,p,resume=cp,force_action=(0,other),stop_slot=600,detailed=True)
    with pytest.raises(ValueError):simulate(w,p,force_action=(0,other))
    with pytest.raises(ValueError):simulate(w,replace(p,roam_cost_s=.03),resume=cp)


class DelayedPrimary:
    """Deterministic queued proposals test execution timing and checkpoint state."""
    def __init__(self,n,delay=20,allow=True):
        self.n=n;self.delay=delay;self.allow=allow;self.pending=[];self.events=[];self.observed=[];self.observations=[]

    def observe(self,public):
        self.observed.append((public['slot'],public['forecast_source']))
        frame=np.zeros((self.n,32),dtype='f4');frame[:,0]=public['current'];frame[:,1:4]=public['pilot_snr']
        self.observations.append({'slot':public['slot'],'frame':frame})
        eligible=np.zeros(self.n,dtype=bool)
        eligible[0]=self.allow and public['current'][0]!=0
        self.pending.append((public['slot']+self.delay,{'current':public['current'].copy(),
                           'target':np.zeros(self.n,dtype=int),'eligible':eligible}))

    def take(self,slot):
        ready=[p for p in self.pending if p[0]<=slot]
        self.pending=[p for p in self.pending if p[0]>slot]
        return ready[-1][1] if ready else None

    def executed(self,slot,ids,targets):
        self.events.append((slot,ids.tolist(),targets.tolist()))

    def summary(self):
        return {'issued':len(self.observed),'executed':len(self.events)}


def test_external_primary_is_charged_controls_roaming_and_replays_exactly():
    torch.set_num_threads(1)
    w=build_world(299044,duration_s=5,n_robots=6)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=0.;w.forecast[:,:,:,1]=40.
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,roam_cost_s=.03,cadence_roam=True)
    # The ordinary planner wants AP1. External primary owns that decision path.
    assert simulate(copy.deepcopy(w),p)['roams']>0
    denied=DelayedPrimary(6,allow=False)
    assert simulate(copy.deepcopy(w),p,external_primary=denied)['roams']==0
    primary=DelayedPrimary(6)
    saved=[]
    whole=simulate(copy.deepcopy(w),p,external_primary=primary,checkpoint_at={200},checkpoint_sink=saved.append,detailed=True)
    assert whole['roams']==1 and len(primary.events)==1
    assert 20<=primary.events[0][0]<25  # queued compute plus actual cadence delivery
    assert [x[0] for x in primary.observed]==[0,100,200,300,400]
    assert whole['packet_detail']['controller_summary']=={'issued':5,'executed':1}
    assert len(whole['packet_detail']['controller_frames'])==5
    assert np.asarray(whole['packet_detail']['controller_frames'][-1]['frame']).shape==(6,32)
    replay=simulate(w,p,resume=saved[0],detailed=True)
    replay['packet_detail']['resume_slot']=0
    replay['packet_detail']['window_max_gap']=whole['packet_detail']['window_max_gap']
    assert replay==whole


def test_suffix_gap_maximum_cannot_be_masked_by_a_historical_peak():
    torch.set_num_threads(1)
    w=build_world(299045,duration_s=7,n_robots=5)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=35.
    p=replace(native(),gate=True,cadence_scan=True,scan_period_s=100.,roam_cost_s=4.)
    saved=[]
    simulate(w,p,roam_gate=reactive_rescue,checkpoint_at={100},checkpoint_sink=saved.append)
    cp=saved[0]
    # A carried historical maximum must not substitute for a suffix maximum.
    cp.bundle['state']['gap_max'][0]=1000
    current=int(cp.observable()['current'][0]);other=(current+1)%len(w.sites)
    stay=simulate(w,p,resume=cp,force_action=(0,current),stop_slot=600,detailed=True)['packet_detail']
    switch=simulate(w,p,resume=cp,force_action=(0,other),stop_slot=600,detailed=True)['packet_detail']
    assert stay['max_gap'][0]==switch['max_gap'][0]==1000
    assert switch['window_max_gap'][0]>=400 and stay['window_max_gap'][0]<50


class ExpiringPrimary(DelayedPrimary):
    def __init__(self,n,ttl):
        super().__init__(n);self.ttl=ttl

    def observe(self,public):
        super().observe(public)
        self.pending[-1][1]['valid_until']=np.full(self.n,public['slot']+self.ttl,dtype=int)


def test_cadence_pending_action_expires_before_delayed_delivery_and_replays():
    torch.set_num_threads(1)
    w=build_world(299044,duration_s=3,n_robots=6)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=35.
    # Robot zero cannot deliver control during the first 400 ms. Its head
    # proposal is ready at 200 ms but expires at 300 ms before cadence fires.
    w.snr[:2,0]=-100.
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,roam_cost_s=.03,cadence_roam=True)
    saved=[];primary=ExpiringPrimary(6,30)
    whole=simulate(copy.deepcopy(w),p,external_primary=primary,checkpoint_at={25},checkpoint_sink=saved.append,
                   stop_slot=100,detailed=True)
    cp=saved[0]
    assert cp.observable()['pending_target'][0]==0 and cp.observable()['pending_valid_until'][0]==30
    assert whole['roams']==0 and primary.events==[]
    replay=simulate(w,p,resume=cp,stop_slot=100,detailed=True)
    replay['packet_detail']['resume_slot']=0
    replay['packet_detail']['window_max_gap']=whole['packet_detail']['window_max_gap']
    assert replay==whole
    # With no deadline, the inherited cadence behavior fires after delivery.
    legacy=DelayedPrimary(6)
    assert simulate(copy.deepcopy(w),p,external_primary=legacy,stop_slot=100)['roams']==1
    assert legacy.events[0][0]>=40
    # Jobs already stale at head completion cannot schedule an immediate move.
    stale=ExpiringPrimary(6,10)
    assert simulate(copy.deepcopy(w),replace(p,cadence_roam=False,roam_cost_s=4.),
                    external_primary=stale,stop_slot=100)['roams']==0
    boundary=ExpiringPrimary(6,20)
    assert simulate(copy.deepcopy(w),replace(p,cadence_roam=False,roam_cost_s=4.),
                    external_primary=boundary,stop_slot=100)['roams']==0


def test_actual_packet_controller_and_live_radiomap_replay_together():
    from wirelessjev.packet_value.study import make_case,policy_for
    torch.set_num_threads(1)
    world,g,controller,kwargs=make_case(299030,8,20,.03)
    saved=[]
    whole=simulate(world,policy_for(.03),checkpoint_at={500},checkpoint_sink=saved.append,
                   detailed=True,**kwargs)
    replay=simulate(world,policy_for(.03),resume=saved[0],detailed=True)
    assert {k:v for k,v in whole.items() if k!='packet_detail'}=={k:v for k,v in replay.items() if k!='packet_detail'}
    a=copy.deepcopy(whole['packet_detail']);b=copy.deepcopy(replay['packet_detail'])
    for x in (a,b):
        x.pop('resume_slot');x.pop('window_max_gap')
        summary=x['controller_summary']
        summary.pop('head_prediction_seconds')
        for d in summary['decisions']:d.pop('actual_head_seconds')
    assert a==b
    assert len(a['controller_frames'])==6 and a['controller_summary']['jobs_issued']>0
