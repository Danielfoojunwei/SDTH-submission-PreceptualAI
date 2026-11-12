"""Native continuation, receiver observability and exact paired interventions."""
import copy
from dataclasses import replace
import numpy as np
import pytest
import torch
from wirelessjev.native.world import build_world
from wirelessjev.native.sim import policies
from wirelessjev.packet_value.sim import simulate as predecessor
from wirelessjev.service_world.sim import simulate
from wirelessjev.gate_repair.gate import reactive_rescue


def native():
    return {p.name:p for p in policies()}['native']


class PublicRecorder:
    def __init__(self):
        self.observations=[]

    def observe(self,public):
        frame=np.stack([public['current'],public['pilot_snr'][:,0],
                        public['misses'],public['sent']],-1)
        self.observations.append({'slot':public['slot'],'frame':frame})


class PassivePrimary:
    def __init__(self):
        self.observations=[]

    def observe(self,public):
        self.observations.append({'slot':public['slot'],'frame':public['current'][:,None].copy()})

    def take(self,slot):return None
    def executed(self,slot,ids,targets):raise AssertionError('No action was proposed')


def without_suffix_metadata(result):
    result=copy.deepcopy(result)
    result['packet_detail'].pop('resume_slot')
    result['packet_detail'].pop('window_max_gap')
    return result


def test_default_options_preserve_predecessor_native_packet_counters():
    torch.set_num_threads(1)
    w=build_world(299050,duration_s=4,n_robots=7)
    for gate in (False,True):
        p=replace(native(),gate=gate,cadence_scan=True,scan_period_s=1.)
        assert simulate(copy.deepcopy(w),p)==predecessor(copy.deepcopy(w),p)
        assert simulate(copy.deepcopy(w),p,detailed=True)==predecessor(copy.deepcopy(w),p,detailed=True)


def test_100ms_observer_is_causal_and_external_primary_stays_at_one_hz():
    torch.set_num_threads(1)
    w=build_world(299051,duration_s=3,n_robots=4)
    p=replace(native(),cadence_scan=True,scan_period_s=100.)
    recorder=PublicRecorder();primary=PassivePrimary();saved=[]
    whole=simulate(w,p,observation_hook=recorder.observe,external_primary=primary,
                   observation_period_slots=10,checkpoint_at={55,100},checkpoint_sink=saved.append,detailed=True)
    assert [x['slot'] for x in recorder.observations]==list(range(0,300,10))
    assert [x['slot'] for x in primary.observations]==[0,100,200]
    for cp in saved:
        public=cp.observable()
        assert public['pilot_snr'].shape==(4,3)
        assert public['pilot_valid'].shape==(4,3)
        assert public['mcs'].shape==(4,3)
        assert public['outcome_history'].shape==(100,4,2)
        valid=public['outcome_valid']
        assert (public['outcome_slot'][valid]<cp.slot).all()
        assert (public['outcome_slot'][~valid]==-1).all()
        assert int(valid.sum())==min(cp.slot,100)
        assert not {'snr','fade_db','blocked','true_snr'} & public.keys()
        replay=simulate(w,p,resume=cp,detailed=True)
        assert without_suffix_metadata(replay)==without_suffix_metadata(whole)
        with pytest.raises(ValueError):simulate(w,p,resume=cp,observation_period_slots=100)
    assert len(whole['packet_detail']['observation_frames'])==30


def test_forced_stay_releases_to_canonical_recovery_instead_of_ten_seconds():
    torch.set_num_threads(1)
    w=build_world(299052,duration_s=4,n_robots=4)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=0.
    w.forecast[0,:,:,0]=40.;w.forecast[1:,:,:,1]=40.
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,stagger=False)
    recorder=PublicRecorder();saved=[]
    simulate(w,p,observation_hook=recorder.observe,observation_period_slots=10,
             checkpoint_at={100},checkpoint_sink=saved.append,stop_slot=101,detailed=True)
    cp=saved[0];current=int(cp.observable()['current'][0]);assert current==0
    factual=simulate(w,p,resume=cp,stop_slot=400,detailed=True)
    stay=simulate(w,p,resume=cp,force_action=(0,current),focal_hold_slots=100,
                  stop_slot=400,detailed=True)
    permanent=simulate(w,p,resume=cp,force_action=(0,current),stop_slot=400,detailed=True)
    def current_at(result,slot):
        row=next(r for r in result['packet_detail']['observation_frames'] if r['slot']==slot)
        return int(row['frame'][0][0])
    assert current_at(factual,110)==1
    assert current_at(stay,190)==0 and current_at(stay,210)==1
    assert current_at(permanent,310)==0
    assert stay['roams']==factual['roams'] and permanent['roams']==factual['roams']-1


def test_resuming_inside_a_short_intervention_preserves_its_release():
    torch.set_num_threads(1)
    w=build_world(299053,duration_s=4,n_robots=4)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=0.
    w.forecast[0,:,:,0]=40.;w.forecast[1:,:,:,1]=40.
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,stagger=False)
    recorder=PublicRecorder();saved=[]
    simulate(w,p,observation_hook=recorder.observe,observation_period_slots=10,
             checkpoint_at={100},checkpoint_sink=saved.append,stop_slot=101,detailed=True)
    nested=[]
    branch=simulate(w,p,resume=saved[0],force_action=(0,0),focal_hold_slots=100,
                    checkpoint_at={150},checkpoint_sink=nested.append,detailed=True)
    cp=nested[0]
    assert cp.bundle['locked_robot']==0 and cp.bundle['locked_until']==200
    replay=simulate(w,p,resume=cp,detailed=True)
    assert without_suffix_metadata(replay)==without_suffix_metadata(branch)


def test_unavailable_receiver_cannot_refresh_pilots_or_measure_a_new_ap():
    torch.set_num_threads(1)
    w=build_world(299054,duration_s=3,n_robots=3)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.forecast[:]=0.;w.forecast[:,:,:,0]=40.
    w.rssi_dl[:,:,0]=0.  # start on AP0; do not consume the test in an initial roam
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,stagger=False,roam_cost_s=4.)
    recorder=PublicRecorder();saved=[]
    simulate(w,p,observation_hook=recorder.observe,observation_period_slots=10,
             checkpoint_at={10},checkpoint_sink=saved.append,stop_slot=11,detailed=True)
    before=saved[0].observable();assert before['current'][0]==0
    after=[]
    simulate(w,p,resume=saved[0],force_action=(0,1),focal_hold_slots=100,
             checkpoint_at={200},checkpoint_sink=after.append,stop_slot=201,detailed=True)
    public=after[0].observable()
    assert public['current'][0]==1 and public['pilot_site'][0]==0
    assert not public['pilot_valid'][0].any() and public['outage_remaining'][0]>0
    np.testing.assert_equal(public['pilot_snr'][0],before['pilot_snr'][0])
    assert public['pilot_age_slots'][0]>before['pilot_age_slots'][0]
    assert public['outcome_slot'][public['outcome_valid']].max()<public['slot']


def test_irregular_observation_clock_does_not_repeat_one_hz_controller_on_resume():
    torch.set_num_threads(1)
    w=build_world(299055,duration_s=3,n_robots=3)
    p=replace(native(),cadence_scan=True,scan_period_s=100.)
    recorder=PublicRecorder();primary=PassivePrimary();saved=[]
    whole=simulate(w,p,observation_hook=recorder.observe,external_primary=primary,
                   observation_period_slots=7,checkpoint_at={100},checkpoint_sink=saved.append,detailed=True)
    assert [row['slot'] for row in primary.observations]==[0,100,200]
    replay=simulate(w,p,resume=saved[0],detailed=True)
    assert without_suffix_metadata(replay)==without_suffix_metadata(whole)


def test_invalid_hold_or_observation_period_is_rejected():
    w=build_world(299056,duration_s=1,n_robots=2);p=native()
    for value in (0,-1,True,1.5):
        with pytest.raises(ValueError):simulate(w,p,observation_period_slots=value)
        with pytest.raises(ValueError):simulate(w,p,focal_hold_slots=value)
        with pytest.raises(ValueError):simulate(w,p,native_proposal_ttl_slots=value)


def test_canonical_native_proposal_expires_before_blocked_delivery_and_replays():
    torch.set_num_threads(1)
    w=build_world(299057,duration_s=3,n_robots=4)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.rssi_dl[:,:,0]=0.
    w.forecast[:]=0.;w.forecast[:,:,:,0]=40.
    w.forecast[0,0,:,0]=0.;w.forecast[0,0,:,1]=40.
    # The initial canonical proposal wants AP1, but subsequent forecasts favor
    # the current AP0. Delivery remains impossible until after the one-second
    # proposal deadline; an old, unbounded queued move would execute at 1.6 s.
    w.snr[:8,0]=-100.
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,
              roam_cost_s=.03,cadence_roam=True,stagger=False,load_penalty=0.)
    saved=[]
    whole=simulate(copy.deepcopy(w),p,native_proposal_ttl_slots=100,
                   checkpoint_at={25,100,170},checkpoint_sink=saved.append,
                   stop_slot=250,detailed=True)
    before,expiry,after=saved
    assert before.observable()['pending_target'][0]==1
    assert before.observable()['pending_valid_until'][0]==100
    assert expiry.observable()['pending_target'][0]==-1
    assert after.observable()['current'][0]==0
    assert whole['roams']==0
    assert before.config['native_proposal_ttl_slots']==100
    replay=simulate(w,p,resume=before,stop_slot=250,detailed=True)
    assert without_suffix_metadata(replay)==without_suffix_metadata(whole)
    with pytest.raises(ValueError):
        simulate(w,p,resume=before,stop_slot=250,native_proposal_ttl_slots=101)
    # With the option absent, preserve the predecessor's unbounded command.
    unbounded=simulate(copy.deepcopy(w),p,stop_slot=250)
    assert unbounded['roams']>0


@pytest.mark.parametrize('ttl,expected_expiry',[(30,55),(200,125)])
def test_forced_native_proposal_uses_earliest_ttl_and_explicit_deadline(ttl,expected_expiry):
    torch.set_num_threads(1)
    w=build_world(299058,duration_s=3,n_robots=3)
    w.snr[:]=35.;w.rssi_dl[:]=-35.;w.rssi_dl[:,:,0]=0.
    w.forecast[:]=0.;w.forecast[:,:,:,0]=40.;w.snr[:8,0]=-100.
    p=replace(native(),gate=False,cadence_scan=True,scan_period_s=100.,
              roam_cost_s=.03,cadence_roam=True,stagger=False)
    prefix=[]
    simulate(w,p,native_proposal_ttl_slots=ttl,checkpoint_at={25},
             checkpoint_sink=prefix.append,stop_slot=26)
    branch=[]
    moved=simulate(w,p,resume=prefix[0],force_action=(0,1),focal_hold_slots=100,
                   checkpoint_at={26},checkpoint_sink=branch.append,
                   stop_slot=250,detailed=True)
    assert branch[0].observable()['pending_target'][0]==1
    assert branch[0].observable()['pending_valid_until'][0]==expected_expiry
    assert moved['roams']==0
