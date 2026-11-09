"""Explicit software-only raw receipts; no model, world or packet execution."""
from copy import deepcopy
import math

import numpy as np
import pytest

from wirelessjev.spatial_control.study import spec
from wirelessjev.spatial_control.validation import document_sha256,validate_result,_policy


def fixture(*,arm='radiomap',parity=None):
    p=spec(); case=deepcopy(p['cases'][0]);raw_arm=arm;profile=p['forecast']['profile_seconds']
    if parity is not None:
        case={'id':'disabled-parity','seed':p['parity']['seed'],'handover_seconds':.03}
        arm='radiomap__'+parity;raw_arm='radiomap';profile={k:4. for k in ('cold','warm','refit')}
    source={'protocol':{'path':'artifacts/spatial_control/protocol.json','sha256':document_sha256(p)},
        'checkpoint':{'path':'artifacts/adaptive_radio/checkpoints/regression-seed0.pt','sha256':'a'*64}}
    world={'seed':case['seed'],'source_sha256':document_sha256(source),'protocol_sha256':document_sha256(p),
        'APs':[{},{}],'world':{'path':f"artifacts/spatial_control/worlds/{case['seed']}/world.npz",'sha256':'b'*64},
        'warm':{'path':f"artifacts/spatial_control/worlds/{case['seed']}/warm.npz",'sha256':'c'*64}}
    prefix='artifacts/spatial_control/cases/'+case['id']+'/'+arm+'/'
    receipt={'case':case,'arm':arm,'parity_kernel':parity,'source_sha256':document_sha256(source),
        'protocol_sha256':document_sha256(p),'world_receipt_sha256':document_sha256(world),
        'raw':{'path':prefix+'raw.json.gz','sha256':'d'*64},
        'arrays':{'path':prefix+'predictions_history.npz','sha256':'e'*64},
        'attempt':{'path':prefix+'attempt.json','sha256':'f'*64},'simulations':1,'simulation_only':True,
        'production_qualified':False,'default_changed':False,'original_SERVICE_opened':False}
    arrays={'forecast_mean_snr':np.zeros((20,100,9,2,3),dtype='f4'),
        'forecast_std_db':np.ones((20,100,9,2,3),dtype='f4'),
        'history_positions':np.zeros((80,100,2),dtype='f4'),
        'history_value':np.zeros((80,100,6),dtype='f4'),
        'history_bound':np.full((80,100,6),-90,dtype='f4'),
        'history_heard':np.zeros((80,100,6),dtype=bool),
        'history_scanned':np.zeros((80,100),dtype=bool)}
    import hashlib
    jobs=[];consumption=[];previous=0.
    for j in range(20):
        kind='cold' if j==0 else 'warm';charged=profile[kind];service=charged+.5
        previous=max(float(j),previous)+service;ready=math.ceil(previous*100-1e-10)
        jobs.append({'decision':j,'history_cutoff_exclusive':60+j,'kind':kind,'actual_prediction_seconds':.001,
            'actual_issue_seconds':.002,'actual_compute_seconds':.002,'charged_compute_seconds':charged,
            'profile_seconds':profile[kind],'shared_extra_seconds':.5,'charged_seconds':service,'runtime_overrun':False,
            'predictor_runtime_overrun':False,'completion_uses_actual_elapsed':True,'offline_recorded_schedule':False,
            'accounting_mode':'actual_cost','ready_slot':ready,'forecast_shape':[100,9,2,3],'forecast_dtype':'float32',
            'forecast_sha256':hashlib.sha256(arrays['forecast_mean_snr'][j].tobytes()).hexdigest(),
            'forecast_std_sha256':hashlib.sha256(arrays['forecast_std_db'][j].tobytes()).hexdigest(),
            'heard_entries':0,'scans_last_bin':0})
        if ready<2000:consumption.append({'forecast_decision':j,'slot':ready,'ready_slot':ready,'age_seconds':ready/100-j})
        if raw_arm.startswith('regression'):
            arrays.update({f'job{j}_context':np.zeros((1,24),dtype='f4'),f'job{j}_known':np.ones(1,bool),
                f'job{j}_scheduled':np.ones(1,bool),f'job{j}_selected_rows':np.array([[0,0,0,0]],dtype='i8'),
                f'job{j}_plan':np.zeros((100,9,2),dtype='f4'),f'job{j}_mean_dbm':np.zeros((100,9,2,3),dtype='f4'),
                f'job{j}_std_db':np.ones((100,9,2,3),dtype='f4'),f'job{j}_belief_mean':np.zeros(8,dtype='f4'),
                f'job{j}_belief_covariance':np.eye(8,dtype='f4')})
    detail={'stop_slot':2000,'resume_slot':0,'observation_period_slots':10,'native_proposal_ttl_slots':100,
        'focal_hold_slots':None,'misses':[0]*100,'sent':[400]*100,'gap':[0]*100,'max_gap':[4]*100,'window_max_gap':[4]*100}
    native={'scene':case['seed'],'policy':'spatial_control_diagnostic','duration_s':20,'collision_lambda':0.,
        'interference':False,'packet_detail':detail,'control_misses':0,'control_sent':40000,'control_miss':0.,
        'control_miss_worst_decile':0.,'control_gap_p99_ms':40.}
    recovery=raw_arm in ('radiomap_recovery','regression_recovery','regression_both')
    if parity is None:
        native['spatial_execution']={'causal_central':True,'local_recovery_enabled':recovery,
            'central_creation_delay_slots':10,'central_ttl_slots':100,'scan_minimum_spacing_slots':500,
            'local_decision_floor_slots':1,'central_decision_budget_seconds':.1,'local_decision_budget_seconds':.005,
            'central_decisions':[],'central_events':[],'local_decisions':[],'local_validation_batches':[],
            'local_events':[],'scan_events':[],'initial_scan_start_upper_slot':[-500]*100,
            'pending_packet_age_end':[-1]*100,'pending_control_packets':0}
    result={'arm':raw_arm,'seed':case['seed'],'handover_seconds':.03,'causal_central':parity is None,
        'production_qualified':False,'policy_config':_policy(.03),'simulation_config':{'seed':0,'duration_seconds':20,
        'collision_lambda':0.,'interference':False,'scan_duration_seconds':.09,'observation_period_slots':10,
        'native_proposal_ttl_slots':100,'planner':'native'},'native':native,'forecast_jobs':jobs,
        'forecast_consumption':consumption,'local_runtime':{} if recovery else None}
    return receipt,result,arrays,source,world,p


@pytest.mark.parametrize('arm',['radiomap','radiomap_recovery','regression','regression_support','regression_recovery','regression_both'])
def test_exact_software_fixture_all_arms(arm):
    assert validate_result(*fixture(arm=arm))['passes']


def test_live_producer_asdict_policy_tuple_before_json_serialization():
    from dataclasses import asdict
    from wirelessjev.spatial_control.kernel import Policy
    values=fixture()
    values[1]['policy_config']=asdict(Policy('spatial_control_diagnostic','native',
        roam_cost_s=.03,scan_period_s=5.,gate=True,cadence_scan=True,
        cadence_roam=True,online_map=False,dual_radio=False))
    assert values[1]['policy_config']['focus_robots']==()
    assert validate_result(*values)['passes']


@pytest.mark.parametrize('parity',['original','disabled'])
def test_disabled_parity_has_its_distinct_identity_and_four_second_profile(parity):
    assert validate_result(*fixture(parity=parity))['causal_central'] is False


@pytest.mark.parametrize('mutate',[
    lambda r,o,a,s,w,p:o.update(handover_seconds=4.),
    lambda r,o,a,s,w,p:o['policy_config'].update(roam_cost_s=4.),
    lambda r,o,a,s,w,p:o['simulation_config'].update(seed=1),
    lambda r,o,a,s,w,p:o['simulation_config'].update(seed=False),
    lambda r,o,a,s,w,p:r['case'].update(handover_seconds=4.),
    lambda r,o,a,s,w,p:r.update(source_sha256='0'*64),
    lambda r,o,a,s,w,p:r.update(world_receipt_sha256='0'*64),
    lambda r,o,a,s,w,p:r['arrays'].update(path='artifacts/other.npz'),
    lambda r,o,a,s,w,p:o['forecast_jobs'][1].update(profile_seconds=.5),
    lambda r,o,a,s,w,p:o['forecast_jobs'][1].update(kind='refit'),
    lambda r,o,a,s,w,p:o['forecast_jobs'].pop(),
    lambda r,o,a,s,w,p:o['forecast_jobs'][1].update(forecast_sha256='0'*64),
    lambda r,o,a,s,w,p:a['forecast_mean_snr'].__setitem__((1,0,0,0,0),1.),
    lambda r,o,a,s,w,p:o['forecast_jobs'][1].update(ready_slot=1),
    lambda r,o,a,s,w,p:o['forecast_consumption'][0].update(age_seconds=0),
    lambda r,o,a,s,w,p:a.update(unregistered_forward=np.zeros(1)),
    lambda r,o,a,s,w,p:o['native'].update(duration_s=True),
    lambda r,o,a,s,w,p:r.update(production_qualified=True),
    lambda r,o,a,s,w,p:o['native']['spatial_execution'].update(pending_control_packets=2),
    lambda r,o,a,s,w,p:o['native']['packet_detail']['sent'].__setitem__(0,399),
])
def test_rehashed_wrong_identity_clock_profile_or_array_is_refused(mutate):
    values=fixture();mutate(*values)
    with pytest.raises(ValueError):validate_result(*values)


def central_fixture():
    values=fixture();_,o,_,_,_,_=values;s=o['native']['spatial_execution'];clock=o['forecast_consumption'][0]['slot']
    s['central_decisions']=[{'slot':clock,'forecast_decision':0,'elapsed_seconds':.01,'budget_seconds':.1,
        'overrun':False,'created_slot':clock+10,'compute_ready_slot':clock+10}]
    command={'command_id':'central-1','robot':0,'current':0,'target':1,'proposed_slot':clock,
        'created_slot':clock+10,'expires_slot':clock+110,'interruption_slots':3}
    s['central_events']=[{'event':'queued','slot':clock,**command},
        {'event':'executed','slot':clock+12,**command,'packet_created_slot':clock+11,
         'delivery_slot':clock+12,'execution_slot':clock+12,'outage_until_slot':clock+16}]
    return values


def test_causal_command_can_use_robot_phased_packet_not_global_modulo_five():
    values=central_fixture()
    assert values[1]['native']['spatial_execution']['central_events'][1]['packet_created_slot']%5!=0
    assert validate_result(*values)['central_commands']==1


@pytest.mark.parametrize('change',[
    lambda e:e.update(packet_created_slot=e['created_slot']-1),
    lambda e:e.update(delivery_slot=e['created_slot']),
    lambda e:e.update(outage_until_slot=e['outage_until_slot']-1),
    lambda e:e.update(interruption_slots=0),
    lambda e:e.update(expires_slot=e['expires_slot']+1),
    lambda e:e.update(command_id='unbound'),
])
def test_central_lifecycle_requires_own_delivery_and_full_cost(change):
    values=central_fixture();change(values[1]['native']['spatial_execution']['central_events'][1])
    with pytest.raises(ValueError):validate_result(*values)


def local_fixture():
    values=fixture(arm='radiomap_recovery');s=values[1]['native']['spatial_execution']
    request={'kind':'move','robot':0,'target':1,'action_id':'local-0-1'}
    queued={**request,'request_slot':20,'ready_slot':21,'current':0}
    s['local_decisions']=[{'slot':20,'elapsed_seconds':.001,'budget_seconds':.005,'overrun':False,
        'ready_slot':21,'requests':[request]}]
    s['local_validation_batches']=[{'slot':21,'elapsed_seconds':.001,'budget_seconds':.005,'overrun':False,
        'checks':[{'action_id':'local-0-1','robot':0,'elapsed_seconds':.001,'evidence_valid':True}]}]
    s['local_events']=[{'event':'queued','slot':20,**queued},{'event':'executed','slot':21,**queued,
        'outage_until_slot':25,'interruption_slots':3,'last_delivery_slot':0}]
    return values


def test_local_execution_needs_paid_delay_validation_and_full_interruption():
    assert validate_result(*local_fixture())['local_requests']==1


@pytest.mark.parametrize('change',[
    lambda s:s['local_events'][1].update(slot=20),
    lambda s:s['local_events'][1].update(outage_until_slot=24),
    lambda s:s['local_events'][1].update(last_delivery_slot=20),
    lambda s:s['local_validation_batches'][0]['checks'][0].update(evidence_valid=False),
    lambda s:s['local_decisions'][0].update(overrun=True),
])
def test_local_requests_cannot_bypass_delay_cpu_or_silence(change):
    values=local_fixture();change(values[1]['native']['spatial_execution'])
    with pytest.raises(ValueError):validate_result(*values)


@pytest.mark.parametrize('change',[
    lambda s:s['scan_events'][0].update(complete_slot=9),
    lambda s:s['scan_events'][0].update(interruption_slots=8),
    lambda s:s['scan_events'].append(dict(s['scan_events'][0],slot=1,measurement_slot=1,interruption_start_slot=2,complete_slot=11)),
])
def test_scans_pay_ninety_ms_and_shared_five_second_spacing(change):
    values=fixture();s=values[1]['native']['spatial_execution']
    s['scan_events']=[{'robot':0,'slot':0,'measurement_slot':0,'interruption_start_slot':1,'complete_slot':10,
        'interruption_slots':9,'reason':'background_after_delivery','action_id':None}]
    assert validate_result(*values)['passes']
    change(s)
    with pytest.raises(ValueError):validate_result(*values)


def test_encoder_rows_are_four_coordinates_with_strict_past_and_source_bands():
    values=fixture(arm='regression');values[2]['job0_selected_rows'][0,0]=60
    with pytest.raises(ValueError):validate_result(*values)


def test_validation_does_not_mutate_software_evidence():
    values=central_fixture();before=deepcopy(values)
    validate_result(*values)
    assert values[:2]==before[:2] and values[3:]==before[3:]
    assert all(np.array_equal(values[2][name],value) for name,value in before[2].items())
