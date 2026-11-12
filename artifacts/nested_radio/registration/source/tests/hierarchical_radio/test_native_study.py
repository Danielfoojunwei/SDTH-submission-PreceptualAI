"""Inert native-runner software fixtures; never scientific evidence or worlds."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import math
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.hierarchical_radio import common as c, native_study as n


def fixture_result(*, parity=False, method='full_radiomap'):
    """Synthetic clock/counter document, without invoking a predictor/simulator."""
    sites=16;case=n.native_spec()['parity_case'] if parity else n.native_spec()['cases'][0]
    shape=(20,100,9,sites,3)
    saved={'forecast_mean_snr':np.zeros(shape,dtype='f4'),'forecast_std_db':np.ones(shape,dtype='f4'),
        'history_positions':np.zeros((80,100,2),dtype='f4'),
        'history_value':np.zeros((80,100,sites*3),dtype='f4'),
        'history_heard':np.zeros((80,100,sites*3),bool),
        'history_bound':np.full((80,100,sites*3),-90.,dtype='f4'),
        'history_scanned':np.zeros((80,100),bool)}
    saved['history_scanned'][55]=True;saved['history_heard'][55]=True;saved['history_value'][55]=-70.
    jobs=[];previous=0.
    for j in range(20):
        kind='cold' if j==0 else 'warm';profile=4. if parity else (1. if j==0 else .25)
        previous=max(float(j),previous)+profile+.5
        jobs.append({'decision':j,'history_cutoff_exclusive':60+j,'kind':kind,'ready_slot':math.ceil(previous*100-1e-10),
            'actual_prediction_seconds':.001,'actual_issue_seconds':.002,'actual_compute_seconds':.002,
            'charged_compute_seconds':profile,'profile_seconds':profile,'shared_extra_seconds':.5,
            'charged_seconds':profile+.5,'runtime_overrun':False,'predictor_runtime_overrun':False,
            'completion_uses_actual_elapsed':True,'offline_recorded_schedule':False,'accounting_mode':'actual_cost',
            'forecast_shape':list(shape[1:]),'forecast_dtype':'float32',
            'forecast_sha256':n._sha(saved['forecast_mean_snr'][j]),
            'forecast_std_sha256':n._sha(saved['forecast_std_db'][j]),'heard_entries':0,'scans_last_bin':0})
    spatial={'causal_central':True,'local_recovery_enabled':False,'local_events':[],'local_decisions':[],
        'local_validation_batches':[],'central_creation_delay_slots':10,'central_ttl_slots':100,
        'scan_minimum_spacing_slots':500,'local_decision_floor_slots':1,'central_decision_budget_seconds':.1,
        'central_decisions':[],'central_events':[],'scan_events':[],'initial_scan_start_upper_slot':[-500]*100,
        'pending_packet_age_end':[-1]*100,'pending_control_packets':0,'central_compute_events':[],
        'local_completed_payloads':[]}
    detail={'misses':[1]*100,'sent':[400]*100,'gap':[0]*100,'max_gap':[2]*100,'window_max_gap':[2]*100,
        'stop_slot':2000,'resume_slot':0,'observation_period_slots':10,'native_proposal_ttl_slots':100,
        'focal_hold_slots':None,'delivered_report_events':[]}
    native={'policy':'hierarchical_radio_transfer','scene':case['seed'],'duration_s':20.,'collision_lambda':0.,
        'interference':False,'packet_detail':detail,'spatial_execution':spatial,'control_misses':100,'control_sent':40000,
        'control_miss':.0025,'control_miss_worst_decile':.0025,'control_gap_p99_ms':20.,'video_on_time':.8,
        'bulk_mbps':1.,'roams':0,'safety_stop_fraction':0.}
    result={'method':method,'seed':case['seed'],'handover_seconds':case['handover_seconds'],'parity':parity,
        'native':native,'forecast_jobs':jobs,'forecast_consumption':[],
        'telemetry':{'events':[],'reports_delivered':0,'scans_started':0},'local_completed_payloads':[],
        'scan_started_events':[],'policy_config':asdict(n._policy(case['handover_seconds'])),
        'simulation_config':n._simulation_config(),'causal_central':True,'production_qualified':False,
        'local_runtime':None,'support_enabled':False,'support':[],'support_refresh':[],
        'support_preparation':[],'model_records':[]}
    public={'sites':np.column_stack((np.arange(sites)*6.+10,np.full(sites,10.))).tolist(),
        'tx':[20.,20.,20.],'noise':[-94.,-87.,-84.],'yard':False}
    world={'APs':[{'software_fixture_AP':i} for i in range(sites)],'public_geometry':public}
    receipt={'case':case,'method':method,'parity':parity}
    return receipt,result,saved,world


def validate_fixture(receipt,result,saved,world):
    warm={name:saved['history_'+name][:60] for name in n.FIELDS}
    return n.validate_result(receipt,result,saved,world,warm_arrays=warm)


def test_fixed_native_membership_is_exactly_32_plus_two():
    p=n.native_spec();assert len(p['cases'])==8
    assert {case['seed'] for case in p['cases']}=={90512001,90512002,90512003,90512004}
    assert len({(case['id'],method) for case in p['cases'] for method in case['order']})==32
    assert all(set(case['order'])==set(n.METHODS) for case in p['cases'])
    assert len(n.PARITY_METHODS)==2 and p['maximum_simulations']==34


@pytest.mark.parametrize('kind',['old_seed','new_seed','extra_method','wrong_ho','extra_field','bool_parity'])
def test_admission_rejects_every_unregistered_identity(kind):
    case=deepcopy(n.native_spec()['cases'][0]);method='hierarchy';parity=False
    if kind=='old_seed':case['seed']=90511001
    elif kind=='new_seed':case['seed']=90512005
    elif kind=='extra_method':method='hierarchy_local_recovery'
    elif kind=='wrong_ho':case['handover_seconds']=.06
    elif kind=='extra_field':case['retry']=True
    else:parity=1
    with pytest.raises(ValueError):n.admit_case(case,method,parity=parity)


def test_publication_guard_runs_before_any_missing_receipt_or_native_world(monkeypatch):
    calls=[]
    def deny():calls.append('publication');raise RuntimeError('not published')
    monkeypatch.setattr(c,'check_sources',deny)
    import wirelessjev.fleetwm.data as fleet
    monkeypatch.setattr(fleet,'episode',lambda *a,**k:pytest.fail('Fresh episode called before authorization'))
    with pytest.raises(RuntimeError,match='not published'):n.prepare_world(c.NATIVE_SEEDS[0])
    with pytest.raises(RuntimeError,match='not published'):n.run_native()
    assert calls==['publication','publication']


@pytest.mark.parametrize('present',['none','selection','field'])
def test_completed_selection_and_field_report_are_both_required(tmp_path,monkeypatch,present):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'check_sources',lambda:None)
    if present=='selection':(tmp_path/'selection.json').write_text('{}')
    elif present=='field':(tmp_path/'field_report.json').write_text('{}')
    with pytest.raises(RuntimeError,match='must precede'):n._prerequisites()


def test_inert_valid_full_map_document_passes_and_metrics_use_raw_counters():
    receipt,result,saved,world=fixture_result()
    proof=validate_fixture(receipt,result,saved,world)
    assert proof['passes'] and proof['jobs']==20 and proof['causal_central']
    values=n.metrics(result)
    assert values['control_misses']==100 and values['control_sent']==40000
    assert values['observed_plus_pending_miss_fraction']==.0025
    assert values['scan_interruption_seconds']==0.


@pytest.mark.parametrize('kind',['early_ready','undercharge','false_overrun','future_history','wrong_forecast_hash',
    'early_consumption','wrong_policy','wrong_packet_count','negative_deviation','extra_arrays',
    'local_authority','unpaid_scan','pending_bound','unexpected_support','mismatched_sensor'])
def test_rehashed_semantically_wrong_documents_are_rejected(kind):
    receipt,result,saved,world=fixture_result()
    if kind=='early_ready':result['forecast_jobs'][0]['ready_slot']-=1
    elif kind=='undercharge':result['forecast_jobs'][0]['charged_compute_seconds']=.25
    elif kind=='false_overrun':result['forecast_jobs'][0]['runtime_overrun']=True
    elif kind=='future_history':result['forecast_jobs'][0]['history_cutoff_exclusive']=61
    elif kind=='wrong_forecast_hash':result['forecast_jobs'][1]['forecast_sha256']='f'*64
    elif kind=='early_consumption':result['forecast_consumption']=[{'forecast_decision':0,'slot':149,'ready_slot':150,'age_seconds':1.49}]
    elif kind=='wrong_policy':result['policy_config']['dual_radio']=True
    elif kind=='wrong_packet_count':result['native']['control_misses']=101
    elif kind=='negative_deviation':saved['forecast_std_db'][0,0,0,0,0]=-1.
    elif kind=='extra_arrays':saved['unregistered_private_labels']=np.zeros(1)
    elif kind=='local_authority':result['native']['spatial_execution']['local_recovery_enabled']=True
    elif kind=='unpaid_scan':result['scan_started_events']=[{'robot':0,'slot':0,'complete_slot':10}]
    elif kind=='pending_bound':result['native']['spatial_execution']['pending_control_packets']=1
    elif kind=='unexpected_support':result['support_enabled']=True
    else:result['native']['packet_detail']['delivered_report_events']=[{'robot':0}]
    with pytest.raises(ValueError):validate_fixture(receipt,result,saved,world)


def test_parity_uses_four_second_jobs_without_releasing_unfinished_forecasts():
    receipt,result,saved,world=fixture_result(parity=True)
    assert validate_fixture(receipt,result,saved,world)['passes']
    assert [j['ready_slot'] for j in result['forecast_jobs'][:3]]==[450,900,1350]
    result['forecast_jobs'][1]['profile_seconds']=.25
    with pytest.raises(ValueError):validate_fixture(receipt,result,saved,world)


def test_undercharged_measured_overrun_fails_even_with_a_recomputed_hash():
    receipt,result,saved,world=fixture_result()
    result['forecast_jobs'][0]['actual_issue_seconds']=2.
    with pytest.raises(ValueError,match='cost accounting'):validate_fixture(receipt,result,saved,world)


def test_central_execution_must_follow_a_packet_created_after_command_creation():
    receipt,result,saved,world=fixture_result();spatial=result['native']['spatial_execution']
    result['forecast_consumption']=[{'forecast_decision':0,'slot':150,'ready_slot':150,'age_seconds':1.5}]
    spatial['central_decisions']=[{'slot':150,'forecast_decision':0,'elapsed_seconds':.01,'budget_seconds':.1,
        'overrun':False,'created_slot':160,'compute_ready_slot':160}]
    base={'command_id':'fixture0','robot':0,'current':0,'target':1,'proposed_slot':150,
        'created_slot':160,'expires_slot':260,'interruption_slots':3}
    spatial['central_events']=[dict(base,event='queued',slot=150),dict(base,event='executed',slot=161,
        delivery_slot=161,execution_slot=161,packet_created_slot=160,outage_until_slot=165)]
    assert validate_fixture(receipt,result,saved,world)['central_commands']==1
    spatial['central_events'][1]['packet_created_slot']=159
    with pytest.raises(ValueError,match='causal paid packet'):validate_fixture(receipt,result,saved,world)


def test_worker_requires_exclusive_once_only_start_before_any_predictor(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'artifacts/hierarchical_radio')
    binding={'fixture':'not research'};monkeypatch.setattr(n,'_prerequisites',lambda:binding)
    case=n.native_spec()['cases'][0];folder=n._case_path(case,'hierarchy')
    world_path=n._world_path(case['seed'])/'receipt.json';c.write(world_path,{'fixture':True})
    identity=n._identity(case,'hierarchy',False,binding)
    c.write(folder/'attempt.json',{'identity':identity,'created_unix_ns':1})
    c.write(folder/'started.json',{'already_started':True})
    with pytest.raises(FileExistsError):n._run_arm(case,'hierarchy',None,None,None,binding)


def test_parity_admission_does_not_consume_a_fresh_case_identity():
    case=n.native_spec()['parity_case']
    assert n.admit_case(case,'radiomap_roundtrip',parity=True)==case
    with pytest.raises(ValueError):n.admit_case(case,'hierarchy',parity=True)
    with pytest.raises(ValueError):n.admit_case(case,'full_radiomap',parity=False)


def archive_fixture(tmp_path,monkeypatch):
    """Write one explicitly synthetic archive inside pytest's temporary root."""
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'artifacts/hierarchical_radio')
    binding={'software_fixture_only':True};monkeypatch.setattr(n,'_prerequisites',lambda:binding)
    receipt,result,saved,world=fixture_result();case=receipt['case'];seed=case['seed']
    folder=n._world_path(seed);public=world['public_geometry']
    physical={key:np.zeros(1,dtype='f4') for key in n.WORLD_FIELDS}
    physical.update(sites=np.asarray(public['sites']),tx_dl=np.asarray(public['tx']),noise_db=np.asarray(public['noise']))
    warm={key:saved['history_'+key][:60] for key in n.FIELDS}
    c.attempt(folder/'attempt.json',{'seed':seed,'binding':binding})
    world.update(seed=seed,robots=100,duration_seconds=20,binding=binding,
        attempt=c.reference(folder/'attempt.json'),world=c.npz(folder/'world.npz',physical),
        warm=c.npz(folder/'warm.npz',warm),world_array_sha256={k:c.array_digest(v) for k,v in physical.items()},
        warm_array_sha256={k:c.array_digest(v) for k,v in warm.items()})
    c.write(folder/'receipt.json',world)
    identity=n._identity(case,receipt['method'],False,binding);output=n._case_path(case,receipt['method'])
    c.attempt(output/'attempt.json',{'identity':identity})
    c.attempt(output/'started.json',{'attempt':c.reference(output/'attempt.json')})
    c.write_bytes(output/'raw.json.gz',gzip.compress(c.encoded(result),mtime=0))
    receipt.update(identity,identity=identity,simulations=1,simulation_only=True,default_changed=False,
        original_SERVICE_opened=False,production_qualified=False,
        arrays=c.npz(output/'predictions_history.npz',saved),raw=c.reference(output/'raw.json.gz'),
        attempt=c.reference(output/'attempt.json'),started=c.reference(output/'started.json'),metrics=n.metrics(result))
    receipt['canonical_validation']=n.validate_result(receipt,result,saved,world)
    c.write(output/'receipt.json',receipt)
    return receipt,result,output


def test_completed_archive_is_passively_verified_without_any_forward_or_new_world(tmp_path,monkeypatch):
    receipt,result,_=archive_fixture(tmp_path,monkeypatch)
    import wirelessjev.fleetwm.data as fleet
    import wirelessjev.hierarchical_radio.training as training
    monkeypatch.setattr(fleet,'episode',lambda *a,**k:pytest.fail('Passive verification generated an episode'))
    monkeypatch.setattr(training,'load',lambda *a,**k:pytest.fail('Passive verification loaded a model'))
    monkeypatch.setattr(n,'_run_arm',lambda *a,**k:pytest.fail('Passive verification reran an arm'))
    restored=n.load_result(receipt)
    assert c.encoded(restored)==c.encoded(result)


def test_rehashing_wrong_raw_clock_cannot_make_an_existing_archive_reusable(tmp_path,monkeypatch):
    receipt,result,folder=archive_fixture(tmp_path,monkeypatch)
    result['forecast_jobs'][0]['ready_slot']=149
    (folder/'raw.json.gz').write_bytes(gzip.compress(c.encoded(result),mtime=0))
    receipt['raw']=c.reference(folder/'raw.json.gz')
    with pytest.raises(ValueError,match='FIFO'):n.load_result(receipt)


def test_rehashing_an_extra_attempt_field_does_not_change_registered_identity(tmp_path,monkeypatch):
    receipt,_,folder=archive_fixture(tmp_path,monkeypatch)
    attempt=c.read(folder/'attempt.json');attempt['retry']=2
    (folder/'attempt.json').write_bytes(c.encoded(attempt));receipt['attempt']=c.reference(folder/'attempt.json')
    with pytest.raises(ValueError,match='durable attempt'):n.load_result(receipt)


def test_passive_matched_control_reconstructs_actual_reports_and_queries():
    from wirelessjev.fleetwm.data import Observations
    from wirelessjev.spatial_control.memory import native_history_memory
    from wirelessjev.native_bridge.predictor import radio_assumptions
    from wirelessjev.hierarchical_radio.native import select_context,make_query
    receipt,result,saved,world=fixture_result(method='matched_radiomap');result['support_enabled']=True
    public=world['public_geometry'];geometry=SimpleNamespace(**{k:np.asarray(public[k]) for k in ('sites','tx','noise')},yard=False)
    for j in range(20):
        past=Observations(*(saved['history_'+name][:60+j] for name in n.FIELDS));plan=np.zeros((100,9,2))
        selected=select_context(past,geometry,plan,issue_time_s=float(j));view=selected['view'];meta=selected['metadata']
        query,shape=make_query(geometry,plan);mean=np.broadcast_to(np.asarray(public['noise'])+5.,shape).copy()
        record={'context':view['context'],'known':view['known'],'scheduled':view['scheduled'],
            'selected_rows':meta['selected_rows'],'selected_scan_rows':meta['selected_scan_rows'],
            'measurement_intervals_s':meta['measurement_intervals_s'],'arrival_intervals_s':meta['arrival_intervals_s'],
            'clock_provenance':meta['clock_provenance'],'context_age_s':meta['conservative_age_s'],
            'context_ap':view['ap'],'context_rx_xyz':view['rx_xyz'],'context_lower':view['lower'],
            'context_upper':view['upper'],'context_floor':view['floor'],'context_fspl':view['fspl'],
            'context_offset':view['offset'],'ap_xyz':selected['ap_xyz'],
            'physical_context_rx_xyz_m':meta['physical_context_rx_xyz_m'],'physical_ap_xyz_m':meta['physical_ap_xyz_m'],
            'physical_query_rx_xyz_m':np.broadcast_to(np.array([0.,0.,1.]),(np.prod(shape),3)).astype('f4'),
            'plan':plan,'mean_dbm':mean,'std_db':np.ones(shape)}
        record.update({'query_'+k:v for k,v in query.items()})
        saved.update({f'job{j}_{k}':v for k,v in record.items()})
        result['model_records'].append({'decision':j,'history_length':60+j,
            'context_metadata':{k:v for k,v in meta.items() if not isinstance(v,np.ndarray)}})
        memory=native_history_memory(past,issue_time_s=float(j),retention_age_s=6.)
        result['support_preparation'].append({'decision':j,'reports':len(memory),'snapshot_metadata':memory.snapshot_metadata,
            'matched_forecast_context_budget':192,'guard_modifies_issued_forecast':False,'packet_risk_qualified':False,
            'retention_age_seconds':6.,'elapsed_seconds':.0001})
        saved['forecast_mean_snr'][j]=radio_assumptions(geometry).convert(mean).astype('f4')
        result['forecast_jobs'][j]['forecast_sha256']=n._sha(saved['forecast_mean_snr'][j])
    # JSON roundtrip also exercises tuple/list normalization in metadata.
    result=__import__('json').loads(c.encoded(result))
    assert validate_fixture(receipt,result,saved,world)['passes']
    saved['job0_context_upper']=saved['job0_context_upper'].copy();saved['job0_context_upper'][0]+=1.
    with pytest.raises(ValueError,match='changed causal context/query context_upper'):
        validate_fixture(receipt,result,saved,world)


def test_rehashed_final_history_cannot_replace_the_bound_warm_observations(tmp_path,monkeypatch):
    receipt,_,folder=archive_fixture(tmp_path,monkeypatch)
    saved=c.arrays(folder/'predictions_history.npz');saved['history_value'][55,0,0]=-69.
    import io
    stream=io.BytesIO();np.savez_compressed(stream,**saved)
    (folder/'predictions_history.npz').write_bytes(stream.getvalue())
    receipt['arrays']=c.reference(folder/'predictions_history.npz')
    with pytest.raises(ValueError,match='warm prefix'):n.load_result(receipt)
