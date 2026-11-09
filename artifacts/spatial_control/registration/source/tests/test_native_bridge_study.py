"""Study registration/aggregation software fixtures, not native case results."""
import pytest
from copy import deepcopy

from wirelessjev.native_bridge import study as s


def test_protocol_is_finite_and_comparisons_are_separated():
    protocol=s.protocol_spec()
    assert len(protocol['cases'])==8 and len(set(c['seed'] for c in protocol['cases']))==4
    assert protocol['maximum_native_simulations']==18 and protocol['duration_seconds']==12
    assert protocol['actual_carriers_hz']==[2.4e9,5.2e9,6.5e9]
    assert protocol['candidate']['native_outcomes_used_for_selection'] is False
    assert protocol['production_qualified'] is False and protocol['default_changed'] is False
    assert protocol['capture']['real_continuous_native_reports_quantization_db']==0.


def test_write_spec_is_json_cli_ready_and_idempotent(tmp_path,monkeypatch):
    import json
    path=tmp_path/'protocol.json';monkeypatch.setattr(s,'PROTOCOL',path)
    value=s.write_spec()
    assert json.loads(path.read_text())==value==s.write_spec()
    json.dumps(value)


def test_native_metrics_recompute_packet_tail_and_silence():
    # Synthetic counters exercise a pure reducer; they are not case evidence.
    import numpy as np
    misses=np.arange(100)%7;sent=np.full(100,20);gap=np.arange(100)
    fraction=misses/sent
    native={'control_miss':float(misses.sum()/sent.sum()),
        'control_miss_worst_decile':float(np.sort(fraction)[-10:].mean()),
        'control_gap_p99_ms':float(np.percentile(gap,99)*10),'video_on_time':.8,'bulk_mbps':1.,
        'airtime_domain_sum':.5,'roams':2,'safety_stop_fraction':.1,
        'control_misses':int(misses.sum()),'control_sent':int(sent.sum()),
        'packet_detail':{'misses':misses.tolist(),'sent':sent.tolist(),'max_gap':gap.tolist()}}
    arm={'native':native,'forecast_jobs':[],'planner_and_admission':[],'forecast_consumption':[]}
    assert s._native_metrics(arm)['maximum_silence_ms']==990
    native['control_miss_worst_decile']+=.01
    with pytest.raises(RuntimeError,match='raw per-robot'):s._native_metrics(arm)


def test_publication_guard_precedes_case_lookup_or_world_generation(monkeypatch):
    def blocked():raise RuntimeError('source not published')
    monkeypatch.setattr(s,'check_sources',blocked)
    with pytest.raises(RuntimeError,match='not published'):s.run_case({'case_id':'not-a-case'})


def test_handover_summaries_preserve_harm_counts_and_four_episodes():
    rows=[]
    for spec in s.protocol_spec()['cases']:
        delta=(1. if spec['seed']==s.SEEDS[0] else -1.)*(1 if spec['handover_seconds']==.03 else 2)
        metrics={'control_miss':0.,'control_miss_worst_decile':0.,'maximum_silence_ms':0.}
        rows.append({'parity':False,'case':spec,'metrics':{'candidate':metrics},
            'paired_delta_candidate_minus_Map':{name:delta for name in metrics}})
    summary=s.summarize(rows)
    assert summary['0.03']['episodes']==4 and summary['4.0']['episodes']==4
    assert summary['0.03']['control_miss_harm_count']==1
    assert summary['0.03']['paired_mean_candidate_minus_Map']['control_miss']==-.5
    assert summary['4.0']['paired_mean_candidate_minus_Map']['control_miss']==-1.
    with pytest.raises(ValueError,match='all four'):s.summarize(rows[:-1])


def binding_fixture():
    """Metadata-only fixture; no fabricated native packet outcomes."""
    spec=s.protocol_spec()['cases'][0];config=s.config_for(spec,False)
    identity={'source_receipt_sha256':'a'*64,'protocol_sha256':'b'*64,
              'world_receipt_sha256':'c'*64,'weights_sha256':'d'*64}
    jobs=[{'decision':i,'history_cutoff_exclusive':60+i,'kind':'cold' if i==0 else 'warm',
           'profile_seconds':config.forecast_profile_seconds['cold' if i==0 else 'warm'],
           'shared_extra_seconds':config.shared_extra_seconds} for i in range(12)]
    native={'scene':spec['seed'],'duration_s':12,'policy':'adaptive_service_development',
            'collision_lambda':0.,'interference':False}
    raw={'case_id':spec['case_id'],'partition':'development','robots':100,'native_packet_simulations':2,
         **{k:v for k,v in identity.items() if k!='weights_sha256'},'simulation_only':True,
         'production_qualified':False,'original_SERVICE_holdouts_opened':False,'default_controller_changed':False,
         'equal_allowed_capture_and_compute_budgets':True,'equal_measured_runtime_claimed':False,
         'config':config.__dict__,'policy':s._policy_for(spec,False),
         'arms':{name:{'native':deepcopy(native),'forecast_jobs':deepcopy(jobs)} for name in ('radiomap','candidate')}}
    directory=s.OUT/'cases'/spec['case_id']
    receipt={'case':spec,'parity':False,**identity,'native_simulations':2,'simulation_only':True,
             'production_qualified':False,'original_SERVICE_holdouts_opened':False,
             'geometry_carrier_scale_transfer_unqualified':True}
    for field,file in (('raw_result','native_result.json.gz'),('candidate_inputs_forecasts','candidate_inputs_forecasts.npz'),('attempt','attempt.json')):
        receipt[field]={'path':str((directory/file).relative_to(s.ROOT)),'sha256':'e'*64}
    return spec,identity,raw,receipt


def test_exact_native_metadata_fixture_accepts_the_registered_case():
    spec,identity,raw,receipt=binding_fixture()
    s._validate_raw_metadata(raw,spec,False,identity)
    s._validate_receipt_metadata(receipt,spec,False,identity)


@pytest.mark.parametrize('duration',(12,12.0))
def test_native_exact_numeric_duration_accepts_int_or_float(duration):
    spec,identity,raw,_=binding_fixture()
    raw['arms']['candidate']['native']['duration_s']=duration
    s._validate_raw_metadata(raw,spec,False,identity)


@pytest.mark.parametrize('duration',(True,float('inf'),11.999,20.))
def test_native_wrong_or_boolean_duration_refuses(duration):
    spec,identity,raw,_=binding_fixture()
    raw['arms']['candidate']['native']['duration_s']=duration
    with pytest.raises(RuntimeError,match='duration'):s._validate_raw_metadata(raw,spec,False,identity)


@pytest.mark.parametrize('field,value',(
    ('case_id','dev-90473001-ho4'),('partition','test'),('robots',99),
    ('source_receipt_sha256','f'*64),('protocol_sha256','f'*64),('world_receipt_sha256','f'*64)))
def test_raw_case_identity_relabels_refuse(field,value):
    spec,identity,raw,_=binding_fixture();raw[field]=value
    with pytest.raises(RuntimeError,match='registration'):s._validate_raw_metadata(raw,spec,False,identity)


def test_checksum_consistent_wrong_handover_case_cannot_be_relabelled():
    spec,identity,raw,_=binding_fixture();other=s.protocol_spec()['cases'][1]
    raw['config']=s.config_for(other,False).__dict__;raw['policy']=s._policy_for(other,False)
    with pytest.raises(RuntimeError,match='HO/profile'):s._validate_raw_metadata(raw,spec,False,identity)


@pytest.mark.parametrize('mutation',('config_profile','job_profile','job_extra','jobs11','decision','futurecutoff',
                                   'scene','duration','collision','interference'))
def test_registered_timing_jobs_and_physics_are_bound(mutation):
    spec,identity,raw,_=binding_fixture();arm=raw['arms']['candidate']
    if mutation=='config_profile':raw['config']['forecast_profile_seconds']['warm']=8.
    if mutation=='job_profile':arm['forecast_jobs'][1]['profile_seconds']=8.
    if mutation=='job_extra':arm['forecast_jobs'][1]['shared_extra_seconds']=8.
    if mutation=='jobs11':arm['forecast_jobs'].pop()
    if mutation=='decision':arm['forecast_jobs'][1]['decision']=0
    if mutation=='futurecutoff':arm['forecast_jobs'][1]['history_cutoff_exclusive']=1000
    if mutation=='scene':arm['native']['scene']=s.SEEDS[1]
    if mutation=='duration':arm['native']['duration_s']=20.
    if mutation=='collision':arm['native']['collision_lambda']=1.
    if mutation=='interference':arm['native']['interference']=True
    with pytest.raises(RuntimeError):s._validate_raw_metadata(raw,spec,False,identity)


@pytest.mark.parametrize('field',('raw_result','candidate_inputs_forecasts','attempt'))
def test_canonical_archive_paths_reject_another_cases_bytes(field):
    spec,identity,_,receipt=binding_fixture()
    receipt[field]['path']=receipt[field]['path'].replace('ho0.03','ho4')
    with pytest.raises(RuntimeError,match='canonical'):s._validate_receipt_metadata(receipt,spec,False,identity)


@pytest.mark.parametrize('field',('case','source_receipt_sha256','protocol_sha256','world_receipt_sha256','weights_sha256'))
def test_cached_receipt_semantics_are_rechecked(field):
    spec,identity,_,receipt=binding_fixture()
    receipt[field]=s.protocol_spec()['cases'][1] if field=='case' else 'f'*64
    with pytest.raises(RuntimeError,match='registered case'):s._validate_receipt_metadata(receipt,spec,False,identity)


def test_durable_attempt_precedes_dependencies_and_blocks_interrupted_retry(tmp_path,monkeypatch):
    root=tmp_path;out=root/'artifacts/native_bridge'
    monkeypatch.setattr(s,'ROOT',root);monkeypatch.setattr(s,'OUT',out);monkeypatch.setattr(s,'PROTOCOL',out/'protocol.json')
    s.write_spec();s.write(out/'registration/source.json',{'software_fixture_only':True})
    monkeypatch.setattr(s,'check_sources',lambda:None)
    spec=s.protocol_spec()['cases'][0];attempt=out/'cases'/spec['case_id']/'attempt.json';calls=[]
    def interrupted():
        assert attempt.exists() and s.read(attempt)['maximum_native_simulations']==2
        calls.append(True)
        raise RuntimeError('software interruption before model or world')
    monkeypatch.setattr(s,'bind_weights',interrupted)
    with pytest.raises(RuntimeError,match='software interruption'):s.run_case(spec)
    first=attempt.read_bytes()
    with pytest.raises(RuntimeError,match='refuse reexecution'):s.run_case(spec)
    assert len(calls)==1 and attempt.read_bytes()==first


def test_partial_case_without_receipt_is_not_reexecuted(tmp_path,monkeypatch):
    out=tmp_path/'artifacts/native_bridge';monkeypatch.setattr(s,'ROOT',tmp_path);monkeypatch.setattr(s,'OUT',out)
    spec=s.protocol_spec()['cases'][0];directory=out/'cases'/spec['case_id'];directory.mkdir(parents=True)
    (directory/'native_result.json.gz').write_bytes(b'partial-software-fixture')
    with pytest.raises(RuntimeError,match='partial native case'):s._begin_attempt(spec,False)


def test_parity_checks_every_native_counter_and_raw_packet_field():
    # Pure equality fixture, not a claimed native simulation result.
    import hashlib
    import numpy as np
    from types import SimpleNamespace
    records=[{'source_snr':np.array((i,),dtype='f4')} for i in range(12)]
    jobs=[{'forecast_sha256':hashlib.sha256(r['source_snr'].tobytes()).hexdigest(),
           'forecast_std_sha256':'a'*64,'ready_slot':450*(i+1)} for i,r in enumerate(records)]
    arm={'forecast_jobs':jobs,'native':{'packet_detail':{'misses':[0],'sent':[2],'max_gap':[1]},'airtime_domain_sum':.5}}
    result={'arms':{'radiomap':deepcopy(arm),'candidate':deepcopy(arm)}}
    assert s._parity(result,SimpleNamespace(records=records))['passes']
    result['arms']['candidate']['native']['packet_detail']['max_gap']=[2]
    proof=s._parity(result,SimpleNamespace(records=records))
    assert not proof['passes'] and not proof['checks']['native_packet_and_service_counters_equal']
