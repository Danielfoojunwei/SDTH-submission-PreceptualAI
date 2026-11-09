"""Only synthetic unit fixtures can pass here; repository candidates stay blocked."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pytest
import numpy as np
from wirelessjev.production_qualification import Limits, digest, identity, subject, qualify, current_repository_status, binomial_upper


def write(root, path, value):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,sort_keys=True)+'\n')
    return {'path':path,'sha256':digest(target)}


def raw_ledger(root,path,content):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    return {'path':path,'sha256':digest(target)}


def bundle(root, mode='active_candidate'):
    """Physically plausible, explicitly fabricated fixture; never persisted evidence."""
    files={group:{} for group in ('source','weights','data','licenses')}
    for group,names in {'source':['candidate.py','baseline.py'],'weights':['seed0.pt','seed1.pt','seed2.pt'],
                        'data':['training.csv'],'licenses':['unit-license.txt']}.items():
        for name in names:
            path=group+'/'+name;(root/path).parent.mkdir(parents=True,exist_ok=True)
            (root/path).write_text('UNIT FIXTURE ONLY '+path+'\n')
            files[group][path]=digest(root/path)
    profile={'profile_id':'unit-fixture-radio','device_model':'Unit fixture','driver':'fixture-1',
             'firmware':'fixture-1','protocol':'WiFi','control_period_ms':50.,'deadline_ms':50.,
             'baseline_endpoint':'02:00:00:00:00:01','interface':'wlan1','role':'WIFI',
             'baseline_source_sha256':files['source']['source/baseline.py']}
    design=write(root,'independent-design.json',{'basis':'independent_randomized_resets','reviewer':'Unit fixture reviewer',
        'reviewed_at':'2025-12-31T00:00:00Z','assumption':'iid_reset_episodes_within_registered_profile',
        'profile_sha256':identity(profile),'shift_status':'no_detected_shift_in_registered_profile','on_shift':'invalidate',
        'justification':'UNIT FIXTURE: every outcome index denotes an independently randomized reset episode and a preregistered uniform command sample; no independence is inferred from packet timestamps.',
        'reset_and_sampling_source':{'path':'source/candidate.py','sha256':files['source']['source/candidate.py']}})
    protocol={'training_sites':[1,2,3],'validation_sites':[10,11],'confirmation_sites':list(range(20,26)),
              'training_seeds':[0,1,2],'sealed_at':'2026-01-01T00:00:00Z',
              'independent_sampling':{'unit':'independently_reset_episode','target':'one_preregistered_uniform_command_per_episode','design_review':design},
              'risk_bins':[{'bin_id':'accepted','admitted':True,'ceiling':1e-5},{'bin_id':'rejected','admitted':False,'ceiling':.05}]}
    p=write(root,'protocol.json',protocol)
    b={'schema_version':1,'candidate_id':'UNIT_FIXTURE_NOT_REAL_EVIDENCE','mode':mode,'files':files,
       'protocol':p,'hardware_profile':profile,'limits':asdict(Limits())}
    sid=subject(b)
    def receipt(role,value):b[role]=write(root,role+'.json',{'subject_sha256':sid,**value})
    receipt('rights',{'grants':[{'artifact_sha256':h,'license_sha256':next(iter(files['licenses'].values())),
        'reviewer':'Unit fixture reviewer','reviewed_at':'2026-01-02T00:00:00Z',
        'allowed_uses':['research','production_inference']} for h in [*files['weights'].values(),*files['data'].values()]]})
    b['fits']=[]
    for seed in range(3):
        path=f'weights/seed{seed}.pt'
        b['fits'].append(write(root,f'fit{seed}.json',{'subject_sha256':sid,'training_seed':seed,'optimizer_steps':100,
            'source_group_sha256':identity(files['source']),'data_group_sha256':identity(files['data']),
            'weights_path':path,'weights_sha256':files['weights'][path]}))
    rows=[]
    control_ledger=raw_ledger(root,'control-episodes.bin',bytes([2])*20+bytes(1_000_000-20))
    reset_ids=raw_ledger(root,'reset-ids.bin',np.arange(1_000_000,dtype='<u8').tobytes())
    for site in protocol['confirmation_sites']:
        for seed in range(3):
            rows.append({'site_id':site,'training_seed':seed,
                'independent_episode_ledger':control_ledger,
                'independent_reset_ids':reset_ids,
                'candidate':{'sent':1000000,'misses':0,'max_gap_ms':50.,'latency_ms':[20.]*100},
                'baseline':{'sent':1000000,'misses':20,'max_gap_ms':100.,'latency_ms':[25.]*100}})
    receipt('evaluation',{'confirmation_opened_at':'2026-02-01T00:00:00Z','rows':rows})
    receipt('risk',{'scope':'empirical_heldout_calibration','target':'deadline_miss','calibration_sites':[10,11],
        'bins':[{'bin_id':'accepted','site_id':10,'trials':1000000,'misses':0,'probability':.000009,
                 'independent_reset_ids':reset_ids,
                 'independent_episode_ledger':raw_ledger(root,'risk-low.bin',bytes(1_000_000))},
                {'bin_id':'rejected','site_id':11,'trials':1000000,'misses':10000,'probability':.02,
                 'independent_reset_ids':reset_ids,
                 'independent_episode_ledger':raw_ledger(root,'risk-high.bin',bytes([1])*10000+bytes(990000))}]})
    receipt('hardware',{'measurement_kind':'physical_device_run','device_id':'UNIT_FIXTURE_ONLY',
        'samples':[{'observation_ms':1.,'inference_ms':2.,'transport_ms':3.,'actuator_ms':5.,
                    'end_to_end_ms':15.,'timed_out':False,'sample_id':str(i),'started_at_ns':1_000_000_000+50_000_000*i} for i in range(1000)]})
    receipt('rollback',{'measurement_kind':'physical_device_run','device_id':'UNIT_FIXTURE_ONLY','events':[
        {'event':'outcome_unknown','at_ms':0.},{'event':'status_read','at_ms':10.},
        {'event':'baseline_verified','at_ms':80.,'controller_sha256':profile['baseline_source_sha256'],
         'observed_baseline_endpoint':profile['baseline_endpoint'],'observed_interface':'wlan1','observed_role':'WIFI',
         'evidence_source':'hardware_status'}]})
    return b


def change(root,b,role,fn):
    value=json.loads((root/b[role]['path']).read_text());fn(value)
    b[role]=write(root,b[role]['path'],value)


def codes(result):return {b['code'] for b in result['blockers']}


def test_complete_unit_fixture_recomputes_counts_intervals_and_never_activates(tmp_path):
    b=bundle(tmp_path);r=qualify(b,tmp_path)
    assert r['execution_eligible'] and r['sandbox_eligible'] and r['latency_shadow_eligible']
    assert not r['runtime_activated'] and not r['scientific_or_safety_certificate']
    assert r['metrics']['control']['paired_site_difference_ppm']==-20.
    assert r['metrics']['control']['paired_site_95_ppm']==[-20.,-20.]
    assert r['metrics']['control']['crossed_site_seed_95_ppm']==[-20.,-20.]
    assert r['metrics']['hardware']['samples']==1000 and r['metrics']['risk']['empirical_ece']==pytest.approx(.0050045)
    assert all(c['candidate_upper_ppm']<10 for c in r['metrics']['control']['absolute_risk']['cells'])


def test_proposal_only_can_qualify_shadow_but_never_execution(tmp_path):
    r=qualify(bundle(tmp_path,'proposal_only'),tmp_path)
    assert r['sandbox_eligible'] and r['latency_shadow_eligible']
    assert not r['execution_eligible'] and codes(r)=={'proposal_only'}


@pytest.mark.parametrize('role',['evaluation','risk','hardware','rollback','rights'])
def test_truthy_booleans_cannot_replace_actual_receipt_files(tmp_path,role):
    b=bundle(tmp_path);b[role]=True
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and role.replace('evaluation','independent_control') in codes(r)


def test_tampered_weights_and_receipts_are_rejected_by_byte_identity(tmp_path):
    b=bundle(tmp_path)
    (tmp_path/'weights/seed0.pt').write_text('changed checkpoint')
    (tmp_path/b['hardware']['path']).write_text('{"passed":true}')
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and {'identity_weights','hardware'} <= codes(r)
    h=r['verified_files']['weights/seed0.pt']
    assert h['expected_sha256']!=h['actual_sha256']


def test_switching_candidate_mode_or_hardware_profile_invalidates_bound_receipts(tmp_path):
    b=bundle(tmp_path,'proposal_only');b['mode']='active_candidate'
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and {'rights','independent_control','hardware'} <= codes(r)
    b=bundle(tmp_path);b['hardware_profile']['driver']='other-driver'
    assert not qualify(b,tmp_path)['execution_eligible']


@pytest.mark.parametrize('mutation',[
    lambda d:d['rows'].pop(),
    lambda d:d['rows'].append(deepcopy(d['rows'][0])),
    lambda d:d['rows'][0]['candidate'].update(sent=False),
    lambda d:d['rows'][0]['candidate'].update(misses=100001),
    lambda d:d['rows'][0]['candidate'].update(latency_ms=[]),
    lambda d:d.update(confirmation_opened_at='2025-01-01T00:00:00Z'),
])
def test_raw_pair_coverage_counts_and_registration_must_validate(tmp_path,mutation):
    b=bundle(tmp_path);change(tmp_path,b,'evaluation',mutation)
    assert 'independent_control' in codes(qualify(b,tmp_path))


def test_harmful_noninferiority_interval_is_rejected_even_if_summary_says_pass(tmp_path):
    b=bundle(tmp_path)
    def harm(d):
        d['passed']=True
        for row in d['rows']:
            row['candidate']['misses']=1;row['baseline']['misses']=0
    change(tmp_path,b,'evaluation',harm)
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and 'independent_control' in codes(r)
    assert r['metrics']['control']['paired_site_95_ppm']==[1.,1.]


@pytest.mark.parametrize('mutation',[
    lambda d:d['rows'][0]['candidate'].update(max_gap_ms=1000),
    lambda d:d['rows'][0]['candidate'].update(latency_ms=[51.]*100),
    lambda d:d['rows'][0]['candidate'].update(max_gap_ms=float('nan')),
])
def test_silence_and_latency_regressions_cannot_hide_behind_low_miss_average(tmp_path,mutation):
    b=bundle(tmp_path);change(tmp_path,b,'evaluation',mutation)
    assert not qualify(b,tmp_path)['execution_eligible']


@pytest.mark.parametrize('mutation',[
    lambda d:d.update(samples=d['samples'][:10]),
    lambda d:d.update(measurement_kind='simulation'),
    lambda d:d['samples'][0].update(timed_out=True),
    lambda d:d['samples'][0].update(end_to_end_ms=1.),
    lambda d:d['samples'][1].update(sample_id='0'),
])
def test_hardware_requires_full_actual_end_to_end_samples(tmp_path,mutation):
    b=bundle(tmp_path);change(tmp_path,b,'hardware',mutation)
    assert 'hardware' in codes(qualify(b,tmp_path))


def test_calibration_is_empirical_and_not_a_distribution_free_certificate(tmp_path):
    b=bundle(tmp_path);change(tmp_path,b,'risk',lambda d:d.update(distribution_free_certificate={'exchangeable':True}))
    assert 'risk' in codes(qualify(b,tmp_path))
    b=bundle(tmp_path);change(tmp_path,b,'risk',lambda d:d['bins'][0].update(probability=.9))
    assert 'risk' in codes(qualify(b,tmp_path))


def test_rollback_requires_actual_reconciliation_to_same_hardware_baseline(tmp_path):
    b=bundle(tmp_path);change(tmp_path,b,'rollback',lambda d:d.update(device_id='other_device'))
    assert 'rollback' in codes(qualify(b,tmp_path))
    b=bundle(tmp_path);change(tmp_path,b,'rollback',lambda d:d['events'][-1].update(at_ms=1000.))
    assert 'rollback' in codes(qualify(b,tmp_path))
    b=bundle(tmp_path);change(tmp_path,b,'rollback',lambda d:d['events'][-1].update(observed_role='HOTPOT'))
    assert 'rollback' in codes(qualify(b,tmp_path))


def test_licenses_are_bound_and_research_permission_is_not_production_permission(tmp_path):
    b=bundle(tmp_path)
    change(tmp_path,b,'rights',lambda d:[g.update(allowed_uses=['research']) for g in d['grants']])
    r=qualify(b,tmp_path)
    assert r['sandbox_eligible'] and not r['execution_eligible'] and 'production_rights' in codes(r)


def test_missing_and_malformed_dossiers_fail_closed(tmp_path):
    for b in [None,True,[],{}, {'schema_version':1,'candidate_id':'missing','files':True}]:
        assert not qualify(b,tmp_path)['execution_eligible']


def test_current_actual_candidates_are_not_qualified_by_primary_receipt_booleans():
    root=Path(__file__).resolve().parents[1]
    status=current_repository_status(root)
    assert not status['runtime_activated'] and len(status['candidates'])==3
    for candidate in status['candidates']:
        r=candidate['qualification']
        assert not r['execution_eligible'] and {'hardware','independent_control','proposal_only'} <= codes(r)
        for path,receipt in candidate['existing_evidence'].items():assert digest(root/path)==receipt['sha256']


def reseal(root,b):
    sid=subject(b)
    for role in ('evaluation','risk','hardware','rollback','rights'):
        change(root,b,role,lambda d:d.update(subject_sha256=sid))
    for i,ref in enumerate(b['fits']):
        d=json.loads((root/ref['path']).read_text());d['subject_sha256']=sid
        b['fits'][i]=write(root,ref['path'],d)


def test_train_confirmation_overlap_and_two_seed_claim_are_not_independent_evidence(tmp_path):
    b=bundle(tmp_path);p=json.loads((tmp_path/b['protocol']['path']).read_text())
    p['training_sites'].append(20);b['protocol']=write(tmp_path,'protocol.json',p);reseal(tmp_path,b)
    r=qualify(b,tmp_path)
    assert 'independent_control' in codes(r)
    assert any('disjoint' in x['detail'] for x in r['blockers'])
    b=bundle(tmp_path);p=json.loads((tmp_path/b['protocol']['path']).read_text())
    p['training_seeds']=[0,1];b['protocol']=write(tmp_path,'protocol.json',p);reseal(tmp_path,b)
    assert 'independent_control' in codes(qualify(b,tmp_path))


def test_three_seed_labels_cannot_reuse_one_trained_checkpoint(tmp_path):
    b=bundle(tmp_path);ref=b['fits'][2]
    fit=json.loads((tmp_path/ref['path']).read_text())
    fit.update(weights_path='weights/seed0.pt',weights_sha256=b['files']['weights']['weights/seed0.pt'])
    b['fits'][2]=write(tmp_path,ref['path'],fit)
    r=qualify(b,tmp_path)
    assert any('same checkpoint' in x['detail'] for x in r['blockers'])


def test_eighteen_zero_miss_commands_cannot_qualify_with_degenerate_bootstrap(tmp_path):
    b=bundle(tmp_path)
    one=raw_ledger(tmp_path,'one-episode.bin',b'\0')
    ids=raw_ledger(tmp_path,'one-reset.bin',np.array([0],dtype='<u8').tobytes())
    def shrink(d):
        for row in d['rows']:
            for arm in ('candidate','baseline'):row[arm].update(sent=1,misses=0)
            row.update(independent_episode_ledger=one,independent_reset_ids=ids)
    change(tmp_path,b,'evaluation',shrink)
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and 'independent_control' in codes(r)
    assert r['metrics']['control']['paired_site_95_ppm']==[0.,0.]
    cells=r['metrics']['control']['absolute_risk']['cells']
    assert sum(c['independent_episodes'] for c in cells)==18
    assert all(c['candidate_upper_ppm']>10 for c in cells)


def test_one_percent_observed_risk_cannot_hide_behind_small_global_ece(tmp_path):
    b=bundle(tmp_path)
    raw=raw_ledger(tmp_path,'one-percent.bin',bytes([1])*1000+bytes(99000))
    ids=raw_ledger(tmp_path,'one-percent-resets.bin',np.arange(100000,dtype='<u8').tobytes())
    def underpredict(d):
        for i,row in enumerate(d['bins']):
            row.update(trials=100000,misses=1000,probability=(i+1)*1e-8,
                       independent_episode_ledger=raw,independent_reset_ids=ids)
    change(tmp_path,b,'risk',underpredict)
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and 'risk' in codes(r)
    assert any('underprediction' in c['detail'] for c in r['blockers'])


@pytest.mark.parametrize('mutate',[
    lambda d:d.update(basis='uncorrected_correlated_packet_trace'),
    lambda d:d.update(assumption='independence_inferred_from_clock_gaps'),
    lambda d:d.update(shift_status='unknown'),
    lambda d:d.update(shift_status='detected_shift'),
    lambda d:d.update(reviewer=''),
])
def test_unsupported_correlation_or_shift_assumptions_fail_closed(tmp_path,mutate):
    b=bundle(tmp_path);p=json.loads((tmp_path/b['protocol']['path']).read_text())
    ref=p['independent_sampling']['design_review'];d=json.loads((tmp_path/ref['path']).read_text());mutate(d)
    p['independent_sampling']['design_review']=write(tmp_path,ref['path'],d)
    b['protocol']=write(tmp_path,'protocol.json',p);reseal(tmp_path,b)
    r=qualify(b,tmp_path)
    assert not r['execution_eligible'] and 'independence' in codes(r)


def test_repeated_reset_ids_and_unlinked_exposure_do_not_inflate_sample_size(tmp_path):
    b=bundle(tmp_path)
    repeated=raw_ledger(tmp_path,'repeated-reset-ids.bin',bytes(8_000_000))
    change(tmp_path,b,'evaluation',lambda d:d['rows'][0].update(independent_reset_ids=repeated))
    r=qualify(b,tmp_path)
    assert any('reset identities' in c['detail'] for c in r['blockers'])
    b=bundle(tmp_path)
    change(tmp_path,b,'evaluation',lambda d:d['rows'][0]['candidate'].update(misses=1))
    assert any('contradicts' in c['detail'] for c in qualify(b,tmp_path)['blockers'])


def test_finite_sample_zero_failure_bound_is_nonzero_and_reaches_ten_ppm_only_with_exposure():
    assert binomial_upper(0,18,.05)>0.15
    assert binomial_upper(0,299571,.05)>1e-5
    assert binomial_upper(0,299572,.05)<=1e-5
    assert binomial_upper(0,1000000,.05/(2*18))<1e-5
