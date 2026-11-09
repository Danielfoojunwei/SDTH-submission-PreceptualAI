"""Independent READ-ONLY FIELD array/metric audit; no science/cache calls."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, re
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr

ROOT=Path('/workspace/SDTH-submission-PreceptualAI')
OUT=ROOT/'artifacts/radiojepa_spectral'
DEST=Path('/workspace/.cloud-setup/spectral-final-independent.json')
RX=re.compile(r'^(?P<scene>[^:]+):pose(?P<pose>\d+):ap(?P<ap>\d+):band(?P<band>\d+):direction(?P<direction>\d+):device(?P<device>\d+)$')
CONTEXT=(0,2,4,6,8,10); QUERY=(1,3,5,7,9,11)
REGIMES=('in_band','held_band','onboarding_0','onboarding_1','onboarding_4','onboarding_16')
REFS=('operator_only','radiomap_operator','bayesian_operator')
HASH={}

def digest(path):
    p=Path(path)
    if p not in HASH: HASH[p]=hashlib.sha256(p.read_bytes()).hexdigest()
    return HASH[p]

def read(path): return json.loads(Path(path).read_text())
def path_ref(value, expected=None):
    p=ROOT/value['path']; assert not Path(value['path']).is_absolute() and '..' not in Path(value['path']).parts
    assert p.is_file() and not p.is_symlink()
    if expected is not None: assert p==expected,(p,expected)
    assert digest(p)==value['sha256'],str(p)
    return p

def key(scene,p,a,b,d,v):return f'{scene}:pose{p}:ap{a}:band{b}:direction{d}:device{v}'
def ids(scene,poses,bands,devices,directions=(0,1)):
    return [key(scene,p,a,b,d,v) for b in bands for d in directions for v in devices for p in poses for a in (0,1)]
def parse(values,scene):
    result=[]
    for value in values:
        m=RX.fullmatch(value);assert m and m['scene']==scene,value
        row=tuple(int(m[k]) for k in ('pose','ap','band','direction','device'))
        assert all(0<=v<n for v,n in zip(row,(12,2,3,2,4)))
        result.append(row)
    assert len(result)==len(set(result))
    return np.asarray(result,dtype=int)
def take(a,index):return a[tuple(index.T)]
def target_arrays(raw,index):
    heard=take(raw['heard'],index);censored=take(raw['censored'],index)
    assert heard.dtype==censored.dtype==bool and not (heard&censored).any()
    interval=take(raw['quantization_interval_dbm'],index)
    return {'center_dbm':np.where(heard,take(raw['rssi_dbm'],index),0.).astype('f4'),
            'lower_dbm':np.where(heard,interval[:,0],0.).astype('f4'),
            'upper_dbm':np.where(heard,interval[:,1],0.).astype('f4'),
            'heard':heard,'censored':censored,
            'censor_upper_dbm':np.where(censored,take(raw['censor_upper_dbm'],index),0.).astype('f4')}
def metrics(mean,std,t):
    mean=np.asarray(mean,dtype='f8');std=np.asarray(std,dtype='f8')
    assert mean.ndim==1 and std.shape==mean.shape and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
    heard=t['heard'];censored=t['censored'];assert heard.any()
    r=mean[heard]-np.asarray(t['center_dbm'],dtype='f8')[heard]
    hi=(t['upper_dbm'][heard].astype('f8')-mean[heard])/std[heard]
    lo=(t['lower_dbm'][heard].astype('f8')-mean[heard])/std[heard]
    assert (hi>lo).all()
    upper=np.where(lo>0,log_ndtr(-lo),log_ndtr(hi))
    lower=np.where(lo>0,log_ndtr(-hi),log_ndtr(lo))
    logmass=upper+np.log(-np.expm1(np.minimum(lower-upper,-np.finfo(float).eps)))
    cnll=log_ndtr((t['censor_upper_dbm'][censored].astype('f8')-mean[censored])/std[censored])
    nll=-np.r_[logmass,cnll];assert np.isfinite(nll).all()
    return {'rmse_db':float(np.sqrt(np.mean(r*r))),'p95_abs_db':float(np.quantile(np.abs(r),.95)),
            'observation_nll':float(nll.mean()),'heard_count':int(heard.sum()),'censored_count':int(censored.sum()),
            'missing_count':int((~(heard|censored)).sum()),'finite':True}
def equal_metrics(actual,expected):
    assert set(actual)==set(expected)
    for k,v in expected.items():
        if isinstance(v,float):assert np.isclose(actual[k],v,rtol=1e-12,atol=1e-12),(k,actual[k],v)
        else:assert actual[k]==v,(k,actual[k],v)
def calibration_for(raw,base,n):
    cal={int(k):dict(v) for k,v in base['estimates'].items()}
    if n is None:return cal
    pairs=[]
    for p in CONTEXT:
        for a in (0,1):
            for b in (0,1):
                for d in (0,1):
                    if raw['heard'][p,a,b,d,3] and raw['heard'][p,a,b,d,0]:
                        pairs.append((raw['rssi_dbm'][p,a,b,d,3],raw['rssi_dbm'][p,a,b,d,0],
                                      raw['quantization_interval_dbm'][p,a,b,d,3],raw['quantization_interval_dbm'][p,a,b,d,0]))
    assert len(pairs)>=n
    selected=pairs[:n]
    if not selected:cal[3]={'offset_db':0.,'pair_count':0,'known':False,'quantization_db':None};return cal
    lo=max(float(p[2][0])-float(p[3][1]) for p in selected)
    hi=min(float(p[2][1])-float(p[3][0]) for p in selected)
    consistent=lo<=hi+1e-9
    median=float(np.median([float(p[0])-float(p[1]) for p in selected]))
    offset=float(np.clip(median,lo,max(lo,hi))) if consistent else median
    cal[3]={'offset_db':offset,'pair_count':n,'known':consistent,
            'quantization_db':float(np.median([float(p[2][1])-float(p[2][0]) for p in selected]))}
    return cal

def physical(raw,index,tx):
    rx=raw['rx_positions_m'][index[:,0]].astype('f8');ap=raw['ap_positions_m'][index[:,1]].astype('f8')
    frequency=raw['carrier_hz'][index[:,2]].astype('f8')
    distance=np.linalg.norm(rx-ap,axis=-1)
    return tx-20*(np.log10(4*np.pi/299792458.)+np.log10(distance)+np.log10(frequency))

def main(final=False):
    protocol=read(OUT/'protocol.json');registry=read(OUT/'field_registry.json');collector=read(OUT/'rt_data/field_test.json')
    source=read(OUT/'registration/source.json');selection=read(OUT/'selection.json');selected=selection['selected_cell']
    assert len(source['source_sha256'])==475
    for name,sha in source['source_sha256'].items():
        assert digest(ROOT/name)==sha,name
        assert digest(OUT/'registration/source'/name)==sha,name
    ident={'source_receipt_sha256':digest(OUT/'registration/source.json'),
           'protocol_sha256':digest(OUT/'protocol.json'),'registry_sha256':digest(OUT/'field_registry.json')}
    assert collector['completed'] and collector['stage']=='field_test'
    assert collector['source_receipt_sha256']==ident['source_receipt_sha256']
    assert collector['registry_sha256']==ident['registry_sha256']
    spec_by_id={r['scene_id']:r for r in registry['stages']['field_test']}
    assert len(spec_by_id)==24 and set(collector['scene_receipts'])==set(spec_by_id)
    raw_by_id={};rt_by_id={};index_by_id={r['scene_id']:i for i,r in enumerate(registry['stages']['field_test'])}
    for scene,specification in spec_by_id.items():
        folder=OUT/'rt_data/field_test'/scene
        receipt_path=path_ref(collector['scene_receipts'][scene],folder/'receipt.json');rt=read(receipt_path)
        assert rt['scene_id']==scene and rt['spec_sha256']==specification['spec_sha256']
        public=folder/'public_views.npz';assert digest(public)==rt['files_sha256']['public_views.npz']
        with np.load(public,allow_pickle=False) as z:
            raw_by_id[scene]={k:z[k].copy() for k in ('rssi_dbm','heard','censored','censor_upper_dbm',
                'quantization_interval_dbm','ap_positions_m','rx_positions_m','carrier_hz','noise_floor_dbm')}
        rt_by_id[scene]=rt
    base=read(OUT/'data/frontend-training.json')
    scores={};expected_files=set();rows_by_key={};context_checks=0
    for regime in REGIMES:
        normal=[(cell,seed,None) for cell in ((selected,) if regime.startswith('onboarding_') else (selected,'matched_regression','geometry_only')) for seed in (0,1,2)]
        normal +=[(name,None,None) for name in REFS]
        if regime=='in_band':normal +=[(selected,seed,i) for i in ('zero','shuffle') for seed in (0,1,2)]
        for cell,seed,intervention in normal:
            member=cell if seed is None else f'{cell}-seed{seed}'+('' if intervention is None else '-'+intervention)
            scores[(regime,member)]={}
            for scene in spec_by_id:
                path=OUT/'predictions/field_test'/regime/member/f'{scene}.json';receipt=read(path)
                path_ref({'path':receipt['npz_path'],'sha256':receipt['npz_sha256']},path.with_suffix('.npz'))
                expected_files.update((path,path.with_suffix('.npz')));digest(path)
                m=receipt['metadata'];assert m['identity']==ident
                assert (m['stage'],m['regime'],m['scene_id'],m['scene_index'])==('field_test',regime,scene,index_by_id[scene])
                assert (m['member'],m['cell'],m['seed'],m['intervention'],m['source_view'])==(member,cell,seed,intervention,'a')
                assert m['predictions_ensembled'] is False and m['uncertainty_calibrated'] is False
                deps={'calibration':OUT/'data/frontend-training.json','scene_data':OUT/'data'/f'scene-field_test-{index_by_id[scene]:02d}-field_test.json','stage':OUT/'rt_data/field_test.json'}
                if seed is not None:deps.update(fit=OUT/'fits'/f'{cell}-seed{seed}.json',checkpoint=OUT/'checkpoints'/f'{cell}-seed{seed}.pt')
                elif cell!='operator_only':deps['baseline']=OUT/'baselines/prior.json'
                assert set(m['dependencies'])==set(deps)
                for name,p in deps.items():path_ref(m['dependencies'][name],p)
                if seed is not None:
                    fit=read(deps['fit']);assert fit['cell']==cell and fit['seed']==seed
                    assert fit['checkpoint_sha256']==digest(deps['checkpoint'])
                pose=CONTEXT if regime=='held_band' else QUERY
                band=(2,) if regime=='held_band' else (0,1)
                device=(3,) if regime.startswith('onboarding_') else (0,1,2)
                expected_ids=ids(scene,pose,band,device)
                assert m['query_measurement_ids']==expected_ids
                expected_a=ids(scene,CONTEXT,(0,1),(0,),directions=(0,))
                expected_b=ids(scene,CONTEXT,(0,1),(1,),directions=(1,))
                assert m['context_a_measurement_ids']==expected_a and m['context_b_measurement_ids']==expected_b
                assert not set(expected_ids)&set(expected_a+expected_b)
                qi=parse(expected_ids,scene);raw=raw_by_id[scene];target=target_arrays(raw,qi)
                cal=calibration_for(raw,base,int(regime.split('_')[1]) if regime.startswith('onboarding_') else None)
                with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                    for name,value in target.items():assert np.array_equal(z[name],value),str(path)+':'+name
                    for name,col in [('pose_index',0),('band_index',2),('direction',3),('device_index',4)]:assert np.array_equal(z[name],qi[:,col])
                    offset=np.asarray([cal[int(v)]['offset_db'] for v in qi[:,4]],dtype='f4')
                    assert np.array_equal(z['query_offset_db'],offset)
                    np.testing.assert_allclose(z['query_fspl_dbm'],physical(raw,qi,rt_by_id[scene]['reference_tx_dbm']).astype('f4'),rtol=0,atol=8.e-6)
                    for view,keys in [('a',expected_a),('b',expected_b)]:
                        ci=parse(keys,scene);ct=target_arrays(raw,ci);ctx=z['context_'+view]
                        assert ctx.shape==(24,20) and np.array_equal(z['mask_'+view],ct['heard']|ct['censored'])
                        ctxoffset=np.array([cal[int(v)]['offset_db'] for v in ci[:,4]])
                        fspl=physical(raw,ci,rt_by_id[scene]['reference_tx_dbm'])
                        rf=np.column_stack((np.where(ct['heard'],(ct['center_dbm']-ctxoffset-fspl)/30,0),
                            np.where(ct['censored'],(ct['censor_upper_dbm']-ctxoffset-fspl)/30,0),ct['heard'],ct['censored'])).astype('f4')
                        np.testing.assert_allclose(ctx[:,:4],rf,rtol=0,atol=1.e-6)
                        context_checks+=1
                    actual=metrics(z['mean'],z['std'],target)
                    if seed is not None:assert z['scene_latent'].shape==(8,) and np.isfinite(z['scene_latent']).all()
                equal_metrics(receipt['metrics'],actual)
                scores[(regime,member)][scene]=actual;rows_by_key[(regime,member,scene)]=receipt
    assert set((OUT/'predictions/field_test').rglob('*.json'))|set((OUT/'predictions/field_test').rglob('*.npz'))==expected_files
    assert len(expected_files)==2592
    summary=read(OUT/'results/field_test.json');assert summary['stage']=='field_test' and summary['identity']==ident
    for regime,group in summary['members'].items():
        for member,block in group.items():
            assert [row['scene_id'] for row in block['layouts']]==list(spec_by_id)
            for row in block['layouts']:
                assert row['available'] is True
                path_ref(row['prediction'],OUT/'predictions/field_test'/regime/member/f'{row["scene_id"]}.json')
                equal_metrics(row['metrics'],scores[(regime,member)][row['scene_id']])
    report=read(OUT/'report.json');assert report['identity']==ident
    scene_ids=list(spec_by_id);contrasts=[]
    def values(regime,cell,metric):
        if cell in REFS:return np.array([scores[(regime,cell)][s][metric] for s in scene_ids])
        return np.array([[scores[(regime,f'{cell}-seed{seed}')][s][metric] for s in scene_ids] for seed in (0,1,2)]).mean(0)
    for slot,official in zip(protocol['field_tests'],report['contrasts']):
        a=values(slot['regime'],selected,slot['metric']);b=values(slot['regime'],slot['reference'],slot['metric']);d=a-b
        rng=np.random.default_rng(protocol['bootstrap']['seed']);ix=rng.integers(len(d),size=(protocol['bootstrap']['samples'],len(d)))
        boot=d[ix].mean(1);adjusted=np.quantile(boot,protocol['bootstrap']['two_sided_quantiles']);rawci=np.quantile(boot,[.025,.975])
        assert all(official[k]==v for k,v in slot.items())
        for name,value in [('candidate_mean',a.mean()),('reference_mean',b.mean()),('mean_paired_difference',d.mean())]:assert np.isclose(official[name],value,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(official['adjusted_interval'],adjusted,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(official['descriptive_95_interval'],rawci,rtol=1e-12,atol=1e-12)
        assert official['candidate_better_adjusted']==bool(adjusted[1]<0)
        contrasts.append({'slot':slot['slot'],'metric':slot['metric'],'regime':slot['regime'],'reference':slot['reference'],
                          'mean_paired_difference':float(d.mean()),'adjusted_interval':adjusted.tolist(),'candidate_better_adjusted':bool(adjusted[1]<0)})
    assert len(contrasts)==4
    for key,index in [('rf_rmse_superiority',0),('matched_alignment_benefit_supported',1),('p95_superiority',2),('held_band_superiority',3)]:assert report[key]==contrasts[index]['candidate_better_adjusted']
    record={'passes':True,'scope':'independent raw FIELD reported-target, context binding, metric and four-bootstrap replay',
            'field_geometries':24,'prediction_receipts':1296,'prediction_arrays':1296,'context_RF_rows_replayed':context_checks*24,
            'targets_exactly_replayed':True,'RMSE_p95_NLL_recomputed':True,'cell_seed_dependencies_bound':True,
            'onboarding_offsets_recomputed_from_exact_context_pairs':True,
            'contrasts':contrasts,'bootstrap_unit':'whole geometry; fixed-three-fit member metrics averaged, no prediction ensemble',
            'adjusted_confidence':.9875,'source_files_and_capsule_unchanged':475,
            'scientific_model_constructors_forwards_fits_RT_or_cache_calls':0,
            'original_service_confirmation_opened':False,'seal_verified':False,
            'audit_script_sha256':digest(Path(__file__)),'input_files_sha256':{str(p.relative_to(ROOT)):s for p,s in HASH.items() if p.is_relative_to(ROOT)}}
    if final:
        assert (OUT/'verification/manifest.json').is_file(),'Wait for original pipeline seal'
        from wirelessjev.spectral_scene import verification
        record['closure']=verification.verify_closure();assert record['closure']['passes']
        helper=ROOT/'scripts/run_scene_field_cached.py';spec=importlib.util.spec_from_file_location('read_only_ops_proof',helper)
        ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
        opdir=ROOT/'artifacts/radiojepa_spectral_ops'
        proof=ops.verify_ops_publication(ROOT,opdir/'amendment.json',opdir/'publication.json')
        record['ops_publication']={'commit':proof['publication']['commit'],'proof_files_sha256':proof['proof_files_sha256']}
        operation_dir=OUT/'operations'
        bootstraps=list(operation_dir.glob('*.bootstrap.json'));finals=list(operation_dir.glob('*.final-uncached.json'))
        assert len(bootstraps)==len(finals)==1
        bootstrap=read(bootstraps[0]);final_receipt=read(finals[0]);handoff=read(operation_dir/'field-cache-handoff.json')
        assert bootstrap['operations_proof']==proof
        assert bootstrap['original_full_verified_development'] is True
        assert bootstrap['scientific_source_changed'] is False
        assert bootstrap['new_training_or_model_fits'] is False
        assert bootstrap['service_confirmation_permitted'] is False and bootstrap['production_qualified'] is False
        ops.gate_permissions(bootstrap['gate'])
        assert ops.value_sha(bootstrap['gate'])==bootstrap['gate_sha256']
        assert ops.value_sha(bootstrap['dependency_files_sha256'])==bootstrap['dependency_set_sha256']
        for name,expected in bootstrap['dependency_files_sha256'].items():assert digest(ROOT/name)==expected,name
        assert final_receipt['bootstrap_sha256']==digest(bootstraps[0])
        assert final_receipt['gate_sha256']==bootstrap['gate_sha256']
        assert final_receipt['before_sha256']==final_receipt['after_sha256']==bootstrap['dependency_files_sha256']
        assert final_receipt['unchanged'] is True and final_receipt['original_full_verified_development'] is True
        assert final_receipt['before_original_passive_audit_and_seal'] is True
        assert handoff['field_prediction_files_at_first_entry']==0 and handoff['field_results_at_first_entry'] is False
        assert handoff['field_collector_sha256']==digest(OUT/'rt_data/field_test.json')
        assert handoff['scientific_source_receipt_sha256']==ident['source_receipt_sha256']
        assert handoff['operations_proof_sha256']==ops.value_sha(proof)
        hits=0;full_replays=0;hitfiles=list(operation_dir.glob('*.hits.jsonl'));assert hitfiles
        for logfile in hitfiles:
            for index,line in enumerate(logfile.read_text().splitlines(),1):
                event=json.loads(line);assert event['process_hit_index']==index
                assert event['cache_id']==bootstrap['cache_id']
                assert event['bootstrap_sha256']==digest(bootstraps[0])
                assert event['gate_sha256']==bootstrap['gate_sha256']
                assert event['dependency_set_sha256']==bootstrap['dependency_set_sha256']
                assert event['all_dependency_bytes_and_membership_checked'] is True and event['source_guard_checked'] is True
                assert type(event['original_full_gate_reexecuted']) is bool
                assert event['final_uncached_before_original_passive_audit']==event['original_full_gate_reexecuted']
                if event['original_full_gate_reexecuted']:
                    assert event['full_gate_elapsed_seconds']==final_receipt['elapsed_seconds']
                    full_replays+=1
                else:assert event['full_gate_elapsed_seconds'] is None
                assert event['service_confirmation_permitted'] is False and event['production_qualified'] is False
                hits+=1
            digest(logfile)
        assert full_replays==1
        digest(finals[0]);digest(operation_dir/'field-cache-handoff.json')
        record['operational_evidence']={'bootstrap_and_final_original_full_gate_same':True,
            'zero_prediction_first_handoff':True,'dependency_files_verified':len(bootstrap['dependency_files_sha256']),
            'sequential_checked_hits':hits,'full_final_original_gate_replays':full_replays,
            'cache_methods_invoked_by_independent_audit':0}
        record['seal_verified']=True;record['sealed_manifest_sha256']=digest(OUT/'verification/manifest.json')
        # verify_closure and publication proof are read-only. Never invoke cache
        # bootstrap/hits/final_uncached against the sealed evidence namespace.
    for path,expected in HASH.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,str(path)
    record['input_files_sha256']={str(p.relative_to(ROOT)):s for p,s in HASH.items() if p.is_relative_to(ROOT)}
    DEST.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:record[k] for k in ('passes','field_geometries','prediction_receipts','targets_exactly_replayed','cell_seed_dependencies_bound','seal_verified','contrasts')},indent=2))
    print('record',str(DEST),'sha256',hashlib.sha256(DEST.read_bytes()).hexdigest())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();main(args.final)
