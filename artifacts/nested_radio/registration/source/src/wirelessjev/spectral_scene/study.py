"""Finite source-bound spectral RF experiment; simulated fields, never SERVICE.

Numerical summaries consume persisted per-layout/member predictions. The FIELD
split stays physically unopened after a development veto. Cached science is
validated and resumed, never silently recomputed or replaced.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

from . import common as c

INITIAL=('joint_none','joint_sigreg','joint_vicreg')
CONTROLS=('matched_regression','geometry_only')
SEEDS=(0,1,2)
REFERENCES=('operator_only','radiomap_operator','bayesian_operator')
METRICS=('rmse_db','p95_abs_db','observation_nll')
TARGETS=('center_dbm','lower_dbm','upper_dbm','heard','censored','censor_upper_dbm')


def reference(path):
    path=Path(path)
    return {'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path)}


def verify_reference(value):
    path=c.safe_path(value['path'])
    if c.digest(path)!=value['sha256']:raise RuntimeError('Spectral evidence reference changed')
    return path


def identity():
    c.check_sources()
    return {'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
            'protocol_sha256':c.digest(c.PROTOCOL),'registry_sha256':c.registry_sha256()}


def _trace_one(task):
    stage,index=task
    c.authorize_stage(stage)
    from ..radiojepa import rt_data as rt
    spec=c.field_spec(stage,index);folder=c.rt_scene_path(stage,index)
    if (folder/'receipt.json').exists():
        rt.validate_scene_receipt(spec,folder,stage)
    else:
        if folder.exists() and any(folder.rglob('*')):
            raise RuntimeError('Partial RT evidence exists; refuse an untracked retrace')
        profiles=tuple(p for p in rt.DEVICE_PROFILES if not p.held_out) if stage in ('training','validation') else rt.DEVICE_PROFILES
        rt._trace_scene(spec,folder,profiles=profiles)
        rt.validate_scene_receipt(spec,folder,stage)
    c.check_sources()
    return spec.scene_id,reference(folder/'receipt.json')


def validate_stage(stage):
    """Passive raw byte membership check, including private solver audit fields."""
    from ..radiojepa import rt_data as rt
    c.authorize_stage(stage)
    path=c.OUT/'rt_data'/f'{stage}.json';value=c.read_json(path)
    expected={c.field_spec(stage,i).scene_id for i in range(c.COUNTS[stage])}
    if (value.get('completed') is not True or value.get('stage')!=stage
            or value.get('source_receipt_sha256')!=c.digest(c.OUT/'registration/source.json')
            or value.get('registry_sha256')!=c.registry_sha256()
            or value.get('authorization')!=c.authorize_stage(stage)
            or set(value.get('scene_receipts',{}))!=expected
            or value.get('real_device_measurements') is not False
            or value.get('original_service_confirmation_consumed') is not False):
        raise RuntimeError('RT stage identity or membership changed')
    actual_dirs={p.name for p in (c.OUT/'rt_data'/stage).iterdir() if p.is_dir()}
    if actual_dirs!=expected:raise RuntimeError('RT stage directory membership changed')
    for i in range(c.COUNTS[stage]):
        spec=c.field_spec(stage,i);folder=c.rt_scene_path(stage,i)
        entry=value['scene_receipts'][spec.scene_id]
        if verify_reference(entry)!=folder/'receipt.json':raise RuntimeError('RT scene reference redirected')
        rt.validate_scene_receipt(spec,folder,stage)
    return value


def collect_stage(stage,workers=2):
    if workers not in (1,2):raise ValueError('Use one or two bounded CPU workers')
    c.authorize_stage(stage)
    path=c.OUT/'rt_data'/f'{stage}.json'
    if path.exists():return validate_stage(stage)
    with c.phase_guard('collect_'+stage):
        tasks=[(stage,i) for i in range(c.COUNTS[stage])]
        if workers==1:entries=list(map(_trace_one,tasks))
        else:
            # No Mitsuba/DrJit solver is initialized in this parent process.
            with ProcessPoolExecutor(max_workers=workers) as pool:entries=list(pool.map(_trace_one,tasks))
        result={'schema':1,'stage':stage,'completed':True,
                'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
                'registry_sha256':c.registry_sha256(),'authorization':c.authorize_stage(stage),
                'scene_receipts':dict(entries),'real_device_measurements':False,
                'original_service_confirmation_consumed':False}
        c.write_once(path,result)
    return validate_stage(stage)


def observation_metrics(mean,std,targets):
    """Heard point errors and proper interval/upper-censor likelihood separately."""
    mean,std=np.asarray(mean,dtype='f8'),np.asarray(std,dtype='f8')
    arrays={name:np.asarray(targets[name]) for name in TARGETS}
    if mean.ndim!=1 or std.shape!=mean.shape or any(a.shape!=mean.shape for a in arrays.values()):
        raise ValueError('Prediction and target axes differ')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise FloatingPointError('Nonfinite prediction or invalid likelihood spread')
    heard=arrays['heard'].astype(bool);censored=arrays['censored'].astype(bool)
    if (heard&censored).any() or not heard.any():raise ValueError('Need non-overlapping heard/censored masks and heard endpoint data')
    residual=mean[heard]-arrays['center_dbm'][heard]
    if not np.isfinite(residual).all():raise FloatingPointError('Nonfinite heard target')
    hi=(arrays['upper_dbm'][heard]-mean[heard])/std[heard]
    lo=(arrays['lower_dbm'][heard]-mean[heard])/std[heard]
    if (hi<=lo).any():raise ValueError('Heard report needs a positive quantization interval')
    # Reflect intervals in the positive tail to avoid catastrophic Phi subtraction.
    positive=lo>0
    upper=np.where(positive,log_ndtr(-lo),log_ndtr(hi))
    lower=np.where(positive,log_ndtr(-hi),log_ndtr(lo))
    interval_logp=upper+np.log(-np.expm1(np.minimum(lower-upper,-np.finfo(float).eps)))
    censor_logp=log_ndtr((arrays['censor_upper_dbm'][censored]-mean[censored])/std[censored])
    nll=-np.concatenate((interval_logp,censor_logp))
    if not np.isfinite(nll).all():raise FloatingPointError('Nonfinite observation likelihood')
    return {'rmse_db':float(np.sqrt(np.mean(residual**2))),
            'p95_abs_db':float(np.quantile(np.abs(residual),.95)),
            'observation_nll':float(nll.mean()),'heard_count':int(heard.sum()),
            'censored_count':int(censored.sum()),'missing_count':int((~(heard|censored)).sum()),
            'finite':True}


def select_prior(fits):
    """Only three independent TRAIN members and their immutable VAL scores."""
    candidates={}
    for cell in INITIAL:
        values=[fits[(cell,seed)] for seed in SEEDS]
        scores=[float(v['validation_mean_layout_rmse_db']) for v in values]
        if not np.isfinite(scores).all():raise ValueError('Nonfinite validation selection')
        candidates[cell]={'mean_rmse_db':float(np.mean(scores)),'seed_rmse_db':scores}
    selected=min(INITIAL,key=lambda name:candidates[name]['mean_rmse_db'])
    return {'selected_cell':selected,'selected_prior':selected.removeprefix('joint_'),
            'score':'mean_seed_mean_layout_validation_rmse_db','candidates':candidates}


def write_selection():
    c.check_sources()
    fits={(cell,seed):c.read_json(c.OUT/'fits'/f'{cell}-seed{seed}.json') for cell in INITIAL for seed in SEEDS}
    result=select_prior(fits)|identity()
    for cell in INITIAL:
        result['candidates'][cell]['fit_receipts_sha256']=[c.digest(c.OUT/'fits'/f'{cell}-seed{s}.json') for s in SEEDS]
    c.write_once(c.OUT/'selection.json',result)
    return result


def verified_selection():
    value=c.read_json(c.OUT/'selection.json')
    fits={(cell,seed):c.read_json(c.OUT/'fits'/f'{cell}-seed{seed}.json') for cell in INITIAL for seed in SEEDS}
    expected=select_prior(fits)|identity()
    for cell in INITIAL:
        expected['candidates'][cell]['fit_receipts_sha256']=[c.digest(c.OUT/'fits'/f'{cell}-seed{s}.json') for s in SEEDS]
    if value!=expected:raise RuntimeError('Validation-only prior choice changed')
    return value


def _keys(regime):
    from .data import ViewKey
    bands=(0,1) if regime=='in_band' or regime.startswith('onboarding_') else (2,) if regime=='held_band' else None
    if bands is None:raise ValueError('Unknown registered field endpoint')
    return tuple(ViewKey(b,d,device) for b in bands for d in (0,1) for device in ((3,) if regime.startswith('onboarding_') else (0,1,2)))



def onboarding_availability(scene_id,index,requested,available,heard):
    """Only descriptive support failure; never drop a primary geometry."""
    if requested not in (0,1,4,16) or available<0 or heard<0:raise ValueError('Invalid genuine onboarding support counts')
    reasons=[]
    if available<requested:reasons.append('insufficient_genuine_context_pairs')
    if heard==0:reasons.append('no_heard_held_device_query')
    if not reasons:return None
    return {'scene_id':scene_id,'scene_index':index,'available':False,
        'requested_context_pairs':requested,'available_context_pairs':available,
        'heard_query_count':heard,'reason':'+'.join(reasons),
        'scope':'descriptive onboarding only; primary geometry retained; no substitutes'}

def _examples(stage,regime):
    from . import data
    role=('onboarding' if regime.startswith('onboarding_') else 'development' if regime=='in_band' else 'held_band') if stage=='development' else 'field_test'
    pairs=int(regime.removeprefix('onboarding_')) if regime.startswith('onboarding_') else None
    if pairs is not None:
        base=data.load_calibration();result=[]
        for index in range(c.COUNTS[stage]):
            scene=data.load_scene(stage,index,role=role)
            available=len(data._paired_reports(scene,3))
            initial=data.make_example(scene,calibration=base,query_keys=_keys(regime))
            heard=int(initial.targets.heard.sum())
            status=onboarding_availability(scene.scene_id,index,pairs,available,heard)
            if status is not None:result.append(status);continue
            current=data.onboard_device(scene,base,n_pairs=pairs)
            result.append(data.make_example(scene,calibration=current,query_keys=_keys(regime)))
        return result
    return data.examples(stage,role=role,query_keys=_keys(regime),
                         onboarding_pairs=pairs,query_poses=data.CONTEXT_POSES if regime=='held_band' else data.QUERY_POSES)


def _cache(path,arrays,metadata):
    receipt=path.with_suffix('.json')
    if receipt.exists():
        saved=c.read_json(receipt)
        if saved['metadata']!=metadata or c.digest(path)!=saved['npz_sha256']:
            raise RuntimeError('Cached predictions changed identity')
        return saved
    if path.exists():raise RuntimeError('Unreceipted prediction bytes exist')
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    c.write_bytes_once(path,stream.getvalue())
    value={'schema':1,'metadata':metadata,'npz_path':str(path.relative_to(c.ROOT)),
           'npz_sha256':c.digest(path),'metrics':observation_metrics(arrays['mean'],arrays['std'],arrays)}
    c.write_once(receipt,value)
    return value


def _predict_member(stage,regime,name,seed,examples,*,intervention=None):
    from . import training
    ident=identity();model=None;codes=None
    if seed is not None:
        model=training.load(name,seed)
        codes=[None if isinstance(e,dict) else training.encode(model,e).numpy()[0] for e in examples]
    if intervention not in (None,'zero','shuffle'):raise ValueError('Unknown latent intervention')
    member=name if seed is None else f'{name}-seed{seed}'
    if intervention:member+='-'+intervention
    if seed is None:
        if intervention:raise ValueError('A deterministic baseline has no learned latent')
        if name!='operator_only':
            from . import baselines
    records=[]
    for index,example in enumerate(examples):
        if isinstance(example,dict):
            if not regime.startswith('onboarding_') or example.get('available') is not False:raise RuntimeError('Only explicit descriptive onboarding support may be unavailable')
            records.append({'scene_id':example['scene_id'],'available':False,'availability':example});continue
        scene=example.query_rows.scene_id
        role=('onboarding' if regime.startswith('onboarding_') else 'development' if regime=='in_band' else 'held_band') if stage=='development' else 'field_test'
        dependencies={'calibration':reference(c.OUT/'data/frontend-training.json'),
            'scene_data':reference(c.OUT/'data'/f'scene-{stage}-{index:02d}-{role}.json'),
            'stage':reference(c.OUT/'rt_data'/f'{stage}.json')}
        if seed is not None:
            dependencies['fit']=reference(c.OUT/'fits'/f'{name}-seed{seed}.json')
            dependencies['checkpoint']=reference(c.OUT/'checkpoints'/f'{name}-seed{seed}.pt')
        elif name!='operator_only':dependencies['baseline']=reference(c.OUT/'baselines/prior.json')
        metadata={'identity':ident,'stage':stage,'regime':regime,'scene_id':scene,'scene_index':index,
            'member':member,'cell':name,'seed':seed,'intervention':intervention,'source_view':'a',
            'dependencies':dependencies,'predictions_ensembled':False,'uncertainty_calibrated':False,
            'query_measurement_ids':list(example.query_rows.measurement_ids),
            'context_a_measurement_ids':list(example.context_rows_a.measurement_ids),
            'context_b_measurement_ids':list(example.context_rows_b.measurement_ids)}
        path=c.OUT/'predictions'/stage/regime/member/f'{scene}.npz'
        if path.with_suffix('.json').exists():
            saved=c.read_json(path.with_suffix('.json'))
            if saved['metadata']!=metadata or c.digest(path)!=saved['npz_sha256']:
                raise RuntimeError('Refuse changed scientific prediction cache')
        else:
            if seed is not None:
                override=np.roll(codes,1,axis=0)[index] if intervention=='shuffle' else None
                mean,std=training.predict(model,example,intervention='zero' if intervention=='zero' else None,z_override=override)
            elif name=='operator_only':
                mean=example.query_fspl_dbm+example.query_offset_db
                std=np.full_like(mean,4.)
            else:mean,std=baselines.predict(name,example)
            arrays={**example.model_inputs(),**example.target_arrays(),'mean':np.asarray(mean),
                'std':np.asarray(std),'scene_latent':np.empty((0,),dtype='f4') if codes is None else codes[index],
                'band_index':example.query_rows.band_index,'device_index':example.query_rows.device,
                'pose_index':example.query_rows.pose_index,'direction':example.query_rows.direction}
            saved=_cache(path,arrays,metadata)
        records.append({'scene_id':scene,'available':True,'prediction':reference(path.with_suffix('.json')),'metrics':saved['metrics']})
    return {'member':member,'cell':name,'seed':seed,'intervention':intervention,'layouts':records}


def _summary_path(stage):return c.OUT/'results'/f'{stage}.json'



def frequency_diagnostic(stage,selected,examples=None,*,write=False):
    """Descriptive identical-link comparison, conditional on real source reports."""
    from . import baselines
    if examples is None:examples=_examples(stage,'held_band')
    arrays={};rows=[];dependencies=[]
    for index,example in enumerate(examples):
        scene=example.query_rows.scene_id
        mean,std,eligible=baselines.direct_same_link_frequency(example)
        models=[]
        for seed in SEEDS:
            receipt_path=c.OUT/'predictions'/stage/'held_band'/f'{selected}-seed{seed}'/f'{scene}.json'
            receipt=c.read_json(receipt_path);dependencies.append(reference(receipt_path))
            with np.load(c.safe_path(receipt['npz_path']),allow_pickle=False) as saved:models.append(saved['mean'])
        models=np.asarray(models);matched=eligible&example.targets.heard
        n=int(matched.sum())
        rows.append({'scene_id':scene,'eligible_source_query_count':int(eligible.sum()),
            'matched_heard_query_count':n,'all_heard_query_count':int(example.targets.heard.sum()),
            'direct_rmse_db':float(np.sqrt(np.mean((mean[matched]-example.targets.center_dbm[matched])**2))) if n else None,
            'selected_seed_rmse_db':[float(np.sqrt(np.mean((m[matched]-example.targets.center_dbm[matched])**2))) for m in models] if n else None})
        for name,value in {'direct_mean':mean,'direct_std':std,'eligible':eligible,'selected_mean':models,
            'target_center':example.targets.center_dbm,'target_heard':example.targets.heard}.items():
            arrays[f'layout{index:02d}_{name}']=value
    path=c.OUT/'diagnostics'/f'{stage}-same-link-frequency.npz'
    metadata={'schema':1,'stage':stage,'identity':identity(),'selected_cell':selected,'layouts':rows,
        'dependencies':dependencies,'cohort':'genuine heard5.2GHz source AND heard6GHz query, known frontend gauge',
        'assumptions':'fixed material/antenna spectral response and calibrated reciprocity; quantization spread only',
        'uncertainty_calibrated':False,'primary_endpoint':False,'npz_path':str(path.relative_to(c.ROOT))}
    if write:
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);c.write_bytes_once(path,stream.getvalue())
        metadata['npz_sha256']=c.digest(path);c.write_once(path.with_suffix('.json'),metadata)
    else:
        actual=c.read_json(path.with_suffix('.json'))
        if actual!={**metadata,'npz_sha256':c.digest(path)}:raise RuntimeError('Direct frequency comparison metadata changed')
        with np.load(path,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays):raise RuntimeError('Direct frequency diagnostic array membership changed')
            for name,value in arrays.items():
                if not np.array_equal(saved[name],value,equal_nan=True):raise RuntimeError('Direct frequency diagnostic does not replay')
    return reference(path.with_suffix('.json'))

def evaluate_stage(stage):
    if stage not in ('development','field_test'):raise ValueError('Not an evaluation stage')
    c.authorize_stage(stage);validate_stage(stage);selection=verified_selection()
    path=_summary_path(stage)
    if path.exists():return verify_evaluation(stage)
    selected=selection['selected_cell'];members={}
    with c.phase_guard('evaluate_'+stage):
        for regime in ('in_band','held_band','onboarding_0','onboarding_1','onboarding_4','onboarding_16'):
            examples=_examples(stage,regime);values={}
            for name in ((selected,) if regime.startswith('onboarding_') else (selected,*CONTROLS)):
                for seed in SEEDS:
                    value=_predict_member(stage,regime,name,seed,examples);values[value['member']]=value
            for name in REFERENCES:
                value=_predict_member(stage,regime,name,None,examples);values[value['member']]=value
            if regime=='in_band':
                for intervention in ('zero','shuffle'):
                    for seed in SEEDS:
                        value=_predict_member(stage,regime,selected,seed,examples,intervention=intervention)
                        values[value['member']]=value
            members[regime]=values
        result={'schema':1,'stage':stage,'identity':identity(),'selection':reference(c.OUT/'selection.json'),
            'members':members,'source_view':'a','seed_metrics_averaged_not_predictions':True,
            'unit':'independent simulated geometry','service_or_production_claim':False,
            'held_device_onboarding_evaluated':True,'source_view_b_evaluated':False,
            'same_link_frequency':frequency_diagnostic(stage,selected,write=True)}
        c.write_once(path,result)
    return verify_evaluation(stage)


def verify_evaluation(stage):
    """Passive summary recomputation; no examples, model imports or forwards."""
    value=c.read_json(_summary_path(stage));ident=identity();expected_prediction_files=set();expected_ids=[c.field_spec(stage,i).scene_id for i in range(c.COUNTS[stage])]
    if value['stage']!=stage or value['identity']!=ident:raise RuntimeError('Evaluation identity changed')
    verify_reference(value['selection'])
    selected=verified_selection()['selected_cell']
    regimes={'in_band','held_band','onboarding_0','onboarding_1','onboarding_4','onboarding_16'}
    if set(value['members'])!=regimes:raise RuntimeError('Evaluation regime membership changed')
    for regime,members in value['members'].items():
        cells=(selected,) if regime.startswith('onboarding_') else (selected,*CONTROLS)
        expected_names={f'{cell}-seed{s}' for cell in cells for s in SEEDS}|set(REFERENCES)
        if regime=='in_band':expected_names|={f'{selected}-seed{s}-{i}' for s in SEEDS for i in ('zero','shuffle')}
        if set(members)!=expected_names:raise RuntimeError('Evaluation member membership changed')
        for name,member in members.items():
            if member['member']!=name or [x['scene_id'] for x in member['layouts']]!=expected_ids:
                raise RuntimeError('Prediction member/layout membership changed')
            for index,row in enumerate(member['layouts']):
                if row.get('available') is False:
                    if (not regime.startswith('onboarding_') or set(row)!={'scene_id','available','availability'}
                        or row['availability'].get('scene_id')!=row['scene_id'] or row['availability'].get('scene_index')!=index):
                        raise RuntimeError('Unavailable descriptive assay membership changed')
                    continue
                if row.get('available') is not True:raise RuntimeError('Primary prediction availability missing')
                receipt_path=verify_reference(row['prediction'])
                expected_path=c.OUT/'predictions'/stage/regime/name/f"{row['scene_id']}.json"
                expected_prediction_files|={expected_path,expected_path.with_suffix('.npz')}
                if receipt_path!=expected_path:raise RuntimeError('Prediction receipt redirected')
                receipt=c.read_json(receipt_path)
                path=c.safe_path(receipt['npz_path'])
                if path!=expected_path.with_suffix('.npz') or c.digest(path)!=receipt['npz_sha256']:raise RuntimeError('Stored predictions changed or redirected')
                meta=receipt['metadata']
                expected_seed=member['seed'];expected_cell=member['cell'];expected_intervention=member['intervention']
                if expected_seed is None:
                    if expected_cell not in REFERENCES or expected_cell!=name or expected_intervention is not None:raise RuntimeError('Baseline identity changed')
                elif expected_seed not in SEEDS or expected_cell not in cells or name!=f'{expected_cell}-seed{expected_seed}'+('' if expected_intervention is None else '-'+expected_intervention):
                    raise RuntimeError('Neural member identity changed')
                expected_dependencies={'calibration','scene_data','stage'}|({'fit','checkpoint'} if expected_seed is not None else {'baseline'} if expected_cell!='operator_only' else set())
                if (set(meta['dependencies'])!=expected_dependencies or meta['cell']!=expected_cell or meta['seed']!=expected_seed
                    or meta['intervention']!=expected_intervention or meta['predictions_ensembled'] is not False
                    or meta['uncertainty_calibrated'] is not False):raise RuntimeError('Prediction metadata/dependency membership changed')
                if (meta['identity']!=ident or meta['stage']!=stage or meta['regime']!=regime
                    or meta['member']!=name or meta['scene_id']!=row['scene_id'] or meta['source_view']!='a'):
                    raise RuntimeError('Prediction identity changed')
                role=('onboarding' if regime.startswith('onboarding_') else 'development' if regime=='in_band' else 'held_band') if stage=='development' else 'field_test'
                expected_refs={'calibration':reference(c.OUT/'data/frontend-training.json'),
                    'scene_data':reference(c.OUT/'data'/f'scene-{stage}-{index:02d}-{role}.json'),
                    'stage':reference(c.OUT/'rt_data'/f'{stage}.json')}
                if expected_seed is not None:
                    expected_refs['fit']=reference(c.OUT/'fits'/f'{expected_cell}-seed{expected_seed}.json')
                    expected_refs['checkpoint']=reference(c.OUT/'checkpoints'/f'{expected_cell}-seed{expected_seed}.pt')
                elif expected_cell!='operator_only':expected_refs['baseline']=reference(c.OUT/'baselines/prior.json')
                if meta['dependencies']!=expected_refs or meta['scene_index']!=index:raise RuntimeError('Prediction dependency paths/bytes changed')
                for dependency in meta['dependencies'].values():verify_reference(dependency)
                with np.load(path,allow_pickle=False) as arrays:
                    metrics=observation_metrics(arrays['mean'],arrays['std'],{k:arrays[k] for k in TARGETS})
                    if np.any(~np.isin(arrays['pose_index'],[0,2,4,6,8,10] if regime=='held_band' else [1,3,5,7,9,11])):
                        raise RuntimeError('Context reports entered query endpoint')
                    if set(arrays['band_index'].tolist())!=({2} if regime=='held_band' else {0,1}):
                        raise RuntimeError('Held frequency leaked or endpoint axes changed')
                    if set(arrays['device_index'].tolist())!=({3} if regime.startswith('onboarding_') else {0,1,2}):raise RuntimeError('Held receiver leaked into primary endpoint')
                if metrics!=receipt['metrics'] or metrics!=row['metrics']:raise RuntimeError('Prediction metrics do not recompute')
                if set(meta['query_measurement_ids'])&set(meta['context_a_measurement_ids']+meta['context_b_measurement_ids']):
                    raise RuntimeError('Targets overlap encoded context')
    actual_prediction_files={p for p in (c.OUT/'predictions'/stage).rglob('*') if p.is_file()}
    if actual_prediction_files!=expected_prediction_files:raise RuntimeError('Prediction file membership changed')
    return value


def layout_values(evaluation,cell,metric,regime='in_band',*,intervention=None):
    members=evaluation['members'][regime]
    if not regime.startswith('onboarding_') and any(r.get('available') is False for m in members.values() for r in m['layouts']):
        raise RuntimeError('Primary RF endpoints cannot drop unsupported geometries')
    if cell in REFERENCES:
        return np.array([r['metrics'][metric] for r in members[cell]['layouts'] if r.get('available',True)],dtype='f8')
    names=[f'{cell}-seed{s}'+('' if intervention is None else '-'+intervention) for s in SEEDS]
    values=np.array([[r['metrics'][metric] for r in members[name]['layouts'] if r.get('available',True)] for name in names],dtype='f8')
    return values.mean(axis=0)



def embedding_diagnostics(evaluation,selected):
    values=[]
    for seed in SEEDS:
        codes=[]
        for row in evaluation['members']['in_band'][f'{selected}-seed{seed}']['layouts']:
            receipt=c.read_json(verify_reference(row['prediction']))
            with np.load(c.safe_path(receipt['npz_path']),allow_pickle=False) as arrays:codes.append(arrays['scene_latent'])
        codes=np.asarray(codes,dtype='f8');singular=np.linalg.svd(codes-codes.mean(0),compute_uv=False)
        energy=singular**2
        values.append({'seed':seed,'distinct_layouts':len(codes),'centered_rank':int(np.sum(singular>1.e-6)),
            'centered_rank_upper_bound':min(len(codes)-1,8),'singular_values':singular.tolist(),
            'participation_rank':float(energy.sum()**2/np.sum(energy**2)) if np.sum(energy**2)>0 else 0.})
    return values


def coverage(evaluation,selected):
    return {regime:[({'scene_id':row['scene_id'],**{k:row['metrics'][k] for k in ('heard_count','censored_count','missing_count')}} if row.get('available',True) else row['availability'])
        for row in evaluation['members'][regime][f'{selected}-seed0']['layouts']] for regime in evaluation['members']}



def onboarding_curve(evaluation,selected):
    rows=[]
    for n in (0,1,4,16):
        regime=f'onboarding_{n}';layouts=evaluation['members'][regime][f'{selected}-seed0']['layouts']
        eligible=[r for r in layouts if r.get('available',True)];unavailable=[r['availability'] for r in layouts if r.get('available') is False]
        rows.append({'context_report_pairs':n,'registered_layouts':len(layouts),'available_layouts':len(eligible),
            'unavailable_layouts':unavailable,
            'selected_mean_rmse_db':float(layout_values(evaluation,selected,'rmse_db',regime).mean()) if eligible else None,
            'radiomap_mean_rmse_db':float(layout_values(evaluation,'radiomap_operator','rmse_db',regime).mean()) if eligible else None,
            'query_devices':[3],'query_bands':[0,1],'query_poses':'odd, disjoint from calibration support',
            'scope':'simulated RSSI offset/quantization inverse from genuine even-pose pairs; known-deviceA encoder unchanged, no neural fine-tuning or real AGC/CFO calibration; nested supports correlated'})
    return rows

def development_gate(evaluation,selected):
    learned=layout_values(evaluation,selected,'rmse_db');reference_values=layout_values(evaluation,'radiomap_operator','rmse_db')
    p95=layout_values(evaluation,selected,'p95_abs_db');refp95=layout_values(evaluation,'radiomap_operator','p95_abs_db')
    zero=layout_values(evaluation,selected,'rmse_db',intervention='zero')
    seed_rmse=[float(np.mean([r['metrics']['rmse_db'] for r in evaluation['members']['in_band'][f'{selected}-seed{s}']['layouts']])) for s in SEEDS]
    finite=all(r['metrics'].get('finite') is True for regime,group in evaluation['members'].items() for member in group.values() for r in member['layouts'] if r.get('available',True))
    rmse=float(learned.mean());map_rmse=float(reference_values.mean())
    deterioration=float(zero.mean()/rmse-1) if rmse>0 else 0.
    checks={'mean_rf_rmse_better':rmse<map_rmse,'mean_p95_better':float(p95.mean())<float(refp95.mean()),
        'every_seed_rmse_no_worse':all(score<=map_rmse for score in seed_rmse),
        'all_predictions_finite':finite,'zero_latent_hurts_at_least_one_percent':deterioration>=.01}
    return {'field_test_permitted':all(checks.values()),'checks':checks,'selected_cell':selected,
        'selected_mean_rmse_db':rmse,'radiomap_mean_rmse_db':map_rmse,
        'selected_mean_p95_abs_db':float(p95.mean()),'radiomap_mean_p95_abs_db':float(refp95.mean()),
        'selected_seed_mean_rmse_db':seed_rmse,'zero_latent_relative_rmse_degradation':deterioration,
        'shuffle_latent_mean_rmse_db':float(layout_values(evaluation,selected,'rmse_db',intervention='shuffle').mean()),
        'rank_is_descriptive_not_gate':True,'original_service_confirmation_permitted':False,
        'production_qualified':False}


def write_development():
    evaluation=verify_evaluation('development');selection=verified_selection()
    from .verification import passive_guard,verify_public_arrays
    with passive_guard():verify_public_arrays('development')
    result=development_gate(evaluation,selection['selected_cell'])|identity()|{
        'embedding_diagnostics':embedding_diagnostics(evaluation,selection['selected_cell']),
        'coverage':coverage(evaluation,selection['selected_cell']),'onboarding':onboarding_curve(evaluation,selection['selected_cell']),
        'same_link_frequency':evaluation['same_link_frequency'],
        'evaluation':reference(_summary_path('development')),'selection':reference(c.OUT/'selection.json')}
    c.write_once(c.OUT/'development.json',result)
    qualification={k:result[k] for k in ('field_test_permitted','source_receipt_sha256','registry_sha256','original_service_confirmation_permitted')}
    qualification.update(development_sha256=c.digest(c.OUT/'development.json'),production_qualified=False)
    c.write_once(c.OUT/'qualification.json',qualification)
    return result


def verified_development():
    validate_stage('development')
    evaluation=verify_evaluation('development');selection=verified_selection()
    from .verification import passive_guard,verify_public_arrays
    with passive_guard():verify_public_arrays('development')
    expected=development_gate(evaluation,selection['selected_cell'])|identity()|{
        'embedding_diagnostics':embedding_diagnostics(evaluation,selection['selected_cell']),
        'coverage':coverage(evaluation,selection['selected_cell']),'onboarding':onboarding_curve(evaluation,selection['selected_cell']),
        'same_link_frequency':evaluation['same_link_frequency'],
        'evaluation':reference(_summary_path('development')),'selection':reference(c.OUT/'selection.json')}
    value=c.read_json(c.OUT/'development.json')
    if value!=expected:raise RuntimeError('Development gate no longer recomputes from stored predictions')
    return value


def paired_bootstrap(candidate,reference_values,*,samples=20000,seed=21749):
    candidate,reference_values=np.asarray(candidate,dtype='f8'),np.asarray(reference_values,dtype='f8')
    if candidate.ndim!=1 or reference_values.shape!=candidate.shape or len(candidate)<2:
        raise ValueError('Bootstrap needs paired independent whole-layout scores')
    differences=candidate-reference_values
    if not np.isfinite(differences).all():raise ValueError('Nonfinite bootstrap endpoint')
    rng=np.random.default_rng(seed);indices=rng.integers(len(differences),size=(samples,len(differences)))
    boot=differences[indices].mean(axis=1)
    adjusted=np.quantile(boot,[.00625,.99375]);raw=np.quantile(boot,[.025,.975])
    return {'layouts':len(differences),'candidate_mean':float(candidate.mean()),'reference_mean':float(reference_values.mean()),
        'mean_paired_difference':float(differences.mean()),'adjusted_confidence':.9875,
        'adjusted_interval':adjusted.tolist(),'descriptive_95_interval':raw.tolist(),
        'candidate_better_adjusted':bool(adjusted[1]<0),'bootstrap_samples':samples,'bootstrap_seed':seed,
        'paired_unit':'whole independent geometry, seed metrics averaged within layout',
        'interval_scope':'layout resampling conditional on three trained members; not uncertainty over retraining'}


def field_results(evaluation,selected):
    contrasts=[]
    for slot in c.protocol()['field_tests']:
        candidate=layout_values(evaluation,selected,slot['metric'],slot['regime'])
        reference_values=layout_values(evaluation,slot['reference'],slot['metric'],slot['regime'])
        contrasts.append(dict(slot)|paired_bootstrap(candidate,reference_values))
    return {'schema':1,'stage':'field_test','identity':identity(),'selected_cell':selected,
        'evaluation':reference(_summary_path('field_test')),'contrasts':contrasts,
        'same_link_frequency':evaluation['same_link_frequency'],
        'embedding_diagnostics':embedding_diagnostics(evaluation,selected),
        'coverage':coverage(evaluation,selected),'onboarding':onboarding_curve(evaluation,selected),
        'seed_main_rmse_db':[float(np.mean([r['metrics']['rmse_db'] for r in evaluation['members']['in_band'][f'{selected}-seed{s}']['layouts']])) for s in SEEDS],
        'rf_rmse_superiority':contrasts[0]['candidate_better_adjusted'],
        'matched_alignment_benefit_supported':contrasts[1]['candidate_better_adjusted'],
        'p95_superiority':contrasts[2]['candidate_better_adjusted'],
        'held_band_superiority':contrasts[3]['candidate_better_adjusted'],
        'production_qualified':False,'original_service_confirmation_permitted':False,
        'calibrated_uncertainty_claim':False,'universal_foundation_model_claim':False}


def write_field_results():
    c.authorize_stage('field_test');evaluation=verify_evaluation('field_test')
    result=field_results(evaluation,verified_selection()['selected_cell'])
    c.write_once(c.OUT/'report.json',result)
    return result


def run(workers=2):
    """Single finite campaign. A veto is a completed, sealed outcome, not a retry."""
    from . import data,training,baselines
    with c.lock(c.WORK/'supervisor.lock',nonblocking=True):
        c.check_sources();c.verify_previous()
        if (c.OUT/'verification/manifest.json').exists():
            from .verification import verify_closure
            return verify_closure()
        for stage in ('training','validation'):collect_stage(stage,workers)
        with c.phase_guard('prepare_training'):data.prepare_training()
        with c.phase_guard('initial9'):training.train_initial(workers)
        selection=write_selection()
        with c.phase_guard('controls6'):training.train_controls(selection['selected_prior'],workers)
        with c.phase_guard('classical_controls'):baselines.prepare()
        collect_stage('development',workers);evaluate_stage('development');gate=write_development()
        if gate['field_test_permitted']:
            collect_stage('field_test',workers);evaluate_stage('field_test');write_field_results()
        else:
            if (c.OUT/'rt_data/field_test').exists():raise RuntimeError('Vetoed FIELD geometries were opened')
            c.write_once(c.OUT/'report.json',{'schema':1,'identity':identity(),'status':'development_veto',
                'development':reference(c.OUT/'development.json'),'field_test_opened':False,
                'rf_rmse_superiority':False,'matched_alignment_benefit_supported':False,
                'production_qualified':False,'original_service_confirmation_permitted':False})
        from . import verification
        verification.run_audit();verification.seal()
        return verification.verify_closure()
