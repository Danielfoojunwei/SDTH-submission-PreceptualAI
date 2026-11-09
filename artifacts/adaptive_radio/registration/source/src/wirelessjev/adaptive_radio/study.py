"""Finite paired-condition RF study on fresh geometries, with no SERVICE claim."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr,ndtr

from . import common as c

CELLS=('joint','regression','rf_erased','pure_geometry')
SEEDS=(0,1,2)
REGIMES=('full','sparse')
BASELINES=('radiomap_operator','bayesian_operator')
TARGET_NAMES=('center_dbm','lower_dbm','upper_dbm','heard','censored','censor_upper_dbm','detection_threshold_dbm')
_CACHE=None
_TEST_DATA_NAMES={}


def reference(path):
    path=Path(path);return {'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path)}


def verify_reference(value):
    path=c.safe_path(value['path'])
    if c.digest(path)!=value['sha256']:raise RuntimeError('Adaptive evidence reference changed')
    return path


def identity():
    c.check_sources()
    return {'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
        'protocol_sha256':c.digest(c.PROTOCOL),'registry_sha256':c.registry_sha256()}


def _trace_one(task):
    from . import simulation as sim
    stage,index,condition=task;c.authorize_stage(stage)
    spec=c.case_spec(stage,index,condition);folder=c.case_path(stage,index,condition)
    if (folder/'receipt.json').exists():sim.validate_case_receipt(spec,folder)
    else:
        if folder.exists() and any(folder.rglob('*')):raise RuntimeError('Partial RT evidence exists; refuse untracked retrace')
        sim.trace_case(spec,folder);sim.validate_case_receipt(spec,folder)
    c.check_sources();return spec.case_id,reference(folder/'receipt.json')


def validate_stage(stage):
    from . import simulation as sim
    c.authorize_stage(stage);value=c.read_json(c.OUT/'rt_data'/f'{stage}.json')
    specs=[c.case_spec(stage,i,k) for i in range(c.COUNTS[stage]) for k in c.CONDITIONS]
    expected={s.case_id for s in specs}
    if (value.get('stage')!=stage or value.get('completed') is not True
        or value.get('source_receipt_sha256')!=c.digest(c.OUT/'registration/source.json')
        or value.get('registry_sha256')!=c.registry_sha256() or value.get('authorization')!=c.authorize_stage(stage)
        or value.get('layouts')!=c.COUNTS[stage] or value.get('conditions')!=3
        or set(value.get('case_receipts',{}))!=expected
        or value.get('real_device_measurements') is not False
        or value.get('original_service_confirmation_consumed') is not False):raise RuntimeError('Adaptive RT stage identity/membership changed')
    folder=c.OUT/'rt_data'/stage
    if {p.name for p in folder.iterdir() if p.is_dir()}!=expected:raise RuntimeError('Adaptive RT case directories changed')
    for spec in specs:
        path=verify_reference(value['case_receipts'][spec.case_id])
        expected_path=folder/spec.case_id/'receipt.json'
        if path!=expected_path:raise RuntimeError('Adaptive RT receipt redirected')
        sim.validate_case_receipt(spec,path.parent)
    return value


def collect_stage(stage,workers=2):
    if workers not in (1,2):raise ValueError('Use one or two bounded CPU workers')
    c.authorize_stage(stage);path=c.OUT/'rt_data'/f'{stage}.json'
    if path.exists():return validate_stage(stage)
    with c.phase_guard('collect_'+stage):
        tasks=[(stage,i,k) for i in range(c.COUNTS[stage]) for k in c.CONDITIONS]
        if workers==1:entries=list(map(_trace_one,tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:entries=list(pool.map(_trace_one,tasks))
        value={'schema':1,'stage':stage,'completed':True,'layouts':c.COUNTS[stage],'conditions':3,
            'case_receipts':dict(entries),'authorization':c.authorize_stage(stage),
            'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),'registry_sha256':c.registry_sha256(),
            'real_device_measurements':False,'original_service_confirmation_consumed':False}
        c.write_once(path,value)
    return validate_stage(stage)


def observation_metrics(mean,std,targets):
    """Probability/interval scores on known reports; no false point label for censoring."""
    mean,std=np.asarray(mean,dtype='f8'),np.asarray(std,dtype='f8')
    t={k:np.asarray(targets[k]) for k in TARGET_NAMES}
    if mean.ndim!=1 or std.shape!=mean.shape or any(a.shape!=mean.shape for a in t.values()):raise ValueError('Prediction/target axes differ')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise FloatingPointError('Invalid finite Gaussian field forecast')
    heard=t['heard'].astype(bool);censored=t['censored'].astype(bool);known=heard|censored
    if (heard&censored).any():raise ValueError('Heard/censored overlap')
    errors=mean[heard]-t['center_dbm'][heard]
    if not np.isfinite(errors).all():raise FloatingPointError('Nonfinite heard target')
    hi=(t['upper_dbm'][heard]-mean[heard])/std[heard];lo=(np.maximum(t['lower_dbm'][heard],t['detection_threshold_dbm'][heard])-mean[heard])/std[heard]
    if (hi<=lo).any():raise ValueError('Quantized report interval must be positive')
    upper=np.where(lo>0,log_ndtr(-lo),log_ndtr(hi));lower=np.where(lo>0,log_ndtr(-hi),log_ndtr(lo))
    logp=upper+np.log(-np.expm1(np.minimum(lower-upper,-np.finfo(float).eps)))
    censorp=log_ndtr((t['censor_upper_dbm'][censored]-mean[censored])/std[censored])
    nll=-np.concatenate((logp,censorp))
    probability=ndtr((mean[known]-t['detection_threshold_dbm'][known])/std[known])
    if not np.isfinite(nll).all() or not np.isfinite(probability).all():raise FloatingPointError('Invalid observation score')
    return {'rmse_db':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        'p95_abs_db':float(np.quantile(np.abs(errors),.95)) if len(errors) else None,
        'observation_nll':float(nll.mean()) if len(nll) else None,
        'detector_brier':float(np.mean((probability-heard[known])**2)) if known.any() else None,
        'heard_count':int(heard.sum()),'censored_count':int(censored.sum()),'missing_count':int((~known).sum()),
        'finite':True}


def _example_id(example):
    return example.layout_id,int(example.condition),example.case_id


def _arrays(example):return example.target_arrays()


def condition_metrics(predictions,examples):
    """Three RF states nest within one statistical geometry; both deltas mandatory."""
    if len(predictions)!=len(examples):raise ValueError('Prediction/case counts differ')
    groups={}
    for prediction,example in zip(predictions,examples):
        layout,condition,case=_example_id(example)
        if condition not in (0,1,2) or condition in groups.setdefault(layout,{}):raise ValueError('Repeated/unregistered RF condition')
        mean,std=prediction;mean=np.asarray(mean,dtype='f8');std=np.asarray(std,dtype='f8')
        target=_arrays(example);score=observation_metrics(mean,std,target)
        # Keys deliberately omit the case/condition identifier. Every physical query
        # must correspond across states even when its detector status changes.
        q=example.query_rows
        keys=list(zip(q.pose_index.tolist(),q.ap_index.tolist(),q.band_index.tolist(),q.direction.tolist(),q.device.tolist()))
        groups[layout][condition]=(mean,std,target,score,keys,case)
    results=[]
    for layout,cases in groups.items():
        if set(cases)!={0,1,2}:raise ValueError('All three registered RF conditions must be retained per geometry')
        anchor=cases[0];errors=[];pairs=[];usable=True
        for condition in (1,2):
            changed=cases[condition]
            if changed[4]!=anchor[4]:raise ValueError('Condition queries refer to different physical links')
            ah=np.asarray(anchor[2]['heard'],bool);bh=np.asarray(changed[2]['heard'],bool)
            ak=ah|np.asarray(anchor[2]['censored'],bool);bk=bh|np.asarray(changed[2]['censored'],bool)
            common=ah&bh;count=int(common.sum());usable &= count>0
            error=(changed[0]-anchor[0])-(changed[2]['center_dbm']-anchor[2]['center_dbm'])
            errors.extend(error[common].tolist())
            pairs.append({'pair':[0,condition],'both_heard_count':count,
                'anchor_heard_only_count':int((ah&~bh&bk).sum()),'changed_heard_only_count':int((bh&~ah&ak).sum()),
                'both_censored_count':int((ak&bk&~ah&~bh).sum()),'missing_pair_count':int((~(ak&bk)).sum())})
        means=np.concatenate([cases[k][0] for k in (0,1,2)]);spreads=np.concatenate([cases[k][1] for k in (0,1,2)])
        targets={name:np.concatenate([cases[k][2][name] for k in (0,1,2)]) for name in TARGET_NAMES}
        metrics=observation_metrics(means,spreads,targets)
        results.append({'layout_id':layout,'case_ids':[cases[k][5] for k in (0,1,2)],
            'delta_available':bool(usable),'delta_rmse_db':float(np.sqrt(np.mean(np.square(errors)))) if usable else None,
            'delta_unavailable_reason':None if usable else 'at_least_one_registered_pair_has_no_both_heard_query',
            'delta_both_heard_count':len(errors),'pair_detection':pairs,**metrics})
    return results


def validation_score(predictions,examples):
    values=[r['delta_rmse_db'] for r in condition_metrics(predictions,examples) if r['delta_available']]
    if not values:raise RuntimeError('Validation has no usable paired-condition geometry')
    return float(np.mean(values))


def paired_bootstrap(candidate,reference_values,*,samples=20000,seed=21749):
    candidate,reference_values=np.asarray(candidate,dtype='f8'),np.asarray(reference_values,dtype='f8')
    if candidate.ndim!=1 or candidate.shape!=reference_values.shape or len(candidate)<2:raise ValueError('Need paired whole-geometry scores')
    difference=candidate-reference_values
    if not np.isfinite(difference).all():raise ValueError('Nonfinite primary endpoint')
    rng=np.random.default_rng(seed);draw=rng.integers(len(difference),size=(samples,len(difference)))
    boot=difference[draw].mean(1);adjusted=np.quantile(boot,[.00625,.99375])
    return {'layouts':len(difference),'candidate_mean':float(candidate.mean()),'reference_mean':float(reference_values.mean()),
        'mean_paired_difference':float(difference.mean()),'adjusted_interval':adjusted.tolist(),'adjusted_confidence':.9875,
        'descriptive_95_interval':np.quantile(boot,[.025,.975]).tolist(),
        'better_adjusted':bool(adjusted[1]<0),'bootstrap_samples':samples,'bootstrap_seed':seed,
        'unit':'whole geometry, three RF conditions nested','scope':'conditional on three trained members; seed spread descriptive'}


def select_branch(fits):
    candidates={}
    for cell in ('joint','regression'):
        scores=[float(fits[(cell,s)]['validation_mean_layout_delta_rmse_db']) for s in SEEDS]
        if not np.isfinite(scores).all():raise ValueError('Nonfinite validation branch scores')
        candidates[cell]={'mean_delta_rmse_db':float(np.mean(scores)),'seed_delta_rmse_db':scores}
    selected=min(('joint','regression'),key=lambda name:candidates[name]['mean_delta_rmse_db'])
    return {'selected_cell':selected,'candidates':candidates,'selection_metric':'mean_member_sparse_VAL_mean_layout_condition_delta_rmse_db'}


def write_selection():
    fits={(cell,s):c.read_json(c.OUT/'fits'/f'{cell}-seed{s}.json') for cell in ('joint','regression') for s in SEEDS}
    value=select_branch(fits)|identity()
    for cell in ('joint','regression'):
        value['candidates'][cell]['fit_receipts_sha256']=[c.digest(c.OUT/'fits'/f'{cell}-seed{s}.json') for s in SEEDS]
    c.write_once(c.OUT/'selection.json',value);return value


def verified_selection():
    saved=c.read_json(c.OUT/'selection.json')
    fits={(cell,s):c.read_json(c.OUT/'fits'/f'{cell}-seed{s}.json') for cell in ('joint','regression') for s in SEEDS}
    expected=select_branch(fits)|identity()
    for cell in ('joint','regression'):
        expected['candidates'][cell]['fit_receipts_sha256']=[c.digest(c.OUT/'fits'/f'{cell}-seed{s}.json') for s in SEEDS]
    if saved!=expected:raise RuntimeError('Validation-only branch selection changed')
    return saved


def examples(stage,regime,*,stale=False,held_band=False):
    from . import data
    return data.examples(stage,regime=regime,stale=stale,held_band=held_band)


def _numpy(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    return np.asarray(value)


def _input_arrays(example):
    arrays=dict(example.model_inputs())
    for name in ('scheduled_mask_a','scheduled_mask_b'):
        if hasattr(example,name):arrays[name]=getattr(example,name)
    return arrays


def _cache_example(receipt,arrays):
    from types import SimpleNamespace
    m=receipt['metadata'];targets={name:arrays[name] for name in TARGET_NAMES}
    rows=SimpleNamespace(**{name:arrays[name] for name in ('pose_index','ap_index','band_index','direction','device')})
    return SimpleNamespace(layout_id=m['layout_id'],case_id=m['case_id'],condition=m['condition'],
        query_rows=rows,target_arrays=lambda:targets)


def _raw_support(records):
    """Actual target coverage even for a supplementary adapter that abstains."""
    by_condition={r[0]['metadata']['condition']:(r[0],r[1]) for r in records}
    if set(by_condition)!={0,1,2}:raise RuntimeError('Condition membership missing')
    anchor=by_condition[0][1];pairs=[]
    for k in (1,2):
        target=by_condition[k][1];ah=anchor['heard'];bh=target['heard'];ak=ah|anchor['censored'];bk=bh|target['censored']
        pairs.append({'pair':[0,k],'both_heard_count':int((ah&bh).sum()),
            'anchor_heard_only_count':int((ah&~bh&bk).sum()),'changed_heard_only_count':int((bh&~ah&ak).sum()),
            'both_censored_count':int((ak&bk&~ah&~bh).sum()),'missing_pair_count':int((~(ak&bk)).sum())})
    heard=sum(int(a['heard'].sum()) for _,a in by_condition.values())
    censored=sum(int(a['censored'].sum()) for _,a in by_condition.values())
    missing=sum(int((~(a['heard']|a['censored'])).sum()) for _,a in by_condition.values())
    return {'pair_detection':pairs,'heard_count':heard,'censored_count':censored,'missing_count':missing}


def _aggregate_cases(records):
    grouped={}
    for receipt,arrays in records:
        grouped.setdefault(receipt['metadata']['layout_id'],[]).append((receipt,arrays))
    result=[]
    for layout,rows in grouped.items():
        if all(receipt['metadata']['available'] for receipt,_ in rows):
            pred=[(a['mean'],a['std']) for _,a in rows];ex=[_cache_example(r,a) for r,a in rows]
            result.extend(condition_metrics(pred,ex))
        else:
            result.append({'layout_id':layout,'case_ids':[r['metadata']['case_id'] for r,_ in rows],
                'delta_available':False,'delta_rmse_db':None,'delta_unavailable_reason':'supplementary_frozen_adapter_abstained',
                'delta_both_heard_count':sum(x['both_heard_count'] for x in _raw_support(rows)['pair_detection']),
                'rmse_db':None,'p95_abs_db':None,'observation_nll':None,'detector_brier':None,
                'finite':True,'supplementary_abstained_cases':sum(not r['metadata']['available'] for r,_ in rows),**_raw_support(rows)})
    return result


def _member(stage,regime,cell,seed=None,*,stale=False,held_band=False,vals=None):
    from . import training,baselines,data
    vals=examples(stage,regime,stale=stale,held_band=held_band) if vals is None else vals
    model=None
    if cell in CELLS:model=training.load(cell,seed)
    name=cell if seed is None else f'{cell}-seed{seed}'
    if stale:name+='-stale'
    assay=regime+('-held_band' if held_band else '')
    records=[];references=[]
    for index,e in enumerate(vals):
        dependencies={'stage':reference(c.OUT/'rt_data'/f'{stage}.json'),
            'frontend':reference(c.OUT/'data/frontend-training.json'),
            'examples':reference(c.OUT/'data'/f"{stage}-{regime}-{'heldband' if held_band else 'main'}-{'stale' if stale else 'current'}.json")}
        if cell in CELLS:
            dependencies['fit']=reference(c.OUT/'fits'/f'{cell}-seed{seed}.json')
            dependencies['checkpoint']=reference(c.OUT/'checkpoints'/f'{cell}-seed{seed}.pt')
        elif cell in BASELINES:dependencies['baseline']=reference(c.OUT/'baselines/prior.json')
        else:
            frozen=baselines.frozen_references()
            # Keep the literal frozen source/weights proof in every adapter record.
            dependencies['frozen_references']=frozen
        eligible={'eligible':True,'reason':None} if cell!='frozen_old' else baselines.frozen_eligibility(e)
        meta={'identity':identity(),'stage':stage,'regime':regime,'assay':assay,'case_index':index,
            'layout_id':e.layout_id,'case_id':e.case_id,'condition':int(e.condition),'context_case_id':e.context_case_id,
            'cell':cell,'seed':seed,'member':name,'stale':stale,'source_view':'a',
            'held_band':held_band,'available':bool(eligible['eligible']),'eligibility':eligible,
            'dependencies':dependencies,'example_binding':data.example_binding(e),'predictions_ensembled':False,'packet_or_production_risk':False}
        path=c.OUT/'predictions'/stage/assay/name/f'{e.case_id}.npz';receipt_path=path.with_suffix('.json')
        if receipt_path.exists():
            saved=c.read_json(receipt_path)
            if saved['metadata']!=meta or c.digest(path)!=saved['npz_sha256']:raise RuntimeError('Prediction cache identity/bytes changed')
        else:
            if path.exists():raise RuntimeError('Unreceipted prediction cache; refuse a hidden rerun')
            posterior={}
            if cell in CELLS:
                mean,std=training.predict(model,e)
                code=training.encode(model,e)
                posterior={'beta_mean':_numpy(code['mean']).reshape(8),
                    'beta_covariance':_numpy(code['covariance']).reshape(8,8)}
            elif cell in BASELINES:mean,std=baselines.predict(cell,e)
            else:mean,std=baselines.predict_frozen(seed,e)
            if not meta['available'] and cell!='frozen_old':raise RuntimeError('Primary member cannot abstain')
            q=e.query_rows
            arrays={**_input_arrays(e),**e.target_arrays(),**posterior,'mean':np.asarray(mean),'std':np.asarray(std),
                **{name:getattr(q,name) for name in ('pose_index','ap_index','band_index','direction','device')}}
            if meta['available']:metrics=observation_metrics(arrays['mean'],arrays['std'],arrays)
            else:metrics=None
            stream=io.BytesIO();np.savez_compressed(stream,**arrays);c.write_bytes_once(path,stream.getvalue())
            saved={'schema':1,'metadata':meta,'npz_path':str(path.relative_to(c.ROOT)),
                'npz_sha256':c.digest(path),'metrics':metrics};c.write_once(receipt_path,saved)
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
        records.append((saved,arrays));references.append(reference(receipt_path))
    return {'member':name,'cell':cell,'seed':seed,'stale':stale,'cases':references,'layouts':_aggregate_cases(records)}


def evaluate_stage(stage):
    if stage not in ('development','test'):raise ValueError('Not an adaptive evaluation stage')
    c.authorize_stage(stage);validate_stage(stage);selection=verified_selection()
    path=c.OUT/'results'/f'{stage}.json'
    if path.exists():return verify_evaluation(stage)
    values={}
    with c.phase_guard('evaluate_'+stage):
        for regime in REGIMES:
            current=examples(stage,regime);stale_examples=examples(stage,regime,stale=True)
            members={}
            for cell in CELLS:
                for seed in SEEDS:
                    member=_member(stage,regime,cell,seed,vals=current);members[member['member']]=member
            for cell in BASELINES:
                member=_member(stage,regime,cell,vals=current);members[member['member']]=member
            for seed in SEEDS:
                member=_member(stage,regime,'frozen_old',seed,vals=current);members[member['member']]=member
                member=_member(stage,regime,selection['selected_cell'],seed,stale=True,vals=stale_examples);members[member['member']]=member
            values[regime]=members
        # SixGHz is a separate same-link diagnostic; it cannot influence the gate.
        held={};held_examples=examples(stage,'sparse',held_band=True)
        for cell in (selection['selected_cell'],*BASELINES):
            for seed in (SEEDS if cell in CELLS else (None,)):
                member=_member(stage,'sparse',cell,seed,held_band=True,vals=held_examples);held[member['member']]=member
        values['sparse-held_band']=held
        value={'schema':1,'stage':stage,'identity':identity(),'selection':reference(c.OUT/'selection.json'),
            'members':values,'seed_metrics_averaged_not_predictions':True,
            'unit':'wholegeometry,3nestedRFconditions','original_service_confirmation_permitted':False,'production_qualified':False}
        c.write_once(path,value)
    return verify_evaluation(stage)


def _all_references(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:verify_reference(value)
        for child in value.values():_all_references(child)
    elif isinstance(value,list):
        for child in value:_all_references(child)


def verify_evaluation(stage):
    """Passive metrics/membership replay. Full raw input replay is separate."""
    from . import baselines
    value=c.read_json(c.OUT/'results'/f'{stage}.json');ident=identity();selection=verified_selection()
    if value['identity']!=ident or value['stage']!=stage:raise RuntimeError('Evaluation source identity changed')
    verify_reference(value['selection']);expected_regimes={'full','sparse','sparse-held_band'}
    if set(value['members'])!=expected_regimes:raise RuntimeError('Evaluation regimes changed')
    expected_cases=[c.case_spec(stage,i,k) for i in range(c.COUNTS[stage]) for k in c.CONDITIONS]
    expected_files=set()
    for assay,members in value['members'].items():
        held=assay.endswith('-held_band');regime='sparse' if held else assay
        if held:
            expected_names={f"{selection['selected_cell']}-seed{s}" for s in SEEDS}|set(BASELINES)
        else:
            expected_names={f'{cell}-seed{s}' for cell in CELLS for s in SEEDS}|set(BASELINES)
            expected_names|={f'frozen_old-seed{s}' for s in SEEDS}|{f"{selection['selected_cell']}-seed{s}-stale" for s in SEEDS}
        if set(members)!=expected_names:raise RuntimeError('Evaluation member membership changed')
        for member_name,member in members.items():
            cell,seed,stale=member['cell'],member['seed'],member['stale']
            if member_name!=member['member'] or member_name!=(cell if seed is None else f'{cell}-seed{seed}')+('-stale' if stale else ''):
                raise RuntimeError('Member cell/seed/intervention identity changed')
            if stale and (cell!=selection['selected_cell'] or held):raise RuntimeError('Unregistered stale intervention')
            if seed is None and cell not in BASELINES or seed is not None and (seed not in SEEDS or cell not in (*CELLS,'frozen_old')):
                raise RuntimeError('Unregistered trained/deterministic member')
            if len(member['cases'])!=len(expected_cases):raise RuntimeError('Condition cases dropped')
            records=[]
            for index,(spec,ref) in enumerate(zip(expected_cases,member['cases'])):
                receipt_path=verify_reference(ref)
                expected_path=c.OUT/'predictions'/stage/assay/member_name/f'{spec.case_id}.json'
                if receipt_path!=expected_path:raise RuntimeError('Prediction receipt redirected')
                receipt=c.read_json(receipt_path);path=expected_path.with_suffix('.npz')
                expected_files|={path,receipt_path};meta=receipt['metadata']
                if receipt['npz_path']!=str(path.relative_to(c.ROOT)) or c.digest(path)!=receipt['npz_sha256']:raise RuntimeError('Prediction bytes/array path changed')
                context_case=c.case_spec(stage,index//3,0).case_id if stale else spec.case_id
                exact={'identity':ident,'stage':stage,'regime':regime,'assay':assay,'case_index':index,
                    'layout_id':spec.layout_id,'case_id':spec.case_id,'condition':index%3,'context_case_id':context_case,
                    'cell':cell,'seed':seed,'member':member_name,'stale':stale,'source_view':'a','held_band':held,
                    'predictions_ensembled':False,'packet_or_production_risk':False}
                if any(meta.get(k)!=v for k,v in exact.items()):raise RuntimeError('Prediction metadata changed')
                deps={'stage':reference(c.OUT/'rt_data'/f'{stage}.json'),'frontend':reference(c.OUT/'data/frontend-training.json'),
                    'examples':reference(c.OUT/'data'/f"{stage}-{regime}-{'heldband' if held else 'main'}-{'stale' if stale else 'current'}.json")}
                if cell in CELLS:
                    deps['fit']=reference(c.OUT/'fits'/f'{cell}-seed{seed}.json');deps['checkpoint']=reference(c.OUT/'checkpoints'/f'{cell}-seed{seed}.pt')
                elif cell in BASELINES:deps['baseline']=reference(c.OUT/'baselines/prior.json')
                else:deps['frozen_references']=baselines.frozen_references()
                if meta['dependencies']!=deps:raise RuntimeError('Prediction canonical dependency paths/bytes changed')
                _all_references(deps)
                if meta['available'] is not True and cell!='frozen_old':raise RuntimeError('Primary model cannot abstain')
                if meta['available']!=meta['eligibility']['eligible']:raise RuntimeError('Adapter availability metadata differs')
                with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
                expected_pose={0,2,4,6,8,10} if held else {1,3,5,7,9,11}
                if set(arrays['pose_index'].tolist())!=expected_pose or set(arrays['band_index'].tolist())!=({2} if held else {0,1}) or set(arrays['device'].tolist())!={0,1,2}:
                    raise RuntimeError('Primary/held endpoint axes changed')
                score=observation_metrics(arrays['mean'],arrays['std'],arrays) if meta['available'] else None
                if score!=receipt['metrics']:raise RuntimeError('Stored case scores do not recompute')
                if cell in CELLS:
                    if arrays['beta_mean'].shape!=(8,) or arrays['beta_covariance'].shape!=(8,8) or not np.isfinite(arrays['beta_mean']).all() or not np.isfinite(arrays['beta_covariance']).all():
                        raise RuntimeError('Invalid stored physics posterior')
                records.append((receipt,arrays))
            if member['layouts']!=_aggregate_cases(records):raise RuntimeError('Geometry-grouped scores do not recompute')
    actual_files={p for p in (c.OUT/'predictions'/stage).rglob('*') if p.is_file()}
    if expected_files!=actual_files:raise RuntimeError('Prediction byte membership changed')
    return value


def layout_values(evaluation,cell,metric,regime='sparse',*,stale=False):
    members=evaluation['members'][regime]
    if cell in BASELINES:return np.array([np.nan if r[metric] is None else r[metric] for r in members[cell]['layouts']],dtype='f8')
    names=[f'{cell}-seed{s}'+('-stale' if stale else '') for s in SEEDS]
    a=np.array([[np.nan if r[metric] is None else r[metric] for r in members[n]['layouts']] for n in names],dtype='f8')
    # Every trained member must support the same target cohort; no selective seed averaging.
    return np.where(np.isfinite(a).all(0),a.mean(0),np.nan)


def _paired_means(a,b):
    known=np.isfinite(a)&np.isfinite(b)
    return (float(a[known].mean()),float(b[known].mean()),int(known.sum())) if known.any() else (None,None,0)


def development_gate(evaluation,selected):
    selected_delta=layout_values(evaluation,selected,'delta_rmse_db')
    map_delta=layout_values(evaluation,'radiomap_operator','delta_rmse_db')
    erased_delta=layout_values(evaluation,'rf_erased','delta_rmse_db')
    stale_delta=layout_values(evaluation,selected,'delta_rmse_db',stale=True)
    cohort=np.isfinite(selected_delta)&np.isfinite(map_delta)&np.isfinite(erased_delta)&np.isfinite(stale_delta)
    n=int(cohort.sum())
    mean=lambda a:float(a[cohort].mean()) if n else None
    current,mapscore,erased,stale=map(mean,(selected_delta,map_delta,erased_delta,stale_delta))
    scalar,scalar_map,scalar_n=_paired_means(layout_values(evaluation,selected,'rmse_db'),layout_values(evaluation,'radiomap_operator','rmse_db'))
    tail,tail_map,tail_n=_paired_means(layout_values(evaluation,selected,'p95_abs_db'),layout_values(evaluation,'radiomap_operator','p95_abs_db'))
    seed_scores=[]
    for seed in SEEDS:
        rows=evaluation['members']['sparse'][f'{selected}-seed{seed}']['layouts']
        seed_scores.append(float(np.mean([rows[i]['delta_rmse_db'] for i in np.flatnonzero(cohort)])) if n else None)
    selected_rows=evaluation['members']['sparse'][f'{selected}-seed0']['layouts']
    censored=sum(r['censored_count'] for r in selected_rows)
    finite=all(r['finite'] is True for regime in REGIMES for name,m in evaluation['members'][regime].items() if m['cell']!='frozen_old' for r in m['layouts'])
    checks={'minimum_three_usable_layouts':n>=3,
        'five_percent_better_than_rf_erased':n>0 and current<=.95*erased and current<erased,
        'five_percent_better_than_radiomap':n>0 and current<=.95*mapscore and current<mapscore,
        'every_seed_delta_no_worse_than_radiomap':n>0 and all(x<=mapscore for x in seed_scores),
        'scalar_rmse_within_two_percent':scalar_n>=3 and scalar<=1.02*scalar_map,
        'p95_within_two_percent':tail_n>=3 and tail<=1.02*tail_map,
        'stale_context_hurts_at_least_five_percent':n>0 and stale>=1.05*current and stale>current,
        'actual_censored_events_present':censored>0,'all_primary_predictions_and_scores_finite':finite}
    return {'test_permitted':all(checks.values()),'checks':checks,'selected_cell':selected,'usable_layouts':n,
        'registered_layouts':len(selected_rows),'unavailable_layouts':[r for r in selected_rows if not r['delta_available']],
        'selected_delta_rmse_db':current,'radiomap_delta_rmse_db':mapscore,'rf_erased_delta_rmse_db':erased,
        'stale_delta_rmse_db':stale,'selected_seed_delta_rmse_db':seed_scores,
        'scalar_rmse_db':scalar,'radiomap_scalar_rmse_db':scalar_map,'scalar_available_layouts':scalar_n,
        'p95_abs_db':tail,'radiomap_p95_abs_db':tail_map,'p95_available_layouts':tail_n,
        'censored_query_events':censored,'original_service_confirmation_permitted':False,'production_qualified':False}


def write_development():
    from . import verification
    validate_stage('development');evaluation=verify_evaluation('development')
    with verification.passive_guard():verification.verify_public_arrays('development')
    value=development_gate(evaluation,verified_selection()['selected_cell'])|identity()|{
        'evaluation':reference(c.OUT/'results/development.json'),'selection':reference(c.OUT/'selection.json')}
    c.write_once(c.OUT/'development.json',value)
    qualification={k:value[k] for k in ('test_permitted','source_receipt_sha256','registry_sha256','original_service_confirmation_permitted','production_qualified')}
    qualification['development_sha256']=c.digest(c.OUT/'development.json')
    c.write_once(c.OUT/'qualification.json',qualification);return value


def _verified_development_uncached():
    from . import verification
    validate_stage('development');evaluation=verify_evaluation('development')
    with verification.passive_guard():verification.verify_public_arrays('development')
    expected=development_gate(evaluation,verified_selection()['selected_cell'])|identity()|{
        'evaluation':reference(c.OUT/'results/development.json'),'selection':reference(c.OUT/'selection.json')}
    value=c.read_json(c.OUT/'development.json')
    if value!=expected:raise RuntimeError('Development gate no longer recomputes from actual reported data')
    return value



def _is_registered_test_data(path,metadata):
    """Future TEST records cannot self-exempt an unrelated gate dependency."""
    relative=Path(path).relative_to(c.OUT/'data')
    if metadata.get('stage')!='test':return False
    configurations={'test-full-main-current.json':('full',False,False),
        'test-full-main-stale.json':('full',False,True),
        'test-sparse-main-current.json':('sparse',False,False),
        'test-sparse-main-stale.json':('sparse',False,True),
        'test-sparse-heldband-current.json':('sparse',True,False)}
    if str(relative) in configurations:
        regime,held,stale=configurations[str(relative)]
        return metadata.get('regime')==regime and metadata.get('held_band') is held and metadata.get('stale') is stale
    key=(c.case_spec,c.COUNTS['test'])
    if key not in _TEST_DATA_NAMES:
        names={}
        for index in range(c.COUNTS['test']):
            for condition in c.CONDITIONS:
                spec=c.case_spec('test',index,condition)
                names[str(Path('scenes')/(spec.case_id+'-field_test.json'))]=(spec.case_id,spec.layout_id,condition)
        _TEST_DATA_NAMES[key]=names
    expected=_TEST_DATA_NAMES[key].get(str(relative))
    if expected is not None:
        return (metadata.get('case_id'),metadata.get('layout_id'),metadata.get('condition'))==expected

    return False

def development_dependencies():
    """Everything that can affect the gate, excluding newly opened TEST outputs."""
    receipt=c.check_sources();paths=[c.safe_path(name) for name in receipt['source_sha256']]
    paths += [c.safe_path(name,c.OUT/'registration/source') for name in receipt['source_sha256']]
    paths += [c.safe_path(name) for name in receipt['preflight_sha256']]
    for name in ('registration/source.json','registration/git.json','selection.json','development.json','qualification.json','results/development.json'):
        paths.append(c.OUT/name)
    for name in ('fits','checkpoints','baselines','rt_data/training','rt_data/validation','rt_data/development','predictions/development'):
        folder=c.OUT/name
        if not folder.is_dir():raise RuntimeError('Missing gate dependency directory')
        paths += [p for p in folder.rglob('*') if p.is_file()]
    for stage in ('training','validation','development'):paths.append(c.OUT/'rt_data'/f'{stage}.json')
    for path in (c.OUT/'data').rglob('*'):
        if not path.is_file():continue
        if path.suffix=='.json' and _is_registered_test_data(path,c.read_json(path)):continue
        paths.append(path)
    from . import baselines
    frozen=baselines.frozen_references()
    for refs in frozen['members'].values():
        for ref in refs.values():paths.append(verify_reference(ref))
    result={}
    for path in sorted(set(paths)):
        name=str(path.relative_to(c.ROOT));path=c.safe_path(name)
        result[name]=c.digest(path)
    return result


def verified_development():
    global _CACHE
    # A veto never gains a cached permission. Its full replay remains authoritative.
    if c.read_json(c.OUT/'development.json').get('test_permitted') is not True:return _verified_development_uncached()
    if _CACHE is None:
        if not (c.OUT/'gates/development-cache.json').exists():
            for path in (c.OUT/'rt_data/test',c.OUT/'rt_data/test.json',c.OUT/'predictions/test',c.OUT/'results/test.json'):
                if path.exists():raise RuntimeError('Original full source-bound gate must precede any TEST evidence')
        _CACHE=c.EvidenceGateCache(_verified_development_uncached,development_dependencies)
    return _CACHE()


def test_results(evaluation,selected):
    primary_rows=evaluation['members']['sparse'][f'{selected}-seed0']['layouts']
    usable=sum(r['delta_available'] for r in primary_rows);cohort_sufficient=usable>=12
    contrasts=[]
    for spec in c.protocol()['primary_tests']:
        candidate=selected if spec['candidate']=='selected' else spec['candidate']
        a=layout_values(evaluation,candidate,spec['metric']);b=layout_values(evaluation,spec['reference'],spec['metric'])
        mask=np.isfinite(a)&np.isfinite(b)
        statistics=paired_bootstrap(a[mask],b[mask]) if mask.sum()>=2 else None
        contrasts.append(dict(spec)|{'candidate_cell':candidate,'usable_geometry_count':int(mask.sum()),
            'unavailable_layout_ids':[r['layout_id'] for i,r in enumerate(primary_rows) if not mask[i]],
            'statistics':statistics,'supported':bool(cohort_sufficient and statistics is not None and statistics['better_adjusted'])})
    return {'schema':1,'identity':identity(),'selected_cell':selected,'registered_test_geometries':24,
        'usable_delta_geometries':usable,'minimum_12_cohort_sufficient':cohort_sufficient,
        'status':'complete' if cohort_sufficient else 'inconclusive_insufficient_delta_cohort',
        'contrasts':contrasts,'RF_adaptation_supported':contrasts[0]['supported'] and contrasts[1]['supported'],
        'matched_alignment_benefit_supported':contrasts[2]['supported'],'detector_brier_benefit_supported':contrasts[3]['supported'],
        'evaluation':reference(c.OUT/'results/test.json'),'original_service_confirmation_permitted':False,'production_qualified':False,
        'packet_probability_or_universal_robot_environment_claim':False}


def write_test_results():
    c.authorize_stage('test');evaluation=verify_evaluation('test')
    value=test_results(evaluation,verified_selection()['selected_cell']);c.write_once(c.OUT/'report.json',value);return value


def run(workers=2):
    from . import data,training,baselines,verification
    with c.lock(c.WORK/'supervisor.lock',nonblocking=True):
        c.check_sources();c.verify_previous()
        if (c.OUT/'verification/manifest.json').exists():return verification.verify_closure()
        for stage in ('training','validation'):collect_stage(stage,workers)
        with c.phase_guard('prepare_training'):data.prepare_training()
        with c.phase_guard('fit12'):training.train_all(workers)
        write_selection()
        with c.phase_guard('classical_priors'):baselines.prepare()
        collect_stage('development',workers);evaluate_stage('development');gate=write_development()
        if gate['test_permitted']:
            verified_development()  # Full original gate and immutable bootstrap before first TEST trace.
            collect_stage('test',workers);evaluate_stage('test');write_test_results()
            if not _CACHE._logging_frozen:_CACHE.finalize()  # Full original gate again before audit/seal.
        else:
            if (c.OUT/'rt_data/test').exists():raise RuntimeError('Vetoed TEST cases opened')
            c.write_once(c.OUT/'report.json',{'schema':1,'identity':identity(),'status':'development_veto',
                'development':reference(c.OUT/'development.json'),'test_opened':False,
                'RF_adaptation_supported':False,'matched_alignment_benefit_supported':False,
                'detector_brier_benefit_supported':False,'original_service_confirmation_permitted':False,'production_qualified':False})
        verification.run_audit();verification.seal();return verification.verify_closure()
