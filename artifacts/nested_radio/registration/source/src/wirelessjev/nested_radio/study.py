"""Prospective zero-capable residual and training-budget experiment, RF only."""
from __future__ import annotations
import argparse
import math
import time
import numpy as np
from . import common as c, data
from ..hierarchical_radio import baselines as numerical, evaluation as ev

ELIGIBLE=('global8_reg','global8_joint','nested_reg','nested_joint')
NUMERICAL=('radiomap-5m','radiomap-10m','radiomap-20m','hierarchy-fixed')
_MODELS={}
_PRIORS={}


def choose_cell(scores):
    from .model import CELLS
    if (set(scores)!=set(CELLS) or any(np.asarray(v).shape!=(3,) or np.asarray(v).dtype.kind not in 'fiu'
            or any(isinstance(x,(bool,np.bool_)) for x in v)
            or not np.isfinite(v).all() or np.any(np.asarray(v)<0) for v in scores.values())):
        raise ValueError('All five fixed cells and three finite member scores required')
    return min(ELIGIBLE,key=lambda cell:float(np.mean(scores[cell])))


def choose_numeric(scores):
    if set(scores)!=set(NUMERICAL) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in scores.values()):
        raise ValueError('All four predeclared numerical candidates required')
    return min(NUMERICAL,key=lambda name:scores[name])


def _parent_ref(name):
    path=c.PARENT/name;ref=c.reference(path)
    parent=c.read(c.PARENT/'verification/manifest.json')
    if parent['evidence_sha256'].get(ref['path'])!=ref['sha256']:
        raise RuntimeError('Reused numerical dependency differs from parent seal')
    return ref


def numerical_candidates():
    from .training import reduce_validation_predictions
    c.check_sources();data.verify_prepared()
    path=c.OUT/'numerical/selection.json'
    if path.exists():return verify_numerical()
    values=data.training_examples('validation','sparse');rows=[]
    for name in NUMERICAL:
        receipt_path=c.OUT/'numerical'/f'{name}.json'
        if not receipt_path.exists():
            attempt=c.OUT/'numerical/attempts'/f'{name}.json'
            if attempt.exists():raise RuntimeError('Incomplete numerical prediction attempt; no replacement forwards')
            prior_ref=_parent_ref('numerical/'+name+'.npz')
            identity={'name':name,'prior':prior_ref,'source_receipt_sha256':c.source_receipt_sha256(),
                'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
            c.attempt(attempt,identity)
            if name!='hierarchy-fixed':
                original=_parent_ref('numerical/'+name+'-validation.npz')
                raw=c.arrays(c.safe_path(original['path']));mean,std=raw['mean'],raw['std']
            else:
                original=None;prior=numerical.load_prior(c.arrays(c.safe_path(prior_ref['path'])))
                predicted=[numerical.predict(prior,e) for e in values]
                mean,std=np.stack([p[0] for p in predicted]),np.stack([p[1] for p in predicted])
            score,nll,per_layout=reduce_validation_predictions(mean,std,values)
            prediction=c.npz(c.OUT/'numerical'/f'{name}-validation.npz',{'mean':mean,'std':std})
            c.write(receipt_path,identity|{'validation_predictions':prediction,'parent_predictions':original,
                'validation_rmse':score,'validation_nll':nll,'validation_per_layout':per_layout,
                'attempt':c.reference(attempt),'new_prior_fit':False})
        rows.append(c.reference(receipt_path))
    scores={name:c.read(c.OUT/'numerical'/f'{name}.json')['validation_rmse'] for name in NUMERICAL}
    chosen=choose_numeric(scores)
    c.write(path,{'schema':1,'selected_name':chosen,'candidates':rows,'scores':scores,
        'source_receipt_sha256':c.source_receipt_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json'),
        'rule':'first minimum equal-layout sparse VAL heard RMSE; all four TRAIN-fit priors retained'})
    return verify_numerical()


def verify_numerical(validation=None):
    from .training import reduce_validation_predictions
    receipt=c.read(c.OUT/'numerical/selection.json')
    if (receipt['source_receipt_sha256']!=c.source_receipt_sha256()
            or receipt['prepared_sha256']!=c.digest(c.OUT/'data/prepared.json') or len(receipt['candidates'])!=4):
        raise RuntimeError('Numerical selection source or membership changed')
    validation=data.training_examples('validation','sparse') if validation is None else validation
    scores={}
    for name,item in zip(NUMERICAL,receipt['candidates']):
        path=c.OUT/'numerical'/f'{name}.json'
        if c.reference(path)!=item:raise RuntimeError('Canonical numerical candidate changed')
        row=c.read(path)
        if (row['name']!=name or row['prior']!=_parent_ref('numerical/'+name+'.npz')
                or row['source_receipt_sha256']!=receipt['source_receipt_sha256']
                or row['prepared_sha256']!=receipt['prepared_sha256'] or row['new_prior_fit'] is not False
                or row['validation_predictions']!=c.reference(c.OUT/'numerical'/f'{name}-validation.npz')
                or row['attempt']!=c.reference(c.OUT/'numerical/attempts'/f'{name}.json')):
            raise RuntimeError('Numerical candidate canonical identity changed')
        attempt=c.read(c.safe_path(row['attempt']['path']));attempt.pop('created_unix_ns')
        if attempt!={k:row[k] for k in ('name','prior','source_receipt_sha256','prepared_sha256')}:
            raise RuntimeError('Numerical attempt identity changed')
        output=c.arrays(c.safe_path(row['validation_predictions']['path']))
        if set(output)!={'mean','std'}:raise RuntimeError('Numerical prediction arrays changed')
        if name!='hierarchy-fixed':
            expected=_parent_ref('numerical/'+name+'-validation.npz')
            if row['parent_predictions']!=expected:raise RuntimeError('Reused validation prediction identity changed')
            parent=c.arrays(c.safe_path(expected['path']))
            if any(not np.array_equal(output[k],parent[k]) for k in output):raise RuntimeError('Reused prediction bytes differ')
        elif row['parent_predictions'] is not None:raise RuntimeError('Fixed hierarchy VAL predictions must be this source-bound forward')
        rmse,nll,per_layout=reduce_validation_predictions(output['mean'],output['std'],validation)
        if (rmse!=row['validation_rmse'] or nll!=row['validation_nll'] or per_layout!=row['validation_per_layout']):
            raise RuntimeError('Numerical validation score is not its raw prediction reduction')
        scores[name]=rmse
    if receipt['scores']!=scores or receipt['selected_name']!=choose_numeric(scores):
        raise RuntimeError('Numerical selection did not use all four predeclared candidates')
    return receipt


def _selection_value():
    from . import training as t
    from .model import CELLS
    c.check_sources();ident=t.identity();datasets=t.datasets();validation=datasets[2]
    expected_members={f'{cell}-seed{s}' for cell in CELLS for s in c.SEEDS}
    expected_epochs={f'{name}-epoch{epoch}' for name in expected_members for epoch in (40,80,120,160)}
    for folder,suffix,expected in (('fits','.json',expected_members),('checkpoints','.pt',expected_members),
            ('validation_predictions','.npz',expected_epochs),('epoch_checkpoints','.pt',expected_epochs)):
        if {p.stem for p in (c.OUT/folder).glob('*'+suffix)}!=expected:
            raise RuntimeError('Exact registered fit/checkpoint/VAL archive membership required: '+folder)
    fits={};weights={};scores={}
    for cell in CELLS:
        scores[cell]=[]
        for seed in c.SEEDS:
            name=f'{cell}-seed{seed}';path=c.OUT/'fits'/f'{name}.json';row=c.read(path)
            t.verify_completed(row,ident,cell,seed,validation_examples=validation,fitting_examples=datasets)
            fits[name]=c.reference(path);weights[name]=c.reference(c.OUT/'checkpoints'/f'{name}.pt')
            scores[cell].append(row['validation_rmse'])
    numeric=verify_numerical(validation)
    return {'schema':1,'selected_cell':choose_cell(scores),'member_VAL_RMSE':scores,
        'mean_VAL_RMSE':{k:float(np.mean(v)) for k,v in scores.items()},'fits':fits,'checkpoints':weights,
        'numerical_selection':c.reference(c.OUT/'numerical/selection.json'),'selected_numerical':numeric['selected_name'],
        'source_receipt_sha256':c.source_receipt_sha256(),'protocol_sha256':c.protocol_sha256(),
        'registry_sha256':c.registry_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json'),
        'technical_gate':True,'efficacy_screening':False,'TEST_or_parent_TEST_used_for_selection':False,
        'all15fits_complete':True,'committed_updates_each':1920,'native_simulations':0}


def select_models():
    value=_selection_value();path=c.OUT/'selection.json'
    if path.exists():
        if c.read(path)!=value:raise RuntimeError('Locked model selection changed')
    else:c.write(path,value)
    return value


def verify_selection():
    path=c.OUT/'selection.json'
    if not path.exists():raise RuntimeError('Lock TRAIN/VAL selection before opening fresh cases')
    value=c.read(path)
    if value!=_selection_value():raise RuntimeError('Selection does not reproduce all fixed raw VAL scores/checkpoints')
    return value


def methods():
    from .model import CELLS
    return [f'{cell}-seed{s}' for cell in CELLS for s in c.SEEDS]+list(NUMERICAL)


def predictor(name):
    from . import training as t
    if name not in methods():raise ValueError('Unknown registered predictor')
    if name in NUMERICAL:
        if name not in _PRIORS:
            ref=_parent_ref('numerical/'+name+'.npz')
            _PRIORS[name]=numerical.load_prior(c.arrays(c.safe_path(ref['path'])))
        return lambda e:numerical.predict(_PRIORS[name],e)
    if name not in _MODELS:
        cell,seed=name.rsplit('-seed',1);_MODELS[name]=t.load(cell,int(seed))
    return lambda e:t.predict(_MODELS[name],e)


def predict_group(stage,regime='sparse',*,held_band=False,stale=False):
    name=data.group_name(stage,regime,held_band,stale);c.check_sources();verify_selection()
    values=data.examples(stage,regime,held_band,stale);names=methods();refs={}
    group_ref=data.group_reference(stage,regime,held_band,stale)
    for index,e in enumerate(values):
        case=e['binding']['case_id'];folder=c.OUT/'predictions'/name;path=folder/f'{case}.json'
        identity={'group':name,'stage':stage,'case_id':case,'layout_id':e['binding']['layout_id'],'state':e['binding']['state'],
            'example_index':index,'example_group':group_ref,'methods':names,
            'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json')}
        if not path.exists():
            attempt=folder/'attempts'/f'{case}.json'
            if attempt.exists():raise RuntimeError('Incomplete prediction attempt; replacement inference forbidden')
            c.attempt(attempt,identity);means=[];stds=[];elapsed=[]
            for method in names:
                began=time.perf_counter();mean,std=predictor(method)(e)
                means.append(mean);stds.append(std);elapsed.append(time.perf_counter()-began)
            ref=c.npz(folder/f'{case}.npz',{'mean':np.stack(means),'std':np.stack(stds)})
            metrics=[ev.metrics(mean,std,e['target']) for mean,std in zip(means,stds)]
            c.write(path,identity|{'predictions':ref,'metrics':metrics,'elapsed_seconds':elapsed,'attempt':c.reference(attempt)})
        value=c.read(path)
        if any(value.get(k)!=v for k,v in identity.items()):raise RuntimeError('Prediction identity changed')
        verify_prediction(value,e);refs[case]=c.reference(path)
    receipt={'schema':1,'group':name,'example_group':group_ref,'case_receipts':refs,'methods':names,
        'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json')}
    path=c.OUT/'predictions'/f'{name}.json'
    if path.exists():
        if c.read(path)!=receipt:raise RuntimeError('Completed prediction group changed')
    else:c.write(path,receipt)
    print('Prediction group',name,len(refs),'complete',flush=True)
    return receipt


def verify_prediction(receipt,e=None):
    group_path=c.safe_path(receipt['example_group']['path'])
    if c.reference(group_path)!=receipt['example_group']:raise RuntimeError('Prediction group reference changed')
    group=data.verify_group(c.read(group_path));name=data.group_name(group['stage'],group['regime'],group['held_band'],group['stale'])
    if group_path!=c.OUT/'data/groups'/f'{name}.json':raise RuntimeError('Canonical example group receipt path required')
    index=receipt['example_index']
    if (type(index) is not int or not 0<=index<len(group['bindings']) or receipt['methods']!=methods()
            or receipt['group']!=name or receipt['stage']!=group['stage']
            or receipt['source_receipt_sha256']!=c.source_receipt_sha256()
            or receipt['selection_sha256']!=c.digest(c.OUT/'selection.json')):
        raise RuntimeError('Prediction canonical membership changed')
    if e is None:e=data.unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])[index]
    spec=data.case_spec(group['stage'],index//3,index%3);folder=c.OUT/'predictions'/name
    if (receipt['case_id']!=spec.case_id or receipt['layout_id']!=spec.layout_id or receipt['state']!=spec.state
            or e['binding']!=group['bindings'][index]
            or receipt['predictions']!=c.reference(folder/f'{spec.case_id}.npz')
            or receipt['attempt']!=c.reference(folder/'attempts'/f'{spec.case_id}.json')):
        raise RuntimeError('Prediction canonical case/arrays changed')
    attempt=c.read(c.safe_path(receipt['attempt']['path']));attempt.pop('created_unix_ns')
    if attempt!={k:receipt[k] for k in ('group','stage','case_id','layout_id','state','example_index','example_group','methods','source_receipt_sha256','selection_sha256')}:
        raise RuntimeError('Prediction attempt identity changed')
    output=c.arrays(c.safe_path(receipt['predictions']['path']))
    if (set(output)!={'mean','std'} or output['mean'].shape!=(len(methods()),len(e['target']['heard']))
            or output['std'].shape!=output['mean'].shape or len(receipt['metrics'])!=len(methods())
            or len(receipt['elapsed_seconds'])!=len(methods()) or not np.isfinite(receipt['elapsed_seconds']).all()
            or min(receipt['elapsed_seconds'])<0):raise RuntimeError('Prediction complete finite axes/timing required')
    actual=[ev.metrics(m,s,e['target']) for m,s in zip(output['mean'],output['std'])]
    if actual!=receipt['metrics']:raise RuntimeError('Metrics differ from raw means/deviations and observed targets')
    return receipt,output,e


def records(name):
    path=c.OUT/'predictions'/f'{name}.json';manifest=c.read(path)
    if (manifest['group']!=name or manifest['methods']!=methods()
            or manifest['source_receipt_sha256']!=c.source_receipt_sha256()
            or manifest['selection_sha256']!=c.digest(c.OUT/'selection.json')):
        raise RuntimeError('Evaluation manifest identity changed')
    group_path=c.safe_path(manifest['example_group']['path'])
    if group_path!=c.OUT/'data/groups'/f'{name}.json':raise RuntimeError('Canonical evaluation example group required')
    if c.reference(group_path)!=manifest['example_group']:raise RuntimeError('Evaluation example group changed')
    group=data.verify_group(c.read(group_path),name)
    values=data.unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])
    if set(manifest['case_receipts'])!={e['binding']['case_id'] for e in values}:raise RuntimeError('Evaluation complete cohort required')
    result=[]
    for e in values:
        item=manifest['case_receipts'][e['binding']['case_id']];p=c.safe_path(item['path'])
        if p!=c.OUT/'predictions'/name/f'{e["binding"]["case_id"]}.json':raise RuntimeError('Canonical case receipt path required')
        if c.reference(p)!=item:raise RuntimeError('Case prediction changed')
        row,output,_=verify_prediction(c.read(p),e)
        for j,method in enumerate(row['methods']):
            result.append({'method':method,'layout_id':row['layout_id'],'state':row['state'],'metrics':row['metrics'][j],
                'mean':output['mean'][j],'std':output['std'][j],'target':e['target']})
    return result


def _ranks(features):
    a=np.asarray(features,dtype='f8');centered=a-a.mean(0,keepdims=True)
    singular=np.linalg.svd(centered,compute_uv=False);power=singular**2
    participation=float(power.sum()**2/(power@power)) if power.sum()>0 else 0.
    return {'rows':len(a),'columns':a.shape[1],'mean_feature_variance':float(np.mean(np.var(a,axis=0))),
        'participation_rank_centered':participation,'singular_values_centered':singular.tolist()}


def representation_diagnostics():
    import torch
    from . import training as t
    from .model import CELLS
    from ..hierarchical_radio.model import HierarchicalField
    c.check_sources();verify_selection();path=c.OUT/'validation_representation.json'
    if path.exists():
        value=c.read(path)
        if value['source_receipt_sha256']!=c.source_receipt_sha256() or value['selection_sha256']!=c.digest(c.OUT/'selection.json'):
            raise RuntimeError('Diagnostic source changed')
        return value
    values=data.training_examples('validation','sparse')[::3]
    rows=[]
    for cell in CELLS:
        for seed in c.SEEDS:
            name=f'{cell}-seed{seed}';model=t.load(cell,seed);parts=[];active=[]
            attempt=c.OUT/'diagnostic_attempts'/f'{name}.json'
            if attempt.exists():raise RuntimeError('Partial representation diagnostic; no silent extra forwards')
            c.attempt(attempt,{'member':name,'source_receipt_sha256':c.source_receipt_sha256()})
            for e in values:
                v=t.batch([e])
                with torch.no_grad():
                    raw=HierarchicalField.local_features(model,v['query']['q'],v['query']['rx_xyz'],v['ap_xyz'],v['query']['ap'])
                    parts.append(raw[0].numpy().copy())
                    gate=model.gates().detach().cpu().numpy()
                    active.append(raw[0].numpy()*gate[None])
            raw=np.concatenate(parts);gated=np.concatenate(active)
            archive=c.npz(c.OUT/'representation_predictions'/f'{name}.npz',
                {'ungated_local_features':raw,'effective_local_features':gated})
            is_active=cell not in ('global8_reg','global8_joint')
            rows.append({'member':name,'fixed_Fourier8':_ranks(raw[:,:8]),
                'learned_geometry8':_ranks(raw[:,8:]),'effective_local16':_ranks(gated),
                'learned_local_features_used_in_this_cell':is_active,
                'inactive_global_features_untrained_not_a_representation':not is_active,
                'gates_and_effective_priors':model.diagnostics(),
                'feature_arrays':archive,'attempt':c.reference(attempt),
                'checkpoint':c.reference(c.OUT/'checkpoints'/f'{name}.pt'),
                'scope':'VAL public query geometry only; not an RF-encoder rank, collapse or physical identifiability certificate'})
    value={'schema':1,'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json'),
        'rows':rows,'used_for_selection':False,'scientific_TEST_inputs_used':False}
    c.write(path,value);return value


def verify_representation():
    import torch
    from .model import CELLS
    value=c.read(c.OUT/'validation_representation.json')
    expected=[f'{cell}-seed{s}' for cell in CELLS for s in c.SEEDS]
    if (value['source_receipt_sha256']!=c.source_receipt_sha256()
            or value['selection_sha256']!=c.digest(c.OUT/'selection.json')
            or [row['member'] for row in value['rows']]!=expected
            or value['used_for_selection'] is not False or value['scientific_TEST_inputs_used'] is not False):
        raise RuntimeError('VAL representation diagnostics identity changed')
    for row in value['rows']:
        name=row['member'];cell=name.rsplit('-seed',1)[0]
        if (row['feature_arrays']!=c.reference(c.OUT/'representation_predictions'/f'{name}.npz')
                or row['attempt']!=c.reference(c.OUT/'diagnostic_attempts'/f'{name}.json')
                or row['checkpoint']!=c.reference(c.OUT/'checkpoints'/f'{name}.pt')):
            raise RuntimeError('Representation archive canonical bytes changed')
        a=c.arrays(c.safe_path(row['feature_arrays']['path']))
        if (set(a)!={'ungated_local_features','effective_local_features'}
                or any(v.shape!=(1152,16) or not np.isfinite(v).all() for v in a.values())):
            raise RuntimeError('Complete finite VAL feature arrays required')
        raw=a['ungated_local_features'];effective=a['effective_local_features']
        gate=np.asarray(row['gates_and_effective_priors']['gates'],dtype=raw.dtype)
        checkpoint=torch.load(c.safe_path(row['checkpoint']['path']),map_location='cpu',weights_only=True)
        stored=checkpoint['state']['local_gates'].detach().cpu().numpy()
        if not np.array_equal(gate,stored):raise RuntimeError('Diagnostic gates differ from the selected checkpoint')
        if gate.shape!=(16,) or not np.array_equal(effective,raw*gate[None]):
            raise RuntimeError('Effective feature design differs from its explicit gates')
        if (row['fixed_Fourier8']!=_ranks(raw[:,:8]) or row['learned_geometry8']!=_ranks(raw[:,8:])
                or row['effective_local16']!=_ranks(effective)
                or row['learned_local_features_used_in_this_cell']!=(cell not in ('global8_reg','global8_joint'))
                or row['inactive_global_features_untrained_not_a_representation']!=(cell in ('global8_reg','global8_joint'))):
            raise RuntimeError('Feature rank/variation statistics differ from raw arrays')
    return value


def report():
    from .model import CELLS
    selection=verify_selection();names={cell:[f'{cell}-seed{s}' for s in c.SEEDS] for cell in CELLS}
    names.update({name:[name] for name in NUMERICAL})
    names.update(selected=names[selection['selected_cell']],best_numerical=names[selection['selected_numerical']])
    ids=[data.geometry_spec('test',i).scene_id for i in range(24)]
    raw_sparse=records('test-sparse');sparse=ev.reduce_layouts(raw_sparse,methods=names,layout_ids=ids)
    full=ev.reduce_layouts(records('test-full'),methods=names,layout_ids=ids)
    contrasts=[]
    for a,b in c.protocol_spec()['evaluation']['contrasts']:
        x=[r['metrics']['rmse_db'] for r in sparse[a]];y=[r['metrics']['rmse_db'] for r in sparse[b]]
        contrasts.append({'candidate':a,'reference':b,**ev.paired_interval(x,y,seed=21771)})
    member_contrasts=[]
    for seed in c.SEEDS:
        mapping={cell:[f'{cell}-seed{seed}'] for cell in CELLS}
        mapping.update({n:[n] for n in NUMERICAL})
        mapping.update(selected=mapping[selection['selected_cell']],best_numerical=mapping[selection['selected_numerical']])
        members=set().union(*map(set,mapping.values()))
        reduced=ev.reduce_layouts([r for r in raw_sparse if r['method'] in members],methods=mapping,layout_ids=ids)
        for a,b in c.protocol_spec()['evaluation']['contrasts']:
            delta=np.array([r['metrics']['rmse_db'] for r in reduced[a]])-np.array([r['metrics']['rmse_db'] for r in reduced[b]])
            member_contrasts.append({'seed':seed,'candidate':a,'reference':b,'mean_delta_db':float(delta.mean()),
                'layouts_lower':int((delta<0).sum()),'layouts_equal':int((delta==0).sum()),'layouts_higher':int((delta>0).sum()),
                'descriptive_no_extra_hypothesis_test':True})
    summary={regime:{name:{key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']}
        for name,rows in values.items()} for regime,values in (('sparse',sparse),('full',full))}
    secondary={}
    for group_name in ('test-sparse-held6GHz','test-sparse-stale','ap_transfer-sparse','ap_transfer-full'):
        stage='ap_transfer' if group_name.startswith('ap_transfer') else 'test'
        values=ev.reduce_layouts(records(group_name),methods=names,
            layout_ids=[data.geometry_spec(stage,i).scene_id for i in range(data.COUNTS[stage])])
        secondary[group_name]={name:{key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']} for name,rows in values.items()}
    delta={}
    for name,members in names.items():
        layouts=[]
        for layout in ids:
            errors=[]
            for member in members:
                state={r['state']:r for r in raw_sparse if r['method']==member and r['layout_id']==layout}
                residual=[]
                for changed in (1,2):
                    a,b=state[0],state[changed];known=a['target']['heard']&b['target']['heard']
                    residual.extend(((b['mean'][known]-a['mean'][known])-(b['target']['center_dbm'][known]-a['target']['center_dbm'][known])).tolist())
                if not residual:raise RuntimeError('No paired heard targets for registered delta endpoint')
                errors.append(float(np.sqrt(np.mean(np.square(residual)))))
            layouts.append({'layout_id':layout,'delta_RMSE_db':float(np.mean(errors))})
        delta[name]={'mean_delta_RMSE_db':float(np.mean([r['delta_RMSE_db'] for r in layouts])),'layouts':layouts}
    result={'schema':1,'complete':True,'source_receipt_sha256':c.source_receipt_sha256(),
        'selection_sha256':c.digest(c.OUT/'selection.json'),'selected_cell':selection['selected_cell'],
        'best_numerical':selection['selected_numerical'],'fits':15,'reused_TRAIN_VAL_RT_cases':90,'fresh_RT_cases':84,
        'primary_layouts':24,'members_each':3,'primary_contrasts':contrasts,'summary':summary,
        'per_member_contrasts_descriptive':member_contrasts,
        'layout_metrics':{'sparse':sparse,'full':full},'secondary_descriptive':secondary,
        'localized_delta_descriptive':delta,'validation_representation':c.reference(c.OUT/'validation_representation.json'),
        'selected_can_equal_global8_then_null_not_residual_gain':selection['selected_cell']=='global8_reg',
        'native_simulations':0,'production_qualified':False,'original_SERVICE_opened':False,'default_changed':False}
    path=c.OUT/'report.json'
    if path.exists():
        if c.read(path)!=result:raise RuntimeError('Final report differs from archived arrays/reductions')
    else:c.write(path,result)
    return result


def seal():
    c.check_sources();verify_selection();verify_representation()
    for stage in data.COUNTS:data.verify_stage(stage,reconstruct_reports=True)
    expected={c.case_path(data.case_spec(stage,i,s)) for stage in data.COUNTS for i in range(data.COUNTS[stage]) for s in data.STATES}
    if (set((c.OUT/'rt_data/cases').iterdir())!=expected
            or set((c.OUT/'rt_data/cases').rglob('attempt.json'))!={p/'attempt.json' for p in expected}):
        raise RuntimeError('Exactly 84 physical case directories and attempts required')
    report();prior=c.prior_identities(verify_bytes=True)
    evidence={str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(c.OUT.rglob('*')) if p.is_file()
        and not str(p.relative_to(c.OUT)).startswith('registration/source/') and p!=c.OUT/'verification/manifest.json'}
    manifest={'schema':1,'source_sha256':c.check_sources()['source_sha256'],'evidence_sha256':evidence,
        'prior':prior,'fresh_RT_cases':84,'fits':15,'native_simulations':0,'production_qualified':False,'original_SERVICE_opened':False}
    path=c.OUT/'verification/manifest.json'
    if path.exists():
        if c.read(path)!=manifest:raise RuntimeError('Sealed evidence changed')
    else:c.write(path,manifest)
    return manifest


def run(workers=3):
    from .training import train_all
    c.check_sources();data.prepare_training();numerical_candidates();train_all(workers);select_models()
    representation_diagnostics()
    for stage in data.COUNTS:data.collect(stage)
    for stage in data.COUNTS:
        for regime in ('full','sparse'):predict_group(stage,regime)
    predict_group('test','sparse',held_band=True);predict_group('test','sparse',stale=True)
    seal();print('Nested radio study complete; all outcomes retained; no packet/default promotion.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('register','publish','run','report'))
    parser.add_argument('--commit');parser.add_argument('--branch');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.command=='register':print('registered',len(c.register()['source_sha256']))
    elif args.command=='publish':print(c.record_publication(args.commit,args.branch))
    elif args.command=='run':run(args.workers)
    else:print(c.encoded(report()).decode())


if __name__=='__main__':main()
