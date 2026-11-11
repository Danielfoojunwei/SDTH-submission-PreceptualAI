"""One published finite shared/AP-local RF study, including honest transfer.

Every raw case and model member is fixed before collection. Selection reads
TRAIN/VAL only. Technical DEV checks do not screen for efficacy. No original
SERVICE case, application default or physical actuator is touched.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import time

import numpy as np

from . import common as c, data, simulation as sim, evaluation as ev

_GROUPS = {}
_MODELS = {}
_NUMERICAL = {}


def stage_path(stage):return c.OUT/'rt_data'/f'{stage}.json'


def collect(stage):
    if stage not in sim.COUNTS:raise ValueError('Unregistered physical stage')
    c.check_sources();refs={}
    for index in range(sim.COUNTS[stage]):
        for state in sim.STATES:
            spec=sim.case_spec(stage,index,state);folder=c.case_path(spec)
            sim.trace_case(spec,folder,authorize=c.authorize)
            refs[spec.case_id]=c.reference(folder/'receipt.json')
            print('RT',spec.case_id,'complete',flush=True)
    receipt={'schema':1,'stage':stage,'complete':True,'case_receipts':refs,
        'source_receipt_sha256':c.source_receipt_sha256(),'registry_sha256':c.registry_sha256(),
        'layouts':sim.COUNTS[stage],'cases':3*sim.COUNTS[stage],
        'static_counterfactuals_not_temporal_events':True}
    if stage_path(stage).exists():
        if c.read(stage_path(stage))!=receipt:raise RuntimeError('Completed RT stage identity changed')
    else:c.write(stage_path(stage),receipt)
    return receipt


def verify_stage(stage):
    value=c.read(stage_path(stage))
    if (value.get('stage')!=stage or value.get('complete') is not True
            or value.get('source_receipt_sha256')!=c.source_receipt_sha256()
            or value.get('registry_sha256')!=c.registry_sha256()):raise RuntimeError('RT stage binding changed')
    expected={sim.case_spec(stage,i,s).case_id for i in range(sim.COUNTS[stage]) for s in sim.STATES}
    if set(value['case_receipts'])!=expected:raise RuntimeError('RT stage membership changed')
    for index in range(sim.COUNTS[stage]):
        for state in sim.STATES:
            spec=sim.case_spec(stage,index,state);folder=c.case_path(spec)
            if c.reference(folder/'receipt.json')!=value['case_receipts'][spec.case_id]:raise RuntimeError('RT case receipt changed')
    return value


def scenes(stage):
    verify_stage(stage)
    return [data.load_scene(c.case_path(sim.case_spec(stage,i,state)),
        expected_spec=sim.case_spec(stage,i,state),source_sha256=c.source_receipt_sha256(),
        registry_sha256=c.registry_sha256()) for i in range(sim.COUNTS[stage]) for state in sim.STATES]


def calibration():
    path=c.OUT/'data/calibration.json'
    value=c.read(path)
    if value.get('source_receipt_sha256')!=c.source_receipt_sha256() or value.get('registry_sha256')!=c.registry_sha256():
        raise RuntimeError('Calibration source changed')
    for name,sha in value['calibration']['source_files_sha256'].items():
        if c.digest(c.safe_path(name))!=sha:raise RuntimeError('Calibration reported sources changed')
    return value['calibration']


def _group_name(stage,regime,held_band,stale):
    if (stage not in sim.COUNTS or regime not in ('full','sparse')
            or type(held_band) is not bool or type(stale) is not bool
            or (held_band or stale) and stage in ('training','validation')):
        raise ValueError('Unknown or forbidden physical example group')
    return stage+'-'+regime+('-held6GHz' if held_band else '')+('-stale' if stale else '')


def _pack_examples(values):
    result={'ap_xyz':np.stack([e['ap_xyz'] for e in values])}
    for name in ('a','b','query','target'):
        for key in values[0][name]:result[name+'__'+key]=np.stack([e[name][key] for e in values])
    return result


def _unpack_examples(arrays,bindings):
    if any(value.shape[0]!=len(bindings) for value in arrays.values()):
        raise RuntimeError('Cached example count differs from array axes')
    values=[]
    for i,binding in enumerate(bindings):
        result={'binding':binding,'ap_xyz':arrays['ap_xyz'][i].copy()}
        for name in ('a','b','query','target'):
            result[name]={key.split('__',1)[1]:value[i].copy() for key,value in arrays.items() if key.startswith(name+'__')}
        for name in ('a','b','query'):
            if {key:data.array_digest(value) for key,value in result[name].items()}!=binding['model_input_sha256'][name]:
                raise RuntimeError('Cached model input array identity changed')
        if (data.array_digest(result['ap_xyz'])!=binding['model_input_sha256']['ap_xyz']
                or {key:data.array_digest(value) for key,value in result['target'].items()}!=binding['target_sha256']):
            raise RuntimeError('Cached query/target identity changed')
        values.append(result)
    return values


def _verify_group(receipt,name=None):
    canonical=_group_name(receipt['stage'],receipt['regime'],receipt['held_band'],receipt['stale'])
    if (name is not None and name!=canonical or receipt['source_receipt_sha256']!=c.source_receipt_sha256()
            or receipt['registry_sha256']!=c.registry_sha256()
            or receipt['calibration_sha256']!=c.digest(c.OUT/'data/calibration.json')
            or receipt['stage_receipt_sha256']!=c.digest(stage_path(receipt['stage']))
            or receipt['arrays']['path']!=str((c.OUT/'data/groups'/f'{canonical}.npz').relative_to(c.ROOT))
            or c.reference(c.safe_path(receipt['arrays']['path']))!=receipt['arrays']):
        raise RuntimeError('Prepared group canonical identity changed')
    bindings=receipt['bindings'];stage=receipt['stage']
    if len(bindings)!=3*sim.COUNTS[stage]:raise RuntimeError('Exact complete geometry/state bindings required')
    for index,binding in enumerate(bindings):
        spec=sim.case_spec(stage,index//3,index%3)
        context_spec=sim.case_spec(stage,index//3,0) if receipt['stale'] else spec
        expected={'stage':stage,'index':index//3,'state':index%3,'condition':index%3,
            'layout_id':spec.layout_id,'case_id':spec.case_id,'context_case_id':context_spec.case_id,
            'regime':receipt['regime'],'held_band':receipt['held_band'],
            'stale_requested':bool(receipt['stale'] and spec.state!=0)}
        if any(binding.get(key)!=value for key,value in expected.items()):
            raise RuntimeError('Exact canonical group layout/state order required')
    return receipt


def examples(stage,regime='full',held_band=False,stale=False):
    name=_group_name(stage,regime,held_band,stale)
    c.check_sources()
    if stage in ('test','ap_transfer'):verify_development()
    path=c.OUT/'data/groups'/f'{name}.json'
    if path.exists():
        receipt=c.read(path)
        identity={'stage':stage,'regime':regime,'held_band':held_band,'stale':stale,
            'source_receipt_sha256':c.source_receipt_sha256(),'registry_sha256':c.registry_sha256(),
            'calibration_sha256':c.digest(c.OUT/'data/calibration.json'),
            'stage_receipt_sha256':c.digest(stage_path(stage))}
        if any(receipt.get(key)!=value for key,value in identity.items()):raise RuntimeError('Prepared group identity changed')
        if c.reference(c.safe_path(receipt['arrays']['path']))!=receipt['arrays']:raise RuntimeError('Prepared group bytes changed')
        _verify_group(receipt,name)
        key=c.digest(path)
        if key not in _GROUPS:_GROUPS[key]=_unpack_examples(c.arrays(c.safe_path(receipt['arrays']['path'])),receipt['bindings'])
        return _GROUPS[key]
    actual=scenes(stage);cal=calibration();values=[]
    for i,scene in enumerate(actual):
        context=actual[(i//3)*3] if stale else scene
        values.append(data.make_example(scene,cal,regime=regime,held_band=held_band,context_scene=context))
    refs=c.npz(c.OUT/'data/groups'/f'{name}.npz',_pack_examples(values))
    receipt={'schema':1,'stage':stage,'regime':regime,'held_band':held_band,'stale':stale,
        'source_receipt_sha256':c.source_receipt_sha256(),'registry_sha256':c.registry_sha256(),
        'calibration_sha256':c.digest(c.OUT/'data/calibration.json'),'stage_receipt_sha256':c.digest(stage_path(stage)),
        'bindings':[e['binding'] for e in values],'arrays':refs,'reported_only':True}
    c.write(path,receipt);_GROUPS[c.digest(path)]=values
    return values


def prepare():
    c.check_sources();verify_stage('training');verify_stage('validation')
    path=c.OUT/'data/calibration.json'
    if not path.exists():
        cal=data.fit_calibration(scenes('training'))
        c.write(path,{'source_receipt_sha256':c.source_receipt_sha256(),'registry_sha256':c.registry_sha256(),'calibration':cal})
    groups={}
    for stage in ('training','validation'):
        for regime in ('full','sparse'):
            examples(stage,regime)
            groups[stage+'-'+regime]=c.reference(c.OUT/'data/groups'/f'{stage}-{regime}.json')
    receipt={'schema':1,'source_receipt_sha256':c.source_receipt_sha256(),'protocol_sha256':c.protocol_sha256(),
        'registry_sha256':c.registry_sha256(),'calibration':c.reference(path),'groups':groups,
        'stages':{stage:c.reference(stage_path(stage)) for stage in ('training','validation')},
        'training_cases':72,'validation_cases':18,'target_band_indices':[0,1],
        'private_materials_true_receiver_offsets_or_query_features_from_query_RSSI':False}
    path=c.OUT/'data/prepared.json'
    if path.exists():
        if c.read(path)!=receipt:raise RuntimeError('Prepared TRAIN/VAL identity changed')
    else:c.write(path,receipt)
    return receipt


def verify_prepared():
    """Bind every fitting input receipt to the immutable TRAIN/VAL identity."""
    receipt=c.read(c.OUT/'data/prepared.json')
    if (receipt.get('source_receipt_sha256')!=c.source_receipt_sha256()
            or receipt.get('protocol_sha256')!=c.protocol_sha256()
            or receipt.get('registry_sha256')!=c.registry_sha256()
            or receipt.get('training_cases')!=72 or receipt.get('validation_cases')!=18
            or receipt.get('target_band_indices')!=[0,1]
            or receipt['calibration']!=c.reference(c.OUT/'data/calibration.json')
            or set(receipt['stages'])!={'training','validation'}
            or set(receipt['groups'])!={f'{stage}-{regime}' for stage in ('training','validation') for regime in ('full','sparse')}):
        raise RuntimeError('Prepared immutable fitting identity changed')
    for stage in ('training','validation'):
        if receipt['stages'][stage]!=c.reference(stage_path(stage)):
            raise RuntimeError('Prepared physical fitting stage changed')
        for regime in ('full','sparse'):
            name=f'{stage}-{regime}';path=c.OUT/'data/groups'/f'{name}.json'
            if receipt['groups'][name]!=c.reference(path):raise RuntimeError('Prepared fitting group receipt changed')
            _verify_group(c.read(path),name)
    return receipt


def numeric_candidates():
    from . import baselines as b
    c.check_sources();verify_prepared();receipt_path=c.OUT/'numerical/selection.json'
    if receipt_path.exists():return verify_numerical()
    training=examples('training','full');validation=examples('validation','sparse')
    candidates=[]
    for length in (5.,10.,20.):
        name=f'radiomap-{length:g}m';path=c.OUT/'numerical'/f'{name}.json'
        if not path.exists():
            c.attempt(c.OUT/'numerical/attempts'/f'{name}.json',{'source':c.source_receipt_sha256(),'lengthscale_m':length})
            began=time.perf_counter();prior=b.fit_prior(training,lengthscale=length)
            refs=c.npz(c.OUT/'numerical'/f'{name}.npz',b.pack_prior(prior))
            means=[];stds=[]
            for e in validation:
                mean,std=b.predict(prior,e);means.append(mean);stds.append(std)
            prediction=c.npz(c.OUT/'numerical'/f'{name}-validation.npz',{'mean':np.stack(means),'std':np.stack(stds)})
            score=ev.validation_nll(np.stack(means),np.stack(stds),{k:np.stack([e['target'][k] for e in validation]) for k in validation[0]['target']})
            c.write(path,{'name':name,'lengthscale_m':length,'prior':refs,'validation_predictions':prediction,
                'validation_nll':score,'elapsed_seconds':time.perf_counter()-began,
                'source_receipt_sha256':c.source_receipt_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json')})
        item=c.read(path);candidates.append(item)
    fixed_path=c.OUT/'numerical/hierarchy-fixed.json'
    if not fixed_path.exists():
        c.attempt(c.OUT/'numerical/attempts/hierarchy-fixed.json',{'source':c.source_receipt_sha256(),'method':'hierarchy_fixed'})
        prior=b.fit_fixed_prior(training);ref=c.npz(c.OUT/'numerical/hierarchy-fixed.npz',b.pack_prior(prior))
        c.write(fixed_path,{'prior':ref,'source_receipt_sha256':c.source_receipt_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json')})
    selected=min(candidates,key=lambda item:item['validation_nll'])
    receipt={'schema':1,'selected_name':selected['name'],'selected_prior':selected['prior'],
        'candidates':[c.reference(c.OUT/'numerical'/f'{item["name"]}.json') for item in candidates],
        'fixed_hierarchy':c.reference(fixed_path),'source_receipt_sha256':c.source_receipt_sha256(),
        'prepared_sha256':c.digest(c.OUT/'data/prepared.json'),'rule':'first_minimum_sparse_VAL_composite_interval_NLL'}
    c.write(receipt_path,receipt);return verify_numerical()


def verify_numerical():
    verify_prepared()
    receipt=c.read(c.OUT/'numerical/selection.json');validation=examples('validation','sparse')
    if (receipt['source_receipt_sha256']!=c.source_receipt_sha256()
            or receipt['prepared_sha256']!=c.digest(c.OUT/'data/prepared.json')):raise RuntimeError('Numerical selection source changed')
    if len(receipt['candidates'])!=3:raise RuntimeError('Exactly three numerical candidates required')
    values=[]
    for length,item in zip((5.,10.,20.),receipt['candidates']):
        path=c.OUT/'numerical'/f'radiomap-{length:g}m.json'
        if c.reference(path)!=item:raise RuntimeError('Numerical candidate receipt changed')
        row=c.read(path)
        if (row.get('name')!=f'radiomap-{length:g}m' or row.get('lengthscale_m')!=length
                or row.get('source_receipt_sha256')!=c.source_receipt_sha256()
                or row.get('prepared_sha256')!=c.digest(c.OUT/'data/prepared.json')):
            raise RuntimeError('Numerical candidate identity changed')
        for key in ('prior','validation_predictions'):
            expected=c.OUT/'numerical'/f'radiomap-{length:g}m{ "-validation" if key=="validation_predictions" else ""}.npz'
            if row[key]['path']!=str(expected.relative_to(c.ROOT)):raise RuntimeError('Numerical canonical path changed')
            if c.reference(c.safe_path(row[key]['path']))!=row[key]:raise RuntimeError('Numerical weights/predictions changed')
        output=c.arrays(c.safe_path(row['validation_predictions']['path']))
        score=ev.validation_nll(output['mean'],output['std'],{k:np.stack([e['target'][k] for e in validation]) for k in validation[0]['target']})
        if not math.isclose(score,row['validation_nll'],rel_tol=0,abs_tol=1e-10):raise RuntimeError('Numerical VAL score changed')
        values.append(row)
    if len(values)!=3 or receipt['selected_name']!=min(values,key=lambda row:row['validation_nll'])['name']:
        raise RuntimeError('Numerical selection did not use all three predeclared candidates')
    if receipt['selected_prior']!=min(values,key=lambda row:row['validation_nll'])['prior']:
        raise RuntimeError('Selected numerical prior binding changed')
    if receipt['fixed_hierarchy']['path']!=str((c.OUT/'numerical/hierarchy-fixed.json').relative_to(c.ROOT)):
        raise RuntimeError('Fixed hierarchy canonical receipt changed')
    fixed=c.read(c.safe_path(receipt['fixed_hierarchy']['path']))
    if (c.reference(c.safe_path(receipt['fixed_hierarchy']['path']))!=receipt['fixed_hierarchy']
            or c.reference(c.safe_path(fixed['prior']['path']))!=fixed['prior']
            or fixed['source_receipt_sha256']!=c.source_receipt_sha256()
            or fixed['prepared_sha256']!=c.digest(c.OUT/'data/prepared.json')
            or fixed['prior']['path']!=str((c.OUT/'numerical/hierarchy-fixed.npz').relative_to(c.ROOT))):
        raise RuntimeError('Fixed hierarchy control changed')
    return receipt


def select_models():
    from . import training as t
    from .model import CELLS
    c.check_sources();identity=t.identity();fits={};checkpoints={};scores={}
    validation=examples('validation','sparse')
    for cell in CELLS:
        scores[cell]=[]
        for seed in c.SEEDS:
            path=c.OUT/'fits'/f'{cell}-seed{seed}.json';fit=c.read(path)
            t.verify_completed(fit,identity,cell,seed,validation_examples=validation)
            fits[f'{cell}-seed{seed}']=c.reference(path)
            checkpoints[f'{cell}-seed{seed}']=c.reference(c.OUT/'checkpoints'/f'{cell}-seed{seed}.pt')
            scores[cell].append(fit['validation_nll'])
    mean={cell:float(np.mean(values)) for cell,values in scores.items()}
    selected=min(('hier_reg','hier_joint'),key=lambda name:mean[name])
    numeric=verify_numerical()
    receipt={'schema':1,'selected_hierarchy':selected,'mean_validation_nll':mean,
        'member_validation_nll':scores,'fits':fits,'checkpoints':checkpoints,
        'numerical_selection':c.reference(c.OUT/'numerical/selection.json'),
        'source_receipt_sha256':c.source_receipt_sha256(),'protocol_sha256':c.protocol_sha256(),
        'registry_sha256':c.registry_sha256(),'prepared_sha256':c.digest(c.OUT/'data/prepared.json'),
        'selection_used_TEST_native_or_original_SERVICE':False}
    path=c.OUT/'selection.json'
    if path.exists():
        if c.read(path)!=receipt:raise RuntimeError('Locked selection changed')
    else:c.write(path,receipt)
    return receipt


def verify_selection():
    # select_models is a passive verifier when the immutable receipt exists;
    # all VAL scores are reconstructed from archived predictions, never forwards.
    return select_models()


def methods_for(*,secondary=False):
    from .model import CELLS
    selected=c.read(c.OUT/'selection.json')['selected_hierarchy']
    cells=(selected,'global_reg') if secondary else CELLS
    return [f'{cell}-seed{seed}' for cell in cells for seed in c.SEEDS]+['radiomap_operator','hierarchy_fixed']


def predictor(method):
    from . import training as t, baselines as b
    if method not in methods_for():raise ValueError('Unknown registered field predictor')
    if '-seed' in method:
        if method not in _MODELS:
            cell,seed=method.rsplit('-seed',1);_MODELS[method]=t.load(cell,int(seed))
        return lambda e:t.predict(_MODELS[method],e)
    if method not in _NUMERICAL:
        selected=c.read(c.OUT/'numerical/selection.json')
        ref=selected['selected_prior'] if method=='radiomap_operator' else c.read(c.safe_path(selected['fixed_hierarchy']['path']))['prior']
        if c.reference(c.safe_path(ref['path']))!=ref:raise RuntimeError('Numerical predictor bytes changed')
        _NUMERICAL[method]=b.load_prior(c.arrays(c.safe_path(ref['path'])))
    return lambda e:b.predict(_NUMERICAL[method],e)


def predict_group(stage,regime='sparse',*,held_band=False,stale=False):
    name=_group_name(stage,regime,held_band,stale);c.check_sources();verify_selection()
    if stage in ('test','ap_transfer'):verify_development()
    values=examples(stage,regime,held_band,stale);methods=methods_for(secondary=held_band or stale)
    refs={}
    for index,e in enumerate(values):
        case=e['binding']['case_id'];folder=c.OUT/'predictions'/name;receipt_path=folder/f'{case}.json'
        identity={'stage':stage,'group':name,'case_id':case,'layout_id':e['binding']['layout_id'],
            'state':e['binding']['state'],'methods':methods,'example_index':index,
            'example_group':c.reference(c.OUT/'data/groups'/f'{name}.json'),
            'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json')}
        if not receipt_path.exists():
            c.attempt(folder/'attempts'/f'{case}.json',identity)
            means=[];stds=[];elapsed=[];metrics=[]
            for method in methods:
                began=time.perf_counter();mean,std=predictor(method)(e)
                elapsed.append(time.perf_counter()-began);means.append(mean);stds.append(std)
                metrics.append(ev.metrics(mean,std,e['target']))
            arrays=c.npz(folder/f'{case}.npz',{'mean':np.stack(means),'std':np.stack(stds)})
            c.write(receipt_path,identity|{'predictions':arrays,'metrics':metrics,'elapsed_seconds':elapsed,
                'attempt':c.reference(folder/'attempts'/f'{case}.json')})
        receipt=c.read(receipt_path)
        if any(receipt.get(key)!=value for key,value in identity.items()):raise RuntimeError('Prediction membership changed')
        verify_prediction(receipt,e)
        refs[case]=c.reference(receipt_path)
    result={'schema':1,'group':name,'stage':stage,'case_receipts':refs,'methods':methods,
        'source_receipt_sha256':c.source_receipt_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json'),
        'example_group':c.reference(c.OUT/'data/groups'/f'{name}.json')}
    path=c.OUT/'predictions'/f'{name}.json'
    if path.exists():
        if c.read(path)!=result:raise RuntimeError('Prediction group changed')
    else:c.write(path,result)
    print('Predictions',name,len(refs),'cases',flush=True)
    return result


def verify_prediction(receipt,e=None):
    if (receipt['source_receipt_sha256']!=c.source_receipt_sha256()
            or receipt['selection_sha256']!=c.digest(c.OUT/'selection.json')):raise RuntimeError('Prediction source changed')
    for key in ('predictions','attempt','example_group'):
        if c.reference(c.safe_path(receipt[key]['path']))!=receipt[key]:raise RuntimeError('Prediction dependency changed')
    group=_verify_group(c.read(c.safe_path(receipt['example_group']['path'])))
    name=_group_name(group['stage'],group['regime'],group['held_band'],group['stale'])
    expected_methods=methods_for(secondary=group['held_band'] or group['stale'])
    if (receipt['group']!=name or receipt['stage']!=group['stage'] or receipt['methods']!=expected_methods
            or receipt['example_group']['path']!=str((c.OUT/'data/groups'/f'{name}.json').relative_to(c.ROOT))
            or group['source_receipt_sha256']!=c.source_receipt_sha256()
            or group['registry_sha256']!=c.registry_sha256()
            or group['calibration_sha256']!=c.digest(c.OUT/'data/calibration.json')
            or group['stage_receipt_sha256']!=c.digest(stage_path(group['stage']))
            or c.reference(c.safe_path(group['arrays']['path']))!=group['arrays']):
        raise RuntimeError('Prediction canonical group or method membership changed')
    index=receipt['example_index']
    if type(index) is not int or not 0<=index<len(group['bindings']):raise RuntimeError('Prediction example index changed')
    if e is None:
        values=_unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])
        e=values[index]
    if (e['binding']['case_id']!=receipt['case_id'] or e['binding']['layout_id']!=receipt['layout_id']
            or e['binding']['state']!=receipt['state'] or e['binding']!=group['bindings'][index]):
        raise RuntimeError('Prediction example identity changed')
    spec=sim.case_spec(group['stage'],index//3,index%3)
    folder=c.OUT/'predictions'/name
    if (receipt['case_id']!=spec.case_id or receipt['layout_id']!=spec.layout_id
            or receipt['state']!=spec.state
            or receipt['predictions']['path']!=str((folder/f'{spec.case_id}.npz').relative_to(c.ROOT))
            or receipt['attempt']['path']!=str((folder/'attempts'/f'{spec.case_id}.json').relative_to(c.ROOT))):
        raise RuntimeError('Prediction canonical case path changed')
    attempt=c.read(c.safe_path(receipt['attempt']['path']))
    expected_attempt={key:receipt[key] for key in ('stage','group','case_id','layout_id','state','methods',
        'example_index','example_group','source_receipt_sha256','selection_sha256')}
    if {key:value for key,value in attempt.items() if key!='created_unix_ns'}!=expected_attempt:
        raise RuntimeError('Prediction attempt identity changed')
    output=c.arrays(c.safe_path(receipt['predictions']['path']))
    if (set(output)!={'mean','std'} or output['mean'].shape!=(len(receipt['methods']),len(e['target']['heard']))
            or output['std'].shape!=output['mean'].shape or len(receipt['metrics'])!=len(expected_methods)
            or len(receipt['elapsed_seconds'])!=len(expected_methods)
            or not np.isfinite(receipt['elapsed_seconds']).all() or min(receipt['elapsed_seconds'])<0):
        raise RuntimeError('Prediction array membership changed')
    actual=[ev.metrics(mean,std,e['target']) for mean,std in zip(output['mean'],output['std'])]
    if actual!=receipt['metrics']:raise RuntimeError('Prediction metrics differ from actual arrays')
    return receipt,output,e


def development():
    for regime in ('full','sparse'):predict_group('development',regime)
    groups={regime:c.reference(c.OUT/'predictions'/f'development-{regime}.json') for regime in ('full','sparse')}
    gate={'schema':1,'technical_pass':True,'efficacy_screening':False,'expected_layouts':4,'expected_states':3,
        'groups':groups,'source_receipt_sha256':c.source_receipt_sha256(),'protocol_sha256':c.protocol_sha256(),
        'registry_sha256':c.registry_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json'),
        'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
    path=c.OUT/'development_gate.json'
    if path.exists():
        if c.read(path)!=gate:raise RuntimeError('Technical gate changed')
    else:c.write(path,gate)
    return verify_development()


def verify_development():
    verify_selection();gate=c.read(c.OUT/'development_gate.json')
    expected={'schema':1,'technical_pass':True,'efficacy_screening':False,'expected_layouts':4,'expected_states':3,
        'source_receipt_sha256':c.source_receipt_sha256(),'protocol_sha256':c.protocol_sha256(),
        'registry_sha256':c.registry_sha256(),'selection_sha256':c.digest(c.OUT/'selection.json'),
        'prepared_sha256':c.digest(c.OUT/'data/prepared.json')}
    if (any(gate.get(key)!=value for key,value in expected.items())
            or gate.get('technical_pass') is not True or gate.get('efficacy_screening') is not False):
        raise RuntimeError('Technical DEV gate source/selection changed')
    if set(gate['groups'])!={'full','sparse'}:raise RuntimeError('Both DEV regimes required')
    for regime,item in gate['groups'].items():
        path=c.OUT/'predictions'/f'development-{regime}.json'
        if c.reference(path)!=item:raise RuntimeError('DEV group receipt changed')
        manifest=c.read(path);expected_cases={sim.case_spec('development',i,s).case_id for i in range(4) for s in sim.STATES}
        if (set(manifest['case_receipts'])!=expected_cases or manifest['methods']!=methods_for()
                or manifest['source_receipt_sha256']!=c.source_receipt_sha256()
                or manifest['selection_sha256']!=c.digest(c.OUT/'selection.json')):
            raise RuntimeError('DEV predictions do not cover the fixed member/case schedule')
        if (manifest.get('group')!=f'development-{regime}' or manifest.get('stage')!='development'
                or manifest['example_group']!=c.reference(c.OUT/'data/groups'/f'development-{regime}.json')):
            raise RuntimeError('DEV canonical example group changed')
        group=_verify_group(c.read(c.safe_path(manifest['example_group']['path'])),f'development-{regime}')
        if c.reference(c.safe_path(group['arrays']['path']))!=group['arrays']:raise RuntimeError('DEV example arrays changed')
        values=_unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])
        for e in values:
            item=manifest['case_receipts'][e['binding']['case_id']]
            if c.reference(c.safe_path(item['path']))!=item:raise RuntimeError('DEV case prediction changed')
            verify_prediction(c.read(c.safe_path(item['path'])),e)
    return gate


def group_records(name):
    manifest=c.read(c.OUT/'predictions'/f'{name}.json');records=[]
    group=_verify_group(c.read(c.safe_path(manifest['example_group']['path'])),name)
    methods=methods_for(secondary=group['held_band'] or group['stale'])
    if (manifest.get('group')!=name or manifest.get('stage')!=group['stage']
            or manifest['methods']!=methods or manifest['source_receipt_sha256']!=c.source_receipt_sha256()
            or manifest['selection_sha256']!=c.digest(c.OUT/'selection.json')
            or manifest['example_group']!=c.reference(c.OUT/'data/groups'/f'{name}.json')
            or set(manifest['case_receipts'])!={b['case_id'] for b in group['bindings']}):
        raise RuntimeError('Evaluation manifest membership changed')
    if c.reference(c.safe_path(group['arrays']['path']))!=group['arrays']:raise RuntimeError('Evaluation group arrays changed')
    values=_unpack_examples(c.arrays(c.safe_path(group['arrays']['path'])),group['bindings'])
    for e in values:
        item=manifest['case_receipts'][e['binding']['case_id']]
        if c.reference(c.safe_path(item['path']))!=item:raise RuntimeError('Evaluation prediction receipt changed')
        receipt,output,_=verify_prediction(c.read(c.safe_path(item['path'])),e)
        for j,method in enumerate(receipt['methods']):
            records.append({'method':method,'layout_id':receipt['layout_id'],'state':receipt['state'],
                'metrics':receipt['metrics'][j],'mean':output['mean'][j],'std':output['std'][j],'target':e['target']})
    return records


def field_report():
    from .model import CELLS
    selected=verify_selection()['selected_hierarchy'];names={cell:[f'{cell}-seed{s}' for s in c.SEEDS] for cell in CELLS}
    names.update(selected_hierarchy=names[selected],radiomap_operator=['radiomap_operator'],hierarchy_fixed=['hierarchy_fixed'])
    layouts=[sim.geometry_spec('test',i).scene_id for i in range(24)]
    sparse=ev.reduce_layouts(group_records('test-sparse'),methods=names,layout_ids=layouts)
    full=ev.reduce_layouts(group_records('test-full'),methods=names,layout_ids=layouts)
    contrasts=[]
    for candidate,reference in c.protocol_spec()['evaluation']['contrasts']:
        a=[row['metrics']['rmse_db'] for row in sparse[candidate]];b=[row['metrics']['rmse_db'] for row in sparse[reference]]
        contrasts.append({'candidate':candidate,'reference':reference,'metric':'sparse_heard_RSSI_RMSE_db',**ev.paired_interval(a,b)})
    raw_sparse=group_records('test-sparse');seed_contrasts=[]
    for seed in c.SEEDS:
        member_names={cell:[f'{cell}-seed{seed}'] for cell in CELLS}
        member_names.update(selected_hierarchy=member_names[selected],radiomap_operator=['radiomap_operator'])
        member_set=set().union(*map(set,member_names.values()))
        member_values=ev.reduce_layouts([r for r in raw_sparse if r['method'] in member_set],methods=member_names,layout_ids=layouts)
        for candidate,reference in c.protocol_spec()['evaluation']['contrasts']:
            deltas=np.array([r['metrics']['rmse_db'] for r in member_values[candidate]])-np.array(
                [r['metrics']['rmse_db'] for r in member_values[reference]])
            seed_contrasts.append({'seed':seed,'candidate':candidate,'reference':reference,
                'mean_delta_db':float(deltas.mean()),'layouts_lower':int((deltas<0).sum()),
                'layouts_equal':int((deltas==0).sum()),'layouts_higher':int((deltas>0).sum()),
                'descriptive_no_additional_hypothesis_test':True})
    summaries={regime:{method:{key:float(np.mean([r['metrics'][key] for r in rows])) for key in rows[0]['metrics']}
        for method,rows in values.items()} for regime,values in (('sparse',sparse),('full',full))}
    delta={}
    raw=raw_sparse
    for method,members in names.items():
        per_layout=[]
        for layout in layouts:
            member_errors=[];pair_counts=[]
            for member in members:
                states={r['state']:r for r in raw if r['layout_id']==layout and r['method']==member}
                errors=[]
                for state in (1,2):
                    a,b=states[0],states[state];known=a['target']['heard']&b['target']['heard']
                    if known.any():
                        truth=b['target']['center_dbm'][known]-a['target']['center_dbm'][known]
                        errors.extend(((b['mean'][known]-a['mean'][known])-truth).tolist())
                if not errors:raise RuntimeError('No heard paired material-change targets for this geometry')
                member_errors.append(float(np.sqrt(np.mean(np.square(errors)))));pair_counts.append(len(errors))
            per_layout.append({'layout_id':layout,'delta_rmse_db':float(np.mean(member_errors)),
                               'paired_targets_per_member':pair_counts})
        delta[method]={'mean_layout_delta_rmse_db':float(np.mean([r['delta_rmse_db'] for r in per_layout])), 'layouts':per_layout}
    secondary={}
    for group_name in ('test-sparse-held6GHz','test-sparse-stale','ap_transfer-sparse','ap_transfer-full'):
        records=group_records(group_name);stage='ap_transfer' if group_name.startswith('ap_transfer') else 'test'
        ids=[sim.geometry_spec(stage,i).scene_id for i in range(sim.COUNTS[stage])]
        present={r['method'] for r in records};mapping={m:members for m,members in names.items() if set(members)<=present}
        values=ev.reduce_layouts(records,methods=mapping,layout_ids=ids)
        secondary[group_name]={m:{k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']} for m,rows in values.items()}
    report={'schema':1,'complete':True,'selected_hierarchy':selected,'source_receipt_sha256':c.source_receipt_sha256(),
        'selection_sha256':c.digest(c.OUT/'selection.json'),'primary_layouts':24,'members':3,
        'actual_neural_fits':15,'actual_RT_cases':186,'primary_contrasts':contrasts,
        'per_training_seed_contrasts_descriptive':seed_contrasts,
        'summary':summaries,'layout_metrics':{'sparse':sparse,'full':full},
        'localized_delta_descriptive':delta,'secondary_descriptive':secondary,
        'topK_heard_Brier_from_floor_not_claimed':True,'packet_risk_or_SOTA_qualified':False,
        'original_SERVICE_opened':False,'default_changed':False}
    path=c.OUT/'field_report.json'
    if path.exists():
        if c.read(path)!=report:raise RuntimeError('Field report changed')
    else:c.write(path,report)
    return report


def seal():
    c.check_sources();verify_development();field=field_report()
    for stage in sim.COUNTS:
        verify_stage(stage)
        for index in range(sim.COUNTS[stage]):
            for state in sim.STATES:
                spec=sim.case_spec(stage,index,state)
                sim.validate_case_receipt(spec,c.case_path(spec),source_sha256=c.source_receipt_sha256(),
                    registry_sha256=c.registry_sha256(),reconstruct_reports=True)
    from .native_study import verify_native
    native=verify_native()
    report={'schema':1,'complete':True,'field_report':c.reference(c.OUT/'field_report.json'),
        'native_report':c.reference(c.OUT/'native/report.json'),'actual_RT_cases':186,
        'neural_fits':15,'native_fresh_simulations':32,'native_parity_simulations':2,
        'source_receipt_sha256':c.source_receipt_sha256(),'production_qualified':False,
        'original_SERVICE_opened':False,'default_changed':False}
    path=c.OUT/'report.json'
    if path.exists():
        if c.read(path)!=report:raise RuntimeError('Final report changed')
    else:c.write(path,report)
    prior=c.prior_identities(verify_bytes=True)
    evidence={str(path.relative_to(c.ROOT)):c.digest(path) for path in sorted(c.OUT.rglob('*'))
        if path.is_file() and not str(path.relative_to(c.OUT)).startswith('registration/source/')
        and path!=c.OUT/'verification/manifest.json'}
    manifest={'schema':1,'source_sha256':c.check_sources()['source_sha256'],
        'evidence_sha256':evidence,'prior':prior,'RT_cases':186,'neural_fits':15,
        'native_simulations':34,'production_qualified':False,'original_SERVICE_opened':False}
    path=c.OUT/'verification/manifest.json'
    if path.exists():
        if c.read(path)!=manifest:raise RuntimeError('Final evidence closure changed')
    else:c.write(path,manifest)
    return report


def run(workers=3):
    c.check_sources()
    for stage in ('training','validation'):collect(stage)
    prepare();numeric_candidates()
    from .training import train_all
    train_all(workers);select_models()
    collect('development');development()
    for stage in ('test','ap_transfer'):collect(stage)
    for stage in ('test','ap_transfer'):
        for regime in ('full','sparse'):predict_group(stage,regime)
    predict_group('test','sparse',held_band=True);predict_group('test','sparse',stale=True)
    field_report()
    from .native_study import run_native
    run_native();seal()
    print('Registered hierarchy study complete; all nulls/harms retained; no promotion.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('write-spec','register','publish','run','report'))
    parser.add_argument('--commit');parser.add_argument('--branch');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.command=='write-spec':c.write_spec();print(c.PROTOCOL)
    elif args.command=='register':print('registered',len(c.register()['source_sha256']),'sources')
    elif args.command=='publish':print(c.record_publication(args.commit,args.branch))
    elif args.command=='run':run(args.workers)
    else:print(c.encoded(field_report()).decode())


if __name__=='__main__':main()
