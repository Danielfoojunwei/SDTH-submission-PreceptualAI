"""Published fixed-budget nested-field training with durable committed steps.

Only the immutable parent TRAIN/validation groups enter optimization or
selection. Every validation candidate retains raw predictions and inert weight
state; point-RMSE selection is reduced from those archives without forwards.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from ..hierarchical_radio.training import (
    batch, select, tensor, validate_examples, atomic_save, state_digest,
    exclusive_fit, _publish_bytes, _publish_torch, _publish_json,
    _finite_tree, _clone_state,
)
from ..hierarchical_radio.model import observation_nll

CELLS=('global8_reg','global8_joint','hier_reg','nested_reg','nested_joint')
SEEDS=(0,1,2)
EPOCHS=160
UPDATES_PER_EPOCH=12
COMMITTED_UPDATES=EPOCHS*UPDATES_PER_EPOCH
VALIDATION_EPOCHS=(40,80,120,160)
BATCH_LAYOUTS=4
TRAIN_LAYOUTS=24
CONDITIONS=3
OPTIMIZER_SPEC={'name':'AdamW','learning_rate':.003,'weight_decay':1.e-4,
                'betas':(.9,.999),'eps':1.e-8,'gradient_clip_norm':5.}


def _common():
    from . import common
    return common


def _model_api():
    from .model import NestedField, objective
    return NestedField, objective


def _canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _member(cell,seed):
    if cell not in CELLS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only five registered cells and seeds0/1/2 may be trained')
    return f'{cell}-seed{seed}'


def settings(cell):
    if cell not in CELLS:raise ValueError('Unknown registered nested-field cell')
    return {'cell':cell,'width':32,'inference_steps':3}


def identity():
    c=_common();c.check_sources();c.verify_parent_data()
    from .data import verify_prepared
    verify_prepared()
    path=c.OUT/'data/prepared.json'
    if not path.is_file() or path.is_symlink() or not isinstance(c.read(path),dict):
        raise RuntimeError('Prepare immutable parent TRAIN/VAL receipts before fitting')
    return {'source_receipt_sha256':c.source_receipt_sha256(),
            'protocol_sha256':c.protocol_sha256(),
            'data_receipt_sha256':c.digest(path)}


def datasets():
    from .data import training_examples
    full=training_examples('training','full');sparse=training_examples('training','sparse')
    validation=training_examples('validation','sparse')
    validate_fitting_examples(full,'training','full');validate_fitting_examples(sparse,'training','sparse')
    validate_fitting_examples(validation,'validation','sparse')
    if any((a['binding']['case_id'],a['binding']['layout_id'])!=(b['binding']['case_id'],b['binding']['layout_id']) for a,b in zip(full,sparse)):
        raise RuntimeError('Full/sparse fitting groups must contain the same physical cases')
    return full,sparse,validation


def validate_fitting_examples(examples,stage,regime):
    """Verify supplied arrays, not only the labels of an optional preload."""
    from ..hierarchical_radio.data import array_digest
    layouts=validate_examples(examples,stage,regime)
    for example in examples:
        binding=example['binding']
        expected={role:{key:array_digest(value) for key,value in example[role].items()}
                  for role in ('a','b','query')}
        expected['ap_xyz']=array_digest(example['ap_xyz'])
        target={key:array_digest(value) for key,value in example['target'].items()}
        if binding.get('model_input_sha256')!=expected or binding.get('target_sha256')!=target:
            raise RuntimeError('Fitting example array bytes differ from bound inputs/targets')
    return layouts


def data_identity(full,sparse,validation):
    return {name:hashlib.sha256(_canonical([e['binding'] for e in values])).hexdigest()
            for name,values in (('training_full',full),('training_sparse',sparse),('validation_sparse',validation))}


def choose_minibatch(rng,committed_updates):
    if type(committed_updates) is not int or not 0<=committed_updates<COMMITTED_UPDATES:
        raise ValueError('Registered committed-update index required')
    layouts=rng.choice(TRAIN_LAYOUTS,size=BATCH_LAYOUTS,replace=False)
    conditions=rng.integers(0,CONDITIONS,size=BATCH_LAYOUTS)
    return layouts*3+conditions,layouts,conditions,'full' if committed_updates%2==0 else 'sparse'


def validate_trace(trace,*,cell=None,complete=False):
    if not isinstance(trace,list) or len(trace)>COMMITTED_UPDATES or complete and len(trace)!=COMMITTED_UPDATES:
        raise RuntimeError('Fixed1920 committed optimizer-step budget changed')
    fields=('loss','measurement_nll','alignment','alignment_weight','gate_penalty',
        'supported_agreement_queries','mean_predictive_std_db','gradient_norm_before_clip',
        'mean_inference_step_size','inference_null_steps','gate_mean','gate_zero_count',
        'effective_prior_rms_normalized_mean','effective_prior_rms_mean_db',
        'gate_penalty_weight','gate_rms_reference')
    for i,row in enumerate(trace):
        if (row.get('step')!=i+1 or row.get('epoch')!=i//12+1 or row.get('update_in_epoch')!=i%12+1
                or row.get('regime')!=('full' if i%2==0 else 'sparse')
                or len(row.get('layout_indices',()))!=4 or len(set(row['layout_indices']))!=4
                or any(type(x) is not int or not 0<=x<24 for x in row['layout_indices'])
                or len(row.get('conditions',()))!=4 or any(type(x) is not int or x not in (0,1,2) for x in row['conditions'])
                or len(row.get('case_ids',()))!=4 or any(not isinstance(x,str) or not x for x in row['case_ids'])
                or any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) for k in fields)):
            raise RuntimeError('Malformed/nonfinite committed optimization trace')
        if (row['gradient_norm_before_clip']<0 or not 0<=row['gate_mean']<=1
                or not 0<=row['gate_zero_count']<=16 or row['gate_penalty']<0
                or row['effective_prior_rms_normalized_mean']<0 or row['effective_prior_rms_mean_db']<0
                or row['gate_rms_reference']!=.5
                or not math.isclose(row['effective_prior_rms_mean_db'],5.*row['effective_prior_rms_normalized_mean'],rel_tol=1e-6,abs_tol=1e-7)
                or not math.isclose(row['gate_penalty'],row['gate_penalty_weight']*row['effective_prior_rms_normalized_mean'],rel_tol=1e-6,abs_tol=1e-8)
                or not any(math.isclose(row['alignment_weight'],v,rel_tol=0,abs_tol=1e-7) for v in (0.,.1))
                or not math.isclose(row['loss'],row['measurement_nll']+row['alignment_weight']*row['alignment']+row['gate_penalty'],rel_tol=1e-7,abs_tol=1e-7)):
            raise RuntimeError('Trace objective or bounded gate changed')
        if cell is not None:
            weight=.1 if cell in ('global8_joint','nested_joint') else 0.
            penalty_weight=.01 if cell in ('nested_reg','nested_joint') else 0.
            if (not math.isclose(row['alignment_weight'],weight,rel_tol=0,abs_tol=1e-7)
                    or not math.isclose(row['gate_penalty_weight'],penalty_weight,rel_tol=0,abs_tol=1e-8)):
                raise RuntimeError('Registered variant alignment/shrinkage penalty changed')


def verify_schedule(trace,seed,full=None,sparse=None):
    rng=np.random.default_rng(seed)
    for i,row in enumerate(trace):
        indices,layouts,conditions,regime=choose_minibatch(rng,i)
        if (row['layout_indices']!=layouts.astype(int).tolist() or row['conditions']!=conditions.astype(int).tolist()
                or row['regime']!=regime):raise RuntimeError('Committed minibatches differ from fixed-seed trajectory')
        values=full if regime=='full' else sparse
        if values is not None and row['case_ids']!=[values[int(j)]['binding']['case_id'] for j in indices]:
            raise RuntimeError('Committed case IDs differ from the admitted TRAIN group')
    return rng.bit_generator.state


def validate_history(history,trace,*,cell=None,complete=False):
    validate_trace(trace,cell=cell,complete=complete)
    maximum=len(trace)//12;allowed={maximum}
    if trace and len(trace)%12==0:allowed.add(maximum-1)
    if not isinstance(history,list) or len(history) not in allowed or complete and len(history)!=EPOCHS:
        raise RuntimeError('Epoch ledger differs from committed updates')
    for i,row in enumerate(history):
        epoch=i+1
        if row.get('epoch')!=epoch or row.get('completed_updates')!=epoch*12 or row.get('mean_training_loss')!=float(np.mean([r['loss'] for r in trace[i*12:(i+1)*12]])):
            raise RuntimeError('Epoch ordering/training reduction changed')
        fields=('validation_rmse','validation_nll','validation_per_layout','validation_predictions','validation_state_sha256','epoch_checkpoint')
        if epoch in VALIDATION_EPOCHS:
            values=row.get('validation_per_layout')
            if (not isinstance(values,list) or len(values)!=6 or len({x.get('layout_id') for x in values})!=6
                    or any(x.get('conditions')!=[0,1,2] or type(x.get('rmse_db')) not in (int,float)
                           or not math.isfinite(x['rmse_db']) or x['rmse_db']<0
                           or type(x.get('nll')) not in (int,float) or not math.isfinite(x['nll']) for x in values)
                    or row.get('validation_rmse')!=float(np.mean([x['rmse_db'] for x in values]))
                    or row.get('validation_nll')!=float(np.mean([x['nll'] for x in values]))
                    or any(not isinstance(row.get(k),dict) or set(row[k])!={'path','sha256'} for k in ('validation_predictions','epoch_checkpoint'))
                    or not isinstance(row.get('validation_state_sha256'),str) or len(row['validation_state_sha256'])!=64):
                raise RuntimeError('Six-layout sparse A-only validation ledger changed')
        elif any(row.get(key) is not None for key in fields):raise RuntimeError('Unregistered validation checkpoint')


def selected_validation(history):
    candidates=[row for row in history if row['epoch'] in VALIDATION_EPOCHS]
    if not candidates:return None,None
    chosen=min(candidates,key=lambda row:row['validation_rmse'])
    return chosen['epoch'],chosen['validation_rmse']


def reduce_validation_predictions(mean,std,examples):
    """Equal query-weight RMSE per state, then equal states and whole layouts."""
    validate_examples(examples,'validation','sparse')
    mean=np.asarray(mean);std=np.asarray(std)
    expected=(18,len(examples[0]['target']['heard']))
    if (mean.shape!=expected or std.shape!=expected or mean.dtype.kind!='f' or std.dtype.kind!='f'
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any()):
        raise RuntimeError('Complete finite positive-spread18xQ validation arrays required')
    values=batch(examples);means=torch.from_numpy(np.array(mean,copy=True));stds=torch.from_numpy(np.array(std,copy=True))
    rows=[]
    with torch.no_grad():
        for i in range(6):
            errors=[]
            for j in range(i*3,(i+1)*3):
                target=examples[j]['target'];heard=np.asarray(target['heard']);center=np.asarray(target['center_dbm'])
                if heard.dtype!=bool or heard.shape!=mean[j].shape or center.shape!=mean[j].shape or not heard.any() or not np.isfinite(center[heard]).all():
                    raise RuntimeError('Every validation state needs genuine finite heard target centers')
                errors.append(float(np.sqrt(np.mean((mean[j,heard]-center[heard].astype('f8'))**2))))
            target={key:value[i*3:(i+1)*3] for key,value in values['target'].items()}
            nll=float(observation_nll(means[i*3:(i+1)*3],stds[i*3:(i+1)*3],target))
            if not math.isfinite(nll):raise FloatingPointError('Nonfinite raw validation composite likelihood')
            rows.append({'layout_id':examples[i*3]['binding']['layout_id'],'conditions':[0,1,2],
                         'rmse_db':float(np.mean(errors)),'nll':nll})
    return float(np.mean([r['rmse_db'] for r in rows])),float(np.mean([r['nll'] for r in rows])),rows


@torch.no_grad()
def validation_score(model,examples,values=None,*,return_predictions=False):
    validate_fitting_examples(examples,'validation','sparse');expected=batch(examples)
    if values is None:values=expected
    elif (set(values)!=set(expected) or any(
            set(values[role])!=set(expected[role]) or any(not torch.equal(values[role][key],value) for key,value in expected[role].items())
            if isinstance(expected[role],dict) else not torch.equal(values[role],expected[role]) for role in expected)):
        raise RuntimeError('Batched validation tensors differ from exact bound examples')
    previous=model.training;model.eval();means=[];stds=[]
    try:
        for i in range(6):
            v=select(values,np.arange(i*3,(i+1)*3));mean,std=model(v['a'],v['query'],v['ap_xyz'])
            means.append(mean.detach().cpu().numpy().copy());stds.append(std.detach().cpu().numpy().copy())
    finally:model.train(previous)
    mean,std=np.concatenate(means),np.concatenate(stds)
    score,nll,rows=reduce_validation_predictions(mean,std,examples)
    return (score,nll,rows,mean,std) if return_predictions else (score,nll,rows)


def validation_metadata(ident,binding,cell,seed,epoch,weights):
    return {'schema':1,'identity':ident,'dataset_binding_sha256':binding,'cell':cell,'seed':seed,
        'epoch':epoch,'committed_updates':epoch*12,'model_state_sha256':weights,
        'source_view':'a','regime':'sparse','target_binding_sha256':binding['validation_sparse'],
        'selection_endpoint':'equal_layout_equal_state_heard_RSSI_RMSE',
        'composite_NLL_for_selection':False}


def read_validation_archive(path,metadata,examples):
    validate_fitting_examples(examples,'validation','sparse')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'mean','std','metadata_utf8'}:raise RuntimeError('Exact validation archive membership changed')
        raw=archive['metadata_utf8']
        if raw.dtype!=np.uint8 or raw.ndim!=1 or json.loads(raw.tobytes())!=metadata:
            raise RuntimeError('Validation archive is not this exact source/data/member/epoch/weight state')
        mean,std=archive['mean'].copy(),archive['std'].copy()
    score,nll,rows=reduce_validation_predictions(mean,std,examples)
    return score,nll,rows,mean,std


def _gates_valid(state,cell):
    value=state.get('local_gates')
    if not isinstance(value,torch.Tensor) or value.shape!=(16,) or not torch.isfinite(value).all() or ((value<0)|(value>1)).any():
        raise RuntimeError('Checkpoint local gates must be finite projected16-vector')
    constant=0. if cell.startswith('global8_') else 1. if cell=='hier_reg' else None
    if constant is not None and not torch.equal(value,torch.full_like(value,constant)):
        raise RuntimeError('Fixed-control gate buffer changed')


def _epoch_valid(checkpoint,ident,binding,cell,seed,epoch,score,nll,weights):
    if (checkpoint.get('identity')!=ident or checkpoint.get('dataset_binding_sha256')!=binding
            or checkpoint.get('cell')!=cell or type(checkpoint.get('seed')) is not int or checkpoint['seed']!=seed
            or checkpoint.get('settings')!=settings(cell) or checkpoint.get('epoch')!=epoch
            or checkpoint.get('committed_updates')!=epoch*12 or checkpoint.get('validation_rmse')!=score
            or checkpoint.get('validation_nll')!=nll or not isinstance(checkpoint.get('state'),dict)
            or not checkpoint['state'] or not all(isinstance(v,torch.Tensor) for v in checkpoint['state'].values())
            or not _finite_tree(checkpoint['state']) or state_digest(checkpoint['state'])!=weights):
        raise RuntimeError('Archived epoch weights differ from actual VAL source/state/score')
    _gates_valid(checkpoint['state'],cell)
    return checkpoint


def validation_at_epoch(model,examples,values,ident,binding,cell,seed,epoch):
    c=_common();weights=state_digest(model.state_dict())
    path=c.OUT/'validation_predictions'/f'{cell}-seed{seed}-epoch{epoch}.npz'
    metadata=validation_metadata(ident,binding,cell,seed,epoch,weights)
    if path.exists():score,nll,rows,mean,std=read_validation_archive(path,metadata,examples)
    else:
        score,nll,rows,mean,std=validation_score(model,examples,values,return_predictions=True)
        stream=io.BytesIO();np.savez_compressed(stream,mean=mean,std=std,metadata_utf8=np.frombuffer(_canonical(metadata),dtype=np.uint8))
        _publish_bytes(path,stream.getvalue(),staging=Path(c.WORK)/'staging')
        actual=read_validation_archive(path,metadata,examples)
        if actual[:3]!=(score,nll,rows):raise RuntimeError('Raw validation archive does not reproduce actual scores')
    reference={'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path)}
    epoch_path=c.OUT/'epoch_checkpoints'/f'{cell}-seed{seed}-epoch{epoch}.pt'
    payload={'identity':ident,'dataset_binding_sha256':binding,'cell':cell,'seed':seed,
        'settings':dict(model.settings),'epoch':epoch,'committed_updates':epoch*12,
        'validation_rmse':score,'validation_nll':nll,'state':_clone_state(model)}
    if epoch_path.exists():_epoch_valid(torch.load(epoch_path,map_location='cpu',weights_only=True),ident,binding,cell,seed,epoch,score,nll,weights)
    else:_publish_torch(epoch_path,payload,staging=Path(c.WORK)/'staging')
    epoch_ref={'path':str(epoch_path.relative_to(c.ROOT)),'sha256':c.digest(epoch_path)}
    return score,nll,rows,reference,weights,epoch_ref


def verify_validation_predictions(receipt,examples=None):
    c=_common()
    if examples is None:
        from .data import training_examples
        examples=training_examples('validation','sparse')
    validate_fitting_examples(examples,'validation','sparse')
    binding=receipt['dataset_binding_sha256']
    if hashlib.sha256(_canonical([e['binding'] for e in examples])).hexdigest()!=binding['validation_sparse']:
        raise RuntimeError('Validation inputs/targets differ from this fitted member')
    refs=[]
    for row in receipt['history']:
        epoch=row['epoch']
        if epoch not in VALIDATION_EPOCHS:continue
        path=c.OUT/'validation_predictions'/f"{receipt['cell']}-seed{receipt['seed']}-epoch{epoch}.npz"
        reference={'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path)}
        if row['validation_predictions']!=reference:raise RuntimeError('Raw validation archive path/bytes changed')
        metadata=validation_metadata(receipt['identity'],binding,receipt['cell'],receipt['seed'],epoch,row['validation_state_sha256'])
        score,nll,rows,_,_=read_validation_archive(path,metadata,examples)
        if (score!=row['validation_rmse'] or nll!=row['validation_nll'] or rows!=row['validation_per_layout']):
            raise RuntimeError('Validation scores differ from actual archived predictions')
        weights=c.OUT/'epoch_checkpoints'/f"{receipt['cell']}-seed{receipt['seed']}-epoch{epoch}.pt"
        if row['epoch_checkpoint']!={'path':str(weights.relative_to(c.ROOT)),'sha256':c.digest(weights)}:
            raise RuntimeError('Archived epoch checkpoint path/bytes changed')
        _epoch_valid(torch.load(weights,map_location='cpu',weights_only=True),receipt['identity'],binding,
            receipt['cell'],receipt['seed'],epoch,score,nll,row['validation_state_sha256'])
        refs.append(reference)
    if len(refs)!=4:raise RuntimeError('All four fixed validation candidates required')
    return refs


def _payload(model,optimizer,rng,ident,binding,cell,seed,trace,history,best_state,elapsed):
    epoch,score=selected_validation(history)
    return {'schema':1,'identity':ident,'dataset_binding_sha256':binding,'cell':cell,'seed':seed,
        'run_id':_member(cell,seed),'settings':dict(model.settings),'optimizer_spec':OPTIMIZER_SPEC,
        'committed_updates':len(trace),'state':_clone_state(model),'optimizer':optimizer.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,
        'update_trace':trace,'history':history,'best_epoch':epoch,'best_score':score,
        'best_state':best_state,'elapsed_seconds':elapsed}


def validate_resume(prior,ident,binding,cell,seed,full=None,sparse=None):
    if (not isinstance(prior,dict) or prior.get('identity')!=ident or prior.get('dataset_binding_sha256')!=binding
            or prior.get('cell')!=cell or type(prior.get('seed')) is not int or prior['seed']!=seed
            or prior.get('run_id')!=_member(cell,seed) or prior.get('settings')!=settings(cell)
            or prior.get('optimizer_spec')!=OPTIMIZER_SPEC or type(prior.get('committed_updates')) is not int
            or prior['committed_updates']!=len(prior.get('update_trace',()))):
        raise RuntimeError('Resume is not this exact source/data/member/budget trajectory')
    validate_history(prior['history'],prior['update_trace'],cell=cell)
    if prior.get('numpy_rng')!=verify_schedule(prior['update_trace'],seed,full,sparse):raise RuntimeError('Committed minibatch RNG changed')
    epoch,score=selected_validation(prior['history'])
    if (prior.get('best_epoch')!=epoch or prior.get('best_score')!=score or (epoch is None)!=(prior.get('best_state') is None)
            or not _finite_tree(prior['state']) or not _finite_tree(prior['optimizer'])
            or prior.get('best_state') is not None and not _finite_tree(prior['best_state'])
            or not math.isfinite(prior.get('elapsed_seconds',math.nan)) or prior['elapsed_seconds']<0):
        raise RuntimeError('Resume selection/finite state changed')
    _gates_valid(prior['state'],cell)
    if prior['best_state'] is not None:
        _gates_valid(prior['best_state'],cell)
        selected=next(row for row in prior['history'] if row['epoch']==epoch)
        if state_digest(prior['best_state'])!=selected['validation_state_sha256']:
            raise RuntimeError('Resume best weights differ from the actual selected validation state')
    groups=prior['optimizer']['param_groups']
    if (len(groups)!=1 or groups[0]['lr']!=.003 or groups[0]['weight_decay']!=1e-4
            or tuple(groups[0]['betas'])!=(.9,.999) or groups[0]['eps']!=1e-8
            or not isinstance(prior.get('torch_rng'),torch.Tensor) or prior['torch_rng'].dtype!=torch.uint8):
        raise RuntimeError('Resume optimizer or Torch RNG changed')
    return prior


def _checkpoint_valid(checkpoint,ident,binding,cell,seed,epoch,score,nll,weights):
    if (checkpoint.get('identity')!=ident or checkpoint.get('dataset_binding_sha256')!=binding
            or checkpoint.get('cell')!=cell or type(checkpoint.get('seed')) is not int or checkpoint['seed']!=seed
            or checkpoint.get('settings')!=settings(cell) or checkpoint.get('selected_epoch')!=epoch
            or checkpoint.get('selected_updates')!=epoch*12 or checkpoint.get('validation_rmse')!=score
            or checkpoint.get('validation_nll')!=nll or checkpoint.get('committed_updates')!=COMMITTED_UPDATES
            or not isinstance(checkpoint.get('state'),dict) or not checkpoint['state']
            or not all(isinstance(v,torch.Tensor) for v in checkpoint['state'].values())
            or not _finite_tree(checkpoint['state']) or state_digest(checkpoint['state'])!=weights):
        raise RuntimeError('Selected inert checkpoint identity/score/weights changed')
    _gates_valid(checkpoint['state'],cell)
    return checkpoint


def verify_completed(receipt,ident,cell,seed,*,validation_examples=None,fitting_examples=None):
    name=_member(cell,seed);c=_common();path=c.OUT/'checkpoints'/f'{name}.pt'
    if (c.OUT/'failures'/f'{name}.json').exists():raise RuntimeError('Terminally failed member cannot be loaded as complete')
    if (receipt.get('complete') is not True or receipt.get('completed') is not True or receipt.get('identity')!=ident
            or receipt.get('cell')!=cell or type(receipt.get('seed')) is not int or receipt['seed']!=seed
            or receipt.get('run_id')!=name or receipt.get('epochs')!=EPOCHS
            or receipt.get('updates')!=COMMITTED_UPDATES or receipt.get('committed_updates')!=COMMITTED_UPDATES
            or receipt.get('settings')!=settings(cell) or receipt.get('optimizer_spec')!=json.loads(json.dumps(OPTIMIZER_SPEC))
            or receipt.get('checkpoint_sha256')!=c.digest(path)):
        raise RuntimeError('Completed fit differs from registered member/budget/checkpoint')
    if fitting_examples is None:full,sparse,validation=datasets()
    else:
        if not isinstance(fitting_examples,(tuple,list)) or len(fitting_examples)!=3:
            raise RuntimeError('Explicit fitting data must contain TRAINfull/sparse and VALsparse')
        full,sparse,validation=fitting_examples
        validate_fitting_examples(full,'training','full');validate_fitting_examples(sparse,'training','sparse')
        validate_fitting_examples(validation,'validation','sparse')
        if any((a['binding']['case_id'],a['binding']['layout_id'])!=(b['binding']['case_id'],b['binding']['layout_id']) for a,b in zip(full,sparse)):
            raise RuntimeError('Explicit full/sparse training cases differ')
    if validation_examples is not None:
        validate_fitting_examples(validation_examples,'validation','sparse');validation=validation_examples
    binding=data_identity(full,sparse,validation)
    if binding!=receipt['dataset_binding_sha256']:raise RuntimeError('Completed fit TRAIN/VAL bindings changed')
    validate_history(receipt['history'],receipt['update_trace'],cell=cell,complete=True)
    verify_schedule(receipt['update_trace'],seed,full,sparse)
    epoch,score=selected_validation(receipt['history'])
    selected=next(row for row in receipt['history'] if row['epoch']==epoch)
    if (receipt.get('selected_epoch')!=epoch or receipt.get('selected_updates')!=epoch*12
            or receipt.get('validation_rmse')!=score or receipt.get('validation_nll')!=selected['validation_nll']):
        raise RuntimeError('Completed fit did not choose first sparse heard-RMSE minimum')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    _checkpoint_valid(checkpoint,ident,binding,cell,seed,epoch,score,selected['validation_nll'],selected['validation_state_sha256'])
    verify_validation_predictions(receipt,validation)
    attempt=c.OUT/'attempts'/name/'start.json'
    if receipt.get('initial_attempt_sha256')!=c.digest(attempt):raise RuntimeError('Initial fit attempt bytes changed')
    start=c.read(attempt)
    if (start.get('identity')!=ident or start.get('dataset_binding_sha256')!=binding or start.get('run_id')!=name
            or start.get('cell')!=cell or type(start.get('seed')) is not int or start['seed']!=seed
            or start.get('maximum_committed_updates')!=COMMITTED_UPDATES
            or start.get('selection_epochs')!=list(VALIDATION_EPOCHS) or start.get('before_first_optimizer_step') is not True):
        raise RuntimeError('Initial fit attempt identity changed')
    return checkpoint


def fit(cell,seed):
    name=_member(cell,seed);c=_common();ident=identity()
    model_class,objective=_model_api()
    work=Path(c.WORK)/f'{name}.last.pt';receipt_path=c.OUT/'fits'/f'{name}.json'
    failure=c.OUT/'failures'/f'{name}.json';attempt=c.OUT/'attempts'/name/'start.json'
    with exclusive_fit(Path(c.WORK)/f'{name}.lock'):
        if failure.exists():raise RuntimeError('Terminal fit failure recorded; no automatic retry/new trial')
        if receipt_path.exists():
            receipt=c.read(receipt_path);verify_completed(receipt,ident,cell,seed);return receipt
        if attempt.exists() and not work.exists():raise RuntimeError('Interrupted attempt has no committed snapshot; replacement refused')
        if work.exists() and not attempt.exists():raise RuntimeError('Unowned resume snapshot cannot create a trial')
        full,sparse,validation=datasets();binding=data_identity(full,sparse,validation)
        if not attempt.exists():_publish_json(c,attempt,{'schema':1,'run_id':name,'cell':cell,'seed':seed,
            'identity':ident,'dataset_binding_sha256':binding,'created_unix_ns':time.time_ns(),
            'maximum_committed_updates':COMMITTED_UPDATES,'selection_epochs':list(VALIDATION_EPOCHS),'before_first_optimizer_step':True})
        start=c.read(attempt)
        if (start.get('identity')!=ident or start.get('dataset_binding_sha256')!=binding or start.get('run_id')!=name
                or start.get('cell')!=cell or type(start.get('seed')) is not int or start['seed']!=seed
                or start.get('maximum_committed_updates')!=COMMITTED_UPDATES or start.get('selection_epochs')!=list(VALIDATION_EPOCHS)
                or start.get('before_first_optimizer_step') is not True):raise RuntimeError('Fit attempt identity changed')
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
        rng=np.random.default_rng(seed);model=model_class(**settings(cell))
        if model.settings!=settings(cell):raise RuntimeError('Model public settings changed')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
        trace=[];history=[];best_state=None;elapsed=0.;resumed=False
        if work.exists():
            old=validate_resume(torch.load(work,map_location='cpu',weights_only=True),ident,binding,cell,seed,full,sparse)
            model.load_state_dict(old['state'],strict=True);optimizer.load_state_dict(old['optimizer'])
            torch.set_rng_state(old['torch_rng']);rng.bit_generator.state=old['numpy_rng']
            trace,history,best_state,elapsed=(old[k] for k in ('update_trace','history','best_state','elapsed_seconds'))
            count=len(list(attempt.parent.glob('resume-*.json')))+1
            _publish_json(c,attempt.parent/f'resume-{count:04d}.json',{'identity':ident,'run_id':name,
                'committed_updates':len(trace),'work_snapshot_sha256':c.digest(work),'created_unix_ns':time.time_ns(),
                'same_rng_and_optimizer_trajectory':True,'new_seed_or_trial':False,'uncommitted_step_may_be_recomputed':True})
            resumed=True
        train={'full':batch(full),'sparse':batch(sparse)};val=batch(validation);began=time.perf_counter()
        def commit():
            atomic_save(work,_payload(model,optimizer,rng,ident,binding,cell,seed,trace,history,best_state,elapsed+time.perf_counter()-began))
        if not resumed:commit()
        try:
            while True:
                completed=len(trace)
                if completed and completed%12==0 and len(history)<completed//12:
                    epoch=completed//12;score=nll=rows=ref=weights=epoch_ref=None
                    if epoch in VALIDATION_EPOCHS:
                        score,nll,rows,ref,weights,epoch_ref=validation_at_epoch(model,validation,val,ident,binding,cell,seed,epoch)
                    _,previous=selected_validation(history)
                    if score is not None and (previous is None or score<previous):best_state=_clone_state(model)
                    history.append({'epoch':epoch,'completed_updates':completed,
                        'mean_training_loss':float(np.mean([r['loss'] for r in trace[-12:]])),
                        'validation_rmse':score,'validation_nll':nll,'validation_per_layout':rows,
                        'validation_predictions':ref,'validation_state_sha256':weights,'epoch_checkpoint':epoch_ref})
                    commit()
                    if epoch%20==0:print(f'TRAIN {name} {completed}/{COMMITTED_UPDATES}',flush=True)
                if completed==COMMITTED_UPDATES:break
                indices,layouts,conditions,regime=choose_minibatch(rng,completed)
                v=select(train[regime],indices);model.train();optimizer.zero_grad(set_to_none=True)
                loss,details=objective(model,v['a'],v['b'],v['query'],v['target'],v['ap_xyz'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fixed training objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();model.project_gates_()
                if not _finite_tree(model.state_dict()) or not _finite_tree(optimizer.state_dict()):raise FloatingPointError('Nonfinite optimizer/model state')
                _gates_valid(model.state_dict(),cell)
                row={key:float(value) for key,value in details.items()}
                row.update(step=completed+1,epoch=completed//12+1,update_in_epoch=completed%12+1,
                    regime=regime,layout_indices=layouts.astype(int).tolist(),conditions=conditions.astype(int).tolist(),
                    case_ids=[(full if regime=='full' else sparse)[int(j)]['binding']['case_id'] for j in indices],gradient_norm_before_clip=float(norm))
                if not _finite_tree(row):raise FloatingPointError('Nonfinite committed trace')
                trace.append(row);commit()
            if identity()!=ident:raise RuntimeError('Source/fitting data changed during training')
            validate_history(history,trace,cell=cell,complete=True);epoch,score=selected_validation(history)
            selected=next(row for row in history if row['epoch']==epoch)
            if best_state is None:raise RuntimeError('No registered validation checkpoint selected')
            checkpoint={'identity':ident,'dataset_binding_sha256':binding,'cell':cell,'seed':seed,'settings':dict(model.settings),
                'state':best_state,'selected_epoch':epoch,'selected_updates':epoch*12,'validation_rmse':score,
                'validation_nll':selected['validation_nll'],'committed_updates':COMMITTED_UPDATES}
            path=c.OUT/'checkpoints'/f'{name}.pt'
            if path.exists():_checkpoint_valid(torch.load(path,map_location='cpu',weights_only=True),ident,binding,cell,seed,epoch,score,selected['validation_nll'],state_digest(best_state))
            else:_publish_torch(path,checkpoint,staging=Path(c.WORK)/'staging')
            receipt={'schema':1,'complete':True,'completed':True,'run_id':name,'cell':cell,'seed':seed,'identity':ident,
                'dataset_binding_sha256':binding,'settings':dict(model.settings),'optimizer_spec':OPTIMIZER_SPEC,
                'epochs':EPOCHS,'updates':COMMITTED_UPDATES,'committed_updates':COMMITTED_UPDATES,'update_trace':trace,'history':history,
                'selected_epoch':epoch,'selected_updates':epoch*12,'validation_rmse':score,'validation_nll':selected['validation_nll'],
                'selection':'FIRST minimum at epochs40/80/120/160; sparseVAL A-only equal-layout/state heard-RSSI RMSE; NLL not selection',
                'checkpoint_sha256':c.digest(path),'elapsed_seconds':elapsed+time.perf_counter()-began,
                'learned_parameters':sum(p.numel() for p in model.parameters()),'batch_distinct_layouts':4,
                'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu','torch_threads':torch.get_num_threads(),
                    'interop_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
                'training_layouts':list(validate_examples(full,'training','full')),'training_bands':[0,1],
                'resume_events':[{'path':str(p.relative_to(c.ROOT)),'sha256':c.digest(p)} for p in sorted(attempt.parent.glob('resume-*.json'))],
                'initial_attempt_sha256':c.digest(attempt),'uncertainty_calibrated':False,'packet_or_production_qualified':False}
            _publish_json(c,receipt_path,receipt);verify_completed(c.read(receipt_path),ident,cell,seed,validation_examples=validation)
            return c.read(receipt_path)
        except KeyboardInterrupt:
            count=len(list(attempt.parent.glob('interrupt-*.json')))+1
            _publish_json(c,attempt.parent/f'interrupt-{count:04d}.json',{'identity':ident,'run_id':name,'created_unix_ns':time.time_ns(),
                'last_snapshot_sha256':c.digest(work),'reason':'KeyboardInterrupt; exact last committed snapshot remains resumable'})
            raise
        except OSError:
            raise  # Atomic snapshots remain resumable; no new seed/trial.
        except Exception as exc:
            _publish_json(c,failure,{'schema':1,'run_id':name,'cell':cell,'seed':seed,'identity':ident,
                'created_unix_ns':time.time_ns(),'error_type':type(exc).__name__,'error':str(exc),
                'committed_snapshot_sha256':c.digest(work),'automatic_retry_permitted':False})
            raise


def load(cell,seed):
    name=_member(cell,seed);c=_common();ident=identity();receipt=c.read(c.OUT/'fits'/f'{name}.json')
    checkpoint=verify_completed(receipt,ident,cell,seed);model_class,_=_model_api()
    with torch.random.fork_rng(devices=[]):model=model_class(**checkpoint['settings'])
    model.load_state_dict(checkpoint['state'],strict=True);model.eval();return model


@torch.no_grad()
def predict(model,example,view='a'):
    if view not in ('a','b'):raise ValueError('Only actual A/B context views are supported')
    v=batch([example]);mean,std=model(v[view],v['query'],v['ap_xyz'])
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():raise FloatingPointError('Invalid field predictions')
    return mean[0].detach().cpu().numpy().copy(),std[0].detach().cpu().numpy().copy()


def _fit_task(task):return fit(*task)


def training_worker_context():return mp.get_context('spawn')


def train_all(workers=3):
    if type(workers) is not int or workers not in (1,2,3):raise ValueError('Only one to three CPU workers are registered')
    tasks=[(cell,seed) for cell in CELLS for seed in SEEDS]
    if workers==1:return [_fit_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers,mp_context=training_worker_context()) as pool:return list(pool.map(_fit_task,tasks))
