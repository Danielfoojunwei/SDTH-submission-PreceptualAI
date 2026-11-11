"""Fixed-budget hierarchical field training with committed-step recovery.

Scientific fitting is guarded by the parent study's published source and
prepared-data identity. An interrupted uncommitted optimizer step may be
recomputed from its last atomic snapshot; it is never a new seed, trial or
selection opportunity. Mathematical/nonfinite failures are terminal.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .model import HierarchicalField, CELLS, objective, observation_nll

SEEDS=(0,1,2)
EPOCHS=40
UPDATES_PER_EPOCH=12
COMMITTED_UPDATES=EPOCHS*UPDATES_PER_EPOCH
VALIDATION_EPOCHS=(10,20,30,40)  # Human, one-based epoch numbers.
BATCH_LAYOUTS=4
TRAIN_LAYOUTS=24
CONDITIONS=3
OPTIMIZER_SPEC={'name':'AdamW','learning_rate':.003,'weight_decay':1.e-4,
                'betas':(.9,.999),'eps':1.e-8,'gradient_clip_norm':5.}
VIEW_KEYS={'context','known','scheduled','ap','rx_xyz','lower','upper','floor','fspl','offset','weights'}
QUERY_KEYS={'q','ap','rx_xyz','basis','fspl','offset'}
TARGET_KEYS={'center_dbm','lower_dbm','upper_dbm','heard','censored','censor_upper_dbm','detection_threshold_dbm','weights'}
MASK_KEYS={'known','scheduled','heard','censored'}


def _common():
    from . import common
    return common


def _member(cell,seed):
    if cell not in CELLS or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only the five registered cells and training seeds0/1/2 are supported')
    return f'{cell}-seed{seed}'


def _canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def tensor(value,name='numeric'):
    """Own a tensor copy, preserving AP indices, masks and interval infinities."""
    a=np.asarray(value)
    if a.dtype.kind not in 'buif':raise ValueError('Only real numeric/bool model arrays are supported')
    if name in MASK_KEYS:
        if a.dtype!=np.dtype(bool):raise ValueError('Observation masks must already be boolean')
        b=np.array(a,dtype=bool,copy=True)
    elif name=='ap':
        if a.dtype.kind not in 'iu':raise ValueError('Physical AP routes must already be integer indices')
        if a.size and (np.min(a)<0 or np.max(a)>np.iinfo(np.int64).max):
            raise ValueError('Physical AP routes must fit nonnegative int64')
        b=np.array(a,dtype='i8',copy=True)
    else:
        with np.errstate(over='ignore',invalid='ignore'):b=np.array(a,dtype='f4',copy=True)
        if np.any(np.isfinite(a)&~np.isfinite(b)):
            raise ValueError('A finite physical value overflowed float32')
    return torch.from_numpy(b)


def batch(examples):
    """Stack one regime only; bindings/IDs never become learned input features."""
    if not isinstance(examples,(tuple,list)) or not examples:raise ValueError('A nonempty single-regime example list is required')
    result={}
    for role,allowed in (('a',VIEW_KEYS),('b',VIEW_KEYS),('query',QUERY_KEYS),('target',TARGET_KEYS)):
        first=examples[0].get(role)
        if not isinstance(first,dict) or not first or set(first)-allowed:raise ValueError('Unknown or missing model '+role+' fields')
        keys=set(first)
        if any(not isinstance(e.get(role),dict) or set(e[role])!=keys for e in examples):
            raise ValueError('Batched '+role+' schemas must match')
        result[role]={key:tensor(np.stack([e[role][key] for e in examples]),key) for key in sorted(keys)}
    result['ap_xyz']=tensor(np.stack([e['ap_xyz'] for e in examples]),'ap_xyz')
    return result


def select(values,indices):
    index=torch.as_tensor(np.array(indices,dtype='i8',copy=True))
    return {name:({key:value.index_select(0,index) for key,value in group.items()}
                  if isinstance(group,dict) else group.index_select(0,index)) for name,group in values.items()}


def choose_minibatch(rng,committed_updates):
    if type(committed_updates) is not int or not 0<=committed_updates<COMMITTED_UPDATES:
        raise ValueError('Fixed committed optimizer-step index required')
    layouts=rng.choice(TRAIN_LAYOUTS,size=BATCH_LAYOUTS,replace=False)
    conditions=rng.integers(0,CONDITIONS,size=BATCH_LAYOUTS)
    return layouts*CONDITIONS+conditions,layouts,conditions,'full' if committed_updates%2==0 else 'sparse'


def validate_examples(examples,stage,regime):
    layouts={'training':24,'validation':6}.get(stage)
    if layouts is None or regime not in ('full','sparse') or len(examples)!=layouts*3:
        raise RuntimeError('Expected registered TRAIN24/VAL6 layouts, each with three RF states')
    names=[]
    for index,e in enumerate(examples):
        b=e.get('binding',{})
        if (b.get('stage')!=stage or b.get('index')!=index//3 or b.get('state')!=index%3
                or b.get('condition')!=index%3 or b.get('regime')!=regime
                or b.get('held_band') is not False or b.get('stale_requested') is not False
                or b.get('case_id')!=b.get('context_case_id')
                or not isinstance(b.get('layout_id'),str) or not b['layout_id']
                or not isinstance(b.get('case_id'),str) or not b['case_id']):
            raise RuntimeError('Only layout-major current-state TRAIN/VAL examples may enter fitting')
        if index%3==0:names.append(b['layout_id'])
        if b['layout_id']!=names[-1]:raise RuntimeError('All three conditions must share one layout identity')
    if len(set(names))!=layouts:raise RuntimeError('Training/validation layouts must be distinct whole geometries')
    return tuple(names)


def identity():
    c=_common();c.check_sources()
    path=c.OUT/'data/prepared.json'
    if not path.is_file() or path.is_symlink() or not isinstance(c.read(path),dict):
        raise RuntimeError('Prepare the exact registered TRAIN/VAL data before training')
    from . import study
    study.verify_prepared()  # Passive canonical references; no examples/selection recursion.
    return {'source_receipt_sha256':c.source_receipt_sha256(),
            'protocol_sha256':c.protocol_sha256(),'data_receipt_sha256':c.digest(path)}


def _data_identity(full,sparse,validation):
    result={}
    for name,examples in (('training_full',full),('training_sparse',sparse),('validation_sparse',validation)):
        result[name]=hashlib.sha256(_canonical([e['binding'] for e in examples])).hexdigest()
    return result


def atomic_save(path,value):
    """A crash leaves either the prior complete snapshot or the new one."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.committed-',suffix='.pt',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def _publish_bytes(path,data,*,staging=None):
    """Publish complete immutable bytes once; staging remains outside OUT."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    staging=path.parent if staging is None else Path(staging)
    staging.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.publication-',dir=staging)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
        descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def _publish_torch(path,value,*,staging=None):
    stream=io.BytesIO();torch.save(value,stream)
    _publish_bytes(path,stream.getvalue(),staging=staging)


def _publish_json(c,path,value):
    _publish_bytes(path,_canonical(value),staging=Path(c.WORK)/'staging')


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if not isinstance(value,torch.Tensor):raise RuntimeError('Only inert model tensors form a weight identity')
        a=np.ascontiguousarray(value.detach().cpu().numpy())
        h.update(name.encode());h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


@contextmanager
def exclusive_fit(path):
    """A kernel file lock releases on process loss; no stale-lock guessing."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        try:fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another process already owns this training member') from exc
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _finite_tree(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,(tuple,list)):return all(_finite_tree(x) for x in value)
    if isinstance(value,dict):return all(_finite_tree(x) for x in value.values())
    if type(value) is float:return math.isfinite(value)
    return True


def _clone_state(model):
    result={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    if not result or not _finite_tree(result):raise FloatingPointError('Nonfinite model weights or buffers')
    return result


def validate_trace(trace,*,complete=False):
    if not isinstance(trace,list) or len(trace)>COMMITTED_UPDATES or (complete and len(trace)!=COMMITTED_UPDATES):
        raise RuntimeError('Fixed480 committed optimizer-step budget changed')
    for index,row in enumerate(trace):
        if (row.get('step')!=index+1 or row.get('epoch')!=index//12+1 or row.get('update_in_epoch')!=index%12+1
                or row.get('regime')!=('full' if index%2==0 else 'sparse')
                or not isinstance(row.get('layout_indices'),list) or len(row['layout_indices'])!=4
                or len(set(row['layout_indices']))!=4 or any(type(x) is not int or not 0<=x<24 for x in row['layout_indices'])
                or len(row.get('conditions',()))!=4 or any(type(x) is not int or x not in (0,1,2) for x in row['conditions'])
                or any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) for k in
                       ('loss','measurement_nll','alignment','alignment_weight','supported_agreement_queries','mean_predictive_std_db','gradient_norm_before_clip'))):
            raise RuntimeError('Malformed/nonfinite committed optimization trace')
        if row['gradient_norm_before_clip']<0 or not any(math.isclose(row['alignment_weight'],x,abs_tol=1e-7,rel_tol=0) for x in (0.,.1)):
            raise RuntimeError('Gradient or fixed agreement weight changed')
        if not math.isclose(row['loss'],row['measurement_nll']+row['alignment_weight']*row['alignment'],abs_tol=1e-7,rel_tol=1e-7):
            raise RuntimeError('Trace objective is not the fixed reported likelihood/alignment loss')


def verify_schedule(trace,seed):
    rng=np.random.default_rng(seed)
    for index,row in enumerate(trace):
        _,layouts,conditions,regime=choose_minibatch(rng,index)
        if (row['layout_indices']!=layouts.astype(int).tolist() or row['conditions']!=conditions.astype(int).tolist()
                or row['regime']!=regime):
            raise RuntimeError('Committed minibatches do not follow this exact fixed-seed sampling trajectory')
    return rng.bit_generator.state


def validate_history(history,trace,*,complete=False):
    validate_trace(trace,complete=complete)
    maximum=len(trace)//12
    allowed={maximum}
    if len(trace) and len(trace)%12==0:allowed.add(maximum-1)
    if (not isinstance(history,list) or len(history) not in allowed or (complete and len(history)!=40)):
        raise RuntimeError('Epoch ledger differs from the committed optimizer steps')
    for i,row in enumerate(history):
        epoch=i+1
        if row.get('epoch')!=epoch or row.get('completed_updates')!=epoch*12:
            raise RuntimeError('Epoch history ordering changed')
        expected=float(np.mean([x['loss'] for x in trace[i*12:(i+1)*12]]))
        if row.get('mean_training_loss')!=expected:raise RuntimeError('Epoch mean does not reduce its12 committed updates')
        if epoch in VALIDATION_EPOCHS:
            values=row.get('validation_per_layout_nll')
            if (not isinstance(values,list) or len(values)!=6 or
                    any(not isinstance(x,dict) or not math.isfinite(x.get('nll',math.nan)) for x in values)
                    or len({x.get('layout_id') for x in values})!=6
                    or row.get('validation_nll')!=float(np.mean([x['nll'] for x in values]))
                    or not isinstance(row.get('validation_predictions'),dict)
                    or set(row['validation_predictions'])!={'path','sha256'}
                    or not isinstance(row.get('validation_state_sha256'),str) or len(row['validation_state_sha256'])!=64):
                raise RuntimeError('Sparse A-only six-layout validation ledger changed')
        elif any(row.get(k) is not None for k in ('validation_nll','validation_per_layout_nll','validation_predictions','validation_state_sha256')):
            raise RuntimeError('Unregistered validation checkpoint')


def selected_validation(history):
    candidates=[row for row in history if row['epoch'] in VALIDATION_EPOCHS]
    if not candidates:return None,None
    # min preserves encounter order, so exact ties retain the FIRST epoch.
    chosen=min(candidates,key=lambda row:row['validation_nll'])
    return chosen['epoch'],chosen['validation_nll']


@torch.no_grad()
def validation_score(model,examples,values=None,*,details=False,return_predictions=False):
    validate_examples(examples,'validation','sparse')
    values=batch(examples) if values is None else values
    previous=model.training;model.eval();rows=[];means=[];stds=[]
    try:
        for index in range(6):
            v=select(values,np.arange(index*3,(index+1)*3))
            mean,std=model(v['a'],v['query'],v['ap_xyz'])
            means.append(mean.detach().cpu().numpy().copy());stds.append(std.detach().cpu().numpy().copy())
            score=float(observation_nll(mean,std,v['target']))
            if not math.isfinite(score):raise FloatingPointError('Nonfinite sparse A-only interval validation score')
            rows.append({'layout_id':examples[index*3]['binding']['layout_id'],'conditions':[0,1,2],'nll':score})
    finally:model.train(previous)
    score=float(np.mean([row['nll'] for row in rows]))
    if return_predictions:return score,rows,np.concatenate(means),np.concatenate(stds)
    return (score,rows) if details else score


def reduce_validation_predictions(mean,std,examples):
    validate_examples(examples,'validation','sparse')
    values=batch(examples);expected=values['target']['heard'].shape
    if (mean.shape!=tuple(expected) or std.shape!=mean.shape or mean.dtype.kind!='f'
            or std.dtype.kind!='f' or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0)):
        raise RuntimeError('Actual raw sparseVAL arrays must be finite positive-spread18xQ')
    means=torch.from_numpy(np.array(mean,copy=True));stds=torch.from_numpy(np.array(std,copy=True))
    rows=[]
    with torch.no_grad():
        for i in range(6):
            target={k:v[i*3:(i+1)*3] for k,v in values['target'].items()}
            score=float(observation_nll(means[i*3:(i+1)*3],stds[i*3:(i+1)*3],target))
            if not math.isfinite(score):raise FloatingPointError('Raw validation score is nonfinite')
            rows.append({'layout_id':examples[i*3]['binding']['layout_id'],'conditions':[0,1,2],'nll':score})
    return float(np.mean([r['nll'] for r in rows])),rows


def validation_metadata(ident,data_identity,cell,seed,epoch,weight_sha256):
    return {'schema':1,'identity':ident,'dataset_binding_sha256':data_identity,
        'cell':cell,'seed':seed,'epoch':epoch,'committed_updates':epoch*12,
        'model_state_sha256':weight_sha256,'source_view':'a','regime':'sparse',
        'target_binding_sha256':data_identity['validation_sparse'],
        'endpoint':'equal_layout_equal_condition_composite_interval_NLL'}


def read_validation_archive(path,metadata,examples):
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'mean','std','metadata_utf8'}:raise RuntimeError('Validation archive exact array membership changed')
        encoded=archive['metadata_utf8']
        if encoded.dtype!=np.uint8 or encoded.ndim!=1:raise RuntimeError('Validation metadata must be inert UTF8 bytes')
        observed=json.loads(encoded.tobytes())
        if observed!=metadata:raise RuntimeError('Validation archive is not this exact source/data/member/epoch/weight state')
        mean,std=archive['mean'].copy(),archive['std'].copy()
    score,rows=reduce_validation_predictions(mean,std,examples)
    return score,rows,mean,std


def validation_at_epoch(model,examples,values,ident,data_identity,cell,seed,epoch):
    c=_common();path=c.OUT/'validation_predictions'/f'{cell}-seed{seed}-epoch{epoch}.npz'
    metadata=validation_metadata(ident,data_identity,cell,seed,epoch,state_digest(model.state_dict()))
    if path.exists():score,rows,mean,std=read_validation_archive(path,metadata,examples)
    else:
        score,rows,mean,std=validation_score(model,examples,values,return_predictions=True)
        stream=io.BytesIO();np.savez_compressed(stream,mean=mean,std=std,
            metadata_utf8=np.frombuffer(_canonical(metadata),dtype=np.uint8))
        _publish_bytes(path,stream.getvalue(),staging=Path(c.WORK)/'staging')
        # The same raw-array reducer, without a second forward, must reproduce
        # the actual validation score before selection can consume it.
        actual,actual_rows,_,_=read_validation_archive(path,metadata,examples)
        if actual!=score or actual_rows!=rows:raise RuntimeError('Raw VAL predictions do not reduce the actual validation score')
    return score,rows,{'path':str(path.relative_to(c.ROOT)),'sha256':c.digest(path)},metadata['model_state_sha256']


def _state_payload(model,optimizer,rng,ident,data_identity,cell,seed,trace,history,best_state,elapsed):
    epoch,best=selected_validation(history)
    return {'schema':1,'identity':ident,'dataset_binding_sha256':data_identity,
        'cell':cell,'seed':seed,'run_id':_member(cell,seed),'settings':dict(model.config),
        'optimizer_spec':OPTIMIZER_SPEC,'committed_updates':len(trace),
        'state':_clone_state(model),'optimizer':optimizer.state_dict(),
        'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,
        'update_trace':trace,'history':history,'best_epoch':epoch,'best_score':best,
        'best_state':best_state,'elapsed_seconds':elapsed}


def validate_resume(prior,ident,data_identity,cell,seed):
    if (not isinstance(prior,dict) or prior.get('identity')!=ident or prior.get('dataset_binding_sha256')!=data_identity
            or prior.get('cell')!=cell or type(prior.get('seed')) is not int or prior['seed']!=seed
            or prior.get('run_id')!=_member(cell,seed) or prior.get('settings')!={'cell':cell,'width':32,'inference_steps':3}
            or prior.get('optimizer_spec')!=OPTIMIZER_SPEC or type(prior.get('committed_updates')) is not int
            or prior['committed_updates']!=len(prior.get('update_trace',()))):
        raise RuntimeError('Resume is not this exact source/data/member/optimizer trajectory')
    validate_history(prior['history'],prior['update_trace'])
    if prior.get('numpy_rng')!=verify_schedule(prior['update_trace'],seed):
        raise RuntimeError('Resume RNG state disagrees with the committed fixed-seed minibatches')
    epoch,score=selected_validation(prior['history'])
    if prior.get('best_epoch')!=epoch or prior.get('best_score')!=score or (epoch is None)!=(prior.get('best_state') is None):
        raise RuntimeError('Resume first-minimum validation selection changed')
    if (not _finite_tree(prior['state']) or not _finite_tree(prior['optimizer'])
            or prior.get('best_state') is not None and not _finite_tree(prior['best_state'])
            or not math.isfinite(prior.get('elapsed_seconds',math.nan)) or prior['elapsed_seconds']<0):
        raise RuntimeError('Nonfinite optimizer/weight state cannot resume')
    groups=prior['optimizer']['param_groups']
    if len(groups)!=1 or groups[0]['lr']!=.003 or groups[0]['weight_decay']!=1.e-4 or tuple(groups[0]['betas'])!=(.9,.999) or groups[0]['eps']!=1.e-8:
        raise RuntimeError('Optimizer hyperparameters changed during resume')
    if not isinstance(prior.get('torch_rng'),torch.Tensor) or prior['torch_rng'].dtype!=torch.uint8 or not isinstance(prior.get('numpy_rng'),dict):
        raise RuntimeError('Both exact random-generator states are required to resume')
    return prior


def _checkpoint_valid(checkpoint,ident,cell,seed,epoch,score):
    if (checkpoint.get('identity')!=ident or checkpoint.get('cell')!=cell or type(checkpoint.get('seed')) is not int
            or checkpoint['seed']!=seed or checkpoint.get('settings')!={'cell':cell,'width':32,'inference_steps':3}
            or checkpoint.get('selected_epoch')!=epoch or checkpoint.get('validation_nll')!=score
            or checkpoint.get('committed_updates')!=480 or not isinstance(checkpoint.get('state'),dict)
            or not checkpoint['state'] or not all(isinstance(x,torch.Tensor) for x in checkpoint['state'].values())
            or not _finite_tree(checkpoint['state'])):
        raise RuntimeError('Inert selected checkpoint identity/selection/finite weights changed')
    return checkpoint


def verify_validation_predictions(receipt,examples=None):
    c=_common()
    if examples is None:
        from . import study
        examples=study.examples('validation',regime='sparse')
    validate_examples(examples,'validation','sparse')
    actual_binding=hashlib.sha256(_canonical([e['binding'] for e in examples])).hexdigest()
    if actual_binding!=receipt['dataset_binding_sha256']['validation_sparse']:
        raise RuntimeError('Sparse validation inputs/targets differ from the fitted member')
    references=[]
    for row in receipt['history']:
        if row['epoch'] not in VALIDATION_EPOCHS:continue
        expected=c.OUT/'validation_predictions'/f"{receipt['cell']}-seed{receipt['seed']}-epoch{row['epoch']}.npz"
        reference=row['validation_predictions']
        if reference['path']!=str(expected.relative_to(c.ROOT)) or reference['sha256']!=c.digest(expected):
            raise RuntimeError('Raw validation archive path/bytes changed')
        metadata=validation_metadata(receipt['identity'],receipt['dataset_binding_sha256'],
            receipt['cell'],receipt['seed'],row['epoch'],row['validation_state_sha256'])
        score,rows,_,_=read_validation_archive(expected,metadata,examples)
        if score!=row['validation_nll'] or rows!=row['validation_per_layout_nll']:
            raise RuntimeError('Selected validation score does not reproduce raw A-only predictions')
        references.append(reference)
    if len(references)!=4:raise RuntimeError('All four fixed validation prediction archives are required')
    return references


def verify_completed(receipt,ident,cell,seed,*,validation_examples=None):
    name=_member(cell,seed);c=_common();path=c.OUT/'checkpoints'/f'{name}.pt'
    if (c.OUT/'failures'/f'{name}.json').exists():raise RuntimeError('A terminally failed member cannot be loaded as complete')
    if (receipt.get('complete') is not True or receipt.get('completed') is not True
            or receipt.get('identity')!=ident or receipt.get('cell')!=cell or type(receipt.get('seed')) is not int
            or receipt['seed']!=seed or receipt.get('run_id')!=name or receipt.get('epochs')!=40
            or receipt.get('updates')!=480 or receipt.get('committed_updates')!=480
            or receipt.get('settings')!={'cell':cell,'width':32,'inference_steps':3}
            or receipt.get('optimizer_spec')!=json.loads(json.dumps(OPTIMIZER_SPEC))
            or receipt.get('checkpoint_sha256')!=c.digest(path)):
        raise RuntimeError('Completed fit does not match exact registered member/budget/checkpoint')
    validate_history(receipt['history'],receipt['update_trace'],complete=True)
    verify_schedule(receipt['update_trace'],seed)
    epoch,score=selected_validation(receipt['history'])
    if receipt.get('selected_epoch')!=epoch or receipt.get('validation_nll')!=score:
        raise RuntimeError('Completed fit did not select the first sparse A-only VAL minimum')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    _checkpoint_valid(checkpoint,ident,cell,seed,epoch,score)
    if checkpoint.get('dataset_binding_sha256')!=receipt['dataset_binding_sha256']:
        raise RuntimeError('Selected checkpoint prepared-data binding changed')
    selected=next(row for row in receipt['history'] if row['epoch']==epoch)
    if state_digest(checkpoint['state'])!=selected['validation_state_sha256']:
        raise RuntimeError('Selected checkpoint is not the actual weights that produced its raw VAL predictions')
    verify_validation_predictions(receipt,examples=validation_examples)
    return checkpoint


def fit(cell,seed):
    """Run exactly480 committed updates, or resume the same interrupted chain."""
    name=_member(cell,seed);c=_common();ident=identity()
    work=Path(c.WORK)/f'{name}.last.pt';receipt_path=c.OUT/'fits'/f'{name}.json'
    failure=c.OUT/'failures'/f'{name}.json';attempt=c.OUT/'attempts'/name/'start.json'
    with exclusive_fit(Path(c.WORK)/f'{name}.lock'):
        if failure.exists():raise RuntimeError('This member has a recorded terminal failure; no automatic retry or new trial')
        if receipt_path.exists():
            receipt=c.read(receipt_path);verify_completed(receipt,ident,cell,seed);return receipt
        if attempt.exists() and not work.exists():raise RuntimeError('Interrupted member has no committed snapshot; refuse a replacement chain')
        if work.exists() and not attempt.exists():raise RuntimeError('Unowned resume state cannot create a new trial')
        from . import study
        full=study.examples('training',regime='full');sparse=study.examples('training',regime='sparse')
        validation=study.examples('validation',regime='sparse')
        layouts=validate_examples(full,'training','full');validate_examples(sparse,'training','sparse');validate_examples(validation,'validation','sparse')
        if any((a['binding']['case_id'],a['binding']['layout_id'])!=(b['binding']['case_id'],b['binding']['layout_id']) for a,b in zip(full,sparse)):
            raise RuntimeError('Full/sparse regimes must share exact state/layout cases')
        data_identity=_data_identity(full,sparse,validation)
        if not attempt.exists():
            _publish_json(c,attempt,{'schema':1,'run_id':name,'cell':cell,'seed':seed,'identity':ident,
                'dataset_binding_sha256':data_identity,'created_unix_ns':time.time_ns(),
                'maximum_committed_updates':480,'selection_epochs':[10,20,30,40],
                'before_first_optimizer_step':True})
        prior_attempt=c.read(attempt)
        if (prior_attempt['identity']!=ident or prior_attempt['dataset_binding_sha256']!=data_identity
                or prior_attempt.get('run_id')!=name or prior_attempt.get('cell')!=cell
                or type(prior_attempt.get('seed')) is not int or prior_attempt['seed']!=seed
                or prior_attempt.get('maximum_committed_updates')!=480
                or prior_attempt.get('selection_epochs')!=[10,20,30,40]
                or prior_attempt.get('before_first_optimizer_step') is not True):
            raise RuntimeError('Attempt source/prepared data/member identity changed')
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
        rng=np.random.default_rng(seed);model=HierarchicalField(cell,width=32,inference_steps=3)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1.e-4,betas=(.9,.999),eps=1.e-8)
        trace=[];history=[];best_state=None;elapsed=0.;resumed=False
        if work.exists():
            prior=validate_resume(torch.load(work,map_location='cpu',weights_only=True),ident,data_identity,cell,seed)
            model.load_state_dict(prior['state'],strict=True);optimizer.load_state_dict(prior['optimizer'])
            torch.set_rng_state(prior['torch_rng']);rng.bit_generator.state=prior['numpy_rng']
            trace,history,best_state,elapsed=(prior[k] for k in ('update_trace','history','best_state','elapsed_seconds'))
            existing=sorted((attempt.parent).glob('resume-*.json'));number=len(existing)+1
            _publish_json(c,attempt.parent/f'resume-{number:04d}.json',{'identity':ident,'run_id':name,
                'committed_updates':len(trace),'work_snapshot_sha256':c.digest(work),'created_unix_ns':time.time_ns(),
                'same_rng_and_optimizer_trajectory':True,'new_seed_or_trial':False,
                'uncommitted_step_may_be_recomputed':True})
            resumed=True
        training={'full':batch(full),'sparse':batch(sparse)};validation_values=batch(validation)
        began=time.perf_counter()
        def commit():
            atomic_save(work,_state_payload(model,optimizer,rng,ident,data_identity,cell,seed,trace,history,best_state,
                                           elapsed+time.perf_counter()-began))
        if not resumed:commit()  # Durable seed/optimizer/RNG before first step.
        try:
            while True:
                completed=len(trace)
                # A committed twelfth step can outlive its uncommitted epoch
                # summary/validation. Recover that metadata without another step.
                if completed and completed%12==0 and len(history)<completed//12:
                    epoch=completed//12;score=rows=validation_ref=validation_weights=None
                    if epoch in VALIDATION_EPOCHS:
                        score,rows,validation_ref,validation_weights=validation_at_epoch(model,validation,validation_values,
                            ident,data_identity,cell,seed,epoch)
                    previous_epoch,previous_score=selected_validation(history)
                    if score is not None and (previous_score is None or score<previous_score):best_state=_clone_state(model)
                    history.append({'epoch':epoch,'completed_updates':completed,
                        'mean_training_loss':float(np.mean([r['loss'] for r in trace[-12:]])),
                        'validation_nll':score,'validation_per_layout_nll':rows,
                        'validation_predictions':validation_ref,'validation_state_sha256':validation_weights})
                    commit()
                if completed==480:break
                indices,chosen_layouts,conditions,regime=choose_minibatch(rng,completed)
                v=select(training[regime],indices);model.train();optimizer.zero_grad(set_to_none=True)
                loss,details=objective(model,v['a'],v['b'],v['query'],v['target'],v['ap_xyz'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fixed training objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
                if not _finite_tree(model.state_dict()) or not _finite_tree(optimizer.state_dict()):
                    raise FloatingPointError('Nonfinite weights/optimizer after committed update')
                row={key:float(value) for key,value in details.items()}
                row.update(step=completed+1,epoch=completed//12+1,update_in_epoch=completed%12+1,
                    regime=regime,layout_indices=chosen_layouts.astype(int).tolist(),conditions=conditions.astype(int).tolist(),
                    case_ids=[(full if regime=='full' else sparse)[int(i)]['binding']['case_id'] for i in indices],
                    gradient_norm_before_clip=float(norm))
                if not _finite_tree(row):raise FloatingPointError('Nonfinite committed-step trace')
                trace.append(row);commit()
            if identity()!=ident:raise RuntimeError('Source/prepared data changed during fitting')
            validate_history(history,trace,complete=True);epoch,score=selected_validation(history)
            if best_state is None:raise RuntimeError('No registered sparse validation checkpoint was selected')
            checkpoint={'identity':ident,'dataset_binding_sha256':data_identity,'cell':cell,'seed':seed,
                'settings':model.config,'state':best_state,'selected_epoch':epoch,
                'validation_nll':score,'committed_updates':480}
            checkpoint_path=c.OUT/'checkpoints'/f'{name}.pt'
            if checkpoint_path.exists():
                existing=_checkpoint_valid(torch.load(checkpoint_path,map_location='cpu',weights_only=True),ident,cell,seed,epoch,score)
                if set(existing['state'])!=set(best_state) or any(not torch.equal(existing['state'][k],v) for k,v in best_state.items()):
                    raise RuntimeError('Existing final checkpoint is not this exact selected snapshot')
            else:_publish_torch(checkpoint_path,checkpoint,staging=Path(c.WORK)/'staging')
            receipt={'schema':1,'complete':True,'completed':True,'run_id':name,'cell':cell,'seed':seed,
                'identity':ident,'dataset_binding_sha256':data_identity,'settings':model.config,
                'optimizer_spec':OPTIMIZER_SPEC,'epochs':40,'updates':480,'committed_updates':480,
                'update_trace':trace,'history':history,'selected_epoch':epoch,'validation_nll':score,
                'selection':'FIRST minimum at1-based epochs10/20/30/40; sparse VAL A-only equal-layout composite interval NLL',
                'checkpoint_sha256':c.digest(checkpoint_path),'elapsed_seconds':elapsed+time.perf_counter()-began,
                'learned_parameters':sum(p.numel() for p in model.parameters()),'batch_distinct_layouts':4,
                'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':'cpu',
                    'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
                    'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()},
                'training_layouts':list(layouts),'training_bands':[0,1],
                'resume_events':[{'path':str(p.relative_to(c.ROOT)),'sha256':c.digest(p)} for p in sorted(attempt.parent.glob('resume-*.json'))],
                'initial_attempt_sha256':c.digest(attempt),'uncertainty_calibrated':False,'packet_or_production_qualified':False}
            _publish_json(c,receipt_path,receipt)
            verify_completed(c.read(receipt_path),ident,cell,seed)
            return c.read(receipt_path)
        except KeyboardInterrupt:
            number=len(list(attempt.parent.glob('interrupt-*.json')))+1
            _publish_json(c,attempt.parent/f'interrupt-{number:04d}.json',{'identity':ident,'run_id':name,
                'created_unix_ns':time.time_ns(),'last_snapshot_sha256':c.digest(work),
                'reason':'KeyboardInterrupt; last fully atomic committed snapshot remains resumable'})
            raise
        except OSError:
            # Filesystem interruptions retain the last atomic snapshot. They
            # never justify replacing a mathematical/nonfinite FAILED member.
            raise
        except Exception as exc:
            _publish_json(c,failure,{'schema':1,'run_id':name,'cell':cell,'seed':seed,'identity':ident,
                'created_unix_ns':time.time_ns(),'error_type':type(exc).__name__,'error':str(exc),
                'committed_snapshot_sha256':c.digest(work),'automatic_retry_permitted':False})
            raise


def load(cell,seed):
    name=_member(cell,seed);c=_common();ident=identity()
    receipt=c.read(c.OUT/'fits'/f'{name}.json');checkpoint=verify_completed(receipt,ident,cell,seed)
    with torch.random.fork_rng(devices=[]):model=HierarchicalField(**checkpoint['settings'])
    model.load_state_dict(checkpoint['state'],strict=True);model.eval();return model


@torch.no_grad()
def predict(model,example,view='a'):
    if view not in ('a','b'):raise ValueError('Only actual view a/b predictions are supported')
    v=batch([example]);mean,std=model(v[view],v['query'],v['ap_xyz'])
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():raise FloatingPointError('Nonfinite field prediction')
    return mean[0].detach().cpu().numpy().copy(),std[0].detach().cpu().numpy().copy()


def _fit_task(task):return fit(*task)


def training_worker_context():
    # LLVM/DrJit/Torch thread pools may already exist after RT collection.
    # Spawn fresh interpreters instead of copying those live runtimes by fork.
    return mp.get_context('spawn')


def train_all(workers=1):
    if type(workers) is not int or workers not in (1,2,3):raise ValueError('Only one to three bounded CPU training workers are supported')
    tasks=[(cell,seed) for cell in CELLS for seed in SEEDS]
    if workers==1:return [_fit_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers,mp_context=training_worker_context()) as pool:return list(pool.map(_fit_task,tasks))
