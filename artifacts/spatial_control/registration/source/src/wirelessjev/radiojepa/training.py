"""Finite registered measured-data experiment; observations are not service labels."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
import io
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import common as c, data
from .model import SceneJEPA


def save_npz(path, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    c.write_bytes_once(path, stream.getvalue())
    return {'path': str(Path(path).relative_to(c.ROOT)), 'sha256': c.digest(path)}


def _check_reference(reference):
    path = c.safe_path(reference['path'])
    if c.digest(path) != reference['sha256']:
        raise RuntimeError('Prepared measured data changed')
    return path


def prepare_data():
    c.check_sources()
    path = c.OUT / 'data/receipt.json'
    if path.exists():
        receipt = c.read_json(path)
        for reference in receipt['files'].values():
            _check_reference(reference)
        if (receipt['datasets_sha256'] != c.digest(c.OUT / 'datasets.json')
                or receipt['source_receipt_sha256']!=c.digest(c.OUT/'registration/source.json')):
            raise RuntimeError('Prepared source identities changed')
        return receipt
    selection, recordings, files = c.read_json(c.OUT / 'datasets.json'), {}, {}
    split_info = None
    for item in selection['selected_measured_recordings']:
        name, root = item['name'], c.data_root(item)
        for key in ('npz', 'receipt', 'binding'):
            if c.digest(root / item[key + '_filename']) != item[key + '_sha256']:
                raise RuntimeError('Acquired source differs from preregistered data: ' + name)
        measured = data.load_recording(name, root)
        group_ids=np.unique(measured.antenna_group_ids)
        array_power=[];array_known=[];centers=[]
        for group in group_ids:
            member=measured.antenna_group_ids==group
            linear=np.power(10.,measured.log_power_db[:,member].astype('f8')/10.).mean(1)
            array_power.append((10*np.log10(linear)).astype('f4'))
            array_known.append(measured.positive_power_mask[:,member].all(1))
            centers.append(measured.antenna_array_centers_m[member][0])
        array_power=np.stack(array_power,1);array_known=np.stack(array_known,1)
        public = {'power_db': array_power, 'xyz': np.array(centers),
                  'groups': np.arange(len(group_ids),dtype='i8'), 'time_s': measured.time_s,
                  'bin_frequency_hz': measured.bin_frequency_hz,
                  'carrier_hz': np.asarray(measured.carrier_hz),
                  'positive_mask': array_known}
        files[name + '.public'] = save_npz(c.OUT / 'data/measured' / (name + '.public.npz'), public)
        files[name + '.labels'] = save_npz(c.OUT / 'data/measured' / (name + '.labels.npz'),
                                          {'positions': data.private_pose_labels(measured)})
        eligible = np.flatnonzero(array_known.all((1, 2)) & (array_power > -120.).all((1, 2)))
        info = {'samples': measured.samples, 'eligible_rows': eligible.tolist(),
                'rejected_incomplete_or_floor_rows': measured.samples - len(eligible),
                'power_unit': measured.power_unit, 'source_sha256': measured.source_sha256,
                'cross_array_eligible': item['cross_array_eligible'],
                'source_antenna_count':measured.log_power_db.shape[1], 'physical_array_count':len(group_ids),
                'aggregation':'mean LINEAR power over actual antenna-group members; then dB; no invented element coordinates'}
        if item['cross_array_eligible']:
            a,b = np.arange(len(group_ids))[::2],np.arange(len(group_ids))[1::2]
            info.update(view_a=a.tolist(), view_b=b.tolist())
        if name == c.protocol()['training_recording']:
            split = data.spatial_split(measured)
            pairs = data.temporal_partners(measured, split.train)
            pairs = pairs[np.isin(pairs[:, 0], eligible) & np.isin(pairs[:, 1], eligible)]
            validation = split.validation[np.isin(split.validation, eligible)]
            test = split.test[np.isin(split.test, eligible)]
            expected = c.protocol()['split']
            if (len(pairs) != expected['common_train'] or len(validation) != expected['validation']
                    or len(test) != expected['test']):
                raise RuntimeError('Published cohort sizes changed; do not retune after fitting')
            files['split'] = save_npz(c.OUT / 'data/measured/factory.split.npz', {
                'train': pairs[:, 0], 'validation': validation, 'test': test,
                'temporal_pairs': pairs, 'full_spatial_train': split.train,
                'excluded': split.excluded})
            split_info = split.report
        recordings[name] = info
    receipt = {'schema': 1, 'files': files, 'recordings': recordings, 'split': split_info,
               'datasets_sha256': c.digest(c.OUT / 'datasets.json'),
               'source_receipt_sha256': c.digest(c.OUT / 'registration/source.json')}
    c.write_once(path, receipt)
    return receipt


def prepared(name):
    receipt = c.read_json(c.OUT / 'data/receipt.json')
    with np.load(_check_reference(receipt['files'][name + '.public']), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def split_arrays():
    receipt = c.read_json(c.OUT / 'data/receipt.json')
    with np.load(_check_reference(receipt['files']['split']), allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def labels(name):
    receipt = c.read_json(c.OUT / 'data/receipt.json')
    with np.load(_check_reference(receipt['files'][name + '.labels']), allow_pickle=False) as saved:
        return saved['positions'].copy()


def settings(cell):
    if cell not in c.protocol()['cells']:
        raise ValueError('Unregistered experiment cell')
    return {'regularizer': 'none' if cell == 'joint_none' else 'vicreg' if cell == 'joint_vicreg' else 'sigreg',
            'decoder': 'unstructured' if cell == 'unstructured_sigreg' else 'physical',
            'alignment': cell != 'regression_sigreg', 'temporal': cell == 'temporal_sigreg'}


def tensor(array):
    return torch.as_tensor(np.array(array, dtype='f4', copy=True))


def operands(public, rows, view):
    rows, view = np.asarray(rows), np.asarray(view)
    power = tensor(public['power_db'][rows][:, view])
    xyz = tensor(public['xyz'][view])
    origin = tensor(np.broadcast_to(public['xyz'].mean(0), (len(rows), 3)))
    return power, xyz, origin, float(public['carrier_hz'])


@torch.no_grad()
def encode(model, public, rows, view, maximum=128):
    result = {}
    for start in range(0, len(rows), maximum):
        a, xyz, origin, carrier = operands(public, rows[start:start+maximum], view)
        encoded = model.encode(a, xyz, origin, carrier)
        for key, value in encoded.items():
            result.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(values) for key, values in result.items()}


@torch.no_grad()
def render(model, public, encoded, source_view, target_view, maximum=128):
    rows, results = len(encoded['z']), []
    offsets = public['bin_frequency_hz'] - float(public['carrier_hz'])
    for start in range(0, rows, maximum):
        end = min(rows, start+maximum)
        chunk = {key:value[start:end] for key,value in encoded.items()}
        origin = tensor(np.broadcast_to(public['xyz'].mean(0), (end-start, 3)))
        prediction = model.render(chunk, tensor(public['xyz'][source_view]), tensor(public['xyz'][target_view]),
                                  origin, float(public['carrier_hz']), offsets)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError('Nonfinite measured-power prediction')
        results.append(prediction.numpy())
    return np.concatenate(results)


@torch.no_grad()
def validation_rmse(model, public, rows, a, b):
    ea,eb = encode(model, public, rows, a), encode(model, public, rows, b)
    pa,pb = render(model, public, eb, b, a), render(model, public, ea, a, b)
    truth_a,truth_b = public['power_db'][rows][:,a], public['power_db'][rows][:,b]
    score=float(np.sqrt((np.square(pa.astype('f8')-truth_a).mean()+np.square(pb.astype('f8')-truth_b).mean())/2))
    if not np.isfinite(score): raise FloatingPointError('Nonfinite validation metric')
    return score


def identity():
    c.check_sources()
    return {'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
            'protocol_sha256':c.digest(c.PROTOCOL), 'data_receipt_sha256':c.digest(c.OUT/'data/receipt.json')}


def _atomic_torch(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.resume-',suffix='.pt',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(value,stream)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def fit(task):
    cell,seed=task
    p, ident = c.protocol(),identity()
    if seed not in p['training_seeds']: raise ValueError('Unregistered seed')
    option=settings(cell)
    run=f'{cell}-seed{seed}'
    complete=c.OUT/'fits'/f'{run}.json'
    if complete.exists():
        saved=c.read_json(complete)
        if (saved.get('completed') is not True or saved.get('cell')!=cell or saved.get('seed')!=seed
                or saved.get('run_id')!=run or any(saved.get(k)!=v for k,v in ident.items())
                or c.digest(c.OUT/'checkpoints'/f'{run}.pt')!=saved['checkpoint_sha256']):
            raise RuntimeError('Completed fit identity/checkpoint changed')
        checkpoint=torch.load(c.OUT/'checkpoints'/f'{run}.pt',map_location='cpu',weights_only=True)
        if checkpoint['identity']!=ident or checkpoint['cell']!=cell or checkpoint['seed']!=seed:
            raise RuntimeError('Completed checkpoint metadata changed')
        return saved
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=SceneJEPA(bins=p['bins'],width=p['width'],paths=p['paths'],regularizer=option['regularizer'],
                    decoder=option['decoder'],alignment_weight=p['alignment_weight'],prior_weight=p['prior_weight'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    public=prepared(p['training_recording']); split=split_arrays()
    info=c.read_json(c.OUT/'data/receipt.json')['recordings'][p['training_recording']]
    a,b=np.array(info['view_a']),np.array(info['view_b'])
    train=split['train']; partners=dict(split['temporal_pairs'].tolist())
    resume=c.WORK/f'{run}.last.pt'
    history=[]; best=float('inf'); best_state=None; best_epoch=None; elapsed=0.
    if resume.exists():
        last=torch.load(resume,map_location='cpu',weights_only=True)
        if last['identity']!=ident or last['cell']!=cell or last['seed']!=seed:
            raise RuntimeError('Resume identity changed')
        model.load_state_dict(last['state']);optimizer.load_state_dict(last['optimizer'])
        history,best,best_state,best_epoch=last['history'],last['best'],last['best_state'],last['best_epoch']
        if (not history or len(history)>p['epochs']
                or [x['epoch'] for x in history]!=list(range(1,len(history)+1))
                or not all(np.isfinite(x['validation_rmse_db']) for x in history)
                or best_epoch!=min(history,key=lambda x:x['validation_rmse_db'])['epoch']
                or best!=min(x['validation_rmse_db'] for x in history)):
            raise RuntimeError('Malformed epoch-resume history or checkpoint choice')
        rng.bit_generator.state=last['rng'];torch.set_rng_state(last['torch_rng']);elapsed=last['elapsed']
    start=time.perf_counter()
    offsets=public['bin_frequency_hz']-float(public['carrier_hz'])
    for epoch in range(len(history),p['epochs']):
        model.train(); losses=[]
        order=rng.permutation(train)
        for begin in range(0,len(order),p['batch_size']):
            rows=order[begin:begin+p['batch_size']]
            if len(rows)<2: raise RuntimeError('Regularizer needs two actual scene rows')
            av,xa,origin,carrier=operands(public,rows,a);bv,xb,_,_=operands(public,rows,b)
            temporal=None
            if option['temporal']:
                temporal=operands(public,np.array([partners[int(i)] for i in rows]),b)[0]
            loss,detail=model.objective(av,bv,xyz_a=xa,xyz_b=xb,origin=origin,carrier_hz=carrier,
                offsets_hz=offsets,alignment=option['alignment'],temporal_b=temporal)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step();losses.append(float(loss.detach()))
        model.eval(); score=validation_rmse(model,public,split['validation'],a,b)
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_rmse_db':score}
        history.append(row)
        if score<best:
            best,best_epoch=score,epoch+1;best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        c.check_sources()
        _atomic_torch(resume,{'identity':ident,'cell':cell,'seed':seed,'state':model.state_dict(),
            'optimizer':optimizer.state_dict(),'history':history,'best':best,'best_state':best_state,
            'best_epoch':best_epoch,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'elapsed':elapsed+time.perf_counter()-start})
        print(json.dumps({'phase':'scene_fit','cell':cell,'seed':seed,**row}),flush=True)
    model.load_state_dict(best_state);model.eval()
    final={'state':best_state,'config':model.config,'epoch':best_epoch,'cell':cell,'seed':seed,'identity':ident}
    checkpoint=c.OUT/'checkpoints'/f'{run}.pt'
    if checkpoint.exists():
        prior=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (prior['identity']!=ident or prior['config']!=model.config
                or any(not torch.equal(prior['state'][k],v) for k,v in best_state.items())):
            raise RuntimeError('Immutable selected checkpoint differs')
    else:
        stream=io.BytesIO();torch.save(final,stream);c.write_bytes_once(checkpoint,stream.getvalue())
    result={'schema':1,'completed':True,'cell':cell,'seed':seed,'run_id':run,**ident,
        'checkpoint_sha256':c.digest(checkpoint),'best_epoch':best_epoch,'best_validation_rmse_db':best,
        'history':history,'parameters':model.inference_parameters,'config':model.config,
        'elapsed_fit_seconds':elapsed+time.perf_counter()-start,
        'training_rows':len(train),'training_labels_are_observed_RF':True,
        'private_pose_used_for_encoder_training':False}
    result.update(evaluate_frozen(model,run))
    c.check_sources();c.write_once(complete,result)
    return result


def evaluate_frozen(model,run):
    from .evaluation import metrics,representation_metrics,ridge_fit
    p,receipt=c.protocol(),c.read_json(c.OUT/'data/receipt.json')
    factory=prepared(p['training_recording']);split=split_arrays()
    fi=receipt['recordings'][p['training_recording']];a,b=np.array(fi['view_a']),np.array(fi['view_b'])
    training=encode(model,factory,split['train'],a);validation=encode(model,factory,split['validation'],a)
    # Position supervision enters only a separate frozen linear probe.
    pose=labels(p['training_recording'])[:,:2]
    head=ridge_fit(training['z'].numpy(),pose[split['train']],validation['z'].numpy(),pose[split['validation']])
    probe_arrays=head.as_arrays()|{'train_z':training['z'].numpy(),'validation_z':validation['z'].numpy(),
                                 'train_rows':split['train'],'validation_rows':split['validation']}
    domains={}
    for name in [p['training_recording'],*p['heldout_recordings']]:
        public=prepared(name);info=receipt['recordings'][name]
        rows=split['test'] if name==p['training_recording'] else np.array(info['eligible_rows'],dtype='i8')
        va,vb=np.array(info['view_a']),np.array(info['view_b'])
        forward_begin_ns=time.perf_counter_ns()
        ea,eb=encode(model,public,rows,va),encode(model,public,rows,vb)
        arrays={'source_rows':rows,'view_a':va,'view_b':vb,
                'pred_ab':render(model,public,ea,va,vb),'pred_ba':render(model,public,eb,vb,va),
                'z_a':ea['z'].numpy(),'z_b':eb['z'].numpy(),
                'projected_a':ea['projected'].numpy(),'projected_b':eb['projected'].numpy()}
        forward_end_ns=time.perf_counter_ns()
        permutation=np.random.default_rng(2168000).permutation(len(rows))
        arrays['shuffle_rows']=permutation
        for kind in p['interventions']:
            changed={k:v.clone() for k,v in ea.items()}
            if kind in ('z_zero','all_zero'): changed['z'].zero_()
            if kind in ('tokens_zero','all_zero'): changed['tokens'].zero_()
            if kind=='power_zero': changed['power_coordinate'].zero_()
            if kind=='all_shuffle':
                changed['z']=changed['z'][permutation];changed['tokens']=changed['tokens'][permutation]
            arrays['intervention_ab_'+kind]=render(model,public,changed,va,vb)
        path=c.OUT/'predictions'/f'{run}-{name}.npz'
        reference=save_npz(path,arrays)
        c.write_once(Path(str(path)+'.before-targets.json'),{'predictions':reference,
            'prepared_public':receipt['files'][name+'.public'],'source_rows':rows.tolist(),
            'prediction_complete_before_metric_computation':True})
        target_a,target_b=public['power_db'][rows][:,va],public['power_db'][rows][:,vb]
        combined=metrics(np.concatenate((arrays['pred_ab'],arrays['pred_ba']),axis=1),np.concatenate((target_b,target_a),axis=1))
        domains[name]={'prediction_path':reference['path'],'prediction_sha256':reference['sha256'],
            'metrics':combined,'representation_z':representation_metrics(arrays['z_a'],arrays['z_b']),
            'ab_metrics':metrics(arrays['pred_ab'],target_b),
            'representation_projected':representation_metrics(arrays['projected_a'],arrays['projected_b']),
            'interventions':{k:metrics(arrays['intervention_ab_'+k],target_b) for k in p['interventions']},
            'intervention_scope':'all_zero/all_shuffle alter both pooledz andreceiver tokens; measuredpower is held fixed; power_zero tests it separately',
            'forward_timing':{'begin_monotonic_ns':forward_begin_ns,'end_monotonic_ns':forward_end_ns,
                             'seconds':(forward_end_ns-forward_begin_ns)/1.e9,
                             'scope':'both views, preparedpower input, encoder+projector+decoder; excludes CSI preprocessing/loading/transport; sharedCPU; not realtime'}}
        if name==p['training_recording']:
            predicted=head.predict(arrays['z_a']);probe_arrays.update(test_z=arrays['z_a'],test_prediction=predicted,test_rows=rows)
            error=np.linalg.norm(predicted-pose[rows],axis=-1)
            domains[name]['frozen_position_probe']={'mean_error_m':float(error.mean()),'median_error_m':float(np.median(error))}
    probe_ref=save_npz(c.OUT/'probes'/f'{run}.npz',probe_arrays)
    return {'domains':domains,'position_probe':{'archive':probe_ref,'metadata':head.metadata,
            'labels_used_after_backbone_frozen':True,'target':'factoryXYmeters;notrobotservice'}}


def train_all(workers=None):
    prepare_data();p=c.protocol();workers=p['workers'] if workers is None else workers
    if workers!=p['workers']: raise ValueError('Use registered worker count')
    tasks=[(cell,seed) for cell in p['cells'] for seed in p['training_seeds']]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reports=list(pool.map(fit,tasks))
    result={'completed':True,'fits':len(reports),'runs':[r['run_id'] for r in reports],
            'source_receipt_sha256':c.digest(c.OUT/'registration/source.json')}
    c.write_once(c.OUT/'training.json',result)
    return result
