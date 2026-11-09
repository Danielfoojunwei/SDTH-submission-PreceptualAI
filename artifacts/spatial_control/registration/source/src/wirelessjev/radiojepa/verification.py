"""Passively verify the finite measured RadioJEPA pilot; never refit or infer.

Read stored arrays/checkpoints and recompute numerical evidence. Hashes and
published source establish byte identity, not independent hardware provenance.
No World, neural module, optimizer, ray tracer or download is constructed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np

from wirelessjev.radiojepa import common as c
from wirelessjev.radiojepa.evaluation import (
    CELLS, POWER_UNIT, RidgeModel, aggregate_fit_reports, mean_template, metrics,
    power_level_transfer, representation_metrics,
)


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def equivalent(actual,expected,label='value'):
    """Compare recomputed floats with roundoff only; identity flags are literal."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),label+': dictionary membership differs')
        for key,value in expected.items(): equivalent(actual[key],value,label+'.'+key)
    elif isinstance(expected,(list,tuple)):
        require(isinstance(actual,(list,tuple)) and len(actual)==len(expected),label+': list length differs')
        for index,(a,b) in enumerate(zip(actual,expected)): equivalent(a,b,label+f'[{index}]')
    elif isinstance(expected,(bool,type(None),str)):
        require(type(actual) is type(expected) and actual==expected,label+': identity differs')
    elif isinstance(expected,(int,np.integer)):
        require(type(actual) is int and actual==int(expected),label+': integer differs')
    elif isinstance(expected,(float,np.floating)):
        require(not isinstance(actual,bool) and isinstance(actual,(int,float,np.number))
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10),label+': numerical value differs')
    else:
        raise RuntimeError('Unsupported comparison type: '+label)


def close(a,b,label,*,atol=1e-10,rtol=1e-10):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=atol,rtol=rtol),label)


def path(name):
    return c.safe_path(name,c.ROOT)


def reference(value):
    require(isinstance(value,dict) and set(value)=={'path','sha256'},'Malformed archive reference')
    p=path(value['path'])
    require(p.is_file() and c.digest(p)==value['sha256'],'Archive bytes changed: '+value['path'])
    return p


def arrays(p):
    with np.load(p,allow_pickle=False) as saved:
        return {name:saved[name].copy() for name in saved.files}


def _finite(a,shape,label):
    a=np.asarray(a)
    require(a.dtype.kind in 'fiu' and a.shape==shape and np.isfinite(a).all(),label)


def _indices(a,expected,label):
    a=np.asarray(a)
    require(a.dtype.kind in 'iu' and np.array_equal(a,expected),label)


def measured_transform(raw,binding):
    """Deterministic front-end arithmetic only; no labels become features."""
    csi=np.asarray(raw['csi'])
    require(csi.dtype==np.float32 and csi.ndim==4 and csi.shape[2:]==(1024,2)
            and np.isfinite(csi).all(),'Original measured CSI representation differs')
    n,source_antennas=csi.shape[:2]
    groups=np.asarray(binding['antenna_group_ids'],dtype='i8')
    centers=np.asarray(binding['array_centers_by_antenna_m'],dtype='f8')
    require(groups.shape==(source_antennas,) and centers.shape==(source_antennas,3),
            'Original physical-array binding axes differ')
    shifted=np.fft.fftshift(csi,axes=2).astype('f8')
    linear=np.square(shifted).sum(-1).reshape(n,source_antennas,64,16).mean(-1)
    positive=linear>0
    individual=(10*np.log10(np.maximum(linear,1e-12))).astype('f4')
    ids=np.unique(groups)
    power=[];known=[];xyz=[]
    for group in ids:
        mask=groups==group
        power.append((10*np.log10(np.power(10.,individual[:,mask].astype('f8')/10.).mean(1))).astype('f4'))
        known.append(positive[:,mask].all(1));xyz.append(centers[mask][0])
    carrier=float(binding['frequency_hz']);bandwidth=float(binding['bandwidth_hz'])
    frequency=carrier+bandwidth*(np.arange(1024,dtype='f8')-512)/1024
    return {'power_db':np.stack(power,1),'positive_mask':np.stack(known,1),
            'xyz':np.array(xyz),'groups':np.arange(len(ids),dtype='i8'),
            'time_s':np.asarray(raw['time'],dtype='f8').reshape(-1),
            'bin_frequency_hz':frequency.reshape(64,16).mean(-1),'carrier_hz':np.asarray(carrier)},source_antennas


def read_data():
    p,receipt=c.protocol(),c.read_json(c.OUT/'data/receipt.json')
    selected=c.read_json(c.OUT/'datasets.json')['selected_measured_recordings']
    names=[item['name'] for item in selected]
    require(set(receipt['recordings'])==set(names),'Measured recording membership changed')
    require(receipt['datasets_sha256']==c.digest(c.OUT/'datasets.json')
            and receipt['source_receipt_sha256']==c.digest(c.OUT/'registration/source.json'),
            'Measured source receipt identity changed')
    expected_files={name+suffix for name in names for suffix in ('.public','.labels')}|{'split'}
    require(set(receipt['files'])==expected_files,'Prepared data archive membership changed')
    public,labels={},{}
    for item in selected:
        name=item['name']; info=receipt['recordings'][name]
        source_root=c.data_root(item)
        for key in ('npz','receipt','binding'):
            source=source_root/item[key+'_filename']
            require(source.is_file() and not source.is_symlink() and c.digest(source)==item[key+'_sha256'],
                    'Restore the preregistered external source cache before raw audit: '+str(source))
        source_receipt=c.read_json(source_root/item['receipt_filename'])
        require(source_receipt['all_length_and_payload_crc32c_pass'] is True
                and source_receipt['acquired_records']==1024,'Measured prefix CRC/record provenance differs')
        raw=arrays(source_root/item['npz_filename'])
        expected,source_antennas=measured_transform(raw,c.read_json(source_root/item['binding_filename']))
        public[name]=arrays(reference(receipt['files'][name+'.public']))
        a=public[name]
        require(set(a)=={'power_db','xyz','groups','time_s','bin_frequency_hz','carrier_hz','positive_mask'},
                'Measured public feature whitelist differs')
        power=a['power_db']
        require(power.ndim==3 and power.shape[0]==1024 and power.shape[-1]==p['bins'],
                'Measured power axes changed')
        n,antennas,bins=power.shape
        _finite(power,(n,antennas,bins),'Nonfinite measured referenced power')
        _finite(a['xyz'],(antennas,3),'Receiver-array center axes changed')
        _finite(a['time_s'],(n,),'Measured timestamps changed')
        _finite(a['bin_frequency_hz'],(bins,),'Measured frequency axes changed')
        require(np.all(np.diff(a['bin_frequency_hz'])>0) and a['carrier_hz'].shape==()
                and np.isfinite(a['carrier_hz']) and a['carrier_hz']>0,'Invalid RF metadata')
        require(a['groups'].shape==(antennas,) and a['groups'].dtype.kind in 'iu'
                and a['positive_mask'].dtype.kind=='b' and a['positive_mask'].shape==power.shape,
                'Physical array groups or positive-power masks differ')
        for key,value in expected.items():
            require(np.array_equal(a[key],value),'Prepared feature differs from measured physical-group transform: '+key)
        require(info['source_antenna_count']==source_antennas and info['physical_array_count']==antennas
                and info['aggregation']=='mean LINEAR power over actual antenna-group members; then dB; no invented element coordinates',
                'Physical-array endpoint metadata differs')
        eligible=np.flatnonzero(a['positive_mask'].all((1,2)) & (power>-120.).all((1,2)))
        require(info['samples']==n and info['power_unit']==POWER_UNIT
                and info['eligible_rows']==eligible.tolist()
                and info['rejected_incomplete_or_floor_rows']==n-len(eligible),
                'Eligibility/units differ from measured positive powers')
        source_files={item['npz_filename']:item['npz_sha256'],item['receipt_filename']:item['receipt_sha256'],
                      item['binding_filename']:item['binding_sha256']}
        require(info['source_sha256']==source_files,'Prepared source hashes differ from preregistration')
        groups=np.unique(a['groups'])
        require(info['cross_array_eligible'] is item['cross_array_eligible']
                and info['cross_array_eligible'] is (len(groups)>=2),'Cross-array eligibility was invented')
        if len(groups)>=2:
            va=np.flatnonzero(np.isin(a['groups'],groups[::2]))
            vb=np.flatnonzero(np.isin(a['groups'],groups[1::2]))
            require(info['view_a']==va.tolist() and info['view_b']==vb.tolist()
                    and not np.intersect1d(va,vb).size,'Views overlap or split a physical array')
        for group in groups:
            xyz=a['xyz'][a['groups']==group]
            require(np.array_equal(xyz,np.broadcast_to(xyz[0],xyz.shape)),
                    'One physical array has conflicting center metadata')
        label=arrays(reference(receipt['files'][name+'.labels']))
        require(set(label)=={'positions'},'Private-label whitelist differs')
        _finite(label['positions'],(n,3),'Invalid private measured XYZ labels')
        close(label['positions'],np.asarray(raw['pos-tachy'],dtype='f8'),'Measured position labels changed',atol=0,rtol=0)
        labels[name]=label['positions']
    split=arrays(reference(receipt['files']['split']))
    require(set(split)=={'train','validation','test','temporal_pairs','full_spatial_train','excluded'},
            'Measured split membership changed')
    for key,count in [('train',p['split']['common_train']),('validation',p['split']['validation']),('test',p['split']['test'])]:
        v=split[key]
        require(v.shape==(count,) and v.dtype.kind in 'iu' and len(np.unique(v))==count
                and np.all((v>=0)&(v<1024)),'Malformed source split: '+key)
    require(not any(np.intersect1d(split[a],split[b]).size for a,b in
                    [('train','validation'),('train','test'),('validation','test')]),'Fitting/test source rows overlap')
    factory=p['training_recording']; coordinates=labels[factory][:,:2]
    # Reconstruct the declared spatial guard without invoking any data fitter.
    lower,upper=np.quantile(coordinates[:,1],[.6,.8]); half=.25/2
    st=np.flatnonzero(coordinates[:,1]<lower-half)
    sv=np.flatnonzero((coordinates[:,1]>lower+half)&(coordinates[:,1]<upper-half))
    ss=np.flatnonzero(coordinates[:,1]>upper+half)
    def near(q,r):
        return (np.square(coordinates[q,None]-coordinates[None,r]).sum(-1)<.25**2).any(-1)
    st=st[~near(st,ss)];sv=sv[~near(sv,ss)];st=st[~near(st,sv)]
    _indices(split['full_spatial_train'],st,'Spatial training exclusion differs')
    _indices(split['validation'],sv,'Spatial validation exclusion differs')
    _indices(split['test'],ss,'Spatial test membership differs')
    excluded=np.setdiff1d(np.arange(1024),np.concatenate((st,sv,ss)))
    _indices(split['excluded'],excluded,'Spatial excluded source rows changed')
    pairs=split['temporal_pairs']
    require(pairs.shape==(len(split['train']),2) and pairs.dtype.kind in 'iu'
            and np.array_equal(pairs[:,0],split['train']) and np.isin(pairs,st).all()
            and np.all(public[factory]['time_s'][pairs[:,1]]-public[factory]['time_s'][pairs[:,0]]>=p['temporal_gap_seconds']),
            'Temporal comparison leaves TRAIN or violates its declared gap')
    return receipt,public,labels,split


def verify_ridge(model_arrays,metadata,train_x,train_y,validation_x,validation_y):
    """Check stored solution via normal equations; no new solve or refit."""
    model=RidgeModel.from_arrays({k:model_arrays[k] for k in
        ('coef','x_mean','x_scale','y_mean','alpha','output_shape')})
    x=np.asarray(train_x,dtype='f8'); y=np.asarray(train_y,dtype='f8').reshape(len(train_y),-1)
    vx=np.asarray(validation_x,dtype='f8');vy=np.asarray(validation_y,dtype='f8')
    close(model.x_mean,x.mean(0),'Ridge feature mean uses nontraining data')
    close(model.x_scale,np.maximum(x.std(0),1e-8),'Ridge feature scale uses nontraining data')
    close(model.y_mean,y.mean(0),'Ridge target centering uses nontraining labels')
    z=(x-model.x_mean)/model.x_scale
    gradient=z.T@(z@model.coef-(y-model.y_mean))+model.alpha*model.coef
    rhs=z.T@(y-model.y_mean)
    require(np.isfinite(gradient).all() and np.max(np.abs(gradient))<=1e-7*max(1.,float(np.abs(rhs).max())),
            'Stored ridge coefficients do not solve the training-only objective')
    require(metadata['test_accessed'] is False and metadata['selected_alpha']==model.alpha
            and metadata['alpha_grid']==c.protocol()['baselines']['ridge_alphas'],
            'Ridge selection scope/grid changed')
    grid=metadata['validation_grid']
    require([entry['alpha'] for entry in grid]==metadata['alpha_grid']
            and all(math.isfinite(row['validation_mse']) and row['validation_mse']>=0 for row in grid),
            'Malformed ridge validation grid')
    winner=min(range(len(grid)),key=lambda i:(grid[i]['validation_mse'],i))
    require(grid[winner]['alpha']==model.alpha,'Ridge alpha differs from declared validation minimum')
    mse=float(np.square(model.predict(vx)-vy).mean())
    equivalent(mse,grid[winner]['validation_mse'],'selected ridge validation MSE')
    return model


def verify_prediction(record,name,pub,info,split):
    p=c.protocol()
    ref={'path':record['prediction_path'],'sha256':record['prediction_sha256']}
    file=reference(ref);a=arrays(file)
    fields={'source_rows','view_a','view_b','pred_ab','pred_ba','z_a','z_b','projected_a','projected_b','shuffle_rows'}
    fields|={'intervention_ab_'+kind for kind in p['interventions']}
    require(set(a)==fields,'Prediction array membership differs')
    rows=split['test'] if name==p['training_recording'] else np.array(info['eligible_rows'],dtype='i8')
    va,vb=np.array(info['view_a']),np.array(info['view_b'])
    _indices(a['source_rows'],rows,'Prediction source rows differ')
    _indices(a['view_a'],va,'Prediction source array partition differs')
    _indices(a['view_b'],vb,'Prediction target array partition differs')
    target_a,target_b=pub['power_db'][rows][:,va],pub['power_db'][rows][:,vb]
    _finite(a['pred_ab'],target_b.shape,'Invalid A-to-B RF forecast')
    _finite(a['pred_ba'],target_a.shape,'Invalid B-to-A RF forecast')
    for key in ('z_a','z_b','projected_a','projected_b'):
        _finite(a[key],(len(rows),p['width']),'Invalid latent source coordinates')
    expected_shuffle=np.random.default_rng(2168000).permutation(len(rows))
    _indices(a['shuffle_rows'],expected_shuffle,'Intervention source permutation changed')
    equivalent(record['metrics'],metrics(np.concatenate((a['pred_ab'],a['pred_ba']),1),
                                        np.concatenate((target_b,target_a),1)),'bidirectional RF metrics')
    equivalent(record['ab_metrics'],metrics(a['pred_ab'],target_b),'unperturbed A-to-B RF metrics')
    equivalent(record['representation_z'],representation_metrics(a['z_a'],a['z_b']),'backbone representation metrics')
    equivalent(record['representation_projected'],representation_metrics(a['projected_a'],a['projected_b']),
               'projector representation metrics')
    require(set(record['interventions'])==set(p['interventions']),'Intervention metrics membership differs')
    for kind in p['interventions']:
        value=a['intervention_ab_'+kind]
        _finite(value,target_b.shape,'Invalid stored intervention forecast')
        equivalent(record['interventions'][kind],metrics(value,target_b),'intervention '+kind)
    require(record['intervention_scope']=='all_zero/all_shuffle alter both pooledz andreceiver tokens; measuredpower is held fixed; power_zero tests it separately',
            'Intervention scope changed')
    boundary=c.read_json(Path(str(file)+'.before-targets.json'))
    require(boundary=={'predictions':ref,'prepared_public':c.read_json(c.OUT/'data/receipt.json')['files'][name+'.public'],
                      'source_rows':rows.tolist(),'prediction_complete_before_metric_computation':True},
            'Pre-metric prediction boundary differs')
    timing=record['forward_timing']
    require(type(timing['begin_monotonic_ns']) is int and type(timing['end_monotonic_ns']) is int
            and 0<=timing['begin_monotonic_ns']<=timing['end_monotonic_ns'],
            'Offline forward clock domains/order differ')
    equivalent(timing['seconds'],(timing['end_monotonic_ns']-timing['begin_monotonic_ns'])/1.e9,
               'Offline forward duration')
    require(timing['scope']=='both views, preparedpower input, encoder+projector+decoder; excludes CSI preprocessing/loading/transport; sharedCPU; not realtime',
            'Forward compute scope changed')
    return a


def verify_fit(fit,receipt,public,labels,split):
    import torch
    p=c.protocol();cell,seed=fit['cell'],fit['seed'];run=f'{cell}-seed{seed}'
    require(fit['completed'] is True and cell in CELLS and seed in (0,1,2) and fit['run_id']==run,
            'Fit identity or completion differs')
    identities={'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
                'protocol_sha256':c.digest(c.PROTOCOL),'data_receipt_sha256':c.digest(c.OUT/'data/receipt.json')}
    require(all(fit.get(k)==v for k,v in identities.items()),'Fit source/data/protocol identity changed')
    checkpoint=c.OUT/'checkpoints'/f'{run}.pt'
    require(c.digest(checkpoint)==fit['checkpoint_sha256'],'Checkpoint bytes changed')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(saved['cell']==cell and saved['seed']==seed and saved['identity']==identities
            and saved['config']==fit['config'] and saved['state'],'Selected checkpoint identity changed')
    require(all(isinstance(t,torch.Tensor) and torch.isfinite(t).all() for t in saved['state'].values()),
            'Invalid or nonfinite checkpoint tensors')
    history=fit['history']
    require([r['epoch'] for r in history]==list(range(1,p['epochs']+1))
            and all(math.isfinite(row['train_loss']) and math.isfinite(row['validation_rmse_db'])
                    and row['validation_rmse_db']>=0 for row in history),'Incomplete or nonfinite training history')
    selected=min(history,key=lambda row:row['validation_rmse_db'])
    require(fit['best_epoch']==saved['epoch']==selected['epoch']
            and fit['best_validation_rmse_db']==selected['validation_rmse_db'],
            'Checkpoint was not the first validation-RMSE minimum')
    regularizer='none' if cell=='joint_none' else 'vicreg' if cell=='joint_vicreg' else 'sigreg'
    require(fit['config']['regularizer']==regularizer and fit['config']['decoder']==
            ('unstructured' if cell=='unstructured_sigreg' else 'physical')
            and fit['config']['bins']==p['bins'] and fit['config']['width']==p['width']
            and fit['config']['paths']==p['paths'],'Registered cell architecture differs')
    require(fit['training_rows']==p['split']['common_train']
            and fit['training_labels_are_observed_RF'] is True and fit['private_pose_used_for_encoder_training'] is False,
            'Encoder training scope changed')
    parameters=sum(value.numel() for key,value in saved['state'].items() if not key.startswith('prior.'))
    require(fit['parameters']==parameters,'Inference parameter count changed')
    domains={p['training_recording'],*p['heldout_recordings']}
    require(set(fit['domains'])==domains,'Fit domain coverage changed')
    predictions={name:verify_prediction(record,name,public[name],receipt['recordings'][name],split)
                 for name,record in fit['domains'].items()}
    probe=fit['position_probe']; a=arrays(reference(probe['archive']))
    require(probe['labels_used_after_backbone_frozen'] is True,'Position supervision entered backbone training')
    factory=p['training_recording'];positions=labels[factory][:,:2]
    for key in ('train','validation','test'):
        _indices(a[key+'_rows'],split[key],'Probe source membership changed')
        _finite(a[key+'_z'],(len(split[key]),p['width']),'Malformed frozen probe coordinates')
    close(a['test_z'],predictions[factory]['z_a'],'Probe test coordinates differ from frozen view-A latent',atol=0,rtol=0)
    model=verify_ridge(a,probe['metadata'],a['train_z'],positions[split['train']],
                       a['validation_z'],positions[split['validation']])
    predicted=model.predict(a['test_z'])
    close(a['test_prediction'],predicted,'Stored frozen XY probe prediction differs')
    errors=np.linalg.norm(predicted-positions[split['test']],axis=-1)
    equivalent(fit['domains'][factory]['frozen_position_probe'],
               {'mean_error_m':float(errors.mean()),'median_error_m':float(np.median(errors))},'Frozen XY probe metrics')
    return {'run':run,'domains':len(domains),'frozen_probe_verified':True,
            'neural_forward_recomputed':False,'new_model_or_optimizer_constructed':False}


def verify_baselines(value,receipt,public,labels,split):
    p=c.protocol();factory=p['training_recording'];f=public[factory]
    require(value['source_receipt_sha256']==c.digest(c.OUT/'registration/source.json')
            and value['test_statistics_used_for_fitting'] is False,'Baseline fitting/source scope changed')
    fi=receipt['recordings'][factory];va=np.array(fi['view_a']);vb=np.array(fi['view_b'])
    train,validation,test=split['train'],split['validation'],split['test']
    data_a,data_b=f['power_db'][:,va],f['power_db'][:,vb]
    domains={factory,*p['heldout_recordings']}
    require(set(value['domains'])==domains,'Baseline domain coverage differs')
    rf=value['factory_rf_ridge']
    require(set(rf)=={'ab','ba'},'Factory RF ridge direction membership changed')
    models={}
    for direction,x,y in [('ab',data_a,data_b),('ba',data_b,data_a)]:
        a=arrays(reference(rf[direction]['archive']))
        for key in ('train','validation','test'):
            _indices(a[key+'_rows'],split[key],'RF ridge fitting source rows differ')
        models[direction]=verify_ridge(a,rf[direction]['metadata'],x[train].reshape(len(train),-1),y[train],
                                       x[validation].reshape(len(validation),-1),y[validation])
    for name,entry in value['domains'].items():
        a=arrays(reference({'path':entry['prediction_path'],'sha256':entry['prediction_sha256']}))
        info=receipt['recordings'][name];pub=public[name]
        rows=test if name==factory else np.array(info['eligible_rows'],dtype='i8')
        qa,qb=np.array(info['view_a']),np.array(info['view_b'])
        expected={'source_rows','view_a','view_b','constant_ab','constant_ba','power_level_ab','power_level_ba'}
        if name==factory: expected|={'ridge_ab','ridge_ba'}
        require(set(a)==expected,'Baseline prediction membership differs')
        _indices(a['source_rows'],rows,'Baseline source rows differ')
        _indices(a['view_a'],qa,'Baseline view-A identity differs')
        _indices(a['view_b'],qb,'Baseline view-B identity differs')
        target_a,target_b=pub['power_db'][rows][:,qa],pub['power_db'][rows][:,qb]
        count=len(rows)
        if name==factory:
            ca,cb=mean_template(data_b[train],count),mean_template(data_a[train],count)
            pa=power_level_transfer(data_a[train],data_b[train],target_a)
            pb=power_level_transfer(data_b[train],data_a[train],target_b)
        else:
            # No matching array identity, target mean or label from the OOD
            # recording is used in these global template controls.
            template=f['power_db'][train].mean((0,1),dtype='f8')
            level=float(f['power_db'][train].mean(dtype='f8'))
            ca=np.broadcast_to(template,target_b.shape)
            cb=np.broadcast_to(template,target_a.shape)
            pa=ca+(target_a.astype('f8').mean((1,2))-level)[:,None,None]
            pb=cb+(target_b.astype('f8').mean((1,2))-level)[:,None,None]
        for key,expected_value in [('constant_ab',ca),('constant_ba',cb),('power_level_ab',pa),('power_level_ba',pb)]:
            close(a[key],expected_value,'Constant/power-level control uses nontraining or target information',atol=2e-5,rtol=1e-7)
        names={'constant','power_level'}|({'ridge'} if name==factory else set())
        require(set(entry['metrics'])==names,'Baseline metric membership differs')
        for kind in names:
            if kind=='ridge':
                close(a['ridge_ab'],models['ab'].predict(target_a.reshape(count,-1)),'Factory A-to-B ridge predictions differ')
                close(a['ridge_ba'],models['ba'].predict(target_b.reshape(count,-1)),'Factory B-to-A ridge predictions differ')
            combined=np.concatenate((a[kind+'_ab'],a[kind+'_ba']),axis=1)
            equivalent(entry['metrics'][kind],metrics(combined,np.concatenate((target_b,target_a),axis=1)),
                       'baseline '+name+'/'+kind)
    position=value['factory_position_ridge'];a=arrays(reference(position['archive']))
    inputs=data_a.reshape(len(data_a),-1)
    for key in ('train','validation','test'):
        _indices(a[key+'_rows'],split[key],'Raw RF position-probe source rows differ')
        close(a[key+'_input'],inputs[split[key]],'Raw RF position probe sees more than measured view-A input',atol=0,rtol=0)
    pose=labels[factory][:,:2]
    model=verify_ridge(a,position['metadata'],inputs[train],pose[train],inputs[validation],pose[validation])
    prediction=model.predict(inputs[test])
    close(a['test_prediction'],prediction,'Raw RF position probe prediction differs')
    errors=np.linalg.norm(prediction-pose[test],axis=-1)
    equivalent(position['metrics'],{'mean_error_m':float(errors.mean()),'median_error_m':float(np.median(errors))},
               'Raw RF position-probe metrics')
    return {'domains':len(domains),'training_only_template_and_level_controls':True,
            'factory_RF_ridge_directions':2,'raw_RF_position_probe':True,
            'ridge_selected_solution_checked_by_normal_equations':True,
            'other_alpha_candidate_solutions_refitted':False}


def _visible():
    excluded={c.OUT/'verification/raw_audit.json',c.OUT/'verification/manifest.json'}
    return {str(p.relative_to(c.ROOT)):c.digest(path(str(p.relative_to(c.ROOT)))) for p in sorted(c.OUT.rglob('*'))
            if p.is_file() and p not in excluded and '__pycache__' not in p.parts
            and not any(part.startswith('.') for part in p.relative_to(c.OUT).parts)
            and not p.name.endswith(('.lock','.part','.pyc'))}


def verify_rt_evidence():
    """Check the pipeline's already-traced stages; never invoke a path solver."""
    from . import rt_data as rt
    registry=rt.geometry_registry();identity=rt.fingerprint(registry)
    equivalent(c.read_json(c.OUT/'rt_registry.json'),registry,'Registered RT geometry')
    root=c.OUT/'rt_data';count=0
    require(not (root/'confirmation').exists() and not (root/'confirmation.json').exists(),
            'Reserved confirmation geometries were opened without service qualification')
    for stage in ('training','validation','development'):
        summary=c.read_json(root/(stage+'.json'))
        require(summary['completed'] is True and summary['stage']==stage
                and summary['registry_sha256']==identity and summary['real_device_measurements'] is False,
                'RT stage identity or simulation status differs')
        specs=registry['stages'][stage]
        require([r['scene_id'] for r in summary['scenes']]==[s['scene_id'] for s in specs],
                'RT scene membership differs')
        for spec,declared in zip(specs,summary['scenes']):
            scene=rt.GeometrySpec(**{k:v for k,v in spec.items() if k!='spec_sha256'})
            folder=root/stage/scene.scene_id
            require(c.digest(folder/'receipt.json')==declared['receipt_sha256'],'RT receipt bytes differ')
            receipt=rt.validate_scene_receipt(scene,folder,stage)
            require(receipt['capture_delivery_clock_claimed'] is False
                    and receipt['interference_or_mac_simulated'] is False
                    and receipt['finite_path_search_converged'] is False,
                    'Static finite-path RT capabilities were overstated')
            physical=arrays(folder/'physical_channels.npz');views=arrays(folder/'public_views.npz')
            gain=np.mean(np.abs(physical['cfr'].astype('complex128'))**2,axis=-1)
            close(physical['channel_power_gain'],gain,'RT channel power differs from stored complex CFR')
            signal=np.zeros_like(gain);solved=physical['solved']
            signal[solved]=rt.REFERENCE_TX_DBM+10*np.log10(gain[solved])
            close(physical['signal_dbm'],signal,'RT signal units differ from physical gain')
            profiles=tuple(x for x in rt.DEVICE_PROFILES if not x.held_out) if stage in ('training','validation') else rt.DEVICE_PROFILES
            expected=rt.device_observations(signal,solved,profiles=profiles)
            for key,value in expected.items():close(views[key],value,'RT receiver transformation differs: '+key)
            count+=1
    return {'scene_count':count,'trace_solver_reexecuted':False,'real_device_measurements':False,
            'reserved_confirmation_opened':False,'scope':'stored CIR/CFR/receiver-probe bytes and deterministic transformations; no MAC/service qualification'}


def verify_campaign(*,write=False):
    """Finite full raw pass. Missing evidence raises; nothing is regenerated."""
    c.check_sources();prior=c.verify_previous();p=c.protocol()
    require(tuple(p['cells'])==CELLS and p['training_seeds']==[0,1,2]
            and p['new_fits']==18 and p['epochs']==40,'Registered finite fit matrix changed')
    before=_visible();sources=c.source_hashes()
    receipt,public,labels,split=read_data()
    rt_proof=verify_rt_evidence()
    expected={f'{cell}-seed{seed}' for cell in CELLS for seed in (0,1,2)}
    require({x.stem for x in (c.OUT/'fits').glob('*.json')}==expected,'Fit evidence membership differs')
    require({x.stem for x in (c.OUT/'checkpoints').glob('*.pt')}==expected,'Checkpoint membership differs')
    require({x.stem for x in (c.OUT/'probes').glob('*.npz')}==expected,'Frozen-probe membership differs')
    completed=c.read_json(c.OUT/'training.json')
    require(completed['completed'] is True and completed['fits']==18 and len(completed['runs'])==18
            and set(completed['runs'])==expected
            and completed['source_receipt_sha256']==c.digest(c.OUT/'registration/source.json'),
            'Finite neural-training completion metadata differs')
    fits=[];proof=[]
    for run in sorted(expected):
        item=c.read_json(c.OUT/'fits'/f'{run}.json')
        proof.append(verify_fit(item,receipt,public,labels,split));fits.append(item)
    domains={p['training_recording'],*p['heldout_recordings']}
    expected_predictions={f'{run}-{name}.npz' for run in expected for name in domains}
    require({x.name for x in (c.OUT/'predictions').glob('*.npz')}==expected_predictions,
            'Prediction stream membership differs')
    require({x.name for x in (c.OUT/'predictions').glob('*.before-targets.json')}==
            {name+'.before-targets.json' for name in expected_predictions},'Pre-target receipt membership differs')
    baselines=c.read_json(c.OUT/'baselines.json')
    baseline_proof=verify_baselines(baselines,receipt,public,labels,split)
    report=c.read_json(c.OUT/'report.json')
    equivalent(report['aggregate'],aggregate_fit_reports(fits,baselines),'Final descriptive aggregate')
    for key in ('hardware_qualified','production_qualified','global_sota_claimed','robot_service_qualified','learned_heads_qualified'):
        if key in report: require(report[key] is False,'Unqualified capability was promoted: '+key)
    if 'radiomap_default' in report: require(report['radiomap_default'] is True,'Default controller changed')
    if 'radio_map_default' in report: require(report['radio_map_default'] is True,'Default controller changed')
    require(p['activation']['radiomap_default'] is True and all(p['activation'][key] is False for key in
            ('hardware_qualified','production_qualified','learned_heads_qualified','main_merge')),
            'Research activation contract changed')
    c.check_sources();after=_visible()
    require(before==after and sources==c.source_hashes(),'Evidence or source changed during passive verification')
    result={'schema':1,'passes':True,'neural_fits':18,'paired_prediction_streams':18*len(domains),
        'completed_prior_evidence':prior,'fit_checks':proof,'baseline_checks':baseline_proof,'rt_checks':rt_proof,
        'source_sha256':sources,'evidence_sha256':before,'report_sha256':c.digest(c.OUT/'report.json'),
        'passive':True,'new_neural_models_or_optimizers_constructed':False,
        'new_fits_or_forward_passes_or_downloads':False,'position_labels_used_to_recompute_probe_errors_only':True,
        'hardware_qualified':False,'production_qualified':False,'global_sota_claimed':False,
        'limits':['Checkpoint hashes and numerical evidence are verified; neural forward passes are not replayed.',
                  'Other ridge alpha candidates are source/receipt-bound; selected solution and selected validation error are independently checked without refits.',
                  'Three training seeds are optimization draws, not three extra independent physical sites.',
                  'Referenced power prediction and constructed position probes do not establish robot-service or official benchmark SOTA.']}
    if write:
        c.write_once(c.OUT/'verification/raw_audit.json',result)
    return result


@contextmanager
def passive_guard():
    import torch
    saved=[(torch.nn.Module,'__init__',torch.nn.Module.__init__),
           (torch.optim.Optimizer,'__init__',torch.optim.Optimizer.__init__),
           (np.linalg,'solve',np.linalg.solve)]
    def prohibited(*args,**kwargs):
        raise RuntimeError('Passive verification attempted model/optimizer construction or a new solve/fit')
    try:
        for owner,name,_ in saved:setattr(owner,name,prohibited)
        yield
    finally:
        for owner,name,original in reversed(saved):setattr(owner,name,original)


def run_audit():
    """One finite numerical pass; existing verified raw evidence is never retimed."""
    target=c.OUT/'verification/raw_audit.json'
    if target.exists():
        c.check_sources();value=c.read_json(target)
        require(value['passes'] is True and value['source_sha256']==c.source_hashes()
                and value['evidence_sha256']==_visible()
                and value['report_sha256']==c.digest(c.OUT/'report.json'),
                'Previously audited source/evidence changed')
        return value
    with passive_guard():
        return verify_campaign(write=True)


def seal(documents=()):
    """Bind an already-audited result and optional interpretation documents."""
    c.check_sources();c.verify_previous()
    audit=c.read_json(c.OUT/'verification/raw_audit.json')
    require(audit['passes'] is True and audit['source_sha256']==c.source_hashes()
            and audit['evidence_sha256']==_visible()
            and audit['report_sha256']==c.digest(c.OUT/'report.json'),
            'Raw-audited inputs changed before closure')
    docs={}
    for name in documents:
        p=Path(name)
        if p.is_absolute():p=p.relative_to(c.ROOT)
        p=path(str(p));docs[str(p.relative_to(c.ROOT))]=c.digest(p)
    value={'schema':1,'source_sha256':c.source_hashes(),'evidence_sha256':_visible(),
           'documents_sha256':docs,'raw_audit_sha256':c.digest(c.OUT/'verification/raw_audit.json'),
           'source_receipt_sha256':c.digest(c.OUT/'registration/source.json'),
           'publication_sha256':c.digest(c.OUT/'registration/git.json'),
           'hardware_qualified':False,'production_qualified':False}
    c.write_once(c.OUT/'verification/manifest.json',value)
    return value


def verify_closure():
    """Byte-only verification; never repeat a fit, forward or raw numerical pass."""
    c.check_sources();c.verify_previous();value=c.read_json(c.OUT/'verification/manifest.json')
    require(value['source_sha256']==c.source_hashes() and value['evidence_sha256']==_visible()
            and value['raw_audit_sha256']==c.digest(c.OUT/'verification/raw_audit.json')
            and value['source_receipt_sha256']==c.digest(c.OUT/'registration/source.json')
            and value['publication_sha256']==c.digest(c.OUT/'registration/git.json'),
            'Scene-model immutable byte/membership closure differs')
    require(all(c.digest(path(name))==sha for name,sha in value['documents_sha256'].items()),
            'Scene-model results document changed')
    return {'passes':True,'source_files':len(value['source_sha256']),
            'evidence_files':len(value['evidence_sha256']),'raw_audit_repeated':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='Seal a completed finite campaign after the passive numerical pass')
    parser.add_argument('--check-closure',action='store_true',help='Check an existing byte closure without repeating numerical verification')
    parser.add_argument('--document',action='append',default=[],help='Interpretation document to bind when writing the closure')
    args=parser.parse_args(argv)
    if args.write and args.check_closure: parser.error('Choose a raw audit or a byte-only closure check')
    if args.check_closure:
        value=verify_closure()
    elif args.write:
        value=run_audit();seal(documents=args.document)
    else:
        with passive_guard():value=verify_campaign(write=False)
    print(json.dumps({'passes':value['passes'],'raw_audit_repeated':not args.check_closure,
                      'hardware_qualified':False,'production_qualified':False},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
