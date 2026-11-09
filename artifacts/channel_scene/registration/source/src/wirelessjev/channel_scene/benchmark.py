"""Finite, independently registered real-CSI same-link pilot-completion study.

Acquired prefixes were inspected in prior work. This is a new task, not globally
unseen data. Publication of these exact sources is required before any fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

import numpy as np
import torch

from ..spectral_scene import common as c
from . import data, baselines, model

ROOT=c.ROOT
OUT=ROOT/'artifacts/channel_scene'
WORK=Path(os.environ.get('CHANNEL_SCENE_CACHE_ROOT',str(ROOT.parent/'training/channel-scene')))
PROTOCOL=OUT/'protocol.json'
INVENTORY=OUT/'inventory.json'
PRIOR=ROOT/'artifacts/radiojepa_spectral/verification/manifest.json'
PRIOR_SHA256='53f329a2ce82d6498116fa098b1231ad31c5826be4435b4610c713701bcf598c'
_PUBLISHED=set()


def protocol():return c.read_json(PROTOCOL)


def previous():
    if c.digest(PRIOR)!=PRIOR_SHA256:raise RuntimeError('Preserved scientific manifest changed')
    value=c.read_json(PRIOR)
    for field,count in [('source_sha256',475),('evidence_sha256',5161)]:
        if len(value[field])!=count:raise RuntimeError('Preserved scientific membership changed')
        for name,expected in value[field].items():
            if c.digest(c.safe_path(name))!=expected:raise RuntimeError('Preserved bytes changed: '+name)
    return {'manifest_sha256':PRIOR_SHA256,'sources':475,'evidence':5161}


def source_hashes():
    result={name:c.digest(c.safe_path(name)) for name in c.read_json(PRIOR)['source_sha256']}
    paths=list((ROOT/'src/wirelessjev/channel_scene').glob('*.py'))
    paths+=list((ROOT/'tests').glob('test_channel_scene*.py'))
    paths+=list((ROOT/'docs/channel_scene').glob('*.md'))
    paths+=[PROTOCOL,INVENTORY,OUT/'.gitignore',PRIOR]
    for path in paths:result[str(path.relative_to(ROOT))]=c.digest(path)
    return dict(sorted(result.items()))


def _admission(name,purpose,split):
    p=protocol()
    return data.ObservationAdmission(p['protocol_id'],split,
        p['observation_sha256'][name],purpose=purpose)


def _spatial(positions,guard=.25):
    """Pose-only partition, identical public quantile/guard rule to prior work."""
    xy=np.asarray(positions)[:,:2];low,high=np.quantile(xy[:,1],[.6,.8]);half=guard/2
    train=np.flatnonzero(xy[:,1]<low-half)
    val=np.flatnonzero((xy[:,1]>low+half)&(xy[:,1]<high-half))
    test=np.flatnonzero(xy[:,1]>high+half)
    def near(query,reference):
        if not len(reference):return np.zeros(len(query),bool)
        return (np.square(xy[query,None]-xy[None,reference]).sum(-1)<guard**2).any(-1)
    train=train[~near(train,test)];val=val[~near(val,test)];train=train[~near(train,val)]
    if not len(train) or not len(val) or not len(test):raise RuntimeError('Empty spatial cohort')
    return {'training':train,'validation':val,'cf06_test':test,
        'excluded':np.setdiff1d(np.arange(len(xy)),np.concatenate([train,val,test]))}


def prepare_inventory():
    """Previously observed TRAIN eligibility only; no scientific model or score."""
    observed=data.load_dichasus('dichasus-cf06',_admission('dichasus-cf06','inspection','training'))
    labels=data.private_labels(observed,purpose='partition')['mobile_position_m']
    split=_spatial(labels,protocol()['guard_m'])
    sampled=np.asarray(protocol()['raw_bin_indices'],int)
    pilots=np.r_[protocol()['pilot_a'],protocol()['pilot_b']]
    # Frozen alignment scale sees TRAIN pilots only, no missing target bins.
    h=observed.cfr[split['training']][:,:,sampled[pilots]]
    scale=float(np.sqrt(np.mean(abs(h.astype('c16'))**2)))
    if not np.isfinite(scale) or scale<=0:raise RuntimeError('Invalid TRAIN-pilot scale')
    value={'schema':1,'source_sha256':observed.source_sha256,
        'rows':{k:v.tolist() for k,v in split.items()},'training_pilot_scale':scale,
        'actual_sampled_offsets_hz':observed.frequency_offsets_hz[sampled].tolist(),
        'source_samples':len(observed.cfr),'source_antennas':observed.cfr.shape[1],
        'held_task_scores_computed':False,'scientific_fits_performed':0,
        'poses_used_for_partition_only':True,'previously_inspected_acquisition':True}
    c.write_once(INVENTORY,value);return value


def preflight():
    before=source_hashes();prior=previous()
    command=['uv','run','--frozen','pytest','-q']+sorted(str(p.relative_to(ROOT))
        for p in (ROOT/'tests').glob('test_channel_scene*.py'))
    done=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if done.returncode or before!=source_hashes():raise RuntimeError(done.stdout.decode(errors='replace'))
    log=OUT/'preflight/tests.txt';c.write_bytes_once(log,done.stdout)
    receipt={'exit_code':0,'command':command,'source_sha256':before,'prior':prior,
        'thread_environment':{k:os.environ.get(k) for k in protocol()['thread_limits']},
        'stdout_path':str(log.relative_to(ROOT)),'stdout_sha256':c.digest(log)}
    c.write_once(OUT/'preflight/tests.json',receipt);return receipt


def register():
    with c.lock(OUT/'registration.lock'):
        if (OUT/'registration/source.json').exists():return check_sources(require_published=False)
        prior=previous();sources=source_hashes();tests=c.read_json(OUT/'preflight/tests.json')
        if (tests.get('exit_code')!=0 or tests.get('source_sha256')!=sources or tests.get('prior')!=prior
                or tests['thread_environment']!=protocol()['thread_limits']
                or c.digest(c.safe_path(tests['stdout_path']))!=tests['stdout_sha256']):
            raise RuntimeError('Final fixture proof must bind the exact registered source')
        if any((OUT/name).exists() for name in ('fits','predictions','report.json')):
            raise RuntimeError('Register before scientific fits/predictions/scores')
        capsule=OUT/'registration/source'
        for name,expected in sources.items():
            destination=c.safe_path(name,capsule);destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copyfile(c.safe_path(name),destination)
            if c.digest(destination)!=expected:raise RuntimeError('Source changed during registration')
        value={'schema':1,'source_sha256':sources,'previous':prior,
            'protocol_sha256':c.digest(PROTOCOL),'inventory_sha256':c.digest(INVENTORY),
            'preflight_sha256':{str((OUT/'preflight/tests.json').relative_to(ROOT)):c.digest(OUT/'preflight/tests.json'),
                                tests['stdout_path']:tests['stdout_sha256']},
            'scientific_fits_or_held_task_scores_before_registration':False,
            'previously_inspected_acquisition_disclosed':True}
        c.write_once(OUT/'registration/source.json',value);return value


def check_sources(*,require_published=True):
    receipt=c.read_json(OUT/'registration/source.json')
    if receipt['source_sha256']!=source_hashes() or c.digest(PRIOR)!=PRIOR_SHA256:
        raise RuntimeError('Registered complex-CSI source changed')
    capsule=OUT/'registration/source'
    if {str(p.relative_to(capsule)) for p in capsule.rglob('*') if p.is_file()}!=set(receipt['source_sha256']):
        raise RuntimeError('Complex-CSI source capsule membership changed')
    for name,expected in receipt['source_sha256'].items():
        if c.digest(c.safe_path(name,capsule))!=expected:raise RuntimeError('Source capsule changed')
    for name,expected in receipt['preflight_sha256'].items():
        if c.digest(c.safe_path(name))!=expected:raise RuntimeError('Complex-CSI fixture proof changed')
    if not require_published:return receipt
    publication=c.read_json(OUT/'registration/git.json');commit=publication.get('commit','')
    if (not re.fullmatch('[0-9a-f]{40}',commit) or publication.get('remote_observed_commit')!=commit
            or not publication.get('branch') or not publication.get('remote')
            or publication.get('source_receipt_sha256')!=c.digest(OUT/'registration/source.json')):
        raise RuntimeError('Publish and observe the exact complex-CSI source before fitting/scoring')
    key=(commit,c.value_digest(publication))
    if key not in _PUBLISHED:
        expected=receipt['source_sha256']|receipt['preflight_sha256']
        expected[str((OUT/'registration/source.json').relative_to(ROOT))]=publication['source_receipt_sha256']
        expected.update({str((capsule/name).relative_to(ROOT)):sha for name,sha in receipt['source_sha256'].items()})
        proc=subprocess.Popen(['git','archive','--format=tar',commit,*expected],cwd=ROOT,stdout=subprocess.PIPE)
        actual=set()
        with tarfile.open(fileobj=proc.stdout,mode='r|') as archive:
            for member in archive:
                if member.isdir():continue
                if (not member.isfile() or member.name not in expected
                        or hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=expected[member.name]):
                    raise RuntimeError('Published complex-CSI source differs')
                actual.add(member.name)
        if proc.wait() or actual!=set(expected):raise RuntimeError('Published complex-CSI source incomplete')
        _PUBLISHED.add(key)
    return receipt


def _npz(path,arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);c.write_bytes_once(path,stream.getvalue())
    external=not path.is_relative_to(ROOT)
    return {'storage':'external_workspace' if external else 'repository',
        'path':str(path.relative_to(WORK if external else ROOT)),'sha256':c.digest(path)}


def _npz_read(reference):
    if reference.get('storage') not in ('repository','external_workspace'):raise RuntimeError('Unknown evidence storage')
    path=c.safe_path(reference['path'],WORK if reference['storage']=='external_workspace' else ROOT)
    if c.digest(path)!=reference['sha256']:raise RuntimeError('Complex-CSI numeric evidence changed')
    with np.load(path,allow_pickle=False) as saved:return {k:saved[k].copy() for k in saved.files}


def _cohort(name,stage):
    observed=data.load_dichasus(name,_admission(name,'training' if stage=='training' else 'evaluation',stage if stage in ('training','validation') else 'test'))
    if name=='dichasus-cf06':rows=np.asarray(c.read_json(INVENTORY)['rows'][stage],int)
    else:rows=np.arange(len(observed.cfr))
    values=np.array(observed.cfr[rows][:,:,protocol()['raw_bin_indices']],dtype='c8',copy=True)
    return {'values':values,'carrier':np.repeat(observed.carrier_hz[rows],values.shape[1]),
        'frequencies':observed.frequency_offsets_hz[protocol()['raw_bin_indices']],
        'frames':rows,'source_sha256':observed.source_sha256}


def metrics(predicted,truth):
    """Packet-level mean NMSE; antenna/bin values remain one correlated packet."""
    p=np.asarray(predicted);h=np.asarray(truth)
    if p.shape!=h.shape or p.ndim!=3 or not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError('Finite predictions and truth must share packet/antenna/frequency axes')
    error=abs(p.astype('c16')-h)**2;energy=abs(h.astype('c16'))**2
    denominator=energy.sum((1,2));valid=denominator>0
    if not valid.any():raise ValueError('No nonzero packet target energy')
    nmse=error[valid].sum((1,2))/denominator[valid]
    return {'mean_packet_nmse':float(nmse.mean()),'median_packet_nmse':float(np.median(nmse)),
        'p90_packet_nmse':float(np.quantile(nmse,.9)),
        'global_nmse':float(error[valid].sum()/energy[valid].sum()),'packets':int(valid.sum()),
        'zero_target_packets_excluded':int((~valid).sum()),'case_quantiles_are_not_iid_confidence_intervals':True}


@torch.no_grad()
def _predict(net,values,carriers,pilots,frequencies=None):
    flat=values.reshape(-1,64);result=[];latent=[]
    for start in range(0,len(flat),512):
        x=torch.tensor(flat[start:start+512,pilots]);f=torch.tensor(carriers[start:start+512])
        y,z,_=net(x,pilots,f,frequency_offsets_hz=frequencies);result.append(y.numpy());latent.append(z.numpy())
    return np.concatenate(result).reshape(values.shape),np.concatenate(latent)


def _validation(net,cohort):
    target=protocol()['target'];scores=[]
    for pilots in (protocol()['pilot_a'],protocol()['pilot_b']):
        y,_=_predict(net,cohort['values'],cohort['carrier'],pilots,cohort['frequencies'])
        scores.append(metrics(y[:,:,target],cohort['values'][:,:,target])['mean_packet_nmse'])
    return float(np.mean(scores))


def fit_all():
    check_sources();p=protocol();inventory=c.read_json(INVENTORY)
    if (OUT/'fits/completed.json').exists():return _fits()
    if (OUT/'predictions').exists() or (WORK/'predictions').exists():
        raise RuntimeError('Do not refit after test prediction access or stray partial outputs')
    train=_cohort('dichasus-cf06','training');val=_cohort('dichasus-cf06','validation')
    flat=train['values'].reshape(-1,64);frequency=train['frequencies'];target=p['target']
    fits=[]
    for seed in p['seeds']:
        for arm in ('joint','regression'):
            check_sources();destination=OUT/f'fits/{arm}-{seed}.json'
            if destination.exists():
                saved=c.read_json(destination)
                _validate_fit(saved,arm,seed);fits.append(saved);continue
            torch.manual_seed(seed);net=model.PilotCompletionModel(frequency)
            optimizer=torch.optim.AdamW(net.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
            rng=np.random.default_rng(seed);best=None;best_score=float('inf');history=[];started=time.perf_counter()
            x=torch.tensor(flat);carriers=torch.tensor(train['carrier'])
            for epoch in range(p['epochs']):
                net.train();losses=[];order=rng.permutation(len(x))
                for start in range(0,len(order),p['batch_size']):
                    index=order[start:start+p['batch_size']];optimizer.zero_grad(set_to_none=True)
                    loss,parts=model.objective(net,x[index],carriers[index],p['pilot_a'],p['pilot_b'],target,
                        joint=arm=='joint',training_pilot_scale=inventory['training_pilot_scale'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite registered complex loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
                    losses.append(float(loss.detach()))
                net.eval();score=_validation(net,val)
                history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_packet_nmse':score})
                if score<best_score:best_score=score;best=deepcopy(net.state_dict())
            net.load_state_dict(best);net.eval();_,z=_predict(net,val['values'],val['carrier'],p['pilot_a'])
            weights=_npz(OUT/f'fits/{arm}-{seed}-weights.npz',{k:v.numpy() for k,v in best.items()})
            saved={'arm':arm,'seed':seed,'weights':weights,'source_sha256':source_hashes(),
                'protocol_sha256':c.digest(PROTOCOL),'inventory_sha256':c.digest(INVENTORY),
                'epochs':p['epochs'],'selected_epoch':min(history,key=lambda row:row['validation_packet_nmse'])['epoch'],
                'history':history,'validation_latent':model.latent_diagnostics(z),
                'training_wall_s':time.perf_counter()-started,'held_recordings_opened':False}
            c.write_once(destination,saved);_validate_fit(saved,arm,seed);fits.append(saved)
    # Strong deterministic baselines use TRAIN-only covariance, VAL-only tuning.
    # They are one fit each, never relabelled as three independently trained models.
    best_fourier=None;best_lmmse=None;fourier_candidates=[];lmmse_candidates=[]
    for taps in p['fourier_taps']:
        for ridge in p['fourier_ridge']:
            scores=[]
            for pilots in (p['pilot_a'],p['pilot_b']):
                y=baselines.fourier_ridge(val['values'].reshape(-1,64)[:,pilots],pilots,frequency,n_taps=taps,ridge=ridge)
                scores.append(metrics(y.reshape(val['values'].shape)[:,:,target],val['values'][:,:,target])['mean_packet_nmse'])
            candidate={'n_taps':taps,'ridge':ridge,'validation_packet_nmse':float(np.mean(scores))}
            fourier_candidates.append(candidate)
            if best_fourier is None or candidate['validation_packet_nmse']<best_fourier['validation_packet_nmse']:best_fourier=candidate
    for shrinkage in p['lmmse_shrinkage']:
        for noise in p['lmmse_noise_fraction']:
            fitted=baselines.TrainLMMSE.fit(flat,frequency,shrinkage=shrinkage,noise_fraction=noise);scores=[]
            for pilots in (p['pilot_a'],p['pilot_b']):
                y=fitted.predict(val['values'].reshape(-1,64)[:,pilots],pilots)
                scores.append(metrics(y.reshape(val['values'].shape)[:,:,target],val['values'][:,:,target])['mean_packet_nmse'])
            candidate={'shrinkage':shrinkage,'noise_fraction':noise,'validation_packet_nmse':float(np.mean(scores))}
            lmmse_candidates.append(candidate)
            if best_lmmse is None or candidate['validation_packet_nmse']<best_lmmse['validation_packet_nmse']:best_lmmse=candidate
    selected=baselines.TrainLMMSE.fit(flat,frequency,shrinkage=best_lmmse['shrinkage'],noise_fraction=best_lmmse['noise_fraction'])
    baseline_state=_npz(OUT/'fits/lmmse-state.npz',{'covariance':selected.covariance,
        'frequency_offsets_hz':selected.frequency_offsets_hz,'training_count':np.array(selected.training_count)})
    value={'schema':1,'fits':fits,'fourier':best_fourier,'lmmse':best_lmmse,'baseline_state':baseline_state,
        'fourier_candidates':fourier_candidates,'lmmse_candidates':lmmse_candidates,
        'source_sha256':source_hashes(),'inventory_sha256':c.digest(INVENTORY),
        'fit_receipts_sha256':{f'fits/{r["arm"]}-{r["seed"]}.json':c.digest(OUT/f'fits/{r["arm"]}-{r["seed"]}.json') for r in fits},
        'held_recordings_opened_before_all_fits':False}
    c.write_once(OUT/'fits/completed.json',value);return value


def _validate_fit(row,arm,seed):
    p=protocol();reference=row.get('weights',{})
    if (row.get('arm')!=arm or row.get('seed')!=seed or row.get('epochs')!=p['epochs']
            or row.get('source_sha256')!=source_hashes()
            or row.get('protocol_sha256')!=c.digest(PROTOCOL)
            or row.get('inventory_sha256')!=c.digest(INVENTORY)
            or row.get('held_recordings_opened') is not False
            or reference.get('storage')!='repository'
            or reference.get('path')!=str((OUT/f'fits/{arm}-{seed}-weights.npz').relative_to(ROOT))):
        raise RuntimeError('Cached fit provenance or canonical checkpoint identity differs')
    history=row.get('history',[])
    if (len(history)!=p['epochs'] or [r.get('epoch') for r in history]!=list(range(1,p['epochs']+1))
            or any(not np.isfinite(r.get(key,np.nan)) or r[key]<0 for r in history for key in ('loss','validation_packet_nmse'))
            or row.get('selected_epoch')!=min(history,key=lambda r:r['validation_packet_nmse'])['epoch']
            or not np.isfinite(row.get('training_wall_s',np.nan)) or row['training_wall_s']<0):
        raise RuntimeError('Cached fit must retain the finite complete budget and first-minimum VAL selection')
    weights=_npz_read(reference);frequency=c.read_json(INVENTORY)['actual_sampled_offsets_hz']
    with torch.random.fork_rng():expected=model.PilotCompletionModel(frequency).state_dict()
    if set(weights)!=set(expected) or any(weights[k].shape!=tuple(v.shape) or not np.isfinite(weights[k]).all() for k,v in expected.items()):
        raise RuntimeError('Checkpoint architecture or finite numeric state differs')
    for key in ('frequencies_hz','canonical_delay_s','fourier_basis'):
        if not np.array_equal(weights[key],expected[key].numpy()):raise RuntimeError('Checkpoint physical grid changed')
    return weights


def _fits():
    value=c.read_json(OUT/'fits/completed.json');p=protocol()
    expected={(arm,seed) for arm in ('joint','regression') for seed in p['seeds']}
    if (value['source_sha256']!=source_hashes() or value['inventory_sha256']!=c.digest(INVENTORY)
            or len(value['fits'])!=len(expected)
            or {(r['arm'],r['seed']) for r in value['fits']}!=expected
            or value.get('held_recordings_opened_before_all_fits') is not False):
        raise RuntimeError('All exact registered fits must be bound before test completion')
    for row in value['fits']:
        _validate_fit(row,row['arm'],row['seed'])
        path=f'fits/{row["arm"]}-{row["seed"]}.json'
        if (c.read_json(OUT/path)!=row or value.get('fit_receipts_sha256',{}).get(path)!=c.digest(OUT/path)):
            raise RuntimeError('Complete fit receipt differs from its immutable per-run artifact')
    if set(value.get('fit_receipts_sha256',{}))!={f'fits/{a}-{s}.json' for a,s in expected}:
        raise RuntimeError('Fit receipt membership changed')
    for name,fields in [('fourier',{'n_taps':p['fourier_taps'],'ridge':p['fourier_ridge']}),
                        ('lmmse',{'shrinkage':p['lmmse_shrinkage'],'noise_fraction':p['lmmse_noise_fraction']})]:
        selected=value.get(name,{})
        if (any(selected.get(k) not in allowed for k,allowed in fields.items())
                or not np.isfinite(selected.get('validation_packet_nmse',np.nan)) or selected['validation_packet_nmse']<0):
            raise RuntimeError('Baseline hyperparameters must belong to the registered finite VAL grid')
        candidates=value.get(name+'_candidates',[]);keys=list(fields)
        ordered=[(a,b) for a in fields[keys[0]] for b in fields[keys[1]]]
        if ([(r.get(keys[0]),r.get(keys[1])) for r in candidates]!=ordered
                or any(not np.isfinite(r.get('validation_packet_nmse',np.nan)) or r['validation_packet_nmse']<0 for r in candidates)
                or selected!=min(candidates,key=lambda r:r['validation_packet_nmse'])):
            raise RuntimeError('Baseline choice must be the first minimum of every registered VAL candidate')
    _baseline_state(value)
    return value


def _baseline_state(fits):
    reference=fits.get('baseline_state',{})
    if reference.get('storage')!='repository' or reference.get('path')!=str((OUT/'fits/lmmse-state.npz').relative_to(ROOT)):
        raise RuntimeError('Baseline must load its canonical pre-test TRAIN fitted state')
    saved=_npz_read(reference);inventory=c.read_json(INVENTORY)
    if (set(saved)!={'covariance','frequency_offsets_hz','training_count'}
            or saved['covariance'].shape!=(64,64) or saved['covariance'].dtype.kind!='c'
            or not np.isfinite(saved['covariance']).all()
            or not np.allclose(saved['covariance'],saved['covariance'].conj().T,atol=1.e-10,rtol=1.e-10)
            or not np.array_equal(saved['frequency_offsets_hz'],inventory['actual_sampled_offsets_hz'])
            or saved['training_count'].shape!=() or int(saved['training_count'])!=len(inventory['rows']['training'])*32):
        raise RuntimeError('Baseline covariance/grid/TRAIN cardinality changed')
    return baselines.TrainLMMSE(saved['frequency_offsets_hz'],saved['covariance'],
        int(saved['training_count']),fits['lmmse']['shrinkage'],fits['lmmse']['noise_fraction'])


def predict_all():
    check_sources();fits=_fits();p=protocol();train=_cohort('dichasus-cf06','training')
    if (OUT/'predictions/completed.json').exists():return _prediction_receipt()
    if (OUT/'predictions').exists() or (WORK/'predictions').exists():
        raise RuntimeError('Stray partial predictions require audit; do not silently regenerate test outputs')
    flat=train['values'].reshape(-1,64);frequency=train['frequencies']
    fitted=_baseline_state(fits)
    references={}
    for name,stage in [('dichasus-cf06','cf06_test'),('dichasus-ad00','test'),('dichasus-dc01','test')]:
        check_sources();cohort=_cohort(name,stage);v=cohort['values'];flat=v.reshape(-1,64);arrays={}
        actual_frequency=cohort['frequencies']
        for view,pilots in [('a',p['pilot_a']),('b',p['pilot_b'])]:
            for label,y in [('linear',baselines.linear_interpolate(flat[:,pilots],pilots,actual_frequency)),
                            ('fourier',baselines.fourier_ridge(flat[:,pilots],pilots,actual_frequency,
                                n_taps=fits['fourier']['n_taps'],ridge=fits['fourier']['ridge'])),
                            ('lmmse',fitted.predict(flat[:,pilots],pilots))]:
                arrays[label+'_'+view]=y.reshape(v.shape)[:,:,p['target']].astype('c8')
            for row in fits['fits']:
                net=model.PilotCompletionModel(frequency)
                net.load_state_dict({k:torch.tensor(value) for k,value in _npz_read(row['weights']).items()});net.eval()
                y,_=_predict(net,v,cohort['carrier'],pilots,actual_frequency)
                arrays[f"{row['arm']}_{row['seed']}_{view}"]=y[:,:,p['target']].astype('c8')
        arrays['frames']=cohort['frames']
        members={key:_npz(WORK/f'predictions/{name}/{key}.npz',{'value':value}) for key,value in arrays.items()}
        references[name]={'members':members,'source_sha256':cohort['source_sha256'],
            'target_shape':[v.shape[0],v.shape[1],len(p['target'])]}
    receipt={'schema':1,'predictions':references,'source_sha256':source_hashes(),
        'fit_receipt_sha256':c.digest(OUT/'fits/completed.json'),
        'task_scores_computed_before_prediction_receipt':False,
        'disclosure':'Acquired task labels share source NPZ with pilots; test values were previously inspected. No globally unseen-data claim.'}
    c.write_once(OUT/'predictions/completed.json',receipt);return receipt


def _prediction_receipt():
    receipt=c.read_json(OUT/'predictions/completed.json');p=protocol()
    expected={f'{label}_{view}' for label in ('linear','fourier','lmmse') for view in ('a','b')}
    expected|={f'{arm}_{seed}_{view}' for arm in ('joint','regression') for seed in p['seeds'] for view in ('a','b')}
    if (receipt.get('source_sha256')!=source_hashes()
            or receipt.get('fit_receipt_sha256')!=c.digest(OUT/'fits/completed.json')
            or receipt.get('task_scores_computed_before_prediction_receipt') is not False
            or set(receipt.get('predictions',{}))!={'dichasus-cf06','dichasus-ad00','dichasus-dc01'}):
        raise RuntimeError('Predictions must bind every registered domain and fit before scoring')
    for name,row in receipt['predictions'].items():
        if set(row.get('members',{}))!=expected|{'frames'}:
            raise RuntimeError('Prediction receipt must include every exact model/seed/view member')
        if row.get('source_sha256')!=p['observation_sha256'][name]:raise RuntimeError('Prediction raw source identity differs')
        count=len(c.read_json(INVENTORY)['rows']['cf06_test']) if name=='dichasus-cf06' else 1024
        antennas=32 if name=='dichasus-cf06' else 64
        if row.get('target_shape')!=[count,antennas,32]:raise RuntimeError('Prediction target shape changed')
        for key,ref in row['members'].items():
            if ref.get('storage')!='external_workspace' or ref.get('path')!=f'predictions/{name}/{key}.npz':
                raise RuntimeError('Prediction member has a noncanonical cache identity')
            member=_npz_read(ref)
            shape=(count,) if key=='frames' else (count,antennas,32)
            if set(member)!={'value'} or member['value'].shape!=shape or not np.isfinite(member['value']).all():
                raise RuntimeError('Prediction cache keys, shapes or values changed')
    return receipt


def evaluate():
    check_sources();_fits();receipt=_prediction_receipt();p=protocol()
    cases={}
    for name,reference in receipt['predictions'].items():
        cohort=_cohort(name,'cf06_test' if name=='dichasus-cf06' else 'test')
        if reference['source_sha256']!=cohort['source_sha256']:raise RuntimeError('Test observations changed')
        frames=_npz_read(reference['members']['frames'])['value']
        if not np.array_equal(frames,cohort['frames']):raise RuntimeError('Prediction packet identities changed')
        truth=cohort['values'][:,:,p['target']]
        cases[name]={key:metrics(_npz_read(ref)['value'],truth) for key,ref in reference['members'].items() if key!='frames'}
    summary={}
    for name,results in cases.items():
        summary[name]={}
        for arm in ('joint','regression'):
            seeds=[float(np.mean([results[f'{arm}_{seed}_{view}']['mean_packet_nmse'] for view in ('a','b')])) for seed in p['seeds']]
            summary[name][arm]={'seed_mean_packet_nmse':float(np.mean(seeds)),
                'seed_std_packet_nmse':float(np.std(seeds,ddof=1)),'individual_seed_nmse':seeds}
        for baseline in ('linear','fourier','lmmse'):
            summary[name][baseline]={'mean_packet_nmse':float(np.mean([results[baseline+'_'+view]['mean_packet_nmse'] for view in ('a','b')])),
                'deterministic_reference_not_independent_training_seeds':True}
    report={'schema':1,'task':p['task'],'summary':summary,'cases':cases,
        'source_sha256':source_hashes(),'predictions_receipt_sha256':c.digest(OUT/'predictions/completed.json'),
        'limitations':p['limitations'],'production_qualified':False,'physical_scatterer_identification_claimed':False}
    c.write_once(OUT/'report.json',report);return report


def run():
    for name,value in protocol()['thread_limits'].items():
        if os.environ.get(name)!=value:raise RuntimeError('Use registered single-thread numerical environment')
    with c.lock(OUT/'run.lock',nonblocking=True):
        check_sources();previous();fit_all();predict_all();result=evaluate();audit();seal();previous();return result


def audit():
    """Replay predictions from actual pilot bytes/frozen weights, never refit."""
    check_sources();fits=_fits();receipt=_prediction_receipt();p=protocol()
    observed=data.load_dichasus('dichasus-cf06',_admission('dichasus-cf06','inspection','training'))
    inventory=c.read_json(INVENTORY);split=_spatial(data.private_labels(observed,purpose='partition')['mobile_position_m'],p['guard_m'])
    if any(not np.array_equal(indices,inventory['rows'][key]) for key,indices in split.items()):
        raise RuntimeError('Pose-only spatial partition does not reconstruct')
    pilots=np.r_[p['pilot_a'],p['pilot_b']];sampled=np.asarray(p['raw_bin_indices'])
    scale=float(np.sqrt(np.mean(abs(observed.cfr[split['training']][:,:,sampled[pilots]].astype('c16'))**2)))
    if (not np.isclose(scale,inventory['training_pilot_scale'],rtol=1.e-12,atol=0)
            or not np.array_equal(observed.frequency_offsets_hz[sampled],inventory['actual_sampled_offsets_hz'])
            or observed.source_sha256!=inventory['source_sha256']):
        raise RuntimeError('TRAIN-pilot scale/frequency/source inventory does not reconstruct')
    del observed
    train=_cohort('dichasus-cf06','training');val=_cohort('dichasus-cf06','validation');flat=train['values'].reshape(-1,64)
    fitted=_baseline_state(fits)
    # Direct sufficient-statistic reconstruction audits the saved TRAIN fit;
    # it does not tune or replace state after test access.
    covariance=flat.astype('c16').T@flat.astype('c16').conj()/len(flat)
    covariance=(covariance+covariance.conj().T)/2
    shrink=fits['lmmse']['shrinkage']
    covariance=(1-shrink)*covariance+shrink*np.diag(covariance.diagonal().real)
    if not np.allclose(covariance,fitted.covariance,rtol=1.e-12,atol=1.e-12):
        raise RuntimeError('Saved covariance does not reconstruct from registered TRAIN bytes')
    validation_proofs={}
    unshrunk=flat.astype('c16').T@flat.astype('c16').conj()/len(flat)
    unshrunk=(unshrunk+unshrunk.conj().T)/2
    for name in ('fourier','lmmse'):
        validation_proofs[name]=[]
        for candidate in fits[name+'_candidates']:
            scores=[]
            for pilots in (p['pilot_a'],p['pilot_b']):
                x=val['values'].reshape(-1,64)[:,pilots]
                if name=='fourier':
                    y=baselines.fourier_ridge(x,pilots,val['frequencies'],n_taps=candidate['n_taps'],ridge=candidate['ridge'])
                else:
                    s=candidate['shrinkage'];cov=(1-s)*unshrunk+s*np.diag(unshrunk.diagonal().real)
                    reference=baselines.TrainLMMSE(train['frequencies'],cov,len(flat),s,candidate['noise_fraction'])
                    y=reference.predict(x,pilots)
                scores.append(metrics(y.reshape(val['values'].shape)[:,:,p['target']],val['values'][:,:,p['target']])['mean_packet_nmse'])
            score=float(np.mean(scores))
            if not np.isclose(score,candidate['validation_packet_nmse'],rtol=1.e-10,atol=1.e-12):
                raise RuntimeError('Registered VAL candidate score does not reconstruct')
            validation_proofs[name].append(dict(candidate))
    for row in fits['fits']:
        net=model.PilotCompletionModel(train['frequencies'])
        net.load_state_dict({k:torch.tensor(value) for k,value in _validate_fit(row,row['arm'],row['seed']).items()});net.eval()
        score=_validation(net,val);selected=row['history'][row['selected_epoch']-1]['validation_packet_nmse']
        if not np.isclose(score,selected,rtol=5.e-6,atol=1.e-7):
            raise RuntimeError('Selected checkpoint does not reconstruct its registered VAL score')
    proofs={}
    for name,reference in receipt['predictions'].items():
        cohort=_cohort(name,'cf06_test' if name=='dichasus-cf06' else 'test')
        values=cohort['values'];x=values.reshape(-1,64);f=cohort['frequencies'];proofs[name]={}
        for view,pilots in [('a',p['pilot_a']),('b',p['pilot_b'])]:
            expected={'linear':baselines.linear_interpolate(x[:,pilots],pilots,f),
                'fourier':baselines.fourier_ridge(x[:,pilots],pilots,f,n_taps=fits['fourier']['n_taps'],ridge=fits['fourier']['ridge']),
                'lmmse':fitted.predict(x[:,pilots],pilots)}
            for row in fits['fits']:
                net=model.PilotCompletionModel(train['frequencies'])
                net.load_state_dict({k:torch.tensor(value) for k,value in _validate_fit(row,row['arm'],row['seed']).items()});net.eval()
                y,_=_predict(net,values,cohort['carrier'],pilots,f)
                expected[f"{row['arm']}_{row['seed']}"]=y.reshape(-1,64)
            for label,y in expected.items():
                key=label+'_'+view;actual=_npz_read(reference['members'][key])['value']
                reconstructed=y.reshape(values.shape)[:,:,p['target']].astype('c8')
                if not np.allclose(actual,reconstructed,rtol=5.e-6,atol=1.e-7):
                    raise RuntimeError('Prediction fails raw-pilot/checkpoint replay: '+name+'/'+key)
                proofs[name][key]={'maximum_abs_error':float(abs(actual-reconstructed).max()),
                    'raw_pilots_only_at_inference':True,'checkpoint_or_TRAIN_state_replayed':True}
    existing=c.read_json(OUT/'report.json');recomputed=evaluate()
    if existing!=recomputed:raise RuntimeError('Report does not reconstruct from bound predictions and actual task labels')
    value={'schema':1,'passes':True,'source_sha256':source_hashes(),'prediction_replay':proofs,
        'train_covariance_reconstructed':True,'pose_split_and_train_pilot_scale_reconstructed':True,
        'all_registered_validation_candidates_reconstructed':validation_proofs,
        'selected_neural_validation_scores_reconstructed':True,'report_sha256':c.digest(OUT/'report.json'),
        'predictions_receipt_sha256':c.digest(OUT/'predictions/completed.json')}
    c.write_once(OUT/'verification/raw.json',value);return value


def _visible_evidence():
    return {str(path.relative_to(ROOT)):c.digest(path) for path in sorted(OUT.rglob('*'))
        if path.is_file() and path.suffix not in ('.lock','.part')
        and path!=OUT/'verification/manifest.json' and not path.is_relative_to(OUT/'registration/source')}


def seal():
    check_sources();proof=c.read_json(OUT/'verification/raw.json')
    if (proof.get('passes') is not True or proof['source_sha256']!=source_hashes()
            or proof['report_sha256']!=c.digest(OUT/'report.json')
            or proof['predictions_receipt_sha256']!=c.digest(OUT/'predictions/completed.json')):
        raise RuntimeError('Raw-pilot prediction replay must precede closure')
    receipt=_prediction_receipt()
    external={ref['path']:ref['sha256'] for row in receipt['predictions'].values() for ref in row['members'].values()}
    manifest={'schema':1,'cache_root_environment_variable':'CHANNEL_SCENE_CACHE_ROOT',
        'source_sha256':source_hashes(),'external_prediction_sha256':external,
        'prediction_receipt':receipt,'raw_audit':proof,
        'missing_shards_are_fatal':True,'self_contained_git_PR':False,
        'restore':'Extract the capsule channel-scene-predictions directory; point CHANNEL_SCENE_CACHE_ROOT to it.'}
    c.write_once(WORK/'deliverables/cache_manifest.json',manifest)
    capsule=WORK/'deliverables/channel-scene-predictions.tar.gz'
    if not capsule.exists():
        capsule.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(capsule,'x:gz',compresslevel=1) as archive:
            files={name:c.safe_path(name,WORK) for name in external}
            files['deliverables/cache_manifest.json']=WORK/'deliverables/cache_manifest.json'
            for name,path in sorted(files.items()):
                info=tarfile.TarInfo('channel-scene-predictions/'+name)
                info.size=path.stat().st_size;info.mode=0o600;info.mtime=0
                with path.open('rb') as stream:archive.addfile(info,stream)
    # A pre-existing deliverable must contain exactly the registered bytes;
    # partial/interrupted archives never receive a successful seal.
    expected_capsule={name:sha for name,sha in external.items()}
    expected_capsule['deliverables/cache_manifest.json']=c.digest(WORK/'deliverables/cache_manifest.json')
    found=set()
    with tarfile.open(capsule,'r:gz') as archive:
        for member in archive:
            name=member.name.removeprefix('channel-scene-predictions/')
            if not member.isfile() or name not in expected_capsule or name in found:
                raise RuntimeError('Prediction deliverable membership changed')
            digest=hashlib.sha256()
            stream=archive.extractfile(member)
            for block in iter(lambda:stream.read(1<<20),b''):
                digest.update(block)
            if digest.hexdigest()!=expected_capsule[name]:raise RuntimeError('Prediction deliverable bytes changed')
            found.add(name)
    if found!=set(expected_capsule):raise RuntimeError('Prediction deliverable incomplete')
    value={'schema':1,'source_sha256':source_hashes(),'evidence_sha256':_visible_evidence(),
        'external_prediction_sha256':external,'previous':previous(),'passes':True,
        'cache_manifest':{'path':'deliverables/cache_manifest.json','sha256':c.digest(WORK/'deliverables/cache_manifest.json')},
        'prediction_capsule':{'path':'deliverables/channel-scene-predictions.tar.gz','sha256':c.digest(capsule),'bytes':capsule.stat().st_size},
        'git_PR_self_contained_without_prediction_capsule':False}
    c.write_once(OUT/'verification/manifest.json',value);return value


def verify_closure():
    check_sources();value=c.read_json(OUT/'verification/manifest.json')
    if value['source_sha256']!=source_hashes() or value['evidence_sha256']!=_visible_evidence() or value['previous']!=previous():
        raise RuntimeError('Complex completion closure changed')
    for name,sha in value['external_prediction_sha256'].items():
        if c.digest(c.safe_path(name,WORK))!=sha:raise RuntimeError('External cache changed')
    for key in ('cache_manifest','prediction_capsule'):
        ref=value[key]
        if c.digest(c.safe_path(ref['path'],WORK))!=ref['sha256']:raise RuntimeError('Prediction deliverable changed or missing')
    return {'passes':True,'source_files':len(value['source_sha256']),'evidence_files':len(value['evidence_sha256']),
        'external_prediction_shards':len(value['external_prediction_sha256']),'production_qualified':False}


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('inventory','preflight','register','run','evaluate','audit','check-closure'))
    args=parser.parse_args(argv)
    value={'inventory':prepare_inventory,'preflight':preflight,'register':register,'run':run,'evaluate':evaluate,
           'audit':audit,'check-closure':verify_closure}[args.phase]()
    print(c.encoded(value).decode())


if __name__=='__main__':main()
