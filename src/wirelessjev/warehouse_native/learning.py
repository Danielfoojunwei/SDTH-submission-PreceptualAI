"""Fit native packet labels with scene-disjoint validation and frozen tests."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import torch
from torch.nn import functional as F
from ..warehouse_v2.experiment import atomic_json,sha256
from ..warehouse_v2.metrics import risk_metrics,cluster_interval
from .data import utility,worst_decile
from .model import NativeModel


def load_dataset(root,phase):
    root=Path(root);path=root/f'dataset-{phase}.json'
    state=json.loads((root/f'collection-{phase}.json').read_text())
    if state['state']!='completed' or sha256(path)!=state['report_sha256']:raise RuntimeError('Incomplete/changed native dataset')
    dataset=json.loads(path.read_text())
    if dataset['protocol_sha256']!=sha256(root/'protocol.json'):raise RuntimeError('Native protocol changed')
    lookup={(r['scene'],r['unique_plan']):r for r in dataset['outcomes']}
    samples=[]
    for row in dataset['scenes']:
        public=root/row['public']
        if sha256(public)!=row['public_sha256']:raise RuntimeError('Public inputs changed')
        value=json.loads(public.read_text());targets=[]
        for alias in row['aliases']:
            report=lookup[row['scene'],alias]
            if sha256(root/report['path'])!=report['report_sha256']:raise RuntimeError('Native label evidence changed')
            targets.append(report['target'])
        unique=np.array([row['aliases'].index(i) for i in sorted(set(row['aliases']))])
        samples.append({'scene':row['scene'],'split':row['split'],'unique':unique,'names':value['names'],
            'observation':{k:np.asarray(v) for k,v in value['observation'].items()},
            'x':np.asarray(value['features'],dtype='float32'),'groups':np.asarray(value['channel_groups']),
            'plans':np.asarray(value['plans']),'prior':np.asarray(value['prior_prediction'],dtype='float32'),
            'target':np.asarray(targets,dtype='float32')})
    return samples


def normalisation(samples):
    x=np.concatenate([s['x'][s['unique']].reshape(-1,s['x'].shape[-1]) for s in samples])
    return x.mean(0),np.maximum(x.std(0),.05)


def tensors(sample,mean,std,unique=False,with_target=True):
    index=sample['unique'] if unique else np.arange(len(sample['plans']))
    result={key:torch.as_tensor(sample[key][index],dtype=torch.long if key in ['plans','groups'] else torch.float32)
            for key in ['plans','groups','prior']}
    result['x']=torch.as_tensor((sample['x'][index]-mean)/std,dtype=torch.float32)
    if with_target:
        result['target']=torch.as_tensor(sample['target'][index],dtype=torch.float32)
        floor=torch.as_tensor(sample['observation']['floor'],dtype=torch.float32)
        result['failure']=(result['target'][...,1]<floor[None]).float()
    return result


def objective(model,batch):
    result=model(batch['x'],batch['plans'],batch['groups'],batch['prior'])
    fractions=result['fractions'];target=batch['target']
    predicted=fractions.mean((1,2));actual=target.mean((1,2))
    # Compare each action with staying, while retaining per-robot supervision.
    advantage=F.mse_loss(predicted-predicted[0],actual-actual[0])
    return F.mse_loss(fractions,target)+.1*F.binary_cross_entropy_with_logits(result['failure_logits'],batch['failure'])+5.*advantage


def train(samples,root,*,seed,graph,epochs=120):
    train=[s for s in samples if s['split']=='train'];validation=[s for s in samples if s['split']=='validation']
    if not train or not validation:raise ValueError('Disjoint training and validation scenes required')
    mean,std=normalisation(train);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=NativeModel(features=len(mean),graph=graph)
    optimiser=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    batches=[tensors(s,mean,std,True) for s in train];vb=[tensors(s,mean,std,True) for s in validation]
    best=float('inf');chosen=None;history=[];started=time.perf_counter()
    for epoch in range(epochs):
        model.train();losses=[]
        for i in rng.permutation(len(batches)):
            loss=objective(model,batches[i]);optimiser.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimiser.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():score=float(torch.stack([objective(model,b) for b in vb]).mean())
        history.append({'epoch':epoch+1,'train':float(np.mean(losses)),'validation':score})
        if score<best:
            best=score;chosen=deepcopy(model.state_dict());best_epoch=epoch+1
    model.load_state_dict(chosen);model.eval();name=f'{"graph" if graph else "mlp"}-{seed}'
    logits=[];events=[]
    with torch.no_grad():
        for b in vb:
            out=model(b['x'],b['plans'],b['groups'],b['prior'])
            logits.extend(out['failure_logits'].ravel().tolist());events.extend(b['failure'].ravel().tolist())
    logits=np.asarray(logits);events=np.asarray(events)
    def nll(log_temperature):
        z=logits/np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0.,z)-events*z))
    temperature=float(np.exp(minimize_scalar(nll,bounds=(-3,3),method='bounded').x))
    path=root/'checkpoints'/f'{name}.pt';path.parent.mkdir(exist_ok=True)
    torch.save({'model':model.state_dict(),'config':model.config,'mean':torch.tensor(mean),'std':torch.tensor(std),
        'seed':seed,'temperature':temperature,'protocol_sha256':sha256(root/'protocol.json')},path)
    atomic_json(root/f'train-{name}.json',{'name':name,'seed':seed,'epochs':epochs,'best_epoch':best_epoch,
        'validation_loss':best,'temperature':temperature,'wall_s':time.perf_counter()-started,
        'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha256(path),
        'history':history,'test_accessed':False})
    return name


def ridge_fit(samples):
    training=[s for s in samples if s['split']=='train'];validation=[s for s in samples if s['split']=='validation']
    mean,std=normalisation(training)
    x=np.concatenate([s['x'][s['unique']].reshape(-1,len(mean)) for s in training])
    y=np.concatenate([(s['target']-s['prior'])[s['unique']].reshape(-1,2) for s in training])
    x=np.c_[(x-mean)/std,np.ones(len(x))];scores=[];weights=[]
    for alpha in [1.,10.,100.,1000.]:
        penalty=np.eye(x.shape[1])*alpha;penalty[-1,-1]=0.
        w=np.linalg.solve(x.T@x+penalty,x.T@y);weights.append(w);errors=[]
        for s in validation:
            xx=np.concatenate([(s['x']-mean)/std,np.ones((*s['x'].shape[:2],1))],-1)
            p=np.clip(s['prior']+xx@w,0,1)
            errors.append(float(((p[s['unique']]-s['target'][s['unique']])**2).mean()))
        scores.append(float(np.mean(errors)))
    i=int(np.argmin(scores))
    return {'mean':mean.tolist(),'std':std.tolist(),'weights':weights[i].tolist(),'alpha':[1.,10.,100.,1000.][i],
            'validation_mse':scores[i],'fit_source':'training only; alpha selected on validation'}


def load_model(root,name):
    saved=torch.load(root/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=True)
    if saved['protocol_sha256']!=sha256(root/'protocol.json'):raise RuntimeError('Checkpoint protocol mismatch')
    model=NativeModel(**saved['config']);model.load_state_dict(saved['model']);model.eval()
    return model,saved


def predict(root,name,sample,*,loaded=None,ridge=None):
    if name=='numerical':
        p=sample['prior'];floor=sample['observation']['floor'][None]
        probability=np.where(p[...,1]<floor,.98,.02)
    elif name=='ridge':
        mean,std,w=[np.asarray(ridge[k]) for k in ['mean','std','weights']]
        x=np.concatenate([(sample['x']-mean)/std,np.ones((*sample['x'].shape[:2],1))],-1)
        p=np.clip(sample['prior']+x@w,0,1);probability=None
    else:
        model,saved=loaded or load_model(root,name)
        batch=tensors(sample,saved['mean'].numpy(),saved['std'].numpy(),with_target=False)
        with torch.no_grad():result=model(batch['x'],batch['plans'],batch['groups'],batch['prior'])
        p=result['fractions'].numpy();probability=expit(result['failure_logits'].numpy()/saved['temperature'])
    return p,probability


def assess(samples,root,names,ridge):
    records=[]
    for name in names:
        loaded=None if name in ['numerical','ridge'] else load_model(root,name)
        for s in samples:
            started=time.perf_counter();p,probability=predict(root,name,s,loaded=loaded,ridge=ridge)
            elapsed=(time.perf_counter()-started)*1000
            index=int(utility(p).argmax());actual=s['target'];gold=utility(actual)
            unique=s['unique'];y=(actual[unique,:,1]<s['observation']['floor'][None]).astype(float)
            records.append({'method':name,'scene':s['scene'],'split':s['split'],'selected':index,'policy':s['names'][index],
                'utility':float(gold[index]),'candidate_oracle_utility':float(gold.max()),
                'candidate_regret':float(gold.max()-gold[index]),'mean_service':float(actual[index,:,0].mean()),
                'critical_ontime_demand_fraction':float(actual[index,:,1].mean()),
                'worst_decile_service':float(worst_decile(actual[index])),
                'critical_floor_miss_fraction':float((actual[index,:,1]<s['observation']['floor']).mean()),
                'moves':int((s['plans'][index]!=s['observation']['current']).sum()),
                'forecast_rmse':float(np.sqrt(((p[unique]-actual[unique])**2).mean())),
                'inference_ms':elapsed,'risk':risk_metrics(probability[unique],y) if probability is not None else None,
                'forecast':p.tolist()})
    for s in samples:
        for name,index in [('stay',0),('budgeted_map_signal',1),('numerical_tail',3),('candidate_oracle',int(utility(s['target']).argmax()))]:
            actual=s['target'];gold=utility(actual)
            records.append({'method':name,'scene':s['scene'],'split':s['split'],'selected':index,
                'policy':s['names'][index],'utility':float(gold[index]),'candidate_oracle_utility':float(gold.max()),
                'candidate_regret':float(gold.max()-gold[index]),'mean_service':float(actual[index,:,0].mean()),
                'critical_ontime_demand_fraction':float(actual[index,:,1].mean()),
                'worst_decile_service':float(worst_decile(actual[index])),
                'critical_floor_miss_fraction':float((actual[index,:,1]<s['observation']['floor']).mean()),
                'moves':int((s['plans'][index]!=s['observation']['current']).sum())})
    return records


def gate(records,name):
    model={r['scene']:r for r in records if r['method']==name}
    numerical={r['scene']:r for r in records if r['method']=='numerical'}
    scenes=sorted(model)
    diff=np.array([model[s]['utility']-numerical[s]['utility'] for s in scenes])
    interval=cluster_interval(diff,np.array(scenes))
    tail=np.array([model[s]['worst_decile_service']-numerical[s]['worst_decile_service'] for s in scenes])
    critical=np.array([model[s]['critical_ontime_demand_fraction']-numerical[s]['critical_ontime_demand_fraction'] for s in scenes])
    accepted=interval['ci95'] is not None and interval['ci95'][0]>0 and (tail>=-1e-8).all() and (critical>=-1e-8).all()
    return {'accepted':bool(accepted),'utility_difference':interval,'minimum_tail_difference':float(tail.min()),
            'minimum_critical_difference':float(critical.min()),'scenes':scenes,
            'rule':'Positive scene-bootstrap lower utility bound AND no scene worsens tail or critical service'}


def aggregate(records):
    result={}
    for name in dict.fromkeys(r['method'] for r in records):
        rows=[r for r in records if r['method']==name]
        keys=['utility','candidate_regret','mean_service','critical_ontime_demand_fraction','worst_decile_service',
              'critical_floor_miss_fraction','moves','forecast_rmse','inference_ms']
        result[name]={k:float(np.mean([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}
        if all(r.get('risk') is not None for r in rows):
            result[name]['risk_brier']=float(np.mean([r['risk']['brier'] for r in rows]))
            result[name]['risk_ece']=float(np.mean([r['risk']['ece'] for r in rows]))
    return result


def crossed_family_interval(records,prefix,draws=4000):
    """Resample training seeds and scenes; robots are not independent trials."""
    numerical={r['scene']:r['utility'] for r in records if r['method']=='numerical'}
    scenes=sorted(numerical);names=[f'{prefix}-{seed}' for seed in [0,1,2]]
    cells=np.array([[next(r['utility'] for r in records if r['method']==name and r['scene']==s)-numerical[s]
                     for s in scenes] for name in names])
    rng=np.random.default_rng(17);means=[]
    for _ in range(draws):
        seeds=rng.integers(3,size=3);envs=rng.integers(len(scenes),size=len(scenes))
        means.append(float(cells[np.ix_(seeds,envs)].mean()))
    return {'mean':float(cells.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
        'training_seeds':3,'test_scenes':len(scenes),'method':'crossed_training_seed_scene_percentile_bootstrap',
        'limits':'Exploratory small-scene interval; one native RNG run per scene/action, no hardware replication'}
