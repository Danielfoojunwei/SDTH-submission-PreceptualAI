"""Actual pretraining, validation-only calibration and compositional evaluation."""
from __future__ import annotations
from copy import deepcopy
from collections import defaultdict
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import expit

from ..warehouse_horizon.model import tensors,prior_prediction
from ..warehouse_horizon.learning import temperature
from ..warehouse_horizon.planning import charge
from ..warehouse_v2.experiment import atomic_json,sha256
from ..warehouse_v2.metrics import risk_metrics,cluster_interval
from .model import RadioModel,link_features,supervised_loss
from .jepa import prediction_loss,momentum_update
from .missions import MISSIONS,compile_contract,select,evaluate,members
from .data import ids,observation_roles

ARMS=['mlp','graph_scratch','graph_masked','graph_jepa']


def family(contract):
    return f'{bool(contract.floors)}-{bool(contract.conditional_floor)}-{contract.priorities[-1].kind}'


def contract_features(contract,obs):
    kinds=['throughput','min_switches','tail','group_service','critical']
    values=[contract.max_moves/100.,max([f.fraction for f in contract.floors]or[0.]),len(contract.floors)/4.,
        contract.conditional_floor.fraction if contract.conditional_floor else 0.]
    values += [float(contract.priorities[-1].kind==kind) for kind in kinds]
    goal=contract.priorities[-1]
    mask=members(goal.group,obs,ids(obs),observation_roles(obs))
    values += [goal.fraction,float(mask.mean()),float(obs['robot'][mask,0].mean()),
               float(obs['robot'][mask,1].mean()),len(contract.priorities)/4.,
               float(goal.group.kind=='zone'),float(goal.group.kind=='teleoperated')]
    return torch.tensor(values,dtype=torch.float32)


def semantic_contexts(root):
    path=root/'semantic_contexts.npz';receipt=root/'semantic_reader.json'
    texts=MISSIONS['train']+MISSIONS['novel']
    if path.exists():
        report=json.loads(receipt.read_text())
        if report['embedding_sha256']!=sha256(path):raise RuntimeError('Semantic embedding cache changed')
        with np.load(path,allow_pickle=False) as data:
            if data['texts'].tolist()!=texts:raise RuntimeError('Semantic mission list changed')
            return {t:torch.tensor(v) for t,v in zip(texts,data['embeddings'])}
    from ..policy import WirelessJEV,MODEL_REPO,MODEL_REVISION
    from ..radiojev.trainable import TrainableD3
    from .missions import Reader
    engine=WirelessJEV(device='cpu');encoder=TrainableD3(engine,lora_layers=0);reader=Reader(engine)
    embeddings=[];rows=[]
    for text in texts:
        started=time.perf_counter()
        with torch.no_grad():value=encoder.encode('Warehouse operator mission:\n'+text,
            ['Does this mission describe warehouse connectivity priorities?'])[0].cpu().numpy()
        row={'text':text,'embedding_ms':(time.perf_counter()-started)*1000,'model_executed':True}
        try:
            contract,r=reader.read(text);row.update(contract=contract.to_dict(),receipt=r,abstained=False)
        except ValueError as error:row.update(abstained=True,reason=str(error))
        embeddings.append(value);rows.append(row)
        print('Actual d3 compositional context',len(rows),'abstained',row['abstained'],flush=True)
    np.savez_compressed(path,texts=np.array(texts),embeddings=np.stack(embeddings))
    atomic_json(receipt,{'source':'actual_pinned_d3_inference','model':{'repo':MODEL_REPO,'revision':MODEL_REVISION},
        'readings':rows,'embedding_sha256':sha256(path),'probabilities_are_physical_risk':False})
    return {t:torch.tensor(v) for t,v in zip(texts,embeddings)}


def train(train_data,validation,contexts,root,*,seed,arm):
    path=root/'checkpoints'/f'{arm}-{seed}.pt';meta_path=root/f'train-{arm}-{seed}.json'
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if sha256(path)!=meta['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True);model=RadioModel(**checkpoint['config'])
        model.load_state_dict(checkpoint['model']);model.eval();return model,meta
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=RadioModel(graph=arm!='mlp')
    target=deepcopy(model.encoder)
    for parameter in target.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    pretraining=[];started=time.perf_counter();pretrained=arm in ['graph_masked','graph_jepa']
    if pretrained:
        for epoch in range(4):
            losses=[]
            for index in rng.permutation(len(train_data)):
                sample=train_data[index];obs=tensors(sample['observation']);features=link_features(obs)
                if arm=='graph_masked':
                    masked=features.clone();valid=obs['history'][...,1].permute(1,2,0)>0
                    chosen=(torch.rand(valid.shape)<.5)&valid
                    if not chosen.any():continue
                    for step in range(4):
                        masked[...,step*3]=torch.where(chosen[...,step],0.,masked[...,step*3])
                        masked[...,step*3+1]=torch.where(chosen[...,step],0.,masked[...,step*3+1])
                    output=model.pretext(model.encoder(masked)).reshape(*chosen.shape,3)[...,0]
                    expected=features[...,:12].reshape(*chosen.shape,3)[...,0]
                    objective=F.smooth_l1_loss(output[chosen],expected[chosen],beta=.1)
                else:
                    if index+1>=len(train_data) or train_data[index+1]['scene']!=sample['scene']:continue
                    following=train_data[index+1];future=tensors(following['observation'])
                    context=model.encoder(features)
                    action=torch.tensor(following['observation']['current'])
                    moved=(torch.arange(len(obs['ap']))[None]==action[:,None]).float()[...,None]
                    route=(obs['route'][-1]-obs['robot'][:,:2])[:,None].expand(-1,len(obs['ap']),-1)
                    predicted=model.latent_predictor(torch.cat([context,moved,route],-1))
                    with torch.no_grad():latent=target(link_features(future))
                    objective=prediction_loss(predicted,latent)+.1*F.relu(.25-context.flatten(0,1).std(0)).mean()
                optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
                momentum_update(model.encoder,target);losses.append(float(objective.detach()))
            pretraining.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'updates':len(losses)})
            print('Pretraining',arm,seed,epoch+1,pretraining[-1]['loss'],flush=True)
    best=float('inf');selected=None;history=[]
    for epoch in range(8 if pretrained else 12):
        model.train();losses=[]
        for index in rng.permutation(len(train_data)):
            sample=train_data[index];obs=tensors(sample['observation']);plans=torch.tensor(sample['plans'])
            text=MISSIONS['train'][int(rng.integers(len(MISSIONS['train'])))];contract=compile_contract(text)
            output=model(obs,plans,torch.tensor(sample['prior']),contexts[text],contract_features(contract,sample['observation']))
            target_data={k:torch.tensor(v) for k,v in sample['target'].items()}
            objective=supervised_loss(output,target_data)
            priorities,allowed,_=evaluate(contract,sample['observation'],sample['plans'],sample['target']['service'],
                sample['target']['failure'],ids(sample['observation']),observation_roles(sample['observation']))
            last=priorities[:,-1];span=max(float(np.ptp(last)),1e-6)
            numeric=torch.tensor((last-last.mean())/span,dtype=torch.float32)
            objective=objective+.1*F.mse_loss(output['value'],numeric)
            optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            losses.append(float(objective.detach()))
        model.eval()
        with torch.inference_mode():
            val=float(np.mean([float(supervised_loss(model(tensors(s['observation']),torch.tensor(s['plans']),torch.tensor(s['prior'])),
                {k:torch.tensor(v) for k,v in s['target'].items()})) for s in validation]))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':val})
        if val<best:best=val;selected=deepcopy(model.state_dict())
        print('Supervised',arm,seed,epoch+1,'validation',val,flush=True)
    model.load_state_dict(selected);model.eval();path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'config':model.config,'model':model.state_dict(),'seed':seed,'arm':arm},path)
    meta={'seed':seed,'arm':arm,'pretraining':pretraining,'supervised':history,'parameters':sum(p.numel() for p in model.parameters()),
        'checkpoint_sha256':sha256(path),'elapsed_s':time.perf_counter()-started,'pretraining_uses_teacher_labels':False,
        'pretraining_observations_from':'training_scenes_only','source':'actual_simulation_training'}
    atomic_json(meta_path,meta);return model,meta


def predict(model,sample,semantic=None,contract=None):
    if model is None:
        p=prior_prediction(sample['prior'],sample['observation'])
        p['quantiles']=np.stack([np.zeros_like(p['service_raw']),p['service_raw'],np.ones_like(p['service_raw'])],-1)
        return p
    with torch.inference_mode():
        output=model(tensors(sample['observation']),torch.tensor(sample['plans']),torch.tensor(sample['prior']),semantic,
            contract_features(contract,sample['observation']) if semantic is not None else None)
    return {k:v.numpy() for k,v in output.items()}


def calibrate(model,samples):
    logits=[];events=[];errors=[]
    for sample in samples:
        p=predict(model,sample);logits.append(p['failure_logits'].ravel());events.append(sample['target']['failure_raw'].ravel())
        errors.append((p['service_raw']-sample['target']['service_raw']).ravel())
    return {'temperature':temperature(np.concatenate(logits),np.concatenate(events)),
        'one_sided_service_margin_q90':float(max(0.,np.quantile(np.concatenate(errors),.9))),
        'fit_on':'validation_only','marginal_not_simultaneous':True,'physical_coverage_guarantee':False}


def cases(model,samples,contexts,calibration,*,texts=None,value_head=False):
    rows=[];risk_p=[];risk_y=[];squared=[];coverage=[]
    for sample in samples:
        p=predict(model,sample);obs=sample['observation']
        service,risk=charge(p,sample['plans'],obs,temperature=calibration['temperature'])
        # Marginal residual calibration is reported, not treated as a joint
        # chance constraint over all robots/offsets.
        low=np.clip(service-calibration['one_sided_service_margin_q90'],0,1)
        squared.append(float(((service-sample['target']['service'])**2).mean()))
        coverage.append(float(((sample['target']['service_raw']>=p['quantiles'][...,0])&
            (sample['target']['service_raw']<=p['quantiles'][...,2])).mean()))
        risk_p.append(risk.ravel());risk_y.append(sample['target']['failure'].ravel())
        for text in texts or MISSIONS['train']+MISSIONS['novel']:
            c=compile_contract(text);r=observation_roles(obs);robot_ids=ids(obs)
            gold=select(c,obs,sample['plans'],sample['target']['service'],sample['target']['failure'],robot_ids,r)
            learned=None
            if value_head and model is not None:learned=predict(model,sample,contexts[text],c)['value']
            receipt=select(c,obs,sample['plans'],service,risk,robot_ids,r,value_override=learned,floor_service=low)
            index=receipt['choice'];actual_feasible=False;regret=None
            scores,feasible,_=evaluate(c,obs,sample['plans'],sample['target']['service'],sample['target']['failure'],robot_ids,r)
            if index is not None:
                actual_feasible=bool(feasible[index])
                if gold['choice'] is not None and actual_feasible:
                    ordering=sorted({tuple(np.round(scores[i],8)) for i in np.flatnonzero(feasible)},reverse=True)
                    regret=ordering.index(tuple(np.round(scores[index],8)))/max(1,len(ordering)-1)
            rows.append({'scene':sample['scene'],'tick':sample['tick'],'text':text,'family':family(c),
                'new_mission':text in MISSIONS['novel'],'choice':index,'gold_choice':gold['choice'],
                'predicted_feasible':index is not None,'actual_feasible':actual_feasible,
                'false_feasible':index is not None and not actual_feasible,'ordinal_regret_when_feasible':regret,
                'mean_service':None if index is None else float(sample['target']['service'][index].mean()),
                'worst_robot':None if index is None else float(sample['target']['service'][index].mean(0).min()),
                'worst_decile':None if index is None else float(np.sort(sample['target']['service'][index].mean(0))[:max(1,len(robot_ids)//10)].mean()),
                'critical_violation_robot_steps':None if index is None else int(sample['target']['failure'][index].sum()),
                'moves':None if index is None else int((sample['plans'][index]!=obs['current']).sum()),
                'maximum_prediction_shortfall':min(receipt['maximum_service_shortfall_by_candidate'])})
    summary={'service_rmse':float(np.sqrt(np.mean(squared))),'raw_quantile_80_coverage':float(np.mean(coverage)),
        'risk':risk_metrics(np.concatenate(risk_p),np.concatenate(risk_y)),**summarise(rows)}
    return summary,rows


def summarise(rows):
    values=[r['ordinal_regret_when_feasible'] for r in rows if r['ordinal_regret_when_feasible'] is not None]
    scenes=[r['scene'] for r in rows if r['ordinal_regret_when_feasible'] is not None]
    return {'cases':len(rows),'proposal_coverage':float(np.mean([r['predicted_feasible'] for r in rows])),
        'false_feasible_fraction':float(np.mean([r['false_feasible'] for r in rows])),
        'truth_has_feasible_candidate_fraction':float(np.mean([r['gold_choice'] is not None for r in rows])),
        'ordinal_regret_when_feasible':cluster_interval(values,scenes) if values else None}
