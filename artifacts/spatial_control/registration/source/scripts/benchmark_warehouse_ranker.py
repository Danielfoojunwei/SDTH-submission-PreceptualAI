"""A full-priority, group-aware cached-mission graph scorer on fresh scenes.

This separately versioned follow-up repairs limitations of the earlier value
head: it sees explicit group membership and learns the complete soft-priority
ordering. Structural constraints and calibrated service floors stay external.
"""
from pathlib import Path
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import shutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v3.data import verified_parent,episode,ids,observation_roles,DOMAINS
from wirelessjev.warehouse_v3.model import RadioModel
from wirelessjev.warehouse_v3.missions import MISSIONS,compile_contract,evaluate,members
from wirelessjev.warehouse_v3.learning import predict,summarise
from wirelessjev.warehouse_horizon.planning import charge


def features(model,sample,contract,embedding,calibration):
    captured=[]
    handle=model.output.register_forward_pre_hook(lambda module,args:captured.append(args[0][:,0,:,:model.config['hidden']].detach()))
    try:prediction=predict(model,sample)
    finally:handle.remove()
    node=captured[0].numpy();obs=sample['observation'];roles=observation_roles(obs);robot_ids=ids(obs)
    groups=[goal.group for goal in contract.priorities if goal.kind=='group_service']+[floor.group for floor in contract.floors]
    if contract.conditional_floor:groups.append(contract.conditional_floor.group)
    focus=np.zeros(len(robot_ids),bool)
    for group in groups:focus|=members(group,obs,robot_ids,roles)
    if not focus.any():focus[:]=True
    service,risk=charge(prediction,sample['plans'],obs,temperature=calibration['temperature'])
    summary=np.c_[service.mean((1,2)),service.min((1,2)),np.quantile(service,.1,axis=(1,2)),
        service[:,:,focus].mean((1,2)),service[:,:,focus].min((1,2)),risk.mean((1,2)),
        risk[:,:,focus].mean((1,2)),(sample['plans']!=obs['current'][None]).sum(1)/100.]
    kinds=['throughput','min_switches','tail','group_service','critical']
    typed=[]
    goals=list(contract.priorities)
    if contract.conditional_floor:
        mask=members(contract.conditional_floor.group,obs,robot_ids,roles)
        if (obs['robot'][mask,6]<contract.conditional_floor.fraction).any():
            from wirelessjev.warehouse_v3.missions import Goal
            goals.insert(0,Goal('group_service',contract.conditional_floor.group))
    for index in range(4):typed += [float(index<len(goals) and goals[index].kind==kind) for kind in kinds]
    typed += [contract.max_moves/100.,max([f.fraction for f in contract.floors]or[0.]),float(focus.mean())]
    x=np.c_[node.mean(1),node[:,focus].mean(1),summary,np.broadcast_to(typed,(len(node),len(typed))),
        np.broadcast_to(embedding,(len(node),len(embedding)))].astype('float32')
    actual,feasible,_=evaluate(contract,obs,sample['plans'],sample['target']['service'],sample['target']['failure'],robot_ids,roles)
    order=sorted({tuple(np.round(row,8)) for row in actual},reverse=True)
    target=np.array([1-order.index(tuple(np.round(row,8)))/max(1,len(order)-1) for row in actual],dtype='float32')
    _,allowed,_=evaluate(contract,obs,sample['plans'],service,risk,robot_ids,roles,
        floor_service=np.clip(service-calibration['one_sided_service_margin_q90'],0,1))
    return x,target,allowed,feasible,actual,service


def pack(rows,mean=None,std=None,text=True):
    width=max(len(r[0]) for r in rows);dim=rows[0][0].shape[1]
    if mean is None:
        joined=np.concatenate([r[0] for r in rows]);mean=joined.mean(0);std=np.maximum(joined.std(0),.05)
    x=np.zeros((len(rows),width,dim),dtype='float32');target=np.full((len(rows),width),-100.,dtype='float32');mask=np.zeros((len(rows),width),bool)
    for i,row in enumerate(rows):
        value=(row[0]-mean)/std
        if not text:value[:,-1024:]=0.
        x[i,:len(value)]=value;target[i,:len(value)]=row[1];mask[i,:len(value)]=True
    return torch.tensor(x),torch.tensor(target),torch.tensor(mask),mean,std


def fit_head(train,validation,*,seed,text):
    x,y,mask,mean,std=pack(train,text=text);vx,vy,vm,_,_=pack(validation,mean,std,text)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);head=nn.Sequential(nn.Linear(x.shape[-1],64),nn.GELU(),nn.Linear(64,1))
    optimizer=torch.optim.AdamW(head.parameters(),lr=5e-4,weight_decay=1e-3);best=float('inf');chosen=None
    def loss(xx,yy,valid):
        score=head(xx).squeeze(-1).masked_fill(~valid,-1e6)
        target=F.softmax(yy.masked_fill(~valid,-1e6)/.25,dim=1)
        return -(target*F.log_softmax(score,dim=1)).sum(1).mean()
    for epoch in range(30):
        for start in range(0,len(x),64):
            index=rng.integers(len(x),size=min(64,len(x)));objective=loss(x[index],y[index],mask[index])
            optimizer.zero_grad(set_to_none=True);objective.backward();optimizer.step()
        with torch.no_grad():value=float(loss(vx,vy,vm))
        if value<best:best=value;chosen=deepcopy(head.state_dict())
    head.load_state_dict(chosen);head.eval();return head,mean,std,best


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('/workspace/training/warehouse-v3'))
    p.add_argument('--parent',type=Path,default=Path('/workspace/training/warehouse-horizon'));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    repo=Path(__file__).resolve().parents[1];paths=[Path(__file__),*repo.glob('src/wirelessjev/**/*.py')]
    for path in paths:
        dest=root/'source_snapshot'/path.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    atomic_json(root/'provenance.json',{'source_sha256':{str(p.relative_to(repo)):sha256(p) for p in paths},
        'designed_after_parent_evaluation':True,'fresh_test_seeds':list(range(180000,180003))+list(range(181000,181003))+list(range(182000,182003))+list(range(183000,183003)),
        'source':'actual_frozen_graph_inference_and_plan_rank_training','hardware_experiments_executed':False})
    atomic_json(root/'status.json',{'state':'running','stage':'full_priority_group_aware_rankers'})
    ds=verified_parent(args.parent)
    with np.load(args.run/'semantic_contexts.npz',allow_pickle=False) as data:embeddings=dict(zip(data['texts'].tolist(),data['embeddings']))
    selection=json.loads((args.run/'selection.json').read_text());models={};heads={};metadata={}
    for seed in [0,1,2]:
        name=f'graph_scratch-{seed}';base_path=args.run/'checkpoints'/f'{name}.pt'
        if sha256(base_path)!=selection['checkpoint_sha256'][base_path.name]:raise RuntimeError('Frozen graph checkpoint changed')
        saved=torch.load(base_path,map_location='cpu',weights_only=True)
        model=RadioModel(**saved['config']);model.load_state_dict(saved['model']);model.eval();models[seed]=model
        rows={split:[features(model,sample,compile_contract(text),embeddings[text],selection['calibration'][name])
            for sample in ds[split] for text in MISSIONS['train']] for split in ['train','validation']}
        for text in [False,True]:
            head,mu,std,best=fit_head(rows['train'],rows['validation'],seed=seed,text=text)
            heads[seed,text]=(head,mu,std);path=root/'checkpoints'/f'ranker-{seed}-{text}.pt';path.parent.mkdir(exist_ok=True)
            torch.save({'model':head.state_dict(),'mean':torch.tensor(mu),'std':torch.tensor(std),'seed':seed,'text_enabled':text},path)
            metadata[f'{seed}-{text}']={'validation_listwise_loss':best,'checkpoint_sha256':sha256(path),'parameters':sum(p.numel() for p in head.parameters())}
            print('Full-priority ranker',seed,'cached_text',text,'trained',flush=True)
    atomic_json(root/'selection.json',{'models':metadata,'sealed_at':datetime.now(timezone.utc).isoformat(),'fresh_test_accessed':False})
    records=[]
    for index,(domain,(n,mode,scale)) in enumerate(DOMAINS.items()):
        samples=[s for offset in range(3) for s in episode(180000+index*1000+offset,n=n,domain=mode,bulk_scale=scale)]
        for seed,model in models.items():
            for text_enabled in [False,True]:
                head,mu,std=heads[seed,text_enabled];cases=[]
                for sample in samples:
                    obs=sample['observation']
                    for text in MISSIONS['train']+MISSIONS['novel']:
                        c=compile_contract(text);row=features(model,sample,c,embeddings[text],selection['calibration'][f'graph_scratch-{seed}'])
                        xx=(row[0]-mu)/std
                        if not text_enabled:xx[:,-1024:]=0.
                        with torch.no_grad():scores=head(torch.tensor(xx)).squeeze(-1).numpy()
                        allowed=row[2];gold_feasible=row[3];choice=int(np.where(allowed,scores,-np.inf).argmax()) if allowed.any() else None
                        soft=row[4];regret=None
                        if choice is not None and gold_feasible[choice]:
                            order=sorted({tuple(np.round(soft[i],8)) for i in np.flatnonzero(gold_feasible)},reverse=True)
                            regret=order.index(tuple(np.round(soft[choice],8)))/max(1,len(order)-1)
                        gold_indices=np.flatnonzero(gold_feasible)
                        gold_choice=int(max(gold_indices,key=lambda i:tuple(np.round(soft[i],8))+(-int(i),))) if len(gold_indices) else None
                        cases.append({'scene':sample['scene'],'tick':sample['tick'],'text':text,'predicted_feasible':choice is not None,
                            'actual_feasible':choice is not None and bool(gold_feasible[choice]),
                            'false_feasible':choice is not None and not bool(gold_feasible[choice]),
                            'ordinal_regret_when_feasible':regret,'gold_choice':gold_choice,'choice':choice})
                records.append({'domain':domain,'seed':seed,'cached_text':text_enabled,'summary':summarise(cases)})
                atomic_json(root/'cases'/f'{domain}-{seed}-{text_enabled}.json',cases)
        print('Fresh group-aware ranker domain',domain,'complete',flush=True)
    atomic_json(root/'ranker.json',{'source':'simulation_and_actual_cached_d3_embeddings','records':records,
        'selection_sha256':sha256(root/'selection.json'),'model_seeds':[0,1,2],
        'limits':['Fresh environments after the parent comparison; known controlled mission texts are reused',
            'This head predicts the full soft-priority ordering; only structural and predicted-floor guards are exact',
            'Group membership is explicit; no hidden robot-ID lookup or future outcome enters features',
            'Text ablation has identical head architecture; cached embeddings are zeroed in the no-text arm',
            'Offline candidate ranking is not a qualified controller or proof of language-model benefit']})
    atomic_json(root/'status.json',{'state':'completed','report_sha256':sha256(root/'ranker.json')})


if __name__=='__main__':main()
