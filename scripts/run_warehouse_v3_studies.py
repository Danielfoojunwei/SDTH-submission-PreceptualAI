"""Execute sensing, true shared-contribution removal and packet-telemetry studies."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import expit

from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v2.adaptation import state_digest,auc
from wirelessjev.warehouse_v2.metrics import risk_metrics
from wirelessjev.warehouse_horizon.world import World,observe
from wirelessjev.warehouse_horizon.belief import RadioBelief
from wirelessjev.warehouse_horizon.model import tensors,prior_prediction
from wirelessjev.warehouse_horizon.planning import Contract,candidate_pool,values,charge
from wirelessjev.warehouse_v3.model import RadioModel,supervised_loss
from wirelessjev.warehouse_v3.data import episode
from wirelessjev.warehouse_v3.learning import predict
from wirelessjev.measurement import packet_features

REPO=Path(__file__).resolve().parents[1]


def sensing():
    rows=[]
    for seed in range(161000,161006):
        world=World(seed,steps=12);history=[];n=100;current=world.initial;previous=np.zeros(n)
        for tick in range(4):
            obs=observe(world.frames[tick],current,tick,previous_service=previous,since_switch=np.full(n,30.),history=history)
            previous,_,_=world.interval(tick,current,current)
        rng=np.random.default_rng(seed+3000);initial=rng.choice(n,12,replace=False)
        uniform=rng.permutation(np.setdiff1d(np.arange(n),initial))
        for policy in ['passive','uniform','decision_aware']:
            selected=list(initial);cost_robots=[]
            for budget in ([0] if policy=='passive' else [0,10,20]):
                while len(cost_robots)<budget:
                    public=deepcopy(obs);available=np.zeros((4,n),bool);available[:,initial]=True;available[-1,cost_robots]=True
                    public['scan_available']=available
                    for i in range(4):public['history'][i,~available[i]]=0.
                    belief=RadioBelief(public);remaining=np.setdiff1d(np.arange(n),selected)
                    if policy=='uniform':robot=int(next(i for i in uniform if i not in selected))
                    else:
                        # Public belief-only heuristic: uncertainty can justify
                        # a scan even when no target is currently visible.
                        mean=belief.mean.mean(0);std=belief.std.mean(0)
                        incumbent=mean[np.arange(n),current]
                        potential=np.maximum(0.,(mean+1.64*std).max(1)-incumbent)
                        importance=1+obs['critical']+obs['bulk']
                        score=potential*std.mean(1)*importance
                        robot=int(remaining[np.argmax(score[remaining])])
                    selected.append(robot);cost_robots.append(robot)
                public=deepcopy(obs);available=np.zeros((4,n),bool);available[:,initial]=True;available[-1,cost_robots]=True
                public['scan_available']=available
                for i in range(4):public['history'][i,~available[i]]=0.
                belief=RadioBelief(public);plans,_=candidate_pool(public,belief,Contract('throughput',12,1.))
                prediction=prior_prediction(belief.forecast(plans),public);service,risk=charge(prediction,plans,public)
                chosen=int(values(Contract('throughput',12,1.),public,service,risk).argmax())
                target=world.outcomes(3,plans,current);truth_score=values(Contract('throughput',12,1.),obs,target['service'],target['failure'])
                for duration in [.02,.1,.5]:
                    charged=target['service'][chosen].copy();interruption=4.*(plans[chosen]!=current)
                    interruption[cost_robots]+=duration
                    for k in range(6):charged[k]=target['service_raw'][chosen,k]*np.clip(1-(interruption-5*k)/5,0,1)
                    # Recompute per-flow violations at the private simulator
                    # boundary; the acquisition never receives these arrays.
                    failed=[]
                    for k in range(6):
                        ev=world.frames[4+k].evaluator(current)
                        terms=ev._snapshot_terms(plans[chosen:chosen+1],ev.snaps[0],False,5.)
                        keep=np.clip(1-(interruption-5*k)/5,0,1)
                        failed.append(((ev.cf_dem*terms['cf'][0,:,None]*keep[:,None]<ev.cf_floor-1e-9)&(ev.cf_dem>0)).any(1))
                    failed=np.stack(failed)
                    utility=float(values(Contract('throughput',12,1.),obs,charged[None],failed[None])[0])
                    truth_dl=world.frames[4].scene.radio().dl-world.frames[4].blockage
                    audible=truth_dl>=-90
                    rows.append({'scene':seed,'policy':policy,'extra_scans':budget,'initial_robots':12,'initial_capped_scans':48,
                        'assumed_scan_seconds':duration,'selected_robots':list(cost_robots),'source':'simulation',
                        'mean_service_after_scan_cost':float(charged.mean()),'critical_violation_robot_steps':int(failed.sum()),
                        'mission_utility_after_cost':utility,'candidate_plan_gap_before_cost':float(truth_score.max()-truth_score[chosen]),
                        'audible_rmse_db':float(np.sqrt(((belief.mean[0]-truth_dl)[audible]**2).mean())) if audible.any() else None,
                        'inaudible_rmse_db':float(np.sqrt(((belief.mean[0]-truth_dl)[~audible]**2).mean())) if (~audible).any() else None})
        print('Sensing environment',seed,'complete',flush=True)
    return {'source':'simulation','records':rows,'limits':['Scan durations are sensitivity assumptions, not measured hardware durations',
        'Initial history contains four scans from twelve robots; each extra query reveals one current capped scan',
        'No-scan is distinct from a scanned but unheard AP; acquisition sees no hidden outcomes',
        'Candidate generation can change with observations; plan gap is within each available candidate set']}


def fit_adapter(base,samples,validation,*,seed,epochs=5,ascent=None):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=deepcopy(base)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);best=float('inf');chosen=None
    for epoch in range(epochs):
        model.train()
        for index in rng.permutation(len(samples)):
            sample=samples[index]
            output=model(tensors(sample['observation']),torch.tensor(sample['plans']),torch.tensor(sample['prior']))
            objective=supervised_loss(output,{k:torch.tensor(v) for k,v in sample['target'].items()})
            if ascent is not None:
                forgotten=ascent[int(rng.integers(len(ascent)))]
                output=model(tensors(forgotten['observation']),torch.tensor(forgotten['plans']),torch.tensor(forgotten['prior']))
                objective=objective-.1*supervised_loss(output,{k:torch.tensor(v) for k,v in forgotten['target'].items()})
            optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        score=metrics(model,validation)['service_rmse']
        if score<best:best=score;chosen=deepcopy(model.state_dict())
    model.load_state_dict(chosen);model.eval();return model


def metrics(model,samples):
    squared=[];p=[];y=[]
    for sample in samples:
        output=predict(model,sample);squared.append(float(((output['service_raw']-sample['target']['service_raw'])**2).mean()))
        p.extend(expit(output['failure_logits']).ravel());y.extend(sample['target']['failure_raw'].ravel())
    return {'service_rmse':float(np.sqrt(np.mean(squared))),'failure_brier':float(np.mean((np.array(p)-y)**2))}


def attack(model,samples):
    signals={'negative_service_mse':[],'failure_confidence':[],'negative_signal_mse':[]}
    for sample in samples:
        output=predict(model,sample)
        signals['negative_service_mse'].append(-float(((output['service_raw']-sample['target']['service_raw'])**2).mean()))
        signals['failure_confidence'].append(float(np.abs(expit(output['failure_logits'])-.5).mean()))
        signals['negative_signal_mse'].append(-float(((output['signal']-sample['target']['signal'])**2).mean()))
    return signals


def adaptation(v3,root):
    sites=[]
    for site in range(4):
        splits={}
        for split,offset in [('train',0),('validation',100),('test',200)]:
            splits[split]=episode(162000+site*1000+offset,domain='shift',environment_seed=166000+site,seconds=150)
        sites.append(splits)
    rows=[]
    for seed in [0,1,2]:
        path=v3/'checkpoints'/f'graph_scratch-{seed}.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        base=RadioModel(**checkpoint['config']);base.load_state_dict(checkpoint['model']);base.eval()
        pooled=[s for site in sites for s in site['train']];validation=[s for site in sites for s in site['validation']]
        shared=fit_adapter(base,pooled,validation,seed=seed+500)
        local=[fit_adapter(base,s['train'],s['validation'],seed=seed+500) for s in sites]
        before=[metrics(base,s['validation']) for s in sites];candidate=[metrics(shared,s['validation']) for s in sites]
        local_val=[metrics(m,s['validation']) for m,s in zip(local,sites)]
        accepted=(np.mean([m['service_rmse'] for m in candidate])<min(np.mean([m['service_rmse'] for m in before]),np.mean([m['service_rmse'] for m in local_val]))
            and all(c['service_rmse']<=b['service_rmse'] and c['failure_brier']<=b['failure_brier'] for b,c in zip(before,candidate)))
        # Pooled training is an explicit specialist reference, not relabelled
        # as federation. Actual FedAvg is evaluated separately below.
        averaged=deepcopy(base);averaged.load_state_dict({k:sum(m.state_dict()[k] for m in local)/len(local) for k in base.state_dict()})
        fed_val=[metrics(averaged,s['validation']) for s in sites]
        fed_accepted=(np.mean([m['service_rmse'] for m in fed_val])<min(np.mean([m['service_rmse'] for m in before]),np.mean([m['service_rmse'] for m in local_val]))
            and all(c['service_rmse']<=b['service_rmse'] and c['failure_brier']<=b['failure_brier'] for b,c in zip(before,fed_val)))
        retained=[s for index,site in enumerate(sites) if index!=1 for s in site['train']]
        retained_val=[s for index,site in enumerate(sites) if index!=1 for s in site['validation']]
        started=time.perf_counter();clean=fit_adapter(base,retained,retained_val,seed=seed+500);clean_s=time.perf_counter()-started
        approximate=fit_adapter(shared,retained,retained_val,seed=seed+800,epochs=2,ascent=sites[1]['train'])
        if state_digest(shared)==state_digest(clean):raise RuntimeError('The shared target did not incorporate the forgotten contribution')
        attacks={}
        for name,model in [('shared_before',shared),('approximate_removal',approximate),('clean_retrain',clean),('initial',base)]:
            member=attack(model,sites[1]['train']);nonmember=attack(model,sites[1]['test'])
            attacks[name]={key:auc(member[key],nonmember[key]) for key in member}
            out=root/'checkpoints'/f'shared-{seed}-{name}.pt';out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'config':model.config,'model':model.state_dict(),'seed':seed},out)
        distance=float(np.sqrt(sum(float((a-b).square().sum()) for a,b in zip(approximate.parameters(),clean.parameters()))))
        rows.append({'seed':seed,'shared_pooled_gate_accepted':bool(accepted),'federation_gate_accepted':bool(fed_accepted),
            'validation':{'initial':before,'local':local_val,'pooled':candidate,'fedavg':fed_val},
            'test':{'initial':[metrics(base,s['test']) for s in sites],'local':[metrics(m,s['test']) for m,s in zip(local,sites)],
                'pooled':[metrics(shared,s['test']) for s in sites],'fedavg':[metrics(averaged,s['test']) for s in sites],
                'approximate_removal':[metrics(approximate,s['test']) for s in sites],'clean_retrain':[metrics(clean,s['test']) for s in sites]},
            'forget_site':1,'shared_target_incorporates_contribution':True,'shared_target_was_gate_accepted':bool(accepted),
            'clean_retraining_seconds':clean_s,'approximate_to_clean_parameter_l2':distance,
            'approximate_equals_clean_weights':state_digest(approximate)==state_digest(clean),
            'attacks':attacks,'shared_state_sha256':state_digest(shared),'clean_state_sha256':state_digest(clean)})
        print('Shared contribution/removal seed',seed,'pooled gate',accepted,'fed gate',fed_accepted,flush=True)
    return {'source':'actual_simulated_multisite_training','records':rows,'sites':4,'model_seeds':[0,1,2],
        'limits':['Pooled training is a central specialist reference, distinct from actual FedAvg',
            'Removal target is the unconditional shared research model; rejected candidates are not released',
            'Clean retained-site retraining is a reference, not an efficient unlearning method',
            'Gradient-ascent removal is approximate; three diagnostic attacks do not certify erasure',
            'No physical fleet and no removal of retained audit data copies']}


class PacketModel(nn.Module):
    def __init__(self,temporal=True):
        super().__init__();self.temporal=temporal;self.gru=nn.GRU(13,24,batch_first=True);self.instant=nn.Linear(13,24)
        self.output=nn.Linear(24,2)
    def forward(self,x):
        if self.temporal:_,h=self.gru(x);h=h[-1]
        else:h=torch.nn.functional.gelu(self.instant(x[:,-1]))
        return self.output(h)


def packet_prediction(packet_root,root):
    report=json.loads((packet_root/'packet_audit.json').read_text())
    splits={'train':[],'validation':[],'test':[]}
    for run in report['records']:
        native_path=packet_root/run['native_report']
        if sha256(native_path)!=run['native_sha256']:raise RuntimeError('Packet training evidence changed')
        native=json.loads(native_path.read_text());n=native['robots'];static=native['per_robot']
        frames=native['telemetry'];scene=run['scene'];split='train' if scene<146002 else ('validation' if scene==146002 else 'test')
        counts=np.bincount([r['ap'] for r in static],minlength=128)
        for tick in range(3,len(frames)-1):
            for robot in range(n):
                current=frames[tick]['robots'][robot];following=frames[tick+1]['robots'][robot]
                # The event is a late arrival observed at the actual endpoint,
                # not unreceived packets inferred to have timed out in one bin.
                sequence=np.array([packet_features(frames[t]['robots'][robot])+[np.log1p(static[robot]['offered_mbps']),counts[static[robot]['ap']]/100.]
                    for t in range(tick-3,tick+1)],dtype='float32')
                target=[np.log1p(following['delivered_bytes']*8/1e6),float(following['critical_late']>0)]
                splits[split].append((sequence,target,scene,float(current['critical_late']>0),np.log1p(current['delivered_bytes']*8/1e6)))
    arrays={key:(torch.tensor(np.stack([r[0] for r in rows])),torch.tensor([r[1] for r in rows],dtype=torch.float32)) for key,rows in splits.items()}
    x,y=arrays['test'];rows=[]
    for seed in [0,1,2]:
        for temporal in [False,True]:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=PacketModel(temporal)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3);best=float('inf');chosen=None
            for epoch in range(20):
                xx,yy=arrays['train']
                for offset in range(0,len(xx),128):
                    idx=rng.integers(len(xx),size=min(128,len(xx)))
                    out=model(xx[idx]);loss=F.mse_loss(out[:,0],yy[idx,0])+F.binary_cross_entropy_with_logits(out[:,1],yy[idx,1])
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                with torch.no_grad():
                    xx,yy=arrays['validation'];out=model(xx)
                    loss=float(F.mse_loss(out[:,0],yy[:,0])+F.binary_cross_entropy_with_logits(out[:,1],yy[:,1]))
                if loss<best:best=loss;chosen=deepcopy(model.state_dict())
            model.load_state_dict(chosen);model.eval()
            with torch.no_grad():out=model(x);prob=out[:,1].sigmoid().numpy()
            path=root/'checkpoints'/f'packet-{seed}-{temporal}.pt';path.parent.mkdir(exist_ok=True)
            torch.save({'model':model.state_dict(),'temporal':temporal,'seed':seed},path)
            rows.append({'seed':seed,'temporal':temporal,'next_second_throughput_mae_mbps':float((out[:,0].exp()-y[:,0].exp()).abs().mean()),
                'late_arrival_risk':risk_metrics(prob,y[:,1].numpy()),'checkpoint_sha256':sha256(path)})
    return {'source':'native_ns3_packet_events','split_samples':{k:len(v) for k,v in splits.items()},
        'test_scenes':[146003,146004,146005],'test_label_counts':{'late':int(y[:,1].sum()),'not_late':int((1-y[:,1]).sum())},
        'persistence':{'late_arrival_risk':risk_metrics([r[3] for r in splits['test']],y[:,1].numpy()),
            'throughput_mae_mbps':float(np.mean([abs(np.expm1(r[4])-np.expm1(r[1][0])) for r in splits['test']]))},
        'records':rows,'limits':['Predicts any late received critical packet in the next one-second bin; not loss probability',
            'All bins are included: no received critical packet means no late-arrival event, not evidence of a healthy link',
            'Rows share scenes and policy interventions; scene-disjoint test has three environments',
            'Missing retry/late-rate denominators use explicit validity masks',
            'These are real simulator events, not physical hardware measurements']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--v3',type=Path,default=Path('/workspace/training/warehouse-v3'))
    p.add_argument('--packets',type=Path,default=Path('/workspace/training/warehouse-v3-packets'))
    p.add_argument('--stages',nargs='+',default=['sensing','adaptation','packet_prediction']);args=p.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    status_path=root/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {'stages':{}}
    sources=[Path(__file__),*REPO.glob('src/wirelessjev/**/*.py')]
    hashes={str(path.relative_to(REPO)):sha256(path) for path in sources}
    if not (root/'provenance.json').exists():
        for path in sources:
            dest=root/'source_snapshot'/path.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        atomic_json(root/'provenance.json',{'source_sha256':hashes,'source':'simulation_and_native_packet_event_training',
            'hardware_experiments_executed':False})
    status.update(state='running',pid=os.getpid());atomic_json(status_path,status)
    try:
        for stage in args.stages:
            if status['stages'].get(stage,{}).get('state')=='completed':
                if sha256(root/f'{stage}.json')!=status['stages'][stage]['sha256']:raise RuntimeError('Completed study changed')
                continue
            if stage in ['adaptation','packet_prediction']:
                dependency=args.v3 if stage=='adaptation' else args.packets
                while not (dependency/'status.json').exists() or json.loads((dependency/'status.json').read_text())['state']!='completed':
                    if (dependency/'status.json').exists() and json.loads((dependency/'status.json').read_text())['state']=='failed':raise RuntimeError('Study dependency failed')
                    time.sleep(5)
            status['stage']=stage;atomic_json(status_path,status)
            result=sensing() if stage=='sensing' else (adaptation(args.v3,root) if stage=='adaptation' else packet_prediction(args.packets,root))
            atomic_json(root/f'{stage}.json',result);status['stages'][stage]={'state':'completed','sha256':sha256(root/f'{stage}.json')};atomic_json(status_path,status)
    except BaseException as error:
        status.update(state='failed',error=str(error),traceback=traceback.format_exc());atomic_json(status_path,status);raise
    status.update(state='completed',stage=None,completed_at=datetime.now(timezone.utc).isoformat());atomic_json(status_path,status)


if __name__=='__main__':main()
