"""Train new models, tune matched controllers, seal, then test fresh sites."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import argparse, hashlib, json, shutil, time
import numpy as np
import torch
from ..contracted.data import ROOT, READINGS, KEYS, runtime_world, examples
from ..contracted.train import write, load_models as old_models
from ..contracted.runtime import DeviceInputs, forecast as old_forecast
from ..native.intent_policy import intent_policy
from ..native.sim import simulate, policies
from ..native.world import HORIZONS
from ..gate_repair.gate import reactive_rescue
from .data import episode, fit_map, inputs, map_prediction, scan_context, training_condition
from .model import FleetJEPA, objective, update_target

OUT=ROOT/'artifacts/fleet_world_model'; RUN=ROOT.parent/'training/fleet-world-model'
P=json.loads((OUT/'protocol.json').read_text())
FIELDS=('context','query','memory','physics','map_mean','map_std','fleet_mean')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths=list((ROOT/'src/wirelessjev/fleetwm').glob('*.py'))+list((ROOT/'src/wirelessjev/native').glob('*.py'))
    paths+=list((ROOT/'src/wirelessjev/twin').glob('*.py'))+list((ROOT/'src/wirelessjev/contracted').glob('*.py'))
    paths += [ROOT/'src/wirelessjev/gate_repair/gate.py',ROOT/'src/wirelessjev/warehouse_v3/jepa.py',
        ROOT/'pyproject.toml',ROOT/'uv.lock',OUT/'protocol.json',ROOT/'artifacts/native/intent_flip.json',
        ROOT/'artifacts/warehouse_v2/studies/wifi_link_curves.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def snapshot(stage):
    hashes=sources();receipt=OUT/(stage+'_source.json')
    if receipt.exists():
        if json.loads(receipt.read_text())['source_sha256']!=hashes:
            raise RuntimeError('Executed source changed; use a new experiment directory')
        return
    for name in hashes:
        dest=OUT/(stage+'_source')/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(receipt,{'source_sha256':hashes,'time_unix':time.time(),'torch':torch.__version__})


def training_episode(task):
    split,index,seed=task;torch.set_num_threads(1)
    condition=training_condition(index) if split=='training' else ['main','main','stress','yard','drift','main'][index]
    n=P['training_robots'];duration=P['training_decisions']
    full,scan,g,obs=episode(seed,n,duration,condition);rows=[];rmap=None
    for t in range(60,60+duration,P['training_query_stride']):
        if rmap is None or (t-60)%P['map_refresh_seconds']==0:
            rmap=fit_map(g,obs.before(t))
        b=inputs(g,obs.before(t),full.plan[t],rmap)
        r={k:getattr(b,k) for k in FIELDS}
        future=t+np.asarray(HORIZONS,dtype=int)
        r['truth']=full.snr[future*5].transpose(1,0,2,3).reshape(n,9,-1).astype('f4')
        r['future_context']=np.stack([scan_context(g,obs.before(int(tau))) for tau in future],1)
        rows.append(r)
    result={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    # Heterogeneous link count in the outdoor environment is zero-padded only
    # at the dataset boundary; valid-link masks exclude padding from losses.
    L=result['context'].shape[1]; result['valid_link']=np.ones((len(result['context']),L),dtype='f4')
    if L<48:
        for k in result:
            axis=1 if k in {'context','memory','valid_link'} else 2
            pads=[(0,0)]*result[k].ndim;pads[axis]=(0,48-L)
            result[k]=np.pad(result[k],pads)
        result['truth'][:,:,L:]=-100
    print(json.dumps({'phase':'data','split':split,'site':seed,'condition':condition,'rows':len(result['context'])}),flush=True)
    return result


def dataset(split,workers):
    path=RUN/(split+'.npz');receipt=RUN/(split+'.json')
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['source_sha256']!=sources() or r['npz_sha256']!=digest(path):raise RuntimeError('Dataset identity changed')
        with np.load(path) as z:return {k:z[k] for k in z.files}
    tasks=[(split,i,s) for i,s in enumerate(P[split+'_sites'])]
    with ProcessPoolExecutor(workers) as pool:pieces=list(pool.map(training_episode,tasks))
    arrays={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    arrays={k:v.astype('f2') for k,v in arrays.items()}
    RUN.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    write(receipt,{'source_sha256':sources(),'npz_sha256':digest(path),'sites':P[split+'_sites'],'rows':len(arrays['context'])})
    return arrays


def batch(arrays,idx):return {k:torch.as_tensor(v[idx].astype('f4')) for k,v in arrays.items()}


def fit_one(task):
    family,seed=task;torch.set_num_threads(1)
    dest=OUT/'checkpoints'/f'{family}-{seed}.pt';report=OUT/'fits'/f'{family}-{seed}.json'
    if report.exists():
        r=json.loads(report.read_text())
        if r['source_sha256']!=sources() or r['checkpoint_sha256']!=digest(dest):raise RuntimeError('Fit identity changed')
        return r
    train=dataset('training',1);dev=dataset('validation',1)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=FleetJEPA(family);optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=7e-4,weight_decay=2e-4)
    history=[];best=float('inf');started=time.time();last=RUN/f'{family}-{seed}.last.pt';first=0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['source_sha256']!=sources():raise RuntimeError('Resume source changed')
        model.load_state_dict(saved['state']);optim.load_state_dict(saved['optimizer']);history=saved['history'];best=saved['best'];first=saved['epoch']+1
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
    for epoch in range(first,P['epochs']):
        model.train();order=rng.permutation(len(train['context']));costs=[]
        for start in range(0,len(order),64):
            b=batch(train,order[start:start+64]);cost,_=objective(model,b)
            if not torch.isfinite(cost):raise FloatingPointError('Nonfinite training loss')
            optim.zero_grad();cost.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();update_target(model);costs.append(float(cost.detach()))
        model.eval();sq=weight=0.;regrets=[]
        with torch.no_grad():
            for start in range(0,len(dev['context']),128):
                b=batch(dev,slice(start,start+128));p=model(b);valid=b['valid_link'][:,None];w=torch.where(b['truth']>0,1.,.2)*valid
                sq+=float(((p['mean']-b['truth']).square()*w).sum());weight+=float(w.sum())
                L=b['truth'].shape[-1];pm=p['mean'][:,:6].reshape(-1,6,L//3,3).max(-1).values.mean(1)
                tm=b['truth'][:,:6].reshape(-1,6,L//3,3).max(-1).values.mean(1)
                pm=pm.masked_fill(~b['valid_link'][:,::3].bool(),-1e6)
                r=tm.max(-1).values-tm.gather(1,pm.argmax(-1)[:,None])[:,0];regrets.extend(r.tolist())
        score=(sq/weight)**.5+.5*float(np.mean(regrets))
        row={'epoch':epoch+1,'train_loss':float(np.mean(costs)),'validation_weighted_rmse':(sq/weight)**.5,'validation_ap_regret':float(np.mean(regrets)),'validation_score':score};history.append(row)
        print(json.dumps({'family':family,'seed':seed,**row}),flush=True)
        if score<best:
            best=score;dest.parent.mkdir(parents=True,exist_ok=True);torch.save({'config':model.config,'state':model.state_dict(),'epoch':epoch+1,'seed':seed},dest)
        torch.save({'state':model.state_dict(),'optimizer':optim.state_dict(),'history':history,'best':best,'epoch':epoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'source_sha256':sources()},last)
        write(RUN/f'{family}-{seed}.status.json',row)
    result={'completed':True,'family':family,'seed':seed,'history':history,'best_validation_score':best,'elapsed_s':time.time()-started,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':digest(dest),'source_sha256':sources(),'test_accessed':False}
    write(report,result);return result


def load(family):
    models=[]
    for seed in P['training_seeds']:
        saved=torch.load(OUT/'checkpoints'/f'{family}-{seed}.pt',weights_only=True,map_location='cpu')
        m=FleetJEPA(**saved['config']);m.load_state_dict(saved['state']);m.eval();models.append(m)
    return models


@torch.no_grad()
def predict(models,b):
    if not hasattr(b,'context'):raise TypeError('Use label-free FleetInputs')
    B=len(b.context);means=[];stds=[]
    for start in range(0,B,128):
        t={k:torch.as_tensor(np.array(getattr(b,k)[start:start+128],dtype='f4')) for k in FIELDS}
        ps=[m(t) for m in models];mu=torch.stack([p['mean'] for p in ps]);v=torch.stack([p['std'].square() for p in ps]).mean(0)
        if len(models)>1:v=v+mu.var(0,unbiased=True)
        means.append(mu.mean(0).numpy());stds.append(v.sqrt().numpy())
    return np.concatenate(means),np.concatenate(stds)


def train(workers):
    snapshot('training');dataset('training',workers);dataset('validation',workers)
    tasks=[(f,s) for f in P['families'] for s in P['training_seeds']]
    with ProcessPoolExecutor(workers) as pool:fits=list(pool.map(fit_one,tasks))
    write(OUT/'training.json',{'completed':True,'fits':len(fits),'checkpoint_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted((OUT/'checkpoints').glob('*.pt'))},'data_receipts':{s:json.loads((RUN/(s+'.json')).read_text()) for s in ['training','validation']},'test_accessed':False,'time_unix':time.time()})
    print('NINE_FLEET_FITS_COMPLETE',flush=True)


def forecast_metrics(mean,std,truth,g):
    error=mean-truth;near=error[:,:,:6];sd=std[:,:,:6]
    audible=truth[:,:,:6]>0
    scores=mean[:,:,:6].max(-1).mean(2);target=truth[:,:,:6].max(-1).mean(2)
    ix=scores.argmax(-1);regret=target.max(-1)-np.take_along_axis(target,ix[...,None],-1)[...,0]
    return {'rmse_near_db':float(np.sqrt(np.mean(near**2))),
            'audible_rmse_db':float(np.sqrt(np.mean(near[audible]**2))),
            'unheard_rmse_db':float(np.sqrt(np.mean(near[~audible]**2))) if (~audible).any() else None,
            'coverage_90':float(np.mean(np.abs(near)<=1.645*sd)),
            'ap_choice_regret_db':float(np.mean(regret))}


def case(task):
    stage,condition,seed=task;torch.set_num_threads(1);dest=OUT/'cases'/f'{stage}-{condition}-{seed}.json'
    if dest.exists():
        r=json.loads(dest.read_text())
        if r['source_sha256']!=sources():raise RuntimeError('Case source changed')
        return r
    started=time.time();n=10 if condition=='small' else 100
    duration=P['primary_duration_s'] if stage=='test' and condition=='main' else P['selection_duration_s'] if stage=='selection' else P['generalisation_duration_s']
    full,scan,g,obs=episode(seed,n,duration,condition)
    initial=fit_map(g,obs.before(60));rmap=initial;models={f:load(f) for f in P['families']}
    predictions={k:[[],[]] for k in ['static_radiomap','rolling_radiomap',*P['families']]};timings={k:0. for k in predictions};fit_seconds=0.
    for d in range(duration):
        t=60+d;plan=full.plan[t]
        if d>0 and d%P['map_refresh_seconds']==0:
            began=time.time();rmap=fit_map(g,obs.before(t));fit_seconds+=time.time()-began
        began=time.time();b=inputs(g,obs.before(t),plan,rmap);input_seconds=time.time()-began
        for name in ['static_radiomap','rolling_radiomap']:
            began=time.time();m,sd=map_prediction(g,initial if name=='static_radiomap' else rmap,plan);timings[name]+=time.time()-began
            predictions[name][0].append(m);predictions[name][1].append(sd)
        for family in P['families']:
            began=time.time();m,sd=predict(models[family],b);timings[family]+=time.time()-began+input_seconds
            predictions[family][0].append(m.reshape(n,9,-1,3));predictions[family][1].append(sd.reshape(n,9,-1,3))
    # Previous JEPA receives its original six-scan inputs and frozen weights.
    old=examples(full,scan,duration,targets=False);weights=[READINGS[P['primary_mission']]['weights'][k] for k in KEYS]
    began=time.time();m,sd=old_forecast(old_models('jepa'),DeviceInputs(old['ctx'],old['route'],old['prior']),weights)
    predictions['previous_jepa']=[m.reshape(duration,n,9,-1,3),sd.reshape(duration,n,9,-1,3)*json.loads((ROOT/'artifacts/contracted/calibration.json').read_text())['jepa']['scale']];timings['previous_jepa']=time.time()-began
    predictions={k:(np.asarray(v[0],dtype='f4'),np.asarray(v[1],dtype='f4')) for k,v in predictions.items()}
    future=np.arange(60,60+duration)[:,None]+np.asarray(HORIZONS,dtype=int)
    truth=full.snr[future*5].transpose(0,2,1,3,4)
    metrics={k:forecast_metrics(m,s,truth,g) for k,(m,s) in predictions.items()}
    base=replace(intent_policy(READINGS[P['primary_mission']]['weights'],READINGS[P['primary_mission']]['focus'],dual_hardware='none'),online_map=False,gate=True)
    selected=json.loads((OUT/'selection.json').read_text()) if stage=='test' else None
    arms=[]
    for family in predictions:
        options=[selected['choices'][family]['gate']] if selected else [*P['gate_multipliers'],'rescue']
        for gate in options:arms.append((family,gate,P['primary_mission']))
    if stage=='test' and condition=='main':
        for f in ['rolling_radiomap','hybrid_jepa','hybrid_supervised','fleet_jepa']:
            arms.append((f,selected['choices'][f]['gate'],P['heldout_mission']))
    rows=[]
    fixed=simulate(runtime_world(full,duration,*[predictions['static_radiomap'][0],predictions['static_radiomap'][1][:,:,0]]),{p.name:p for p in policies()}['mlo_split'],collision_lambda=.3 if condition=='stress' else 0)
    rows.append({'family':'fixed','gate':None,'mission':P['primary_mission'],'result':fixed})
    for family,gate,mission in arms:
        mean,std=predictions[family]
        pol=replace(intent_policy(READINGS[mission]['weights'],READINGS[mission]['focus'],dual_hardware='none'),online_map=False,gate=True,belief_scale=1 if gate=='rescue' else gate,name=family+'_'+str(gate))
        hook=reactive_rescue if gate=='rescue' else None
        result=simulate(runtime_world(full,duration,mean,std[:,:,0]),pol,roam_gate=hook,collision_lambda=.3 if condition=='stress' else 0)
        rows.append({'family':family,'gate':gate,'mission':mission,'forecast_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),'result':result})
        print(json.dumps({'stage':stage,'condition':condition,'site':seed,'family':family,'gate':gate,'mission':mission,'misses':result['control_misses']}),flush=True)
    receipt={'completed':True,'stage':stage,'scene':seed,'condition':condition,'robots':n,'duration_s':duration,'rows':rows,'forecast_metrics':metrics,'predict_seconds':timings,'rolling_refits_seconds':fit_seconds,'elapsed_s':time.time()-started,'source_sha256':sources(),'checkpoint_sha256':json.loads((OUT/'training.json').read_text())['checkpoint_sha256'],'runtime_inputs_have_truth':False,'latest_observation_strictly_before_decision':True}
    write(dest,receipt);return receipt


def execute(tasks,workers):
    with ProcessPoolExecutor(workers) as pool:return list(pool.map(case,tasks))


def select(workers):
    if (OUT/'selection.json').exists():return
    if not json.loads((OUT/'training.json').read_text())['completed']:raise RuntimeError('Finish training')
    snapshot('selection')
    parts=execute([('selection',c,s) for c in ['main','stress','yard','small','drift'] for s in P['selection_'+c+'_sites']],workers)
    choices={}
    for f in ['static_radiomap','rolling_radiomap','previous_jepa',*P['families']]:
        candidates=[]
        for gate in [*P['gate_multipliers'],'rescue']:
            ratios={}
            for condition in ['main','stress','yard','small','drift']:
                cs=[c for c in parts if c['condition']==condition]
                denom=sum(next(r['result']['control_misses'] for r in c['rows'] if r['family']=='rolling_radiomap' and r['gate']==1.) for c in cs)
                misses=sum(next(r['result']['control_misses'] for r in c['rows'] if r['family']==f and r['gate']==gate) for c in cs)
                ratios[condition]=(misses+1)/(denom+1)
            candidates.append({'gate':gate,'score':float(np.mean(list(ratios.values()))),'ratios':ratios})
        choices[f]=min(candidates,key=lambda r:r['score'])|{'candidates':candidates}
    candidate=min(P['families'],key=lambda f:choices[f]['score'])
    write(OUT/'selection.json',{'sealed':True,'sealed_at_unix':time.time(),'choices':choices,'development_best_model':candidate,'source_sha256':sources(),'checkpoint_sha256':json.loads((OUT/'training.json').read_text())['checkpoint_sha256'],'test_accessed':False})
    print('FLEET_SELECTION_SEALED',flush=True)


def test(workers):
    selected=json.loads((OUT/'selection.json').read_text())
    if not selected['sealed'] or selected['source_sha256']!=sources():raise RuntimeError('Selection not sealed or source changed')
    for name,h in selected['checkpoint_sha256'].items():
        if digest(ROOT/name)!=h:raise RuntimeError('Checkpoint changed after selection')
    snapshot('test')
    parts=execute([('test',c,s) for c in ['main','stress','yard','small','drift'] for s in P['test_'+c+'_sites']],workers)
    write(OUT/'completion.json',{'completed':True,'fresh_scenes':len(parts),'test_rollouts':sum(len(c['rows']) for c in parts),'selection_rollouts':sum(len(json.loads(p.read_text())['rows']) for p in (OUT/'cases').glob('selection-*.json')),'completed_at_unix':time.time()})
    print('FLEET_WORLD_MODEL_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['train','select','test','all'],default='all');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    if args.phase in ['train','all']:train(args.workers)
    if args.phase in ['select','all']:select(args.workers)
    if args.phase in ['test','all']:test(args.workers)


if __name__=='__main__':main()
