"""Finite, source-sealed optimisation frontier; every selected model gets fresh tests."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields,replace
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np
import torch
from ..contracted.data import ROOT,READINGS,KEYS,runtime_world
from ..contracted.train import write,load_models as load_previous
from ..contracted.runtime import DeviceInputs,forecast as old_forecast
from ..fleetwm.data import episode,Observations
from ..fleetwm.study import load as load_teacher,predict as teacher_forecast
from ..native.intent_policy import intent_policy
from ..native.world import HORIZONS
from ..gate_repair.gate import reactive_rescue
from .fast import FleetMemory,FastMap,fit_reference
from .timing import ForecastQueue
from .sim import simulate
from .model import FrontierModel,objective

OUT=ROOT/'artifacts/world_frontier';RUN=ROOT.parent/'training/world-frontier'
P=json.loads((OUT/'protocol.json').read_text());METHODS=P['comparators']
WEIGHTS=[READINGS['bulk_uploads']['weights'][k] for k in KEYS]


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sources():
    paths=list((ROOT/'src/wirelessjev/frontier').glob('*.py'))
    for name in ['native','twin','contracted','fleetwm']:paths+=list((ROOT/'src/wirelessjev'/name).glob('*.py'))
    paths += [ROOT/'src/wirelessjev/fleet_strong_reference.py',ROOT/'src/wirelessjev/gate_repair/gate.py',ROOT/'src/wirelessjev/warehouse_v3/jepa.py',
        ROOT/'artifacts/native/foundation.json',ROOT/'artifacts/native/intent_flip.json',ROOT/'artifacts/contracted/calibration.json',
        ROOT/'artifacts/warehouse_v2/studies/wifi_link_curves.json',ROOT/'pyproject.toml',ROOT/'uv.lock',OUT/'protocol.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def checkpoint_hashes():
    paths=list((ROOT/'artifacts/contracted/checkpoints').glob('jepa-*.pt'))+list((ROOT/'artifacts/fleet_world_model/checkpoints').glob('hybrid_supervised-*.pt'))+list((OUT/'checkpoints').glob('*.pt'))
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}


def snapshot(stage):
    dest=OUT/(stage+'_source.json');hashes=sources()
    if dest.exists():
        if json.loads(dest.read_text())['source_sha256']!=hashes:raise RuntimeError('Executed source changed; start a separate experiment')
        return
    for name in hashes:
        target=OUT/(stage+'_source')/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write(dest,{'source_sha256':hashes,'checkpoint_sha256':checkpoint_hashes(),'time_unix':time.time(),'torch':torch.__version__})


def mask_scans(obs,cadence):
    if cadence==1:return obs
    scanned=obs.scanned & (np.arange(len(obs.value))[:,None]%cadence==0)
    return Observations(obs.positions,obs.value,obs.heard&scanned[...,None],obs.bound,scanned)


def tensors(b,count):
    return {f.name:torch.tensor(np.array(getattr(b,f.name),dtype='f4')) for f in fields(b)}|{'count':torch.tensor(count,dtype=torch.float32)}


def training_case(task):
    split,i,seed=task;torch.set_num_threads(1)
    n=P['training_robots_cycle'][i%3];cadence=P['training_scan_period_cycle'][(i//3)%3]
    condition='drift' if i%4==3 else 'main';D=P['training_decisions']
    full,scan,g,obs=episode(seed,n,D,condition);obs=mask_scans(obs,cadence)
    memory=FleetMemory(g);models=load_teacher('hybrid_supervised');pieces=[]
    for t in range(60,60+D,P['training_query_stride']):
        past=obs.before(t)
        if t==60 or (t-60)%20==0:rmap=fit_reference(g,past)
        b,c=memory.inputs(past,full.plan[t],rmap)
        teacher,_=teacher_forecast(models,b)
        future=t+np.asarray(HORIZONS,dtype=int)
        r={f.name:getattr(b,f.name) for f in fields(b)}|{'count':c,'teacher':teacher,
            'truth':full.snr[future*5].transpose(1,0,2,3).reshape(n,9,-1).astype('f4')}
        pieces.append(r)
    print(json.dumps({'data':split,'site':seed,'robots':n,'scan_period':cadence,'condition':condition}),flush=True)
    return {k:np.concatenate([r[k] for r in pieces]) for k in pieces[0]}


def dataset(split,workers):
    path=RUN/(split+'.npz');receipt=RUN/(split+'.json')
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['source_sha256']!=sources() or old['data_sha256']!=digest(path):raise RuntimeError('Dataset identity changed')
        with np.load(path) as z:return {k:z[k] for k in z.files}
    with ProcessPoolExecutor(workers) as pool:pieces=list(pool.map(training_case,[(split,i,s) for i,s in enumerate(P[split+'_sites'])]))
    arrays={k:np.concatenate([r[k] for r in pieces]).astype('f2') for k in pieces[0]}
    RUN.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
    write(receipt,{'sites':P[split+'_sites'],'source_sha256':sources(),'data_sha256':digest(path),'examples':len(arrays['context'])})
    return arrays


def batch(a,index):return {k:torch.as_tensor(v[index].astype('f4')) for k,v in a.items()}


@torch.no_grad()
def validate(model,a):
    absolute=[];delta=[];ranking=[]
    for start in range(0,len(a['context']),128):
        b=batch(a,slice(start,start+128));p=model(b)['mean'];y=b['truth']
        w=torch.where(y[:,:6]>0,1.,.2)
        absolute.append(float(((p[:,:6]-y[:,:6]).square()*w).mean()))
        delta.append(float((((p[:,:6]-p[:,:1])-(y[:,:6]-y[:,:1])).square()*w).mean()))
        B,H,L=y.shape;s=p[:,:6].reshape(B,6,L//3,3).max(-1).values.mean(1);t=y[:,:6].reshape(B,6,L//3,3).max(-1).values.mean(1)
        ranking.append(float((t.max(-1).values-t.gather(1,s.argmax(-1)[:,None])[:,0]).mean()))
    result={'weighted_rmse':float(np.sqrt(np.mean(absolute))),'weighted_delta_rmse':float(np.sqrt(np.mean(delta))),'ap_regret':float(np.mean(ranking))}
    result['score']=result['weighted_delta_rmse']+.2*result['weighted_rmse']+.1*result['ap_regret'];return result


def fit_one(task):
    family,seed=task;torch.set_num_threads(1);dest=OUT/'checkpoints'/f'{family}-{seed}.pt';report=OUT/'fits'/f'{family}-{seed}.json'
    if report.exists():
        r=json.loads(report.read_text());assert r['source_sha256']==sources() and r['checkpoint_sha256']==digest(dest);return r
    train=dataset('training',1);dev=dataset('validation',1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=FrontierModel(family);optim=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=2e-4);best=float('inf');history=[];started=time.time();first=0
    last=RUN/f'{family}-{seed}.last.pt'
    if last.exists():
        s=torch.load(last,weights_only=False,map_location='cpu');assert s['source_sha256']==sources();model.load_state_dict(s['state']);optim.load_state_dict(s['optimizer'])
        history=s['history'];best=s['best'];first=s['epoch']+1;rng.bit_generator.state=s['numpy_rng'];torch.set_rng_state(s['torch_rng'])
    for epoch in range(first,P['epochs']):
        model.train();order=rng.permutation(len(train['context']));losses=[]
        for start in range(0,len(order),64):
            b=batch(train,order[start:start+64]);cost=objective(model,b)
            if not torch.isfinite(cost):raise FloatingPointError('Nonfinite training loss')
            optim.zero_grad();cost.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optim.step();losses.append(float(cost.detach()))
        model.eval();v=validate(model,dev);r={'family':family,'seed':seed,'epoch':epoch+1,'train_loss':float(np.mean(losses)),**v};history.append(r);print(json.dumps(r),flush=True)
        if v['score']<best:
            best=v['score'];dest.parent.mkdir(parents=True,exist_ok=True);torch.save({'config':model.config,'state':model.state_dict(),'epoch':epoch+1,'seed':seed},dest)
        torch.save({'state':model.state_dict(),'optimizer':optim.state_dict(),'history':history,'best':best,'epoch':epoch,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'source_sha256':sources()},last)
    r={'completed':True,'family':family,'seed':seed,'best_validation_score':best,'history':history,'parameters':sum(p.numel() for p in model.parameters()),'elapsed_s':time.time()-started,'source_sha256':sources(),'checkpoint_sha256':digest(dest),'test_accessed':False}
    write(report,r);return r


def load_student(family):
    trained=json.loads((OUT/'training.json').read_text());seed=trained['single_members'][family]
    s=torch.load(OUT/'checkpoints'/f'{family}-{seed}.pt',weights_only=True,map_location='cpu');m=FrontierModel(**s['config']);m.load_state_dict(s['state']);m.eval();return m


def train(workers):
    snapshot('training');dataset('training',workers);dataset('validation',workers)
    with ProcessPoolExecutor(workers) as pool:rs=list(pool.map(fit_one,[(f,s) for f in P['families'] for s in P['training_seeds']]))
    choices={f:min([r for r in rs if r['family']==f],key=lambda r:(r['best_validation_score'],r['seed']))['seed'] for f in P['families']}
    write(OUT/'training.json',{'completed':True,'fits':len(rs),'single_members':choices,'checkpoint_sha256':checkpoint_hashes(),'source_sha256':sources(),'data_receipts':{s:json.loads((RUN/(s+'.json')).read_text()) for s in ['training','validation']},'time_unix':time.time(),'test_accessed':False})
    print('FRONTIER_NINE_FITS_COMPLETE',flush=True)


class Predictors:
    def __init__(self,g):
        self.g=g;self.memory=FleetMemory(g);self.teacher=load_teacher('hybrid_supervised');self.previous=load_previous('jepa')
        self.students={f:load_student(f) for f in P['families']};self.generic=None;self.strong=None;self.refit_at=None
        self.last_t=None;self.temporal_previous=None;self.innovation=np.zeros(len(g.sites)*3,dtype='f4')

    @torch.no_grad()
    def step(self,past,plan):
        g=self.g;t=len(past.value);fit_g=fit_s=0.
        if self.generic is None or t-self.refit_at>=20:
            a=time.perf_counter();self.generic=fit_reference(g,past);fit_g=time.perf_counter()-a
            a=time.perf_counter();self.strong=fit_reference(g,past,strong=True);fit_s=time.perf_counter()-a;self.refit_at=t
        a=time.perf_counter();b,count=self.memory.inputs(past,plan);feature=time.perf_counter()-a
        predictions={};timings={}
        def mapped(rmap):
            m,s=FastMap(rmap).predict(plan[...,0],plan[...,1]);return (m-g.tx+15-g.noise).reshape(len(plan),9,-1),s.reshape(len(plan),9,-1)
        a=time.perf_counter();predictions['radiomap']=mapped(self.strong);timings['radiomap']=time.perf_counter()-a+fit_s
        a=time.perf_counter();mm,ms=mapped(self.generic);map_time=time.perf_counter()-a
        from dataclasses import replace as dc_replace
        assisted=dc_replace(b,map_mean=mm,map_std=ms)
        a=time.perf_counter();predictions['confirmed_hybrid']=teacher_forecast(self.teacher,assisted);timings['confirmed_hybrid']=time.perf_counter()-a+map_time+fit_g+feature
        a=time.perf_counter();old_prior=b.physics+b.query[...,3]*5*np.tile([2.5,3.5,4.5],len(g.sites))
        route=np.stack([b.query[...,0],b.query[...,1],b.query[...,2],b.query[...,4],old_prior/30],-1).astype('f4')
        predictions['previous_jepa']=old_forecast(self.previous,DeviceInputs(b.context,route,old_prior),WEIGHTS)
        scale=json.loads((ROOT/'artifacts/contracted/calibration.json').read_text())['jepa']['scale'];m,s=predictions['previous_jepa'];predictions['previous_jepa']=(m,s*scale)
        # Conservative charge includes the full causal feature preparation.
        timings['previous_jepa']=time.perf_counter()-a+feature
        tb=tensors(b,count)
        for family,model in self.students.items():
            a=time.perf_counter();p=model(tb);mean=p['mean'].numpy();sd=p['std'].numpy()
            if family=='temporal':
                if self.last_t==t-1 and self.temporal_previous is not None:
                    # Only actual strong reports at t-1, compared with a forecast
                    # issued before those reports existed. No twin labels enter.
                    heard=past.heard[-1]&(past.value[-1]>-75)
                    observed=past.value[-1]+np.tile(15-g.tx-g.noise,len(g.sites))+1.174
                    error=np.where(heard,observed-self.temporal_previous,0);count=heard.sum(0)
                    update=error.sum(0)/np.maximum(count,1)
                    self.innovation=np.clip(self.innovation+np.where(count>=3,.2*np.clip(update,-6,6),0),-10,10)
                elif self.last_t is not None:self.innovation[:]=0
                mean=mean+self.innovation[None,None,:]
                self.temporal_previous=mean[:,0].copy()
            predictions[family]=(mean,sd);timings[family]=time.perf_counter()-a+feature
        self.last_t=t
        return predictions,timings,(fit_g>0)


def benchmark():
    if (OUT/'latency.json').exists():return
    torch.set_num_threads(1);full,scan,g,original=episode(P['validation_sites'][0],100,50,'main')
    values={};profile={}
    for cadence in [1,5]:
        obs=mask_scans(original,cadence);pred=Predictors(g)
        measured={f:{'cold':[],'warm':[],'refit':[]} for f in METHODS}
        for d in [*range(24),40,41]:
            _,timing,refit=pred.step(obs.before(60+d),full.plan[60+d]);kind='cold' if d==0 else 'refit' if refit else 'warm'
            for f in METHODS:measured[f][kind].append(timing[f])
        values[str(cadence)]=measured
        profile[str(cadence)]={f:{k:float(np.quantile(v,.95)) for k,v in groups.items()} for f,groups in measured.items()}
    write(OUT/'latency.json',{'sealed':True,'time_unix':time.time(),'robots':100,'torch_threads':1,'raw_seconds':values,'p95_seconds':profile,'source_sha256':sources(),'checkpoint_sha256':checkpoint_hashes(),'test_accessed':False})
    print('MEASURED_LATENCY_PROFILE_SEALED',flush=True)


def metrics(mean,std,truth):
    e=mean[:,:,:6]-truth[:,:,:6];delta=(mean[:,:,:6]-mean[:,:,:1])-(truth[:,:,:6]-truth[:,:,:1])
    s=mean[:,:,:6].max(-1).mean(2);t=truth[:,:,:6].max(-1).mean(2)
    regret=t.max(-1)-np.take_along_axis(t,s.argmax(-1)[...,None],-1)[...,0]
    return {'rmse_db':float(np.sqrt(np.mean(e**2))),'delta_rmse_db':float(np.sqrt(np.mean(delta**2))),'ap_regret':float(np.mean(regret)),
        'coverage_90':float(np.mean(np.abs(e)<=1.645*std[:,:,:6]))}


def case(task):
    stage,condition,seed=task;torch.set_num_threads(1);dest=OUT/'cases'/f'{stage}-{condition}-{seed}.json'
    if dest.exists():
        old=json.loads(dest.read_text());assert old['source_sha256']==sources();return old
    D=P['main_duration_seconds'] if stage=='test' and condition=='main' else P['selection_duration_seconds'] if stage=='selection' else P['other_duration_seconds']
    cadence=5 if condition=='sparse' else 1
    full,scan,g,obs=episode(seed,100,D,'drift' if condition=='drift' else 'main');obs=mask_scans(obs,cadence);pred=Predictors(g)
    predictions={f:[[],[]] for f in METHODS};actual={f:[] for f in METHODS};service={f:[] for f in METHODS}
    latency=json.loads((OUT/'latency.json').read_text())['p95_seconds'][str(cadence)];began=time.time()
    for d in range(D):
        ps,ts,refit=pred.step(obs.before(60+d),full.plan[60+d]);kind='cold' if d==0 else 'refit' if refit else 'warm'
        for f,(m,s) in ps.items():
            predictions[f][0].append(m.reshape(100,9,-1,3));predictions[f][1].append(s.reshape(100,9,-1,3));actual[f].append(ts[f])
            service[f].append(latency[f][kind]+(P['slow_extra_inference_seconds'] if condition=='slow' else 0))
    predictions={f:(np.asarray(x[0],dtype='f4'),np.asarray(x[1],dtype='f4')) for f,x in predictions.items()}
    future=np.arange(60,60+D)[:,None]+np.asarray(HORIZONS,dtype=int);truth=full.snr[future*5].transpose(0,2,1,3,4)
    forecast_stats={f:metrics(m,s,truth) for f,(m,s) in predictions.items()}
    selected=json.loads((OUT/'selection.json').read_text()) if stage=='test' else None;rows=[]
    reading=READINGS['bulk_uploads'];base=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),online_map=False,gate=True,scan_period_s=cadence,cadence_scan=True,
        roam_cost_s=P['slow_roam_seconds'] if condition=='slow' else P['normal_roam_seconds'])
    for f in METHODS:
        for planner in [selected['choices'][f]['planner']] if selected else P['planner_choices']:
            mean,std=predictions[f];world=runtime_world(full,D,mean,std[:,:,0]);queue=ForecastQueue(service[f])
            r=simulate(world,replace(base,name=f+'_'+planner),roam_gate=reactive_rescue,forecast_queue=queue,planner=planner,
                scan_duration_s=P['slow_scan_seconds'] if condition=='slow' else P['normal_scan_seconds'])
            rows.append({'family':f,'planner':planner,'timing_cost':True,'forecast_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),'result':r,'forecasts_consumed':queue.last+1,'last_forecast_age_seconds':queue.age})
            print(json.dumps({'stage':stage,'condition':condition,'site':seed,'family':f,'planner':planner,'misses':r['control_misses']}),flush=True)
        if stage=='test' and condition=='main' and f in ['radiomap',selected['selected_family']]:
            planner=selected['choices'][f]['planner'];mean,std=predictions[f]
            r=simulate(runtime_world(full,D,mean,std[:,:,0]),replace(base,name=f+'_free',scan_period_s=30.,cadence_scan=False),roam_gate=reactive_rescue,planner=planner)
            rows.append({'family':f,'planner':planner,'timing_cost':False,'forecast_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),'result':r})
    result={'completed':True,'stage':stage,'scene':seed,'condition':condition,'duration_s':D,'robots':100,'scan_period_s':cadence,'rows':rows,
        'forecast_metrics':forecast_stats,'actual_prediction_seconds':actual,'charged_service_seconds':service,'elapsed_s':time.time()-began,
        'source_sha256':sources(),'checkpoint_sha256':checkpoint_hashes(),'no_truth_in_forecast_inputs':True,'latency_sha256':digest(OUT/'latency.json')}
    write(dest,result);return result


def select(workers):
    if (OUT/'selection.json').exists():return
    benchmark();snapshot('selection')
    tasks=[('selection',c,s) for c in ['main','drift','sparse'] for s in P['selection_'+c+'_sites']]
    with ProcessPoolExecutor(workers) as pool:parts=list(pool.map(case,tasks))
    choices={}
    for f in METHODS:
        candidates=[]
        for planner in P['planner_choices']:
            ratios={}
            for condition in ['main','drift','sparse']:
                cs=[c for c in parts if c['condition']==condition]
                def misses(family,which):return sum(next(r['result']['control_misses'] for r in c['rows'] if r['family']==family and r['planner']==which) for c in cs)
                ratios[condition]=(misses(f,planner)+1)/(misses('radiomap','native')+1)
            candidates.append({'planner':planner,'score':float(np.mean(list(ratios.values()))),'condition_ratios':ratios})
        choices[f]=min(candidates,key=lambda r:r['score'])|{'candidates':candidates}
    family=min(P['families'],key=lambda f:choices[f]['score'])
    write(OUT/'selection.json',{'sealed':True,'sealed_at_unix':time.time(),'choices':choices,'selected_family':family,'source_sha256':sources(),
        'checkpoint_sha256':checkpoint_hashes(),'latency_sha256':digest(OUT/'latency.json'),'test_accessed':False})
    print('FRONTIER_SELECTION_SEALED',flush=True)


def test(workers):
    selected=json.loads((OUT/'selection.json').read_text());assert selected['source_sha256']==sources() and selected['latency_sha256']==digest(OUT/'latency.json')
    for name,h in selected['checkpoint_sha256'].items():assert digest(ROOT/name)==h
    snapshot('test');tasks=[('test',c,s) for c in ['main','drift','sparse','slow'] for s in P['test_'+c+'_sites']]
    with ProcessPoolExecutor(workers) as pool:rs=list(pool.map(case,tasks))
    write(OUT/'completion.json',{'completed':True,'fresh_sites':len(rs),'test_runs':sum(len(r['rows']) for r in rs),'selection_runs':96,'time_unix':time.time()})
    print('FRONTIER_ALL_TESTS_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','select','test','all'],default='all');p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    if a.phase in ['train','all']:train(a.workers)
    if a.phase in ['select','all']:select(a.workers)
    if a.phase in ['test','all']:test(a.workers)


if __name__=='__main__':main()
