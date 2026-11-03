"""Source-sealed packet-action study; no test access before selection."""
import argparse,json,shutil,time,gzip
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
import torch
from ..frontier.study import ROOT,sources as prior_sources,digest,mask_scans,READINGS,runtime_world,write
from ..closed_loop.runtime import SinglePredictor,LiveForecasts
from ..closed_loop.telemetry import DeliveredScans
from ..fleetwm.data import episode,Geometry
from ..native.intent_policy import intent_policy
from ..gate_repair.gate import reactive_rescue
from ..hardware_budget import OutageBudgetGate
from .features import action_queries
from .model import PacketValueModel,objective,update_target
from .control import PacketController
from .sim import simulate

OUT=ROOT/'artifacts/packet_value';P=json.loads((OUT/'protocol.json').read_text())


def source_hashes():
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src/wirelessjev/closed_loop').glob('*.py'))
    files += [OUT/'protocol.json',ROOT/'src/wirelessjev/hardware_budget.py']
    return prior_sources()|{str(f.relative_to(ROOT)):digest(f) for f in files}


def snapshot(stage):
    dest=OUT/(stage+'_source.json');hashes=source_hashes()
    if dest.exists():
        assert json.loads(dest.read_text())['source_sha256']==hashes;return
    for name in hashes:
        path=OUT/(stage+'_source')/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    write(dest,{'source_sha256':hashes,'time_unix':time.time()})


def policy_for(cost,name='packet_value'):
    r=READINGS['bulk_uploads']
    return replace(intent_policy(r['weights'],r['focus'],dual_hardware='none'),name=name,
                   online_map=False,gate=True,scan_period_s=5.,cadence_scan=True,roam_cost_s=cost)


def make_case(seed,n,duration,cost,*,model=None,correction=0.,head_seconds=.01,final_timing=False):
    full,_,g,o=episode(seed,n,duration,'drift' if seed%2 else 'main');o=mask_scans(o,5)
    delivered=DeliveredScans(o.before(60),duration,seed)
    profile=json.loads((ROOT/'artifacts/closed_loop/latency.json').read_text())['p95_seconds']['radiomap']
    if final_timing:profile=json.loads((OUT/'latency.json').read_text())['forecast_seconds']
    live=LiveForecasts(SinglePredictor(g,'radiomap'),delivered,duration,profile,.5)
    controller=PacketController(g,cost,model=model,correction=correction,head_seconds=head_seconds)
    world=runtime_world(full,duration,live.mean,live.std[:,:,0])
    kwargs={'roam_gate':reactive_rescue,'forecast_queue':live,'forecast_source':live,
            'scan_observer':delivered,'scan_duration_s':.09,'external_primary':controller}
    return world,g,controller,kwargs


def label_case(task):
    split,index,seed,cost=task;torch.set_num_threads(1)
    key=f'{split}-{seed}-{cost}';receipt=OUT/'labels'/(key+'.json');data=OUT/'data'/(key+'.npz')
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_sha256']==source_hashes() and r['data_sha256']==digest(data);return r
    started=time.time();n=100 if split=='validation' or index%2 else 24
    world,g,controller,kwargs=make_case(seed,n,30,cost);cps=[];p=policy_for(cost)
    factual=simulate(world,p,checkpoint_at={500,1500},checkpoint_sink=cps.append,detailed=True,**kwargs)
    pieces=[];records=[];rng=np.random.default_rng(seed+345000);stay_runs=0
    for cp in cps:
        obs=cp.observable();ctrl=cp.bundle['hooks'][-1]
        context=ctrl.history.context();query=action_queries(g,obs,cost)
        available=np.flatnonzero((obs['outage_remaining']==0)&(obs['scan_remaining']==0))
        if split=='training':
            weak=available[np.argsort(obs['pilot_snr'][available].max(-1))[:2]]
            others=np.setdiff1d(available,weak)
            focus=np.r_[weak,rng.choice(others,min(2,len(others)),replace=False)]
        else:focus=rng.choice(available,min(4,len(available)),replace=False)
        for robot in focus:
            robot=int(robot);current=int(obs['current'][robot]);local=obs['local_rssi'][robot]
            candidates=np.flatnonzero((local>=-90).any(-1)&(np.arange(len(g.sites))!=current))
            quality=np.where(local>=-90,local-g.tx+15-g.noise,-np.inf).max(-1)
            ordered=candidates[np.argsort(-quality[candidates])];best=ordered[:2];rest=np.setdiff1d(candidates,best)
            targets=np.r_[best,rng.choice(rest,min(1,len(rest)),replace=False)]
            if not len(targets):continue
            stay=simulate(world,p,resume=cp,stop_slot=cp.slot+1000,force_action=(robot,current),scan_duration_s=.09,detailed=True)
            stay_runs+=1;sp=stay['packet_detail'];base=float(obs['misses'][robot]);stay_miss=sp['misses'][robot]-base
            numeric=ctrl.numeric(obs)
            for target in targets:
                target=int(target)
                move=simulate(world,p,resume=cp,stop_slot=cp.slot+1000,force_action=(robot,target),scan_duration_s=.09,detailed=True)
                mp=move['packet_detail'];move_miss=mp['misses'][robot]-base
                frames=mp['controller_frames'];future=np.stack([x['frame'][robot] for x in frames])
                if len(future)<6:future=np.r_[np.repeat(future[:1],6-len(future),axis=0),future]
                delta=float(move_miss-stay_miss);gap=(mp['window_max_gap'][robot]-sp['window_max_gap'][robot])*.01
                pieces.append({'context':context[robot],'query':query[robot,target],'delta_misses':delta,'delta_gap':gap,
                    'future_context':future,'future_valid':np.ones(6,dtype=bool),'numeric_delta':numeric[robot,target],
                    'cost':cost,'scene':seed})
                records.append({'fork_slot':cp.slot,'robot':robot,'current':current,'candidate':target,
                    'prefix_misses':base,'stay_misses':stay_miss,'move_misses':move_miss,'delta_misses':delta,
                    'stay_window_max_gap':sp['window_max_gap'][robot],'move_window_max_gap':mp['window_max_gap'][robot],
                    'delta_gap_seconds':gap,'future_slots':[x['slot'] for x in frames]})
        replay=simulate(world,p,resume=cp,stop_slot=cp.slot+1000,scan_duration_s=.09,detailed=True)
        endpoint=next(row for row in factual['packet_detail']['cumulative'] if row['slot_end']==cp.slot+1000)
        assert replay['packet_detail']['misses']==endpoint['misses']
    arrays={k:np.asarray([row[k] for row in pieces]) for k in pieces[0]} if pieces else {}
    data.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(data,**arrays)
    if split=='validation' and index==0 and cost==.03:
        cp=cps[0];ctrl=cp.bundle['hooks'][-1];public=cp.observable()
        needed=['slot','current','forecast','pilot_snr','local_rssi','positions','plan','gap','misses','sent',
                'outage_remaining','scan_remaining','roams']
        np.savez_compressed(OUT/'data/calibration_inputs.npz',context=ctrl.history.context(),
            query=action_queries(g,public,cost),sites=g.sites,tx=g.tx,noise=g.noise,
            **{'obs_'+k:public[k] for k in needed})
    r={'completed':True,'split':split,'scene':seed,'cost':cost,'robots':n,'examples':len(pieces),'records':records,
       'source_sha256':source_hashes(),'data_sha256':digest(data),'factual_replays':len(cps),
       'elapsed_seconds':time.time()-started,'native_runs':len(records)+stay_runs+len(cps)+1,
       'forecast_timing':kwargs['forecast_source'].records}
    write(receipt,r);print(json.dumps({'labels':key,'examples':len(pieces),'elapsed':round(time.time()-started,1)}),flush=True);return r


def arrays_for(split):
    paths=sorted((OUT/'data').glob(split+'-*.npz'))
    rows=[]
    for path in paths:
        with np.load(path) as z:
            if z.files:rows.append({k:z[k] for k in z.files})
    if not rows:raise RuntimeError('No factual action examples')
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def diagnostic():
    a=arrays_for('training');v=arrays_for('validation');groups={}
    for split,x in [('training',a),('validation',v)]:
        groups[split]={}
        for cost in P['handover_seconds']:
            y=x['delta_misses'][x['cost']==cost]
            groups[split][str(cost)]={'examples':len(y),'beneficial':int((y<0).sum()),
                'harmful':int((y>0).sum()),'zero':int((y==0).sum()),
                'minimum_delta':float(y.min()),'maximum_delta':float(y.max())}
    passed=all(groups['training'][str(c)]['beneficial']>0 for c in P['handover_seconds'])
    r={'completed':True,'passed':passed,'groups':groups,'source_sha256':source_hashes(),
        'native_runs':sum(json.loads(p.read_text())['native_runs'] for p in (OUT/'labels').glob('*.json')),
        'no_all_stay_success_claim':True}
    write(OUT/'diagnostic.json',r);print('PACKET_DIAGNOSTIC',json.dumps(r['groups']),flush=True)
    return passed


def generate_data(workers):
    snapshot('data')
    tasks=[(split,i,s,c) for split in ['training','validation'] for i,s in enumerate(P[split+'_sites']) for c in P['handover_seconds']]
    with ProcessPoolExecutor(workers) as pool:list(pool.map(label_case,tasks))
    return diagnostic()


def batch(a,index):
    return {k:torch.as_tensor(v[index].astype('f4')) for k,v in a.items() if k in (
        'context','query','delta_misses','delta_gap','future_context','future_valid')}


@torch.no_grad()
def validation_score(model,a):
    misses=[];gaps=[]
    for start in range(0,len(a['context']),128):
        b=batch(a,slice(start,start+128));p=model(b['context'],b['query'])
        misses.extend((p['delta_misses']-b['delta_misses']).abs().tolist())
        gaps.extend((p['delta_gap']-b['delta_gap']).abs().tolist())
    return float(np.mean(misses)+5*np.mean(gaps))


def fit_one(task):
    family,seed=task;torch.set_num_threads(1);dest=OUT/'checkpoints'/f'{family}-{seed}.pt';report=OUT/'fits'/f'{family}-{seed}.json'
    if report.exists():
        old=json.loads(report.read_text());assert old['source_sha256']==source_hashes() and old['checkpoint_sha256']==digest(dest);return old
    train=arrays_for('training');val=arrays_for('validation');torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=PacketValueModel(family=family);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float('inf');history=[];dest.parent.mkdir(parents=True,exist_ok=True);started=time.time()
    for epoch in range(P['epochs']):
        model.train();order=rng.permutation(len(train['context']));losses=[]
        for start in range(0,len(order),P['batch_size']):
            b=batch(train,order[start:start+P['batch_size']]);loss,_=objective(model,b)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite factual packet loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if family=='jepa':update_target(model)
            losses.append(float(loss.detach()))
        model.eval();score=validation_score(model,val)
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_score':score};history.append(row)
        if score<best:
            best=score;torch.save({'config':model.config,'state':model.state_dict(),'seed':seed,'epoch':epoch+1},dest)
        print(json.dumps({'fit':family,'seed':seed,**row}),flush=True)
    r={'completed':True,'family':family,'seed':seed,'history':history,'best_validation_score':best,
       'inference_parameters':model.inference_parameters,'checkpoint_sha256':digest(dest),
       'source_sha256':source_hashes(),'elapsed_seconds':time.time()-started,'test_accessed':False}
    write(report,r);return r


def load_model(family,seed):
    state=torch.load(OUT/'checkpoints'/f'{family}-{seed}.pt',weights_only=True,map_location='cpu')
    model=PacketValueModel(**state['config']);model.load_state_dict(state['state']);model.eval();return model


def weights():return {str(p.relative_to(ROOT)):digest(p) for p in sorted((OUT/'checkpoints').glob('*.pt'))}


def train(workers):
    assert json.loads((OUT/'diagnostic.json').read_text())['passed']
    snapshot('training')
    with ProcessPoolExecutor(workers) as pool:rows=list(pool.map(fit_one,[(f,s) for f in P['families'] for s in P['training_seeds']]))
    write(OUT/'training.json',{'completed':True,'fits':len(rows),'source_sha256':source_hashes(),
        'checkpoint_sha256':weights(),'test_accessed':False,'time_unix':time.time()})
    print('SIX_MATCHED_PACKET_FITS_COMPLETE',flush=True)


@torch.no_grad()
def calibrate():
    if (OUT/'calibration.json').exists() and (OUT/'latency.json').exists():
        for name in ['calibration','latency']:
            old=json.loads((OUT/(name+'.json')).read_text())
            assert old['source_sha256']==source_hashes() and old['checkpoint_sha256']==weights()
        return
    if (OUT/'selection.json').exists():raise RuntimeError('No recalibration after development')
    torch.set_num_threads(1);a=arrays_for('validation');predictions={'radiomap':a['numeric_delta']};models={}
    for family in P['families']:
        for seed in P['training_seeds']:
            key=f'{family}-{seed}';model=load_model(family,seed);models[key]=model;pieces=[]
            for start in range(0,len(a['context']),128):
                b=batch(a,slice(start,start+128));pieces.append(model(b['context'],b['query'])['delta_misses'].numpy())
            predictions[key]=np.concatenate(pieces)
    corrections={};raw={}
    for key,pred in predictions.items():
        corrections[key]={};raw[key]={}
        for cost in P['handover_seconds']:
            mask=a['cost']==cost;error=a['delta_misses'][mask]-pred[mask]
            corrections[key][str(cost)]=max(0.,float(np.quantile(error,.95,method='higher')))
            raw[key][str(cost)]={'examples':int(mask.sum()),'residuals':error.tolist()}
    write(OUT/'calibration.json',{'sealed':True,'time_unix':time.time(),'corrections':corrections,'raw':raw,
        'source_sha256':source_hashes(),'checkpoint_sha256':weights(),'test_accessed':False})
    with np.load(OUT/'data/calibration_inputs.npz') as z:
        public={k[4:]:z[k].copy() for k in z.files if k.startswith('obs_')}
        g=Geometry(z['sites'].copy(),z['tx'].copy(),z['noise'].copy(),())
    public['slot']=int(public['slot']);first_slot=public['slot']
    times={}
    for key,model in {'radiomap':None,**models}.items():
        ctrl=PacketController(g,.03,model=model,head_seconds=.01);values=[]
        for i in range(32):
            public['slot']=first_slot+i*100
            if ctrl.next_decision is not None:ctrl.next_decision[:]=0
            t=time.perf_counter();ctrl.observe(public);elapsed=time.perf_counter()-t
            if i>=2:values.append(elapsed)
        times[key]=values
    neural=max(float(np.quantile(v,.95)) for k,v in times.items() if k!='radiomap')
    old=json.loads((ROOT/'artifacts/closed_loop/latency.json').read_text())['p95_seconds']['radiomap']
    measured={k:[] for k in old}
    for p in (OUT/'labels').glob('validation-*.json'):
        for r in json.loads(p.read_text())['forecast_timing']:measured[r['kind']].append(r['actual_prediction_seconds'])
    forecast={k:max(v,float(np.quantile(measured[k],.95)) if measured[k] else v) for k,v in old.items()}
    head={'radiomap':float(np.quantile(times['radiomap'],.95)),'neural':neural}
    write(OUT/'latency.json',{'sealed':True,'time_unix':time.time(),'head_seconds':head,
        'forecast_seconds':forecast,'model_head_raw_seconds':times,'forecast_raw_seconds':measured,
        'source_sha256':source_hashes(),'checkpoint_sha256':weights(),'test_accessed':False,
        'full_feature_guard_and_ledger_measured':True})
    print('PACKET_CALIBRATION_AND_TIMING_SEALED',flush=True)


def evaluation_case(task):
    stage,site,cost,family,model_seed=task;torch.set_num_threads(1)
    key=f'{stage}-{site}-{cost}-{family}-{model_seed}';dest=OUT/'cases'/(key+'.json')
    cp_hash=weights();latency_sha=digest(OUT/'latency.json');calibration_sha=digest(OUT/'calibration.json')
    if dest.exists():
        old=json.loads(dest.read_text());assert old['source_sha256']==source_hashes() and old['checkpoint_sha256']==cp_hash
        assert old['latency_sha256']==latency_sha and old['calibration_sha256']==calibration_sha;return old
    began=time.time();duration=P['confirmation_duration_seconds'] if stage=='test' else P['development_duration_seconds']
    key_model=f'{family}-{model_seed}' if family in P['families'] else 'radiomap'
    cal=json.loads((OUT/'calibration.json').read_text())['corrections'];profile=json.loads((OUT/'latency.json').read_text())['head_seconds']
    model=load_model(family,model_seed) if family in P['families'] else None
    world,g,ctrl,kwargs=make_case(site,100,duration,cost,model=model,correction=cal[key_model][str(cost)],
        head_seconds=profile['neural' if model is not None else 'radiomap'],final_timing=True)
    if family=='canonical':
        kwargs.pop('external_primary');kwargs['handover_admission']=OutageBudgetGate(world.forecast,kwargs['forecast_queue'],cost)
    result=simulate(world,policy_for(cost,family),detailed=True,**kwargs)
    detail=result['packet_detail'];detail.pop('controller_frames',None);detail.pop('controller_summary',None)
    events=detail.pop('delivered_report_events');telemetry=OUT/'telemetry'/(key+'.json.gz');telemetry.parent.mkdir(exist_ok=True)
    telemetry.write_bytes(gzip.compress(json.dumps(events,separators=(',',':')).encode(),mtime=0))
    detail['cumulative']=[{'slot_end':r['slot_end'],'misses':int(sum(r['misses'])),'sent':int(sum(r['sent'])),
        'gap':int(max(r['gap'])),'max_gap':int(max(r['max_gap']))} for r in detail['cumulative']]
    detail['cumulative_scope']='Fleet totals; final per-robot counters remain explicit'
    r={'completed':True,'stage':stage,'scene':site,'cost':cost,'family':family,'model_seed':model_seed,
       'duration_seconds':duration,'result':result,'controller':None if family=='canonical' else ctrl.summary(),
       'source_sha256':source_hashes(),'checkpoint_sha256':cp_hash,'latency_sha256':latency_sha,
       'calibration_sha256':calibration_sha,'simulation_only':True,'elapsed_seconds':time.time()-began,
       'telemetry_file':str(telemetry.relative_to(OUT)),'telemetry_sha256':digest(telemetry)}
    write(dest,r);print(json.dumps({'evaluation':key,'misses':result['control_misses'],'roams':result['roams'],
        'elapsed':round(time.time()-began,1)}),flush=True);return r


def evaluation_tasks(stage):
    return [(stage,s,c,f,seed) for s in P[('test' if stage=='test' else 'development')+'_sites'] for c in P['handover_seconds']
        for f in P['families']+['radiomap','canonical'] for seed in (P['training_seeds'] if f in P['families'] else [None])]


def development(workers):
    if (OUT/'selection.json').exists():
        old=json.loads((OUT/'selection.json').read_text());assert old['source_sha256']==source_hashes();return
    snapshot('development')
    with ProcessPoolExecutor(workers) as pool:list(pool.map(evaluation_case,evaluation_tasks('development')))
    write(OUT/'selection.json',{'sealed':True,'sealed_at_unix':time.time(),'source_sha256':source_hashes(),
        'checkpoint_sha256':weights(),'latency_sha256':digest(OUT/'latency.json'),'calibration_sha256':digest(OUT/'calibration.json'),
        'test_accessed':False,'no_development_tuning':True,'all_training_seeds_retained':P['training_seeds'],
        'rule':'Prespecified common controller and validation corrections; no family-specific policy tuning'})
    print('PACKET_DEVELOPMENT_SEALED',flush=True)


def confirmation(workers):
    seal=json.loads((OUT/'selection.json').read_text());assert seal['sealed'] and seal['source_sha256']==source_hashes()
    assert seal['checkpoint_sha256']==weights() and seal['latency_sha256']==digest(OUT/'latency.json')
    assert seal['calibration_sha256']==digest(OUT/'calibration.json');snapshot('test');tasks=evaluation_tasks('test')
    with ProcessPoolExecutor(workers) as pool:list(pool.map(evaluation_case,tasks))
    write(OUT/'completion.json',{'completed':True,'diagnostic_failed':False,'fits':6,
        'label_native_runs':json.loads((OUT/'diagnostic.json').read_text())['native_runs'],
        'development_runs':len(evaluation_tasks('development')),'test_runs':len(tasks),'fresh_sites':len(P['test_sites'])})
    print('PACKET_JEPA_CONFIRMATION_COMPLETE',flush=True)


def verify_seeds():
    keys=['training_sites','validation_sites','development_sites','test_sites'];sets=[set(P[k]) for k in keys]
    for i,a in enumerate(sets):
        for b in sets[i+1:]:assert not a&b
    def values(x):
        if isinstance(x,dict):return set().union(*(values(v) for v in x.values())) if x else set()
        if isinstance(x,list):return set().union(*(values(v) for v in x)) if x else set()
        return {x} if isinstance(x,int) else set()
    used=set.union(*sets)
    for path in (ROOT/'artifacts').glob('*/protocol.json'):
        if path.parent!=OUT:assert not used&values(json.loads(path.read_text())),str(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','data','train','calibrate','development','test'],default='all')
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();verify_seeds()
    if args.phase in ('all','data'):
        if not generate_data(args.workers):
            write(OUT/'completion.json',{'completed':True,'diagnostic_failed':True,'fits':0,'test_runs':0,
                'reason':'No beneficial factual training action at one or more physical costs; no all-stay success claim'})
            return
    if args.phase in ('all','train'):train(args.workers)
    if args.phase in ('all','calibrate'):calibrate()
    if args.phase in ('all','development'):development(args.workers)
    if args.phase in ('all','test'):confirmation(args.workers)


if __name__=='__main__':main()
