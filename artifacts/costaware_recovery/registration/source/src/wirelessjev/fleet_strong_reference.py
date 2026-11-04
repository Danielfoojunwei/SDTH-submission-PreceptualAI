"""Stronger cross-site RadioMap comparison, added before test access."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,json,time,hashlib,shutil
import numpy as np
import torch
from .fleetwm.study import ROOT,OUT,P,digest,sources,forecast_metrics
from .fleetwm.data import episode,map_prediction
from .contracted.data import YardRadioMap,runtime_world,READINGS
from .contracted.train import write
from .native.radiomap import RadioMap
from .native.world import HORIZONS
from .native.intent_policy import intent_policy
from .native.sim import simulate
from .gate_repair.gate import reactive_rescue

REF=OUT/'strong_reference'


def prior():
    raw=json.loads((ROOT/'artifacts/native/foundation.json').read_text())['prior']
    return {int(k):{key:np.array(value) if isinstance(value,list) else value for key,value in p.items()} for k,p in raw.items()}


def fit_strong_map(g,past):
    start=max(0,len(past.value)-60);pos=past.positions[start:];obs={};cens={}
    robots=np.broadcast_to(np.arange(pos.shape[1]),pos.shape[:2]);times=np.arange(start,len(past.value))[:,None]
    for s in range(len(g.sites)):
        for b in range(3):
            j=s*3+b;h=past.heard[start:,:,j];c=past.scanned[start:]&~h&(times%5==0)
            obs[s,b]=(pos[...,0][h],pos[...,1][h],past.value[start:,:,j][h],robots[h])
            cens[s,b]=(pos[...,0][c],pos[...,1][c],past.bound[start:,:,j][c],robots[c])
    cls=YardRadioMap if g.yard else RadioMap
    return cls(g.sites,g.aps,0,prior()).fit(obs,cens,np.arange(max(1,pos.shape[1]//5)))


def ref_sources():
    paths=[ROOT/'src/wirelessjev/fleet_strong_reference.py',ROOT/'artifacts/native/foundation.json',OUT/'strong_reference_amendment.json']
    return sources()|{str(p.relative_to(ROOT)):digest(p) for p in paths}


def snapshot(stage):
    receipt=REF/(stage+'_source.json');hashes=ref_sources()
    if receipt.exists():
        if json.loads(receipt.read_text())['source_sha256']!=hashes:raise RuntimeError('Reference source changed')
        return
    for name in hashes:
        dest=REF/(stage+'_source')/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(receipt,{'source_sha256':hashes,'time_unix':time.time()})


def case(task):
    stage,condition,seed=task;torch.set_num_threads(1);path=REF/'cases'/f'{stage}-{condition}-{seed}.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['source_sha256']!=ref_sources():raise RuntimeError('Reference case source changed')
        return old
    n=10 if condition=='small' else 100
    duration=P['primary_duration_s'] if stage=='test' and condition=='main' else P['selection_duration_s'] if stage=='selection' else P['generalisation_duration_s']
    full,scan,g,obs=episode(seed,n,duration,condition);means=[];stds=[];elapsed=0;refits=0;rmap=None
    for d in range(duration):
        t=60+d
        if d%20==0:
            a=time.time();rmap=fit_strong_map(g,obs.before(t));refits+=time.time()-a
        a=time.time();m,s=map_prediction(g,rmap,full.plan[t]);elapsed+=time.time()-a
        means.append(m);stds.append(s)
    mean=np.asarray(means,dtype='f4');std=np.asarray(stds,dtype='f4')
    future=np.arange(60,60+duration)[:,None]+np.asarray(HORIZONS,dtype=int);truth=full.snr[future*5].transpose(0,2,1,3,4)
    gates=[*P['gate_multipliers'],'rescue'] if stage=='selection' else [json.loads((REF/'selection.json').read_text())['gate']]
    missions=[P['primary_mission']]+([P['heldout_mission']] if stage=='test' and condition=='main' else [])
    rows=[]
    for mission in missions:
        for gate in gates:
            pol=replace(intent_policy(READINGS[mission]['weights'],READINGS[mission]['focus'],dual_hardware='none'),online_map=False,gate=True,belief_scale=1 if gate=='rescue' else gate,name='learned_radiomap_'+str(gate))
            r=simulate(runtime_world(full,duration,mean,std[:,:,0]),pol,roam_gate=reactive_rescue if gate=='rescue' else None,collision_lambda=.3 if condition=='stress' else 0)
            rows.append({'family':'learned_radiomap','gate':gate,'mission':mission,'forecast_sha256':hashlib.sha256(mean.tobytes()).hexdigest(),'result':r})
            print(json.dumps({'reference_stage':stage,'site':seed,'condition':condition,'gate':gate,'mission':mission,'misses':r['control_misses']}),flush=True)
    result={'stage':stage,'scene':seed,'condition':condition,'robots':n,'duration_s':duration,'rows':rows,'forecast_metrics':forecast_metrics(mean,std,truth,g),'predict_seconds':elapsed,'refit_seconds':refits,'source_sha256':ref_sources()}
    write(path,result);return result


def run(stage,workers):
    if stage=='selection':
        if (REF/'selection.json').exists():return
        core=json.loads((OUT/'selection.json').read_text());snapshot(stage)
    else:
        seal=json.loads((OUT/'comparison_seal.json').read_text())
        assert seal['core_selection_sha256']==digest(OUT/'selection.json') and seal['reference_selection_sha256']==digest(REF/'selection.json')
        snapshot(stage)
    tasks=[(stage,c,s) for c in ['main','stress','yard','small','drift'] for s in P[('selection' if stage=='selection' else 'test')+'_'+c+'_sites']]
    with ProcessPoolExecutor(workers) as pool:parts=list(pool.map(case,tasks))
    if stage=='selection':
        candidates=[]
        for gate in [*P['gate_multipliers'],'rescue']:
            ratios={}
            for c in ['main','stress','yard','small','drift']:
                cs=[x for x in parts if x['condition']==c]
                denominator=0
                for x in cs:
                    old=json.loads((OUT/'cases'/f'selection-{c}-{x["scene"]}.json').read_text())
                    denominator+=next(r['result']['control_misses'] for r in old['rows'] if r['family']=='rolling_radiomap' and r['gate']==1.)
                misses=sum(next(r['result']['control_misses'] for r in x['rows'] if r['gate']==gate) for x in cs)
                ratios[c]=(misses+1)/(denominator+1)
            candidates.append({'gate':gate,'score':float(np.mean(list(ratios.values()))),'ratios':ratios})
        selected=min(candidates,key=lambda r:r['score'])
        write(REF/'selection.json',selected|{'candidates':candidates,'sealed':True,'sealed_at_unix':time.time(),'source_sha256':ref_sources(),'test_accessed':False})
        scores={f:core['choices'][f]['score'] for f in ['static_radiomap','rolling_radiomap']};scores['learned_radiomap']=selected['score']
        write(OUT/'comparison_seal.json',{'sealed':True,'sealed_at_unix':time.time(),'core_selection_sha256':digest(OUT/'selection.json'),'reference_selection_sha256':digest(REF/'selection.json'),'strongest_development_radiomap':min(scores,key=scores.get),'reference_scores':scores,'test_accessed':False})
        print('STRONG_REFERENCE_AND_COMPARISON_SEALED',flush=True)
    else:
        write(REF/'completion.json',{'completed':True,'fresh_scenes':len(parts),'test_rollouts':sum(len(p['rows']) for p in parts),'selection_rollouts':60})
        print('STRONG_REFERENCE_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['selection','test'],required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();run(args.phase,args.workers)


if __name__=='__main__':main()
