"""Independent validation of the fixed compact model; preserves frontier results."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import argparse,json,time,shutil
import numpy as np
import torch
from .frontier.study import ROOT,OUT as DISC,sources,checkpoint_hashes,digest,Predictors,metrics,READINGS,runtime_world,write
from .frontier.fast import FleetMemory
from .frontier.timing import ForecastQueue
from .frontier.sim import simulate
from .fleetwm.data import episode
from .native.world import HORIZONS
from .native.intent_policy import intent_policy
from .gate_repair.gate import reactive_rescue

OUT=ROOT/'artifacts/frontier_confirmation';P=json.loads((OUT/'protocol.json').read_text())


def source_hashes():
    return sources()|{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),OUT/'protocol.json']}


def seal():
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())['source_sha256']==source_hashes();return
    counts={f:0 for f in ['distilled','packet','temporal']};discovery={}
    for s in P['discovery_sites']:
        p=DISC/'cases'/f'test-main-{s}.json';r=json.loads(p.read_text());discovery[str(p.relative_to(ROOT))]=digest(p)
        for f in counts:counts[f]+=next(x['result']['control_misses'] for x in r['rows'] if x['family']==f and x['timing_cost'])
    candidate=min(counts,key=lambda f:(counts[f],f));assert candidate==P['candidate'] and counts[candidate]==P['candidate_discovery_misses']
    for name in source_hashes():
        target=OUT/'executed_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write(path,{'sealed':True,'sealed_at_unix':time.time(),'candidate':candidate,'discovery_counts':counts,'discovery_sha256':discovery,
        'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'latency_sha256':digest(DISC/'latency.json'),'test_accessed':False})


def case(seed):
    torch.set_num_threads(1);dest=OUT/'cases'/f'{seed}.json'
    if dest.exists():
        r=json.loads(dest.read_text());assert r['source_sha256']==source_hashes();return r
    D=P['duration_seconds'];full,scan,g,obs=episode(seed,100,D);pred=Predictors(g)
    forecasts={f:[[],[]] for f in P['arms']};service={f:[] for f in P['arms']}
    profile=json.loads((DISC/'latency.json').read_text())['p95_seconds']['1']
    for d in range(D):
        ps,ts,refit=pred.step(obs.before(60+d),full.plan[60+d]);kind='cold' if d==0 else 'refit' if refit else 'warm'
        for f in P['arms']:
            m,s=ps[f];forecasts[f][0].append(m.reshape(100,9,-1,3));forecasts[f][1].append(s.reshape(100,9,-1,3));service[f].append(profile[f][kind])
    future=np.arange(60,60+D)[:,None]+np.asarray(HORIZONS,dtype=int);truth=full.snr[future*5].transpose(0,2,1,3,4)
    reading=READINGS['bulk_uploads'];base=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),online_map=False,gate=True,scan_period_s=1.,cadence_scan=True,roam_cost_s=.03)
    rows=[]
    for f in P['arms']:
        m,s=[np.asarray(v,dtype='f4') for v in forecasts[f]];q=ForecastQueue(service[f])
        result=simulate(runtime_world(full,D,m,s[:,:,0]),replace(base,name='compact_confirmation_'+f),roam_gate=reactive_rescue,forecast_queue=q,planner='native',scan_duration_s=.03)
        rows.append({'family':f,'result':result,'forecast_metrics':metrics(m,s,truth),'charged_service_seconds':service[f]})
        print(json.dumps({'confirmation_site':seed,'family':f,'misses':result['control_misses']}),flush=True)
    r={'completed':True,'scene':seed,'rows':rows,'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'latency_sha256':digest(DISC/'latency.json'),'no_truth_in_runtime_inputs':True}
    write(dest,r);return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();seal()
    selected=json.loads((OUT/'selection.json').read_text());assert selected['source_sha256']==source_hashes() and selected['checkpoint_sha256']==checkpoint_hashes()
    if not (OUT/'test_opened.json').exists():write(OUT/'test_opened.json',{'time_unix':time.time(),'selection_sha256':digest(OUT/'selection.json')})
    with ProcessPoolExecutor(a.workers) as pool:list(pool.map(case,P['test_sites']))
    print('COMPACT_CONFIRMATION_NATIVE_COMPLETE',flush=True)


if __name__=='__main__':main()
