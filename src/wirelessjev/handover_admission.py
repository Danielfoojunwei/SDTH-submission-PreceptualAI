"""Frozen all-handover admission experiment; no new learned parameters."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
import torch
from .frontier.study import ROOT,OUT as FRONT,sources,checkpoint_hashes,digest,Predictors,mask_scans,READINGS,runtime_world,write
from .frontier.timing import ForecastQueue
from .fleetwm.data import episode
from .hardware_budget import OutageBudgetGate
from .admission_sim import simulate
from .native.intent_policy import intent_policy
from .gate_repair.gate import reactive_rescue

OUT=ROOT/'artifacts/handover_admission';P=json.loads((OUT/'protocol.json').read_text())


def source_hashes():
    return sources()|{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/wirelessjev/admission_sim.py',ROOT/'src/wirelessjev/hardware_budget.py',OUT/'protocol.json']}


def case(seed):
    torch.set_num_threads(1);dest=OUT/'cases'/f'{seed}.json'
    if dest.exists():
        r=json.loads(dest.read_text());assert r['source_sha256']==source_hashes();return r
    D=60;full,scan,g,obs=episode(seed,100,D);obs=mask_scans(obs,5);pred=Predictors(g)
    forecast={f:[[],[]] for f in P['methods']};cost={f:[] for f in P['methods']};profile=json.loads((FRONT/'latency.json').read_text())['p95_seconds']['5']
    for d in range(D):
        ps,ts,refit=pred.step(obs.before(60+d),full.plan[60+d]);kind='cold' if d==0 else 'refit' if refit else 'warm'
        for f in P['methods']:
            m,s=ps[f];forecast[f][0].append(m.reshape(100,9,-1,3));forecast[f][1].append(s.reshape(100,9,-1,3));cost[f].append(profile[f][kind]+.5)
    reading=READINGS['bulk_uploads'];base=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),online_map=False,gate=True,scan_period_s=5.,cadence_scan=True,roam_cost_s=4.)
    rows=[]
    for f in P['methods']:
        mean,std=[np.asarray(v,dtype='f4') for v in forecast[f]]
        for variant in P['variants']:
            q=ForecastQueue(cost[f]);budget=OutageBudgetGate(mean,q,4.)
            r=simulate(runtime_world(full,D,mean,std[:,:,0]),replace(base,name=f+'_'+variant),roam_gate=budget if variant=='planned_only' else reactive_rescue,
                handover_admission=budget if variant=='all_handovers' else None,forecast_queue=q,scan_duration_s=.09)
            rows.append({'family':f,'variant':variant,'result':r});print(json.dumps({'admission_site':seed,'family':f,'variant':variant,'misses':r['control_misses']}),flush=True)
    result={'completed':True,'scene':seed,'rows':rows,'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'no_truth_in_admission_inputs':True,'latency_sha256':digest(FRONT/'latency.json')}
    write(dest,result);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();receipt=OUT/'test_source.json'
    if receipt.exists():assert json.loads(receipt.read_text())['source_sha256']==source_hashes()
    else:
        for name in source_hashes():
            p=OUT/'executed_source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        write(receipt,{'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'time_unix':time.time(),'latency_sha256':digest(FRONT/'latency.json')})
    with ProcessPoolExecutor(a.workers) as pool:list(pool.map(case,P['test_sites']))
    write(OUT/'completion.json',{'completed':True,'fresh_sites':6,'native_runs':48});print('ALL_HANDOVER_ADMISSION_COMPLETE',flush=True)


if __name__=='__main__':main()
