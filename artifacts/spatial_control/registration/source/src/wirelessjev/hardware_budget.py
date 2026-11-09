"""Charge slow sensing/handovers and admit only useful planned outages."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import argparse,json,time,shutil
import numpy as np
import torch
from .frontier.study import ROOT,OUT as FRONT,sources,checkpoint_hashes,digest,Predictors,mask_scans,READINGS,runtime_world,write
from .frontier.model import failure_table
from .frontier.sim import simulate
from .frontier.timing import ForecastQueue
from .fleetwm.data import episode
from .native.world import HORIZONS
from .native.intent_policy import intent_policy
from .gate_repair.gate import reactive_rescue

OUT=ROOT/'artifacts/hardware_budget';P=json.loads((OUT/'protocol.json').read_text())


class OutageBudgetGate:
    def __init__(self,mean,queue,cost):
        self.mean=mean;self.queue=queue;self.cost=cost;self.table=failure_table();self.grid=np.linspace(-40,60,501)

    def __call__(self,ctx):
        fc=self.queue.advance(self.mean[ctx.decision]);ar=np.arange(len(fc))
        def failure(site):
            snr=fc[ar,:,site]
            return np.stack([np.interp(snr[...,b],self.grid,self.table[b]) for b in range(3)],-1).min(-1)
        current=failure(ctx.current);candidate=failure(ctx.candidate);h=np.asarray(HORIZONS)
        stay=np.trapezoid(current,h,axis=1)
        after=h>=self.cost
        switch=self.cost+np.trapezoid(candidate[:,after],h[after],axis=1)
        return ctx.eligible & np.isfinite(stay-switch) & (stay>switch)


def source_hashes():return sources()|{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),OUT/'protocol.json']}


def snapshot(stage):
    receipt=OUT/(stage+'_source.json')
    if receipt.exists():assert json.loads(receipt.read_text())['source_sha256']==source_hashes();return
    for name in source_hashes():
        path=OUT/(stage+'_source')/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    write(receipt,{'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'time_unix':time.time(),'latency_sha256':digest(FRONT/'latency.json')})


def case(task):
    stage,seed=task;torch.set_num_threads(1);dest=OUT/'cases'/f'{stage}-{seed}.json'
    if dest.exists():
        r=json.loads(dest.read_text());assert r['source_sha256']==source_hashes();return r
    D=P['duration_seconds'];full,scan,g,original=episode(seed,100,D);latency=json.loads((FRONT/'latency.json').read_text())['p95_seconds']
    selected=json.loads((OUT/'selection.json').read_text()) if stage=='test' else None;rows=[]
    for cadence in P['scan_period_options']:
        needed=P['methods'] if stage=='development' else [f for f in P['methods'] if selected['choices'][f]['scan_period']==cadence or cadence==1]
        if not needed:continue
        obs=mask_scans(original,cadence);pred=Predictors(g);forecast={f:[[],[]] for f in needed};service={f:[] for f in needed}
        for d in range(D):
            ps,ts,refit=pred.step(obs.before(60+d),full.plan[60+d]);kind='cold' if d==0 else 'refit' if refit else 'warm'
            for f in needed:
                m,s=ps[f];forecast[f][0].append(m.reshape(100,9,-1,3));forecast[f][1].append(s.reshape(100,9,-1,3));service[f].append(latency[str(cadence)][f][kind]+.5)
        reading=READINGS['bulk_uploads'];base=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),online_map=False,gate=True,scan_period_s=cadence,cadence_scan=True,roam_cost_s=4.)
        for f in needed:
            gates=P['gate_options'] if stage=='development' else [g for g in P['gate_options'] if (cadence==1 and g=='rescue') or (selected['choices'][f]['scan_period']==cadence and selected['choices'][f]['gate']==g)]
            mean,std=[np.asarray(v,dtype='f4') for v in forecast[f]]
            for gate in gates:
                q=ForecastQueue(service[f]);hook=reactive_rescue if gate=='rescue' else OutageBudgetGate(mean,q,4.)
                result=simulate(runtime_world(full,D,mean,std[:,:,0]),replace(base,name=f+'_'+gate),roam_gate=hook,forecast_queue=q,scan_duration_s=.09)
                rows.append({'family':f,'scan_period':cadence,'gate':gate,'result':result})
                print(json.dumps({'hardware_stage':stage,'scene':seed,'family':f,'scan_period':cadence,'gate':gate,'misses':result['control_misses']}),flush=True)
    r={'completed':True,'stage':stage,'scene':seed,'rows':rows,'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'no_truth_in_runtime_inputs':True,'latency_sha256':digest(FRONT/'latency.json')}
    write(dest,r);return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();snapshot('development')
    if not (OUT/'selection.json').exists():
        with ProcessPoolExecutor(a.workers) as pool:dev=list(pool.map(case,[('development',s) for s in P['development_sites']]))
        choices={}
        for f in P['methods']:
            options=[]
            for cadence in P['scan_period_options']:
                for gate in P['gate_options']:
                    rs=[next(r['result'] for r in c['rows'] if r['family']==f and r['scan_period']==cadence and r['gate']==gate) for c in dev]
                    options.append({'scan_period':cadence,'gate':gate,'misses':sum(r['control_misses'] for r in rs),'mean_bulk_mbps':float(np.mean([r['bulk_mbps'] for r in rs]))})
            choices[f]=min(options,key=lambda r:(r['misses'],-r['mean_bulk_mbps']))|{'candidates':options}
        write(OUT/'selection.json',{'sealed':True,'sealed_at_unix':time.time(),'choices':choices,'source_sha256':source_hashes(),'checkpoint_sha256':checkpoint_hashes(),'test_accessed':False})
    choice=json.loads((OUT/'selection.json').read_text());assert choice['source_sha256']==source_hashes()
    snapshot('test')
    with ProcessPoolExecutor(a.workers) as pool:tests=list(pool.map(case,[('test',s) for s in P['test_sites']]))
    write(OUT/'completion.json',{'completed':True,'development_runs':64,'fresh_sites':4,'test_runs':sum(len(c['rows']) for c in tests)})
    print('HARDWARE_BUDGET_COMPLETE',flush=True)


if __name__=='__main__':main()
