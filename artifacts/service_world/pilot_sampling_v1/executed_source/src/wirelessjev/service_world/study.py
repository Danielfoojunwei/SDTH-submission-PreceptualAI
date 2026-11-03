"""Finite native capability audit; hindsight headroom never qualifies a policy."""
import argparse,json,shutil,time
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from ..packet_value.study import ROOT,policy_for
from ..frontier.study import sources as prior_sources,digest,mask_scans,runtime_world,write
from ..fleetwm.data import episode
from ..closed_loop.runtime import SinglePredictor,LiveForecasts
from ..closed_loop.telemetry import DeliveredScans
from ..hardware_budget import OutageBudgetGate
from ..gate_repair.gate import reactive_rescue
from ..frontier.timing import ForecastQueue
from .features import RadioHistory
from .model import ServiceWorldModel
from .physics import forced_deadline_misses
from .sim import simulate

OUT=ROOT/'artifacts/service_world';P=json.loads((OUT/'protocol.json').read_text())


def sources():
    files=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'src/wirelessjev/closed_loop').glob('*.py'))
    files += [ROOT/'src/wirelessjev/packet_value/study.py',ROOT/'src/wirelessjev/packet_value/model.py',
              ROOT/'src/wirelessjev/packet_value/control.py',ROOT/'src/wirelessjev/packet_value/features.py',
              ROOT/'src/wirelessjev/hardware_budget.py',OUT/'protocol.json',ROOT/'artifacts/packet_value/latency.json']
    return prior_sources()|{str(p.relative_to(ROOT)):digest(p) for p in files}


def snapshot():
    dest=OUT/'executed_source.json';hashes=sources()
    if dest.exists():assert json.loads(dest.read_text())['source_sha256']==hashes;return
    for name in hashes:
        target=OUT/'executed_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write(dest,{'source_sha256':hashes,'time_unix':time.time(),'learned_policy_qualified':False})


def make_case(seed,cost):
    full,_,g,obs=episode(seed,P['robots'],P['duration_seconds'],'drift' if seed%2 else 'main')
    obs=mask_scans(obs,5);sensor=DeliveredScans(obs.before(60),P['duration_seconds'],seed)
    profile=json.loads((ROOT/'artifacts/packet_value/latency.json').read_text())['forecast_seconds']
    live=LiveForecasts(SinglePredictor(g,'radiomap'),sensor,P['duration_seconds'],profile,.5)
    world=runtime_world(full,P['duration_seconds'],live.mean,live.std[:,:,0]);recorder=RadioHistory(g)
    kwargs={'roam_gate':reactive_rescue,'handover_admission':OutageBudgetGate(live.mean,live,cost),
            'forecast_queue':live,'forecast_source':live,'scan_observer':sensor,'scan_duration_s':.09,
            'observation_hook':recorder.observe,'observation_period_slots':10,'focal_hold_slots':100,
            'native_proposal_ttl_slots':100}
    service=[profile['cold' if d==0 else 'refit' if d%20==0 else 'warm']+.5 for d in range(P['duration_seconds'])]
    forks=ForecastQueue(service).ready[np.asarray(P['checkpoint_forecast_issues_seconds'])].tolist()
    return world,g,recorder,kwargs,forks


@torch.no_grad()
def prototype_probe(recorder,public):
    """Untrained real-input shape/invariance check, not an accuracy result."""
    torch.manual_seed(0);model=ServiceWorldModel();model.eval()
    window=recorder.window();ids=np.arange(min(8,len(public['current'])))
    b={k:torch.as_tensor(v[ids]) for k,v in window.items()}
    reference=np.zeros((len(ids),3,2),dtype='f4')
    for row,r in enumerate(ids):
        remaining=int(max(public['outage_remaining'][r],public['scan_remaining'][r]))
        for h,horizon in enumerate((100,500,1000)):
            end=min(remaining,horizon)
            # Unknown command phase: minimum guaranteed loss across phases.
            reference[row,h,0]=min(forced_deadline_misses([(0,end)],phase=p,
                                      start_slot=0,stop_slot=horizon) for p in range(5))
            reference[row,h,1]=end*.01
    action=torch.zeros((len(ids),3,12));action[:,:,3]=1.
    prior=torch.as_tensor(reference)
    p=model(**b,actions=action,ap_index=torch.as_tensor(public['current'][ids]),
            physics_mean=prior,action_elapsed=torch.tensor([[1.,4.,5.]]).expand(len(ids),-1))
    assert torch.equal(p['mean'],prior) and p['hazard'] is None
    assert torch.isfinite(p['latent']).all()
    return {'untrained':True,'accuracy_or_control_result':False,'clients':len(ids),
            'inference_parameters':model.inference_parameters,'zero_residual_exact':True,
            'unknown_hazard_is_none':True,'physical_reference':'Known-unavailability lower bounds only; not expected service'}


def label_case(task):
    seed,cost=task;torch.set_num_threads(1);started=time.time()
    key=f'{seed}-{cost:g}';receipt=OUT/'cases'/(key+'.json');data=OUT/'inputs'/(key+'.npz')
    hashes=sources()
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        assert previous['source_sha256']==hashes and previous['inputs_sha256']==digest(data)
        return previous
    world,g,recorder,kwargs,forks=make_case(seed,cost);p=policy_for(cost)
    p=replace(p,name='canonical_expiring_proposals')
    cps=[]
    # Forecast completions need not fall on whole seconds. Exact endpoint
    # checkpoints avoid silently rounding the ten-second comparison window.
    factual=simulate(world,p,**kwargs,checkpoint_at=set(forks)|{k+1000 for k in forks},
                     checkpoint_sink=cps.append,detailed=True)
    saved={cp.slot:cp for cp in cps};rows=[];features=[];native_runs=1;probe=None
    rng=np.random.default_rng(seed+round(cost*1000))
    for slot in forks:
        cp=saved[slot];public=cp.observable();prefix=np.asarray(public['misses'])
        baseline=simulate(world,p,resume=cp,stop_slot=slot+1000,detailed=True);native_runs+=1
        bp=baseline['packet_detail'];base=np.asarray(bp['misses'])
        np.testing.assert_array_equal(base,saved[slot+1000].observable()['misses'])
        # Availability and weakest *measured* pilot selection use only public
        # pre-intervention state. No future outcome selects the focal robot.
        available=np.flatnonzero((public['outage_remaining']==0)&(public['scan_remaining']==0))
        detectable=np.asarray(public['pilot_valid']) & (np.asarray(public['pilot_snr'])+g.noise+g.tx-15>=-90.)
        margin=np.where(detectable,public['pilot_snr'],-np.inf).max(-1)
        rankable=available[detectable[available].any(-1)]
        weak=rankable[np.argsort(margin[rankable],kind='stable')[:2]]
        rest=np.setdiff1d(available,weak);chosen=weak.tolist()
        if len(rest):chosen.append(int(rng.choice(rest)))
        history=cp.bundle['hooks'][7].__self__;window=history.window()
        if probe is None:probe=prototype_probe(history,public)
        for r in chosen:
            features.append({'slot':slot,'robot':r,**{k:v[r] for k,v in window.items()}})
            current=int(public['current'][r]);actions=[('stay',current)]
            heard=np.asarray(public['local_heard'][r]).any(-1)
            candidates=np.flatnonzero(heard & (np.arange(len(g.sites))!=current))
            if len(candidates):
                score=np.where(public['local_heard'][r],public['local_rssi'][r],-np.inf).max(-1)
                actions.append(('move',int(candidates[np.argmax(score[candidates])])))
            for kind,target in actions:
                result=simulate(world,p,resume=cp,force_action=(r,target),focal_hold_slots=100,
                                stop_slot=slot+1000,detailed=True);native_runs+=1
                detail=result['packet_detail'];misses=int(detail['misses'][r]-prefix[r])
                baseline_misses=int(base[r]-prefix[r]);gap=int(detail['window_max_gap'][r])
                baseline_gap=int(bp['window_max_gap'][r])
                rows.append({'slot':slot,'robot':r,'current':current,'target':target,'kind':kind,
                    'hold_slots':100,'canonical_recovery':True,'baseline_misses':baseline_misses,
                    'intervention_misses':misses,'delta_misses':misses-baseline_misses,
                    'baseline_window_max_gap_slots':baseline_gap,'intervention_window_max_gap_slots':gap,
                    'delta_window_max_gap_slots':gap-baseline_gap})
    arrays={k:np.asarray([row[k] for row in features]) for k in features[0]} if features else {}
    data.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(data,**arrays)
    # Same information and timing, classical pending-command-expiry ablation.
    legacy_world,_,_,legacy_kwargs,_=make_case(seed,cost)
    legacy_kwargs['native_proposal_ttl_slots']=None
    legacy=simulate(legacy_world,replace(p,name='canonical_legacy_unbounded'),**legacy_kwargs);native_runs+=1
    factual['packet_detail'].pop('observation_frames',None)
    factual['packet_detail'].pop('delivered_report_events',None)
    for row in factual['packet_detail']['cumulative']:
        row['misses']=int(sum(row['misses']));row['sent']=int(sum(row['sent']))
    receipt_row={'completed':True,'scene':seed,'cost':cost,'robots':P['robots'],
        'factual':factual,'legacy':legacy,'records':rows,'input_windows':len(features),
        'factual_replays':len(forks),'native_runs':native_runs,'prototype':probe,
        'forecast_timing':kwargs['forecast_source'].records,'source_sha256':hashes,
        'inputs_sha256':digest(data),'elapsed_seconds':time.time()-started,
        'simulation_only':True,'no_learned_policy_evaluated':True,'no_heldout_access':True}
    write(receipt,receipt_row)
    print(json.dumps({'native_case':key,'interventions':len(rows),'runs':native_runs,
                      'seconds':round(time.time()-started,1)}),flush=True)
    return receipt_row


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    snapshot();tasks=[(s,c) for s in P['diagnostic_sites'] for c in P['handover_seconds']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:rows=list(pool.map(label_case,tasks))
    write(OUT/'completion.json',{'completed':True,'cases':len(rows),
        'native_runs':sum(r['native_runs'] for r in rows),'factual_replays':sum(r['factual_replays'] for r in rows),
        'trained_models':0,'learned_policy_evaluations':0,'source_sha256':sources(),
        'original_holdouts_unused':True,'sota_established':False,'novelty_established':False,
        'time_unix':time.time()})
    print('SERVICE_NATIVE_DIAGNOSTIC_COMPLETE',flush=True)


if __name__=='__main__':main()
