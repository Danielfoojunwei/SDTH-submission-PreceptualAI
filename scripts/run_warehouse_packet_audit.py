"""Execute paired native packet experiments with matching public plans/routes."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
import torch

from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_horizon.world import World,observe
from wirelessjev.warehouse_horizon.belief import RadioBelief
from wirelessjev.warehouse_horizon.model import prior_prediction
from wirelessjev.warehouse_horizon.planning import Contract,candidate_pool,charge,values,budgeted_signal
from wirelessjev.warehouse_v3.model import RadioModel
from wirelessjev.warehouse_v3.learning import predict
from wirelessjev.hardware import RASPBOT

REPO=Path(__file__).resolve().parents[1]


def observation(world):
    current=world.initial.copy();history=[];previous=np.zeros(len(current))
    for tick in range(4):
        obs=observe(world.frames[tick],current,tick,previous_service=previous,since_switch=np.full(len(current),30.),history=history)
        previous,_,_=world.interval(tick,current,current)
    return obs


def topology(obs,action):
    lines=[f'WIRELESSJEV_PACKET_V2 {len(action)} {len(obs["ap"])} {RASPBOT.client_tx_power_dbm}']
    for a in obs['ap']:lines.append(f'{a[0]*120:.9f} {a[1]*80:.9f} {round(a[3]*165)} {int(a[2])} {a[4]:.9f} {a[5]*30:.9f}')
    for i in range(len(action)):
        route=np.vstack([obs['robot'][i,:2],obs['route'][:,i]])*np.array([120.,80.])
        positions=' '.join(f'{x:.9f} {y:.9f}' for x,y in route)
        lines.append(f'{obs["critical"][i]+obs["bulk"][i]:.9f} {obs["critical"][i]:.9f} {action[i]} {4 if action[i]!=obs["current"][i] else 0} {positions}')
    return '\n'.join(lines)+'\n'


def analyse(report):
    n=report['robots'];duration=report['duration_s'];delivered=np.zeros(n)
    for frame in report['telemetry']:
        for row in frame['robots']:delivered[row['robot']]+=row['delivered_bytes']
    per=[]
    for i,row in enumerate(report['per_robot']):
        service=min(1.,delivered[i]*8/(duration*row['offered_mbps']*1e6))
        # Counts include endpoint arrivals during the explicit drain period.
        # Muted application intervals have no generated packets, so their
        # missing demand is charged through service, not invented packets.
        misses=row['critical_tx']-row['critical_rx']+row['critical_late']
        if misses<0:raise RuntimeError('Endpoint counter mismatch')
        per.append({'robot':i,'service':service,'critical_generated':row['critical_tx'],
            'critical_deadline_misses':misses,'longest_critical_silence_s':row['longest_critical_silence_s'],
            'first_critical_packet_s':row['first_critical_packet_s'],'retries':row['phy_retry_attempts'],
            'p99_delay_s':row['p99_delay_s']})
    service=np.array([r['service'] for r in per])
    return {'mean_service':float(service.mean()),'worst_decile_service':float(np.sort(service)[:max(1,int(np.ceil(n*.1)))].mean()),
        'critical_deadline_miss_fraction_generated':sum(r['critical_deadline_misses'] for r in per)/max(1,sum(r['critical_generated'] for r in per)),
        'maximum_critical_silence_s':max(r['longest_critical_silence_s'] for r in per),
        'retry_attempts':sum(r['retries'] for r in per),'per_robot':per}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--v3',type=Path,default=Path('/workspace/training/warehouse-v3'))
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--binary',type=Path,default=Path('/workspace/upstream/warehouse-v2/ns3/build-contracts/scratch/ns3.49-warehouse-contracts-optimized'))
    args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    if not 1<=args.workers<=4:raise ValueError('Use one to four native simulation workers')
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=root/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state')=='completed':
        state=json.loads(status_path.read_text())
        if sha256(root/'packet_audit.json')!=state['report_sha256']:raise RuntimeError('Completed packet audit changed')
        print('Completed packet audit verified');return
    torch.set_num_threads(1)
    if not args.binary.exists():raise RuntimeError('Build the real native simulator first')
    hashes={'third_party/ns3/warehouse-contracts.cc':sha256(REPO/'third_party/ns3/warehouse-contracts.cc'),
        'scripts/run_warehouse_packet_audit.py':sha256(Path(__file__))}
    status={'state':'running','pid':os.getpid(),'stage':'native_packet_experiments','started_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(status_path,status)
    receipt={'binary_sha256':sha256(args.binary),'source_sha256':hashes,'version':'ns3-3.49',
        'native_workers':args.workers,
        'phy_profile':'vht80_ht20_2ss','guard_interval_ns':800,
        'duration_s':8,'seeds':list(range(146000,146006)),'rts_cts':[False,True],
        'training_scene_seeds':[146000,146001],'validation_scene_seeds':[146002],
        'test_scene_seeds':[146003,146004,146005],
        'source':'ns3_packet_events','hardware_experiments_executed':False,
        'learned_policy':'predeclared graph_jepa seed 0; research arm, not qualified',
        'protocol_limits':['Eight-second prefix of the committed 30-second route',
            'Geometry, routes, demands, channels and action assignments match; log-distance propagation lacks racks',
            'Assumed application muting is not Wi-Fi authentication or DHCP emulation',
            'Deadline labels concern generated packets; absent muted demand is charged in service',
            'External busy metadata parameterises an independent burst source, not calibrated hardware airtime']}
    atomic_json(root/'provenance.json',receipt)
    for relative in hashes:
        dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,dest)
    try:
        jobs=[];plans_receipts=[]
        for seed in receipt['seeds']:
            obs=observation(World(seed,steps=12));belief=RadioBelief(obs)
            candidates,_=candidate_pool(obs,belief,Contract('throughput',12,1.));prior=belief.forecast(candidates)
            raw=prior_prediction(prior,obs);service,risk=charge(raw,candidates,obs)
            chosen=int(values(Contract('throughput',12,1.),obs,service,risk).argmax())
            arms={'stay':obs['current'],'budgeted_rssi':budgeted_signal(obs,12),'numerical':candidates[chosen]}
            checkpoint=args.v3/'checkpoints/graph_jepa-0.pt'
            while not checkpoint.exists():
                state=json.loads((args.v3/'status.json').read_text())
                if state['state']=='failed':raise RuntimeError('Required predeclared model failed')
                time.sleep(5)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model=RadioModel(**saved['config'])
            model.load_state_dict(saved['model']);model.eval()
            predicted=predict(model,{'observation':obs,'plans':candidates,'prior':prior})
            learned_service,learned_risk=charge(predicted,candidates,obs)
            index=int(values(Contract('throughput',12,1.),obs,learned_service,learned_risk).argmax())
            arms['radio_jepa_seed0']=candidates[index]
            for name,action in arms.items():
                path=root/'inputs'/f'{seed}-{name}.txt';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(topology(obs,action))
                payload={'scene':seed,'policy':name,'topology_sha256':sha256(path),'switches':int((action!=obs['current']).sum()),
                    'selected_action':action.tolist(),'forecast_mean_service_first_8s':float((service[np.flatnonzero((candidates==action).all(1))[0],0]*5+service[np.flatnonzero((candidates==action).all(1))[0],1]*3).mean()/8) if ((candidates==action).all(1)).any() else None,
                    'checkpoint_sha256':sha256(checkpoint) if name=='radio_jepa_seed0' else None}
                plans_receipts.append(payload)
                for rts in [False,True]:jobs.append((seed,name,rts,path))
        atomic_json(root/'plans.json',plans_receipts)
        def execute(job):
            seed,policy,rts,input_path=job;path=root/'native'/f'{seed}-{policy}-{rts}.json';path.parent.mkdir(exist_ok=True)
            marker=path.with_suffix('.receipt.json');key={'binary_sha256':sha256(args.binary),'input_sha256':sha256(input_path),'rts':rts,'seed':seed}
            if path.exists() and marker.exists():
                old=json.loads(marker.read_text())
                if any(old[k]!=v for k,v in key.items()) or sha256(path)!=old['report_sha256']:raise RuntimeError('Cached native run mismatch')
                elapsed=old['wall_s']
            else:
                started=time.perf_counter()
                result=subprocess.run([str(args.binary),f'--input={input_path.resolve()}',f'--output={path.resolve()}',f'--seed={seed}',
                    '--run=1','--duration=8',f'--rts={int(rts)}'],capture_output=True,text=True,timeout=900)
                (path.with_suffix('.log')).write_text(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError(f'Native packet simulator failed: {path.with_suffix(".log")}')
                elapsed=time.perf_counter()-started
                atomic_json(marker,{**key,'report_sha256':sha256(path),'wall_s':elapsed})
            report=json.loads(path.read_text())
            if report['phy_profile']!='vht80_ht20_2ss':raise RuntimeError('Packet run used the wrong PHY profile')
            return {'scene':seed,'policy':policy,'rts_cts':rts,'native_report':str(path.relative_to(root)),
                'native_sha256':sha256(path),'wall_s':elapsed,'metrics':analyse(report)}
        records=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(execute,job) for job in jobs]
            for future in as_completed(futures):
                records.append(future.result());print('Native packet run completed',len(records),'/',len(jobs),flush=True)
                atomic_json(root/'progress.json',{'completed':len(records),'total':len(jobs),'records':records})
        atomic_json(root/'packet_audit.json',{'source':'ns3_packet_events','records':records,'plans':plans_receipts,
            'provenance_sha256':sha256(root/'provenance.json'),'hardware_qualified':False})
    except BaseException as error:
        status.update(state='failed',error=str(error),traceback=traceback.format_exc());atomic_json(status_path,status);raise
    status.update(state='completed',stage=None,completed_at=datetime.now(timezone.utc).isoformat(),
        report_sha256=sha256(root/'packet_audit.json'));atomic_json(status_path,status)


if __name__=='__main__':main()
