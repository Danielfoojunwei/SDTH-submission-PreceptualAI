"""Collect immutable, full-window native counterfactual labels for warehouse v4."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import numpy as np
from wirelessjev.warehouse_v2.experiment import atomic_json, sha256
from wirelessjev.warehouse_native.data import (DURATION, BUDGET, MapPrior, public_observation,
    candidates, topology, features, endpoint_target)

REPO=Path(__file__).resolve().parents[1]
SPLITS={'train':list(range(190000,190008)), 'validation':list(range(191000,191003)),
        'test':list(range(192000,192006))}


def immutable_json(path, content):
    if path.exists():
        if json.loads(path.read_text())!=content:raise RuntimeError(f'Immutable protocol/data changed: {path}')
    else:atomic_json(path,content)


def protocol(root,binary):
    sources=[Path(__file__),REPO/'third_party/ns3/warehouse-contracts.cc',
             *sorted((REPO/'src/wirelessjev').rglob('*.py'))]
    # Model/training code is frozen separately before test. These files do not
    # affect the declared public inputs, target definitions or native execution.
    sources=[p for p in sources if p.relative_to(REPO).as_posix() not in
             ['src/wirelessjev/warehouse_native/model.py','src/wirelessjev/warehouse_native/learning.py']]
    scientific={str(p.relative_to(REPO)):sha256(p) for p in sources}
    receipt={'version':4,'source':'ns3_packet_events','duration_s':DURATION,
        'robots':100,'max_switches':BUDGET,'native_seed_splits':SPLITS,
        'binary_sha256':sha256(binary),'source_sha256':scientific,
        'rts_cts':True,'phy_profile':'vht80_ht20_2ss','assumed_application_mute_s':4.,
        'objective':'0.5 * mean demand service + 0.5 * mean deadline-met critical demand service',
        'critical_deadline_s':.05,'critical_packet_bytes':240,
        'target_window':'All thirty one-second bins; denominator includes nominal demand during muting',
        'candidate_policy_names':['stay','budgeted_map_signal','numerical','numerical_tail'],
        'observations':'Declared positions, APs, route, demand, current assignment; no scan or future telemetry',
        'native_model_limits':['Known log-distance map without racks/materials',
            'One fixed-association decision per thirty-second simulation, not persistent closed-loop roaming',
            'Four-second application mute is assumed, not actual Wi-Fi association/WPA/DHCP',
            'Map eligibility is not measured scan availability; no car action is generated',
            'Eight training, three validation, six held-out simulated environments; correlated robot labels',
            'RTS enabled for all arms, chosen using the completed earlier audit, not this holdout'],
        'hardware_experiments_executed':False}
    immutable_json(root/'protocol.json',receipt)
    for source in sources:
        target=root/'source_snapshot'/source.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=sha256(source):raise RuntimeError('Executed source capsule changed')
        if not target.exists():shutil.copyfile(source,target)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=['development','test'],default='development')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--binary',type=Path,default=Path('/workspace/upstream/warehouse-v2/ns3/build-contracts/scratch/ns3.49-warehouse-contracts-optimized'))
    args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    if not 1<=args.workers<=4:raise ValueError('One to four workers required')
    lock=(root/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=protocol(root,args.binary)
    status_path=root/f'collection-{args.split}.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state')=='completed':
        state=json.loads(status_path.read_text())
        if sha256(root/f'dataset-{args.split}.json')!=state['report_sha256']:raise RuntimeError('Completed dataset changed')
        print('Completed collection verified');return
    if args.split=='test':
        selection=json.loads((root/'selection.json').read_text())
        if not selection.get('frozen_before_test') or selection['protocol_sha256']!=sha256(root/'protocol.json'):
            raise RuntimeError('Freeze model selection before accessing the test outcomes')
        for name,digest in selection['checkpoints'].items():
            if sha256(root/'checkpoints'/name)!=digest:raise RuntimeError('Frozen checkpoint changed')
    state={'state':'running','phase':args.split,'pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(status_path,state)
    try:
        jobs=[];scenes=[]
        for split in (['test'] if args.split=='test' else ['train','validation']):
            for seed in SPLITS[split]:
                obs=public_observation(seed);prior=MapPrior(obs)
                plans,names,prediction,diagnostics=candidates(obs,prior)
                x,groups=features(obs,plans,prediction)
                (root/'public').mkdir(exist_ok=True)
                public=root/'public'/f'{seed}.json'
                obs_dict={key:val.tolist() for key,val in obs.items()}
                content={'scene':seed,'split':split,'observation':obs_dict,'plans':plans.tolist(),
                    'names':names,'prior_prediction':prediction.tolist(),'features':x.tolist(),
                    'channel_groups':groups.tolist(),'diagnostics':diagnostics}
                immutable_json(public,content)
                aliases=[];seen={}
                for j,plan in enumerate(plans):
                    if int((plan!=obs['current']).sum())>BUDGET:raise RuntimeError('Hard policy budget exceeded')
                    text=topology(obs,plan);key=plan.tobytes()
                    if key not in seen:
                        unique=len(seen);seen[key]=unique
                        path=root/'inputs'/f'{seed}-{unique}.txt';path.parent.mkdir(exist_ok=True)
                        if path.exists() and path.read_text()!=text:raise RuntimeError('Native input changed')
                        path.write_text(text)
                        jobs.append((seed,unique,path,obs['critical']))
                    aliases.append(seen[key])
                scenes.append({'scene':seed,'split':split,'public':str(public.relative_to(root)),
                    'public_sha256':sha256(public),'aliases':aliases})
        atomic_json(root/f'collection-plan-{args.split}.json',{'scenes':scenes,'native_runs':len(jobs),
            'planned_at':datetime.now(timezone.utc).isoformat(),'test_selection_sha256':sha256(root/'selection.json') if args.split=='test' else None})
        print('Native thirty-second jobs:',len(jobs),'phase',args.split,flush=True)
        def execute(job):
            seed,index,path,critical=job
            output=root/'native'/f'{seed}-{index}.json';output.parent.mkdir(exist_ok=True)
            marker=output.with_suffix('.receipt.json')
            key={'binary_sha256':receipt['binary_sha256'],'input_sha256':sha256(path),'seed':seed,
                 'run':1,'duration_s':DURATION,'rts_cts':True,'protocol_sha256':sha256(root/'protocol.json')}
            if output.exists() and marker.exists():
                saved=json.loads(marker.read_text())
                if any(saved[k]!=v for k,v in key.items()) or saved['report_sha256']!=sha256(output):
                    raise RuntimeError('Native cached report changed')
            else:
                started=time.perf_counter()
                result=subprocess.run([str(args.binary),f'--input={path.resolve()}',f'--output={output.resolve()}',
                    f'--seed={seed}','--run=1',f'--duration={DURATION}','--rts=1'],capture_output=True,text=True,timeout=2400)
                output.with_suffix('.log').write_text(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError(f'Native execution failed: {output.with_suffix(".log")}')
                saved={**key,'report_sha256':sha256(output),'wall_s':time.perf_counter()-started}
                atomic_json(marker,saved)
            report=json.loads(output.read_text())
            if report['phy_profile']!=receipt['phy_profile'] or report['seed']!=seed or report['robots']!=100:
                raise RuntimeError('Native configuration mismatch')
            target=endpoint_target(report,critical)
            return {'scene':seed,'unique_plan':index,'path':str(output.relative_to(root)),
                    'receipt_sha256':sha256(marker),'report_sha256':sha256(output),'wall_s':saved['wall_s'],
                    'target':target.tolist()}
        results=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(execute,job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result());print('Native completed',len(results),'/',len(jobs),flush=True)
                atomic_json(root/f'progress-{args.split}.json',{'completed':len(results),'total':len(jobs)})
        atomic_json(root/f'dataset-{args.split}.json',{'source':'ns3_packet_events','scenes':scenes,'outcomes':results,
            'protocol_sha256':sha256(root/'protocol.json')})
        state.update(state='completed',completed_at=datetime.now(timezone.utc).isoformat(),
            report_sha256=sha256(root/f'dataset-{args.split}.json'),native_runs=len(results))
    except BaseException as error:
        state.update(state='failed',error=str(error),traceback=traceback.format_exc());raise
    finally:atomic_json(status_path,state)


if __name__=='__main__':main()
