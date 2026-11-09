"""Train six native-packet models, freeze selection, then collect fresh tests."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v2.metrics import cluster_interval
from wirelessjev.warehouse_native.learning import load_dataset,train,ridge_fit,assess,gate,aggregate,crossed_family_interval

REPO=Path(__file__).resolve().parents[1]


def freeze_sources(root):
    paths=[Path(__file__),*sorted((REPO/'src/wirelessjev/warehouse_native').glob('*.py'))]
    hashes={str(p.relative_to(REPO)):sha256(p) for p in paths}
    target=root/'training-protocol.json'
    protocol={'source_sha256':hashes,'model_seeds':[0,1,2],'arms':['mlp','graph'],'epochs':120,
        'target':'Native endpoint service and critical demand met within 50 ms, over thirty seconds',
        'loss':'Per-robot fraction MSE + 0.1 floor-event BCE + 5 candidate-advantage MSE',
        'selection':'Minimum validation loss per checkpoint; temperature fitted on validation only',
        'numerical_baseline':'Known native log-distance map, NIST VHT80/HT20 two-stream curves, matched noise figure and mute window',
        'specialist_baseline':'Training-fitted linear residual; ridge penalty chosen on validation',
        'default_selection':'Best validation utility among candidates passing all gates, else numerical',
        'gate':'Positive scene-bootstrap lower utility bound and no validation scene loses tail or critical demand service',
        'test_accessed_before_selection':False,'hardware_qualified':False}
    if target.exists() and json.loads(target.read_text())!=protocol:raise RuntimeError('Frozen training source or protocol changed')
    if not target.exists():atomic_json(target,protocol)
    for p in paths:
        dest=root/'training_source_snapshot'/p.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(p,dest)
        if sha256(dest)!=sha256(p):raise RuntimeError('Training source capsule changed')


def wait_development(root):
    while True:
        path=root/'collection-development.json'
        if not path.exists():raise RuntimeError('Start native development collection first')
        state=json.loads(path.read_text())
        if state['state']=='completed':return
        if state['state']=='failed':raise RuntimeError('Native development collection failed: '+state.get('error',''))
        pid=state['pid']
        try:
            os.kill(pid,0)
            stat=Path(f'/proc/{pid}/stat').read_text()
            if stat.split(') ',1)[1].startswith('Z'):raise ProcessLookupError()
        except (ProcessLookupError,FileNotFoundError):raise RuntimeError('Native collector exited without completion receipt')
        time.sleep(15)


def frozen_selection(root):
    selection=json.loads((root/'selection.json').read_text())
    if not selection['frozen_before_test'] or selection['test_used_for_model_selection']:
        raise RuntimeError('Invalid frozen selection')
    for file,key in [('protocol.json','protocol_sha256'),('training-protocol.json','training_protocol_sha256'),
                     ('dataset-development.json','development_sha256'),('validation.json','validation_sha256'),
                     ('ridge.json','ridge_sha256')]:
        if sha256(root/file)!=selection[key]:raise RuntimeError('Frozen selection dependency changed: '+file)
    for name,digest in selection['checkpoints'].items():
        if sha256(root/'checkpoints'/name)!=digest:raise RuntimeError('Frozen checkpoint changed')
    return selection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args();root=args.out
    lock=(root/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    freeze_sources(root)
    status_path=root/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state')=='completed':
        status=json.loads(status_path.read_text())
        if sha256(root/'results.json')!=status['report_sha256']:raise RuntimeError('Completed results changed')
        print('Completed v4 campaign verified');return
    status={'state':'running','stage':'collecting_native_development','pid':os.getpid(),
            'started_at':datetime.now(timezone.utc).isoformat()};atomic_json(status_path,status)
    try:
        wait_development(root);development=load_dataset(root,'development')
        status.update(stage='training_on_native_packets');atomic_json(status_path,status)
        names=[]
        for graph in [False,True]:
            for seed in [0,1,2]:
                name=f'{"graph" if graph else "mlp"}-{seed}'
                checkpoint=root/'checkpoints'/f'{name}.pt';meta=root/f'train-{name}.json'
                if checkpoint.exists() and meta.exists():
                    if sha256(checkpoint)!=json.loads(meta.read_text())['checkpoint_sha256']:raise RuntimeError('Cached checkpoint changed')
                else:train(development,root,seed=seed,graph=graph)
                names.append(name);print('Native model trained',name,flush=True)
        if (root/'selection.json').exists():
            selection=frozen_selection(root)
            # Validation inference timings are operational measurements. Reuse
            # the sealed report instead of changing its hash during recovery.
            ridge=json.loads((root/'ridge.json').read_text())
            selected=selection['research_default']
        else:
            ridge=ridge_fit(development);atomic_json(root/'ridge.json',ridge)
            validation=[s for s in development if s['split']=='validation']
            records=assess(validation,root,['numerical','ridge']+names,ridge)
            atomic_json(root/'validation.json',{'records':records,'summary':aggregate(records)})
            gates={name:gate(records,name) for name in ['ridge']+names}
            summary=aggregate(records);accepted=[name for name,g in gates.items() if g['accepted']]
            selected=max(accepted,key=lambda n:summary[n]['utility']) if accepted else 'numerical'
            selection={'frozen_before_test':True,'frozen_at':datetime.now(timezone.utc).isoformat(),
                'protocol_sha256':sha256(root/'protocol.json'),'training_protocol_sha256':sha256(root/'training-protocol.json'),
                'development_sha256':sha256(root/'dataset-development.json'),'validation_sha256':sha256(root/'validation.json'),
                'ridge_sha256':sha256(root/'ridge.json'),'gates':gates,'research_default':selected,
                'checkpoints':{f'{name}.pt':sha256(root/'checkpoints'/f'{name}.pt') for name in names},
                'hardware_qualified':False,'test_used_for_model_selection':False}
            atomic_json(root/'selection.json',selection)
        status.update(stage='collecting_fresh_native_test');atomic_json(status_path,status)
        subprocess.run([sys.executable,str(REPO/'scripts/collect_warehouse_native.py'),'--out',str(root),
            '--split','test','--workers',str(args.workers)],cwd=REPO,check=True)
        tests=load_dataset(root,'test');status.update(stage='evaluating_frozen_models');atomic_json(status_path,status)
        records=assess(tests,root,['numerical','ridge']+names,ridge)
        comparisons={}
        numeric={r['scene']:r for r in records if r['method']=='numerical'}
        for name in ['ridge']+names:
            rows=[r for r in records if r['method']==name]
            delta=[r['utility']-numeric[r['scene']]['utility'] for r in rows]
            comparisons[name]=cluster_interval(np.array(delta),np.array([r['scene'] for r in rows]))
        results={'version':4,'source':'actual_ns3_packet_events_and_native_supervised_training',
            'selection_sha256':sha256(root/'selection.json'),'test_dataset_sha256':sha256(root/'dataset-test.json'),
            'test_records':records,'summary':aggregate(records),'paired_utility_difference_vs_numerical':comparisons,
            'crossed_family_utility_difference_vs_numerical':{arm:crossed_family_interval(records,arm) for arm in ['mlp','graph']},
            'test_gates_diagnostic_only':{name:gate(records,name) for name in ['ridge']+names},
            'research_default_selected_on_validation':selected,'hardware_qualified':False,
            'sota_established':False,'persistent_closed_loop_executed':False,
            'limits':json.loads((root/'protocol.json').read_text())['native_model_limits']}
        atomic_json(root/'results.json',results)
        status.update(state='completed',stage=None,completed_at=datetime.now(timezone.utc).isoformat(),report_sha256=sha256(root/'results.json'))
    except BaseException as error:
        status.update(state='failed',error=str(error),traceback=traceback.format_exc());raise
    finally:atomic_json(status_path,status)


if __name__=='__main__':main()
