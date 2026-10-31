"""Snapshot live status; archive completed native evidence after verification."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
import torch
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_native.data import endpoint_target


def read(path):return json.loads(path.read_text())


def finite(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():raise RuntimeError('Nonfinite checkpoint')
    if isinstance(value,dict):
        for v in value.values():finite(v)
    if isinstance(value,(list,tuple)):
        for v in value:finite(v)
    if isinstance(value,float) and not np.isfinite(value):raise RuntimeError('Nonfinite report')


def live_snapshot(root,out):
    state=read(root/'status.json');protocol=read(root/'protocol.json')
    snapshot={'version':4,'source':'ns3_packet_events','snapshot_at':datetime.now(timezone.utc).isoformat(),
        'state':state['state'],'stage':state.get('stage'),'robots':100,'duration_s':30,
        'planned_model_checkpoints':6,'model_checkpoints_finished':len(list(root.glob('train-*.json'))),
        'training_seeds':3,'training_scenes':8,'validation_scenes':3,'test_scenes':6,
        'hardware_qualified':False,'sota_established':False,'persistent_closed_loop_executed':False,
        'limits':protocol['native_model_limits'],'native_progress':{}}
    for phase in ['development','test']:
        path=root/f'progress-{phase}.json'
        if path.exists():snapshot['native_progress'][phase]=read(path)
        else:snapshot['native_progress'][phase]={'completed':0,'total':None}
    if state['state']=='failed':snapshot['error']=state.get('error')
    if state['state']=='running':
        try:
            command=Path(f'/proc/{state["pid"]}/cmdline').read_bytes()
            alive=b'run_warehouse_native.py' in command and str(root.resolve()).encode() in command
        except (FileNotFoundError,ProcessLookupError):alive=False
        snapshot['supervisor_alive']=alive
        if not alive:snapshot['state']='interrupted'
    if state['state']=='completed':
        if sha256(root/'results.json')!=state['report_sha256']:raise RuntimeError('Completion hash mismatch')
        result=read(root/'results.json')
        snapshot.update(summary=result['summary'],paired_differences=result['paired_utility_difference_vs_numerical'],
            crossed_family_differences=result['crossed_family_utility_difference_vs_numerical'],
            research_default=result['research_default_selected_on_validation'])
    atomic_json(out/'snapshot.json',snapshot)
    return snapshot


def capsule(source,dest,hashes):
    for rel,digest in hashes.items():
        if sha256(source/rel)!=digest:raise RuntimeError('Executed source capsule mismatch')
    with dest.open('wb') as output,gzip.GzipFile(fileobj=output,filename='',mode='wb',mtime=0) as stream:
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file():
                    info=archive.gettarinfo(str(path),arcname=str(path.relative_to(source)))
                    info.mtime=info.uid=info.gid=0;info.uname=info.gname=''
                    with path.open('rb') as data:archive.addfile(info,data)
    return sha256(dest)


def archive(root,out):
    state=read(root/'status.json')
    if state['state']!='completed':raise RuntimeError('Only complete results can be archived as complete')
    if sha256(root/'results.json')!=state['report_sha256']:raise RuntimeError('Result checksum changed')
    selection=read(root/'selection.json')
    if not selection['frozen_before_test'] or selection['test_used_for_model_selection']:raise RuntimeError('Test leakage flag')
    for phase in ['development','test']:
        report=root/f'dataset-{phase}.json';receipt=read(root/f'collection-{phase}.json')
        if receipt['state']!='completed' or sha256(report)!=receipt['report_sha256']:raise RuntimeError('Incomplete native phase')
        dataset=read(report);public={r['scene']:read(root/r['public']) for r in dataset['scenes']}
        for row in dataset['outcomes']:
            path=root/row['path']
            if sha256(path)!=row['report_sha256']:raise RuntimeError('Native output changed')
            target=endpoint_target(read(path),public[row['scene']]['observation']['critical'])
            if not np.array_equal(target,np.asarray(row['target'],dtype='float32')):raise RuntimeError('Targets disagree with endpoint events')
    manifest={'source':'actual_ns3_packet_events_and_native_supervised_training','hardware_qualified':False,
        'reports':{},'checkpoints':{},'source_capsules':{}}
    names=['protocol.json','training-protocol.json','status.json','collection-development.json','collection-test.json',
        'collection-plan-development.json','collection-plan-test.json','selection.json','ridge.json','validation.json','results.json',
        'dataset-development.json','dataset-test.json','runtime.json']
    names += [str(p.relative_to(root)) for p in sorted(root.glob('train-*.json'))]
    names += [str(p.relative_to(root)) for folder in ['native','public','inputs'] for p in sorted((root/folder).glob('*'))
              if p.suffix in ['.json','.txt']]
    for relative in names:
        path=root/relative
        if path.suffix=='.json':finite(read(path))
        compressed=path.stat().st_size>50_000
        dest=out/(relative+'.gz' if compressed else relative);dest.parent.mkdir(parents=True,exist_ok=True)
        if compressed:
            with dest.open('wb') as output,gzip.GzipFile(fileobj=output,filename='',mode='wb',mtime=0) as stream:stream.write(path.read_bytes())
        else:shutil.copyfile(path,dest)
        manifest['reports'][str(dest.relative_to(out))]={'sha256':sha256(dest),'uncompressed_sha256':sha256(path)}
    for name,digest in selection['checkpoints'].items():
        path=root/'checkpoints'/name
        if sha256(path)!=digest:raise RuntimeError('Frozen checkpoint changed')
        finite(torch.load(path,map_location='cpu',weights_only=True));dest=out/'checkpoints'/name
        dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest);manifest['checkpoints'][name]=digest
    for folder,protocol in [('source_snapshot','protocol.json'),('training_source_snapshot','training-protocol.json')]:
        dest=out/f'{folder}.tar.gz'
        manifest['source_capsules'][dest.name]=capsule(root/folder,dest,read(root/protocol)['source_sha256'])
    for attempt in sorted((root/'attempts').glob('*')):
        relative=Path('attempts')/attempt.name;dest=out/relative;dest.mkdir(parents=True,exist_ok=True)
        for name in ['superseded.json','training-protocol.json','status.json']:
            path=attempt/name
            if path.exists():
                shutil.copyfile(path,dest/name)
                manifest['reports'][str(relative/name)]={'sha256':sha256(dest/name),'uncompressed_sha256':sha256(path)}
        if (attempt/'training_source_snapshot').exists():
            target=dest/'executed_sources.tar.gz'
            manifest['source_capsules'][str(relative/target.name)]=capsule(attempt/'training_source_snapshot',target,
                read(attempt/'training-protocol.json')['source_sha256'])
    live_snapshot(root,out);manifest['snapshot_sha256']=sha256(out/'snapshot.json')
    atomic_json(out/'manifest.json',manifest)
    result=read(root/'results.json')
    lines=['# Warehouse native packet results (v4)','',
        'Actual ns-3 packet simulations with 100 robots, a full thirty-second decision window, eight training scenes, three validation scenes and six new test scenes. Three independent training seeds per neural architecture. No physical car or persistent roaming loop was tested.','',
        '| Method | Utility | Mean service | Timely critical demand | Worst-decile service | Candidate regret | Forecast RMSE |',
        '|---|---|---|---|---|---|---|']
    for name,row in result['summary'].items():
        values=[f'{row[k]:.4f}' if k in row else '—' for k in ['utility','mean_service','critical_ontime_demand_fraction','worst_decile_service','candidate_regret','forecast_rmse']]
        lines.append('| '+' | '.join([name,*values])+' |')
    lines += ['',f'Validation-selected research default: `{result["research_default_selected_on_validation"]}`. Raw research-model scores do not override failed gates. The candidate oracle is a retrospective upper bound within four proposals, not a global optimum.','',
        'Utility is half mean service plus half critical demand delivered within fifty milliseconds. Denominators include nominal demand during the assumed four-second application mute. These metrics differ from historical E1/E2 regret and v3 generated-packet miss fractions.','',
        'The map, PHY configuration and future route are known simulation declarations. There are no RF scan inputs. Propagation lacks racks/material calibration; association, WPA and DHCP are not simulated as actual handovers. Six scenes and one packet realisation per candidate give exploratory evidence. See `WAREHOUSE_NATIVE.md` and the archived raw reports, checkpoints and source capsules.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('artifacts/warehouse_native'));p.add_argument('--follow',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    while True:
        monitor=args.root/'archive-monitor' if args.follow else args.out
        monitor.mkdir(parents=True,exist_ok=True)
        snapshot=live_snapshot(args.root,monitor)
        if snapshot['state']=='completed':
            manifest=archive(args.root,args.out);print('Verified native evidence archived:',len(manifest['reports']),'reports',flush=True);return
        if not args.follow:return
        if snapshot['state'] in ['failed','interrupted']:raise RuntimeError('Campaign needs recovery: '+snapshot['state'])
        time.sleep(30)


if __name__=='__main__':main()
