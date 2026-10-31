"""Validation-only calibration and scene-cluster audit of actual late events."""
from pathlib import Path
import argparse
from collections import defaultdict
import importlib.util
import json
import time
import numpy as np
import torch
from scipy.special import expit
from wirelessjev.measurement import packet_features
from wirelessjev.warehouse_horizon.learning import temperature
from wirelessjev.warehouse_v2.metrics import risk_metrics
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=Path('/workspace/training/warehouse-v3-packet-learning-unconditional'))
    p.add_argument('--packets',type=Path,default=Path('/workspace/training/warehouse-v3-packets-matched'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    while not (args.models/'status.json').exists() or json.loads((args.models/'status.json').read_text())['state']!='completed':
        if (args.models/'status.json').exists() and json.loads((args.models/'status.json').read_text())['state']=='failed':raise RuntimeError('Packet learning failed')
        time.sleep(5)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    repo=Path(__file__).resolve().parents[1];path=repo/'scripts/run_warehouse_v3_studies.py'
    spec=importlib.util.spec_from_file_location('trained_packet_architecture',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=json.loads((args.models/'packet_prediction.json').read_text())
    native=json.loads((args.packets/'packet_audit.json').read_text());splits=defaultdict(list)
    for run in native['records']:
        path=args.packets/run['native_report']
        if sha256(path)!=run['native_sha256']:raise RuntimeError('Native event report changed')
        report=json.loads(path.read_text());static=report['per_robot'];frames=report['telemetry'];n=report['robots']
        counts=np.bincount([r['ap'] for r in static],minlength=128)
        split='train' if run['scene']<146002 else ('validation' if run['scene']==146002 else 'test')
        for tick in range(3,len(frames)-1):
            for robot in range(n):
                following=frames[tick+1]['robots'][robot];now=frames[tick]['robots'][robot]
                features=[packet_features(frames[t]['robots'][robot])+[np.log1p(static[robot]['offered_mbps']),counts[static[robot]['ap']]/100.] for t in range(tick-3,tick+1)]
                splits[split].append((features,float(following['critical_late']>0),float(now['critical_late']>0),run['scene']))
    if {k:len(v) for k,v in splits.items()}!=original['split_samples']:raise RuntimeError('Event dataset differs from the trained study')
    arrays={k:(torch.tensor(np.array([r[0] for r in rows],dtype='float32')),np.array([r[1] for r in rows]),np.array([r[3] for r in rows])) for k,rows in splits.items()}
    models={};calibration={}
    for row in original['records']:
        key=f'{row["seed"]}-{row["temporal"]}';path=args.models/'checkpoints'/f'packet-{key}.pt'
        if sha256(path)!=row['checkpoint_sha256']:raise RuntimeError('Packet model changed')
        saved=torch.load(path,map_location='cpu',weights_only=True);model=module.PacketModel(saved['temporal']);model.load_state_dict(saved['model']);model.eval();models[key]=model
        with torch.inference_mode():logits=model(arrays['validation'][0])[:,1].numpy()
        calibration[key]=temperature(logits,arrays['validation'][1])
    atomic_json(root/'calibration.json',{'temperature':calibration,'fit_on':'validation_scene_146002_only',
        'test_labels_used_for_fit':False,'designed_after_initial_packet_results':True,
        'target_event':'at least one late received critical packet in the next one-second bin; no arrivals means false',
        'physical_calibration_guarantee':False})
    x,y,scenes=arrays['test'];persistence=np.array([r[2] for r in splits['test']]);records=[];matrix={}
    for key,model in models.items():
        with torch.inference_mode():logits=model(x)[:,1].numpy()
        raw=expit(logits);calibrated=expit(logits/calibration[key]);per_scene=[]
        source=next(r for r in original['records'] if key==f'{r["seed"]}-{r["temporal"]}')
        if abs(risk_metrics(raw,y)['brier']-source['late_arrival_risk']['brier'])>1e-7:raise RuntimeError('Frozen raw probabilities did not reproduce')
        for scene in sorted(set(scenes)):
            mask=scenes==scene
            per_scene.append({'scene':int(scene),'raw':risk_metrics(raw[mask],y[mask]),'calibrated':risk_metrics(calibrated[mask],y[mask]),'persistence':risk_metrics(persistence[mask],y[mask])})
        records.append({'model':key,'raw':risk_metrics(raw,y),'calibrated':risk_metrics(calibrated,y),'per_scene':per_scene})
        matrix[key]=[r['calibrated']['brier']-r['persistence']['brier'] for r in per_scene]
    intervals={};rng=np.random.default_rng(0)
    for temporal in [False,True]:
        differences=np.array([matrix[f'{seed}-{temporal}'] for seed in [0,1,2]])
        draws=[differences[np.ix_(rng.integers(3,size=3),rng.integers(3,size=3))].mean() for _ in range(4000)]
        intervals[str(temporal)]={'mean_scene_brier_difference_vs_persistence':float(differences.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'method':'crossed_model_seed_and_environment_percentile_bootstrap'}
    atomic_json(root/'packet_calibration.json',{'source':'actual_native_packet_events_and_frozen_model_inference',
        'parent_model_report_sha256':sha256(args.models/'packet_prediction.json'),'calibration_sha256':sha256(root/'calibration.json'),
        'records':records,'paired_intervals':intervals,'test_scenes':[146003,146004,146005],
        'limits':['Post-run audit reuses the original packet holdout; no new independent replication',
            'Temperature fitted only on one validation scene; physical transfer remains unqualified',
            'A no-arrival bin is not a late-arrival event and does not imply a healthy link',
            'This is not packet-loss, switch-success or critical-rate-violation probability']})
    atomic_json(root/'status.json',{'state':'completed','report_sha256':sha256(root/'packet_calibration.json')})


if __name__=='__main__':main()
