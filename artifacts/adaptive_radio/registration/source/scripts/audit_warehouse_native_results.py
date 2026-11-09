"""Audit completed portable native evidence without changing frozen results."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from wirelessjev.warehouse_native.data import endpoint_target
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256


def read(root,relative):
    path=root/relative
    if not path.exists():path=path.with_name(path.name+'.gz')
    data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    return json.loads(data)


def event_counts(samples):
    result={}
    for split,labels in samples.items():
        labels=np.asarray(labels,dtype=bool)
        result[split]={'miss':int(labels.sum()),'met':int((~labels).sum()),'n':len(labels),
            'discrimination_testable':bool(labels.any() and (~labels).any()),
            'always_miss_brier':float((~labels).mean()) if len(labels) else None}
    return result


def audit(root):
    manifest=read(root,'manifest.json')
    for name,entry in manifest['reports'].items():
        path=root/name
        if sha256(path)!=entry['sha256']:raise RuntimeError('Archived report changed: '+name)
        data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['uncompressed_sha256']:raise RuntimeError('Raw report changed: '+name)
    for name,digest in manifest['checkpoints'].items():
        path=root/'checkpoints'/name
        if sha256(path)!=digest:raise RuntimeError('Checkpoint changed: '+name)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if any(not torch.isfinite(t).all() for t in saved['model'].values()):raise RuntimeError('Nonfinite checkpoint')
    for name,digest in manifest['source_capsules'].items():
        if sha256(root/name)!=digest:raise RuntimeError('Executed source changed: '+name)
    if sha256(root/'snapshot.json')!=manifest['snapshot_sha256']:raise RuntimeError('Snapshot changed')
    labels=defaultdict(list);native_runs=0;floor_ratios=set()
    for phase in ['development','test']:
        dataset=read(root,f'dataset-{phase}.json')
        scenes={r['scene']:r for r in dataset['scenes']}
        public={s:read(root,r['public']) for s,r in scenes.items()}
        for row in dataset['outcomes']:
            scene=public[row['scene']];raw=read(root,row['path'])
            floor_ratios.update(map(float,scene['observation']['floor']))
            target=endpoint_target(raw,scene['observation']['critical'])
            if not np.array_equal(target,np.asarray(row['target'],dtype='float32')):raise RuntimeError('Endpoint target mismatch')
            event=target[:,1]<np.asarray(scene['observation']['floor'])
            labels[scenes[row['scene']]['split']].extend(event.tolist());native_runs+=1
    result=read(root,'results.json');records=result['test_records']
    names=[f'{arm}-{seed}' for arm in ['mlp','graph'] for seed in [0,1,2]]
    scenes=sorted({r['scene'] for r in records})
    decisions={n:[next(r['selected'] for r in records if r['method']==n and r['scene']==s) for s in scenes] for n in names}
    identical=len({tuple(v) for v in decisions.values()})==1
    counts=event_counts(labels)
    return {'source':'post_completion_audit_of_unchanged_native_evidence','audited_at':datetime.now(timezone.utc).isoformat(),
        'archive_manifest_sha256':sha256(root/'manifest.json'),'audit_source_sha256':sha256(Path(__file__)),
        'verified_reports':len(manifest['reports']),'verified_checkpoints':len(manifest['checkpoints']),
        'verified_source_capsules':len(manifest['source_capsules']),'verified_native_runs':native_runs,
        'endpoint_targets_recomputed':True,'critical_floor_event_counts':counts,
        'risk_discrimination_testable':all(c['discrimination_testable'] for c in counts.values()),
        'observed_critical_floor_ratios':sorted(floor_ratios),
        'risk_limit':f'Observed aggregate critical floor ratios are {sorted(floor_ratios)}; all cases miss the floor. Zero neural Brier/ECE matches a constant always-miss predictor; it does not establish event discrimination or calibrated physical risk.',
        'all_six_models_choose_identical_test_plans':identical,'test_decisions':decisions,
        'frozen_default':result['research_default_selected_on_validation'],
        'trained_models_accepted_on_validation':sum(v['accepted'] for k,v in read(root,'selection.json')['gates'].items() if k in names),
        'baseline_specialist_accepted_on_validation':read(root,'selection.json')['gates']['ridge']['accepted'],
        'original_results_changed':False,'test_used_to_change_thresholds_or_selection':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/warehouse_native'));args=p.parse_args()
    result=audit(args.root);atomic_json(args.root/'postrun_audit.json',result)
    print('Verified:',result['verified_native_runs'],'native runs,',result['verified_checkpoints'],'checkpoints')
    print('Risk event discrimination testable:',result['risk_discrimination_testable'])


if __name__=='__main__':main()
