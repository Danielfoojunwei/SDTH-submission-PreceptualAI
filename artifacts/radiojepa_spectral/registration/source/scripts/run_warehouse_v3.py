"""Run real compositional-mission and radio-representation ablations."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
import time
import numpy as np
import torch

from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v3.data import verified_parent,held_out
from wirelessjev.warehouse_v3.learning import (ARMS,semantic_contexts,train,calibrate,cases,summarise)
from wirelessjev.warehouse_v3.missions import MISSIONS

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=Path('/workspace/training/warehouse-horizon'))
    args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=root/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('state')=='completed':
        for name,digest in status['reports'].items():
            if sha256(root/name)!=digest:raise RuntimeError('Completed evidence changed')
        print('Completed v3 campaign verified; no training restarted');return
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sources=[*REPO.glob('src/wirelessjev/**/*.py'),Path(__file__),REPO/'uv.lock',REPO/'pyproject.toml',
        REPO/'artifacts/warehouse_v2/studies/wifi_link_curves.json',REPO/'third_party/vjepa2/SOURCE.json']
    hashes={str(path.relative_to(REPO)):sha256(path) for path in sources}
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    provenance=root/'provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text())['fingerprint']!=fingerprint:raise RuntimeError('Changed unfinished campaign; use a new output directory')
    else:
        for path in sources:
            dest=root/'source_snapshot'/path.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        atomic_json(provenance,{'source':'simulation_and_actual_pinned_pretrained_inference','source_sha256':hashes,
            'fingerprint':fingerprint,'parent_selection_sha256':sha256(args.parent/'selection.json'),
            'hardware_experiments_executed':False,'model_seeds':[0,1,2],'train_scenes':18,'validation_scenes':6,
            'test_scenes_per_domain':3,'test_domains':['ordinary','layout','fleet200','traffic'],
            'pretraining_reference':'MIT-licensed V-JEPA2 loss/EMA; new radio encoders, no pretrained video weights',
            'masked_pretraining_reference':'original implementation of masked observed-link reconstruction; no LWM code or weights'})
    def stage(name):
        status.update(state='running',stage=name,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat());atomic_json(status_path,status)
    try:
        stage('actual_d3_contexts');contexts=semantic_contexts(root)
        stage('verified_training_observations');data=verified_parent(args.parent)
        models={};calibration={}
        stage('independent_pretraining_and_supervised_ablations')
        for seed in [0,1,2]:
            for arm in ARMS:
                key=f'{arm}-{seed}';models[key],_=train(data['train'],data['validation'],contexts,root,seed=seed,arm=arm)
                calibration[key]=calibrate(models[key],data['validation'])
        calibration['numerical']=calibrate(None,data['validation'])
        stage('validation_routing')
        validation={};details={}
        for key,model in [('numerical',None),*models.items()]:
            summary,rows=cases(model,data['validation'],contexts,calibration[key],texts=MISSIONS['train'])
            validation[key]=summary;details[key]=rows
        routes={}
        for family in sorted({r['family'] for r in details['numerical']}):
            candidates=[]
            for key,rows in details.items():
                subset=[r for r in rows if r['family']==family]
                # Defer is charged. An unsafe proposal carries a much larger
                # penalty; routing never relaxes a contract or its guard.
                penalty=np.mean([10*r['false_feasible']+(not r['predicted_feasible'])+
                    (r['ordinal_regret_when_feasible'] or 0.) for r in subset])
                candidates.append((float(penalty),key!='numerical',key))
            routes[family]=min(candidates)[2]
        sealed={'calibration':calibration,'routes':routes,'validation':validation,
            'checkpoint_sha256':{p.name:sha256(p) for p in (root/'checkpoints').glob('*.pt')},
            'test_accessed_before_selection':False,'route_fit_mission_texts':MISSIONS['train']}
        selection=root/'selection.json'
        if selection.exists():
            old=json.loads(selection.read_text())
            if any(old[k]!=v for k,v in sealed.items()):raise RuntimeError('Selection cannot change after held-out access')
        else:
            if any((root/'cache').glob('15*.npz')):raise RuntimeError('Test cache preceded selection seal')
            atomic_json(selection,{**sealed,'sealed_at':datetime.now(timezone.utc).isoformat()})
        stage('new_domain_evaluation');test=held_out(root,fingerprint);results=[]
        for domain,samples in test.items():
            all_rows={}
            for key,model in [('numerical',None),*models.items()]:
                started=time.perf_counter();summary,rows=cases(model,samples,contexts,calibration[key])
                all_rows[key]=rows;results.append({'domain':domain,'method':key,'summary':summary,
                    'evaluation_elapsed_s':time.perf_counter()-started})
                atomic_json(root/'cases'/f'{domain}-{key}.json',rows)
            routed=[]
            for i,row in enumerate(all_rows['numerical']):
                key=routes.get(row['family'],'numerical');routed.append({**all_rows[key][i],'selected_method':key})
            atomic_json(root/'cases'/f'{domain}-router.json',routed)
            results.append({'domain':domain,'method':'validation_router','summary':summarise(routed)})
            print('Held-out domain complete',domain,flush=True)
        atomic_json(root/'results.json',{'source':'simulation','results':results,'model_seeds':[0,1,2],
            'selection_sha256':sha256(selection),'future_truth_in_policy_inputs':False,
            'limits':['Only the declared complete mission grammar is supported',
              'Candidate-set infeasibility is not global physical infeasibility',
              'Marginal intervals do not ensure joint service floors',
              'No physical-car or foundation-model claim; new radio encoders are trained from scratch']})
    except BaseException as error:
        status.update(state='failed',error=str(error),traceback=traceback.format_exc());atomic_json(status_path,status);raise
    status.update(state='completed',stage=None,completed_at=datetime.now(timezone.utc).isoformat(),
        reports={name:sha256(root/name) for name in ['semantic_reader.json','selection.json','results.json']});atomic_json(status_path,status)


if __name__=='__main__':main()
