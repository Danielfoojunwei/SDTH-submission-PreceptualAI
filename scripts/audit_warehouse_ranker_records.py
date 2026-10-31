"""Recover and verify exact plan indices from frozen ranker checkpoints.

The first executed ranker script stored a feasible-set indicator under the
misleading name gold_choice. Its regret calculation used the actual plans.
This audit retains that raw evidence, reconstructs actual choices and verifies
every published quality field before adding the corrected indices.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import time
import numpy as np
import torch
from torch import nn
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v3.data import DOMAINS,episode
from wirelessjev.warehouse_v3.model import RadioModel
from wirelessjev.warehouse_v3.missions import MISSIONS,compile_contract


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ranker',type=Path,default=Path('/workspace/training/warehouse-v3-ranker'))
    p.add_argument('--core',type=Path,default=Path('/workspace/training/warehouse-v3'));p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    while not (args.ranker/'status.json').exists() or json.loads((args.ranker/'status.json').read_text())['state']!='completed':time.sleep(5)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    repo=Path(__file__).resolve().parents[1];path=repo/'scripts/benchmark_warehouse_ranker.py'
    spec=importlib.util.spec_from_file_location('ranker_feature_extraction',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources=[Path(__file__),path,*repo.glob('src/wirelessjev/**/*.py')]
    for source in sources:
        dest=root/'source_snapshot'/source.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    atomic_json(root/'provenance.json',{'source_sha256':{str(p.relative_to(repo)):sha256(p) for p in sources},
        'scope':'Re-evaluation of frozen models on the original ranker holdout; not a new replication',
        'original_field_issue':'gold_choice was a feasible-set indicator, not a plan index',
        'source':'actual_frozen_model_re_evaluation','hardware_experiments_executed':False})
    atomic_json(root/'status.json',{'state':'running','stage':'verify_and_reconstruct_plan_indices'})
    core_selection=json.loads((args.core/'selection.json').read_text());rank_selection=json.loads((args.ranker/'selection.json').read_text())
    with np.load(args.core/'semantic_contexts.npz',allow_pickle=False) as data:embeddings=dict(zip(data['texts'].tolist(),data['embeddings']))
    models={};heads={}
    for seed in [0,1,2]:
        path=args.core/'checkpoints'/f'graph_scratch-{seed}.pt'
        if sha256(path)!=core_selection['checkpoint_sha256'][path.name]:raise RuntimeError('Frozen backbone changed')
        saved=torch.load(path,map_location='cpu',weights_only=True);model=RadioModel(**saved['config']);model.load_state_dict(saved['model']);model.eval();models[seed]=model
        for text in [False,True]:
            path=args.ranker/'checkpoints'/f'ranker-{seed}-{text}.pt'
            if sha256(path)!=rank_selection['models'][f'{seed}-{text}']['checkpoint_sha256']:raise RuntimeError('Frozen ranker changed')
            saved=torch.load(path,map_location='cpu',weights_only=True);state=saved['model']
            head=nn.Sequential(nn.Linear(state['0.weight'].shape[1],64),nn.GELU(),nn.Linear(64,1));head.load_state_dict(state);head.eval()
            heads[seed,text]=(head,saved['mean'].numpy(),saved['std'].numpy())
    verified=0
    for index,(domain,(n,mode,scale)) in enumerate(DOMAINS.items()):
        samples=[s for offset in range(3) for s in episode(180000+index*1000+offset,n=n,domain=mode,bulk_scale=scale)]
        corrected={(seed,text):[] for seed in models for text in [False,True]}
        original={(seed,text):json.loads((args.ranker/'cases'/f'{domain}-{seed}-{text}.json').read_text()) for seed in models for text in [False,True]}
        position=0
        for sample in samples:
            for mission in MISSIONS['train']+MISSIONS['novel']:
                for seed,model in models.items():
                    row=module.features(model,sample,compile_contract(mission),embeddings[mission],core_selection['calibration'][f'graph_scratch-{seed}'])
                    soft=row[4];gold_indices=np.flatnonzero(row[3])
                    gold=int(max(gold_indices,key=lambda i:tuple(np.round(soft[i],8))+(-int(i),))) if len(gold_indices) else None
                    order=sorted({tuple(np.round(soft[i],8)) for i in gold_indices},reverse=True)
                    for text in [False,True]:
                        head,mu,std=heads[seed,text];xx=(row[0]-mu)/std
                        if not text:xx[:,-1024:]=0.
                        with torch.no_grad():score=head(torch.tensor(xx)).squeeze(-1).numpy()
                        choice=int(np.where(row[2],score,-np.inf).argmax()) if row[2].any() else None
                        actual=choice is not None and bool(row[3][choice]);regret=None
                        if actual:regret=order.index(tuple(np.round(soft[choice],8)))/max(1,len(order)-1)
                        old=original[seed,text][position]
                        if old['scene']!=sample['scene'] or old['tick']!=sample['tick'] or old['text']!=mission:raise RuntimeError('Raw row identity mismatch')
                        if (old['predicted_feasible']!=(choice is not None) or old['actual_feasible']!=actual or
                            old['false_feasible']!=((choice is not None) and not actual) or old['ordinal_regret_when_feasible']!=regret):
                            raise RuntimeError('Fixed-model re-evaluation changed a reported quality field')
                        if (old['gold_choice'] is not None)!=(gold is not None):raise RuntimeError('Original feasibility indicator disagreed')
                        corrected[seed,text].append({**old,'choice':choice,'gold_choice':gold,'truth_has_feasible_candidate':gold is not None})
                        verified+=1
                position+=1
        for (seed,text),rows in corrected.items():atomic_json(root/'cases'/f'{domain}-{seed}-{text}.json',rows)
        print('Verified ranker plan records',domain,flush=True)
    atomic_json(root/'record_audit.json',{'source':'actual_frozen_model_re_evaluation','verified_quality_rows':verified,
        'parent_report_sha256':sha256(args.ranker/'ranker.json'),'quality_fields_changed':False,
        'corrected_fields':['choice','gold_choice'],'source_holdout_reused':True,'original_raw_evidence_retained':True})
    atomic_json(root/'status.json',{'state':'completed','report_sha256':sha256(root/'record_audit.json')})


if __name__=='__main__':main()
