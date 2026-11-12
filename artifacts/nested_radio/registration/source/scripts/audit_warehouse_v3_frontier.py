"""Post-run audit of frozen models: crossed intervals and per-domain full cost."""
from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np
import torch
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256
from wirelessjev.warehouse_v3.data import held_out
from wirelessjev.warehouse_v3.learning import ARMS,predict
from wirelessjev.warehouse_v3.model import RadioModel
from wirelessjev.warehouse_horizon.planning import charge


def crossed(matrix,draws=4000):
    matrix=np.asarray(matrix);rng=np.random.default_rng(0)
    values=[matrix[np.ix_(rng.integers(matrix.shape[0],size=matrix.shape[0]),rng.integers(matrix.shape[1],size=matrix.shape[1]))].mean() for _ in range(draws)]
    return {'difference':float(matrix.mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),
        'matrix':matrix.tolist(),'method':'crossed_model_seed_and_environment_percentile_bootstrap'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('/workspace/training/warehouse-v3'))
    p.add_argument('--frontier',type=Path,default=Path('/workspace/training/warehouse-v3-frontier-matched'))
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    while not (args.frontier/'status.json').exists() or json.loads((args.frontier/'status.json').read_text())['state']!='completed':
        if (args.frontier/'status.json').exists() and json.loads((args.frontier/'status.json').read_text())['state']=='failed':raise RuntimeError('Frontier failed')
        time.sleep(5)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('full_pipeline_benchmark',repo/'scripts/benchmark_warehouse_v3.py')
    benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
    selection=json.loads((args.run/'selection.json').read_text())
    provenance=json.loads((args.run/'provenance.json').read_text());test=held_out(args.run,provenance['fingerprint'])
    with np.load(args.run/'semantic_contexts.npz',allow_pickle=False) as data:
        contexts={text:torch.tensor(value) for text,value in zip(data['texts'].tolist(),data['embeddings'])}
    models={'numerical':None}
    for seed in [0,1,2]:
        for arm in ARMS:
            key=f'{arm}-{seed}';path=args.run/'checkpoints'/f'{key}.pt'
            if sha256(path)!=selection['checkpoint_sha256'][path.name]:raise RuntimeError('Frozen model changed')
            saved=torch.load(path,map_location='cpu',weights_only=True);model=RadioModel(**saved['config']);model.load_state_dict(saved['model']);model.eval();models[key]=model
    rows=[];timings=[]
    for domain,samples in test.items():
        for key,model in models.items():
            calibration=selection['calibration'][key]
            for scene in sorted({s['scene'] for s in samples}):
                subset=[s for s in samples if s['scene']==scene];errors=[]
                for sample in subset:
                    prediction=predict(model,sample);service,_=charge(prediction,sample['plans'],sample['observation'],temperature=calibration['temperature'])
                    errors.append(float(((service-sample['target']['service'])**2).mean()))
                rows.append({'domain':domain,'scene':scene,'method':key,'service_rmse':float(np.sqrt(np.mean(errors)))})
            timings.append({'domain':domain,'method':key,**benchmark.latency(model,samples,contexts,calibration)})
        print('Fixed-model interval/timing audit',domain,flush=True)
    intervals={}
    for domain in test:
        scenes=sorted({r['scene'] for r in rows if r['domain']==domain})
        lookup={(r['scene'],r['method']):r['service_rmse'] for r in rows if r['domain']==domain}
        for arm in ARMS:
            matrix=[[lookup[scene,f'{arm}-{seed}']-lookup[scene,'numerical'] for scene in scenes] for seed in [0,1,2]]
            intervals[f'{domain}/{arm}-minus-numerical']=crossed(matrix)
        for arm in ['graph_masked','graph_jepa']:
            matrix=[[lookup[scene,f'{arm}-{seed}']-lookup[scene,f'graph_scratch-{seed}'] for scene in scenes] for seed in [0,1,2]]
            intervals[f'{domain}/{arm}-minus-scratch']=crossed(matrix)
    atomic_json(root/'intervals_and_cost.json',{'source':'frozen_model_simulation_re_evaluation_and_actual_cpu_timing',
        'selection_sha256':sha256(args.run/'selection.json'),'per_environment':rows,'intervals':intervals,'timings':timings,
        'limits':['This re-evaluates the same frozen models and held-out scenes; it is not an independent replication',
            'Intervals describe mean per-environment RMSE differences, not the pooled-RMSE statistic',
            'Model seeds and environments are resampled separately; three environments per domain remain small',
            'Local timing includes decoding, public belief, candidates, prediction and checks; physical sensing/actuation remain unmeasured']})
    atomic_json(root/'status.json',{'state':'completed','report_sha256':sha256(root/'intervals_and_cost.json')})


if __name__=='__main__':main()
