"""Durable, fail-fast v2 research runner. Executes real models and simulators."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch

from wirelessjev.warehouse_v2.data import build_dataset
from wirelessjev.warehouse_v2.experiment import (atomic_json, sha256, semantic_contexts,
                                               fit_model, evaluate_model)

REPO = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--stages', nargs='+', default=['semantic', 'training', 'compatibility', 'sensing', 'transfer', 'multitask', 'federation', 'phy', 'ns3'])
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.epochs < 1 or args.threads < 1:
        p.error('Unique seeds, positive epochs and threads are required')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = {'epochs': args.epochs, 'seeds': args.seeds, 'threads': args.threads, 'stages': args.stages}
    config_path = args.out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Use a new output directory for a different campaign configuration')
    atomic_json(config_path, config)
    sources = [*REPO.glob('src/wirelessjev/warehouse_v2/*.py'), Path(__file__), REPO/'uv.lock']
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in sources}
    provenance_path = args.out/'provenance.json'
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous['source_sha256'] != hashes:
            raise RuntimeError('Campaign source changed; use a new directory or explicitly migrate validated stages')
    else:
        atomic_json(provenance_path, {'created_at': stamp(), 'source': 'simulation_and_pretrained_inference',
                    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)),
                    'source_sha256': hashes, 'dependencies': {k: version(k) for k in ['torch', 'sionna', 'numpy', 'scipy', 'transformers']},
                    'hardware': {'cuda': torch.cuda.is_available(), 'cpu_threads': args.threads},
                    'physical_car_access': False, 'hardware_experiments_executed': False})
    status_path = args.out/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'created_at': stamp(), 'stages': {}}
    status.update(state='running', pid=os.getpid(), updated_at=stamp())
    atomic_json(status_path, status)
    data, manifest, embeddings, models = None, None, None, {}
    def dataset():
        nonlocal data, manifest
        if data is None:
            data, manifest = build_dataset(args.out/'cache'/'observations')
            atomic_json(args.out/'dataset.json', manifest)
        return data
    def contexts():
        nonlocal embeddings
        if embeddings is None:
            embeddings = semantic_contexts(args.out)
        return embeddings
    for stage in args.stages:
        if status['stages'].get(stage, {}).get('state') == 'completed':
            print(f'Already completed: {stage}', flush=True)
            continue
        status['stages'][stage] = {'state': 'running', 'started_at': stamp()}
        status.update(active_stage=stage, updated_at=stamp())
        atomic_json(status_path, status)
        started = time.perf_counter()
        try:
            if stage == 'semantic':
                contexts()
            elif stage == 'training':
                ds = dataset()
                for seed in args.seeds:
                    for arm in ['instant_mlp', 'temporal_graph', 'semantic_graph']:
                        model, meta = fit_model(ds['train'], ds['validation'], contexts(), args.out,
                                                seed=seed, arm=arm, epochs=args.epochs)
                        models[arm, seed] = model, meta
                        for split in ['test', 'shift', 'fleet200']:
                            report_path = args.out/f'eval-{arm}-{seed}-{split}.json'
                            if not report_path.exists():
                                report = evaluate_model(model, ds[split], contexts(), split=split, arm=arm,
                                                        calibration=meta['calibration'])
                                # Preserve every per-case prediction outside Git, with a digest.
                                records = report.pop('records')
                                records_path = args.out/'cache'/f'predictions-{arm}-{seed}-{split}.json'
                                atomic_json(records_path, records)
                                report.update(predictions=str(records_path), predictions_sha256=sha256(records_path),
                                              checkpoint_sha256=meta['checkpoint_sha256'], seed=seed)
                                atomic_json(report_path, report)
            elif stage in ['compatibility', 'sensing', 'transfer', 'multitask', 'federation', 'ns3']:
                from wirelessjev.warehouse_v2 import studies
                getattr(studies, f'run_{stage}')(args.out, dataset(), contexts(), args.seeds)
            elif stage == 'phy':
                from wirelessjev.warehouse_v2.phy import run_phy
                atomic_json(args.out/'phy.json', run_phy())
            else:
                raise ValueError(f'Unknown stage {stage}')
        except BaseException as error:
            status['stages'][stage].update(state='failed', error=str(error), traceback=traceback.format_exc())
            status.update(state='failed', updated_at=stamp())
            atomic_json(status_path, status)
            raise
        status['stages'][stage].update(state='completed', completed_at=stamp(), elapsed_s=time.perf_counter()-started)
        atomic_json(status_path, status)
    status.update(state='completed', active_stage=None, completed_at=stamp(), updated_at=stamp())
    atomic_json(status_path, status)


if __name__ == '__main__':
    main()
