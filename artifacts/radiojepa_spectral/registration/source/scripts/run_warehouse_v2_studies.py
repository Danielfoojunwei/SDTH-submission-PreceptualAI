"""Run the extension studies against completed, immutable core checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import torch

from wirelessjev.warehouse_v2.data import build_dataset
from wirelessjev.warehouse_v2.experiment import atomic_json, semantic_contexts, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', default=['compatibility', 'quality', 'closedloop', 'sensing', 'transfer', 'multitask', 'federation', 'phy', 'link_curves', 'ns3'])
    args = parser.parse_args()
    if json.loads((args.core_run/'status.json').read_text())['state'] != 'completed':
        raise RuntimeError('Core training must complete and validate before extension studies')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    core = {'path': str(args.core_run.resolve()), 'core_provenance_sha256': sha256(args.core_run/'provenance.json')}
    if (args.out/'core_run.json').exists() and json.loads((args.out/'core_run.json').read_text()) != core:
        raise RuntimeError('Refusing to mix different core model runs')
    atomic_json(args.out/'core_run.json', core)
    repo = Path(__file__).resolve().parents[1]
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    snapshot = args.out/'source_snapshots'/attempt
    sources = [*repo.glob('src/wirelessjev/**/*.py'), *repo.glob('third_party/ns3/*.cc'),
               repo/'docs/experiments/e7_calibration_study.py', Path(__file__), repo/'uv.lock', repo/'pyproject.toml']
    hashes = {}
    for source in sources:
        relative = source.relative_to(repo)
        dest = snapshot/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[str(relative)] = sha256(source)
    atomic_json(snapshot/'provenance.json', {'created_at': now(), 'source_sha256': hashes, 'core': core,
                'dependencies': {name: version(name) for name in ['torch', 'sionna', 'chronos-forecasting']},
                'hardware_experiments_executed': False})
    path = args.out/'status.json'
    status = json.loads(path.read_text()) if path.exists() else {'created_at': now(), 'stages': {}}
    status.update(state='running', pid=os.getpid(), updated_at=now(), source_attempt=attempt)
    atomic_json(path, status)
    data, _ = build_dataset(args.core_run/'cache'/'observations')
    embeddings = semantic_contexts(args.core_run)
    from wirelessjev.warehouse_v2 import studies
    for stage in args.stages:
        previous = status['stages'].get(stage, {})
        report_path = args.out/f'{stage}.json'
        if previous.get('state') == 'completed':
            if previous['report_sha256'] != sha256(report_path):
                raise RuntimeError(f'Completed report changed: {stage}')
            print(f'Already completed: {stage}', flush=True)
            continue
        status['stages'][stage] = {'state': 'running', 'started_at': now(), 'source_attempt': attempt}
        status.update(active_stage=stage, updated_at=now())
        atomic_json(path, status)
        started = time.perf_counter()
        try:
            if stage in ['quality', 'closedloop']:
                import importlib
                module = importlib.import_module('wirelessjev.warehouse_v2.'+stage)
                result = module.study(args.out, data, embeddings, [0, 1, 2]) if stage == 'quality' else module.study(args.out, [0, 1, 2])
                atomic_json(report_path, result)
            elif stage == 'phy':
                from wirelessjev.warehouse_v2.phy import run_phy
                atomic_json(report_path, run_phy())
            elif stage == 'link_curves':
                from wirelessjev.warehouse_v2.link_curves import study
                atomic_json(report_path, study(args.out, data))
            else:
                getattr(studies, 'run_'+stage)(args.out, data, embeddings, [0, 1, 2])
        except BaseException as error:
            status['stages'][stage].update(state='failed', error=str(error), traceback=traceback.format_exc())
            status.update(state='failed', updated_at=now())
            atomic_json(path, status)
            raise
        status['stages'][stage].update(state='completed', completed_at=now(), elapsed_s=time.perf_counter()-started,
                                       report_sha256=sha256(report_path))
        atomic_json(path, status)
    status.update(state='completed', active_stage=None, updated_at=now(), completed_at=now())
    atomic_json(path, status)


if __name__ == '__main__':
    main()
