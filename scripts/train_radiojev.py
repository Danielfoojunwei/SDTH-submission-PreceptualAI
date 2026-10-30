"""Run full RadioJEV E1-E5 in an isolated directory, retaining completed work."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ('https://huggingface.co/vllm-sr/d3-lite/resolve/'
                '03560c26f5199c3c04129a2ccb38960c913635e0/config.json')
STAGES = [
    ('e1_no_llm', 'e1', ['--arm', 'no_llm']),
    ('e1_radio_lora', 'e1', ['--arm', 'radio_lora']),
    ('e1_radio_frozen', 'e1', ['--arm', 'radio_frozen']),
    ('e1_intent', 'e1', ['--arm', 'intent']),
    ('e2_radio_q', 'e2', ['--arm', 'radio_q']),
    ('e2_e1_zero', 'e2', ['--arm', 'e1_zero']),
    ('e2_intent', 'e2', ['--arm', 'intent']),
    ('e3', 'e3', []),
    ('e45', 'e45', []),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path, value):
    atomic(path, (json.dumps(value, indent=2) + '\n').encode())


def scene_job(kind, seed, site_options):
    if kind == 'e1':
        from wirelessjev.radiojev.data import build_scene
        return build_scene(seed)
    from wirelessjev.radiojev.e2 import build_scene
    if kind == 'e2':
        return build_scene(seed)
    from functools import partial
    from wirelessjev.twin.warehouse import make_warehouse
    return build_scene(seed, scene_fn=partial(make_warehouse, **site_options), linear=True)


def prepare(kind, run, workers, status, save):
    from wirelessjev.radiojev import e1, e2, e45
    target = run / f'{kind}_data.pkl'
    if target.exists():
        data = pickle.loads(target.read_bytes())
        assert data
        return
    tasks = []
    if kind in ('e1', 'e2'):
        seeds = (e1 if kind == 'e1' else e2).SEEDS
        for split, entries in seeds.items():
            for seed in entries:
                tasks.append((split, seed, None))
    else:
        for si, (site, options) in enumerate(e45.SITES.items()):
            base = 70000 + 1000 * si
            for split, entries in [('train', range(base, base + e45.N_TRAIN)),
                                   ('test', range(base + 500, base + 500 + e45.N_TEST))]:
                tasks.extend((site + '/' + split, seed, options) for seed in entries)
    parts = run / 'scene_cache' / kind
    rows = {}
    pending = []
    for split, seed, options in tasks:
        path = parts / split / f'{seed}.pkl'
        if path.exists():
            row = pickle.loads(path.read_bytes())
            assert row['seed'] == seed and row['graphs'] and row['util']
            rows[(split, seed)] = row
        else:
            pending.append((split, seed, options))
    status['data'] = {'kind': kind, 'completed': len(rows), 'total': len(tasks)}
    save()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scene_job, kind, seed, options): (split, seed)
                   for split, seed, options in pending}
        for future in as_completed(futures):
            split, seed = futures[future]
            row = future.result()
            assert row['seed'] == seed and row['graphs'] and row['util']
            atomic(parts / split / f'{seed}.pkl', pickle.dumps(row))
            rows[(split, seed)] = row
            status['data']['completed'] = len(rows)
            save()
            if len(rows) == 1 or len(rows) % 10 == 0 or len(rows) == len(tasks):
                print(f'[{now()}] {kind} data {len(rows)}/{len(tasks)} scenes', flush=True)
    data = {}
    for split, seed, _ in tasks:
        row = rows[(split, seed)]
        if kind == 'e45':
            site, subset = split.split('/')
            data.setdefault(site, {}).setdefault(subset, []).append(row)
        else:
            data.setdefault(split, []).append(row)
    atomic(target, pickle.dumps(data))


def execute_stage(stage, run, threads):
    import random
    import numpy as np
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(threads)
    modules = {name: importlib.import_module('wirelessjev.radiojev.' + name)
               for name in ('e1', 'e2', 'e3', 'e45')}
    for module in modules.values():
        module.OUT = run
        assert not module.SMOKE
    _, name, arguments = next(entry for entry in STAGES if entry[0] == stage)
    sys.argv = [str(modules[name].__file__), *arguments]
    modules[name].main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--stage', choices=[s[0] for s in STAGES])
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error('workers and threads must be positive')
    run = args.run_dir.resolve()
    os.chdir(ROOT)
    os.environ.pop('WJ_SMOKE', None)
    os.environ.pop('HF_HUB_OFFLINE', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if args.plan:
        print(json.dumps({'run_dir': str(run), 'workers': args.workers,
                          'threads': args.threads, 'smoke': False, 'seed': 0,
                          'stages': [s[0] for s in STAGES]}, indent=2))
        return
    run.mkdir(parents=True, exist_ok=True)
    if args.stage:
        execute_stage(args.stage, run, args.threads)
        return
    lock = (run / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        'created_at': now(), 'run_dir': str(run), 'seed': 0,
        'stages': {s[0]: {'status': 'pending'} for s in STAGES}}
    status.update(status='running', pid=os.getpid(), workers=args.workers,
                  threads=args.threads, smoke=False)

    def save():
        if status['status'] not in ('blocked', 'failed'):
            status.pop('blocked_stage', None)
            status.pop('reason', None)
        status['updated_at'] = now()
        write_json(status_path, status)

    save()
    try:
        prepare('e1', run, args.workers, status, save)
        for key, _, _ in STAGES:
            result = run / f'{key}.json'
            if status['stages'][key]['status'] == 'completed':
                assert result.exists() and json.loads(result.read_text())
                continue
            if key == 'e1_radio_lora':
                # This preparation remains useful while model access is blocked.
                prepare('e2', run, args.workers, status, save)
                prepare('e45', run, args.workers, status, save)
            if key != 'e1_no_llm':
                try:
                    request = urllib.request.Request(MODEL_CONFIG, method='HEAD')
                    with urllib.request.urlopen(request, timeout=30) as response:
                        assert response.status == 200
                except (OSError, AssertionError) as error:
                    status.update(status='blocked', blocked_stage=key,
                                  reason='Pinned Hugging Face model config is inaccessible: ' + str(error))
                    save()
                    print(status['reason'], flush=True)
                    return 2
            entry = status['stages'][key]
            for stale in ('pid', 'returncode', 'finished_at'):
                entry.pop(stale, None)
            entry.update(status='running', started_at=now(), log=str(run / f'{key}.log'))
            save()
            print(f'[{now()}] START full {key}', flush=True)
            with (run / f'{key}.log').open('a') as log:
                process = subprocess.Popen([sys.executable, '-u', __file__, '--run-dir', str(run),
                                            '--stage', key, '--threads', str(args.threads)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                entry['pid'] = process.pid
                save()
                returncode = process.wait()
            if returncode:
                entry.update(status='failed', returncode=returncode, finished_at=now())
                status['status'] = 'failed'
                save()
                return returncode
            report = json.loads(result.read_text())
            assert isinstance(report, dict) and report
            if key in ('e1_radio_lora', 'e2_radio_q', 'e3'):
                checkpoint = run / {'e1_radio_lora': 'ckpt_radio_lora.pt',
                                    'e2_radio_q': 'ckpt_radio_q.pt', 'e3': 'ckpt_e3.pt'}[key]
                assert checkpoint.exists() and checkpoint.stat().st_size > 0, checkpoint
            entry.update(status='completed', returncode=0, finished_at=now())
            save()
            print(f'[{now()}] COMPLETED full {key}', flush=True)
        status['status'] = 'completed'
        save()
        return 0
    except BaseException as error:
        status.update(status='failed', reason=type(error).__name__ + ': ' + str(error))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
