"""Execute the sealed multi-step warehouse campaign with actual training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import torch

from wirelessjev.warehouse_v2.experiment import atomic_json, sha256
from wirelessjev.warehouse_horizon.data import CONFIG, dataset
from wirelessjev.warehouse_horizon.learning import fit, select_residual, model_prediction, blend, assess
from wirelessjev.warehouse_horizon.world import World
from wirelessjev.warehouse_horizon.evaluation import rollout

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Completed runs are immutable evidence, even after the working tree has
    # evolved. Verify the executed source capsule instead of restarting them.
    status_path = root/'status.json'
    if status_path.exists():
        done = json.loads(status_path.read_text())
        if done.get('state') == 'completed':
            previous = json.loads((root/'provenance.json').read_text())
            if any(sha256(root/'source_snapshot'/name) != digest for name, digest in previous['source_sha256'].items()):
                raise RuntimeError('Completed run source capsule changed')
            if any(sha256(root/name) != digest for name, digest in done['reports'].items()):
                raise RuntimeError('Completed run report changed')
            print('Completed campaign and executed source/report hashes verified; no training restarted')
            return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sources = [*REPO.glob('src/wirelessjev/**/*.py'), Path(__file__), REPO/'uv.lock', REPO/'pyproject.toml',
               REPO/'artifacts/warehouse_v2/studies/wifi_link_curves.json']
    hashes = {str(p.relative_to(REPO)): sha256(p) for p in sources}
    fingerprint = hashlib.sha256(json.dumps({'sources': hashes, 'config': CONFIG}, sort_keys=True).encode()).hexdigest()
    label_paths = [p for p in sources if p.name in ['world.py', 'belief.py', 'planning.py', 'data.py', 'fastplan.py',
                                                  'radio.py', 'warehouse.py', 'envlearn.py', 'hardware.py', 'wifi_link_curves.json']]
    label_hashes = {str(p.relative_to(REPO)): sha256(p) for p in label_paths}
    data_fingerprint = hashlib.sha256(json.dumps({'sources': label_hashes, 'config': CONFIG}, sort_keys=True).encode()).hexdigest()
    provenance_path = root/'provenance.json'
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous['fingerprint'] != fingerprint:
            raise RuntimeError('Campaign source changed; use a new output directory with explicit lineage')
    else:
        for p in sources:
            dest = root/'source_snapshot'/p.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        atomic_json(provenance_path, {'fingerprint': fingerprint, 'data_fingerprint': data_fingerprint,
                    'source_sha256': hashes, 'source': 'simulation_and_actual_pretrained_inference',
                    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                    'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)),
                    'created_at': now(), 'dependencies': {k: version(k) for k in ['torch', 'numpy', 'scipy', 'transformers']},
                    'hardware_experiments_executed': False})
    atomic_json(root/'config.json', CONFIG)
    status_path = root/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {'created_at': now()}
    if status.get('state') == 'completed':
        if any(sha256(root/name) != expected for name, expected in status['reports'].items()):
            raise RuntimeError('Completed campaign report changed')
        print('Campaign already completed and report hashes verify; no work restarted')
        return
    status.update(state='running', pid=os.getpid(), updated_at=now())
    atomic_json(status_path, status)
    def stage(name):
        status.update(stage=name, updated_at=now())
        atomic_json(status_path, status)
    try:
        stage('actual_typed_mission_reader')
        reader_path = root/'reader.json'
        if not reader_path.exists():
            from wirelessjev.policy import WirelessJEV, MODEL_REPO, MODEL_REVISION
            from wirelessjev.warehouse_horizon.planning import read_contracts
            engine = WirelessJEV(device='cpu')
            readings = read_contracts(engine)
            atomic_json(reader_path, {'model': {'repo': MODEL_REPO, 'revision': MODEL_REVISION},
                                      'model_executed': True, 'readings': readings})
            del engine
        readings = json.loads(reader_path.read_text())['readings']
        stage('long_episode_training_data')
        ds = dataset(root/'cache', data_fingerprint)
        models, selections = {}, {}
        selection_path = root/'selection.json'
        stage('independent_training_and_validation_selection')
        baseline = select_residual(None, ds['validation'])
        for seed in CONFIG['model_seeds']:
            for graph in [False, True]:
                model, metadata = fit(ds['train'], ds['validation'], root, seed=seed, graph=graph, fingerprint=fingerprint)
                models[seed, graph] = model
                key = f'{seed}-{graph}'
                selections[key] = select_residual(model, ds['validation'])
                atomic_json(root/f'selection-{key}.json', selections[key])
        # This receipt is written before reading/generating any test outcome.
        sealed = {'baseline': baseline, 'models': selections,
                  'checkpoint_sha256': {p.name: sha256(p) for p in (root/'checkpoints').glob('*.pt')}}
        if selection_path.exists():
            previous = json.loads(selection_path.read_text())
            if any(previous[key] != value for key, value in sealed.items()):
                raise RuntimeError('Refusing to change selection after held-out access')
        else:
            if any((root/'cache'/f'{seed}.npz').exists() for seed in CONFIG['test']):
                raise RuntimeError('Test cache exists before selection was sealed; use a fresh run')
            atomic_json(selection_path, {**sealed, 'sealed_at': now(), 'test_outcomes_accessed_before_selection': False})
        stage('sealed_held_out_forecasting')
        test = dataset(root/'cache', data_fingerprint, splits=('test',))['test']
        offline = {'baseline': assess(test, [model_prediction(None, s) for s in test], risk_temperature=baseline['temperature']), 'models': {}}
        for (seed, graph), model in models.items():
            selected = selections[f'{seed}-{graph}']
            bases = [model_prediction(None, sample) for sample in test]
            learned = [model_prediction(model, sample) for sample in test]
            predictions = [blend(b, p, selected['selected_strength']) for b, p in zip(bases, learned)]
            offline['models'][f'{seed}-{graph}'] = {
                'raw': assess(test, learned),
                'validated_blend': assess(test, predictions, risk_temperature=selected['temperature'])}
        atomic_json(root/'offline.json', offline)
        stage('full_cross_closed_loop')
        records = []
        for scene in CONFIG['test']:
            world = World(scene)
            arms = [('stay', None, baseline, None), ('budgeted_strongest', None, baseline, None),
                    ('unconstrained_strongest', None, baseline, None), ('numerical_planner', None, baseline, None)]
            arms += [('residual_graph' if graph else 'residual_mlp', model, selections[f'{seed}-{graph}'], seed)
                     for (seed, graph), model in models.items()]
            for method, model, selected, seed in arms:
                path = root/'rollouts'/f'{scene}-{method}-{seed}.json'
                if path.exists():
                    report = json.loads(path.read_text())
                    if report.get('fingerprint') != fingerprint or report.get('selection_sha256') != sha256(selection_path):
                        raise RuntimeError('Saved rollout provenance mismatch')
                else:
                    report = rollout(world, method, readings, model=model, selection=selected)
                    report['model_seed'] = seed
                    report['fingerprint'] = fingerprint
                    report['selection_sha256'] = sha256(selection_path)
                    atomic_json(path, report)
                records.append(report)
                print(f'Horizon closed loop scene={scene} seed={seed} {method}: utility={report["mean_mission_utility"]:.4f} service={report["metrics"]["mean_service"]:.4f}', flush=True)
        atomic_json(root/'closedloop.json', {'source': 'multi_step_warehouse_simulation', 'records': records,
                    'test_environments': CONFIG['test'], 'model_seeds': CONFIG['model_seeds'],
                    'selection_sha256': sha256(selection_path), 'full_model_environment_cross': True,
                    'physical_car_trial': False})
    except BaseException as error:
        status.update(state='failed', error=str(error), traceback=traceback.format_exc(), updated_at=now())
        atomic_json(status_path, status)
        raise
    status.update(state='completed', stage=None, completed_at=now(), updated_at=now(),
                  reports={name: sha256(root/name) for name in ['reader.json', 'selection.json', 'offline.json', 'closedloop.json']})
    atomic_json(status_path, status)


if __name__ == '__main__':
    main()
