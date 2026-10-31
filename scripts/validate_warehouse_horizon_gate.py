"""Select mission branches on validation rollouts, then audit new environments.

This selector was designed after the original held-out horizon study exposed
lower-tail regressions. Its prospective audit uses disjoint environments.
Rejected learned branches remain available as research evidence, not defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from wirelessjev.warehouse_v2.experiment import atomic_json, sha256
from wirelessjev.warehouse_horizon.data import CONFIG
from wirelessjev.warehouse_horizon.world import World
from wirelessjev.warehouse_horizon.model import HorizonModel
from wirelessjev.warehouse_horizon.planning import compile_contract, MISSION_TEXTS
from wirelessjev.warehouse_horizon.evaluation import rollout
from dataclasses import asdict

AUDIT_SEEDS = [143000, 143001, 143002]


def now():
    return datetime.now(timezone.utc).isoformat()


def candidate_gate(baseline, candidate):
    result = {}
    for mission in ['throughput', 'tail']:
        differences = []
        for old, new in zip(baseline, candidate):
            assert old['scene'] == new['scene']
            a, b = old['phase_metrics'][mission], new['phase_metrics'][mission]
            differences.append({'scene': old['scene'],
                                'utility_delta': new['phase_utility'][mission]-old['phase_utility'][mission],
                                'critical_violation_delta': b['critical_violation_robot_steps']-a['critical_violation_robot_steps'],
                                'worst_decile_delta': b['worst_decile_mean_service']-a['worst_decile_mean_service']})
        mean_gain = float(np.mean([r['utility_delta'] for r in differences]))
        reasons = []
        if mean_gain <= 0: reasons.append('mean_validation_utility_did_not_improve')
        if any(r['critical_violation_delta'] > 0 for r in differences): reasons.append('a_validation_environment_has_more_critical_violations')
        if any(r['worst_decile_delta'] < -1e-6 for r in differences): reasons.append('a_validation_environment_has_lower_worst_decile_service')
        result[mission] = {'accepted': not reasons, 'reasons': reasons, 'validation_differences': differences,
                           'mean_utility_delta': mean_gain}
    return result


def whole_schedule_recipe(gates, baseline, candidate):
    # Mixing independently validated phases changes the state at the mission
    # transition. Accept the already-tested whole policy only when every
    # phase passes; otherwise use the whole numerical policy.
    accepted = all(gates[mission]['accepted'] for mission in ['throughput', 'tail'])
    branch = candidate if accepted else baseline
    return {'by_mission': {mission: {'selected_strength': branch['selected_strength'],
                                     'temperature': branch['temperature'], 'accepted': accepted}
                           for mission in ['throughput', 'tail']},
            'whole_schedule_accepted': accepted, 'independent_phase_mixing': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, source = args.out, args.run
    if json.loads((source/'status.json').read_text())['state'] != 'completed':
        raise RuntimeError('The original horizon study must finish first')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('state') == 'completed':
        if (sha256(root/'gate_audit.json') != status['report_sha256'] or
                sha256(root/'gate_policy.json') != status['gate_policy_sha256']):
            raise RuntimeError('Completed selector evidence changed')
        provenance = json.loads((root/'provenance.json').read_text())
        if provenance['parent_selection_sha256'] != sha256(source/'selection.json'):
            raise RuntimeError('Completed selector parent changed')
        for relative, digest in provenance['source_sha256'].items():
            if sha256(root/'source_snapshot'/relative) != digest:
                raise RuntimeError('Completed selector executed sources changed')
        print('Selector audit already completed; no rollout restarted'); return
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    repo = Path(__file__).resolve().parents[1]
    paths = [*repo.glob('src/wirelessjev/**/*.py'), Path(__file__), repo/'uv.lock',
             repo/'artifacts/warehouse_v2/studies/wifi_link_curves.json']
    hashes = {str(p.relative_to(repo)): sha256(p) for p in paths}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    provenance_path = root/'provenance.json'
    if provenance_path.exists():
        if json.loads(provenance_path.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError('Selector source changed; use a new audit directory')
    else:
        for p in paths:
            dest = root/'source_snapshot'/p.relative_to(repo)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
        atomic_json(provenance_path, {'fingerprint': fingerprint, 'source_sha256': hashes,
                    'parent_selection_sha256': sha256(source/'selection.json'), 'audit_environments': AUDIT_SEEDS,
                    'designed_after_initial_holdout': True, 'hardware_experiments_executed': False})
    status.update(state='running', stage='closed_loop_validation', pid=os.getpid(), updated_at=now())
    atomic_json(status_path, status)
    original = json.loads((source/'selection.json').read_text())
    readings = json.loads((source/'reader.json').read_text())['readings']
    models = {}
    for seed in [0, 1, 2]:
        for graph in [False, True]:
            name = f'horizon-{seed}-{graph}.pt'
            if sha256(source/'checkpoints'/name) != original['checkpoint_sha256'][name]:
                raise RuntimeError('Parent model changed')
            checkpoint = torch.load(source/'checkpoints'/name, map_location='cpu', weights_only=True)
            model = HorizonModel(**checkpoint['config']); model.load_state_dict(checkpoint['model']); model.eval()
            models[f'{seed}-{graph}'] = model
    def run_cached(scene, method, key, selection):
        path = root/'rollouts'/f'{scene.seed}-{method}-{key}.json'
        if path.exists():
            report = json.loads(path.read_text())
            if report['fingerprint'] != fingerprint:
                raise RuntimeError('Cached selector rollout source mismatch')
            return report
        result = rollout(scene, method, readings, model=models.get(key), selection=selection)
        result.update(model_seed=int(key.split('-')[0]) if key in models else None,
                      model_key=key, fingerprint=fingerprint)
        atomic_json(path, result)
        return result
    try:
        baselines, candidates = [], {key: [] for key in models}
        for seed in CONFIG['validation']:
            world = World(seed)
            baselines.append(run_cached(world, 'numerical_planner', 'baseline', original['baseline']))
            for key in models:
                candidates[key].append(run_cached(world, 'candidate_residual', key, original['models'][key]))
            print(f'Closed-loop selector validation environment {seed}', flush=True)
        gates, recipes = {}, {}
        for key in models:
            gates[key] = candidate_gate(baselines, candidates[key])
            recipes[key] = whole_schedule_recipe(gates[key], original['baseline'], original['models'][key])
        policy_path = root/'gate_policy.json'
        policy = {'models': recipes, 'validation_evidence': gates,
                  'contracts': {k: asdict(compile_contract(text)) for k, text in MISSION_TEXTS.items()},
                  'fit_on': 'six_validation_environments_closed_loop',
                  'whole_schedule_gate': True,
                  'parent_selection_sha256': sha256(source/'selection.json'),
                  'physical_safety_guarantee': False}
        if policy_path.exists():
            previous = json.loads(policy_path.read_text())
            if any(previous[k] != v for k, v in policy.items()):
                raise RuntimeError('Refusing to change the selector after audit access')
        else:
            if any((root/'rollouts'/f'{s}-numerical_planner-baseline.json').exists() for s in AUDIT_SEEDS):
                raise RuntimeError('Audit data existed before selector sealing')
            atomic_json(policy_path, {**policy, 'sealed_at': now()})
        status.update(stage='prospective_audit', updated_at=now()); atomic_json(status_path, status)
        audit = []
        for seed in AUDIT_SEEDS:
            world = World(seed)
            audit.append(run_cached(world, 'numerical_planner', 'baseline', original['baseline']))
            audit.append(run_cached(world, 'budgeted_strongest', 'baseline', original['baseline']))
            for key in models:
                audit.append(run_cached(world, 'guarded_residual', key, recipes[key]))
            print(f'Prospective selector audit environment {seed}', flush=True)
        atomic_json(root/'gate_audit.json', {'records': audit, 'validation_baseline': baselines,
                    'validation_candidates': candidates, 'gate_policy_sha256': sha256(policy_path),
                    'audit_environments': AUDIT_SEEDS, 'model_seeds': [0, 1, 2], 'source': 'simulation',
                    'designed_after_original_holdout': True, 'parent_horizon_run': str(source),
                    'hardware_qualified': False})
    except BaseException as error:
        status.update(state='failed', error=str(error), traceback=traceback.format_exc(), updated_at=now())
        atomic_json(status_path, status); raise
    status.update(state='completed', stage=None, completed_at=now(), updated_at=now(),
                  report_sha256=sha256(root/'gate_audit.json'), gate_policy_sha256=sha256(policy_path))
    atomic_json(status_path, status)


if __name__ == '__main__':
    main()
