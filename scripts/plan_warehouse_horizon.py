"""Validate an observation and produce a local warehouse plan; never actuate.

Use --mission for actual pinned d3 interpretation of the declared controlled
language, or --contract for an explicitly supplied typed program. The latter
is labelled as caller-provided input, never as model inference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from wirelessjev.twin.radio import CHANNELS
from wirelessjev.warehouse_horizon.belief import RadioBelief
from wirelessjev.warehouse_horizon.evaluation import Predictor
from wirelessjev.warehouse_horizon.model import HorizonModel
from wirelessjev.warehouse_horizon.planning import (Contract, CRITERIA, compile_contract,
                                                   choose, feasible, charge)
from wirelessjev.warehouse_v2.experiment import atomic_json, sha256

SCHEMA = 'wirelessjev.horizon_observation.v1'
NUMBER = r'\d+(?:\.\d+)?'
CONTROLLED = re.compile(
    rf'reconnect at most \d+ robots per decision;\s*'
    rf'(?:maximise total data delivered|maximise average service of the worst-served {NUMBER}% of robots)\.\s*'
    rf'apply a penalty of {NUMBER} per fraction of robots with critical-link failures\.\s*', re.I)


def validate_observation(document, *, now=None):
    if document.get('schema') != SCHEMA:
        raise ValueError('Unsupported observation schema')
    source = document.get('source')
    if source not in ['simulation', 'hardware']:
        raise ValueError('Input source must be explicit')
    if document.get('warehouse_size_m') != [120, 80] or document.get('offsets_s') != [5, 10, 15, 20, 25, 30]:
        raise ValueError('This model requires the declared warehouse dimensions and six offsets')
    raw = document.get('observation')
    required = {'history', 'past_xy', 'robot', 'ap', 'route', 'current', 'critical', 'bulk', 'video', 'tick'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError('Observation fields differ from the trained schema')
    obs = {key: np.asarray(value) for key, value in raw.items()}
    n = len(document.get('robot_ids', [])); m = len(document.get('ap_ids', []))
    if n < 1 or n > 200 or m < 1 or m > 128:
        raise ValueError('Unsupported fleet or AP count')
    shapes = {'history': (4, n, m, 3), 'past_xy': (4, n, 2), 'robot': (n, 8),
              'ap': (m, 6), 'route': (6, n, 2), 'current': (n,), 'critical': (n,),
              'bulk': (n,), 'video': (n,), 'tick': ()}
    for key, shape in shapes.items():
        if obs[key].shape != shape or not np.issubdtype(obs[key].dtype, np.number) and obs[key].dtype != np.bool_:
            raise ValueError(f'Invalid observation shape/type: {key}')
        if not np.isfinite(obs[key]).all():
            raise ValueError(f'Nonfinite observation field: {key}')
    if not np.issubdtype(obs['current'].dtype, np.integer) or not np.issubdtype(obs['tick'].dtype, np.integer):
        raise ValueError('Association indices and tick must be integers')
    if (obs['current'] < 0).any() or (obs['current'] >= m).any() or obs['tick'] < 0:
        raise ValueError('Invalid association or tick')
    for key in ['robot_ids', 'ap_ids']:
        ids = document[key]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError('Node identities must be nonempty and unique')
    if source == 'hardware':
        if any(not re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', value) for value in document['ap_ids']):
            raise ValueError('Hardware AP identities must be actual BSSIDs')
        if len(set(v.lower() for v in document['ap_ids'])) != m:
            raise ValueError('Hardware BSSIDs differ only in case')
        observed_at = datetime.fromisoformat(document.get('observed_at', '').replace('Z', '+00:00'))
        if observed_at.tzinfo is None:
            raise ValueError('Hardware observation timestamp needs a timezone')
        age = ((now or datetime.now(timezone.utc))-observed_at).total_seconds()
        if age < -5 or age > 30:
            raise ValueError('Hardware observation is stale or from the future')
    for coordinates in [obs['past_xy'], obs['route'], obs['robot'][:, :2], obs['ap'][:, :2]]:
        if ((coordinates < 0) | (coordinates > 1)).any():
            raise ValueError('Coordinates must use the trained warehouse normalisation')
    if (obs['critical'] <= 0).any() or (obs['bulk'] < 0).any():
        raise ValueError('This warehouse profile requires positive critical demand and nonnegative bulk demand')
    if not np.isin(obs['video'], [0, 1]).all():
        raise ValueError('Video flags must be binary')
    if ((obs['robot'][:, 4] < 0) | (obs['robot'][:, 4] > 1)).any():
        raise ValueError('Critical floor ratios must be fractions')
    if ((obs['robot'][:, 6:] < 0) | (obs['robot'][:, 6:] > 1)).any():
        raise ValueError('Delivery fractions and normalised switch ages must be bounded')
    if not np.allclose(obs['robot'][:, 2], np.log1p(obs['critical'])/3., atol=1e-5) or not np.allclose(obs['robot'][:, 3], np.log1p(obs['bulk'])/3., atol=1e-5):
        raise ValueError('Robot features and declared demands have inconsistent units')
    if not np.allclose(obs['robot'][:, :2], obs['past_xy'][-1], atol=1e-5) or not np.array_equal(obs['robot'][:, 5], obs['video']):
        raise ValueError('Robot features disagree with pose/video metadata')
    if not np.isin(obs['ap'][:, 2], [0, 1]).all() or ((obs['ap'][:, 4] < 0) | (obs['ap'][:, 4] > 1)).any():
        raise ValueError('Invalid band or busy-time values')
    for ap in obs['ap']:
        band = '5g' if ap[2] else '2g'
        channel = ap[3]*165.
        if int(round(channel)) not in CHANNELS[band] or abs(channel-round(channel)) > 1e-3 or not 0 <= ap[5] <= 1:
            raise ValueError('Invalid channel or normalised AP power')
    for scan in obs['history']:
        if not np.isin(scan[..., 1], [0, 1]).all():
            raise ValueError('Scan masks must be binary')
        for row in scan:
            heard = row[:, 1] > 0
            if heard.sum() > 8 or (row[~heard, 0] != 0).any():
                raise ValueError('Scan cap or missing-value representation is invalid')
            if (row[heard, 0] < -1/3-1e-5).any():
                raise ValueError('A reported scan value is below the declared floor')
            bound = row[heard, 0].min() if heard.sum() == 8 else -1/3
            if not np.allclose(row[:, 2], bound, atol=1e-5):
                raise ValueError('Scan censoring bounds are inconsistent with the result cap')
    for key in required-{'current', 'tick', 'video'}:
        obs[key] = obs[key].astype('float32')
    obs['current'] = obs['current'].astype('int64')
    return obs


def validate_controlled_text(text):
    if not CONTROLLED.fullmatch(text.strip()):
        raise ValueError('Mission is outside the declared complete grammar; no clause was discarded')
    return compile_contract(text)


def load_forecaster(root, key):
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'completed':
        raise ValueError('Use a completed and archived research run')
    selection_path = root/'selection.json'
    if sha256(selection_path) != status['reports']['selection.json']:
        raise ValueError('Selection receipt mismatch')
    selection = json.loads(selection_path.read_text())
    if key == 'numerical':
        return None, selection['baseline'], {'mode': 'numerical_belief', 'selection_sha256': sha256(selection_path)}
    if not re.fullmatch(r'[012]-(?:True|False)', key):
        raise ValueError('Unknown model key')
    checkpoint_path = root/'checkpoints'/f'horizon-{key}.pt'
    digest = sha256(checkpoint_path)
    if selection['checkpoint_sha256'][checkpoint_path.name] != digest:
        raise ValueError('Checkpoint hash does not match sealed selection')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if any(not torch.isfinite(t).all() for t in checkpoint['model'].values() if t.is_floating_point()):
        raise ValueError('Checkpoint has nonfinite weights')
    model = HorizonModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['model']); model.eval()
    return model, selection['models'][key], {'mode': 'offline_validated_residual', 'model_key': key,
                                           'checkpoint_sha256': digest, 'selection_sha256': sha256(selection_path)}


def guarded_recipe(root, key, contract, observation, *, research=False, source='simulation'):
    """Fail closed to the numerical recipe if closed-loop evidence is absent.

    The optional research path is deliberately limited to simulated inputs.
    An empirical acceptance check is never a physical safety certificate.
    """
    if research and source != 'simulation':
        raise ValueError('Unguarded research residuals accept simulation input only')
    parent = json.loads((root/'selection.json').read_text())
    if key == 'numerical':
        return parent['baseline'], {'mode': 'numerical_belief', 'model_executed': False}
    if research:
        return parent['models'][key], {'mode': 'research_residual', 'model_executed': True,
                                      'closed_loop_qualified': False}
    policy_path, status_path = root/'selector/gate_policy.json', root/'selector/status.json'
    reason = 'closed_loop_validation_evidence_missing'
    policy = None
    if policy_path.exists() and status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('state') != 'completed' or sha256(policy_path) != status['gate_policy_sha256']:
            raise ValueError('Closed-loop selector receipt mismatch')
        policy = json.loads(policy_path.read_text())
        if policy['parent_selection_sha256'] != sha256(root/'selection.json'):
            raise ValueError('Closed-loop selector belongs to different model selection')
        recipe = policy['models'][key]
        if not recipe['whole_schedule_accepted']:
            reason = 'closed_loop_validation_rejected'
        elif policy['contracts'].get(contract.objective) != asdict(contract):
            reason = 'contract_outside_closed_loop_validation'
        elif len(observation['current']) != 100 or len(observation['ap']) != 32:
            reason = 'fleet_shape_outside_closed_loop_validation'
        else:
            return recipe['by_mission'][contract.objective], {
                'mode': 'closed_loop_validated_residual', 'model_executed': True,
                'gate_policy_sha256': sha256(policy_path), 'physical_safety_guarantee': False}
    receipt = {'mode': 'numerical_fallback', 'model_executed': False, 'reason': reason,
               'requested_model_key': key, 'physical_safety_guarantee': False}
    if policy is not None:
        receipt['gate_policy_sha256'] = sha256(policy_path)
    return parent['baseline'], receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observation', type=Path, required=True)
    mission = parser.add_mutually_exclusive_group(required=True)
    mission.add_argument('--mission')
    mission.add_argument('--contract', type=Path)
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/warehouse_horizon')
    parser.add_argument('--model-key', default='numerical')
    parser.add_argument('--research-residual', action='store_true',
                        help='Explicit simulation-only use of a residual without closed-loop acceptance')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    document = json.loads(args.observation.read_text())
    obs = validate_observation(document)
    if args.mission:
        validate_controlled_text(args.mission)
        from wirelessjev.policy import WirelessJEV, MODEL_REPO, MODEL_REVISION
        engine = WirelessJEV(device='cpu')
        objective, probabilities, duration = engine.choose('Operator mission (data):\n'+args.mission,
            'Select the primary objective; switch budgets and critical penalties are compatible supplementary clauses.', CRITERIA)
        contract = compile_contract(args.mission, objective)
        reading = {'reader': 'actual_d3', 'model_executed': True,
                   'model': {'repo': MODEL_REPO, 'revision': MODEL_REVISION},
                   'answer_probabilities': probabilities, 'interpretation_ms': duration}
        reading['probabilities_are_physical_risk'] = False
    else:
        contract = Contract(**json.loads(args.contract.read_text()))
        reading = {'reader': 'caller_provided_typed_contract', 'model_executed': False}
    if contract.max_moves > 12:
        raise ValueError('This research proposal interface supports budgets up to twelve; this is not a hardware switching-rate limit')
    model, selection, model_receipt = load_forecaster(args.run, args.model_key)
    selection, guard_receipt = guarded_recipe(args.run, args.model_key, contract, obs,
        research=args.research_residual, source=document['source'])
    model_receipt.update(guard_receipt)
    if not model_receipt['model_executed']:
        model = None
    start = time.perf_counter()
    belief = RadioBelief(obs)
    predictor = Predictor(obs, belief, model, strength=selection['selected_strength'])
    action, receipt = choose(obs, contract, belief, predictor, temperature=selection['temperature'])
    if not feasible(action[None], obs, contract)[0]:
        raise RuntimeError('Independent proposal guard failed')
    forecast = predictor(action[None])
    service, risk = charge(forecast, action[None], obs, temperature=selection['temperature'])
    assignments = {robot: document['ap_ids'][int(ap)] for robot, ap in zip(document['robot_ids'], action)}
    result = {'schema': 'wirelessjev.horizon_plan.v1', 'proposal_only': True, 'actuation_performed': False,
              'input_source_self_reported': document['source'], 'input_origin_authenticated': False,
              'observation_sha256': sha256(args.observation), 'contract': asdict(contract),
              'reader': reading, 'forecaster': model_receipt, 'selection_strength': selection['selected_strength'],
              'assignments': assignments, 'decision': receipt, 'local_compute_ms': (time.perf_counter()-start)*1000,
              'predicted_service_fraction_by_offset': service[0].tolist(),
              'critical_rate_violation_probability_by_offset': risk[0].tolist(),
              'offsets_s': [5, 10, 15, 20, 25, 30], 'assumed_switch_interruption_s': 4.,
              'model_validated_on': 'simulation_only', 'physical_safety_guarantee': False}
    result['training_fleet_size'] = 100
    result['input_fleet_size'] = len(obs['current'])
    atomic_json(args.out, result)
    print(f'Local proposal written: {receipt["moves"]} switches; no car command sent')


if __name__ == '__main__':
    main()
