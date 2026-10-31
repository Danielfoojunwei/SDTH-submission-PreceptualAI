"""Long-rollout intervention data with sealed environment splits and receipts."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..warehouse_v2.experiment import atomic_json, sha256
from .belief import RadioBelief
from .planning import Contract, candidate_pool
from .world import World, observe

CONFIG = {'source': 'simulation', 'schema': 'warehouse.route_horizon.v1',
          'train': list(range(140000, 140018)), 'validation': list(range(141000, 141006)),
          'test': list(range(142000, 142003)), 'model_seeds': [0, 1, 2],
          'episode_s': 600, 'horizon_s': 30, 'step_s': 5, 'scan_sigma_db': 3,
          'max_training_candidates': 12, 'max_moves': 12, 'assumed_switch_outage_s': 4,
          'primary_scan_interruption_modelled': False, 'epochs': 14}


def diverse_candidates(plans, maximum=12):
    if len(plans) <= maximum:
        return plans
    selected = [0, 1]
    distance = (plans[:, None] != plans[None]).sum(2)
    while len(selected) < maximum:
        closest = distance[:, selected].min(1).astype(float)
        closest[selected] = -1
        selected.append(int(closest.argmax()))
    return plans[selected]


def episode(seed, *, domain='train'):
    world = World(seed, steps=126, domain=domain)
    current = world.initial.copy()
    history, examples = [], []
    previous_service = np.zeros(len(current), dtype='float32')
    since = np.full(len(current), 30., dtype='float32')
    outage = np.zeros(len(current))
    rng = np.random.default_rng(seed+7_000_000)
    for tick in range(120):
        observation = observe(world.frames[tick], current, tick, previous_service=previous_service,
                              since_switch=since, history=history)
        chosen = current.copy()
        if tick >= 3 and (tick-3) % 6 == 0:
            belief = RadioBelief(observation)
            contract = Contract('throughput', CONFIG['max_moves'], 1.)
            candidates, _ = candidate_pool(observation, belief, contract)
            candidates = diverse_candidates(candidates, CONFIG['max_training_candidates'])
            # Every plan is an intervention into the same exogenous future.
            target = world.outcomes(tick, candidates, current)
            prior = belief.forecast(candidates)
            examples.append({'observation': observation, 'plans': candidates, 'prior': prior,
                             'target': target, 'scene': seed, 'tick': tick})
            # Exercise full/partial reassociations during long episodes.
            chosen = candidates[int(rng.integers(len(candidates)))].copy()
        moved = chosen != current
        outage = np.where(moved, 4., outage)
        previous_service, failure, outage = world.interval(tick, chosen, current, outage_remaining=outage)
        since = np.where(moved, 0., since)+5.
        current = chosen
    return examples


def save_examples(path, examples, fingerprint):
    arrays = {'count': np.array(len(examples)), 'fingerprint': np.array(fingerprint)}
    for i, sample in enumerate(examples):
        for key, value in sample['observation'].items():
            arrays[f'{i}_obs_{key}'] = value
        for key, value in sample['target'].items():
            arrays[f'{i}_target_{key}'] = value
        arrays[f'{i}_plans'] = sample['plans']
        arrays[f'{i}_prior'] = sample['prior']
    np.savez_compressed(path, **arrays)


def load_examples(path, fingerprint, seed):
    examples = []
    with np.load(path, allow_pickle=False) as data:
        if str(data['fingerprint']) != fingerprint:
            raise RuntimeError('Dataset cache source/configuration mismatch')
        for i in range(int(data['count'])):
            observation = {key.removeprefix(f'{i}_obs_'): data[key].copy() for key in data.files if key.startswith(f'{i}_obs_')}
            target = {key.removeprefix(f'{i}_target_'): data[key].copy() for key in data.files if key.startswith(f'{i}_target_')}
            examples.append({'observation': observation, 'plans': data[f'{i}_plans'].copy(),
                             'prior': data[f'{i}_prior'].copy(), 'target': target,
                             'scene': seed, 'tick': int(observation['tick'])})
    return examples


def dataset(root, fingerprint, *, splits=('train', 'validation')):
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in splits:
        result[split] = []
        for seed in CONFIG[split]:
            path = root/f'{seed}.npz'
            if path.exists():
                examples = load_examples(path, fingerprint, seed)
            else:
                print(f'Generating full 600 s route interventions: {split} scene={seed}', flush=True)
                examples = episode(seed)
                save_examples(path, examples, fingerprint)
            result[split].extend(examples)
    return result
