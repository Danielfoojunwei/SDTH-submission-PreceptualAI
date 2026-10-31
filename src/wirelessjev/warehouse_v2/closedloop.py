"""Actual closed-loop execution inside the declared warehouse simulator."""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..envlearn import external_noise, leak_matrix
from ..fastplan import BatchEvaluator
from .data import make_environment, measure_scan, observed_inputs, observed_candidates
from .experiment import atomic_json, make_outcomes
from .metrics import robot_metrics
from .model import tensor_inputs
from .studies import load_model


def rollout(seed, policy, model=None, *, duration_s=600):
    scene, spec = make_environment(seed, n=100)
    noise_rng = np.random.default_rng(seed+10000)
    plan_rng = np.random.default_rng(seed+20000)
    label_rng = np.random.default_rng(seed+30000)
    history, service, failures, moves, decisions, times = [], [], [], [], [], []
    for step in range(int(duration_s/5)):
        history.append(measure_scan(scene, noise_rng))
        history = history[-4:]
        inp, demand, video = observed_inputs(scene, spec, history, noise_rng)
        current = inp['current']
        chosen = current.copy()
        if len(history) == 4 and (step-3) % 6 == 0:
            start = time.perf_counter()
            plans = observed_candidates(inp['history'], current, demand, plan_rng)
            if policy == 'stay':
                choice = 0
            elif policy == 'strongest_scan':
                choice = 1
            elif policy == 'legacy_hysteresis':
                signal = inp['history'][-1]
                strongest = plans[1]
                selected = signal[np.arange(100), strongest, 0]
                old = signal[np.arange(100), current, 0]
                heard = signal[np.arange(100), current, 1] > 0
                change = (~heard) | ((selected-old > 5/30.) & (old < 5/30.))
                chosen = np.where(change, strongest, current)
                choice = None
            elif policy == 'observed_load_balance':
                choice = 2
            else:
                with torch.inference_mode():
                    output = model({k: torch.as_tensor(v) for k, v in inp.items()}, torch.tensor(plans))
                    value = (output['service'].numpy()*demand).sum(1)
                    choice = int(value.argmax())
            if choice is not None:
                chosen = plans[choice]
            elapsed = (time.perf_counter()-start)*1000
            times.append(elapsed)
            decisions.append({'time_s': step*5, 'choice': choice, 'moves': int((chosen != current).sum()), 'compute_ms': elapsed})
        future = scene.advance(5.)
        ev = BatchEvaluator(future, [])
        ev.client_itf = True
        ev.ext_noise_mw = external_noise(future, spec)
        ev.leak = leak_matrix(future, spec.leak_db)
        blocked = label_rng.random(ev.snaps[0].ul.shape) < spec.blockage_p
        ev.snaps[0].ul -= blocked*spec.blockage_db
        terms = ev._snapshot_terms(chosen[None], ev.snaps[0], True, 5.)
        moved = chosen != current
        critical = terms['cf'][0]*np.maximum(0, 1-moved*.8)
        violation = ((ev.cf_dem*critical[:, None] < ev.cf_floor-1e-9) & (ev.cf_dem > 0)).any(1)
        service.append(terms['sat'][0]); failures.append(violation); moves.append(moved)
        scene = future.with_associations({c.car_id: future.aps[j].ap_id for c, j in zip(future.cars, chosen)})
    metrics = robot_metrics(np.stack(service), np.stack(failures), np.stack(moves),
                            robot_ids=[c.car_id for c in scene.cars])
    return {'source': 'closed_loop_warehouse_simulation', 'seed': seed, 'policy': policy,
            'duration_s': duration_s, 'robots': 100, 'decision_interval_s': 30,
            'assumed_switch_interruption_s': 4., 'metrics': metrics, 'decisions': decisions,
            'compute_ms_mean': float(np.mean(times)), 'compute_ms_p95': float(np.quantile(times, .95)),
            'limits': ['Sensor collection is simulated and scan interruption is not charged in this rollout',
                       'Gaussian scan noise and stochastic blockage; no real vehicle or network transport',
                       'Learned models predict 5 s outcomes while decisions are held for 30 s; horizon mismatch is explicit',
                       'Objective is fleet delivered data, not the original language-mission schedule']}


def study(root, seeds):
    rows = []
    for seed in seeds:
        for policy in ['stay', 'legacy_hysteresis', 'observed_load_balance', 'instant_mlp', 'temporal_graph']:
            path = root/'cache'/f'closedloop-{seed}-{policy}.json'
            if path.exists():
                row = json.loads(path.read_text())
            else:
                model = load_model(root, seed, policy)[0] if policy in ['instant_mlp', 'temporal_graph'] else None
                row = rollout(130000+seed, policy, model)
                atomic_json(path, row)
            rows.append(row)
            print(f'Closed loop seed={seed} {policy}: service={row["metrics"]["mean_service"]:.4f}', flush=True)
    return {'source': 'closed_loop_warehouse_simulation', 'records': rows,
            'limits': ['Each learned model seed is paired with one rollout seed; this does not separate both variance sources',
                       'Per-robot results describe simulated robots, not the physical fleet']}
