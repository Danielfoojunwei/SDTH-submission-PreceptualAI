"""Full-cross model/environment closed-loop evaluation on fresh scenes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import time

import numpy as np
import torch

from ..warehouse_v2.metrics import robot_metrics
from .belief import RadioBelief
from .learning import blend
from .model import tensors, prior_prediction
from .planning import (Contract, MISSION_TEXTS, compile_contract, budgeted_signal,
                       choose, feasible, values)
from .world import World, observe


class Predictor:
    def __init__(self, observation, belief, model=None, *, strength=0.):
        self.observation, self.belief = observation, belief
        self.model, self.strength = model, strength
        self.tensor_observation = tensors(observation) if model is not None and strength else None

    def __call__(self, plans):
        prior = self.belief.forecast(plans)
        base = prior_prediction(prior, self.observation)
        if self.model is None or not self.strength:
            return base
        with torch.inference_mode():
            result = self.model(self.tensor_observation, torch.as_tensor(plans), torch.as_tensor(prior))
        prediction = {k: v.numpy() for k, v in result.items()}
        return blend(base, prediction, self.strength)


def rollout(world, method, readings, *, model=None, selection=None, outage_s=4., duration_s=600):
    current = world.initial.copy()
    previous_service = np.zeros(len(current), dtype='float32')
    since, outage = np.full(len(current), 30.), np.zeros(len(current))
    history, services, failures, moves, decisions = [], [], [], [], []
    scores, delivered, phase_scores = [], [], defaultdict(list)
    last_decision, last_mission = None, None
    selection = selection or {'selected_strength': 0., 'temperature': 1.}
    for tick in range(int(duration_s/5)):
        observation = observe(world.frames[tick], current, tick, previous_service=previous_service,
                              since_switch=since, history=history)
        key = 'throughput' if tick < 60 else 'tail'
        gold_contract = compile_contract(MISSION_TEXTS[key])
        contract = Contract(**readings[key]['contract']) if readings[key].get('contract') else None
        action = current.copy()
        due = tick >= 3 and (last_decision is None or tick-last_decision >= 6 or key != last_mission)
        if due:
            start = time.perf_counter()
            receipt = {'mission': key, 'time_s': tick*5, 'method': method}
            if contract is None:
                receipt.update(status='semantic_abstention', moves=0)
            elif method == 'stay':
                receipt.update(status='selected', moves=0)
            elif method in ['budgeted_strongest', 'unconstrained_strongest']:
                limit = contract.max_moves if method == 'budgeted_strongest' else len(current)
                action = budgeted_signal(observation, limit)
                receipt.update(status='selected', moves=int((action != current).sum()))
            else:
                belief = RadioBelief(observation)
                active_selection = selection.get('by_mission', {}).get(key, selection)
                predictor = Predictor(observation, belief, model, strength=active_selection['selected_strength'])
                action, result = choose(observation, contract, belief, predictor, outage_s=outage_s,
                                        temperature=active_selection['temperature'])
                receipt.update(result)
                receipt['selected_branch'] = 'residual' if active_selection['selected_strength'] else 'numerical'
            budget_ok = feasible(action[None], observation, gold_contract)[0]
            if method != 'unconstrained_strongest' and not budget_ok:
                raise RuntimeError('Budget/visibility guard failed before simulated actuation')
            receipt.update(contract_satisfied=bool(budget_ok),
                           compute_ms=(time.perf_counter()-start)*1000)
            decisions.append(receipt)
            last_decision, last_mission = tick, key
        moved = action != current
        outage = np.where(moved, outage_s, outage)
        service, failure, outage = world.interval(tick, action, current, outage_remaining=outage)
        score = float(values(gold_contract, observation, service[None, None], failure[None, None])[0])
        scores.append(score); phase_scores[key].append(score)
        demand = observation['critical']+observation['bulk']
        delivered.append(float((service*demand).sum()/demand.sum()))
        services.append(service); failures.append(failure); moves.append(moved)
        previous_service = service
        since = np.where(moved, 0., since)+5.
        current = action
    metrics = robot_metrics(np.stack(services), np.stack(failures), np.stack(moves),
                            robot_ids=[f'amr-{i+1:03d}' for i in range(len(current))])
    compute = [r['compute_ms'] for r in decisions]
    phase_metrics = {}
    for name, begin, end in [('throughput', 0, min(60, len(services))), ('tail', 60, len(services))]:
        if end > begin:
            phase_metrics[name] = robot_metrics(np.stack(services[begin:end]), np.stack(failures[begin:end]),
                                                np.stack(moves[begin:end]))
    return {'source': 'multi_step_warehouse_simulation', 'scene': world.seed, 'method': method,
            'duration_s': duration_s, 'robots': len(current), 'outage_s': outage_s,
            'mean_mission_utility': float(np.mean(scores)),
            'phase_utility': {k: float(np.mean(v)) for k, v in phase_scores.items()}, 'phase_metrics': phase_metrics,
            'mean_delivered_demand_fraction': float(np.mean(delivered)), 'metrics': metrics,
            'decisions': decisions, 'constraint_violations': sum(not r['contract_satisfied'] for r in decisions),
            'semantic_abstentions': sum(r['status'] == 'semantic_abstention' for r in decisions),
            'compute_ms_mean': float(np.mean(compute)), 'compute_ms_p95': float(np.quantile(compute, .95)),
            'residual_selection': selection,
            'timing_includes': ['cached_mission_lookup', 'belief_construction_for_planners', 'candidate_generation',
                               'numerical_and_neural_forecasting', 'local_search', 'feasibility_guard'],
            'timing_excludes': ['physical_sensor_collection', 'first_use_language_model_inference', 'network_actuation'],
            'limits': ['Simulation only; critical-rate proxy is not a packet-deadline guarantee',
                       'Scan interruption and actuation failures are not modelled',
                       'Longest violation duration is quantised at five-second intervals',
                       'The unconstrained strongest-scan reference can violate the declared switch budget']}
