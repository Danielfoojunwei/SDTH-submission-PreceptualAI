"""Composable typed missions, partial moves and observation-only local search."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import time

import numpy as np

from .world import interruption_keep

MISSION_TEXTS = {
    'throughput': 'Reconnect at most 12 robots per decision; maximise total data delivered. Apply a penalty of 1 per fraction of robots with critical-link failures.',
    'tail': 'Reconnect at most 12 robots per decision; maximise average service of the worst-served 10% of robots. Apply a penalty of 1 per fraction of robots with critical-link failures.',
}
CRITERIA = {
    'throughput': 'Maximise total fleet data, possibly with a switch budget and a critical-link penalty',
    'tail': 'Maximise average service of a specified lowest-served percentage of robots, possibly with budgets and penalties',
    'unsupported': 'Neither supported objective, conflicting objective, or ambiguous request',
}


@dataclass(frozen=True)
class Contract:
    objective: str
    max_moves: int
    critical_penalty: float
    tail_fraction: float = .1

    def __post_init__(self):
        if self.objective not in ('throughput', 'tail') or type(self.max_moves) is not int or self.max_moves < 0:
            raise ValueError('Unsupported objective or invalid hard move budget')
        if not np.isfinite(self.critical_penalty) or self.critical_penalty < 0 or not 0 < self.tail_fraction <= 1:
            raise ValueError('Invalid penalty or tail fraction')


def compile_contract(text, objective=None):
    text = text.lower()
    limits = re.findall(r'at most\s+(\d+)\s+robots', text)
    penalties = re.findall(r'penalty of\s+(\d+(?:\.\d+)?)\s+per fraction', text)
    if len(limits) != 1 or len(penalties) != 1:
        raise ValueError('An explicit, unambiguous budget and critical penalty are required')
    detected = 'tail' if re.search(r'worst.served\s+\d+(?:\.\d+)?%', text) else (
        'throughput' if 'maximise total data' in text else None)
    if detected is None or (objective is not None and objective != detected):
        raise ValueError('Unsupported or conflicting typed objective')
    fractions = re.findall(r'worst.served\s+(\d+(?:\.\d+)?)%', text)
    if detected == 'tail' and len(fractions) != 1:
        raise ValueError('Exactly one tail fraction is required')
    return Contract(detected, int(limits[0]), float(penalties[0]),
                    float(fractions[0])/100 if fractions else .1)


def read_contracts(engine):
    result = {}
    for key, text in MISSION_TEXTS.items():
        start = time.perf_counter()
        choice, probabilities, _ = engine.choose(
            'Operator mission (data):\n'+text,
            'Choose the primary objective. Switch budgets and critical-link penalties are compatible supplementary clauses, not conflicting objectives.',
            CRITERIA)
        row = {'text': text, 'choice': choice, 'probabilities': probabilities,
               'model_executed': True, 'latency_ms': (time.perf_counter()-start)*1000,
               'probabilities_are_physical_risk': False}
        try:
            row['contract'] = asdict(compile_contract(text, choice))
        except ValueError as error:
            row['contract'] = None
            row['abstention'] = str(error)
        result[key] = row
    return result


def values(contract, observation, service, failure):
    service, failure = np.asarray(service), np.asarray(failure)
    if service.ndim != 3 or failure.shape != service.shape or not np.isfinite(service).all() or not np.isfinite(failure).all():
        raise ValueError('Finite [plan, time, robot] outcomes required')
    if np.any((service < 0) | (service > 1)) or np.any((failure < 0) | (failure > 1)):
        raise ValueError('Invalid service fraction or failure probability')
    demand = observation['critical']+observation['bulk']
    if contract.objective == 'throughput':
        objective = ((service*demand[None, None]).sum(2)/max(float(demand.sum()), 1e-9)).mean(1)
    else:
        count = max(1, int(math.ceil(contract.tail_fraction*len(demand))))
        objective = np.sort(service, axis=2)[:, :, :count].mean((1, 2))
    return objective-contract.critical_penalty*failure.mean((1, 2))


def feasible(plans, observation, contract):
    A = np.asarray(plans)
    n, m = len(observation['current']), len(observation['ap'])
    if A.ndim != 2 or A.shape[1] != n or not np.issubdtype(A.dtype, np.integer) or (A < 0).any() or (A >= m).any():
        raise ValueError('Malformed candidate assignment')
    moved = A != observation['current'][None]
    heard = observation['history'][-1, np.arange(n)[None], A, 1] > 0
    return (moved.sum(1) <= contract.max_moves) & (~moved | heard).all(1)


def charge(prediction, plans, obs, *, outage_s=4., temperature=1., strength=1.):
    from scipy.special import expit
    raw = prediction['service_raw']
    failure = expit(prediction['failure_logits']/temperature)
    keep = interruption_keep(plans != obs['current'][None], outage_s=outage_s)
    # If even perfect critical delivery cannot meet a declared flow floor
    # during the known interruption, the simulated violation is certain.
    impossible = keep < obs['robot'][None, None, :, 4]-1e-9
    return raw*keep, np.where(impossible, 1., failure)


def budgeted_signal(observation, limit):
    current = observation['current']
    scan = observation['history'][-1]
    signal = np.where(scan[..., 1] > 0, scan[..., 0]*30.-80., -100.)
    strong = signal.argmax(1)
    strong[~(scan[..., 1] > 0).any(1)] = current[~(scan[..., 1] > 0).any(1)]
    gain = signal[np.arange(len(current)), strong]-signal[np.arange(len(current)), current]
    changed = np.flatnonzero(strong != current)
    changed = changed[np.argsort(-gain[changed], kind='stable')]
    action = current.copy()
    action[changed[:limit]] = strong[changed[:limit]]
    return action


def candidate_pool(obs, belief, contract, *, include_training_budgets=True):
    current = obs['current']
    n = len(current)
    heard = obs['history'][-1, :, :, 1] > 0
    visible = [np.union1d(np.flatnonzero(row), [current[i]]) for i, row in enumerate(heard)]
    measured = np.where(heard, obs['history'][-1, :, :, 0]*30.-80., belief.current_mean)
    strong, route = current.copy(), current.copy()
    future = belief.mean.mean(0)
    for i, valid in enumerate(visible):
        strong[i] = valid[np.argmax(measured[i, valid])]
        route[i] = valid[np.argmax(future[i, valid])]
    demand = obs['critical']+obs['bulk']
    # A public-observation heuristic, not a hidden simulator outcome.
    balanced = current.copy()
    load = np.zeros(len(obs['ap']))
    for i in np.argsort(-demand):
        valid = visible[i]
        score = future[i, valid] - 2.*load[valid] - 10.*obs['ap'][valid, 4]
        j = int(valid[np.argmax(score)])
        balanced[i] = j
        load[j] += demand[i]
    pool = [current.copy()]
    limits = sorted(set([min(contract.max_moves, x) for x in [2, 5, 10, 20]]+[min(n, contract.max_moves)]))
    for proposed in [strong, route, balanced]:
        gain = future[np.arange(n), proposed]-future[np.arange(n), current]
        for ordering, order in enumerate([gain, gain*(1.+demand/np.maximum(demand.mean(), 1e-6)),
                                         gain*(2.-obs['robot'][:, 6])]):
            changed = np.flatnonzero(proposed != current)
            changed = changed[np.argsort(-order[changed], kind='stable')]
            for limit in (limits if ordering == 0 else [contract.max_moves]):
                plan = current.copy()
                plan[changed[:limit]] = proposed[changed[:limit]]
                pool.append(plan)
    # Keep the strongest-scan endpoint itself as a budgeted baseline.
    strongest_budgeted = budgeted_signal(obs, contract.max_moves)
    pool.insert(1, strongest_budgeted)
    unique, seen = [], set()
    for plan in pool:
        key = plan.tobytes()
        if key not in seen:
            seen.add(key); unique.append(plan)
    plans = np.stack(unique)
    if not feasible(plans, obs, contract).all():
        raise RuntimeError('Candidate generator violated its declared contract')
    return plans, strongest_budgeted


def neighbours(incumbent, observation, belief, contract):
    current = observation['current']
    future = belief.mean.mean(0)
    current_signal = future[np.arange(len(current)), current]
    urgency = (1.-observation['robot'][:, 6])*20.-current_signal/10.
    robots = np.argsort(-urgency, kind='stable')[:8]
    moved = np.flatnonzero(incumbent != current)
    benefit = future[np.arange(len(current)), incumbent]-current_signal
    remove = int(moved[np.argmin(benefit[moved])]) if len(moved) else None
    pool = [incumbent.copy()]
    for i in robots:
        valid = np.flatnonzero(observation['history'][-1, i, :, 1] > 0)
        valid = valid[np.argsort(-future[i, valid])[:2]]
        for target in valid:
            plan = incumbent.copy()
            if plan[i] == current[i] and target != current[i] and len(moved) >= contract.max_moves:
                if remove is None:
                    continue
                plan[remove] = current[remove]
            plan[i] = target
            pool.append(plan)
    A = np.stack(pool)
    return A[feasible(A, observation, contract)]


def choose(observation, contract, belief, forecast, *, outage_s=4., temperature=1., rounds=2):
    start = time.perf_counter()
    plans, _ = candidate_pool(observation, belief, contract)
    evaluated = 0
    best = observation['current'].copy()
    best_score = -np.inf
    for iteration in range(rounds+1):
        prediction = forecast(plans)
        service, failure = charge(prediction, plans, observation, outage_s=outage_s, temperature=temperature)
        utility = values(contract, observation, service, failure)
        allowed = feasible(plans, observation, contract)
        if not allowed.any():
            return observation['current'].copy(), {'status': 'deferred', 'candidates': evaluated}
        index = int(np.argmax(np.where(allowed, utility, -np.inf)))
        evaluated += len(plans)
        if utility[index] > best_score+1e-10:
            best, best_score = plans[index].copy(), float(utility[index])
        if iteration < rounds:
            plans = neighbours(best, observation, belief, contract)
    if not feasible(best[None], observation, contract)[0]:
        raise RuntimeError('Final action failed independent feasibility validation')
    return best, {'status': 'selected', 'predicted_utility': best_score, 'candidates': evaluated,
                  'moves': int((best != observation['current']).sum()),
                  'compute_ms': (time.perf_counter()-start)*1000, 'physical_safety_guarantee': False}
