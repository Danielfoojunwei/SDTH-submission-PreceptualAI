"""Equal-budget active scanning and matched calibration/planning study."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np

from ..envlearn import Survey, RadioMap, truth_evaluator
from ..fastplan import BatchEvaluator, Snapshot
from ..missions import WAREHOUSE_MISSIONS
from ..oracle import horizon_snapshots
from .data import make_environment, measure_scan, observed_candidates
from .experiment import atomic_json
from .metrics import cluster_interval


def calibrated_class():
    path = Path(__file__).resolve().parents[3]/'docs/experiments/e7_calibration_study.py'
    spec = importlib.util.spec_from_file_location('warehouse_v2_calibration_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CalibratedMap


def append_scan(survey, scene, robot, *, seed):
    # Identical robot/query pairs yield identical sensor noise in every arm.
    rng = np.random.default_rng(seed)
    row = measure_scan(scene, rng)[robot]
    car = scene.cars[robot]
    for j, (value, heard, bound) in enumerate(row):
        if heard:
            survey.x.append(car.x); survey.y.append(car.y); survey.ap.append(j)
            survey.rssi.append(float(value*30.-80.))
        else:
            survey.cx.append(car.x); survey.cy.append(car.y); survey.cap.append(j)
            survey.cbound.append(float(bound*30.-80.))
    for key, duty in scene.external_occupancy.items():
        survey.busy.setdefault(key, []).append(float(np.clip(duty+rng.normal(0, .02), 0, 1)))


def belief(scene, radio_map, survey):
    # Construct from nominal public metadata, never by copying a truth evaluator.
    nominal = replace(scene, obstruction=None, shadow_db=0., pl_coef=30., ap_height_m=5.,
                      external_occupancy={k: float(np.mean(v)) for k, v in survey.busy.items()},
                      _radio=None, _radio_key=None)
    planned = [scene]+horizon_snapshots(scene)
    ev = BatchEvaluator(nominal, [])
    ap_power = np.array([a.tx_power_dbm for a in scene.aps])
    ev.snaps = []
    uncertainty = []
    for snapshot in planned:
        mean, std = radio_map.predict(np.array([c.x for c in snapshot.cars]),
                                      np.array([c.y for c in snapshot.cars]))
        ev.snaps.append(Snapshot(mean-ap_power[None]+scene.cars[0].tx_power_dbm, mean))
        uncertainty.append(std)
    ev.client_itf = True
    # Nominal AP/channel metadata determines contention; hidden interference
    # strengths and leakage are not copied from truth.
    return ev, uncertainty


def fit(scene, survey, kind, seed):
    cls = RadioMap if kind == 'corrected_v2' else calibrated_class()
    model = cls(scene, 5., D=48, seed=seed)
    model.fit(survey)
    return model


def decision_probe_scores(ev, uncertainty, plans, weights):
    """Declared heuristic, not exact expected information gain.

    Prefer uncertain links on candidate plans close in utility to the incumbent,
    weighted by public demand. Queries are entire capped scans, not hidden-link
    RSSI requests. No truth outcome enters the acquisition function.
    """
    values = ev.utility(plans, weights)
    best = int(values.argmax())
    temperature = max(float(np.std(values)), .02)
    competition = np.exp((values-values.max())/temperature)
    competition[best] = 0.
    scores = np.zeros(ev.n)
    for k, plan in enumerate(plans):
        changed = plan != plans[best]
        std = uncertainty[0][np.arange(ev.n), plan]
        scores += competition[k]*changed*std*(1.+ev.demand/max(ev.demand.mean(), 1e-9))
    # If plans agree, uncertainty still breaks ties without truth access.
    return scores+1e-6*np.mean(uncertainty[0], axis=1)


def one_environment(seed):
    scene, spec = make_environment(90_000+seed, n=100)
    initial = Survey(censor_weight=1.)
    for tick in range(12):
        append_scan(initial, scene, (tick*7) % 100, seed=seed*1000+tick)
        scene = scene.advance(5.)
    basic = fit(scene, initial, 'corrected_v2', seed)
    base_ev, _ = belief(scene, basic, initial)
    signal = base_ev.snaps[0].dl
    order = np.argsort(-signal, axis=1)
    # Four fixed candidates for every arm, using only the initial map.
    pseudo_scan = np.stack([(signal+80.)/30., (signal >= -90.).astype(float),
                            np.full(signal.shape, -1/3)], -1).astype('float32')
    plans = observed_candidates(np.stack([pseudo_scan]*4), base_ev.current, base_ev.demand,
                                np.random.default_rng(seed+333))
    truth = truth_evaluator(scene, spec, np.random.default_rng(seed+999))
    truth_dl = truth.snaps[0].dl
    records = []
    for kind in ['corrected_v2', 'geometry_censored']:
        for acquisition in ['uniform', 'decision_aware']:
            survey = deepcopy(initial)
            selected = []
            rng = np.random.default_rng(seed+555)
            for budget in [0, 10, 20, 40]:
                while len(selected) < budget:
                    model = fit(scene, survey, kind, seed)
                    ev, uncertainty = belief(scene, model, survey)
                    remaining = np.array([i for i in range(100) if i not in selected])
                    batch = min(10, budget-len(selected))
                    if acquisition == 'uniform':
                        queries = rng.choice(remaining, batch, replace=False)
                    else:
                        scores = decision_probe_scores(ev, uncertainty, plans,
                                                       WAREHOUSE_MISSIONS['shift_fairness'].weights)
                        queries = remaining[np.argsort(-scores[remaining])[:batch]]
                    for robot in queries:
                        append_scan(survey, scene, int(robot), seed=seed*1000+100+int(robot))
                        selected.append(int(robot))
                model = fit(scene, survey, kind, seed)
                ev, uncertainty = belief(scene, model, survey)
                mu = ev.snaps[0].dl
                row = {'environment': seed, 'map': kind, 'acquisition': acquisition,
                       'extra_robot_scans': budget, 'initial_robot_scans': 12,
                       'selected_robots': list(selected),
                       'audible_rmse_db': float(np.sqrt(np.mean((mu[truth_dl >= -90]-truth_dl[truth_dl >= -90])**2))) if (truth_dl >= -90).any() else None,
                       'inaudible_rmse_db': float(np.sqrt(np.mean((mu[truth_dl < -90]-truth_dl[truth_dl < -90])**2))) if (truth_dl < -90).any() else None,
                       'missions': {}}
                for name, mission in list(WAREHOUSE_MISSIONS.items())[:5]:
                    predicted = ev.utility(plans, mission.weights)
                    chosen = int(predicted.argmax())
                    actual = truth.utility(plans, mission.weights)
                    terms = truth._snapshot_terms(plans, truth.snaps[0], True, 30.)
                    # Cost sensitivity, not a measured scan duration. Charge the
                    # scanned robots' lost delivered data within a 30 s window.
                    charged = {}
                    for cost in [.02, .1, .5]:
                        loss = (terms['sat'][chosen, selected]*truth.demand[selected]).sum()*cost/30. if selected else 0.
                        charged[str(cost)] = float(terms['U'][chosen]-loss/max(truth.demand.sum(), 1e-9))
                    row['missions'][name] = {'choice': chosen, 'plan_gap': float(actual.max()-actual[chosen]),
                                             'realised_mission_utility': float(actual[chosen]),
                                             'scan_cost_sensitivity_data_fraction': charged}
                records.append(row)
    return records


def study(root, environments=15):
    rows = []
    for seed in range(40, 40+environments):
        path = root/'cache'/f'sensing-environment-{seed}.json'
        if path.exists():
            import json
            result = json.loads(path.read_text())
        else:
            print(f'Active sensing/calibration environment {seed}', flush=True)
            result = one_environment(seed)
            atomic_json(path, result)
        rows.extend(result)
    summary = {}
    for kind in ['corrected_v2', 'geometry_censored']:
        for acquisition in ['uniform', 'decision_aware']:
            for budget in [0, 10, 20, 40]:
                subset = [r for r in rows if (r['map'], r['acquisition'], r['extra_robot_scans']) == (kind, acquisition, budget)]
                gaps = [np.mean([m['plan_gap'] for m in r['missions'].values()]) for r in subset]
                summary[f'{kind}/{acquisition}/{budget}'] = cluster_interval(gaps, [r['environment'] for r in subset])
    return {'source': 'paired_warehouse_simulation', 'environments': environments,
            'random_features': 48, 'assumed_ap_height_m': 5., 'summary': summary, 'records': rows,
            'limits': ['Independent v2 protocol, not the historical 200-feature E7 survey sweep',
                       'Scan costs are sensitivity assumptions, not measured hardware durations',
                       'Decision-aware acquisition is a candidate-disagreement heuristic',
                       'Gap is to the best of four fixed candidates, not a global optimum',
                       'Public nominal layout is available; hidden materials, walls and propagation are excluded from the map fit']}
