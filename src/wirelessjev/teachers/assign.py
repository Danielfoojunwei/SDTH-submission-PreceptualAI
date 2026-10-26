"""Association teachers (car -> BSSID) and baselines.

The only fleet action available on the SmartCar hardware is choosing a BSSID
for each car's BE200, so every teacher returns {car_id: ap_id}; AP channels
stay as the customer configured them.

- milp_assignment: R6 joint planner over per-(car, BSS) candidates with
  per-domain airtime budgets and mission-weighted linearised utility.
- hungarian_access: max sum SINR * idle with one association per car,
  from Shen, Goh, Lam & Li, "Efficient and Secure Federated Dynamic Spectrum
  Access for LEO Satellite IoT" (ICCT 2024); BSS capacity is expanded into
  slots so a linear assignment applies.
- coloring_assignment: collaborative max-min labelling argmax b/(D+1) from Li,
  Sun, Lam, Sun, Shen & Peng (Wireless Pers. Commun. 2024).
- local_search: exact-evaluator hill climbing; used as the global reference
  when measuring how good the candidate set is.
- legacy_policy: the existing SmartCar rule (keep unless < -80 dBm; choose by
  SSID priority, then strongest signal).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..hardware import RASPBOT
from ..missions import Mission
from ..oracle import evaluate, mission_utility
from ..twin.radio import (HANDOVER_OUTAGE_S, DECISION_HORIZON_S,
                          MAC_EFFICIENCY, Scene, phy_rate_mbps)
from .joint_planner import Candidate, plan_joint


def _channels(scene: Scene) -> dict[str, int]:
    return {ap.ap_id: ap.channel for ap in scene.aps}


def _visible_ids(scene: Scene, car) -> set[str]:
    return {ap.ap_id for ap, _ in scene.visible_aps(car)}


def _solo_rate(scene: Scene, car, ap_id: str) -> float:
    """Uplink rate (Mbps after MAC efficiency) with no co-channel activity."""
    channels = _channels(scene)
    sinr = scene.sinr_db(car, ap_id, channels, {})
    return phy_rate_mbps(sinr, scene.ap(ap_id).band) * MAC_EFFICIENCY


def milp_assignment(scene: Scene, mission: Mission) -> dict[str, str]:
    channels = _channels(scene)
    domains = scene.domains(channels)
    domain_of = {ap: f"airtime.d{i}" for i, d in enumerate(domains) for ap in d}
    budgets = {}
    for i, d in enumerate(domains):
        ap = scene.ap(next(iter(d)))
        budgets[f"airtime.d{i}"] = max(0.0, 1.0 - scene.occupancy(ap, channels))
    w = mission.weights
    keep = 1.0 - HANDOVER_OUTAGE_S / DECISION_HORIZON_S
    cands = []
    for car in scene.cars:
        visible = _visible_ids(scene, car)
        crit_floor = sum(f.floor_mbps for f in car.flows if f.critical)
        for ap_id in visible:
            rate = _solo_rate(scene, car, ap_id)
            if rate <= 0:
                continue
            moved = car.current_ap is not None and car.current_ap != ap_id
            # Airtime for critical floors is mandatory; best-effort is scaled
            # into the objective rather than reserved.
            airtime = crit_floor / rate
            served = car.demand_mbps * (keep if moved else 1.0)
            focus = w["focus"] if car.car_id in mission.focus_cars else 0.0
            utility = (w["U"] * min(served, rate) / max(car.demand_mbps, 1e-9)
                       + focus * (keep if moved else 1.0)
                       - w["ho"] * (1.0 if moved else 0.0)
                       - w["viol"] * (1.0 if moved and crit_floor > 0 else 0.0)
                       - w["prot"] * (1.0 if scene.occupancy(scene.ap(ap_id), channels) > 0 else 0.0)
                       + 0.01 * math.log10(rate))
            cands.append(Candidate(f"{car.car_id}@{ap_id}", (car.car_id,),
                                   {domain_of[ap_id]: airtime}, utility))
    plan = plan_joint([c.car_id for c in scene.cars], budgets, cands)
    if not plan["selected"]:
        return {c.car_id: c.current_ap for c in scene.cars}
    return dict(sel.split("@") for sel in plan["selected"])


def hungarian_access(scene: Scene) -> dict[str, str]:
    channels = _channels(scene)
    cars = scene.cars
    slots_per_ap = math.ceil(len(cars) / max(1, len(scene.aps))) + 1
    slots = [ap.ap_id for ap in scene.aps for _ in range(slots_per_ap)]
    cost = np.full((len(cars), len(slots)), 1e6)
    for i, car in enumerate(cars):
        visible = _visible_ids(scene, car)
        for j, ap_id in enumerate(slots):
            if ap_id not in visible:
                continue
            idle = 1.0 - scene.occupancy(scene.ap(ap_id), channels)
            sinr = scene.sinr_db(car, ap_id, channels, {})
            if phy_rate_mbps(sinr, scene.ap(ap_id).band) <= 0:
                continue
            cost[i, j] = -(max(sinr, 0.0) * idle)
    rows, cols = linear_sum_assignment(cost)
    out = {c.car_id: c.current_ap for c in cars}
    for i, j in zip(rows, cols):
        if cost[i, j] < 1e6:
            out[cars[i].car_id] = slots[j]
    return out


def coloring_assignment(scene: Scene) -> dict[str, str]:
    """Collaborative max-min: highest-demand cars are labelled first; each
    picks argmax b/(D+1), D = cars already labelled in that BSS's domain."""
    channels = _channels(scene)
    domains = scene.domains(channels)
    domain_of = {ap: i for i, d in enumerate(domains) for ap in d}
    count = [0] * len(domains)
    out = {}
    for car in sorted(scene.cars, key=lambda c: -c.demand_mbps):
        best, best_score = car.current_ap, -1.0
        for ap_id in _visible_ids(scene, car):
            b = _solo_rate(scene, car, ap_id) * (
                1.0 - scene.occupancy(scene.ap(ap_id), channels))
            score = b / (count[domain_of[ap_id]] + 1)
            if score > best_score:
                best, best_score = ap_id, score
        out[car.car_id] = best
        count[domain_of[best]] += 1
    return out


def local_search(scene: Scene, mission: Mission, start: dict[str, str],
                 max_sweeps: int = 6) -> dict[str, str]:
    channels = _channels(scene)
    n = len(scene.cars)
    current = dict(start)

    def score(assoc):
        return mission_utility(evaluate(scene, {"assoc": assoc,
                                                "channels": channels}),
                               mission, n)

    best = score(current)
    for _ in range(max_sweeps):
        improved = False
        for car in scene.cars:
            for ap_id in _visible_ids(scene, car):
                if ap_id == current[car.car_id]:
                    continue
                trial = dict(current)
                trial[car.car_id] = ap_id
                s = score(trial)
                if s > best + 1e-9:
                    best, current, improved = s, trial, True
        if not improved:
            break
    return current


def legacy_policy(scene: Scene) -> dict[str, str]:
    out = {}
    for car in scene.cars:
        visible = scene.visible_aps(car)
        current = car.current_ap
        rssi = {ap.ap_id: r for ap, r in visible}
        if current in rssi and rssi[current] >= RASPBOT.weak_signal_dbm:
            out[car.car_id] = current
        else:
            out[car.car_id] = max(visible, key=lambda s: (s[0].priority, s[1]))[0].ap_id
    return out


def strongest_rssi(scene: Scene) -> dict[str, str]:
    return {car.car_id: scene.visible_aps(car)[0][0].ap_id for car in scene.cars}


def random_assignment(scene: Scene, seed: int = 0) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    return {car.car_id: str(rng.choice(sorted(_visible_ids(scene, car))))
            for car in scene.cars}


def local_greedy(scene: Scene) -> dict[str, str]:
    """Each car selfishly maximises its own expected share, in turn."""
    channels = _channels(scene)
    domains = scene.domains(channels)
    domain_of = {ap: i for i, d in enumerate(domains) for ap in d}
    load = [0.0] * len(domains)
    out = {}
    for car in scene.cars:
        best, best_share = car.current_ap, -1.0
        for ap_id in _visible_ids(scene, car):
            rate = _solo_rate(scene, car, ap_id)
            if rate <= 0:
                continue
            free = 1.0 - scene.occupancy(scene.ap(ap_id), channels) - load[domain_of[ap_id]]
            share = max(0.0, free) * rate
            if share > best_share:
                best, best_share = ap_id, share
        out[car.car_id] = best
        rate = _solo_rate(scene, car, best)
        if rate > 0:
            load[domain_of[best]] += car.demand_mbps / rate
    return out


BASELINES = {
    "legacy": lambda scene, mission: legacy_policy(scene),
    "strongest_rssi": lambda scene, mission: strongest_rssi(scene),
    "random": lambda scene, mission: random_assignment(scene, scene.seed),
    "local_greedy": lambda scene, mission: local_greedy(scene),
    "hungarian": lambda scene, mission: hungarian_access(scene),
    "coloring": lambda scene, mission: coloring_assignment(scene),
    "milp": milp_assignment,
}
