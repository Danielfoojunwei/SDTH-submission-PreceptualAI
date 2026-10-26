"""Exact evaluator for a complete fleet configuration.

A configuration is {"assoc": {car_id: ap_id}, "channels": {ap_id: channel}}.
Evaluation is deterministic given the scene, so it serves as ground truth for
regret and as the teacher for distillation.

Reward families implemented (each from a Li/Shen/Lam paper):
- composite  R = B + aU - gW + bV       (Li, Shui, Lam, Shen, Wang; ICAIIC 2025)
- piecewise  SNR / SINR / 0 / -10       (Zhang, Lam, Shen, Wang, Li; Ad Hoc Netw 2023)
- qos_margin SINR - mu                  (Li, Shen, Guo, Lam, Wei, Wang; IEEE TVT 2022)
- pm1        +1 / -0.5 / -1             (Wang, Shen, Li, Lam; WCNC 2025)
mission_utility() is the composite reward extended with mission weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .missions import Mission
from .twin.observe import channel_status_term, interference_state
from .twin.radio import (DECISION_HORIZON_S, HANDOVER_OUTAGE_S,
                         MAC_EFFICIENCY, Scene, phy_rate_mbps, rate_table)

RETUNE_OUTAGE_S = 0.05


@dataclass
class Evaluation:
    delivered: dict            # car_id -> {flow_name: Mbps}
    total_mbps: float
    critical_violations: int
    worst_critical_margin: float
    satisfaction: dict         # car_id -> [0, 1]
    U: float
    V: float
    W: float
    B: float
    jain: float
    handovers: int
    retunes: int
    protected_hits: int
    max_domain_load: float
    access_rate: float
    piecewise: float
    qos_margin: float
    pm1: float
    sinr_db: dict
    domain_load: dict

    def summary(self) -> dict:
        return {
            "total_mbps": round(self.total_mbps, 2),
            "critical_violations": self.critical_violations,
            "worst_critical_margin_mbps": round(self.worst_critical_margin, 2),
            "handovers": self.handovers,
            "retunes": self.retunes,
            "protected_hits": self.protected_hits,
            "max_domain_load": round(self.max_domain_load, 3),
            "jain": round(self.jain, 3),
            "access_rate": round(self.access_rate, 3),
        }


def _snr_needed(demand_mbps: float, band: str) -> float:
    table = rate_table(band)
    for threshold, mbps in table:
        if mbps * MAC_EFFICIENCY >= demand_mbps:
            return threshold
    return table[-1][0]


def evaluate(scene: Scene, config: dict, horizon_s: float = DECISION_HORIZON_S) -> Evaluation:
    assoc: dict = config["assoc"]
    channels: dict = config["channels"]
    domains = scene.domains(channels)
    domain_of = {ap: i for i, d in enumerate(domains) for ap in d}

    activity = {ap.ap_id: 0.5 for ap in scene.aps}
    for _ in range(3):  # fixed point: interference depends on airtime use
        sinr = {c.car_id: scene.sinr_db(c, assoc[c.car_id], channels, activity,
                                        domains) for c in scene.cars}
        rate = {cid: phy_rate_mbps(s, scene.ap(assoc[cid]).band) * MAC_EFFICIENCY
                for cid, s in sinr.items()}
        crit_need = [0.0] * len(domains)
        be_need = [0.0] * len(domains)
        for car in scene.cars:
            d = domain_of[assoc[car.car_id]]
            r = rate[car.car_id]
            for f in car.flows:
                need = f.demand_mbps / r if r > 0 else math.inf
                if f.critical:
                    crit_need[d] += need
                else:
                    be_need[d] += need
        crit_frac, be_frac, load = [], [], []
        for d, members in enumerate(domains):
            avail = 1.0 - scene.occupancy(scene.ap(next(iter(members))), channels)
            c, b = crit_need[d], be_need[d]
            load.append((c + b) / avail if avail > 0 else math.inf)
            if c <= avail:
                crit_frac.append(1.0)
                rest = avail - c
                be_frac.append(1.0 if b <= rest else (rest / b if b > 0 else 1.0))
            else:
                crit_frac.append(avail / c if math.isfinite(c) and c > 0 else 0.0)
                be_frac.append(0.0)
        new_activity = {ap.ap_id: 0.0 for ap in scene.aps}
        for car in scene.cars:
            ap_id = assoc[car.car_id]
            d = domain_of[ap_id]
            r = rate[car.car_id]
            if r <= 0:
                continue
            for f in car.flows:
                frac = crit_frac[d] if f.critical else be_frac[d]
                new_activity[ap_id] += frac * f.demand_mbps / r
        activity = {k: min(1.0, v) for k, v in new_activity.items()}

    current_channels = {ap.ap_id: ap.channel for ap in scene.aps}
    retuned = {ap for ap, ch in channels.items() if current_channels[ap] != ch}
    handovers = 0
    delivered, satisfaction = {}, {}
    violations, worst_margin = 0, math.inf
    for car in scene.cars:
        ap_id = assoc[car.car_id]
        d = domain_of[ap_id]
        outage = 0.0
        if car.current_ap is not None and car.current_ap != ap_id:
            handovers += 1
            outage += HANDOVER_OUTAGE_S
        if ap_id in retuned:
            outage += RETUNE_OUTAGE_S
        keep = max(0.0, 1.0 - outage / horizon_s)
        per_flow = {}
        for f in car.flows:
            frac = crit_frac[d] if f.critical else be_frac[d]
            if rate[car.car_id] <= 0:
                frac = 0.0
            got = f.demand_mbps * frac * keep
            per_flow[f.name] = got
            if f.critical:
                margin = got - f.floor_mbps
                worst_margin = min(worst_margin, margin)
                if margin < -1e-9:
                    violations += 1
        delivered[car.car_id] = per_flow
        satisfaction[car.car_id] = min(1.0, sum(per_flow.values()) /
                                       max(car.demand_mbps, 1e-9))

    state = interference_state(scene, assoc, channels)
    B = sum(channel_status_term(state[c.car_id][assoc[c.car_id]])
            for c in scene.cars) / len(scene.cars)
    protected_hits = sum(1 for c in scene.cars
                         if state[c.car_id][assoc[c.car_id]] == -1)
    total = sum(sum(v.values()) for v in delivered.values())
    demand = sum(c.demand_mbps for c in scene.cars)
    sats = list(satisfaction.values())
    jain = (sum(sats) ** 2) / (len(sats) * sum(s * s for s in sats)) \
        if any(sats) else 0.0

    # W: mean SINR degradation (dB) caused by co-channel interference.
    W = 0.0
    no_interference = {ap.ap_id: 0.0 for ap in scene.aps}
    for car in scene.cars:
        snr = scene.sinr_db(car, assoc[car.car_id], channels, no_interference,
                            domains)
        W += max(0.0, snr - sinr[car.car_id]) / 10.0
    W /= len(scene.cars)

    piecewise = qos = pm1 = 0.0
    access_ok = 0
    for car in scene.cars:
        code = state[car.car_id][assoc[car.car_id]]
        s = sinr[car.car_id]
        piecewise += {2: s / 10, 1: s / 10, 0: 0.0, -1: -10.0}[code]
        qos += (s - _snr_needed(car.demand_mbps, scene.ap(assoc[car.car_id]).band)) / 10
        crit_ok = all(
            delivered[car.car_id][f.name] >= f.floor_mbps - 1e-9
            for f in car.flows if f.critical)
        access_ok += crit_ok
        if crit_ok and satisfaction[car.car_id] >= 0.9:
            pm1 += 1.0
        elif crit_ok:
            pm1 -= 0.5
        else:
            pm1 -= 1.0

    return Evaluation(
        delivered=delivered, total_mbps=total,
        critical_violations=violations,
        worst_critical_margin=worst_margin if math.isfinite(worst_margin) else 0.0,
        satisfaction=satisfaction, U=total / max(demand, 1e-9),
        V=sum(sats) / len(sats), W=W, B=B, jain=jain, handovers=handovers,
        retunes=len(retuned), protected_hits=protected_hits,
        max_domain_load=max(load), access_rate=access_ok / len(scene.cars),
        piecewise=piecewise, qos_margin=qos, pm1=pm1, sinr_db=sinr,
        domain_load={i: l for i, l in enumerate(load)},
    )


def mission_utility(ev: Evaluation, mission: Mission, n_cars: int) -> float:
    w = mission.weights
    focus = 0.0
    if mission.focus_cars:
        focus = sum(ev.satisfaction[c] for c in mission.focus_cars) / len(
            mission.focus_cars)
    return (w["B"] * ev.B + w["U"] * ev.U + w["V"] * ev.V - w["W"] * ev.W
            + w["fair"] * ev.jain
            - w["ho"] * (ev.handovers + 0.5 * ev.retunes) / n_cars
            + w["focus"] * focus
            - w["viol"] * ev.critical_violations / n_cars
            - w["prot"] * ev.protected_hits / n_cars)


REWARDS = {
    "composite": lambda ev: ev.B + ev.U - 0.3 * ev.W + ev.V,
    "piecewise": lambda ev: ev.piecewise,
    "qos_margin": lambda ev: ev.qos_margin,
    "pm1": lambda ev: ev.pm1,
}
