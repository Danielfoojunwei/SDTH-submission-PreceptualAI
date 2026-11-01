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

import numpy as np

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


def evaluate_reference(scene: Scene, config: dict,
                       horizon_s: float = DECISION_HORIZON_S) -> Evaluation:
    """Original per-car implementation, kept as the ground truth that the
    vectorised evaluate() is tested against."""
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
                # A client with no usable rate is disconnected: it fails
                # itself but consumes no airtime.
                need = f.demand_mbps / r if r > 0 else 0.0
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


_TABLES = {band: (np.array([t for t, _ in rate_table(band)]),
                  np.array([m for _, m in rate_table(band)]))
           for band in ("5g", "2g")}


def evaluate(scene: Scene, config: dict, horizon_s: float = DECISION_HORIZON_S) -> Evaluation:
    """Vectorised evaluator (same model and outputs as evaluate_reference)."""
    assoc: dict = config["assoc"]
    channels: dict = config["channels"]
    r = scene.radio()
    cars, aps = scene.cars, scene.aps
    n, m = len(cars), len(aps)
    a = np.array([r.ap_index[assoc[c.car_id]] for c in cars])
    domains = scene.domains(channels)
    dom_of_ap = np.empty(m, dtype=int)
    for d, members in enumerate(domains):
        for ap_id in members:
            dom_of_ap[r.ap_index[ap_id]] = d
    nd = len(domains)
    dcar = dom_of_ap[a]
    chan = np.array([channels[ap.ap_id] for ap in aps])
    same_bc = (r.band[:, None] == r.band[None, :]) & (chan[:, None] == chan[None, :])
    interferer = same_bc & (dom_of_ap[:, None] != dom_of_ap[None, :])
    apap_mw = np.where(interferer, 10 ** (r.apap / 10), 0.0)
    noise_mw = 10 ** (r.noise / 10)
    ul = r.ul[np.arange(n), a]
    crit = np.array([sum(f.demand_mbps for f in c.flows if f.critical) for c in cars])
    be = np.array([sum(f.demand_mbps for f in c.flows if not f.critical) for c in cars])
    avail = np.array([1.0 - scene.occupancy(scene.ap(next(iter(dm))), channels)
                      for dm in domains])
    band_of_car = r.band[a]

    activity = np.full(m, 0.5)
    for _ in range(3):
        interference = activity @ apap_mw                  # per receiving AP
        sinr = ul - 10 * np.log10(noise_mw[a] + interference[a])
        rate = np.zeros(n)
        for band, (thr, mbps) in _TABLES.items():
            sel = band_of_car == band
            idx = np.searchsorted(thr, sinr[sel], side="right") - 1
            rate[sel] = np.where(idx >= 0, mbps[np.clip(idx, 0, None)], 0.0) * MAC_EFFICIENCY
        pos = rate > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            cn = np.where(pos, crit / np.where(pos, rate, 1), 0.0)   # disconnected: no airtime
            bn = np.where(pos, be / np.where(pos, rate, 1), 0.0)
        crit_need = np.zeros(nd)
        be_need = np.zeros(nd)
        np.add.at(crit_need, dcar, cn)
        np.add.at(be_need, dcar, bn)
        with np.errstate(divide="ignore", invalid="ignore"):
            load = np.where(avail > 0, (crit_need + be_need) / np.where(avail > 0, avail, 1), np.inf)
            fits = crit_need <= avail
            rest = avail - crit_need
            be_frac = np.where(fits, np.where(be_need <= rest, 1.0,
                                              np.where(be_need > 0, rest / np.where(be_need > 0, be_need, 1), 1.0)), 0.0)
            crit_frac = np.where(fits, 1.0, np.where(np.isfinite(crit_need) & (crit_need > 0),
                                                     avail / np.where(crit_need > 0, crit_need, 1), 0.0))
            per_car = np.where(pos, (crit_frac[dcar] * crit + be_frac[dcar] * be)
                               / np.where(pos, rate, 1), 0.0)
        new_activity = np.zeros(m)
        np.add.at(new_activity, a, per_car)
        activity = np.minimum(1.0, new_activity)

    current_channels = {ap.ap_id: ap.channel for ap in aps}
    retuned = {ap for ap, ch in channels.items() if current_channels[ap] != ch}
    cur = [c.current_ap for c in cars]
    moved = np.array([c is not None and c != assoc[cid] for c, cid in
                      zip(cur, (c.car_id for c in cars))])
    retune_hit = np.array([assoc[c.car_id] in retuned for c in cars])
    outage = moved * HANDOVER_OUTAGE_S + retune_hit * RETUNE_OUTAGE_S
    keep = np.maximum(0.0, 1.0 - outage / horizon_s)
    cfrac = np.where(pos, crit_frac[dcar], 0.0)
    bfrac = np.where(pos, be_frac[dcar], 0.0)

    delivered, satisfaction = {}, {}
    violations, worst_margin = 0, math.inf
    for i, car in enumerate(cars):
        per_flow = {}
        for f in car.flows:
            got = f.demand_mbps * (cfrac[i] if f.critical else bfrac[i]) * keep[i]
            per_flow[f.name] = got
            if f.critical:
                margin = got - f.floor_mbps
                worst_margin = min(worst_margin, margin)
                if margin < -1e-9:
                    violations += 1
        delivered[car.car_id] = per_flow
        satisfaction[car.car_id] = min(1.0, sum(per_flow.values()) / max(car.demand_mbps, 1e-9))

    key_bc = list(zip(band_of_car, chan[a]))
    count = {}
    for k in key_bc:
        count[k] = count.get(k, 0) + 1
    ext = np.array([scene.external_occupancy.get((b, int(ch)), 0.0) > 0 for b, ch in key_bc])
    shared = np.array([count[k] > 1 for k in key_bc])
    code = np.where(ext, -1, np.where(shared, 1, 2))
    B = float(np.mean(np.select([code == 2, code == 1, code == -1], [1.0, -0.5, -1.0], 0.0)))
    protected_hits = int((code == -1).sum())
    total = sum(sum(v.values()) for v in delivered.values())
    demand = float((crit + be).sum())
    sats = np.array(list(satisfaction.values()))
    jain = float(sats.sum() ** 2 / (n * (sats ** 2).sum())) if sats.any() else 0.0
    snr = ul - 10 * np.log10(noise_mw[a])
    W = float(np.mean(np.maximum(0.0, snr - sinr) / 10.0))
    piecewise = float(np.sum(np.where(code == -1, -10.0, sinr / 10)))
    qos = pm1 = 0.0
    access_ok = 0
    sinr_d = {}
    for i, car in enumerate(cars):
        qos += (sinr[i] - _snr_needed(car.demand_mbps, band_of_car[i])) / 10
        crit_ok = all(delivered[car.car_id][f.name] >= f.floor_mbps - 1e-9
                      for f in car.flows if f.critical)
        access_ok += crit_ok
        pm1 += 1.0 if crit_ok and satisfaction[car.car_id] >= 0.9 else (-0.5 if crit_ok else -1.0)
        sinr_d[car.car_id] = float(sinr[i])
    return Evaluation(
        delivered=delivered, total_mbps=total, critical_violations=violations,
        worst_critical_margin=worst_margin if math.isfinite(worst_margin) else 0.0,
        satisfaction=satisfaction, U=total / max(demand, 1e-9),
        V=float(sats.mean()), W=W, B=B, jain=jain, handovers=int(moved.sum()),
        retunes=len(retuned), protected_hits=protected_hits,
        max_domain_load=float(load.max()), access_rate=access_ok / n,
        piecewise=piecewise, qos_margin=qos, pm1=pm1, sinr_db=sinr_d,
        domain_load={i: float(l) for i, l in enumerate(load)},
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


# --- route-aware horizon -----------------------------------------------------
HORIZON_OFFSETS_S = (10.0, 20.0)


def horizon_snapshots(scene: Scene, offsets=HORIZON_OFFSETS_S) -> list[Scene]:
    """Future snapshots along each robot's committed plan (no new goals).
    Empty when the scene has no route-aware mobility model."""
    mob = scene.mobility
    if mob is None or not hasattr(mob, "planned_positions"):
        return []
    times = list(offsets)
    future = {c.car_id: mob.planned_positions(c, times) for c in scene.cars}
    from dataclasses import replace
    snaps = []
    for k in range(len(times)):
        cars = [replace(c, x=future[c.car_id][k][0], y=future[c.car_id][k][1])
                for c in scene.cars]
        snap = scene._copy(cars=cars)
        snap.radio()          # build once; copies made from it reuse it
        snaps.append(snap)
    return snaps


def horizon_utility(scene: Scene, snaps: list[Scene], assoc: dict, mission,
                    channels: dict | None = None) -> float:
    """Mean mission utility now and at the planned future positions. The
    handover cost is charged once (now); later snapshots treat the plan's
    associations as already in place."""
    channels = channels or {ap.ap_id: ap.channel for ap in scene.aps}
    n = len(scene.cars)
    vals = [mission_utility(evaluate(scene, {"assoc": assoc, "channels": channels}),
                            mission, n)]
    for snap in snaps:
        later = snap.with_associations(assoc)
        vals.append(mission_utility(evaluate(later, {"assoc": assoc, "channels": channels}),
                                    mission, n))
    return float(sum(vals) / len(vals))


def future_link_losses(snaps: list[Scene], assoc: dict) -> int:
    """Robots whose assigned BSS stops being usable (no rate, or below the
    -80 dBm legacy threshold) somewhere on their planned route."""
    lost = np.zeros(0, dtype=bool)
    for snap in snaps:
        r = snap.radio()
        idx = np.arange(len(snap.cars))
        a = np.array([r.ap_index[assoc[c.car_id]] for c in snap.cars])
        snr = r.ul[idx, a] - r.noise[a]               # interference-free SNR
        min_thr = np.array([_TABLES[b][0][0] for b in r.band[a]])
        bad = (r.dl[idx, a] < -80.0) | (snr < min_thr)
        lost = bad if lost.size == 0 else (lost | bad)
    return int(lost.sum())
