"""Candidate generation: a short list of complete 10-car configurations.

The numeric layer does the arithmetic; the decision model only chooses. Each
option carries digested, human-readable consequences (reconnections, critical
floors, fleet throughput, least-served car, airtime, protected-channel use)
plus option-wise relative descriptors, because Jev-class models are weak at
raw numeric comparison (arXiv 2610.18529) and option-wise normalisation helps.

Options never say which mission or teacher produced them, so the model must
match consequences to the intent rather than match labels.

Hard filters follow the constraints of Shen, Goh, Lam & Li (ICCT 2024): the
association must exist (BSS visible), the channel must be usable (non-zero
rate, not saturated by protected traffic), and the car's transmit power must
respect its cap.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from .missions import MISSIONS, Mission
from .oracle import (Evaluation, evaluate, future_link_losses,
                     horizon_snapshots, horizon_utility, mission_utility)
from .teachers.assign import (coloring_assignment, hungarian_access,
                              legacy_policy, local_greedy, local_search,
                              milp_assignment, troubled_cars)
from .twin.radio import MAC_EFFICIENCY, Scene, phy_rate_mbps

MAX_OPTIONS = 7          # plus DEFER
DEFER_ID = "DEFER"


@dataclass
class Option:
    option_id: str
    assoc: dict[str, str]
    evaluation: Evaluation | None
    text: str = ""
    source: str = ""          # kept for audit only; never shown to the model
    future_losses: int = 0    # robots losing a usable link on their route

    @property
    def is_defer(self) -> bool:
        return self.option_id == DEFER_ID


def _key(assoc: dict[str, str]) -> str:
    return hashlib.sha256(repr(sorted(assoc.items())).encode()).hexdigest()[:12]


def hard_filter(scene: Scene, assoc: dict[str, str],
                pinned: frozenset[str] = frozenset()) -> str | None:
    channels = {ap.ap_id: ap.channel for ap in scene.aps}
    for car in scene.cars:
        if car.car_id in pinned:
            if assoc[car.car_id] != car.current_ap:
                return f"{car.car_id}: quarantined car must keep its BSSID"
            continue
        ap_id = assoc[car.car_id]
        visible = {ap.ap_id for ap, _ in scene.visible_aps(car)}
        if ap_id not in visible:
            return f"{car.car_id}: BSS {ap_id} not visible"
        ap = scene.ap(ap_id)
        if scene.occupancy(ap, channels) >= 0.95:
            return f"{car.car_id}: channel saturated by protected traffic"
        sinr = scene.sinr_db(car, ap_id, channels, {})
        if phy_rate_mbps(sinr, ap.band) * MAC_EFFICIENCY <= 0:
            return f"{car.car_id}: no usable rate on {ap_id}"
        if car.tx_power_dbm > car.tx_power_cap_dbm:
            return f"{car.car_id}: transmit power above contract cap"
    return None


def generate(scene: Scene, *, seed: int = 0,
             missions: dict[str, Mission] | None = None,
             pinned: frozenset[str] = frozenset(),
             horizon: bool = True) -> tuple[list[Option], list[dict]]:
    """Return (options incl. DEFER, rejected) for one fleet snapshot.

    The same option set is produced regardless of which mission will be
    asked, so the intent-flip test compares choices over identical options.
    """
    missions = missions or MISSIONS
    channels = {ap.ap_id: ap.channel for ap in scene.aps}
    hold = {c.car_id: c.current_ap for c in scene.cars}
    proposals: list[tuple[str, dict]] = [("hold", hold),
                                         ("legacy", legacy_policy(scene))]
    snaps = horizon_snapshots(scene) if horizon else []
    large = len(scene.cars) > 20
    for m in missions.values():
        start = milp_assignment(scene, m)
        proposals.append((f"milp:{m.mission_id}", start))
        scorer = ((lambda a, m=m: horizon_utility(scene, snaps, a, m, channels))
                  if snaps else None)
        if large:
            focus = [c for c in troubled_cars(scene, start) if c not in pinned]
            best = local_search(scene, m, start, max_sweeps=3, scorer=scorer,
                                max_bss=4, focus=focus)
        else:
            best = local_search(scene, m, start, scorer=scorer)
        proposals.append((f"ls:{m.mission_id}", best))
    proposals += [("hungarian", hungarian_access(scene)),
                  ("coloring", coloring_assignment(scene)),
                  ("local_greedy", local_greedy(scene))]

    # Quarantined (low-reputation) cars keep their BSSID in every option.
    proposals = [(src, {**a, **{c: hold[c] for c in pinned}}) for src, a in proposals]
    unique: dict[str, tuple[str, dict]] = {}
    rejected = []
    for source, assoc in proposals:
        k = _key(assoc)
        if k in unique:
            continue
        reason = hard_filter(scene, assoc, pinned)
        if reason:
            rejected.append({"source": source, "reason": reason})
            continue
        unique[k] = (source, assoc)

    # Keep the per-mission optima and HOLD first, then fill with the rest.
    def rank(item):
        source = item[0]
        return (0 if source.startswith("ls:") else
                1 if source == "hold" else
                2 if source.startswith("milp:") else 3)

    chosen = sorted(unique.values(), key=rank)[:MAX_OPTIONS]
    rng = random.Random(seed)
    rng.shuffle(chosen)

    options = []
    for i, (source, assoc) in enumerate(chosen):
        ev = evaluate(scene, {"assoc": assoc, "channels": channels})
        opt = Option(f"P{i + 1}", assoc, ev, source=source)
        opt.future_losses = future_link_losses(snaps, assoc) if snaps else 0
        options.append(opt)
    if options:
        _describe(scene, options, horizon=bool(snaps))
    for o in options:
        o.snaps = snaps
    options.append(Option(DEFER_ID, hold, None, source="defer",
                          text="Defer: keep every car's current BSSID and "
                               "escalate to the operator for review."))
    return options, rejected


def _describe(scene: Scene, options: list[Option], horizon: bool = False) -> None:
    hold = {c.car_id: c.current_ap for c in scene.cars}
    critical_cars = [c for c in scene.cars if any(f.critical and f.name ==
                                                  "video_uplink" for f in c.flows)]
    evs = [o.evaluation for o in options]
    max_total = max(e.total_mbps for e in evs)
    min_ho = min(e.handovers for e in evs)
    max_min_sat = max(min(e.satisfaction.values()) for e in evs)
    min_prot = min(e.protected_hits for e in evs)
    for opt in options:
        e = opt.evaluation
        moved = sorted(c for c in opt.assoc if opt.assoc[c] != hold[c])
        parts = []
        if moved:
            parts.append(f"reconnects {len(moved)} car(s)")
        else:
            parts.append("no reconnections")
        if e.critical_violations:
            parts.append(f"{e.critical_violations} critical flow(s) fall below "
                         f"their floor this period")
        else:
            parts.append("all critical floors met")
        if 0 < len(critical_cars) <= 4:
            parts.append("video service: " + ", ".join(
                f"{c.car_id} video {round(100 * e.satisfaction[c.car_id])}%"
                for c in critical_cars))
        elif critical_cars:
            sat = {c.car_id: e.satisfaction[c.car_id] for c in critical_cars}
            low = sorted((k for k, v in sat.items() if v < 0.9), key=sat.get)[:3]
            parts.append(f"video service across {len(sat)} streaming cars: lowest "
                         f"{round(100 * min(sat.values()))}%"
                         + (f" ({', '.join(low)} below 90%)" if low else ", all at 90% or more"))
        demand = sum(c.demand_mbps for c in scene.cars)
        parts.append(f"fleet delivers {e.total_mbps:.0f} of {demand:.0f} Mbps "
                     f"({round(100 * e.U)}%)")
        worst = min(e.satisfaction, key=e.satisfaction.get)
        parts.append(f"least-served car {worst} at "
                     f"{round(100 * e.satisfaction[worst])}% of its demand")
        parts.append(f"busiest channel airtime {round(100 * min(e.max_domain_load, 9.99))}%")
        parts.append(f"{e.protected_hits} car(s) on channels shared with "
                     f"protected neighbour equipment")
        level = "low" if e.W < 0.1 else "medium" if e.W < 0.5 else "high"
        parts.append(f"co-channel interference {level}")
        if horizon:
            parts.append(f"{opt.future_losses} car(s) lose a usable link within "
                         f"20 s on their planned route")
        tags = []
        if e.total_mbps >= max_total - 1e-6:
            tags.append("most total data")
        if e.handovers <= min_ho:
            tags.append("fewest reconnections")
        if min(e.satisfaction.values()) >= max_min_sat - 1e-6:
            tags.append("best for the least-served car")
        if e.protected_hits <= min_prot:
            tags.append("least use of protected channels")
        if horizon and opt.future_losses <= min(o.future_losses for o in options):
            tags.append("fewest upcoming link losses")
        if tags:
            parts.append("relative: " + "; ".join(tags))
        opt.text = "; ".join(parts) + "."


def oracle_scores(scene: Scene, options: list[Option], mission: Mission) -> dict[str, float]:
    """Exact mission utility per option; DEFER equals HOLD minus a small
    escalation cost (it delays any improvement)."""
    n = len(scene.cars)
    channels = {ap.ap_id: ap.channel for ap in scene.aps}
    snaps = next((getattr(o, "snaps", []) for o in options if getattr(o, "snaps", [])), [])
    scores = {}
    for o in options:
        if snaps:
            u = horizon_utility(scene, snaps, o.assoc, mission, channels)
        else:
            ev = o.evaluation or evaluate(scene, {"assoc": o.assoc, "channels": channels})
            u = mission_utility(ev, mission, n)
        scores[o.option_id] = u - (0.05 if o.is_defer else 0.0)
    return scores
