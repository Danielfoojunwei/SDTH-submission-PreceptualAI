"""Closed-loop 10-car swarm simulation (digital twin, dry-run transport).

Timeline (5 s policy steps, 300 s total by default):
- 0-150 s   mission "protect_video", 150-300 s mission "convoy_continuity";
- a fleet decision every 30 s (the commitment horizon) from WirelessJEV with
  the distilled option-wise adapter, executed through the transaction ledger
  and a SmartCarCmd-compatible dry-run transport (5% clean failures, 5%
  unknown outcomes, reconciled from status on the next step);
- car-10 starts transmitting above its power cap at t=120 s: reputation drops,
  it is quarantined (pinned to its BSSID) and excluded from swarm merges;
- the coordinator is lost from 180-210 s: no fleet decisions; each car runs
  only its local legacy rule (reconnect below -80 dBm).

The same timeline is replayed with the legacy policy and the MILP teacher in
the decision slot, so realised mission utility can be compared step by step.

    uv run python -m wirelessjev.closedloop --adapter artifacts/bench/adapters/fleet
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .candidates import DEFER_ID, generate
from .execution import (DryRunTransport, Ledger, Transaction, build_connect,
                        execute, reconcile)
from .hardware import RASPBOT
from .missions import MISSIONS
from .oracle import evaluate, mission_utility
from .swarm import Reputation
from .teachers import legacy_policy, milp_assignment
from .twin import make_scene
from .twin.radio import DECISION_INTERVAL_S, MAC_EFFICIENCY, phy_rate_mbps

STEP = DECISION_INTERVAL_S


def mission_at(t: float, total: float):
    return MISSIONS["protect_video"] if t < total / 2 else MISSIONS["convoy_continuity"]


def run(policy: str, *, seed: int, total_s: float, engine=None, log=print) -> dict:
    scene = make_scene(seed, hotspot=True)
    channels = {a.ap_id: a.channel for a in scene.aps}
    domains = scene.domains(channels)
    domain_of = {ap: f"d{i}" for i, d in enumerate(domains) for ap in d}
    budgets = {f"d{i}": 1.0 - scene.occupancy(scene.ap(next(iter(d))), channels)
               for i, d in enumerate(domains)}
    actual = {c.car_id: c.current_ap for c in scene.cars}
    transport = DryRunTransport(state=dict(actual), failure_rate=0.05,
                                unknown_rate=0.05, seed=seed)
    ledger = Ledger(budgets=budgets, max_in_flight=4)
    rep = Reputation()
    timeline, decisions, txn_log = [], [], []
    n_steps = int(total_s / STEP)
    decide_every = 6
    for k in range(n_steps):
        t = k * STEP
        mission = mission_at(t, total_s)
        coordinator_up = not (180 <= t < 210)

        # misbehaving car from t=120 s
        if t >= 120:
            scene = scene._copy(cars=[replace(c, tx_power_dbm=24.0) if c.car_id == "car-10"
                                      else c for c in scene.cars])
        for car in scene.cars:
            v = rep.check(car.car_id, tx_power_dbm=car.tx_power_dbm,
                          cap_dbm=car.tx_power_cap_dbm, reconnects=0,
                          used_mbps=car.demand_mbps, declared_mbps=car.demand_mbps)
            rep.update(car.car_id, v)
        # Quarantine on detection (power over cap) or on low reputation.
        pinned = frozenset(c.car_id for c in scene.cars
                           if not rep.trusted(c.car_id)
                           or c.tx_power_dbm > c.tx_power_cap_dbm)

        # resolve unknown outcomes from fresh status
        for txn in list(ledger.in_flight()):
            if txn.state == "UNKNOWN":
                reconcile(txn, ledger, transport)
        actual = dict(transport.state)
        scene = scene.with_associations(actual)

        target = None
        if coordinator_up and k % decide_every == 0:
            started = time.perf_counter()
            if policy == "wirelessjev":
                options, _ = generate(scene, seed=seed + k, pinned=pinned)
                d = engine.score(scene, mission, options)
                chosen = next(o for o in options if o.option_id == d.option_id)
                target = None if chosen.is_defer else chosen.assoc
                decisions.append({"t": t, "mission": mission.mission_id,
                                  "choice": d.option_id, "source": chosen.source,
                                  "latency_ms": round(d.latency_ms, 1),
                                  "pinned": sorted(pinned)})
            elif policy == "milp":
                target = milp_assignment(scene, mission)
                target.update({c: actual[c] for c in pinned})
                decisions.append({"t": t, "mission": mission.mission_id,
                                  "latency_ms": round((time.perf_counter() - started) * 1000, 1)})
        if not coordinator_up or policy == "legacy" or target is None:
            # local fallback on each car: legacy rule only below -80 dBm
            local = legacy_policy(scene)
            target = {c: local[c] if c not in pinned else actual[c] for c in actual}

        for car_id, bssid in target.items():
            if bssid == actual[car_id]:
                continue
            car = scene.car(car_id)
            rate = phy_rate_mbps(scene.sinr_db(car, bssid, channels, {}),
                                 scene.ap(bssid).band) * MAC_EFFICIENCY
            airtime = min(1.0, car.demand_mbps / max(rate, 1e-6)) * 0.2
            txn = Transaction(f"{policy}-{k}-{car_id}", car_id, actual[car_id], bssid,
                              build_connect(scene.ap(bssid).ssid, bssid))
            if not ledger.prepare(txn, domain_of[bssid], airtime):
                txn_log.append({"t": t, "car": car_id, "outcome": "NOT_ADMITTED"})
                continue
            execute(txn, ledger, transport)
            if txn.state in ("COMMITTED", "ABORTED"):
                ledger.release(txn)
            txn_log.append({"t": t, "car": car_id, "outcome": txn.outcome})

        new_assoc = dict(transport.state)
        ev = evaluate(scene, {"assoc": new_assoc, "channels": channels}, horizon_s=STEP)
        timeline.append({"t": t, "mission": mission.mission_id,
                         "utility": mission_utility(ev, mission, len(scene.cars)),
                         "handovers": ev.handovers,
                         "critical_violations": ev.critical_violations,
                         "total_mbps": ev.total_mbps, "coordinator_up": coordinator_up,
                         "quarantined": sorted(pinned)})
        scene = scene.with_associations(new_assoc).advance(STEP)

    outcomes = {}
    for x in txn_log:
        outcomes[x["outcome"]] = outcomes.get(x["outcome"], 0) + 1
    unknown_resent = sum(1 for car_id, req in transport.sent
                         if req["action"] == "connect") - sum(
        1 for x in txn_log if x["outcome"] not in ("NOT_ADMITTED",))
    return {"policy": policy, "timeline": timeline, "decisions": decisions,
            "transactions": outcomes, "connect_requests_sent": sum(
                1 for _, r in transport.sent if r["action"] == "connect"),
            "connects_beyond_one_per_txn": unknown_resent,
            "mean_utility": float(np.mean([s["utility"] for s in timeline])),
            "total_handovers": int(sum(s["handovers"] for s in timeline)),
            "critical_violation_steps": int(sum(s["critical_violations"] > 0
                                                for s in timeline)),
            "final_reputation": {k: round(v, 3) for k, v in rep.scores.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="artifacts/bench/adapters/fleet")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7001, 7002, 7003])
    ap.add_argument("--total", type=float, default=300.0)
    ap.add_argument("--out", default="artifacts/closedloop.json")
    args = ap.parse_args()
    from .policy import WirelessJEV
    engine = WirelessJEV(adapter=args.adapter)
    results = []
    for seed in args.seeds:
        for policy in ("wirelessjev", "milp", "legacy"):
            r = run(policy, seed=seed, total_s=args.total, engine=engine)
            r["seed"] = seed
            results.append(r)
            print(f"seed {seed} {policy}: mean utility {r['mean_utility']:.3f}, "
                  f"handovers {r['total_handovers']}, txns {r['transactions']}", flush=True)
    summary = {}
    for policy in ("wirelessjev", "milp", "legacy"):
        rs = [r for r in results if r["policy"] == policy]
        summary[policy] = {
            "mean_utility": float(np.mean([r["mean_utility"] for r in rs])),
            "total_handovers": int(sum(r["total_handovers"] for r in rs)),
            "critical_violation_steps": int(sum(r["critical_violation_steps"] for r in rs)),
            "connects_beyond_one_per_txn": int(sum(r["connects_beyond_one_per_txn"] for r in rs)),
        }
    out = {"summary": summary, "adapter": args.adapter,
           "hardware": RASPBOT.customer_radio, "runs": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
