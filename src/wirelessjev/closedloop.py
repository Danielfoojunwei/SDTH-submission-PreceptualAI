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
from dataclasses import dataclass, field, replace
from typing import Callable
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


@dataclass
class Scenario:
    scene_fn: Callable = field(default=lambda seed: make_scene(seed, hotspot=True))
    schedule: tuple = ((0.0, "protect_video"), (150.0, "convoy_continuity"))
    missions: dict = field(default_factory=lambda: dict(MISSIONS))
    bad_car: str = "car-10"
    bad_from_s: float = 120.0
    outage_s: tuple = (180.0, 210.0)
    max_in_flight: int = 4
    decide_every: int = 6            # policy steps (5 s) per fleet decision


TEN_CAR = Scenario()


def warehouse_scenario() -> Scenario:
    from .missions import WAREHOUSE_MISSIONS
    from .twin.warehouse import make_warehouse
    return Scenario(scene_fn=make_warehouse,
                    schedule=((0.0, "peak_picking"), (200.0, "teleop_assist"),
                              (400.0, "map_rollout")),
                    missions=dict(WAREHOUSE_MISSIONS), bad_car="amr-090",
                    bad_from_s=300.0, outage_s=(450.0, 480.0), max_in_flight=12)


def mission_at(t: float, sc: Scenario):
    name = [m for start, m in sc.schedule if t >= start][-1]
    return sc.missions[name]


def run(policy: str, *, seed: int, total_s: float, engine=None, log=print,
        scenario: Scenario | None = None, record: bool = False,
        intent_reader=None) -> dict:
    """record=True also stores per-step replay frames (robot positions,
    associations, satisfaction) and full decision details for the Studio."""
    sc = scenario or TEN_CAR
    scene = sc.scene_fn(seed)
    frames = []
    channels = {a.ap_id: a.channel for a in scene.aps}
    domains = scene.domains(channels)
    domain_of = {ap: f"d{i}" for i, d in enumerate(domains) for ap in d}
    budgets = {f"d{i}": 1.0 - scene.occupancy(scene.ap(next(iter(d))), channels)
               for i, d in enumerate(domains)}
    actual = {c.car_id: c.current_ap for c in scene.cars}
    transport = DryRunTransport(state=dict(actual), failure_rate=0.05,
                                unknown_rate=0.05, seed=seed)
    ledger = Ledger(budgets=budgets, max_in_flight=sc.max_in_flight)
    rep = Reputation()
    timeline, decisions, txn_log = [], [], []
    n_steps = int(total_s / STEP)
    decide_every = sc.decide_every
    for k in range(n_steps):
        t = k * STEP
        mission = mission_at(t, sc)
        coordinator_up = not (sc.outage_s[0] <= t < sc.outage_s[1])

        # misbehaving car from bad_from_s
        if t >= sc.bad_from_s:
            scene = scene._copy(cars=[replace(c, tx_power_dbm=24.0) if c.car_id == sc.bad_car
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
        if coordinator_up and policy == "wirelessjev_fast":
            # Intent read once per mission text (cached); plan every step.
            from .fastplan import FastPlanner
            reading = intent_reader.read(mission.intent)
            started = time.perf_counter()
            fp = FastPlanner(scene, pinned=pinned)
            a, u, st = fp.search(reading.weights, reading.focus, starts=[fp.ev.current],
                                 max_iter=60)
            target = fp.ev.to_assoc(a)
            decisions.append({"t": t, "mission": mission.mission_id,
                              "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                              "intent_ms": round(reading.latency_ms, 1),
                              "moves": int((a != fp.ev.current).sum()),
                              "iterations": st["iterations"]})
        elif coordinator_up and k % decide_every == 0:
            started = time.perf_counter()
            if policy == "wirelessjev":
                options, _ = generate(scene, seed=seed + k, pinned=pinned,
                                      missions=sc.missions)
                d = engine.score(scene, mission, options)
                chosen = next(o for o in options if o.option_id == d.option_id)
                target = None if chosen.is_defer else chosen.assoc
                entry = {"t": t, "mission": mission.mission_id,
                         "choice": d.option_id, "source": chosen.source,
                         "latency_ms": round(d.latency_ms, 1),
                         "pinned": sorted(pinned)}
                if record:
                    from .candidates import oracle_scores
                    scores = oracle_scores(scene, options, mission)
                    entry.update(
                        intent=mission.intent,
                        options=[{"id": o.option_id, "text": o.text,
                                  "p_true": round(d.probabilities[o.option_id], 4),
                                  "oracle": round(scores[o.option_id], 4),
                                  "moves": sum(o.assoc[c] != actual[c] for c in actual)}
                                 for o in options],
                        oracle_best=max(scores, key=scores.get),
                        regret=round(max(scores.values()) - scores[d.option_id], 4))
                decisions.append(entry)
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
        if record:
            r = scene.radio()
            frames.append({
                "t": t,
                "xy": [[round(c.x, 1), round(c.y, 1)] for c in scene.cars],
                "ap": [r.ap_index[new_assoc[c.car_id]] for c in scene.cars],
                "rssi": [round(float(r.dl[i, r.ap_index[new_assoc[c.car_id]]]), 1)
                         for i, c in enumerate(scene.cars)],
                "sat": [round(ev.satisfaction[c.car_id], 2) for c in scene.cars],
                "quarantined": sorted(pinned),
            })
        scene = scene.with_associations(new_assoc).advance(STEP)

    outcomes = {}
    for x in txn_log:
        outcomes[x["outcome"]] = outcomes.get(x["outcome"], 0) + 1
    unknown_resent = sum(1 for car_id, req in transport.sent
                         if req["action"] == "connect") - sum(
        1 for x in txn_log if x["outcome"] not in ("NOT_ADMITTED",))
    extra = {}
    if record:
        extra = {"frames": frames,
                 "robots": [c.car_id for c in scene.cars],
                 "aps": [{"id": a.ap_id, "x": a.x, "y": a.y, "band": a.band,
                          "channel": a.channel, "site": a.site} for a in scene.aps],
                 "occupied_channels": [[b, ch] for (b, ch), v in
                                       scene.external_occupancy.items() if v > 0],
                 "txn_log": txn_log}
    return {**extra, "policy": policy, "timeline": timeline, "decisions": decisions,
            "transactions": outcomes, "connect_requests_sent": sum(
                1 for _, r in transport.sent if r["action"] == "connect"),
            "connects_beyond_one_per_txn": unknown_resent,
            "mean_utility": float(np.mean([s["utility"] for s in timeline])),
            "total_handovers": int(sum(s["handovers"] for s in timeline)),
            "mean_critical_violations": float(np.mean([s["critical_violations"]
                                                       for s in timeline])),
            "critical_violation_steps": int(sum(s["critical_violations"] > 0
                                                for s in timeline)),
            "final_reputation": {k: round(v, 3) for k, v in rep.scores.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="artifacts/bench/adapters/fleet")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7001, 7002, 7003])
    ap.add_argument("--total", type=float, default=300.0)
    ap.add_argument("--out", default="artifacts/closedloop.json")
    ap.add_argument("--warehouse", action="store_true",
                    help="100-robot AMR warehouse shift instead of the 10-car scenario")
    ap.add_argument("--policies", nargs="+", default=["wirelessjev", "milp", "legacy"])
    ap.add_argument("--intent-adapter", default=None,
                    help="distilled score readout for wirelessjev_fast")
    args = ap.parse_args()
    scenario = warehouse_scenario() if args.warehouse else TEN_CAR
    from .policy import WirelessJEV
    engine = WirelessJEV(adapter=args.adapter if "wirelessjev" in args.policies else None)
    intent_reader = None
    if "wirelessjev_fast" in args.policies:
        from .intent import IntentReader
        readout = None
        if args.intent_adapter:
            from safetensors.torch import load_file
            readout = load_file(str(Path(args.intent_adapter) / "readout.safetensors"))["readout"]
        intent_reader = IntentReader(engine, readout=readout)
    results = []
    for seed in args.seeds:
        for policy in args.policies:
            r = run(policy, seed=seed, total_s=args.total, engine=engine,
                    scenario=scenario, record=(seed == args.seeds[0]),
                    intent_reader=intent_reader)
            r["seed"] = seed
            results.append(r)
            print(f"seed {seed} {policy}: mean utility {r['mean_utility']:.3f}, "
                  f"handovers {r['total_handovers']}, txns {r['transactions']}", flush=True)
    summary = {}
    for policy in args.policies:
        rs = [r for r in results if r["policy"] == policy]
        summary[policy] = {
            "mean_utility": float(np.mean([r["mean_utility"] for r in rs])),
            "total_handovers": int(sum(r["total_handovers"] for r in rs)),
            "critical_violation_steps": int(sum(r["critical_violation_steps"] for r in rs)),
            "mean_critical_violations": float(np.mean([r["mean_critical_violations"] for r in rs])),
            "connects_beyond_one_per_txn": int(sum(r["connects_beyond_one_per_txn"] for r in rs)),
            "decision_latency_ms_mean": float(np.mean([d["latency_ms"] for r in rs
                                                       for d in r["decisions"]] or [0.0])),
            "decisions": int(sum(len(r["decisions"]) for r in rs)),
        }
    out = {"summary": summary, "adapter": args.adapter,
           "hardware": RASPBOT.customer_radio, "runs": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
