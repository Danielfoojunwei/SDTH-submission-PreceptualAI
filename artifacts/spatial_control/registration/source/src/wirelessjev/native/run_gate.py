"""Trust-gate and safety-broadcast campaign (docs/NATIVE_GATE.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import time

from .broadcast import study
from .run import OUT
from .sim import policies, simulate
from .world import build_world

TEST_SCENES = tuple(range(510, 516))
CONDITIONS = {"sigma8": (8.0, 0.0), "sigma4": (4.0, 0.0), "sigma4_2g_noisy": (4.0, 10.0)}


def arms():
    pol = {p.name: p for p in policies()}
    nat = pol["native"]
    return {"mlo_split": pol["mlo_split"], "native": nat,
            "native_gated": replace(nat, name="native_gated", gate=True),
            "native_gated_overconfident": replace(nat, name="native_gated_overconfident",
                                                  gate=True, belief_scale=0.5),
            "native_gated_z1": replace(nat, name="native_gated_z1", gate=True, gate_z=1.0)}


def run_scene(seed: int, duration: float):
    rows, bcast = [], []
    for cond, (sig, extra) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, sigma_map=sig, hetero=True, noise_2g_extra_db=extra)
        for name, p in arms().items():
            rows.append(simulate(w, p) | {"arm": name, "condition": cond})
        if cond != "sigma8":
            bcast.append(study(w) | {"condition": cond})
    return rows, bcast


def evaluate(rows, bcast):
    def get(cond, arm):
        return {r["scene"]: r for r in rows if r["condition"] == cond and r["arm"] == arm}
    sc = sorted(get("sigma8", "native"))
    m = lambda cond, arm, s, k="control_miss": get(cond, arm)[s][k]
    g1 = sum(m("sigma8", "native_gated", s) < m("sigma8", "mlo_split", s) for s in sc)
    g2a = sum(m("sigma4", "native_gated", s) <= 1.2 * m("sigma4", "native", s) for s in sc)
    g2b = sum(m("sigma4", "native_gated", s) < m("sigma4", "mlo_split", s) for s in sc)
    b = {(x["condition"], x["scene"]): x for x in bcast}
    b1 = sum(b[("sigma4", s)]["layered_0.5"]["miss"] <= b[("sigma4", s)]["multicast_mcs0"]["miss"] for s in sc)
    return {"scenes": sc,
            "G1": {"gated_beats_split_at_sigma8": g1, "pass": g1 >= 5},
            "G2": {"gated_within_1.2x_of_ungated": g2a, "gated_beats_split": g2b,
                   "pass": g2a >= 5 and g2b >= 5},
            "B1": {"layered_0.5_no_worse_than_multicast": b1, "pass": b1 >= 5},
            "C_noisy_2g": {arm: sum(m("sigma4_2g_noisy", arm, s) < m("sigma4_2g_noisy", "mlo_split", s)
                                    for s in sc) for arm in arms() if arm != "mlo_split"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(run_scene, TEST_SCENES, [args.duration] * len(TEST_SCENES)))
    rows = [r for p in parts for r in p[0]]
    bcast = [b for p in parts for b in p[1]]
    rep = {"protocol": "docs/NATIVE_GATE.md", "duration_s": args.duration, "elapsed_s": time.time() - t,
           "rows": rows, "broadcast": bcast, "evaluation": evaluate(rows, bcast)}
    (OUT / "gate_campaign.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["evaluation"], indent=1, default=float))


if __name__ == "__main__":
    main()
