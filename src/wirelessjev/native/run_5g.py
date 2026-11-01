"""Vertical Wi-Fi stack vs private 5G (docs/WIFI_VS_5G.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import time

from .run import OUT
from .sim import Policy, policies, simulate
from .world import build_world

TEST_SCENES = tuple(range(530, 536))
CONDITIONS = {"main": dict(collision_lambda=0.01, interference=True, noise_2g=0.0),
              "stress": dict(collision_lambda=0.03, interference=True, noise_2g=10.0),
              "lenient": dict(collision_lambda=0.0, interference=False, noise_2g=0.0)}


def arms(cond: str) -> dict:
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True)
    vert = replace(nat, online_map=True, power_control=True, cadence_roam=True, adaptive_video=True)
    out = {"nr_30ms": Policy("nr_30ms", "nr", nr_ho_cost_s=0.03),
           "nr_daps": Policy("nr_daps", "nr", nr_ho_cost_s=0.0),
           "mlo_split": pol["mlo_split"], "native_gated": replace(nat, name="native_gated"),
           "vertical_single": replace(vert, name="vertical_single"),
           "vertical_client": replace(vert, name="vertical_client", dual_radio=True),
           "vertical_full": replace(vert, name="vertical_full", dual_radio=True, sched_control=True)}
    if cond == "main":
        base = out["vertical_client"]
        for comp, field in (("online", "online_map"), ("power", "power_control"),
                            ("cadence", "cadence_roam"), ("video", "adaptive_video")):
            out[f"vertical_client-{comp}"] = replace(base, name=f"vertical_client-{comp}", **{field: False})
    return out


def run_scene(seed: int, duration: float):
    rows = []
    for cond, c in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=c["noise_2g"])
        for name, p in arms(cond).items():
            rows.append(simulate(w, p, collision_lambda=c["collision_lambda"],
                                 interference=c["interference"]) | {"arm": name, "condition": cond})
    return rows


def evaluate(rows):
    def v(cond, arm, s, k):
        return next(r[k] for r in rows if r["condition"] == cond and r["arm"] == arm and r["scene"] == s)
    sc = sorted({r["scene"] for r in rows})
    w = lambda cond, a, b, k, le=True: sum((v(cond, a, s, k) <= v(cond, b, s, k)) if le else
                                          (v(cond, a, s, k) >= v(cond, b, s, k)) for s in sc)
    res = {"V1": w("main", "vertical_client", "nr_30ms", "control_miss"),
           "V2": w("main", "vertical_client", "nr_30ms", "availability_1s", le=False),
           "V3": w("main", "vertical_client", "nr_30ms", "safety_stop_fraction"),
           "V4": w("main", "vertical_single", "nr_30ms", "control_miss"),
           "V5": w("stress", "vertical_client", "nr_30ms", "control_miss")}
    return {"scenes": sc, **{k: {"scenes": n, "pass": n >= 5} for k, n in res.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(run_scene, TEST_SCENES, [args.duration] * len(TEST_SCENES))
                for r in part]
    rep = {"protocol": "docs/WIFI_VS_5G.md", "duration_s": args.duration, "elapsed_s": time.time() - t,
           "rows": rows, "evaluation": evaluate(rows)}
    (OUT / "wifi_vs_5g.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["evaluation"], indent=1))


if __name__ == "__main__":
    main()
