"""Cross-AP diversity without extra hardware, vs private 5G and Wi-Fi 8
(docs/CROSS_AP.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import time

import numpy as np

from .run import OUT
from .sim import Policy, policies, simulate
from .world import build_world

TEST_SCENES = tuple(range(540, 546))
CONDITIONS = {"main": dict(lam=0.01, noise=0.0, heavy=False),
              "stress": dict(lam=0.03, noise=10.0, heavy=False),
              "heavy": dict(lam=0.01, noise=0.0, heavy=True)}


def arms() -> dict:
    pol = {p.name: p for p in policies()}
    vert = replace(pol["native"], gate=True, online_map=True, power_control=True,
                   cadence_roam=True, adaptive_video=True)
    return {"nr_30ms": Policy("nr_30ms", "nr", nr_ho_cost_s=0.03),
            "nr_daps": Policy("nr_daps", "nr", nr_ho_cost_s=0.0),
            "wifi8": Policy("wifi8", "wifi8", roam_cost_s=0.0, scan_period_s=30.0),
            "mlo_split": pol["mlo_split"],
            "vertical_single": replace(vert, name="vertical_single"),
            "vertical_shared": replace(vert, name="vertical_shared", dual_radio=True, radio2="shared"),
            "vertical_onboard": replace(vert, name="vertical_onboard", dual_radio=True, radio2="onboard"),
            "vertical_client": replace(vert, name="vertical_client", dual_radio=True),
            "vertical_shared-levers": replace(pol["native"], name="vertical_shared-levers", gate=True,
                                              dual_radio=True, radio2="shared")}


def heavy(world):
    """15% of robots stream 6 Mb/s video; bulk uploads doubled."""
    rng = np.random.default_rng(60_000 + world.seed)
    video = world.video_mbps.copy()
    extra = rng.choice(world.n, int(0.15 * world.n), replace=False)
    video[extra] = np.maximum(video[extra], 6.0)
    world.video_mbps = video
    world.bulk_mbps = world.bulk_mbps * 2.0
    return world


def run_scene(seed: int, duration: float, robots: int = 100):
    rows = []
    for cond, c in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=c["noise"],
                        n_robots=robots)
        if c["heavy"]:
            w = heavy(w)
        for name, p in arms().items():
            r = simulate(w, p, collision_lambda=c["lam"], interference=True)
            r.pop("per_robot_control_miss", None)
            r.pop("per_robot_video_on_time", None)
            rows.append(r | {"arm": name, "condition": cond})
    return rows


def evaluate(rows):
    def v(cond, arm, s, k="control_miss"):
        return next(r[k] for r in rows if r["condition"] == cond and r["arm"] == arm and r["scene"] == s)
    sc = sorted({r["scene"] for r in rows})
    le = lambda cond, a, b: int(sum(v(cond, a, s) <= v(cond, b, s) for s in sc))
    res = {"X1": le("main", "vertical_shared", "nr_30ms"),
           "X2": le("main", "vertical_onboard", "nr_30ms"),
           "X3": le("main", "vertical_shared", "wifi8"),
           "X4": le("stress", "vertical_shared", "nr_30ms"),
           "X5": le("stress", "vertical_onboard", "nr_30ms"),
           "X6": le("heavy", "vertical_shared", "nr_30ms")}
    return {"scenes": sc, **{k: {"scenes": n, "pass": n >= 5} for k, n in res.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--robots", type=int, default=100)
    args = ap.parse_args()
    t = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(run_scene, TEST_SCENES, [args.duration] * len(TEST_SCENES),
                                       [args.robots] * len(TEST_SCENES)) for r in part]
    rep = {"protocol": "docs/CROSS_AP.md", "robots": args.robots, "duration_s": args.duration,
           "elapsed_s": time.time() - t,
           "rows": rows, "evaluation": evaluate(rows)}
    name = "cross_ap.json" if args.robots == 100 else f"cross_ap_{args.robots}robots.json"
    (OUT / name).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["evaluation"], indent=1, default=int))


if __name__ == "__main__":
    main()
