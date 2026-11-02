"""Learned-map campaign (docs/NATIVE_LEARNED_MAP.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import time

import numpy as np

from .run import OUT
from .sim import policies, simulate
from .world import build_world

TEST_SCENES = tuple(range(520, 526))
CONDITIONS = {"survey60": (60, 0.0), "survey300": (300, 0.0), "survey300_2g_noisy": (300, 10.0)}


def arms():
    pol = {p.name: p for p in policies()}
    nat = pol["native"]
    return {"mlo_split": pol["mlo_split"], "native_learned": nat,
            "gated_learned": replace(nat, name="gated_learned", gate=True),
            "gated_learned_x1.5": replace(nat, name="gated_learned_x1.5", gate=True, belief_scale=1.5)}


def run_scene(seed: int, duration: float):
    rows, calib = [], []
    for cond, (sv, extra) in CONDITIONS.items():
        w = build_world(seed, duration_s=duration, survey_s=sv, noise_2g_extra_db=extra)
        calib.append(w.map_calibration | {"scene": seed, "condition": cond})
        for name, p in arms().items():
            rows.append(simulate(w, p) | {"arm": name, "condition": cond})
    return rows, calib


def evaluate(rows, calib):
    def m(cond, arm, s):
        return next(r["control_miss"] for r in rows
                    if r["condition"] == cond and r["arm"] == arm and r["scene"] == s)
    sc = sorted({r["scene"] for r in rows})
    l1 = sum(m("survey300", "gated_learned", s) < m("survey300", "mlo_split", s) for s in sc)
    l2 = sum(m("survey60", "gated_learned", s) < m("survey60", "mlo_split", s) for s in sc)
    l3 = sum(m("survey300_2g_noisy", "gated_learned", s) < m("survey300_2g_noisy", "native_learned", s)
             for s in sc)
    cov = float(np.mean([c["coverage_90"] for c in calib if c["condition"] != "survey300_2g_noisy"]))
    return {"scenes": sc, "L1": {"wins": l1, "pass": l1 >= 5}, "L2": {"wins": l2, "pass": l2 >= 5},
            "L3": {"wins": l3, "pass": l3 >= 5}, "L4": {"coverage_90": cov, "pass": cov >= 0.70}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(run_scene, TEST_SCENES, [args.duration] * len(TEST_SCENES)))
    rows = [r for p in parts for r in p[0]]
    calib = [c for p in parts for c in p[1]]
    rep = {"protocol": "docs/NATIVE_LEARNED_MAP.md", "duration_s": args.duration,
           "elapsed_s": time.time() - t, "rows": rows, "calibration": calib,
           "evaluation": evaluate(rows, calib)}
    (OUT / "learned_campaign.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep["evaluation"], indent=1, default=float))


if __name__ == "__main__":
    main()
