"""Pre-registered native-stack campaign (see docs/NATIVE_STACK.md)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .sim import policies, simulate
from .world import build_world

OUT = Path(__file__).resolve().parents[3] / "artifacts" / "native"
TEST_SCENES = tuple(range(500, 506))


def run_scene(seed: int, duration: float) -> list[dict]:
    rows = []
    pol = {p.name: p for p in policies()}
    w = build_world(seed, duration_s=duration, sigma_map=4.0)
    for p in pol.values():
        rows.append(simulate(w, p) | {"arm": p.name})
    w8 = build_world(seed, duration_s=duration, sigma_map=8.0)
    rows.append(simulate(w8, pol["native"]) | {"arm": "native_sigma8"})
    w0 = build_world(seed, duration_s=duration, sigma_map=0.0)
    rows.append(simulate(w0, pol["native"]) | {"arm": "native_oracle"})
    wb = build_world(seed, duration_s=duration, sigma_map=4.0, blind=True)
    rows.append(simulate(wb, pol["native"]) | {"arm": "native_blind"})
    rows.append(simulate(w, replace(pol["legacy"], name="legacy_1s", roam_cost_s=1.0)) | {"arm": "legacy_1s"})
    for name in ("mlo_default", "mlo_split", "native"):     # exploratory stress
        rows.append(simulate(w, pol[name], blockage_rate=4.5) | {"arm": f"{name}_heavy_blockage"})
    return rows


def evaluate(rows: list[dict]) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["scene"]] = r
    scenes = sorted(by["native"])

    def wins(a, b, key="control_miss", better=lambda x, y: x < y):
        return sum(better(by[a][s][key], by[b][s][key]) for s in scenes)

    red = [1 - by["native"][s]["control_miss"] / max(by["mlo_default"][s]["control_miss"], 1e-12)
           for s in scenes]
    air = [by["native"][s]["airtime_domain_sum"] / by["mlo_default"][s]["airtime_domain_sum"] - 1
           for s in scenes]
    h1 = sum(r >= 0.5 and a <= 0.15 for r, a in zip(red, air))
    gain_nat = [by["mlo_default"][s]["control_miss"] - by["native"][s]["control_miss"] for s in scenes]
    gain_blind = [by["mlo_default"][s]["control_miss"] - by["native_blind"][s]["control_miss"] for s in scenes]
    kept = float(np.sum(gain_blind) / max(np.sum(gain_nat), 1e-12))
    return {
        "scenes": scenes,
        "H1": {"scenes_passing": h1, "pass": h1 >= 5, "control_miss_reduction": red,
               "airtime_change": air},
        "H2": {"scenes_better": wins("native", "mlo_default", "control_miss_worst_decile"),
               "pass": wins("native", "mlo_default", "control_miss_worst_decile") >= 5},
        "H3": {"scenes_better": wins("native_sigma8", "mlo_default"),
               "pass": wins("native_sigma8", "mlo_default") >= 5},
        "H4": {"blind_share_of_gain_kept": kept, "pass": kept <= 0.5},
        "H5": {"native_beats_ft11r": wins("native", "ft11r"),
               "native_beats_legacy_1s": wins("native", "legacy_1s"),
               "pass": wins("native", "ft11r") >= 5},
        "post_hoc_baseline_mlo_split": {
            "native_beats_mlo_split": wins("native", "mlo_split"),
            "native_less_airtime_than_mlo_split": wins("native", "mlo_split", "airtime_domain_sum")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--scenes", type=str, default=",".join(map(str, TEST_SCENES)))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = [int(x) for x in args.scenes.split(",")]
    t = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for part in pool.map(run_scene, scenes, [args.duration] * len(scenes)) for r in part]
    report = {"protocol": "docs/NATIVE_STACK.md", "duration_s": args.duration, "elapsed_s": time.time() - t,
              "rows": rows, "evaluation": evaluate(rows)}
    (OUT / "campaign.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["evaluation"], indent=1, default=float))


if __name__ == "__main__":
    main()
