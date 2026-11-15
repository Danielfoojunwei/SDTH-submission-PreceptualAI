"""Foresight from obstacle sensing: train the predictor, score it, and run it
in the control loop (docs/FORESIGHT.md). Development scenes only.

    python -m wirelessjev.outage_risk.run_foresight train
    python -m wirelessjev.outage_risk.run_foresight loop --scene 402 --condition main
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np

from ..native.blockers import blocker_tracks, truth_foresight
from ..native.run import OUT
from ..native.run_fleet import CONDITIONS, _vert
from ..native.sim import BLOCK_SLOTS, SLOT, Policy, hidden_draws, simulate
from ..native.world import POS_DT, build_world
from . import foresight as fo

TRAIN_SCENES, TEST_SCENES = (400, 401), (402, 403)
MODEL = OUT / "foresight_model.pt"
SOURCES = ("truth", "geometric_site_known", "geometric_whole_line", "learned")
# Frozen for the fresh-scene test (docs/FORESIGHT.md): chosen on development scenes 402-403.
STRICT_THRESHOLD, MAX_MOVE_M = 0.8, 20.0


def block_positions(world, n_blocks: int) -> np.ndarray:
    idx = np.minimum((np.arange(n_blocks) * (BLOCK_SLOTS * SLOT / POS_DT)).astype(int), world.pos.shape[0] - 1)
    return world.pos[idx].astype(np.float32)


def foresights_for(world, duration: float, model: fo.LearnedForesight | None = None, sources=SOURCES) -> dict:
    """Every warning source for one world, as (blocks, N, S) arrays the simulator can read."""
    n_blocks = int(round(duration / SLOT)) // BLOCK_SLOTS + 1
    pos, sites = block_positions(world, n_blocks), world.sites.astype(np.float32)
    tracks = blocker_tracks(world.seed, n_blocks)
    out = {}
    if "truth" in sources:
        _, blocked, _ = hidden_draws(world, int(round(duration / SLOT)), blockage_model="moving")
        out["truth"] = truth_foresight(blocked)
    sensed = fo.sense(pos, tracks, world.seed)
    if "geometric_site_known" in sources:
        out["geometric_site_known"] = fo.geometric_foresight(pos, sites, sensed)
    if "geometric_whole_line" in sources:
        out["geometric_whole_line"] = fo.geometric_foresight(pos, sites, sensed, frac=1.0)
    if "learned" in sources and model is not None:
        out["learned"] = model.foresight(pos, sites, sensed)
    return out


def frozen_model() -> fo.LearnedForesight:
    """The predictor trained on scenes 400-401, with the stricter warning threshold."""
    m = fo.LearnedForesight.load(MODEL)
    m.threshold = STRICT_THRESHOLD
    return m


def train(duration: float = 120.0, seed: int = 0) -> dict:
    xs, ys = [], []
    for sc in TRAIN_SCENES:
        w = build_world(sc, duration_s=duration, survey_s=60)
        n_blocks = int(round(duration / SLOT)) // BLOCK_SLOTS + 1
        pos, sites, tracks = block_positions(w, n_blocks), w.sites.astype(np.float32), blocker_tracks(sc, n_blocks)
        x, y = fo.training_rows(pos, sites, fo.sense(pos, tracks, sc), fo.truth_blocked(pos, sites, tracks))
        xs.append(x), ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    model = fo.LearnedForesight(seed=seed).fit(x, y)
    model.save(MODEL)
    return {"rows": int(len(x)), "positive_share": float(y.mean()), "model": str(MODEL)}


def loop(scene: int, cond: str, duration: float = 120.0) -> list:
    lam, noise = CONDITIONS[cond]
    _, vert = _vert()
    shared = replace(vert, dual_radio=True, radio2="shared")
    w = build_world(scene, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
    fs = foresights_for(w, duration, fo.LearnedForesight.load(MODEL))
    truth = fs["truth"]
    rows = []

    def run(name, p, f=None, source=None):
        r = simulate(w, replace(p, name=name), collision_lambda=lam, interference=True, blockage_model="moving", foresight=f)
        row = {"scene": scene, "condition": cond, "arm": name, "source": source, "control_miss": r["control_miss"],
               "control_misses": r["control_misses"], "control_sent": r["control_sent"], "availability_1s": r["availability_1s"],
               "foresight_moves": r.get("foresight_moves", 0), "miss_causes": r.get("miss_causes")}
        if f is not None:
            row["warning_quality_all_paths"] = fo.score(f[:-10], truth[:-10])
        rows.append(row)
        print(json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items() if k != "warning_quality_all_paths"}), flush=True)

    run("nr_30ms", Policy("nr_30ms", "nr", nr_ho_cost_s=0.03))
    for base_name, base in (("single", vert), ("shared", shared)):
        run(base_name, base)
        for src, f in fs.items():
            run(f"{base_name}+foresee", replace(base, foresee=True), f, src)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("train", "loop"))
    ap.add_argument("--scene", type=int, default=402)
    ap.add_argument("--condition", default="main", choices=tuple(CONDITIONS))
    ap.add_argument("--seconds", type=float, default=120.0)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        print(json.dumps(train(args.seconds), indent=1))
    else:
        rows = loop(args.scene, args.condition, args.seconds)
        (OUT / f"foresight_dev_{args.scene}_{args.condition}.json").write_text(json.dumps(rows, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
