"""Site transfer for foresight (docs/SITE_TRANSFER.md): a new site with a
different AP height. Does a predictor learned from that site's own radio
readings beat a hand-written rule that still carries the old site's geometry?

    python -m wirelessjev.outage_risk.run_transfer collect --scene 400
    python -m wirelessjev.outage_risk.run_transfer train
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np

from ..native.blockers import REACH_FRAC, blocker_tracks, reach_frac_for
from ..native.run import OUT
from ..native.run_fleet import CONDITIONS, _vert
from ..native.sim import BLOCK_SLOTS, SLOT
from ..native.world import build_world
from . import dataset as ds
from . import foresight as fo
from .run_foresight import TRAIN_SCENES, block_positions

NEW_SITE_AP_HEIGHT_M = 10.0
NEW_SITE_FRAC = reach_frac_for(NEW_SITE_AP_HEIGHT_M)        # 0.227, against 0.386 at the old site
MODEL_RADIO = OUT / "foresight_model_newsite_radio.pt"      # learned from the new site's radio rule
MODEL_TRUE = OUT / "foresight_model_newsite_true.pt"        # learned from the new site's true blockage (reference)
COLLECT_S = 300.0


def telemetry_path(scene: int):
    return OUT / f"transfer_telemetry_{scene}.npz"


def collect(scene: int, duration: float = COLLECT_S) -> dict:
    """Drive the new site with no foresight and keep what the robots saw."""
    _, vert = _vert()
    lam, noise = CONDITIONS["main"]
    w = build_world(scene, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
    tr = ds.collect(w, replace(vert, name="single"), collision_lambda=lam, interference=True, blockage_model="moving",
                    reach_frac=NEW_SITE_FRAC)
    np.savez_compressed(telemetry_path(scene), sites=w.sites.astype(np.float32), **tr)
    return {"scene": scene, "ticks": int(len(tr["slot"])), "blocked_share": float(tr["blocked"][:, :, 0].mean())}


def train(scenes=TRAIN_SCENES, seed: int = 0) -> dict:
    xr, yr, xt, yt = [], [], [], []
    for sc in scenes:
        tr = dict(np.load(telemetry_path(sc)))
        sites = tr.pop("sites")
        blocks = tr["slot"] // BLOCK_SLOTS
        tracks = blocker_tracks(sc, int(blocks.max()) + 1)[blocks]
        sensed = fo.sense(tr["pos"], tracks, sc)
        x, y = fo.radio_label_rows(tr, sites, sensed)
        xr.append(x), yr.append(y)
        x, y = fo.training_rows(tr["pos"], sites, sensed, fo.truth_blocked(tr["pos"], sites, tracks, NEW_SITE_FRAC))
        xt.append(x), yt.append(y)
    xr, yr, xt, yt = (np.concatenate(a) for a in (xr, yr, xt, yt))
    fo.LearnedForesight(seed=seed).fit(xr, yr).save(MODEL_RADIO)
    fo.LearnedForesight(seed=seed).fit(xt, yt).save(MODEL_TRUE)
    return {"radio_label_rows": int(len(xr)), "radio_label_positive_share": float(yr.mean()),
            "true_label_rows": int(len(xt)), "true_label_positive_share": float(yt.mean())}


RADIO_THRESHOLD = 0.5            # set on tuning scenes 402-403 before the fresh-scene test (docs/SITE_TRANSFER.md)
SOURCES = ("old_learned", "rule_old_geometry", "rule_new_geometry", "new_radio", "new_true", "truth")


def warnings_for(world, duration: float) -> dict:
    """Every warning source at the new site, as (blocks, N, S) arrays."""
    from .run_foresight import foresights_for, frozen_model
    n_blocks = int(round(duration / SLOT)) // BLOCK_SLOTS + 1
    pos, sites = block_positions(world, n_blocks), world.sites.astype(np.float32)
    sensed = fo.sense(pos, blocker_tracks(world.seed, n_blocks), world.seed)
    radio, true = fo.LearnedForesight.load(MODEL_RADIO), fo.LearnedForesight.load(MODEL_TRUE)
    radio.threshold, true.threshold = RADIO_THRESHOLD, 0.8
    return {"old_learned": frozen_model().foresight(pos, sites, sensed),
            "rule_old_geometry": fo.geometric_foresight(pos, sites, sensed, frac=REACH_FRAC),
            "rule_new_geometry": fo.geometric_foresight(pos, sites, sensed, frac=NEW_SITE_FRAC),
            "new_radio": radio.foresight(pos, sites, sensed), "new_true": true.foresight(pos, sites, sensed),
            "truth": foresights_for(world, duration, None, sources=("truth",), reach_frac=NEW_SITE_FRAC)["truth"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("collect", "train"))
    ap.add_argument("--scene", type=int, default=400)
    args = ap.parse_args(argv)
    print(json.dumps(collect(args.scene) if args.cmd == "collect" else train(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
