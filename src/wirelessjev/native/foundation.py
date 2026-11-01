"""RadioMap v1 (cross-site prior): a radio prior learned across many sites.

Every warehouse differs in rack density, path-loss slope and shadowing, but
the *way* radio behaves in this class of building transfers. Past deployments
(here: 40 digital twins with randomised physics, fully surveyed) give, per
band, the distribution of the geometry coefficients (intercept, log-distance
slope, loss per rack row) and the typical shadowing strength. A new site's
map starts from that learned prior instead of free-space physics, so a short
survey suffices. Protocol and verdicts: docs/RADIOJEV_FOUNDATION.md.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..twin.warehouse import HEIGHT, RACK_DB, WIDTH, make_warehouse, rack_obstruction
from .radiomap import N_RFF
from .run import OUT
from .world import BANDS, _path_losses, build_world

TRAIN_SEEDS = tuple(range(700, 740))
TEST_SEEDS = tuple(range(600, 606))
SURVEYS_S = (10, 30, 60)


def random_site(seed: int) -> dict:
    rng = np.random.default_rng(90_000 + seed)
    return {"rack_scale": float(rng.uniform(0.5, 1.6)), "pl_coef": float(rng.uniform(28, 38)),
            "shadow_db": float(rng.uniform(2.0, 7.0))}


def site_coefficients(seed: int):
    """Least-squares geometry law per BSS on a dense truth grid of one site."""
    p = random_site(seed)
    scene = make_warehouse(seed, n_robots=1, rack_scale=p["rack_scale"])
    scene.pl_coef, scene.shadow_db = p["pl_coef"], p["shadow_db"]
    aps5 = [a for a in scene.aps if a.band == "5g"]
    aps2 = [a for a in scene.aps if a.band == "2g"]
    gx, gy = np.meshgrid(np.arange(2, WIDTH, 4.0), np.arange(2, HEIGHT, 4.0))
    x, y = gx.ravel(), gy.ravel()
    tx = np.array([aps2[0].tx_power_dbm, aps5[0].tx_power_dbm, aps5[0].tx_power_dbm])
    rssi = tx - _path_losses(scene, aps5, aps2, x, y)                          # (P,S,3)
    coefs, resid = {b: [] for b in range(3)}, {b: [] for b in range(3)}
    for s, a5 in enumerate(aps5):
        d = np.maximum(1.0, np.hypot(x - a5.x, y - a5.y))
        X = np.column_stack([np.ones_like(x), np.log10(d), rack_obstruction(a5, x, y) / RACK_DB["5g"]])
        for b in range(3):
            c, *_ = np.linalg.lstsq(X, rssi[:, s, b], rcond=None)
            coefs[b].append(c)
            resid[b].append(np.std(rssi[:, s, b] - X @ c))
    return coefs, resid


def learn_prior(seeds=TRAIN_SEEDS) -> dict:
    allc = {b: [] for b in range(3)}
    allr = {b: [] for b in range(3)}
    for seed in seeds:
        c, r = site_coefficients(seed)
        for b in range(3):
            allc[b] += c[b]
            allr[b] += r[b]
    prior = {}
    for b in range(3):
        C = np.array(allc[b])
        prior[b] = {"mean": C.mean(0), "var": C.var(0) + 1e-3,
                    "shadow_db": float(np.sqrt(np.mean(np.square(allr[b]))))}
    return prior


def evaluate(prior: dict, seeds=TEST_SEEDS, surveys=SURVEYS_S) -> list[dict]:
    rows = []
    for seed in seeds:
        params = random_site(seed)
        for sv in surveys:
            for name, pr in (("generic", None), ("learned", prior)):
                w = build_world(seed, duration_s=10.0, survey_s=sv, site_params=params, map_prior=pr)
                rows.append({"seed": seed, "survey_s": sv, "prior": name, **params, **w.map_calibration})
                print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=len(TRAIN_SEEDS))
    args = ap.parse_args()
    t = time.time()
    prior = learn_prior(TRAIN_SEEDS[: args.train])
    print("prior", {b: {k: (np.round(v, 2).tolist() if isinstance(v, np.ndarray) else round(v, 2))
                        for k, v in p.items()} for b, p in prior.items()}, flush=True)
    rows = evaluate(prior)
    ser = {b: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
           for b, p in prior.items()}
    (OUT / "foundation.json").write_text(json.dumps(
        {"protocol": "docs/RADIOJEV_FOUNDATION.md", "train_sites": args.train, "prior": ser,
         "rows": rows, "elapsed_s": time.time() - t}, indent=1, default=float))


if __name__ == "__main__":
    main()
