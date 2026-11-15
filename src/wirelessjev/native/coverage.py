"""Site coverage report from what the robots measured (docs/COVERAGE_REPORT.md).

The fleet's RadioMap exists to plan robot links. The same map answers
questions the site owner has about every other device: where is coverage
weak, where is there no backup AP, and what happens if an AP fails.

    python -m wirelessjev.native.coverage --scene 400 --out artifacts/native/coverage_400

In the simulator the report is scored against ground truth. On a real site
there is no ground truth; the same functions take any `predict(x, y)`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..twin.warehouse import HEIGHT, WIDTH, make_warehouse
from .world import BANDS, _path_losses, build_world

SERVICE_DBM = {"basic": -75.0, "control": -67.0}       # weakest usable level for each service
CELL_M = 2.0
NEAR_ROUTE_M = 3.0                                     # "where the fleet drives": within this of a robot position
CLEAR_DB = 3.0                                         # "clearly" weak or fine: this far from the threshold
UNSURE_DB = 6.0                                        # map uncertainty above this is reported as "not enough data"
# The report states the level a device typically *measures*. That is lower than the mean signal
# power, because readings are averaged in dB over fading and the occasional blocked scan. In the
# simulator's survey: -1.17 dB (Nakagami m=2, dB average) and -0.75 dB (5% of scans 15 dB down).
TYPICAL_MINUS_MEAN_DB = -1.17 - 0.05 * 15.0


def grid(cell_m: float = CELL_M, width: float = WIDTH, height: float = HEIGHT):
    xs = np.arange(cell_m / 2, width, cell_m)
    ys = np.arange(cell_m / 2, height, cell_m)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy                                        # (ny, nx) each


def per_site(level: np.ndarray) -> np.ndarray:
    """(..., S, 3) RSSI per band -> (..., S): each AP's best band."""
    return level.max(axis=-1)


def best_and_backup(site_level: np.ndarray, without: int | None = None):
    """Strongest and second-strongest AP per cell; `without` removes one AP (what if it fails)."""
    lv = site_level.copy()
    if without is not None:
        lv[..., without] = -np.inf
    part = np.sort(lv, axis=-1)
    return part[..., -1], part[..., -2], np.argmax(lv, axis=-1)


def route_mask(gx, gy, positions: np.ndarray, near_m: float = NEAR_ROUTE_M) -> np.ndarray:
    """Cells within `near_m` of anywhere a robot has been: the area the fleet can vouch for."""
    from scipy.spatial import cKDTree
    pts = positions.reshape(-1, 2)
    d = cKDTree(pts[:: max(1, len(pts) // 200_000)]).query(np.column_stack([gx.ravel(), gy.ravel()]))[0]
    return (d <= near_m).reshape(gx.shape)


def holes(weak: np.ndarray, gx, gy, best: np.ndarray, cell_m: float = CELL_M, min_cells: int = 2) -> list[dict]:
    """Connected weak areas, largest first."""
    from scipy import ndimage
    lab, n = ndimage.label(weak)
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= min_cells:
            out.append({"area_m2": float(m.sum() * cell_m ** 2), "centre_m": [round(float(gx[m].mean()), 1), round(float(gy[m].mean()), 1)],
                        "weakest_dbm": round(float(best[m].min()), 1), "typical_dbm": round(float(np.median(best[m])), 1)})
    return sorted(out, key=lambda h: -h["area_m2"])


def report(site_level: np.ndarray, site_std: np.ndarray | None, gx, gy, area: np.ndarray, sites: np.ndarray,
           cell_m: float = CELL_M) -> dict:
    """The site report. site_level, site_std: (ny, nx, S); area: (ny, nx) bool."""
    best, backup, server = best_and_backup(site_level)
    n_area = max(int(area.sum()), 1)
    out = {"area_m2": float(n_area * cell_m ** 2), "aps": int(site_level.shape[-1]), "services": {}}
    if site_std is not None:
        unsure = np.take_along_axis(site_std, server[..., None], axis=-1)[..., 0] > UNSURE_DB
        out["not_enough_data_share"] = float((unsure & area).sum() / n_area)
    for name, thr in SERVICE_DBM.items():
        weak = (best < thr) & area
        no_backup = (backup < thr) & area
        out["services"][name] = {"threshold_dbm": thr, "covered_share": float(1 - weak.sum() / n_area),
                                 "with_backup_share": float(1 - no_backup.sum() / n_area),
                                 "holes": holes(weak, gx, gy, best, cell_m)[:10]}
    # What if one AP fails: how much covered floor is lost (control level).
    thr = SERVICE_DBM["control"]
    base_weak = (best < thr) & area
    fail = []
    for k in range(site_level.shape[-1]):
        b2, _, _ = best_and_backup(site_level, without=k)
        new = ((b2 < thr) & area) & ~base_weak
        fail.append({"ap": k, "xy_m": [float(sites[k, 0]), float(sites[k, 1])], "new_weak_m2": float(new.sum() * cell_m ** 2),
                     "serves_share": float(((server == k) & area).sum() / n_area)})
    out["if_an_ap_fails"] = sorted(fail, key=lambda f: -f["new_weak_m2"])
    return out


def score(pred: np.ndarray, true: np.ndarray, area: np.ndarray) -> dict:
    """Simulator only: how well the report matches ground truth. pred, true: (ny, nx, S)."""
    pb, pk, _ = best_and_backup(pred)
    tb, tk, _ = best_and_backup(true)
    out = {"best_level_bias_db": float(np.mean((pb - tb)[area])),
           "best_level_rmse_db": float(np.sqrt(np.mean((pb - tb)[area] ** 2))),
           "backup_level_rmse_db": float(np.sqrt(np.mean((pk - tk)[area] ** 2)))}

    def pr(p, t):
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        return {"cells_true": tp + fn, "cells_flagged": tp + fp, "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan")}

    for name, thr in SERVICE_DBM.items():
        out[f"weak_{name}"] = pr((pb < thr) & area, (tb < thr) & area)
        out[f"no_backup_{name}"] = pr((pk < thr) & area, (tk < thr) & area)
    # Cells that are clearly one side of the line (3 dB or more): the ones a report must get right.
    thr = SERVICE_DBM["control"]
    clear_weak, clear_fine = (tb < thr - CLEAR_DB) & area, (tb > thr + CLEAR_DB) & area
    out["clearly_weak_found"] = {"cells": int(clear_weak.sum()), "share": float((pb < thr)[clear_weak].mean()) if clear_weak.any() else float("nan")}
    out["clearly_fine_flagged"] = {"cells": int(clear_fine.sum()), "share": float((pb < thr)[clear_fine].mean()) if clear_fine.any() else float("nan")}
    # AP-failure what-ifs, pooled over every AP (control level), and whether the worst failure is ranked high.
    tp = fp = fn = 0
    new_p, new_t = [], []
    base_p, base_t = (pb < thr) & area, (tb < thr) & area
    for k in range(pred.shape[-1]):
        p = (best_and_backup(pred, without=k)[0] < thr) & area
        t = (best_and_backup(true, without=k)[0] < thr) & area
        tp, fp, fn = tp + int((p & t).sum()), fp + int((p & ~t).sum()), fn + int((~p & t).sum())
        new_p.append(int((p & ~base_p).sum()))
        new_t.append(int((t & ~base_t).sum()))
    worst = int(np.argmax(new_t))
    out["worst_ap_failure"] = {"true_worst_ap": worst, "report_rank_of_it": int(np.argsort(-np.array(new_p)).tolist().index(worst)) + 1,
                               "in_report_top3": bool(worst in np.argsort(-np.array(new_p))[:3].tolist())}
    out["weak_if_an_ap_fails"] = {"cells_true": tp + fn, "cells_flagged": tp + fp,
                                  "precision": tp / (tp + fp) if tp + fp else float("nan"),
                                  "recall": tp / (tp + fn) if tp + fn else float("nan")}
    return out


def truth_levels(seed: int, gx, gy, n_robots: int = 100, typical: bool = False) -> np.ndarray:
    """Ground-truth RSSI per AP and band on the grid (simulator only): the mean signal power,
    or with `typical` the level a device typically measures (see TYPICAL_MINUS_MEAN_DB)."""
    scene = make_warehouse(seed, n_robots=n_robots)
    aps5 = [a for a in scene.aps if a.band == "5g"]
    aps2 = [a for a in scene.aps if a.band == "2g"]
    tx = {"2g": aps2[0].tx_power_dbm, "5g": aps5[0].tx_power_dbm, "6g": aps5[0].tx_power_dbm}
    out = np.array([tx[b] for b in BANDS], dtype=np.float32) - _path_losses(scene, aps5, aps2, gx, gy)
    return out + (TYPICAL_MINUS_MEAN_DB if typical else 0.0)


def scene_report(seed: int, survey_s: float = 60.0, drive_s: float = 60.0, cell_m: float = CELL_M) -> dict:
    """Simulator: survey, build the report from the learned map, and score it against truth."""
    w = build_world(seed, duration_s=drive_s, survey_s=survey_s)
    gx, gy = grid(cell_m)
    mean, std = w.rmap.predict(gx, gy)
    area = route_mask(gx, gy, w.pos)
    pred, true = per_site(mean), per_site(truth_levels(seed, gx, gy, w.n, typical=True))
    power = per_site(truth_levels(seed, gx, gy, w.n))
    server = np.argmax(pred, axis=-1)
    site_std = np.take_along_axis(std, np.argmax(mean, axis=-1)[..., None], axis=-1)[..., 0]
    rep = report(pred, site_std, gx, gy, area, w.sites, cell_m)
    return {"scene": seed, "survey_s": survey_s, "cell_m": cell_m, "report": rep, "accuracy": score(pred, true, area),
            "accuracy_against_mean_power": score(pred, power, area),
            "_grids": {"gx": gx, "gy": gy, "pred": pred, "true": true, "area": area, "sites": w.sites, "server": server}}


def figure(res: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = res["_grids"]
    best, backup, _ = best_and_backup(g["pred"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    ext = [0, WIDTH, 0, HEIGHT]
    for ax, data, title in ((axes[0], best, "Strongest AP (dBm)"), (axes[1], backup, "Backup AP (dBm)")):
        im = ax.imshow(np.where(g["area"], data, np.nan), origin="lower", extent=ext, cmap="viridis", vmin=-85, vmax=-45)
        ax.contour(g["gx"], g["gy"], np.where(g["area"], data, 0.0), levels=[SERVICE_DBM["control"]], colors="#E2702F", linewidths=1.5)
        ax.scatter(g["sites"][:, 0], g["sites"][:, 1], marker="^", c="white", edgecolors="black", s=70, zorder=3)
        ax.set_title(title + f"; orange line: {SERVICE_DBM['control']:.0f} dBm")
        ax.set_xlabel("m")
        ax.set_ylabel("m")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Coverage from robot measurements (scene {res['scene']}, {res['survey_s']:.0f} s survey); floor the fleet drives")
    fig.savefig(path, dpi=110)
    plt.close(fig)


COVERAGE_SCENES = tuple(range(610, 616))


def campaign(scenes=COVERAGE_SCENES, survey_s: float = 60.0) -> dict:
    """Pre-registered check of the report against ground truth (docs/COVERAGE_REPORT.md)."""
    rows = []
    for sc in scenes:
        r = scene_report(sc, survey_s)
        rows.append({"scene": sc, "report": r["report"], "accuracy": r["accuracy"],
                     "accuracy_against_mean_power": r["accuracy_against_mean_power"]})
    acc = [r["accuracy"] for r in rows]

    def pooled_pr(key):
        t = sum(a[key]["cells_true"] for a in acc)
        f = sum(a[key]["cells_flagged"] for a in acc)
        tp = sum(a[key]["precision"] * a[key]["cells_flagged"] for a in acc if a[key]["cells_flagged"])
        return {"precision": tp / f if f else float("nan"), "recall": tp / t if t else float("nan")}

    def pooled_share(key):
        n = sum(a[key]["cells"] for a in acc)
        return sum(a[key]["share"] * a[key]["cells"] for a in acc if a[key]["cells"]) / n if n else float("nan")

    rmse = float(np.sqrt(np.mean([a["best_level_rmse_db"] ** 2 for a in acc])))
    backup, weak = pooled_pr("no_backup_control"), pooled_pr("weak_control")
    found, flagged = pooled_share("clearly_weak_found"), pooled_share("clearly_fine_flagged")
    top3 = int(sum(a["worst_ap_failure"]["in_report_top3"] for a in acc))
    ev = {"K1": {"best_level_rmse_db": rmse, "pass": bool(rmse <= 4.0)},
          "K2": {**backup, "pass": bool(backup["precision"] >= 0.8 and backup["recall"] >= 0.9)},
          "K3": {"clearly_weak_found": found, "clearly_fine_flagged": flagged, "pass": bool(found >= 0.6 and flagged <= 0.05)},
          "K4": {**weak, "pass": bool(weak["precision"] >= 0.7 and weak["recall"] >= 0.7)},
          "K5": {"scenes": top3, "pass": bool(top3 >= 5)}}
    return {"protocol": "docs/COVERAGE_REPORT.md", "survey_s": survey_s, "rows": rows, "evaluation": ev}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=int, default=400)
    ap.add_argument("--survey-s", type=float, default=60.0)
    ap.add_argument("--out", default=None, help="path prefix for .json and .png")
    ap.add_argument("--campaign", action="store_true", help="run the pre-registered check on fresh scenes")
    ap.add_argument("--dev", action="store_true", help="with --campaign: use development scenes 400-403 instead")
    args = ap.parse_args(argv)
    if args.campaign:
        from .run import OUT
        res = campaign((400, 401, 402, 403) if args.dev else COVERAGE_SCENES, args.survey_s)
        if not args.dev:
            (OUT / "coverage_report.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res["evaluation"], indent=1))
        return 0
    res = scene_report(args.scene, args.survey_s)
    public = {k: v for k, v in res.items() if not k.startswith("_")}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out + ".json").write_text(json.dumps(public, indent=1))
        figure(res, args.out + ".png")
    print(json.dumps(public, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
