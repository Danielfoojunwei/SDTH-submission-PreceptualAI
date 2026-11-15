"""Development comparison: the hand-written blockage rule against a logistic
detector trained on the same telemetry. Development scenes only (400-403);
nothing here is a confirmation.

    python -m wirelessjev.outage_risk.run_dev
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ..native.run import OUT
from ..native.run_fleet import CONDITIONS, _vert
from ..native.world import build_world
from . import dataset as ds
from .models import Logistic, SmallNet, at_false_alarm_rate, auc, brier, yes_no

TRAIN, TEST = (400, 401), (402, 403)
NET_SEEDS = (0, 1, 2)


def gather(scenes, cond: str, duration: float, cache=None, model: str = "independent"):
    """Own-link features, the same rows with the fleet's reports appended, labels and the rule."""
    lam, noise = CONDITIONS[cond]
    _, vert = _vert()
    xs, xfs, ys, rules = [], [], [], []
    for seed in scenes:
        f = None if cache is None else Path(cache) / f"telemetry_{model}_{seed}_{cond}_{int(duration)}.npz"
        if f is not None and f.exists():
            tr = dict(np.load(f))
        else:
            w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
            tr = ds.collect(w, replace(vert, name="single"), collision_lambda=lam, interference=True,
                            blockage_model=model)
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f, **tr)
        x, y, r = ds.examples(tr)
        xs.append(x), ys.append(y), rules.append(r), xfs.append(ds.examples(tr, fleet=True)[0])
    return np.concatenate(xs), np.concatenate(xfs), np.concatenate(ys), np.concatenate(rules)


def sensing_check(scenes, cond: str, duration: float, cache) -> dict:
    """Moving blockers only: the radio rule against a predictor that uses the robot's own obstacle sensing."""
    from ..native.blockers import blocker_tracks
    from ..native.sim import TELEMETRY_AHEAD_BLOCKS
    from ..twin.warehouse import AP_SITES
    sites = np.array(AP_SITES, dtype=np.float32)
    ys, rules, alarms, los = [], [], {h: [] for h in TELEMETRY_AHEAD_BLOCKS}, {h: [] for h in TELEMETRY_AHEAD_BLOCKS}
    for seed in scenes:
        tr = dict(np.load(Path(cache) / f"telemetry_moving_{seed}_{cond}_{int(duration)}.npz"))
        tracks = blocker_tracks(seed, int(duration / 0.1) + 1)[tr["slot"] // 10]
        _, y, r = ds.examples(tr)
        ys.append(y), rules.append(r)
        for h in TELEMETRY_AHEAD_BLOCKS:
            alarms[h].append(ds.sensed_blocker_alarm(tr, tracks, sites, h))
            los[h].append(ds.sensed_blocker_alarm(tr, tracks, sites, h, line_of_sight=True))
    y, rule = np.concatenate(ys), np.concatenate(rules)
    return {f"{0.1 * h:.1f}": {"radio_rule": yes_no(rule, y[:, j]), "sensed_blockers_10m": yes_no(np.concatenate(alarms[h]), y[:, j]),
                                "sensed_blockers_10m_line_of_sight": yes_no(np.concatenate(los[h]), y[:, j])}
            for j, h in enumerate(TELEMETRY_AHEAD_BLOCKS)}


def compare(train, test) -> dict:
    (xa, xfa, ya, _), (xb, xfb, yb, rule) = train, test
    out = {"train_rows": int(len(xa)), "test_rows": int(len(xb)), "horizons": {}}
    newest = slice(0, ds.PER_TICK)                      # the latest tick only: no history
    for j, h in enumerate(ds.HORIZONS_S):
        y = yb[:, j]
        r = yes_no(rule, y)
        base = float(ya[:, j].mean())
        # The rule as a probability: the blocked share of training rows in each of its two states.
        p_rule = np.where(rule, r["precision"] if np.isfinite(r["precision"]) else base, base)
        full = Logistic().fit(xa, ya[:, j]).predict(xb)
        last = Logistic().fit(xa[:, newest], ya[:, j]).predict(xb[:, newest])
        row = {"blocked_share": float(y.mean()), "rule": r | {"brier": brier(p_rule, y)},
               "brier_base_rate": brier(np.full(len(y), base), y)}
        for name, p in (("logistic_window", full), ("logistic_last_tick", last)):
            row[name] = {"auc": auc(p, y), "brier": brier(p, y),
                         "at_rule_false_alarm_rate": at_false_alarm_rate(p, y, r["false_alarm_rate"])}
        pf = Logistic().fit(xfa, ya[:, j]).predict(xfb)
        row["logistic_fleet"] = {"auc": auc(pf, y), "brier": brier(pf, y),
                                 "at_rule_false_alarm_rate": at_false_alarm_rate(pf, y, r["false_alarm_rate"])}
        for name, a, b in (("small_net", xa, xb), ("small_net_fleet", xfa, xfb)):
            nets = [SmallNet(seed=s).fit(a, ya[:, j]).predict(b) for s in NET_SEEDS]
            row[name] = {"seeds": list(NET_SEEDS), "auc": [auc(p, y) for p in nets], "brier": [brier(p, y) for p in nets],
                         "caught_at_rule_false_alarm_rate":
                             [at_false_alarm_rate(p, y, r["false_alarm_rate"])["caught"] for p in nets]}
        out["horizons"][f"{h:.1f}"] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--blockage-model", default="independent", choices=("independent", "moving"))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--sensing-only", action="store_true",
                    help="moving blockers, cached telemetry: score a predictor that senses blockers within 10 m")
    ap.add_argument("--cache", help="directory for the simulated telemetry, so models can be rerun without the simulator")
    args = ap.parse_args(argv)
    rep = {"scenes": {"train": TRAIN, "test": TEST}, "seconds": args.seconds, "policy": "single radio, no detector",
           "blockage_model": args.blockage_model,
           "note": "development scenes only; telemetry comes from runs without a detector in the loop"}
    conds = args.conditions.split(",")
    if args.sensing_only:
        out = {c: sensing_check(TEST, c, args.seconds, args.cache) for c in conds}
        (OUT / "outage_risk_dev_moving_sensing.json").write_text(json.dumps(out, indent=1, default=float))
        print(json.dumps(out, indent=1, default=float))
        return 0
    for cond in conds:
        rep[cond] = compare(gather(TRAIN, cond, args.seconds, args.cache, args.blockage_model),
                            gather(TEST, cond, args.seconds, args.cache, args.blockage_model))
        for h, row in rep[cond]["horizons"].items():
            ru, lw, sn, lf, sf = (row[k] for k in ("rule", "logistic_window", "small_net", "logistic_fleet", "small_net_fleet"))
            print(f"{cond} +{h}s blocked {row['blocked_share']:.4f} | rule caught {ru['caught']:.3f} "
                  f"false alarms {ru['false_alarm_rate']:.5f} brier {ru['brier']:.5f} | logistic caught "
                  f"{lw['at_rule_false_alarm_rate']['caught']:.3f} brier {lw['brier']:.5f} | net caught "
                  f"{[round(c, 3) for c in sn['caught_at_rule_false_alarm_rate']]} brier "
                  f"{[round(b, 5) for b in sn['brier']]} || with fleet reports: logistic caught "
                  f"{lf['at_rule_false_alarm_rate']['caught']:.3f} brier {lf['brier']:.5f} | net caught "
                  f"{[round(c, 3) for c in sf['caught_at_rule_false_alarm_rate']]} brier "
                  f"{[round(b, 5) for b in sf['brier']]}", flush=True)
    name = "outage_risk_dev" + ("" if args.blockage_model == "independent" else "_" + args.blockage_model)
    name += "" if len(conds) == len(CONDITIONS) else "_" + "_".join(conds)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
