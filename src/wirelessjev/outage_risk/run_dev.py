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


def gather(scenes, cond: str, duration: float, cache=None):
    lam, noise = CONDITIONS[cond]
    _, vert = _vert()
    xs, ys, rules = [], [], []
    for seed in scenes:
        f = None if cache is None else Path(cache) / f"telemetry_{seed}_{cond}_{int(duration)}.npz"
        if f is not None and f.exists():
            tr = dict(np.load(f))
        else:
            w = build_world(seed, duration_s=duration, survey_s=60, noise_2g_extra_db=noise)
            tr = ds.collect(w, replace(vert, name="single"), collision_lambda=lam, interference=True)
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(f, **tr)
        x, y, r = ds.examples(tr)
        xs.append(x), ys.append(y), rules.append(r)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(rules)


def compare(train, test) -> dict:
    (xa, ya, _), (xb, yb, rule) = train, test
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
        nets = [SmallNet(seed=s).fit(xa, ya[:, j]).predict(xb) for s in NET_SEEDS]
        row["small_net"] = {"seeds": list(NET_SEEDS), "auc": [auc(p, y) for p in nets], "brier": [brier(p, y) for p in nets],
                            "caught_at_rule_false_alarm_rate":
                                [at_false_alarm_rate(p, y, r["false_alarm_rate"])["caught"] for p in nets]}
        out["horizons"][f"{h:.1f}"] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--cache", help="directory for the simulated telemetry, so models can be rerun without the simulator")
    args = ap.parse_args(argv)
    rep = {"scenes": {"train": TRAIN, "test": TEST}, "seconds": args.seconds, "policy": "single radio, no detector",
           "note": "development scenes only; telemetry comes from runs without a detector in the loop"}
    for cond in CONDITIONS:
        rep[cond] = compare(gather(TRAIN, cond, args.seconds, args.cache), gather(TEST, cond, args.seconds, args.cache))
        for h, row in rep[cond]["horizons"].items():
            ru, lw, sn = row["rule"], row["logistic_window"], row["small_net"]
            print(f"{cond} +{h}s blocked {row['blocked_share']:.4f} | rule caught {ru['caught']:.3f} "
                  f"false alarms {ru['false_alarm_rate']:.5f} brier {ru['brier']:.5f} | logistic caught "
                  f"{lw['at_rule_false_alarm_rate']['caught']:.3f} brier {lw['brier']:.5f} | net caught "
                  f"{[round(c, 3) for c in sn['caught_at_rule_false_alarm_rate']]} brier "
                  f"{[round(b, 5) for b in sn['brier']]}", flush=True)
    (OUT / "outage_risk_dev.json").write_text(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
