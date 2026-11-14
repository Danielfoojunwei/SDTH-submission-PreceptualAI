"""Offline analysis of recorded laps: outages, the safe-plan question, a replay
of the simulator's blockage rule against real outages, and the labels and
baseline for short-horizon outage prediction.

    python -m wirelessjev.hil.analyse artifacts/onecar/*-lap-*.jsonl

An *outage* is defined from the 20 Hz gateway ping: a gap between replies
longer than `gap_s` (default 0.2 s, the fleet's stop rule). Nothing here is
simulated; laps with no pings produce no outage statistics.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .telemetry import load_lap

PLAN_LENGTHS_S = (0.2, 0.5, 1.0, 2.0)
HORIZONS_S = (0.1, 0.5, 1.0)
# Same rule and settings as the simulator's detector (docs/BLOCKAGE_DETECTOR.md).
DROP_DB, CONFIRM, EARLY_S = 10.0, 2, 1.0


def ping_gaps(lap: dict) -> np.ndarray:
    """(start, duration) of every gap between consecutive ping replies."""
    t = np.sort(np.array([p["t_car"] for p in lap["pings"]], dtype=float))
    if len(t) < 2:
        return np.zeros((0, 2))
    return np.column_stack([t[:-1], np.diff(t)])


def lap_summary(lap: dict, gap_s: float = 0.2) -> dict:
    gaps = ping_gaps(lap)
    span = float(gaps[:, 1].sum()) if len(gaps) else 0.0
    out = {"name": lap["meta"].get("name"), "samples": len(lap["samples"]), "pings": len(lap["pings"]),
           "scans": len(lap["scans"]), "marks": len(lap["marks"]), "ping_span_s": round(span, 2)}
    if not len(gaps):
        return out
    long = gaps[gaps[:, 1] > gap_s]
    out.update({
        "outages": int(len(long)), "longest_gap_s": round(float(gaps[:, 1].max()), 3),
        "outage_time_fraction": round(float(long[:, 1].sum() / span), 5) if span else None,
        # Direction 1: a robot carrying a plan of this length would have stopped this often.
        "stops_if_plan_s": {str(p): int((gaps[:, 1] > p).sum()) for p in PLAN_LENGTHS_S},
        "bssids": sorted({s["bssid"] for s in lap["samples"] if s.get("bssid")}),
        "roams_seen": int(sum(bool(a.get("bssid") and b.get("bssid") and a["bssid"] != b["bssid"])
                              for a, b in zip(lap["samples"], lap["samples"][1:]))),
    })
    return out


def outage_dataset(lap: dict, gap_s: float = 0.2):
    """Per telemetry sample: features known at that moment and, for each
    horizon, whether an outage (reply gap > gap_s) is in progress or starts
    within it. Features use only the past."""
    gaps = ping_gaps(lap)
    long = gaps[gaps[:, 1] > gap_s]
    rows, labels = [], []
    last_sig, last_retry, last_t = None, None, None
    for s in lap["samples"]:
        t = s["t_car"]
        sig = s.get("signal_dbm")
        retry = s.get("tx_retries")
        d_sig = 0.0 if last_sig is None or sig is None else sig - last_sig
        d_retry = 0.0 if last_retry is None or retry is None or t == last_t else (retry - last_retry) / (t - last_t)
        in_gap = bool(len(long) and ((long[:, 0] <= t) & (long[:, 0] + long[:, 1] > t)).any())
        rows.append([sig if sig is not None else -100.0, d_sig, d_retry, float(in_gap)])
        labels.append([bool(len(long) and ((long[:, 0] < t + h) & (long[:, 0] + long[:, 1] > t)).any())
                       for h in HORIZONS_S])
        if sig is not None:
            last_sig = sig
        if retry is not None:
            last_retry, last_t = retry, t
    return np.array(rows, dtype=float).reshape(-1, 4), np.array(labels, dtype=bool).reshape(-1, len(HORIZONS_S))


def detector_replay(lap: dict, drop_db: float = DROP_DB, confirm: int = CONFIRM) -> list[float]:
    """Times at which the simulator's blockage rule would have fired on this lap.

    The rule: the signal is more than `drop_db` below a running baseline of the
    car's own readings for `confirm` consecutive samples. One car reports only
    the link it is using, so this is the single-link form of the rule; the
    "every band at once" check needs a multi-link client. The baseline restarts
    when the car changes AP and is not updated while the signal is low."""
    out, base, bssid, run = [], None, None, 0
    for s in lap["samples"]:
        sig = s.get("signal_dbm")
        if sig is None or not s.get("connected"):
            continue
        if base is None or s.get("bssid") != bssid:
            base, bssid, run = sig, s.get("bssid"), 0
            continue
        if base - sig > drop_db:
            run += 1
            if run >= confirm:
                out.append(s["t_car"])
                run = 0
        else:
            run = 0
            base = 0.8 * base + 0.2 * sig
    return out


def detector_score(lap: dict, gap_s: float = 0.2, **rule) -> dict:
    """Score the replayed rule against real outages (ping reply gaps > gap_s).

    An outage counts as caught if the rule fired from `EARLY_S` before its
    start until its end. `lead_s` is positive when the rule fired before the
    last reply that preceded the outage. Any other firing is a false alarm."""
    fired = np.array(detector_replay(lap, **rule), dtype=float)
    gaps = ping_gaps(lap)
    long = gaps[gaps[:, 1] > gap_s] if len(gaps) else np.zeros((0, 2))
    used = np.zeros(len(fired), dtype=bool)
    leads = []
    for start, dur in long:
        hit = (fired >= start - EARLY_S) & (fired <= start + dur)
        used |= hit
        if hit.any():
            leads.append(float(start - fired[hit].min()))
    t = [s["t_car"] for s in lap["samples"]]
    minutes = (max(t) - min(t)) / 60.0 if len(t) > 1 else 0.0
    return {"fired": int(len(fired)), "outages": int(len(long)), "caught": len(leads),
            "median_lead_s": round(float(np.median(leads)), 3) if leads else None,
            "false_alarms": int((~used).sum()),
            "false_alarms_per_min": round(float((~used).sum() / minutes), 3) if minutes > 0 else None}


def persistence_brier(features: np.ndarray, labels: np.ndarray, base_rate: np.ndarray) -> dict:
    """Baseline any learned outage model must beat: "in an outage now means an
    outage next", otherwise the base rate. Lower Brier is better."""
    out = {}
    for j, h in enumerate(HORIZONS_S):
        p = np.where(features[:, 3] > 0, 1.0, base_rate[j])
        out[str(h)] = {"brier_persistence": float(np.mean((p - labels[:, j]) ** 2)),
                       "brier_base_rate": float(np.mean((base_rate[j] - labels[:, j]) ** 2)),
                       "positives": int(labels[:, j].sum())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("laps", nargs="+")
    ap.add_argument("--gap", type=float, default=0.2)
    args = ap.parse_args(argv)
    laps = [load_lap(p) for p in args.laps]
    report = {"laps": [lap_summary(lap, args.gap) for lap in laps]}
    scores = [detector_score(lap, args.gap) for lap in laps if lap["pings"]]
    if scores:
        tot = {k: sum(s[k] for s in scores) for k in ("fired", "outages", "caught", "false_alarms")}
        report["blockage_rule_replay"] = {"rule": {"drop_db": DROP_DB, "confirm": CONFIRM, "early_window_s": EARLY_S},
                                          "form": "single link (one car reports only the link in use)",
                                          "total": tot, "per_lap": scores}
    data = [outage_dataset(lap, args.gap) for lap in laps if lap["pings"]]
    if data:
        X = np.concatenate([d[0] for d in data])
        Y = np.concatenate([d[1] for d in data])
        report["outage_prediction_baseline"] = persistence_brier(X, Y, Y.mean(0))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
