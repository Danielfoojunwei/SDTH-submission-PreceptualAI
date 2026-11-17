"""The site map: where the car was, what it heard, where it roamed and where the link dropped.

Positions come from the operator's waypoint marks, interpolated in time (the same rule as the
coverage report and the 10-robot replay): readings before the first mark or after the last one
have no position. While a lap records, the car's current place after its last mark is shown as
an estimate (progress towards the next waypoint at the speed driven so far) and is labelled so.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from ..hil import telemetry as tel
from ..hil import viewer

MAX_POINTS = 2500
MAX_SERIES = 1500


def _split(events: list[dict]) -> dict:
    head = next((e for e in events if e.get("event") == "lap_start"), {})
    return {"meta": head, "end": next((e for e in events if e.get("event") == "lap_end"), {}),
            "samples": [e for e in events if e.get("event") == "sample" and "t_car" in e],
            "scans": [e for e in events if e.get("event") == "scan"],
            "pings": sorted((e for e in events if e.get("event") == "ping" and "t_car" in e), key=lambda e: e["t_car"]),
            "marks": [e for e in events if e.get("event") == "mark"]}


def _thin(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def _r(v, n=2):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)


def channel_of(freq):
    if not freq:
        return None
    f = float(freq)
    if 2412 <= f <= 2484:
        return 14 if f == 2484 else int((f - 2407) // 5)
    if 5000 <= f <= 5900:
        return int((f - 5000) // 5)
    if 5925 <= f <= 7125:
        return int((f - 5950) // 5)
    return None


def band_of(freq):
    if not freq:
        return None
    f = float(freq)
    return "2.4 GHz" if f < 3000 else "5 GHz" if f < 5925 else "6 GHz"


def _estimate_now(marks, waypoints, offset, t_now):
    """After the last mark: progress towards the next waypoint in sheet order at the speed so far."""
    order = list(waypoints)
    pts = [(m["t_mac"] + offset, k) for m in marks if (k := tel.waypoint_of(m["label"], waypoints))]
    if not pts:
        return None
    pts.sort()
    t_last, k_last = pts[-1]
    here = np.array(waypoints[k_last][:2], float)
    dist = sum(math.dist(waypoints[a][:2], waypoints[b][:2]) for (_, a), (_, b) in zip(pts, pts[1:]))
    span = pts[-1][0] - pts[0][0]
    speed = dist / span if span > 0 and dist > 0 else None
    nxt = order[(order.index(k_last) + 1) % len(order)] if len(order) > 1 else None
    if speed is None or nxt is None:
        return {"x": float(here[0]), "y": float(here[1]), "estimated": True, "since_mark_s": _r(t_now - t_last, 1),
                "last_mark": k_last, "towards": nxt}
    there = np.array(waypoints[nxt][:2], float)
    seg = float(np.linalg.norm(there - here)) or 1.0
    frac = min(1.0, max(0.0, speed * (t_now - t_last) / seg))
    p = here + frac * (there - here)
    return {"x": float(p[0]), "y": float(p[1]), "estimated": True, "since_mark_s": _r(t_now - t_last, 1),
            "last_mark": k_last, "towards": nxt}


def lap_map(events: list[dict], site: dict | None, waypoints: dict | None, cfg: dict, now_mac: float | None = None) -> dict:
    lap = _split(events)
    head, samples, pings, marks = lap["meta"], lap["samples"], lap["pings"], lap["marks"]
    offset = float(head.get("car_minus_mac_s") or 0.0)
    gap_s, good, fair = float(cfg["gap_s"]), float(cfg["good_dbm"]), float(cfg["fair_dbm"])
    wps = waypoints or {}
    t0 = (head.get("t_mac") + offset) if head.get("t_mac") else (samples[0]["t_car"] if samples else 0.0)
    times = np.array([s["t_car"] for s in samples], float)
    xy = tel.positions(times, marks, wps, offset) if len(samples) and len(wps) >= 2 else np.full((len(samples), 2), np.nan)

    # ---- positioned readings ----------------------------------------------------------------
    pts = [[_r(xy[i, 0]), _r(xy[i, 1]), s.get("signal_dbm"), s.get("bssid"), _r(s["t_car"] - t0, 1)]
           for i, s in enumerate(samples) if np.isfinite(xy[i]).all() and s.get("connected") and s.get("signal_dbm") is not None]

    # ---- outages: gaps in gateway replies ---------------------------------------------------
    gaps = [(a["t_car"], b["t_car"]) for a, b in zip(pings, pings[1:]) if b["t_car"] - a["t_car"] > gap_s]
    live_gap = None
    if pings and not lap["end"]:
        now = (time.time() if now_mac is None else now_mac) + offset
        if now - pings[-1]["t_car"] > gap_s:
            live_gap = (pings[-1]["t_car"], now)

    def where(t):
        if not len(samples) or len(wps) < 2:
            return None, None
        p = tel.positions([t], marks, wps, offset)[0]
        return (_r(p[0]), _r(p[1])) if np.isfinite(p).all() else (None, None)

    outages = []
    for a, b in gaps + ([live_gap] if live_gap else []):
        x, y = where(a)
        outages.append({"t_s": _r(a - t0, 1), "seconds": _r(b - a), "x": x, "y": y, "ongoing": (a, b) == live_gap})

    # ---- roams: the serving BSSID changes between connected samples ---------------------------
    conn = [s for s in samples if s.get("connected") and s.get("bssid")]
    roams = []
    for a, b in zip(conn, conn[1:]):
        if a["bssid"] != b["bssid"]:
            silent = max([g[1] - g[0] for g in gaps if g[0] <= b["t_car"] + 1.0 and g[1] >= a["t_car"] - 1.0] or [0.0])
            x, y = where(b["t_car"])
            roams.append({"t_s": _r(b["t_car"] - t0, 1), "from": a["bssid"], "to": b["bssid"], "x": x, "y": y,
                          "reply_gap_s": _r(silent), "bad": silent > float(cfg["bad_roam_s"]),
                          "from_dbm": a.get("signal_dbm"), "to_dbm": b.get("signal_dbm")})

    # ---- per AP -------------------------------------------------------------------------------
    sheet = {a["bssid"].lower(): a for a in (site or {}).get("aps", [])}
    period = float(head.get("period_s") or 0.1)
    stats = {}
    for s in conn:
        st = stats.setdefault(s["bssid"].lower(), {"n": 0, "sig": [], "freq": s.get("freq_mhz")})
        st["n"] += 1
        if s.get("signal_dbm") is not None:
            st["sig"].append(s["signal_dbm"])
    last_scan = {a["bssid"].lower(): a for a in (lap["scans"][-1]["aps"] if lap["scans"] else [])}
    aps = []
    for b in list(sheet) + [b for b in set(stats) | set(last_scan) if b not in sheet]:
        st, sh, sc = stats.get(b, {"n": 0, "sig": [], "freq": None}), sheet.get(b, {}), last_scan.get(b, {})
        freq = sh.get("frequency_mhz") or st["freq"] or sc.get("freq_mhz")
        sig = st["sig"]
        aps.append({"bssid": b, "in_sheet": b in sheet, "ssid": sh.get("ssid"), "xyz_m": sh.get("xyz_m"),
                    "freq_mhz": freq, "channel": channel_of(freq), "band": band_of(freq),
                    "serving_s": _r(st["n"] * period, 1), "serving_share": _r(st["n"] / len(conn), 3) if conn else 0.0,
                    "mean_dbm": _r(float(np.mean(sig)), 1) if sig else None, "min_dbm": min(sig) if sig else None,
                    "max_dbm": max(sig) if sig else None,
                    "roams_in": sum(r["to"].lower() == b for r in roams), "roams_out": sum(r["from"].lower() == b for r in roams),
                    "scan_dbm": sc.get("signal_dbm"), "seen_in_scan": bool(sc)})

    # ---- health (shown with its definition) -----------------------------------------------------
    span = (max(samples[-1]["t_car"], pings[-1]["t_car"] if pings else 0) - t0) if samples else 0.0
    sig_all = [s["signal_dbm"] for s in conn if s.get("signal_dbm") is not None]
    rtts = [p["rtt_ms"] for p in pings if p.get("rtt_ms") is not None]
    out_s = sum(o["seconds"] or 0 for o in outages)
    good_share = (sum(v >= good for v in sig_all) / len(sig_all)) if sig_all else None
    health = {
        "lap_s": _r(span, 1),
        "signal_good_share": _r(good_share, 3), "signal_fair_share": _r(sum(fair <= v < good for v in sig_all) / len(sig_all), 3) if sig_all else None,
        "signal_median_dbm": _r(float(np.median(sig_all)), 1) if sig_all else None,
        "latency_median_ms": _r(float(np.median(rtts)), 1) if rtts else None,
        "latency_p95_ms": _r(float(np.percentile(rtts, 95)), 1) if rtts else None,
        "latency_good_share": _r(sum(v <= float(cfg["latency_good_ms"]) for v in rtts) / len(rtts), 3) if rtts else None,
        "outages": len(outages), "outage_s": _r(out_s), "longest_outage_s": _r(max([o["seconds"] or 0 for o in outages] or [0])),
        "reply_share": _r(1 - out_s / span, 4) if span > 0 else None,
        "roams": len(roams), "bad_roams": sum(r["bad"] for r in roams),
        "mean_roam_gap_s": _r(float(np.mean([r["reply_gap_s"] for r in roams])), 2) if roams else None,
        "positioned_share": _r(len(pts) / len(conn), 3) if conn else None,
    }
    if good_share is not None and health["reply_share"] is not None:
        health["score"] = round(100 * good_share * health["reply_share"])
    health["definition"] = ("Score = share of link samples at or above Good signal x share of the lap with gateway replies "
                            "(no outage). 100 means good signal and no reply gaps all lap.")

    # ---- measured heatmap: mean signal per cell from positioned readings --------------------------
    cell = float(cfg["heat_cell_m"])
    heat = {}
    for x, y, sig, _, _ in pts:
        k = (int(x // cell), int(y // cell))
        h = heat.setdefault(k, [0.0, 0])
        h[0] += sig
        h[1] += 1
    cells = [[_r((i + 0.5) * cell), _r((j + 0.5) * cell), _r(v[0] / v[1], 1), v[1]] for (i, j), v in heat.items()]

    # ---- where the car is now -------------------------------------------------------------------
    car = None
    if samples:
        last = samples[-1]
        p = xy[-1]
        if np.isfinite(p).all():
            car = {"x": float(p[0]), "y": float(p[1]), "estimated": False}
        elif not lap["end"] and len(wps) >= 2:
            car = _estimate_now(marks, wps, offset, last["t_car"])
        if car:
            car |= {"bssid": last.get("bssid"), "signal_dbm": last.get("signal_dbm"), "connected": bool(last.get("connected"))}

    timeline = _thin([[_r(s["t_car"] - t0, 1), s.get("signal_dbm"), s.get("bssid")] for s in samples], MAX_SERIES)
    rtt = _thin([[_r(p["t_car"] - t0, 1), p.get("rtt_ms")] for p in pings], MAX_SERIES)
    mk = [{"label": m["label"], "t_s": _r(m["t_mac"] + offset - t0, 1), "known": tel.waypoint_of(m["label"], wps) is not None}
          for m in marks]
    events_out = sorted(
        [{"t_s": m["t_s"], "kind": "mark", "text": f"Passed {m['label']}" + ("" if m["known"] else " (not on the site sheet)")} for m in mk] +
        [{"t_s": r["t_s"], "kind": "bad_roam" if r["bad"] else "roam",
          "text": f"Roamed {r['from']} -> {r['to']}, reply gap {r['reply_gap_s']} s"} for r in roams] +
        [{"t_s": o["t_s"], "kind": "outage", "text": f"No gateway replies for {o['seconds']} s" + (" (still going)" if o["ongoing"] else "")}
         for o in outages],
        key=lambda e: e["t_s"] or 0)
    return {"lap": {"name": head.get("name"), "note": head.get("note") or "", "finished": bool(lap["end"]),
                    "marks": len(marks), "samples": len(samples)},
            "site": {"dimensions_m": (site or {}).get("dimensions_m"), "aps": (site or {}).get("aps", [])},
            "waypoints": wps, "points": _thin(pts, MAX_POINTS), "cells": cells, "cell_m": cell,
            "outages": outages, "roams": roams, "aps": aps, "health": health, "car": car,
            "timeline": timeline, "rtt": rtt, "marks": mk, "events": events_out[-300:],
            "thresholds": {"good_dbm": good, "fair_dbm": fair, "gap_s": gap_s}}


def merged_map(log_dir: Path, site, waypoints, cfg) -> dict:
    """All laps on one map: the radio map the car has built so far."""
    pts, cells, roams, outages, n = [], {}, [], [], 0
    cell = float(cfg["heat_cell_m"])
    for lap in viewer.list_laps(log_dir):
        if "-lap-check" in lap["file"]:
            continue
        m = lap_map(viewer.read_events(log_dir / lap["file"]), site, waypoints, cfg, now_mac=None)
        n += 1
        pts += m["points"]
        roams += [r | {"lap": m["lap"]["name"]} for r in m["roams"]]
        outages += [o | {"lap": m["lap"]["name"]} for o in m["outages"]]
    for x, y, sig, _, _ in pts:
        k = (int(x // cell), int(y // cell))
        h = cells.setdefault(k, [0.0, 0])
        h[0] += sig
        h[1] += 1
    return {"laps": n, "readings": len(pts), "points": _thin(pts, MAX_POINTS), "cell_m": cell,
            "cells": [[_r((i + 0.5) * cell), _r((j + 0.5) * cell), _r(v[0] / v[1], 1), v[1]] for (i, j), v in cells.items()],
            "roams": roams, "outages": outages}
