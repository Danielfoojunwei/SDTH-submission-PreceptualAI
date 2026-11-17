"""Rehearsal data: a site sheet and a synthetic lap that agree with each other, so the map,
roams, outages and health can be tried without the car. Everything here is made up and labelled so."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time

SITE = {"schema": 1, "site_id": "demo-hall (synthetic)", "dimensions_m": [24.0, 14.0, 4.0],
        "aps": [{"bssid": "aa:bb:cc:00:00:01", "ssid": "fleet", "xyz_m": [3.0, 2.0, 3.0], "frequency_mhz": 5180, "tx_power_dbm": 12.0},
                {"bssid": "aa:bb:cc:00:00:02", "ssid": "fleet", "xyz_m": [21.0, 2.0, 3.0], "frequency_mhz": 5260, "tx_power_dbm": 12.0},
                {"bssid": "aa:bb:cc:00:00:03", "ssid": "fleet", "xyz_m": [12.0, 12.5, 3.0], "frequency_mhz": 5500, "tx_power_dbm": 12.0}],
        "cad_boxes_m": [], "receiver": {"noise_floor_dbm": -92.0, "offset_db": 0.0, "quantization_db": 1.0, "reference_gauge_assumed": True}}
WAYPOINTS = {"A": [2.0, 4.0], "B": [12.0, 4.0], "C": [22.0, 4.0], "D": [22.0, 10.0], "E": [12.0, 10.0], "F": [2.0, 10.0]}
RACK = (6.0, 18.0, 6.6, 7.4)               # a shelf row between the two aisles: x0, x1, y0, y1


def _crosses_rack(p, q):
    x0, x1, y0, y1 = RACK
    for k in range(21):
        t = k / 20
        x, y = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False


def rssi(xy, ap):
    d = math.dist(xy, ap["xyz_m"][:2])
    level = ap["tx_power_dbm"] - (46.7 + 35 * math.log10(max(1.0, d)))
    return level - (9.0 if _crosses_rack(xy, ap["xyz_m"]) else 0.0)


def write_site(log_dir: Path) -> None:
    (log_dir / "site.json").write_text(json.dumps(SITE, indent=1))
    (log_dir / "waypoints.json").write_text(json.dumps(WAYPOINTS, indent=1))


def demo_lap(log_dir: Path, seconds: float = 900.0, speed_mps: float = 0.6, stop: threading.Event | None = None,
             realtime: bool = True, name: str = "demo") -> Path:
    """Drive the loop A-B-C-D-E-F-A at `speed_mps`, writing the same events a real lap writes:
    10 Hz link samples, 20 Hz gateway pings, a scan every 3 s and a mark at each waypoint.
    A person steps into the path every 40 s (the link dips and replies stop for 1.2 s)."""
    stop = stop or threading.Event()
    path = log_dir / time.strftime(f"%Y%m%d-%H%M%S-lap-{name}.jsonl")
    route = list(WAYPOINTS) + [list(WAYPOINTS)[0]]
    legs = [(WAYPOINTS[a], WAYPOINTS[b], b) for a, b in zip(route, route[1:])]
    loop_m = sum(math.dist(p, q) for p, q, _ in legs)
    start = time.time() if realtime else time.time() - seconds
    aps = SITE["aps"]
    serving = aps[0]

    def place(t):
        s = (t * speed_mps) % loop_m
        for p, q, label in legs:
            L = math.dist(p, q)
            if s <= L:
                f = s / L
                return (p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])), label, f
            s -= L
        return tuple(legs[-1][1]), legs[-1][2], 1.0

    with path.open("a", encoding="utf-8") as f:
        def w(ev):
            f.write(json.dumps(ev) + "\n")
            f.flush()
        w({"event": "lap_start", "name": name, "t_mac": start, "car_minus_mac_s": 0.0, "gateway": "10.0.0.1", "period_s": 0.1,
           "seconds": int(seconds), "note": "synthetic demo lap (made-up data)"})
        w({"event": "mark", "t_mac": start, "label": "A"})
        k, last_label, roam_until = 0, None, -1.0
        while not stop.is_set() and k * 0.1 < seconds:
            t_rel = k * 0.1
            t = start + t_rel
            xy, heading_to, frac = place(t_rel)
            levels = {a["bssid"]: rssi(xy, a) for a in aps}
            best = max(aps, key=lambda a: levels[a["bssid"]])
            if levels[best["bssid"]] > levels[serving["bssid"]] + 6:
                serving = best
                roam_until = t_rel + 0.3                          # a roam costs a short reply gap
            phase = t_rel % 40.0
            blocked = 20.0 <= phase < 22.5
            silent = (20.4 <= phase < 21.6) or t_rel < roam_until
            sig = round(levels[serving["bssid"]] - (13.0 if blocked else 0.0))
            w({"event": "sample", "t_car": t, "connected": True, "bssid": serving["bssid"], "ssid": "fleet",
               "freq_mhz": serving["frequency_mhz"], "signal_dbm": sig, "tx_bitrate_mbps": 390.0 if sig > -60 else 173.0 if sig > -72 else 58.0,
               "tx_retries": float(k * 2), "tx_failed": 0.0})
            if not silent:
                for j in (0, 1):
                    w({"event": "ping", "t_car": t + 0.05 * j, "rtt_ms": round(2.2 + (9.0 if blocked else 0.0) + (k % 7) * 0.15, 1)})
            if k % 30 == 0:
                w({"event": "scan", "t_car": t, "aps": [{"bssid": a["bssid"], "freq_mhz": float(a["frequency_mhz"]),
                                                         "signal_dbm": round(levels[a["bssid"]], 1), "age_ms": 200} for a in aps]})
            # The operator presses the waypoint letter as the car arrives at it.
            nxt = place(t_rel + 0.1)
            if nxt[1] != heading_to and heading_to != last_label:
                w({"event": "mark", "t_mac": t + 0.1, "label": heading_to})
                last_label = heading_to
            k += 1
            if realtime:
                time.sleep(max(0.0, start + k * 0.1 - time.time()))
        w({"event": "lap_end", "t_mac": start + k * 0.1})
    return path
