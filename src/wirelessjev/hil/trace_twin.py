"""Trace-driven twin: recorded laps of one real car become a small fleet.

Each recorded lap (wirelessjev.hil.telemetry) is one virtual robot. Laps that
were driven one after another are overlaid on a common clock, and the result
is a `native.world.World`, so the unchanged slot simulator and every policy
run on *measured* signal levels instead of a propagation model.

What is measured: each robot's signal to every AP it heard (10 Hz for the
associated AP, scan-cache snapshots for the others), its positions from
waypoint marks, and the AP sheet. What is not: laps were not simultaneous,
so robot-to-robot interference comes from the simulator's model; fading and
blockage between samples come from the simulator's hidden draws until those
are calibrated from the same logs.

Forecasts are honest: a robot's forecast at its future position uses only
*other* laps' measurements near that position (leave-one-lap-out), with the
local spread as its uncertainty.
"""
from __future__ import annotations

import numpy as np

from ..native.world import HORIZONS, POS_DT, World
from .telemetry import positions

FLOOR_DBM = -105.0
CLIENT_TX_DBM = 15.0
OWN_WINDOW_S = 3.0          # use a robot's own reading if one is this recent
KNN, KNN_RADIUS_M = 5, 5.0
BLIND_SIGMA_DB = 8.0


def band_of(freq_mhz: float) -> int:
    return 0 if freq_mhz < 3000 else 1 if freq_mhz < 5925 else 2


def site_index(site: dict):
    """Group BSSIDs into physical AP sites by position; map BSSID -> (site, band)."""
    places, index = [], {}
    for ap in site["aps"]:
        xy = tuple(round(v, 2) for v in ap["xyz_m"][:2])
        if xy not in places:
            places.append(xy)
        index[ap["bssid"].lower()] = (places.index(xy), band_of(ap["frequency_mhz"]), ap)
    return np.array(places, dtype=float), index


def _lap_observations(lap: dict, index: dict):
    """(t_car, site, band, rssi) from link samples and scan snapshots."""
    rows = []
    for s in lap["samples"]:
        key = (s.get("bssid") or "").lower()
        if s.get("connected") and key in index and s.get("signal_dbm") is not None:
            rows.append((s["t_car"], index[key][0], index[key][1], s["signal_dbm"]))
    for sc in lap["scans"]:
        for ap in sc["aps"]:
            key = ap["bssid"].lower()
            if key in index and (ap.get("age_ms") is None or ap["age_ms"] <= 10_000):
                rows.append((sc["t_car"] - (ap.get("age_ms") or 0) / 1000.0, index[key][0],
                             index[key][1], ap["signal_dbm"]))
    return np.array(rows, dtype=float).reshape(-1, 4)


def build_trace_world(laps: list[dict], site: dict, waypoints: dict, *, duration_s: float | None = None,
                      seed: int = 0, video_robots=(0,), video_mbps: float = 5.0,
                      bulk_mbps: float = 8.0) -> World:
    sites, index = site_index(site)
    n, s_n = len(laps), len(sites)
    if n < 2:
        raise ValueError("a trace twin needs at least two laps (leave-one-lap-out forecasts)")
    obs, spans, own_pos = [], [], []
    for lap in laps:
        o = _lap_observations(lap, index)
        off = lap["meta"].get("car_minus_mac_s", 0.0)
        marks = sorted(lap["marks"], key=lambda m: m["t_mac"])
        usable = [m for m in marks if m["label"] in waypoints]
        if len(usable) < 2 or len(o) == 0:
            raise ValueError(f"lap {lap['meta'].get('name')!r} needs two waypoint marks and measurements")
        t0, t1 = usable[0]["t_mac"] + off, usable[-1]["t_mac"] + off
        o = o[(o[:, 0] >= t0) & (o[:, 0] <= t1)]
        xy = positions(o[:, 0], marks, waypoints, off)
        obs.append(np.column_stack([o, xy]))                 # t, site, band, rssi, x, y
        spans.append((t0, t1))
        own_pos.append((marks, off))
    duration = duration_s or min(t1 - t0 for t0, t1 in spans)
    steps = int(duration / POS_DT)
    if steps < int(10 / POS_DT):
        raise ValueError("laps are shorter than 10 s between first and last mark")

    def knn(i_exclude, x, y, sb):
        """Leave-one-lap-out spatial estimate: mean, std of nearest readings."""
        pool = np.concatenate([o for j, o in enumerate(obs) if j != i_exclude])
        pool = pool[(pool[:, 1] == sb[0]) & (pool[:, 2] == sb[1])]
        if len(pool) == 0:
            return np.full(len(x), FLOOR_DBM), np.full(len(x), BLIND_SIGMA_DB)
        d = np.hypot(pool[None, :, 4] - x[:, None], pool[None, :, 5] - y[:, None])
        order = np.argsort(d, axis=1)[:, :KNN]
        dn = np.take_along_axis(d, order, axis=1)
        vals = pool[order, 3]
        ok = dn <= KNN_RADIUS_M
        cnt = ok.sum(1)
        mean = np.where(cnt > 0, (vals * ok).sum(1) / np.maximum(cnt, 1), FLOOR_DBM)
        std = np.where(cnt > 1, np.sqrt(((vals - mean[:, None]) ** 2 * ok).sum(1) / np.maximum(cnt, 1)), BLIND_SIGMA_DB)
        return mean, np.maximum(std, 1.0)

    rssi = np.full((steps, n, s_n, 3), FLOOR_DBM, dtype=np.float32)
    pos = np.zeros((steps, n, 2))
    per_dec = int(round(1.0 / POS_DT))
    decs = steps // per_dec
    plan = np.zeros((decs, n, len(HORIZONS), 2))
    fc = np.full((decs, n, len(HORIZONS), s_n, 3), FLOOR_DBM, dtype=np.float32)
    sig = np.full((decs, n, s_n, 3), BLIND_SIGMA_DB, dtype=np.float32)
    for i in range(n):
        t0 = spans[i][0]
        marks, off = own_pos[i]
        grid = t0 + np.arange(steps) * POS_DT
        pos[:, i] = positions(grid, marks, waypoints, off)
        for d in range(decs):
            th = np.minimum(t0 + d * 1.0 + np.array(HORIZONS), spans[i][1])
            plan[d, i] = positions(th, marks, waypoints, off)
        o = obs[i]
        for s in range(s_n):
            for b in range(3):
                m, sd = knn(i, pos[:, i, 0], pos[:, i, 1], (s, b))        # fallback: other laps
                series = m.copy()
                mine = o[(o[:, 1] == s) & (o[:, 2] == b)]
                if len(mine):
                    order = np.argsort(mine[:, 0])
                    tt, vv = mine[order, 0], mine[order, 3]
                    near = np.abs(tt[np.clip(np.searchsorted(tt, grid), 0, len(tt) - 1)] - grid) <= OWN_WINDOW_S
                    prev = np.abs(tt[np.clip(np.searchsorted(tt, grid) - 1, 0, len(tt) - 1)] - grid) <= OWN_WINDOW_S
                    use = near | prev
                    series[use] = np.interp(grid[use], tt, vv)
                rssi[:, i, s, b] = series
                pm, ps = knn(i, plan[:, i, :, 0].ravel(), plan[:, i, :, 1].ravel(), (s, b))
                fc[:, i, :, s, b] = pm.reshape(decs, len(HORIZONS))
                sig[:, i, s, b] = ps.reshape(decs, len(HORIZONS))[:, 0]

    noise = np.full(3, float(site["receiver"]["noise_floor_dbm"]), dtype=np.float32)
    tx = np.zeros(3, dtype=np.float32)
    channel = np.zeros((s_n, 3), dtype=int)
    present = np.zeros((s_n, 3), dtype=bool)
    for _, (s, b, ap) in index.items():
        tx[b] = ap["tx_power_dbm"]
        channel[s, b] = int(ap["frequency_mhz"])
        present[s, b] = True
    rssi[:, :, ~present] = -150.0                              # bands an AP does not have
    fc[:, :, :, ~present] = -150.0
    # Uplink SNR from the downlink reading: same path, client transmits at 15 dBm.
    to_snr = lambda r: r - tx[None, :] + CLIENT_TX_DBM - noise[None, :]
    domain = np.zeros((s_n, 3), dtype=int)
    ids = {}
    for s in range(s_n):
        for b in range(3):
            domain[s, b] = ids.setdefault((b, channel[s, b]), len(ids))
    video = np.zeros(n)
    video[list(video_robots)] = video_mbps
    bulk = np.where(np.arange(n) % 2 == 1, bulk_mbps, 0.0)
    return World(seed, n, sites, to_snr(rssi).astype(np.float32), rssi, to_snr(fc).astype(np.float32),
                 domain, np.ones(len(ids)), video, bulk, float(steps * POS_DT), float("nan"), False,
                 sig, noise, {"source": "trace", "laps": n}, channel, None, plan, pos, tx)


def main(argv=None) -> int:
    """Run the standard policies on a fleet built from recorded laps."""
    import argparse
    from dataclasses import replace
    import json

    from ..native.sim import policies, simulate
    from ..smartcar_runtime.contracts import validate_site
    from .telemetry import load_lap

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("laps", nargs="+", help="lap logs; each becomes one virtual robot")
    ap.add_argument("--site", required=True, help="site.json (same file as the deployment bundle)")
    ap.add_argument("--waypoints", required=True, help='JSON object: {"A": [x, y], ...} in site metres')
    ap.add_argument("--collisions", type=float, default=0.01)
    args = ap.parse_args(argv)
    site = validate_site(json.load(open(args.site)))
    world = build_trace_world([load_lap(p) for p in args.laps], site, json.load(open(args.waypoints)))
    pol = {p.name: p for p in policies()}
    vert = replace(pol["native"], gate=True, cadence_roam=True)
    arms = {"wifi7_fixed_rule": pol["mlo_split"], "single_radio": vert,
            "second_path_same_card": replace(vert, dual_radio=True, radio2="shared")}
    out = {"robots": world.n, "evidence": "1 real car, laps replayed as a fleet; interference, fading "
           "between samples and blockage are simulated", "duration_s": world.duration_s, "arms": {}}
    for name, p in arms.items():
        r = simulate(world, replace(p, name=name), collision_lambda=args.collisions, interference=True)
        out["arms"][name] = {k: r[k] for k in ("control_miss", "control_misses", "control_sent", "availability_1s",
                                               "safety_stop_fraction", "control_gap_p99_ms", "roams", "miss_causes")}
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
