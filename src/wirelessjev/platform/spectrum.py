"""The site's radio map from the car's own readings, using RadioMap.

Every positioned reading (the link sample of the serving AP, and every AP in each scan
snapshot) is fitted with RadioMap's Bayesian model per AP and band: log distance, the site
sheet's obstructions crossed (CAD boxes), and random features for the shadowing nobody knows.
The model is then evaluated over the whole floor, which gives what a site-survey tool shows:
signal from the best AP and from each AP, which AP serves where, the backup signal from a
second AP location, SNR, SINR with co-channel APs, how many co-channel APs are heard, and
RadioMap's own uncertainty, so the page can fade what is guessed rather than measured.

The subclass changes only where obstructions come from (the site sheet instead of the
simulator's warehouse racks); RadioMap itself is unchanged."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..hil import telemetry as tel
from ..hil import viewer
from ..hil.trace_twin import _lap_observations, site_index
from ..native.radiomap import N_RFF, RadioMap
from .mapview import band_of, channel_of

CCA_DBM = -82.0            # a 20 MHz Wi-Fi preamble at or above this makes other radios defer: co-channel contention
MAX_CELLS = 9000
SPATIAL_MIN_READINGS = 200


def obstruction(ax: float, ay: float, xs: np.ndarray, ys: np.ndarray, boxes: list, steps: int = 32) -> np.ndarray:
    """How many of the site sheet's boxes the straight path from the AP to each point crosses."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    out = np.zeros(xs.shape)
    if not boxes:
        return out
    t = np.linspace(0.0, 1.0, steps)[:, None]
    px, py = ax + t * (xs[None, :] - ax), ay + t * (ys[None, :] - ay)
    for (lo, hi) in boxes:
        x0, x1 = sorted((lo[0], hi[0]))
        y0, y1 = sorted((lo[1], hi[1]))
        inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
        out += inside.any(axis=0)
    return out


class SiteRadioMap(RadioMap):
    """RadioMap with obstructions from the site sheet's CAD boxes."""

    def __init__(self, sites: np.ndarray, boxes: list, seed: int = 0):
        super().__init__(sites, aps5=None, seed=seed)
        self.boxes = boxes

    def _features(self, s, b, x, y):
        d = np.maximum(1.0, np.hypot(x - self.sites[s, 0], y - self.sites[s, 1]))
        crossed = obstruction(self.sites[s, 0], self.sites[s, 1], x, y, self.boxes)
        arg = x[:, None] * self.w[s, b, :, 0] + y[:, None] * self.w[s, b, :, 1] + self.ph[s, b]
        rff = math.sqrt(2.0 / N_RFF) * np.cos(arg)
        return np.column_stack([np.ones_like(x), np.log10(d), crossed, rff])


def readings(laps: list[dict], site: dict, waypoints: dict):
    """Positioned readings: rows of (x, y, site, band, rssi, lap)."""
    sites, index = site_index(site)
    rows = []
    for k, lap in enumerate(laps):
        o = _lap_observations(lap, index)
        if not len(o):
            continue
        xy = tel.positions(o[:, 0], lap["marks"], waypoints, lap["meta"].get("car_minus_mac_s", 0.0))
        ok = np.isfinite(xy).all(axis=1)
        if ok.any():
            rows.append(np.column_stack([xy[ok], o[ok, 1], o[ok, 2], o[ok, 3], np.full(ok.sum(), k)]))
    return sites, index, (np.concatenate(rows) if rows else np.zeros((0, 6)))


def contours(z: np.ndarray, level: float, x0: float, y0: float, cell: float) -> list:
    """Marching squares: line segments where the grid crosses `level` (cell centres as corners)."""
    segs = []
    ny, nx = z.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            c = [z[j, i], z[j, i + 1], z[j + 1, i + 1], z[j + 1, i]]
            if any(not np.isfinite(v) for v in c):
                continue
            pts = []
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            for a in range(4):
                b = (a + 1) % 4
                va, vb = c[a], c[b]
                if (va >= level) != (vb >= level):
                    f = (level - va) / (vb - va)
                    (ia, ja), (ib, jb) = corners[a], corners[b]
                    pts.append((x0 + (ia + f * (ib - ia) + 0.5) * cell, y0 + (ja + f * (jb - ja) + 0.5) * cell))
            for p, q in zip(pts[0::2], pts[1::2]):
                segs.append([round(float(v), 2) for v in (p[0], p[1], q[0], q[1])])
    return segs


def _q(a: np.ndarray, nd: int = 0) -> list:
    """Grid values as a flat list, rounded, with None where unknown."""
    flat = a.ravel()
    return [None if not np.isfinite(v) else (round(float(v), nd) if nd else int(round(float(v)))) for v in flat]


def build(laps: list[dict], site: dict, waypoints: dict, cfg: dict) -> dict:
    W, H = float(site["dimensions_m"][0]), float(site["dimensions_m"][1])
    boxes = [[b[0][:2], b[1][:2]] for b in site.get("cad_boxes_m", [])]
    aps = site["aps"]
    noise = float(site.get("receiver", {}).get("noise_floor_dbm", -92.0))
    sites, index, R = readings(laps, site, waypoints or {})

    # Each AP's own transmit power, so RadioMap's generic prior (a 20 dBm AP) fits every AP.
    tx = {(s, b): float(ap.get("tx_power_dbm", 20.0)) - 20.0 for _, (s, b, ap) in index.items()}
    rm = SiteRadioMap(sites, boxes)
    obs = {}
    for (s, b) in tx:
        k = (R[:, 2] == s) & (R[:, 3] == b)
        obs[(s, b)] = (R[k, 0], R[k, 1], R[k, 4] - tx[(s, b)], R[k, 5])
    holdout = "spatial" if len(R) >= SPATIAL_MIN_READINGS else None
    rm.fit(obs, holdout=holdout)

    cell = max(0.25, math.sqrt(W * H / MAX_CELLS))
    nx, ny = max(2, int(math.ceil(W / cell))), max(2, int(math.ceil(H / cell)))
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell)
    mean, std = rm.predict(gx, gy)                                          # (ny, nx, S, 3)

    per_ap, sd_ap = [], []
    for ap in aps:
        s, b, _ = index[ap["bssid"].lower()]
        off = tx[(s, b)]
        per_ap.append(mean[..., s, b] + off)
        sd_ap.append(std[..., s, b])
    P, SD = np.stack(per_ap, -1), np.stack(sd_ap, -1)                        # (ny, nx, n_ap)
    best_i = np.argmax(P, -1)
    best = np.take_along_axis(P, best_i[..., None], -1)[..., 0]
    best_sd = np.take_along_axis(SD, best_i[..., None], -1)[..., 0]
    # Backup: the strongest AP at a different physical place (the second path the system uses).
    site_of = np.array([index[a["bssid"].lower()][0] for a in aps])
    best_site = site_of[best_i]
    other = np.where(site_of[None, None, :] != best_site[..., None], P, -np.inf)
    backup = other.max(-1)
    backup[~np.isfinite(backup)] = np.nan
    # Co-channel: other BSSIDs on the serving AP's channel that are heard at or above CCA.
    chan = np.array([(band_of(a["frequency_mhz"]), channel_of(a["frequency_mhz"])) for a in aps], dtype=object)
    same = np.array([[chan[i][0] == chan[j][0] and chan[i][1] == chan[j][1] and i != j for j in range(len(aps))]
                     for i in range(len(aps))])
    co = same[best_i]                                                       # (ny, nx, n_ap)
    cochan = (co & (P >= CCA_DBM)).sum(-1)
    lin = np.where(co, 10 ** (P / 10.0), 0.0).sum(-1) + 10 ** (noise / 10.0)
    sinr = best - 10 * np.log10(lin)
    snr = best - noise

    xy = R[:, :2]
    if len(xy):
        from scipy.spatial import cKDTree
        near = cKDTree(xy).query(np.column_stack([gx.ravel(), gy.ravel()]))[0].reshape(gx.shape)
    else:
        near = np.full(gx.shape, np.inf)
    # How good is the map? Error on each quadrant when fitted without it (RadioMap's spatial hold-out).
    fit = {"readings": int(len(R)), "laps_used": int(len(set(R[:, 5].tolist()))) if len(R) else 0,
           "aps_measured": int(sum(len(o[0]) > 0 for o in obs.values())),
           "discrepancy_db": round(math.sqrt(max(0.0, getattr(rm, "disc_var", 0.0))), 1),
           "median_uncertainty_db_where_measured": round(float(np.median(best_sd[near <= 2.0])), 1) if (near <= 2.0).any() else None,
           "median_uncertainty_db_elsewhere": round(float(np.median(best_sd[near > 2.0])), 1) if (near > 2.0).any() else None,
           "model": "RadioMap (Bayesian: log distance + site-sheet obstructions + shadowing features), fitted per AP and band",
           "source": "measured" if len(R) else "site sheet only"}
    good, fair = float(cfg["good_dbm"]), float(cfg["fair_dbm"])
    return {"grid": {"cell": round(cell, 3), "nx": nx, "ny": ny, "width": W, "height": H},
            "layers": {"best": _q(best), "best_ap": [int(v) for v in best_i.ravel()], "backup": _q(backup),
                       "snr": _q(snr), "sinr": _q(sinr), "cochan": [int(v) for v in cochan.ravel()],
                       "uncertainty": _q(best_sd, 1), "near_m": _q(np.minimum(near, 99.0), 1),
                       "per_ap": [_q(P[..., i]) for i in range(len(aps))]},
            "aps": [{"bssid": a["bssid"].lower(), "ssid": a.get("ssid"), "xyz_m": a["xyz_m"], "freq_mhz": a["frequency_mhz"],
                     "channel": channel_of(a["frequency_mhz"]), "band": band_of(a["frequency_mhz"]),
                     "tx_power_dbm": a.get("tx_power_dbm"),
                     "readings": int(((R[:, 2] == index[a["bssid"].lower()][0]) & (R[:, 3] == index[a["bssid"].lower()][1])).sum()) if len(R) else 0}
                    for a in aps],
            "boxes": boxes, "noise_floor_dbm": noise,
            "contours": {"good": contours(best, good, 0.0, 0.0, cell), "fair": contours(best, fair, 0.0, 0.0, cell)},
            "thresholds": {"good_dbm": good, "fair_dbm": fair, "cca_dbm": CCA_DBM}, "fit": fit}


class Cache:
    """Refit only when a lap or the site sheet changes (and at most every 10 s while a lap grows)."""

    def __init__(self):
        self.key, self.value, self.at = None, None, 0.0

    def get(self, log_dir: Path, site: dict | None, waypoints: dict | None, cfg: dict, now: float) -> dict:
        if not site:
            raise ValueError("save the site sheet first: the map's size, APs and obstructions come from it")
        files = [l for l in viewer.list_laps(log_dir) if "-lap-check" not in l["file"]]
        key = (tuple((l["file"], round(l["modified"])) for l in files), repr(site), repr(waypoints),
               cfg["good_dbm"], cfg["fair_dbm"])
        if self.value is not None and (key == self.key or (now - self.at < 10 and key[1:] == self.key[1:])):
            return self.value
        from .mapview import _split
        laps = [_split(viewer.read_events(log_dir / l["file"])) for l in files]          # tolerates a lap still being written
        self.value, self.key, self.at = build(laps, site, waypoints or {}, cfg), key, now
        return self.value
