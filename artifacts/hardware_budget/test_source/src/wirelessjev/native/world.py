"""Ground truth and route forecasts for the cross-layer slot simulator.

Every policy sees the same truth, fading and blockage for a scene. Truth is
the warehouse twin's propagation (log-distance, shadowing, racks) for 2.4 and
5 GHz BSSs, plus a 6 GHz link per site (160 MHz, +1.9 dB free-space loss, +1 dB
per rack row crossed beyond the 5 GHz rack loss). Forecasts evaluate the same
law at the robots' *planned* positions and add a smooth map-error field
(sigma_map, matching corrected E7). Fast fading and blockage are never
forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..twin.radio import CS_THRESHOLD_DBM
from ..twin.warehouse import RACK_DB, make_warehouse, rack_obstruction

BANDS = ("2g", "5g", "6g")
NOISE_DBM = {"2g": -174 + 73.0 + 7.0, "5g": -174 + 79.0 + 7.0, "6g": -174 + 82.0 + 7.0}
CLIENT_TX_DBM = 15.0                    # BE200 client (hardware profile)
POS_DT = 0.2                            # truth sampled every 200 ms
HORIZONS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0)
EXTRA_6G_FSPL_DB = 20 * math.log10(6.5 / 5.2)
EXTRA_6G_RACK_DB = 1.0
CH6 = 7                                 # 160 MHz channels available at 6 GHz


@dataclass
class World:
    seed: int
    n: int
    sites: np.ndarray            # (S, 2)
    snr: np.ndarray              # (T, N, S, 3) uplink SNR dB, float32
    rssi_dl: np.ndarray          # (T, N, S, 3) downlink RSSI dBm (what scans report)
    forecast: np.ndarray         # (D, N, H, S, 3) forecast uplink SNR at decision d
    domain: np.ndarray           # (S, 3) contention-domain id
    avail: np.ndarray            # (n_domains,) airtime left after external occupancy
    video_mbps: np.ndarray       # (N,)
    bulk_mbps: np.ndarray        # (N,)
    duration_s: float
    sigma_map: float
    blind: bool
    sigma_belief: np.ndarray = None   # (D, N, S, 3) true local map-error std at decision positions
    noise_db: np.ndarray = None       # (3,) noise plus interference floor per band
    map_calibration: dict = None      # learned map only: error and interval coverage
    channel: np.ndarray = None        # (S, 3) channel number per site and band
    rmap: object = None               # learned RadioMap (survey_s > 0), for online updates
    plan: np.ndarray = None           # (D, N, H, 2) planned positions at each decision
    pos: np.ndarray = None            # (T, N, 2) true positions every POS_DT
    tx_dl: np.ndarray = None          # (3,) AP transmit power per band


def _error_field(rng, sites, sigma):
    """Smooth random map error per (site, band): random Fourier features with
    ~10 m correlation length, scaled to std sigma."""
    s, b = sites.shape[0], len(BANDS)
    k = 24
    w = rng.normal(0, 1 / 10.0, size=(s, b, k, 2))
    ph = rng.uniform(0, 2 * math.pi, size=(s, b, k))
    amp = sigma * math.sqrt(2.0 / k)

    def at(x, y):   # x, y arrays of the same shape -> (..., S, 3)
        arg = (x[..., None, None, None] * w[..., 0] + y[..., None, None, None] * w[..., 1]
               + ph)
        return amp * np.cos(arg).sum(-1)
    return at


def _path_losses(scene, aps5, aps2, xs, ys):
    """(..., S, 3) path loss for 2.4, 5 and 6 GHz links of each site."""
    flat_x, flat_y = xs.ravel(), ys.ravel()
    out = np.empty(flat_x.shape + (len(aps5), 3), dtype=np.float32)
    for s, (a5, a2) in enumerate(zip(aps5, aps2)):
        pl5 = scene._path_loss_vec(a5, flat_x, flat_y)
        out[:, s, 0] = scene._path_loss_vec(a2, flat_x, flat_y)
        out[:, s, 1] = pl5
        rows = rack_obstruction(a5, flat_x, flat_y) / RACK_DB["5g"]
        out[:, s, 2] = pl5 + EXTRA_6G_FSPL_DB + EXTRA_6G_RACK_DB * rows
    return out.reshape(xs.shape + (len(aps5), 3))


def build_world(seed: int, *, duration_s: float = 300.0, n_robots: int = 100,
                sigma_map: float = 4.0, blind: bool = False, hetero: bool = False,
                noise_2g_extra_db: float = 0.0, survey_s: float = 0.0,
                map_calibration: str = "robots", site_params: dict | None = None,
                map_prior: dict | None = None) -> World:
    """survey_s > 0: robots first drive for survey_s seconds reporting capped
    scans; a learned RadioMap then supplies forecasts and their uncertainty
    (sigma_map/hetero are ignored)."""
    sp = site_params or {}
    scene = make_warehouse(seed, n_robots=n_robots, rack_scale=sp.get("rack_scale", 1.0))
    scene.pl_coef = sp.get("pl_coef", scene.pl_coef)
    scene.shadow_db = sp.get("shadow_db", scene.shadow_db)
    aps5 = [a for a in scene.aps if a.band == "5g"]
    aps2 = [a for a in scene.aps if a.band == "2g"]
    sites = np.array([[a.x, a.y] for a in aps5])
    tx = {"2g": aps2[0].tx_power_dbm, "5g": aps5[0].tx_power_dbm, "6g": aps5[0].tx_power_dbm}

    steps = int(round(duration_s / POS_DT))
    per_decision = int(round(1.0 / POS_DT))
    pos = np.zeros((steps, n_robots, 2))
    plan = np.zeros((steps // per_decision, n_robots, len(HORIZONS), 2))
    cars = list(scene.cars)
    mob = scene.mobility
    survey_pos = []
    for _ in range(int(survey_s)):              # survey drive, 1 scan per second
        survey_pos.append([(c.x, c.y) for c in cars])
        for _ in range(per_decision):
            cars = mob.advance(cars, POS_DT)
    for k in range(steps):
        if k % per_decision == 0:
            d = k // per_decision
            for i, car in enumerate(cars):
                if blind:
                    plan[d, i] = (car.x, car.y)
                else:
                    plan[d, i] = mob.planned_positions(car, list(HORIZONS))
        pos[k] = [(c.x, c.y) for c in cars]
        cars = mob.advance(cars, POS_DT)

    pl = _path_losses(scene, aps5, aps2, pos[..., 0], pos[..., 1])
    # Optional 2.4 GHz interference-plus-noise rise (Bluetooth, scanners,
    # microwaves): the base twin models 2.4 GHz occupancy as airtime only.
    noise = np.array([NOISE_DBM[b] for b in BANDS], dtype=np.float32)
    noise[0] += noise_2g_extra_db
    snr = (CLIENT_TX_DBM - pl - noise).astype(np.float32)
    rssi_dl = (np.array([tx[b] for b in BANDS], dtype=np.float32) - pl).astype(np.float32)

    rng = np.random.default_rng(10_000 + seed)
    field = _error_field(rng, sites, sigma_map) if sigma_map > 0 else None
    plf = _path_losses(scene, aps5, aps2, plan[..., 0], plan[..., 1])
    fc = CLIENT_TX_DBM - plf - noise
    mult = np.ones(plan.shape[:-1] + (len(sites), 3))
    if hetero:
        # Spatially varying map quality: log-normal multiplier with RMS 1.
        g = _error_field(np.random.default_rng(20_000 + seed), sites, 0.8)
        lg = g(plan[..., 0], plan[..., 1])
        mult = np.exp(lg) / math.sqrt(np.mean(np.exp(2 * lg)))
    if field is not None:
        fc = fc + field(plan[..., 0], plan[..., 1]) * mult
    forecast = fc.astype(np.float32)
    sigma_belief = (sigma_map * mult[:, :, 0]).astype(np.float32)
    calib = None
    rmap = None
    if survey_s > 0:
        from .radiomap import RadioMap, calibration, survey
        sp = np.array(survey_pos)
        txv = np.array([tx[b] for b in BANDS], dtype=np.float32)
        truth_s = txv - _path_losses(scene, aps5, aps2, sp[..., 0], sp[..., 1])
        obs, cens = survey(truth_s, sp, np.random.default_rng(40_000 + seed))
        holdout = ("spatial" if map_calibration == "spatial" else
                   np.random.default_rng(50_000 + seed).choice(n_robots, n_robots // 5, replace=False))
        rmap = RadioMap(sites, aps5, seed, map_prior).fit(obs, cens, holdout)
        mean, std = rmap.predict(plan[..., 0], plan[..., 1])
        forecast = (mean - txv + CLIENT_TX_DBM - noise).astype(np.float32)
        sigma_belief = std[:, :, 0].astype(np.float32)
        sub = pos[::25]
        m_e, s_e = rmap.predict(sub[..., 0], sub[..., 1])
        calib = calibration(rssi_dl[::25], m_e, s_e)
        sigma_map = float("nan")

    # Contention domains per band: same channel and carrier-sensed (as twin).
    r = scene.radio()
    ap_index = {a.ap_id: j for j, a in enumerate(scene.aps)}
    domain = np.zeros((len(sites), 3), dtype=int)
    channel = np.zeros((len(sites), 3), dtype=int)
    avail = []
    for b, band in enumerate(BANDS):
        if band == "6g":
            chans = [(k % 4 + 2 * (k // 4)) % CH6 for k in range(len(sites))]
            ref = aps5
            extra = EXTRA_6G_FSPL_DB
            occ = [0.0] * len(sites)
        else:
            ref = aps5 if band == "5g" else aps2
            chans = [a.channel for a in ref]
            extra = 0.0
            occ = [scene.external_occupancy.get((band, a.channel), 0.0) for a in ref]
        channel[:, b] = chans
        parent = list(range(len(sites)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                heard = r.apap[ap_index[ref[i].ap_id], ap_index[ref[j].ap_id]] - extra
                if chans[i] == chans[j] and heard >= CS_THRESHOLD_DBM:
                    parent[find(i)] = find(j)
        roots = {}
        for i in range(len(sites)):
            root = find(i)
            if root not in roots:
                roots[root] = len(avail)
                avail.append(1.0 - occ[i])
            domain[i, b] = roots[root]

    video = np.array([sum(f.demand_mbps for f in c.flows if f.name == "video_uplink")
                      for c in scene.cars])
    bulk = np.array([sum(f.demand_mbps for f in c.flows if f.name == "map_upload")
                     for c in scene.cars])
    return World(seed, n_robots, sites, snr, rssi_dl, forecast, domain, np.array(avail),
                 video, bulk, duration_s, sigma_map, blind, sigma_belief, noise, calib, channel,
                 rmap, plan, pos, np.array([tx[b] for b in BANDS], dtype=np.float32))
