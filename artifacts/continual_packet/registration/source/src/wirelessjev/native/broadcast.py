"""Fleet safety broadcast without retries: multicast at MCS 0 vs a message
carried on the base layer of every downlink frame (NATIVE_GATE.md, part B)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .phy import LinkCurves
from .sim import BLOCK_SLOTS, BLOCKAGE_DB, SLOT, hidden_draws
from .world import BANDS, NOISE_DBM, POS_DT, World

PHY = Path(__file__).resolve().parents[3] / "artifacts" / "native" / "phy_layered.json"
WINDOW = 5                     # slots (50 ms)
MSG_BITS = 100 * 8
OVERHEAD_S = 100e-6


def layered_success_curve(path: Path = PHY, alpha: str = "layered_alpha3"):
    rows = json.loads(Path(path).read_text())["awgn"][alpha]
    snr = np.array([r["snr_db"] for r in rows])
    ok = 1 - np.array([r["critical_bler"] for r in rows])
    return lambda x: np.interp(x, snr, ok, left=0.0, right=1.0)


def study(world: World, *, curves: LinkCurves | None = None, fs=(0.2, 0.5, 1.0),
          event_period_s: float = 1.0) -> dict:
    curves = curves or LinkCurves()
    n_slots = int(round(world.duration_s / SLOT))
    fade_db, blocked, _ = hidden_draws(world, n_slots)
    layered = layered_success_curve()
    noise = world.noise_db
    per = int(round(event_period_s / SLOT))
    misses = {"multicast_mcs0": [], "multicast_x3": []} | {f"layered_{f:g}": [] for f in fs}
    for k0 in range(0, n_slots - WINDOW, per):
        kp = min(k0 * SLOT // POS_DT, world.rssi_dl.shape[0] - 1)
        kp = int(kp)
        site = np.argmax(world.rssi_dl[kp].max(axis=2), axis=1)               # (N,)
        ar = np.arange(world.n)
        snr = []
        for t in range(WINDOW):
            kb = (k0 + t) // BLOCK_SLOTS
            x = (world.rssi_dl[kp][ar, site] + fade_db[kb][ar, site]
                 - BLOCKAGE_DB * blocked[kb][ar, site][:, None] - noise[None, :])
            snr.append(x)
        snr = np.stack(snr, 1)                                                 # (N,T,3)
        p0 = np.stack([curves.success(BANDS[b], np.zeros(world.n, dtype=int), snr[:, 0, b])
                       for b in range(3)], 1)
        misses["multicast_mcs0"].append(np.prod(1 - p0, axis=1))
        p3 = np.stack([np.stack([curves.success(BANDS[b], np.zeros(world.n, dtype=int), snr[:, t, b])
                                 for b in range(3)], 1) for t in range(3)], 1)
        misses["multicast_x3"].append(np.prod(1 - p3, axis=(1, 2)))
        pl = layered(snr)
        for f in fs:
            misses[f"layered_{f:g}"].append(np.prod(1 - f * pl, axis=(1, 2)))
    out = {"scene": world.seed, "events": len(misses["multicast_mcs0"])}
    for name, m in misses.items():
        m = np.array(m)                                                        # (E,N)
        per_robot = m.mean(0)
        out[name] = {"miss": float(m.mean()),
                     "worst_robot_miss": float(per_robot.max()),
                     "events_any_robot_missed": float((1 - np.prod(1 - m, axis=1)).mean())}
    t0 = (MSG_BITS / (curves.rates("2g")[0] * 1e6) + OVERHEAD_S)
    out["added_airtime_per_ap"] = {
        "multicast_mcs0": 3 * t0 / event_period_s, "multicast_x3": 9 * t0 / event_period_s,
        **{f"layered_{f:g}": 0.0 for f in fs}}
    return out
