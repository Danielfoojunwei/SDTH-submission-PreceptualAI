"""Foresight from the robot's own obstacle sensing (docs/FORESIGHT.md).

Radio telemetry cannot see a blockage coming (docs/OUTAGE_RISK.md). A robot's
lidar or camera can see the forklift. This module turns sensed obstacles into
a one-second warning per robot-AP path, three ways:

- `geometric_foresight`: a hand-written rule that needs the site's geometry
  (how much of the line to an AP an obstacle can block);
- `LearnedForesight`: a small network that learns that mapping from data, with
  no geometry given;
- `blockers.truth_foresight`: the perfect warning, for headroom.

Sensing is imperfect on purpose: limited range, hidden by racks, missed
detections, position noise, and velocity from differencing noisy positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..native.blockers import FORESEE_BLOCKS, RADIUS_M, REACH_FRAC
from .dataset import _rack_between
from .models import SmallNet

RANGE_M, P_DETECT, POS_SIGMA_M = 10.0, 0.9, 0.2
VEL_BASE_TICKS = 5                     # velocity from two detections 0.5 s apart
TICK_S = 0.1
HORIZONS = (2, 4, 6, 8, 10)            # ticks ahead checked by the geometric rule
MAX_OBJECTS = 2                        # nearest sensed obstacles given to the learned model
N_FEATURES = 3 + 5 * MAX_OBJECTS


@dataclass
class Sensed:
    seen: np.ndarray        # (T, N, K) bool
    pos: np.ndarray         # (T, N, K, 2) estimated position
    vel: np.ndarray         # (T, N, K, 2) estimated velocity, m/s (0 when unknown)


def sense(pos: np.ndarray, tracks: np.ndarray, seed: int = 0, *, range_m: float = RANGE_M,
          p_detect: float = P_DETECT, sigma_m: float = POS_SIGMA_M) -> Sensed:
    """What each robot's obstacle sensing reports. pos (T, N, 2), tracks (T, K, 2)."""
    rng = np.random.default_rng(90_001 + seed)
    t_n, n, k_n = pos.shape[0], pos.shape[1], tracks.shape[1]
    seen = np.zeros((t_n, n, k_n), dtype=bool)
    est = np.zeros((t_n, n, k_n, 2), dtype=np.float32)
    for t in range(t_n):
        d = np.hypot(tracks[t][None, :, 0] - pos[t][:, None, 0], tracks[t][None, :, 1] - pos[t][:, None, 1])
        seen[t] = (d < range_m) & ~_rack_between(pos[t], tracks[t]) & (rng.random((n, k_n)) < p_detect)
        est[t] = tracks[t][None, :, :] + rng.normal(0, sigma_m, size=(n, k_n, 2))
    vel = np.zeros_like(est)
    both = seen[VEL_BASE_TICKS:] & seen[:-VEL_BASE_TICKS]
    vel[VEL_BASE_TICKS:] = np.where(both[..., None], (est[VEL_BASE_TICKS:] - est[:-VEL_BASE_TICKS]) / (VEL_BASE_TICKS * TICK_S), 0.0)
    return Sensed(seen, est, vel)


FLEET_LATENCY_TICKS = 2                # a shared report is 200 ms old when it arrives
FLEET_NEAR_M = 25.0                    # the fleet model only looks at obstacles this close to the robot


def sense_fleet(pos: np.ndarray, tracks: np.ndarray, seed: int = 0, *, latency_ticks: int = FLEET_LATENCY_TICKS, **kw) -> Sensed:
    """Fleet-shared sensing: every robot is told about each forklift that any
    robot detects. A shared report arrives `latency_ticks` late and is moved
    forward along its estimated velocity; a robot's own fresh detection wins
    over a shared one. No robot sees a forklift that no robot detects."""
    own = sense(pos, tracks, seed, **kw)
    t_n, n, k_n = own.seen.shape
    seen, est, vel = own.seen.copy(), own.pos.copy(), own.vel.copy()
    cols = np.arange(k_n)
    for t in range(latency_ticks, t_n):
        det = own.seen[t - latency_ticks]                          # (N, K)
        who = np.argmax(det, axis=0)                               # one reporting robot per forklift
        known = det.any(0)
        v = own.vel[t - latency_ticks, who, cols]
        p = own.pos[t - latency_ticks, who, cols] + v * (latency_ticks * TICK_S)
        use = known[None, :] & ~own.seen[t]                        # shared report where the robot has none of its own
        seen[t] |= use
        est[t] = np.where(use[..., None], p[None], est[t])
        vel[t] = np.where(use[..., None], v[None], vel[t])
    return Sensed(seen, est, vel)


def _blocked_by(start, sites, blk, mask, frac, radius):
    """start (N, 2), sites (S, 2), blk (N, K, 2), mask (N, K) -> (N, S)."""
    d = sites[None, :, :] - start[:, None, :]                      # (N, S, 2)
    rel = blk[:, None, :, :] - start[:, None, None, :]             # (N, 1, K, 2)
    len2 = np.maximum((d * d).sum(-1), 1e-9)[:, :, None]           # (N, S, 1)
    t = (rel * d[:, :, None, :]).sum(-1) / len2                    # (N, S, K)
    perp = np.abs(rel[..., 0] * d[:, :, None, 1] - rel[..., 1] * d[:, :, None, 0]) / np.sqrt(len2)
    return ((t > 0) & (t <= frac) & (perp < radius) & mask[:, None, :]).any(-1)


def truth_blocked(pos: np.ndarray, sites: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """(T, N, S) bool: the moving-blocker truth, from positions alone."""
    full = np.ones((pos.shape[1], tracks.shape[1]), dtype=bool)
    return np.stack([_blocked_by(pos[t], sites, np.broadcast_to(tracks[t][None], (pos.shape[1],) + tracks[t].shape), full,
                                 REACH_FRAC, RADIUS_M) for t in range(pos.shape[0])])


def geometric_foresight(pos: np.ndarray, sites: np.ndarray, s: Sensed, *, frac: float = REACH_FRAC,
                        radius: float = RADIUS_M) -> np.ndarray:
    """(T, N, S) bool. Each sensed obstacle keeps its estimated velocity; the
    robot follows its own plan; a path is flagged if an obstacle will sit on
    the first `frac` of the line to that AP at any checked time in the next
    second. `frac` is site knowledge (AP height against obstacle height)."""
    t_n = pos.shape[0]
    out = np.zeros((t_n, pos.shape[1], sites.shape[0]), dtype=bool)
    for t in range(t_n):
        if not s.seen[t].any():
            continue
        for h in HORIZONS:
            own = pos[min(t + h, t_n - 1)]
            out[t] |= _blocked_by(own, sites, s.pos[t] + s.vel[t] * (h * TICK_S), s.seen[t], frac, radius)
    return out


def pair_features(pos: np.ndarray, sites: np.ndarray, s: Sensed, t: int):
    """Features for every robot-AP pair at tick t, in a frame aligned with the
    line to the AP: line length; the robot's own planned velocity; then for the
    nearest sensed obstacles: distance along the line, distance off it, and
    velocity along and across it. No site geometry is given.
    Returns (N, S, N_FEATURES) float32 and (N,) distance to the nearest sensed obstacle (inf if none)."""
    t_n, n = pos.shape[0], pos.shape[1]
    d = sites[None, :, :] - pos[t][:, None, :]                              # (N, S, 2)
    length = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-6)
    ux, uy = d[..., 0] / length, d[..., 1] / length                         # unit vector to the AP
    own_v = (pos[min(t + VEL_BASE_TICKS, t_n - 1)] - pos[t]) / (VEL_BASE_TICKS * TICK_S)
    f = np.zeros((n, sites.shape[0], N_FEATURES), dtype=np.float32)
    f[..., 0] = length
    f[..., 1] = own_v[:, None, 0] * ux + own_v[:, None, 1] * uy
    f[..., 2] = -own_v[:, None, 0] * uy + own_v[:, None, 1] * ux
    dist = np.where(s.seen[t], np.hypot(s.pos[t][..., 0] - pos[t][:, None, 0], s.pos[t][..., 1] - pos[t][:, None, 1]), np.inf)
    order = np.argsort(dist, axis=1)[:, :MAX_OBJECTS]                       # (N, M)
    for m in range(MAX_OBJECTS):
        idx = order[:, m]
        ok = np.isfinite(dist[np.arange(n), idx])
        rel = s.pos[t][np.arange(n), idx] - pos[t]                          # (N, 2)
        v = s.vel[t][np.arange(n), idx]
        base = 3 + 5 * m
        f[..., base] = np.where(ok[:, None], rel[:, None, 0] * ux + rel[:, None, 1] * uy, 0.0)
        f[..., base + 1] = np.where(ok[:, None], -rel[:, None, 0] * uy + rel[:, None, 1] * ux, 0.0)
        f[..., base + 2] = np.where(ok[:, None], v[:, None, 0] * ux + v[:, None, 1] * uy, 0.0)
        f[..., base + 3] = np.where(ok[:, None], -v[:, None, 0] * uy + v[:, None, 1] * ux, 0.0)
        f[..., base + 4] = ok[:, None]
    return f, dist.min(axis=1)


def training_rows(pos, sites, s: Sensed, truth: np.ndarray, every: int = 2, near_m: float = np.inf):
    """(rows, N_FEATURES), (rows,) labels: pairs whose robot knows of an obstacle within `near_m`,
    every `every` ticks. Label: the path is blocked at some time in the next second."""
    from ..native.blockers import truth_foresight
    target = truth_foresight(truth, FORESEE_BLOCKS)
    xs, ys = [], []
    for t in range(0, pos.shape[0] - FORESEE_BLOCKS, every):
        f, nearest = pair_features(pos, sites, s, t)
        has = nearest < near_m
        if has.any():
            xs.append(f[has].reshape(-1, N_FEATURES))
            ys.append(target[t][has].reshape(-1))
    return np.concatenate(xs), np.concatenate(ys)


class LearnedForesight:
    """Small network from sensed obstacles to "this path is blocked within a second"."""

    def __init__(self, seed: int = 0, hidden: int = 64, epochs: int = 20, threshold: float = 0.5,
                 near_m: float = float("inf")):
        self.net = SmallNet(hidden=hidden, epochs=epochs, seed=seed)
        self.threshold = threshold
        self.near_m = near_m           # only robots with a known obstacle this close are scored

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LearnedForesight":
        self.net.fit(x, y)
        return self

    def foresight(self, pos, sites, s: Sensed) -> np.ndarray:
        t_n = pos.shape[0]
        out = np.zeros((t_n, pos.shape[1], sites.shape[0]), dtype=bool)
        for t in range(t_n):
            f, nearest = pair_features(pos, sites, s, t)
            has = nearest < self.near_m
            if has.any():
                p = self.net.predict(f[has].reshape(-1, N_FEATURES)).reshape(-1, sites.shape[0])
                out[t][has] = p > self.threshold
        return out

    def save(self, path) -> None:
        import torch
        torch.save({"state": self.net.net.state_dict(), "mu": self.net.mu, "sd": self.net.sd, "hidden": self.net.hidden,
                    "threshold": self.threshold, "near_m": self.near_m}, path)

    @classmethod
    def load(cls, path) -> "LearnedForesight":
        import torch
        d = torch.load(path, weights_only=False)
        m = cls(hidden=d["hidden"], threshold=d["threshold"], near_m=d.get("near_m", float("inf")))
        m.net.mu, m.net.sd = d["mu"], d["sd"]
        m.net.net = torch.nn.Sequential(torch.nn.Linear(N_FEATURES, d["hidden"]), torch.nn.ReLU(),
                                        torch.nn.Linear(d["hidden"], d["hidden"]), torch.nn.ReLU(),
                                        torch.nn.Linear(d["hidden"], 1))
        m.net.net.load_state_dict(d["state"])
        return m


def score(pred: np.ndarray, truth_ahead: np.ndarray) -> dict:
    """pred, truth_ahead: same-shape bool arrays."""
    y, p = truth_ahead.astype(bool), pred.astype(bool)
    return {"caught": float(p[y].mean()) if y.any() else float("nan"),
            "false_alarm_rate": float(p[~y].mean()) if (~y).any() else float("nan"),
            "precision": float(y[p].mean()) if p.any() else float("nan"), "warnings": int(p.sum()), "events": int(y.sum())}
