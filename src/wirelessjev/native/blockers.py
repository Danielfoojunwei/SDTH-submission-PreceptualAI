"""Moving blockers: forklifts that drive the aisles and block every robot
whose line to an AP they cross (docs/MOVING_BLOCKERS.md).

The default blockage model draws an independent on/off state per robot and
AP. Real blockers are objects: one forklift blocks several robots at once,
can block a robot's paths to two APs in the same direction, and sweeps along
an aisle, so one robot's blockage tells its neighbours what is coming.

Geometry: APs are mounted high and robots are low, so the line between them
is below the top of a forklift only near the robot. A path is blocked when a
blocker is within `RADIUS_M` of that line, inside the first `REACH_FRAC` of
it measured from the robot.
"""
from __future__ import annotations

import numpy as np

from ..twin.warehouse import AISLES, CROSS_AISLES

BLOCKER_H_M, ROBOT_H_M, AP_H_M = 2.5, 0.3, 6.0
REACH_FRAC = (BLOCKER_H_M - ROBOT_H_M) / (AP_H_M - ROBOT_H_M)     # 0.386
RADIUS_M = 0.75                                                  # half the width of a forklift
SPEED_MPS = (1.0, 2.5)
DWELL_S, DWELL_PROB = (0.0, 20.0), 0.5
N_BLOCKERS = 13                                                  # calibrated on development scenes 400-401
BLOCK_S = 0.1


def blocker_tracks(seed: int, n_blocks: int, n_blockers: int = N_BLOCKERS) -> np.ndarray:
    """(n_blocks, K, 2) positions every 100 ms. Each blocker repeats: drive
    along its aisle to a random spot, maybe stop there, drive to a cross
    aisle, drive along it to another aisle."""
    rng = np.random.default_rng(31337 + seed)
    out = np.zeros((n_blocks, n_blockers, 2), dtype=np.float32)
    lo, hi = CROSS_AISLES[0] + 2.0, CROSS_AISLES[-1] - 2.0
    for b in range(n_blockers):
        x, y = float(rng.uniform(lo, hi)), float(rng.choice(AISLES))
        speed = float(rng.uniform(*SPEED_MPS))
        leg, dwell = 0, 0.0
        tx, ty = float(rng.uniform(lo, hi)), y
        for k in range(n_blocks):
            out[k, b] = (x, y)
            if dwell > 0:
                dwell -= BLOCK_S
                continue
            step = speed * BLOCK_S
            dx, dy = tx - x, ty - y
            dist = abs(dx) + abs(dy)                              # legs are axis-aligned
            if dist > step:
                x += step * np.sign(dx)
                y += step * np.sign(dy)
                continue
            x, y = tx, ty
            leg = (leg + 1) % 3
            if leg == 1:                                          # reached the spot in the aisle
                if rng.random() < DWELL_PROB:
                    dwell = float(rng.uniform(*DWELL_S))
                tx = float(rng.choice(CROSS_AISLES))
            elif leg == 2:                                        # at a cross aisle: pick the next aisle
                ty = float(rng.choice(AISLES))
            else:                                                 # in the new aisle: pick a spot
                tx = float(rng.uniform(lo, hi))
                speed = float(rng.uniform(*SPEED_MPS))
    return out


def lines_blocked(start: np.ndarray, end: np.ndarray, blockers: np.ndarray, frac: float = REACH_FRAC,
                  radius: float = RADIUS_M, per_blocker: bool = False) -> np.ndarray:
    """start (..., 2), end (..., 2) broadcastable; blockers (K, 2). True where
    any blocker is within `radius` of the line start->end, inside its first
    `frac` (measured from `start`) and ahead of `start`. `per_blocker` keeps
    the last axis (..., K) instead of reducing over blockers."""
    d = (end - start)[..., None, :]                               # (..., 1, 2)
    rel = blockers - start[..., None, :]                          # (..., K, 2)
    len2 = np.maximum((d * d).sum(-1), 1e-9)
    t = (rel * d).sum(-1) / len2                                  # (..., K) position along the line
    perp = np.abs(rel[..., 0] * d[..., 1] - rel[..., 1] * d[..., 0]) / np.sqrt(len2)
    hit = (t > 0) & (t <= frac) & (perp < radius)
    return hit if per_blocker else hit.any(-1)


def blocked_moving(world, n_blocks: int, n_blockers: int = N_BLOCKERS, pos_per_block: float = 0.5) -> np.ndarray:
    """(n_blocks, N, S) bool: robot-AP paths blocked by the moving blockers."""
    if world.pos is None:
        raise ValueError("moving blockers need robot positions (a warehouse world)")
    tracks = blocker_tracks(world.seed, n_blocks, n_blockers)
    sites = world.sites.astype(np.float32)
    out = np.zeros((n_blocks, world.n, sites.shape[0]), dtype=bool)
    last = world.pos.shape[0] - 1
    for k in range(n_blocks):
        p = world.pos[min(int(k * pos_per_block), last)].astype(np.float32)
        out[k] = lines_blocked(p[:, None, :], sites[None, :, :], tracks[k])
    return out
