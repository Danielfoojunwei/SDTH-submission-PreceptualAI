"""Telemetry from the slot simulator -> features a robot could compute and
the hidden truth (is the serving path blocked now, in 0.5 s, in 1 s).

Features use only the robot's own 100 ms-old readings, the map's expectation
and its delivery counters. Nothing here reads the blockage state."""
from __future__ import annotations

import numpy as np

from ..native.sim import TELEMETRY_AHEAD_BLOCKS, simulate

WINDOW = 5                      # ticks of history per example (0.5 s)
PER_TICK = 9
HORIZONS_S = tuple(0.1 * h for h in TELEMETRY_AHEAD_BLOCKS)
RULE_DROP_DB, RULE_CONFIRM = 10.0, 2


def collect(world, policy, **sim_kwargs) -> dict:
    """Run one policy and return its telemetry as (ticks, robots, ...) arrays."""
    tel: list = []
    simulate(world, policy, telemetry=tel, **sim_kwargs)
    if not tel:
        raise ValueError("no telemetry: the policy must be a native one and the run longer than 1 s")
    return {k: np.stack([row[k] for row in tel]) for k in tel[0]}


def _baselines(level: np.ndarray, site: np.ndarray, drop_db: float):
    """Running level per robot and band, the way the rule keeps it: restart on
    a move, frozen while the reading is low. Returns the drop below it."""
    t_n, n, _ = level.shape
    base = level[0].copy()
    drop = np.zeros_like(level)
    for t in range(1, t_n):
        moved = site[t] != site[t - 1]
        base = np.where(moved[:, None], level[t], base)
        drop[t] = base - level[t]
        low = drop[t].mean(axis=1) > drop_db
        base = np.where(low[:, None], base, 0.8 * base + 0.2 * level[t])
    return drop


def rule_state(tr: dict, drop_db: float = RULE_DROP_DB, confirm: int = RULE_CONFIRM) -> np.ndarray:
    """The hand-written rule as a per-tick yes/no: the band-average level has
    been more than `drop_db` below the running level for `confirm` ticks."""
    low = _baselines(tr["level"], tr["site"], drop_db).mean(axis=2) > drop_db
    run = np.zeros(low.shape, dtype=int)
    for t in range(1, len(low)):
        run[t] = np.where(low[t], run[t - 1] + 1, 0)
    return run >= confirm


def tick_features(tr: dict) -> np.ndarray:
    """(ticks, robots, PER_TICK): shortfall against the map, drop below the
    robot's own running level, delivery misses, and move state."""
    level, expected, site = tr["level"], tr["expected"], tr["site"]
    short = expected - level                                   # below what the map expects, per band
    drop = _baselines(level, site, RULE_DROP_DB)               # below the robot's own recent level
    sent = np.diff(tr["sent"], axis=0, prepend=tr["sent"][:1])
    missed = np.diff(tr["missed"], axis=0, prepend=tr["missed"][:1])
    miss_rate = missed / np.maximum(sent, 1)
    since = np.zeros(site.shape)
    for t in range(1, len(site)):
        since[t] = np.where(site[t] != site[t - 1], 0, np.minimum(since[t - 1] + 1, 20))
    # A blocker lowers every band; a fade lowers one. min over bands separates them.
    f = [short.mean(2), short.min(2), short.max(2), drop.mean(2), drop.min(2), drop.max(2),
         miss_rate, tr["roaming"].astype(float), since / 20.0]
    return np.stack(f, axis=2).astype(np.float32)


def examples(tr: dict, window: int = WINDOW):
    """Flatten to (examples, window * PER_TICK) features, (examples, 3) labels
    and the rule's yes/no for the same rows. The first `window` ticks are dropped."""
    f = tick_features(tr)
    t_n, n, _ = f.shape
    x = np.concatenate([f[window - 1 - j: t_n - j] for j in range(window)], axis=2)   # newest first
    y = tr["blocked"][window - 1:]
    rule = rule_state(tr)[window - 1:]
    return x.reshape(-1, window * PER_TICK), y.reshape(-1, y.shape[2]), rule.reshape(-1)
