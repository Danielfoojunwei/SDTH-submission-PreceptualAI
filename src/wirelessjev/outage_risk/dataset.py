"""Telemetry from the slot simulator -> features a robot could compute and
the hidden truth (is the serving path blocked now, in 0.5 s, in 1 s).

Features use only the robot's own 100 ms-old readings, the map's expectation
and its delivery counters. Nothing here reads the blockage state."""
from __future__ import annotations

import numpy as np

from ..native_ext.sim_fleet import TELEMETRY_AHEAD_BLOCKS, simulate

WINDOW = 5                      # ticks of history per example (0.5 s)
PER_TICK = 9
FLEET_PER_TICK = 4
NEAR_M, RECENT_TICKS = 10.0, 5      # a neighbour is within 10 m; a report is recent for 0.5 s
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
    """The simulator's hand-written rule (docs/BLOCKAGE_DETECTOR.md, version 3)
    as a per-tick yes/no. The band-average level has been more than `drop_db`
    below a running level for `confirm` ticks. After a move the running level
    starts at the higher of the first reading and what the map expects there,
    and it is not updated while the reading is low."""
    level, expected, site = tr["level"].mean(axis=2), tr["expected"].mean(axis=2), tr["site"]
    out = np.zeros(level.shape, dtype=bool)
    base = np.maximum(expected[0], level[0])
    run = np.zeros(level.shape[1], dtype=int)
    for t in range(1, len(level)):
        moved = site[t] != site[t - 1]
        base = np.where(moved, np.maximum(expected[t], level[t]), base)
        low = base - level[t] > drop_db
        run = np.where(low, run + 1, 0)
        out[t] = run >= confirm
        base = np.where(low, base, 0.8 * base + 0.2 * level[t])
    return out


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


def fleet_features(tr: dict) -> np.ndarray:
    """(ticks, robots, FLEET_PER_TICK): what other robots' alarms say.

    Each robot shares its position and whether the hand-written rule is firing.
    Reports reach the others one tick (100 ms) late. Per robot: neighbours
    within 10 m alarming now; neighbours alarming in the last 0.5 s; robots on
    the same AP alarming in the last 0.5 s; closeness of the nearest robot that
    alarmed in the last 0.5 s. A robot's own alarm is never counted."""
    alarm, pos, site = rule_state(tr), tr["pos"], tr["site"]
    t_n, n = alarm.shape
    recent = np.zeros_like(alarm)
    for j in range(RECENT_TICKS):
        recent[j + 1:] |= alarm[:t_n - j - 1] if j + 1 < t_n else False
    late = np.zeros_like(alarm)
    late[1:] = alarm[:-1]
    out = np.zeros((t_n, n, FLEET_PER_TICK), dtype=np.float32)
    other = ~np.eye(n, dtype=bool)
    for t in range(t_n):
        if not recent[t].any():
            continue
        dist = np.hypot(pos[t][:, None, 0] - pos[t][None, :, 0], pos[t][:, None, 1] - pos[t][None, :, 1])
        near = (dist < NEAR_M) & other
        out[t, :, 0] = (near & late[t][None, :]).sum(1)
        out[t, :, 1] = (near & recent[t][None, :]).sum(1)
        out[t, :, 2] = ((site[t][:, None] == site[t][None, :]) & other & recent[t][None, :]).sum(1)
        d = np.where(other & recent[t][None, :], dist, np.inf).min(1)
        out[t, :, 3] = 1.0 / (1.0 + d)
    return out


def examples(tr: dict, window: int = WINDOW, fleet: bool = False):
    """Flatten to (examples, window * features per tick) features, (examples, 3)
    labels and the rule's yes/no for the same rows. `fleet` appends the other
    robots' reports to each tick. The first `window` ticks are dropped."""
    f = tick_features(tr)
    if fleet:
        f = np.concatenate([f, fleet_features(tr)], axis=2)
    t_n, n, _ = f.shape
    x = np.concatenate([f[window - 1 - j: t_n - j] for j in range(window)], axis=2)   # newest first
    y = tr["blocked"][window - 1:]
    rule = rule_state(tr)[window - 1:]
    return x.reshape(-1, window * f.shape[2]), y.reshape(-1, y.shape[2]), rule.reshape(-1)


def _rack_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n, 2), b (K, 2) -> (n, K): a rack stands on the straight line between them."""
    from ..twin.warehouse import RACK_ROWS, RACK_SEGMENTS
    ax, ay, bx, by = a[:, None, 0], a[:, None, 1], b[None, :, 0], b[None, :, 1]
    out = np.zeros((len(a), len(b)), dtype=bool)
    for yr in RACK_ROWS:
        cross = (ay - yr) * (by - yr) < 0
        x_at = ax + (yr - ay) / np.where(by == ay, 1.0, by - ay) * (bx - ax)
        for x0, x1 in RACK_SEGMENTS:
            out |= cross & (x_at >= x0) & (x_at <= x1)
    return out


def sensed_blocker_alarm(tr: dict, tracks: np.ndarray, sites: np.ndarray, ahead_ticks: int,
                         range_m: float = 10.0, window: int = WINDOW, line_of_sight: bool = False) -> np.ndarray:
    """A geometric predictor from the robot's own obstacle sensing, not from radio.

    The robot sees blockers within `range_m` (a lidar), assumes each keeps its
    current velocity, knows its own planned position `ahead_ticks` later, and
    asks whether one of them will then sit on the line to its serving AP.
    `tracks` is (ticks, K, 2) blocker positions at the telemetry ticks. Returns
    rows aligned with `examples` (the first `window` - 1 ticks dropped).
    `line_of_sight` drops blockers hidden from the robot behind a rack."""
    from ..native_ext.blockers import lines_blocked
    pos, site = tr["pos"], tr["site"]
    t_n, n = site.shape
    out = np.zeros((t_n, n), dtype=bool)
    for t in range(1, t_n):
        vel = tracks[t] - tracks[t - 1]
        future = tracks[t] + vel * ahead_ticks
        own = pos[min(t + ahead_ticks, t_n - 1)]
        seen = np.hypot(tracks[t][None, :, 0] - pos[t][:, None, 0], tracks[t][None, :, 1] - pos[t][:, None, 1]) < range_m
        if line_of_sight:
            seen &= ~_rack_between(pos[t], tracks[t])
        hit = lines_blocked(own, sites[site[t]].astype(np.float32), future, per_blocker=True)        # (n, K)
        out[t] = (hit & seen).any(1)
    return out[window - 1:].reshape(-1)

