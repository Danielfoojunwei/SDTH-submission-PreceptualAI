"""Plan -> radio graph features (robots, BSSIDs, global terms).

Everything is read from the exact batched evaluator, so the model sees the
same physics the oracle scores: per-robot SINR, service and reconnection,
per-BSSID airtime and protection, and the plan-level terms of the mission
utility, each averaged over now + the planned-route snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from ..fastplan import BatchEvaluator

ROBOT_RE = re.compile(r"\b(?:amr-\d{3}|car-\d{2})\b")
ROBOT_F = 12
ROBOT_F_POS = 14   # + normalised x, y (E2 onward)
BSS_F = 7
GLOBAL_F = 10


@dataclass
class PlanGraph:
    robot: np.ndarray        # (n, ROBOT_F)
    bss: np.ndarray          # (m, BSS_F)
    assign: np.ndarray       # (n,) BSS index of each robot
    glob: np.ndarray         # (GLOBAL_F,)


def named_robots(ev: BatchEvaluator, mission_text: str) -> np.ndarray:
    names = set(ROBOT_RE.findall(mission_text))
    return np.array([c in names for c in ev.car_ids], dtype=float)


def plan_graph(ev: BatchEvaluator, a: np.ndarray, named: np.ndarray,
               positions: tuple[np.ndarray, np.ndarray] | None = None) -> PlanGraph:
    A = a[None, :]
    terms = [ev._snapshot_terms(A, snap, k == 0, 30.0) for k, snap in enumerate(ev.snaps)]
    t0 = terms[0]
    n = ev.n
    sat_all = np.stack([t["sat"][0] for t in terms])           # (S, n)
    lost = np.zeros(n, dtype=bool)
    for t in terms[1:]:
        lost |= ~t["pos"][0]
    rows = np.arange(n)
    dl = ev.snaps[0].dl[rows, a]
    video = (ev.cf_dem[:, 1:] > 0).any(1) if ev.cf_dem.shape[1] > 1 else np.zeros(n, bool)
    moved = (a != ev.current).astype(float)
    robot = np.stack([
        t0["sinr"][0] / 40.0, t0["pos"][0].astype(float), sat_all[0], sat_all.min(0),
        moved, ev.crit / 10.0, ev.be / 20.0, video.astype(float), named,
        ev.is5[a].astype(float), (dl + 60.0) / 30.0, lost.astype(float)], axis=1)
    if positions is not None:
        robot = np.concatenate([robot, np.stack([positions[0] / 120.0,
                                                 positions[1] / 80.0], 1)], 1)
    m = ev.m
    act = t0["activity_in"][0]
    count = np.bincount(a, minlength=m)
    d = ev.dom_of_ap
    need = (t0["crit_need"][0] + t0["be_need"][0])[d] / np.maximum(ev.avail[d], 1e-6)
    occ = 1.0 - ev.avail[d]
    bss = np.stack([act, occ, ev.is5.astype(float), count / 20.0,
                    np.minimum(need, 3.0) / 3.0, ev.ext.astype(float),
                    (count > 0).astype(float)], axis=1)
    keys = ("U", "V", "jain", "W", "B")
    glob_terms = [np.mean([t[k][0] for t in terms]) for k in keys]
    glob = np.array(glob_terms + [
        t0["handovers"][0] / n * 10, np.mean([t["viol"][0] for t in terms]) / n * 10,
        np.mean([t["prot"][0] for t in terms]) / n, lost.mean(),
        float((sat_all.min(0) * named).sum() / max(named.sum(), 1))])
    return PlanGraph(robot.astype(np.float32), bss.astype(np.float32),
                     a.astype(np.int64), glob.astype(np.float32))
