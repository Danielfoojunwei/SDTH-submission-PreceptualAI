"""Fast candidate plans for training and evaluation (FastPlanner based)."""
from __future__ import annotations

import numpy as np

from ..fastplan import FastPlanner
from ..teachers.assign import legacy_policy, local_greedy


def candidate_plans(scene, missions: dict, *, k: int = 8, seed: int = 0,
                    planner: FastPlanner | None = None) -> tuple[FastPlanner, list[np.ndarray]]:
    """Per-mission optima under the true weights, plus hold, legacy, greedy
    and perturbations of the optima; unique, at most k plans."""
    fp = planner or FastPlanner(scene)
    ev = fp.ev
    starts = [ev.current, ev.from_assoc(legacy_policy(scene))]
    plans = [ev.current.copy(), starts[1].copy(), ev.from_assoc(local_greedy(scene))]
    for m in missions.values():
        a, _, _ = fp.search(m.weights, m.focus_cars, starts=starts, max_iter=60)
        plans.append(a)
    rng = np.random.default_rng(seed)
    base = list(plans[3:])
    while len(plans) < 3 * k and base:
        a = base[int(rng.integers(len(base)))].copy()
        for _ in range(int(rng.integers(2, 8))):
            i = int(rng.integers(ev.n))
            a[i] = int(rng.choice(ev.visible[i]))
        plans.append(a)
    uniq, seen = [], set()
    for a in plans:
        key = a.tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(a)
    head, tail = uniq[: 3 + len(missions)], uniq[3 + len(missions):]
    rng.shuffle(tail)
    out = (head + tail)[:k]
    order = rng.permutation(len(out))
    return fp, [out[i] for i in order]
