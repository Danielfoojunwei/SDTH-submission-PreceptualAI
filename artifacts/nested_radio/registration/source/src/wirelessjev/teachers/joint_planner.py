"""Finite joint planner (MILP over complete-configuration candidates).

Ported from the R6 `joint_planner.py` design: each car occurs in exactly one
selected candidate, resource budgets and minimum service are linear
constraints, incompatible pairs are excluded, and the solver output is
re-validated after rounding. Solved with SciPy's HiGHS-backed `milp`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cars: tuple[str, ...]
    resource_use: Mapping[str, float]
    utility: float
    service_bits: Mapping[str, float] = field(default_factory=dict)


def plan_joint(fleet: Sequence[str], budgets: Mapping[str, float],
               candidates: Sequence[Candidate], *,
               minimum_service: Mapping[str, float] | None = None,
               incompatible: Sequence[tuple[str, str]] = (),
               time_limit_s: float = 2.0) -> dict:
    cars = tuple(fleet)
    if not cars or len(set(cars)) != len(cars):
        raise ValueError("fleet must contain unique cars")
    if not candidates:
        return {"status": "NO_CANDIDATES", "selected": []}
    minimum_service = dict(minimum_service or {})
    seen = set()
    for c in candidates:
        if c.candidate_id in seen:
            raise ValueError(f"duplicate candidate {c.candidate_id}")
        seen.add(c.candidate_id)
        if not set(c.cars) <= set(cars):
            raise ValueError(f"{c.candidate_id}: unknown car")
        if not isfinite(c.utility):
            raise ValueError(f"{c.candidate_id}: non-finite utility")
        for r, v in c.resource_use.items():
            if r not in budgets or not isfinite(v) or v < 0:
                raise ValueError(f"{c.candidate_id}: bad resource {r}")

    index = {c.candidate_id: j for j, c in enumerate(candidates)}
    rows, cols, vals, lo, hi = [], [], [], [], []

    def add_row(coeffs: Mapping[int, float], lower: float, upper: float):
        scale = max(1.0, max((abs(v) for v in coeffs.values()), default=0.0))
        r = len(lo)
        for j, v in coeffs.items():
            if v:
                rows.append(r)
                cols.append(j)
                vals.append(v / scale)
        lo.append(lower / scale if isfinite(lower) else lower)
        hi.append(upper / scale if isfinite(upper) else upper)

    for car in cars:
        add_row({j: 1.0 for j, c in enumerate(candidates) if car in c.cars}, 1, 1)
    for resource, cap in budgets.items():
        add_row({j: c.resource_use.get(resource, 0.0)
                 for j, c in enumerate(candidates)}, -np.inf, cap)
    for flow, need in minimum_service.items():
        add_row({j: c.service_bits.get(flow, 0.0)
                 for j, c in enumerate(candidates)}, need, np.inf)
    for a, b in incompatible:
        add_row({index[a]: 1.0, index[b]: 1.0}, -np.inf, 1.0)

    matrix = coo_matrix((vals, (rows, cols)),
                        shape=(len(lo), len(candidates))).tocsc()
    result = milp(
        c=-np.array([c.utility for c in candidates]),
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lo, hi),
        options={"time_limit": time_limit_s},
    )
    if result.status not in (0, 1) or result.x is None:
        return {"status": "INFEASIBLE" if result.status == 2 else "NO_PLAN",
                "selected": []}
    x = np.rint(result.x)
    measured = matrix @ x
    tol = 1e-6
    if (np.max(np.abs(result.x - x)) > tol
            or np.any(measured < np.array(lo) - tol)
            or np.any(measured > np.array(hi) + tol)):
        return {"status": "INVALID_SOLVER_OUTPUT", "selected": []}
    chosen = [c.candidate_id for c, v in zip(candidates, x) if v == 1]
    return {"status": "OPTIMAL" if result.status == 0 else "FEASIBLE",
            "selected": chosen}
