"""Bounded discrete, one-action receding-horizon planning proposals.

The world model is a parameter, so representation attribution swaps only it.
This first slice holds one candidate action over a multi-horizon forecast;
multi-step CEM/MPPI is rejected until a validated transition model exists.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
from .cost import evaluate
from .spec import MissionSpec
from ..worldmodel.interface import CandidateAction, SceneState, immutable_array


@dataclass(frozen=True)
class PlanProposal:
    action_index: np.ndarray
    objective: np.ndarray
    reason: tuple[str, ...]
    evaluated_actions: int
    elapsed_seconds: float
    deadline_overrun: bool
    requires_existing_gate: bool = True
    production_qualified: bool = False
    actuation_authorized: bool = False

    def __post_init__(self):
        raw = np.asarray(self.action_index)
        if raw.dtype.kind not in 'iu' or np.any(raw < -1):
            raise ValueError('Plan indices must be integers or -1 for no proposal')
        index = immutable_array(raw, dtype=np.int64, ndim=1, name='plan indices')
        objective = np.array(self.objective, dtype=float, copy=True)
        if (objective.shape != index.shape or not np.isfinite(objective[index >= 0]).all()
                or not np.all(objective[index < 0] == np.inf)):
            raise ValueError('Malformed plan objective')
        if (len(self.reason) != len(index) or any(not isinstance(x,str) or not x for x in self.reason)
                or type(self.evaluated_actions) is not int or self.evaluated_actions < 0
                or not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0
                or type(self.deadline_overrun) is not bool):
            raise ValueError('Malformed plan evidence')
        if (self.requires_existing_gate is not True or self.production_qualified is not False
                or self.actuation_authorized is not False):
            raise ValueError('A proposal cannot authorize actuation or bypass the independent gate')
        objective = np.frombuffer(objective.tobytes(), dtype=objective.dtype)
        object.__setattr__(self, 'action_index', index)
        object.__setattr__(self, 'objective', objective)


class DiscreteMPC:
    def __init__(self, *, max_candidates=64, deadline_seconds=.1, depth=1, clock=time.monotonic):
        if type(max_candidates) is not int or max_candidates < 1:
            raise ValueError('A finite positive candidate budget is required')
        if (isinstance(deadline_seconds, bool) or not math.isfinite(deadline_seconds)
                or deadline_seconds <= 0):
            raise ValueError('A positive deadline is required')
        if depth != 1:
            raise ValueError('Multi-step transitions/CEM/MPPI are not validated in this slice')
        self.max_candidates, self.deadline_seconds, self.clock = max_candidates, deadline_seconds, clock

    def plan(self, world_model, state: SceneState, actions, mission: MissionSpec,
             *, horizon=30., timescale='mid', independently_qualified_heads=()):
        if not isinstance(state, SceneState) or not isinstance(mission, MissionSpec):
            raise TypeError('Use a predicted SceneState and a typed mission')
        actions = tuple(actions)
        if not actions or any(not isinstance(action, CandidateAction) for action in actions):
            raise TypeError('Use explicit candidate actions')
        if len({a.action_id for a in actions}) != len(actions):
            raise ValueError('Duplicate action IDs')
        n = len(state.current_ap)
        best, cost = np.full(n, -1, dtype=np.int64), np.full(n, np.inf)
        reasons = ['no_feasible_supported_action'] * n
        start = self.clock()
        count, overrun = 0, False
        for j, action in enumerate(actions[:self.max_candidates]):
            if self.clock() - start >= self.deadline_seconds:
                break
            outcome = world_model.query(state, (action,), horizon, timescale)
            if (outcome.action_ids != (action.action_id,) or len(outcome.outage_probability) != n
                    or not np.array_equal(outcome.times_s,state.times_s)):
                raise ValueError('World-model outcome does not match requested action/fleet/source horizons')
            count += 1
            now = self.clock()
            if now - start > self.deadline_seconds:
                # A synchronous model call cannot be preempted here. Its late
                # result is discarded; no hard real-time claim is made.
                overrun = True
                break
            value = evaluate(outcome, mission, horizon,
                             independently_qualified_heads=independently_qualified_heads)
            improve = value.feasible[:, 0] & (value.objective[:, 0] < cost)
            best[improve], cost[improve] = j, value.objective[improve, 0]
            for robot in np.flatnonzero(improve):
                reasons[robot] = 'proposal_requires_independent_existing_gate'
            for robot in np.flatnonzero((best < 0) & bool(value.unavailable_heads)):
                reasons[robot] = 'unqualified_or_missing_head:' + ','.join(value.unavailable_heads)
        elapsed = self.clock() - start
        if count == 0:
            reasons = ['budget_exhausted_hold_external_fallback'] * n
        elif overrun:
            for robot in np.flatnonzero(best < 0):
                reasons[robot] = 'late_forecast_discarded_hold_external_fallback'
        return PlanProposal(best, cost, tuple(reasons), count, elapsed, overrun)
