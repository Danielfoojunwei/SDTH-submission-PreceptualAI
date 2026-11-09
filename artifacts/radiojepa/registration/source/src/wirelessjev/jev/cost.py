"""Mission costs over the same world-model interface for every model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .spec import MissionSpec
from ..worldmodel.interface import OutcomeDistribution
from ..worldmodel.radiomap_adapter import integrate


@dataclass(frozen=True)
class CostEvaluation:
    objective: np.ndarray
    feasible: np.ndarray
    unavailable_heads: tuple[str, ...]


def evaluate(outcome: OutcomeDistribution, mission: MissionSpec, horizon: float,
             *, independently_qualified_heads=()):
    if not isinstance(outcome, OutcomeDistribution) or not isinstance(mission, MissionSpec):
        raise TypeError('Use validated outcomes and a typed mission')
    if not 0 < horizon <= outcome.times_s[-1] or abs(horizon - outcome.horizon_seconds) > 1e-12:
        raise ValueError('Objective horizon is outside forecast support')
    # PHY proxy only: the model metadata explicitly excludes queues and MAC.
    rate = integrate(outcome.rate_mbps.transpose(0, 2, 1), outcome.times_s, 0, horizon)
    objective = (mission.outage_weight * outcome.outage_integral_seconds
                 + mission.switching_weight * outcome.switch_seconds
                 - mission.rate_weight * rate)
    feasible = np.ones(objective.shape, dtype=bool)
    unknown = []
    qualified = set(independently_qualified_heads)
    if mission.max_outage_probability is not None:
        if mission.require_calibrated_outage and ('outage' not in qualified or outcome.outage_upper is None):
            feasible[:] = False
            unknown.append('outage')
        else:
            p = outcome.outage_upper if mission.require_calibrated_outage else outcome.outage_probability
            # Interpolate exactly to the requested endpoint, excluding later knots.
            end = np.searchsorted(outcome.times_s, horizon, side='right')
            past = p[..., :end]
            left = min(end - 1, len(outcome.times_s) - 2)
            fraction = ((horizon - outcome.times_s[left]) /
                        (outcome.times_s[left+1] - outcome.times_s[left]))
            endpoint = p[..., left] * (1-fraction) + p[..., left+1] * fraction
            feasible &= np.maximum(past.max(-1), endpoint) <= mission.max_outage_probability
    if mission.max_silence_seconds is not None:
        if 'silence' not in qualified or outcome.silence_upper_seconds is None:
            feasible[:] = False
            unknown.append('silence')
        else:
            feasible &= outcome.silence_upper_seconds <= mission.max_silence_seconds
    return CostEvaluation(objective, feasible, tuple(unknown))
