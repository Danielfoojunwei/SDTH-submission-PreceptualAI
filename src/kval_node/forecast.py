"""Per-flow traffic forecasting with particle-swarm-tuned parameters.

Adapted from Feng Li, Wei Nie, Kwok-Yan Lam, Bowen Shen and Xiuhua Li,
"Network Traffic Prediction Using PSO-LightGBM-TM", IEEE INFOCOM 2024
Workshops, doi:10.1109/INFOCOMWKSHPS61880.2024.10620828 (journal version:
"Network traffic prediction based on PSO-LightGBM-TM", Computer Networks,
doi:10.1016/j.comnet.2024.110838).

**What is taken.** Their central insight is not the base learner: it is that a
single model fitted across all flows is wrong, because flows differ, and that
**each flow deserves its own tuned parameters**, found by particle swarm
optimisation. They report this beating deep-learning baselines while being
structurally simpler. That argument transfers exactly to a connectivity node:
each profile — a Wi-Fi link, a cellular link, an NTN link — has its own latency
regime, and one shared smoothing constant fits none of them well.

**What is not taken, and why.** Their base learner is LightGBM, a PyPI package.
This repository has a deliberate no-PyPI-runtime-dependency policy (see
``docs/DEPENDENCIES.md``): a Jetson node installs from signed Ubuntu/NVIDIA
repositories only. Rather than break that, PSO here tunes a **damped Holt
linear-trend exponential smoother**, which is a few lines of standard-library
arithmetic. So this is *their per-flow PSO tuning strategy over a different,
dependency-free base learner* — not their model, and it inherits none of their
reported accuracy.

The forecaster is descriptive, not a learned policy: it is fitted on a
profile's own recent history at call time, holds no persistent weights, and
needs no signed model manifest. It informs an operator and can raise an early
warning; it does not select profiles. Profile selection stays with the
deterministic optimizer and the Safety Shield.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .validation import bounded_number, require_finite

#: Minimum history before a forecast means anything. Below this the series
#: cannot distinguish trend from noise.
MIN_HISTORY = 6

#: PSO population and iteration count. Small on purpose: the search space is
#: three bounded parameters, and this runs on an edge node inside a poll cycle.
DEFAULT_PARTICLES = 12
DEFAULT_ITERATIONS = 24

#: Standard PSO coefficients — inertia, cognitive pull, social pull.
INERTIA = 0.72
COGNITIVE = 1.4
SOCIAL = 1.4


@dataclass(frozen=True)
class SmoothingParameters:
    """Damped Holt linear-trend parameters, one set per flow."""

    alpha: float
    beta: float
    damping: float

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "SmoothingParameters":
        return cls(
            alpha=min(0.999, max(0.001, vector[0])),
            beta=min(0.999, max(0.0, vector[1])),
            damping=min(1.0, max(0.5, vector[2])),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "damping": round(self.damping, 6),
        }


def holt_forecast(
    series: Sequence[float], parameters: SmoothingParameters, *, horizon: int = 1
) -> tuple[float, list[float]]:
    """Damped Holt linear trend. Returns ``(forecast, one_step_fitted)``.

    The fitted values are one-step-ahead predictions made using only data
    strictly before each point, so scoring against them cannot leak the future.
    """
    values = [require_finite(value, "series value") for value in series]
    if len(values) < 2:
        raise ValueError("need at least two points to smooth")
    level = values[0]
    trend = values[1] - values[0]
    fitted: list[float] = []
    for value in values[1:]:
        prediction = level + parameters.damping * trend
        fitted.append(prediction)
        previous_level = level
        level = parameters.alpha * value + (1.0 - parameters.alpha) * prediction
        trend = parameters.beta * (level - previous_level) + (1.0 - parameters.beta) * (
            parameters.damping * trend
        )
    forecast = level
    damped = trend
    for _ in range(horizon):
        forecast += damped
        damped *= parameters.damping
    return forecast, fitted


def _mean_absolute_error(series: Sequence[float], parameters: SmoothingParameters) -> float:
    try:
        _forecast, fitted = holt_forecast(series, parameters)
    except ValueError:
        return math.inf
    actual = series[1:]
    if not fitted:
        return math.inf
    return statistics.fmean(abs(a - f) for a, f in zip(actual, fitted))


def particle_swarm_minimise(
    objective: Callable[[Sequence[float]], float],
    bounds: Sequence[tuple[float, float]],
    *,
    particles: int = DEFAULT_PARTICLES,
    iterations: int = DEFAULT_ITERATIONS,
) -> tuple[list[float], float]:
    """Minimise ``objective`` over ``bounds`` by particle swarm optimisation.

    The swarm is initialised on a deterministic lattice rather than randomly.
    ``Math.random`` would make a node's forecast unreproducible from its own
    audit record, and every other decision this node makes is reconstructable
    from the ledger; a forecast should be no different.
    """
    dimensions = len(bounds)
    if dimensions == 0:
        raise ValueError("bounds cannot be empty")

    # Deterministic spread across each dimension.
    positions: list[list[float]] = []
    for index in range(particles):
        fraction = (index + 0.5) / particles
        positions.append(
            [
                low + ((fraction * (dimension + 1)) % 1.0) * (high - low)
                for dimension, (low, high) in enumerate(bounds)
            ]
        )
    velocities = [[0.0] * dimensions for _ in range(particles)]

    personal_best = [list(position) for position in positions]
    personal_score = [objective(position) for position in positions]
    best_index = min(range(particles), key=lambda i: personal_score[i])
    global_best = list(personal_best[best_index])
    global_score = personal_score[best_index]

    for step in range(iterations):
        # Deterministic surrogate for PSO's stochastic pull coefficients.
        cognitive_pull = 0.5 + 0.5 * math.sin(step * 1.7)
        social_pull = 0.5 + 0.5 * math.cos(step * 2.3)
        for index in range(particles):
            for dimension in range(dimensions):
                low, high = bounds[dimension]
                velocity = (
                    INERTIA * velocities[index][dimension]
                    + COGNITIVE * cognitive_pull * (personal_best[index][dimension] - positions[index][dimension])
                    + SOCIAL * social_pull * (global_best[dimension] - positions[index][dimension])
                )
                span = high - low
                velocity = max(-span, min(span, velocity))
                velocities[index][dimension] = velocity
                positions[index][dimension] = max(low, min(high, positions[index][dimension] + velocity))
            score = objective(positions[index])
            if score < personal_score[index]:
                personal_score[index] = score
                personal_best[index] = list(positions[index])
                if score < global_score:
                    global_score = score
                    global_best = list(positions[index])
    return global_best, global_score


@dataclass(frozen=True)
class Forecast:
    flow_id: str
    samples: int
    predicted: float
    recent_mean: float
    mean_absolute_error: float
    parameters: SmoothingParameters
    degrading: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "samples": self.samples,
            "predicted": round(self.predicted, 6),
            "recent_mean": round(self.recent_mean, 6),
            "mean_absolute_error": round(self.mean_absolute_error, 6),
            "parameters": self.parameters.to_dict(),
            "degrading": self.degrading,
            "reasons": list(self.reasons),
        }


class PerFlowForecaster:
    """Fits and tunes one parameter set per flow, as the source paper argues.

    ``degradation_ratio`` is the fraction by which a predicted value must
    exceed the recent mean before the flow is called degrading — so ordinary
    jitter does not raise a warning.
    """

    PARAMETER_BOUNDS = ((0.001, 0.999), (0.0, 0.999), (0.5, 1.0))

    def __init__(self, *, degradation_ratio: float = 0.25) -> None:
        self.degradation_ratio = bounded_number(degradation_ratio, "degradation_ratio", 0.0, 10.0)

    def fit(self, flow_id: str, series: Sequence[float], *, horizon: int = 1) -> Forecast:
        values = [require_finite(value, "series value") for value in series]
        if len(values) < MIN_HISTORY:
            return Forecast(
                flow_id=flow_id,
                samples=len(values),
                predicted=values[-1] if values else 0.0,
                recent_mean=statistics.fmean(values) if values else 0.0,
                mean_absolute_error=math.inf,
                parameters=SmoothingParameters(0.5, 0.1, 0.9),
                degrading=False,
                reasons=(f"NEED_AT_LEAST_{MIN_HISTORY}_SAMPLES",),
            )

        best_vector, error = particle_swarm_minimise(
            lambda vector: _mean_absolute_error(values, SmoothingParameters.from_vector(vector)),
            self.PARAMETER_BOUNDS,
        )
        parameters = SmoothingParameters.from_vector(best_vector)
        predicted, _fitted = holt_forecast(values, parameters, horizon=horizon)
        recent_mean = statistics.fmean(values[-MIN_HISTORY:])

        reasons: list[str] = []
        degrading = predicted > recent_mean * (1.0 + self.degradation_ratio)
        if degrading:
            reasons.append("FORECAST_ABOVE_RECENT_MEAN_BY_MARGIN")
        if not math.isfinite(error):
            reasons.append("COULD_NOT_SCORE_FIT")
        if not reasons:
            reasons.append("WITHIN_RECENT_BEHAVIOUR")

        return Forecast(
            flow_id=flow_id,
            samples=len(values),
            predicted=predicted,
            recent_mean=recent_mean,
            mean_absolute_error=error,
            parameters=parameters,
            degrading=degrading,
            reasons=tuple(reasons),
        )

    def fit_all(self, series_by_flow: dict[str, Sequence[float]], *, horizon: int = 1) -> list[Forecast]:
        """One tuned parameter set per flow — never one shared across flows."""
        return [self.fit(flow_id, series, horizon=horizon) for flow_id, series in sorted(series_by_flow.items())]
