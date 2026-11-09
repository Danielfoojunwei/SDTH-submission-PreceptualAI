"""Independent, scoped calibration tools; no drift or conditional SLA theorem.

One-sided conformal bounds below concern a scalar prediction error (for
example candidate-minus-current utility), not a calibrated Bernoulli event
probability. They require exchangeable calibration units. Correlated robots
are not silently counted as independent calibration units.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import re
import numpy as np
from ..worldmodel.interface import OutcomeDistribution, immutable_array


@dataclass(frozen=True)
class CalibrationScope:
    head: str
    site: str
    device: str

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in (self.head, self.site, self.device)):
            raise ValueError('Explicit head/site/device calibration scope is required')


@dataclass(frozen=True)
class SplitConformalBound:
    scope: CalibrationScope
    alpha: float
    quantile: float
    samples: int
    receipt_sha256: str
    observed_calibration_coverage: float

    def __post_init__(self):
        if (not isinstance(self.scope,CalibrationScope) or isinstance(self.alpha,bool)
                or not 0 < self.alpha < 1 or type(self.samples) is not int or self.samples < 1
                or not (math.isfinite(self.quantile) or self.quantile == math.inf)
                or not 0 <= self.observed_calibration_coverage <= 1
                or not isinstance(self.receipt_sha256,str)
                or not re.fullmatch('[0-9a-f]{64}',self.receipt_sha256)):
            raise ValueError('Invalid conformal calibration evidence')

    @classmethod
    def fit(cls, prediction, observation, *, unit_ids, alpha, scope, receipt_sha256):
        prediction = immutable_array(prediction, ndim=1, name='calibration prediction')
        observation = immutable_array(observation, shape=prediction.shape, name='calibration outcome')
        if (not len(prediction) or len(unit_ids) != len(prediction)
                or len(set(unit_ids)) != len(unit_ids)):
            raise ValueError('One record per distinct exchangeable calibration unit is required')
        if not isinstance(scope, CalibrationScope) or not 0 < alpha < 1:
            raise ValueError('A valid scope and alpha in (0,1) are required')
        if not isinstance(receipt_sha256, str) or not re.fullmatch('[0-9a-f]{64}', receipt_sha256):
            raise ValueError('Bind the held-out calibration evidence by SHA256')
        scores = prediction - observation  # positive = gain overprediction
        k = math.ceil((len(scores) + 1) * (1 - alpha))
        q = float(np.sort(scores)[k - 1]) if k <= len(scores) else float('inf')
        return cls(scope, float(alpha), q, len(scores), receipt_sha256, float(np.mean(scores <= q)))

    def lower(self, prediction, *, scope):
        if scope != self.scope:
            raise ValueError('Unseen site/device/head requires fallback or new calibration')
        value = np.asarray(prediction, dtype=float)
        if not np.isfinite(value).all():
            raise ValueError('Finite predictions are required')
        return value - self.quantile

    def monitor(self, prediction, observation, *, scope, tolerance=.05):
        """Empirical drift veto, not restored conformal validity under drift."""
        if not 0 <= tolerance < 1:
            raise ValueError('Invalid coverage tolerance')
        lower = self.lower(prediction, scope=scope)
        actual = np.asarray(observation, dtype=float)
        if actual.shape != lower.shape or not np.isfinite(actual).all() or not actual.size:
            raise ValueError('Aligned nonempty observed outcomes are required')
        coverage = float(np.mean(actual >= lower))
        return {'samples': actual.size, 'empirical_coverage': coverage,
                'informative_bound': math.isfinite(self.quantile),
                'fallback_required': bool(not math.isfinite(self.quantile)
                                          or coverage < 1 - self.alpha - tolerance),
                'coverage_guaranteed_under_drift': False,
                'probability_of_SLA_violation_bounded': False}


def route_heads(primary: OutcomeDistribution, fallback: OutcomeDistribution,
                *, independently_qualified_heads=()):
    """Per-head routing controlled outside the model; default is all fallback.

    The caller must obtain qualifications from its independent promotion
    contract. A model's own SceneState.qualified_head cannot activate a head.
    This function neither creates qualifications nor mutates existing gates.
    """
    if (primary.action_ids != fallback.action_ids or not np.array_equal(primary.times_s, fallback.times_s)
            or primary.outage_probability.shape != fallback.outage_probability.shape):
        raise ValueError('Head routing requires identical candidate/fleet/time axes')
    heads = set(independently_qualified_heads)
    if not heads <= {'outage', 'rate', 'silence'}:
        raise ValueError('Unsupported head qualification')
    risk = primary if 'outage' in heads else fallback
    rate = primary if 'rate' in heads else fallback
    silence = primary if 'silence' in heads else fallback
    # Switching cost is a separately declared physical action cost, not a head.
    if not np.array_equal(primary.switch_seconds, fallback.switch_seconds):
        raise ValueError('A learned head cannot change declared switching costs')
    metadata = dict(fallback.metadata)
    metadata.update({f'{head}_source': primary.source_id if head in heads else fallback.source_id
                     for head in ('outage', 'rate', 'silence')})
    metadata['qualification'] = 'independent_caller_supplied_per_head_permission'
    return replace(fallback, outage_probability=risk.outage_probability,
        outage_integral_seconds=risk.outage_integral_seconds, outage_upper=risk.outage_upper,
        rate_mbps=rate.rate_mbps, silence_upper_seconds=silence.silence_upper_seconds,
        source_id='per-head-routing', metadata=metadata)
