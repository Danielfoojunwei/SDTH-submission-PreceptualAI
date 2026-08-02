"""Local differential privacy for federated contributions.

Adapted from Zhao, Zhao, Yang, Wang, Wang, Lyu, Niyato and **Kwok-Yan Lam**,
"Local Differential Privacy based Federated Learning for Internet of Things",
IEEE IoT-J, 2021 ([arXiv:2004.08856](https://arxiv.org/abs/2004.08856)), which
perturbs each participant's shared values locally — before they leave the
device — using the Piecewise Mechanism and variants, so the aggregator never
sees any individual's true contribution.

**Why this system needs it.** The machine-unlearning paper in the same corpus
(ICAIIC 2025) makes the point that a spectrum model trained on behaviour and
location data leaks that data. Applied here: a fleet baseline built from each
node's exact feature means encodes **where the sensors are and what each one
sees**. For a defensive receive-only fleet whose value depends partly on siting
discretion, that is a counter-intelligence exposure, not merely a privacy one.
Sharing only sufficient statistics narrows it; it does not close it, because a
node's exact mean is still its exact mean.

The Piecewise Mechanism (Wang et al., ICDE 2019, as used by the paper above)
gives an **unbiased** perturbation of a bounded numeric value: individual
values are protected, while the aggregate mean remains correct in expectation
and its error shrinks as more nodes contribute.

Three properties are enforced, and one deliberate inconsistency is explained:

**Randomness here is real, unlike elsewhere in this repository.** Particle-swarm
tuning in ``forecast.py`` is seeded deterministically so a forecast is
reconstructable from the audit ledger. Privacy is the opposite requirement: a
deterministic "perturbation" is not a perturbation at all, and an adversary who
knows the seed recovers the true value exactly. This module uses
``secrets.SystemRandom``. Auditability is preserved at the correct level — the
ledger records *that* perturbation was applied and with what budget, never the
unperturbed value, since recording that would defeat the mechanism.

**The budget composes and is not free.** Each release spends privacy budget;
repeated rounds over the same underlying data compose, and ``epsilon`` should
be understood as per-release. Utility cost is reported alongside every
perturbation so the trade is visible rather than assumed.

**Bounds are operator-declared.** The mechanism requires a known range per
feature. A value outside its declared range is clamped before perturbation —
never used to widen the range, which would itself leak.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any, Sequence

from .validation import bounded_number, require_finite

#: Default per-release privacy budget. Smaller is more private and noisier.
#: 1.0 is a commonly used middle setting; it is a policy choice, not a
#: universally correct value, and an operator should set it deliberately.
DEFAULT_EPSILON = 1.0

#: Budget below which the noise dominates any realistic fleet size.
MINIMUM_USEFUL_EPSILON = 0.01


@dataclass(frozen=True)
class FeatureRange:
    """Operator-declared bounds for a feature, required by the mechanism."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("feature range bounds must be finite")
        if self.high <= self.low:
            raise ValueError("feature range high must exceed low")

    def normalise(self, value: float) -> float:
        """Map into [-1, 1], clamping out-of-range values rather than widening."""
        midpoint = (self.high + self.low) / 2.0
        half_span = (self.high - self.low) / 2.0
        scaled = (require_finite(value, "value") - midpoint) / half_span
        return max(-1.0, min(1.0, scaled))

    def denormalise(self, value: float) -> float:
        midpoint = (self.high + self.low) / 2.0
        half_span = (self.high - self.low) / 2.0
        return midpoint + value * half_span


class PiecewiseMechanism:
    """Unbiased LDP perturbation of a value in [-1, 1].

    For budget ``epsilon`` the perturbed output has the same expectation as the
    input, with variance that falls as the budget grows. Averaging over ``n``
    contributors reduces the aggregate error as O(1/sqrt(n)) — which is why
    this is usable for a fleet baseline but not for a single node's reading.
    """

    def __init__(self, *, epsilon: float = DEFAULT_EPSILON) -> None:
        self.epsilon = bounded_number(epsilon, "epsilon", MINIMUM_USEFUL_EPSILON, 50.0)
        self._random = secrets.SystemRandom()

    def _constant(self) -> float:
        half = math.exp(self.epsilon / 2.0)
        return (half + 1.0) / (half - 1.0)

    def perturb(self, value: float) -> float:
        """Perturb a value already normalised to [-1, 1]."""
        t = max(-1.0, min(1.0, require_finite(value, "value")))
        constant = self._constant()
        left = (constant + 1.0) / 2.0 * t - (constant - 1.0) / 2.0
        right = left + constant - 1.0
        half = math.exp(self.epsilon / 2.0)
        if self._random.random() < half / (half + 1.0):
            # Inside the high-probability interval around the true value.
            return self._random.uniform(left, right)
        # Outside it: one of the two tails, chosen in proportion to their width.
        left_width = left + constant
        right_width = constant - right
        total = left_width + right_width
        if total <= 0.0:
            return t
        if self._random.random() < left_width / total:
            return self._random.uniform(-constant, left)
        return self._random.uniform(right, constant)

    def expected_absolute_error(self, contributors: int) -> float:
        """Approximate error of the aggregate mean, in normalised units.

        Variance of one perturbed value under the piecewise mechanism scales
        roughly as 4/(e^{eps/2} - 1)^2; averaging n of them divides it by n.
        This is an estimate for operator guidance, not a tight bound.
        """
        if contributors < 1:
            return math.inf
        half = math.exp(self.epsilon / 2.0)
        single = 2.0 / max(1e-9, half - 1.0)
        return single / math.sqrt(contributors)


@dataclass(frozen=True)
class PrivatisedValue:
    feature: str
    perturbed: float
    epsilon: float
    contributors_needed_for_1pct: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "perturbed": round(self.perturbed, 6),
            "epsilon": self.epsilon,
            "contributors_needed_for_1pct": self.contributors_needed_for_1pct,
            "reasons": list(self.reasons),
            # The true value is deliberately absent. Recording it would defeat
            # the mechanism this record exists to attest.
        }


class LocalPrivatiser:
    """Applies LDP to a node's feature means before they leave the node."""

    def __init__(
        self,
        ranges: dict[str, FeatureRange],
        *,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        if not ranges:
            raise ValueError("at least one feature range is required")
        self.ranges = ranges
        self.epsilon = epsilon
        # The budget is split across the features released together, because
        # releasing k values at full budget each spends k*epsilon overall.
        self._per_feature = PiecewiseMechanism(epsilon=max(MINIMUM_USEFUL_EPSILON, epsilon / len(ranges)))

    @property
    def per_feature_epsilon(self) -> float:
        return self._per_feature.epsilon

    def privatise(self, feature: str, value: float) -> PrivatisedValue:
        feature_range = self.ranges.get(feature)
        if feature_range is None:
            raise KeyError(f"no declared range for feature {feature!r}")
        normalised = feature_range.normalise(value)
        perturbed = feature_range.denormalise(self._per_feature.perturb(normalised))
        reasons: list[str] = ["PERTURBED_LOCALLY_BEFORE_RELEASE"]
        if self.per_feature_epsilon <= 0.1:
            reasons.append("BUDGET_VERY_SMALL_EXPECT_HEAVY_NOISE")
        needed = self._contributors_for_target(0.01)
        return PrivatisedValue(
            feature=feature,
            perturbed=perturbed,
            epsilon=self.per_feature_epsilon,
            contributors_needed_for_1pct=needed,
            reasons=tuple(reasons),
        )

    def _contributors_for_target(self, target_normalised_error: float) -> int:
        """How many contributors the fleet needs for a given aggregate error."""
        single = self._per_feature.expected_absolute_error(1)
        if not math.isfinite(single) or target_normalised_error <= 0:
            return 0
        return max(1, math.ceil((single / target_normalised_error) ** 2))

    def privatise_all(self, values: dict[str, float]) -> list[PrivatisedValue]:
        return [self.privatise(name, value) for name, value in sorted(values.items()) if name in self.ranges]


def aggregate_privatised(values: Sequence[float]) -> float:
    """Mean of perturbed contributions.

    The piecewise mechanism is unbiased, so the plain mean is the right
    estimator: the injected noise cancels in expectation. Note the tension with
    robust aggregation — trimming or consensus filtering *also* removes
    legitimate noise and biases the result. An operator must choose: LDP for
    privacy, or consensus filtering for poisoning resistance. Applying both to
    the same values weakens each.
    """
    if not values:
        raise ValueError("no values to aggregate")
    return sum(require_finite(value, "value") for value in values) / len(values)
