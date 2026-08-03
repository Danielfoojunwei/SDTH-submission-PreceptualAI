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

**Where the budget goes is a judgement, and it is written down.** Tian, Wang,
Jia, Wang, Chen and Kam, "Privacy-Preserving Federated Radio Map Learning for
Wireless Digital Twins via Adaptive Noise Allocation"
([arXiv:2605.15560](https://arxiv.org/abs/2605.15560)), find that location
leakage is *concentrated* rather than spread evenly over what a node releases:
"where noise is injected matters more than how much noise is injected".
Perturbing every coordinate identically spends effort on quantities that say
little about where a node is, and under-protects the few that say a great deal.
:data:`SITING_SENSITIVITY` classifies each feature by how far releasing it
narrows the node's physical position, and :func:`allocate_budget` gives the
sensitive ones a *smaller* share of epsilon — a tighter budget, therefore more
noise — funded entirely by the coordinates that carry little siting
information. The total is unchanged; only its distribution moves. Uniform
allocation stays the default because it is the behaviour the composition
argument below is simplest about, and the one an operator can reason through
unaided.

**What is guaranteed and what is not.** Stated precisely, because implying more
than was derived is exactly how a privacy claim becomes a liability.

*Guaranteed.* Every released value is perturbed by the Piecewise Mechanism at
that feature's assigned budget, and the mechanism's per-value property does not
depend on how the budget was divided — allocation decides only which epsilon
lands where. The assigned budgets sum to the declared total, which is asserted
by test rather than asserted in prose. There is one exception and it is
reported rather than hidden: a total too small to give every feature the
minimum usable budget leaves every feature at the floor, so the release spends
*more* than was requested. Both allocations behave identically in that regime,
so choosing an allocation never changes what is spent.

*Claimed with care.* Under sequential composition, releasing k coordinates
perturbed with independent randomness at eps_1..eps_k is LDP with epsilon equal
to their sum. Three conditions carry that argument here and each is a real
assumption rather than a formality: per-feature ranges are operator-declared
rather than derived from the data; the allocation is static — a table of
feature names, not a function of any observed value, unlike the learned
allocator in the paper above, which is precisely why their accounting is harder
than this one; and the *set* of features released must be fixed by policy,
because which coordinates appear is itself a release and no budget here covers
it. The sum is an upper bound, and a loose one. No (epsilon, delta) statement
is made, no advanced composition is used, and nothing accounts across rounds —
repeated releases over overlapping data compose, and this module does not track
that.

*Not claimed.* That concentrating the noise gives a better privacy-utility
trade in this system. The paper measured that in its own setting, against its
own attacker (45.63 m attacker localisation error against 36.78 m for uniform,
at better reconstruction); nothing here reproduces it, and the classification
below is reasoning about radio physics rather than a measurement on this fleet.
The conserved quantity is not even the same: theirs is noise energy, ours is
epsilon, and the two do not move together. Because the mechanism's variance is
convex in epsilon, an unequal split at matched total epsilon injects *more*
total noise than a uniform one. The gain is that the noise now sits on the
coordinates that reveal position; the cost is accuracy on those same
coordinates. That is a redistribution, not a free lunch, and the tests assert
the cost so this docstring cannot quietly imply it away. Nor is any formal
guarantee claimed for the implementation itself: the mechanism is proved over
the reals and implemented in binary floating point, and implementations of
continuous mechanisms are known to leak through their low-order bits (Mironov,
CCS 2012) — nothing here defends against that. Finally, the budget covers the
values passed to :meth:`LocalPrivatiser.privatise` and nothing else. Any
statistic released beside them — a count, a spread — is outside it.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .validation import bounded_number, require_finite

#: Default per-release privacy budget. Smaller is more private and noisier.
#: 1.0 is a commonly used middle setting; it is a policy choice, not a
#: universally correct value, and an operator should set it deliberately.
DEFAULT_EPSILON = 1.0

#: Budget below which the noise dominates any realistic fleet size.
MINIMUM_USEFUL_EPSILON = 0.01

#: Split the budget evenly. Simple, well understood, and what every existing
#: deployment already gets: no feature is treated as more revealing than any
#: other, because no judgement about which ones are has been applied.
UNIFORM_ALLOCATION = "uniform"

#: Concentrate the noise on the features that narrow where the node is, under
#: the same total budget. Called adaptive allocation in arXiv:2605.15560; named
#: for what it keys on here, since this allocation is static rather than
#: learned from the data.
SITING_AWARE_ALLOCATION = "siting_aware"

BUDGET_ALLOCATIONS = (UNIFORM_ALLOCATION, SITING_AWARE_ALLOCATION)

#: Uniform remains the default. An allocation that concentrates noise rests on
#: the classification below being right about a particular deployment's radios
#: and surroundings; that is an operator's call to make deliberately, not a
#: behaviour to inherit by upgrading.
DEFAULT_ALLOCATION = UNIFORM_ALLOCATION

SITING_LEAK_HIGH = "high"
SITING_LEAK_MODERATE = "moderate"
SITING_LEAK_LOW = "low"

#: How much releasing a feature narrows *where the node physically is*. The
#: question behind every entry is the same: given this number, a map, and the
#: published positions of the emitters in the area, how much smaller does the
#: set of places this node could be become? Quantities that are monotone in
#: path loss shrink it hard; quantities determined by how the node was built or
#: by the network behind it barely shrink it at all. Each entry says which and
#: why, because this table is the judgement the mechanism rests on.
SITING_SENSITIVITY: dict[str, str] = {
    # A direct read-out of path loss between this node and an emitter whose
    # position is frequently public. One value gives a range ring; values over
    # time, or across bands, intersect those rings into a fix. This is exactly
    # the input that signal-strength fingerprint localisation is built on, so
    # rating it below the top tier would be ignoring an entire literature.
    "signal_percent": SITING_LEAK_HIGH,
    # Adaptive modulation makes achieved rate a quantised read-out of SNR, and
    # SNR is path loss again: a link sustaining its top rate is short, or has
    # clear line of sight, or both. Demand-limited links weaken the inference,
    # because an idle link looks like a distant one — but they do not reverse
    # it, and the direction that matters for siting (high rate implies close)
    # survives the confounding.
    "throughput_mbps": SITING_LEAK_HIGH,
    # Loss is what happens when link margin runs out, so it sits downstream of
    # the same path loss. It is held below the top tier because it is bursty,
    # quantised by retry limits, and driven as much by local congestion and
    # offered traffic as by geometry: it narrows a region rather than fixing a
    # point.
    "loss_percent": SITING_LEAK_MODERATE,
    # How much of the local spectrum is unusable is a description of the place.
    # Occupancy is geography — a crowded band count separates a port from open
    # water. What keeps it out of the top tier is that only the *count* leaves
    # the node: band identities would name the emitters overhead, and if this
    # fleet ever releases those they belong in the tier above.
    # How many provisioned links are actually usable is a coverage-map lookup.
    # A node that can use one profile of six is somewhere very different from
    # one that can use all six. Coarse — a small integer, floored and capped by
    # policy — so it selects a region rather than a position.
    "eligible_count": SITING_LEAK_MODERATE,
    # Second-order: confidence comes from the spread of scores across
    # candidates, which comes from the same measured link quality. Passing
    # through the scorer compresses it into [0, 1] and mixes in policy, which
    # costs an attacker a great deal of resolution — but a derived quantity is
    # still a function of what it was derived from.
    "confidence": SITING_LEAK_MODERATE,
    # Nominally configuration, and that reading is wrong here. On a platform
    # disciplining time from GNSS, trust lapses when GNSS does, and GNSS
    # degradation is geographically clustered: terrain masking, structures,
    # deliberate denial. Averaged over a window this becomes the fraction of
    # cycles a node could see the sky, which is a statement about its
    # surroundings. One bit per cycle narrows a region rather than fixing a
    # position, which is what keeps it out of the top tier.
    "clock_trusted": SITING_LEAK_MODERATE,
    # Dominated by the path through the network — queueing, scheduling, the
    # backhaul hop — rather than by the node's coordinates. Two nodes hundreds
    # of kilometres apart behind one gateway measure the same round-trip time,
    # and a single node's RTT moves more across a day than across a
    # redeployment. Not zero: a stable RTT floor is a coarse proxy for distance
    # to the gateway, and would matter more if per-target times were released
    # separately. Released as one aggregate, it is the cheapest coordinate on
    # which to buy back accuracy.
    "rtt_ms": SITING_LEAK_LOW,
    # The variance of that same quantity, and dominated by contention and
    # scheduler behaviour. It carries a weak busy/quiet discriminator, since a
    # crowded shared medium jitters more — but that is a statement about how
    # crowded a place is, not about which place it is.
    "jitter_ms": SITING_LEAK_LOW,
    # How many radio profiles are provisioned: a property of the build, not the
    # location, and near-constant for a given node. It does fingerprint the
    # platform type, which is a fleet-composition exposure worth knowing about
    # — but this mechanism protects values, not identity, and the update
    # carrying them is signed and roster-bound, so the node is named already.
    "candidate_count": SITING_LEAK_LOW,
    # Set by battery and thermal state under an operator power policy. The only
    # route from here to a location runs through ambient temperature, and duty
    # cycle, airflow and platform type swamp that long before it says anything
    # about where the platform is.
    "power_duty_limit": SITING_LEAK_LOW,
    # Whether the selection stayed where it was: a churn rate. Sustained churn
    # indicates a volatile RF environment, which is a condition rather than a
    # place, and selector hysteresis dominates it in any case.
    "is_current_profile": SITING_LEAK_LOW,
}

#: What an unclassified feature is assumed to be. Fail safe: a feature nobody
#: has classified is assumed to leak. The cost of that assumption is accuracy
#: on one coordinate; the cost of the opposite assumption is the fleet's siting,
#: which cannot be recalled once it has been released.
UNCLASSIFIED_SENSITIVITY = SITING_LEAK_HIGH

#: Relative share of the total budget by tier, before normalisation. The
#: direction is the whole point and is easy to invert by accident: a *smaller*
#: epsilon is a *tighter* budget and therefore *more* noise, so the
#: siting-sensitive tiers take the small shares and the rest fund them. The
#: ratio is deliberately modest — four to one across the range — because the
#: classification above is a judgement, and a judgement that turns out wrong
#: should cost one feature some accuracy rather than strip a coordinate of
#: protection outright.
TIER_BUDGET_SHARE: dict[str, float] = {
    SITING_LEAK_HIGH: 1.0,
    SITING_LEAK_MODERATE: 2.0,
    SITING_LEAK_LOW: 4.0,
}


def siting_sensitivity(feature: str) -> str:
    """Tier for a feature, defaulting to the most sensitive when unclassified."""
    return SITING_SENSITIVITY.get(feature, UNCLASSIFIED_SENSITIVITY)


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
class BudgetAllocation:
    """How one release's total budget was divided, and what it actually spent.

    ``requested_epsilon`` is what the operator declared. ``spent_epsilon`` is
    the sum of what was handed to the mechanisms. They are equal except in the
    floor regime below, and the difference is reported rather than smoothed
    over, because a mechanism that silently spends more than it was given has
    made the operator's budget decision meaningless.
    """

    strategy: str
    requested_epsilon: float
    per_feature: dict[str, float]
    reasons: tuple[str, ...]

    @property
    def spent_epsilon(self) -> float:
        return sum(self.per_feature.values())

    @property
    def conserved(self) -> bool:
        """Whether this release spends exactly what was declared."""
        return math.isclose(
            self.spent_epsilon, self.requested_epsilon, rel_tol=1e-9, abs_tol=1e-12
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "requested_epsilon": round(self.requested_epsilon, 9),
            # Rounding here is presentational. The equality that matters is
            # asserted on the unrounded values by ``conserved``.
            "spent_epsilon": round(self.spent_epsilon, 9),
            "conserved": self.conserved,
            "per_feature_epsilon": {
                name: round(value, 9) for name, value in sorted(self.per_feature.items())
            },
            "siting_sensitivity": {
                name: siting_sensitivity(name) for name in sorted(self.per_feature)
            },
            "reasons": list(self.reasons),
        }


def allocate_budget(
    total_epsilon: float,
    features: Iterable[str],
    *,
    allocation: str = DEFAULT_ALLOCATION,
) -> BudgetAllocation:
    """Divide one release's budget across the features released together.

    Both allocations spend the same total; only its distribution differs.
    Under :data:`SITING_AWARE_ALLOCATION` the shares are inversely related to
    how much a feature narrows the node's position, so the noise concentrates
    where the leakage does.

    Two properties are enforced against the obvious ways this goes wrong:

    **No feature drops below a usable budget.** A share that rounds down past
    :data:`MINIMUM_USEFUL_EPSILON` is raised to it and the shortfall is taken
    back proportionally from the features still above the floor, rather than
    added on top — which is how an allocation quietly overspends.

    **A budget too small to divide is refused as a division.** When the total
    cannot give every feature the floor, every feature is held at the floor,
    the release spends more than was asked, and the reasons say so. Both
    allocations produce exactly the same thing there, so the choice of
    allocation still cannot change what a release costs.
    """
    if allocation not in BUDGET_ALLOCATIONS:
        raise ValueError(
            f"unknown budget allocation {allocation!r}; expected one of {list(BUDGET_ALLOCATIONS)}"
        )
    names = tuple(sorted({str(name) for name in features}))
    if not names:
        raise ValueError("at least one feature is required to allocate a budget")
    total = bounded_number(total_epsilon, "epsilon", MINIMUM_USEFUL_EPSILON, 50.0)

    reasons: list[str] = [
        # The composition bound, named so nobody has to infer which one is in
        # force. It is the sum, and the sum is an upper bound rather than an
        # accountant's figure.
        "SEQUENTIAL_COMPOSITION_BOUND_IS_THE_SUM_OF_PER_FEATURE_EPSILON",
    ]

    floor_total = MINIMUM_USEFUL_EPSILON * len(names)
    if total <= floor_total:
        # Nothing to allocate: any split puts some feature below the point
        # where its own noise swamps any realistic fleet. Refusing to pretend
        # otherwise, and saying plainly that the release costs more than asked.
        reasons.append("TOTAL_BUDGET_TOO_SMALL_TO_SPLIT_ACROSS_THESE_FEATURES")
        reasons.append("EVERY_FEATURE_HELD_AT_THE_MINIMUM_USEFUL_BUDGET")
        if floor_total > total:
            reasons.append("RELEASE_SPENDS_MORE_EPSILON_THAN_WAS_REQUESTED")
        reasons.append("ALLOCATION_HAS_NO_EFFECT_AT_THIS_BUDGET")
        return BudgetAllocation(
            strategy=allocation,
            requested_epsilon=total,
            per_feature={name: MINIMUM_USEFUL_EPSILON for name in names},
            reasons=tuple(reasons),
        )

    if allocation == UNIFORM_ALLOCATION:
        weights = {name: 1.0 for name in names}
        reasons.append("BUDGET_SPLIT_EVENLY_ACROSS_RELEASED_FEATURES")
    else:
        weights = {name: TIER_BUDGET_SHARE[siting_sensitivity(name)] for name in names}
        reasons.append("NOISE_CONCENTRATED_ON_SITING_SENSITIVE_FEATURES")
        # The paper this follows declines to claim an end-to-end accountant for
        # its adaptive allocation. Neither does this, and it says so on every
        # release rather than only in a docstring nobody reads at runtime.
        reasons.append("NO_END_TO_END_DP_ACCOUNTANT_CLAIMED_FOR_THIS_ALLOCATION")
        unclassified = [name for name in names if name not in SITING_SENSITIVITY]
        if unclassified:
            reasons.append(
                "UNCLASSIFIED_FEATURES_TREATED_AS_SITING_SENSITIVE:" + ",".join(unclassified)
            )

    assigned: dict[str, float] = {}
    free = list(names)
    remaining = total
    while free:
        weight_total = sum(weights[name] for name in free)
        shares = {name: remaining * weights[name] / weight_total for name in free}
        below = [name for name in free if shares[name] < MINIMUM_USEFUL_EPSILON]
        if not below:
            assigned.update(shares)
            break
        for name in below:
            assigned[name] = MINIMUM_USEFUL_EPSILON
            remaining -= MINIMUM_USEFUL_EPSILON
        free = [name for name in free if name not in below]
        if "SOME_FEATURES_HELD_AT_THE_MINIMUM_USEFUL_BUDGET" not in reasons:
            reasons.append("SOME_FEATURES_HELD_AT_THE_MINIMUM_USEFUL_BUDGET")

    # The declared total must be exactly what is spent, not approximately: a
    # conservation claim that is only true to within floating-point residue is
    # a claim about floating point. The residue goes to the largest share,
    # which is the one that can absorb it without approaching the floor.
    drift = total - sum(assigned.values())
    if drift and assigned:
        largest = max(sorted(assigned), key=lambda name: assigned[name])
        if assigned[largest] + drift >= MINIMUM_USEFUL_EPSILON:
            assigned[largest] += drift

    return BudgetAllocation(
        strategy=allocation,
        requested_epsilon=total,
        per_feature=assigned,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class PrivatisedValue:
    feature: str
    perturbed: float
    epsilon: float
    contributors_needed_for_1pct: int
    reasons: tuple[str, ...]
    allocation: str = DEFAULT_ALLOCATION
    siting_sensitivity: str = UNCLASSIFIED_SENSITIVITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "perturbed": round(self.perturbed, 6),
            "epsilon": self.epsilon,
            "contributors_needed_for_1pct": self.contributors_needed_for_1pct,
            # Which allocation was in force, and what this feature was judged
            # to reveal under it. A release that does not say where its budget
            # went cannot be audited for whether it went to the right place.
            "allocation": self.allocation,
            "siting_sensitivity": self.siting_sensitivity,
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
        allocation: str = DEFAULT_ALLOCATION,
    ) -> None:
        if not ranges:
            raise ValueError("at least one feature range is required")
        self.ranges = ranges
        self.epsilon = epsilon
        # The budget is split across the features released together, because
        # releasing k values at full budget each spends k*epsilon overall. How
        # it is split is the operator's declared choice; that it sums to the
        # declared total is not negotiable either way.
        self.budget = allocate_budget(epsilon, ranges.keys(), allocation=allocation)
        self._mechanisms = {
            name: PiecewiseMechanism(epsilon=share)
            for name, share in self.budget.per_feature.items()
        }

    @property
    def allocation(self) -> str:
        return self.budget.strategy

    @property
    def per_feature_epsilon(self) -> float:
        """The weakest per-coordinate budget in force.

        Under uniform allocation every feature shares one budget and this is
        simply that number. Under a concentrated allocation there is no single
        per-feature epsilon, and a caller that has room for exactly one figure
        gets the *largest* — the loosest budget, the coordinate with the least
        protection. Reporting the smallest would let a release describe itself
        by its most private coordinate, which understates the exposure. Callers
        that can carry the whole picture should read :attr:`budget` instead.
        """
        return max(self.budget.per_feature.values())

    def epsilon_for(self, feature: str) -> float:
        """Budget assigned to one feature under the allocation in force."""
        try:
            return self.budget.per_feature[feature]
        except KeyError:
            raise KeyError(f"no declared range for feature {feature!r}") from None

    def privatise(self, feature: str, value: float) -> PrivatisedValue:
        feature_range = self.ranges.get(feature)
        if feature_range is None:
            raise KeyError(f"no declared range for feature {feature!r}")
        mechanism = self._mechanisms[feature]
        normalised = feature_range.normalise(value)
        perturbed = feature_range.denormalise(mechanism.perturb(normalised))
        sensitivity = siting_sensitivity(feature)
        reasons: list[str] = ["PERTURBED_LOCALLY_BEFORE_RELEASE"]
        if mechanism.epsilon <= 0.1:
            reasons.append("BUDGET_VERY_SMALL_EXPECT_HEAVY_NOISE")
        if self.allocation == SITING_AWARE_ALLOCATION:
            # Which direction this feature's budget moved, so the record shows
            # what was traded rather than only that something was.
            if sensitivity == SITING_LEAK_HIGH:
                reasons.append("NOISE_CONCENTRATED_HERE_FEATURE_NARROWS_NODE_POSITION")
            elif sensitivity == SITING_LEAK_LOW:
                reasons.append("BUDGET_RELEASED_HERE_FEATURE_SAYS_LITTLE_ABOUT_POSITION")
            if feature not in SITING_SENSITIVITY:
                reasons.append("FEATURE_UNCLASSIFIED_SO_TREATED_AS_SITING_SENSITIVE")
        needed = self._contributors_for_target(feature, 0.01)
        return PrivatisedValue(
            feature=feature,
            perturbed=perturbed,
            epsilon=mechanism.epsilon,
            contributors_needed_for_1pct=needed,
            reasons=tuple(reasons),
            allocation=self.allocation,
            siting_sensitivity=sensitivity,
        )

    def _contributors_for_target(self, feature: str, target_normalised_error: float) -> int:
        """How many contributors the fleet needs for a given aggregate error.

        Per feature, because under a concentrated allocation the answer differs
        between them: a feature carrying more noise needs a larger fleet before
        its aggregate means anything.
        """
        single = self._mechanisms[feature].expected_absolute_error(1)
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
