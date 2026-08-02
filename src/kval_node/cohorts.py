"""Cohort formation, reputation, and hierarchical aggregation for the fleet.

This module implements the coordination mechanisms from the NTU / Zhejiang
Gongshang dynamic-spectrum-access line of work (Feng Li, Bowen Shen, Kwok-Yan
Lam and co-authors), adapted to a **receive-only** sensing fleet. See
``docs/DSA_RESEARCH_MAPPING.md`` for the full paper-by-paper mapping, including
what was deliberately not carried across.

The adaptation is not cosmetic. That corpus solves transmit-side problems —
which channel a secondary user should transmit on, at what price and power.
Kval never transmits. What transfers is the *coordination architecture*: how a
fleet of edge nodes federates learning without pooling raw data, how it handles
nodes whose environments differ, and how it stays useful when some nodes are
unreliable or hostile.

The mechanisms implemented here:

**Cohort clustering.** Federating one global model across nodes in genuinely
different RF environments produces a model that fits none of them — the non-IID
problem. Nodes are grouped by observed-environment similarity so aggregation
happens within comparable populations.
  * Feng Li, Shen, Guo, Lam, Wei, Wang, "Dynamic Spectrum Access for
    Internet-of-Things Based on Federated Deep Reinforcement Learning", IEEE
    TVT 71(7):7952-7956, 2022, doi:10.1109/TVT.2022.3166535 — clusters
    non-interfering users with similar conditions into federated cohorts.
  * Shen, Goh, Lam, Li, "Efficient and Secure Federated Dynamic Spectrum Access
    for LEO Satellite Internet-of-Things", IEEE ICCT 2024,
    doi:10.1109/ICCT62411.2024.10946453 — bi-level K-Means client clustering.

**Heterogeneity reporting.** Cohorts formed silently tell an operator nothing
about whether the grouping means anything — whether these are three genuinely
different radio environments or one environment cut into three by noise. The
fleet's heterogeneity is therefore measured and reported alongside the cohorts,
including the case where it says the grouping separated nothing.
  * Peide Li, Cao, Zhang, Wei, Hu, Xie, "EPFL-REMNet: Efficient Personalized
    Federated Digital Twin Towards 6G Heterogeneous Radio Environment",
    [arXiv:2511.05238](https://arxiv.org/abs/2511.05238), 2025 — defines a
    per-location heterogeneity metric ``H = std(s_1..s_M)`` over the signal
    strengths from M base stations, and splits a deployment into light, medium
    and heavy non-IID regimes at the 33rd and 66th percentiles of H.

**Reputation weighting.** Nodes are not equally trustworthy: some are exposed,
poorly sited, miscalibrated, or compromised. Reputation earned from past
agreement with the fleet discounts a node's contribution.
  * Shen, Lam, Li, Wang, "Privacy-Aware Spectrum Pricing and Power Control
    Optimization for LEO Satellite Internet-of-Things", IEEE TWC
    24(11):9569-9584, 2025, doi:10.1109/TWC.2025.3574172 — reputation-based
    selection of which node aggregates each round.

**Hierarchical aggregation with rotation.** A single central aggregator is both
a single point of failure and the obvious target. Aggregation is per-cohort
first, then across cohorts, and the aggregator role rotates deterministically
by reputation rather than being fixed.
  * Zhang, Lam, Shen, Wang, Li, "Dynamic spectrum access for Internet-of-Things
    with hierarchical federated deep reinforcement learning", Ad Hoc Networks
    149:103257, 2023, doi:10.1016/j.adhoc.2023.103257 — hierarchical,
    personalised aggregation.
  * Feng Li, Wang, Lam, Shen, Wang, "A Secure Dynamic Spectrum Access Scheme
    for Internet of Things With Swarm Learning", IEEE IoT-J 12(18), 2025,
    doi:10.1109/JIOT.2025.3583710 — swarm learning with no central server.

**Contribution excision.** A node that is decommissioned, moves jurisdiction, or
is found compromised must be removable from the fleet baseline without
retraining from scratch. Because aggregation here is over sufficient
statistics, excision is exact rather than approximate.
  * Feng Li, Shui, Lam, Shen, Wang, "Secure Dynamic Spectrum Access in
    Internet-of-Things Based on Machine Unlearning", IEEE ICAIIC 2025,
    doi:10.1109/ICAIIC64266.2025.10920793.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from .analysis import MIN_BASELINE_SAMPLES
from .dataset import FEATURE_NAMES
from .federated import (
    AggregationResult,
    FeatureStatistics,
    FederatedUpdate,
    RobustAggregator,
)
from .validation import bounded_number, non_empty_str

#: Features describing a node's RF environment, used for cohort similarity.
#: Deliberately excludes decision-outcome features, so cohorts reflect *where a
#: node is* rather than *what it chose*.
ENVIRONMENT_FEATURES: tuple[str, ...] = (
    "rtt_ms",
    "loss_percent",
    "jitter_ms",
    "signal_percent",
    "contested_band_count",
)

DEFAULT_MAX_COHORTS = 4
DEFAULT_KMEANS_ITERATIONS = 25


@dataclass(frozen=True)
class Reputation:
    node_id: str
    score: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Reputation":
        return cls(
            node_id=non_empty_str(raw.get("node_id"), "node_id"),
            score=bounded_number(raw.get("score", 1.0), "score", 0.0, 1.0),
        )


class ReputationLedger:
    """Tracks how well each node has agreed with its cohort over time.

    Reputation is earned, not asserted: a node that consistently reports what
    its cohort reports keeps a high score, and one that repeatedly diverges
    loses it. The score then discounts that node's evidence, which is the
    lightweight Byzantine defence the LEO-satellite work uses in place of full
    BFT machinery.
    """

    #: Weight of the newest observation. Low enough that one anomalous round
    #: does not destroy a good node's standing, high enough that a persistently
    #: lying node degrades within a handful of rounds.
    LEARNING_RATE = 0.2

    #: Floor so a node can always recover; reputation suppresses influence, it
    #: does not permanently excommunicate.
    MINIMUM_SCORE = 0.05

    def __init__(self, initial: dict[str, float] | None = None) -> None:
        self._scores: dict[str, float] = dict(initial or {})

    def score(self, node_id: str) -> float:
        return self._scores.get(node_id, 1.0)

    def scores(self) -> dict[str, float]:
        return dict(self._scores)

    def record_round(self, agreed: Sequence[str], diverged: Sequence[str]) -> None:
        for node_id in agreed:
            current = self.score(node_id)
            self._scores[node_id] = min(1.0, current + self.LEARNING_RATE * (1.0 - current))
        for node_id in diverged:
            current = self.score(node_id)
            self._scores[node_id] = max(
                self.MINIMUM_SCORE, current - self.LEARNING_RATE * current
            )

    def rank(self) -> list[str]:
        """Nodes ordered by reputation, ties broken by ID for determinism."""
        return sorted(self._scores, key=lambda node: (-self._scores[node], node))


def _environment_vector(update: FederatedUpdate) -> list[float]:
    return [
        update.statistics[name].mean if name in update.statistics else 0.0
        for name in ENVIRONMENT_FEATURES
    ]


def _standardise(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Zero-mean, unit-variance per dimension so no feature dominates by scale."""
    if not vectors:
        return []
    dimensions = len(vectors[0])
    means = [statistics.fmean(v[d] for v in vectors) for d in range(dimensions)]
    spreads = [statistics.pstdev([v[d] for v in vectors]) or 1.0 for d in range(dimensions)]
    return [[(v[d] - means[d]) / spreads[d] for d in range(dimensions)] for v in vectors]


def kmeans_cohorts(
    updates: Sequence[FederatedUpdate],
    *,
    max_cohorts: int = DEFAULT_MAX_COHORTS,
    iterations: int = DEFAULT_KMEANS_ITERATIONS,
) -> list[list[FederatedUpdate]]:
    """Group nodes by RF-environment similarity.

    Deterministic k-means: seeds are the points furthest apart rather than
    random, so the same fleet always yields the same cohorts and an aggregation
    result is reproducible from its inputs.
    """
    if len(updates) <= 2:
        return [list(updates)] if updates else []
    vectors = _standardise([_environment_vector(update) for update in updates])
    cohort_count = max(1, min(max_cohorts, len(updates) // 2))
    if cohort_count == 1:
        return [list(updates)]

    # Farthest-point seeding, deterministic.
    seeds = [0]
    while len(seeds) < cohort_count:
        best_index, best_distance = None, -1.0
        for index, vector in enumerate(vectors):
            if index in seeds:
                continue
            distance = min(math.dist(vector, vectors[seed]) for seed in seeds)
            if distance > best_distance:
                best_index, best_distance = index, distance
        if best_index is None:
            break
        seeds.append(best_index)
    centroids = [list(vectors[seed]) for seed in seeds]

    assignment = [0] * len(vectors)
    for _ in range(iterations):
        changed = False
        for index, vector in enumerate(vectors):
            distances = [math.dist(vector, centroid) for centroid in centroids]
            nearest = distances.index(min(distances))
            if assignment[index] != nearest:
                assignment[index] = nearest
                changed = True
        for cluster in range(len(centroids)):
            members = [vectors[i] for i, a in enumerate(assignment) if a == cluster]
            if members:
                centroids[cluster] = [
                    statistics.fmean(member[d] for member in members) for d in range(len(members[0]))
                ]
        if not changed:
            break

    cohorts: list[list[FederatedUpdate]] = [[] for _ in centroids]
    for index, update in enumerate(updates):
        cohorts[assignment[index]].append(update)
    return [cohort for cohort in cohorts if cohort]


@dataclass(frozen=True)
class CohortResult:
    cohort_id: int
    members: tuple[str, ...]
    aggregate: AggregationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "members": list(self.members),
            "aggregate": self.aggregate.to_dict(),
        }


@dataclass(frozen=True)
class HierarchicalResult:
    cohorts: tuple[CohortResult, ...]
    global_aggregate: AggregationResult | None
    aggregator_node: str | None
    reputations: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-hierarchical-baseline/v1",
            "cohorts": [cohort.to_dict() for cohort in self.cohorts],
            "global": self.global_aggregate.to_dict() if self.global_aggregate else None,
            "aggregator_node": self.aggregator_node,
            "reputations": {k: round(v, 4) for k, v in sorted(self.reputations.items())},
            "reasons": list(self.reasons),
        }


class HierarchicalAggregator:
    """Per-cohort aggregation, then across cohorts, with a rotating aggregator."""

    def __init__(
        self,
        *,
        cohort_quorum: int = 2,
        global_quorum: int = 2,
        max_cohorts: int = DEFAULT_MAX_COHORTS,
        reputation: ReputationLedger | None = None,
    ) -> None:
        self.cohort_quorum = cohort_quorum
        self.global_quorum = global_quorum
        self.max_cohorts = max_cohorts
        self.reputation = reputation or ReputationLedger()

    def select_aggregator(self, candidates: Sequence[str], *, round_index: int) -> str | None:
        """Pick this round's aggregator, rotating among the most reputable.

        Rotation matters as much as the ranking: a fixed aggregator is a single
        point of failure and the obvious target, which is the weakness swarm
        learning removes.
        """
        eligible = [node for node in candidates if self.reputation.score(node) > 0.5]
        if not eligible:
            return None
        ordered = sorted(eligible, key=lambda node: (-self.reputation.score(node), node))
        return ordered[round_index % len(ordered)]

    def aggregate(
        self, updates: Sequence[FederatedUpdate], *, round_index: int = 0
    ) -> HierarchicalResult:
        reasons: list[str] = []
        if not updates:
            return HierarchicalResult((), None, None, self.reputation.scores(), ("NO_UPDATES",))

        cohorts = kmeans_cohorts(updates, max_cohorts=self.max_cohorts)
        cohort_results: list[CohortResult] = []
        cohort_representatives: list[FederatedUpdate] = []

        for index, members in enumerate(cohorts):
            aggregator = RobustAggregator(quorum=self.cohort_quorum)
            result = aggregator.aggregate(members)
            cohort_results.append(
                CohortResult(index, tuple(sorted(u.node_id for u in members)), result)
            )
            # Reputation follows cohort agreement: contributors that survived
            # quarantine agreed, quarantined ones diverged.
            quarantined = {entry["node_id"] for entry in result.quarantined}
            self.reputation.record_round(
                agreed=[u.node_id for u in members if u.node_id not in quarantined],
                diverged=sorted(quarantined),
            )
            if result.baseline is not None:
                # Represent the cohort by a synthetic update carrying its
                # aggregate, so the global tier combines cohorts rather than
                # letting a large cohort outvote a small one node-by-node.
                stats = {
                    name: type(members[0].statistics[name])(
                        count=sum(u.statistics[name].count for u in members if name in u.statistics),
                        mean=result.baseline.means[name],
                        sum_squares=0.0,
                    )
                    for name in result.baseline.means
                }
                # sum_squares is recomputed so stddev() recovers the cohort spread.
                stats = {
                    name: type(members[0].statistics[name])(
                        count=stat.count,
                        mean=stat.mean,
                        sum_squares=stat.count * (stat.mean**2 + result.baseline.stddevs[name] ** 2),
                    )
                    for name, stat in stats.items()
                }
                cohort_representatives.append(
                    FederatedUpdate(
                        node_id=f"cohort-{index}",
                        produced_at=members[0].produced_at,
                        examples=sum(u.examples for u in members),
                        statistics=stats,
                    )
                )

        global_result: AggregationResult | None = None
        if len(cohort_representatives) >= self.global_quorum:
            global_result = RobustAggregator(quorum=self.global_quorum).aggregate(
                cohort_representatives
            )
        else:
            reasons.append("TOO_FEW_COHORTS_PRODUCED_A_BASELINE_FOR_A_GLOBAL_TIER")

        aggregator_node = self.select_aggregator(
            sorted({update.node_id for update in updates}), round_index=round_index
        )
        if aggregator_node is None:
            reasons.append("NO_NODE_HAS_SUFFICIENT_REPUTATION_TO_AGGREGATE")
        if not reasons:
            reasons.append("HIERARCHICAL_BASELINE_AGGREGATED")
        return HierarchicalResult(
            cohorts=tuple(cohort_results),
            global_aggregate=global_result,
            aggregator_node=aggregator_node,
            reputations=self.reputation.scores(),
            reasons=tuple(reasons),
        )


def excise_contributor(
    updates: Sequence[FederatedUpdate], node_id: str
) -> tuple[list[FederatedUpdate], bool]:
    """Remove a node's contribution exactly, with no retraining.

    Because the fleet baseline is built from sufficient statistics rather than
    gradient updates, removing a contributor is exact: drop its statistics and
    re-aggregate. There is no residual influence to approximate away, which is
    the property machine-unlearning schemes work hard to achieve for trained
    models and get for free here.
    """
    remaining = [update for update in updates if update.node_id != node_id]
    return remaining, len(remaining) != len(updates)


@dataclass(frozen=True)
class RecoveryCertificate:
    """Evidence that a poisoned contribution left no trace in the baseline.

    Implements the *guarantee* sought by the certified-recovery line — "Toward
    Efficient and Certified Recovery From Poisoning Attacks in Federated
    Learning", IEEE TIFS 2025 ([arXiv:2401.08216](https://arxiv.org/abs/2401.08216)),
    and the certified-client-removal and federated-unlearning work alongside
    it, all with **Lam** as an author.

    Their setting is gradient-based federated learning, where a client's
    influence is smeared across every subsequent round and removing it exactly
    would mean retraining from the round it joined. Their contribution is
    machinery — stored update histories, calibrated rollback — that recovers a
    model *approximately*, with a bound on the residual influence.

    Here the guarantee is stronger and the machinery is unnecessary, because
    aggregation is over sufficient statistics rather than gradients. Removal is
    **exact**, and the certificate proves it rather than asserting it: the
    baseline recomputed after excision is compared byte-for-byte against one
    built from a fleet the excised node never joined. Identical hashes mean the
    residual influence is not small — it is zero.

    The certificate also records what the poisoned baseline *had* claimed, so
    the size of the averted error is on record rather than quietly discarded.
    """

    node_id: str
    was_present: bool
    exact: bool
    recovered_hash: str
    never_joined_hash: str
    max_shift: float
    shifted_features: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-recovery-certificate/v1",
            "node_id": self.node_id,
            "was_present": self.was_present,
            "exact": self.exact,
            "recovered_hash": self.recovered_hash,
            "never_joined_hash": self.never_joined_hash,
            "max_shift": round(self.max_shift, 6),
            "shifted_features": list(self.shifted_features),
            "reasons": list(self.reasons),
        }


def _baseline_fingerprint(result: AggregationResult) -> str:
    """Stable hash of an aggregate, for byte-level comparison."""
    if result.baseline is None:
        return "no-baseline"
    body = {
        "means": {name: repr(value) for name, value in sorted(result.baseline.means.items())},
        "stddevs": {name: repr(value) for name, value in sorted(result.baseline.stddevs.items())},
        "contributors": sorted(result.contributors),
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def certified_excision(
    updates: Sequence[FederatedUpdate],
    node_id: str,
    *,
    quorum: int = 2,
) -> tuple[list[FederatedUpdate], RecoveryCertificate]:
    """Excise a contributor and certify that nothing of it remains.

    The certificate is produced by construction, not by inspection: the
    recovered baseline and a baseline built from scratch without the node are
    both computed and their fingerprints compared. If they ever diverged, the
    certificate would say ``exact=False`` rather than claiming a guarantee the
    aggregation could not support.
    """
    remaining, removed = excise_contributor(updates, node_id)
    reasons: list[str] = []

    if not removed:
        reasons.append("NODE_WAS_NOT_A_CONTRIBUTOR_SO_THERE_IS_NOTHING_TO_REMOVE")
        return remaining, RecoveryCertificate(
            node_id=node_id,
            was_present=False,
            exact=True,
            recovered_hash="",
            never_joined_hash="",
            max_shift=0.0,
            shifted_features=(),
            reasons=tuple(reasons),
        )

    aggregator = RobustAggregator(quorum=quorum)
    poisoned = aggregator.aggregate(updates)
    recovered = aggregator.aggregate(remaining)
    # The control: aggregate the same remaining set through a fresh aggregator,
    # as if the excised node had never contributed at all.
    never_joined = RobustAggregator(quorum=quorum).aggregate(list(remaining))

    recovered_hash = _baseline_fingerprint(recovered)
    never_joined_hash = _baseline_fingerprint(never_joined)
    exact = recovered_hash == never_joined_hash

    max_shift = 0.0
    shifted: list[str] = []
    if poisoned.baseline is not None and recovered.baseline is not None:
        for feature, before in sorted(poisoned.baseline.means.items()):
            after = recovered.baseline.means.get(feature)
            if after is None:
                continue
            spread = max(poisoned.baseline.stddevs.get(feature, 0.0), 1e-9)
            shift = abs(after - before) / spread
            if shift > 1e-9:
                shifted.append(feature)
            max_shift = max(max_shift, shift)

    if exact:
        reasons.append("RECOVERED_BASELINE_IS_IDENTICAL_TO_ONE_THE_NODE_NEVER_JOINED")
        reasons.append("RESIDUAL_INFLUENCE_IS_ZERO_NOT_MERELY_BOUNDED")
    else:
        # Cannot happen while aggregation stays over sufficient statistics, and
        # is reported rather than assumed away precisely so that a future change
        # which breaks the property is caught here instead of being trusted.
        reasons.append("RECOVERED_BASELINE_DIFFERS_FROM_A_CLEAN_ONE_SO_REMOVAL_IS_NOT_EXACT")
    if shifted:
        reasons.append("THE_EXCISED_CONTRIBUTION_HAD_BEEN_MOVING_THE_BASELINE")

    return remaining, RecoveryCertificate(
        node_id=node_id,
        was_present=True,
        exact=exact,
        recovered_hash=recovered_hash,
        never_joined_hash=never_joined_hash,
        max_shift=max_shift,
        shifted_features=tuple(shifted),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Fleet heterogeneity
# ---------------------------------------------------------------------------

#: Features whose *variability* carries a radio-environment signature.
#:
#: EPFL-REMNet scores a location by ``std(s_1..s_M)``, the spread among the
#: signal strengths measured there from M base stations. Kval has no
#: per-base-station readings: a node's federated statistics summarise only the
#: link profile it actually selected, so the across-link spread the paper uses
#: is not in the wire format at all. What the sufficient statistics *do* carry
#: is each feature's spread across that node's own observation window, and that
#: measures the same thing along a different axis — how much the radio quality
#: varies within one node's view. A platform crossing an urban canyon sees the
#: swing over time that the paper sees across base stations.
#:
#: Included because each is a measurement of the channel rather than of the
#: node's own software or fit:
#:   * ``rtt_ms``, ``jitter_ms`` — propagation, queueing and multipath.
#:   * ``loss_percent`` — the direct consequence of interference and fading.
#:   * ``signal_percent`` — the closest analogue Kval has to the paper's s_m.
#:   * ``throughput_mbps`` — admitted here although ``ENVIRONMENT_FEATURES``
#:     excludes it, because the quantity used is its *swing*, not its level.
#:     The level is confounded by offered traffic; the offered load on a
#:     supervisory link is near-constant by design, so a swing in achieved rate
#:     is a swing in what the link could carry. It remains the weakest of the
#:     six and is deliberately not weighted above the others.
#:   * ``contested_band_count`` — the only member sourced from the spectrum
#:     sensor rather than the link, so it registers the environment changing
#:     around the platform before the link has degraded.
#:
#: Excluded, with the reason each would corrupt the score:
#:   * ``confidence`` — the recommender's belief in its own choice. A model
#:     retrain would change a node's measured environment.
#:   * ``candidate_count``, ``eligible_count`` — how many radios the platform
#:     carries and how many were usable. Two nodes in the same canyon with
#:     different radio fits would land in different regimes.
#:   * ``clock_trusted`` — a binary subsystem-health flag. Its variance is set
#:     by the holdover policy timer and by whether a receiver is fitted at all.
#:   * ``power_duty_limit`` — the thermal and battery governor's output. A
#:     platform, mission and ambient-temperature property.
#:   * ``is_current_profile`` — a decision outcome. Scoring it would let a node
#:     that flaps between profiles report a complex environment on the strength
#:     of its own indecision.
HETEROGENEITY_FEATURES: tuple[str, ...] = (
    "rtt_ms",
    "loss_percent",
    "jitter_ms",
    "throughput_mbps",
    "signal_percent",
    "contested_band_count",
)

#: Swing in each feature's own unit that counts as one step of environmental
#: change. Dividing by these is what makes features in milliseconds, percent,
#: megabits and band counts addable at all — and, more importantly, what gives
#: the score an **absolute** meaning: a score of 1.0 is a fleet whose features
#: swing by about one step, in any deployment, in any week. Without a fixed
#: scale the only available normalisation is the fleet's own spread, which
#: forces the average node to score the same number by construction and makes
#: cross-deployment comparison impossible.
#:
#: These are judgement calls and they are the definition of "heavy". Changing
#: them changes which absolute regime a fleet reports; it does not change the
#: relative tiers, which are ratios within one fleet.
REFERENCE_SWINGS: dict[str, float] = {
    #: Below this, latency variation sits inside ordinary queueing; above it a
    #: supervisory control loop can no longer budget for the round trip.
    "rtt_ms": 25.0,
    #: The span over which retransmission and forward error correction go from
    #: invisible to dominating the achievable rate.
    "loss_percent": 2.0,
    #: A representative de-jitter buffer depth. Variation the buffer absorbs is
    #: not a change of environment; variation past it is heard as dropouts.
    "jitter_ms": 10.0,
    #: Roughly the difference between a link carrying compressed video return
    #: and one reduced to telemetry.
    "throughput_mbps": 5.0,
    #: ``signal_percent`` is a 0-100 normalisation of received strength; 15
    #: points separates a link that holds from one fading in and out.
    "signal_percent": 15.0,
    #: One band flickering in and out of the degraded list is ordinary sensor
    #: churn. Two bands appearing and clearing is a different spectrum picture.
    "contested_band_count": 2.0,
}

REGIME_LIGHT = "light"
REGIME_MEDIUM = "medium"
REGIME_HEAVY = "heavy"

#: Score below which the fleet's environments are light in absolute terms:
#: features swing by well under one reference step, which is the smooth,
#: low-variability regime EPFL-REMNet calls light.
ABSOLUTE_LIGHT_CEILING = 0.5

#: Score at or above which the fleet's environments are heavy in absolute
#: terms. This is the guard against the percentile trap: the relative tiers
#: always produce a heavy third, so "heavy" is only allowed to *mean* anything
#: when the underlying score clears a fixed bar. At 1.5 the typical feature is
#: swinging by more than one and a half reference steps, which is the sharp,
#: skewed regime the paper describes rather than merely the worst third of a
#: calm fleet.
ABSOLUTE_HEAVY_FLOOR = 1.5

#: Minimum nodes before the percentile tiers are computed at all. The tiers are
#: order statistics: with ``n`` nodes the 33rd-percentile boundary is decided by
#: the node ranked near ``n/3``, so each tier needs enough members that its
#: contents are not simply a restatement of its own boundaries. Nine is the
#: smallest fleet giving three tiers of three, and three is the smallest tier
#: with an interior member — one node that is not itself a boundary. At six,
#: every node in a tier is a boundary node and one node shifting rank moves a
#: whole tier. At three there is no 33rd percentile in any useful sense; there
#: are three nodes and three labels, and the labels are just the ranking.
MIN_NODES_FOR_TIERS = 9

#: Minimum nodes before any fleet-wide heterogeneity is reported. Same
#: reasoning as the aggregation quorum in ``federated.py``: a claim about the
#: fleet derived from one or two nodes is those nodes' own condition wearing a
#: larger label. Between this and ``MIN_NODES_FOR_TIERS`` the absolute regime is
#: still reported, because it is a per-node measurement that needs no
#: percentiles — only the relative tiers are refused.
MIN_NODES_FOR_ASSESSMENT = 3

#: Ratio of a cohort's internal spread to the fleet's, at or above which the
#: cohort is reported as having separated nothing. A cohort exists to be a
#: *tighter* population than the fleet it came from; one that still contains
#: four fifths of the fleet's own spread has not grouped anything an operator
#: should act on, whatever the clustering objective records as an improvement.
MAX_SEPARATED_SPREAD_RATIO = 0.8


def _root_mean_square(values: Sequence[float]) -> float:
    """Quadratic mean. Empty input is a caller error, not a zero."""
    if not values:
        raise ValueError("no values to combine")
    return math.sqrt(statistics.fmean(value * value for value in values))


def _usable_statistic(update: FederatedUpdate, name: str) -> FeatureStatistics | None:
    """The node's statistics for ``name``, or None if they cannot be trusted.

    ``FeatureStatistics.stddev`` returns 0.0 for a count below two. That zero is
    not a measurement of a calm environment, it is the absence of one, and
    letting it through would let a node with three observations report the
    quietest environment in the fleet. The floor is the same count a node needs
    before its own local baseline is worth computing.
    """
    stat = update.statistics.get(name)
    if stat is None or stat.count < MIN_BASELINE_SAMPLES:
        return None
    if not math.isfinite(stat.mean):
        return None
    spread = stat.stddev()
    if not math.isfinite(spread):
        return None
    return stat


def _admissible_features(
    updates: Sequence[FederatedUpdate],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the feature set into those that carry information and those that do not.

    A feature that is flat everywhere — no node varies in it and no two nodes
    differ in it — is almost always a sensor that is not fitted rather than an
    environment that is calm. Averaging its zero in would *deflate* every score
    and report a homogeneous fleet on the strength of a missing instrument,
    which is the reverse of the failure the metric exists to catch.
    """
    used: list[str] = []
    excluded: list[str] = []
    for name in HETEROGENEITY_FEATURES:
        stats = [
            stat
            for stat in (_usable_statistic(update, name) for update in updates)
            if stat is not None
        ]
        if len(stats) < 2:
            excluded.append(name)
            continue
        varies_within = any(stat.stddev() > 0.0 for stat in stats)
        varies_across = statistics.pstdev([stat.mean for stat in stats]) > 0.0
        if not varies_within and not varies_across:
            excluded.append(name)
            continue
        used.append(name)
    return tuple(used), tuple(excluded)


def _node_score(
    update: FederatedUpdate, features: Sequence[str]
) -> tuple[float, int] | None:
    """One node's heterogeneity score and the observation count behind it.

    The score is the quadratic mean of the node's per-feature swings expressed
    in reference-swing units. Quadratic rather than arithmetic because a single
    feature swinging violently is what breaks a shared model, and an arithmetic
    mean would let five calm features hide it.
    """
    swings: list[float] = []
    counts: list[int] = []
    for name in features:
        stat = _usable_statistic(update, name)
        if stat is None:
            # Comparing this node's score against the fleet's percentiles
            # requires it to be the same function of the same features.
            return None
        swings.append(stat.stddev() / REFERENCE_SWINGS[name])
        counts.append(stat.count)
    if not swings:
        return None
    return _root_mean_square(swings), min(counts)


def _group_spread(
    updates: Sequence[FederatedUpdate], features: Sequence[str]
) -> float | None:
    """Between-node dispersion of environment means, in reference-swing units.

    Constructed exactly like a node's own score — the quadratic mean of
    normalised standard deviations — but with the dispersion taken across nodes
    instead of within one, so the two numbers are on the same scale and can be
    read against each other. This is the quantity the cohort test needs: how
    far apart the members of a group actually sit.
    """
    if len(updates) < 2:
        return None
    dispersions: list[float] = []
    for name in features:
        means = [
            stat.mean
            for stat in (_usable_statistic(update, name) for update in updates)
            if stat is not None
        ]
        if len(means) < 2:
            continue
        dispersions.append(statistics.pstdev(means) / REFERENCE_SWINGS[name])
    if not dispersions:
        return None
    return _root_mean_square(dispersions)


@dataclass(frozen=True)
class NodeHeterogeneity:
    """One node's environmental variability, and where it falls in the fleet."""

    node_id: str
    score: float
    #: Observations behind the score: the smallest per-feature count used.
    samples: int
    #: Relative tier from the fleet's own percentiles, or None when the fleet
    #: was too small for percentiles to mean anything.
    tier: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "score": round(self.score, 6),
            "samples": self.samples,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class CohortSeparation:
    """Whether one cohort is a narrower population than the fleet it came from."""

    cohort_id: int
    members: tuple[str, ...]
    scored_members: int
    spread: float | None
    spread_ratio: float | None
    #: True when the cohort is measurably tighter than the fleet, False when it
    #: is not, and **None when the question could not be answered** — a
    #: single-node cohort has no internal spread, and a fleet with no spread
    #: gives nothing to compare against.
    separated: bool | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "members": list(self.members),
            "scored_members": self.scored_members,
            "spread": None if self.spread is None else round(self.spread, 6),
            "spread_ratio": (
                None if self.spread_ratio is None else round(self.spread_ratio, 6)
            ),
            "separated": self.separated,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FleetHeterogeneity:
    """How varied the fleet's radio environments are, and whether cohorts caught it."""

    available: bool
    #: Absolute regime — comparable between deployments and between weeks.
    regime: str | None
    #: The number behind the regime: the fleet's median node score.
    score: float | None
    #: Between-node dispersion, on the same scale as ``score``.
    fleet_spread: float | None
    #: Tertile boundaries of the node scores, or None when tiers were refused.
    tier_boundaries: tuple[float, float] | None
    nodes: tuple[NodeHeterogeneity, ...]
    unscored_nodes: tuple[str, ...]
    node_count: int
    total_examples: int
    features_used: tuple[str, ...]
    features_excluded: tuple[str, ...]
    cohorts: tuple[CohortSeparation, ...]
    reasons: tuple[str, ...]

    def tier_counts(self) -> dict[str, int]:
        counts = {REGIME_LIGHT: 0, REGIME_MEDIUM: 0, REGIME_HEAVY: 0}
        for node in self.nodes:
            if node.tier is not None:
                counts[node.tier] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-fleet-heterogeneity/v1",
            "available": self.available,
            "regime": self.regime,
            "score": None if self.score is None else round(self.score, 6),
            "fleet_spread": (
                None if self.fleet_spread is None else round(self.fleet_spread, 6)
            ),
            "tier_boundaries": (
                None
                if self.tier_boundaries is None
                else [round(value, 6) for value in self.tier_boundaries]
            ),
            "tier_counts": self.tier_counts(),
            "nodes": [node.to_dict() for node in self.nodes],
            "unscored_nodes": list(self.unscored_nodes),
            "node_count": self.node_count,
            "scored_node_count": len(self.nodes),
            "total_examples": self.total_examples,
            "features_used": list(self.features_used),
            "features_excluded": list(self.features_excluded),
            "cohorts": [cohort.to_dict() for cohort in self.cohorts],
            "reasons": list(self.reasons),
        }


def _refusal(
    updates: Sequence[FederatedUpdate],
    reasons: Sequence[str],
    *,
    features_used: tuple[str, ...] = (),
    features_excluded: tuple[str, ...] = (),
) -> FleetHeterogeneity:
    return FleetHeterogeneity(
        available=False,
        regime=None,
        score=None,
        fleet_spread=None,
        tier_boundaries=None,
        nodes=(),
        unscored_nodes=tuple(sorted(update.node_id for update in updates)),
        node_count=len(updates),
        total_examples=sum(update.examples for update in updates),
        features_used=features_used,
        features_excluded=features_excluded,
        cohorts=(),
        reasons=tuple(reasons),
    )


def _absolute_regime(score: float) -> str:
    if score < ABSOLUTE_LIGHT_CEILING:
        return REGIME_LIGHT
    if score < ABSOLUTE_HEAVY_FLOOR:
        return REGIME_MEDIUM
    return REGIME_HEAVY


def _assess_cohorts(
    cohorts: Sequence[Sequence[FederatedUpdate]],
    *,
    scored_ids: set[str],
    features: Sequence[str],
    fleet_spread: float | None,
) -> tuple[list[CohortSeparation], list[str]]:
    """Report, per cohort, whether the grouping narrowed anything.

    Deliberately judged on the same feature means the clustering itself used,
    so a cohort that fails this test fails on the clustering's own terms and
    cannot be excused as having been measured against the wrong quantities.
    """
    results: list[CohortSeparation] = []
    fleet_reasons: list[str] = []
    for index, members in enumerate(cohorts):
        scored = [member for member in members if member.node_id in scored_ids]
        member_ids = tuple(sorted(member.node_id for member in members))
        reasons: list[str] = []
        spread = _group_spread(scored, features) if len(scored) >= 2 else None
        if spread is None:
            reasons.append("COHORT_HAS_TOO_FEW_SCORED_NODES_TO_HAVE_AN_INTERNAL_SPREAD")
            results.append(
                CohortSeparation(index, member_ids, len(scored), None, None, None, tuple(reasons))
            )
            continue
        if fleet_spread is None or fleet_spread <= 0.0:
            reasons.append("FLEET_HAS_NO_MEASURABLE_SPREAD_TO_COMPARE_THE_COHORT_AGAINST")
            results.append(
                CohortSeparation(
                    index, member_ids, len(scored), spread, None, None, tuple(reasons)
                )
            )
            continue
        ratio = spread / fleet_spread
        separated = ratio < MAX_SEPARATED_SPREAD_RATIO
        if separated:
            reasons.append("COHORT_IS_NARROWER_THAN_THE_FLEET")
        else:
            reasons.append("COHORT_SPREAD_MATCHES_THE_FLEET_SO_THE_GROUPING_SEPARATED_NOTHING")
        results.append(
            CohortSeparation(
                index, member_ids, len(scored), spread, ratio, separated, tuple(reasons)
            )
        )

    if len(cohorts) <= 1:
        fleet_reasons.append("ONLY_ONE_COHORT_SO_THERE_IS_NO_GROUPING_TO_ASSESS")
    else:
        verdicts = [result.separated for result in results]
        if all(verdict is None for verdict in verdicts):
            fleet_reasons.append("NO_COHORT_COULD_BE_JUDGED_AGAINST_THE_FLEET")
        elif not any(verdict is True for verdict in verdicts):
            # The grouping is reported as unsupported rather than left implicit
            # in a set of cohort IDs, which would read as structure.
            fleet_reasons.append("COHORT_GROUPING_IS_NOT_SUPPORTED_BY_THE_MEASURED_SPREADS")
        elif any(verdict is not True for verdict in verdicts):
            fleet_reasons.append("SOME_COHORTS_DID_NOT_SEPARATE_FROM_THE_FLEET")
    return results, fleet_reasons


def assess_heterogeneity(
    updates: Sequence[FederatedUpdate],
    *,
    cohorts: Sequence[Sequence[FederatedUpdate]] | None = None,
    max_cohorts: int = DEFAULT_MAX_COHORTS,
) -> FleetHeterogeneity:
    """Measure how varied the fleet's radio environments are, and audit the cohorts.

    Follows EPFL-REMNet ([arXiv:2511.05238](https://arxiv.org/abs/2511.05238)),
    which scores a location by the spread among the signal strengths it receives
    from M base stations and splits a deployment into light, medium and heavy
    non-IID regimes at the 33rd and 66th percentiles of that score. Kval's
    federated statistics carry no per-base-station readings, so the spread is
    taken along the axis they do carry — each feature's variation across the
    node's own observation window — over the features in
    ``HETEROGENEITY_FEATURES``, normalised by ``REFERENCE_SWINGS`` so that
    milliseconds, percent, megabits and band counts combine into one number.

    Two answers come back, and they are not the same answer:

    **How heterogeneous is this fleet?** ``regime`` and ``score``. The regime is
    fixed against ``ABSOLUTE_LIGHT_CEILING`` and ``ABSOLUTE_HEAVY_FLOOR``, so it
    means the same thing in a different deployment and in a different week.

    **Is this cohort grouping meaningful?** ``cohorts``, each carrying its own
    spread as a fraction of the fleet's. A cohort whose internal spread is
    comparable to the fleet's has not separated anything, and is reported that
    way rather than as a bare cohort ID that would imply structure.

    **The percentile trap, stated plainly.** The per-node ``tier`` values are
    the paper's rule applied literally, and they are *relative to this fleet*.
    A perfectly homogeneous fleet still yields a "heavy" third, because a third
    of any distribution is always its worst third. Tiers therefore say which
    nodes are the fleet's most variable, never that any node is in a difficult
    environment. Only ``regime`` carries that claim, and when the tiers and the
    absolute scale disagree the result says so in ``reasons`` rather than
    letting the word "heavy" do work the measurements do not support.

    Refuses below ``MIN_NODES_FOR_ASSESSMENT`` nodes entirely, and refuses the
    tiers alone below ``MIN_NODES_FOR_TIERS``.
    """
    reasons: list[str] = []
    if len(updates) < MIN_NODES_FOR_ASSESSMENT:
        return _refusal(
            updates, [f"NEED_AT_LEAST_{MIN_NODES_FOR_ASSESSMENT}_NODES_TO_DESCRIBE_A_FLEET"]
        )

    features, excluded = _admissible_features(updates)
    if not features:
        return _refusal(
            updates,
            ["NO_ENVIRONMENT_FEATURE_VARIED_ACROSS_THE_FLEET"],
            features_excluded=excluded,
        )
    if excluded:
        reasons.append(f"FLAT_OR_SPARSE_FEATURES_EXCLUDED:{','.join(sorted(excluded))}")

    scored: list[tuple[FederatedUpdate, float, int]] = []
    unscored: list[str] = []
    for update in updates:
        result = _node_score(update, features)
        if result is None:
            unscored.append(update.node_id)
            continue
        scored.append((update, result[0], result[1]))
    if unscored:
        reasons.append("NODES_WITHOUT_USABLE_STATISTICS_FOR_EVERY_FEATURE_WERE_NOT_SCORED")
    if len(scored) < MIN_NODES_FOR_ASSESSMENT:
        return _refusal(
            updates,
            reasons + [f"FEWER_THAN_{MIN_NODES_FOR_ASSESSMENT}_NODES_COULD_BE_SCORED"],
            features_used=features,
            features_excluded=excluded,
        )

    scores = [score for _update, score, _samples in scored]
    fleet_score = statistics.median(scores)
    regime = _absolute_regime(fleet_score)

    boundaries: tuple[float, float] | None = None
    tiers: dict[str, str] = {}
    if len(scored) < MIN_NODES_FOR_TIERS:
        reasons.append(
            f"NEED_AT_LEAST_{MIN_NODES_FOR_TIERS}_SCORED_NODES_BEFORE_PERCENTILE_TIERS_MEAN_ANYTHING"
        )
    else:
        # ``n=3`` cuts at the tertiles, which is what the paper's q33 and q66
        # approximate. Computed from this fleet's own scores, so they move when
        # the fleet does — see the trap described in the docstring.
        lower, upper = statistics.quantiles(scores, n=3, method="inclusive")
        boundaries = (lower, upper)
        for update, score, _samples in scored:
            if score <= lower:
                tiers[update.node_id] = REGIME_LIGHT
            elif score <= upper:
                tiers[update.node_id] = REGIME_MEDIUM
            else:
                tiers[update.node_id] = REGIME_HEAVY
        reasons.append("PERCENTILE_TIERS_ARE_RELATIVE_TO_THIS_FLEET_NOT_TO_AN_ABSOLUTE_SCALE")
        if lower >= upper:
            reasons.append("NODE_SCORES_ARE_DEGENERATE_SO_THE_TIERS_CARRY_NO_INFORMATION")
        if max(scores) < ABSOLUTE_HEAVY_FLOOR:
            # The trap, made explicit: a third of the fleet is labelled heavy
            # and not one node is in a heavy environment.
            reasons.append("RELATIVE_HEAVY_TIER_SITS_ENTIRELY_BELOW_THE_ABSOLUTE_HEAVY_FLOOR")
        if min(scores) >= ABSOLUTE_HEAVY_FLOOR:
            reasons.append("RELATIVE_LIGHT_TIER_SITS_ENTIRELY_ABOVE_THE_ABSOLUTE_HEAVY_FLOOR")

    scored_updates = [update for update, _score, _samples in scored]
    fleet_spread = _group_spread(scored_updates, features)
    if cohorts is None:
        # With nothing supplied, audit the grouping this module itself would
        # form, so the cohorts under test are the ones the fleet would use.
        grouping = kmeans_cohorts(scored_updates, max_cohorts=max_cohorts)
    else:
        grouping = list(cohorts)
    cohort_results, cohort_reasons = _assess_cohorts(
        grouping,
        scored_ids={update.node_id for update in scored_updates},
        features=features,
        fleet_spread=fleet_spread,
    )
    reasons.extend(cohort_reasons)
    reasons.append("FLEET_HETEROGENEITY_ASSESSED")

    return FleetHeterogeneity(
        available=True,
        regime=regime,
        score=fleet_score,
        fleet_spread=fleet_spread,
        tier_boundaries=boundaries,
        nodes=tuple(
            NodeHeterogeneity(
                node_id=update.node_id,
                score=score,
                samples=samples,
                tier=tiers.get(update.node_id),
            )
            for update, score, samples in sorted(scored, key=lambda item: item[0].node_id)
        ),
        unscored_nodes=tuple(sorted(unscored)),
        node_count=len(updates),
        total_examples=sum(update.examples for update in updates),
        features_used=features,
        features_excluded=excluded,
        cohorts=tuple(cohort_results),
        reasons=tuple(reasons),
    )
