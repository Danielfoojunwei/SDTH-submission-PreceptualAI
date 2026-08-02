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

Three mechanisms are implemented here:

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

import math
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from .dataset import FEATURE_NAMES
from .federated import AggregationResult, FederatedUpdate, RobustAggregator
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
