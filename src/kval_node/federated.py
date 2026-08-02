"""Federated statistics across the node fleet, with robust aggregation.

No model is trained in this repository, so there are no weights to average. What
the fleet *does* have, on every node, is a local feature envelope computed from
that node's own decision history — and sharing those is federated learning in
the sense that matters operationally: **raw telemetry never leaves the node**,
only sufficient statistics (count, mean, and sum of squares per feature), and
every node ends up with a fleet-wide baseline it could not have computed alone.

That baseline is what :class:`~kval_node.learning.DriftDetector` needs to say
whether conditions are unusual *for the fleet* rather than merely unusual for
one node that has only ever seen one location.

The security properties are the point, and they are the same ones a weight-
averaging scheme would need:

**Updates are signed and roster-bound.** An update is verified against its
author's key from the operator-signed peer roster, exactly like a fabric report.
An unrostered author is refused.

**Aggregation is consensus-based, not a mean.** A plain average lets a single
node move the result arbitrarily far — the classic model-poisoning attack, and
threat 16 in the EW threat register. Contributors are first reduced to the
largest group that *agrees with each other* per feature, and only that group is
averaged (with the extremes trimmed).

Agreement is used rather than deviation-from-the-median deliberately. Outlier
detection against a median has a breakdown point: once adversarial contributors
reach half the fleet the median is itself corrupted, the attackers no longer
look like outliers, and the poisoning succeeds. Asking which contributors agree
with one another has no such collapse — a colluding half that disagrees with
the honest half simply forms a second cluster, and neither can pass as fleet
consensus without meeting the quorum.

**Outliers are quarantined and named.** A contributor outside the agreeing
cluster on any feature is excluded and reported, so a node that is compromised —
or simply faulty — is visible rather than silently averaged in.

The honest limit: this defends the *baseline*, not the fleet. If enough nodes
are compromised that no honest quorum remains, aggregation refuses to publish a
baseline rather than publishing a poisoned one — but it cannot tell you which
cluster was the honest one. That determination is an operator's.

**One node cannot form a fleet.** Aggregation below a quorum is refused: a
"fleet" baseline derived from a single contributor is just that node's baseline
wearing a larger label.
"""

from __future__ import annotations

import base64
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .analysis import MIN_BASELINE_SAMPLES
from .dataset import FEATURE_NAMES, Example
from .fabric import FabricError, PeerRoster
from .learning import FeatureBaseline
from .models import parse_timestamp, utc_now
from .signing import canonical_json, public_key_id
from .validation import non_empty_str, positive_int, require_finite

UPDATE_SCHEMA = "kval-federated-update/v1"

#: Contributors required before a fleet baseline means anything.
DEFAULT_QUORUM = 3

#: Fraction trimmed from each end per coordinate before averaging. At 0.25 a
#: quarter of contributors at each extreme are discarded, so a minority of
#: malicious updates cannot move the result.
DEFAULT_TRIM_FRACTION = 0.25

#: Fractional difference within which two contributors' means are treated as
#: agreeing. Nodes in one fleet observing the same conditions should land well
#: inside this; a poisoned value will not.
DEFAULT_RELATIVE_TOLERANCE = 0.5

#: Scale floor so that near-zero means are not split apart by relative
#: differences that carry no information.
DEFAULT_ABSOLUTE_FLOOR = 1e-6

#: Reports older than this cannot describe the fleet's current conditions.
DEFAULT_UPDATE_MAX_AGE_SECONDS = 86_400.0


@dataclass(frozen=True)
class FeatureStatistics:
    """Sufficient statistics for one feature. Never raw observations."""

    count: int
    mean: float
    #: Sum of squared feature values, i.e. sum(x_i^2) — not the mean of
    #: squares. Paired with ``count`` and ``mean`` this is enough to recover
    #: the variance without sharing any individual observation.
    sum_squares: float

    def stddev(self) -> float:
        if self.count < 2:
            return 0.0
        variance = max(0.0, self.sum_squares / self.count - self.mean**2)
        return math.sqrt(variance)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "sum_squares": self.sum_squares}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureStatistics":
        return cls(
            count=positive_int(raw["count"], "count"),
            mean=require_finite(raw["mean"], "mean"),
            sum_squares=require_finite(raw["sum_squares"], "sum_squares"),
        )


@dataclass(frozen=True)
class FederatedUpdate:
    node_id: str
    produced_at: datetime
    examples: int
    statistics: dict[str, FeatureStatistics]

    @classmethod
    def from_examples(cls, examples: Sequence[Example], *, node_id: str, now: datetime | None = None) -> "FederatedUpdate":
        if len(examples) < MIN_BASELINE_SAMPLES:
            raise FabricError(
                f"need at least {MIN_BASELINE_SAMPLES} local examples to contribute; got {len(examples)}"
            )
        stats: dict[str, FeatureStatistics] = {}
        for name in FEATURE_NAMES:
            values = [example.features.get(name, 0.0) for example in examples]
            stats[name] = FeatureStatistics(
                count=len(values),
                mean=statistics.fmean(values),
                sum_squares=sum(value**2 for value in values),
            )
        return cls(
            node_id=non_empty_str(node_id, "node_id"),
            produced_at=now or utc_now(),
            examples=len(examples),
            statistics=stats,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FederatedUpdate":
        stats_raw = raw.get("statistics")
        if not isinstance(stats_raw, dict) or not stats_raw:
            raise ValueError("update carries no statistics")
        return cls(
            node_id=non_empty_str(raw.get("node_id"), "node_id"),
            produced_at=parse_timestamp(str(raw["produced_at"])),
            examples=positive_int(raw["examples"], "examples"),
            statistics={str(k): FeatureStatistics.from_dict(v) for k, v in stats_raw.items()},
        )

    def to_body(self) -> dict[str, Any]:
        return {
            "schema": UPDATE_SCHEMA,
            "node_id": self.node_id,
            "produced_at": self.produced_at.isoformat(),
            "examples": self.examples,
            "statistics": {name: self.statistics[name].to_dict() for name in sorted(self.statistics)},
        }


def sign_update(update: FederatedUpdate, private_key: Ed25519PrivateKey) -> dict[str, Any]:
    body = update.to_body()
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "body": body,
        "signature": base64.b64encode(private_key.sign(canonical_json(body))).decode("ascii"),
    }


def verify_update(
    envelope: dict[str, Any],
    roster: PeerRoster,
    *,
    now: datetime | None = None,
    max_age_seconds: float = DEFAULT_UPDATE_MAX_AGE_SECONDS,
) -> FederatedUpdate:
    current = now or utc_now()
    if envelope.get("algorithm") != "Ed25519":
        raise FabricError("unsupported update signature algorithm")
    body = envelope.get("body")
    if not isinstance(body, dict) or body.get("schema") != UPDATE_SCHEMA:
        raise FabricError("unsupported or malformed update body")
    try:
        update = FederatedUpdate.from_dict(body)
    except (ValueError, KeyError, TypeError) as exc:
        raise FabricError(f"update body is malformed: {exc}") from exc
    if body != update.to_body():
        raise FabricError("update body contains unauthenticated or non-canonical fields")

    peer = roster.by_id().get(update.node_id)
    if peer is None:
        raise FabricError(f"update author {update.node_id!r} is not on the signed roster")
    if envelope.get("key_id") != public_key_id(peer.public_key):
        raise FabricError("update key ID does not match the roster entry for this node")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise FabricError("update signature is missing or malformed") from exc
    try:
        peer.public_key.verify(signature, canonical_json(update.to_body()))
    except InvalidSignature as exc:
        raise FabricError("update signature verification failed") from exc

    age = (current - update.produced_at).total_seconds()
    if age > max_age_seconds:
        raise FabricError("update is older than the aggregation window")
    if age < -1.0:
        raise FabricError("update is dated in the future")
    return update


def trimmed_mean(values: Sequence[float], *, trim_fraction: float) -> float:
    """Coordinate-wise trimmed mean: discard the extremes, average the rest."""
    ordered = sorted(require_finite(value, "value") for value in values)
    if not ordered:
        raise ValueError("no values to aggregate")
    trim = int(len(ordered) * trim_fraction)
    kept = ordered[trim : len(ordered) - trim] or ordered
    return statistics.fmean(kept)


def median_absolute_deviation(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    return median, statistics.median(deviations)


def largest_consensus_cluster(
    values: Sequence[float],
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    absolute_floor: float = DEFAULT_ABSOLUTE_FLOOR,
) -> list[int]:
    """Indices of the largest group of contributors that agree with each other.

    Outlier detection against the median has a breakdown point: once half the
    contributors are malicious the median is itself corrupted, the "outliers"
    look normal, and the attack succeeds. Asking instead which contributors
    *agree with one another* has no such failure — a colluding half that
    disagrees with the honest half simply produces two clusters, and neither
    can be passed off as fleet consensus without meeting the quorum.

    Two values agree when they lie within ``relative_tolerance`` of each other,
    with ``absolute_floor`` preventing near-zero values from being split by
    meaningless relative differences.
    """
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    best: list[int] = []
    for start in range(len(ordered)):
        cluster = [ordered[start]]
        low = values[ordered[start]]
        for candidate in ordered[start + 1 :]:
            high = values[candidate]
            span = abs(high - low)
            scale = max(abs(low), abs(high), absolute_floor)
            if span > relative_tolerance * scale:
                break
            cluster.append(candidate)
        if len(cluster) > len(best):
            best = cluster
    return sorted(best)


@dataclass(frozen=True)
class AggregationResult:
    baseline: FeatureBaseline | None
    contributors: tuple[str, ...]
    quarantined: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, str], ...]
    total_examples: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-fleet-baseline/v1",
            "available": self.baseline is not None,
            "contributors": list(self.contributors),
            "quarantined": list(self.quarantined),
            "rejected": list(self.rejected),
            "total_examples": self.total_examples,
            "features": (
                {
                    name: {"mean": round(self.baseline.means[name], 6), "stddev": round(self.baseline.stddevs[name], 6)}
                    for name in sorted(self.baseline.means)
                }
                if self.baseline
                else {}
            ),
            "reasons": list(self.reasons),
        }


class RobustAggregator:
    """Combines verified updates into a fleet baseline, resisting poisoning."""

    def __init__(
        self,
        *,
        quorum: int = DEFAULT_QUORUM,
        trim_fraction: float = DEFAULT_TRIM_FRACTION,
        relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    ) -> None:
        if quorum < 2:
            raise ValueError("a fleet baseline needs at least two contributors")
        if not 0.0 <= trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        self.quorum = quorum
        self.trim_fraction = trim_fraction
        self.relative_tolerance = relative_tolerance

    def _quarantine(self, updates: Sequence[FederatedUpdate]) -> tuple[list[FederatedUpdate], list[dict[str, Any]]]:
        """Keep the contributors that agree with each other; exclude the rest.

        A contributor is quarantined when it falls outside the largest
        agreeing cluster on any feature. This does not rely on the fleet median
        being honest, so it does not collapse when adversarial contributors
        reach half the fleet.
        """
        if len(updates) < 3:
            # With too few contributors there is no consensus to measure; the
            # quorum check is what protects this case.
            return list(updates), []
        flagged: dict[str, list[str]] = {}
        for name in FEATURE_NAMES:
            present = [update for update in updates if name in update.statistics]
            if len(present) < 3:
                continue
            values = [update.statistics[name].mean for update in present]
            cluster = set(
                largest_consensus_cluster(values, relative_tolerance=self.relative_tolerance)
            )
            if len(cluster) == len(present):
                continue
            for index, update in enumerate(present):
                if index not in cluster:
                    flagged.setdefault(update.node_id, []).append(name)
        kept = [update for update in updates if update.node_id not in flagged]
        quarantined = [
            {"node_id": node_id, "outlying_features": sorted(features)}
            for node_id, features in sorted(flagged.items())
        ]
        return kept, quarantined

    def aggregate(self, updates: Sequence[FederatedUpdate]) -> AggregationResult:
        reasons: list[str] = []
        if len(updates) < self.quorum:
            return AggregationResult(
                baseline=None,
                contributors=tuple(sorted(u.node_id for u in updates)),
                quarantined=(),
                rejected=(),
                total_examples=sum(u.examples for u in updates),
                reasons=(f"NEED_AT_LEAST_{self.quorum}_CONTRIBUTORS",),
            )

        kept, quarantined = self._quarantine(updates)
        if quarantined:
            reasons.append("OUTLYING_CONTRIBUTORS_QUARANTINED")
        if len(kept) < self.quorum:
            return AggregationResult(
                baseline=None,
                contributors=tuple(sorted(u.node_id for u in kept)),
                quarantined=tuple(quarantined),
                rejected=(),
                total_examples=sum(u.examples for u in kept),
                reasons=tuple(reasons + ["TOO_FEW_CONTRIBUTORS_SURVIVED_QUARANTINE"]),
            )

        means: dict[str, float] = {}
        stddevs: dict[str, float] = {}
        excluded: list[str] = []
        for name in FEATURE_NAMES:
            contributing = [update.statistics[name] for update in kept if name in update.statistics]
            if len(contributing) < self.quorum:
                excluded.append(name)
                continue
            mean = trimmed_mean([stat.mean for stat in contributing], trim_fraction=self.trim_fraction)
            spread = trimmed_mean([stat.stddev() for stat in contributing], trim_fraction=self.trim_fraction)
            if spread <= 0.0:
                # A feature constant across the whole fleet carries no envelope;
                # FeatureBaseline would reject a zero standard deviation.
                excluded.append(name)
                continue
            means[name] = mean
            stddevs[name] = spread

        if not means:
            return AggregationResult(
                baseline=None,
                contributors=tuple(sorted(u.node_id for u in kept)),
                quarantined=tuple(quarantined),
                rejected=(),
                total_examples=sum(u.examples for u in kept),
                reasons=tuple(reasons + ["NO_FEATURE_VARIED_ACROSS_THE_FLEET"]),
            )

        if excluded:
            reasons.append(f"CONSTANT_OR_SPARSE_FEATURES_EXCLUDED:{','.join(sorted(excluded))}")
        reasons.append("FLEET_BASELINE_AGGREGATED")
        return AggregationResult(
            baseline=FeatureBaseline(means=means, stddevs=stddevs),
            contributors=tuple(sorted(update.node_id for update in kept)),
            quarantined=tuple(quarantined),
            rejected=(),
            total_examples=sum(update.examples for update in kept),
            reasons=tuple(reasons),
        )
