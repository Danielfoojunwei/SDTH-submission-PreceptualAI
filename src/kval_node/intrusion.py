"""Fleet-level anomaly detection without a shared model.

Adapted from **Lam** and co-authors, "Effective Intrusion Detection in
Heterogeneous Internet-of-Things Networks via Ensemble Knowledge
Distillation-based Federated Learning", IEEE ICC 2024
([arXiv:2401.11968](https://arxiv.org/abs/2401.11968)). Their problem is that
IoT nodes are heterogeneous — different hardware, different data distributions
— so forcing one architecture on all of them, as standard federated averaging
requires, fits none of them. Their answer is ensemble knowledge distillation:
each node keeps **its own** detector, and the fleet shares *outputs* rather than
*parameters*.

**What is taken, and what is not.** The distillation machinery is not built; it
needs trained student and teacher models and none ship. What transfers is the
architectural claim, which is the part that matters here and is independently
right for this fleet:

**Share verdicts, not parameters.** Every node's normal is genuinely different —
a node on a quiet rural hilltop and one beside an airport radar have nothing to
average. A shared model would be wrong for both. Each node instead judges its
own readings against **its own** baseline, and the fleet combines the verdicts.
This also means a node's raw environment never leaves it, which is the same
exposure ``privacy.py`` exists to limit.

**The distinction the fleet exists to draw.** One node reporting an anomaly is a
local condition: a truck parked in front of the antenna, a failing LNA, a
neighbour's microwave. The *same* anomaly appearing simultaneously at nodes in
**different cohorts** — nodes whose environments were already established to be
unalike — cannot be a local condition. That is the signature of something
acting on the whole area, and it is exactly the wideband jamming and sweep-jam
case the EW threat register cares about. A single node cannot draw that
distinction no matter how good its detector is; only the fleet can.

**Correlation is required, not just count.** Five nodes anomalous for five
unrelated reasons is not an event. Agreement is scored on *which features*
moved and *in which direction*, so unrelated faults do not accumulate into a
false fleet alarm.

**What it will not do.** No detection here identifies a cause, names a
technique, or attributes an actor. It reports that several environments changed
together in a way local conditions do not explain. Everything beyond that is an
operator's judgement, and the output is worded so it cannot be mistaken for
more.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from .validation import bounded_number, non_empty_str, require_finite

#: Deviation from a node's own baseline, in its own standard deviations, above
#: which a feature is anomalous *for that node*. Deliberately per-node: a
#: fleet-wide absolute threshold would be wrong everywhere at once.
DEFAULT_ANOMALY_SIGMAS = 3.0

#: Nodes that must agree before a fleet-level event is declared.
DEFAULT_MIN_CORROBORATION = 3

#: Distinct cohorts that must be involved. Anomalies confined to one cohort
#: are, by construction, explainable by that cohort's shared environment.
DEFAULT_MIN_COHORTS = 2

#: Window within which reports are treated as describing the same moment.
DEFAULT_CORRELATION_WINDOW = timedelta(minutes=5)

#: Share of anomalous features that must match in direction across nodes. Below
#: this the nodes are anomalous for unrelated reasons.
DEFAULT_MIN_AGREEMENT = 0.5

#: Samples below which a node's baseline is not stable enough to judge against.
#: Matches the corpus analyser's own floor.
MIN_BASELINE_SAMPLES = 20


@dataclass(frozen=True)
class NodeBaseline:
    """A node's own notion of normal, derived from its own history."""

    node_id: str
    samples: int
    means: dict[str, float]
    stddevs: dict[str, float]

    @classmethod
    def from_observations(
        cls, node_id: str, observations: Sequence[dict[str, float]]
    ) -> "NodeBaseline":
        node = non_empty_str(node_id, "node_id")
        if not observations:
            return cls(node, 0, {}, {})
        features = sorted({name for row in observations for name in row})
        means: dict[str, float] = {}
        spreads: dict[str, float] = {}
        for name in features:
            values = [
                require_finite(row[name], name) for row in observations if name in row
            ]
            if len(values) < 2:
                continue
            means[name] = statistics.fmean(values)
            spreads[name] = statistics.pstdev(values)
        return cls(node, len(observations), means, spreads)

    def usable(self) -> bool:
        return self.samples >= MIN_BASELINE_SAMPLES and bool(self.means)


@dataclass(frozen=True)
class FeatureDeviation:
    feature: str
    sigmas: float
    direction: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "sigmas": round(self.sigmas, 3),
            "direction": self.direction,
        }


@dataclass(frozen=True)
class LocalVerdict:
    """One node's judgement about its own readings. No raw values leave here."""

    node_id: str
    cohort_id: str
    observed_at: datetime
    anomalous: bool
    deviations: tuple[FeatureDeviation, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-local-verdict/v1",
            "node_id": self.node_id,
            "cohort_id": self.cohort_id,
            "observed_at": self.observed_at.isoformat(),
            "anomalous": self.anomalous,
            "deviations": [deviation.to_dict() for deviation in self.deviations],
            "reasons": list(self.reasons),
            # Raw feature values are deliberately absent: sharing them would
            # publish what each sensor sees and therefore where it is.
        }


class LocalDetector:
    """Judges a node's readings against that node's own baseline."""

    def __init__(self, baseline: NodeBaseline, *, sigmas: float = DEFAULT_ANOMALY_SIGMAS) -> None:
        self.baseline = baseline
        self.sigmas = bounded_number(sigmas, "sigmas", 0.5, 20.0)

    def judge(
        self, reading: dict[str, float], *, cohort_id: str, observed_at: datetime
    ) -> LocalVerdict:
        if not self.baseline.usable():
            # Without a stable baseline there is no "normal" to depart from.
            # Reporting "not anomalous" would be a fail-open, so the verdict
            # says it abstained.
            return LocalVerdict(
                node_id=self.baseline.node_id,
                cohort_id=non_empty_str(cohort_id, "cohort_id"),
                observed_at=observed_at,
                anomalous=False,
                deviations=(),
                reasons=(f"BASELINE_NEEDS_AT_LEAST_{MIN_BASELINE_SAMPLES}_SAMPLES_SO_ABSTAINING",),
            )

        deviations: list[FeatureDeviation] = []
        for feature in sorted(reading):
            if feature not in self.baseline.means:
                continue
            spread = self.baseline.stddevs.get(feature, 0.0)
            value = require_finite(reading[feature], feature)
            difference = value - self.baseline.means[feature]
            if spread <= 0.0:
                # A feature that never varied has no scale. Any change is
                # notable but cannot be expressed in sigmas without inventing
                # one, so it is reported at the threshold rather than at
                # infinity.
                if difference != 0.0:
                    deviations.append(
                        FeatureDeviation(feature, self.sigmas, 1 if difference > 0 else -1)
                    )
                continue
            sigmas = difference / spread
            if abs(sigmas) >= self.sigmas:
                deviations.append(
                    FeatureDeviation(feature, abs(sigmas), 1 if sigmas > 0 else -1)
                )

        anomalous = bool(deviations)
        reasons = (
            ("READINGS_DEPART_FROM_THIS_NODE_S_OWN_NORMAL",)
            if anomalous
            else ("READINGS_WITHIN_THIS_NODE_S_OWN_NORMAL",)
        )
        return LocalVerdict(
            node_id=self.baseline.node_id,
            cohort_id=non_empty_str(cohort_id, "cohort_id"),
            observed_at=observed_at,
            anomalous=anomalous,
            deviations=tuple(deviations),
            reasons=reasons,
        )


def verdict_agreement(first: LocalVerdict, second: LocalVerdict) -> float:
    """Share of anomalous features that moved the same way at both nodes.

    Direction matters. Two nodes whose noise floor rose together are seeing one
    thing; one rising and one falling are seeing two.
    """
    left = {deviation.feature: deviation.direction for deviation in first.deviations}
    right = {deviation.feature: deviation.direction for deviation in second.deviations}
    union = set(left) | set(right)
    if not union:
        return 0.0
    matching = sum(1 for feature in union if left.get(feature) == right.get(feature) is not None)
    return matching / len(union)


@dataclass(frozen=True)
class FleetFinding:
    event: bool
    corroborating_nodes: tuple[str, ...]
    cohorts: tuple[str, ...]
    shared_features: tuple[str, ...]
    agreement: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-fleet-anomaly/v1",
            "event": self.event,
            "corroborating_nodes": list(self.corroborating_nodes),
            "cohorts": list(self.cohorts),
            "shared_features": list(self.shared_features),
            "agreement": round(self.agreement, 4),
            "reasons": list(self.reasons),
        }


class FleetDetector:
    """Combines local verdicts into a fleet-level finding, or declines to.

    The bar is deliberately high. A false fleet-level alarm is expensive — it
    is the finding an operator acts on — and the cheap ways to reach one
    (counting anomalies, ignoring cohorts, ignoring direction) are exactly the
    ways to reach a wrong one.
    """

    def __init__(
        self,
        *,
        min_corroboration: int = DEFAULT_MIN_CORROBORATION,
        min_cohorts: int = DEFAULT_MIN_COHORTS,
        correlation_window: timedelta = DEFAULT_CORRELATION_WINDOW,
        min_agreement: float = DEFAULT_MIN_AGREEMENT,
        reputations: dict[str, float] | None = None,
    ) -> None:
        self.min_corroboration = max(2, min_corroboration)
        self.min_cohorts = max(1, min_cohorts)
        self.correlation_window = correlation_window
        self.min_agreement = bounded_number(min_agreement, "min_agreement", 0.0, 1.0)
        self.reputations = reputations or {}

    def _weight(self, node_id: str) -> float:
        return self.reputations.get(node_id, 1.0)

    def assess(self, verdicts: Sequence[LocalVerdict]) -> FleetFinding:
        anomalous = [verdict for verdict in verdicts if verdict.anomalous]
        if len(anomalous) < self.min_corroboration:
            return FleetFinding(
                False,
                tuple(sorted(v.node_id for v in anomalous)),
                (),
                (),
                0.0,
                ("TOO_FEW_NODES_ANOMALOUS_TO_RULE_OUT_LOCAL_CONDITIONS",),
            )

        # Largest set of anomalies that fall inside one correlation window.
        ordered = sorted(anomalous, key=lambda v: v.observed_at)
        best: list[LocalVerdict] = []
        for start in range(len(ordered)):
            window = [
                verdict
                for verdict in ordered[start:]
                if verdict.observed_at - ordered[start].observed_at <= self.correlation_window
            ]
            if len(window) > len(best):
                best = window
        if len(best) < self.min_corroboration:
            return FleetFinding(
                False,
                tuple(sorted(v.node_id for v in anomalous)),
                (),
                (),
                0.0,
                ("ANOMALIES_ARE_SPREAD_OVER_TIME_NOT_SIMULTANEOUS",),
            )

        cohorts = sorted({verdict.cohort_id for verdict in best})
        reasons: list[str] = []
        if len(cohorts) < self.min_cohorts:
            # Anomalies confined to one cohort are explainable by that cohort's
            # shared environment, which is what a cohort *is*.
            return FleetFinding(
                False,
                tuple(sorted(v.node_id for v in best)),
                tuple(cohorts),
                (),
                0.0,
                ("ANOMALY_CONFINED_TO_ONE_COHORT_SO_A_SHARED_LOCAL_CAUSE_EXPLAINS_IT",),
            )

        pairs = [
            verdict_agreement(best[i], best[j])
            for i in range(len(best))
            for j in range(i + 1, len(best))
        ]
        agreement = statistics.fmean(pairs) if pairs else 0.0
        if agreement < self.min_agreement:
            return FleetFinding(
                False,
                tuple(sorted(v.node_id for v in best)),
                tuple(cohorts),
                (),
                agreement,
                ("NODES_ARE_ANOMALOUS_FOR_UNRELATED_REASONS_NOT_ONE_CAUSE",),
            )

        shared = sorted(
            set.intersection(*({d.feature for d in verdict.deviations} for verdict in best))
        )
        weighted = sum(self._weight(verdict.node_id) for verdict in best)
        if weighted < self.min_corroboration:
            # Reputation-discounted corroboration: nodes that have repeatedly
            # diverged from their cohorts do not carry a fleet-level finding.
            return FleetFinding(
                False,
                tuple(sorted(v.node_id for v in best)),
                tuple(cohorts),
                tuple(shared),
                agreement,
                ("CORROBORATION_COMES_FROM_NODES_WITH_TOO_LITTLE_STANDING",),
            )

        reasons.append("MULTIPLE_COHORTS_CHANGED_TOGETHER_BEYOND_LOCAL_EXPLANATION")
        reasons.append("THIS_REPORTS_A_CORRELATED_CHANGE_NOT_A_CAUSE_OR_AN_ACTOR")
        return FleetFinding(
            event=True,
            corroborating_nodes=tuple(sorted(v.node_id for v in best)),
            cohorts=tuple(cohorts),
            shared_features=tuple(shared),
            agreement=agreement,
            reasons=tuple(reasons),
        )


def expected_false_alarm_rate(
    *, sigmas: float, features: int, nodes: int, min_corroboration: int
) -> float:
    """Rough probability that corroboration arises by chance alone.

    Assumes independent, normally distributed features — which real RF
    measurements are **not**; they are autocorrelated and cross-correlated, so
    the true rate is higher than this. It is a floor for setting thresholds
    deliberately, not a performance claim, and no detection result quotes it.
    """
    if sigmas <= 0 or features < 1 or nodes < 1:
        return 1.0
    per_feature = math.erfc(sigmas / math.sqrt(2.0))
    per_node = min(1.0, 1.0 - (1.0 - per_feature) ** features)
    total = 0.0
    for count in range(max(2, min_corroboration), nodes + 1):
        total += (
            math.comb(nodes, count)
            * per_node**count
            * (1.0 - per_node) ** (nodes - count)
        )
    return min(1.0, total)
