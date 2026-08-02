"""Where the research mechanisms are made to work on each other.

Each module in this package reproduces one paper's mechanism, and each was
independently correct and independently useless. A cohort clusterer that
nothing clusters, a forecaster whose forecasts reach no decision, an anomaly
detector with no fleet to detect across — those are libraries, not
capabilities. This module is the wiring that turns them into a system, and the
compositions are where the value actually is:

**Federation is hierarchical, reputation-weighted and privacy-preserving at
once.** A node's contribution is perturbed locally before release (LDP,
IoT-J 2021), aggregated within an RF-environment cohort rather than globally
(TVT 2022, ICCT 2024), combined across cohorts so a large one cannot outvote a
small one (Ad Hoc 2023), weighted by reputation earned from past agreement
(TWC 2025), by an aggregator that rotates rather than sitting still (IoT-J
2025) — and a contributor found hostile is excised with a certificate that its
influence is gone (TIFS 2025, ICAIIC 2025). Seven papers, one pipeline. None of
them delivers that alone.

**Prediction reaches a decision.** Per-flow forecasting (INFOCOM 2024) is worth
having only if a predicted degradation demotes a link *before* it fails. The
forecast is bounded the same way sensing is: it may demote, never promote.

**Localisation feeds tracking.** NLOS-rejected position estimates (Sensors
2018) accumulate in the ledger, and the mobility model (WPC 2024) reads that
history to predict where an emitter goes next. Neither paper produces a track
on its own — one gives a point, the other needs a history of points.

**Detection is a fleet property.** Per-node detectors (ICC 2024) produce
verdicts; cohort membership decides whether agreement between them is
meaningful; reputation decides whether the agreeing nodes carry any weight.
A single-node anomaly stays a local condition, as it should.

**Integrity guards the learning path.** Baselines are derived from the ledger,
so a handful of injected records shifts what "normal" means (MemMorph). Before
a baseline is used, the ledger's authorship is audited and the influence of any
single record is bounded — the learning path refuses rather than inheriting a
poisoned normal.

**Certification survives partition.** When the operator is unreachable and a
grant is expiring, a quorum of roster peers can attest to extending it (MANET
2009) — bounded, never widening scope, so the fleet degrades gracefully instead
of disarming on a timer.

Every composition below fails closed. A missing piece produces an unavailable
result naming what was missing, never a plausible one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from .analysis import build_baseline
from .cohorts import (
    HierarchicalAggregator,
    RecoveryCertificate,
    ReputationLedger,
    certified_excision,
    kmeans_cohorts,
)
from .dataset import Example
from .evidence import BeliefMass
from .fabric import FabricReport
from .federated import FederatedUpdate
from .forecast import MIN_HISTORY, PerFlowForecaster
from .intrusion import FleetDetector, LocalDetector, LocalVerdict, NodeBaseline
from .memory_integrity import bounded_influence
from .mobility import Fix, TrackPredictor
from .privacy import FeatureRange, LocalPrivatiser
from .spectrum import BandStatus
from .threshold_ca import Attestation, CertifierRoster, Grant, QuorumRenewer
from .validation import bounded_number

#: Declared bounds for the features a node may release under local differential
#: privacy. The mechanism needs a known range per feature, and it must come
#: from the operator: deriving it from the data would leak the outlier it was
#: meant to hide.
DEFAULT_FEATURE_RANGES: dict[str, FeatureRange] = {
    "rtt_ms": FeatureRange(0.0, 2000.0),
    "loss_percent": FeatureRange(0.0, 100.0),
    "jitter_ms": FeatureRange(0.0, 500.0),
    "signal_percent": FeatureRange(0.0, 100.0),
    "contested_band_count": FeatureRange(0.0, 32.0),
}

#: Share of a feature's privacy budget spent protecting its mean, the rest
#: going to its spread. Both moments leave the node, so both are perturbed and
#: the two shares sum to the declared budget under sequential composition.
#:
#: The mean takes the larger share because it is the primary quantity every
#: consumer reads, and because a spread is already a second-order figure that
#: tolerates more noise before it stops being useful for grouping. This is a
#: stated engineering judgement, not a derived optimum.
MEAN_BUDGET_SHARE = 0.6


def _spread_range(span: FeatureRange) -> FeatureRange:
    """The interval a standard deviation of this feature can occupy.

    A bounded feature has a bounded spread: the most dispersed distribution on
    ``[low, high]`` puts half its mass at each end, giving a standard deviation
    of half the width. Deriving the bound rather than declaring a second set of
    operator ranges keeps the two from drifting apart, and a perturbation
    calibrated to too wide an interval would add noise nothing required.
    """
    return FeatureRange(0.0, (span.high - span.low) / 2.0)

#: Records the influence bound is computed over before a baseline is trusted.
MIN_RECORDS_FOR_INFLUENCE_BOUND = 3


# --- federation: privacy, cohorts, reputation and excision as one pipeline --------


@dataclass(frozen=True)
class PrivacyDisclosure:
    """What a privatised release costs, stated rather than assumed."""

    applied: bool
    epsilon: float
    features: tuple[str, ...]
    contributors_needed_for_1pct: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "epsilon": round(self.epsilon, 6),
            "features": list(self.features),
            "contributors_needed_for_1pct": self.contributors_needed_for_1pct,
            "reasons": list(self.reasons),
        }


def privatise_update(
    update: FederatedUpdate,
    *,
    epsilon: float | None,
    ranges: dict[str, FeatureRange] | None = None,
) -> tuple[FederatedUpdate, PrivacyDisclosure]:
    """Perturb a node's means locally, before they leave the node.

    A node's exact feature mean encodes where that sensor is and what it sees.
    Sharing sufficient statistics narrows the exposure; it does not close it.

    The cost is real and is disclosed rather than hidden. Perturbed
    contributions are noisy by construction, so **consensus filtering at the
    aggregator will treat honest noise as divergence**. An operator gets
    privacy or poisoning-resistance for a given release, not both, and the
    disclosure says which one is in force.
    """
    if epsilon is None:
        return update, PrivacyDisclosure(
            applied=False,
            epsilon=0.0,
            features=(),
            contributors_needed_for_1pct=0,
            reasons=("NO_PRIVACY_BUDGET_DECLARED_SO_EXACT_MEANS_ARE_RELEASED",),
        )

    declared = ranges or DEFAULT_FEATURE_RANGES
    matched = {name: span for name, span in declared.items() if name in update.statistics}
    # The spread is released too, so it is privatised too. Reconstructing
    # ``sum_squares`` from the *true* spread published every node's exact
    # per-feature variance while claiming the release was perturbed: the mean
    # was protected and the second moment walked out untouched, recoverable to
    # zero error by any aggregator. That is not a small leak here. Variance in
    # signal is the multipath signature of an enclosed or reflective site, and
    # it is the quantity cohort clustering and the heterogeneity assessment use
    # to tell one environment from another — so the exact thing the fleet
    # fingerprints environments by was the thing being handed over intact.
    #
    # Each feature's budget is therefore split across the two moments it
    # actually releases. Sequential composition over the pair costs the sum, so
    # the split is what keeps the declared total honest rather than doubling
    # the spend behind the operator's back.
    spread_ranges = {name: _spread_range(span) for name, span in matched.items()}
    budget = bounded_number(epsilon, "epsilon", 0.01, 50.0)
    privatiser = LocalPrivatiser(matched, epsilon=budget * MEAN_BUDGET_SHARE)
    spread_privatiser = LocalPrivatiser(
        spread_ranges, epsilon=budget * (1.0 - MEAN_BUDGET_SHARE)
    )
    perturbed: dict[str, Any] = {}
    touched: list[str] = []
    needed = 0
    for name, statistic in update.statistics.items():
        if name not in privatiser.ranges:
            perturbed[name] = statistic
            continue
        released = privatiser.privatise(name, statistic.mean)
        released_spread = spread_privatiser.privatise(name, statistic.stddev())
        needed = max(
            needed,
            released.contributors_needed_for_1pct,
            released_spread.contributors_needed_for_1pct,
        )
        touched.append(name)
        # A spread cannot be negative, so the perturbation is clamped at zero.
        # The clamp is itself a disclosure — a released zero is more likely to
        # have come from a small true spread — but it is bounded and visible,
        # which an impossible variance would not be.
        spread = max(0.0, released_spread.perturbed)
        # Recomputed from the perturbed moments so the released statistics stay
        # internally consistent rather than describing a distribution that
        # cannot exist.
        perturbed[name] = type(statistic)(
            count=statistic.count,
            mean=released.perturbed,
            sum_squares=statistic.count * (released.perturbed**2 + spread**2),
        )

    disclosure = PrivacyDisclosure(
        applied=bool(touched),
        epsilon=privatiser.per_feature_epsilon,
        features=tuple(sorted(touched)),
        contributors_needed_for_1pct=needed,
        reasons=(
            "PERTURBED_LOCALLY_SO_NO_EXACT_NODE_MEAN_LEAVES_THIS_NODE",
            "CONSENSUS_FILTERING_WILL_READ_THIS_NOISE_AS_DIVERGENCE",
        )
        if touched
        else ("NO_DECLARED_FEATURE_RANGE_MATCHED_THIS_UPDATE",),
    )
    return (
        FederatedUpdate(
            node_id=update.node_id,
            produced_at=update.produced_at,
            examples=update.examples,
            statistics=perturbed,
        ),
        disclosure,
    )


class FleetLearning:
    """The federated pipeline, with every stage's paper doing its job.

    Reputation persists between rounds in a small JSON file beside the audit
    ledger. Without persistence a node that lied last round starts every round
    with a clean slate, which makes reputation decorative.
    """

    def __init__(self, reputation_path=None) -> None:
        self.reputation_path = reputation_path
        self.reputation = ReputationLedger(self._load_reputation())

    def _load_reputation(self) -> dict[str, float]:
        if self.reputation_path is None or not self.reputation_path.exists():
            return {}
        try:
            raw = json.loads(self.reputation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Unreadable reputation means unknown reputation, not good
            # reputation: everyone starts neutral rather than trusted.
            return {}
        return {
            str(node): bounded_number(float(score), "score", 0.0, 1.0)
            for node, score in raw.get("scores", {}).items()
        }

    def _save_reputation(self) -> None:
        if self.reputation_path is None:
            return
        self.reputation_path.parent.mkdir(parents=True, exist_ok=True)
        self.reputation_path.write_text(
            json.dumps({"scores": self.reputation.scores()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def aggregate(
        self, updates: Sequence[FederatedUpdate], *, round_index: int = 0
    ) -> dict[str, Any]:
        aggregator = HierarchicalAggregator(reputation=self.reputation)
        result = aggregator.aggregate(updates, round_index=round_index)
        self._save_reputation()
        payload = result.to_dict()
        payload["reasons"] = list(payload["reasons"]) + [
            "COHORTS_FORMED_BY_RF_ENVIRONMENT_NOT_BY_ADMINISTRATIVE_GROUPING",
            "AGGREGATOR_ROTATES_SO_NO_NODE_IS_A_PERMANENT_TARGET",
        ]
        return payload

    def excise(
        self, updates: Sequence[FederatedUpdate], node_id: str
    ) -> tuple[list[FederatedUpdate], RecoveryCertificate]:
        remaining, certificate = certified_excision(updates, node_id)
        # A node removed from the fleet must also lose its standing, or it
        # returns under the same reputation it had before it was excised — but
        # only if it was actually there. Demoting unconditionally meant a
        # typo'd node ID silently degraded a healthy node that had done nothing.
        if certificate.was_present:
            self.reputation.record_round(agreed=[], diverged=[node_id])
            self._save_reputation()
        return remaining, certificate


# --- forecasting that reaches a decision -------------------------------------------


@dataclass(frozen=True)
class ForecastAdvice:
    """Profiles a forecast says to demote, and why."""

    degrading_profiles: frozenset[str]
    per_flow: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-forecast-advice/v1",
            "degrading_profiles": sorted(self.degrading_profiles),
            "per_flow": list(self.per_flow),
            "reasons": list(self.reasons),
        }


def forecast_advice(history: dict[str, Sequence[float]]) -> ForecastAdvice:
    """Turn per-flow forecasts into a bounded instruction to the optimizer.

    The bound matters more than the forecast. A prediction may **demote** a
    profile and may never **promote** one, exactly as spectrum sensing may. A
    mechanism that could promote on a prediction could talk the node onto a
    link that has not yet demonstrated it works, and a wrong forecast would
    then cost connectivity rather than merely cost a ranking place.
    """
    forecaster = PerFlowForecaster()
    usable = {flow: list(values) for flow, values in history.items() if len(values) >= MIN_HISTORY}
    # The bound is a property of the mechanism, not of this particular result,
    # so it is stated on every advice including an empty one.
    if not usable:
        return ForecastAdvice(
            frozenset(),
            (),
            (
                "FORECAST_MAY_DEMOTE_A_PROFILE_AND_MAY_NEVER_PROMOTE_ONE",
                f"NO_FLOW_HAS_THE_{MIN_HISTORY}_SAMPLES_A_FORECAST_NEEDS",
            ),
        )

    results = forecaster.fit_all(usable)
    degrading = frozenset(result.flow_id for result in results if result.degrading)
    reasons = ["FORECAST_MAY_DEMOTE_A_PROFILE_AND_MAY_NEVER_PROMOTE_ONE"]
    if degrading:
        reasons.append("A_FORECAST_DEMOTION_IS_A_PREDICTION_NOT_AN_OBSERVED_FAILURE")
    else:
        reasons.append("NO_FLOW_IS_FORECAST_TO_DEGRADE")
    return ForecastAdvice(
        degrading_profiles=degrading,
        per_flow=tuple(result.to_dict() for result in results),
        reasons=tuple(reasons),
    )


# --- localisation feeding track prediction -----------------------------------------


def predict_track(
    fixes: Sequence[Fix], *, horizon: timedelta, place_radius_m: float
) -> dict[str, Any]:
    """Predict an emitter's next position from a history of NLOS-rejected fixes.

    The place radius is tied to localisation uncertainty rather than guessed:
    two fixes are the same place when they are closer together than the
    accuracy that produced them. Using a tighter radius would turn measurement
    noise into a tour of imaginary locations.
    """
    predictor = TrackPredictor(place_radius_m=place_radius_m)
    result = predictor.predict(fixes, horizon=horizon).to_dict()
    result["place_radius_m"] = round(place_radius_m, 2)
    result["fixes"] = len(fixes)
    return result


def fixes_from_estimates(records: Sequence[dict[str, Any]]) -> list[Fix]:
    """Read past localisation results out of recorded history.

    Estimates the locator refused are absent from history rather than recorded
    as a position, so a refusal cannot become a track point.
    """
    fixes: list[Fix] = []
    for row in records:
        payload = row.get("payload") or {}
        if payload.get("available") is not True:
            continue
        latitude = payload.get("latitude_deg")
        longitude = payload.get("longitude_deg")
        if latitude is None or longitude is None:
            continue
        from .models import parse_timestamp

        fixes.append(
            Fix(
                observed_at=parse_timestamp(str(row["recorded_at"])),
                latitude_deg=float(latitude),
                longitude_deg=float(longitude),
                uncertainty_radius_m=float(payload.get("uncertainty_radius_m") or 0.0),
            )
        )
    return fixes


# --- detection as a fleet property --------------------------------------------------

#: Band statuses that count as an anomalous observation for a node.
_ANOMALOUS_STATUSES = frozenset(
    {BandStatus.JAMMING_SUSPECTED.value, BandStatus.ANOMALOUS.value}
)


def cohort_assignments(updates: Sequence[FederatedUpdate]) -> dict[str, str]:
    """Which RF-environment cohort each node belongs to.

    Cohort membership is what makes fleet-level corroboration meaningful. Nodes
    in one cohort share an environment by construction, so their agreement is
    explainable locally; agreement *across* cohorts is not.
    """
    assignments: dict[str, str] = {}
    for index, cohort in enumerate(kmeans_cohorts(updates)):
        for update in cohort:
            assignments[update.node_id] = f"cohort-{index}"
    return assignments


def verdicts_from_reports(
    reports: Sequence[FabricReport],
    baselines: dict[str, NodeBaseline],
    cohorts: dict[str, str],
) -> list[LocalVerdict]:
    """Each node's own judgement of its own readings, per the ICC 2024 shape.

    Where a node has a usable baseline of its own, its detector decides. Where
    it does not, the band status it reported stands in — a weaker signal, and
    one that carries no feature directions, so it can corroborate a detector's
    finding but cannot manufacture one on its own.
    """
    from .intrusion import FeatureDeviation

    verdicts: list[LocalVerdict] = []
    for report in reports:
        cohort = cohorts.get(report.node_id, "unassigned")
        baseline = baselines.get(report.node_id)
        if baseline is not None and baseline.usable():
            reading = {
                "contested_band_count": float(
                    sum(1 for band in report.bands if band.status.value in _ANOMALOUS_STATUSES)
                ),
                "noise_floor_dbm": _mean_noise_excess(report),
            }
            verdicts.append(
                LocalDetector(baseline).judge(
                    reading, cohort_id=cohort, observed_at=report.produced_at
                )
            )
            continue

        anomalous = [band for band in report.bands if band.status.value in _ANOMALOUS_STATUSES]
        verdicts.append(
            LocalVerdict(
                node_id=report.node_id,
                cohort_id=cohort,
                observed_at=report.produced_at,
                anomalous=bool(anomalous),
                deviations=tuple(
                    FeatureDeviation(f"band:{band.band_id}", band.confidence, 1)
                    for band in anomalous
                ),
                reasons=("NO_LOCAL_BASELINE_SO_THE_REPORTED_BAND_STATUS_STANDS_IN",),
            )
        )
    return verdicts


def _mean_noise_excess(report: FabricReport) -> float:
    values = [
        band.noise_floor_excess_db
        for band in report.bands
        if band.noise_floor_excess_db is not None
    ]
    return sum(values) / len(values) if values else 0.0


def detect_fleet_event(
    verdicts: Sequence[LocalVerdict], reputations: dict[str, float]
) -> dict[str, Any]:
    """Combine per-node verdicts, discounted by earned standing."""
    finding = FleetDetector(reputations=reputations).assess(verdicts)
    payload = finding.to_dict()
    payload["verdicts"] = [verdict.to_dict() for verdict in verdicts]
    return payload


# --- integrity guarding the learning path -------------------------------------------


@dataclass(frozen=True)
class BaselineTrust:
    trusted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"trusted": self.trusted, "reasons": list(self.reasons)}


def trusted_baseline(
    examples: Sequence[Example], *, memory_report: dict[str, Any] | None
) -> tuple[Any, BaselineTrust]:
    """Build a feature baseline only from history that survives integrity checks.

    Two gates, in order. History containing records this node did not write is
    not this node's history, so no baseline is derived from it at all. And a
    baseline that one record could overturn was never adequately evidenced —
    precisely the leverage a small injection buys.

    The influence bound is applied per feature rather than to one summary
    number, because an injection that shifts a single feature would otherwise
    hide behind the others staying steady.
    """
    reasons: list[str] = []
    if memory_report is not None and memory_report.get("available") is True:
        if memory_report.get("trustworthy") is not True:
            return None, BaselineTrust(
                False,
                ("MEMORY_AUDIT_FAILED_SO_NO_BASELINE_IS_DERIVED_FROM_THIS_HISTORY",),
            )
        reasons.append("HISTORY_PASSED_THE_MEMORY_AUDIT")
    else:
        reasons.append("MEMORY_AUDIT_UNAVAILABLE_SO_AUTHORSHIP_IS_UNVERIFIED")

    dominated: list[str] = []
    for feature in sorted({name for example in examples for name in example.features}):
        values = [
            float(example.features[feature])
            for example in examples
            if feature in example.features
        ]
        if len(values) < MIN_RECORDS_FOR_INFLUENCE_BOUND:
            continue
        if not bounded_influence(values).within_bound:
            dominated.append(feature)

    if dominated:
        return None, BaselineTrust(
            False,
            tuple(
                reasons
                + [
                    "A_SINGLE_RECORD_DOMINATES_THE_BASELINE_SO_IT_IS_NOT_USED",
                    f"DOMINATED_FEATURES:{','.join(dominated)}",
                ]
            ),
        )
    reasons.append("NO_SINGLE_RECORD_DOMINATES_ANY_FEATURE")

    baseline, baseline_reasons = build_baseline(examples)
    if baseline is None:
        return None, BaselineTrust(False, tuple(reasons) + baseline_reasons)
    return baseline, BaselineTrust(True, tuple(reasons) + baseline_reasons)


# --- certification surviving partition ----------------------------------------------


def renew_under_partition(
    grant: Grant,
    attestations: Sequence[Attestation],
    roster: CertifierRoster,
    *,
    new_expiry: datetime,
    revoked: frozenset[str] = frozenset(),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Extend an expiring grant on the strength of reachable peers.

    Reached only when the operator cannot be. The renewal moves one field of an
    already-signed grant and nothing else, so the worst outcome of a
    compromised quorum is that a capability the operator already approved lives
    a few hours longer than intended.
    """
    result = QuorumRenewer(roster, revoked=revoked).renew(
        grant, attestations, new_expiry=new_expiry, now=now
    )
    payload = result.to_dict()
    payload["available"] = True
    return payload


def propagated_to_belief(entry: dict[str, Any]) -> BeliefMass:
    """Read a propagated-belief record back into a mass assignment."""
    belief = entry.get("belief") or {}
    return BeliefMass(
        clear=float(belief.get("clear", 0.0)),
        busy=float(belief.get("busy", 0.0)),
        unknown=float(belief.get("unknown", 1.0)),
    )
