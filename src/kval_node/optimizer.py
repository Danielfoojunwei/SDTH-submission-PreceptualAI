from __future__ import annotations

import uuid
from dataclasses import dataclass

from .models import (
    Catalog,
    LinkObservation,
    Recommendation,
    ScoredProfile,
    ServiceClass,
    TelemetrySnapshot,
    utc_now,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


#: The band reserved for measurements that are already past their limit.
#:
#: A hard clamp to zero made every failing link score identically, and a node
#: whose whole purpose is contested spectrum spends its worst moments with
#: *every* link past its limits. Measured before the change: a link at 300 ms /
#: 25% loss and one at 6000 ms / 95% loss both scored 0.20600, so the selection
#: fell through to catalogue order — swapping the two entries in the catalogue
#: swapped which link the node chose, while it reported
#: ``HIGHEST_POLICY_CONSTRAINED_UTILITY`` as though a comparison had decided it.
#:
#: This contradicted the principle ``c2link`` already states for command links:
#: the budget is a floor, because best-of-a-bad-set is how a vehicle keeps
#: flying. Ranking has to survive past the limit or it is absent exactly when it
#: is needed.
SATURATION_FLOOR = 0.05


def _lower_is_better(value: float, unacceptable: float) -> float:
    """Score a measurement where less is better, ordered past the limit too.

    Acceptable readings occupy ``[1.0, SATURATION_FLOOR]`` and readings past the
    limit decay asymptotically below it, so nothing failing can ever outrank
    anything acceptable. The transform is order-preserving: within the
    acceptable region it is a positive affine map of the old score, and across
    the boundary every acceptable reading still beats every unacceptable one. It
    only refines comparisons that used to be ties, so no ranking that was
    correct before can invert.
    """
    if unacceptable <= 0.0:
        return 0.0
    ratio = value / unacceptable
    if ratio <= 1.0:
        return _clamp(1.0 - ratio) * (1.0 - SATURATION_FLOOR) + SATURATION_FLOOR
    return SATURATION_FLOOR / ratio


def _higher_is_better(value: float, target: float) -> float:
    """Score a measurement where more is better.

    Deliberately left saturating. This one clamps at the *good* end — links that
    all exceed the throughput target tie at 1.0 — and indifference between links
    that have all met the target is a defensible answer, unlike indifference
    between links that have all failed.
    """
    return _clamp(value / target)


@dataclass(frozen=True)
class Objective:
    latency_limit_ms: float
    loss_limit_percent: float
    jitter_limit_ms: float
    throughput_target_mbps: float
    weights: dict[str, float]
    switch_margin: float


OBJECTIVES: dict[ServiceClass, Objective] = {
    ServiceClass.CRITICAL_DATA: Objective(
        latency_limit_ms=250.0,
        loss_limit_percent=20.0,
        jitter_limit_ms=100.0,
        throughput_target_mbps=2.0,
        weights={"latency": 0.29, "loss": 0.29, "jitter": 0.17, "throughput": 0.08, "signal": 0.07, "cost": 0.04, "energy": 0.03, "stability": 0.03},
        switch_margin=0.08,
    ),
    ServiceClass.TELEMETRY: Objective(
        latency_limit_ms=500.0,
        loss_limit_percent=30.0,
        jitter_limit_ms=200.0,
        throughput_target_mbps=0.5,
        weights={"latency": 0.22, "loss": 0.27, "jitter": 0.12, "throughput": 0.10, "signal": 0.08, "cost": 0.08, "energy": 0.08, "stability": 0.05},
        switch_margin=0.10,
    ),
    ServiceClass.BULK_DATA: Objective(
        latency_limit_ms=1000.0,
        loss_limit_percent=40.0,
        jitter_limit_ms=500.0,
        throughput_target_mbps=25.0,
        weights={"latency": 0.08, "loss": 0.14, "jitter": 0.04, "throughput": 0.46, "signal": 0.08, "cost": 0.10, "energy": 0.07, "stability": 0.03},
        switch_margin=0.06,
    ),
}


#: Utility multiplier applied to a profile whose declared spectrum band is
#: showing corroborated interference. Passive sensing is coarse, so it is given
#: enough weight to break a near-tie and steer away from a contested band, but
#: never enough to override complete live probe evidence on its own.
SPECTRUM_DEGRADED_MULTIPLIER = 0.6

#: Penalty applied to a profile a forecast expects to degrade. Weaker than the
#: sensing penalty on purpose: sensing reports what *is* happening, a forecast
#: reports what might. A prediction should be able to break a tie, not to
#: overrule measured evidence.
FORECAST_DEGRADED_MULTIPLIER = 0.85


class ProfileOptimizer:
    """Transparent constrained optimizer over real observations.

    Spectrum evidence enters in exactly one direction: it can **demote** a
    profile whose operator-declared band is contested. It can never promote one.
    A band that merely looks quiet is not evidence that a link works — only the
    live probe is — so sensing is allowed to warn us off a medium and nothing
    more.
    """

    @staticmethod
    def _score(
        observation: LinkObservation,
        *,
        monetary_cost: float,
        energy_cost: float,
        is_current: bool,
        objective: Objective,
        spectrum_degraded: bool = False,
        forecast_degrading: bool = False,
    ) -> ScoredProfile:
        reasons: list[str] = []
        if not observation.interface_present:
            reasons.append("INTERFACE_MISSING")
        if not observation.link_up:
            reasons.append("LINK_DOWN")
        if not observation.connection_active:
            reasons.append("CONNECTION_NOT_ACTIVE")
        if not observation.standards_verified:
            reasons.append("STANDARDS_REQUIREMENTS_NOT_VERIFIED")
        if observation.rtt_ms is None or observation.loss_percent is None or observation.jitter_ms is None:
            reasons.append("PROBE_EVIDENCE_INCOMPLETE")
        if reasons:
            return ScoredProfile(observation.profile_id, -1.0, observation.confidence(), {}, tuple(reasons))

        throughput = (observation.rx_mbps or 0.0) + (observation.tx_mbps or 0.0)
        components = {
            "latency": _lower_is_better(observation.rtt_ms, objective.latency_limit_ms),
            "loss": _lower_is_better(observation.loss_percent, objective.loss_limit_percent),
            "jitter": _lower_is_better(observation.jitter_ms, objective.jitter_limit_ms),
            "throughput": _higher_is_better(throughput, objective.throughput_target_mbps),
            "signal": _higher_is_better(observation.signal_percent or 0.0, 100.0),
            "cost": 1.0 - monetary_cost,
            "energy": 1.0 - energy_cost,
            "stability": 1.0 if is_current else 0.0,
        }
        utility = sum(components[name] * weight for name, weight in objective.weights.items())
        utility *= observation.confidence()
        scored_reasons = ["LIVE_EVIDENCE_SCORED"]
        if spectrum_degraded:
            utility *= SPECTRUM_DEGRADED_MULTIPLIER
            components["spectrum_penalty"] = SPECTRUM_DEGRADED_MULTIPLIER
            scored_reasons.append("SPECTRUM_BAND_CONTESTED")
        if forecast_degrading:
            utility *= FORECAST_DEGRADED_MULTIPLIER
            components["forecast_penalty"] = FORECAST_DEGRADED_MULTIPLIER
            scored_reasons.append("FORECAST_EXPECTS_THIS_LINK_TO_DEGRADE")
        return ScoredProfile(
            profile_id=observation.profile_id,
            utility=round(utility, 8),
            confidence=observation.confidence(),
            components={key: round(value, 8) for key, value in components.items()},
            reasons=tuple(scored_reasons),
        )

    @staticmethod
    def _empty_recommendation(
        catalog: Catalog,
        snapshot: TelemetrySnapshot,
        service_class: ServiceClass,
        *,
        scores: tuple[ScoredProfile, ...],
        reason_codes: tuple[str, ...],
    ) -> Recommendation:
        """A recommendation that selects nothing, built from the real dataclass.

        Exists so an early exit cannot quietly diverge from ``Recommendation``'s
        required fields — which is how the emission-control branch shipped with
        a constructor call that could never have succeeded.
        """
        return Recommendation(
            recommendation_id=str(uuid.uuid4()),
            created_at=utc_now(),
            catalog_version=catalog.version,
            selected_profile_id=None,
            current_profile_id=snapshot.current_profile_id,
            service_class=service_class,
            confidence=0.0,
            scores=scores,
            reason_codes=reason_codes,
        )

    def recommend(
        self,
        catalog: Catalog,
        snapshot: TelemetrySnapshot,
        service_class: ServiceClass,
        *,
        degraded_band_ids: frozenset[str] = frozenset(),
        forecast_degrading_profiles: frozenset[str] = frozenset(),
        role: str = "payload",
        suppressed_roles: frozenset[str] = frozenset(),
    ) -> Recommendation:
        """Rank approved profiles.

        ``degraded_band_ids`` names spectrum bands this node's own sensing found
        contested. Not corroborated: the loop passes the result of its local
        ``sense_spectrum`` only, and no peer report reaches this parameter. A
        profile that declared one of those bands in the signed catalogue is
        demoted; profiles that declared no bands are unaffected, so a node
        without spectrum sensing behaves exactly as before.

        ``forecast_degrading_profiles`` names links a per-flow forecast expects
        to worsen. Both inputs may only **demote**. Neither may promote, and the
        forecast penalty is the weaker of the two, because a prediction that
        could outrank measured evidence would let the node talk itself onto a
        link that has not demonstrated it works.

        ``role`` confines ranking to one link role. This is a safety property,
        not a filter for convenience: comparing a command link against a
        payload link on one utility scale lets a high-value sensor downlink
        outscore the link that steers the vehicle, and utility weights are
        exactly the sort of thing that gets retuned for throughput by someone
        who does not know that. Candidates of another role are refused with a
        reason rather than silently dropped.

        ``suppressed_roles`` comes from the emission-control posture. A role
        the platform is ordered not to emit on has no eligible candidates at
        all, which is different from having bad ones.
        """
        observations = {item.profile_id: item for item in snapshot.observations}
        objective = OBJECTIVES[service_class]
        scores: list[ScoredProfile] = []
        if role in suppressed_roles:
            # This branch had never executed: it built a Recommendation with a
            # keyword the dataclass does not have and five required fields
            # missing, so the first emission-control order would have crashed
            # the cycle — failing *open into a crash* at exactly the moment the
            # control exists for. Constructed through the same helper as every
            # other exit now, so it cannot drift from the dataclass again.
            return self._empty_recommendation(
                catalog,
                snapshot,
                service_class,
                scores=tuple(
                    ScoredProfile(
                        profile.profile_id,
                        -1.0,
                        0.0,
                        {},
                        ("ROLE_SUPPRESSED_BY_EMISSION_CONTROL",),
                    )
                    for profile in catalog.profiles
                    if profile.link_role == role
                ),
                reason_codes=("EMISSION_CONTROL_SUPPRESSES_THIS_ROLE_SO_NO_LINK_IS_SELECTED",),
            )
        for profile in catalog.profiles:
            if profile.link_role != role:
                scores.append(
                    ScoredProfile(profile.profile_id, -1.0, 0.0, {}, ("DIFFERENT_LINK_ROLE",))
                )
                continue
            if service_class not in profile.allowed_service_classes:
                scores.append(ScoredProfile(profile.profile_id, -1.0, 0.0, {}, ("SERVICE_CLASS_NOT_ALLOWED",)))
                continue
            observation = observations.get(profile.profile_id)
            if observation is None:
                scores.append(ScoredProfile(profile.profile_id, -1.0, 0.0, {}, ("OBSERVATION_MISSING",)))
                continue
            scores.append(
                self._score(
                    observation,
                    monetary_cost=profile.monetary_cost,
                    energy_cost=profile.energy_cost,
                    is_current=profile.profile_id == snapshot.current_profile_id,
                    objective=objective,
                    spectrum_degraded=bool(profile.spectrum_band_ids & degraded_band_ids),
                    forecast_degrading=profile.profile_id in forecast_degrading_profiles,
                )
            )

        # Ties break on profile_id, not on catalogue order. Identical readings
        # do still tie, and when they did the winner used to be whichever entry
        # the catalogue happened to list first — which meant an operator could
        # change the node's choice by reordering a file. Deterministic and
        # declared beats incidental and silent.
        eligible = sorted(
            (score for score in scores if score.utility >= 0.0),
            key=lambda score: (-score.utility, score.profile_id),
        )
        selected = eligible[0] if eligible else None
        reasons: list[str] = []
        if selected is None:
            reasons.append("NO_PROFILE_WITH_COMPLETE_LIVE_EVIDENCE")
        else:
            current = next((score for score in eligible if score.profile_id == snapshot.current_profile_id), None)
            if current is not None and selected.profile_id != current.profile_id:
                if selected.utility - current.utility < objective.switch_margin:
                    selected = current
                    reasons.append("CURRENT_PROFILE_RETAINED_BY_HYSTERESIS")
            if sum(1 for score in eligible if score.utility == selected.utility) > 1:
                # Saying "highest utility" when nothing was higher is the same
                # class of untrue report as naming a corroborator that measured
                # the band clear. The operator needs to know the evidence did
                # not separate these links.
                reasons.append("TOP_PROFILES_TIED_ON_UTILITY_SO_THE_EVIDENCE_DID_NOT_CHOOSE")
            if not reasons:
                reasons.append("HIGHEST_POLICY_CONSTRAINED_UTILITY")
        return Recommendation(
            recommendation_id=str(uuid.uuid4()),
            created_at=utc_now(),
            catalog_version=catalog.version,
            selected_profile_id=selected.profile_id if selected else None,
            current_profile_id=snapshot.current_profile_id,
            service_class=service_class,
            confidence=selected.confidence if selected else 0.0,
            scores=tuple(sorted(scores, key=lambda score: score.utility, reverse=True)),
            reason_codes=tuple(reasons),
        )
