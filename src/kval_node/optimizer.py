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


def _lower_is_better(value: float, unacceptable: float) -> float:
    return _clamp(1.0 - value / unacceptable)


def _higher_is_better(value: float, target: float) -> float:
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


class ProfileOptimizer:
    """Transparent constrained optimizer over real observations."""

    @staticmethod
    def _score(
        observation: LinkObservation,
        *,
        monetary_cost: float,
        energy_cost: float,
        is_current: bool,
        objective: Objective,
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
        return ScoredProfile(
            profile_id=observation.profile_id,
            utility=round(utility, 8),
            confidence=observation.confidence(),
            components={key: round(value, 8) for key, value in components.items()},
            reasons=("LIVE_EVIDENCE_SCORED",),
        )

    def recommend(
        self,
        catalog: Catalog,
        snapshot: TelemetrySnapshot,
        service_class: ServiceClass,
    ) -> Recommendation:
        observations = {item.profile_id: item for item in snapshot.observations}
        objective = OBJECTIVES[service_class]
        scores: list[ScoredProfile] = []
        for profile in catalog.profiles:
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
                )
            )

        eligible = sorted((score for score in scores if score.utility >= 0.0), key=lambda score: score.utility, reverse=True)
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
