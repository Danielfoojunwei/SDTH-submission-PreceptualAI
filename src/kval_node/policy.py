from __future__ import annotations

from datetime import datetime, timezone

from .audit import AuditLedger
from .models import (
    Catalog,
    Recommendation,
    ShieldAction,
    ShieldDecision,
    TelemetrySnapshot,
)


class SafetyShield:
    def __init__(
        self,
        catalog: Catalog,
        audit: AuditLedger,
        *,
        observation_max_age_seconds: float,
        minimum_confidence: float,
    ) -> None:
        self.catalog = catalog
        self.audit = audit
        self.observation_max_age_seconds = observation_max_age_seconds
        self.minimum_confidence = minimum_confidence

    def evaluate(
        self,
        recommendation: Recommendation,
        snapshot: TelemetrySnapshot,
        *,
        apply_requested: bool,
        now: datetime | None = None,
    ) -> ShieldDecision:
        current = now or datetime.now(timezone.utc)
        failures: list[str] = []
        if current >= self.catalog.expires_at:
            failures.append("CATALOG_EXPIRED")
        if recommendation.catalog_version != self.catalog.version:
            failures.append("CATALOG_VERSION_MISMATCH")
        recommendation_age = (current - recommendation.created_at).total_seconds()
        if recommendation_age < -1.0 or recommendation_age > self.observation_max_age_seconds:
            failures.append("RECOMMENDATION_STALE")
        snapshot_age = (current - snapshot.observed_at).total_seconds()
        if snapshot_age < -1.0 or snapshot_age > self.observation_max_age_seconds:
            failures.append("OBSERVATION_STALE")
        if recommendation.selected_profile_id is None:
            return ShieldDecision(ShieldAction.ABSTAIN, recommendation, ("NO_SELECTED_PROFILE",))

        profiles = self.catalog.by_id()
        profile = profiles.get(recommendation.selected_profile_id)
        if profile is None:
            failures.append("PROFILE_NOT_IN_SIGNED_CATALOG")
        elif recommendation.service_class not in profile.allowed_service_classes:
            failures.append("SERVICE_CLASS_NOT_AUTHORIZED")
        if recommendation.confidence < self.minimum_confidence:
            failures.append("CONFIDENCE_BELOW_POLICY")

        observation = next(
            (item for item in snapshot.observations if item.profile_id == recommendation.selected_profile_id),
            None,
        )
        if observation is None:
            failures.append("SELECTED_OBSERVATION_MISSING")
        else:
            observation_age = (current - observation.observed_at).total_seconds()
            if observation_age < -1.0 or observation_age > self.observation_max_age_seconds:
                failures.append("SELECTED_OBSERVATION_STALE")
            if not observation.interface_present:
                failures.append("SELECTED_INTERFACE_MISSING")
            if not observation.link_up:
                failures.append("SELECTED_LINK_DOWN")
            if not observation.connection_active:
                failures.append("SELECTED_CONNECTION_INACTIVE")
            if not observation.standards_verified:
                failures.append("SELECTED_STANDARDS_REQUIREMENTS_UNVERIFIED")
            if observation.rtt_ms is None or observation.loss_percent is None or observation.jitter_ms is None:
                failures.append("SELECTED_PROBE_EVIDENCE_INCOMPLETE")

        if profile is not None and recommendation.selected_profile_id != snapshot.current_profile_id:
            activations = self.audit.recent_activations(now=current)
            if len(activations) >= profile.maximum_switches_per_hour:
                failures.append("SWITCH_RATE_LIMIT")
            if activations:
                last_activation = datetime.fromisoformat(activations[-1]["created_at"])
                if (current - last_activation).total_seconds() < profile.minimum_dwell_seconds:
                    failures.append("MINIMUM_DWELL_NOT_MET")

        if failures:
            return ShieldDecision(ShieldAction.REJECT_RETAIN_CURRENT, recommendation, tuple(sorted(set(failures))))
        if recommendation.selected_profile_id == snapshot.current_profile_id:
            return ShieldDecision(ShieldAction.PRESENT_TO_HOST, recommendation, ("CURRENT_PROFILE_ALREADY_SELECTED",))
        if apply_requested:
            return ShieldDecision(ShieldAction.APPLY_APPROVED_PROFILE, recommendation, ("ALL_SAFETY_CHECKS_PASSED",))
        return ShieldDecision(ShieldAction.PRESENT_TO_HOST, recommendation, ("ADVISORY_MODE",))
