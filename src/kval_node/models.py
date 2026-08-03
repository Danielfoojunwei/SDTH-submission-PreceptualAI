from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .standards import StandardsRequirement


class AccessType(str, Enum):
    WIFI = "wifi"
    THREE_G = "3g"
    FOUR_G = "4g"
    FIVE_G = "5g"
    NTN = "ntn"
    ETHERNET = "ethernet"


class ServiceClass(str, Enum):
    CRITICAL_DATA = "critical_data"
    TELEMETRY = "telemetry"
    BULK_DATA = "bulk_data"


class ShieldAction(str, Enum):
    PRESENT_TO_HOST = "PRESENT_TO_HOST"
    APPLY_APPROVED_PROFILE = "APPLY_APPROVED_PROFILE"
    REJECT_RETAIN_CURRENT = "REJECT_RETAIN_CURRENT"
    ABSTAIN = "ABSTAIN"


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must contain a timezone")
    return parsed.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return number


@dataclass(frozen=True)
class Profile:
    profile_id: str
    access_type: AccessType
    connection_uuid: str
    interface: str
    probe_target: str
    monetary_cost: float
    energy_cost: float
    minimum_dwell_seconds: int
    maximum_switches_per_hour: int
    allowed_service_classes: frozenset[ServiceClass]
    standards: StandardsRequirement
    #: Spectrum policy band IDs this profile's radio depends on. Declared by the
    #: operator in the signed catalogue rather than inferred, so passive sensing
    #: can warn the optimizer off a contested band without the node guessing
    #: which frequencies a connection actually uses. Empty means "no spectrum
    #: evidence applies", which is the pre-existing behaviour.
    #: What this link is *for*. Declared in the signed catalogue, never
    #: inferred. The default is deliberately ``payload``: a command link must
    #: be declared explicitly, so a profile can never become safety-critical by
    #: omission — and an undeclared link can never be selected as C2.
    link_role: str = "payload"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Profile":
        required = {
            "profile_id",
            "access_type",
            "connection_uuid",
            "interface",
            "probe_target",
            "monetary_cost",
            "energy_cost",
            "minimum_dwell_seconds",
            "maximum_switches_per_hour",
            "allowed_service_classes",
            "standards",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"profile is missing fields: {', '.join(missing)}")
        profile_id = str(raw["profile_id"]).strip()
        interface = str(raw["interface"]).strip()
        probe_target = str(raw["probe_target"]).strip()
        connection_uuid = str(raw["connection_uuid"]).strip()
        for name, value in {
            "profile_id": profile_id,
            "interface": interface,
            "probe_target": probe_target,
            "connection_uuid": connection_uuid,
        }.items():
            if not value or "REPLACE_" in value:
                raise ValueError(f"{name} must be provisioned with a real value")
        uuid.UUID(connection_uuid)
        dwell = int(raw["minimum_dwell_seconds"])
        maximum_switches = int(raw["maximum_switches_per_hour"])
        if dwell < 0:
            raise ValueError("minimum_dwell_seconds cannot be negative")
        if maximum_switches < 1:
            raise ValueError("maximum_switches_per_hour must be positive")
        classes = frozenset(ServiceClass(value) for value in raw["allowed_service_classes"])
        if not classes:
            raise ValueError("allowed_service_classes cannot be empty")
        standards = StandardsRequirement.from_dict(raw["standards"])
        access_type = AccessType(str(raw["access_type"]).lower())
        expected_rat = {
            AccessType.THREE_G: "utran",
            AccessType.FOUR_G: "eutran",
            AccessType.FIVE_G: "nr",
        }.get(access_type)
        if expected_rat is not None:
            if standards.required_rat is None or standards.required_rat.value != expected_rat:
                raise ValueError(f"{access_type.value} profiles require RAT {expected_rat}")
            if standards.dnn is None:
                raise ValueError(f"{access_type.value} profiles require a provisioned DNN/APN")
            if "3gpp-ts-23.003" not in {item.value for item in standards.required_identifiers}:
                raise ValueError(f"{access_type.value} profiles must require 3gpp-ts-23.003")
        link_role = str(raw.get("link_role", "payload")).strip().lower()
        if link_role not in {"c2_cnpc", "payload", "bulk"}:
            raise ValueError(f"unknown link_role {link_role!r}")
        return cls(
            profile_id=profile_id,
            access_type=access_type,
            connection_uuid=connection_uuid,
            interface=interface,
            probe_target=probe_target,
            monetary_cost=_bounded_number(raw["monetary_cost"], "monetary_cost", 0.0, 1.0),
            energy_cost=_bounded_number(raw["energy_cost"], "energy_cost", 0.0, 1.0),
            minimum_dwell_seconds=dwell,
            maximum_switches_per_hour=maximum_switches,
            allowed_service_classes=classes,
            standards=standards,
            link_role=link_role,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            # Without this an operator running verify-catalog could not see
            # which profile is the command link.
            "link_role": self.link_role,
            "profile_id": self.profile_id,
            "access_type": self.access_type.value,
            "interface": self.interface,
            "standards": sorted(identifier.value for identifier in self.standards.required_identifiers),
        }


@dataclass(frozen=True)
class Catalog:
    version: str
    created_at: datetime
    expires_at: datetime
    profiles: tuple[Profile, ...]
    payload_hash: str
    key_id: str

    def by_id(self) -> dict[str, Profile]:
        return {profile.profile_id: profile for profile in self.profiles}


@dataclass(frozen=True)
class LinkObservation:
    profile_id: str
    interface: str
    observed_at: datetime
    interface_present: bool
    link_up: bool
    connection_active: bool
    rtt_ms: float | None
    loss_percent: float | None
    jitter_ms: float | None
    rx_mbps: float | None
    tx_mbps: float | None
    signal_percent: float | None
    access_technology: str | None
    standards_verified: bool
    standards_evidence: tuple[str, ...]
    standards_errors: tuple[str, ...]
    evidence: tuple[str, ...]
    errors: tuple[str, ...] = ()

    def confidence(self) -> float:
        mandatory = (self.rtt_ms, self.loss_percent, self.jitter_ms)
        mandatory_fraction = sum(value is not None for value in mandatory) / len(mandatory)
        optional = (self.rx_mbps, self.tx_mbps, self.signal_percent)
        optional_fraction = sum(value is not None for value in optional) / len(optional)
        base = 0.7 * mandatory_fraction + 0.3 * optional_fraction
        if not self.interface_present or not self.link_up or not self.connection_active:
            return 0.0
        return round(base, 6)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["observed_at"] = self.observed_at.isoformat()
        result["confidence"] = self.confidence()
        return result


@dataclass(frozen=True)
class TelemetrySnapshot:
    observed_at: datetime
    current_profile_id: str | None
    observations: tuple[LinkObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat(),
            "current_profile_id": self.current_profile_id,
            "observations": [observation.to_dict() for observation in self.observations],
        }


@dataclass(frozen=True)
class ScoredProfile:
    profile_id: str
    utility: float
    confidence: float
    components: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    recommendation_id: str
    created_at: datetime
    catalog_version: str
    selected_profile_id: str | None
    current_profile_id: str | None
    service_class: ServiceClass
    confidence: float
    scores: tuple[ScoredProfile, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "created_at": self.created_at.isoformat(),
            "catalog_version": self.catalog_version,
            "selected_profile_id": self.selected_profile_id,
            "current_profile_id": self.current_profile_id,
            "service_class": self.service_class.value,
            "confidence": self.confidence,
            "scores": [score.to_dict() for score in self.scores],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ShieldDecision:
    action: ShieldAction
    recommendation: Recommendation
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason_codes": list(self.reason_codes),
            "recommendation": self.recommendation.to_dict(),
        }
