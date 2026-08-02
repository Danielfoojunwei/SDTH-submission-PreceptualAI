"""Multimodal state fusion with provenance.

Autonomous decisions need more than one link metric. This module binds together
the operational modalities Kval can observe — network, radio, compute, power,
thermal, service, security, fleet, spectrum, and context — into a single
time-bounded state. Every modality observation carries its own source identity,
timestamp, quality, missingness map, and provenance hash. Fusion is fail-closed:
a mandatory modality that is stale, missing, or without provenance drops the
state to "not decision-ready" instead of quietly filling a gap.

This is deliberately not a learned encoder. It is a transparent, auditable
container so that a later learned state estimator has clean, provenance-tagged
inputs — and so that "multimodal" does not simply mean "more inputs for an
attacker to spoof".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .models import utc_now
from .validation import bounded_number, non_empty_str, sha256_hex


class Modality(str, Enum):
    NETWORK = "network"
    RADIO = "radio"
    COMPUTE = "compute"
    POWER = "power"
    THERMAL = "thermal"
    SERVICE = "service"
    SECURITY = "security"
    FLEET = "fleet"
    SPECTRUM = "spectrum"
    CONTEXT = "context"


@dataclass(frozen=True)
class ModalityObservation:
    modality: Modality
    source_id: str
    observed_at: datetime
    features: dict[str, float]
    quality: float
    missing: tuple[str, ...]
    provenance_hash: str

    @classmethod
    def create(
        cls,
        modality: Modality,
        *,
        source_id: str,
        observed_at: datetime,
        features: dict[str, float],
        provenance_hash: str,
        quality: float = 1.0,
        missing: tuple[str, ...] = (),
    ) -> "ModalityObservation":
        source = non_empty_str(source_id, "source_id")
        validated: dict[str, float] = {}
        for key, value in features.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"feature {key} for modality {modality.value} must be finite")
            validated[str(key)] = number
        return cls(
            modality=modality,
            source_id=source,
            observed_at=observed_at,
            features=validated,
            quality=bounded_number(quality, "quality", 0.0, 1.0),
            missing=tuple(str(item) for item in missing),
            provenance_hash=sha256_hex(provenance_hash, "provenance_hash"),
        )

    def age_seconds(self, now: datetime) -> float:
        return (now - self.observed_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "source_id": self.source_id,
            "observed_at": self.observed_at.isoformat(),
            "features": {key: self.features[key] for key in sorted(self.features)},
            "quality": self.quality,
            "missing": list(self.missing),
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class FusedState:
    produced_at: datetime
    window_seconds: float
    decision_ready: bool
    present_modalities: tuple[str, ...]
    stale_modalities: tuple[str, ...]
    missing_mandatory: tuple[str, ...]
    provenance_coverage: float
    completeness: float
    observations: tuple[ModalityObservation, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-multimodal-state/v1",
            "produced_at": self.produced_at.isoformat(),
            "window_seconds": self.window_seconds,
            "decision_ready": self.decision_ready,
            "present_modalities": list(self.present_modalities),
            "stale_modalities": list(self.stale_modalities),
            "missing_mandatory": list(self.missing_mandatory),
            "provenance_coverage": self.provenance_coverage,
            "completeness": self.completeness,
            "observations": [item.to_dict() for item in self.observations],
            "reasons": list(self.reasons),
        }


#: Modalities without which a bounded autonomous action must not be taken.
DEFAULT_MANDATORY: frozenset[Modality] = frozenset({Modality.NETWORK, Modality.POWER, Modality.SECURITY})


class MultimodalFuser:
    def __init__(
        self,
        *,
        window_seconds: float,
        mandatory: frozenset[Modality] = DEFAULT_MANDATORY,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self.mandatory = mandatory

    def fuse(
        self,
        observations: tuple[ModalityObservation, ...] | list[ModalityObservation],
        *,
        now: datetime | None = None,
    ) -> FusedState:
        current = now or utc_now()
        # Keep only the freshest observation per modality, and only within window.
        freshest: dict[Modality, ModalityObservation] = {}
        stale: set[Modality] = set()
        for observation in observations:
            age = observation.age_seconds(current)
            if age < -1.0 or age > self.window_seconds:
                stale.add(observation.modality)
                continue
            existing = freshest.get(observation.modality)
            if existing is None or observation.observed_at > existing.observed_at:
                freshest[observation.modality] = observation

        present = frozenset(freshest)
        missing_mandatory = tuple(sorted(m.value for m in self.mandatory - present))
        stale_only = tuple(sorted(m.value for m in stale if m not in present))

        total_modalities = len(Modality)
        completeness = round(len(present) / total_modalities, 6)
        provenance_coverage = 1.0 if present else 0.0

        reasons: list[str] = []
        decision_ready = not missing_mandatory
        if missing_mandatory:
            reasons.append("MANDATORY_MODALITY_ABSENT")
        if stale_only:
            reasons.append("MODALITY_OBSERVATIONS_STALE")
        if decision_ready:
            reasons.append("MANDATORY_MODALITIES_PRESENT_AND_FRESH")

        return FusedState(
            produced_at=current,
            window_seconds=self.window_seconds,
            decision_ready=decision_ready,
            present_modalities=tuple(sorted(m.value for m in present)),
            stale_modalities=stale_only,
            missing_mandatory=missing_mandatory,
            provenance_coverage=provenance_coverage,
            completeness=completeness,
            observations=tuple(freshest[m] for m in sorted(present, key=lambda item: item.value)),
            reasons=tuple(reasons),
        )
