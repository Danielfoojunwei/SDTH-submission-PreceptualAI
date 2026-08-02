"""Turn the audit ledger into a training corpus.

The learning guardrails in ``learning.py`` are ready for a predictive model that
had no way to come into existence: the audit ledger accumulated decision records
every cycle, and nothing turned them into a dataset. This module is that path.

Each exported example pairs the features available *at decision time* with an
outcome measured from the **following** cycle, so a model trained on it learns
to predict rather than to describe. The pairing is what makes the corpus usable:
a record that has no successor cycle within the horizon is dropped rather than
labelled from nothing.

Two properties are enforced:

**No leakage.** Features come only from the record being labelled. The outcome
comes only from later records. The exporter never reads a field from the future
into the feature vector.

**Verified provenance.** Export refuses to run against a ledger whose hash chain
does not verify, so a tampered ledger cannot silently become training data —
NIST AI 100-2e2025 names poisoning as a primary attack class, and the ledger is
the obvious place to attack.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from .audit import AuditLedger
from .models import parse_timestamp

#: Feature names exported for every example, in a stable order. A model's signed
#: feature schema is expected to hash this list.
FEATURE_NAMES: tuple[str, ...] = (
    "rtt_ms",
    "loss_percent",
    "jitter_ms",
    "throughput_mbps",
    "signal_percent",
    "confidence",
    "candidate_count",
    "eligible_count",
    "contested_band_count",
    "clock_trusted",
    "power_duty_limit",
    "is_current_profile",
)


class DatasetError(RuntimeError):
    """Raised when a corpus cannot be exported honestly."""


@dataclass(frozen=True)
class Example:
    """One (features, outcome) pair drawn from consecutive decision records."""

    node_id: str
    decided_at: datetime
    selected_profile_id: str | None
    features: dict[str, float]
    outcome_rtt_ms: float | None
    outcome_loss_percent: float | None
    outcome_link_up: bool
    label_improved: bool | None
    horizon_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "decided_at": self.decided_at.isoformat(),
            "selected_profile_id": self.selected_profile_id,
            "features": {name: self.features.get(name, 0.0) for name in FEATURE_NAMES},
            "outcome": {
                "rtt_ms": self.outcome_rtt_ms,
                "loss_percent": self.outcome_loss_percent,
                "link_up": self.outcome_link_up,
            },
            "label_improved": self.label_improved,
            "horizon_seconds": self.horizon_seconds,
        }


@dataclass
class ExportStats:
    records_read: int = 0
    examples: int = 0
    dropped_no_successor: int = 0
    dropped_incomplete: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_read": self.records_read,
            "examples": self.examples,
            "dropped_no_successor": self.dropped_no_successor,
            "dropped_incomplete": self.dropped_incomplete,
            "reasons": self.reasons,
        }


def _selected_observation(payload: dict[str, Any], profile_id: str | None) -> dict[str, Any] | None:
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict):
        return None
    observations = telemetry.get("observations")
    if not isinstance(observations, list):
        return None
    if profile_id is None:
        return None
    return next(
        (item for item in observations if isinstance(item, dict) and item.get("profile_id") == profile_id),
        None,
    )


def _features_from(payload: dict[str, Any]) -> tuple[str | None, dict[str, float]] | None:
    """Extract decision-time features. Returns None when evidence is incomplete."""
    decision = payload.get("decision")
    if not isinstance(decision, dict):
        return None
    recommendation = decision.get("recommendation")
    if not isinstance(recommendation, dict):
        return None
    selected = recommendation.get("selected_profile_id")
    observation = _selected_observation(payload, selected)
    if observation is None:
        return None
    if observation.get("rtt_ms") is None or observation.get("loss_percent") is None:
        return None

    scores = recommendation.get("scores")
    scores = scores if isinstance(scores, list) else []
    capabilities = payload.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    clock = capabilities.get("clock") if isinstance(capabilities.get("clock"), dict) else {}
    power = capabilities.get("power") if isinstance(capabilities.get("power"), dict) else {}
    recommendation_power = power.get("recommendation") if isinstance(power.get("recommendation"), dict) else {}
    degraded = capabilities.get("degraded_band_ids")

    telemetry = payload.get("telemetry") or {}
    current_profile = telemetry.get("current_profile_id") if isinstance(telemetry, dict) else None

    features = {
        "rtt_ms": float(observation["rtt_ms"]),
        "loss_percent": float(observation["loss_percent"]),
        "jitter_ms": float(observation.get("jitter_ms") or 0.0),
        "throughput_mbps": float(observation.get("rx_mbps") or 0.0) + float(observation.get("tx_mbps") or 0.0),
        "signal_percent": float(observation.get("signal_percent") or 0.0),
        "confidence": float(recommendation.get("confidence") or 0.0),
        "candidate_count": float(len(scores)),
        "eligible_count": float(sum(1 for s in scores if isinstance(s, dict) and float(s.get("utility", -1)) >= 0.0)),
        "contested_band_count": float(len(degraded) if isinstance(degraded, list) else 0),
        "clock_trusted": 1.0 if clock.get("trusted") else 0.0,
        "power_duty_limit": float(recommendation_power.get("max_inference_duty") or 1.0),
        "is_current_profile": 1.0 if selected is not None and selected == current_profile else 0.0,
    }
    return selected, features


def _outcome_from(payload: dict[str, Any], profile_id: str | None) -> tuple[float | None, float | None, bool] | None:
    observation = _selected_observation(payload, profile_id)
    if observation is None:
        return None
    return (
        float(observation["rtt_ms"]) if observation.get("rtt_ms") is not None else None,
        float(observation["loss_percent"]) if observation.get("loss_percent") is not None else None,
        bool(observation.get("link_up")),
    )


class CorpusExporter:
    """Builds a labelled corpus from a verified audit ledger."""

    def __init__(self, ledger_path: Path, *, horizon_seconds: float = 60.0) -> None:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        self.ledger_path = ledger_path
        self.horizon_seconds = horizon_seconds

    def _decision_records(self) -> Iterator[tuple[datetime, str, dict[str, Any]]]:
        connection = sqlite3.connect(self.ledger_path, timeout=30.0)
        try:
            rows = connection.execute(
                """
                SELECT created_at, node_id, payload_json
                FROM audit_records
                WHERE record_type = 'decision'
                ORDER BY id ASC
                """
            ).fetchall()
        finally:
            connection.close()
        for created_at, node_id, payload_json in rows:
            try:
                payload = json.loads(payload_json)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield parse_timestamp(str(created_at)), str(node_id), payload

    def export(self) -> tuple[list[Example], ExportStats]:
        stats = ExportStats()
        # A corpus is only as trustworthy as the ledger it came from.
        ledger = AuditLedger(self.ledger_path)
        try:
            verification = ledger.verify()
        finally:
            ledger.close()
        if not verification.valid:
            raise DatasetError(
                f"refusing to export from an unverified audit chain: {verification.error}"
            )

        records = list(self._decision_records())
        stats.records_read = len(records)
        horizon = timedelta(seconds=self.horizon_seconds)
        examples: list[Example] = []

        for index, (created_at, node_id, payload) in enumerate(records):
            extracted = _features_from(payload)
            if extracted is None:
                stats.dropped_incomplete += 1
                continue
            selected, features = extracted

            successor: tuple[float | None, float | None, bool] | None = None
            successor_at: datetime | None = None
            for later_at, _later_node, later_payload in records[index + 1 :]:
                if later_at - created_at > horizon:
                    break
                outcome = _outcome_from(later_payload, selected)
                if outcome is not None:
                    successor = outcome
                    successor_at = later_at
                    break
            if successor is None or successor_at is None:
                stats.dropped_no_successor += 1
                continue

            outcome_rtt, outcome_loss, outcome_up = successor
            # "Improved" compares like with like: the same profile's measured
            # latency and loss before and after. Missing either side means no
            # label rather than an assumed one.
            if outcome_rtt is None or outcome_loss is None:
                label: bool | None = None
            else:
                label = (outcome_rtt <= features["rtt_ms"]) and (outcome_loss <= features["loss_percent"])

            examples.append(
                Example(
                    node_id=node_id,
                    decided_at=created_at,
                    selected_profile_id=selected,
                    features=features,
                    outcome_rtt_ms=outcome_rtt,
                    outcome_loss_percent=outcome_loss,
                    outcome_link_up=outcome_up,
                    label_improved=label,
                    horizon_seconds=(successor_at - created_at).total_seconds(),
                )
            )

        stats.examples = len(examples)
        if not examples:
            stats.reasons.append("NO_LABELLED_EXAMPLES_AVAILABLE")
        if stats.dropped_incomplete:
            stats.reasons.append("RECORDS_WITHOUT_COMPLETE_DECISION_EVIDENCE_DROPPED")
        if stats.dropped_no_successor:
            stats.reasons.append("RECORDS_WITHOUT_AN_OUTCOME_WITHIN_HORIZON_DROPPED")
        return examples, stats


def write_jsonl(examples: list[Example], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
