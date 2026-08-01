from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from .audit import AuditLedger
from .config import NodeConfig
from .executor import NetworkManagerExecutor
from .hardware import PlatformReport, verify_jetson_platform
from .identity import RuntimeContext, load_verified_node_manifest
from .models import (
    Recommendation,
    ScoredProfile,
    ShieldAction,
    TelemetrySnapshot,
    utc_now,
)
from .optimizer import ProfileOptimizer
from .policy import SafetyShield
from .signing import load_verified_catalog
from .telemetry import TelemetryCollector


class KvalNode:
    def __init__(self, config: NodeConfig) -> None:
        self.config = config
        self.platform: PlatformReport = verify_jetson_platform()
        self.identity = load_verified_node_manifest(
            config.node_manifest_path,
            config.node_manifest_public_key_path,
        )
        if self.identity.node_id != config.node_id:
            raise ValueError("configured node_id does not match the signed node manifest")
        self.catalog = load_verified_catalog(config.catalog_path, config.catalog_public_key_path)
        self.audit = AuditLedger(config.audit_database_path)
        self.collector = TelemetryCollector()
        self.optimizer = ProfileOptimizer()
        self.shield = SafetyShield(
            self.catalog,
            self.audit,
            observation_max_age_seconds=config.observation_max_age_seconds,
            minimum_confidence=config.minimum_confidence,
        )
        self.executor = NetworkManagerExecutor()
        self._stop = False

    def close(self) -> None:
        self.audit.close()

    def _write_latest(self, payload: dict[str, object]) -> None:
        destination = self.config.latest_result_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, sort_keys=True, separators=(",", ":"))
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)

    def _finish_cycle(
        self,
        snapshot: TelemetrySnapshot,
        recommendation: Recommendation,
        *,
        apply_requested: bool,
    ) -> dict[str, object]:
        decision = self.shield.evaluate(recommendation, snapshot, apply_requested=apply_requested)
        context = RuntimeContext.build(
            self.identity,
            platform=self.platform.to_dict(),
            catalog_hash=self.catalog.payload_hash,
            snapshot=snapshot,
            service_class=self.config.service_class.value,
            decision_engine="deterministic-baseline-v1",
        )
        activation: dict[str, object] | None = None
        activation_error: str | None = None
        if decision.action is ShieldAction.APPLY_APPROVED_PROFILE:
            profile = self.catalog.by_id()[recommendation.selected_profile_id or ""]
            try:
                result = self.executor.activate(profile, catalog_hash=self.catalog.payload_hash)
                activation = result.to_dict()
                self.audit.append(self.config.node_id, "activation", activation)
            except Exception as exc:
                activation_error = str(exc)
                failure = {
                    "profile_id": profile.profile_id,
                    "connection_uuid": profile.connection_uuid,
                    "interface": profile.interface,
                    "success": False,
                    "error": activation_error,
                }
                self.audit.append(self.config.node_id, "activation", failure)

        payload: dict[str, object] = {
            "schema": "kval-node-result/v1",
            "node_id": self.config.node_id,
            "context": context.to_dict(),
            "platform": self.platform.to_dict(),
            "catalog": {
                "version": self.catalog.version,
                "payload_hash": self.catalog.payload_hash,
                "key_id": self.catalog.key_id,
                "expires_at": self.catalog.expires_at.isoformat(),
            },
            "telemetry": snapshot.to_dict(),
            "decision": decision.to_dict(),
            "activation": activation,
            "activation_error": activation_error,
        }
        record_hash = self.audit.append(self.config.node_id, "decision", payload)
        payload["audit_record_hash"] = record_hash
        self._write_latest(payload)
        return payload

    def run_once(self) -> dict[str, object]:
        snapshot = self.collector.collect(self.catalog)
        recommendation = self.optimizer.recommend(self.catalog, snapshot, self.config.service_class)
        return self._finish_cycle(
            snapshot,
            recommendation,
            apply_requested=self.config.apply_network_changes,
        )

    def activate_for_hil(self, profile_id: str) -> dict[str, object]:
        profile = self.catalog.by_id().get(profile_id)
        if profile is None:
            raise ValueError("requested HIL profile is not in the signed catalogue")
        snapshot = self.collector.collect(self.catalog)
        observation = next((item for item in snapshot.observations if item.profile_id == profile_id), None)
        confidence = observation.confidence() if observation else 0.0
        recommendation = Recommendation(
            recommendation_id=f"hil-{int(time.time_ns())}",
            created_at=utc_now(),
            catalog_version=self.catalog.version,
            selected_profile_id=profile_id,
            current_profile_id=snapshot.current_profile_id,
            service_class=self.config.service_class,
            confidence=confidence,
            scores=(
                ScoredProfile(profile_id, 0.0, confidence, {}, ("OPERATOR_REQUESTED_HIL",)),
            ),
            reason_codes=("OPERATOR_REQUESTED_HIL",),
        )
        return self._finish_cycle(snapshot, recommendation, apply_requested=True)

    def benchmark_optimizer(self, iterations: int) -> dict[str, object]:
        if iterations < 10:
            raise ValueError("benchmark requires at least 10 iterations")
        snapshot = self.collector.collect(self.catalog)
        latencies_ms: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            recommendation = self.optimizer.recommend(self.catalog, snapshot, self.config.service_class)
            self.shield.evaluate(recommendation, snapshot, apply_requested=False)
            latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        ordered = sorted(latencies_ms)

        def percentile(value: float) -> float:
            index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * value))))
            return ordered[index]

        return {
            "device_model": self.platform.model,
            "jetpack_version": self.platform.jetpack_version,
            "iterations": iterations,
            "p50_ms": round(percentile(0.50), 6),
            "p95_ms": round(percentile(0.95), 6),
            "p99_ms": round(percentile(0.99), 6),
            "maximum_ms": round(max(ordered), 6),
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "scope": "optimizer_and_safety_shield_only",
        }

    def run_forever(self) -> None:
        def request_stop(_signum: int, _frame: object) -> None:
            self._stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not self._stop:
            started = time.monotonic()
            self.run_once()
            remaining = self.config.poll_interval_seconds - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
