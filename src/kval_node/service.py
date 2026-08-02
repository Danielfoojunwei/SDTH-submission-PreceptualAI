from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .audit import AuditLedger
from .capabilities import CapabilityServices
from .config import NodeConfig
from .executor import NetworkManagerExecutor
from .hardware import PlatformReport, verify_jetson_platform
from .models import (
    LinkObservation,
    Recommendation,
    ScoredProfile,
    ShieldAction,
    TelemetrySnapshot,
    utc_now,
)
from .multimodal import Modality, ModalityObservation, MultimodalFuser
from .optimizer import ProfileOptimizer
from .policy import SafetyShield
from .recovery import LinkStateMachine, RecoveryAction
from .signing import load_private_key, load_verified_catalog
from .spectrum import BandStatus
from .telemetry import TelemetryCollector
from .timesync import ClockMonitor

#: Provenance marker for observations Kval derived itself rather than received
#: from a signed external adapter. It is a real digest of the deriving module's
#: identity, so an auditor can tell locally-derived features from adapter ones.
_LOCAL_PROVENANCE = hashlib.sha256(b"kval-node/service/local-derivation/v1").hexdigest()


def _optional_node_key(config: NodeConfig) -> Ed25519PrivateKey | None:
    """Load the node's signing key, or run unsigned rather than not at all.

    A node without a fabric key is a valid deployment — a standalone node with
    no peers. It loses authorship binding on its audit records, and the memory
    audit reports that it is unavailable rather than passing vacuously.
    """
    if config.fabric_node_key_path is None:
        return None
    try:
        return load_private_key(config.fabric_node_key_path)
    except (OSError, ValueError, TypeError):
        return None


class KvalNode:
    def __init__(self, config: NodeConfig) -> None:
        self.config = config
        self.platform: PlatformReport = verify_jetson_platform()
        self.catalog = load_verified_catalog(config.catalog_path, config.catalog_public_key_path)
        # The node key binds each audit record to its author. Without one the
        # ledger still records and still hash-chains; it just cannot answer
        # whether history was *appended* to, only whether it was rewritten.
        self.audit = AuditLedger(
            config.audit_database_path, signing_key=_optional_node_key(config)
        )
        self.collector = TelemetryCollector()
        self.optimizer = ProfileOptimizer()
        self.shield = SafetyShield(
            self.catalog,
            self.audit,
            observation_max_age_seconds=config.observation_max_age_seconds,
            minimum_confidence=config.minimum_confidence,
        )
        self.executor = NetworkManagerExecutor()
        # Capability loops. Each is optional and fails closed when its signed
        # policy or evidence is absent, so a node configured exactly as it was
        # before this change behaves exactly as it did before this change.
        self.capabilities = CapabilityServices(config)
        self.clock = ClockMonitor()
        self.fuser = MultimodalFuser(window_seconds=config.observation_max_age_seconds)
        self._link_machines: dict[str, LinkStateMachine] = {}
        # Last contested-band evidence, carried across cycles where sensing was
        # unavailable so a refused scan cannot look like an all-clear.
        self._last_degraded_bands: frozenset[str] = frozenset()
        self._degraded_until: datetime | None = None
        self._stop = False

    def c2_status(self) -> dict[str, object]:
        """Assess the command link using this process's long-lived monitor.

        Must run inside ``KvalNode``: the CLI builds a fresh
        ``CapabilityServices`` per invocation, so a command-shaped C2 check
        would reset the contingency timers to zero every call and could never
        declare a contingency at all.
        """
        snapshot = self.collector.collect(self.catalog)
        return self.capabilities.c2_status_from_telemetry(snapshot)

    def close(self) -> None:
        self.audit.close()

    # --- capability loops ----------------------------------------------------

    def _machine_for(self, profile_id: str) -> LinkStateMachine:
        machine = self._link_machines.get(profile_id)
        if machine is None:
            machine = LinkStateMachine()
            self._link_machines[profile_id] = machine
        return machine

    def _run_recovery(self, snapshot: TelemetrySnapshot) -> dict[str, object]:
        """Advance the link-health state machine for the profile in use."""
        current_id = snapshot.current_profile_id
        if current_id is None:
            return {"available": False, "reason": "no current profile to track"}
        observation = next(
            (item for item in snapshot.observations if item.profile_id == current_id), None
        )
        if observation is None:
            return {"available": False, "reason": "no observation for the current profile"}
        decision = self._machine_for(current_id).observe(observation, now=snapshot.observed_at)
        return {"available": True, "profile_id": current_id, **decision.to_dict()}

    def _guarded(self, name: str, call, **kwargs) -> dict[str, object]:
        """Run a capability so that its failure degrades the node, not kills it.

        A capability that raises used to take the whole daemon down for good:
        ``forecast_links`` referenced a field that does not exist, which was
        invisible while the corpus was empty and fatal the moment it was not —
        and because the ledger is durable, every restart died in the same place.
        A resilient-communications node must lose the capability, not the loop.

        The failure is recorded to the audit ledger rather than swallowed, so a
        silently-degraded node is still a visibly-degraded node.
        """
        try:
            return call(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a capability must never kill the loop
            detail = {
                "available": False,
                "reason": f"capability raised {type(exc).__name__}: {exc}",
                "capability": name,
            }
            try:
                self.audit.append(self.config.node_id, "capability_failure", detail)
            except Exception:  # noqa: BLE001 - never let auditing a fault raise a fault
                pass
            return detail

    @staticmethod
    def _suppressed_roles(emcon: dict[str, object]) -> frozenset[str]:
        """Link roles the standing emission-control order forbids.

        An unavailable posture suppresses nothing. Inventing suppression from
        a missing or unreadable order would silence a platform nobody ordered
        silent, which is a denial of service the node would inflict on itself.
        """
        if emcon.get("available") is not True:
            return frozenset()
        roles = emcon.get("suppressed_roles")
        if not isinstance(roles, list):
            return frozenset()
        return frozenset(str(item) for item in roles)

    @staticmethod
    def _forecast_degrading(forecast: dict[str, object]) -> frozenset[str]:
        """Profiles the forecast expects to worsen, if forecasting ran at all."""
        if forecast.get("available") is not True:
            return frozenset()
        profiles = forecast.get("degrading_profiles")
        if not isinstance(profiles, list):
            return frozenset()
        return frozenset(str(item) for item in profiles)

    def _degraded_band_ids(self, spectrum: dict[str, object]) -> frozenset[str]:
        """Bands the local sensor reports as contested, if sensing is available.

        When sensing is unavailable this returns the *last* contested set rather
        than an empty one. "Not allowed to scan yet" is not "nothing is
        contested", and treating them alike made the node re-select a link it
        had just judged jammed: the rate limiter refuses a scan, the evidence
        vanishes, and the contested profile's penalty lifts. With a 2 s poll
        against a 5 s minimum scan interval — the shipped example config —
        that happened on two cycles in every three.

        The retained set is bounded by the same observation-age rule the
        analyzer uses, so stale evidence expires rather than pinning a profile
        down forever.
        """
        if spectrum.get("available") is not True:
            if self._degraded_until is not None and utc_now() < self._degraded_until:
                return self._last_degraded_bands
            return frozenset()
        contested = {BandStatus.JAMMING_SUSPECTED.value, BandStatus.ANOMALOUS.value}
        bands = spectrum.get("bands")
        if not isinstance(bands, list):
            return frozenset()
        current = frozenset(
            str(band.get("band_id"))
            for band in bands
            if isinstance(band, dict)
            and band.get("status") in contested
            and float(band.get("confidence") or 0.0) > 0.0
        )
        self._last_degraded_bands = current
        self._degraded_until = utc_now() + timedelta(
            seconds=self.config.observation_max_age_seconds
        )
        return current

    def _fuse_state(
        self,
        snapshot: TelemetrySnapshot,
        spectrum: dict[str, object],
        power: dict[str, object],
        clock: dict[str, object],
    ) -> dict[str, object]:
        """Bind this cycle's modalities into one provenance-tagged state."""
        observations: list[ModalityObservation] = []

        best: LinkObservation | None = None
        for item in snapshot.observations:
            if item.confidence() > 0.0 and (best is None or item.confidence() > best.confidence()):
                best = item
        if best is not None:
            observations.append(
                ModalityObservation.create(
                    Modality.NETWORK,
                    source_id=f"telemetry:{best.profile_id}",
                    observed_at=best.observed_at,
                    features={
                        "rtt_ms": best.rtt_ms or 0.0,
                        "loss_percent": best.loss_percent or 0.0,
                        "jitter_ms": best.jitter_ms or 0.0,
                    },
                    provenance_hash=_LOCAL_PROVENANCE,
                    quality=best.confidence(),
                )
            )

        if power.get("available") is True:
            reading = power.get("reading")
            recommendation = power.get("recommendation")
            if isinstance(reading, dict) and isinstance(recommendation, dict):
                features = {"max_inference_duty": float(recommendation.get("max_inference_duty") or 0.0)}
                for key in ("cpu_temp_c", "gpu_temp_c", "junction_temp_c", "module_watts"):
                    value = reading.get(key)
                    if value is not None:
                        features[key] = float(value)
                observations.append(
                    ModalityObservation.create(
                        Modality.POWER,
                        source_id=str(power.get("policy_id", "power")),
                        observed_at=snapshot.observed_at,
                        features=features,
                        provenance_hash=str(power.get("policy_hash") or _LOCAL_PROVENANCE),
                    )
                )

        if spectrum.get("available") is True:
            bands = spectrum.get("bands")
            band_list = bands if isinstance(bands, list) else []
            observations.append(
                ModalityObservation.create(
                    Modality.SPECTRUM,
                    source_id=str(spectrum.get("policy_id", "spectrum")),
                    observed_at=snapshot.observed_at,
                    features={
                        "contested_bands": float(len(self._degraded_band_ids(spectrum))),
                        "clear_comms_bands": float(len(spectrum.get("clear_comms_band_ids") or [])),
                        "assessed_bands": float(len(band_list)),
                    },
                    provenance_hash=str(spectrum.get("policy_hash") or _LOCAL_PROVENANCE),
                )
            )

        # Clock integrity is the security modality: an undisciplined clock is
        # exactly the state in which freshness checks stop meaning anything.
        observations.append(
            ModalityObservation.create(
                Modality.SECURITY,
                source_id="clock-integrity",
                observed_at=snapshot.observed_at,
                features={
                    "clock_trusted": 1.0 if clock.get("trusted") else 0.0,
                    "observed_step_seconds": float(clock.get("observed_step_seconds") or 0.0),
                },
                provenance_hash=_LOCAL_PROVENANCE,
                quality=1.0 if clock.get("trusted") else 0.0,
            )
        )

        return self.fuser.fuse(observations, now=snapshot.observed_at).to_dict()

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
        capabilities: dict[str, object] | None = None,
    ) -> dict[str, object]:
        decision = self.shield.evaluate(recommendation, snapshot, apply_requested=apply_requested)
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
            "capabilities": capabilities or {},
        }
        record_hash = self.audit.append(self.config.node_id, "decision", payload)
        payload["audit_record_hash"] = record_hash
        self._write_latest(payload)
        return payload

    def run_once(self) -> dict[str, object]:
        """One full cycle: observe -> sense -> govern -> fuse -> decide -> audit.

        Each capability loop is optional and independently fail-closed. When a
        capability is unconfigured it reports ``available: false`` with a reason
        and contributes nothing to the decision, so the loop degrades to the
        deterministic connectivity behaviour rather than stalling.
        """
        snapshot = self.collector.collect(self.catalog)

        # Clock integrity gates everything that compares timestamps, so it runs
        # first and its verdict travels with the cycle.
        clock = self.clock.check(now=snapshot.observed_at).to_dict()

        spectrum = self._guarded("spectrum", self.capabilities.sense_spectrum, now=snapshot.observed_at)
        power = self._guarded("power", self.capabilities.plan_power)
        recovery = self._run_recovery(snapshot)

        # Emission control decides which link roles may be used at all, so it
        # is resolved before ranking rather than reported alongside it. This
        # was previously CLI-only: a signed SILENT order sat on disk while the
        # loop went on selecting a payload link.
        emcon = self._guarded("emcon", self.capabilities.emcon_posture, now=snapshot.observed_at)
        suppressed = self._suppressed_roles(emcon)

        degraded_bands = self._degraded_band_ids(spectrum)
        # A per-flow forecast over this node's own recorded history. It may
        # demote a link it expects to worsen, and can never promote one.
        forecast = self._guarded("forecast", self.capabilities.forecast_links)
        forecast_degrading = self._forecast_degrading(forecast)
        recommendation = self.optimizer.recommend(
            self.catalog,
            snapshot,
            self.config.service_class,
            degraded_band_ids=degraded_bands,
            forecast_degrading_profiles=forecast_degrading,
            role=self.config.link_role,
            suppressed_roles=suppressed,
        )

        fused = self._fuse_state(snapshot, spectrum, power, clock)
        capabilities: dict[str, object] = {
            "clock": clock,
            "spectrum": spectrum,
            "power": power,
            "recovery": recovery,
            "state": fused,
            "forecast": forecast,
            "emcon": emcon,
            "link_role": self.config.link_role,
            "suppressed_roles": sorted(suppressed),
            "degraded_band_ids": sorted(degraded_bands),
            "forecast_degrading_profiles": sorted(forecast_degrading),
        }

        # A recovery escalation is an operator matter, not something the node
        # resolves by switching again; record it in its own audit entry so it is
        # visible without reading every decision record.
        if recovery.get("action") == RecoveryAction.ESCALATE.value:
            self.audit.append(self.config.node_id, "escalation", recovery)

        payload = self._finish_cycle(
            snapshot,
            recommendation,
            apply_requested=self.config.apply_network_changes,
            capabilities=capabilities,
        )
        # An activation is a link switch; the state machine must know so its
        # flap ceiling reflects reality.
        activation = payload.get("activation")
        if isinstance(activation, dict) and activation.get("success") is True:
            self._machine_for(str(activation.get("profile_id"))).record_switch(snapshot.observed_at)
        return payload

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
