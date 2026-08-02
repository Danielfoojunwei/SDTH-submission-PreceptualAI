"""Capability orchestration for the deeper node agent.

Loads operator-signed spectrum, power, and transport policies and runs each new
capability fail-closed. Every method returns a structured result whose
``available``/``verified`` flag is ``False`` (with a reason) when signed policy,
a sensor adapter, or live hardware evidence is missing — nothing is fabricated.

These services do not require the Jetson platform gate, so an operator can
verify signed policy on a control laptop. Live evidence (sensor socket,
``tegrastats``, ``wg``, ``nft``) simply reports unavailable off the node.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import NodeConfig
from .lease import LeaseAction, LeaseError, LeaseNonceStore, redeem_lease, verify_lease
from .models import utc_now
from .power import PowerGovernor, PowerPolicy, PowerReading, parse_tegrastats
from .signing import load_public_key, load_verified_document
from .spectrum import SpectrumAnalyzer, SpectrumPolicy, SpectrumSensorClient
from .transport_security import TransportPolicy, TransportVerifier

SPECTRUM_DOCUMENT_TYPE = "kval-spectrum-policy/v1"
POWER_DOCUMENT_TYPE = "kval-power-policy/v1"
TRANSPORT_DOCUMENT_TYPE = "kval-transport-policy/v1"

#: Lease actions governed by the signed power policy.
_POWER_ACTIONS = frozenset(
    {
        LeaseAction.SET_POWER_MODE,
        LeaseAction.SET_INFERENCE_DUTY,
        LeaseAction.SET_SCAN_INTERVAL,
        LeaseAction.SET_TELEMETRY_INTERVAL,
    }
)

#: Sentinel used when a governing policy exists but cannot be loaded. It can
#: never equal a real SHA-256 digest, so the lease comparison always refuses.
_UNRESOLVABLE_POLICY_HASH = "policy-unresolvable"


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


class CapabilityServices:
    def __init__(self, config: NodeConfig) -> None:
        self.config = config

    # --- spectrum -------------------------------------------------------------

    def load_spectrum_policy(self) -> tuple[SpectrumPolicy, str]:
        if self.config.spectrum_policy_path is None:
            raise FileNotFoundError("spectrum_policy_path is not configured")
        payload, payload_hash = load_verified_document(
            self.config.spectrum_policy_path,
            self.config.catalog_public_key_path,
            expected_type=SPECTRUM_DOCUMENT_TYPE,
        )
        return SpectrumPolicy.from_dict(payload), payload_hash

    def sense_spectrum(self, *, now: datetime | None = None) -> dict[str, Any]:
        if self.config.spectrum_policy_path is None:
            return _unavailable("spectrum sensing is not configured on this node")
        try:
            policy, policy_hash = self.load_spectrum_policy()
        except Exception as exc:
            return {"available": False, "reason": f"spectrum policy invalid: {exc}"}
        if self.config.spectrum_sensor_socket is None:
            return {
                "available": False,
                "reason": "no spectrum sensor adapter socket configured",
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
            }
        try:
            observations = SpectrumSensorClient().scan(self.config.spectrum_sensor_socket, policy)
        except Exception as exc:  # fail closed on any adapter error
            return {
                "available": False,
                "reason": f"spectrum sensor unavailable: {exc}",
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
            }
        analyzer = SpectrumAnalyzer(max_observation_age_seconds=self.config.observation_max_age_seconds)
        intelligence = analyzer.analyze(policy, observations, now=now)
        result = intelligence.to_dict()
        result["available"] = True
        result["policy_hash"] = policy_hash
        result["observations"] = [observation.to_dict() for observation in observations]
        return result

    # --- power ----------------------------------------------------------------

    def load_power_policy(self) -> tuple[PowerPolicy, str]:
        if self.config.power_policy_path is None:
            raise FileNotFoundError("power_policy_path is not configured")
        payload, payload_hash = load_verified_document(
            self.config.power_policy_path,
            self.config.catalog_public_key_path,
            expected_type=POWER_DOCUMENT_TYPE,
        )
        return PowerPolicy.from_dict(payload), payload_hash

    def plan_power(
        self,
        *,
        reading: PowerReading | None = None,
        tegrastats_line: str | None = None,
        current_nvpmodel_id: int | None = None,
    ) -> dict[str, Any]:
        if self.config.power_policy_path is None:
            return _unavailable("power governance is not configured on this node")
        try:
            policy, policy_hash = self.load_power_policy()
        except Exception as exc:
            return {"available": False, "reason": f"power policy invalid: {exc}"}
        if reading is None:
            if tegrastats_line is None:
                tegrastats_line = read_tegrastats_sample()
            if tegrastats_line is None:
                return {
                    "available": False,
                    "reason": "no power/thermal reading available (tegrastats not readable)",
                    "policy_id": policy.policy_id,
                    "policy_hash": policy_hash,
                }
            reading = parse_tegrastats(tegrastats_line)
        recommendation = PowerGovernor().recommend(policy, reading, current_nvpmodel_id=current_nvpmodel_id)
        return {
            "available": True,
            "policy_id": policy.policy_id,
            "policy_hash": policy_hash,
            "reading": reading.to_dict(),
            "recommendation": recommendation.to_dict(),
            "lease_bounds_required": recommendation.to_lease_bounds(),
        }

    # --- transport ------------------------------------------------------------

    def load_transport_policy(self) -> tuple[TransportPolicy, str]:
        if self.config.transport_policy_path is None:
            raise FileNotFoundError("transport_policy_path is not configured")
        payload, payload_hash = load_verified_document(
            self.config.transport_policy_path,
            self.config.catalog_public_key_path,
            expected_type=TRANSPORT_DOCUMENT_TYPE,
        )
        return TransportPolicy.from_dict(payload), payload_hash

    def transport_preflight(self) -> dict[str, Any]:
        if self.config.transport_policy_path is None:
            return _unavailable("secure transport is not configured on this node")
        try:
            policy, policy_hash = self.load_transport_policy()
        except Exception as exc:
            return {"available": False, "reason": f"transport policy invalid: {exc}"}
        report = TransportVerifier(policy).preflight(
            wifi_connection_uuid=self.config.transport_wifi_connection_uuid
        )
        report["available"] = True
        report["policy_hash"] = policy_hash
        return report

    # --- leases ---------------------------------------------------------------

    def _active_policy_hash(self, action: LeaseAction) -> str | None:
        """Hash of the signed policy that governs ``action``, if one applies."""
        try:
            if action in _POWER_ACTIONS:
                return self.load_power_policy()[1]
            if action is LeaseAction.ACTIVATE_APPROVED_SPECTRUM_PROFILE:
                return self.load_spectrum_policy()[1]
        except Exception:
            # A policy that cannot be loaded cannot authorise anything; the
            # caller turns this into a refusal rather than an unchecked pass.
            return _UNRESOLVABLE_POLICY_HASH
        return None

    def verify_lease_envelope(
        self,
        envelope: dict[str, Any],
        *,
        expected_action: LeaseAction | None = None,
        redeem: bool = False,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if self.config.lease_public_key_path is None:
            return _unavailable("lease verification is not configured (no lease public key)")
        try:
            public_key = load_public_key(self.config.lease_public_key_path)
        except Exception as exc:  # unreadable/invalid key must fail closed
            return {"available": True, "valid": False, "redeemed": False, "error": f"lease public key unusable: {exc}"}
        current = now or utc_now()
        try:
            if redeem:
                if self.config.lease_nonce_database_path is None:
                    return _unavailable("lease redemption requires a lease_nonce_database_path")
                if expected_action is None:
                    return _unavailable("lease redemption requires an expected action")
                # Bind the lease to the policy actually in force, so a lease
                # naming a superseded policy cannot redeem against a new one.
                expected_policy_hash = self._active_policy_hash(expected_action)
                store = LeaseNonceStore(self.config.lease_nonce_database_path)
                try:
                    lease = redeem_lease(
                        envelope,
                        public_key,
                        store,
                        node_id=self.config.node_id,
                        expected_action=expected_action,
                        expected_policy_hash=expected_policy_hash,
                        now=current,
                    )
                finally:
                    store.close()
            else:
                lease = verify_lease(envelope, public_key, node_id=self.config.node_id, now=current)
        except LeaseError as exc:
            return {"available": True, "valid": False, "redeemed": False, "error": str(exc)}
        return {"available": True, "valid": True, "redeemed": redeem, "lease": lease.to_body()}


def read_tegrastats_sample(*, timeout_seconds: float = 3.0) -> str | None:
    """Read a single ``tegrastats`` sample, or ``None`` if it cannot be read.

    ``tegrastats`` streams forever, so we start it at a 1 s interval, take the
    first complete line, and terminate it. Any failure (missing binary, no
    output, permission) fails closed to ``None``.
    """
    try:
        process = subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return None
    try:
        line = _read_line_with_timeout(process, timeout_seconds)
        return (line.strip() or None) if line else None
    finally:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()


def _read_line_with_timeout(process: "subprocess.Popen[str]", timeout_seconds: float) -> str | None:
    """Read one complete line within a real deadline.

    ``select`` only reports that *some* bytes are ready; a blocking
    ``readline()`` afterwards can still hang on a partial line. Read
    byte-at-a-time against a monotonic deadline so the timeout is genuinely
    enforced.
    """
    import os
    import selectors
    import time

    stream = process.stdout
    if stream is None:
        return None
    deadline = time.monotonic() + timeout_seconds
    selector = selectors.DefaultSelector()
    try:
        selector.register(stream, selectors.EVENT_READ)
    except (ValueError, OSError):
        return None
    buffer: list[str] = []
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if not selector.select(remaining):
                return None
            chunk = os.read(stream.fileno(), 4096).decode("utf-8", errors="replace")
            if not chunk:
                return "".join(buffer) or None
            buffer.append(chunk)
            if "\n" in chunk:
                return "".join(buffer).split("\n", 1)[0]
    finally:
        selector.close()
