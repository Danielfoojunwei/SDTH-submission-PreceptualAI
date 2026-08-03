"""The capabilities the autonomous loop needs, and nothing else.

``service.py`` used to reach every capability through one class, and that class
imported the whole package. A node whose job is choosing a radio link therefore
loaded cohort clustering, federated aggregation, differential privacy, emitter
localisation, mobility prediction and threshold cryptography before it made its
first decision — 44 modules to run a loop that calls five methods.

This is the split. The loop imports this module; the operator tools import
:mod:`kval_node.capabilities`, which extends it. Nothing was deleted and no
behaviour changed: the same methods run the same way, and the only difference is
what a flying node has to load to reach them.

The boundary is worth keeping because it is the one an auditor cares about.
Everything reachable from here runs unattended on a platform in flight. Anything
past it runs when an operator asks, on a machine where a person is watching.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit import AuditLedger
from .c2link import (
    C2Budget,
    C2LinkMonitor,
    DirectionalHealth,
    InteroperabilityDeclaration,
)
from .config import NodeConfig
from .dataset import FEATURE_NAMES, CorpusExporter, DatasetError
from .emcon import EmconOrder, assess_emcon
from .forecast import forecast_advice
from .lease import LeaseAction, LeaseError, LeaseNonceStore, redeem_lease, verify_lease
from .models import TelemetrySnapshot, parse_timestamp, utc_now
from .power import PowerGovernor, PowerPolicy, PowerReading, parse_tegrastats
from .pqc import LMS_ALGORITHM, LmsPublicKey
from .signing import (
    SignaturePolicy,
    load_lms_public_key,
    load_public_key,
    load_verified_catalog,
    verify_document_agile,
)
from .spectrum import (
    ScanRateExceeded,
    ScanRateLimiter,
    SpectrumAnalyzer,
    SpectrumObservation,
    SpectrumPolicy,
    SpectrumSensorClient,
)

SPECTRUM_DOCUMENT_TYPE = "kval-spectrum-policy/v1"
POWER_DOCUMENT_TYPE = "kval-power-policy/v1"
EMCON_DOCUMENT_TYPE = "kval-emcon-order/v1"
C2_DOCUMENT_TYPE = "kval-c2-budget/v1"

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


class FlightServices:
    """Capabilities that run unattended, in the loop, on the platform."""

    def __init__(self, config: NodeConfig) -> None:
        self.config = config

    def _verify_document(self, path: Path, *, expected_type: str) -> tuple[dict[str, Any], str]:
        """Verify a signed document under whichever approved algorithm signed it.

        Every loader used to call ``load_verified_document``, which hardcodes
        Ed25519. That made the whole LMS half of ``pqc.py`` unreachable from
        production and made ``require_cnsa_2_0`` a display field — a node with
        the flag set accepted classical signatures without comment, while the
        docs claimed it refused them.

        Both anchors are offered when configured, so a fleet can migrate
        document by document rather than in one flag day.
        """
        envelope = json.loads(path.read_text(encoding="utf-8"))
        ed25519_key = None
        lms_key = None
        try:
            ed25519_key = load_public_key(self.config.catalog_public_key_path)
        except Exception:  # noqa: BLE001 - an LMS-only node has no PEM anchor
            pass
        if self.config.lms_public_key_path is not None:
            lms_key = load_lms_public_key(self.config.lms_public_key_path)
        return verify_document_agile(
            envelope,
            expected_type=expected_type,
            ed25519_key=ed25519_key,
            lms_key=lms_key,
            policy=SignaturePolicy(require_cnsa_2_0=self.config.require_cnsa_2_0),
        )


    def scan_state_path(self) -> Path:
        """Durable store for spectrum scan-rate accounting."""
        if self.config.spectrum_scan_state_path is not None:
            return self.config.spectrum_scan_state_path
        return self.config.audit_database_path.parent / "spectrum-scans.db"

    # --- spectrum -------------------------------------------------------------


    def load_spectrum_policy(self) -> tuple[SpectrumPolicy, str]:
        if self.config.spectrum_policy_path is None:
            raise FileNotFoundError("spectrum_policy_path is not configured")
        payload, payload_hash = self._verify_document(self.config.spectrum_policy_path, expected_type=SPECTRUM_DOCUMENT_TYPE)
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
        # Enforce the policy's scan duty limits before touching the adapter.
        # These are emission-duty and energy controls, so they are checked
        # against durable state rather than a per-process counter.
        limiter = ScanRateLimiter(self.scan_state_path())
        try:
            limiter.check(policy, now=now)
        except ScanRateExceeded as exc:
            limiter.close()
            return {
                "available": False,
                "reason": str(exc),
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
            }
        try:
            observations = SpectrumSensorClient().scan(self.config.spectrum_sensor_socket, policy)
        except Exception as exc:  # fail closed on any adapter error
            # Deliberately *not* recorded against the duty budget: an attempt
            # that never reached the adapter emitted nothing, so charging it
            # would let a brief sensor outage spend the whole hourly quota and
            # lock out sensing long after the adapter recovered.
            limiter.close()
            return {
                "available": False,
                "reason": f"spectrum sensor unavailable: {exc}",
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
            }
        # The scan happened, so now it counts.
        limiter.record(policy_id=policy.policy_id, now=now)
        limiter.close()
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
        payload, payload_hash = self._verify_document(self.config.power_policy_path, expected_type=POWER_DOCUMENT_TYPE)
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


    def forecast_links(self, *, horizon_multiplier: int = 24) -> dict[str, Any]:
        """Forecast each link's near-term behaviour from recorded history.

        The result is an instruction the optimizer honours: a profile forecast
        to degrade is demoted. It can never be promoted on a forecast, so a
        wrong prediction costs a ranking place rather than connectivity.
        """
        try:
            examples, _stats = CorpusExporter(
                self.config.audit_database_path,
                horizon_seconds=self.config.observation_max_age_seconds * horizon_multiplier,
            ).export()
        except DatasetError as exc:
            return {"available": False, "reason": str(exc)}

        history: dict[str, list[float]] = {}
        for example in examples:
            # ``Example.features`` is already a mapping and the profile field is
            # ``selected_profile_id``. Getting either wrong is silent while the
            # corpus is empty and fatal the moment it is not, which is how this
            # shipped: the loop body never ran in any test.
            rtt = example.features.get("rtt_ms")
            profile_id = example.selected_profile_id
            if rtt is None or profile_id is None:
                # A cycle that selected nothing has no flow to attribute the
                # reading to. Grouping those under a null key would forecast a
                # profile that does not exist.
                continue
            history.setdefault(profile_id, []).append(float(rtt))

        advice = forecast_advice(history)
        result = advice.to_dict()
        result["available"] = True
        result["flows"] = len(history)
        result["examples"] = len(examples)
        return result

    # --- emitter tracking ------------------------------------------------------


    def load_emcon_order(self) -> EmconOrder | None:
        """Read the standing emission-control order, if the operator signed one.

        Signed, because an unauthenticated EMCON order is a denial-of-service
        primitive: anyone who can assert "you are silent" suppresses this
        platform's lost-link response for as long as they like.
        """
        if self.config.emcon_order_path is None:
            return None
        payload, _hash = self._verify_document(
            self.config.emcon_order_path, expected_type=EMCON_DOCUMENT_TYPE
        )
        return EmconOrder.from_dict(payload)

    def _emcon_or_none(self) -> EmconOrder | None:
        try:
            return self.load_emcon_order()
        except Exception:
            return None

    def emcon_posture(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Where this platform sits on the emission-control spectrum."""
        try:
            order = self.load_emcon_order()
        except Exception as exc:
            # An unreadable order is not an order. Treating it as silence would
            # let a corrupted file suppress the lost-link contingency.
            return {
                "available": False,
                "reason": f"emission control order invalid: {exc}",
                "silence_commanded": False,
            }
        result = assess_emcon(order, now=now).to_dict()
        result["available"] = True
        return result


    def _catalog_profiles(self):
        try:
            catalog = load_verified_catalog(
                self.config.catalog_path, self.config.catalog_public_key_path
            )
        except Exception:  # noqa: BLE001 - an unverifiable catalogue declares nothing
            return ()
        return catalog.profiles

    def _c2_monitor(self, budget: C2Budget) -> C2LinkMonitor:
        # One monitor per process: the contingency timers are the mechanism,
        # and a fresh monitor each cycle would never accumulate elapsed time.
        existing = getattr(self, "_c2_monitor_instance", None)
        if existing is None or existing.budget != budget:
            existing = C2LinkMonitor(budget)
            self._c2_monitor_instance = existing
        return existing

    def c2_status(
        self,
        *,
        command_path: DirectionalHealth,
        return_path: DirectionalHealth,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Assess the command link and declare the pre-planned contingency.

        The emission posture is consulted first, and only ever to explain a
        dark *return* path. A missing command path is a lost link whatever the
        EMCON state, because nothing about a silence order stops an operator
        transmitting to the vehicle.
        """
        if self.config.c2_budget_path is None:
            return _unavailable(
                "no C2 budget is provisioned; a command link with no declared budget "
                "and no pre-planned contingency cannot be assessed"
            )
        try:
            payload, budget_hash = self._verify_document(
                self.config.c2_budget_path, expected_type=C2_DOCUMENT_TYPE
            )
            budget = C2Budget.from_dict(payload)
        except Exception as exc:
            return {"available": False, "reason": f"C2 budget invalid: {exc}"}

        posture = assess_emcon(self._emcon_or_none(), now=now)
        monitor = self._c2_monitor(budget)
        assessment = monitor.observe(
            command_path=command_path,
            return_path=return_path,
            now=now or utc_now(),
            emission_silence_commanded=posture.silence_commanded(),
        )
        result = assessment.to_dict()
        result["available"] = True
        result["budget"] = budget.to_dict()
        result["budget_hash"] = budget_hash
        result["emcon"] = posture.to_dict()
        result["interoperability"] = InteroperabilityDeclaration().to_dict()
        return result

    def c2_status_from_telemetry(
        self, snapshot: TelemetrySnapshot, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Assess the command link from the telemetry the node already collects.

        ``c2_status`` took two ``DirectionalHealth`` values that **nothing in
        production ever built**, so the whole C2 monitor — budget floor, loss
        confirmation, pre-planned contingency — was unreachable. This is the
        adapter that was missing.

        The two directions are read from different evidence on purpose. The
        *command* path is the probe result for the profile carrying C2: if the
        node can reach its control endpoint, commands can arrive. The *return*
        path is whether the node's own reports are being accepted, which is the
        half emission control is allowed to silence.
        """
        current = now or utc_now()
        c2_profiles = {
            profile.profile_id
            for profile in self._catalog_profiles()
            if profile.link_role == "c2_cnpc"
        }
        observation = next(
            (item for item in snapshot.observations if item.profile_id in c2_profiles), None
        )
        if observation is None:
            return _unavailable(
                "no observation for any profile declared as c2_cnpc in the signed catalogue"
            )
        health = DirectionalHealth(
            present=observation.link_up and observation.connection_active,
            latency_ms=observation.rtt_ms,
            loss_percent=observation.loss_percent,
            last_seen=observation.observed_at,
        )
        return self.c2_status(command_path=health, return_path=health, now=current)


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
