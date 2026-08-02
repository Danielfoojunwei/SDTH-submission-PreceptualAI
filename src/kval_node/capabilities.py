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

import json
import sqlite3
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .analysis import OutcomeAnalyzer
from .audit import AuditLedger
from .config import NodeConfig
from .dataset import FEATURE_NAMES, CorpusExporter, DatasetError
from .federated import FederatedUpdate, RobustAggregator, sign_update, verify_update
from .geolocate import (
    EmitterLocator,
    GeolocationError,
    NodePosition,
    PathLossModel,
    SignalSighting,
)
from .fabric import (
    FabricAggregator,
    FabricError,
    FabricReport,
    PeerRoster,
    sign_report,
    verify_report,
)
from .evidence import BeliefMass
from .intrusion import NodeBaseline
from .learning import ModelTrustError, verify_model_manifest
from .lease import LeaseAction, LeaseError, LeaseNonceStore, redeem_lease, verify_lease
from .memory_integrity import MemoryAuditor, SignedRecord
from .mobility import DEFAULT_PLACE_RADIUS_M
from .models import parse_timestamp, utc_now
from .orchestration import (
    FleetLearning,
    cohort_assignments,
    detect_fleet_event,
    fixes_from_estimates,
    forecast_advice,
    predict_track,
    privatise_update,
    renew_under_partition,
    verdicts_from_reports,
)
from .power import PowerGovernor, PowerPolicy, PowerReading, parse_tegrastats
from .pqc import LMS_ALGORITHM
from .signing import (
    CNSA_2_0_SIGNATURE_ALGORITHMS,
    ED25519_ALGORITHM,
    SUPPORTED_SIGNATURE_ALGORITHMS,
    load_private_key,
    load_public_key,
    load_verified_document,
)
from .threshold_ca import Attestation, CertificationError, CertifierRoster, Grant
from .timesync import ClockMonitor
from .topology import EvidencePropagator, FleetGraph, assess_fragility
from .spectrum import (
    ScanRateExceeded,
    ScanRateLimiter,
    SpectrumAnalyzer,
    SpectrumObservation,
    SpectrumPolicy,
    SpectrumSensorClient,
)
from .transport_security import TransportPolicy, TransportVerifier

SPECTRUM_DOCUMENT_TYPE = "kval-spectrum-policy/v1"
POWER_DOCUMENT_TYPE = "kval-power-policy/v1"
TRANSPORT_DOCUMENT_TYPE = "kval-transport-policy/v1"
ROSTER_DOCUMENT_TYPE = "kval-peer-roster/v1"
PATH_LOSS_DOCUMENT_TYPE = "kval-path-loss-model/v1"

#: Record type under which localisation results are written to the ledger, so
#: the mobility model has a history of fixes to read back.
EMITTER_RECORD_TYPE = "emitter_estimate"

#: Records the memory audit examines. Bounded so the O(n^2) near-duplicate scan
#: stays cheap on a node; the burst and duplicate patterns it looks for are
#: local in time, so a recent window is where they show.
MEMORY_AUDIT_RECORDS = 500

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

    def scan_state_path(self) -> Path:
        """Durable store for spectrum scan-rate accounting."""
        if self.config.spectrum_scan_state_path is not None:
            return self.config.spectrum_scan_state_path
        return self.config.audit_database_path.parent / "spectrum-scans.db"

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
        # Enforce the policy's scan duty limits before touching the adapter.
        # These are emission-duty and energy controls, so they are checked
        # against durable state rather than a per-process counter.
        limiter = ScanRateLimiter(self.scan_state_path())
        try:
            limiter.check_and_record(policy, now=now)
        except ScanRateExceeded as exc:
            return {
                "available": False,
                "reason": str(exc),
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
            }
        finally:
            limiter.close()
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

    # --- outcome analysis and the model contract ------------------------------

    def analyze_outcomes(self, *, window: int = 8, now: datetime | None = None) -> dict[str, Any]:
        """Descriptive statistics over recorded decisions. Needs no model."""
        return OutcomeAnalyzer(
            self.config.audit_database_path,
            horizon_seconds=self.config.observation_max_age_seconds * 8,
        ).analyze(window=window, now=now)

    def verify_model(self) -> dict[str, Any]:
        """Verify a signed model manifest, if one is configured.

        This is the point at which a predictive model would be admitted. With no
        manifest configured the learned path is simply absent and the node stays
        on the deterministic optimizer; with one configured, every element of
        the contract must verify or the model is refused. Either way the node
        never runs an unverified model.
        """
        if self.config.model_manifest_path is None:
            return {
                "available": False,
                "reason": "no model manifest configured; learned path abstains",
                "learned_path": "ABSTAIN",
            }
        try:
            envelope = json.loads(self.config.model_manifest_path.read_text(encoding="utf-8"))
            manifest = verify_model_manifest(
                envelope, load_public_key(self.config.catalog_public_key_path)
            )
        except (ModelTrustError, json.JSONDecodeError, OSError) as exc:
            return {
                "available": True,
                "verified": False,
                "learned_path": "ABSTAIN",
                "error": str(exc),
            }
        return {
            "available": True,
            "verified": True,
            # A verified manifest proves provenance, not fitness. Promotion
            # still requires evaluation evidence outside this node.
            "learned_path": "ADVISORY_UNTIL_PROMOTED",
            "manifest": manifest.canonical(),
        }

    # --- cross-node fabric ----------------------------------------------------

    def load_peer_roster(self) -> tuple[PeerRoster, str]:
        if self.config.fabric_roster_path is None:
            raise FileNotFoundError("fabric_roster_path is not configured")
        payload, payload_hash = load_verified_document(
            self.config.fabric_roster_path,
            self.config.catalog_public_key_path,
            expected_type=ROSTER_DOCUMENT_TYPE,
        )
        return PeerRoster.from_dict(payload), payload_hash

    def publish_report(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Sense locally and emit this node's signed report for its peers."""
        if self.config.fabric_node_key_path is None:
            return _unavailable("fabric publishing is not configured (no node key)")
        spectrum = self.sense_spectrum(now=now)
        if spectrum.get("available") is not True:
            return {"available": False, "reason": f"cannot report: {spectrum.get('reason')}"}
        try:
            private_key = load_private_key(self.config.fabric_node_key_path)
        except Exception as exc:
            return {"available": False, "reason": f"fabric node key unusable: {exc}"}

        clock = ClockMonitor().check(now=now)
        policy, policy_hash = self.load_spectrum_policy()
        intelligence = SpectrumAnalyzer(
            max_observation_age_seconds=self.config.observation_max_age_seconds
        ).analyze(
            policy,
            [SpectrumObservation.from_dict(item) for item in spectrum.get("observations", [])],
            now=now,
        )
        report = FabricReport.from_intelligence(
            intelligence,
            node_id=self.config.node_id,
            policy_hash=policy_hash,
            clock_trusted=clock.trusted,
        )
        return {
            "available": True,
            "clock": clock.to_dict(),
            "report": sign_report(report, private_key),
        }

    def aggregate_fleet(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Verify peer reports from the inbox and fuse them into a fleet view."""
        if self.config.fabric_roster_path is None or self.config.fabric_inbox_path is None:
            return _unavailable("fabric aggregation is not configured (roster and inbox required)")
        try:
            roster, roster_hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        inbox = self.config.fabric_inbox_path
        candidates = sorted(inbox.glob("*.json")) if inbox.is_dir() else []
        verified: list[FabricReport] = []
        rejected: list[dict[str, str]] = []
        for path in candidates:
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                verified.append(verify_report(envelope, roster, now=now))
            except (FabricError, json.JSONDecodeError, OSError) as exc:
                # A bad report is dropped with a reason; it never degrades the
                # view of the peers that did report correctly.
                rejected.append({"source": path.name, "reason": str(exc)})

        view = FabricAggregator().aggregate(
            verified,
            expected_nodes=frozenset(peer.node_id for peer in roster.peers) - {self.config.node_id},
            now=now,
        )
        result = view.to_dict()
        result["available"] = True
        result["roster_id"] = roster.roster_id
        result["roster_hash"] = roster_hash
        result["rejected_reports"] = rejected
        return result

    def _verified_reports(self, roster: PeerRoster, *, now: datetime | None) -> tuple[list[FabricReport], list[dict[str, str]]]:
        inbox = self.config.fabric_inbox_path
        verified: list[FabricReport] = []
        rejected: list[dict[str, str]] = []
        for path in sorted(inbox.glob("*.json")) if inbox and inbox.is_dir() else []:
            try:
                verified.append(verify_report(json.loads(path.read_text(encoding="utf-8")), roster, now=now))
            except (FabricError, json.JSONDecodeError, OSError) as exc:
                rejected.append({"source": path.name, "reason": str(exc)})
        return verified, rejected

    def locate_emitter(self, band_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        """Estimate an emitter's position in one band from peer reports."""
        if self.config.fabric_roster_path is None or self.config.fabric_inbox_path is None:
            return _unavailable("emitter localisation needs the fabric roster and inbox")
        try:
            roster, _hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        reports, rejected = self._verified_reports(roster, now=now)
        positions = [
            NodePosition(peer.node_id, peer.latitude_deg, peer.longitude_deg, peer.altitude_m)
            for peer in roster.peers
            if peer.has_position()
        ]
        sightings: list[SignalSighting] = []
        for report in reports:
            band = next((item for item in report.bands if item.band_id == band_id), None)
            if band is None or band.peak_power_dbm is None or band.confidence <= 0.0:
                continue
            sightings.append(SignalSighting(report.node_id, band.peak_power_dbm, report.clock_trusted))

        path_loss = None
        if self.config.path_loss_model_path is not None:
            try:
                payload, _ = load_verified_document(
                    self.config.path_loss_model_path,
                    self.config.catalog_public_key_path,
                    expected_type=PATH_LOSS_DOCUMENT_TYPE,
                )
                path_loss = PathLossModel.from_dict(payload)
            except Exception as exc:
                return {"available": False, "reason": f"path-loss model invalid: {exc}"}

        try:
            estimate = EmitterLocator().locate(sightings, positions, path_loss=path_loss)
        except GeolocationError as exc:
            return {
                "available": False,
                "reason": str(exc),
                "band_id": band_id,
                "sightings": len(sightings),
                "positioned_nodes": len(positions),
                "rejected_reports": rejected,
            }
        result = estimate.to_dict()
        result["available"] = True
        result["band_id"] = band_id
        result["rejected_reports"] = rejected
        self._record_fix(result)
        return result

    def _record_fix(self, estimate: dict[str, Any]) -> None:
        """Persist a successful estimate so a track can be built from fixes.

        Only accepted estimates are written. A refusal is not a position, and
        recording one would let the mobility model treat "we could not tell"
        as a place the emitter had been.
        """
        if estimate.get("available") is not True:
            return
        ledger = AuditLedger(
            self.config.audit_database_path, signing_key=self._optional_node_key()
        )
        try:
            ledger.append(
                self.config.node_id,
                EMITTER_RECORD_TYPE,
                {
                    "available": True,
                    "band_id": estimate["band_id"],
                    "latitude_deg": estimate["latitude_deg"],
                    "longitude_deg": estimate["longitude_deg"],
                    "uncertainty_radius_m": estimate["uncertainty_radius_m"],
                    "method": estimate["method"],
                    "contributing_nodes": estimate["contributing_nodes"],
                    "excluded": estimate.get("excluded", []),
                },
            )
        except sqlite3.Error:
            # A ledger that cannot be written is a recording failure, not a
            # localisation failure: the estimate the caller asked for is still
            # correct and is still returned.
            pass
        finally:
            ledger.close()

    def _optional_node_key(self):
        if self.config.fabric_node_key_path is None:
            return None
        try:
            return load_private_key(self.config.fabric_node_key_path)
        except (OSError, ValueError, TypeError):
            return None

    def fabric_topology(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Report where the fabric would break, and what each node can see.

        Two answers from one graph. **Fragility** names the nodes whose loss
        partitions the fleet, which is the question worth asking before a
        capture rather than after one. **Propagation** gives every node a view
        of every band, including nodes with no working radio of their own — the
        eyes-on-the-ground case the fabric exists for.
        """
        if self.config.fabric_roster_path is None or self.config.fabric_inbox_path is None:
            return _unavailable("fabric topology needs the roster and inbox")
        try:
            roster, roster_hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        positioned = [
            NodePosition(peer.node_id, peer.latitude_deg, peer.longitude_deg, peer.altitude_m)
            for peer in roster.peers
            if peer.has_position()
        ]
        if len(positioned) < 2:
            return _unavailable(
                "fabric topology needs at least two peers with declared positions in the roster"
            )
        graph = FleetGraph.from_positions(positioned, link_range_m=self.config.fabric_link_range_m)

        reports, rejected = self._verified_reports(roster, now=now)
        band_ids = sorted({band.band_id for report in reports for band in report.bands})
        propagator = EvidencePropagator(graph)
        bands: dict[str, Any] = {}
        for band_id in band_ids:
            observations: dict[str, BeliefMass] = {}
            for report in reports:
                if report.node_id not in graph.nodes:
                    continue
                band = next((item for item in report.bands if item.band_id == band_id), None)
                if band is None:
                    continue
                observations[report.node_id] = BeliefMass.from_status(band.status, band.confidence)
            bands[band_id] = [
                belief.to_dict() for belief in propagator.propagate(observations)
            ]

        return {
            "available": True,
            "roster_id": roster.roster_id,
            "roster_hash": roster_hash,
            "graph": graph.to_dict(),
            "fragility": assess_fragility(graph).to_dict(),
            "bands": bands,
            "rejected_reports": rejected,
        }

    def audit_memory(self) -> dict[str, Any]:
        """Check that recorded history is this node's own and was not injected.

        The hash chain already proves the sequence has not been *modified*.
        This asks the question the chain cannot: whether records were
        *appended* that the node never wrote, or written in the burst-and-
        near-duplicate pattern that memory poisoning produces.
        """
        if self.config.fabric_node_key_path is None:
            return _unavailable("memory audit needs the node key to check authorship")
        try:
            private_key = load_private_key(self.config.fabric_node_key_path)
        except (OSError, ValueError, TypeError) as exc:
            return {"available": False, "reason": f"node key unusable: {exc}"}

        ledger = AuditLedger(self.config.audit_database_path)
        try:
            chain = ledger.verify()
            records = [
                SignedRecord(
                    node_id=str(row["node_id"]),
                    recorded_at=parse_timestamp(str(row["recorded_at"])),
                    record_type=str(row["record_type"]),
                    payload=row["payload"],
                    signature=str(row["signature"]),
                    key_id=str(row["key_id"]),
                )
                for row in ledger.signed_records(limit=MEMORY_AUDIT_RECORDS)
            ]
        except (sqlite3.Error, KeyError, ValueError) as exc:
            return {"available": False, "reason": f"audit ledger unreadable: {exc}"}
        finally:
            ledger.close()

        report = MemoryAuditor().audit(
            records, private_key.public_key(), node_id=self.config.node_id
        )
        result = report.to_dict()
        result["available"] = True
        # Both answers matter and they are different questions: the chain says
        # history was not rewritten, the audit says it was not padded.
        result["chain_intact"] = chain.valid
        result["chain_records"] = chain.records
        return result

    # --- federated statistics -------------------------------------------------

    def federated_update(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Produce this node's signed contribution: statistics, never raw data.

        When the operator has declared a privacy budget, the statistics are
        perturbed **before** they leave the node, so the aggregator never sees
        this node's exact means — which would encode where this sensor is and
        what it can hear.
        """
        if self.config.fabric_node_key_path is None:
            return _unavailable("federated contribution needs a fabric node key")
        try:
            examples, _stats = CorpusExporter(
                self.config.audit_database_path,
                horizon_seconds=self.config.observation_max_age_seconds * 8,
            ).export()
        except DatasetError as exc:
            return {"available": False, "reason": str(exc)}
        try:
            update = FederatedUpdate.from_examples(examples, node_id=self.config.node_id, now=now)
            private_key = load_private_key(self.config.fabric_node_key_path)
        except (FabricError, OSError, ValueError) as exc:
            return {"available": False, "reason": str(exc)}

        released, disclosure = privatise_update(
            update, epsilon=self.config.federated_privacy_epsilon
        )
        return {
            "available": True,
            "update": sign_update(released, private_key),
            "privacy": disclosure.to_dict(),
        }

    def _reputation_path(self) -> Path:
        return self.config.audit_database_path.parent / "fleet-reputation.json"

    def _load_updates(
        self, roster: PeerRoster, *, now: datetime | None
    ) -> tuple[list[FederatedUpdate], list[dict[str, str]]]:
        inbox = self.config.federated_inbox_path
        updates: list[FederatedUpdate] = []
        rejected: list[dict[str, str]] = []
        for path in sorted(inbox.glob("*.json")) if inbox and inbox.is_dir() else []:
            try:
                updates.append(
                    verify_update(json.loads(path.read_text(encoding="utf-8")), roster, now=now)
                )
            except (FabricError, json.JSONDecodeError, OSError) as exc:
                rejected.append({"source": path.name, "reason": str(exc)})
        return updates, rejected

    def federated_aggregate(self, *, now: datetime | None = None, round_index: int = 0) -> dict[str, Any]:
        """Combine peer contributions through the full federated pipeline.

        Not a flat average of everything received. Contributions are clustered
        by RF environment, aggregated within cohort, combined across cohorts so
        a large cohort cannot outvote a small one, and weighted by reputation
        that persists between rounds — with the aggregating node rotating so
        none of them is a permanent target.
        """
        if self.config.fabric_roster_path is None or self.config.federated_inbox_path is None:
            return _unavailable("federated aggregation needs the roster and a federated inbox")
        try:
            roster, roster_hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        updates, rejected = self._load_updates(roster, now=now)
        learning = FleetLearning(self._reputation_path())
        result = learning.aggregate(updates, round_index=round_index)
        # The flat aggregate is retained alongside it: a hierarchical result
        # that differs sharply from the flat one is informative, and hiding the
        # comparison would make the hierarchy unfalsifiable.
        result["flat_baseline"] = RobustAggregator().aggregate(updates).to_dict()
        result["available"] = True
        result["roster_id"] = roster.roster_id
        result["roster_hash"] = roster_hash
        result["rejected"] = rejected
        return result

    def excise_node(self, node_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        """Remove a contributor and certify that nothing of it remains."""
        if self.config.fabric_roster_path is None or self.config.federated_inbox_path is None:
            return _unavailable("excision needs the roster and a federated inbox")
        try:
            roster, _hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        updates, rejected = self._load_updates(roster, now=now)
        learning = FleetLearning(self._reputation_path())
        remaining, certificate = learning.excise(updates, node_id)
        return {
            "available": True,
            "certificate": certificate.to_dict(),
            "remaining_contributors": sorted(update.node_id for update in remaining),
            "reputation": learning.reputation.scores(),
            "rejected": rejected,
        }

    # --- prediction that reaches a decision -----------------------------------

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
            features = dict(zip(FEATURE_NAMES, example.features))
            rtt = features.get("rtt_ms")
            if rtt is None:
                continue
            history.setdefault(example.profile_id, []).append(float(rtt))

        advice = forecast_advice(history)
        result = advice.to_dict()
        result["available"] = True
        result["flows"] = len(history)
        return result

    # --- emitter tracking ------------------------------------------------------

    def track_emitter(
        self, band_id: str, *, horizon_seconds: float = 600.0, now: datetime | None = None
    ) -> dict[str, Any]:
        """Predict where an emitter goes next, from its recorded fixes.

        Reads the localisation results this node has already recorded. A
        refused estimate never entered history, so a refusal cannot become a
        track point.
        """
        ledger = AuditLedger(self.config.audit_database_path)
        try:
            records = [
                row
                for row in ledger.signed_records(limit=MEMORY_AUDIT_RECORDS)
                if row["record_type"] == EMITTER_RECORD_TYPE
                and (row["payload"] or {}).get("band_id") == band_id
            ]
        except (sqlite3.Error, KeyError, ValueError) as exc:
            return {"available": False, "reason": f"audit ledger unreadable: {exc}"}
        finally:
            ledger.close()

        fixes = fixes_from_estimates(records)
        if not fixes:
            return _unavailable(
                f"no recorded emitter fixes for band {band_id!r}; run locate-emitter first"
            )
        # Tie the place radius to the accuracy that produced the fixes rather
        # than guessing it: a tighter radius turns noise into a tour of places
        # the emitter never visited.
        radius = max(
            DEFAULT_PLACE_RADIUS_M,
            max(fix.uncertainty_radius_m for fix in fixes) * 2.0,
        )
        result = predict_track(
            fixes, horizon=timedelta(seconds=horizon_seconds), place_radius_m=radius
        )
        result["available"] = True
        result["band_id"] = band_id
        return result

    # --- fleet-level detection --------------------------------------------------

    def detect_fleet_anomaly(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Ask whether several environments changed together beyond local cause.

        Composes three mechanisms because none of them answers the question
        alone: per-node detectors decide what is anomalous *for that node*,
        cohort membership decides whether their agreement is meaningful, and
        reputation decides whether the agreeing nodes carry any weight.
        """
        if self.config.fabric_roster_path is None or self.config.fabric_inbox_path is None:
            return _unavailable("fleet detection needs the roster and inbox")
        try:
            roster, _hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        reports, rejected = self._verified_reports(roster, now=now)
        if not reports:
            return _unavailable("no verified peer reports to assess")

        updates, _rejected_updates = self._load_updates(roster, now=now)
        cohorts = cohort_assignments(updates)
        baselines = {
            update.node_id: NodeBaseline(
                node_id=update.node_id,
                samples=update.examples,
                means={name: stat.mean for name, stat in update.statistics.items()},
                stddevs={name: stat.stddev() for name, stat in update.statistics.items()},
            )
            for update in updates
        }
        verdicts = verdicts_from_reports(reports, baselines, cohorts)
        reputation = FleetLearning(self._reputation_path()).reputation.scores()

        result = detect_fleet_event(verdicts, reputation)
        result["available"] = True
        result["cohorts_assigned"] = cohorts
        result["rejected_reports"] = rejected
        return result

    # --- certification under partition -------------------------------------------

    def renew_grant(
        self,
        grant: Grant,
        attestations: Sequence[Attestation],
        *,
        new_expiry: datetime,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Extend an expiring grant when the operator cannot be reached."""
        if self.config.fabric_roster_path is None:
            return _unavailable("quorum renewal needs the operator-signed roster")
        try:
            roster, _hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        certifiers = {
            peer.node_id: peer.public_key
            for peer in roster.peers
            if peer.node_id != self.config.node_id
        }
        if len(certifiers) < self.config.quorum_threshold:
            return _unavailable(
                f"roster has {len(certifiers)} eligible certifiers, "
                f"below the declared quorum of {self.config.quorum_threshold}"
            )
        try:
            certifier_roster = CertifierRoster(certifiers, self.config.quorum_threshold)
        except CertificationError as exc:
            return {"available": False, "reason": str(exc)}
        return renew_under_partition(
            grant, attestations, certifier_roster, new_expiry=new_expiry, now=now
        )

    # --- cryptographic posture --------------------------------------------------

    def crypto_posture(self, *, today: date | None = None) -> dict[str, Any]:
        """Report this node against the CNSA 2.0 transition, with dates.

        Every trust decision here rests on a signature, and the algorithm those
        signatures use is a procurement fact before it is a technical one: from
        **1 January 2027** a new National Security System acquisition must
        support CNSA 2.0 or carry a documented exception, and software and
        firmware signing — which is what a signed policy root is — must use it
        exclusively from **2030**.

        The report names each trust anchor, the algorithm it actually uses, and
        whether that algorithm is approved. It does not grade itself: an
        operator reads which anchors are still classical and how long they have.
        """
        current = today or date.today()
        anchors: list[dict[str, Any]] = []

        def record(name: str, path: Path | None, purpose: str) -> None:
            if path is None:
                anchors.append(
                    {"anchor": name, "purpose": purpose, "algorithm": None,
                     "cnsa_2_0_approved": None, "reason": "NOT_CONFIGURED_ON_THIS_NODE"}
                )
                return
            algorithm = LMS_ALGORITHM if path.suffix == ".lms" else ED25519_ALGORITHM
            anchors.append(
                {
                    "anchor": name,
                    "purpose": purpose,
                    "path": str(path),
                    "algorithm": algorithm,
                    "cnsa_2_0_approved": algorithm in CNSA_2_0_SIGNATURE_ALGORITHMS,
                    "reason": (
                        "APPROVED_FOR_NATIONAL_SECURITY_SYSTEMS"
                        if algorithm in CNSA_2_0_SIGNATURE_ALGORITHMS
                        else "CLASSICAL_ASYMMETRIC_RETIRED_BY_CNSA_2_0"
                    ),
                }
            )

        record("catalogue_and_policy", self.config.catalog_public_key_path,
               "authorises every profile and policy the node may act on")
        record("capability_lease", self.config.lease_public_key_path,
               "authorises bounded, single-use actions")
        record("fabric_node", self.config.fabric_node_key_path,
               "binds this node's reports and audit records to it")

        classical = [a for a in anchors if a["cnsa_2_0_approved"] is False]
        reasons: list[str] = []
        acquisition_gate = date(2027, 1, 1)
        signing_deadline = date(2030, 12, 31)
        days_to_gate = (acquisition_gate - current).days

        if classical:
            reasons.append("SOME_TRUST_ANCHORS_STILL_USE_CLASSICAL_ASYMMETRIC_SIGNATURES")
            if current < acquisition_gate:
                reasons.append(
                    f"{days_to_gate}_DAYS_UNTIL_NEW_NSS_ACQUISITIONS_MUST_SUPPORT_CNSA_2_0"
                )
            else:
                reasons.append("PAST_THE_NSS_ACQUISITION_GATE_A_DOCUMENTED_EXCEPTION_IS_REQUIRED")
        else:
            reasons.append("EVERY_CONFIGURED_TRUST_ANCHOR_USES_A_CNSA_2_0_APPROVED_ALGORITHM")

        # Stated on every report, because the algorithm is the easy half.
        reasons.append("SP_800_208_ALSO_REQUIRES_HARDWARE_STATE_CUSTODY_NOT_MET_BY_A_STATE_FILE")

        return {
            "available": True,
            "schema": "kval-crypto-posture/v1",
            "assessed_on": current.isoformat(),
            "require_cnsa_2_0": self.config.require_cnsa_2_0,
            "anchors": anchors,
            "classical_anchors": [a["anchor"] for a in classical],
            "nss_acquisition_gate": acquisition_gate.isoformat(),
            "signing_exclusive_deadline": signing_deadline.isoformat(),
            "days_to_acquisition_gate": days_to_gate,
            "supported_algorithms": sorted(SUPPORTED_SIGNATURE_ALGORITHMS),
            "cnsa_2_0_algorithms": sorted(CNSA_2_0_SIGNATURE_ALGORITHMS),
            "reasons": reasons,
        }

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
