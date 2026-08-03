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

import base64
import json
import sqlite3
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from .analysis import OutcomeAnalyzer
from .audit import AuditLedger
from .config import NodeConfig
from .intrusion import LocalDetector, NodeBaseline
from .flight import (
    C2_DOCUMENT_TYPE,
    FLOW_DOCUMENT_TYPE,
    EMCON_DOCUMENT_TYPE,
    POWER_DOCUMENT_TYPE,
    FlightServices,
    _unavailable,
)
from .dataset import FEATURE_NAMES, CorpusExporter, DatasetError
from .roster import PeerRoster, RosterError
from .federated import FederatedUpdate, RobustAggregator, sign_update, verify_update
from .c2link import (
    C2Budget,
    C2LinkMonitor,
    DirectionalHealth,
    InteroperabilityDeclaration,
)
from .emcon import EmconOrder, assess_emcon
from .learning import ModelTrustError, verify_model_manifest
from .lease import LeaseAction, LeaseError, LeaseNonceStore, redeem_lease, verify_lease
from .memory_integrity import MemoryAuditor, SignedRecord
from .models import TelemetrySnapshot, parse_timestamp, utc_now
from .cohorts import assess_heterogeneity
from .orchestration import (
    FleetLearning,
    cohort_assignments,
    forecast_advice,
    privatise_update,
    renew_under_partition,
)
from .power import PowerGovernor, PowerPolicy, PowerReading, parse_tegrastats
from .pqc import LMS_ALGORITHM, LmsPublicKey
from .signing import (
    CNSA_2_0_SIGNATURE_ALGORITHMS,
    ED25519_ALGORITHM,
    SUPPORTED_SIGNATURE_ALGORITHMS,
    SignaturePolicy,
    load_lms_public_key,
    load_private_key,
    load_public_key,
    load_verified_catalog,
    verify_document_agile,
)
from .threshold_ca import Attestation, CertificationError, CertifierRoster, Grant
from .timesync import ClockMonitor
from .transport_security import TransportPolicy, TransportVerifier

TRANSPORT_DOCUMENT_TYPE = "kval-transport-policy/v1"
ROSTER_DOCUMENT_TYPE = "kval-peer-roster/v1"
#: Signed document types for the uncrewed-platform controls.

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




def _anchor_algorithm(path: Path) -> tuple[str | None, str]:
    """Identify a trust anchor by parsing it, never by its filename.

    The earlier version inferred the algorithm from ``path.suffix == ".lms"``,
    which is forgeable in both directions and grades a file it never opens:
    copying an Ed25519 PEM to ``decoy.lms`` produced
    ``EVERY_CONFIGURED_TRUST_ANCHOR_USES_A_CNSA_2_0_APPROVED_ALGORITHM`` on a
    wholly classical node, and a real LMS key named ``.pub`` was reported as
    retired. A compliance report that can be changed with ``mv`` is worse than
    no report, because it is believed.

    An anchor the node cannot load is reported as **unusable**, not as
    approved — the previous behaviour would call a node compliant while it
    could not verify its own catalogue.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"ANCHOR_IS_UNREADABLE:{type(exc).__name__}"
    try:
        LmsPublicKey.from_bytes(base64.b64decode(raw.strip(), validate=True))
        return LMS_ALGORITHM, "APPROVED_FOR_NATIONAL_SECURITY_SYSTEMS"
    except Exception:  # noqa: BLE001 - not an LMS anchor; try the classical one
        pass
    try:
        load_public_key(path)
        return ED25519_ALGORITHM, "CLASSICAL_ASYMMETRIC_RETIRED_BY_CNSA_2_0"
    except Exception:  # noqa: BLE001 - neither format parses
        return None, "ANCHOR_PARSES_AS_NEITHER_LMS_NOR_ED25519_AND_CANNOT_BE_USED"


class CapabilityServices(FlightServices):
    """Everything an operator can ask for, on top of what the loop runs."""

    def load_transport_policy(self) -> tuple[TransportPolicy, str]:
        if self.config.transport_policy_path is None:
            raise FileNotFoundError("transport_policy_path is not configured")
        payload, payload_hash = self._verify_document(self.config.transport_policy_path, expected_type=TRANSPORT_DOCUMENT_TYPE)
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
        payload, payload_hash = self._verify_document(self.config.fabric_roster_path, expected_type=ROSTER_DOCUMENT_TYPE)
        return PeerRoster.from_dict(payload), payload_hash






    def _optional_node_key(self):
        if self.config.fabric_node_key_path is None:
            return None
        try:
            return load_private_key(self.config.fabric_node_key_path)
        except (OSError, ValueError, TypeError):
            return None


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
        except (RosterError, OSError, ValueError) as exc:
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
            except (RosterError, json.JSONDecodeError, OSError) as exc:
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
        # An empty inbox is not a successful aggregation. Reporting available
        # with zero cohorts exited 0, so a systemd timer or the HIL script read
        # "no peer ever contributed" as a healthy fleet.
        result["available"] = bool(updates)
        if not updates:
            result["reason"] = "no verified peer contributions to aggregate"
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
            "available": certificate.was_present,
            "reason": (
                None if certificate.was_present
                else f"{node_id!r} is not a contributor; nothing was excised"
            ),
            "certificate": certificate.to_dict(),
            "remaining_contributors": sorted(update.node_id for update in remaining),
            "reputation": learning.reputation.scores(),
            "rejected": rejected,
        }

    # --- prediction that reaches a decision -----------------------------------

    def _recorded_spectrum_observations(self) -> tuple[list[dict[str, Any]], str | None]:
        """Every spectrum observation this node has recorded, oldest first.

        Read from the decision records rather than re-sensed, because the
        question is what changed over time and a fresh scan is one sample. A
        cycle where sensing was unavailable recorded no observations, so gaps
        are absent rather than filled — which is what lets the discriminator
        refuse on a gap instead of interpolating across an outage.
        """
        ledger = AuditLedger(self.config.audit_database_path)
        try:
            rows = [
                row
                for row in ledger.signed_records(limit=MEMORY_AUDIT_RECORDS)
                if row["record_type"] == "decision"
            ]
        except (sqlite3.Error, KeyError, ValueError) as exc:
            return [], f"audit ledger unreadable: {exc}"
        finally:
            ledger.close()

        observations: list[dict[str, Any]] = []
        for row in rows:
            spectrum = (row["payload"] or {}).get("spectrum") or {}
            if spectrum.get("available") is not True:
                continue
            observations.extend(spectrum.get("observations", []))
        return observations, None



    def assess_fleet_heterogeneity(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Report how different the fleet's radio environments actually are.

        Cohorts form whether or not they mean anything. This says whether the
        grouping separated environments or split one environment by noise, and
        reports the regime on an absolute scale so "heavy" means the same thing
        next week and in another deployment.
        """
        if self.config.fabric_roster_path is None or self.config.federated_inbox_path is None:
            return _unavailable("heterogeneity assessment needs the roster and a federated inbox")
        try:
            roster, roster_hash = self.load_peer_roster()
        except Exception as exc:
            return {"available": False, "reason": f"peer roster invalid: {exc}"}

        updates, rejected = self._load_updates(roster, now=now)
        if not updates:
            # An empty inbox is not a homogeneous fleet. Reporting a regime here
            # would describe a fleet that never contributed anything.
            return {
                "available": False,
                "reason": "no verified peer contributions to assess",
                "roster_id": roster.roster_id,
                "rejected": rejected,
            }
        assessment = assess_heterogeneity(updates)
        return {
            "available": True,
            "roster_id": roster.roster_id,
            "roster_hash": roster_hash,
            "rejected": rejected,
            "assessment": assessment.to_dict(),
        }


    # --- fleet-level detection --------------------------------------------------


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

    # --- uncrewed-platform command link -------------------------------------------




    def assess_link_anomaly(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Is this node's link behaviour a departure from its own normal?

        The per-node half of the ICC 2024 shape: each node judges its own
        readings against its own history, because what counts as anomalous is a
        property of where that node is sitting. It used to be fed band statuses
        and noise floors; it is fed link measurements now, which is the evidence
        that survived — and which is what a flow actually rides on.

        Abstains rather than reassures when the baseline is too thin. A detector
        with nothing to compare against reporting "not anomalous" is a fail-open,
        and this one says it abstained instead.
        """
        try:
            exporter = CorpusExporter(self.config.audit_database_path, node_id=self.config.node_id)
            examples = exporter.examples()
        except Exception as exc:
            return {"available": False, "reason": f"corpus unavailable: {exc}"}
        if len(examples) < 2:
            return _unavailable("too little history to judge this node against itself")

        history = [dict(example.features) for example in examples[:-1]]
        latest = dict(examples[-1].features)
        baseline = NodeBaseline.from_observations(self.config.node_id, history)
        verdict = LocalDetector(baseline).judge(
            latest, cohort_id=self.config.link_role, observed_at=now or utc_now()
        )
        result = verdict.to_dict()
        result["available"] = True
        result["baseline_samples"] = baseline.samples
        return result

    def renew_grant_from_files(
        self, grant_path: Path, attestation_paths: Sequence[Path], *, extend_seconds: float
    ) -> dict[str, Any]:
        """Drive quorum renewal from operator artefacts on disk."""
        try:
            grant = Grant.from_dict(json.loads(grant_path.read_text(encoding="utf-8")))
            attestations = [
                Attestation.from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in attestation_paths
            ]
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            return {"available": False, "reason": f"grant or attestation unreadable: {exc}"}
        return self.renew_grant(
            grant,
            attestations,
            new_expiry=grant.expires_at + timedelta(seconds=extend_seconds),
        )




    def _band_frequency(self, band_id: str) -> tuple[int, int, tuple[str, ...]]:
        """Centre frequency and bandwidth from the signed spectrum policy.

        The band plan is operator-signed and is the only authoritative source
        for what a band ID means in hertz. Without it the report says zero *and
        says why*, rather than presenting zero as a measurement.
        """
        try:
            policy, _hash = self.load_spectrum_policy()
        except Exception as exc:
            return 0, 0, (f"NO_SIGNED_BAND_PLAN_SO_FREQUENCY_IS_UNKNOWN:{exc}",)
        band = next((item for item in policy.bands if item.band_id == band_id), None)
        if band is None:
            return 0, 0, (f"BAND_{band_id}_IS_NOT_IN_THE_SIGNED_BAND_PLAN",)
        centre = (band.freq_low_hz + band.freq_high_hz) // 2
        return centre, band.freq_high_hz - band.freq_low_hz, ()


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
            algorithm, detail = _anchor_algorithm(path)
            anchors.append(
                {
                    "anchor": name,
                    "purpose": purpose,
                    "path": str(path),
                    "algorithm": algorithm,
                    "cnsa_2_0_approved": (
                        algorithm in CNSA_2_0_SIGNATURE_ALGORITHMS
                        if algorithm is not None
                        else None
                    ),
                    "reason": detail,
                }
            )

        record("catalogue_and_policy", self.config.catalog_public_key_path,
               "authorises every profile and policy the node may act on")
        record("capability_lease", self.config.lease_public_key_path,
               "authorises bounded, single-use actions")
        record("fabric_node", self.config.fabric_node_key_path,
               "binds this node's reports and audit records to it")

        classical = [a for a in anchors if a["cnsa_2_0_approved"] is False]
        unusable = [
            a for a in anchors
            if a["algorithm"] is None and a["reason"] != "NOT_CONFIGURED_ON_THIS_NODE"
        ]
        reasons: list[str] = []
        acquisition_gate = date(2027, 1, 1)
        signing_deadline = date(2030, 12, 31)
        days_to_gate = (acquisition_gate - current).days

        if unusable:
            reasons.append("SOME_TRUST_ANCHORS_CANNOT_BE_PARSED_AND_THE_NODE_CANNOT_USE_THEM")
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
            "unusable_anchors": [a["anchor"] for a in unusable],
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
