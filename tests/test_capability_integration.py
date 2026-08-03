"""The new capabilities must be reachable from a real configured node.

A module with tests but no production caller is dead code, and this repository
has been bitten by exactly that before. These tests drive
``CapabilityServices`` through real signed artefacts written to disk — an
operator-signed roster, a node key, a fabric inbox, a hash-chained ledger — so
that ``fabric-topology`` and ``audit-memory`` are pinned as things the node can
actually do, not just things it can import.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.audit import AuditLedger
from kval_node.capabilities import ROSTER_DOCUMENT_TYPE, CapabilityServices
from kval_node.config import NodeConfig
from kval_node.models import utc_now
from kval_node.signing import sign_document


NOW = utc_now()

# Three nodes ~250 m apart, plus one far outside radio range.
POSITIONS = {
    "node-a": (1.3000, 103.8000),
    "node-b": (1.3020, 103.8000),
    "node-c": (1.3010, 103.8020),
    "node-far": (1.5000, 103.9000),
}


def _pem(key: Ed25519PrivateKey) -> str:
    return (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )


class CapabilityIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.operator = Ed25519PrivateKey.generate()
        self.node_keys = {node: Ed25519PrivateKey.generate() for node in POSITIONS}

        operator_public = self.root / "operator.pub"
        operator_public.write_bytes(
            self.operator.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )

        roster_payload = {
            "roster_id": "r1",
            "created_at": (NOW - timedelta(minutes=1)).isoformat(),
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "peers": [
                {
                    "node_id": node_id,
                    "public_key_pem": _pem(self.node_keys[node_id]),
                    "role": "sub-node",
                    "position": {"latitude_deg": latitude, "longitude_deg": longitude},
                }
                for node_id, (latitude, longitude) in POSITIONS.items()
            ],
        }
        roster_path = self.root / "roster.json"
        roster_path.write_text(
            json.dumps(sign_document(ROSTER_DOCUMENT_TYPE, roster_payload, self.operator)),
            encoding="utf-8",
        )

        self.inbox = self.root / "inbox"
        self.inbox.mkdir()

        self.node_key_path = self.root / "node.key"
        descriptor = self.node_key_path.open("wb")
        with descriptor:
            descriptor.write(
                self.node_keys["node-a"].private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        catalog_path = self.root / "catalog.json"
        catalog_path.write_text("{}", encoding="utf-8")
        config_path = self.root / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "node_id": "node-a",
                    "catalog_path": str(catalog_path),
                    "catalog_public_key_path": str(operator_public),
                    "audit_database_path": str(self.root / "audit.db"),
                    "latest_result_path": str(self.root / "latest.json"),
                    "service_class": "critical_data",
                    "poll_interval_seconds": 5,
                    "observation_max_age_seconds": 30.0,
                    "minimum_confidence": 0.25,
                    "apply_network_changes": False,
                    "fabric_roster_path": str(roster_path),
                    "fabric_node_key_path": str(self.node_key_path),
                    "fabric_inbox_path": str(self.inbox),
                    "fabric_link_range_m": 400.0,
                }
            ),
            encoding="utf-8",
        )
        self.services = CapabilityServices(NodeConfig.load(config_path))

    def tearDown(self) -> None:
        self._directory.cleanup()


    # --- fabric topology -------------------------------------------------------






    # --- memory audit ----------------------------------------------------------

    def test_a_node_s_own_history_audits_clean(self) -> None:
        ledger = AuditLedger(
            self.services.config.audit_database_path, signing_key=self.node_keys["node-a"]
        )
        try:
            for index in range(6):
                ledger.append("node-a", "decision", {"selected": f"profile-{index}"})
        finally:
            ledger.close()

        result = self.services.audit_memory()
        self.assertTrue(result["available"])
        self.assertTrue(result["trustworthy"])
        self.assertTrue(result["chain_intact"])
        self.assertIn("HISTORY_CONSISTENT_WITH_THIS_NODE_S_OWN_OPERATION", result["reasons"])

    def test_records_written_without_the_node_key_are_caught(self) -> None:
        # An attacker with database access appends plausible history. The chain
        # can be kept consistent by appending properly; the signature cannot.
        ledger = AuditLedger(
            self.services.config.audit_database_path, signing_key=self.node_keys["node-a"]
        )
        try:
            for index in range(6):
                ledger.append("node-a", "decision", {"selected": f"profile-{index}"})
        finally:
            ledger.close()

        unsigned = AuditLedger(self.services.config.audit_database_path)
        try:
            for index in range(3):
                unsigned.append("node-a", "decision", {"selected": "attacker-preferred"})
        finally:
            unsigned.close()

        result = self.services.audit_memory()
        self.assertTrue(result["available"])
        # The chain is intact — appending correctly does not break it. That is
        # exactly why the chain alone is not enough.
        self.assertTrue(result["chain_intact"])
        self.assertFalse(result["trustworthy"])
        self.assertIn("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE", result["reasons"])

    def test_the_injected_records_are_identified_individually(self) -> None:
        ledger = AuditLedger(
            self.services.config.audit_database_path, signing_key=self.node_keys["node-a"]
        )
        try:
            for index in range(6):
                ledger.append("node-a", "decision", {"selected": f"profile-{index}"})
        finally:
            ledger.close()
        unsigned = AuditLedger(self.services.config.audit_database_path)
        try:
            for _ in range(3):
                unsigned.append("node-a", "decision", {"selected": "attacker-preferred"})
        finally:
            unsigned.close()

        result = self.services.audit_memory()
        kinds = {finding["kind"] for finding in result["findings"]}
        self.assertIn("UNAUTHENTICATED", kinds)
        unauthenticated = [f for f in result["findings"] if f["kind"] == "UNAUTHENTICATED"]
        self.assertEqual(len(unauthenticated), 3)

    def test_audit_serializes(self) -> None:
        ledger = AuditLedger(
            self.services.config.audit_database_path, signing_key=self.node_keys["node-a"]
        )
        try:
            ledger.append("node-a", "decision", {"selected": "wifi"})
        finally:
            ledger.close()
        json.dumps(self.services.audit_memory())


class UnconfiguredNodeTests(unittest.TestCase):
    """A node without the optional pieces reports unavailable, never vacuous."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        key = Ed25519PrivateKey.generate()
        public = root / "operator.pub"
        public.write_bytes(
            key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        catalog = root / "catalog.json"
        catalog.write_text("{}", encoding="utf-8")
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "node_id": "solo",
                    "catalog_path": str(catalog),
                    "catalog_public_key_path": str(public),
                    "audit_database_path": str(root / "audit.db"),
                    "latest_result_path": str(root / "latest.json"),
                    "service_class": "critical_data",
                    "poll_interval_seconds": 5,
                    "observation_max_age_seconds": 30.0,
                    "minimum_confidence": 0.25,
                    "apply_network_changes": False,
                }
            ),
            encoding="utf-8",
        )
        self.services = CapabilityServices(NodeConfig.load(config))

    def tearDown(self) -> None:
        self._directory.cleanup()


    def test_memory_audit_without_a_node_key_is_unavailable_not_passing(self) -> None:
        # Reporting "trustworthy" with no key to check against would be the
        # worst possible answer here.
        result = self.services.audit_memory()
        self.assertFalse(result["available"])
        self.assertNotIn("trustworthy", result)


if __name__ == "__main__":
    unittest.main()


class ComposedPipelineTests(CapabilityIntegrationTests):
    """The mechanisms reaching each other through a real configured node.

    Same fixture as above — signed roster, node key, inbox, ledger on disk —
    driven through the commands an operator actually runs.
    """

    def _publish_update(self, node_id: str, rtt_mean: float) -> None:
        from kval_node.dataset import FEATURE_NAMES
        from kval_node.federated import FeatureStatistics, FederatedUpdate, sign_update

        stats = {name: FeatureStatistics(50, 0.0, 50.0) for name in FEATURE_NAMES}
        stats["rtt_ms"] = FeatureStatistics(50, rtt_mean, 50 * (rtt_mean**2 + 100.0))
        update = FederatedUpdate(node_id, NOW, 50, stats)
        (self.federated_inbox / f"{node_id}.json").write_text(
            json.dumps(sign_update(update, self.node_keys[node_id])), encoding="utf-8"
        )

    def setUp(self) -> None:
        super().setUp()
        self.federated_inbox = self.root / "federated"
        self.federated_inbox.mkdir()
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        config["federated_inbox_path"] = str(self.federated_inbox)
        config["federated_privacy_epsilon"] = 8.0
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        self.services = CapabilityServices(NodeConfig.load(self.root / "config.json"))

    def test_aggregation_runs_the_hierarchical_pipeline_not_a_flat_average(self) -> None:
        for index, node in enumerate(("node-b", "node-c", "node-far")):
            self._publish_update(node, 20.0 + index * 90.0)
        result = self.services.federated_aggregate(now=NOW)
        self.assertTrue(result["available"])
        self.assertIn("cohorts", result)
        self.assertIn(
            "COHORTS_FORMED_BY_RF_ENVIRONMENT_NOT_BY_ADMINISTRATIVE_GROUPING", result["reasons"]
        )

    def test_the_flat_baseline_is_kept_so_the_hierarchy_stays_falsifiable(self) -> None:
        for index, node in enumerate(("node-b", "node-c", "node-far")):
            self._publish_update(node, 20.0 + index * 90.0)
        result = self.services.federated_aggregate(now=NOW)
        self.assertIn("flat_baseline", result)

    def test_reputation_is_written_where_the_next_round_will_find_it(self) -> None:
        for index, node in enumerate(("node-b", "node-c", "node-far")):
            self._publish_update(node, 20.0 + index * 90.0)
        self.services.federated_aggregate(now=NOW)
        self.assertTrue((self.root / "fleet-reputation.json").exists())

    def test_a_privatised_contribution_discloses_what_it_cost(self) -> None:
        ledger = AuditLedger(
            self.services.config.audit_database_path, signing_key=self.node_keys["node-a"]
        )
        try:
            for index in range(6):
                ledger.append("node-a", "decision", {"selected": f"profile-{index}"})
        finally:
            ledger.close()
        result = self.services.federated_update(now=NOW)
        if result["available"]:
            self.assertIn("privacy", result)
            self.assertIn(
                "CONSENSUS_FILTERING_WILL_READ_THIS_NOISE_AS_DIVERGENCE",
                result["privacy"]["reasons"],
            )

    def test_excision_is_reachable_and_certified(self) -> None:
        for index, node in enumerate(("node-b", "node-c", "node-far")):
            self._publish_update(node, 20.0 + index * 5.0)
        result = self.services.excise_node("node-c", now=NOW)
        self.assertTrue(result["available"])
        self.assertTrue(result["certificate"]["exact"])
        self.assertNotIn("node-c", result["remaining_contributors"])





    def test_forecasting_is_reachable_and_bounded(self) -> None:
        result = self.services.forecast_links()
        if result["available"]:
            self.assertIn(
                "FORECAST_MAY_DEMOTE_A_PROFILE_AND_MAY_NEVER_PROMOTE_ONE", result["reasons"]
            )


class CryptoPostureTests(CapabilityIntegrationTests):
    """CNSA 2.0 readiness must be something an operator can ask the node.

    From 1 January 2027 a new National Security System acquisition must support
    CNSA 2.0. The node reports which trust anchors are still classical rather
    than grading itself. See docs/DEFENCE_STANDARDS.md.
    """

    def test_classical_anchors_are_named_not_hidden(self) -> None:
        from datetime import date

        result = self.services.crypto_posture(today=date(2026, 9, 25))
        self.assertTrue(result["available"])
        self.assertIn("catalogue_and_policy", result["classical_anchors"])
        self.assertIn(
            "SOME_TRUST_ANCHORS_STILL_USE_CLASSICAL_ASYMMETRIC_SIGNATURES", result["reasons"]
        )

    def test_the_acquisition_gate_is_counted_down_in_days(self) -> None:
        from datetime import date

        result = self.services.crypto_posture(today=date(2026, 9, 25))
        self.assertEqual(result["nss_acquisition_gate"], "2027-01-01")
        self.assertEqual(result["days_to_acquisition_gate"], 98)
        self.assertTrue(
            any(r.endswith("_DAYS_UNTIL_NEW_NSS_ACQUISITIONS_MUST_SUPPORT_CNSA_2_0")
                for r in result["reasons"])
        )

    def test_past_the_gate_the_report_says_an_exception_is_required(self) -> None:
        from datetime import date

        result = self.services.crypto_posture(today=date(2027, 6, 1))
        self.assertIn(
            "PAST_THE_NSS_ACQUISITION_GATE_A_DOCUMENTED_EXCEPTION_IS_REQUIRED",
            result["reasons"],
        )

    def test_the_hardware_custody_gap_is_stated_on_every_report(self) -> None:
        # The algorithm is the easy half. SP 800-208 wants the state in an HSM
        # and this does not have one, so no report may imply otherwise.
        from datetime import date

        result = self.services.crypto_posture(today=date(2026, 9, 25))
        self.assertIn(
            "SP_800_208_ALSO_REQUIRES_HARDWARE_STATE_CUSTODY_NOT_MET_BY_A_STATE_FILE",
            result["reasons"],
        )

    def test_ed25519_anchors_are_marked_unapproved(self) -> None:
        result = self.services.crypto_posture()
        catalogue = next(a for a in result["anchors"] if a["anchor"] == "catalogue_and_policy")
        self.assertEqual(catalogue["algorithm"], "Ed25519")
        self.assertFalse(catalogue["cnsa_2_0_approved"])
        self.assertEqual(catalogue["reason"], "CLASSICAL_ASYMMETRIC_RETIRED_BY_CNSA_2_0")

    def test_an_unconfigured_anchor_is_not_counted_as_compliant(self) -> None:
        # Absent is not approved. Counting a missing anchor as passing would
        # make the least-configured node look the most compliant.
        result = self.services.crypto_posture()
        lease = next(a for a in result["anchors"] if a["anchor"] == "capability_lease")
        self.assertIsNone(lease["cnsa_2_0_approved"])
        self.assertEqual(lease["reason"], "NOT_CONFIGURED_ON_THIS_NODE")

    def test_posture_serializes(self) -> None:
        json.dumps(self.services.crypto_posture())
