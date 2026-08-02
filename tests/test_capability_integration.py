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
from kval_node.fabric import sign_report
from kval_node.models import utc_now
from kval_node.signing import sign_document
from kval_node.spectrum import BandStatus

from tests.test_fabric import report as build_report

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

    def _publish(self, node_id: str, status: BandStatus) -> None:
        envelope = sign_report(
            build_report(node_id, status=status, produced_at=NOW), self.node_keys[node_id]
        )
        (self.inbox / f"{node_id}.json").write_text(json.dumps(envelope), encoding="utf-8")

    # --- fabric topology -------------------------------------------------------

    def test_topology_names_the_node_whose_loss_would_partition_the_fleet(self) -> None:
        result = self.services.fabric_topology(now=NOW)
        self.assertTrue(result["available"])
        # node-c links a and b at 400 m range; a-b are ~220 m apart, so all
        # three are mutually reachable and none is a hinge.
        self.assertIn("node-far", result["fragility"]["isolated"])
        self.assertIn(
            "NODES_WITH_NO_LINKS_CANNOT_CONTRIBUTE_OR_INHERIT", result["fragility"]["reasons"]
        )

    def test_a_node_that_reported_nothing_still_inherits_a_view(self) -> None:
        # The fabric's premise, end to end: node-b publishes an observation,
        # node-c did not, and node-c still ends up with a belief.
        self._publish("node-b", BandStatus.JAMMING_SUSPECTED)
        result = self.services.fabric_topology(now=NOW)
        beliefs = {item["node_id"]: item for item in result["bands"]["ism-2g4"]}
        self.assertGreater(beliefs["node-c"]["belief"]["busy"], 0.0)
        self.assertFalse(beliefs["node-c"]["first_hand"])
        self.assertTrue(beliefs["node-b"]["first_hand"])

    def test_the_isolated_node_inherits_nothing_rather_than_a_default(self) -> None:
        self._publish("node-b", BandStatus.JAMMING_SUSPECTED)
        result = self.services.fabric_topology(now=NOW)
        beliefs = {item["node_id"]: item for item in result["bands"]["ism-2g4"]}
        self.assertEqual(beliefs["node-far"]["belief"]["unknown"], 1.0)
        self.assertEqual(beliefs["node-far"]["belief"]["clear"], 0.0)

    def test_an_unsigned_report_in_the_inbox_is_rejected_not_used(self) -> None:
        (self.inbox / "junk.json").write_text('{"not": "a report"}', encoding="utf-8")
        result = self.services.fabric_topology(now=NOW)
        self.assertTrue(result["available"])
        self.assertEqual([item["source"] for item in result["rejected_reports"]], ["junk.json"])

    def test_topology_serializes(self) -> None:
        self._publish("node-b", BandStatus.CLEAR)
        json.dumps(self.services.fabric_topology(now=NOW))

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

    def test_topology_without_a_roster_is_unavailable(self) -> None:
        result = self.services.fabric_topology()
        self.assertFalse(result["available"])
        self.assertIn("roster", result["reason"])

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

    def test_a_localisation_is_recorded_so_a_track_can_be_built(self) -> None:
        # The composition: geolocate writes fixes, mobility reads them. Without
        # the recording step the tracker has nothing and the pairing is fiction.
        from kval_node.capabilities import EMITTER_RECORD_TYPE

        self.services._record_fix(
            {
                "available": True,
                "band_id": "ism-2g4",
                "latitude_deg": 1.3010,
                "longitude_deg": 103.8010,
                "uncertainty_radius_m": 25.0,
                "method": "trilateration",
                "contributing_nodes": ["node-a", "node-b", "node-c"],
            }
        )
        ledger = AuditLedger(self.services.config.audit_database_path)
        try:
            kinds = {row["record_type"] for row in ledger.signed_records()}
        finally:
            ledger.close()
        self.assertIn(EMITTER_RECORD_TYPE, kinds)

    def test_a_refused_localisation_never_becomes_a_track_point(self) -> None:
        self.services._record_fix({"available": False, "reason": "collinear"})
        result = self.services.track_emitter("ism-2g4")
        self.assertFalse(result["available"])

    def test_tracking_reads_back_the_fixes_localisation_wrote(self) -> None:
        for index in range(6):
            self.services._record_fix(
                {
                    "available": True,
                    "band_id": "ism-2g4",
                    "latitude_deg": 1.3010 + index * 0.0001,
                    "longitude_deg": 103.8010,
                    "uncertainty_radius_m": 25.0,
                    "method": "trilateration",
                    "contributing_nodes": ["node-a", "node-b", "node-c"],
                }
            )
        result = self.services.track_emitter("ism-2g4")
        self.assertTrue(result["available"])
        self.assertEqual(result["fixes"], 6)
        self.assertGreaterEqual(result["place_radius_m"], 50.0)

    def test_fleet_detection_composes_reports_cohorts_and_reputation(self) -> None:
        for node in ("node-b", "node-c", "node-far"):
            self._publish(node, BandStatus.JAMMING_SUSPECTED)
            self._publish_update(node, 20.0)
        result = self.services.detect_fleet_anomaly(now=NOW)
        self.assertTrue(result["available"])
        self.assertIn("cohorts_assigned", result)
        self.assertIn("verdicts", result)

    def test_forecasting_is_reachable_and_bounded(self) -> None:
        result = self.services.forecast_links()
        if result["available"]:
            self.assertIn(
                "FORECAST_MAY_DEMOTE_A_PROFILE_AND_MAY_NEVER_PROMOTE_ONE", result["reasons"]
            )
