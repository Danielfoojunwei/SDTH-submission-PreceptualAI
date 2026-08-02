from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.fabric import (
    BandReport,
    FabricAggregator,
    FabricError,
    FabricReport,
    PeerRoster,
    sign_report,
    verify_report,
)
from kval_node.spectrum import BandStatus

NOW = datetime.now(timezone.utc)


def pem(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def roster(keys: dict[str, Ed25519PrivateKey]) -> PeerRoster:
    return PeerRoster.from_dict(
        {
            "roster_id": "r1",
            "created_at": (NOW - timedelta(minutes=1)).isoformat(),
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "peers": [
                {"node_id": node_id, "public_key_pem": pem(key), "role": "sub-node"}
                for node_id, key in keys.items()
            ],
        },
        now=NOW,
    )


def report(
    node_id: str,
    *,
    status: BandStatus = BandStatus.CLEAR,
    band_id: str = "ism-2g4",
    produced_at: datetime | None = None,
    clock_trusted: bool = True,
    confidence: float = 1.0,
    noise_excess: float | None = 0.0,
) -> FabricReport:
    return FabricReport(
        node_id=node_id,
        produced_at=produced_at or NOW,
        policy_id="sp",
        policy_hash="a" * 64,
        clock_trusted=clock_trusted,
        bands=(
            BandReport(
                band_id=band_id,
                status=status,
                occupancy_percent=50.0,
                noise_floor_excess_db=noise_excess,
                detected_emitters=1,
                confidence=confidence,
            ),
        ),
    )


class RosterTests(unittest.TestCase):
    def test_expired_roster_is_rejected(self) -> None:
        key = Ed25519PrivateKey.generate()
        with self.assertRaisesRegex(ValueError, "expired"):
            PeerRoster.from_dict(
                {
                    "roster_id": "r",
                    "created_at": (NOW - timedelta(days=2)).isoformat(),
                    "expires_at": (NOW - timedelta(minutes=1)).isoformat(),
                    "peers": [{"node_id": "a", "public_key_pem": pem(key)}],
                },
                now=NOW,
            )

    def test_duplicate_node_ids_rejected(self) -> None:
        key = Ed25519PrivateKey.generate()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PeerRoster.from_dict(
                {
                    "roster_id": "r",
                    "created_at": (NOW - timedelta(minutes=1)).isoformat(),
                    "expires_at": (NOW + timedelta(days=1)).isoformat(),
                    "peers": [
                        {"node_id": "a", "public_key_pem": pem(key)},
                        {"node_id": "a", "public_key_pem": pem(key)},
                    ],
                },
                now=NOW,
            )


class ReportVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.roster = roster({"node-a": self.key})

    def test_round_trip(self) -> None:
        envelope = sign_report(report("node-a"), self.key)
        verified = verify_report(envelope, self.roster, now=NOW)
        self.assertEqual(verified.node_id, "node-a")

    def test_unknown_author_refused(self) -> None:
        stranger = Ed25519PrivateKey.generate()
        envelope = sign_report(report("node-z"), stranger)
        with self.assertRaisesRegex(FabricError, "not on the signed roster"):
            verify_report(envelope, self.roster, now=NOW)

    def test_impersonation_refused(self) -> None:
        # A stranger signs a report claiming to be a rostered node.
        stranger = Ed25519PrivateKey.generate()
        envelope = sign_report(report("node-a"), stranger)
        with self.assertRaisesRegex(FabricError, "key ID does not match"):
            verify_report(envelope, self.roster, now=NOW)

    def test_tampered_body_refused(self) -> None:
        envelope = sign_report(report("node-a"), self.key)
        altered = copy.deepcopy(envelope)
        altered["body"]["bands"][0]["status"] = BandStatus.JAMMING_SUSPECTED.value
        with self.assertRaisesRegex(FabricError, "signature verification failed"):
            verify_report(altered, self.roster, now=NOW)

    def test_unauthenticated_extra_field_refused(self) -> None:
        envelope = sign_report(report("node-a"), self.key)
        envelope["body"]["smuggled"] = "x"
        with self.assertRaisesRegex(FabricError, "unauthenticated|malformed"):
            verify_report(envelope, self.roster, now=NOW)

    def test_stale_report_refused(self) -> None:
        envelope = sign_report(report("node-a", produced_at=NOW - timedelta(hours=1)), self.key)
        with self.assertRaisesRegex(FabricError, "older than the correlation window"):
            verify_report(envelope, self.roster, now=NOW)

    def test_future_report_refused(self) -> None:
        envelope = sign_report(report("node-a", produced_at=NOW + timedelta(hours=1)), self.key)
        with self.assertRaisesRegex(FabricError, "future"):
            verify_report(envelope, self.roster, now=NOW)


class AggregationTests(unittest.TestCase):
    def test_quorum_confirms_interference(self) -> None:
        reports = [
            report(node, status=BandStatus.JAMMING_SUSPECTED, noise_excess=12.0)
            for node in ("a", "b")
        ]
        view = FabricAggregator(quorum=2).aggregate(reports, now=NOW)
        band = view.bands[0]
        self.assertTrue(band.interference_confirmed)
        self.assertEqual(band.corroborating_nodes, ("a", "b"))
        self.assertIn("FLEET_CONFIRMED_INTERFERENCE", view.reasons)

    def test_single_node_does_not_confirm(self) -> None:
        reports = [report("a", status=BandStatus.JAMMING_SUSPECTED, noise_excess=12.0)]
        view = FabricAggregator(quorum=2).aggregate(reports, now=NOW)
        self.assertFalse(view.bands[0].interference_confirmed)
        self.assertIn("INTERFERENCE_REPORTED_BELOW_QUORUM", view.bands[0].reasons)

    def test_untrusted_clock_cannot_corroborate(self) -> None:
        # Two nodes both claim jamming, but one cannot be aligned in time, so
        # the quorum is not met and the report is marked excluded.
        reports = [
            report("a", status=BandStatus.JAMMING_SUSPECTED, noise_excess=12.0),
            report("b", status=BandStatus.JAMMING_SUSPECTED, noise_excess=12.0, clock_trusted=False),
        ]
        view = FabricAggregator(quorum=2).aggregate(reports, now=NOW)
        self.assertFalse(view.bands[0].interference_confirmed)
        self.assertEqual(view.bands[0].corroborating_nodes, ("a",))
        self.assertIn("b", view.excluded_nodes)

    def test_clear_everywhere_becomes_a_fleet_candidate(self) -> None:
        reports = [report(node, status=BandStatus.CLEAR) for node in ("a", "b")]
        view = FabricAggregator(quorum=2).aggregate(reports, now=NOW)
        self.assertTrue(view.bands[0].clear_everywhere)
        self.assertEqual(view.fleet_clear_band_ids, ("ism-2g4",))

    def test_one_bad_node_removes_the_band_from_fleet_clear(self) -> None:
        reports = [
            report("a", status=BandStatus.CLEAR),
            report("b", status=BandStatus.JAMMING_SUSPECTED, noise_excess=12.0),
        ]
        view = FabricAggregator(quorum=2).aggregate(reports, now=NOW)
        self.assertFalse(view.bands[0].clear_everywhere)
        self.assertEqual(view.fleet_clear_band_ids, ())

    def test_silent_nodes_are_named(self) -> None:
        view = FabricAggregator().aggregate(
            [report("a")], expected_nodes=frozenset({"a", "b", "c"}), now=NOW
        )
        self.assertEqual(view.silent_nodes, ("b", "c"))
        self.assertIn("EXPECTED_NODES_SILENT", view.reasons)

    def test_stale_reports_are_discarded(self) -> None:
        reports = [report("a"), report("b", produced_at=NOW - timedelta(hours=2))]
        view = FabricAggregator().aggregate(reports, now=NOW)
        self.assertEqual(view.contributing_nodes, ("a",))
        self.assertIn("b", view.excluded_nodes)

    def test_zero_confidence_band_is_not_usable_evidence(self) -> None:
        view = FabricAggregator().aggregate(
            [report("a", status=BandStatus.CLEAR, confidence=0.0)], now=NOW
        )
        self.assertFalse(view.bands[0].clear_everywhere)
        self.assertIn("NO_NODE_PRODUCED_USABLE_EVIDENCE", view.bands[0].reasons)


if __name__ == "__main__":
    unittest.main()
