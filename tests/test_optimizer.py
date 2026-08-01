from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from kval_node.models import (
    AccessType,
    Catalog,
    LinkObservation,
    Profile,
    ServiceClass,
    TelemetrySnapshot,
)
from kval_node.optimizer import ProfileOptimizer
from kval_node.standards import StandardsRequirement


NOW = datetime.now(timezone.utc)


def make_profile(profile_id: str, connection_uuid: str, interface: str, cost: float = 0.1) -> Profile:
    return Profile(
        profile_id=profile_id,
        access_type=AccessType.WIFI,
        connection_uuid=connection_uuid,
        interface=interface,
        probe_target="127.0.0.1",
        monetary_cost=cost,
        energy_cost=cost,
        minimum_dwell_seconds=30,
        maximum_switches_per_hour=8,
        allowed_service_classes=frozenset(ServiceClass),
        standards=StandardsRequirement.from_dict({}),
    )


def make_observation(profile_id: str, interface: str, rtt_ms: float, loss: float, throughput: float) -> LinkObservation:
    return LinkObservation(
        profile_id=profile_id,
        interface=interface,
        observed_at=NOW,
        interface_present=True,
        link_up=True,
        connection_active=True,
        rtt_ms=rtt_ms,
        loss_percent=loss,
        jitter_ms=2.0,
        rx_mbps=throughput,
        tx_mbps=1.0,
        signal_percent=90.0,
        access_technology=None,
        standards_verified=True,
        standards_evidence=("NO_ADDITIONAL_STANDARDS_CONSTRAINT",),
        standards_errors=(),
        evidence=("ip-link", "networkmanager", "bound-icmp-probe"),
    )


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fast = make_profile("fast", "11111111-1111-4111-8111-111111111111", "wlan0")
        self.slow = make_profile("slow", "22222222-2222-4222-8222-222222222222", "wlan1")
        self.catalog = Catalog("v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1), (self.fast, self.slow), "a" * 64, "key")

    def test_selects_best_complete_live_evidence(self) -> None:
        snapshot = TelemetrySnapshot(
            NOW,
            "slow",
            (
                make_observation("fast", "wlan0", 15.0, 0.0, 10.0),
                make_observation("slow", "wlan1", 150.0, 10.0, 1.0),
            ),
        )
        result = ProfileOptimizer().recommend(self.catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertEqual(result.selected_profile_id, "fast")
        self.assertGreater(result.confidence, 0.99)

    def test_rejects_unverified_standards_evidence(self) -> None:
        failed = make_observation("fast", "wlan0", 10.0, 0.0, 10.0)
        failed = LinkObservation(**{**failed.__dict__, "standards_verified": False, "standards_errors": ("missing attestation",)})
        snapshot = TelemetrySnapshot(NOW, None, (failed,))
        one_profile_catalog = Catalog("v1", self.catalog.created_at, self.catalog.expires_at, (self.fast,), "b" * 64, "key")
        result = ProfileOptimizer().recommend(one_profile_catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertIsNone(result.selected_profile_id)
        self.assertIn("STANDARDS_REQUIREMENTS_NOT_VERIFIED", result.scores[0].reasons)

    def test_abstains_without_probe_evidence(self) -> None:
        incomplete = make_observation("fast", "wlan0", 10.0, 0.0, 10.0)
        incomplete = LinkObservation(**{**incomplete.__dict__, "rtt_ms": None})
        snapshot = TelemetrySnapshot(NOW, None, (incomplete,))
        one_profile_catalog = Catalog("v1", self.catalog.created_at, self.catalog.expires_at, (self.fast,), "b" * 64, "key")
        result = ProfileOptimizer().recommend(one_profile_catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertIsNone(result.selected_profile_id)


if __name__ == "__main__":
    unittest.main()

