from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kval_node.audit import AuditLedger
from kval_node.models import (
    AccessType,
    Catalog,
    LinkObservation,
    Profile,
    Recommendation,
    ScoredProfile,
    ServiceClass,
    ShieldAction,
    TelemetrySnapshot,
)
from kval_node.policy import SafetyShield
from kval_node.standards import StandardsRequirement


class PolicyTests(unittest.TestCase):
    def test_valid_profile_can_be_presented_or_applied(self) -> None:
        now = datetime.now(timezone.utc)
        profile = Profile(
            "wifi",
            AccessType.WIFI,
            "11111111-1111-4111-8111-111111111111",
            "wlan0",
            "127.0.0.1",
            0.1,
            0.1,
            10,
            5,
            frozenset({ServiceClass.CRITICAL_DATA}),
            StandardsRequirement.from_dict({}),
        )
        catalog = Catalog("v1", now - timedelta(minutes=1), now + timedelta(days=1), (profile,), "a" * 64, "key")
        observation = LinkObservation(
            "wifi",
            "wlan0",
            now,
            True,
            True,
            True,
            10.0,
            0.0,
            1.0,
            2.0,
            1.0,
            90.0,
            None,
            True,
            ("NO_ADDITIONAL_STANDARDS_CONSTRAINT",),
            (),
            ("ip-link", "networkmanager", "bound-icmp-probe"),
        )
        snapshot = TelemetrySnapshot(now, None, (observation,))
        recommendation = Recommendation(
            "rec-1",
            now,
            "v1",
            "wifi",
            None,
            ServiceClass.CRITICAL_DATA,
            1.0,
            (ScoredProfile("wifi", 0.9, 1.0, {}, ("LIVE_EVIDENCE_SCORED",)),),
            ("HIGHEST_POLICY_CONSTRAINED_UTILITY",),
        )
        with tempfile.TemporaryDirectory() as directory:
            ledger = AuditLedger(Path(directory) / "audit.db")
            try:
                shield = SafetyShield(catalog, ledger, observation_max_age_seconds=8.0, minimum_confidence=0.75)
                advisory = shield.evaluate(recommendation, snapshot, apply_requested=False, now=now)
                apply = shield.evaluate(recommendation, snapshot, apply_requested=True, now=now)
                self.assertIs(advisory.action, ShieldAction.PRESENT_TO_HOST)
                self.assertIs(apply.action, ShieldAction.APPLY_APPROVED_PROFILE)
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()

