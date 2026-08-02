"""Interference reports in a shape a spectrum manager can ingest.

Joint Spectrum Interference Resolution reporting under CJCSM 3320.01D (Joint
Electromagnetic Spectrum Operations, 24 January 2025). Kval does the hard part
— detecting and corroborating across nodes — and previously emitted its own
JSON, which reaches nobody's tooling. See docs/DEFENCE_STANDARDS.md.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from kval_node.interference import (
    EFFECT_DEGRADED,
    EFFECT_INTERMITTENT_OUTAGE,
    EFFECT_LOSS_OF_SERVICE,
    EFFECT_NONE,
    build_report,
    effect_from_evidence,
)

NOW = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)


def _report(**overrides):
    base = dict(
        report_id="node-a-ism-2g4-1",
        observed_at=NOW,
        reporting_node="node-a",
        corroborating_nodes=["node-b", "node-c"],
        band_id="ism-2g4",
        centre_frequency_hz=2_442_000_000,
        bandwidth_hz=22_000_000,
        victim_role="payload",
        link_available=True,
    )
    base.update(overrides)
    return build_report(**base)


class EffectTests(unittest.TestCase):
    def test_a_lost_link_is_loss_of_service(self) -> None:
        self.assertEqual(
            effect_from_evidence(corroborating_nodes=0, victim_role="payload", link_available=False),
            EFFECT_LOSS_OF_SERVICE,
        )

    def test_a_corroborated_c2_victim_is_escalated(self) -> None:
        # A command link degraded across several nodes is the report a spectrum
        # manager needs on their desk first.
        self.assertEqual(
            effect_from_evidence(corroborating_nodes=2, victim_role="c2_cnpc", link_available=True),
            EFFECT_INTERMITTENT_OUTAGE,
        )

    def test_a_corroborated_payload_victim_is_degraded(self) -> None:
        self.assertEqual(
            effect_from_evidence(corroborating_nodes=2, victim_role="payload", link_available=True),
            EFFECT_DEGRADED,
        )

    def test_no_corroboration_and_a_working_link_is_no_effect(self) -> None:
        self.assertEqual(
            effect_from_evidence(corroborating_nodes=0, victim_role="payload", link_available=True),
            EFFECT_NONE,
        )


class ReportTests(unittest.TestCase):
    def test_the_report_carries_the_jsir_fields(self) -> None:
        payload = _report().to_dict()
        for field in (
            "observed_at", "reporting_node", "band_id", "centre_frequency_hz",
            "bandwidth_hz", "victim_role", "effect",
        ):
            self.assertIn(field, payload)

    def test_intent_is_never_classified(self) -> None:
        # Nothing in a noise floor says whether a microwave oven, a faulty
        # amplifier or an adversary put it there.
        payload = _report().to_dict()
        self.assertIsNone(payload["intent"])
        self.assertIn(
            "INTENT_IS_NOT_CLASSIFIED_BECAUSE_A_MEASUREMENT_CANNOT_ESTABLISH_IT",
            payload["reasons"],
        )

    def test_no_source_is_named(self) -> None:
        payload = _report(
            estimated_position={
                "latitude_deg": 1.3, "longitude_deg": 103.8, "uncertainty_radius_m": 250.0
            }
        ).to_dict()
        self.assertIn(
            "NO_SOURCE_IS_NAMED_A_POSITION_WITH_UNCERTAINTY_IS_A_PLACE_NOT_AN_ACTOR",
            payload["reasons"],
        )
        self.assertIn("uncertainty_radius_m", payload["estimated_position"])

    def test_the_reporting_node_is_not_its_own_corroborator(self) -> None:
        report = _report(corroborating_nodes=["node-a", "node-b"])
        self.assertEqual(report.corroborating_nodes, ("node-b",))

    def test_an_uncorroborated_report_says_so(self) -> None:
        report = _report(corroborating_nodes=[])
        self.assertIn("SINGLE_NODE_OBSERVATION_UNCORROBORATED_BY_ANY_PEER", report.reasons)

    def test_a_missing_position_is_stated_not_omitted(self) -> None:
        report = _report(estimated_position=None)
        self.assertIn("NO_POSITION_ESTIMATE_SURVIVED_THE_LOCATOR_S_CHECKS", report.reasons)

    def test_an_out_of_range_excess_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _report(noise_floor_excess_db=9999.0)

    def test_report_serializes(self) -> None:
        json.dumps(_report().to_dict())


if __name__ == "__main__":
    unittest.main()
