"""Outcome analysis must use the learning primitives on real recorded history."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from kval_node.analysis import MIN_BASELINE_SAMPLES, OutcomeAnalyzer, build_baseline
from kval_node.audit import AuditLedger
from kval_node.dataset import FEATURE_NAMES, Example
from kval_node.learning import LearnedDecision

NOW = datetime.now(timezone.utc)


def decision_payload(
    *,
    profile_id: str = "wifi",
    rtt: float = 20.0,
    loss: float = 0.0,
    current: str | None = "wifi",
) -> dict:
    """A decision record in the shape ``service.py`` writes to the ledger."""
    return {
        "schema": "kval-node-result/v1",
        "telemetry": {
            "observed_at": NOW.isoformat(),
            "current_profile_id": current,
            "observations": [
                {
                    "profile_id": profile_id,
                    "rtt_ms": rtt,
                    "loss_percent": loss,
                    "jitter_ms": 2.0,
                    "rx_mbps": 10.0,
                    "tx_mbps": 1.0,
                    "signal_percent": 80.0,
                    "link_up": True,
                }
            ],
        },
        "decision": {
            "action": "PRESENT_TO_HOST",
            "recommendation": {
                "selected_profile_id": profile_id,
                "confidence": 0.9,
                "scores": [{"profile_id": profile_id, "utility": 0.8}],
            },
        },
        "capabilities": {
            "clock": {"trusted": True},
            "power": {"recommendation": {"max_inference_duty": 1.0}},
            "degraded_band_ids": [],
        },
    }


def example(rtt: float, *, profile: str = "wifi", loss: float = 0.0) -> Example:
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["rtt_ms"] = rtt
    features["loss_percent"] = loss
    features["confidence"] = 0.9
    return Example(
        node_id="n",
        decided_at=NOW,
        selected_profile_id=profile,
        features=features,
        outcome_rtt_ms=rtt,
        outcome_loss_percent=loss,
        outcome_link_up=True,
        label_improved=True,
        horizon_seconds=1.0,
    )


class BaselineTests(unittest.TestCase):
    def test_too_few_samples_yields_no_baseline(self) -> None:
        baseline, reasons = build_baseline([example(20.0) for _ in range(3)])
        self.assertIsNone(baseline)
        self.assertIn(f"NEED_AT_LEAST_{MIN_BASELINE_SAMPLES}_EXAMPLES", reasons)

    def test_constant_features_are_excluded_not_divided_by(self) -> None:
        # Only rtt_ms varies; every other feature is constant and must be
        # dropped rather than yielding a zero standard deviation.
        examples = [example(float(20 + index)) for index in range(MIN_BASELINE_SAMPLES + 5)]
        baseline, reasons = build_baseline(examples)
        self.assertIsNotNone(baseline)
        self.assertIn("rtt_ms", baseline.means)
        self.assertNotIn("jitter_ms", baseline.means)
        self.assertTrue(any("CONSTANT_FEATURES_EXCLUDED" in reason for reason in reasons))
        self.assertTrue(all(value > 0 for value in baseline.stddevs.values()))

    def test_all_constant_history_yields_no_baseline(self) -> None:
        baseline, reasons = build_baseline([example(20.0) for _ in range(MIN_BASELINE_SAMPLES + 5)])
        self.assertIsNone(baseline)
        self.assertIn("NO_FEATURE_VARIED_ACROSS_HISTORY", reasons)


class OutcomeAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "audit.db"

    def _write(self, payloads: list[dict]) -> None:
        ledger = AuditLedger(self.path)
        try:
            for payload in payloads:
                ledger.append("node-1", "decision", payload)
        finally:
            ledger.close()

    def test_empty_history_is_unavailable_not_an_error(self) -> None:
        self._write([decision_payload()])
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertFalse(result["available"])
        self.assertIn("no labelled history", result["reason"])

    def test_detects_a_latency_regime_change(self) -> None:
        good = [decision_payload(rtt=20.0) for _ in range(10)]
        bad = [decision_payload(rtt=400.0) for _ in range(10)]
        self._write(good + bad)
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertTrue(result["available"])
        self.assertTrue(result["rtt_change_point"]["detected"])
        self.assertIn("REGIME_CHANGE_DETECTED", result["reasons"])

    # Real telemetry always jitters. A perfectly constant series is degenerate
    # and CausalGate refuses to attribute anything to it, so these fixtures
    # carry the small variation a live link actually produces.
    JITTER = (0.0, 3.0, -2.0, 1.0, -3.0, 2.0, -1.0, 4.0)

    def test_a_switch_that_helped_is_attributed(self) -> None:
        before = [
            decision_payload(profile_id="slow", current="slow", rtt=300.0 + offset)
            for offset in self.JITTER
        ]
        after = [
            decision_payload(profile_id="fast", current="fast", rtt=20.0 + offset)
            for offset in self.JITTER
        ]
        self._write(before + after)
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertEqual(len(result["switches"]), 1)
        switch = result["switches"][0]
        self.assertTrue(switch["attributed"])
        self.assertEqual(switch["profile_id"], "fast")
        self.assertGreater(switch["improvement_ms"], 100.0)
        self.assertEqual(result["switches_that_helped"], 1)

    def test_a_switch_that_changed_nothing_is_not_credited(self) -> None:
        before = [
            decision_payload(profile_id="a", current="a", rtt=100.0 + offset) for offset in self.JITTER
        ]
        after = [
            decision_payload(profile_id="b", current="b", rtt=101.0 + offset) for offset in self.JITTER
        ]
        self._write(before + after)
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertFalse(result["switches"][0]["attributed"])
        self.assertEqual(result["switches_that_helped"], 0)
        self.assertIn("NO_SWITCH_MEASURABLY_IMPROVED_LATENCY", result["reasons"])

    def test_a_degenerate_constant_series_is_refused(self) -> None:
        # A stuck sensor reporting an identical value forever is not evidence
        # that a switch worked, however large the apparent difference.
        before = [decision_payload(profile_id="a", current="a", rtt=300.0) for _ in range(8)]
        after = [decision_payload(profile_id="b", current="b", rtt=20.0) for _ in range(8)]
        self._write(before + after)
        result = OutcomeAnalyzer(self.path).analyze()
        switch = result["switches"][0]
        self.assertFalse(switch["attributed"])
        self.assertIn("ZERO_VARIANCE_IN_BOTH_SAMPLES", switch["reasons"])

    def test_drift_is_reported_once_history_is_long_enough(self) -> None:
        payloads = [decision_payload(rtt=float(20 + index)) for index in range(MIN_BASELINE_SAMPLES + 6)]
        self._write(payloads)
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertTrue(result["drift"]["available"])
        self.assertIn(result["drift"]["decision"], {item.value for item in LearnedDecision})

    def test_short_history_reports_why_drift_is_unavailable(self) -> None:
        self._write([decision_payload(rtt=float(20 + index)) for index in range(5)])
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertFalse(result["drift"]["available"])
        self.assertTrue(result["drift"]["reasons"])

    def test_refuses_a_tampered_ledger(self) -> None:
        self._write([decision_payload(), decision_payload()])
        connection = sqlite3.connect(self.path)
        connection.execute("UPDATE audit_records SET payload_json = ? WHERE id = 1", ('{"x":1}',))
        connection.commit()
        connection.close()
        result = OutcomeAnalyzer(self.path).analyze()
        self.assertFalse(result["available"])
        self.assertIn("unverified audit chain", result["reason"])

    def test_result_serializes(self) -> None:
        self._write([decision_payload(rtt=20.0), decision_payload(rtt=30.0)])
        json.dumps(OutcomeAnalyzer(self.path).analyze())


if __name__ == "__main__":
    unittest.main()
