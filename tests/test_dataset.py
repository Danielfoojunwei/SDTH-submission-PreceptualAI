from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kval_node.audit import AuditLedger
from kval_node.dataset import FEATURE_NAMES, CorpusExporter, DatasetError, write_jsonl

NOW = datetime.now(timezone.utc)


def decision_payload(
    *,
    profile_id: str = "wifi",
    rtt: float = 20.0,
    loss: float = 0.0,
    current: str | None = "wifi",
    link_up: bool = True,
    contested: list[str] | None = None,
    clock_trusted: bool = True,
) -> dict:
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
                    "link_up": link_up,
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
            "clock": {"trusted": clock_trusted},
            "power": {"recommendation": {"max_inference_duty": 1.0}},
            "degraded_band_ids": contested or [],
        },
    }


class CorpusExportTests(unittest.TestCase):
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

    def test_consecutive_records_produce_a_labelled_example(self) -> None:
        self._write([decision_payload(rtt=100.0, loss=5.0), decision_payload(rtt=20.0, loss=0.0)])
        examples, stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        self.assertEqual(len(examples), 1)
        self.assertEqual(stats.examples, 1)
        example = examples[0]
        self.assertEqual(example.features["rtt_ms"], 100.0)
        self.assertEqual(example.outcome_rtt_ms, 20.0)
        self.assertTrue(example.label_improved)

    def test_worse_outcome_is_labelled_not_improved(self) -> None:
        self._write([decision_payload(rtt=20.0, loss=0.0), decision_payload(rtt=400.0, loss=30.0)])
        examples, _stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        self.assertFalse(examples[0].label_improved)

    def test_features_never_contain_the_outcome(self) -> None:
        # A distinctive successor value that cannot collide with a legitimate
        # feature (counts and flags are small integers).
        self._write([decision_payload(rtt=100.0, loss=5.0), decision_payload(rtt=7777.0, loss=88.0)])
        examples, _stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        self.assertNotIn(7777.0, examples[0].features.values())
        self.assertNotIn(88.0, examples[0].features.values())
        self.assertEqual(examples[0].features["rtt_ms"], 100.0)
        self.assertEqual(examples[0].features["loss_percent"], 5.0)
        # The outcome is still captured — just not as a feature.
        self.assertEqual(examples[0].outcome_rtt_ms, 7777.0)

    def test_every_declared_feature_is_present(self) -> None:
        self._write([decision_payload(), decision_payload()])
        examples, _stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        exported = examples[0].to_dict()["features"]
        self.assertEqual(sorted(exported), sorted(FEATURE_NAMES))

    def test_contested_bands_and_clock_reach_the_features(self) -> None:
        self._write(
            [
                decision_payload(contested=["ism-2g4", "unii-1"], clock_trusted=False),
                decision_payload(),
            ]
        )
        examples, _stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        self.assertEqual(examples[0].features["contested_band_count"], 2.0)
        self.assertEqual(examples[0].features["clock_trusted"], 0.0)

    def test_record_without_a_successor_is_dropped(self) -> None:
        self._write([decision_payload()])
        examples, stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        self.assertEqual(examples, [])
        self.assertEqual(stats.dropped_no_successor, 1)
        self.assertIn("NO_LABELLED_EXAMPLES_AVAILABLE", stats.reasons)

    def test_incomplete_evidence_is_dropped_not_guessed(self) -> None:
        broken = decision_payload()
        broken["telemetry"]["observations"][0]["rtt_ms"] = None
        self._write([broken, decision_payload()])
        examples, stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        self.assertEqual(examples, [])
        self.assertEqual(stats.dropped_incomplete, 1)

    def test_export_refuses_a_tampered_ledger(self) -> None:
        self._write([decision_payload(), decision_payload()])
        connection = sqlite3.connect(self.path)
        connection.execute("UPDATE audit_records SET payload_json = ? WHERE id = 1", ('{"x":1}',))
        connection.commit()
        connection.close()
        with self.assertRaisesRegex(DatasetError, "unverified audit chain"):
            CorpusExporter(self.path).export()

    def test_jsonl_round_trips(self) -> None:
        import json

        self._write([decision_payload(rtt=100.0), decision_payload(rtt=10.0)])
        examples, _stats = CorpusExporter(self.path, horizon_seconds=3600.0).export()
        destination = Path(self._dir.name) / "corpus.jsonl"
        write_jsonl(examples, destination)
        lines = destination.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["features"]["rtt_ms"], 100.0)
        self.assertTrue(parsed["label_improved"])

    def test_horizon_excludes_a_distant_successor(self) -> None:
        # Timestamps are controlled rather than taken from write latency: two
        # consecutive ledger writes can complete inside a millisecond on a fast
        # machine, so racing the clock would make this test flaky. Editing
        # created_at in place is not an option either, because that breaks the
        # hash chain and export would refuse. The ledger itself is real, so the
        # chain-verification step still runs for real.
        self._write([decision_payload(rtt=100.0), decision_payload(rtt=10.0)])
        spaced = [
            (NOW, "node-1", decision_payload(rtt=100.0)),
            (NOW + timedelta(hours=1), "node-1", decision_payload(rtt=10.0)),
        ]

        class SpacedExporter(CorpusExporter):
            def _decision_records(self):
                return iter(spaced)

        far = SpacedExporter(self.path, horizon_seconds=60.0).export()
        self.assertEqual(far[0], [])
        # Both records drop: the first has no successor inside the window, and
        # the last has no successor at all.
        self.assertEqual(far[1].dropped_no_successor, 2)

        near = SpacedExporter(self.path, horizon_seconds=7200.0).export()
        self.assertEqual(len(near[0]), 1)
        self.assertEqual(near[0][0].horizon_seconds, 3600.0)


if __name__ == "__main__":
    unittest.main()
