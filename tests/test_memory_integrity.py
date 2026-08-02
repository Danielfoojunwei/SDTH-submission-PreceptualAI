"""Memory-level integrity safeguards.

Defends the attack in Zhang, Zheng, Xu, Zhou, Shen, Ou, Zhang and Lam,
"MemMorph: Tool Hijacking in LLM Agents via Memory Poisoning" (arXiv:2605.26154),
which reports up to 85.9% attack success from three injected records. See
docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import unittest
from datetime import timedelta

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.memory_integrity import (
    DEFAULT_BURST_THRESHOLD,
    MemoryAuditor,
    MemoryIntegrityError,
    SignedRecord,
    bounded_influence,
    payload_similarity,
    sign_record,
    verify_record,
)
from kval_node.models import utc_now

NODE = "kval-node-01"


def _payload(rtt: float, band: str = "band-2400") -> dict[str, object]:
    return {"decision": "hold", "rtt_ms": rtt, "band_id": band, "passive": True}


class SignedRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key()

    def test_round_trip(self) -> None:
        record = sign_record(_payload(42.0), self.key, node_id=NODE)
        verify_record(record, self.public, node_id=NODE)

    def test_a_record_this_node_did_not_write_is_refused(self) -> None:
        # The MemMorph adversary has write access to the memory store but not
        # to the node's signing key.
        attacker = Ed25519PrivateKey.generate()
        forged = sign_record(_payload(9999.0), attacker, node_id=NODE)
        with self.assertRaises(MemoryIntegrityError):
            verify_record(forged, self.public, node_id=NODE)

    def test_editing_the_payload_after_signing_is_refused(self) -> None:
        record = sign_record(_payload(42.0), self.key, node_id=NODE)
        tampered = SignedRecord(
            node_id=record.node_id,
            recorded_at=record.recorded_at,
            payload=_payload(9999.0),
            signature=record.signature,
            key_id=record.key_id,
        )
        with self.assertRaises(MemoryIntegrityError):
            verify_record(tampered, self.public, node_id=NODE)

    def test_backdating_a_record_is_refused(self) -> None:
        # Timestamp is inside the signed body, so a record cannot be moved in
        # time to sit alongside plausible history.
        record = sign_record(_payload(42.0), self.key, node_id=NODE)
        moved = SignedRecord(
            node_id=record.node_id,
            recorded_at=record.recorded_at - timedelta(hours=6),
            payload=record.payload,
            signature=record.signature,
            key_id=record.key_id,
        )
        with self.assertRaises(MemoryIntegrityError):
            verify_record(moved, self.public, node_id=NODE)

    def test_a_record_claiming_another_author_is_refused(self) -> None:
        record = sign_record(_payload(42.0), self.key, node_id="kval-node-99")
        with self.assertRaises(MemoryIntegrityError):
            verify_record(record, self.public, node_id=NODE)

    def test_malformed_signature_is_refused_not_crashed_on(self) -> None:
        record = sign_record(_payload(42.0), self.key, node_id=NODE)
        broken = SignedRecord(
            node_id=record.node_id,
            recorded_at=record.recorded_at,
            payload=record.payload,
            signature="not base64!!",
            key_id=record.key_id,
        )
        with self.assertRaises(MemoryIntegrityError):
            verify_record(broken, self.public, node_id=NODE)

    def test_record_serializes(self) -> None:
        json.dumps(sign_record(_payload(1.0), self.key, node_id=NODE).to_dict())


class PayloadSimilarityTests(unittest.TestCase):
    def test_identical_payloads_are_fully_similar(self) -> None:
        self.assertEqual(payload_similarity(_payload(1.0), _payload(1.0)), 1.0)

    def test_unrelated_payloads_are_dissimilar(self) -> None:
        self.assertLess(payload_similarity({"a": 1}, {"z": 99}), 0.1)

    def test_templated_variants_score_high(self) -> None:
        # One field changed out of four: the shape MemMorph's generated records
        # take, since they come from a single template.
        self.assertGreaterEqual(payload_similarity(_payload(10.0), _payload(11.0)), 0.5)

    def test_nested_structure_is_compared_not_ignored(self) -> None:
        left = {"outer": {"inner": [1, 2, 3]}}
        right = {"outer": {"inner": [1, 2, 4]}}
        similarity = payload_similarity(left, right)
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)


class MemoryAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key()
        self.auditor = MemoryAuditor()
        self.base = utc_now()

    def _genuine_history(self, count: int = 8) -> list[SignedRecord]:
        # Real operation writes one record per cycle, minutes apart, with
        # readings that differ from each other.
        return [
            sign_record(
                _payload(40.0 + index * 3.0, band=f"band-{2400 + index}"),
                self.key,
                node_id=NODE,
                now=self.base + timedelta(minutes=index * 5),
            )
            for index in range(count)
        ]

    def test_genuine_history_is_trusted(self) -> None:
        report = self.auditor.audit(self._genuine_history(), self.public, node_id=NODE)
        self.assertTrue(report.trustworthy)
        self.assertEqual(report.findings, ())
        self.assertIn("HISTORY_CONSISTENT_WITH_THIS_NODE_S_OWN_OPERATION", report.reasons)

    def test_the_paper_s_three_record_injection_is_caught(self) -> None:
        # MemMorph's headline result: 85.9% attack success from three records.
        # Three is exactly the burst size this auditor is tuned to catch.
        attacker = Ed25519PrivateKey.generate()
        history = self._genuine_history()
        injected_at = self.base + timedelta(minutes=100)
        history += [
            sign_record(
                {"decision": "hold", "rtt_ms": 900.0, "band_id": "band-2400", "passive": True},
                attacker,
                node_id=NODE,
                now=injected_at + timedelta(seconds=index),
            )
            for index in range(DEFAULT_BURST_THRESHOLD)
        ]
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertFalse(report.trustworthy)
        kinds = {finding.kind for finding in report.findings}
        self.assertIn("UNAUTHENTICATED", kinds)
        self.assertIn("BURST", kinds)
        self.assertIn("NEAR_DUPLICATE", kinds)
        self.assertIn("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE", report.reasons)

    def test_a_signed_burst_is_still_flagged(self) -> None:
        # Authorship binding is not sufficient on its own: a compromised node
        # can sign its own poisoned records. The behavioural checks are what
        # catch that case, so they must not depend on the signature failing.
        history = self._genuine_history()
        burst_at = self.base + timedelta(minutes=100)
        history += [
            sign_record(
                _payload(500.0 + index, band=f"band-{9000 + index}"),
                self.key,
                node_id=NODE,
                now=burst_at + timedelta(seconds=index),
            )
            for index in range(DEFAULT_BURST_THRESHOLD)
        ]
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertFalse(report.trustworthy)
        kinds = {finding.kind for finding in report.findings}
        self.assertNotIn("UNAUTHENTICATED", kinds)
        self.assertIn("BURST", kinds)

    def test_near_duplicates_spread_over_time_are_still_flagged(self) -> None:
        # An attacker who paces the injection defeats the burst test. Records
        # generated from one template still resemble each other.
        history = self._genuine_history()
        history += [
            sign_record(
                _payload(900.0, band="band-2400"),
                self.key,
                node_id=NODE,
                now=self.base + timedelta(hours=index + 3),
            )
            for index in range(3)
        ]
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertFalse(report.trustworthy)
        self.assertIn("NEAR_DUPLICATE", {f.kind for f in report.findings})
        self.assertIn("NEAR_DUPLICATE_RECORDS_SUGGEST_TEMPLATED_INJECTION", report.reasons)

    def test_records_are_ordered_before_the_burst_test(self) -> None:
        # A store that returns rows out of order must not hide a burst.
        history = self._genuine_history(4)
        burst_at = self.base + timedelta(minutes=200)
        burst = [
            sign_record(
                _payload(700.0 + index, band=f"band-{7000 + index}"),
                self.key,
                node_id=NODE,
                now=burst_at + timedelta(seconds=index),
            )
            for index in range(DEFAULT_BURST_THRESHOLD)
        ]
        shuffled = [burst[1], history[0], burst[2], history[1], burst[0], *history[2:]]
        report = self.auditor.audit(shuffled, self.public, node_id=NODE)
        self.assertIn("BURST", {f.kind for f in report.findings})

    def test_empty_history_is_not_silently_trusted_as_clean(self) -> None:
        report = self.auditor.audit([], self.public, node_id=NODE)
        self.assertEqual(report.records, 0)
        self.assertEqual(report.findings, ())

    def test_report_serializes(self) -> None:
        report = self.auditor.audit(self._genuine_history(), self.public, node_id=NODE)
        json.dumps(report.to_dict())


class BoundedInfluenceTests(unittest.TestCase):
    def test_a_dominating_record_is_flagged(self) -> None:
        # One extreme record among a tight cluster is exactly the leverage
        # MemMorph buys with a small injection.
        verdict = bounded_influence([10.0, 10.5, 9.5, 10.2, 900.0])
        self.assertFalse(verdict.within_bound)
        self.assertEqual(verdict.most_influential_index, 4)
        self.assertIn("A_SINGLE_RECORD_MOVES_THE_CONCLUSION_TOO_FAR", verdict.reasons)

    def test_a_well_evidenced_conclusion_passes(self) -> None:
        verdict = bounded_influence([10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.4, 9.7])
        self.assertTrue(verdict.within_bound)
        self.assertIn("NO_SINGLE_RECORD_DOMINATES", verdict.reasons)

    def test_too_few_records_cannot_bound_influence(self) -> None:
        # Fail closed: with two values each accounts for half the conclusion.
        verdict = bounded_influence([10.0, 11.0])
        self.assertFalse(verdict.within_bound)
        self.assertIn("TOO_FEW_RECORDS_TO_BOUND_INFLUENCE", verdict.reasons)

    def test_identical_values_have_no_dominant_record(self) -> None:
        verdict = bounded_influence([5.0, 5.0, 5.0, 5.0])
        self.assertTrue(verdict.within_bound)
        self.assertEqual(verdict.max_influence, 0.0)

    def test_influence_falls_as_evidence_grows(self) -> None:
        # The same outlier matters less when more genuine records surround it.
        small = bounded_influence([10.0, 10.0, 10.0, 40.0])
        large = bounded_influence([10.0] * 40 + [40.0])
        self.assertLess(large.max_influence, small.max_influence)

    def test_verdict_serializes(self) -> None:
        json.dumps(bounded_influence([1.0, 2.0, 3.0, 4.0]).to_dict())


if __name__ == "__main__":
    unittest.main()
