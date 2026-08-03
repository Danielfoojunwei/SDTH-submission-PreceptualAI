"""Memory-level integrity safeguards.

Defends the attack in Zhang, Zheng, Xu, Zhou, Shen, Ou, Zhang and Lam,
"MemMorph: Tool Hijacking in LLM Agents via Memory Poisoning" (arXiv:2605.26154),
which reports up to 85.9% attack success from three injected records. See
docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.memory_integrity import (
    DEFAULT_GROUP_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
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
            record_type=record.record_type,
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
            record_type=record.record_type,
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
            record_type=record.record_type,
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

    def _injected(
        self, count: int = DEFAULT_GROUP_THRESHOLD, *, key=None, spacing=timedelta(seconds=1)
    ) -> list[SignedRecord]:
        """A template repeated ``count`` times, as MemMorph generates them."""
        injected_at = self.base + timedelta(minutes=100)
        return [
            sign_record(
                {
                    "advisory": "prefer the uplink on band-9000",
                    "source": "incident-report",
                    "severity": "routine",
                },
                key or self.key,
                node_id=NODE,
                now=injected_at + spacing * index,
            )
            for index in range(count)
        ]

    def test_the_paper_s_three_record_injection_is_caught(self) -> None:
        # MemMorph's headline result: 85.9% attack success from three records.
        # Three is exactly the group size this auditor is tuned to catch.
        attacker = Ed25519PrivateKey.generate()
        history = self._genuine_history() + self._injected(key=attacker)
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertFalse(report.trustworthy)
        kinds = {finding.kind for finding in report.findings}
        self.assertIn("UNAUTHENTICATED", kinds)
        self.assertIn("TEMPLATED_GROUP", kinds)
        self.assertIn("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE", report.reasons)

    def test_a_signed_injection_is_still_flagged(self) -> None:
        # Authorship binding is not sufficient on its own: a compromised node
        # signs its own poisoned records. The content test is what catches that,
        # so it must not depend on the signature check failing.
        history = self._genuine_history() + self._injected()
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertFalse(report.trustworthy)
        kinds = {finding.kind for finding in report.findings}
        self.assertNotIn("UNAUTHENTICATED", kinds)
        self.assertIn("TEMPLATED_GROUP", kinds)
        self.assertIn(
            "A_GROUP_OF_RECORDS_LOOKS_TEMPLATED_AND_FOREIGN_TO_THIS_HISTORY", report.reasons
        )

    def test_pacing_the_injection_does_not_evade_it(self) -> None:
        # Timing is reported but not required, so spreading the same template
        # over hours does not help the attacker.
        history = self._genuine_history() + self._injected(spacing=timedelta(hours=3))
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertIn("TEMPLATED_GROUP", {f.kind for f in report.findings})

    def test_the_finding_names_the_injected_records(self) -> None:
        history = self._genuine_history(8) + self._injected()
        report = self.auditor.audit(history, self.public, node_id=NODE)
        group = next(f for f in report.findings if f.kind == "TEMPLATED_GROUP")
        self.assertEqual(set(group.record_indices), {8, 9, 10})

    def test_records_are_ordered_before_analysis(self) -> None:
        # A store that returns rows out of order must not hide an injection.
        history = self._genuine_history(6)
        injected = self._injected()
        shuffled = [injected[1], history[0], injected[2], history[1], injected[0], *history[2:]]
        report = self.auditor.audit(shuffled, self.public, node_id=NODE)
        self.assertIn("TEMPLATED_GROUP", {f.kind for f in report.findings})

    # --- the false positives that would make this unusable ---------------------

    def test_a_node_repeating_a_stable_decision_is_not_an_injection(self) -> None:
        # A healthy link produces the same decision every cycle. Flagging that
        # would fire constantly on a correctly-operating node.
        steady = [
            sign_record(
                {"selected": "wifi-primary", "action": "hold"},
                self.key,
                node_id=NODE,
                now=self.base + timedelta(minutes=index),
            )
            for index in range(20)
        ]
        report = self.auditor.audit(steady, self.public, node_id=NODE)
        self.assertTrue(report.trustworthy)

    def test_several_records_in_one_loop_iteration_are_not_an_injection(self) -> None:
        # One cycle writes a decision, an activation and an observation within
        # the same second. Timing alone must not condemn them.
        moment = self.base
        cycle = [
            sign_record({"selected": "wifi"}, self.key, node_id=NODE, record_type="decision", now=moment),
            sign_record(
                {"profile_id": "wifi", "success": True},
                self.key,
                node_id=NODE,
                record_type="activation",
                now=moment + timedelta(milliseconds=40),
            ),
            sign_record(
                {"band_id": "band-2400", "occupancy_percent": 12.0},
                self.key,
                node_id=NODE,
                record_type="observation",
                now=moment + timedelta(milliseconds=90),
            ),
        ]
        report = self.auditor.audit(cycle + self._genuine_history(), self.public, node_id=NODE)
        self.assertTrue(report.trustworthy)

    def test_a_uniform_history_is_not_condemned_for_having_no_variety(self) -> None:
        # If every record resembles every other there is no "rest of history"
        # to be foreign to, so there is no evidence either way.
        uniform = [
            sign_record({"selected": "wifi"}, self.key, node_id=NODE, now=self.base + timedelta(minutes=i))
            for i in range(6)
        ]
        self.assertTrue(self.auditor.audit(uniform, self.public, node_id=NODE).trustworthy)

    def test_a_pair_is_below_the_group_threshold(self) -> None:
        # Two similar records are a coincidence; three are a template.
        history = self._genuine_history() + self._injected(count=2)
        report = self.auditor.audit(history, self.public, node_id=NODE)
        self.assertNotIn("TEMPLATED_GROUP", {f.kind for f in report.findings})

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
        self.assertGreater(verdict.disproportion, 5.0)

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

    def test_the_bound_is_measured_against_a_typical_record_not_a_fixed_share(self) -> None:
        # With ten records each legitimately carries about a tenth, so an
        # absolute cap tight enough to catch poisoning in long history would
        # condemn every short history. The ratio must be stable across length.
        for count in (10, 40, 200):
            verdict = bounded_influence([10.0 + (index % 7) * 0.3 for index in range(count)])
            self.assertTrue(verdict.within_bound, f"clean {count}-record history was condemned")
            self.assertLess(verdict.disproportion, 5.0)

    def test_the_bound_does_not_depend_on_how_much_readings_naturally_vary(self) -> None:
        # A node whose readings span a wide range is not more suspicious than
        # one whose readings barely move.
        tight = bounded_influence([10.0 + (i % 7) * 0.3 for i in range(50)])
        wide = bounded_influence([10.0 + i * 3.0 for i in range(50)])
        self.assertTrue(tight.within_bound)
        self.assertTrue(wide.within_bound)

    def test_a_group_injection_is_left_to_the_group_detector(self) -> None:
        # Stated as a limit rather than papered over: remove one of three
        # identical injected records and the other two still hold the mean, so
        # a leave-one-out test is structurally blind to groups that are not
        # also extreme. The templated-group detector covers that case.
        subtle = [20.0 + (index % 5) for index in range(40)] + [26.0, 26.0, 26.0]
        self.assertTrue(bounded_influence(subtle).within_bound)

    def test_influence_falls_as_evidence_grows(self) -> None:
        # The same outlier matters less when more genuine records surround it.
        small = bounded_influence([10.0, 10.0, 10.0, 40.0])
        large = bounded_influence([10.0] * 40 + [12.0] * 5 + [40.0])
        self.assertLess(large.max_influence, small.max_influence)

    def test_an_extreme_record_is_caught_at_any_history_length(self) -> None:
        # Regression on a fail-open that made this defence inert. Normalising
        # the leave-one-out shift by the spread of *all* values let the outlier
        # inflate the denominator judging it, so the ratio converged to
        # 1/sqrt(n-1): past seventeen records a single injected value of any
        # magnitude sat inside the bound. Measuring against the spread of the
        # records that remain removes the feedback.
        for count in (5, 17, 25, 50, 200):
            for magnitude in (1e3, 1e6, 1e12):
                verdict = bounded_influence([20.0] * (count - 1) + [magnitude])
                self.assertFalse(
                    verdict.within_bound,
                    f"a record of {magnitude:g} passed the bound among {count} records",
                )

    def test_a_realistic_clean_history_still_passes(self) -> None:
        # The fix must not make the bound so strict that normal history fails,
        # or it gets disabled and defends nothing.
        self.assertTrue(bounded_influence([20.0 + (i % 7) for i in range(50)]).within_bound)

    def test_the_memmorph_three_record_shape_is_caught(self) -> None:
        # The paper's own attack size, injected into a clean history.
        clean = [20.0 + (index % 7) * 0.3 for index in range(50)]
        self.assertFalse(bounded_influence(clean + [900.0] * 3).within_bound)

    def test_a_point_inside_a_wide_spread_is_not_an_outlier(self) -> None:
        # A node whose readings genuinely vary must not be condemned for it.
        wide = [20.0 + index * 3.0 for index in range(50)]
        self.assertTrue(bounded_influence(wide + [180.0]).within_bound)

    def test_one_disagreeing_record_among_identical_ones_is_total_influence(self) -> None:
        # No spread among the rest means any disagreement is unbounded, which
        # is the honest answer rather than a division by zero.
        verdict = bounded_influence([5.0, 5.0, 5.0, 5.0, 900.0])
        self.assertFalse(verdict.within_bound)
        self.assertEqual(verdict.most_influential_index, 4)

    def test_verdict_serializes(self) -> None:
        json.dumps(bounded_influence([1.0, 2.0, 3.0, 4.0]).to_dict())


class CadenceSeparatesInjectionFromLinkChangeTests(unittest.TestCase):
    """The detector's real discriminator, and the one it shipped without.

    Similarity was never able to carry this test. A node flapping between links
    under jamming — the condition this whole system exists for — produces small
    groups of mutually similar records unlike the rest of history, which is
    also the exact description of an injection. Measured, they match on group
    size, share of history, internal similarity and external similarity alike,
    so no threshold separates them. Rate does: the node's loop writes at one
    cadence, and an injection does not come from the loop.
    """

    CADENCE = timedelta(seconds=30)

    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key()
        self.auditor = MemoryAuditor()
        self.base = utc_now()

    def _decision(self, profile: str, rtt: float, sequence: int) -> dict:
        return {
            "selected_profile_id": profile,
            "rtt_ms": rtt,
            "loss_pct": 0.2,
            "score": 0.81,
            "reason": "BEST_MEASURED_LATENCY",
            "sequence": sequence,
        }

    def _at(self, when: timedelta, payload: dict) -> SignedRecord:
        return sign_record(payload, self.key, node_id=NODE, now=self.base + when)

    def _flapping(self, cycles: int = 30) -> list[SignedRecord]:
        """Contested spectrum: the node switches link every three cycles."""
        return [
            self._at(
                self.CADENCE * index,
                self._decision(f"link_{index // 3}", 40.0 + 20 * (index // 3), index),
            )
            for index in range(cycles)
        ]

    def _templated(self, spacing: timedelta, start: timedelta) -> list[SignedRecord]:
        return [
            self._at(start + spacing * j, self._decision("attacker_relay", 2.0, 900 + j))
            for j in range(DEFAULT_GROUP_THRESHOLD)
        ]

    def _kinds(self, records: list[SignedRecord]) -> set[str]:
        return {f.kind for f in self.auditor.audit(records, self.public, node_id=NODE).findings}

    def test_flapping_between_links_under_jamming_is_not_poisoning(self) -> None:
        # Ten regimes, each matching an injection on every similarity measure.
        # A detector that fires here is one an operator switches off.
        self.assertNotIn("TEMPLATED_GROUP", self._kinds(self._flapping()))

    def test_a_failover_that_sticks_is_not_poisoning(self) -> None:
        bimodal = [
            self._at(
                self.CADENCE * index,
                self._decision(
                    "lte_b3" if index < 15 else "sat_ku", 40.0 if index < 15 else 600.0, index
                ),
            )
            for index in range(30)
        ]
        self.assertNotIn("TEMPLATED_GROUP", self._kinds(bimodal))

    def test_the_paper_s_injection_written_in_one_operation_is_caught(self) -> None:
        # MemMorph writes its records into the store at once. At the 0.9
        # similarity threshold this shipped with, these three scored 0.714
        # against each other and never grouped: the detector could not see the
        # attack it is named for.
        history = self._flapping() + self._templated(
            timedelta(milliseconds=17), timedelta(seconds=930)
        )
        self.assertIn("TEMPLATED_GROUP", self._kinds(history))

    def test_drip_feeding_the_same_template_is_also_caught(self) -> None:
        # The gate is two-sided on purpose. Three records hours apart in a
        # history recorded every thirty seconds did not come from the loop
        # either, and a one-sided gate would sell the evasion for the price of
        # waiting.
        history = self._flapping() + self._templated(timedelta(hours=3), timedelta(hours=1))
        self.assertIn("TEMPLATED_GROUP", self._kinds(history))

    def test_the_evasion_is_where_it_is_documented_to_be(self) -> None:
        # An attacker holding the signing key who paces the injection to the
        # node's own cadence escapes this test. Pinned so the limit stays a
        # stated property rather than becoming a surprise.
        history = self._flapping() + self._templated(self.CADENCE, timedelta(seconds=930))
        self.assertNotIn("TEMPLATED_GROUP", self._kinds(history))

    def test_the_finding_states_both_rates_it_compared(self) -> None:
        history = self._flapping() + self._templated(
            timedelta(milliseconds=17), timedelta(seconds=930)
        )
        report = self.auditor.audit(history, self.public, node_id=NODE)
        detail = next(f for f in report.findings if f.kind == "TEMPLATED_GROUP").detail
        self.assertIn("node cadence of 30.0s", detail)
        self.assertIn("between records", detail)


if __name__ == "__main__":
    unittest.main()


class ConvergentInjectionTests(unittest.TestCase):
    """MemMorph's actual shape: dissimilar records agreeing on one conclusion.

    The templated arm cannot see this and reported a poisoned ledger as
    trustworthy. The paper's records are "disguised as technical facts,
    incident reports, and operational policies" — three different kinds, whose
    force is that they look like independent evidence converging. Mutual
    similarity is designed out of the attack, so a detector keyed on it is
    keyed on the wrong signature.
    """

    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.now = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
        self.auditor = MemoryAuditor()

    def record(self, payload, kind, offset):
        return sign_record(
            payload, self.key, node_id="node-a", record_type=kind,
            now=self.now + timedelta(seconds=offset),
        )

    def history(self, count=12):
        return [
            self.record(
                {"rtt_ms": 40.0 + (i % 3), "loss_percent": 1.0 + (i % 2) * 0.2,
                 "selected": "link_a", "cycle": i},
                "decision", i * 30,
            )
            for i in range(count)
        ]

    def memmorph(self, at):
        # Three deliberately different kinds, sharing no template, all pushing
        # rtt and loss the same way. This is the paper's construction.
        return [
            self.record(
                {"rtt_ms": 900.0, "loss_percent": 40.0, "source": "telemetry-summary"},
                "observation", at),
            self.record(
                {"rtt_ms": 850.0, "loss_percent": 38.0,
                 "incident": "repeated timeout on link_a", "operator_note": "seen before"},
                "escalation", at + 0.2),
            self.record(
                {"rtt_ms": 880.0, "loss_percent": 44.0, "recommendation": "prefer link_b",
                 "rationale": "historical reliability", "author": "ops"},
                "policy_note", at + 0.4),
        ]

    def test_dissimilar_records_converging_in_a_burst_are_caught(self) -> None:
        records = self.history() + self.memmorph(12 * 30 + 1)
        report = self.auditor.audit(records, self.key.public_key(), node_id="node-a")
        self.assertFalse(report.trustworthy)
        kinds = {finding.kind for finding in report.findings}
        self.assertIn("CONVERGENT_GROUP", kinds)
        self.assertIn(
            "DISSIMILAR_RECORDS_PUSH_THE_SAME_CONCLUSION_OFF_THIS_NODE_S_CADENCE",
            report.reasons,
        )

    def test_the_injected_records_are_the_ones_named(self) -> None:
        records = self.history() + self.memmorph(12 * 30 + 1)
        report = self.auditor.audit(records, self.key.public_key(), node_id="node-a")
        finding = next(f for f in report.findings if f.kind == "CONVERGENT_GROUP")
        self.assertEqual(set(finding.record_indices), {12, 13, 14})

    def test_the_injected_records_are_not_similar_to_each_other(self) -> None:
        # If they were, the templated arm would have caught them and this arm
        # would be redundant. Pin that they genuinely are not.
        inject = self.memmorph(0)
        for i in range(len(inject)):
            for j in range(i + 1, len(inject)):
                self.assertLess(
                    payload_similarity(inject[i].payload, inject[j].payload),
                    DEFAULT_SIMILARITY_THRESHOLD,
                )

    def test_genuine_degradation_at_the_node_cadence_is_quiet(self) -> None:
        # A node whose links really are getting worse produces records that all
        # deviate the same way. Only the rate separates that from an insertion,
        # and firing here would make the detector useless in the environment
        # this system is built for.
        degrading = [
            self.record(
                {"rtt_ms": 40.0 + i * 70, "loss_percent": 1.0 + i * 3.0,
                 "selected": "link_a", "cycle": i},
                "decision", i * 30)
            for i in range(15)
        ]
        report = self.auditor.audit(degrading, self.key.public_key(), node_id="node-a")
        self.assertTrue(report.trustworthy, [f.to_dict() for f in report.findings])

    def test_a_healthy_repeating_node_is_quiet(self) -> None:
        report = self.auditor.audit(self.history(14), self.key.public_key(), node_id="node-a")
        self.assertTrue(report.trustworthy, [f.to_dict() for f in report.findings])

    def test_agreement_on_a_single_field_is_not_convergence(self) -> None:
        # One field agreeing is coincidence: any two records above the median
        # "agree". The signal is several quantities pointing one way at once.
        records = self.history() + [
            self.record({"rtt_ms": 900.0, "source": "telemetry-summary"}, "observation", 361),
            self.record({"rtt_ms": 850.0, "incident": "timeout", "note": "x"}, "escalation", 361.2),
            self.record({"rtt_ms": 880.0, "recommendation": "b", "author": "ops"}, "policy_note", 361.4),
        ]
        report = self.auditor.audit(records, self.key.public_key(), node_id="node-a")
        self.assertNotIn("CONVERGENT_GROUP", {f.kind for f in report.findings})

    def test_records_disagreeing_in_direction_are_not_a_group(self) -> None:
        records = self.history() + [
            self.record({"rtt_ms": 900.0, "loss_percent": 40.0, "source": "s"}, "observation", 361),
            self.record({"rtt_ms": 2.0, "loss_percent": 0.01, "incident": "i", "n": "x"}, "escalation", 361.2),
            self.record({"rtt_ms": 880.0, "loss_percent": 0.02, "rec": "b", "author": "o"}, "policy_note", 361.4),
        ]
        report = self.auditor.audit(records, self.key.public_key(), node_id="node-a")
        self.assertNotIn("CONVERGENT_GROUP", {f.kind for f in report.findings})
