"""Cohort formation, reputation, hierarchical aggregation, and DS fusion.

Mechanisms adapted from the Feng Li / Bowen Shen / Kwok-Yan Lam dynamic-spectrum
-access corpus; see docs/DSA_RESEARCH_MAPPING.md for the paper-by-paper mapping.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from kval_node.cohorts import (
    HierarchicalAggregator,
    ReputationLedger,
    excise_contributor,
    kmeans_cohorts,
)
from kval_node.dataset import FEATURE_NAMES
from kval_node.evidence import BeliefMass, combine_pair, fuse
from kval_node.federated import FeatureStatistics, FederatedUpdate
from kval_node.spectrum import BandStatus

NOW = datetime.now(timezone.utc)


def update(node: str, rtt_mean: float, *, count: int = 50, variance: float = 100.0) -> FederatedUpdate:
    stats = {name: FeatureStatistics(count, 0.0, float(count)) for name in FEATURE_NAMES}
    stats["rtt_ms"] = FeatureStatistics(count, rtt_mean, count * (rtt_mean**2 + variance))
    return FederatedUpdate(node, NOW, count, stats)


class CohortClusteringTests(unittest.TestCase):
    def test_distinct_environments_separate(self) -> None:
        urban = [update(f"urban{i}", 200.0 + i) for i in range(4)]
        rural = [update(f"rural{i}", 20.0 + i) for i in range(4)]
        cohorts = kmeans_cohorts(urban + rural, max_cohorts=2)
        self.assertEqual(len(cohorts), 2)
        for cohort in cohorts:
            prefixes = {u.node_id[:5] for u in cohort}
            self.assertEqual(len(prefixes), 1, "a cohort mixed two environments")

    def test_clustering_is_deterministic(self) -> None:
        nodes = [update(f"n{i}", 20.0 + i * 30) for i in range(8)]
        first = [[u.node_id for u in c] for c in kmeans_cohorts(nodes)]
        second = [[u.node_id for u in c] for c in kmeans_cohorts(nodes)]
        self.assertEqual(first, second)

    def test_tiny_fleet_is_one_cohort(self) -> None:
        self.assertEqual(len(kmeans_cohorts([update("a", 10.0), update("b", 20.0)])), 1)

    def test_empty_fleet_yields_no_cohorts(self) -> None:
        self.assertEqual(kmeans_cohorts([]), [])


class ReputationTests(unittest.TestCase):
    def test_agreement_raises_and_divergence_lowers(self) -> None:
        ledger = ReputationLedger()
        for _ in range(5):
            ledger.record_round(agreed=["good"], diverged=["liar"])
        self.assertGreater(ledger.score("good"), 0.9)
        self.assertLess(ledger.score("liar"), 0.5)

    def test_reputation_never_reaches_zero_so_recovery_is_possible(self) -> None:
        ledger = ReputationLedger()
        for _ in range(200):
            ledger.record_round(agreed=[], diverged=["unlucky"])
        self.assertGreaterEqual(ledger.score("unlucky"), ReputationLedger.MINIMUM_SCORE)
        for _ in range(50):
            ledger.record_round(agreed=["unlucky"], diverged=[])
        self.assertGreater(ledger.score("unlucky"), 0.9)

    def test_unknown_node_starts_neutral(self) -> None:
        self.assertEqual(ReputationLedger().score("newcomer"), 1.0)


class HierarchicalAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fleet = [update(f"urban{i}", 200.0 + i) for i in range(4)] + [
            update(f"rural{i}", 20.0 + i) for i in range(4)
        ]

    def test_produces_cohorts_and_a_global_tier(self) -> None:
        result = HierarchicalAggregator().aggregate(self.fleet)
        self.assertGreaterEqual(len(result.cohorts), 2)
        self.assertIsNotNone(result.global_aggregate)

    def test_aggregator_rotates_between_rounds(self) -> None:
        aggregator = HierarchicalAggregator()
        chosen = {aggregator.aggregate(self.fleet, round_index=r).aggregator_node for r in range(3)}
        self.assertGreater(len(chosen), 1, "the aggregator role did not rotate")

    def test_a_disreputable_node_is_not_chosen_to_aggregate(self) -> None:
        ledger = ReputationLedger()
        for _ in range(10):
            ledger.record_round(agreed=["good"], diverged=["bad"])
        aggregator = HierarchicalAggregator(reputation=ledger)
        self.assertNotEqual(aggregator.select_aggregator(["good", "bad"], round_index=0), "bad")

    def test_aggregation_tolerates_nodes_dropping_out(self) -> None:
        # Dropout resilience: a fleet that loses members between rounds must
        # still aggregate, not fail closed.
        full = HierarchicalAggregator().aggregate(self.fleet)
        reduced = HierarchicalAggregator().aggregate(self.fleet[:5])
        self.assertIsNotNone(full.global_aggregate)
        self.assertTrue(reduced.cohorts)

    def test_result_serializes(self) -> None:
        json.dumps(HierarchicalAggregator().aggregate(self.fleet).to_dict())


class UnlearningTests(unittest.TestCase):
    def test_excision_is_exact_and_reported(self) -> None:
        fleet = [update(f"n{i}", 20.0 + i) for i in range(4)]
        remaining, found = excise_contributor(fleet, "n2")
        self.assertTrue(found)
        self.assertEqual(len(remaining), 3)
        self.assertNotIn("n2", [u.node_id for u in remaining])

    def test_excising_an_absent_node_reports_false(self) -> None:
        fleet = [update("n0", 20.0)]
        remaining, found = excise_contributor(fleet, "ghost")
        self.assertFalse(found)
        self.assertEqual(len(remaining), 1)


class DempsterShaferTests(unittest.TestCase):
    """The distinction vote-counting cannot express: ignorance is not disbelief."""

    def test_blind_nodes_yield_ignorance_not_clear(self) -> None:
        blind = [BeliefMass.from_status(BandStatus.INSUFFICIENT_EVIDENCE, 0.0) for _ in range(3)]
        result = fuse(blind)
        self.assertEqual(result.unknown, 1.0)
        self.assertEqual(result.clear, 0.0)
        self.assertEqual(result.busy, 0.0)
        self.assertIn("NO_NODE_PRODUCED_COMMITTED_EVIDENCE", result.reasons)

    def test_agreeing_nodes_produce_strong_belief(self) -> None:
        result = fuse([BeliefMass.from_status(BandStatus.CLEAR, 1.0) for _ in range(3)])
        self.assertGreater(result.clear, 0.99)
        self.assertTrue(result.trustworthy)

    def test_jamming_reports_accumulate(self) -> None:
        result = fuse([BeliefMass.from_status(BandStatus.JAMMING_SUSPECTED, 1.0) for _ in range(3)])
        self.assertGreater(result.busy, 0.99)

    def test_irreconcilable_sources_are_flagged_not_normalised_away(self) -> None:
        result = fuse(
            [
                BeliefMass.from_status(BandStatus.JAMMING_SUSPECTED, 1.0),
                BeliefMass.from_status(BandStatus.CLEAR, 1.0),
            ]
        )
        self.assertFalse(result.trustworthy)
        self.assertIn("SOURCES_IRRECONCILABLY_CONFLICTED", result.reasons)

    def test_credibility_discounts_an_untrusted_source(self) -> None:
        # A low-credibility node claiming jamming cannot override a trusted
        # node reporting clear — the SSDF defence.
        result = fuse(
            [
                BeliefMass.from_status(BandStatus.JAMMING_SUSPECTED, 1.0, credibility=0.1),
                BeliefMass.from_status(BandStatus.CLEAR, 1.0, credibility=1.0),
            ]
        )
        self.assertGreater(result.clear, result.busy)

    def test_low_confidence_commits_little_mass(self) -> None:
        weak = BeliefMass.from_status(BandStatus.CLEAR, 0.1)
        strong = BeliefMass.from_status(BandStatus.CLEAR, 1.0)
        self.assertLess(weak.clear, strong.clear)
        self.assertGreater(weak.unknown, strong.unknown)

    def test_mass_always_sums_to_one(self) -> None:
        for status in BandStatus:
            for confidence in (0.0, 0.3, 1.0):
                mass = BeliefMass.from_status(status, confidence)
                self.assertAlmostEqual(mass.clear + mass.busy + mass.unknown, 1.0, places=9)

    def test_total_conflict_returns_ignorance_not_a_division_error(self) -> None:
        combined, conflict = combine_pair(BeliefMass(1.0, 0.0, 0.0), BeliefMass(0.0, 1.0, 0.0))
        self.assertEqual(conflict, 1.0)
        self.assertEqual(combined.unknown, 1.0)

    def test_ignorance_is_the_identity_element(self) -> None:
        evidence = BeliefMass.from_status(BandStatus.CLEAR, 0.8)
        combined, _conflict = combine_pair(evidence, BeliefMass(0.0, 0.0, 1.0))
        self.assertAlmostEqual(combined.clear, evidence.clear, places=9)

    def test_result_serializes(self) -> None:
        json.dumps(fuse([BeliefMass.from_status(BandStatus.CLEAR, 1.0)]).to_dict())


if __name__ == "__main__":
    unittest.main()
