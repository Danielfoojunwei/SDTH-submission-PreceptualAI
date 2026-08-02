"""Graph-structured evidence propagation across the fabric.

Graph framing from Zhang, Lam, Shen, Wang and Li, "Dynamic spectrum access for
Internet-of-Things with graph neural network and deep Q-network", Ad Hoc
Networks 163:103596, 2024. See docs/DSA_RESEARCH_MAPPING.md for what is and is
not carried across.
"""

from __future__ import annotations

import json
import unittest

from kval_node.evidence import BeliefMass
from kval_node.geolocate import NodePosition
from kval_node.topology import (
    EvidencePropagator,
    FleetGraph,
    TopologyError,
    assess_fragility,
    discount,
)

BUSY = BeliefMass(0.0, 0.9, 0.1)
CLEAR = BeliefMass(0.9, 0.0, 0.1)
IGNORANT = BeliefMass(0.0, 0.0, 1.0)

# a - b - c - d, a chain, so hop distance is unambiguous.
CHAIN = FleetGraph.from_pairs([("a", "b"), ("b", "c"), ("c", "d")])

# A triangle plus a pendant: a-b-c fully connected, d hanging off c.
TRIANGLE_PLUS = FleetGraph.from_pairs([("a", "b"), ("b", "c"), ("a", "c"), ("c", "d")])


class FleetGraphTests(unittest.TestCase):
    def test_links_are_undirected(self) -> None:
        self.assertIn("b", CHAIN.neighbours("a"))
        self.assertIn("a", CHAIN.neighbours("b"))

    def test_duplicate_links_collapse(self) -> None:
        graph = FleetGraph.from_pairs([("a", "b"), ("b", "a"), ("a", "b")])
        self.assertEqual(len(graph.links), 1)

    def test_self_link_is_refused(self) -> None:
        with self.assertRaises(TopologyError):
            FleetGraph.from_pairs([("a", "a")])

    def test_malformed_pair_is_refused(self) -> None:
        with self.assertRaises(TopologyError):
            FleetGraph.from_pairs([("a", "b", "c")])

    def test_hop_distances_follow_the_chain(self) -> None:
        distances = CHAIN.hop_distances("a")
        self.assertEqual(distances, {"a": 0, "b": 1, "c": 2, "d": 3})

    def test_hop_distance_takes_the_shortcut(self) -> None:
        # In the triangle, a reaches c directly rather than through b.
        self.assertEqual(TRIANGLE_PLUS.hop_distances("a")["c"], 1)

    def test_unknown_source_is_refused(self) -> None:
        with self.assertRaises(TopologyError):
            CHAIN.hop_distances("nobody")

    def test_partition_is_reported_as_separate_components(self) -> None:
        split = FleetGraph.from_pairs([("a", "b"), ("c", "d")])
        self.assertEqual(split.components(), (("a", "b"), ("c", "d")))

    def test_serializes(self) -> None:
        json.dumps(CHAIN.to_dict())


class RangeGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        # Three nodes on a line, ~110 m apart each.
        self.positions = [
            NodePosition("a", 1.3000, 103.8000),
            NodePosition("b", 1.3010, 103.8000),
            NodePosition("c", 1.3020, 103.8000),
            NodePosition("far", 1.4000, 103.8000),
        ]

    def test_only_pairs_within_range_are_linked(self) -> None:
        graph = FleetGraph.from_positions(self.positions, link_range_m=150.0)
        self.assertIn("b", graph.neighbours("a"))
        self.assertNotIn("c", graph.neighbours("a"))

    def test_a_node_out_of_everyone_s_range_still_appears(self) -> None:
        # Dropping it would silently shrink the fleet the operator declared.
        graph = FleetGraph.from_positions(self.positions, link_range_m=150.0)
        self.assertIn("far", graph.nodes)
        self.assertEqual(graph.neighbours("far"), ())

    def test_a_wider_range_links_more(self) -> None:
        wide = FleetGraph.from_positions(self.positions, link_range_m=300.0)
        self.assertIn("c", wide.neighbours("a"))

    def test_absurd_range_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            FleetGraph.from_positions(self.positions, link_range_m=0.0)


class DiscountTests(unittest.TestCase):
    def test_weakening_moves_mass_to_ignorance_not_to_the_opposite(self) -> None:
        weakened = discount(BUSY, 0.5)
        self.assertAlmostEqual(weakened.busy, 0.45)
        self.assertEqual(weakened.clear, 0.0)
        self.assertGreater(weakened.unknown, BUSY.unknown)

    def test_full_retention_is_a_no_op(self) -> None:
        self.assertAlmostEqual(discount(BUSY, 1.0).busy, BUSY.busy)

    def test_zero_retention_is_total_ignorance(self) -> None:
        self.assertAlmostEqual(discount(BUSY, 0.0).unknown, 1.0)


class PropagationTests(unittest.TestCase):
    def test_a_node_that_cannot_sense_still_gets_a_view(self) -> None:
        # The fabric's whole premise: d has no radio of its own, and a saw the
        # band. d must end up believing the band is busy, not knowing nothing.
        results = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})}
        self.assertGreater(results["d"].belief.busy, 0.0)
        self.assertFalse(results["d"].first_hand)
        self.assertEqual(results["d"].hops_to_nearest_observer, 3)

    def test_inherited_belief_is_weaker_the_further_away_it_came_from(self) -> None:
        results = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})}
        self.assertGreater(results["a"].belief.busy, results["b"].belief.busy)
        self.assertGreater(results["b"].belief.busy, results["c"].belief.busy)
        self.assertGreater(results["c"].belief.busy, results["d"].belief.busy)

    def test_a_distant_node_is_never_more_certain_than_the_observer(self) -> None:
        results = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})}
        for node in ("b", "c", "d"):
            self.assertLessEqual(results[node].belief.busy, results["a"].belief.busy)
            self.assertGreaterEqual(results[node].belief.unknown, results["a"].belief.unknown)

    def test_inheritance_is_labelled_so_it_is_never_mistaken_for_sensing(self) -> None:
        results = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})}
        self.assertIn("OBSERVED_DIRECTLY", results["a"].reasons)
        self.assertIn("INHERITED_FROM_NEIGHBOURS_AT_2_HOPS", results["c"].reasons)

    def test_belief_does_not_circulate_around_a_cycle(self) -> None:
        # a, b and c form a triangle. One observation must not be counted three
        # times just because the graph has a loop.
        cycle = {r.node_id: r for r in EvidencePropagator(TRIANGLE_PLUS).propagate({"a": BUSY})}
        line = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})}
        self.assertAlmostEqual(cycle["b"].belief.busy, line["b"].belief.busy, places=9)
        self.assertEqual(cycle["b"].observers_used, ("a",))

    def test_independent_observers_reinforce(self) -> None:
        # Two genuinely separate observations should make a node more sure than
        # one does. This is the case that must still work after cycle
        # suppression, or the graph would be useless.
        one = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})}
        two = {
            r.node_id: r
            for r in EvidencePropagator(CHAIN).propagate({"a": BUSY, "d": BUSY})
        }
        self.assertGreater(two["b"].belief.busy, one["b"].belief.busy)
        self.assertEqual(two["b"].observers_used, ("a", "d"))

    def test_conflicting_neighbours_are_flagged_not_averaged(self) -> None:
        results = {
            r.node_id: r
            for r in EvidencePropagator(CHAIN).propagate({"a": BUSY, "b": CLEAR})
        }
        self.assertGreater(results["a"].conflict, 0.4)
        self.assertIn("NEIGHBOURS_MATERIALLY_DISAGREE_ABOUT_THIS_BAND", results["a"].reasons)

    def test_agreeing_neighbours_report_no_conflict(self) -> None:
        results = {
            r.node_id: r
            for r in EvidencePropagator(CHAIN).propagate({"a": BUSY, "d": BUSY})
        }
        self.assertEqual(results["a"].conflict, 0.0)
        self.assertNotIn("NEIGHBOURS_MATERIALLY_DISAGREE_ABOUT_THIS_BAND", results["a"].reasons)

    def test_discounting_caps_how_much_conflict_distance_can_express(self) -> None:
        # Worth pinning: an inherited disagreement can never look as sharp as a
        # first-hand one, so a threshold tuned for direct sources under-fires.
        near = EvidencePropagator(CHAIN).propagate({"a": BUSY, "b": CLEAR})
        far = EvidencePropagator(CHAIN).propagate({"a": BUSY, "d": CLEAR})
        self.assertGreater(
            next(r for r in near if r.node_id == "a").conflict,
            next(r for r in far if r.node_id == "a").conflict,
        )

    def test_an_unreachable_node_ends_at_ignorance_not_at_clear(self) -> None:
        # Failing to "clear" would let a partition look like an all-clear.
        split = FleetGraph.from_pairs([("a", "b"), ("c", "d")])
        results = {r.node_id: r for r in EvidencePropagator(split).propagate({"a": BUSY})}
        self.assertEqual(results["c"].belief.unknown, 1.0)
        self.assertEqual(results["c"].belief.clear, 0.0)
        self.assertIn("UNREACHABLE_FROM_ANY_OBSERVER_SO_TOTAL_IGNORANCE", results["c"].reasons)

    def test_no_observations_leaves_everyone_ignorant(self) -> None:
        results = EvidencePropagator(CHAIN).propagate({})
        self.assertTrue(all(r.belief.unknown == 1.0 for r in results))
        self.assertTrue(
            all("NO_OBSERVER_WITHIN_REACH_SO_TOTAL_IGNORANCE" in r.reasons for r in results)
        )

    def test_an_ignorant_observation_contributes_nothing(self) -> None:
        results = {r.node_id: r for r in EvidencePropagator(CHAIN).propagate({"a": IGNORANT})}
        self.assertEqual(results["b"].belief.unknown, 1.0)

    def test_hop_limit_stops_propagation(self) -> None:
        results = {
            r.node_id: r
            for r in EvidencePropagator(CHAIN, max_hops=1).propagate({"a": BUSY})
        }
        self.assertGreater(results["b"].belief.busy, 0.0)
        self.assertEqual(results["d"].belief.unknown, 1.0)

    def test_an_observation_from_outside_the_graph_is_refused(self) -> None:
        with self.assertRaises(TopologyError):
            EvidencePropagator(CHAIN).propagate({"stranger": BUSY})

    def test_every_node_gets_a_result(self) -> None:
        results = EvidencePropagator(CHAIN).propagate({"a": BUSY})
        self.assertEqual({r.node_id for r in results}, set(CHAIN.nodes))

    def test_result_serializes(self) -> None:
        json.dumps([r.to_dict() for r in EvidencePropagator(CHAIN).propagate({"a": BUSY})])


class FragilityTests(unittest.TestCase):
    def test_a_chain_s_middle_nodes_are_single_points_of_failure(self) -> None:
        report = assess_fragility(CHAIN)
        self.assertEqual(set(report.articulation_points), {"b", "c"})
        self.assertIn("LOSS_OF_A_SINGLE_NODE_WOULD_PARTITION_THE_FABRIC", report.reasons)

    def test_a_ring_tolerates_any_single_loss(self) -> None:
        ring = FleetGraph.from_pairs([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
        report = assess_fragility(ring)
        self.assertEqual(report.articulation_points, ())
        self.assertIn("FABRIC_TOLERATES_THE_LOSS_OF_ANY_SINGLE_NODE", report.reasons)

    def test_the_hinge_of_two_clusters_is_identified(self) -> None:
        # c is the only route between the a-b cluster and the d-e cluster.
        graph = FleetGraph.from_pairs(
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "c"), ("a", "c")]
        )
        self.assertEqual(assess_fragility(graph).articulation_points, ("c",))

    def test_an_existing_partition_is_reported(self) -> None:
        split = FleetGraph.from_pairs([("a", "b"), ("c", "d")])
        report = assess_fragility(split)
        self.assertEqual(len(report.components), 2)
        self.assertIn("FABRIC_IS_ALREADY_PARTITIONED", report.reasons)

    def test_isolated_nodes_are_called_out(self) -> None:
        graph = FleetGraph.from_positions(
            [
                NodePosition("a", 1.3000, 103.8000),
                NodePosition("b", 1.3010, 103.8000),
                NodePosition("lonely", 1.5000, 103.8000),
            ],
            link_range_m=200.0,
        )
        report = assess_fragility(graph)
        self.assertEqual(report.isolated, ("lonely",))
        self.assertIn("NODES_WITH_NO_LINKS_CANNOT_CONTRIBUTE_OR_INHERIT", report.reasons)

    def test_report_serializes(self) -> None:
        json.dumps(assess_fragility(CHAIN).to_dict())


if __name__ == "__main__":
    unittest.main()
