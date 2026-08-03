"""The allocator's claim is survivability, so that is what these pin.

A plan that carries a flow is only correct if the flow survives losing the
single slice carrying the most of it. Every test here is written against that
property rather than against a particular placement, because the placement is an
implementation detail and the survivability is the product.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from kval_node.allocation import (
    DEFAULT_QUANTUM_KBPS,
    AllocationError,
    Demand,
    FlowTable,
    Slice,
    plan_allocation,
)

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)


def slices(count: int, *, role: str = "c2_cnpc", capacity: float = 500.0, health: float = 1.0, prefix: str = "s"):
    return [Slice(f"{prefix}{i}", f"p{prefix}{i}", role, capacity, health) for i in range(count)]


class SurvivabilityTests(unittest.TestCase):
    def test_a_carried_flow_survives_losing_its_largest_slice(self) -> None:
        plan = plan_allocation([Demand("c2", "c2_cnpc", 200.0, minimum_slices=3)], slices(4), now=NOW)
        flow = plan.flows[0]
        self.assertTrue(flow.carried, flow.reasons)
        total = sum(p.kbps for p in flow.placements)
        largest = max(p.kbps for p in flow.placements)
        self.assertGreaterEqual(total - largest + 1e-9, 200.0)
        self.assertGreaterEqual(len(flow.placements), 3)

    def test_the_headroom_cost_of_diversity_is_actually_paid(self) -> None:
        # Surviving one loss across n slices costs n/(n-1) of the rate. If the
        # plan quietly allocated only the bare rate it would look carried and
        # fail on the first loss.
        plan = plan_allocation([Demand("c2", "c2_cnpc", 300.0, minimum_slices=3)], slices(3), now=NOW)
        flow = plan.flows[0]
        self.assertTrue(flow.carried, flow.reasons)
        self.assertGreaterEqual(sum(p.kbps for p in flow.placements) + 1e-6, 300.0 * 3 / 2)

    def test_a_flow_that_cannot_survive_one_loss_is_refused_not_half_carried(self) -> None:
        thin = [Slice("a", "pa", "c2_cnpc", 60.0, 1.0), Slice("b", "pb", "c2_cnpc", 60.0, 1.0)]
        plan = plan_allocation([Demand("c2", "c2_cnpc", 200.0)], thin, now=NOW)
        flow = plan.flows[0]
        self.assertFalse(flow.carried)
        self.assertEqual(flow.placements, ())
        self.assertIn("LOSING_THE_LARGEST_SLICE_WOULD_DROP_THE_FLOW_BELOW_ITS_RATE", flow.reasons)

    def test_refused_capacity_is_handed_back_for_other_flows(self) -> None:
        # A refused flow must not squat on capacity a flow that *can* be
        # carried needs. Otherwise one impossible demand starves the rest.
        pool = slices(3, capacity=200.0)
        plan = plan_allocation(
            [
                Demand("impossible", "c2_cnpc", 5000.0, minimum_slices=3),
                Demand("modest", "c2_cnpc", 100.0, minimum_slices=3),
            ],
            pool,
            now=NOW,
        )
        by_id = {flow.flow_id: flow for flow in plan.flows}
        self.assertFalse(by_id["impossible"].carried)
        self.assertTrue(by_id["modest"].carried, by_id["modest"].reasons)

    def test_no_single_placement_can_exceed_what_the_others_cover(self) -> None:
        # One fat slice must not swallow the flow: that is the single point of
        # failure the whole module exists to prevent.
        lopsided = [
            Slice("fat", "pf", "c2_cnpc", 10_000.0, 1.0),
            Slice("thin1", "pt1", "c2_cnpc", 500.0, 1.0),
            Slice("thin2", "pt2", "c2_cnpc", 500.0, 1.0),
        ]
        plan = plan_allocation([Demand("c2", "c2_cnpc", 300.0, minimum_slices=3)], lopsided, now=NOW)
        flow = plan.flows[0]
        self.assertTrue(flow.carried, flow.reasons)
        largest = max(p.kbps for p in flow.placements)
        self.assertLessEqual(largest, 300.0 + 1e-6)


class RoleSegregationTests(unittest.TestCase):
    def test_a_command_flow_is_never_placed_on_a_payload_slice(self) -> None:
        mixed = slices(1, role="c2_cnpc") + slices(4, role="payload", prefix="pay")
        plan = plan_allocation([Demand("c2", "c2_cnpc", 100.0)], mixed, now=NOW)
        flow = plan.flows[0]
        self.assertFalse(flow.carried)
        self.assertIn("FEWER_SLICES_IN_THIS_ROLE_THAN_THE_FLOW_REQUIRES_DIVERSITY", flow.reasons)

    def test_command_traffic_is_placed_before_payload(self) -> None:
        pool = slices(3, capacity=200.0)
        payload_pool = [Slice(f"q{i}", f"pq{i}", "payload", 200.0, 1.0) for i in range(3)]
        plan = plan_allocation(
            [
                Demand("bulk", "payload", 250.0, minimum_slices=3),
                Demand("command", "c2_cnpc", 250.0, minimum_slices=3),
            ],
            pool + payload_pool,
            now=NOW,
        )
        self.assertEqual(plan.flows[0].flow_id, "command")


class EvidenceTests(unittest.TestCase):
    def test_capacity_is_discounted_by_probe_confidence(self) -> None:
        weak = slices(3, capacity=400.0, health=0.25)
        self.assertAlmostEqual(weak[0].usable_kbps(), 100.0)
        plan = plan_allocation([Demand("c2", "c2_cnpc", 400.0, minimum_slices=3)], weak, now=NOW)
        self.assertFalse(plan.flows[0].carried)

    def test_no_slices_at_all_is_reported_not_silently_empty(self) -> None:
        plan = plan_allocation([Demand("c2", "c2_cnpc", 10.0)], [], now=NOW)
        self.assertIn("NO_MEASURED_SLICES_WERE_OFFERED", plan.reasons)
        self.assertFalse(plan.flows[0].carried)

    def test_a_plan_is_deterministic_across_runs(self) -> None:
        # Two nodes reading the same evidence must produce the same placement,
        # or a fleet cannot reason about what its peers are doing.
        first = plan_allocation([Demand("c2", "c2_cnpc", 300.0, minimum_slices=3)], slices(5), now=NOW)
        second = plan_allocation([Demand("c2", "c2_cnpc", 300.0, minimum_slices=3)], slices(5), now=NOW)
        self.assertEqual(first.to_dict()["flows"], second.to_dict()["flows"])

    def test_duplicate_slice_ids_are_refused(self) -> None:
        duped = [Slice("same", "p1", "c2_cnpc", 100.0, 1.0), Slice("same", "p2", "c2_cnpc", 100.0, 1.0)]
        with self.assertRaises(AllocationError):
            plan_allocation([Demand("c2", "c2_cnpc", 10.0)], duped, now=NOW)

    def test_a_flow_on_one_slice_is_not_expressible(self) -> None:
        with self.assertRaises(ValueError):
            Demand("c2", "c2_cnpc", 100.0, minimum_slices=1)


class FlowTableTests(unittest.TestCase):
    def body(self, **overrides) -> dict:
        raw = {
            "table_id": "flows-1",
            "created_at": (NOW - timedelta(hours=1)).isoformat(),
            "expires_at": (NOW + timedelta(hours=1)).isoformat(),
            "demands": [
                {"flow_id": "c2", "link_role": "c2_cnpc", "required_kbps": 64.0, "minimum_slices": 3}
            ],
        }
        raw.update(overrides)
        return raw

    def test_a_valid_table_parses(self) -> None:
        table = FlowTable.from_dict(self.body(), now=NOW)
        self.assertEqual(table.demands[0].minimum_slices, 3)

    def test_an_expired_table_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            FlowTable.from_dict(
                self.body(expires_at=(NOW - timedelta(minutes=1)).isoformat()), now=NOW
            )

    def test_a_table_with_no_demands_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            FlowTable.from_dict(self.body(demands=[]), now=NOW)

    def test_duplicate_flow_ids_are_refused(self) -> None:
        one = {"flow_id": "c2", "link_role": "c2_cnpc", "required_kbps": 64.0}
        with self.assertRaises(ValueError):
            FlowTable.from_dict(self.body(demands=[one, dict(one)]), now=NOW)


class QuantumTests(unittest.TestCase):
    def test_a_smaller_quantum_spreads_a_flow_more_thinly(self) -> None:
        coarse = plan_allocation(
            [Demand("c2", "c2_cnpc", 200.0, minimum_slices=2)], slices(8), quantum_kbps=100.0, now=NOW
        )
        fine = plan_allocation(
            [Demand("c2", "c2_cnpc", 200.0, minimum_slices=2)], slices(8), quantum_kbps=10.0, now=NOW
        )
        self.assertGreater(len(fine.flows[0].placements), len(coarse.flows[0].placements))

    def test_a_non_positive_quantum_is_refused(self) -> None:
        with self.assertRaises(AllocationError):
            plan_allocation([Demand("c2", "c2_cnpc", 10.0)], slices(3), quantum_kbps=0.0, now=NOW)

    def test_the_default_quantum_is_small_relative_to_a_command_flow(self) -> None:
        self.assertLess(DEFAULT_QUANTUM_KBPS, 64.0)


if __name__ == "__main__":
    unittest.main()
