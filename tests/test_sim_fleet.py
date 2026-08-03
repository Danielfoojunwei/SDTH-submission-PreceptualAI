"""The simulator's own tests. These test the harness, not the node.

Two things need pinning. First that the harness is *directionally* right — rain
must hurt Ka more than UHF, or a run tells you nothing. Second that the boundary
holds: `sim` may import `kval_node`, never the reverse.

Nothing here is evidence about radio. A passing suite means the harness
degrades links in a defensible direction and drives the real allocator; it does
not mean any number resembles a measurement.
"""

from __future__ import annotations

import ast
import pathlib
import unittest
from datetime import datetime, timezone

from sim.fleet import demands_for, run_fleet, run_vehicle, slices_for, summarise
from sim.vehicles import CONDITIONS, VEHICLE_TYPES, Bearer, Condition, degraded_link

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
ROOT = pathlib.Path(__file__).resolve().parent.parent


class BoundaryTests(unittest.TestCase):
    def test_no_production_module_imports_the_simulator(self) -> None:
        offenders = []
        for path in sorted((ROOT / "src" / "kval_node").glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(name == "sim" or name.startswith("sim.") for name in names):
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "production code must not reach the simulator")

    def test_the_simulator_drives_the_real_allocator(self) -> None:
        # If this ever stops importing the production symbol, the harness has
        # quietly become a model of the allocator rather than a driver of it.
        import sim.fleet as fleet

        self.assertEqual(fleet.plan_allocation.__module__, "kval_node.allocation")


class PropagationDirectionTests(unittest.TestCase):
    def test_rain_hurts_high_frequencies_more(self) -> None:
        ka = Bearer("ka", "w0", "payload", 20.0, 50.0, 100.0)
        uhf = Bearer("uhf", "w1", "payload", 0.4, 50.0, 100.0)
        self.assertGreater(ka.rain_attenuation_db(40.0), uhf.rain_attenuation_db(40.0) * 100)

    def test_no_rain_is_no_attenuation(self) -> None:
        ka = Bearer("ka", "w0", "payload", 20.0, 50.0, 100.0)
        self.assertEqual(ka.rain_attenuation_db(0.0), 0.0)

    def test_terrain_spares_a_satcom_bearer(self) -> None:
        masked = Condition("masked", terrain_blockage=0.8)
        satcom = Bearer("sat", "w0", "payload", 20.0, 50.0, 100.0, terrain_sensitivity=0.05)
        mesh = Bearer("mesh", "w1", "payload", 5.8, 50.0, 100.0, terrain_sensitivity=1.0)
        self.assertLess(masked.degradation_db(satcom), masked.degradation_db(mesh))

    def test_a_jammer_only_touches_bearers_inside_its_span(self) -> None:
        jam = Condition("ism", jammer_db=30.0, jammer_low_ghz=2.0, jammer_high_ghz=6.0)
        inside = Bearer("in", "w0", "c2_cnpc", 2.4, 5.0, 50.0)
        outside = Bearer("out", "w1", "c2_cnpc", 0.9, 5.0, 50.0)
        self.assertTrue(jam.jams(inside))
        self.assertFalse(jam.jams(outside))
        self.assertEqual(jam.degradation_db(outside), 0.0)

    def test_a_link_past_its_margin_is_reported_down_not_merely_slow(self) -> None:
        # A link that cannot pass a probe is a different fact from a slow link,
        # and the node has to be able to tell them apart.
        killed = Condition("kill", jammer_db=40.0, jammer_low_ghz=0.0, jammer_high_ghz=30.0)
        bearer = Bearer("b", "w0", "c2_cnpc", 2.4, 5.0, 50.0)
        reading = degraded_link(bearer, killed)
        self.assertEqual(reading["up"], 0.0)
        self.assertIsNone(reading["rtt_ms"])


class HealthMappingTests(unittest.TestCase):
    def test_health_is_evidence_completeness_not_signal_strength(self) -> None:
        # The first version of this harness fed signal strength in as health,
        # which discounted the already-degraded capacity a second time and made
        # every outcome pessimistic. Health mirrors LinkObservation.confidence,
        # which counts *present fields*.
        vehicle = VEHICLE_TYPES[0]
        heavy = next(c for c in CONDITIONS if c.name == "heavy_rain")
        for item in slices_for(vehicle, heavy):
            if item.capacity_kbps > 0.0:
                self.assertAlmostEqual(item.health, 1.0, places=6)

    def test_a_downed_bearer_is_offered_with_zero_capacity_not_omitted(self) -> None:
        vehicle = VEHICLE_TYPES[0]
        killed = Condition("kill", jammer_db=50.0, jammer_low_ghz=0.0, jammer_high_ghz=30.0)
        slices = slices_for(vehicle, killed)
        self.assertEqual(len(slices), len(vehicle.bearers))
        self.assertTrue(all(item.usable_kbps() == 0.0 for item in slices))


class FleetRunTests(unittest.TestCase):
    def test_clear_air_carries_every_flow_on_every_type(self) -> None:
        # If a platform cannot meet its own flow table in clear air, the fixture
        # is wrong and every condition result below it is uninterpretable.
        for vehicle in VEHICLE_TYPES:
            clear = next(c for c in CONDITIONS if c.name == "clear_air")
            outcome, _ = run_vehicle(vehicle.type_id, vehicle, clear, now=NOW)
            self.assertEqual(outcome.flows_refused, (), f"{vehicle.type_id}: {outcome.refusal_reasons}")

    def test_rain_takes_payload_before_it_takes_command(self) -> None:
        # The asymmetry that matters: command links sit low in frequency and
        # survive a cell that removes a Ka downlink.
        heavy = next(c for c in CONDITIONS if c.name == "heavy_rain")
        multirotor = next(v for v in VEHICLE_TYPES if v.type_id == "multirotor")
        outcome, _ = run_vehicle("mr", multirotor, heavy, now=NOW)
        self.assertTrue(outcome.c2_carried)
        self.assertIn("video", outcome.flows_refused)

    def test_wideband_jamming_takes_command_and_says_so(self) -> None:
        wide = next(c for c in CONDITIONS if c.name == "wideband_jammed")
        for vehicle in VEHICLE_TYPES:
            outcome, _ = run_vehicle(vehicle.type_id, vehicle, wide, now=NOW)
            self.assertFalse(outcome.c2_carried, vehicle.type_id)
            self.assertIn("c2-primary", outcome.flows_refused)

    def test_a_ground_vehicle_is_unbothered_by_rain(self) -> None:
        heavy = next(c for c in CONDITIONS if c.name == "heavy_rain")
        ground = next(v for v in VEHICLE_TYPES if v.type_id == "ground")
        outcome, _ = run_vehicle("gv", ground, heavy, now=NOW)
        self.assertEqual(outcome.flows_refused, ())

    def test_a_run_is_deterministic(self) -> None:
        first, _ = run_fleet()
        second, _ = run_fleet()
        self.assertEqual([o.to_dict() for o in first], [o.to_dict() for o in second])

    def test_the_summary_states_what_it_is_not(self) -> None:
        outcomes, _ = run_fleet()
        note = summarise(outcomes)["note"]
        self.assertIn("fabricated", note)
        self.assertIn("not evidence", note.replace("no number here is evidence", "not evidence"))

    def test_every_type_meets_every_condition(self) -> None:
        outcomes, plans = run_fleet()
        self.assertEqual(len(outcomes), len(VEHICLE_TYPES) * len(CONDITIONS))
        self.assertEqual(len(plans), len(outcomes))

    def test_demands_come_from_the_vehicle_type(self) -> None:
        for vehicle in VEHICLE_TYPES:
            demands = demands_for(vehicle)
            self.assertTrue(demands)
            self.assertTrue(any(d.link_role == "c2_cnpc" for d in demands))


if __name__ == "__main__":
    unittest.main()
