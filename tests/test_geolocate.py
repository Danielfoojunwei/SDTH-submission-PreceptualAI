from __future__ import annotations

import math
import unittest

from kval_node.geolocate import (
    EARTH_RADIUS_M,
    EmitterLocator,
    GeolocationError,
    NodePosition,
    PathLossModel,
    SignalSighting,
    classify_residuals,
    geometry_ratio,
    robust_residual_scale,
)

# A roughly equilateral triangle of nodes, ~200 m on a side.
NODE_A = NodePosition("a", 1.3000, 103.8000)
NODE_B = NodePosition("b", 1.3020, 103.8000)
NODE_C = NodePosition("c", 1.3010, 103.8020)
TRIANGLE = [NODE_A, NODE_B, NODE_C]


def sighting(node: str, dbm: float, *, trusted: bool = True) -> SignalSighting:
    return SignalSighting(node, dbm, trusted)


class GeometryTests(unittest.TestCase):
    def test_triangle_has_usable_geometry(self) -> None:
        from kval_node.geolocate import to_local_xy

        points, _lat, _lon = to_local_xy(TRIANGLE)
        self.assertGreater(geometry_ratio(points), 0.1)

    def test_collinear_nodes_have_near_zero_ratio(self) -> None:
        from kval_node.geolocate import to_local_xy

        line = [
            NodePosition("a", 1.3000, 103.8000),
            NodePosition("b", 1.3010, 103.8000),
            NodePosition("c", 1.3020, 103.8000),
        ]
        points, _lat, _lon = to_local_xy(line)
        self.assertLess(geometry_ratio(points), 0.01)


class WeightedCentroidTests(unittest.TestCase):
    def test_estimate_is_pulled_toward_the_strongest_node(self) -> None:
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)], TRIANGLE
        )
        self.assertEqual(estimate.method, "weighted_centroid")
        # Much closer to A than to B or C.
        self.assertLess(abs(estimate.latitude_deg - NODE_A.latitude_deg), 0.0005)
        self.assertLess(abs(estimate.longitude_deg - NODE_A.longitude_deg), 0.0005)

    def test_equal_signals_give_the_geometric_centre(self) -> None:
        estimate = EmitterLocator().locate(
            [sighting("a", -60.0), sighting("b", -60.0), sighting("c", -60.0)], TRIANGLE
        )
        expected_lat = sum(p.latitude_deg for p in TRIANGLE) / 3
        self.assertAlmostEqual(estimate.latitude_deg, expected_lat, places=5)

    def test_structural_limit_is_always_stated(self) -> None:
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)], TRIANGLE
        )
        self.assertIn("CENTROID_CANNOT_PLACE_AN_EMITTER_OUTSIDE_THE_NODE_HULL", estimate.reasons)
        self.assertIn("NO_CALIBRATED_PATH_LOSS_MODEL_SO_CENTROID_ONLY", estimate.reasons)

    def test_uncertainty_is_reported_and_positive(self) -> None:
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)], TRIANGLE
        )
        self.assertGreater(estimate.uncertainty_radius_m, 0.0)


class RefusalTests(unittest.TestCase):
    def test_two_nodes_is_refused(self) -> None:
        with self.assertRaisesRegex(GeolocationError, "at least 3"):
            EmitterLocator().locate([sighting("a", -45.0), sighting("b", -70.0)], TRIANGLE)

    def test_collinear_geometry_is_refused(self) -> None:
        line = [
            NodePosition("a", 1.3000, 103.8000),
            NodePosition("b", 1.3010, 103.8000),
            NodePosition("c", 1.3020, 103.8000),
        ]
        with self.assertRaisesRegex(GeolocationError, "collinear"):
            EmitterLocator().locate(
                [sighting("a", -45.0), sighting("b", -70.0), sighting("c", -72.0)], line
            )

    def test_untrusted_clock_nodes_are_excluded(self) -> None:
        # Excluding one leaves two, which is below the minimum.
        with self.assertRaisesRegex(GeolocationError, "at least 3"):
            EmitterLocator().locate(
                [sighting("a", -45.0), sighting("b", -70.0, trusted=False), sighting("c", -72.0)],
                TRIANGLE,
            )

    def test_node_without_a_declared_position_is_excluded(self) -> None:
        with self.assertRaisesRegex(GeolocationError, "at least 3"):
            EmitterLocator().locate(
                [sighting("a", -45.0), sighting("b", -70.0), sighting("stranger", -50.0)],
                [NODE_A, NODE_B],
            )


class PathLossTests(unittest.TestCase):
    def test_distance_grows_as_power_falls(self) -> None:
        model = PathLossModel(reference_power_dbm=-40.0, path_loss_exponent=2.0)
        near = model.distance_m(-50.0)
        far = model.distance_m(-80.0)
        self.assertLess(near, far)
        # Free-space: 20 dB down is 10x the distance.
        self.assertAlmostEqual(model.distance_m(-60.0) / model.distance_m(-40.0), 10.0, places=3)

    def test_exponent_is_range_checked(self) -> None:
        with self.assertRaises(ValueError):
            PathLossModel.from_dict({"reference_power_dbm": -40.0, "path_loss_exponent": 0.1})

    def test_trilateration_runs_when_calibrated(self) -> None:
        model = PathLossModel(reference_power_dbm=-40.0, path_loss_exponent=2.5)
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)],
            TRIANGLE,
            path_loss=model,
        )
        self.assertEqual(estimate.method, "trilateration")
        self.assertNotIn("NO_CALIBRATED_PATH_LOSS_MODEL_SO_CENTROID_ONLY", estimate.reasons)
        self.assertGreater(estimate.uncertainty_radius_m, 0.0)

    def test_result_serializes(self) -> None:
        import json

        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)], TRIANGLE
        )
        json.dumps(estimate.to_dict())


# --- NLOS identification (Hua, Yin, Lu, Zhang, Li, Sensors 18(9):2991, 2018) ---

# Five nodes around a common area, ~300 m across, with usable geometry.
PENTAGON = [
    NodePosition("n1", 1.3000, 103.8000),
    NodePosition("n2", 1.3030, 103.8005),
    NodePosition("n3", 1.3025, 103.8032),
    NodePosition("n4", 1.2998, 103.8030),
    NodePosition("n5", 1.3012, 103.8016),
]

EMITTER_LAT, EMITTER_LON = 1.3014, 103.8012

MODEL = PathLossModel(reference_power_dbm=-40.0, path_loss_exponent=2.5)


def metres_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    east = math.radians(lon2 - lon1) * EARTH_RADIUS_M * cos_lat
    north = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    return math.hypot(east, north)


def received_power(node: NodePosition, *, obstruction_db: float = 0.0) -> float:
    """Power a node would measure under the log-distance model.

    This is the model's own forward direction, so a sighting built this way is
    exactly consistent with what the locator inverts. ``obstruction_db`` adds
    the excess loss a blocked path introduces — the physical effect the paper's
    residual test exists to identify.
    """
    distance = metres_between(node.latitude_deg, node.longitude_deg, EMITTER_LAT, EMITTER_LON)
    free = MODEL.reference_power_dbm - 10.0 * MODEL.path_loss_exponent * math.log10(
        distance / MODEL.reference_distance_m
    )
    return free - obstruction_db


class ResidualScaleTests(unittest.TestCase):
    def test_scale_is_not_inflated_by_the_outlier_it_must_find(self) -> None:
        # The whole point of a robust scale: a plain stddev would let the
        # blocked anchor raise its own threshold and escape detection.
        clean = [2.0, -1.0, 0.5, -2.0, 1.0]
        with_outlier = clean + [400.0]
        self.assertLess(robust_residual_scale(with_outlier), 2.0 * robust_residual_scale(clean))

    def test_perfectly_consistent_residuals_do_not_give_a_zero_scale(self) -> None:
        # A zero scale would make every sighting an infinite-sigma outlier.
        self.assertGreater(robust_residual_scale([0.0, 0.0, 0.0, 0.0]), 0.0)

    def test_empty_input_is_handled(self) -> None:
        self.assertGreater(robust_residual_scale([]), 0.0)


class ResidualClassificationTests(unittest.TestCase):
    def test_a_longer_measured_range_is_read_as_obstruction(self) -> None:
        # Negative residual: the node measured further than the fleet implies,
        # which is what extra attenuation looks like.
        findings = classify_residuals(
            ["a", "b", "c", "d"], [1.0, -0.5, 0.5, -300.0]
        )
        blocked = next(f for f in findings if f.node_id == "d")
        self.assertEqual(blocked.classification, "NLOS_SUSPECTED")
        self.assertTrue(all(f.classification == "CONSISTENT" for f in findings if f.node_id != "d"))

    def test_a_shorter_measured_range_cannot_be_obstruction(self) -> None:
        # Positive residual: heard more strongly than the geometry allows.
        # Obstruction cannot do this, so it gets its own louder label.
        findings = classify_residuals(["a", "b", "c", "d"], [1.0, -0.5, 0.5, 300.0])
        self.assertEqual(
            next(f for f in findings if f.node_id == "d").classification, "IMPLAUSIBLY_STRONG"
        )

    def test_a_shared_offset_is_calibration_not_a_blocked_node(self) -> None:
        # Every node off by the same amount means the path-loss model is
        # mis-calibrated. Blaming a node for it would exclude a good anchor.
        findings = classify_residuals(["a", "b", "c", "d"], [50.0, 51.0, 49.5, 50.5])
        self.assertTrue(all(f.classification == "CONSISTENT" for f in findings))

    def test_mismatched_lengths_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            classify_residuals(["a", "b"], [1.0])


class NlosRejectionTests(unittest.TestCase):
    def _sightings(
        self, obstructed: dict[str, float] | None = None, *, nodes: list[NodePosition] = PENTAGON
    ) -> list[SignalSighting]:
        obstructed = obstructed or {}
        return [
            SignalSighting(
                node.node_id,
                received_power(node, obstruction_db=obstructed.get(node.node_id, 0.0)),
                True,
            )
            for node in nodes
        ]

    def test_clean_measurements_lose_no_nodes(self) -> None:
        estimate = EmitterLocator().locate(self._sightings(), PENTAGON, path_loss=MODEL)
        self.assertEqual(estimate.excluded, ())
        self.assertEqual(len(estimate.contributing_nodes), 5)

    def test_clean_measurements_recover_the_true_position(self) -> None:
        # Confirms the fixture is sound before asking what obstruction does to it.
        estimate = EmitterLocator().locate(self._sightings(), PENTAGON, path_loss=MODEL)
        error = metres_between(
            estimate.latitude_deg, estimate.longitude_deg, EMITTER_LAT, EMITTER_LON
        )
        self.assertLess(error, 1.0)

    def test_an_obstructed_node_is_identified_and_excluded(self) -> None:
        estimate = EmitterLocator().locate(
            self._sightings({"n3": 18.0}), PENTAGON, path_loss=MODEL
        )
        self.assertNotIn("n3", estimate.contributing_nodes)
        self.assertEqual([f.node_id for f in estimate.excluded], ["n3"])
        self.assertEqual(estimate.excluded[0].classification, "NLOS_SUSPECTED")
        self.assertIn("EXCLUDED_NLOS_SUSPECTED_SIGHTING", estimate.reasons)

    def test_rejection_recovers_the_accuracy_obstruction_destroyed(self) -> None:
        # The paper's actual claim: identifying the NLOS anchor and dropping it
        # beats solving with it included.
        obstructed = self._sightings({"n3": 18.0})
        with_rejection = EmitterLocator().locate(obstructed, PENTAGON, path_loss=MODEL)
        # The counterfactual needs the consistency gate relaxed too, or the
        # naive solve is refused before it can be compared against.
        without = EmitterLocator(reject_nlos=False, max_residual_ratio=math.inf).locate(
            obstructed, PENTAGON, path_loss=MODEL
        )
        rejected_error = metres_between(
            with_rejection.latitude_deg, with_rejection.longitude_deg, EMITTER_LAT, EMITTER_LON
        )
        naive_error = metres_between(
            without.latitude_deg, without.longitude_deg, EMITTER_LAT, EMITTER_LON
        )
        self.assertEqual(without.excluded, ())
        self.assertIn("NLOS_REJECTION_DISABLED_BY_CONFIGURATION", without.reasons)
        self.assertLess(rejected_error, naive_error)

    def test_a_report_stronger_than_propagation_allows_is_flagged_distinctly(self) -> None:
        # Negative obstruction is not physical attenuation; it is what a
        # repeater, a strong reflection or a fabricated report looks like.
        estimate = EmitterLocator().locate(
            self._sightings({"n4": -20.0}), PENTAGON, path_loss=MODEL
        )
        self.assertNotIn("n4", estimate.contributing_nodes)
        self.assertEqual(estimate.excluded[0].classification, "IMPLAUSIBLY_STRONG")
        self.assertIn("EXCLUDED_SIGHTING_STRONGER_THAN_PROPAGATION_ALLOWS", estimate.reasons)

    def test_three_anchors_always_fit_so_no_test_is_possible(self) -> None:
        # The structural limit: a three-anchor system has as many equations as
        # unknowns, so it fits any measurements exactly, blocked or not.
        estimate = EmitterLocator().locate(
            [
                SignalSighting("a", -45.0, True),
                SignalSighting("b", -71.0, True),
                SignalSighting("c", -73.0, True),
            ],
            TRIANGLE,
            path_loss=MODEL,
        )
        self.assertIn("TOO_FEW_ANCHORS_TO_IDENTIFY_A_BLOCKED_ONE", estimate.reasons)
        self.assertEqual(estimate.excluded, ())

    def test_rejection_stops_at_the_minimum_anchor_count(self) -> None:
        # Five anchors survive exactly one blocked node: dropping to four is
        # the floor, because three would fit anything. The result says so.
        estimate = EmitterLocator().locate(
            self._sightings({"n3": 18.0}), PENTAGON, path_loss=MODEL
        )
        self.assertEqual(len(estimate.contributing_nodes), 4)
        self.assertIn("AT_MINIMUM_ANCHOR_COUNT_NO_FURTHER_REJECTION_POSSIBLE", estimate.reasons)

    def test_two_blocked_anchors_are_refused_not_answered_badly(self) -> None:
        # No single removal reconciles the rest, so nothing is excluded and the
        # surviving set still disagrees. The least-squares point in this case is
        # kilometres wrong; returning it with a warning string would be a
        # fail-open, so no position is returned at all.
        with self.assertRaisesRegex(GeolocationError, "aperture"):
            EmitterLocator().locate(
                self._sightings({"n2": 18.0, "n3": 22.0}), PENTAGON, path_loss=MODEL
            )

    def test_the_refused_case_really_would_have_been_badly_wrong(self) -> None:
        # Establishes that the refusal above is protecting against a real
        # error rather than discarding a usable answer.
        naive = EmitterLocator(reject_nlos=False, max_residual_ratio=math.inf).locate(
            self._sightings({"n2": 18.0, "n3": 22.0}),
            PENTAGON,
            path_loss=MODEL,
        )
        self.assertGreater(
            metres_between(naive.latitude_deg, naive.longitude_deg, EMITTER_LAT, EMITTER_LON),
            1000.0,
        )

    def test_surviving_anchors_all_report_their_agreement(self) -> None:
        estimate = EmitterLocator().locate(
            self._sightings({"n3": 18.0}), PENTAGON, path_loss=MODEL
        )
        self.assertEqual(
            {f.node_id for f in estimate.residuals}, set(estimate.contributing_nodes)
        )
        self.assertTrue(all(f.classification == "CONSISTENT" for f in estimate.residuals))

    def test_a_miscalibrated_model_is_not_blamed_on_one_node(self) -> None:
        # Every anchor equally wrong is a model problem. Excluding anchors one
        # by one would destroy a good fleet to flatter a bad model.
        wrong_model = PathLossModel(reference_power_dbm=-40.0, path_loss_exponent=3.5)
        estimate = EmitterLocator().locate(self._sightings(), PENTAGON, path_loss=wrong_model)
        self.assertEqual(estimate.excluded, ())
        self.assertIn("ANCHORS_DISAGREE_BUT_NO_SINGLE_ONE_EXPLAINS_IT", estimate.reasons)

    def test_centroid_states_that_rejection_is_unavailable(self) -> None:
        # Without ranges there are no residuals, so the test cannot run and the
        # result must not imply otherwise.
        estimate = EmitterLocator().locate(self._sightings({"n3": 18.0}), PENTAGON)
        self.assertIn("NLOS_REJECTION_NEEDS_RANGES_SO_IS_UNAVAILABLE_FOR_CENTROID", estimate.reasons)
        self.assertEqual(estimate.excluded, ())
        self.assertEqual(estimate.residuals, ())

    def test_excluded_findings_serialize(self) -> None:
        import json

        estimate = EmitterLocator().locate(
            self._sightings({"n3": 18.0}), PENTAGON, path_loss=MODEL
        )
        payload = json.loads(json.dumps(estimate.to_dict()))
        self.assertEqual(payload["excluded"][0]["node_id"], "n3")
        self.assertEqual(len(payload["residuals"]), 4)


if __name__ == "__main__":
    unittest.main()
