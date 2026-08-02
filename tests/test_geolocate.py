from __future__ import annotations

import unittest

from kval_node.geolocate import (
    EmitterLocator,
    GeolocationError,
    NodePosition,
    PathLossModel,
    SignalSighting,
    geometry_ratio,
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


if __name__ == "__main__":
    unittest.main()
