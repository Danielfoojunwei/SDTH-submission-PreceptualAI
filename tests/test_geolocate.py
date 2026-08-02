from __future__ import annotations

import math
import unittest

from dataclasses import dataclass

from kval_node.geolocate import (
    EARTH_RADIUS_M,
    MAX_PLAUSIBLE_PATH_LOSS_EXPONENT,
    MIN_CALIBRATION_POINTS,
    MIN_PLAUSIBLE_PATH_LOSS_EXPONENT,
    CalibrationObservation,
    EmitterLocator,
    GeolocationError,
    NodePosition,
    PathLossCalibrator,
    PathLossModel,
    SignalSighting,
    SurveyedReference,
    classify_residuals,
    environment_band,
    geometry_ratio,
    robust_residual_scale,
    slant_range_m,
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


# --- path-loss calibration from surveyed references --------------------------------

REFERENCE = SurveyedReference(
    reference_id="beacon-1",
    latitude_deg=1.3000,
    longitude_deg=103.8000,
    survey_uncertainty_m=0.5,
    survey_provenance="OPERATOR_SURVEYED",
)

#: A reference surveyed to centimetres, which is what a short-range calibration
#: needs: two metres from a beacon known to half a metre, the survey error *is*
#: the measurement, and the fit excludes the point for saying so.
PRECISE_REFERENCE = SurveyedReference(
    reference_id="beacon-1",
    latitude_deg=1.3000,
    longitude_deg=103.8000,
    survey_uncertainty_m=0.05,
    survey_provenance="SURVEY_CONTROL_POINT",
)

#: Distances spanning 1.6 decades, which is what a fleet spread around a
#: surveyed beacon actually looks like.
SPREAD_M = (50.0, 150.0, 400.0, 1000.0, 2000.0)


def observer_at(
    distance_m: float, *, index: int = 0, bearing_deg: float = 0.0, altitude_m: float = 0.0
) -> NodePosition:
    """A node ``distance_m`` from ``REFERENCE`` on the given bearing."""
    north = distance_m * math.cos(math.radians(bearing_deg))
    east = distance_m * math.sin(math.radians(bearing_deg))
    cos_lat = math.cos(math.radians(REFERENCE.latitude_deg))
    return NodePosition(
        f"cal{index}",
        REFERENCE.latitude_deg + math.degrees(north / EARTH_RADIUS_M),
        REFERENCE.longitude_deg + math.degrees(east / (EARTH_RADIUS_M * cos_lat)),
        altitude_m,
    )


def calibration_set(
    distances=SPREAD_M,
    *,
    exponent: float = 2.5,
    reference_power_dbm: float = -40.0,
    reference: SurveyedReference = REFERENCE,
    noise: tuple[float, ...] | None = None,
) -> list[CalibrationObservation]:
    """Measurements a surveyed reference would produce under a known model.

    This is the model's forward direction, so a clean set is exactly what the
    fit has to invert — which is what makes recovery of the input parameters a
    meaningful check rather than a tautology about least squares.
    """
    observations = []
    for index, distance in enumerate(distances):
        power = reference_power_dbm - 10.0 * exponent * math.log10(distance)
        if noise is not None:
            power += noise[index]
        observations.append(
            CalibrationObservation(reference, observer_at(distance, index=index), power)
        )
    return observations


class SurveyedReferenceTests(unittest.TestCase):
    def test_a_derived_position_cannot_be_declared_as_a_survey(self) -> None:
        # The vocabulary is closed so that "estimated" has no spelling that
        # passes for surveyed ground truth.
        with self.assertRaises(ValueError):
            SurveyedReference("r", 1.3, 103.8, 1.0, "ESTIMATED_BY_THIS_NODE")

    def test_survey_uncertainty_cannot_be_omitted(self) -> None:
        # No default: a caller must state how well the position is known,
        # because that number is what rejects an estimate offered as a survey.
        with self.assertRaises(TypeError):
            SurveyedReference("r", 1.3, 103.8, survey_provenance="OPERATOR_SURVEYED")

    def test_latitude_is_range_checked(self) -> None:
        with self.assertRaises(ValueError):
            SurveyedReference("r", 91.0, 103.8, 1.0, "OPERATOR_SURVEYED")

    def test_from_dict_refuses_an_emitter_estimate(self) -> None:
        # The realistic accident: this module's own output piped back into its
        # own input. Fitting against it would tune the model on its own errors.
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)], TRIANGLE
        )
        payload = estimate.to_dict()
        payload.update(
            {
                "reference_id": "beacon-1",
                "survey_uncertainty_m": 1.0,
                "survey_provenance": "OPERATOR_SURVEYED",
            }
        )
        with self.assertRaisesRegex(ValueError, "derived"):
            SurveyedReference.from_dict(payload)

    def test_from_dict_refuses_a_document_carrying_a_derived_schema(self) -> None:
        # Belt and braces on the key check: renaming a field must not open the
        # loop the field names close.
        with self.assertRaisesRegex(ValueError, "derived"):
            SurveyedReference.from_dict(
                {
                    "schema": "kval-track-prediction/v1",
                    "reference_id": "beacon-1",
                    "latitude_deg": 1.3,
                    "longitude_deg": 103.8,
                    "survey_uncertainty_m": 1.0,
                    "survey_provenance": "OPERATOR_SURVEYED",
                }
            )

    def test_from_dict_accepts_a_survey_document(self) -> None:
        reference = SurveyedReference.from_dict(
            {
                "reference_id": "beacon-1",
                "latitude_deg": 1.3,
                "longitude_deg": 103.8,
                "altitude_m": 12.0,
                "survey_uncertainty_m": 0.25,
                "survey_provenance": "SURVEY_CONTROL_POINT",
            }
        )
        self.assertEqual(reference.reference_id, "beacon-1")
        self.assertEqual(reference.altitude_m, 12.0)


class CalibrationInputTests(unittest.TestCase):
    """The surveyed-only rule, enforced where a caller would otherwise slip."""

    def test_an_emitter_estimate_cannot_be_used_as_a_reference(self) -> None:
        # An Estimate has latitude_deg and longitude_deg, so duck typing would
        # let the circular case straight through if the type were not checked.
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)], TRIANGLE
        )
        with self.assertRaises(TypeError):
            CalibrationObservation(estimate, observer_at(500.0), -80.0)

    def test_a_roster_node_position_cannot_be_used_as_a_reference(self) -> None:
        # A roster position is surveyed, but it is a *receiver*. Using one as
        # the emitter would fit the model against the wrong end of the path.
        with self.assertRaises(TypeError):
            CalibrationObservation(NODE_A, observer_at(500.0), -80.0)

    def test_a_duck_typed_position_cannot_be_used_as_a_reference(self) -> None:
        @dataclass(frozen=True)
        class LooksLikeAPosition:
            reference_id: str = "beacon-1"
            latitude_deg: float = 1.3
            longitude_deg: float = 103.8
            altitude_m: float = 0.0
            survey_uncertainty_m: float = 0.5
            survey_provenance: str = "OPERATOR_SURVEYED"

        with self.assertRaises(TypeError):
            CalibrationObservation(LooksLikeAPosition(), observer_at(500.0), -80.0)

    def test_the_observer_must_be_a_roster_position(self) -> None:
        with self.assertRaises(TypeError):
            CalibrationObservation(REFERENCE, REFERENCE, -80.0)

    def test_the_fitter_refuses_anything_that_is_not_a_calibration_observation(self) -> None:
        with self.assertRaises(TypeError):
            PathLossCalibrator().fit([(REFERENCE, observer_at(500.0), -80.0)])


class SlantRangeTests(unittest.TestCase):
    def test_horizontal_distance_is_recovered(self) -> None:
        self.assertAlmostEqual(slant_range_m(REFERENCE, observer_at(1000.0)), 1000.0, places=3)

    def test_altitude_is_not_ignored(self) -> None:
        # A node above a reference is further from it than the map says, and
        # using the ground distance would shift the fit's x axis by half a
        # decade here — a whole environment band on its own.
        aloft = observer_at(100.0, altitude_m=300.0)
        self.assertAlmostEqual(slant_range_m(REFERENCE, aloft), math.hypot(100.0, 300.0), places=3)


class PathLossFitTests(unittest.TestCase):
    def test_the_fit_recovers_the_parameters_that_generated_the_data(self) -> None:
        fit = PathLossCalibrator().fit(calibration_set(exponent=2.8, reference_power_dbm=-35.0))
        self.assertTrue(fit.fitted)
        self.assertAlmostEqual(fit.path_loss_exponent, 2.8, places=9)
        self.assertAlmostEqual(fit.reference_power_dbm, -35.0, places=9)
        self.assertEqual(fit.reference_id, "beacon-1")

    def test_the_fitted_model_drives_the_locator(self) -> None:
        # The point of the whole exercise: a model nobody measured by hand
        # produces the same position an operator-calibrated one does.
        fit = PathLossCalibrator().fit(
            calibration_set(
                exponent=MODEL.path_loss_exponent,
                reference_power_dbm=MODEL.reference_power_dbm,
            )
        )
        sightings = [SignalSighting(node.node_id, received_power(node), True) for node in PENTAGON]
        estimate = EmitterLocator().locate(sightings, PENTAGON, path_loss=fit.model)
        self.assertEqual(estimate.method, "trilateration")
        error = metres_between(
            estimate.latitude_deg, estimate.longitude_deg, EMITTER_LAT, EMITTER_LON
        )
        self.assertLess(error, 1.0)

    def test_the_fitted_model_round_trips_through_the_signed_document_loader(self) -> None:
        # The fit's ``model`` block is keyed for PathLossModel.from_dict, so an
        # operator can sign an accepted fit into the same path-loss document the
        # hand-measured route uses, with no transcription step to get wrong.
        fit = PathLossCalibrator().fit(calibration_set())
        loaded = PathLossModel.from_dict(fit.to_dict()["model"])
        self.assertAlmostEqual(loaded.path_loss_exponent, 2.5, places=4)
        self.assertAlmostEqual(loaded.reference_power_dbm, -40.0, places=3)

    def test_the_residual_and_both_standard_errors_are_reported(self) -> None:
        # Least squares always returns numbers. These are what say whether the
        # numbers are evidence.
        fit = PathLossCalibrator().fit(calibration_set(noise=(2.0, -1.5, 0.5, 1.0, -2.0)))
        self.assertGreater(fit.residual_rms_db, 0.0)
        self.assertGreater(fit.exponent_standard_error, 0.0)
        self.assertGreater(fit.reference_power_standard_error_db, 0.0)

    def test_the_residual_is_reported_as_a_range_error_an_operator_can_act_on(self) -> None:
        clean = PathLossCalibrator().fit(calibration_set())
        scattered = PathLossCalibrator().fit(
            calibration_set(noise=(4.0, -4.0, 4.0, -4.0, 4.0))
        )
        self.assertAlmostEqual(clean.range_error_factor, 1.0, places=6)
        self.assertGreater(scattered.range_error_factor, 1.0)

    def test_the_sample_count_and_every_exclusion_are_reported(self) -> None:
        fit = PathLossCalibrator().fit(calibration_set())
        self.assertEqual(fit.samples, len(SPREAD_M))
        self.assertEqual(fit.excluded_samples, 0)

    def test_a_fit_at_the_minimum_sample_count_says_so(self) -> None:
        fit = PathLossCalibrator().fit(calibration_set())
        self.assertIn(
            "AT_THE_MINIMUM_SAMPLE_COUNT_SO_ONE_MEASUREMENT_MOVES_THE_RESIDUAL", fit.reasons
        )

    def test_a_narrow_but_usable_span_is_flagged_rather_than_hidden(self) -> None:
        fit = PathLossCalibrator().fit(
            calibration_set((200.0, 300.0, 500.0, 700.0, 1000.0), exponent=3.0)
        )
        self.assertTrue(fit.fitted)
        self.assertLess(fit.distance_span_decades, 1.0)
        self.assertIn("DISTANCE_SPAN_IS_NARROW_SO_THE_EXPONENT_IS_WEAKLY_CONSTRAINED", fit.reasons)

    def test_the_fit_never_claims_to_have_identified_the_place(self) -> None:
        # The exponent-to-environment mapping is indicative. Two different
        # environments can produce the same exponent, and the result says so.
        fit = PathLossCalibrator().fit(calibration_set())
        self.assertIn(
            "ENVIRONMENT_BAND_IS_INDICATIVE_NOT_AN_IDENTIFICATION_OF_THE_PLACE", fit.reasons
        )
        self.assertIn(
            "FIT_DESCRIBES_THIS_ENVIRONMENT_ONLY_AND_DOES_NOT_TRANSFER_TO_A_PEER", fit.reasons
        )

    def test_results_serialize_without_infinity_or_nan(self) -> None:
        import json

        fitted = json.loads(json.dumps(PathLossCalibrator().fit(calibration_set()).to_dict()))
        refused = json.loads(json.dumps(PathLossCalibrator().fit([]).to_dict()))
        self.assertEqual(fitted["schema"], "kval-path-loss-fit/v1")
        self.assertTrue(fitted["fitted"])
        self.assertFalse(refused["fitted"])
        self.assertIsNone(refused["model"])
        self.assertIsNone(refused["residual_rms_db"])


class PathLossFitRefusalTests(unittest.TestCase):
    def test_too_few_measurements_are_refused(self) -> None:
        fit = PathLossCalibrator().fit(calibration_set(SPREAD_M[:4]))
        self.assertFalse(fit.fitted)
        self.assertIsNone(fit.model)
        self.assertIn(
            f"NEED_AT_LEAST_{MIN_CALIBRATION_POINTS}_SURVEYED_MEASUREMENTS_TO_FIT", fit.reasons
        )

    def test_measurements_at_one_radius_cannot_give_a_slope(self) -> None:
        # The degeneracy: a ring of nodes all 1 km from the beacon constrains
        # nothing about how power falls with distance, however many there are.
        ring = [
            CalibrationObservation(
                REFERENCE,
                observer_at(1000.0, index=index, bearing_deg=bearing),
                -40.0 - 25.0 * math.log10(1000.0),
            )
            for index, bearing in enumerate((0.0, 72.0, 144.0, 216.0, 288.0))
        ]
        fit = PathLossCalibrator().fit(ring)
        self.assertFalse(fit.fitted)
        self.assertIn("DISTANCE_SPAN_TOO_NARROW_TO_SEPARATE_A_SLOPE_FROM_NOISE", fit.reasons)

    def test_a_narrow_cluster_of_distances_is_refused(self) -> None:
        fit = PathLossCalibrator().fit(calibration_set((950.0, 975.0, 1000.0, 1025.0, 1050.0)))
        self.assertFalse(fit.fitted)
        self.assertIn("DISTANCE_SPAN_TOO_NARROW_TO_SEPARATE_A_SLOPE_FROM_NOISE", fit.reasons)
        self.assertLess(fit.distance_span_decades, 0.1)

    def test_a_non_finite_measurement_refuses_the_whole_set(self) -> None:
        # Dropping it and fitting the rest would be the flattering choice: a
        # receiver reporting a non-finite power is broken, so nothing else it
        # reported in that batch is trustworthy either.
        observations = calibration_set()
        observations[2] = CalibrationObservation(
            REFERENCE, observer_at(400.0, index=2), float("nan")
        )
        fit = PathLossCalibrator().fit(observations)
        self.assertFalse(fit.fitted)
        self.assertIn("NON_FINITE_RECEIVED_POWER_IN_THE_CALIBRATION_SET", fit.reasons)

    def test_two_reference_emitters_cannot_share_one_fit(self) -> None:
        # The intercept is that emitter's own radiated power. Two emitters have
        # two intercepts, and one fit through both trades slope against the
        # difference: a weaker beacon further out reads as a steeper
        # environment.
        other = SurveyedReference("beacon-2", 1.31, 103.81, 0.5, "OPERATOR_SURVEYED")
        observations = calibration_set() + calibration_set(
            (300.0, 900.0), reference=other
        )
        fit = PathLossCalibrator().fit(observations)
        self.assertFalse(fit.fitted)
        self.assertIn("OBSERVATIONS_MIX_MORE_THAN_ONE_SURVEYED_REFERENCE_EMITTER", fit.reasons)

    def test_an_observer_standing_on_the_reference_is_refused(self) -> None:
        observations = calibration_set()
        observations[0] = CalibrationObservation(
            REFERENCE, observer_at(0.0, index=0), -20.0
        )
        fit = PathLossCalibrator().fit(observations)
        self.assertFalse(fit.fitted)
        self.assertIn("AN_OBSERVER_IS_AT_THE_SURVEYED_REFERENCE_SO_NO_DISTANCE_EXISTS", fit.reasons)

    def test_an_implausibly_shallow_exponent_is_refused_and_not_clamped(self) -> None:
        # A fitted 0.8 means the measurements are wrong — a mislabelled
        # reference, a distance in the wrong unit. Clamping it to 1.5 would
        # hand back a fabricated number wearing a plausible value.
        fit = PathLossCalibrator().fit(calibration_set(exponent=0.8))
        self.assertFalse(fit.fitted)
        self.assertIsNone(fit.model)
        self.assertIn("FITTED_EXPONENT_OUTSIDE_PHYSICALLY_PLAUSIBLE_RANGE", fit.reasons)
        self.assertAlmostEqual(fit.path_loss_exponent, 0.8, places=9)

    def test_an_implausibly_steep_exponent_is_refused(self) -> None:
        # The powers an exponent of 7 implies are themselves absurd, which is
        # the point: propagation is not exotic, the data is wrong.
        fit = PathLossCalibrator().fit(
            calibration_set(
                (2.0, 4.0, 8.0, 16.0, 32.0),
                exponent=7.0,
                reference_power_dbm=-30.0,
                reference=PRECISE_REFERENCE,
            )
        )
        self.assertFalse(fit.fitted)
        self.assertIn("FITTED_EXPONENT_OUTSIDE_PHYSICALLY_PLAUSIBLE_RANGE", fit.reasons)

    def test_power_rising_with_distance_is_named_for_what_it_is(self) -> None:
        # Near and far transposed in the survey looks exactly like this.
        fit = PathLossCalibrator().fit(calibration_set(exponent=-2.0))
        self.assertFalse(fit.fitted)
        self.assertIn("FITTED_POWER_RISES_WITH_DISTANCE_WHICH_IS_NOT_PROPAGATION", fit.reasons)

    def test_an_implausible_reference_power_is_refused(self) -> None:
        # The reference power is extrapolated back to 1 m, far outside the
        # data, so a poor fit runs away there first. A receiver cannot report
        # +60 dBm, and a model claiming it would scale every range wrongly.
        fit = PathLossCalibrator().fit(calibration_set(reference_power_dbm=60.0))
        self.assertFalse(fit.fitted)
        self.assertIn("FITTED_REFERENCE_POWER_OUTSIDE_ANY_PLAUSIBLE_RECEIVER_RANGE", fit.reasons)

    def test_scatter_too_large_for_a_log_distance_law_is_refused(self) -> None:
        fit = PathLossCalibrator().fit(
            calibration_set(noise=(15.0, -15.0, 15.0, -15.0, 15.0))
        )
        self.assertFalse(fit.fitted)
        self.assertIn(
            "RESIDUAL_TOO_LARGE_FOR_A_LOG_DISTANCE_LAW_TO_DESCRIBE_THESE_POINTS", fit.reasons
        )
        # A refused fit still reports what it computed: an operator who can see
        # the residual can find the cause, one told only "false" cannot.
        self.assertIsNotNone(fit.residual_rms_db)
        self.assertIsNotNone(fit.path_loss_exponent)

    def test_a_flat_noisy_set_is_refused_rather_than_overflowing(self) -> None:
        # A stuck receiver reads nearly the same power at every distance, so the
        # fitted exponent lands within a hair of zero. The implied range error
        # is then 10**(residual / ~0), which in Python *raises* rather than
        # saturating to infinity — so the arithmetic that describes how useless
        # the fit is would crash before the fit could be refused for being
        # useless. The noise below is orthogonal to the fit's x axis, which
        # keeps a real residual while leaving the tiny slope intact.
        import json

        xs = [math.log10(distance) for distance in SPREAD_M]
        mean_x = sum(xs) / len(xs)
        centred = [x - mean_x for x in xs]
        noise = [0.5, -0.5, 0.5, -0.5, 0.0]
        noise[4] = -sum(c * n for c, n in zip(centred, noise)) / centred[4]

        fit = PathLossCalibrator().fit(calibration_set(exponent=1e-5, noise=tuple(noise)))
        self.assertGreater(fit.residual_rms_db / (10.0 * fit.path_loss_exponent), 308.0)
        self.assertFalse(fit.fitted)
        self.assertIn("FITTED_EXPONENT_OUTSIDE_PHYSICALLY_PLAUSIBLE_RANGE", fit.reasons)
        self.assertIsNone(json.loads(json.dumps(fit.to_dict()))["range_error_factor"])

    def test_a_refused_fit_claims_no_environment_band(self) -> None:
        fit = PathLossCalibrator().fit(calibration_set(exponent=0.8))
        self.assertEqual(fit.environment_band, "UNCLASSIFIED")
        self.assertIn("NO_ENVIRONMENT_BAND_IS_CLAIMED_FOR_A_REFUSED_FIT", fit.reasons)

    def test_an_estimated_position_offered_as_a_survey_fails_on_its_own_numbers(self) -> None:
        # The last line of defence behind the type check. Someone who copies an
        # emitter estimate into a SurveyedReference and declares its uncertainty
        # honestly — tens or hundreds of metres — is refused, because a survey
        # that vague resolves none of these distances.
        vague = SurveyedReference("beacon-1", 1.3000, 103.8000, 300.0, "OPERATOR_SURVEYED")
        fit = PathLossCalibrator().fit(calibration_set(reference=vague))
        self.assertFalse(fit.fitted)
        self.assertEqual(fit.excluded_samples, len(SPREAD_M))
        self.assertIn("MEASUREMENT_TOO_CLOSE_FOR_THE_SURVEY_TO_RESOLVE_EXCLUDED", fit.reasons)

    def test_one_unresolved_measurement_does_not_lose_the_whole_fit(self) -> None:
        coarse = SurveyedReference("beacon-1", 1.3000, 103.8000, 6.0, "OPERATOR_SURVEYED")
        fit = PathLossCalibrator().fit(
            calibration_set((50.0, 150.0, 400.0, 1000.0, 2000.0, 3000.0), reference=coarse)
        )
        self.assertTrue(fit.fitted)
        self.assertEqual(fit.samples, 6)
        self.assertEqual(fit.excluded_samples, 1)
        self.assertIn("MEASUREMENT_TOO_CLOSE_FOR_THE_SURVEY_TO_RESOLVE_EXCLUDED", fit.reasons)

    def test_the_minimum_sample_count_cannot_be_configured_away(self) -> None:
        # Below five points the residual is a property of one measurement
        # rather than of the data, so the floor is not a default.
        calibrator = PathLossCalibrator(min_points=2)
        self.assertEqual(calibrator.min_points, MIN_CALIBRATION_POINTS)


class EnvironmentBandTests(unittest.TestCase):
    def test_a_free_space_exponent_reads_as_free_space(self) -> None:
        self.assertEqual(environment_band(2.0, 0.1), "FREE_SPACE_LIKE")

    def test_an_urban_exponent_reads_as_outdoor_clutter(self) -> None:
        self.assertEqual(environment_band(3.1, 0.2), "OUTDOOR_CLUTTER_LIKE")

    def test_a_steep_exponent_reads_as_heavy_obstruction(self) -> None:
        self.assertEqual(environment_band(4.6, 0.3), "HEAVY_OBSTRUCTION_LIKE")

    def test_no_band_is_claimed_when_the_error_bar_crosses_an_edge(self) -> None:
        # 2.4 +/- 0.3 is consistent with free space and with clutter alike.
        # Naming either would be a claim the data does not make.
        self.assertEqual(environment_band(2.4, 0.3), "UNCLASSIFIED")

    def test_a_non_finite_exponent_is_unclassified(self) -> None:
        self.assertEqual(environment_band(float("nan"), 0.1), "UNCLASSIFIED")

    def test_a_node_inside_a_structure_learns_that_as_a_number(self) -> None:
        # The capability in one sentence: nobody told this node it was indoors.
        fit = PathLossCalibrator().fit(
            calibration_set(
                (2.0, 5.0, 12.0, 30.0, 80.0),
                exponent=4.5,
                reference_power_dbm=-30.0,
                reference=PRECISE_REFERENCE,
            )
        )
        self.assertTrue(fit.fitted)
        self.assertEqual(fit.environment_band, "HEAVY_OBSTRUCTION_LIKE")
        self.assertAlmostEqual(fit.path_loss_exponent, 4.5, places=9)


class OperatorCalibratedPathIsUnchangedTests(unittest.TestCase):
    """A deployment with no surveyed references must behave exactly as before."""

    def test_a_hand_measured_model_still_needs_no_fit(self) -> None:
        model = PathLossModel(reference_power_dbm=-40.0, path_loss_exponent=2.5)
        self.assertEqual(model.reference_distance_m, 1.0)
        self.assertAlmostEqual(model.distance_m(-40.0), 1.0, places=9)
        estimate = EmitterLocator().locate(
            [sighting("a", -45.0), sighting("b", -71.0), sighting("c", -73.0)],
            TRIANGLE,
            path_loss=model,
        )
        self.assertEqual(estimate.method, "trilateration")

    def test_the_fits_exponent_bounds_match_the_signed_document_loader(self) -> None:
        # They have to: a fit an operator accepts gets signed into the same
        # path-loss document, and a fit the loader would then reject is a fit
        # that should never have been offered.
        for edge in (MIN_PLAUSIBLE_PATH_LOSS_EXPONENT, MAX_PLAUSIBLE_PATH_LOSS_EXPONENT):
            PathLossModel.from_dict(
                {"reference_power_dbm": -40.0, "path_loss_exponent": edge}
            )
        with self.assertRaises(ValueError):
            PathLossModel.from_dict(
                {
                    "reference_power_dbm": -40.0,
                    "path_loss_exponent": MAX_PLAUSIBLE_PATH_LOSS_EXPONENT + 0.01,
                }
            )


if __name__ == "__main__":
    unittest.main()
