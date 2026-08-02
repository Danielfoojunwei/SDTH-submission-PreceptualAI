"""Emitter track prediction via the Exploration and Preferential Return model.

EPR as used by Shen et al., "Dynamic Spectrum Optimization for
Internet-of-Vehicles with Deep-Learning-Based Mobility Prediction", Wireless
Personal Communications 137, 2024; the model itself is from Song, Koren, Wang
and Barabasi, Nature Physics 6:818-823, 2010. See docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import math
import unittest
from datetime import datetime, timedelta, timezone

from kval_node.mobility import (
    DEFAULT_GAMMA,
    DEFAULT_RHO,
    MIN_TRACK_LENGTH,
    Fix,
    MobilityError,
    TrackPredictor,
    cluster_places,
    exploration_probability,
    metres_between,
)

START = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc)

DEPOT = (1.3000, 103.8000)
CHECKPOINT = (1.3050, 103.8000)
OUTPOST = (1.3000, 103.8060)


def at(minutes: float, place: tuple[float, float], *, uncertainty: float = 20.0) -> Fix:
    return Fix(START + timedelta(minutes=minutes), place[0], place[1], uncertainty)


def moving(minutes: float, index: int, *, uncertainty: float = 20.0) -> Fix:
    """A fix on a steady northward run at roughly 10 m/s."""
    return Fix(
        START + timedelta(minutes=minutes),
        DEPOT[0] + index * 0.0054,  # ~600 m per step
        DEPOT[1],
        uncertainty,
    )


class ExplorationLawTests(unittest.TestCase):
    def test_probability_falls_as_territory_grows(self) -> None:
        self.assertGreater(exploration_probability(2), exploration_probability(20))

    def test_matches_the_published_form(self) -> None:
        # p = rho * S^-gamma, with the values reported by Song et al.
        self.assertAlmostEqual(
            exploration_probability(10), DEFAULT_RHO * 10 ** (-DEFAULT_GAMMA), places=9
        )

    def test_no_known_places_means_certain_exploration(self) -> None:
        self.assertEqual(exploration_probability(0), 1.0)

    def test_probability_never_exceeds_one(self) -> None:
        self.assertLessEqual(exploration_probability(1, rho=1.0), 1.0)


class PlaceClusteringTests(unittest.TestCase):
    def test_nearby_fixes_collapse_to_one_place(self) -> None:
        fixes = [at(0, DEPOT), at(5, (DEPOT[0] + 0.0002, DEPOT[1]))]
        self.assertEqual(len(cluster_places(fixes)), 1)

    def test_distant_fixes_stay_separate(self) -> None:
        self.assertEqual(len(cluster_places([at(0, DEPOT), at(5, CHECKPOINT)])), 2)

    def test_a_stationary_emitter_has_visited_once_not_many_times(self) -> None:
        # Counting each reading as a visit would make sitting still look like
        # habitual returning, which is the opposite of what it is.
        fixes = [at(minute, DEPOT) for minute in range(0, 50, 5)]
        places = cluster_places(fixes)
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0].visits, 1)

    def test_returning_increments_the_visit_count(self) -> None:
        fixes = [at(0, DEPOT), at(10, CHECKPOINT), at(20, DEPOT), at(30, CHECKPOINT), at(40, DEPOT)]
        by_visits = {round(p.latitude_deg, 4): p.visits for p in cluster_places(fixes)}
        self.assertEqual(by_visits[1.3000], 3)
        self.assertEqual(by_visits[1.3050], 2)

    def test_clustering_is_order_independent(self) -> None:
        fixes = [at(0, DEPOT), at(10, CHECKPOINT), at(20, DEPOT)]
        self.assertEqual(
            [p.visits for p in cluster_places(fixes)],
            [p.visits for p in cluster_places(list(reversed(fixes)))],
        )

    def test_place_serializes(self) -> None:
        json.dumps(cluster_places([at(0, DEPOT)])[0].to_dict())


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = TrackPredictor()
        self.horizon = timedelta(minutes=10)

    def test_a_short_track_is_refused(self) -> None:
        result = self.predictor.predict([at(0, DEPOT), at(5, CHECKPOINT)], horizon=self.horizon)
        self.assertFalse(result.predicted)
        self.assertIsNone(result.latitude_deg)
        self.assertIn(f"NEED_AT_LEAST_{MIN_TRACK_LENGTH}_FIXES_TO_MODEL_A_TRACK", result.reasons)

    def test_a_long_silence_is_not_bridged(self) -> None:
        # Predicting through an unobserved hour would invent movement.
        fixes = [at(0, DEPOT), at(5, CHECKPOINT), at(10, DEPOT), at(90, DEPOT)]
        result = self.predictor.predict(fixes, horizon=self.horizon)
        self.assertFalse(result.predicted)
        self.assertIn("TRACK_HAS_A_GAP_TOO_LONG_TO_BRIDGE", result.reasons)

    def test_too_few_known_places_means_exploration_dominates(self) -> None:
        # Two places give p ~= 0.52, above the 0.5 cut: the model's own answer
        # is "somewhere new", so it must not name a place.
        fixes = [at(0, DEPOT), at(10, CHECKPOINT), at(20, DEPOT), at(30, CHECKPOINT)]
        result = self.predictor.predict(fixes, horizon=self.horizon)
        self.assertFalse(result.predicted)
        self.assertGreater(result.exploration_probability, 0.5)
        self.assertIn(
            "EXPLORATION_MORE_LIKELY_THAN_RETURN_SO_NO_PLACE_IS_PREDICTED", result.reasons
        )

    def test_the_exploration_probability_is_reported_even_when_refusing(self) -> None:
        # The probability *is* the answer in this case, so it must survive.
        fixes = [at(0, DEPOT), at(10, CHECKPOINT), at(20, DEPOT), at(30, CHECKPOINT)]
        result = self.predictor.predict(fixes, horizon=self.horizon)
        self.assertGreater(result.exploration_probability, 0.0)
        self.assertEqual(result.distinct_places, 2)

    def test_a_non_positive_horizon_is_an_error(self) -> None:
        with self.assertRaises(MobilityError):
            self.predictor.predict([at(0, DEPOT)] * 5, horizon=timedelta(0))

    def test_refusal_serializes(self) -> None:
        json.dumps(self.predictor.predict([at(0, DEPOT)], horizon=self.horizon).to_dict())


class PreferentialReturnTests(unittest.TestCase):
    def setUp(self) -> None:
        # A patrol with enough distinct places that returning beats exploring.
        self.places = [DEPOT, CHECKPOINT, OUTPOST, (1.3040, 103.8050), (1.2960, 103.8030)]
        self.predictor = TrackPredictor()
        self.horizon = timedelta(minutes=10)

    def _patrol(self, extra_depot_visits: int = 0) -> list[Fix]:
        fixes = [at(index * 10, place) for index, place in enumerate(self.places)]
        minute = len(self.places) * 10
        for _ in range(extra_depot_visits):
            fixes.append(at(minute, DEPOT))
            minute += 10
            fixes.append(at(minute, CHECKPOINT))
            minute += 10
        return fixes

    def test_a_settled_emitter_is_predicted_to_return(self) -> None:
        result = self.predictor.predict(self._patrol(), horizon=self.horizon)
        self.assertTrue(result.predicted)
        self.assertEqual(result.method, "preferential_return")
        self.assertLess(result.exploration_probability, 0.5)

    def test_the_prediction_is_pulled_toward_the_most_visited_place(self) -> None:
        # EPR's return rule: proportional to visit frequency, not uniform.
        even = self.predictor.predict(self._patrol(), horizon=self.horizon)
        weighted = self.predictor.predict(self._patrol(extra_depot_visits=4), horizon=self.horizon)
        self.assertLess(
            metres_between(weighted.latitude_deg, weighted.longitude_deg, *DEPOT),
            metres_between(even.latitude_deg, even.longitude_deg, *DEPOT),
        )

    def test_more_places_make_exploration_less_likely(self) -> None:
        few = self.predictor.predict(
            [at(i * 10, p) for i, p in enumerate(self.places[:4])] * 1, horizon=self.horizon
        )
        many = self.predictor.predict(self._patrol(), horizon=self.horizon)
        self.assertLess(many.exploration_probability, few.exploration_probability)

    def test_a_stationary_emitter_is_predicted_to_stay(self) -> None:
        fixes = [at(minute, DEPOT) for minute in range(0, 60, 5)]
        result = TrackPredictor(max_exploration=1.0).predict(fixes, horizon=self.horizon)
        self.assertTrue(result.predicted)
        self.assertLess(metres_between(result.latitude_deg, result.longitude_deg, *DEPOT), 1.0)
        self.assertIn("ONLY_ONE_KNOWN_PLACE_SO_THE_PREDICTION_IS_THAT_IT_STAYS", result.reasons)

    def test_uncertainty_covers_the_spread_of_known_places(self) -> None:
        result = self.predictor.predict(self._patrol(), horizon=self.horizon)
        self.assertGreater(result.uncertainty_radius_m, 100.0)

    def test_result_serializes(self) -> None:
        json.dumps(self.predictor.predict(self._patrol(), horizon=self.horizon).to_dict())


class KinematicContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = TrackPredictor()
        self.horizon = timedelta(minutes=1)

    def _run(self) -> list[Fix]:
        # Steady northward movement, each fix a new place, so the emitter has
        # plenty of distinct places and low exploration probability.
        return [moving(index, index) for index in range(7)]

    def test_an_emitter_in_transit_continues_its_leg(self) -> None:
        result = self.predictor.predict(self._run(), horizon=self.horizon)
        self.assertTrue(result.predicted)
        self.assertEqual(result.method, "kinematic_continuation")
        self.assertIn("EMITTER_IS_IN_TRANSIT_SO_THE_CURRENT_LEG_CONTINUES", result.reasons)

    def test_the_prediction_is_ahead_of_the_last_fix_not_behind_it(self) -> None:
        # The failure mode this exists to prevent: predicting the depot an
        # emitter is currently driving away from.
        fixes = self._run()
        result = self.predictor.predict(fixes, horizon=self.horizon)
        self.assertGreater(result.latitude_deg, fixes[-1].latitude_deg)

    def test_reaching_further_ahead_is_less_certain(self) -> None:
        near = self.predictor.predict(self._run(), horizon=timedelta(minutes=1))
        far = self.predictor.predict(self._run(), horizon=timedelta(minutes=5))
        self.assertGreater(far.uncertainty_radius_m, near.uncertainty_radius_m)

    def test_noise_at_a_standstill_is_not_mistaken_for_motion(self) -> None:
        # Fixes jittering inside their own uncertainty must not extrapolate.
        jitter = [
            Fix(START + timedelta(minutes=i), DEPOT[0] + 0.00002 * i, DEPOT[1], 50.0)
            for i in range(8)
        ]
        result = TrackPredictor(max_exploration=1.0).predict(jitter, horizon=self.horizon)
        self.assertEqual(result.method, "preferential_return")

    def test_a_manoeuvring_emitter_falls_back_to_the_return_rule(self) -> None:
        # Legs that disagree in direction mean extrapolation is guessing.
        zigzag = [
            at(0, DEPOT),
            at(1, CHECKPOINT),
            at(2, OUTPOST),
            at(3, (1.3040, 103.8050)),
            at(4, (1.2960, 103.8030)),
            at(5, DEPOT),
            at(6, OUTPOST),
        ]
        result = self.predictor.predict(zigzag, horizon=self.horizon)
        self.assertEqual(result.method, "preferential_return")

    def test_the_two_mechanisms_are_chosen_between_never_averaged(self) -> None:
        # Averaging "carry on north" with "return to the depot" gives a point
        # the emitter will not be at under either hypothesis.
        result = self.predictor.predict(self._run(), horizon=self.horizon)
        self.assertIn(result.method, {"kinematic_continuation", "preferential_return"})
        self.assertNotIn("blend", result.method)

    def test_deferring_the_return_rule_is_stated(self) -> None:
        result = self.predictor.predict(self._run(), horizon=self.horizon)
        self.assertTrue(
            any(r.startswith("PREFERENTIAL_RETURN_DEFERRED_WITH_") for r in result.reasons)
        )

    def test_prediction_is_deterministic(self) -> None:
        first = self.predictor.predict(self._run(), horizon=self.horizon)
        second = self.predictor.predict(self._run(), horizon=self.horizon)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_predicted_distance_matches_the_observed_speed(self) -> None:
        fixes = self._run()
        speed = metres_between(
            fixes[-2].latitude_deg, fixes[-2].longitude_deg,
            fixes[-1].latitude_deg, fixes[-1].longitude_deg,
        ) / (fixes[-1].observed_at - fixes[-2].observed_at).total_seconds()
        result = self.predictor.predict(fixes, horizon=timedelta(minutes=1))
        travelled = metres_between(
            fixes[-1].latitude_deg, fixes[-1].longitude_deg,
            result.latitude_deg, result.longitude_deg,
        )
        self.assertAlmostEqual(travelled, speed * 60.0, delta=1.0)


class FixValidationTests(unittest.TestCase):
    def test_an_impossible_latitude_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Fix(START, 91.0, 103.8, 10.0)

    def test_a_non_finite_uncertainty_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            Fix(START, 1.3, 103.8, math.nan)


if __name__ == "__main__":
    unittest.main()
