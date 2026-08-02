"""Per-flow PSO-tuned forecasting.

Mechanism adapted from Li, Nie, Lam, Shen and Li, "Network Traffic Prediction
Using PSO-LightGBM-TM", IEEE INFOCOM 2024 Workshops. Their base learner is
LightGBM; this implements their per-flow PSO tuning strategy over a
dependency-free smoother. See docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import math
import unittest

from kval_node.forecast import (
    MIN_HISTORY,
    PerFlowForecaster,
    SmoothingParameters,
    holt_forecast,
    particle_swarm_minimise,
)

RISING = [20.0, 22.0, 25.0, 29.0, 34.0, 40.0, 47.0, 55.0]
STABLE = [20.0, 21.0, 19.0, 20.0, 21.0, 19.0, 20.0, 20.0]
FALLING = [80.0, 70.0, 61.0, 53.0, 46.0, 40.0, 35.0, 31.0]


class HoltTests(unittest.TestCase):
    def test_forecast_follows_a_trend(self) -> None:
        parameters = SmoothingParameters(0.8, 0.5, 1.0)
        forecast, _fitted = holt_forecast(RISING, parameters)
        self.assertGreater(forecast, RISING[-1])

    def test_fitted_values_are_one_step_ahead_predictions(self) -> None:
        # One fewer fitted value than observations: nothing is predicted for
        # the first point, and each prediction uses only earlier data.
        _forecast, fitted = holt_forecast(RISING, SmoothingParameters(0.5, 0.2, 0.9))
        self.assertEqual(len(fitted), len(RISING) - 1)

    def test_damping_flattens_the_projection(self) -> None:
        undamped, _ = holt_forecast(RISING, SmoothingParameters(0.8, 0.5, 1.0), horizon=5)
        damped, _ = holt_forecast(RISING, SmoothingParameters(0.8, 0.5, 0.5), horizon=5)
        self.assertLess(damped, undamped)

    def test_too_short_a_series_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            holt_forecast([1.0], SmoothingParameters(0.5, 0.1, 0.9))


class ParticleSwarmTests(unittest.TestCase):
    def test_finds_the_minimum_of_a_known_bowl(self) -> None:
        best, score = particle_swarm_minimise(
            lambda v: (v[0] - 0.3) ** 2 + (v[1] + 0.4) ** 2, [(-1.0, 1.0), (-1.0, 1.0)]
        )
        self.assertLess(score, 0.01)
        self.assertAlmostEqual(best[0], 0.3, delta=0.1)
        self.assertAlmostEqual(best[1], -0.4, delta=0.1)

    def test_is_deterministic(self) -> None:
        # A node's forecast must be reconstructable from its audit record, so
        # the search cannot depend on an unseeded random source.
        objective = lambda v: (v[0] - 0.25) ** 2  # noqa: E731
        first = particle_swarm_minimise(objective, [(0.0, 1.0)])
        second = particle_swarm_minimise(objective, [(0.0, 1.0)])
        self.assertEqual(first, second)

    def test_respects_bounds(self) -> None:
        best, _score = particle_swarm_minimise(lambda v: -v[0], [(0.0, 1.0)])
        self.assertLessEqual(best[0], 1.0)
        self.assertGreaterEqual(best[0], 0.0)

    def test_empty_bounds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            particle_swarm_minimise(lambda v: 0.0, [])


class PerFlowForecasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.forecaster = PerFlowForecaster()

    def test_a_worsening_flow_is_flagged(self) -> None:
        result = self.forecaster.fit("wifi", RISING)
        self.assertTrue(result.degrading)
        self.assertGreater(result.predicted, result.recent_mean)
        self.assertIn("FORECAST_ABOVE_RECENT_MEAN_BY_MARGIN", result.reasons)

    def test_a_stable_flow_is_not_flagged(self) -> None:
        result = self.forecaster.fit("eth", STABLE)
        self.assertFalse(result.degrading)
        self.assertIn("WITHIN_RECENT_BEHAVIOUR", result.reasons)

    def test_an_improving_flow_is_not_flagged(self) -> None:
        self.assertFalse(self.forecaster.fit("cell", FALLING).degrading)

    def test_each_flow_gets_its_own_parameters(self) -> None:
        # The source paper's central claim: one parameter set per flow, not one
        # shared across flows.
        rising = self.forecaster.fit("a", RISING).parameters
        stable = self.forecaster.fit("b", STABLE).parameters
        self.assertNotEqual(rising.to_dict(), stable.to_dict())

    def test_tuning_beats_an_arbitrary_default(self) -> None:
        from kval_node.forecast import _mean_absolute_error

        tuned = self.forecaster.fit("a", RISING)
        default = _mean_absolute_error(RISING, SmoothingParameters(0.5, 0.5, 0.9))
        self.assertLessEqual(tuned.mean_absolute_error, default)

    def test_short_history_refuses_to_forecast(self) -> None:
        result = self.forecaster.fit("x", [1.0, 2.0, 3.0])
        self.assertFalse(result.degrading)
        self.assertIn(f"NEED_AT_LEAST_{MIN_HISTORY}_SAMPLES", result.reasons)
        self.assertTrue(math.isinf(result.mean_absolute_error))

    def test_is_deterministic(self) -> None:
        self.assertEqual(
            self.forecaster.fit("a", RISING).predicted,
            self.forecaster.fit("a", RISING).predicted,
        )

    def test_fit_all_is_per_flow_and_sorted(self) -> None:
        results = self.forecaster.fit_all({"b": STABLE, "a": RISING})
        self.assertEqual([r.flow_id for r in results], ["a", "b"])
        self.assertTrue(results[0].degrading)
        self.assertFalse(results[1].degrading)

    def test_result_serializes(self) -> None:
        json.dumps(self.forecaster.fit("a", RISING).to_dict())


if __name__ == "__main__":
    unittest.main()
