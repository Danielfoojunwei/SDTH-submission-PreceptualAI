"""Local differential privacy on federated contributions.

Mechanism from Zhao, Zhao, Yang, Wang, Wang, Lyu, Niyato and Lam, "Local
Differential Privacy based Federated Learning for Internet of Things",
arXiv:2004.08856. See docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import statistics
import unittest

from kval_node.privacy import (
    FeatureRange,
    LocalPrivatiser,
    PiecewiseMechanism,
    aggregate_privatised,
)


class FeatureRangeTests(unittest.TestCase):
    def test_normalise_maps_to_unit_interval(self) -> None:
        span = FeatureRange(0.0, 100.0)
        self.assertAlmostEqual(span.normalise(50.0), 0.0)
        self.assertAlmostEqual(span.normalise(0.0), -1.0)
        self.assertAlmostEqual(span.normalise(100.0), 1.0)

    def test_round_trip(self) -> None:
        span = FeatureRange(-20.0, 60.0)
        self.assertAlmostEqual(span.denormalise(span.normalise(17.0)), 17.0, places=9)

    def test_out_of_range_is_clamped_not_widened(self) -> None:
        # Widening the range from data would itself leak the outlier.
        span = FeatureRange(0.0, 10.0)
        self.assertEqual(span.normalise(999.0), 1.0)
        self.assertEqual(span.normalise(-999.0), -1.0)

    def test_degenerate_range_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FeatureRange(5.0, 5.0)


class PiecewiseMechanismTests(unittest.TestCase):
    def test_perturbation_is_unbiased_in_aggregate(self) -> None:
        # The individual outputs are noisy; their mean converges to the truth.
        mechanism = PiecewiseMechanism(epsilon=2.0)
        truth = 0.4
        samples = [mechanism.perturb(truth) for _ in range(20_000)]
        self.assertAlmostEqual(statistics.fmean(samples), truth, delta=0.05)

    def test_individual_outputs_actually_differ_from_the_input(self) -> None:
        # If perturbation were a no-op there would be no privacy at all.
        mechanism = PiecewiseMechanism(epsilon=1.0)
        samples = [mechanism.perturb(0.0) for _ in range(200)]
        self.assertGreater(len(set(samples)), 100)
        self.assertGreater(statistics.pstdev(samples), 0.1)

    def test_a_larger_budget_is_less_noisy(self) -> None:
        tight = PiecewiseMechanism(epsilon=0.5)
        loose = PiecewiseMechanism(epsilon=8.0)
        tight_spread = statistics.pstdev([tight.perturb(0.2) for _ in range(3000)])
        loose_spread = statistics.pstdev([loose.perturb(0.2) for _ in range(3000)])
        self.assertLess(loose_spread, tight_spread)

    def test_error_estimate_falls_with_fleet_size(self) -> None:
        mechanism = PiecewiseMechanism(epsilon=1.0)
        self.assertLess(
            mechanism.expected_absolute_error(100), mechanism.expected_absolute_error(4)
        )

    def test_output_stays_within_the_mechanism_range(self) -> None:
        mechanism = PiecewiseMechanism(epsilon=1.0)
        constant = mechanism._constant()
        for _ in range(500):
            value = mechanism.perturb(1.0)
            self.assertGreaterEqual(value, -constant - 1e-9)
            self.assertLessEqual(value, constant + 1e-9)

    def test_unusable_budget_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PiecewiseMechanism(epsilon=0.0)


class LocalPrivatiserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ranges = {"rtt_ms": FeatureRange(0.0, 500.0), "loss_percent": FeatureRange(0.0, 100.0)}

    def test_budget_is_split_across_features_released_together(self) -> None:
        # Releasing k values at full budget each would spend k*epsilon.
        privatiser = LocalPrivatiser(self.ranges, epsilon=2.0)
        self.assertAlmostEqual(privatiser.per_feature_epsilon, 1.0)

    def test_record_never_contains_the_true_value(self) -> None:
        privatiser = LocalPrivatiser(self.ranges, epsilon=4.0)
        record = privatiser.privatise("rtt_ms", 123.456).to_dict()
        self.assertNotIn("value", record)
        self.assertNotIn("true_value", record)
        self.assertNotEqual(record["perturbed"], 123.456)

    def test_fleet_mean_recovers_the_truth(self) -> None:
        privatiser = LocalPrivatiser(self.ranges, epsilon=8.0)
        truth = 200.0
        released = [privatiser.privatise("rtt_ms", truth).perturbed for _ in range(4000)]
        self.assertAlmostEqual(aggregate_privatised(released), truth, delta=15.0)

    def test_reports_how_many_contributors_are_needed(self) -> None:
        privatiser = LocalPrivatiser(self.ranges, epsilon=1.0)
        result = privatiser.privatise("rtt_ms", 100.0)
        self.assertGreater(result.contributors_needed_for_1pct, 1)

    def test_unknown_feature_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            LocalPrivatiser(self.ranges).privatise("not_declared", 1.0)

    def test_privatise_all_skips_undeclared_features(self) -> None:
        privatiser = LocalPrivatiser(self.ranges, epsilon=4.0)
        results = privatiser.privatise_all({"rtt_ms": 50.0, "unknown": 1.0})
        self.assertEqual([r.feature for r in results], ["rtt_ms"])

    def test_no_ranges_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            LocalPrivatiser({})

    def test_result_serializes(self) -> None:
        json.dumps(LocalPrivatiser(self.ranges).privatise("rtt_ms", 10.0).to_dict())


if __name__ == "__main__":
    unittest.main()
