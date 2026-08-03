"""Local differential privacy on federated contributions.

Mechanism from Zhao, Zhao, Yang, Wang, Wang, Lyu, Niyato and Lam, "Local
Differential Privacy based Federated Learning for Internet of Things",
arXiv:2004.08856. See docs/DSA_RESEARCH_MAPPING.md.

Budget allocation follows Tian, Wang, Jia, Wang, Chen and Kam, "Privacy-
Preserving Federated Radio Map Learning for Wireless Digital Twins via Adaptive
Noise Allocation", arXiv:2605.15560: location leakage is concentrated, so the
noise should be too. The tests below hold that to the same standard the module
claims — the total budget is conserved, and the cost of concentrating it is
asserted rather than assumed away.
"""

from __future__ import annotations

import json
import statistics
import unittest

from kval_node.dataset import FEATURE_NAMES
from kval_node.privacy import (
    BUDGET_ALLOCATIONS,
    MINIMUM_USEFUL_EPSILON,
    SITING_AWARE_ALLOCATION,
    SITING_LEAK_HIGH,
    SITING_LEAK_LOW,
    SITING_LEAK_MODERATE,
    SITING_SENSITIVITY,
    TIER_BUDGET_SHARE,
    UNIFORM_ALLOCATION,
    FeatureRange,
    LocalPrivatiser,
    PiecewiseMechanism,
    aggregate_privatised,
    allocate_budget,
    siting_sensitivity,
)

#: The features a node actually releases in production, so the allocation is
#: exercised on the set that matters rather than a convenient one.
RELEASED_FEATURES = ("rtt_ms", "loss_percent", "jitter_ms", "signal_percent", "throughput_mbps")

RELEASED_RANGES = {
    "rtt_ms": FeatureRange(0.0, 2000.0),
    "loss_percent": FeatureRange(0.0, 100.0),
    "jitter_ms": FeatureRange(0.0, 500.0),
    "signal_percent": FeatureRange(0.0, 100.0),
    "throughput_mbps": FeatureRange(0.0, 1000.0),
}


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


class BudgetConservationTests(unittest.TestCase):
    """The total budget is the operator's decision, not the allocator's."""

    def test_total_epsilon_is_identical_however_it_is_allocated(self) -> None:
        # The defect: an allocation that improves privacy by quietly spending
        # more budget has improved nothing, it has changed the question. Every
        # allocation must sum to the same declared total.
        for total in (0.06, 0.5, 1.0, 2.0, 7.5, 50.0):
            spent = {
                allocation: allocate_budget(
                    total, RELEASED_FEATURES, allocation=allocation
                ).spent_epsilon
                for allocation in BUDGET_ALLOCATIONS
            }
            for allocation, value in spent.items():
                with self.subTest(total=total, allocation=allocation):
                    self.assertAlmostEqual(value, total, places=12)
            self.assertAlmostEqual(
                spent[UNIFORM_ALLOCATION], spent[SITING_AWARE_ALLOCATION], places=12
            )

    def test_conservation_holds_for_any_number_of_features(self) -> None:
        # Water-filling at the floor is where a naive implementation starts
        # adding budget on top instead of taking it from elsewhere.
        for count in range(1, len(FEATURE_NAMES) + 1):
            names = FEATURE_NAMES[:count]
            for allocation in BUDGET_ALLOCATIONS:
                for total in (0.02 * count + 0.001, 0.3, 4.0):
                    with self.subTest(count=count, allocation=allocation, total=total):
                        budget = allocate_budget(total, names, allocation=allocation)
                        self.assertAlmostEqual(budget.spent_epsilon, total, places=12)
                        self.assertTrue(budget.conserved)

    def test_no_feature_is_left_below_a_usable_budget(self) -> None:
        # A share small enough that its own noise swamps any realistic fleet is
        # not privacy, it is a discarded coordinate. The shortfall is taken
        # from the features above the floor, which is why conservation holds.
        budget = allocate_budget(0.08, RELEASED_FEATURES, allocation=SITING_AWARE_ALLOCATION)
        self.assertGreaterEqual(min(budget.per_feature.values()), MINIMUM_USEFUL_EPSILON)
        self.assertAlmostEqual(budget.spent_epsilon, 0.08, places=12)
        self.assertIn("SOME_FEATURES_HELD_AT_THE_MINIMUM_USEFUL_BUDGET", budget.reasons)

    def test_an_indivisible_budget_is_declared_rather_than_silently_overspent(self) -> None:
        # The one case where spending exceeds the request. It must say so: an
        # operator who declared 0.02 and was charged 0.05 has to be told.
        budget = allocate_budget(0.02, RELEASED_FEATURES, allocation=SITING_AWARE_ALLOCATION)
        self.assertFalse(budget.conserved)
        self.assertGreater(budget.spent_epsilon, budget.requested_epsilon)
        self.assertIn("RELEASE_SPENDS_MORE_EPSILON_THAN_WAS_REQUESTED", budget.reasons)
        self.assertIn("ALLOCATION_HAS_NO_EFFECT_AT_THIS_BUDGET", budget.reasons)

    def test_allocations_agree_exactly_when_the_budget_cannot_be_split(self) -> None:
        # Even in the overspending regime the choice of allocation must not
        # change what a release costs, or the two are not comparable.
        uniform = allocate_budget(0.01, RELEASED_FEATURES, allocation=UNIFORM_ALLOCATION)
        concentrated = allocate_budget(0.01, RELEASED_FEATURES, allocation=SITING_AWARE_ALLOCATION)
        self.assertEqual(uniform.per_feature, concentrated.per_feature)

    def test_a_total_below_the_usable_floor_is_refused_not_clamped(self) -> None:
        # Refusing beats returning a plausible allocation of a budget that does
        # not exist.
        with self.assertRaises(ValueError):
            allocate_budget(0.0, RELEASED_FEATURES)

    def test_unknown_allocation_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            allocate_budget(1.0, RELEASED_FEATURES, allocation="whatever_sounds_private")

    def test_no_features_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            allocate_budget(1.0, [])


class SitingSensitivityTests(unittest.TestCase):
    """Which features reveal where a node is — the judgement being relied on."""

    def test_every_released_feature_is_classified_explicitly(self) -> None:
        # A feature added to the corpus without a classification silently
        # inherits the fail-safe tier. That is the right runtime behaviour and
        # the wrong permanent state, so the omission fails here instead.
        for name in FEATURE_NAMES:
            with self.subTest(feature=name):
                self.assertIn(name, SITING_SENSITIVITY)
                self.assertIn(
                    SITING_SENSITIVITY[name],
                    (SITING_LEAK_HIGH, SITING_LEAK_MODERATE, SITING_LEAK_LOW),
                )

    def test_received_power_features_are_treated_as_the_most_revealing(self) -> None:
        # Both are monotone in path loss, which is what turns a released number
        # into a range ring around an emitter.
        self.assertEqual(siting_sensitivity("signal_percent"), SITING_LEAK_HIGH)
        self.assertEqual(siting_sensitivity("throughput_mbps"), SITING_LEAK_HIGH)

    def test_round_trip_time_is_not_treated_as_a_position(self) -> None:
        # RTT is a property of the path through the network, not of where the
        # node stands; spending noise on it buys little.
        self.assertEqual(siting_sensitivity("rtt_ms"), SITING_LEAK_LOW)

    def test_an_unclassified_feature_fails_safe_to_the_most_sensitive_tier(self) -> None:
        # Assuming a new feature is harmless is the assumption that cannot be
        # taken back once the release has happened.
        self.assertEqual(siting_sensitivity("some_future_feature"), SITING_LEAK_HIGH)

    def test_the_sensitive_tiers_receive_the_tighter_budgets(self) -> None:
        # The direction is trivial to invert by accident and inverting it would
        # strip protection from exactly the coordinates that need it: a smaller
        # epsilon is a tighter budget and therefore more noise.
        self.assertLess(TIER_BUDGET_SHARE[SITING_LEAK_HIGH], TIER_BUDGET_SHARE[SITING_LEAK_MODERATE])
        self.assertLess(TIER_BUDGET_SHARE[SITING_LEAK_MODERATE], TIER_BUDGET_SHARE[SITING_LEAK_LOW])

    def test_concentrated_allocation_moves_budget_off_the_revealing_features(self) -> None:
        budget = allocate_budget(2.0, RELEASED_FEATURES, allocation=SITING_AWARE_ALLOCATION)
        uniform = allocate_budget(2.0, RELEASED_FEATURES, allocation=UNIFORM_ALLOCATION)
        self.assertLess(budget.per_feature["signal_percent"], uniform.per_feature["signal_percent"])
        self.assertGreater(budget.per_feature["rtt_ms"], uniform.per_feature["rtt_ms"])
        self.assertLess(budget.per_feature["signal_percent"], budget.per_feature["loss_percent"])
        self.assertLess(budget.per_feature["loss_percent"], budget.per_feature["rtt_ms"])

    def test_unclassified_features_are_named_in_the_reasons(self) -> None:
        budget = allocate_budget(
            2.0, ("rtt_ms", "an_unmapped_feature"), allocation=SITING_AWARE_ALLOCATION
        )
        self.assertTrue(
            any(
                reason.startswith("UNCLASSIFIED_FEATURES_TREATED_AS_SITING_SENSITIVE:")
                and "an_unmapped_feature" in reason
                for reason in budget.reasons
            )
        )


class ConcentratedAllocationCostTests(unittest.TestCase):
    """What concentrating the noise costs, asserted rather than implied away."""

    def test_concentration_injects_more_total_noise_at_the_same_total_budget(self) -> None:
        # The honest half of the claim. The mechanism's variance is convex in
        # epsilon, so an unequal split at a matched total injects *more* noise
        # overall than an even one. Redistribution, not a free lunch — and a
        # future edit that starts describing this as strictly better should
        # fail here.
        def total_noise(allocation: str) -> float:
            budget = allocate_budget(2.0, RELEASED_FEATURES, allocation=allocation)
            return sum(
                PiecewiseMechanism(epsilon=share).expected_absolute_error(1) ** 2
                for share in budget.per_feature.values()
            )

        self.assertGreater(total_noise(SITING_AWARE_ALLOCATION), total_noise(UNIFORM_ALLOCATION))

    def test_the_revealing_feature_is_measurably_noisier_and_the_rest_cleaner(self) -> None:
        # The allocation has to reach the mechanism, not just the report.
        concentrated = LocalPrivatiser(
            RELEASED_RANGES, epsilon=2.0, allocation=SITING_AWARE_ALLOCATION
        )
        uniform = LocalPrivatiser(RELEASED_RANGES, epsilon=2.0, allocation=UNIFORM_ALLOCATION)

        def spread(privatiser: LocalPrivatiser, feature: str, value: float) -> float:
            return statistics.pstdev(
                [privatiser.privatise(feature, value).perturbed for _ in range(2000)]
            )

        self.assertGreater(
            spread(concentrated, "signal_percent", 50.0),
            1.5 * spread(uniform, "signal_percent", 50.0),
        )
        self.assertLess(
            spread(concentrated, "rtt_ms", 1000.0),
            0.8 * spread(uniform, "rtt_ms", 1000.0),
        )

    def test_a_noisier_feature_needs_a_larger_fleet_and_says_so(self) -> None:
        # The cost has to be reported per feature; one number for the release
        # would understate it for the coordinate carrying the extra noise.
        privatiser = LocalPrivatiser(
            RELEASED_RANGES, epsilon=2.0, allocation=SITING_AWARE_ALLOCATION
        )
        revealing = privatiser.privatise("signal_percent", 50.0)
        cheap = privatiser.privatise("rtt_ms", 1000.0)
        self.assertGreater(
            revealing.contributors_needed_for_1pct, cheap.contributors_needed_for_1pct
        )

    def test_perturbation_stays_unbiased_under_a_concentrated_budget(self) -> None:
        # Reallocation must not break the property the whole mechanism rests
        # on: noisier per node, still correct in aggregate. The tolerance comes
        # from the mechanism's own error estimate, not from a number picked to
        # make this pass.
        privatiser = LocalPrivatiser(
            RELEASED_RANGES, epsilon=20.0, allocation=SITING_AWARE_ALLOCATION
        )
        contributors = 4000
        truth = 50.0
        released = [
            privatiser.privatise("signal_percent", truth).perturbed for _ in range(contributors)
        ]
        half_span = (RELEASED_RANGES["signal_percent"].high - RELEASED_RANGES["signal_percent"].low) / 2.0
        expected = (
            PiecewiseMechanism(epsilon=privatiser.epsilon_for("signal_percent"))
            .expected_absolute_error(contributors)
            * half_span
        )
        self.assertAlmostEqual(aggregate_privatised(released), truth, delta=expected * 3.0)


class AllocationReportingTests(unittest.TestCase):
    """A release that does not say where its budget went cannot be audited."""

    def test_uniform_remains_the_default(self) -> None:
        # Concentrating noise rests on a judgement about a deployment's radios.
        # Inheriting that by upgrading would make it nobody's decision.
        privatiser = LocalPrivatiser(RELEASED_RANGES, epsilon=2.0)
        self.assertEqual(privatiser.allocation, UNIFORM_ALLOCATION)
        self.assertEqual(len(set(privatiser.budget.per_feature.values())), 1)

    def test_the_privatiser_reports_the_allocation_and_the_split(self) -> None:
        privatiser = LocalPrivatiser(
            RELEASED_RANGES, epsilon=2.0, allocation=SITING_AWARE_ALLOCATION
        )
        report = privatiser.budget.to_dict()
        self.assertEqual(report["strategy"], SITING_AWARE_ALLOCATION)
        self.assertTrue(report["conserved"])
        self.assertEqual(report["siting_sensitivity"]["signal_percent"], SITING_LEAK_HIGH)
        self.assertIn(
            "NO_END_TO_END_DP_ACCOUNTANT_CLAIMED_FOR_THIS_ALLOCATION", report["reasons"]
        )
        json.dumps(report)

    def test_the_composition_bound_in_force_is_named_on_every_release(self) -> None:
        # The sum is an upper bound, not an accountant's figure. Saying which
        # bound is in force stops a reader assuming a tighter one.
        for allocation in BUDGET_ALLOCATIONS:
            with self.subTest(allocation=allocation):
                budget = allocate_budget(1.0, RELEASED_FEATURES, allocation=allocation)
                self.assertIn(
                    "SEQUENTIAL_COMPOSITION_BOUND_IS_THE_SUM_OF_PER_FEATURE_EPSILON",
                    budget.reasons,
                )

    def test_a_single_reported_epsilon_is_the_weakest_one_in_force(self) -> None:
        # Reporting the smallest would let a release describe itself by its
        # most private coordinate and understate what it exposed.
        privatiser = LocalPrivatiser(
            RELEASED_RANGES, epsilon=2.0, allocation=SITING_AWARE_ALLOCATION
        )
        self.assertEqual(privatiser.per_feature_epsilon, max(privatiser.budget.per_feature.values()))
        self.assertGreater(privatiser.per_feature_epsilon, privatiser.epsilon_for("signal_percent"))

    def test_uniform_reports_one_epsilon_for_every_feature(self) -> None:
        # The historical single-number disclosure must stay exactly true when
        # the historical allocation is in use.
        privatiser = LocalPrivatiser(RELEASED_RANGES, epsilon=2.0)
        self.assertAlmostEqual(privatiser.per_feature_epsilon, 0.4)
        for name in RELEASED_FEATURES:
            self.assertAlmostEqual(privatiser.epsilon_for(name), 0.4)

    def test_each_released_value_carries_its_allocation_and_still_no_truth(self) -> None:
        privatiser = LocalPrivatiser(
            RELEASED_RANGES, epsilon=2.0, allocation=SITING_AWARE_ALLOCATION
        )
        record = privatiser.privatise("signal_percent", 63.5).to_dict()
        self.assertEqual(record["allocation"], SITING_AWARE_ALLOCATION)
        self.assertEqual(record["siting_sensitivity"], SITING_LEAK_HIGH)
        self.assertIn("NOISE_CONCENTRATED_HERE_FEATURE_NARROWS_NODE_POSITION", record["reasons"])
        self.assertNotIn("value", record)
        self.assertNotIn("true_value", record)
        self.assertNotEqual(record["perturbed"], 63.5)
        json.dumps(record)

    def test_a_feature_funding_the_concentration_says_that_it_did(self) -> None:
        privatiser = LocalPrivatiser(
            RELEASED_RANGES, epsilon=2.0, allocation=SITING_AWARE_ALLOCATION
        )
        record = privatiser.privatise("rtt_ms", 120.0).to_dict()
        self.assertIn("BUDGET_RELEASED_HERE_FEATURE_SAYS_LITTLE_ABOUT_POSITION", record["reasons"])

    def test_epsilon_for_an_undeclared_feature_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            LocalPrivatiser(RELEASED_RANGES).epsilon_for("not_declared")

    def test_privatise_all_covers_every_declared_feature_under_either_allocation(self) -> None:
        values = {name: 1.0 for name in RELEASED_FEATURES}
        for allocation in BUDGET_ALLOCATIONS:
            with self.subTest(allocation=allocation):
                privatiser = LocalPrivatiser(RELEASED_RANGES, epsilon=2.0, allocation=allocation)
                released = privatiser.privatise_all(values)
                self.assertEqual(
                    sorted(item.feature for item in released), sorted(RELEASED_FEATURES)
                )


if __name__ == "__main__":
    unittest.main()
