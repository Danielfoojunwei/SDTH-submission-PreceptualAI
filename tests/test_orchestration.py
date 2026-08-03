"""The mechanisms have to work on each other, not merely exist.

Every module here reproduced one paper and was independently correct and
independently useless. These tests pin the *compositions* — the places where
one paper's output becomes another's input — because that wiring is what
silently regresses while every unit test stays green.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.dataset import FEATURE_NAMES, Example
from kval_node.federated import FeatureStatistics, FederatedUpdate
from kval_node.orchestration import (
    DEFAULT_FEATURE_RANGES,
    FleetLearning,
    cohort_assignments,
    forecast_advice,
    privatise_update,
    trusted_baseline,
)


NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def update(node: str, rtt_mean: float, *, count: int = 50) -> FederatedUpdate:
    stats = {name: FeatureStatistics(count, 0.0, float(count)) for name in FEATURE_NAMES}
    stats["rtt_ms"] = FeatureStatistics(count, rtt_mean, count * (rtt_mean**2 + 100.0))
    return FederatedUpdate(node, NOW, count, stats)


class PrivacyThenAggregationTests(unittest.TestCase):
    """LDP (IoT-J 2021) feeding the federated pipeline."""

    def test_the_exact_mean_never_leaves_the_node(self) -> None:
        original = update("n1", 200.0)
        released, disclosure = privatise_update(original, epsilon=4.0)
        self.assertTrue(disclosure.applied)
        self.assertNotEqual(released.statistics["rtt_ms"].mean, 200.0)

    def test_no_declared_budget_releases_exact_means_and_says_so(self) -> None:
        # The historical behaviour must stay reachable and must not pretend to
        # be private.
        original = update("n1", 200.0)
        released, disclosure = privatise_update(original, epsilon=None)
        self.assertFalse(disclosure.applied)
        self.assertEqual(released.statistics["rtt_ms"].mean, 200.0)
        self.assertIn(
            "NO_PRIVACY_BUDGET_DECLARED_SO_EXACT_MEANS_ARE_RELEASED", disclosure.reasons
        )

    def test_the_tension_with_consensus_filtering_is_disclosed(self) -> None:
        # Applying both weakens each. An operator must be told, not discover it.
        _released, disclosure = privatise_update(update("n1", 200.0), epsilon=4.0)
        self.assertIn("CONSENSUS_FILTERING_WILL_READ_THIS_NOISE_AS_DIVERGENCE", disclosure.reasons)

    def test_released_statistics_stay_internally_consistent(self) -> None:
        # A perturbed mean with an unperturbed sum of squares describes a
        # distribution that cannot exist, and stddev() would go imaginary.
        released, _ = privatise_update(update("n1", 200.0), epsilon=4.0)
        self.assertGreaterEqual(released.statistics["rtt_ms"].stddev(), 0.0)

    def test_the_fleet_mean_still_recovers_the_truth(self) -> None:
        # The whole point of an unbiased mechanism: privacy per node, accuracy
        # in aggregate. The tolerance comes from the mechanism's own error
        # estimate rather than a number chosen to make the test pass.
        from kval_node.privacy import PiecewiseMechanism

        contributors = 4000
        releases = [
            privatise_update(update(f"n{i}", 200.0), epsilon=8.0)[0].statistics["rtt_ms"].mean
            for i in range(contributors)
        ]
        _released, disclosure = privatise_update(update("n0", 200.0), epsilon=8.0)
        half_span = (
            DEFAULT_FEATURE_RANGES["rtt_ms"].high - DEFAULT_FEATURE_RANGES["rtt_ms"].low
        ) / 2.0
        expected = (
            PiecewiseMechanism(epsilon=disclosure.epsilon).expected_absolute_error(contributors)
            * half_span
        )
        self.assertAlmostEqual(
            sum(releases) / len(releases), 200.0, delta=max(expected * 3.0, 5.0)
        )

    def test_undeclared_features_pass_through_untouched(self) -> None:
        released, _ = privatise_update(update("n1", 200.0), epsilon=4.0)
        undeclared = set(FEATURE_NAMES) - set(DEFAULT_FEATURE_RANGES)
        for name in undeclared:
            self.assertEqual(released.statistics[name].mean, 0.0)


class FleetLearningTests(unittest.TestCase):
    """Cohorts, reputation, rotation and certified excision as one pipeline."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "reputation.json"
        self.fleet = [update(f"urban{i}", 200.0 + i) for i in range(4)] + [
            update(f"rural{i}", 20.0 + i) for i in range(4)
        ]

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_aggregation_is_hierarchical_not_flat(self) -> None:
        result = FleetLearning(self.path).aggregate(self.fleet)
        self.assertGreaterEqual(len(result["cohorts"]), 2)
        self.assertIn(
            "COHORTS_FORMED_BY_RF_ENVIRONMENT_NOT_BY_ADMINISTRATIVE_GROUPING", result["reasons"]
        )

    def test_reputation_survives_between_rounds(self) -> None:
        # Without persistence a node that lied last round starts clean, which
        # makes reputation decorative.
        first = FleetLearning(self.path)
        first.aggregate(self.fleet)
        self.assertTrue(self.path.exists())
        second = FleetLearning(self.path)
        self.assertEqual(second.reputation.scores(), first.reputation.scores())

    def test_unreadable_reputation_starts_neutral_not_trusted(self) -> None:
        self.path.write_text("{ not json", encoding="utf-8")
        self.assertEqual(FleetLearning(self.path).reputation.scores(), {})

    def test_the_aggregator_rotates_between_rounds(self) -> None:
        learning = FleetLearning(self.path)
        chosen = {
            learning.aggregate(self.fleet, round_index=index)["aggregator_node"]
            for index in range(3)
        }
        self.assertGreater(len(chosen), 1)

    def test_excising_a_node_also_costs_it_its_standing(self) -> None:
        # Otherwise it returns under the reputation it had before excision.
        learning = FleetLearning(self.path)
        learning.aggregate(self.fleet)
        before = learning.reputation.score("urban0")
        _remaining, certificate = learning.excise(self.fleet, "urban0")
        self.assertTrue(certificate.exact)
        self.assertLess(learning.reputation.score("urban0"), before)

    def test_the_excision_certificate_survives_a_restart(self) -> None:
        learning = FleetLearning(self.path)
        learning.excise(self.fleet, "urban0")
        reloaded = FleetLearning(self.path)
        self.assertLess(reloaded.reputation.score("urban0"), 1.0)


class ForecastReachesADecisionTests(unittest.TestCase):
    """INFOCOM 2024 feeding the optimizer, with the bound that makes it safe."""

    RISING = [20.0, 22.0, 25.0, 29.0, 34.0, 40.0, 47.0, 55.0]
    STABLE = [20.0, 21.0, 19.0, 20.0, 21.0, 19.0, 20.0, 20.0]

    def test_a_worsening_flow_becomes_a_demotion(self) -> None:
        advice = forecast_advice({"wifi": self.RISING, "eth": self.STABLE})
        self.assertIn("wifi", advice.degrading_profiles)
        self.assertNotIn("eth", advice.degrading_profiles)

    def test_the_demote_only_bound_is_stated_on_every_result(self) -> None:
        advice = forecast_advice({"eth": self.STABLE})
        self.assertIn("FORECAST_MAY_DEMOTE_A_PROFILE_AND_MAY_NEVER_PROMOTE_ONE", advice.reasons)

    def test_a_short_history_produces_no_advice(self) -> None:
        advice = forecast_advice({"wifi": [1.0, 2.0]})
        self.assertEqual(advice.degrading_profiles, frozenset())


    def test_advice_serializes(self) -> None:
        json.dumps(forecast_advice({"wifi": self.RISING}).to_dict())







class DetectionIsAFleetPropertyTests(unittest.TestCase):
    """ICC 2024 detectors, TVT 2022 cohorts and TWC 2025 reputation together."""

    def setUp(self) -> None:
        self.updates = [update(f"urban{i}", 200.0 + i) for i in range(4)] + [
            update(f"rural{i}", 20.0 + i) for i in range(4)
        ]

    def test_cohorts_come_from_rf_environment(self) -> None:
        # The property that matters is separation, not a particular cohort
        # count: the clusterer may split a wide environment further, but it
        # must never place an urban and a rural node together.
        assignments = cohort_assignments(self.updates)
        urban = {assignments[f"urban{i}"] for i in range(4)}
        rural = {assignments[f"rural{i}"] for i in range(4)}
        self.assertEqual(urban & rural, set())







class IntegrityGuardsLearningTests(unittest.TestCase):
    """MemMorph protecting the baseline the learning path derives."""

    @staticmethod
    def _example(rtt: float) -> Example:
        return Example(
            node_id="n1",
            decided_at=NOW,
            selected_profile_id="wifi",
            features={name: float(index + 1) for index, name in enumerate(FEATURE_NAMES)}
            | {"rtt_ms": rtt, "loss_percent": rtt / 10.0},
            outcome_rtt_ms=rtt,
            outcome_loss_percent=0.0,
            outcome_link_up=True,
            label_improved=True,
            horizon_seconds=60.0,
        )

    def test_a_failed_memory_audit_stops_the_baseline_being_built(self) -> None:
        examples = [self._example(20.0 + index) for index in range(30)]
        baseline, trust = trusted_baseline(
            examples, memory_report={"available": True, "trustworthy": False}
        )
        self.assertIsNone(baseline)
        self.assertFalse(trust.trusted)
        self.assertIn(
            "MEMORY_AUDIT_FAILED_SO_NO_BASELINE_IS_DERIVED_FROM_THIS_HISTORY", trust.reasons
        )

    def test_a_baseline_one_record_could_overturn_is_refused(self) -> None:
        # Exactly the leverage a small injection buys: 24 steady records and
        # one extreme one, which the aggregate would otherwise absorb.
        examples = [self._example(20.0 + (index % 5)) for index in range(24)]
        examples.append(self._example(90000.0))
        baseline, trust = trusted_baseline(
            examples, memory_report={"available": True, "trustworthy": True}
        )
        self.assertIsNone(baseline)
        self.assertIn(
            "A_SINGLE_RECORD_DOMINATES_THE_BASELINE_SO_IT_IS_NOT_USED", trust.reasons
        )
        self.assertTrue(any(r.startswith("DOMINATED_FEATURES:") for r in trust.reasons))

    def test_well_evidenced_history_produces_a_baseline(self) -> None:
        examples = [self._example(20.0 + (index % 7)) for index in range(30)]
        baseline, trust = trusted_baseline(
            examples, memory_report={"available": True, "trustworthy": True}
        )
        self.assertIsNotNone(baseline)
        self.assertTrue(trust.trusted)
        self.assertIn("HISTORY_PASSED_THE_MEMORY_AUDIT", trust.reasons)
        self.assertIn("NO_SINGLE_RECORD_DOMINATES_ANY_FEATURE", trust.reasons)

    def test_an_unavailable_audit_is_stated_not_assumed_to_pass(self) -> None:
        examples = [self._example(20.0 + (index % 7)) for index in range(30)]
        _baseline, trust = trusted_baseline(examples, memory_report=None)
        self.assertIn("MEMORY_AUDIT_UNAVAILABLE_SO_AUTHORSHIP_IS_UNVERIFIED", trust.reasons)

    def test_too_little_history_is_refused_by_the_baseline_builder(self) -> None:
        _baseline, trust = trusted_baseline([self._example(20.0)], memory_report=None)
        self.assertFalse(trust.trusted)

    def test_trust_serializes(self) -> None:
        examples = [self._example(20.0 + (index % 7)) for index in range(30)]
        json.dumps(trusted_baseline(examples, memory_report=None)[1].to_dict())


class BothReleasedMomentsArePrivatisedTests(unittest.TestCase):
    """The spread leaves the node too, so the spread is perturbed too.

    ``sum_squares`` used to be reconstructed from the *true* standard deviation
    while the mean was perturbed, so every node published its exact per-feature
    variance and any aggregator could recover it to zero error. The reason code
    said no exact mean left the node, which was true and beside the point.

    It matters more than a second-order statistic usually would: variance in
    signal is the multipath signature of an enclosed or reflective site, and it
    is the quantity cohort clustering and the heterogeneity assessment use to
    tell one environment from another. The exact thing the fleet fingerprints
    environments by was the thing walking out intact.
    """

    TRUE_MEAN = 62.0
    TRUE_SPREAD = 17.5
    COUNT = 200

    def _release(self, epsilon: float = 0.5):
        statistics = {
            "signal_percent": FeatureStatistics(
                self.COUNT,
                self.TRUE_MEAN,
                self.COUNT * (self.TRUE_MEAN**2 + self.TRUE_SPREAD**2),
            )
        }
        released, disclosure = privatise_update(
            FederatedUpdate("n1", NOW, self.COUNT, statistics), epsilon=epsilon
        )
        return released.statistics["signal_percent"], disclosure

    def test_the_exact_spread_never_leaves_the_node(self) -> None:
        # Many draws, because a single one could coincide by luck. Before the
        # fix every draw returned the true value to zero error.
        recovered = [self._release()[0].stddev() for _ in range(200)]
        exact = [value for value in recovered if abs(value - self.TRUE_SPREAD) < 1e-9]
        self.assertEqual(exact, [], "the true per-node spread was released unperturbed")

    def test_the_mean_is_still_perturbed(self) -> None:
        means = [self._release()[0].mean for _ in range(200)]
        exact = [value for value in means if abs(value - self.TRUE_MEAN) < 1e-9]
        self.assertEqual(exact, [])

    def test_a_released_spread_is_never_negative(self) -> None:
        # A negative variance describes a distribution that cannot exist, and
        # would make stddev() return a complex or NaN value downstream.
        for _ in range(200):
            self.assertGreaterEqual(self._release()[0].stddev(), 0.0)

    def test_the_count_is_still_released_exactly(self) -> None:
        # Deliberate and unchanged: the observation count is not a siting
        # signal, and the aggregator needs it to weight contributions.
        self.assertEqual(self._release()[0].count, self.COUNT)

    def test_the_release_stays_json_serialisable(self) -> None:
        released, disclosure = self._release()
        json.dumps(released.to_dict())
        json.dumps(disclosure.to_dict())


if __name__ == "__main__":
    unittest.main()
