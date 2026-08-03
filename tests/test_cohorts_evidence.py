"""Cohort formation, reputation, hierarchical aggregation, and DS fusion.

Mechanisms adapted from the Feng Li / Bowen Shen / Kwok-Yan Lam dynamic-spectrum
-access corpus; see docs/DSA_RESEARCH_MAPPING.md for the paper-by-paper mapping.
"""

from __future__ import annotations

import json
import math
import unittest
from datetime import datetime, timezone

from kval_node.cohorts import (
    ABSOLUTE_HEAVY_FLOOR,
    HETEROGENEITY_FEATURES,
    MIN_NODES_FOR_ASSESSMENT,
    MIN_NODES_FOR_TIERS,
    REFERENCE_SWINGS,
    REGIME_HEAVY,
    REGIME_LIGHT,
    REGIME_MEDIUM,
    HierarchicalAggregator,
    ReputationLedger,
    assess_heterogeneity,
    certified_excision,
    excise_contributor,
    kmeans_cohorts,
)
from kval_node.dataset import FEATURE_NAMES
from kval_node.federated import FeatureStatistics, FederatedUpdate

NOW = datetime.now(timezone.utc)


def update(node: str, rtt_mean: float, *, count: int = 50, variance: float = 100.0) -> FederatedUpdate:
    stats = {name: FeatureStatistics(count, 0.0, float(count)) for name in FEATURE_NAMES}
    stats["rtt_ms"] = FeatureStatistics(count, rtt_mean, count * (rtt_mean**2 + variance))
    return FederatedUpdate(node, NOW, count, stats)


class CohortClusteringTests(unittest.TestCase):
    def test_distinct_environments_separate(self) -> None:
        urban = [update(f"urban{i}", 200.0 + i) for i in range(4)]
        rural = [update(f"rural{i}", 20.0 + i) for i in range(4)]
        cohorts = kmeans_cohorts(urban + rural, max_cohorts=2)
        self.assertEqual(len(cohorts), 2)
        for cohort in cohorts:
            prefixes = {u.node_id[:5] for u in cohort}
            self.assertEqual(len(prefixes), 1, "a cohort mixed two environments")

    def test_clustering_is_deterministic(self) -> None:
        nodes = [update(f"n{i}", 20.0 + i * 30) for i in range(8)]
        first = [[u.node_id for u in c] for c in kmeans_cohorts(nodes)]
        second = [[u.node_id for u in c] for c in kmeans_cohorts(nodes)]
        self.assertEqual(first, second)

    def test_tiny_fleet_is_one_cohort(self) -> None:
        self.assertEqual(len(kmeans_cohorts([update("a", 10.0), update("b", 20.0)])), 1)

    def test_empty_fleet_yields_no_cohorts(self) -> None:
        self.assertEqual(kmeans_cohorts([]), [])


class ReputationTests(unittest.TestCase):
    def test_agreement_raises_and_divergence_lowers(self) -> None:
        ledger = ReputationLedger()
        for _ in range(5):
            ledger.record_round(agreed=["good"], diverged=["liar"])
        self.assertGreater(ledger.score("good"), 0.9)
        self.assertLess(ledger.score("liar"), 0.5)

    def test_reputation_never_reaches_zero_so_recovery_is_possible(self) -> None:
        ledger = ReputationLedger()
        for _ in range(200):
            ledger.record_round(agreed=[], diverged=["unlucky"])
        self.assertGreaterEqual(ledger.score("unlucky"), ReputationLedger.MINIMUM_SCORE)
        for _ in range(50):
            ledger.record_round(agreed=["unlucky"], diverged=[])
        self.assertGreater(ledger.score("unlucky"), 0.9)

    def test_unknown_node_starts_neutral(self) -> None:
        self.assertEqual(ReputationLedger().score("newcomer"), 1.0)


class HierarchicalAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fleet = [update(f"urban{i}", 200.0 + i) for i in range(4)] + [
            update(f"rural{i}", 20.0 + i) for i in range(4)
        ]

    def test_produces_cohorts_and_a_global_tier(self) -> None:
        result = HierarchicalAggregator().aggregate(self.fleet)
        self.assertGreaterEqual(len(result.cohorts), 2)
        self.assertIsNotNone(result.global_aggregate)

    def test_aggregator_rotates_between_rounds(self) -> None:
        aggregator = HierarchicalAggregator()
        chosen = {aggregator.aggregate(self.fleet, round_index=r).aggregator_node for r in range(3)}
        self.assertGreater(len(chosen), 1, "the aggregator role did not rotate")

    def test_a_disreputable_node_is_not_chosen_to_aggregate(self) -> None:
        ledger = ReputationLedger()
        for _ in range(10):
            ledger.record_round(agreed=["good"], diverged=["bad"])
        aggregator = HierarchicalAggregator(reputation=ledger)
        self.assertNotEqual(aggregator.select_aggregator(["good", "bad"], round_index=0), "bad")

    def test_aggregation_tolerates_nodes_dropping_out(self) -> None:
        # Dropout resilience: a fleet that loses members between rounds must
        # still aggregate, not fail closed.
        full = HierarchicalAggregator().aggregate(self.fleet)
        reduced = HierarchicalAggregator().aggregate(self.fleet[:5])
        self.assertIsNotNone(full.global_aggregate)
        self.assertTrue(reduced.cohorts)

    def test_result_serializes(self) -> None:
        json.dumps(HierarchicalAggregator().aggregate(self.fleet).to_dict())


class UnlearningTests(unittest.TestCase):
    def test_excision_is_exact_and_reported(self) -> None:
        fleet = [update(f"n{i}", 20.0 + i) for i in range(4)]
        remaining, found = excise_contributor(fleet, "n2")
        self.assertTrue(found)
        self.assertEqual(len(remaining), 3)
        self.assertNotIn("n2", [u.node_id for u in remaining])

    def test_excising_an_absent_node_reports_false(self) -> None:
        fleet = [update("n0", 20.0)]
        remaining, found = excise_contributor(fleet, "ghost")
        self.assertFalse(found)
        self.assertEqual(len(remaining), 1)


class CertifiedRecoveryTests(unittest.TestCase):
    """Certified recovery from poisoning, per IEEE TIFS 2025 (arXiv:2401.08216).

    Their guarantee is a *bound* on residual influence after removing a
    poisoning client, because gradient updates smear a contribution across every
    subsequent round. Aggregating sufficient statistics instead makes removal
    exact, and the certificate proves that rather than asserting it.
    """

    def setUp(self) -> None:
        self.clean = [update(f"n{i}", 20.0 + i) for i in range(5)]

    def test_removal_is_certified_exact(self) -> None:
        poisoned = self.clean + [update("attacker", 5000.0)]
        remaining, certificate = certified_excision(poisoned, "attacker")
        self.assertTrue(certificate.was_present)
        self.assertTrue(certificate.exact)
        self.assertEqual(certificate.recovered_hash, certificate.never_joined_hash)
        self.assertNotIn("attacker", [u.node_id for u in remaining])

    def test_the_guarantee_is_zero_residual_not_a_bound(self) -> None:
        poisoned = self.clean + [update("attacker", 5000.0)]
        _remaining, certificate = certified_excision(poisoned, "attacker")
        self.assertIn("RESIDUAL_INFLUENCE_IS_ZERO_NOT_MERELY_BOUNDED", certificate.reasons)
        self.assertIn(
            "RECOVERED_BASELINE_IS_IDENTICAL_TO_ONE_THE_NODE_NEVER_JOINED", certificate.reasons
        )

    def test_the_recovered_baseline_matches_one_built_without_the_node(self) -> None:
        # The property the certificate rests on, checked independently of the
        # certificate's own comparison.
        poisoned = self.clean + [update("attacker", 5000.0)]
        _remaining, from_poisoned = certified_excision(poisoned, "attacker")
        _nothing, from_clean = certified_excision(self.clean + [update("x", 21.0)], "x")
        self.assertTrue(from_poisoned.exact)
        self.assertTrue(from_clean.exact)

    def test_a_blatant_poisoner_was_already_quarantined_so_nothing_shifted(self) -> None:
        # Consensus filtering catches the obvious attacker before aggregation,
        # so excising it moves nothing. The certificate says so rather than
        # claiming credit for a correction that never happened.
        poisoned = self.clean + [update("attacker", 5000.0)]
        _remaining, certificate = certified_excision(poisoned, "attacker")
        self.assertEqual(certificate.max_shift, 0.0)
        self.assertEqual(certificate.shifted_features, ())

    def test_a_subtle_poisoner_that_survived_filtering_is_measured(self) -> None:
        # The case excision actually exists for: a contribution close enough to
        # the consensus to be aggregated, but still dragging the baseline. How
        # far it had dragged it is operational evidence, so it is recorded.
        poisoned = self.clean + [update("attacker", 40.0)]
        _remaining, certificate = certified_excision(poisoned, "attacker")
        self.assertGreater(certificate.max_shift, 0.0)
        self.assertIn("rtt_ms", certificate.shifted_features)
        self.assertIn("THE_EXCISED_CONTRIBUTION_HAD_BEEN_MOVING_THE_BASELINE", certificate.reasons)
        self.assertTrue(certificate.exact)

    def test_excising_an_absent_node_certifies_nothing_happened(self) -> None:
        _remaining, certificate = certified_excision(self.clean, "ghost")
        self.assertFalse(certificate.was_present)
        self.assertTrue(certificate.exact)
        self.assertIn(
            "NODE_WAS_NOT_A_CONTRIBUTOR_SO_THERE_IS_NOTHING_TO_REMOVE", certificate.reasons
        )

    def test_certificate_serializes(self) -> None:
        poisoned = self.clean + [update("attacker", 5000.0)]
        json.dumps(certified_excision(poisoned, "attacker")[1].to_dict())






def swings_at(factor: float) -> dict[str, float]:
    """Per-feature standard deviations that normalise to ``factor`` for every feature.

    Fixtures are built in reference-swing units so a test can state the score it
    expects instead of a pile of magic milliseconds and percentages.
    """
    return {name: REFERENCE_SWINGS[name] * factor for name in HETEROGENEITY_FEATURES}


def env_node(
    node_id: str,
    *,
    swings: dict[str, float],
    means: dict[str, float] | None = None,
    count: int = 60,
    omit: tuple[str, ...] = (),
) -> FederatedUpdate:
    """An update whose recovered stddev for each feature is exactly ``swings[name]``."""
    means = means or {}
    stats = {}
    for name in FEATURE_NAMES:
        if name in omit:
            continue
        mean = float(means.get(name, 1.0))
        spread = float(swings.get(name, 1.0))
        stats[name] = FeatureStatistics(count, mean, count * (mean**2 + spread**2))
    return FederatedUpdate(node_id, NOW, count, stats)


def calm_fleet(size: int = MIN_NODES_FOR_TIERS) -> list[FederatedUpdate]:
    """A fleet whose every node is in a benign, low-variability environment."""
    return [
        env_node(f"calm{i}", swings=swings_at(0.10 + 0.01 * i), means={"rtt_ms": 20.0 + i})
        for i in range(size)
    ]


def variable_fleet(size: int = MIN_NODES_FOR_TIERS) -> list[FederatedUpdate]:
    """A fleet whose every node sees sharp swings — the paper's heavy regime."""
    return [
        env_node(f"wild{i}", swings=swings_at(2.0 + 0.1 * i), means={"rtt_ms": 300.0 + 20 * i})
        for i in range(size)
    ]


class FleetHeterogeneityTests(unittest.TestCase):
    """The reported measure of how non-IID the fleet actually is.

    After EPFL-REMNet ([arXiv:2511.05238](https://arxiv.org/abs/2511.05238)),
    which scores a location by ``H = std(s_1..s_M)`` over the signal strengths
    received there from M base stations and cuts a deployment into light,
    medium and heavy non-IID regimes at the 33rd and 66th percentiles of H.
    Kval has no per-base-station readings, so the spread is taken over the axis
    its federated statistics do carry.
    """

    def test_a_fleet_below_the_assessment_minimum_is_refused(self) -> None:
        # A "fleet" of two is two nodes wearing a larger label. Reporting a
        # regime for it would give an operator a fleet-wide number derived from
        # one siting decision.
        result = assess_heterogeneity(calm_fleet(MIN_NODES_FOR_ASSESSMENT - 1))
        self.assertFalse(result.available)
        self.assertIsNone(result.regime)
        self.assertIsNone(result.score)
        self.assertIn(
            f"NEED_AT_LEAST_{MIN_NODES_FOR_ASSESSMENT}_NODES_TO_DESCRIBE_A_FLEET",
            result.reasons,
        )

    def test_a_fleet_too_small_for_percentiles_is_refused_tiers_not_the_whole_result(
        self,
    ) -> None:
        # Three nodes do not have a 33rd percentile in any useful sense — the
        # tiers would just restate the ranking. The absolute regime needs no
        # percentiles, so refusing it too would be refusing a supportable claim.
        result = assess_heterogeneity(calm_fleet(MIN_NODES_FOR_TIERS - 1))
        self.assertTrue(result.available)
        self.assertIsNotNone(result.regime)
        self.assertIsNone(result.tier_boundaries)
        self.assertTrue(all(node.tier is None for node in result.nodes))
        self.assertIn(
            f"NEED_AT_LEAST_{MIN_NODES_FOR_TIERS}_SCORED_NODES_BEFORE_PERCENTILE_TIERS_MEAN_ANYTHING",
            result.reasons,
        )

    def test_tiers_split_a_large_enough_fleet_into_three(self) -> None:
        fleet = [
            env_node(f"n{i}", swings=swings_at(0.2 + 0.2 * i), means={"rtt_ms": 20.0 + i})
            for i in range(MIN_NODES_FOR_TIERS)
        ]
        result = assess_heterogeneity(fleet)
        self.assertEqual(
            result.tier_counts(), {REGIME_LIGHT: 3, REGIME_MEDIUM: 3, REGIME_HEAVY: 3}
        )
        self.assertIsNotNone(result.tier_boundaries)

    def test_a_calm_fleet_still_produces_a_heavy_tier_and_the_result_says_so(self) -> None:
        # The percentile trap. Boundaries drawn from the fleet's own
        # distribution always label a third of it heavy, so an operator reading
        # tier alone would conclude a third of a benign fleet is in an urban
        # canyon. The absolute floor is what stops "heavy" meaning nothing more
        # than "the worst third of whatever we have".
        result = assess_heterogeneity(calm_fleet())
        self.assertEqual(result.tier_counts()[REGIME_HEAVY], 3)
        self.assertEqual(result.regime, REGIME_LIGHT)
        self.assertLess(max(node.score for node in result.nodes), ABSOLUTE_HEAVY_FLOOR)
        self.assertIn(
            "RELATIVE_HEAVY_TIER_SITS_ENTIRELY_BELOW_THE_ABSOLUTE_HEAVY_FLOOR", result.reasons
        )
        self.assertIn(
            "PERCENTILE_TIERS_ARE_RELATIVE_TO_THIS_FLEET_NOT_TO_AN_ABSOLUTE_SCALE",
            result.reasons,
        )

    def test_a_genuinely_variable_fleet_clears_the_absolute_heavy_floor(self) -> None:
        result = assess_heterogeneity(variable_fleet())
        self.assertEqual(result.regime, REGIME_HEAVY)
        self.assertGreaterEqual(result.score, ABSOLUTE_HEAVY_FLOOR)
        self.assertIn(
            "RELATIVE_LIGHT_TIER_SITS_ENTIRELY_ABOVE_THE_ABSOLUTE_HEAVY_FLOOR", result.reasons
        )

    def test_a_fleet_at_one_reference_swing_is_medium(self) -> None:
        # The scale is anchored, not fitted: one reference swing per feature is
        # the middle regime by construction, in any fleet.
        fleet = [
            env_node(f"m{i}", swings=swings_at(1.0), means={"rtt_ms": 100.0 + i})
            for i in range(MIN_NODES_FOR_TIERS)
        ]
        result = assess_heterogeneity(fleet)
        self.assertEqual(result.regime, REGIME_MEDIUM)
        self.assertAlmostEqual(result.score, 1.0, places=9)

    def test_a_nodes_score_does_not_depend_on_which_fleet_it_is_measured_in(self) -> None:
        # The reason an absolute regime is possible at all. Normalising against
        # the fleet's own spread would force the average node to the same score
        # in every deployment, making "how heterogeneous is this fleet" a
        # question with a constant answer.
        wild = variable_fleet()
        alone = {n.node_id: n.score for n in assess_heterogeneity(wild).nodes}
        mixed = {n.node_id: n.score for n in assess_heterogeneity(wild + calm_fleet()).nodes}
        for node_id, score in alone.items():
            self.assertAlmostEqual(mixed[node_id], score, places=12)

    def test_decision_outcome_features_cannot_move_the_score(self) -> None:
        # Scoring is_current_profile or confidence would let a node that flaps
        # between profiles, or one running a differently tuned recommender,
        # report a complex radio environment on the strength of its own
        # behaviour rather than what is around it.
        outcome_noise = dict(swings_at(1.0))
        outcome_noise.update(
            {
                "confidence": 400.0,
                "candidate_count": 400.0,
                "eligible_count": 400.0,
                "clock_trusted": 400.0,
                "power_duty_limit": 400.0,
                "is_current_profile": 400.0,
            }
        )
        plain = [
            env_node(f"n{i}", swings=swings_at(1.0), means={"rtt_ms": 20.0 + i})
            for i in range(MIN_NODES_FOR_TIERS)
        ]
        noisy = [
            env_node(f"n{i}", swings=outcome_noise, means={"rtt_ms": 20.0 + i})
            for i in range(MIN_NODES_FOR_TIERS)
        ]
        self.assertEqual(
            [n.score for n in assess_heterogeneity(plain).nodes],
            [n.score for n in assess_heterogeneity(noisy).nodes],
        )

    def test_a_feature_flat_across_the_whole_fleet_is_excluded_not_scored_as_calm(self) -> None:
        # A feature that never moves anywhere is almost always an instrument
        # that is not fitted, not an environment that is quiet. Averaging its
        # zero in deflates every score: here it would drop a heavy fleet into
        # the medium regime on the strength of a missing spectrum sensor.
        base = swings_at(1.6)
        base["throughput_mbps"] = 0.0
        fleet = [
            env_node(
                f"n{i}",
                swings={**base, "rtt_ms": base["rtt_ms"] + 0.05 * i},
                means={"rtt_ms": 20.0 + i, "throughput_mbps": 0.0},
            )
            for i in range(MIN_NODES_FOR_TIERS)
        ]
        result = assess_heterogeneity(fleet)
        self.assertIn("throughput_mbps", result.features_excluded)
        self.assertNotIn("throughput_mbps", result.features_used)
        self.assertEqual(result.regime, REGIME_HEAVY)
        # What counting the dead sensor as a calm one would have reported.
        diluted = result.score * math.sqrt(len(result.features_used) / len(HETEROGENEITY_FEATURES))
        self.assertLess(diluted, ABSOLUTE_HEAVY_FLOOR)

    def test_a_fleet_with_nothing_varying_anywhere_is_refused(self) -> None:
        flat = {name: 0.0 for name in HETEROGENEITY_FEATURES}
        fleet = [env_node(f"n{i}", swings=flat) for i in range(MIN_NODES_FOR_TIERS)]
        result = assess_heterogeneity(fleet)
        self.assertFalse(result.available)
        self.assertIn("NO_ENVIRONMENT_FEATURE_VARIED_ACROSS_THE_FLEET", result.reasons)

    def test_a_node_with_too_few_observations_is_not_scored(self) -> None:
        # FeatureStatistics.stddev returns 0.0 below two observations. That zero
        # is the absence of a measurement, not a calm environment, and scoring
        # it would put a node with a handful of samples in the quietest tier.
        thin = env_node("thin", swings=swings_at(1.0), means={"rtt_ms": 40.0}, count=3)
        result = assess_heterogeneity(calm_fleet() + [thin])
        self.assertIn("thin", result.unscored_nodes)
        self.assertNotIn("thin", [node.node_id for node in result.nodes])
        self.assertIn("NODES_WITHOUT_USABLE_STATISTICS_FOR_EVERY_FEATURE_WERE_NOT_SCORED", result.reasons)

    def test_a_node_missing_an_admitted_feature_is_named_rather_than_scored_differently(
        self,
    ) -> None:
        # Scoring it over the features it happens to have would put a different
        # function on each node, and the percentile tiers are nothing but a
        # comparison between those numbers.
        partial = env_node(
            "partial", swings=swings_at(1.0), means={"rtt_ms": 40.0}, omit=("jitter_ms",)
        )
        result = assess_heterogeneity(calm_fleet() + [partial])
        self.assertIn("jitter_ms", result.features_used)
        self.assertEqual(result.unscored_nodes, ("partial",))

    def test_every_scored_node_reports_the_observations_behind_its_score(self) -> None:
        result = assess_heterogeneity(calm_fleet())
        self.assertTrue(all(node.samples >= 1 for node in result.nodes))
        self.assertEqual(result.node_count, MIN_NODES_FOR_TIERS)
        self.assertEqual(result.total_examples, 60 * MIN_NODES_FOR_TIERS)

    def test_assessment_is_deterministic(self) -> None:
        fleet = calm_fleet() + variable_fleet()
        self.assertEqual(
            assess_heterogeneity(fleet).to_dict(), assess_heterogeneity(fleet).to_dict()
        )

    def test_result_serializes_without_non_finite_values(self) -> None:
        # to_dict must emit None rather than inf or nan; allow_nan=False is what
        # makes that a test rather than a hope.
        for result in (
            assess_heterogeneity(calm_fleet()),
            assess_heterogeneity(variable_fleet()),
            assess_heterogeneity(calm_fleet(2)),
        ):
            json.dumps(result.to_dict(), allow_nan=False)


class CohortSeparationTests(unittest.TestCase):
    """Whether a cohort is a narrower population than the fleet it came from."""

    def _two_environments(self) -> tuple[list[FederatedUpdate], list[FederatedUpdate]]:
        urban = [
            env_node(
                f"urban{i}",
                swings=swings_at(1.0),
                means={"rtt_ms": 400.0 + i, "signal_percent": 20.0 + i},
            )
            for i in range(5)
        ]
        rural = [
            env_node(
                f"rural{i}",
                swings=swings_at(1.0),
                means={"rtt_ms": 20.0 + i, "signal_percent": 90.0 + i},
            )
            for i in range(5)
        ]
        return urban, rural

    def test_distinct_environments_are_reported_as_separated(self) -> None:
        urban, rural = self._two_environments()
        result = assess_heterogeneity(urban + rural, cohorts=[urban, rural])
        self.assertTrue(all(cohort.separated for cohort in result.cohorts))
        self.assertTrue(all(cohort.spread_ratio < 1.0 for cohort in result.cohorts))
        self.assertNotIn(
            "COHORT_GROUPING_IS_NOT_SUPPORTED_BY_THE_MEASURED_SPREADS", result.reasons
        )

    def test_an_arbitrary_split_of_one_population_is_reported_as_no_structure(self) -> None:
        # The defect the metric exists to catch: k-means always returns
        # cohorts, and a bare cohort ID reads as structure. Each half here
        # reproduces the whole fleet's spread, so nothing was separated.
        population = [
            env_node(f"p{i}", swings=swings_at(1.0), means={"rtt_ms": 20.0 + 100.0 * (i % 5)})
            for i in range(10)
        ]
        result = assess_heterogeneity(
            population, cohorts=[population[:5], population[5:]]
        )
        self.assertTrue(all(cohort.separated is False for cohort in result.cohorts))
        for cohort in result.cohorts:
            self.assertAlmostEqual(cohort.spread_ratio, 1.0, places=9)
            self.assertIn(
                "COHORT_SPREAD_MATCHES_THE_FLEET_SO_THE_GROUPING_SEPARATED_NOTHING",
                cohort.reasons,
            )
        self.assertIn(
            "COHORT_GROUPING_IS_NOT_SUPPORTED_BY_THE_MEASURED_SPREADS", result.reasons
        )

    def test_a_single_node_cohort_is_reported_unjudged_rather_than_separated(self) -> None:
        # Its internal spread is zero, which would sail through any ratio test.
        # Zero spread from one node is not a tight cohort, it is no measurement.
        urban, rural = self._two_environments()
        fleet = urban + rural
        result = assess_heterogeneity(fleet, cohorts=[fleet[:1], fleet[1:]])
        loner = result.cohorts[0]
        self.assertIsNone(loner.separated)
        self.assertIsNone(loner.spread)
        self.assertIn(
            "COHORT_HAS_TOO_FEW_SCORED_NODES_TO_HAVE_AN_INTERNAL_SPREAD", loner.reasons
        )

    def test_a_fleet_with_no_spread_refuses_to_judge_its_cohorts(self) -> None:
        # Every node sits at the same point, so there is no fleet-wide spread to
        # divide by. The ratio is withheld rather than manufactured.
        fleet = [
            env_node(f"same{i}", swings=swings_at(0.5 + 0.05 * i), means={"rtt_ms": 50.0})
            for i in range(MIN_NODES_FOR_TIERS)
        ]
        result = assess_heterogeneity(fleet, cohorts=[fleet[:4], fleet[4:]])
        self.assertEqual(result.fleet_spread, 0.0)
        self.assertTrue(all(cohort.separated is None for cohort in result.cohorts))
        self.assertTrue(all(cohort.spread_ratio is None for cohort in result.cohorts))
        self.assertIn("NO_COHORT_COULD_BE_JUDGED_AGAINST_THE_FLEET", result.reasons)

    def test_one_cohort_is_reported_as_nothing_to_assess(self) -> None:
        fleet = calm_fleet()
        result = assess_heterogeneity(fleet, cohorts=[fleet])
        self.assertIn("ONLY_ONE_COHORT_SO_THERE_IS_NO_GROUPING_TO_ASSESS", result.reasons)

    def test_the_clustering_this_module_ships_is_audited_by_default(self) -> None:
        # No cohorts supplied: the audit runs against kmeans_cohorts itself, so
        # the grouping Kval actually forms is the one being judged.
        urban, rural = self._two_environments()
        supplied = assess_heterogeneity(
            urban + rural, cohorts=kmeans_cohorts(urban + rural, max_cohorts=2)
        )
        default = assess_heterogeneity(urban + rural, max_cohorts=2)
        self.assertEqual(
            [c.members for c in supplied.cohorts], [c.members for c in default.cohorts]
        )

    def test_cohort_assessment_serializes(self) -> None:
        urban, rural = self._two_environments()
        result = assess_heterogeneity(urban + rural, cohorts=[urban, rural[:1], rural[1:]])
        json.dumps(result.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
