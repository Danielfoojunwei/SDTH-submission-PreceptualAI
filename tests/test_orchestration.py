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
from kval_node.intrusion import NodeBaseline
from kval_node.mobility import Fix
from kval_node.orchestration import (
    DEFAULT_FEATURE_RANGES,
    FleetLearning,
    cohort_assignments,
    detect_fleet_event,
    fixes_from_estimates,
    forecast_advice,
    predict_track,
    privatise_update,
    trusted_baseline,
    verdicts_from_reports,
)
from kval_node.spectrum import BandStatus

from tests.test_fabric import report as build_report

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

    def test_a_forecast_demotion_is_weaker_than_measured_evidence(self) -> None:
        # A prediction must be able to break a tie, never to overrule what the
        # sensor actually measured.
        from kval_node.optimizer import (
            FORECAST_DEGRADED_MULTIPLIER,
            SPECTRUM_DEGRADED_MULTIPLIER,
        )

        self.assertGreater(FORECAST_DEGRADED_MULTIPLIER, SPECTRUM_DEGRADED_MULTIPLIER)

    def test_advice_serializes(self) -> None:
        json.dumps(forecast_advice({"wifi": self.RISING}).to_dict())


class LocalisationFeedsTrackingTests(unittest.TestCase):
    """Sensors 2018 producing the history WPC 2024 consumes."""

    def _record(self, minutes: float, latitude: float, *, available: bool = True) -> dict:
        payload = {"available": available, "band_id": "b"}
        if available:
            payload |= {
                "latitude_deg": latitude,
                "longitude_deg": 103.8,
                "uncertainty_radius_m": 30.0,
            }
        return {"recorded_at": (NOW + timedelta(minutes=minutes)).isoformat(), "payload": payload}

    def test_recorded_estimates_become_track_fixes(self) -> None:
        fixes = fixes_from_estimates([self._record(i, 1.30 + i * 0.001) for i in range(5)])
        self.assertEqual(len(fixes), 5)
        self.assertEqual(fixes[0].uncertainty_radius_m, 30.0)

    def test_a_refused_estimate_never_becomes_a_track_point(self) -> None:
        # "We could not tell" is not a place the emitter has been.
        records = [self._record(0, 1.30), self._record(1, 0.0, available=False)]
        self.assertEqual(len(fixes_from_estimates(records)), 1)

    def test_the_place_radius_follows_localisation_uncertainty(self) -> None:
        # A radius tighter than the accuracy that produced the fixes turns
        # noise into a tour of places the emitter never visited.
        jittering = [
            Fix(NOW + timedelta(minutes=index), 1.3000 + 0.0001 * (index % 2), 103.8, 60.0)
            for index in range(10)
        ]
        loose = predict_track(jittering, horizon=timedelta(minutes=5), place_radius_m=150.0)
        tight = predict_track(jittering, horizon=timedelta(minutes=5), place_radius_m=1.0)
        self.assertLess(loose["distinct_places"], tight["distinct_places"])

    def test_a_track_predicts_or_declines_never_guesses(self) -> None:
        fixes = [Fix(NOW + timedelta(minutes=i), 1.30, 103.8, 30.0) for i in range(6)]
        result = predict_track(fixes, horizon=timedelta(minutes=5), place_radius_m=150.0)
        if result["predicted"]:
            self.assertIsNotNone(result["latitude_deg"])
        else:
            self.assertIsNone(result["latitude_deg"])

    def test_result_serializes(self) -> None:
        fixes = [Fix(NOW + timedelta(minutes=i), 1.30, 103.8, 30.0) for i in range(6)]
        json.dumps(predict_track(fixes, horizon=timedelta(minutes=5), place_radius_m=150.0))


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

    def test_a_node_without_a_baseline_still_contributes_its_band_status(self) -> None:
        reports = [
            build_report(f"n{i}", status=BandStatus.JAMMING_SUSPECTED, produced_at=NOW)
            for i in range(3)
        ]
        verdicts = verdicts_from_reports(reports, {}, {"n0": "c1", "n1": "c2", "n2": "c3"})
        self.assertTrue(all(v.anomalous for v in verdicts))
        self.assertIn(
            "NO_LOCAL_BASELINE_SO_THE_REPORTED_BAND_STATUS_STANDS_IN", verdicts[0].reasons
        )

    def test_correlated_anomalies_across_cohorts_become_an_event(self) -> None:
        reports = [
            build_report(f"n{i}", status=BandStatus.JAMMING_SUSPECTED, produced_at=NOW)
            for i in range(3)
        ]
        verdicts = verdicts_from_reports(reports, {}, {"n0": "c1", "n1": "c2", "n2": "c3"})
        finding = detect_fleet_event(verdicts, {})
        self.assertTrue(finding["event"])

    def test_a_clear_fleet_is_not_an_event(self) -> None:
        reports = [
            build_report(f"n{i}", status=BandStatus.CLEAR, produced_at=NOW) for i in range(3)
        ]
        verdicts = verdicts_from_reports(reports, {}, {"n0": "c1", "n1": "c2", "n2": "c3"})
        self.assertFalse(detect_fleet_event(verdicts, {})["event"])

    def test_reputation_gates_who_can_raise_a_fleet_finding(self) -> None:
        reports = [
            build_report(f"n{i}", status=BandStatus.JAMMING_SUSPECTED, produced_at=NOW)
            for i in range(3)
        ]
        verdicts = verdicts_from_reports(reports, {}, {"n0": "c1", "n1": "c2", "n2": "c3"})
        discredited = {"n0": 0.05, "n1": 0.05, "n2": 0.05}
        self.assertFalse(detect_fleet_event(verdicts, discredited)["event"])

    def test_a_node_with_its_own_baseline_uses_its_own_detector(self) -> None:
        baseline = NodeBaseline(
            node_id="n0",
            samples=50,
            means={"contested_band_count": 0.0, "noise_floor_dbm": 0.0},
            stddevs={"contested_band_count": 0.2, "noise_floor_dbm": 1.0},
        )
        reports = [build_report("n0", status=BandStatus.JAMMING_SUSPECTED, produced_at=NOW)]
        verdicts = verdicts_from_reports(reports, {"n0": baseline}, {"n0": "c1"})
        self.assertIn("READINGS_DEPART_FROM_THIS_NODE_S_OWN_NORMAL", verdicts[0].reasons)


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


if __name__ == "__main__":
    unittest.main()
