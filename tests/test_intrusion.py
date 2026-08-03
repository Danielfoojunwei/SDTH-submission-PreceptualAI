"""Fleet-level anomaly detection without a shared model.

Architecture from Lam et al., "Effective Intrusion Detection in Heterogeneous
IoT Networks via Ensemble Knowledge Distillation-based Federated Learning",
IEEE ICC 2024 (arXiv:2401.11968) — nodes keep their own detectors and share
outputs rather than parameters. See docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from kval_node.intrusion import (
    MIN_BASELINE_SAMPLES,
    FleetDetector,
    LocalDetector,
    LocalVerdict,
    NodeBaseline,
    expected_false_alarm_rate,
    verdict_agreement,
)

NOW = datetime(2026, 6, 1, 9, 0, 0, tzinfo=timezone.utc)


def quiet_history(noise_floor: float, count: int = 40) -> list[dict[str, float]]:
    """A node's own history, jittering around its own normal."""
    jitter = (0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.8)
    return [
        {
            "noise_floor_dbm": noise_floor + jitter[index % len(jitter)],
            "rtt_ms": 40.0 + jitter[index % len(jitter)],
        }
        for index in range(count)
    ]


def detector(node_id: str, noise_floor: float, count: int = 40) -> LocalDetector:
    return LocalDetector(NodeBaseline.from_observations(node_id, quiet_history(noise_floor, count)))


class BaselineTests(unittest.TestCase):
    def test_each_node_gets_its_own_normal(self) -> None:
        # The heterogeneity the source paper is about: a hilltop node and an
        # airport-adjacent node have nothing to average.
        quiet = NodeBaseline.from_observations("quiet", quiet_history(-100.0))
        loud = NodeBaseline.from_observations("loud", quiet_history(-70.0))
        self.assertNotAlmostEqual(
            quiet.means["noise_floor_dbm"], loud.means["noise_floor_dbm"]
        )

    def test_a_short_history_is_not_usable(self) -> None:
        self.assertFalse(NodeBaseline.from_observations("n", quiet_history(-100.0, 5)).usable())

    def test_an_empty_history_is_not_usable(self) -> None:
        self.assertFalse(NodeBaseline.from_observations("n", []).usable())

    def test_a_feature_seen_once_is_excluded(self) -> None:
        # One sample gives no spread, so it cannot be judged against.
        rows = quiet_history(-100.0) + [{"one_off": 5.0}]
        self.assertNotIn("one_off", NodeBaseline.from_observations("n", rows).means)


class LocalDetectionTests(unittest.TestCase):
    def test_normal_readings_are_not_anomalous(self) -> None:
        verdict = detector("n1", -100.0).judge(
            {"noise_floor_dbm": -100.5, "rtt_ms": 40.2}, cohort_id="c1", observed_at=NOW
        )
        self.assertFalse(verdict.anomalous)
        self.assertIn("READINGS_WITHIN_THIS_NODE_S_OWN_NORMAL", verdict.reasons)

    def test_a_departure_from_this_node_s_normal_is_anomalous(self) -> None:
        verdict = detector("n1", -100.0).judge(
            {"noise_floor_dbm": -70.0, "rtt_ms": 40.0}, cohort_id="c1", observed_at=NOW
        )
        self.assertTrue(verdict.anomalous)
        self.assertEqual([d.feature for d in verdict.deviations], ["noise_floor_dbm"])
        self.assertEqual(verdict.deviations[0].direction, 1)

    def test_the_same_reading_is_normal_for_one_node_and_not_another(self) -> None:
        # A fleet-wide absolute threshold would be wrong at both nodes. This is
        # the whole argument for per-node detectors.
        reading = {"noise_floor_dbm": -70.0}
        self.assertTrue(detector("quiet", -100.0).judge(reading, cohort_id="c1", observed_at=NOW).anomalous)
        self.assertFalse(detector("loud", -70.0).judge(reading, cohort_id="c2", observed_at=NOW).anomalous)

    def test_direction_is_recorded(self) -> None:
        verdict = detector("n1", -100.0).judge(
            {"noise_floor_dbm": -130.0}, cohort_id="c1", observed_at=NOW
        )
        self.assertEqual(verdict.deviations[0].direction, -1)

    def test_an_unusable_baseline_abstains_rather_than_reporting_normal(self) -> None:
        # Reporting "not anomalous" without a baseline would be a fail-open.
        verdict = detector("n1", -100.0, count=5).judge(
            {"noise_floor_dbm": 0.0}, cohort_id="c1", observed_at=NOW
        )
        self.assertFalse(verdict.anomalous)
        self.assertIn(
            f"BASELINE_NEEDS_AT_LEAST_{MIN_BASELINE_SAMPLES}_SAMPLES_SO_ABSTAINING",
            verdict.reasons,
        )

    def test_a_feature_with_no_history_is_ignored(self) -> None:
        verdict = detector("n1", -100.0).judge(
            {"unheard_of": 9999.0}, cohort_id="c1", observed_at=NOW
        )
        self.assertFalse(verdict.anomalous)

    def test_a_constant_feature_reports_at_the_threshold_not_at_infinity(self) -> None:
        rows = [{"flag": 1.0, "noise_floor_dbm": -100.0 + (i % 3)} for i in range(40)]
        local = LocalDetector(NodeBaseline.from_observations("n", rows))
        verdict = local.judge({"flag": 5.0}, cohort_id="c1", observed_at=NOW)
        self.assertTrue(verdict.anomalous)
        self.assertEqual(verdict.deviations[0].sigmas, local.sigmas)

    def test_raw_values_never_leave_the_node(self) -> None:
        # Publishing readings would publish what each sensor sees, and so where
        # it is — the exposure privacy.py exists to limit.
        payload = json.dumps(
            detector("n1", -100.0)
            .judge({"noise_floor_dbm": -70.0}, cohort_id="c1", observed_at=NOW)
            .to_dict()
        )
        self.assertNotIn("-70.0", payload)
        self.assertNotIn("noise_floor_dbm\": -", payload)

    def test_verdict_serializes(self) -> None:
        json.dumps(
            detector("n1", -100.0)
            .judge({"noise_floor_dbm": -70.0}, cohort_id="c1", observed_at=NOW)
            .to_dict()
        )


def verdict(
    node: str,
    cohort: str,
    features: dict[str, int],
    *,
    minutes: float = 0.0,
    anomalous: bool = True,
) -> LocalVerdict:
    from kval_node.intrusion import FeatureDeviation

    return LocalVerdict(
        node_id=node,
        cohort_id=cohort,
        observed_at=NOW + timedelta(minutes=minutes),
        anomalous=anomalous,
        deviations=tuple(
            FeatureDeviation(name, 4.0, direction) for name, direction in sorted(features.items())
        ),
        reasons=(),
    )


class AgreementTests(unittest.TestCase):
    def test_the_same_features_moving_the_same_way_fully_agree(self) -> None:
        left = verdict("a", "c1", {"noise_floor_dbm": 1})
        right = verdict("b", "c2", {"noise_floor_dbm": 1})
        self.assertEqual(verdict_agreement(left, right), 1.0)

    def test_opposite_directions_do_not_agree(self) -> None:
        left = verdict("a", "c1", {"noise_floor_dbm": 1})
        right = verdict("b", "c2", {"noise_floor_dbm": -1})
        self.assertEqual(verdict_agreement(left, right), 0.0)

    def test_different_features_do_not_agree(self) -> None:
        left = verdict("a", "c1", {"noise_floor_dbm": 1})
        right = verdict("b", "c2", {"rtt_ms": 1})
        self.assertEqual(verdict_agreement(left, right), 0.0)

    def test_partial_overlap_scores_partially(self) -> None:
        left = verdict("a", "c1", {"noise_floor_dbm": 1, "rtt_ms": 1})
        right = verdict("b", "c2", {"noise_floor_dbm": 1})
        self.assertAlmostEqual(verdict_agreement(left, right), 0.5)


class FleetDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = FleetDetector()

    def test_a_correlated_change_across_cohorts_is_an_event(self) -> None:
        # The case only the fleet can see: three nodes in two established
        # cohorts, noise floor up together, within minutes.
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c1", {"noise_floor_dbm": 1}, minutes=1),
                verdict("c", "c2", {"noise_floor_dbm": 1}, minutes=2),
            ]
        )
        self.assertTrue(finding.event)
        self.assertEqual(finding.shared_features, ("noise_floor_dbm",))
        self.assertEqual(finding.cohorts, ("c1", "c2"))

    def test_the_finding_refuses_to_name_a_cause(self) -> None:
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c1", {"noise_floor_dbm": 1}),
                verdict("c", "c2", {"noise_floor_dbm": 1}),
            ]
        )
        self.assertIn("THIS_REPORTS_A_CORRELATED_CHANGE_NOT_A_CAUSE_OR_AN_ACTOR", finding.reasons)
        payload = json.dumps(finding.to_dict()).lower()
        for word in ("jam", "attack", "adversary", "enemy"):
            self.assertNotIn(word, payload)

    def test_one_anomalous_node_is_a_local_condition(self) -> None:
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c1", {}, anomalous=False),
                verdict("c", "c2", {}, anomalous=False),
            ]
        )
        self.assertFalse(finding.event)
        self.assertIn("TOO_FEW_NODES_ANOMALOUS_TO_RULE_OUT_LOCAL_CONDITIONS", finding.reasons)

    def test_an_anomaly_confined_to_one_cohort_is_explained_by_that_cohort(self) -> None:
        # A cohort is by definition a set of nodes with a shared environment,
        # so a shared local cause is the simpler explanation.
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c1", {"noise_floor_dbm": 1}),
                verdict("c", "c1", {"noise_floor_dbm": 1}),
            ]
        )
        self.assertFalse(finding.event)
        self.assertIn(
            "ANOMALY_CONFINED_TO_ONE_COHORT_SO_A_SHARED_LOCAL_CAUSE_EXPLAINS_IT", finding.reasons
        )

    def test_unrelated_faults_do_not_accumulate_into_an_event(self) -> None:
        # Three nodes anomalous for three different reasons is three faults,
        # not one event.
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c2", {"rtt_ms": 1}),
                verdict("c", "c3", {"loss_percent": -1}),
            ]
        )
        self.assertFalse(finding.event)
        self.assertIn("NODES_ARE_ANOMALOUS_FOR_UNRELATED_REASONS_NOT_ONE_CAUSE", finding.reasons)

    def test_opposite_directions_are_not_one_cause(self) -> None:
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c2", {"noise_floor_dbm": -1}),
                verdict("c", "c3", {"noise_floor_dbm": 1}),
            ]
        )
        self.assertFalse(finding.event)

    def test_anomalies_spread_over_hours_are_not_simultaneous(self) -> None:
        finding = self.detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}, minutes=0),
                verdict("b", "c2", {"noise_floor_dbm": 1}, minutes=120),
                verdict("c", "c3", {"noise_floor_dbm": 1}, minutes=240),
            ]
        )
        self.assertFalse(finding.event)
        self.assertIn("ANOMALIES_ARE_SPREAD_OVER_TIME_NOT_SIMULTANEOUS", finding.reasons)

    def test_a_burst_inside_a_longer_record_is_still_found(self) -> None:
        # The correlated group must be found even when surrounded by unrelated
        # anomalies at other times.
        finding = self.detector.assess(
            [
                verdict("old", "c1", {"noise_floor_dbm": 1}, minutes=-500),
                verdict("a", "c1", {"noise_floor_dbm": 1}, minutes=0),
                verdict("b", "c2", {"noise_floor_dbm": 1}, minutes=1),
                verdict("c", "c3", {"noise_floor_dbm": 1}, minutes=2),
                verdict("later", "c1", {"noise_floor_dbm": 1}, minutes=500),
            ]
        )
        self.assertTrue(finding.event)
        self.assertEqual(finding.corroborating_nodes, ("a", "b", "c"))

    def test_nodes_with_no_standing_cannot_carry_a_finding(self) -> None:
        # Reputation from cohorts.py: nodes that repeatedly diverged do not get
        # to raise a fleet-level alarm between them.
        detector = FleetDetector(reputations={"a": 0.1, "b": 0.1, "c": 0.1})
        finding = detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c2", {"noise_floor_dbm": 1}),
                verdict("c", "c3", {"noise_floor_dbm": 1}),
            ]
        )
        self.assertFalse(finding.event)
        self.assertIn("CORROBORATION_COMES_FROM_NODES_WITH_TOO_LITTLE_STANDING", finding.reasons)

    def test_reputable_nodes_still_raise_the_finding(self) -> None:
        detector = FleetDetector(reputations={"a": 1.0, "b": 1.0, "c": 1.0})
        finding = detector.assess(
            [
                verdict("a", "c1", {"noise_floor_dbm": 1}),
                verdict("b", "c2", {"noise_floor_dbm": 1}),
                verdict("c", "c3", {"noise_floor_dbm": 1}),
            ]
        )
        self.assertTrue(finding.event)

    def test_no_verdicts_is_not_an_event(self) -> None:
        self.assertFalse(self.detector.assess([]).event)

    def test_finding_serializes(self) -> None:
        json.dumps(
            self.detector.assess(
                [
                    verdict("a", "c1", {"noise_floor_dbm": 1}),
                    verdict("b", "c2", {"noise_floor_dbm": 1}),
                    verdict("c", "c3", {"noise_floor_dbm": 1}),
                ]
            ).to_dict()
        )


class FalseAlarmEstimateTests(unittest.TestCase):
    def test_a_higher_threshold_gives_fewer_chance_alarms(self) -> None:
        loose = expected_false_alarm_rate(sigmas=2.0, features=5, nodes=10, min_corroboration=3)
        tight = expected_false_alarm_rate(sigmas=4.0, features=5, nodes=10, min_corroboration=3)
        self.assertLess(tight, loose)

    def test_requiring_more_corroboration_gives_fewer_chance_alarms(self) -> None:
        self.assertLess(
            expected_false_alarm_rate(sigmas=3.0, features=5, nodes=10, min_corroboration=6),
            expected_false_alarm_rate(sigmas=3.0, features=5, nodes=10, min_corroboration=2),
        )

    def test_the_estimate_stays_a_probability(self) -> None:
        for sigmas in (0.5, 1.0, 3.0, 8.0):
            value = expected_false_alarm_rate(
                sigmas=sigmas, features=12, nodes=20, min_corroboration=3
            )
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_degenerate_inputs_return_certainty_not_a_crash(self) -> None:
        self.assertEqual(
            expected_false_alarm_rate(sigmas=0.0, features=5, nodes=5, min_corroboration=2), 1.0
        )


if __name__ == "__main__":
    unittest.main()
