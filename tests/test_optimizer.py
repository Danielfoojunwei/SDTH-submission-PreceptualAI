from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from kval_node.models import (
    AccessType,
    Catalog,
    LinkObservation,
    Profile,
    ServiceClass,
    TelemetrySnapshot,
)
from kval_node.optimizer import ProfileOptimizer
from kval_node.standards import StandardsRequirement


NOW = datetime.now(timezone.utc)


def make_profile(profile_id: str, connection_uuid: str, interface: str, cost: float = 0.1) -> Profile:
    return Profile(
        profile_id=profile_id,
        access_type=AccessType.WIFI,
        connection_uuid=connection_uuid,
        interface=interface,
        probe_target="127.0.0.1",
        monetary_cost=cost,
        energy_cost=cost,
        minimum_dwell_seconds=30,
        maximum_switches_per_hour=8,
        allowed_service_classes=frozenset(ServiceClass),
        standards=StandardsRequirement.from_dict({}),
    )


def make_observation(profile_id: str, interface: str, rtt_ms: float, loss: float, throughput: float) -> LinkObservation:
    return LinkObservation(
        profile_id=profile_id,
        interface=interface,
        observed_at=NOW,
        interface_present=True,
        link_up=True,
        connection_active=True,
        rtt_ms=rtt_ms,
        loss_percent=loss,
        jitter_ms=2.0,
        rx_mbps=throughput,
        tx_mbps=1.0,
        signal_percent=90.0,
        access_technology=None,
        standards_verified=True,
        standards_evidence=("NO_ADDITIONAL_STANDARDS_CONSTRAINT",),
        standards_errors=(),
        evidence=("ip-link", "networkmanager", "bound-icmp-probe"),
    )


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fast = make_profile("fast", "11111111-1111-4111-8111-111111111111", "wlan0")
        self.slow = make_profile("slow", "22222222-2222-4222-8222-222222222222", "wlan1")
        self.catalog = Catalog("v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1), (self.fast, self.slow), "a" * 64, "key")

    def test_selects_best_complete_live_evidence(self) -> None:
        snapshot = TelemetrySnapshot(
            NOW,
            "slow",
            (
                make_observation("fast", "wlan0", 15.0, 0.0, 10.0),
                make_observation("slow", "wlan1", 150.0, 10.0, 1.0),
            ),
        )
        result = ProfileOptimizer().recommend(self.catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertEqual(result.selected_profile_id, "fast")
        self.assertGreater(result.confidence, 0.99)

    def test_rejects_unverified_standards_evidence(self) -> None:
        failed = make_observation("fast", "wlan0", 10.0, 0.0, 10.0)
        failed = LinkObservation(**{**failed.__dict__, "standards_verified": False, "standards_errors": ("missing attestation",)})
        snapshot = TelemetrySnapshot(NOW, None, (failed,))
        one_profile_catalog = Catalog("v1", self.catalog.created_at, self.catalog.expires_at, (self.fast,), "b" * 64, "key")
        result = ProfileOptimizer().recommend(one_profile_catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertIsNone(result.selected_profile_id)
        self.assertIn("STANDARDS_REQUIREMENTS_NOT_VERIFIED", result.scores[0].reasons)

    def test_abstains_without_probe_evidence(self) -> None:
        incomplete = make_observation("fast", "wlan0", 10.0, 0.0, 10.0)
        incomplete = LinkObservation(**{**incomplete.__dict__, "rtt_ms": None})
        snapshot = TelemetrySnapshot(NOW, None, (incomplete,))
        one_profile_catalog = Catalog("v1", self.catalog.created_at, self.catalog.expires_at, (self.fast,), "b" * 64, "key")
        result = ProfileOptimizer().recommend(one_profile_catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertIsNone(result.selected_profile_id)


class RankingSurvivesPastTheLimitTests(unittest.TestCase):
    """Ranking when every link has already failed — the jammed case.

    Scoring clamped to zero at the limit, so all failing links scored the same
    and selection fell through to catalogue order: the same two links, swapped
    in the file, produced a different choice, reported as
    ``HIGHEST_POLICY_CONSTRAINED_UTILITY`` as though a comparison had made it.
    A node built for contested spectrum lost its ranking exactly when contested
    spectrum arrived.
    """

    def setUp(self) -> None:
        self.a = make_profile("link_a", "11111111-1111-4111-8111-111111111111", "wlan0")
        self.b = make_profile("link_b", "22222222-2222-4222-8222-222222222222", "wlan1")
        self.catalog = Catalog(
            "v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1), (self.a, self.b), "a" * 64, "key"
        )

    def _obs(self, profile_id: str, interface: str, rtt: float, loss: float, jitter: float) -> LinkObservation:
        base = make_observation(profile_id, interface, rtt, loss, 10.0)
        return LinkObservation(**{**base.__dict__, "jitter_ms": jitter})

    def _pick(self, a: tuple[float, float, float], b: tuple[float, float, float]):
        snapshot = TelemetrySnapshot(
            NOW, None, (self._obs("link_a", "wlan0", *a), self._obs("link_b", "wlan1", *b))
        )
        return ProfileOptimizer().recommend(self.catalog, snapshot, ServiceClass.CRITICAL_DATA)

    BAD = (300.0, 25.0, 200.0)
    WORSE = (6000.0, 95.0, 4000.0)

    def test_the_least_bad_link_is_chosen_when_all_of_them_have_failed(self) -> None:
        result = self._pick(self.BAD, self.WORSE)
        self.assertEqual(result.selected_profile_id, "link_a")

    def test_the_choice_does_not_depend_on_catalogue_order(self) -> None:
        # The same pair with the roles swapped between the entries. Before, both
        # arrangements selected link_a because the utilities were identical.
        self.assertEqual(self._pick(self.BAD, self.WORSE).selected_profile_id, "link_a")
        self.assertEqual(self._pick(self.WORSE, self.BAD).selected_profile_id, "link_b")

    def test_a_failing_link_never_outranks_an_acceptable_one(self) -> None:
        # The boundary is what the saturation floor must not disturb: readings
        # either side of the limit have to stay on their own sides.
        result = self._pick((149.0, 0.9, 29.0), (151.0, 1.1, 31.0))
        scores = {score.profile_id: score.utility for score in result.scores}
        self.assertGreater(scores["link_a"], scores["link_b"])
        self.assertEqual(result.selected_profile_id, "link_a")

    def test_worse_readings_always_score_lower_never_equal(self) -> None:
        deep = self._pick((3000.0, 80.0, 2000.0), (9000.0, 95.0, 6000.0))
        scores = {score.profile_id: score.utility for score in deep.scores}
        self.assertNotEqual(scores["link_a"], scores["link_b"])
        self.assertGreater(scores["link_a"], scores["link_b"])

    def test_a_real_tie_is_declared_rather_than_dressed_up_as_a_ranking(self) -> None:
        # Identical readings genuinely tie. Reporting that as "highest utility"
        # is the same class of untrue statement as naming a corroborator that
        # measured the band clear.
        result = self._pick(self.BAD, self.BAD)
        self.assertIn("TOP_PROFILES_TIED_ON_UTILITY_SO_THE_EVIDENCE_DID_NOT_CHOOSE", result.reason_codes)
        self.assertNotIn("HIGHEST_POLICY_CONSTRAINED_UTILITY", result.reason_codes)

    def test_a_clear_winner_is_still_reported_as_a_ranking(self) -> None:
        result = self._pick((10.0, 0.0, 1.0), self.WORSE)
        self.assertIn("HIGHEST_POLICY_CONSTRAINED_UTILITY", result.reason_codes)
        self.assertNotIn("TOP_PROFILES_TIED_ON_UTILITY_SO_THE_EVIDENCE_DID_NOT_CHOOSE", result.reason_codes)


class PreviouslyUnexercisedRankingBranchesTests(unittest.TestCase):
    """Three control paths no test had ever executed.

    Found by tracing the suite and diffing executed lines against production
    statements. All three turned out to be correct — but the optimizer's
    emission-control branch, found the same way, crashed the moment it was
    forced, so "never executed" is not a state worth leaving them in.
    """

    def setUp(self) -> None:
        self.fast = make_profile("fast", "11111111-1111-4111-8111-111111111111", "wlan0")
        self.slow = make_profile("slow", "22222222-2222-4222-8222-222222222222", "wlan1")
        self.catalog = Catalog(
            "v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1), (self.fast, self.slow), "a" * 64, "key"
        )
        self.observations = (
            make_observation("fast", "wlan0", 15.0, 0.0, 10.0),
            make_observation("slow", "wlan1", 150.0, 10.0, 1.0),
        )

    def test_a_forecast_of_degradation_penalises_the_link(self) -> None:
        # The Part 3 forecast-to-optimizer composition, never once executed.
        snapshot = TelemetrySnapshot(NOW, "slow", self.observations)
        clean = ProfileOptimizer().recommend(self.catalog, snapshot, ServiceClass.CRITICAL_DATA)
        warned = ProfileOptimizer().recommend(
            self.catalog, snapshot, ServiceClass.CRITICAL_DATA,
            forecast_degrading_profiles=frozenset({"fast"}),
        )
        before = next(s for s in clean.scores if s.profile_id == "fast")
        after = next(s for s in warned.scores if s.profile_id == "fast")
        self.assertLess(after.utility, before.utility)
        self.assertIn("FORECAST_EXPECTS_THIS_LINK_TO_DEGRADE", after.reasons)
        self.assertIn("forecast_penalty", after.components)

    def test_a_profile_that_forbids_the_service_class_is_rejected(self) -> None:
        narrow = Profile(**{**self.fast.__dict__, "allowed_service_classes": frozenset({ServiceClass.BULK_DATA})})
        catalog = Catalog("v1", self.catalog.created_at, self.catalog.expires_at, (narrow, self.slow), "b" * 64, "key")
        result = ProfileOptimizer().recommend(
            catalog, TelemetrySnapshot(NOW, "slow", self.observations), ServiceClass.CRITICAL_DATA
        )
        rejected = next(s for s in result.scores if s.profile_id == "fast")
        self.assertEqual(rejected.utility, -1.0)
        self.assertIn("SERVICE_CLASS_NOT_ALLOWED", rejected.reasons)
        self.assertNotEqual(result.selected_profile_id, "fast")

    def test_a_marginal_gain_does_not_move_the_node_off_its_current_link(self) -> None:
        # Anti-flapping. A link selector without working hysteresis oscillates,
        # and every switch costs a dwell period.
        snapshot = TelemetrySnapshot(
            NOW, "slow",
            (make_observation("fast", "wlan0", 15.0, 0.0, 10.0),
             make_observation("slow", "wlan1", 90.0, 0.0, 10.0)),
        )
        result = ProfileOptimizer().recommend(self.catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertEqual(result.selected_profile_id, "slow")
        self.assertIn("CURRENT_PROFILE_RETAINED_BY_HYSTERESIS", result.reason_codes)

    def test_a_decisive_gain_still_moves_the_node(self) -> None:
        # Hysteresis must not become a trap: the same node on a badly degraded
        # link has to be able to leave it.
        snapshot = TelemetrySnapshot(
            NOW, "slow",
            (make_observation("fast", "wlan0", 15.0, 0.0, 10.0),
             make_observation("slow", "wlan1", 500.0, 0.0, 10.0)),
        )
        result = ProfileOptimizer().recommend(self.catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertEqual(result.selected_profile_id, "fast")


if __name__ == "__main__":
    unittest.main()

