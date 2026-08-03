from __future__ import annotations

import copy
import statistics
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.dataset import FEATURE_NAMES
from kval_node.roster import PeerRoster, RosterError
from kval_node.federated import (
    FeatureStatistics,
    FederatedUpdate,
    RobustAggregator,
    sign_update,
    trimmed_mean,
    verify_update,
)

NOW = datetime.now(timezone.utc)


def pem(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")


def roster(keys: dict[str, Ed25519PrivateKey]) -> PeerRoster:
    return PeerRoster.from_dict(
        {
            "roster_id": "r",
            "created_at": (NOW - timedelta(minutes=1)).isoformat(),
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "peers": [{"node_id": n, "public_key_pem": pem(k)} for n, k in keys.items()],
        },
        now=NOW,
    )


def update(node: str, rtt_mean: float, *, count: int = 50, variance: float = 100.0) -> FederatedUpdate:
    stats = {name: FeatureStatistics(count, 0.0, float(count)) for name in FEATURE_NAMES}
    stats["rtt_ms"] = FeatureStatistics(count, rtt_mean, count * (rtt_mean**2 + variance))
    return FederatedUpdate(node, NOW, count, stats)


class SufficientStatisticsTests(unittest.TestCase):
    def test_stddev_recovers_from_count_mean_and_sum_squares(self) -> None:
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        stat = FeatureStatistics(
            count=len(values),
            mean=statistics.fmean(values),
            sum_squares=sum(v**2 for v in values),
        )
        self.assertAlmostEqual(stat.stddev(), statistics.pstdev(values), places=9)

    def test_single_sample_has_no_spread(self) -> None:
        self.assertEqual(FeatureStatistics(1, 5.0, 25.0).stddev(), 0.0)

    def test_update_carries_no_raw_observations(self) -> None:
        body = update("n", 50.0).to_body()
        for stat in body["statistics"].values():
            self.assertEqual(sorted(stat), ["count", "mean", "sum_squares"])


class UpdateVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.roster = roster({"node-a": self.key})

    def test_round_trip(self) -> None:
        envelope = sign_update(update("node-a", 50.0), self.key)
        self.assertEqual(verify_update(envelope, self.roster, now=NOW).node_id, "node-a")

    def test_unrostered_author_refused(self) -> None:
        stranger = Ed25519PrivateKey.generate()
        envelope = sign_update(update("node-z", 50.0), stranger)
        with self.assertRaisesRegex(RosterError, "not on the signed roster"):
            verify_update(envelope, self.roster, now=NOW)

    def test_impersonation_refused(self) -> None:
        stranger = Ed25519PrivateKey.generate()
        envelope = sign_update(update("node-a", 50.0), stranger)
        with self.assertRaisesRegex(RosterError, "key ID does not match"):
            verify_update(envelope, self.roster, now=NOW)

    def test_tampered_statistics_refused(self) -> None:
        envelope = sign_update(update("node-a", 50.0), self.key)
        altered = copy.deepcopy(envelope)
        altered["body"]["statistics"]["rtt_ms"]["mean"] = 999.0
        with self.assertRaisesRegex(RosterError, "signature verification failed"):
            verify_update(altered, self.roster, now=NOW)

    def test_stale_update_refused(self) -> None:
        old = FederatedUpdate("node-a", NOW - timedelta(days=30), 50, update("node-a", 50.0).statistics)
        with self.assertRaisesRegex(RosterError, "older than the aggregation window"):
            verify_update(sign_update(old, self.key), self.roster, now=NOW)


class TrimmedMeanTests(unittest.TestCase):
    def test_extremes_are_discarded(self) -> None:
        values = [10.0, 10.0, 10.0, 10.0, 1_000_000.0]
        self.assertLess(trimmed_mean(values, trim_fraction=0.25), 100.0)

    def test_zero_trim_is_a_plain_mean(self) -> None:
        values = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(trimmed_mean(values, trim_fraction=0.0), 2.0)


class RobustAggregationTests(unittest.TestCase):
    def test_honest_fleet_aggregates(self) -> None:
        updates = [update(f"n{i}", 50.0 + i) for i in range(6)]
        result = RobustAggregator().aggregate(updates)
        self.assertIsNotNone(result.baseline)
        self.assertAlmostEqual(result.baseline.means["rtt_ms"], 52.5, places=1)
        self.assertGreater(result.baseline.stddevs["rtt_ms"], 0.0)
        self.assertIn("FLEET_BASELINE_AGGREGATED", result.reasons)

    def test_a_poisoner_cannot_move_the_baseline(self) -> None:
        honest = [update(f"n{i}", 50.0 + i) for i in range(6)]
        poisoned = honest + [update("attacker", 999_999.0)]
        result = RobustAggregator().aggregate(poisoned)
        # A plain average would be dragged into the hundreds of thousands.
        naive = statistics.fmean([u.statistics["rtt_ms"].mean for u in poisoned])
        self.assertGreater(naive, 100_000.0)
        self.assertLess(result.baseline.means["rtt_ms"], 60.0)

    def test_the_poisoner_is_named_not_just_ignored(self) -> None:
        honest = [update(f"n{i}", 50.0 + i) for i in range(6)]
        result = RobustAggregator().aggregate(honest + [update("attacker", 999_999.0)])
        self.assertEqual([q["node_id"] for q in result.quarantined], ["attacker"])
        self.assertIn("rtt_ms", result.quarantined[0]["outlying_features"])
        self.assertNotIn("attacker", result.contributors)
        self.assertIn("OUTLYING_CONTRIBUTORS_QUARANTINED", result.reasons)

    def test_one_node_cannot_form_a_fleet(self) -> None:
        result = RobustAggregator().aggregate([update("solo", 50.0)])
        self.assertIsNone(result.baseline)
        self.assertIn("NEED_AT_LEAST_3_CONTRIBUTORS", result.reasons)

    def test_a_fleet_of_identical_nodes_has_no_envelope(self) -> None:
        # Every contributor identical and internally constant: nothing varies,
        # so there is no envelope to build rather than a zero-width one.
        updates = [
            FederatedUpdate(
                f"n{i}", NOW, 50, {name: FeatureStatistics(50, 1.0, 50.0) for name in FEATURE_NAMES}
            )
            for i in range(4)
        ]
        result = RobustAggregator().aggregate(updates)
        self.assertIsNone(result.baseline)
        self.assertIn("NO_FEATURE_VARIED_ACROSS_THE_FLEET", result.reasons)

    def test_quorum_is_enforced_after_quarantine(self) -> None:
        # Three honest, three wild: too few survive to form a fleet.
        updates = [update(f"n{i}", 50.0 + i) for i in range(3)] + [
            update(f"bad{i}", 10.0**9 * (i + 1)) for i in range(3)
        ]
        result = RobustAggregator(quorum=5).aggregate(updates)
        self.assertIsNone(result.baseline)

    def test_result_serializes(self) -> None:
        import json

        result = RobustAggregator().aggregate([update(f"n{i}", 50.0 + i) for i in range(4)])
        json.dumps(result.to_dict())


if __name__ == "__main__":
    unittest.main()
