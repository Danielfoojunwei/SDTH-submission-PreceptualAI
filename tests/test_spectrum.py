from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kval_node.spectrum import (
    BandStatus,
    ScanRateExceeded,
    ScanRateLimiter,
    SpectrumAnalyzer,
    SpectrumObservation,
    SpectrumPolicy,
)

NOW = datetime.now(timezone.utc)


def policy_dict() -> dict:
    return {
        "policy_id": "spectrum-test-v1",
        "region": "TEST",
        "created_at": (NOW - timedelta(minutes=1)).isoformat(),
        "expires_at": (NOW + timedelta(days=1)).isoformat(),
        "passive_only": True,
        "bands": [
            {
                "band_id": "wifi-2g4",
                "label": "ISM 2.4",
                "freq_low_hz": 2400000000,
                "freq_high_hz": 2483500000,
                "max_resolution_bandwidth_hz": 1000000,
                "occupancy_baseline_percent": 20.0,
                "occupancy_alert_percent": 75.0,
                "noise_floor_baseline_dbm": -95.0,
                "comms_candidate": True,
            }
        ],
    }


def observation(**overrides) -> SpectrumObservation:
    base = {
        "band_id": "wifi-2g4",
        "observed_at": NOW.isoformat(),
        "source_adapter_id": "sensor-01",
        "conformance_hash": "a" * 64,
        "passive": True,
        "occupancy_percent": 10.0,
        "noise_floor_dbm": -95.0,
        "detected_emitters": 0,
        "quality": 1.0,
    }
    base.update(overrides)
    return SpectrumObservation.from_dict(base)


class SpectrumPolicyTests(unittest.TestCase):
    def test_valid_policy_parses(self) -> None:
        policy = SpectrumPolicy.from_dict(policy_dict(), now=NOW)
        self.assertEqual(policy.approved_band_ids(), frozenset({"wifi-2g4"}))
        self.assertTrue(policy.passive_only)

    def test_passive_only_cannot_be_disabled(self) -> None:
        raw = policy_dict()
        raw["passive_only"] = False
        with self.assertRaisesRegex(ValueError, "receive-only"):
            SpectrumPolicy.from_dict(raw, now=NOW)

    def test_passive_only_must_be_exactly_true(self) -> None:
        for value in ("true", 1, None):
            with self.subTest(value=value):
                raw = policy_dict()
                raw["passive_only"] = value
                with self.assertRaisesRegex(ValueError, "receive-only"):
                    SpectrumPolicy.from_dict(raw, now=NOW)

    def test_expired_policy_is_rejected(self) -> None:
        raw = policy_dict()
        raw["created_at"] = (NOW - timedelta(days=2)).isoformat()
        raw["expires_at"] = (NOW - timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "expired"):
            SpectrumPolicy.from_dict(raw, now=NOW)


class SpectrumObservationTests(unittest.TestCase):
    def test_non_passive_observation_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "passive"):
            observation(passive=False)

    def test_truthy_non_true_passive_value_is_refused(self) -> None:
        for value in ("yes", 1, [1]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "passive"):
                    observation(passive=value)

    def test_occupancy_bounds_enforced(self) -> None:
        with self.assertRaises(ValueError):
            observation(occupancy_percent=150.0)


class SpectrumAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SpectrumPolicy.from_dict(policy_dict(), now=NOW)
        self.analyzer = SpectrumAnalyzer()

    def test_clear_band_is_comms_candidate(self) -> None:
        intel = self.analyzer.analyze(self.policy, [observation(occupancy_percent=5.0)], now=NOW)
        self.assertEqual(intel.bands[0].status, BandStatus.CLEAR)
        self.assertIn("wifi-2g4", intel.clear_comms_band_ids)
        self.assertFalse(intel.jamming_suspected)

    def test_high_occupancy_with_noise_is_jamming_suspected(self) -> None:
        intel = self.analyzer.analyze(
            self.policy,
            [observation(occupancy_percent=95.0, noise_floor_dbm=-80.0)],
            now=NOW,
        )
        self.assertEqual(intel.bands[0].status, BandStatus.JAMMING_SUSPECTED)
        self.assertTrue(intel.jamming_suspected)
        self.assertEqual(intel.clear_comms_band_ids, ())

    def test_congested_without_noise_is_not_jamming(self) -> None:
        intel = self.analyzer.analyze(
            self.policy,
            [observation(occupancy_percent=90.0, noise_floor_dbm=-95.0)],
            now=NOW,
        )
        self.assertEqual(intel.bands[0].status, BandStatus.CONGESTED)
        self.assertFalse(intel.jamming_suspected)

    def test_emitter_presence_is_reported(self) -> None:
        intel = self.analyzer.analyze(
            self.policy,
            [observation(occupancy_percent=30.0, detected_emitters=2, bearing_deg=120.0)],
            now=NOW,
        )
        self.assertTrue(intel.presence_detected)
        self.assertTrue(intel.bands[0].emitter_present)
        self.assertEqual(intel.bands[0].bearing_deg, 120.0)

    def test_missing_observation_is_insufficient_evidence(self) -> None:
        intel = self.analyzer.analyze(self.policy, [], now=NOW)
        self.assertEqual(intel.bands[0].status, BandStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(intel.bands[0].confidence, 0.0)

    def test_stale_observation_is_not_reported_clear(self) -> None:
        old = observation(observed_at=(NOW - timedelta(days=10)).isoformat(), occupancy_percent=0.0)
        intel = self.analyzer.analyze(self.policy, [old], now=NOW)
        self.assertEqual(intel.bands[0].status, BandStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(intel.clear_comms_band_ids, ())
        self.assertIn("STALE_OR_FUTURE_OBSERVATIONS_DISCARDED", intel.reasons)

    def test_future_dated_observation_cannot_mask_live_jamming(self) -> None:
        live_jamming = observation(occupancy_percent=95.0, noise_floor_dbm=-80.0)
        forged_future = observation(
            observed_at=(NOW + timedelta(hours=1)).isoformat(),
            occupancy_percent=1.0,
            noise_floor_dbm=-95.0,
        )
        intel = self.analyzer.analyze(self.policy, [live_jamming, forged_future], now=NOW)
        self.assertEqual(intel.bands[0].status, BandStatus.JAMMING_SUSPECTED)
        self.assertTrue(intel.jamming_suspected)

    def test_low_quality_observation_is_insufficient(self) -> None:
        poor = observation(occupancy_percent=0.0, quality=0.05)
        intel = self.analyzer.analyze(self.policy, [poor], now=NOW)
        self.assertEqual(intel.bands[0].status, BandStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(intel.clear_comms_band_ids, ())

    def test_observation_outside_policy_is_rejected(self) -> None:
        rogue = observation(band_id="wifi-2g4")
        # Force a band id not in the policy by rebuilding with an approved id
        # then analyzing a different policy where it is unknown.
        other_policy = SpectrumPolicy.from_dict(
            {**policy_dict(), "bands": [{**policy_dict()["bands"][0], "band_id": "other"}]},
            now=NOW,
        )
        intel = self.analyzer.analyze(other_policy, [rogue], now=NOW)
        self.assertIn("OBSERVATIONS_OUTSIDE_POLICY_REJECTED", intel.reasons)
        self.assertEqual(intel.bands[0].status, BandStatus.INSUFFICIENT_EVIDENCE)


class ScanRateLimiterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "scans.db"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _policy(self, **overrides) -> SpectrumPolicy:
        raw = policy_dict()
        raw.update(overrides)
        return SpectrumPolicy.from_dict(raw, now=NOW)

    def test_minimum_interval_is_enforced(self) -> None:
        policy = self._policy(min_scan_interval_seconds=5.0)
        limiter = ScanRateLimiter(self.path)
        try:
            limiter.check_and_record(policy, now=NOW)
            with self.assertRaisesRegex(ScanRateExceeded, "below the policy minimum"):
                limiter.check_and_record(policy, now=NOW + timedelta(seconds=1))
            # Once the interval has elapsed the scan is permitted again.
            limiter.check_and_record(policy, now=NOW + timedelta(seconds=6))
        finally:
            limiter.close()

    def test_hourly_cap_is_enforced(self) -> None:
        policy = self._policy(max_scans_per_hour=3, min_scan_interval_seconds=0.05)
        limiter = ScanRateLimiter(self.path)
        try:
            for index in range(3):
                limiter.check_and_record(policy, now=NOW + timedelta(seconds=index))
            with self.assertRaisesRegex(ScanRateExceeded, "reaches the policy limit"):
                limiter.check_and_record(policy, now=NOW + timedelta(seconds=4))
        finally:
            limiter.close()

    def test_limits_survive_a_restart(self) -> None:
        # Duty limits are emission controls; a per-process counter would reset
        # on every CLI invocation, so the state must be durable.
        policy = self._policy(min_scan_interval_seconds=5.0)
        first = ScanRateLimiter(self.path)
        try:
            first.check_and_record(policy, now=NOW)
        finally:
            first.close()
        second = ScanRateLimiter(self.path)
        try:
            with self.assertRaises(ScanRateExceeded):
                second.check_and_record(policy, now=NOW + timedelta(seconds=1))
        finally:
            second.close()

    def test_old_scans_leave_the_hourly_window(self) -> None:
        policy = self._policy(max_scans_per_hour=2, min_scan_interval_seconds=0.05)
        limiter = ScanRateLimiter(self.path)
        try:
            limiter.check_and_record(policy, now=NOW)
            limiter.check_and_record(policy, now=NOW + timedelta(seconds=1))
            with self.assertRaises(ScanRateExceeded):
                limiter.check_and_record(policy, now=NOW + timedelta(seconds=2))
            limiter.check_and_record(policy, now=NOW + timedelta(hours=1, seconds=5))
        finally:
            limiter.close()


if __name__ == "__main__":
    unittest.main()
