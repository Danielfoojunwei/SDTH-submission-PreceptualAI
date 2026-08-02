from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from kval_node.power import (
    PowerGovernor,
    PowerPolicy,
    PowerReading,
    PowerReason,
    parse_tegrastats,
)

NOW = datetime.now(timezone.utc)


def policy() -> PowerPolicy:
    return PowerPolicy.from_dict(
        {
            "policy_id": "power-test-v1",
            "created_at": (NOW - timedelta(minutes=1)).isoformat(),
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "thermal_ceiling_c": 84.0,
            "thermal_resume_c": 72.0,
            "battery_floor_percent": 20.0,
            "max_sustained_watts": 15.0,
            "allowed_nvpmodel_ids": [0, 1, 2],
            "economy_nvpmodel_id": 1,
            "performance_nvpmodel_id": 0,
            "min_telemetry_interval_seconds": 1.0,
            "min_scan_interval_seconds": 1.0,
            "max_inference_duty": 1.0,
        },
        now=NOW,
    )


def reading(**overrides) -> PowerReading:
    base = dict(
        observed_at=NOW,
        source="test",
        cpu_temp_c=50.0,
        gpu_temp_c=48.0,
        module_watts=8.0,
        battery_percent=80.0,
        hardware_throttled=False,
        nvpmodel_id=0,
    )
    base.update(overrides)
    return PowerReading(**base)


class PowerPolicyTests(unittest.TestCase):
    def test_hysteresis_must_be_below_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "hysteresis"):
            PowerPolicy.from_dict(
                {
                    "policy_id": "bad",
                    "created_at": (NOW - timedelta(minutes=1)).isoformat(),
                    "expires_at": (NOW + timedelta(days=1)).isoformat(),
                    "thermal_ceiling_c": 70.0,
                    "thermal_resume_c": 80.0,
                    "max_sustained_watts": 15.0,
                    "allowed_nvpmodel_ids": [0, 1],
                    "economy_nvpmodel_id": 1,
                    "performance_nvpmodel_id": 0,
                },
                now=NOW,
            )

    def test_modes_must_be_in_allowlist(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowed_nvpmodel_ids"):
            PowerPolicy.from_dict(
                {
                    "policy_id": "bad",
                    "created_at": (NOW - timedelta(minutes=1)).isoformat(),
                    "expires_at": (NOW + timedelta(days=1)).isoformat(),
                    "thermal_ceiling_c": 84.0,
                    "thermal_resume_c": 72.0,
                    "max_sustained_watts": 15.0,
                    "allowed_nvpmodel_ids": [1, 2],
                    "economy_nvpmodel_id": 1,
                    "performance_nvpmodel_id": 0,
                },
                now=NOW,
            )


class TegrastatsParserTests(unittest.TestCase):
    def test_parses_temps_and_power(self) -> None:
        line = "RAM 1024/7620MB CPU@55.5C GPU@60C VDD_IN 8200mW/8200mW"
        result = parse_tegrastats(line, observed_at=NOW)
        self.assertEqual(result.cpu_temp_c, 55.5)
        self.assertEqual(result.gpu_temp_c, 60.0)
        self.assertEqual(result.module_watts, 8.2)
        self.assertTrue(result.has_thermal_evidence())

    def test_line_without_thermal_has_no_evidence(self) -> None:
        result = parse_tegrastats("RAM 1024/7620MB", observed_at=NOW)
        self.assertIsNone(result.hottest())
        self.assertFalse(result.has_thermal_evidence())


class PowerGovernorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.governor = PowerGovernor()
        self.policy = policy()

    def test_nominal_selects_performance(self) -> None:
        rec = self.governor.recommend(self.policy, reading(), current_nvpmodel_id=0)
        self.assertIs(rec.reason, PowerReason.NOMINAL)
        self.assertEqual(rec.target_nvpmodel_id, 0)

    def test_thermal_ceiling_forces_economy(self) -> None:
        rec = self.governor.recommend(self.policy, reading(cpu_temp_c=90.0), current_nvpmodel_id=0)
        self.assertIs(rec.reason, PowerReason.THERMAL_LIMIT)
        self.assertEqual(rec.target_nvpmodel_id, 1)
        self.assertLessEqual(rec.max_inference_duty, 0.25)

    def test_battery_floor_forces_economy(self) -> None:
        rec = self.governor.recommend(self.policy, reading(battery_percent=10.0), current_nvpmodel_id=0)
        self.assertIs(rec.reason, PowerReason.BATTERY_FLOOR)
        self.assertEqual(rec.target_nvpmodel_id, 1)

    def test_power_budget_exceeded(self) -> None:
        rec = self.governor.recommend(self.policy, reading(module_watts=20.0), current_nvpmodel_id=0)
        self.assertIs(rec.reason, PowerReason.POWER_BUDGET)

    def test_throttle_reported(self) -> None:
        rec = self.governor.recommend(self.policy, reading(hardware_throttled=True))
        self.assertIs(rec.reason, PowerReason.THROTTLED_BY_HARDWARE)

    def test_hysteresis_holds_economy_until_resume(self) -> None:
        # In economy at 78C (above resume 72C) -> hold economy.
        rec = self.governor.recommend(self.policy, reading(cpu_temp_c=78.0, gpu_temp_c=78.0), current_nvpmodel_id=1)
        self.assertIs(rec.reason, PowerReason.THERMAL_RECOVERING)
        self.assertEqual(rec.target_nvpmodel_id, 1)
        # Cooled to resume -> allowed back to performance.
        rec2 = self.governor.recommend(self.policy, reading(cpu_temp_c=70.0, gpu_temp_c=70.0), current_nvpmodel_id=1)
        self.assertIs(rec2.reason, PowerReason.NOMINAL)
        self.assertEqual(rec2.target_nvpmodel_id, 0)

    def test_no_evidence_is_insufficient(self) -> None:
        rec = self.governor.recommend(
            self.policy,
            reading(cpu_temp_c=None, gpu_temp_c=None, battery_percent=None, module_watts=None),
        )
        self.assertIs(rec.reason, PowerReason.INSUFFICIENT_EVIDENCE)
        self.assertEqual(rec.target_nvpmodel_id, 1)


if __name__ == "__main__":
    unittest.main()
