from __future__ import annotations

import unittest

from kval_node.timesync import (
    ClockMonitor,
    ClockSource,
    parse_chronyc_tracking,
    parse_timedatectl,
)

CHRONY_TRACKING = """Reference ID    : C0A80001 (ntp.example.net)
Stratum         : 3
Ref time (UTC)  : Sat Aug 02 02:30:00 2026
System time     : 0.000012345 seconds fast of NTP time
Last offset     : -0.000031234 seconds
RMS offset      : 0.000045678 seconds
Frequency       : 12.345 ppm slow
Residual freq   : +0.001 ppm
Skew            : 0.123 ppm
Root delay      : 0.012345678 seconds
Root dispersion : 0.001234567 seconds
Update interval : 64.2 seconds
Leap status     : Normal
"""

CHRONY_UNSYNCED = CHRONY_TRACKING.replace("Leap status     : Normal", "Leap status     : Not synchronised")
CHRONY_FAR_OFF = CHRONY_TRACKING.replace("Last offset     : -0.000031234 seconds", "Last offset     : 4.250000000 seconds")

TIMEDATECTL_YES = """               Local time: Sat 2026-08-02 02:30:00 UTC
           Universal time: Sat 2026-08-02 02:30:00 UTC
                 RTC time: Sat 2026-08-02 02:30:00
                Time zone: Etc/UTC (UTC, +0000)
System clock synchronized: yes
              NTP service: active
"""
TIMEDATECTL_NO = TIMEDATECTL_YES.replace("synchronized: yes", "synchronized: no")


class ChronyParserTests(unittest.TestCase):
    def test_parses_offset_dispersion_and_leap(self) -> None:
        offset, dispersion, leap_normal = parse_chronyc_tracking(CHRONY_TRACKING)
        self.assertAlmostEqual(offset, 0.000031234)
        self.assertAlmostEqual(dispersion, 0.001234567)
        self.assertTrue(leap_normal)

    def test_offset_is_absolute(self) -> None:
        offset, _dispersion, _leap = parse_chronyc_tracking(CHRONY_FAR_OFF)
        self.assertAlmostEqual(offset, 4.25)

    def test_non_normal_leap_status_detected(self) -> None:
        _offset, _dispersion, leap_normal = parse_chronyc_tracking(CHRONY_UNSYNCED)
        self.assertFalse(leap_normal)

    def test_unparseable_output_yields_nothing(self) -> None:
        offset, dispersion, leap_normal = parse_chronyc_tracking("nonsense")
        self.assertIsNone(offset)
        self.assertIsNone(dispersion)
        self.assertFalse(leap_normal)


class TimedatectlParserTests(unittest.TestCase):
    def test_parses_yes_and_no(self) -> None:
        self.assertTrue(parse_timedatectl(TIMEDATECTL_YES))
        self.assertFalse(parse_timedatectl(TIMEDATECTL_NO))

    def test_absent_field_is_unknown(self) -> None:
        self.assertIsNone(parse_timedatectl("Local time: whenever"))


class ClockMonitorTests(unittest.TestCase):
    def test_step_starts_near_zero(self) -> None:
        monitor = ClockMonitor()
        self.assertLess(monitor.observed_step_seconds(), 1.0)

    def test_stepped_wall_clock_is_detected(self) -> None:
        monitor = ClockMonitor()
        # Simulate the wall clock having been stepped backwards an hour by
        # moving the recorded baseline forward.
        monitor._wall_at_start += 3600.0
        self.assertGreater(monitor.observed_step_seconds(), 3599.0)

    def test_no_time_daemon_is_not_trusted(self) -> None:
        # This host has neither chronyc nor timedatectl, so the check must fail
        # closed rather than assuming the clock is fine.
        verdict = ClockMonitor().check()
        self.assertFalse(verdict.trusted)
        self.assertIs(verdict.source, ClockSource.UNKNOWN)
        self.assertIn("NO_TIME_DAEMON_EVIDENCE", verdict.reasons)

    def test_verdict_serializes(self) -> None:
        import json

        json.dumps(ClockMonitor().check().to_dict())


if __name__ == "__main__":
    unittest.main()
