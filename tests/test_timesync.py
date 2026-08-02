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


def _monitor(*, chrony=None, ntp=None, **kwargs) -> ClockMonitor:
    """A real ``ClockMonitor`` fed known daemon responses.

    The verdict must depend on what the time daemon reports, not on whatever
    the machine running the tests happens to have installed — a CI runner with
    a healthy NTP client and a container with none must reach the same verdict
    from the same inputs. The two daemon queries are constructor arguments, so
    every line of decision logic exercised here is the real implementation and
    nothing is overridden.
    """
    return ClockMonitor(
        chrony_source=lambda: chrony,
        ntp_sync_source=lambda: ntp,
        **kwargs,
    )


class ClockMonitorTests(unittest.TestCase):
    def test_step_starts_near_zero(self) -> None:
        monitor = ClockMonitor()
        self.assertLess(monitor.observed_step_seconds(), 1.0)

    def test_stepped_wall_clock_is_detected(self) -> None:
        monitor = ClockMonitor()
        # Simulate the wall clock having been stepped by moving the baseline.
        monitor._wall_at_start += 3600.0
        self.assertGreater(monitor.observed_step_seconds(), 3599.0)

    def test_no_time_daemon_fails_closed(self) -> None:
        verdict = _monitor(chrony=None, ntp=None).check()
        self.assertFalse(verdict.trusted)
        self.assertIs(verdict.source, ClockSource.UNKNOWN)
        self.assertIn("NO_TIME_DAEMON_EVIDENCE", verdict.reasons)

    def test_healthy_chrony_is_trusted(self) -> None:
        verdict = _monitor(chrony=(0.0001, 0.0005, True)).check()
        self.assertTrue(verdict.trusted)
        self.assertIs(verdict.source, ClockSource.CHRONY)
        self.assertIn("CLOCK_DISCIPLINED", verdict.reasons)

    def test_chrony_offset_above_limit_is_not_trusted(self) -> None:
        verdict = _monitor(chrony=(5.0, 0.0005, True)).check()
        self.assertFalse(verdict.trusted)
        self.assertIn("CLOCK_OFFSET_ABOVE_LIMIT", verdict.reasons)

    def test_chrony_root_dispersion_above_limit_is_not_trusted(self) -> None:
        verdict = _monitor(chrony=(0.0001, 9.0, True)).check()
        self.assertFalse(verdict.trusted)
        self.assertIn("ROOT_DISPERSION_ABOVE_LIMIT", verdict.reasons)

    def test_chrony_not_synchronised_is_not_trusted(self) -> None:
        verdict = _monitor(chrony=(0.0001, 0.0005, False)).check()
        self.assertFalse(verdict.trusted)
        self.assertIn("CHRONY_LEAP_STATUS_NOT_NORMAL", verdict.reasons)

    def test_timesyncd_synchronised_is_trusted(self) -> None:
        verdict = _monitor(chrony=None, ntp=True).check()
        self.assertTrue(verdict.trusted)
        self.assertIs(verdict.source, ClockSource.SYSTEMD_TIMESYNCD)

    def test_timesyncd_unsynchronised_is_not_trusted(self) -> None:
        verdict = _monitor(chrony=None, ntp=False).check()
        self.assertFalse(verdict.trusted)
        self.assertIn("SYSTEM_CLOCK_NOT_SYNCHRONISED", verdict.reasons)

    def test_a_stepped_clock_overrides_a_healthy_daemon(self) -> None:
        # A daemon claiming everything is fine does not make a stepped wall
        # clock safe to correlate against a peer.
        monitor = _monitor(chrony=(0.0001, 0.0005, True))
        monitor._wall_at_start += 3600.0
        verdict = monitor.check()
        self.assertFalse(verdict.trusted)
        self.assertIn("WALL_CLOCK_STEPPED", verdict.reasons)

    def test_verdict_serializes_on_any_host(self) -> None:
        import json

        # The real monitor, whatever this machine has installed.
        json.dumps(ClockMonitor().check().to_dict())


if __name__ == "__main__":
    unittest.main()
