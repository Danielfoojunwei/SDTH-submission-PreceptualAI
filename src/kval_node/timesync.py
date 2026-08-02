"""Clock-integrity checks for cross-node correlation.

Threat 11 in ``docs/SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md`` is GNSS/time
spoofing. Two nodes' spectrum observations can only be compared if their clocks
are comparable, so anything that fuses observations across nodes has to know
whether the local clock is actually disciplined — and refuse to correlate when
it is not.

This module measures that from real system state:

* ``timedatectl`` / ``systemd-timesyncd`` for NTP synchronisation status;
* ``chronyc tracking`` for a chrony-managed host, including the estimated
  offset and root dispersion;
* the divergence between ``CLOCK_REALTIME`` and ``CLOCK_MONOTONIC`` across the
  process lifetime, which catches a wall clock that has been stepped.

Nothing here trusts a timestamp because it looks plausible. An undisciplined
clock yields ``TimeIntegrity.trusted == False`` and the fabric layer degrades to
local-only decisions rather than fusing observations it cannot align.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Any

from .commands import CommandError, run_command
from .models import utc_now


class ClockSource(str, Enum):
    CHRONY = "chrony"
    SYSTEMD_TIMESYNCD = "systemd-timesyncd"
    UNKNOWN = "unknown"


#: Maximum estimated offset, in seconds, at which cross-node correlation is
#: still meaningful. Spectrum observations are compared in windows of seconds,
#: so a clock more than this far out cannot be aligned against a peer.
DEFAULT_MAX_OFFSET_SECONDS = 0.25

#: Maximum tolerated drift between the wall clock and the monotonic clock over
#: the process lifetime before the wall clock is treated as having been stepped.
DEFAULT_MAX_STEP_SECONDS = 2.0


@dataclass(frozen=True)
class TimeIntegrity:
    checked_at: datetime
    trusted: bool
    source: ClockSource
    synchronised: bool
    offset_seconds: float | None
    root_dispersion_seconds: float | None
    observed_step_seconds: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_at": self.checked_at.isoformat(),
            "trusted": self.trusted,
            "source": self.source.value,
            "synchronised": self.synchronised,
            "offset_seconds": self.offset_seconds,
            "root_dispersion_seconds": self.root_dispersion_seconds,
            "observed_step_seconds": self.observed_step_seconds,
            "reasons": list(self.reasons),
        }


_TIMEDATECTL_NTP = re.compile(r"(?:NTP synchronized|System clock synchronized):\s*(?P<value>yes|no)", re.IGNORECASE)
_CHRONY_OFFSET = re.compile(r"^Last offset\s*:\s*(?P<value>[-+]?[0-9.]+)\s*seconds", re.MULTILINE)
_CHRONY_DISPERSION = re.compile(r"^Root dispersion\s*:\s*(?P<value>[0-9.]+)\s*seconds", re.MULTILINE)
_CHRONY_LEAP = re.compile(r"^Leap status\s*:\s*(?P<value>.+)$", re.MULTILINE)


def parse_timedatectl(output: str) -> bool | None:
    """Return whether ``timedatectl`` reports the system clock synchronised."""
    match = _TIMEDATECTL_NTP.search(output)
    if match is None:
        return None
    return match.group("value").lower() == "yes"


def parse_chronyc_tracking(output: str) -> tuple[float | None, float | None, bool]:
    """Return ``(offset_seconds, root_dispersion_seconds, leap_normal)``."""
    offset_match = _CHRONY_OFFSET.search(output)
    dispersion_match = _CHRONY_DISPERSION.search(output)
    leap_match = _CHRONY_LEAP.search(output)
    offset = abs(float(offset_match.group("value"))) if offset_match else None
    dispersion = float(dispersion_match.group("value")) if dispersion_match else None
    leap_normal = bool(leap_match and leap_match.group("value").strip().lower() == "normal")
    return offset, dispersion, leap_normal


#: What a chrony query yields: (offset seconds, root dispersion, leap normal).
ChronyReading = tuple[float | None, float | None, bool]

#: A source of time-daemon evidence. Returning ``None`` means the daemon could
#: not be reached, which is different from it reporting bad news.
ChronySource = Callable[[], "ChronyReading | None"]
NtpSyncSource = Callable[[], "bool | None"]


def query_chrony() -> ChronyReading | None:
    """Ask ``chronyc`` what the disciplined clock looks like, or admit it cannot."""
    try:
        result = run_command(["chronyc", "tracking"], timeout_seconds=5.0)
    except CommandError:
        return None
    if result.returncode != 0:
        return None
    return parse_chronyc_tracking(result.stdout)


def query_timedatectl() -> bool | None:
    """Ask systemd whether the clock is NTP-synchronised, or admit it cannot."""
    try:
        result = run_command(
            ["timedatectl", "show", "--property=NTPSynchronized", "--value"],
            timeout_seconds=5.0,
        )
    except CommandError:
        return None
    if result.returncode == 0:
        value = result.stdout.strip().lower()
        if value in {"yes", "true"}:
            return True
        if value in {"no", "false"}:
            return False
    try:
        fallback = run_command(["timedatectl", "status"], timeout_seconds=5.0)
    except CommandError:
        return None
    if fallback.returncode != 0:
        return None
    return parse_timedatectl(fallback.stdout)


class ClockMonitor:
    """Tracks wall-vs-monotonic divergence and queries the host time daemon.

    The two daemon queries are injected rather than hard-wired. A clock verdict
    must depend on what the daemons report, not on what the machine running the
    code happens to have installed — a node with a healthy NTP client and a
    container with none must reach deterministic verdicts from the same inputs.
    Defaulting to the real queries keeps production behaviour unchanged; being
    able to supply them is what makes the decision logic testable without
    subclassing around it.
    """

    def __init__(
        self,
        *,
        max_offset_seconds: float = DEFAULT_MAX_OFFSET_SECONDS,
        max_step_seconds: float = DEFAULT_MAX_STEP_SECONDS,
        chrony_source: ChronySource | None = None,
        ntp_sync_source: NtpSyncSource | None = None,
    ) -> None:
        self.max_offset_seconds = max_offset_seconds
        self.max_step_seconds = max_step_seconds
        self._chrony_source = chrony_source or query_chrony
        self._ntp_sync_source = ntp_sync_source or query_timedatectl
        self._wall_at_start = time.time()
        self._monotonic_at_start = time.monotonic()

    def observed_step_seconds(self) -> float:
        """Wall-clock movement not explained by monotonic elapsed time."""
        wall_elapsed = time.time() - self._wall_at_start
        monotonic_elapsed = time.monotonic() - self._monotonic_at_start
        return abs(wall_elapsed - monotonic_elapsed)

    def reset_baseline(self) -> None:
        self._wall_at_start = time.time()
        self._monotonic_at_start = time.monotonic()

    def check(self, *, now: datetime | None = None) -> TimeIntegrity:
        checked_at = now or utc_now()
        reasons: list[str] = []
        step = self.observed_step_seconds()

        chrony = self._chrony_source()
        if chrony is not None:
            offset, dispersion, leap_normal = chrony
            source = ClockSource.CHRONY
            synchronised = leap_normal and offset is not None
            if not leap_normal:
                reasons.append("CHRONY_LEAP_STATUS_NOT_NORMAL")
            if offset is None:
                reasons.append("CHRONY_REPORTED_NO_OFFSET")
        else:
            offset = None
            dispersion = None
            source = ClockSource.UNKNOWN
            ntp = self._ntp_sync_source()
            if ntp is None:
                synchronised = False
                reasons.append("NO_TIME_DAEMON_EVIDENCE")
            else:
                source = ClockSource.SYSTEMD_TIMESYNCD
                synchronised = ntp
                if not ntp:
                    reasons.append("SYSTEM_CLOCK_NOT_SYNCHRONISED")

        if offset is not None and offset > self.max_offset_seconds:
            reasons.append("CLOCK_OFFSET_ABOVE_LIMIT")
        if dispersion is not None and dispersion > self.max_offset_seconds:
            reasons.append("ROOT_DISPERSION_ABOVE_LIMIT")
        if step > self.max_step_seconds:
            reasons.append("WALL_CLOCK_STEPPED")

        trusted = synchronised and not reasons
        if trusted:
            reasons.append("CLOCK_DISCIPLINED")
        return TimeIntegrity(
            checked_at=checked_at,
            trusted=trusted,
            source=source,
            synchronised=synchronised,
            offset_seconds=offset,
            root_dispersion_seconds=dispersion,
            observed_step_seconds=round(step, 6),
            reasons=tuple(reasons),
        )
