"""Passive spectrum-sensing fabric.

Kval spectrum sensing is strictly *receive-only*. The node never transmits,
jams, generates a waveform, or sets frequency, gain, or transmit power. It
consumes occupancy evidence produced by a root-owned, operator-approved
spectrum sensor adapter and derives bounded intelligence from it: which
approved bands are clear for communication, whether interference or jamming is
suspected, and whether emitters are present ("eyes on the ground") when cameras
are unavailable.

Two facts bound every claim here:

* The sensor adapter, not Kval, owns the radio. Kval verifies provenance and
  refuses any observation that is not marked passive.
* Occupancy sensing is coarse. It can indicate presence, congestion, and
  interference; it is not a substitute for computer vision and is not reported
  as one.
"""

from __future__ import annotations

import json
import re
import socket
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .models import parse_timestamp, utc_now
from .validation import bounded_number, non_empty_str, positive_int, sha256_hex

SPECTRUM_AGENT_SCHEMA = "kval-spectrum-agent/v1"


class BandStatus(str, Enum):
    CLEAR = "CLEAR"
    OCCUPIED = "OCCUPIED"
    CONGESTED = "CONGESTED"
    ANOMALOUS = "ANOMALOUS"
    JAMMING_SUSPECTED = "JAMMING_SUSPECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class SpectrumBand:
    band_id: str
    label: str
    freq_low_hz: int
    freq_high_hz: int
    max_resolution_bandwidth_hz: int
    occupancy_baseline_percent: float
    occupancy_alert_percent: float
    noise_floor_baseline_dbm: float
    comms_candidate: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SpectrumBand":
        band_id = non_empty_str(raw.get("band_id"), "band_id")
        label = non_empty_str(raw.get("label"), "label")
        low = positive_int(raw["freq_low_hz"], "freq_low_hz")
        high = positive_int(raw["freq_high_hz"], "freq_high_hz")
        if high <= low:
            raise ValueError("freq_high_hz must be greater than freq_low_hz")
        resolution = positive_int(raw["max_resolution_bandwidth_hz"], "max_resolution_bandwidth_hz")
        if resolution > (high - low):
            raise ValueError("resolution bandwidth cannot exceed the band width")
        baseline = bounded_number(raw.get("occupancy_baseline_percent", 0.0), "occupancy_baseline_percent", 0.0, 100.0)
        alert = bounded_number(raw.get("occupancy_alert_percent", 80.0), "occupancy_alert_percent", 0.0, 100.0)
        if alert < baseline:
            raise ValueError("occupancy_alert_percent cannot be below occupancy_baseline_percent")
        noise_floor = bounded_number(
            raw.get("noise_floor_baseline_dbm", -95.0), "noise_floor_baseline_dbm", -160.0, 0.0
        )
        return cls(
            band_id=band_id,
            label=label,
            freq_low_hz=low,
            freq_high_hz=high,
            max_resolution_bandwidth_hz=resolution,
            occupancy_baseline_percent=baseline,
            occupancy_alert_percent=alert,
            noise_floor_baseline_dbm=noise_floor,
            comms_candidate=bool(raw.get("comms_candidate", False)),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "label": self.label,
            "freq_low_hz": self.freq_low_hz,
            "freq_high_hz": self.freq_high_hz,
            "comms_candidate": self.comms_candidate,
        }


@dataclass(frozen=True)
class SpectrumPolicy:
    policy_id: str
    region: str
    created_at: datetime
    expires_at: datetime
    passive_only: bool
    max_scans_per_hour: int
    min_scan_interval_seconds: float
    bands: tuple[SpectrumBand, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, now: datetime | None = None) -> "SpectrumPolicy":
        current = now or utc_now()
        policy_id = non_empty_str(raw.get("policy_id"), "policy_id")
        region = non_empty_str(raw.get("region"), "region")
        created_at = parse_timestamp(str(raw["created_at"]))
        expires_at = parse_timestamp(str(raw["expires_at"]))
        if expires_at <= created_at:
            raise ValueError("spectrum policy expiry must be after creation")
        if expires_at <= current:
            raise ValueError("spectrum policy has expired")
        # Receive-only is not optional. A policy that tries to disable it — or
        # that supplies anything other than an explicit true — is rejected.
        if raw.get("passive_only", True) is not True:
            raise ValueError("passive_only cannot be disabled; Kval spectrum sensing is receive-only")
        max_scans = positive_int(raw.get("max_scans_per_hour", 120), "max_scans_per_hour")
        min_interval = bounded_number(
            raw.get("min_scan_interval_seconds", 1.0), "min_scan_interval_seconds", 0.05, 3600.0
        )
        bands = tuple(SpectrumBand.from_dict(item) for item in raw.get("bands", []))
        if not bands:
            raise ValueError("spectrum policy must define at least one approved band")
        band_ids = [band.band_id for band in bands]
        if len(band_ids) != len(set(band_ids)):
            raise ValueError("spectrum policy contains duplicate band IDs")
        return cls(
            policy_id=policy_id,
            region=region,
            created_at=created_at,
            expires_at=expires_at,
            passive_only=True,
            max_scans_per_hour=max_scans,
            min_scan_interval_seconds=min_interval,
            bands=bands,
        )

    def band(self, band_id: str) -> SpectrumBand | None:
        return next((band for band in self.bands if band.band_id == band_id), None)

    def approved_band_ids(self) -> frozenset[str]:
        return frozenset(band.band_id for band in self.bands)


@dataclass(frozen=True)
class SpectrumObservation:
    band_id: str
    observed_at: datetime
    source_adapter_id: str
    conformance_hash: str
    passive: bool
    occupancy_percent: float
    noise_floor_dbm: float
    peak_power_dbm: float | None
    detected_emitters: int
    bearing_deg: float | None
    bearing_confidence: float | None
    quality: float
    errors: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SpectrumObservation":
        band_id = non_empty_str(raw.get("band_id"), "band_id")
        observed_at = parse_timestamp(str(raw["observed_at"]))
        source_adapter_id = non_empty_str(raw.get("source_adapter_id"), "source_adapter_id")
        conformance_hash = sha256_hex(raw.get("conformance_hash"), "conformance_hash")
        # An observation that is not marked passive is refused outright. The
        # check is identity against True so that a truthy string or non-zero
        # number cannot stand in for an explicit passive declaration.
        if raw.get("passive") is not True:
            raise ValueError("spectrum observation must be marked passive; transmit evidence is refused")
        occupancy = bounded_number(raw["occupancy_percent"], "occupancy_percent", 0.0, 100.0)
        noise_floor = bounded_number(raw["noise_floor_dbm"], "noise_floor_dbm", -160.0, 30.0)
        peak_raw = raw.get("peak_power_dbm")
        peak = bounded_number(peak_raw, "peak_power_dbm", -160.0, 60.0) if peak_raw is not None else None
        emitters = int(raw.get("detected_emitters", 0))
        if emitters < 0:
            raise ValueError("detected_emitters cannot be negative")
        bearing_raw = raw.get("bearing_deg")
        bearing = bounded_number(bearing_raw, "bearing_deg", 0.0, 360.0) if bearing_raw is not None else None
        bearing_conf_raw = raw.get("bearing_confidence")
        bearing_conf = (
            bounded_number(bearing_conf_raw, "bearing_confidence", 0.0, 1.0)
            if bearing_conf_raw is not None
            else None
        )
        quality = bounded_number(raw.get("quality", 1.0), "quality", 0.0, 1.0)
        errors = tuple(str(item) for item in raw.get("errors", []))
        return cls(
            band_id=band_id,
            observed_at=observed_at,
            source_adapter_id=source_adapter_id,
            conformance_hash=conformance_hash,
            passive=True,
            occupancy_percent=occupancy,
            noise_floor_dbm=noise_floor,
            peak_power_dbm=peak,
            detected_emitters=emitters,
            bearing_deg=bearing,
            bearing_confidence=bearing_conf,
            quality=quality,
            errors=errors,
        )

    def confidence(self) -> float:
        if self.errors:
            return 0.0
        return round(self.quality, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "observed_at": self.observed_at.isoformat(),
            "source_adapter_id": self.source_adapter_id,
            "conformance_hash": self.conformance_hash,
            "passive": self.passive,
            "occupancy_percent": self.occupancy_percent,
            "noise_floor_dbm": self.noise_floor_dbm,
            "peak_power_dbm": self.peak_power_dbm,
            "detected_emitters": self.detected_emitters,
            "bearing_deg": self.bearing_deg,
            "bearing_confidence": self.bearing_confidence,
            "quality": self.quality,
            "confidence": self.confidence(),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class BandAssessment:
    band_id: str
    label: str
    status: BandStatus
    occupancy_percent: float | None
    noise_floor_excess_db: float | None
    peak_power_dbm: float | None
    detected_emitters: int | None
    comms_candidate: bool
    emitter_present: bool
    bearing_deg: float | None
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "label": self.label,
            "status": self.status.value,
            "occupancy_percent": self.occupancy_percent,
            "noise_floor_excess_db": self.noise_floor_excess_db,
            "peak_power_dbm": self.peak_power_dbm,
            "detected_emitters": self.detected_emitters,
            "comms_candidate": self.comms_candidate,
            "emitter_present": self.emitter_present,
            "bearing_deg": self.bearing_deg,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SpectrumIntelligence:
    policy_id: str
    region: str
    produced_at: datetime
    bands: tuple[BandAssessment, ...]
    jamming_suspected: bool
    clear_comms_band_ids: tuple[str, ...]
    presence_detected: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-spectrum-intelligence/v1",
            "policy_id": self.policy_id,
            "region": self.region,
            "produced_at": self.produced_at.isoformat(),
            "bands": [band.to_dict() for band in self.bands],
            "jamming_suspected": self.jamming_suspected,
            "clear_comms_band_ids": list(self.clear_comms_band_ids),
            "presence_detected": self.presence_detected,
            "reasons": list(self.reasons),
        }


class SpectrumAnalyzer:
    """Derives bounded intelligence from passive occupancy observations.

    All outputs are deterministic heuristics over policy thresholds. A band with
    no fresh, error-free observation is reported as INSUFFICIENT_EVIDENCE rather
    than assumed clear. Jamming is reported as *suspected*, never asserted.
    """

    #: Noise-floor rise (dB above the policy baseline) that, together with high
    #: occupancy, is treated as a jamming indicator rather than ordinary traffic.
    JAMMING_NOISE_EXCESS_DB = 8.0

    #: An observation older than this is not evidence about the current RF
    #: environment. Spectrum occupancy changes in seconds.
    DEFAULT_MAX_OBSERVATION_AGE_SECONDS = 30.0

    #: Clock skew tolerated on an adapter timestamp before it is treated as
    #: implausibly future-dated and discarded.
    FUTURE_TOLERANCE_SECONDS = 1.0

    #: Below this observation quality there is not enough signal to call a band
    #: usable, so the band is reported as insufficient rather than clear.
    MINIMUM_CONFIDENCE = 0.25

    def __init__(
        self,
        *,
        jamming_noise_excess_db: float | None = None,
        max_observation_age_seconds: float | None = None,
        minimum_confidence: float | None = None,
    ) -> None:
        self.jamming_noise_excess_db = (
            jamming_noise_excess_db if jamming_noise_excess_db is not None else self.JAMMING_NOISE_EXCESS_DB
        )
        self.max_observation_age_seconds = (
            max_observation_age_seconds
            if max_observation_age_seconds is not None
            else self.DEFAULT_MAX_OBSERVATION_AGE_SECONDS
        )
        self.minimum_confidence = (
            minimum_confidence if minimum_confidence is not None else self.MINIMUM_CONFIDENCE
        )

    def _insufficient(self, band: SpectrumBand, reason: str) -> BandAssessment:
        return BandAssessment(
            band_id=band.band_id,
            label=band.label,
            status=BandStatus.INSUFFICIENT_EVIDENCE,
            occupancy_percent=None,
            noise_floor_excess_db=None,
            peak_power_dbm=None,
            detected_emitters=None,
            comms_candidate=band.comms_candidate,
            emitter_present=False,
            bearing_deg=None,
            confidence=0.0,
            reasons=(reason,),
        )

    def _assess_band(
        self, band: SpectrumBand, observation: SpectrumObservation | None
    ) -> BandAssessment:
        if observation is None or observation.confidence() <= 0.0:
            reason = "NO_FRESH_OBSERVATION" if observation is None else "OBSERVATION_HAD_ERRORS"
            return BandAssessment(
                band_id=band.band_id,
                label=band.label,
                status=BandStatus.INSUFFICIENT_EVIDENCE,
                occupancy_percent=None,
                noise_floor_excess_db=None,
                peak_power_dbm=None,
                detected_emitters=None,
                comms_candidate=band.comms_candidate,
                emitter_present=False,
                bearing_deg=None,
                confidence=0.0,
                reasons=(reason,),
            )

        if observation.confidence() < self.minimum_confidence:
            return self._insufficient(band, "OBSERVATION_QUALITY_BELOW_MINIMUM")

        occupancy = observation.occupancy_percent
        noise_excess = round(observation.noise_floor_dbm - band.noise_floor_baseline_dbm, 3)
        emitter_present = observation.detected_emitters > 0
        reasons: list[str] = []

        elevated_noise = noise_excess >= self.jamming_noise_excess_db
        high_occupancy = occupancy >= band.occupancy_alert_percent
        if high_occupancy and elevated_noise:
            status = BandStatus.JAMMING_SUSPECTED
            reasons.append("HIGH_OCCUPANCY_WITH_ELEVATED_NOISE_FLOOR")
        elif elevated_noise:
            status = BandStatus.ANOMALOUS
            reasons.append("NOISE_FLOOR_ABOVE_BASELINE")
        elif high_occupancy:
            status = BandStatus.CONGESTED
            reasons.append("OCCUPANCY_AT_OR_ABOVE_ALERT_THRESHOLD")
        elif occupancy > band.occupancy_baseline_percent:
            status = BandStatus.OCCUPIED
            reasons.append("OCCUPANCY_ABOVE_BASELINE")
        else:
            status = BandStatus.CLEAR
            reasons.append("OCCUPANCY_AT_OR_BELOW_BASELINE")
        if emitter_present:
            reasons.append("EMITTER_PRESENT")

        return BandAssessment(
            band_id=band.band_id,
            label=band.label,
            status=status,
            occupancy_percent=occupancy,
            noise_floor_excess_db=noise_excess,
            peak_power_dbm=observation.peak_power_dbm,
            detected_emitters=observation.detected_emitters,
            comms_candidate=band.comms_candidate,
            emitter_present=emitter_present,
            bearing_deg=observation.bearing_deg,
            confidence=observation.confidence(),
            reasons=tuple(reasons),
        )

    def analyze(
        self,
        policy: SpectrumPolicy,
        observations: tuple[SpectrumObservation, ...] | list[SpectrumObservation],
        *,
        now: datetime | None = None,
    ) -> SpectrumIntelligence:
        produced_at = now or utc_now()
        approved = policy.approved_band_ids()
        # Refuse any observation that is outside the signed allowlist, older
        # than the freshness window, or implausibly future-dated. Occupancy
        # changes in seconds, so stale evidence is not evidence.
        by_band: dict[str, SpectrumObservation] = {}
        rejected = 0
        stale = 0
        for observation in observations:
            if observation.band_id not in approved:
                rejected += 1
                continue
            age = (produced_at - observation.observed_at).total_seconds()
            if age > self.max_observation_age_seconds or age < -self.FUTURE_TOLERANCE_SECONDS:
                stale += 1
                continue
            existing = by_band.get(observation.band_id)
            # Strictly-newer wins, so an equal-timestamp duplicate cannot
            # displace the observation already selected.
            if existing is None or observation.observed_at > existing.observed_at:
                by_band[observation.band_id] = observation

        assessments = tuple(self._assess_band(band, by_band.get(band.band_id)) for band in policy.bands)
        jamming = any(a.status is BandStatus.JAMMING_SUSPECTED for a in assessments)
        clear = tuple(
            a.band_id
            for a in assessments
            if a.comms_candidate and a.status in {BandStatus.CLEAR, BandStatus.OCCUPIED}
        )
        presence = any(a.emitter_present for a in assessments)

        reasons: list[str] = []
        if jamming:
            reasons.append("JAMMING_SUSPECTED_ON_AT_LEAST_ONE_BAND")
        if not clear:
            reasons.append("NO_CLEAR_APPROVED_COMMS_BAND")
        if presence:
            reasons.append("EMITTER_PRESENCE_DETECTED")
        if rejected:
            reasons.append("OBSERVATIONS_OUTSIDE_POLICY_REJECTED")
        if stale:
            reasons.append("STALE_OR_FUTURE_OBSERVATIONS_DISCARDED")
        if not reasons:
            reasons.append("NOMINAL")

        return SpectrumIntelligence(
            policy_id=policy.policy_id,
            region=policy.region,
            produced_at=produced_at,
            bands=assessments,
            jamming_suspected=jamming,
            clear_comms_band_ids=clear,
            presence_detected=presence,
            reasons=tuple(reasons),
        )


class ScanRateExceeded(RuntimeError):
    """Raised when a scan would exceed the signed policy's duty limits."""


class ScanRateLimiter:
    """Durable enforcement of a spectrum policy's scan duty limits.

    ``max_scans_per_hour`` and ``min_scan_interval_seconds`` are emission-duty
    and energy controls, so they have to hold across process restarts — a
    per-process counter would reset on every CLI invocation. Scan times are
    recorded in SQLite alongside the node's other durable state.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS spectrum_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL,
                scanned_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _recent(self, policy_id: str, since: datetime) -> list[datetime]:
        rows = self._connection.execute(
            "SELECT scanned_at FROM spectrum_scans WHERE policy_id = ? AND scanned_at >= ? ORDER BY scanned_at ASC",
            (policy_id, since.isoformat()),
        ).fetchall()
        return [parse_timestamp(str(row[0])) for row in rows]

    def check_and_record(self, policy: SpectrumPolicy, *, now: datetime | None = None) -> None:
        """Raise :class:`ScanRateExceeded` if a scan now would breach policy."""
        current = now or utc_now()
        window_start = current - timedelta(hours=1)
        recent = self._recent(policy.policy_id, window_start)
        if len(recent) >= policy.max_scans_per_hour:
            raise ScanRateExceeded(
                f"scan refused: {len(recent)} scans in the last hour reaches the policy limit "
                f"of {policy.max_scans_per_hour}"
            )
        if recent:
            elapsed = (current - recent[-1]).total_seconds()
            if elapsed < policy.min_scan_interval_seconds:
                raise ScanRateExceeded(
                    f"scan refused: {elapsed:.3f}s since the last scan is below the policy minimum "
                    f"of {policy.min_scan_interval_seconds}s"
                )
        with self._connection:
            self._connection.execute(
                "INSERT INTO spectrum_scans (policy_id, scanned_at) VALUES (?, ?)",
                (policy.policy_id, current.isoformat()),
            )
            # Keep the table bounded; anything older than the window is unused.
            self._connection.execute(
                "DELETE FROM spectrum_scans WHERE scanned_at < ?",
                ((current - timedelta(hours=2)).isoformat(),),
            )


class SpectrumSensorClient:
    """Fail-closed client for a root-owned, receive-only spectrum sensor adapter.

    Mirrors the trust model of the radio standards agent: the socket must be a
    root-owned Unix socket, requests only ever ask for occupancy on approved
    band IDs, and any response that is malformed, wrong-schema, or non-passive
    is rejected.
    """

    MAX_RESPONSE_BYTES = 262_144
    _HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")

    @staticmethod
    def _validate_socket(path: Path) -> None:
        try:
            metadata = path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"spectrum sensor socket is missing: {path}") from exc
        if not stat.S_ISSOCK(metadata.st_mode):
            raise RuntimeError(f"spectrum sensor endpoint is not a Unix socket: {path}")
        if metadata.st_uid != 0:
            raise RuntimeError("spectrum sensor socket must be owned by root")
        if metadata.st_mode & 0o022:
            raise RuntimeError("spectrum sensor socket cannot be group/world writable")

    def _request(self, path: Path, request: dict[str, Any]) -> dict[str, Any]:
        self._validate_socket(path)
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(str(path))
            client.sendall(encoded)
            response = bytearray()
            while len(response) <= self.MAX_RESPONSE_BYTES:
                chunk = client.recv(8192)
                if not chunk:
                    break
                response.extend(chunk)
                if b"\n" in chunk:
                    break
        if len(response) > self.MAX_RESPONSE_BYTES:
            raise RuntimeError("spectrum sensor response exceeded size limit")
        try:
            parsed = json.loads(bytes(response).split(b"\n", 1)[0])
        except json.JSONDecodeError as exc:
            raise RuntimeError("spectrum sensor returned malformed JSON") from exc
        if not isinstance(parsed, dict) or parsed.get("schema") != SPECTRUM_AGENT_SCHEMA:
            raise RuntimeError("spectrum sensor returned an unsupported schema")
        return parsed

    def scan(self, path: Path, policy: SpectrumPolicy) -> tuple[SpectrumObservation, ...]:
        response = self._request(
            path,
            {
                "operation": "scan",
                "mode": "passive",
                "policy_id": policy.policy_id,
                "region": policy.region,
                "band_ids": sorted(policy.approved_band_ids()),
            },
        )
        if response.get("success") is not True:
            raise RuntimeError(f"spectrum sensor rejected scan: {response.get('error', 'unknown error')}")
        raw_observations = response.get("observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise RuntimeError("spectrum sensor returned no observations")
        observations = tuple(SpectrumObservation.from_dict(item) for item in raw_observations)
        approved = policy.approved_band_ids()
        for observation in observations:
            if observation.band_id not in approved:
                raise RuntimeError("spectrum sensor returned an observation outside the signed policy")
        return observations
