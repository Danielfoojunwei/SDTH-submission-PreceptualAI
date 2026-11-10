"""Causal, target-specific support from delivered RSSI reports.

This is observation retrieval, not a calibrated RF or service-risk predictor.
A censor bound remains a bound. Unknown measurement time cannot pass a strict
freshness test. Native delivery bins are never called measurement timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

NATIVE_CARRIERS_HZ = (2.4e9, 5.2e9, 6.5e9)
NATIVE_GAUGE = "native_zero_device0_assumed"


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _identity(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _interval(value, name):
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be an ordered pair or unknown")
    lo, hi = (_finite(x, name) for x in value)
    if lo > hi:
        raise ValueError(f"{name} must be ordered")
    return (lo, hi)


def _position(value):
    try:
        if len(value) != 3:
            raise ValueError("position_m must contain XYZ")
        return tuple(_finite(x, "position_m") for x in value)
    except TypeError as exc:
        raise ValueError("position_m must contain XYZ") from exc


@dataclass(frozen=True)
class RFReport:
    report_id: str
    robot_id: int
    site_id: int
    band_index: int
    carrier_hz: float
    position_m: tuple[float, float, float]
    kind: str                         # heard, interval, censored, or missing
    gauge: str
    available_at_s: float             # known available BY this time, not an invented arrival
    measured_interval_s: tuple[float, float] | None
    arrival_interval_s: tuple[float, float] | None = None
    value_dbm: float | None = None
    lower_dbm: float | None = None
    upper_dbm: float | None = None
    position_error_m: float = 0.0
    clock_provenance: str = "explicit"
    source_row: tuple[int, int, int, int] | None = None

    def __post_init__(self):
        for name in ("report_id", "gauge", "clock_provenance"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("robot_id", "site_id", "band_index"):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        carrier = _finite(self.carrier_hz, "carrier_hz")
        if carrier <= 0:
            raise ValueError("carrier_hz must be positive")
        object.__setattr__(self, "carrier_hz", carrier)
        object.__setattr__(self, "position_m", _position(self.position_m))
        available = _finite(self.available_at_s, "available_at_s")
        object.__setattr__(self, "available_at_s", available)
        measured = _interval(self.measured_interval_s, "measured_interval_s")
        arrival = _interval(self.arrival_interval_s, "arrival_interval_s")
        if measured is not None and measured[1] > available:
            raise ValueError("Measurement cannot follow known availability")
        if arrival is not None and (arrival[1] > available or (measured is not None and measured[1] > arrival[0])):
            raise ValueError("Measurement, arrival and availability intervals must be causal")
        object.__setattr__(self, "measured_interval_s", measured)
        object.__setattr__(self, "arrival_interval_s", arrival)
        error = _finite(self.position_error_m, "position_error_m")
        if error < 0:
            raise ValueError("position_error_m must be nonnegative")
        object.__setattr__(self, "position_error_m", error)
        if self.kind not in ("heard", "interval", "censored", "missing"):
            raise ValueError("Unknown report semantics")
        for name in ("value_dbm", "lower_dbm", "upper_dbm"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        fields = (self.value_dbm is not None, self.lower_dbm is not None, self.upper_dbm is not None)
        expected = {"heard": (True, False, False), "interval": (False, True, True),
                    "censored": (False, False, True), "missing": (False, False, False)}
        if fields != expected[self.kind]:
            raise ValueError("Point, interval, censor and missing semantics cannot be interchanged")
        if self.kind == "interval" and self.lower_dbm > self.upper_dbm:
            raise ValueError("Quantization interval must be ordered")
        if self.source_row is not None:
            if not isinstance(self.source_row, (tuple, list)) or len(self.source_row) != 4:
                raise ValueError("source_row must identify history, robot, AP and band")
            object.__setattr__(self, "source_row", tuple(_identity(x, "source_row") for x in self.source_row))


@dataclass(frozen=True)
class SupportResult:
    reports: tuple[RFReport, ...]
    distances_m: tuple[float, ...]             # conservative upper distances
    ages_s: tuple[float, ...]                  # conservative upper ages
    robot_id: int
    candidate_count: int
    eligible_count: int
    unknown_age_count: int
    missing_count: int
    unavailable_count: int
    gauge_mismatch_count: int
    clock_scope: str = "measurement intervals; unknown ages excluded"
    risk_qualified: bool = False

    @property
    def own_reports(self):
        return tuple(row for row in self.reports if row.robot_id == self.robot_id)

    @property
    def latest_own(self):
        own = self.own_reports
        return own[0] if own else None


class SpatialReportMemory:
    """Bounded indexed immutable reports; queries allocate only one link's rows.

    The newest eligible own report is reserved first. Remaining slots minimize
    squared normalized distance plus conservative age. No heard-before-censor
    tie-break can discard a newer own negative report. Bounds are not risk.
    """
    def __init__(self, *, maximum_reports=2_000_000):
        if type(maximum_reports) is not int or maximum_reports < 1:
            raise ValueError("maximum_reports must be a positive integer")
        self.maximum_reports = maximum_reports
        self._reports = {}
        self._groups = {}
        self._arrays = {}
        self.snapshot_metadata = {}

    def __len__(self):
        return len(self._reports)

    def ingest(self, report: RFReport, *, at_s):
        if not isinstance(report, RFReport):
            raise TypeError("Only explicit RFReport observations can enter memory")
        at = _finite(at_s, "at_s")
        if report.available_at_s > at:
            raise ValueError("Cannot ingest a future/unavailable report")
        old = self._reports.get(report.report_id)
        if old is not None:
            if old != report:
                raise ValueError("Conflicting duplicate report identity")
            return False
        if len(self) >= self.maximum_reports:
            raise ValueError("Explicit report capacity exceeded; no evidence silently discarded")
        self._reports[report.report_id] = report
        key = (report.site_id, report.band_index, report.carrier_hz)
        self._groups.setdefault(key, []).append(report)
        self._arrays.pop(key, None)
        return True

    def _group_arrays(self, key):
        if key not in self._arrays:
            rows = tuple(self._groups.get(key, ()))
            self._arrays[key] = (rows, np.asarray([r.position_m for r in rows], dtype=float).reshape(-1, 3),
                np.asarray([r.available_at_s for r in rows]),
                np.asarray([np.nan if r.measured_interval_s is None else r.measured_interval_s[0] for r in rows]),
                np.asarray([r.position_error_m for r in rows]))
        return self._arrays[key]

    def query(self, *, robot_id, site_id, band_index, carrier_hz, position_m,
              issue_time_s, max_age_s, max_distance_m, limit=8, gauge=NATIVE_GAUGE,
              position_error_m=0.0):
        robot = _identity(robot_id, "robot_id")
        site = _identity(site_id, "site_id")
        band = _identity(band_index, "band_index")
        carrier = _finite(carrier_hz, "carrier_hz")
        if carrier <= 0:
            raise ValueError("carrier_hz must be positive")
        position = np.asarray(_position(position_m))
        issue = _finite(issue_time_s, "issue_time_s")
        age_limit = _finite(max_age_s, "max_age_s")
        distance_limit = _finite(max_distance_m, "max_distance_m")
        query_error = _finite(position_error_m, "position_error_m")
        if min(age_limit, distance_limit, query_error) < 0:
            raise ValueError("Age, distance and position-error limits must be nonnegative")
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("limit must be an integer in 1..4096")
        if not isinstance(gauge, str) or not gauge.strip():
            raise ValueError("A nonempty common gauge is required")
        retention = self.snapshot_metadata.get("retention_age_s")
        if retention is not None and (age_limit > retention or issue < self.snapshot_metadata["issue_time_s"]):
            raise ValueError("Query exceeds the explicit retained-age/snapshot-time scope")
        rows, xyz, available, measured_lower, errors = self._group_arrays((site, band, carrier))
        if not rows:
            return SupportResult((), (), (), robot, 0, 0, 0, 0, 0, 0)
        same_gauge = np.asarray([r.gauge == gauge for r in rows])
        known = np.asarray([r.kind != "missing" for r in rows])
        usable = available <= issue
        candidates = same_gauge & usable
        distance = np.linalg.norm(xyz - position, axis=1) + errors + query_error
        age = issue - measured_lower
        eligible = candidates & known & np.isfinite(age) & (age >= 0) & (age <= age_limit) & (distance <= distance_limit)
        indices = np.flatnonzero(eligible).tolist()
        def freshness(i):
            return (age[i], distance[i], rows[i].report_id)
        own = sorted((i for i in indices if rows[i].robot_id == robot), key=freshness)
        reserved = own[:1]
        reserved_set = set(reserved)
        def order(i):
            score = (distance[i] / max(distance_limit, 1e-12))**2 + (age[i] / max(age_limit, 1e-12))**2
            return (score, age[i], distance[i], rows[i].robot_id, rows[i].report_id)
        chosen = reserved + sorted((i for i in indices if i not in reserved_set), key=order)
        chosen = chosen[:limit]
        return SupportResult(tuple(rows[i] for i in chosen), tuple(float(distance[i]) for i in chosen),
            tuple(float(age[i]) for i in chosen), robot, int(candidates.sum()), int(eligible.sum()),
            int((candidates & known & ~np.isfinite(age)).sum()), int((candidates & ~known).sum()),
            int((~usable).sum()), int((usable & ~same_gauge).sum()))


def _native_arrays(past):
    names = ("positions", "value", "heard", "bound", "scanned")
    if not all(hasattr(past, name) for name in names):
        raise TypeError("Explicit observable history required; no World/private field input")
    position, value, heard, bound, scanned = (np.asarray(getattr(past, name)) for name in names)
    if (value.ndim != 3 or value.shape[2] == 0 or value.shape[2] % 3 or
            heard.shape != value.shape or bound.shape != value.shape or
            position.shape != value.shape[:2] + (2,) or scanned.shape != value.shape[:2] or
            heard.dtype != np.dtype(bool) or scanned.dtype != np.dtype(bool) or
            not np.isfinite(position).all() or np.any(heard & ~scanned[..., None])):
        raise ValueError("Native observable axes/masks/positions are inconsistent")
    if not np.isfinite(value[heard]).all() or not np.isfinite(bound[scanned[..., None] & ~heard]).all():
        raise ValueError("Known heard reports and censor bounds must be finite")
    return position, value, heard, bound, scanned


def _native_events(events, *, issue, warm_bins, bin_seconds, slots_per_second, n_robots, history_length):
    latest = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Actual native scan-event dictionaries required")
        keys = ("robot", "measurement_slot", "complete_slot", "delivery_slot")
        if not all(key in event for key in keys):
            raise ValueError("Incomplete native scan clock evidence")
        robot, measured, complete, delivered = (_identity(event[key], key) for key in keys)
        if robot >= n_robots or not measured < complete <= delivered:
            raise ValueError("Native scan measurement/completion/delivery is inconsistent")
        if delivered / slots_per_second > issue:
            raise ValueError("Future scan-event ledger entry supplied to causal snapshot")
        if ("first_usable_decision" in event and
                _identity(event["first_usable_decision"], "first_usable_decision") != int(delivered / slots_per_second / bin_seconds) + 1):
            raise ValueError("Native first-usable-decision disagrees with delivery bin")
        payload = event.get("payload_sha256")
        if not isinstance(payload, str) or len(payload) != 64 or any(x not in "0123456789abcdef" for x in payload):
            raise ValueError("Native scan payload identity required for exact clock reconciliation")
        row = warm_bins + int(delivered / slots_per_second / bin_seconds)
        if row >= history_length:
            # Delivery in the current bin may already be known but is excluded
            # by the forecast prefix's strictly earlier-bin contract.
            continue
        key = (row, robot)
        old = latest.get(key)
        if old is not None and old["delivery_slot"] == delivered and old != event:
            raise ValueError("Conflicting events share one native delivery identity")
        if old is None or delivered > old["delivery_slot"]:
            latest[key] = event
    return latest


def native_history_memory(past, *, issue_time_s, events: Iterable[dict] = (),
                          bands=(0, 1), gauge=NATIVE_GAUGE, warm_seconds=60,
                          bin_seconds=1.0, slots_per_second=100, receiver_height_m=1.0,
                          retention_age_s=None):
    """Build one snapshot index from actual native delivered-history rows.

    Warm rows retain conservative one-second measurement-bin intervals. With
    actual payload-matched events, postwarm rows retain exact measured/arrival
    clocks. Otherwise their measurement time is unknown; delivery-bin indices
    supply only an availability upper bound. Such rows cannot pass freshness.
    The caller must count index construction as real compute, once per issue.

    An explicit retention age materializes only reports whose conservative
    measurement age is within that limit. Older and unknown-age rows are
    counted separately in snapshot_metadata, never relabeled missing. This
    preserves any query with max_age_s <= retention_age_s; wider queries need
    an untruncated memory. Default None retains every available report.
    """
    issue = _finite(issue_time_s, "issue_time_s")
    width = _finite(bin_seconds, "bin_seconds")
    height = _finite(receiver_height_m, "receiver_height_m")
    warm = _finite(warm_seconds, "warm_seconds")
    slots = _identity(slots_per_second, "slots_per_second")
    retention = None if retention_age_s is None else _finite(retention_age_s, "retention_age_s")
    if retention is not None and retention < 0:
        raise ValueError("retention_age_s must be nonnegative or None")
    if issue < 0 or width <= 0 or warm <= 0 or slots == 0 or not math.isclose(warm / width, round(warm / width), abs_tol=1e-10):
        raise ValueError("Native snapshot needs nonnegative issue and positive compatible clocks")
    warm_bins = int(round(warm / width))
    position, value, heard, bound, scanned = _native_arrays(past)
    if len(value) < warm_bins or len(value) > warm_bins + math.floor(issue / width + 1e-10):
        raise ValueError("Native history must contain only complete bins strictly before issue")
    if not isinstance(bands, (tuple, list)) or not bands or len(set(bands)) != len(bands):
        raise ValueError("Distinct native source bands required")
    selected_bands = tuple(_identity(b, "band") for b in bands)
    if any(b >= 3 for b in selected_bands):
        raise ValueError("Unsupported native band")
    event_index = _native_events(events, issue=issue, warm_bins=warm_bins, bin_seconds=width,
        slots_per_second=slots, n_robots=value.shape[1], history_length=len(value))
    if any(not scanned[t, robot] for t, robot in event_index):
        raise ValueError("Delivered native event is absent from its history prefix")
    memory = SpatialReportMemory()
    observed = retained = old_excluded = unknown_excluded = 0
    for t, robot in zip(*np.nonzero(scanned)):
        observed += 1
        t, robot = int(t), int(robot)
        payload = hashlib.sha256(np.ascontiguousarray(value[t, robot]).tobytes() +
                                  np.ascontiguousarray(heard[t, robot]).tobytes()).hexdigest()
        event = event_index.get((t, robot))
        if event is not None and event["payload_sha256"] != payload:
            raise ValueError("Native latest event payload does not match its delivered history row")
        if event is not None and "heard" in event and _identity(event["heard"], "heard") != int(heard[t, robot].sum()):
            raise ValueError("Native event heard count disagrees with its delivered payload")
        if t < warm_bins:
            start = (t - warm_bins) * width
            measured = (start, start + width)
            arrival = None
            available = 0.0
            provenance = "warm_measurement_bin; arrival_unknown_but_available_by_zero"
        elif event is not None:
            mt = event["measurement_slot"] / slots
            dt = event["delivery_slot"] / slots
            measured, arrival, available = (mt, mt), (dt, dt), dt
            provenance = "payload_matched_native_scan_event_exact_slots"
        else:
            measured = arrival = None
            available = (t - warm_bins + 1) * width
            provenance = "delivery_bin_only; measurement_and_exact_arrival_unknown"
        if retention is not None:
            if measured is None:
                unknown_excluded += 1
                continue
            if issue - measured[0] > retention:
                old_excluded += 1
                continue
        retained += 1
        # A stable scan identity includes actual observed values/masks/bounds
        # and position, so two deliveries overwriting a bin cannot alias.
        scan_digest = hashlib.sha256(bytes.fromhex(payload) + np.ascontiguousarray(bound[t, robot]).tobytes() +
            np.ascontiguousarray(position[t, robot]).tobytes()).hexdigest()[:20]
        for site in range(value.shape[2] // 3):
            for band in selected_bands:
                link = 3 * site + band
                is_heard = bool(heard[t, robot, link])
                report = RFReport(f"native:{t}:{robot}:{site}:{band}:{scan_digest}", robot, site, band,
                    NATIVE_CARRIERS_HZ[band], (*position[t, robot], height),
                    "heard" if is_heard else "censored", gauge, available, measured, arrival,
                    value_dbm=float(value[t, robot, link]) if is_heard else None,
                    upper_dbm=None if is_heard else float(bound[t, robot, link]),
                    clock_provenance=provenance, source_row=(t, robot, site, band))
                memory.ingest(report, at_s=issue)
    memory.snapshot_metadata = {
        "retention_age_s": retention,
        "issue_time_s": issue,
        "history_length": len(value),
        "robots": value.shape[1],
        "sites": value.shape[2] // 3,
        "observed_scan_rows": observed,
        "retained_scan_rows": retained,
        "retention_excluded_old_scan_rows": old_excluded,
        "retention_excluded_unknown_clock_scan_rows": unknown_excluded,
        "retained_reports": len(memory),
        "bands": selected_bands,
        "discarded_rows_are_not_missing_observations": True,
    }
    return memory


def inspect_native_support(memory: SpatialReportMemory, *, robot_id, site_id,
                           band_index, position_m, issue_time_s, max_age_s=6.0,
                           max_distance_m=5.0, limit=8, gauge=NATIVE_GAUGE):
    """Inspect one target using an already-built index, without model execution."""
    if not isinstance(memory, SpatialReportMemory):
        raise TypeError("Build one SpatialReportMemory per issue before target inspection")
    band = _identity(band_index, "band_index")
    if band >= 3:
        raise ValueError("Unsupported native band")
    return memory.query(robot_id=robot_id, site_id=site_id, band_index=band,
        carrier_hz=NATIVE_CARRIERS_HZ[band], position_m=position_m,
        issue_time_s=issue_time_s, max_age_s=max_age_s, max_distance_m=max_distance_m,
        limit=limit, gauge=gauge)
