"""Measured RSSI/FTM pairs from the official UVIndoorLoc raw archive.

Only two quantities from the SAME AP ranging result form a pair. Neither a
common request callback nor matching reference locations establishes simultaneous
captures across APs or phones. Signed device ranges are observations, not latent
physical path lengths. This module performs no downloads, fits or imputation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Iterator, Literal, Sequence
import zipfile

import numpy as np

SOURCE_URL = 'https://doi.org/10.5281/zenodo.20051265'
SOURCE_SHA256 = '0ff469b8bd9d58a7d37c77423fc40450916bf26bbe208010bfc2bdf075ccf9f5'
SOURCE_LICENSE = 'CC-BY-4.0'
HEADER = (
    'Request Start Unix Timestamp', 'Request Completed Unix Timestamp',
    'Time Since System Boot (ms)', 'Device', 'Android Version', 'AP MAC Address',
    'Status', 'Distance (mm)', 'Std Dev (mm)', 'Attempted Measurements',
    'Successful Measurements', 'Bandwidth', 'Frequency (MHz)', 'Average RSSI (dbM)',
)
MEMBER = re.compile(r'raw/(?P<scenario>S0[123])_(?P<phone>POCO|GP10|S24U)_'
                    r'(?P<point>[01][0-9]{2})_(?P<session>0[12])\.csv\Z')
MAC = re.compile(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
UNSUPPORTED = 'Requires Android >= 14'


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(path, expected):
    if not isinstance(expected, str) or not SHA.fullmatch(expected):
        raise ValueError('An explicit SHA256 archive identity is required')
    path = Path(path)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError('Measured source archive is missing or its SHA256 differs')
    return path


def _identity(archive_sha256, member, row):
    value = f'{archive_sha256}\0{member}\0{row}'.encode()
    return hashlib.sha256(value).hexdigest()


def _integer(row, name, *, minimum=None):
    value = row.get(name)
    if not isinstance(value, str) or not re.fullmatch(r'-?[0-9]+', value.strip()):
        raise ValueError('Missing or noninteger measured field: ' + name)
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError('Measured integer outside its documented domain: ' + name)
    return result


def _number(row, name, *, minimum=None):
    try:
        result = float(row[name])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Missing or nonnumeric measured field: ' + name) from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError('Nonfinite or out-of-domain measured field: ' + name)
    return result


def _metadata_frequency(row):
    value = row.get('Frequency (MHz)')
    if value == UNSUPPORTED:
        return 0., False
    frequency = _integer(row, 'Frequency (MHz)', minimum=0)
    if frequency == 0:  # Actual provider records include zero-valued metadata.
        return 0., False
    if not 2300 <= frequency <= 7200:
        raise ValueError('UV Wi-Fi frequency is not expressed in the declared MHz domain')
    return float(frequency) * 1.e6, True


def _metadata_bandwidth(row):
    value = row.get('Bandwidth')
    if value == UNSUPPORTED:
        return -1, False
    code = _integer(row, 'Bandwidth', minimum=-1)
    # Preserve Android's code. Do not turn it into hertz without its SDK mapping.
    return code, code >= 0


@dataclass(frozen=True)
class SourceKey:
    archive_sha256: str
    source_file: str
    source_row: int
    record_id: str

    def __post_init__(self):
        if (not SHA.fullmatch(self.archive_sha256) or not MEMBER.fullmatch(self.source_file)
                or type(self.source_row) is not int or self.source_row < 2
                or self.record_id != _identity(self.archive_sha256, self.source_file, self.source_row)):
            raise ValueError('Invalid opaque measured-row identity')


@dataclass
class FilterSummary:
    """Acquisition accounting, not an outage or command-service endpoint."""
    source_rows: int = 0
    accepted: int = 0
    excluded_status: int = 0
    excluded_zero_success: int = 0
    negative_measured_ranges: int = 0
    large_absolute_measured_ranges: int = 0
    unknown_frequency: int = 0


@dataclass(frozen=True)
class PairedRSSIFTMRecord:
    key: SourceKey
    rssi_dbm: float
    measured_range_m: float
    reported_std_m: float
    status: int
    attempted_measurements: int
    successful_measurements: int
    request_start_unix_ms: int
    completion_unix_ms: int
    capture_boot_ms: int
    device_id: str
    android_version: int
    ap_id: str
    scenario: str
    session: int
    day_utc: str
    partition: str
    frequency_hz: float
    frequency_known: bool
    bandwidth_code: int
    bandwidth_known: bool

    @property
    def negative_measured_range(self):
        return self.measured_range_m < 0.

    @property
    def large_absolute_measured_range(self):
        """A fixed diagnostic flag only; never a geometry-based rejection."""
        return abs(self.measured_range_m) > 100.


def _members(archive, selected):
    names = archive.namelist()
    actual = set(names)
    if len(names) != len(actual):
        raise ValueError('Duplicate archive member identities are not permitted')
    if selected is None:
        return sorted(name for name in actual if MEMBER.fullmatch(name))
    selected = tuple(selected)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select unique existing raw measurement members')
    if any(not isinstance(name, str) or not MEMBER.fullmatch(name) or name not in actual
           for name in selected):
        raise ValueError('Selected member is not a raw UV ranging-measurement file')
    return sorted(selected)


def _records(archive, member, source_sha256, summary):
    match = MEMBER.fullmatch(member)
    with archive.open(member) as binary:
        reader = csv.DictReader(io.TextIOWrapper(binary, encoding='utf-8-sig', newline=''))
        if tuple(reader.fieldnames or ()) != HEADER:
            raise ValueError('Raw ranging CSV schema differs: ' + member)
        for row in reader:
            summary.source_rows += 1
            if None in row:
                raise ValueError('Unexpected extra CSV fields: ' + member)
            status = _integer(row, 'Status', minimum=0)
            if status != 0:
                # Provider error rows legitimately omit measurement columns.
                summary.excluded_status += 1
                continue
            attempted = _integer(row, 'Attempted Measurements', minimum=0)
            successes = _integer(row, 'Successful Measurements', minimum=0)
            if successes > attempted:
                raise ValueError('Successful ranging count exceeds attempts')
            if successes == 0:
                summary.excluded_zero_success += 1
                continue
            start = _integer(row, 'Request Start Unix Timestamp')
            end = _integer(row, 'Request Completed Unix Timestamp')
            if not 946684800000 <= start <= end <= 4102444800000:
                raise ValueError('Invalid UNIX millisecond request/completion clocks')
            boot = _integer(row, 'Time Since System Boot (ms)', minimum=0)
            # Boot and UNIX clocks remain separate. No unknown offset is invented.
            device = row.get('Device', '').strip()
            ap = row.get('AP MAC Address', '').strip().lower()
            if not device or len(device) > 256 or not MAC.fullmatch(ap):
                raise ValueError('Missing device identity or malformed AP address')
            frequency, frequency_known = _metadata_frequency(row)
            bandwidth, bandwidth_known = _metadata_bandwidth(row)
            key = SourceKey(source_sha256, member, reader.line_num,
                            _identity(source_sha256, member, reader.line_num))
            session = int(match['session'])
            partition = ('training' if match['point'][0] == '0' else f'test_day{session}')
            record = PairedRSSIFTMRecord(key=key,
                rssi_dbm=_number(row, 'Average RSSI (dbM)'),
                measured_range_m=_number(row, 'Distance (mm)') / 1000.,
                reported_std_m=_number(row, 'Std Dev (mm)', minimum=0) / 1000.,
                status=status, attempted_measurements=attempted, successful_measurements=successes,
                request_start_unix_ms=start, completion_unix_ms=end, capture_boot_ms=boot,
                device_id=device, android_version=_integer(row, 'Android Version', minimum=1),
                ap_id=ap, scenario=match['scenario'], session=session,
                day_utc=datetime.fromtimestamp(start / 1000., timezone.utc).date().isoformat(),
                partition=partition, frequency_hz=frequency, frequency_known=frequency_known,
                bandwidth_code=bandwidth, bandwidth_known=bandwidth_known)
            summary.accepted += 1
            summary.negative_measured_ranges += record.negative_measured_range
            summary.large_absolute_measured_ranges += record.large_absolute_measured_range
            summary.unknown_frequency += not frequency_known
            yield record


def iter_public_records(archive_path, *, expected_sha256=SOURCE_SHA256,
                        members: Sequence[str] | None = None,
                        summary: FilterSummary | None = None) -> Iterator[PairedRSSIFTMRecord]:
    """Read raw radio observations only; never opens location/position labels.

    The declared eligibility filter is status==0 and successful_measurements>0.
    Invalid numeric/schema/clock fields raise, rather than manufacturing values.
    Negative or large reported ranges remain in the accepted observations.
    """
    path = _verify_archive(archive_path, expected_sha256)
    summary = FilterSummary() if summary is None else summary
    with zipfile.ZipFile(path) as archive:
        names = _members(archive, members)
        if not names:
            raise ValueError('No raw ranging measurements in the source archive')
        for member in names:
            yield from _records(archive, member, expected_sha256, summary)


def _sealed(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class MeasurementView:
    kind: str
    values: np.ndarray
    value_names: tuple[str, ...]
    device_ids: tuple[str, ...]
    ap_ids: tuple[str, ...]
    frequency_hz: np.ndarray
    frequency_known: np.ndarray


@dataclass(frozen=True)
class PairedRSSIFTMBatch:
    keys: tuple[SourceKey, ...]
    rssi_dbm: np.ndarray
    measured_range_m: np.ndarray
    reported_std_m: np.ndarray
    statuses: np.ndarray
    attempted_measurements: np.ndarray
    successful_measurements: np.ndarray
    request_start_unix_ms: np.ndarray
    completion_unix_ms: np.ndarray
    capture_boot_ms: np.ndarray
    device_ids: tuple[str, ...]
    android_versions: np.ndarray
    ap_ids: tuple[str, ...]
    scenarios: tuple[str, ...]
    sessions: np.ndarray
    days_utc: tuple[str, ...]
    partitions: tuple[str, ...]
    frequency_hz: np.ndarray
    frequency_known: np.ndarray
    bandwidth_codes: np.ndarray
    bandwidth_known: np.ndarray
    negative_measured_ranges: np.ndarray
    large_absolute_measured_ranges: np.ndarray

    def view(self, modality: Literal['rssi', 'ftm']) -> MeasurementView:
        """Whitelisted modalities; provenance/location tags are not features."""
        if modality == 'rssi':
            values, names = self.rssi_dbm[:, None], ('rssi_dbm',)
        elif modality == 'ftm':
            values = np.stack((self.measured_range_m, self.reported_std_m), -1)
            names = ('signed_measured_range_m', 'reported_std_m')
        else:
            raise ValueError('Choose an observed RSSI or FTM view')
        return MeasurementView(modality, _sealed(values, 'f8'), names,
                               self.device_ids, self.ap_ids,
                               self.frequency_hz, self.frequency_known)


def _batch(records):
    def array(name, dtype):
        return _sealed([getattr(record, name) for record in records], dtype)
    def values(name):
        return tuple(getattr(record, name) for record in records)
    return PairedRSSIFTMBatch(keys=values('key'), rssi_dbm=array('rssi_dbm', 'f8'),
        measured_range_m=array('measured_range_m', 'f8'), reported_std_m=array('reported_std_m', 'f8'),
        statuses=array('status', 'i8'),
        attempted_measurements=array('attempted_measurements', 'i8'),
        successful_measurements=array('successful_measurements', 'i8'),
        request_start_unix_ms=array('request_start_unix_ms', 'i8'),
        completion_unix_ms=array('completion_unix_ms', 'i8'), capture_boot_ms=array('capture_boot_ms', 'i8'),
        device_ids=values('device_id'), android_versions=array('android_version', 'i8'),
        ap_ids=values('ap_id'), scenarios=values('scenario'), sessions=array('session', 'i8'),
        days_utc=values('day_utc'), partitions=values('partition'),
        frequency_hz=array('frequency_hz', 'f8'), frequency_known=array('frequency_known', '?'),
        bandwidth_codes=array('bandwidth_code', 'i8'), bandwidth_known=array('bandwidth_known', '?'),
        negative_measured_ranges=array('negative_measured_range', '?'),
        large_absolute_measured_ranges=array('large_absolute_measured_range', '?'))


def iter_public_batches(archive_path, *, batch_size=4096, expected_sha256=SOURCE_SHA256,
                        members=None, summary=None):
    """Bounded-memory measurement batches; preserves file order and source IDs."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('A positive integer batch size is required')
    pending = []
    for record in iter_public_records(archive_path, expected_sha256=expected_sha256,
                                      members=members, summary=summary):
        pending.append(record)
        if len(pending) == batch_size:
            yield _batch(pending)
            pending = []
    if pending:
        yield _batch(pending)


@dataclass(frozen=True)
class PrivateReferenceLabels:
    keys: tuple[SourceKey, ...]
    reference_ids: tuple[str, ...]
    xyz_m: np.ndarray


def private_reference_labels(archive_path, keys: Sequence[SourceKey], *, expected_sha256=SOURCE_SHA256):
    """Explicit probe-label read, separate from every public measurement API.

    Labels are reference-position tags, not measured propagation ranges. Source
    rows must exist and pass the same eligibility filter; fabricated keys fail.
    """
    path = _verify_archive(archive_path, expected_sha256)
    keys = tuple(keys)
    if (not keys or len(set(keys)) != len(keys)
            or any(not isinstance(key, SourceKey) or key.archive_sha256 != expected_sha256 for key in keys)):
        raise ValueError('Unique keys from this measured source are required')
    members = sorted({key.source_file for key in keys})
    actual = {record.key for record in iter_public_records(path, expected_sha256=expected_sha256,
                                                          members=members)}
    if not set(keys) <= actual:
        raise ValueError('Requested probe row is missing or was not an eligible measured result')
    locations = {}
    with zipfile.ZipFile(path) as archive:
        with archive.open('00_locations.csv') as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
            if not {'id', 'x', 'y', 'z'} <= set(reader.fieldnames or ()):
                raise ValueError('Private reference-position schema differs')
            for row in reader:
                identifier = row['id']
                if identifier in locations:
                    raise ValueError('Duplicated private reference-position identity')
                locations[identifier] = [_number(row, name) for name in ('x', 'y', 'z')]
    identifiers, labels = [], []
    for key in keys:
        match = MEMBER.fullmatch(key.source_file)
        identifier = match['scenario'] + 'P' + match['point']
        if identifier not in locations:
            raise ValueError('Measured row lacks its private reference-position label')
        identifiers.append(identifier)
        labels.append(locations[identifier])
    return PrivateReferenceLabels(keys, tuple(identifiers), _sealed(labels, 'f8'))
