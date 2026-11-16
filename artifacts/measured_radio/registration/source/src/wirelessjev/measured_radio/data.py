"""Pinned UV reported-RSSI summaries, separate from hardware capture records.

Metadata inspection opens only public coordinate catalogues and archive member
names. Preparing labels is an explicit later operation. The registered unit is
one arithmetic mean of at most eight identity-selected reported dBm readings,
not an instantaneous radio measurement or a standard error of that mean.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import math
from pathlib import Path
import re
import zipfile

NAMESPACE = 'uv-measured-v1'
SOURCE_URL = 'https://doi.org/10.5281/zenodo.20051265'
SOURCE_SHA256 = '0ff469b8bd9d58a7d37c77423fc40450916bf26bbe208010bfc2bdf075ccf9f5'
SOURCE_LICENSE = 'CC-BY-4.0'
METADATA_MEMBER_SHA256 = {
    'raw/aps/S01.csv': '2f24fe042f6c4c304e5186a63d5f259552baa4d0c0b964d04d6780f6c88c068d',
    'raw/aps/S02.csv': 'a06a32a891910e2e46d867a91e6981402825e11fa245cb71321c09df972d2c5d',
    'raw/aps/S03.csv': '944e24b38f1cd2fdfde40a06be2c00389e1ec26e3600adcd7999629137263eae',
    'raw/locations/S01.csv': '6e8676315cb1ec4781155922365fb4795230a25f45ad699b4fd3b2df9da38048',
    'raw/locations/S02.csv': '45560b1c6b1d6fc026e199fa0496b99c3058f23fc063bc3e543faab6c3c6e07a',
    'raw/locations/S03.csv': 'ab8b1fe6373fb21b83d8954a9de0a2f19ecb630a8be28feb2a467673c21807eb',
}
RAW_HEADER = (
    'Request Start Unix Timestamp', 'Request Completed Unix Timestamp',
    'Time Since System Boot (ms)', 'Device', 'Android Version', 'AP MAC Address',
    'Status', 'Distance (mm)', 'Std Dev (mm)', 'Attempted Measurements',
    'Successful Measurements', 'Bandwidth', 'Frequency (MHz)', 'Average RSSI (dbM)',
)
AP_HEADER = ('id', 'mac', 'x', 'y', 'z', 'orientation')
LOCATION_HEADER = ('id', 'x', 'y', 'z', 'orientation', 'type')
SCENARIOS = ('S01', 'S02', 'S03')
PHONES = {'GP10': 'GooglePixel 10', 'S24U': 'samsungSM-S928B', 'POCO': 'XiaomiM2102J20SG'}
UNSUPPORTED = 'Requires Android >= 14'
MEMBER = re.compile(r'raw/(?P<scenario>S0[123])_(?P<phone>POCO|GP10|S24U)_'
                    r'(?P<point>[01][0-9]{2})_(?P<session>0[12])\.csv\Z')
MAC = re.compile(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _verified_archive(path, expected_sha256):
    if not isinstance(expected_sha256, str) or not SHA.fullmatch(expected_sha256):
        raise ValueError('An explicit source SHA256 is required')
    path = Path(path)
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError('Measured archive SHA256 differs or source is missing')
    return path


def _pins(expected_sha256, member_pins):
    if member_pins is None:
        if expected_sha256 != SOURCE_SHA256:
            raise ValueError('Nonofficial archives require explicit metadata member SHA256 pins')
        member_pins = METADATA_MEMBER_SHA256
    if (not isinstance(member_pins, dict) or set(member_pins) != set(METADATA_MEMBER_SHA256)
            or any(not isinstance(v, str) or not SHA.fullmatch(v) for v in member_pins.values())):
        raise ValueError('All six coordinate metadata member SHA256 pins are required')
    if expected_sha256 == SOURCE_SHA256 and member_pins != METADATA_MEMBER_SHA256:
        raise ValueError('Official coordinate metadata pins cannot be changed')
    return dict(member_pins)


def _table(archive, member, header, member_pins):
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError('Missing coordinate metadata member: ' + member) from exc
    if hashlib.sha256(payload).hexdigest() != member_pins[member]:
        raise ValueError('Coordinate metadata member SHA256 differs: ' + member)
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    if tuple(reader.fieldnames or ()) != header:
        raise ValueError('Coordinate metadata schema differs: ' + member)
    rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError('Empty or malformed coordinate metadata: ' + member)
    return rows


def _number(value, name):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Nonnumeric ' + name) from exc
    if not math.isfinite(result):
        raise ValueError('Nonfinite ' + name)
    return result


def _integer(value, name, minimum=0):
    if not isinstance(value, str) or not re.fullmatch(r'-?[0-9]+', value.strip()):
        raise ValueError('Noninteger ' + name)
    result = int(value)
    if result < minimum:
        raise ValueError('Out-of-domain ' + name)
    return result


def _xyz(row):
    return [_number(row[c], 'coordinate ' + c) for c in ('x', 'y', 'z')]


def _rank_locations(scenario, locations):
    """Nested maximin order from TRAIN coordinate metadata alone."""
    candidates = {r['id']: r['xyz_provider'] for r in locations if r['partition'] == 'train'}
    tie = {key: _digest(f'{NAMESPACE}:support:{scenario}:{key}') for key in candidates}
    order, selected_points = [], []
    while candidates:
        if not order:
            chosen = min(candidates, key=lambda key: (tie[key], key))
        else:
            def key(identifier):
                distance = min(sum((a - b) ** 2 for a, b in zip(candidates[identifier], selected_xyz))
                               for selected_xyz in selected_points)
                return (-distance, tie[identifier], identifier)
            chosen = min(candidates, key=key)
        selected_points.append(candidates.pop(chosen))
        order.append(chosen)
    return order


def _metadata(archive, source_sha256, member_pins):
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive member identities')
    scenarios, partitions, ap_by_mac = {}, {}, {}
    physical_ids, used_ap_ids = set(), set()
    for scenario in SCENARIOS:
        aps, locations, seen_xyz = [], [], set()
        for row in _table(archive, f'raw/aps/{scenario}.csv', AP_HEADER, member_pins):
            integer_id = _integer(row['id'], 'logical AP id', 1)
            ap_id = f'AP{integer_id}'
            mac = row['mac'].lower()
            if not MAC.fullmatch(mac) or ap_id in used_ap_ids or (scenario, mac) in ap_by_mac:
                raise ValueError('Duplicate or invalid logical AP identity')
            used_ap_ids.add(ap_id)
            physical_id = 'uv-physical-' + _digest(NAMESPACE + ':physical:' + mac)[:24]
            physical_ids.add(physical_id)
            ap = {'id': ap_id, 'xyz_provider': _xyz(row),
                  'orientation': _number(row['orientation'], 'AP orientation'),
                  'physical_ap_id': physical_id}
            aps.append(ap)
            ap_by_mac[scenario, mac] = ap_id
        seen_ids = set()
        for row in _table(archive, f'raw/locations/{scenario}.csv', LOCATION_HEADER, member_pins):
            identifier, kind = row['id'], row['type']
            match = re.fullmatch(scenario + r'P([01])[0-9]{2}', identifier)
            xyz = _xyz(row)
            if (not match or identifier in seen_ids or tuple(xyz) in seen_xyz
                    or kind != ('training' if match[1] == '0' else 'test')):
                raise ValueError('Duplicate, overlapping or invalid reference location identity')
            seen_ids.add(identifier)
            seen_xyz.add(tuple(xyz))
            locations.append({'id': identifier, 'xyz_provider': xyz,
                              'orientation': _number(row['orientation'], 'receiver orientation'),
                              'provider_partition': kind})
        training = [r for r in locations if r['provider_partition'] == 'training']
        validation = set(r['id'] for r in sorted(training, key=lambda r: (
            _digest(f'{NAMESPACE}:validation:{scenario}:{r["id"]}'), r['id']))[:math.ceil(.2 * len(training))])
        for location in locations:
            location['partition'] = ('test' if location['provider_partition'] == 'test' else
                                     'validation' if location['id'] in validation else 'train')
            partitions[location['id']] = location['partition']
        all_xyz = [r['xyz_provider'] for r in aps + locations]
        minimum = [min(xyz[d] for xyz in all_xyz) for d in range(3)]
        maximum = [max(xyz[d] for xyz in all_xyz) for d in range(3)]
        origin = [(minimum[0] + maximum[0]) / 2, (minimum[1] + maximum[1]) / 2, minimum[2]]
        dimensions = [max(maximum[d] - minimum[d], lower) for d, lower in enumerate((1., 1., 3.))]
        for item in aps + locations:
            item['xyz_model'] = [value - offset for value, offset in zip(item['xyz_provider'], origin)]
        scenarios[scenario] = {
            'aps': sorted(aps, key=lambda r: int(r['id'][2:])),
            'locations': sorted(locations, key=lambda r: r['id']),
            'frame': {'origin_provider_xyz': origin, 'dimensions_m': dimensions,
                      'semantics': 'Computational catalogue frame; not a measured floor'},
        }
    members = []
    for member in sorted(names):
        match = MEMBER.fullmatch(member)
        if not match:
            if member.startswith('raw/') and member.endswith('.csv') and member.count('/') == 1:
                raise ValueError('Unknown raw measurement member name: ' + member)
            continue
        scenario, phone, point, session = (match[k] for k in ('scenario', 'phone', 'point', 'session'))
        location_id = f'{scenario}P{point}'
        if location_id not in partitions or (point[0] == '0' and session != '01'):
            raise ValueError('Raw member has missing location or invalid training session')
        members.append({'source_member': member, 'scenario': scenario, 'phone': phone,
                        'session': session, 'location_id': location_id,
                        'partition': partitions[location_id]})
    orders = {scenario: _rank_locations(scenario, value['locations'])
              for scenario, value in scenarios.items()}
    metadata = {
        'namespace': NAMESPACE,
        'source': {'url': SOURCE_URL, 'sha256': source_sha256, 'license': SOURCE_LICENSE,
                   'metadata_member_sha256': dict(member_pins)},
        'scenarios': scenarios, 'partitions': partitions, 'raw_members': members,
        'support_location_ids': {scenario: {str(budget): order[:budget] for budget in (4, 8, 16)}
                                 for scenario, order in orders.items()},
        'support_location_order': orders, 'physical_ap_count': len(physical_ids),
        'logical_ap_count': len(used_ap_ids),
        'summary_definition': 'Arithmetic mean of at most eight identity-selected reported RSSI dBm readings',
    }
    return metadata, ap_by_mac


def inspect_metadata(archive_path, *, expected_sha256=SOURCE_SHA256, metadata_member_sha256=None):
    """Inspect public coordinates/identities without opening raw measurement CSVs.

    Both explicit hash overrides are required for nonofficial unit archives.
    The official source always requires the six pinned provider catalogues.
    """
    path = _verified_archive(archive_path, expected_sha256)
    pins = _pins(expected_sha256, metadata_member_sha256)
    with zipfile.ZipFile(path) as archive:
        return _metadata(archive, expected_sha256, pins)[0]


def _protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError('An explicit registered protocol is required')
    if (protocol.get('namespace') != NAMESPACE or protocol.get('validation_fraction') != .2
            or protocol.get('support_budgets') != [4, 8, 16]
            or type(protocol.get('rows_per_group')) is not int or protocol['rows_per_group'] != 8):
        raise ValueError('Protocol differs from the registered metadata split/summary choices')
    source = protocol.get('source_sha256')
    return source, _pins(source, protocol.get('metadata_member_sha256'))


def prepare_dataset(archive_path, protocol):
    """Explicitly prepare registered observed-RSSI summaries after registration."""
    source_sha256, pins = _protocol(protocol)
    path = _verified_archive(archive_path, source_sha256)
    audit = {name: 0 for name in (
        'source_files', 'source_rows', 'failed_status_rows', 'zero_success_rows',
        'successful_rows', 'invalid_rssi_rows', 'sentinel_rssi_rows', 'frequency_unknown_rows',
        'frequency_zero_rows', 'frequency_missing_rows', 'frequency_other_rows',
        'eligible_rows', 'eligible_files', 'eligible_groups', 'selected_rows', 'summary_records')}
    groups = {}
    with zipfile.ZipFile(path) as archive:
        metadata, ap_by_mac = _metadata(archive, source_sha256, pins)
        locations = {row['id']: row for scenario in metadata['scenarios'].values()
                     for row in scenario['locations']}
        for member_info in metadata['raw_members']:
            member, scenario = member_info['source_member'], member_info['scenario']
            payload = archive.read(member)
            member_sha256 = hashlib.sha256(payload).hexdigest()
            reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
            if tuple(reader.fieldnames or ()) != RAW_HEADER:
                raise ValueError('Raw measurement schema differs: ' + member)
            audit['source_files'] += 1
            had_eligible = False
            for row in reader:
                audit['source_rows'] += 1
                if None in row:
                    raise ValueError('Unexpected extra raw CSV fields: ' + member)
                frequency = row.get('Frequency (MHz)')
                if frequency in (None, ''):
                    audit['frequency_missing_rows'] += 1
                elif frequency == UNSUPPORTED:
                    audit['frequency_unknown_rows'] += 1
                elif frequency == '0':
                    audit['frequency_zero_rows'] += 1
                elif frequency != '5180':
                    _integer(frequency, 'carrier frequency')
                    audit['frequency_other_rows'] += 1
                status = _integer(row.get('Status'), 'status')
                if status != 0:
                    audit['failed_status_rows'] += 1
                    continue  # Provider failure rows legitimately omit trailing fields.
                attempted = _integer(row.get('Attempted Measurements'), 'attempted measurements')
                successful = _integer(row.get('Successful Measurements'), 'successful measurements')
                if successful > attempted:
                    raise ValueError('Successful measurement count exceeds attempts')
                if successful == 0:
                    audit['zero_success_rows'] += 1
                    continue
                audit['successful_rows'] += 1
                try:
                    rssi = _number(row.get('Average RSSI (dbM)'), 'reported RSSI')
                except ValueError:
                    audit['invalid_rssi_rows'] += 1
                    continue
                if rssi == -200:
                    audit['sentinel_rssi_rows'] += 1
                    continue
                if frequency != '5180':
                    continue
                start = _integer(row.get('Request Start Unix Timestamp'), 'request start clock')
                end = _integer(row.get('Request Completed Unix Timestamp'), 'completion clock')
                if not 946684800000 <= start <= end <= 4102444800000:
                    raise ValueError('Invalid UNIX millisecond clocks')
                boot = _integer(row.get('Time Since System Boot (ms)'), 'boot clock')
                android = _integer(row.get('Android Version'), 'Android version', 1)
                if row.get('Device') != PHONES[member_info['phone']]:
                    raise ValueError('Raw device differs from source member phone')
                ap_id = ap_by_mac.get((scenario, (row.get('AP MAC Address') or '').lower()))
                if ap_id is None:
                    raise ValueError('Raw result has no scenario-specific logical AP placement')
                identity = _digest(f'{source_sha256}\0{member}\0{reader.line_num}')
                selected_row = {
                    'row': reader.line_num, 'record_id': identity, 'rssi_dbm': rssi,
                    'request_start_unix_ms': start, 'completion_unix_ms': end,
                    'capture_boot_ms': boot, 'android_version': android,
                    'attempted_measurements': attempted, 'successful_measurements': successful,
                }
                key = member, ap_id
                if key not in groups:
                    groups[key] = {'member_info': member_info, 'ap_id': ap_id,
                                   'source_member_sha256': member_sha256,
                                   'selected': [], 'eligible_source_row_numbers': [],
                                   'android_versions': set()}
                group = groups[key]
                group['android_versions'].add(android)
                group['eligible_source_row_numbers'].append(reader.line_num)
                group['selected'].append(selected_row)
                group['selected'].sort(key=lambda r: r['record_id'])
                del group['selected'][8:]
                audit['eligible_rows'] += 1
                had_eligible = True
            audit['eligible_files'] += had_eligible
        records = []
        for (member, ap_id), group in sorted(groups.items()):
            info, selected = group['member_info'], group['selected']
            versions = group['android_versions']
            if len(versions) != 1:
                raise ValueError('A summary cannot mix receiver Android calibration groups')
            location = locations[info['location_id']]
            record = {
                'id': _digest(f'{NAMESPACE}\0{source_sha256}\0{member}\0{ap_id}'),
                **info, 'ap_id': ap_id,
                'xyz_provider': list(location['xyz_provider']),
                'xyz_model': list(location['xyz_model']), 'orientation': location['orientation'],
                'rssi_dbm': math.fsum(row['rssi_dbm'] for row in selected) / len(selected),
                'carrier_hz': 5.18e9, 'android_version': next(iter(versions)),
                'source_archive_sha256': source_sha256,
                'source_member_sha256': group['source_member_sha256'],
                'selected_source_rows': selected,
                'selected_count': len(selected),
                'eligible_count': len(group['eligible_source_row_numbers']),
                'eligible_source_row_numbers': group['eligible_source_row_numbers'],
                'is_raw_measurement': False,
                'summary_kind': 'arithmetic_mean_reported_rssi_dbm',
            }
            records.append(record)
            audit['selected_rows'] += len(selected)
    audit['eligible_groups'] = audit['summary_records'] = len(records)
    return {'metadata': metadata, 'records': records, 'audit': audit}


PUBLIC_QUERY_FIELDS = (
    'id', 'scenario', 'phone', 'session', 'location_id', 'partition', 'ap_id',
    'xyz_provider', 'xyz_model', 'orientation', 'carrier_hz', 'android_version',
    'source_archive_sha256', 'source_member', 'source_member_sha256',
)


def query_metadata(record):
    """Whitelist geometry/acquisition identity; never forward labels or counts."""
    if not isinstance(record, dict) or any(name not in record for name in PUBLIC_QUERY_FIELDS):
        raise ValueError('A complete summary record is required for public query metadata')
    return {name: copy.deepcopy(record[name]) for name in PUBLIC_QUERY_FIELDS}
