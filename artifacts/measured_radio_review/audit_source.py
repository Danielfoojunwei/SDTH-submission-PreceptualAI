#!/usr/bin/env python3
"""Independent stdlib replay of the published UV data preparation contract.

Imports no study, model, reader, or training code. This verifies frozen reported
RSSI summaries against their original source; it never predicts or fits them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/measured_radio_counterfix'
SOURCE = Path('/workspace/upstream/scene-radio/rssi/downloads/20051265-UVIndoorLoc-RTT&RSSI.zip')
SOURCE_SHA = '0ff469b8bd9d58a7d37c77423fc40450916bf26bbe208010bfc2bdf075ccf9f5'
META_SHA = '36539227b9ede5413b974de1e7fb91be1e17df4b841936df208706018414c931'
PROTOCOL_SHA = '37c9520331486a173e948b583278ad160245076a534ab0939d67160b98f9fdb0'
SOURCE_COMMIT = '5faa3d4380eb0773772d407e58f06080e36e9fa7'
PINS = {
    'raw/aps/S01.csv': '2f24fe042f6c4c304e5186a63d5f259552baa4d0c0b964d04d6780f6c88c068d',
    'raw/aps/S02.csv': 'a06a32a891910e2e46d867a91e6981402825e11fa245cb71321c09df972d2c5d',
    'raw/aps/S03.csv': '944e24b38f1cd2fdfde40a06be2c00389e1ec26e3600adcd7999629137263eae',
    'raw/locations/S01.csv': '6e8676315cb1ec4781155922365fb4795230a25f45ad699b4fd3b2df9da38048',
    'raw/locations/S02.csv': '45560b1c6b1d6fc026e199fa0496b99c3058f23fc063bc3e543faab6c3c6e07a',
    'raw/locations/S03.csv': 'ab8b1fe6373fb21b83d8954a9de0a2f19ecb630a8be28feb2a467673c21807eb',
}
SCENARIOS = ('S01', 'S02', 'S03')
PHONES = {'GP10': 'GooglePixel 10', 'S24U': 'samsungSM-S928B', 'POCO': 'XiaomiM2102J20SG'}
RAW_HEADER = (
    'Request Start Unix Timestamp', 'Request Completed Unix Timestamp',
    'Time Since System Boot (ms)', 'Device', 'Android Version', 'AP MAC Address',
    'Status', 'Distance (mm)', 'Std Dev (mm)', 'Attempted Measurements',
    'Successful Measurements', 'Bandwidth', 'Frequency (MHz)', 'Average RSSI (dbM)',
)
MEMBER = re.compile(r'raw/(S0[123])_(POCO|GP10|S24U)_([01][0-9]{2})_(0[12])\.csv\Z')


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(archive, member, header):
    payload = archive.read(member)
    if member in PINS:
        require(hashlib.sha256(payload).hexdigest() == PINS[member], 'Coordinate member hash: ' + member)
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    require(tuple(reader.fieldnames or ()) == header, 'CSV schema: ' + member)
    return payload, reader


def catalogue(archive, metadata):
    """Derive the data-independent catalogue and verify every metadata field."""
    require(metadata['source'] == {
        'url': 'https://doi.org/10.5281/zenodo.20051265', 'sha256': SOURCE_SHA,
        'license': 'CC-BY-4.0', 'metadata_member_sha256': PINS}, 'Source metadata identity')
    ap_map, locations, partitions, physical = {}, {}, {}, set()
    for scenario in SCENARIOS:
        _, ap_reader = rows(archive, f'raw/aps/{scenario}.csv', ('id', 'mac', 'x', 'y', 'z', 'orientation'))
        _, location_reader = rows(archive, f'raw/locations/{scenario}.csv', ('id', 'x', 'y', 'z', 'orientation', 'type'))
        ap_raw, loc_raw = list(ap_reader), list(location_reader)
        sc = metadata['scenarios'][scenario]
        raw_points = [tuple(float(r[k]) for k in ('x', 'y', 'z')) for r in ap_raw + loc_raw]
        lo = [min(r[k] for r in raw_points) for k in range(3)]
        hi = [max(r[k] for r in raw_points) for k in range(3)]
        origin = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
        dimensions = [max(hi[k] - lo[k], minimum) for k, minimum in enumerate((1., 1., 3.))]
        require(sc['frame'] == {'origin_provider_xyz': origin, 'dimensions_m': dimensions,
                'semantics': 'Computational catalogue frame; not a measured floor'}, 'Coordinate frame: ' + scenario)
        official_train = [r['id'] for r in loc_raw if r['type'] == 'training']
        validation = set(sorted(official_train, key=lambda key: (
            h(f'uv-measured-v1:validation:{scenario}:{key}'), key))[:math.ceil(.2 * len(official_train))])
        expected_aps, expected_locations = [], []
        for r in ap_raw:
            ap_id, mac = 'AP' + str(int(r['id'])), r['mac'].lower()
            provider = [float(r[k]) for k in ('x', 'y', 'z')]
            physical_id = 'uv-physical-' + h('uv-measured-v1:physical:' + mac)[:24]
            physical.add(physical_id)
            require((scenario, mac) not in ap_map, 'Repeated AP source join')
            ap_map[scenario, mac] = ap_id
            expected_aps.append({'id': ap_id, 'xyz_provider': provider,
                'orientation': float(r['orientation']), 'physical_ap_id': physical_id,
                'xyz_model': [x - offset for x, offset in zip(provider, origin)]})
        require(sc['aps'] == sorted(expected_aps, key=lambda r: int(r['id'][2:])), 'AP catalogue: ' + scenario)
        for r in loc_raw:
            identifier, provider = r['id'], [float(r[k]) for k in ('x', 'y', 'z')]
            partition = 'test' if r['type'] == 'test' else 'validation' if identifier in validation else 'train'
            expected = {'id': identifier, 'xyz_provider': provider, 'orientation': float(r['orientation']),
                'provider_partition': r['type'], 'partition': partition,
                'xyz_model': [x - offset for x, offset in zip(provider, origin)]}
            locations[identifier] = expected
            partitions[identifier] = partition
            expected_locations.append(expected)
        require(sc['locations'] == sorted(expected_locations, key=lambda r: r['id']), 'Location catalogue: ' + scenario)
        # Recompute the complete deterministic order, without calling the reader.
        remaining = {r['id']: r['xyz_provider'] for r in expected_locations if r['partition'] == 'train'}
        ordered = []
        while remaining:
            distances = {key: (min(sum((a - b) ** 2 for a, b in zip(xyz, locations[chosen]['xyz_provider']))
                              for chosen in ordered) if ordered else 0.) for key, xyz in remaining.items()}
            choice = sorted(remaining, key=lambda key: (
                -distances[key], h(f'uv-measured-v1:support:{scenario}:{key}'), key))[0]
            ordered.append(choice)
            del remaining[choice]
        require(metadata['support_location_order'][scenario] == ordered, 'Full maximin order: ' + scenario)
        require(metadata['support_location_ids'][scenario] == {str(b): ordered[:b] for b in (4, 8, 16)},
                'Nested support budgets: ' + scenario)
    require(metadata['partitions'] == partitions, 'Whole-location partitions')
    require(metadata['physical_ap_count'] == len(physical) == 8, 'Physical AP reuse')
    require(metadata['logical_ap_count'] == len(ap_map) == 20, 'Logical AP placement count')
    members = []
    for name in sorted(archive.namelist()):
        match = MEMBER.fullmatch(name)
        if match:
            scenario, phone, point, session = match.groups()
            location_id = f'{scenario}P{point}'
            members.append({'source_member': name, 'scenario': scenario, 'phone': phone,
                'session': session, 'location_id': location_id, 'partition': partitions[location_id]})
    require(metadata['raw_members'] == members, 'Raw member provenance/partition/session catalogue')
    require('raw/S03_S24U_020_01.csv' not in archive.namelist(), 'Known missing capture unexpectedly replaced')
    require('S03P020' in metadata['support_location_ids']['S03']['16'], 'Known missing location removed from plan')
    return ap_map, locations, members


def audit(source, dataset_path):
    require(sha(source) == SOURCE_SHA, 'Pinned raw archive hash')
    protocol_path = OUT / 'protocol.json'
    require(sha(protocol_path) == PROTOCOL_SHA, 'Published protocol hash')
    protocol = read(protocol_path)
    receipt_path = OUT / 'data/receipt.json'
    require(receipt_path.is_file(), 'Prepared data receipt is not yet present')
    receipt = read(receipt_path)
    require(receipt['dataset_sha256'] == sha(dataset_path), 'Dataset receipt hash')
    require(receipt['protocol_sha256'] == PROTOCOL_SHA, 'Dataset/protocol binding')
    require(receipt['source_archive_sha256'] == SOURCE_SHA, 'Dataset/source binding')
    metadata_receipt = OUT / 'registration/zenodo-20051265.json'
    require(sha(metadata_receipt) == META_SHA, 'Official license metadata hash')
    require(read(metadata_receipt)['metadata']['license']['id'] == 'cc-by-4.0', 'Official source license')
    require(receipt['source_metadata_sha256'] == META_SHA, 'Dataset/license binding')
    require(sha(OUT / 'registration/catalogue.json') == protocol['catalogue_sha256'], 'Registered catalogue hash')
    publication = read(OUT / 'registration/git.json')
    require(publication['protocol_sha256'] == PROTOCOL_SHA, 'Publication/protocol binding')
    require(publication['source_commit'] == SOURCE_COMMIT, 'Published source commit')
    source_receipt_path = OUT / 'registration/source.json'
    require(sha(source_receipt_path) == protocol['source_receipt_sha256'], 'Registered source receipt hash')
    source_receipt = read(source_receipt_path)
    for name, expected_hash in source_receipt['files_sha256'].items():
        require(sha(ROOT / name) == expected_hash, 'Live registered source changed: ' + name)
        require(sha(OUT / 'registration/source' / name) == expected_hash, 'Frozen source capsule changed: ' + name)
    for name, expected_hash in publication['published_files_sha256'].items():
        require(sha(ROOT / name) == expected_hash, 'Published evidence file changed: ' + name)
    value = json.loads(gzip.decompress(dataset_path.read_bytes()))
    require(value['metadata'] == read(OUT / 'registration/catalogue.json'), 'Prepared/frozen catalogue equality')
    metadata, counts, groups = value['metadata'], Counter(), {}
    versions, known_phones = defaultdict(set), set()
    with zipfile.ZipFile(source) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate archive members')
        ap_map, locations, members = catalogue(archive, metadata)
        for info in members:
            member, scenario = info['source_member'], info['scenario']
            payload, reader = rows(archive, member, RAW_HEADER)
            member_sha = hashlib.sha256(payload).hexdigest()
            counts['source_files'] += 1
            file_eligible = False
            for row in reader:
                counts['source_rows'] += 1
                require(None not in row, 'Extra raw CSV fields: ' + member)
                freq = row.get('Frequency (MHz)')
                frequency_flag = ('frequency_missing_rows' if freq in (None, '') else
                    'frequency_unknown_rows' if freq == 'Requires Android >= 14' else
                    'frequency_zero_rows' if freq == '0' else
                    'frequency_other_rows' if freq != '5180' else None)
                if frequency_flag:
                    counts[frequency_flag] += 1
                if int(row['Status']) != 0:
                    counts['failed_status_rows'] += 1
                    continue
                attempts, success = int(row['Attempted Measurements']), int(row['Successful Measurements'])
                require(success >= 0 and attempts >= 0, 'Negative source measurement counts')
                counter_inconsistent = success > attempts
                counts['counter_inconsistent_rows'] += counter_inconsistent
                if success == 0:
                    counts['zero_success_rows'] += 1
                    continue
                counts['successful_rows'] += 1
                try:
                    signal = float(row['Average RSSI (dbM)'])
                except (ValueError, TypeError):
                    signal = float('nan')
                if not math.isfinite(signal):
                    counts['invalid_rssi_rows'] += 1
                    continue
                if signal == -200:
                    counts['sentinel_rssi_rows'] += 1
                    continue
                if freq != '5180':
                    continue
                require(row['Device'] == PHONES[info['phone']], 'Source device/member join')
                ap_id = ap_map[scenario, row['AP MAC Address'].lower()]
                start, end, boot = (int(row[k]) for k in ('Request Start Unix Timestamp',
                    'Request Completed Unix Timestamp', 'Time Since System Boot (ms)'))
                android = int(row['Android Version'])
                require(946684800000 <= start <= end <= 4102444800000 and boot >= 0 and android >= 1,
                        'Source time/device validity')
                source_row = {'row': reader.line_num,
                    'record_id': h(f'{SOURCE_SHA}\0{member}\0{reader.line_num}'), 'rssi_dbm': signal,
                    'request_start_unix_ms': start, 'completion_unix_ms': end, 'capture_boot_ms': boot,
                    'android_version': android, 'attempted_measurements': attempts,
                    'successful_measurements': success, 'counter_inconsistent': counter_inconsistent}
                key = member, ap_id
                group = groups.setdefault(key, {'selected': [], 'all_rows': [], 'versions': set(),
                    'counter_inconsistent_source_row_numbers': [],
                    'info': info, 'member_sha256': member_sha})
                group['selected'] = sorted(group['selected'] + [source_row],
                                            key=lambda r: r['record_id'])[:8]
                group['all_rows'].append(reader.line_num)
                if counter_inconsistent:
                    group['counter_inconsistent_source_row_numbers'].append(reader.line_num)
                group['versions'].add(android)
                versions[f'{scenario}/{info["phone"]}'].add(android)
                known_phones.add(info['phone'])
                counts['eligible_rows'] += 1
                counts['eligible_counter_inconsistent_rows'] += counter_inconsistent
                file_eligible = True
            counts['eligible_files'] += file_eligible
    require(known_phones == {'GP10', 'S24U'}, 'Actual known-carrier phone cohort')
    require(len(groups) == len(value['records']) == 4255, 'Complete summary group count')
    expected = []
    for (member, ap_id), group in sorted(groups.items()):
        require(len(group['versions']) == 1, 'Mixed Android versions in source summary group')
        info, selected = group['info'], group['selected']
        location = locations[info['location_id']]
        expected.append({'id': h(f'uv-measured-v1\0{SOURCE_SHA}\0{member}\0{ap_id}'), **info,
            'ap_id': ap_id, 'xyz_provider': location['xyz_provider'], 'xyz_model': location['xyz_model'],
            'orientation': location['orientation'],
            'rssi_dbm': math.fsum(r['rssi_dbm'] for r in selected) / len(selected),
            'carrier_hz': 5.18e9, 'android_version': next(iter(group['versions'])),
            'source_archive_sha256': SOURCE_SHA, 'source_member_sha256': group['member_sha256'],
            'selected_source_rows': selected, 'selected_count': len(selected),
            'eligible_count': len(group['all_rows']), 'eligible_source_row_numbers': group['all_rows'],
            'counter_inconsistent_source_row_numbers': group['counter_inconsistent_source_row_numbers'],
            'is_raw_measurement': False, 'summary_kind': 'arithmetic_mean_reported_rssi_dbm'})
        counts['selected_rows'] += len(selected)
    for index, (actual, reference) in enumerate(zip(value['records'], expected)):
        require(actual == reference, 'Summary differs from independent source replay at index ' + str(index))
    counts['eligible_groups'] = counts['summary_records'] = len(expected)
    require(set(value['audit']) == set(counts) | {
        'invalid_rssi_rows', 'sentinel_rssi_rows', 'frequency_other_rows'}, 'Unexpected audit counter schema')
    require(all(value['audit'][key] == counts[key] for key in value['audit']), 'Full source eligibility audit counters')
    require(all(counts[key] == expected_count for key, expected_count in protocol['expected_audit'].items()),
            'Registered expected source/selection counts')
    require(receipt['audit'] == value['audit'], 'Receipt/full audit equality')
    require(all(len(v) == 1 for v in versions.values()), 'Scenario/phone Android consistency across sessions')
    return {'schema': 1, 'status': 'passed', 'independent_stdlib_source_replay': True,
        'study_data_or_model_modules_imported': False, 'model_forward_or_fit_calls': 0,
        'protocol_sha256': PROTOCOL_SHA, 'published_source_commit': publication['source_commit'],
        'source_archive_sha256': SOURCE_SHA, 'official_license_metadata_sha256': META_SHA,
        'registered_source_files_verified': len(source_receipt['files_sha256']),
        'published_evidence_files_verified': len(publication['published_files_sha256']),
        'original_failed_attempt_sha256': sha(ROOT / 'artifacts/measured_radio/failure.json'),
        'source_license': 'CC-BY-4.0', 'dataset_sha256': sha(dataset_path),
        'data_receipt_sha256': sha(receipt_path), 'catalogue_sha256': protocol['catalogue_sha256'],
        'audit_script_sha256': sha(__file__), 'all_summary_records_match_raw_source': True,
        'all_selected_rows_are_smallest_eight_eligible_identities': True,
        'all_selected_source_values_status_carrier_clock_device_ap_and_hashes_match': True,
        'all_coordinate_joins_partitions_support_orders_and_frames_match': True,
        'audit': dict(value['audit']), 'actual_admitted_phones': sorted(known_phones),
        'android_versions_by_scenario_phone': {key: sorted(v) for key, v in sorted(versions.items())},
        'partition_locations': dict(Counter(metadata['partitions'].values())),
        'selected_summary_units_are_raw_measurements': False,
        'predictions_results_or_hyperparameters_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--dataset', type=Path, default=OUT / 'data/dataset.json.gz')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('audit_source.json'))
    arguments = parser.parse_args()
    if not arguments.dataset.is_file() or not (OUT / 'data/receipt.json').is_file():
        print(json.dumps({'status': 'not_ready', 'reason': 'Prepared dataset and receipt are required'}))
        return 2
    result = audit(arguments.source, arguments.dataset)
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if arguments.output.exists():
        require(arguments.output.read_bytes() == payload, 'Existing independent source audit differs; preserve it')
    else:
        with arguments.output.open('xb') as stream:
            stream.write(payload)
    print(json.dumps({'status': result['status'], 'summary_records': result['audit']['summary_records'],
                      'selected_rows': result['audit']['selected_rows'], 'output': str(arguments.output)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
