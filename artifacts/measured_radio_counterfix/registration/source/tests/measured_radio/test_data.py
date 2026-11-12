"""Tiny explicit unit archives only; no official measured labels are opened."""
import copy
import csv
import hashlib
import io
import math
import zipfile

import pytest

from wirelessjev.measured_radio import data as d


MAC = '02:00:00:00:00:01'
MEMBER = 'raw/S01_GP10_001_01.csv'


def csv_text(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def measured_row(**updates):
    row = dict(zip(d.RAW_HEADER, [
        '1770729860186', '1770729861078', '1044616896', 'GooglePixel 10', '16',
        MAC, '0', '-1200', '384', '8', '7', '2', '5180', '-60',
    ]))
    row.update(updates)
    return row


def make_archive(tmp_path, *, raw=None, edit_catalogue=None, name='fixture.zip'):
    payloads = {}
    for scenario, ap_id in zip(d.SCENARIOS, (1, 9, 15)):
        payloads[f'raw/aps/{scenario}.csv'] = csv_text(
            d.AP_HEADER, [[ap_id, MAC, -1, -2, -.5, 90]])
        locations = [[f'{scenario}P{i:03}', i - 1, 0, .8, 0, 'training'] for i in range(1, 9)]
        locations += [[f'{scenario}P10{i}', 7.5 + i, 1, -.8, 90, 'test'] for i in (1, 2)]
        payloads[f'raw/locations/{scenario}.csv'] = csv_text(d.LOCATION_HEADER, locations)
    if edit_catalogue:
        edit_catalogue(payloads)
    if raw is None:
        raw = {MEMBER: [measured_row(**{'Average RSSI (dbM)': str(-40 - i)}) for i in range(12)]}
    for member, rows in raw.items():
        payloads[member] = rows if isinstance(rows, str) else csv_text(
            d.RAW_HEADER, [[row.get(key, '') for key in d.RAW_HEADER] for row in rows])
    path = tmp_path / name
    with zipfile.ZipFile(path, 'w') as archive:
        for member, payload in payloads.items():
            archive.writestr(member, payload)
    pins = {member: hashlib.sha256(payloads[member].encode()).hexdigest()
            for member in d.METADATA_MEMBER_SHA256}
    protocol = {'namespace': d.NAMESPACE, 'validation_fraction': .2,
                'support_budgets': [4, 8, 16], 'rows_per_group': 8,
                'source_sha256': d.sha256(path), 'metadata_member_sha256': pins}
    return path, protocol


def inspect(path, protocol):
    return d.inspect_metadata(path, expected_sha256=protocol['source_sha256'],
                              metadata_member_sha256=protocol['metadata_member_sha256'])


def test_metadata_never_opens_measurement_payload_or_rssi(tmp_path):
    path, protocol = make_archive(tmp_path, raw={MEMBER: 'NOT A MEASUREMENT CSV'})
    metadata = inspect(path, protocol)
    assert len(metadata['scenarios']) == 3
    assert metadata['raw_members'][0]['location_id'] == 'S01P001'
    assert metadata['raw_members'][0]['session'] == '01'
    assert not any('rssi_dbm' in row for sc in metadata['scenarios'].values() for row in sc['locations'])
    with pytest.raises(ValueError, match='schema'):
        d.prepare_dataset(path, protocol)


def test_archive_and_individual_coordinate_pins_are_strict(tmp_path):
    path, protocol = make_archive(tmp_path)
    with pytest.raises(ValueError, match='SHA256'):
        d.inspect_metadata(path)
    with pytest.raises(ValueError, match='explicit metadata'):
        d.inspect_metadata(path, expected_sha256=protocol['source_sha256'])
    changed = copy.deepcopy(protocol)
    changed['metadata_member_sha256']['raw/aps/S01.csv'] = '0' * 64
    with pytest.raises(ValueError, match='member SHA256'):
        inspect(path, changed)
    changed['metadata_member_sha256'].pop('raw/aps/S02.csv')
    with pytest.raises(ValueError, match='six'):
        inspect(path, changed)


def test_coordinate_frame_preserves_signed_provider_data_and_rigid_distances(tmp_path):
    path, protocol = make_archive(tmp_path)
    sc = inspect(path, protocol)['scenarios']['S01']
    frame = sc['frame']
    assert frame['origin_provider_xyz'] == [4.25, -.5, -.8]
    assert frame['dimensions_m'] == [10.5, 3., 3.]
    assert 'not a measured floor' in frame['semantics']
    ap = sc['aps'][0]
    assert ap['xyz_provider'] == [-1., -2., -.5]
    for location in sc['locations']:
        provider = math.dist(ap['xyz_provider'], location['xyz_provider'])
        model = math.dist(ap['xyz_model'], location['xyz_model'])
        assert model == pytest.approx(provider)
        assert location['xyz_model'] == [a - b for a, b in zip(
            location['xyz_provider'], frame['origin_provider_xyz'])]


def test_split_uses_location_hash_and_keeps_all_official_test_locations(tmp_path):
    path, protocol = make_archive(tmp_path)
    metadata = inspect(path, protocol)
    for scenario in d.SCENARIOS:
        training_ids = [f'{scenario}P{i:03}' for i in range(1, 9)]
        expected = sorted(training_ids, key=lambda loc: hashlib.sha256(
            f'uv-measured-v1:validation:{scenario}:{loc}'.encode()).hexdigest())[:2]
        assert sorted(loc for loc in training_ids if metadata['partitions'][loc] == 'validation') == sorted(expected)
        assert all(metadata['partitions'][f'{scenario}P10{i}'] == 'test' for i in (1, 2))


def test_support_order_is_nested_farthest_point_metadata_order(tmp_path):
    path, protocol = make_archive(tmp_path)
    metadata = inspect(path, protocol)
    for scenario in d.SCENARIOS:
        locations = metadata['scenarios'][scenario]['locations']
        eligible = {r['id']: r['xyz_provider'] for r in locations if r['partition'] == 'train'}
        order = metadata['support_location_order'][scenario]
        digest = lambda key: hashlib.sha256(f'uv-measured-v1:support:{scenario}:{key}'.encode()).hexdigest()
        assert order[0] == min(eligible, key=digest)
        for rank, chosen in enumerate(order[1:], 1):
            remaining = set(eligible) - set(order[:rank])
            def criterion(identifier):
                nearest = min(math.dist(eligible[identifier], eligible[seen]) for seen in order[:rank])
                return -nearest, digest(identifier)
            assert chosen == min(remaining, key=criterion)
        assert metadata['support_location_ids'][scenario]['4'] == order[:4]
        assert metadata['support_location_ids'][scenario]['8'] == order[:8]
        assert metadata['support_location_ids'][scenario]['16'] == order[:16]


def test_maximin_ties_resolve_by_registered_metadata_hash():
    rows = [{'id': identifier, 'xyz_provider': xyz, 'partition': 'train'}
            for identifier, xyz in [('S01P001', [0., 0., 0.]),
                                    ('S01P002', [1., 0., 0.]),
                                    ('S01P003', [-1., 0., 0.])]]
    order = d._rank_locations('S01', rows)
    assert order == d._rank_locations('S01', list(reversed(rows)))
    assert set(order) == {row['id'] for row in rows}


def test_missing_phone_files_do_not_replace_registered_support_locations(tmp_path):
    first, p1 = make_archive(tmp_path, name='first.zip')
    second, p2 = make_archive(tmp_path, raw={}, name='missing.zip')
    m1, m2 = inspect(first, p1), inspect(second, p2)
    assert m1['support_location_ids'] == m2['support_location_ids']
    assert m1['partitions'] == m2['partitions']
    assert not m2['raw_members']


def test_identity_selection_is_exact_and_summary_is_arithmetic_db_mean(tmp_path):
    path, protocol = make_archive(tmp_path)
    prepared = d.prepare_dataset(path, protocol)
    assert len(prepared['records']) == 1
    record = prepared['records'][0]
    expected_rows = sorted(range(2, 14), key=lambda number: hashlib.sha256(
        f'{protocol["source_sha256"]}\0{MEMBER}\0{number}'.encode()).hexdigest())[:8]
    assert [row['row'] for row in record['selected_source_rows']] == expected_rows
    assert record['rssi_dbm'] == pytest.approx(sum(-40 - (number - 2) for number in expected_rows) / 8)
    assert record['selected_count'] == 8 and record['eligible_count'] == 12
    assert record['eligible_source_row_numbers'] == list(range(2, 14))
    assert not record['is_raw_measurement']
    assert record['summary_kind'] == 'arithmetic_mean_reported_rssi_dbm'
    assert record['carrier_hz'] == 5.18e9
    assert record['source_member_sha256'] == hashlib.sha256(
        zipfile.ZipFile(path).read(MEMBER)).hexdigest()
    assert prepared['audit']['selected_rows'] == 8
    assert prepared['audit']['eligible_rows'] == 12
    assert prepared['audit']['eligible_groups'] == 1
    assert d.prepare_dataset(path, protocol) == prepared


def test_exact_status_count_frequency_and_rssi_eligibility_masks(tmp_path):
    failure = measured_row(**{'Status': '1'})
    for key in d.RAW_HEADER[7:]:
        failure[key] = ''
    rows = [failure, measured_row(**{'Successful Measurements': '0'}),
            measured_row(**{'Frequency (MHz)': d.UNSUPPORTED}),
            measured_row(**{'Frequency (MHz)': '0'}),
            measured_row(**{'Frequency (MHz)': '5200'}),
            measured_row(**{'Average RSSI (dbM)': '-200'}),
            measured_row(**{'Average RSSI (dbM)': 'nan'}),
            measured_row(**{'Average RSSI (dbM)': 'inf'}), measured_row()]
    path, protocol = make_archive(tmp_path, raw={MEMBER: rows})
    prepared = d.prepare_dataset(path, protocol)
    assert len(prepared['records']) == 1
    audit = prepared['audit']
    assert audit['source_rows'] == 9 and audit['eligible_rows'] == 1
    assert audit['failed_status_rows'] == 1 and audit['zero_success_rows'] == 1
    assert audit['frequency_unknown_rows'] == audit['frequency_zero_rows'] == 1
    assert audit['frequency_missing_rows'] == audit['frequency_other_rows'] == 1
    assert audit['invalid_rssi_rows'] == 2 and audit['sentinel_rssi_rows'] == 1
    assert audit['successful_rows'] == 7


def test_all_phones_and_sessions_at_one_location_share_partition(tmp_path):
    raw = {
        'raw/S01_GP10_101_01.csv': [measured_row()],
        'raw/S01_GP10_101_02.csv': [measured_row()],
        'raw/S01_S24U_101_01.csv': [measured_row(Device='samsungSM-S928B')],
        'raw/S01_POCO_101_01.csv': [measured_row(Device='XiaomiM2102J20SG')],
    }
    path, protocol = make_archive(tmp_path, raw=raw)
    records = d.prepare_dataset(path, protocol)['records']
    assert len(records) == 4  # Eligibility is by carrier metadata, never hard-coded phone exclusion.
    assert {record['partition'] for record in records} == {'test'}
    assert {record['location_id'] for record in records} == {'S01P101'}
    assert {record['session'] for record in records} == {'01', '02'}


def test_synthetic_reader_query_whitelist_and_episode_admission_integrate(tmp_path):
    """Exercise the real adapters together, without loading or running a model."""
    from wirelessjev.measured_radio.model import prepare_episode

    raw = {f'raw/S01_GP10_{i:03}_01.csv': [measured_row()] for i in range(1, 9)}
    raw.update({f'raw/S01_GP10_101_{session}.csv': [measured_row()]
                for session in ('01', '02')})
    path, protocol = make_archive(tmp_path, raw=raw)
    prepared = d.prepare_dataset(path, protocol)
    support = [row for row in prepared['records'] if row['partition'] == 'train']
    queries = [d.query_metadata(row) for row in prepared['records'] if row['partition'] == 'test']
    assert {row['session'] for row in support} == {'01'}
    assert {row['session'] for row in queries} == {'01', '02'}
    episode = prepare_episode(support, queries, prepared['metadata'])
    assert episode['support_ids'] == [row['id'] for row in support]
    assert episode['query_ids'] == [row['id'] for row in queries]
    assert episode['query_indices'] == [0, 1]
    assert episode['view']['context'].shape == (6, 20)
    assert episode['assumptions']['query_labels_used'] is False


def test_public_query_whitelist_removes_all_label_and_selection_information(tmp_path):
    path, protocol = make_archive(tmp_path)
    record = d.prepare_dataset(path, protocol)['records'][0]
    record['future_secret_rssi'] = -42
    query = d.query_metadata(record)
    assert set(query) == set(d.PUBLIC_QUERY_FIELDS)
    assert not any('rssi' in key or 'count' in key or 'selected' in key for key in query)
    assert 'eligible_source_row_numbers' not in query
    query['xyz_model'][0] += 1
    assert query['xyz_model'] != record['xyz_model']
    del record['xyz_model']
    with pytest.raises(ValueError, match='complete'):
        d.query_metadata(record)


def test_one_summary_cannot_mix_android_calibration_groups(tmp_path):
    rows = [measured_row(), measured_row(**{'Android Version': '15'})]
    path, protocol = make_archive(tmp_path, raw={MEMBER: rows})
    with pytest.raises(ValueError, match='calibration groups'):
        d.prepare_dataset(path, protocol)


@pytest.mark.parametrize('updates,match', [
    ({'Successful Measurements': '-1'}, 'Out-of-domain'),
    ({'Status': 'not-a-status'}, 'Noninteger'),
    ({'Request Start Unix Timestamp': '1770729860'}, 'clocks'),
    ({'Request Completed Unix Timestamp': '1770729850000'}, 'clocks'),
    ({'Time Since System Boot (ms)': '-1'}, 'boot'),
    ({'AP MAC Address': '02:00:00:00:00:99'}, 'logical AP'),
    ({'Device': 'unrecognized receiver'}, 'phone'),
    ({'Frequency (MHz)': 'not-a-frequency'}, 'carrier'),
])
def test_relevant_corruption_fails_closed(tmp_path, updates, match):
    path, protocol = make_archive(tmp_path, raw={MEMBER: [measured_row(**updates)]})
    with pytest.raises(ValueError, match=match):
        d.prepare_dataset(path, protocol)


@pytest.mark.parametrize('attempted,successful', [(0, 1), (0, 7), (8, 9)])
def test_reported_counters_preserved_without_assuming_subset_relation(tmp_path, attempted, successful):
    path, protocol = make_archive(tmp_path, raw={MEMBER: [measured_row(**{
        'Attempted Measurements': str(attempted), 'Successful Measurements': str(successful)})]})
    result = d.prepare_dataset(path, protocol)
    assert result['audit']['counter_inconsistent_rows'] == 1
    assert result['audit']['eligible_counter_inconsistent_rows'] == 1
    record = result['records'][0]
    assert record['counter_inconsistent_source_row_numbers'] == [2]
    selected = record['selected_source_rows'][0]
    assert (selected['attempted_measurements'], selected['successful_measurements']) == (attempted, successful)
    assert selected['counter_inconsistent'] is True


@pytest.mark.parametrize('field,value', [
    ('namespace', 'different'), ('validation_fraction', .25),
    ('support_budgets', [4, 16]), ('rows_per_group', 7), ('rows_per_group', True),
])
def test_protocol_choices_cannot_silently_change(tmp_path, field, value):
    path, protocol = make_archive(tmp_path)
    protocol[field] = value
    with pytest.raises(ValueError, match='Protocol differs'):
        d.prepare_dataset(path, protocol)


def test_duplicate_archive_members_are_rejected(tmp_path):
    path, protocol = make_archive(tmp_path)
    with pytest.warns(UserWarning, match='Duplicate name'):
        with zipfile.ZipFile(path, 'a') as archive:
            archive.writestr(MEMBER, 'duplicate')
    protocol['source_sha256'] = d.sha256(path)
    with pytest.raises(ValueError, match='Duplicate archive'):
        inspect(path, protocol)


@pytest.mark.parametrize('bad_member', ['raw/S01_GP10_999_01.csv', 'raw/S01_GP10_001_02.csv',
                                      'raw/unknown_format.csv'])
def test_unknown_member_location_or_training_session_is_rejected(tmp_path, bad_member):
    path, protocol = make_archive(tmp_path, raw={bad_member: [measured_row()]})
    with pytest.raises(ValueError, match='member|location|session'):
        inspect(path, protocol)


def test_overlapping_location_coordinates_fail_before_split(tmp_path):
    def edit(payloads):
        payloads['raw/locations/S01.csv'] = payloads['raw/locations/S01.csv'].replace(
            'S01P102,9.5,1,-0.8', 'S01P102,8.5,1,-0.8')
    path, protocol = make_archive(tmp_path, edit_catalogue=edit)
    with pytest.raises(ValueError, match='overlapping'):
        inspect(path, protocol)


def test_public_metadata_pseudonymizes_reused_hardware_addresses(tmp_path):
    path, protocol = make_archive(tmp_path)
    metadata = inspect(path, protocol)
    assert metadata['physical_ap_count'] == 1 and metadata['logical_ap_count'] == 3
    assert len({sc['aps'][0]['physical_ap_id'] for sc in metadata['scenarios'].values()}) == 1
    assert MAC not in repr(metadata)
