"""Explicit tiny unit fixtures; no real-data corpus or experiment is produced."""
import csv
from dataclasses import fields
import io
import zipfile

import numpy as np
import pytest

from wirelessjev.radiojepa import rssi_data as d


MEMBER = 'raw/S01_POCO_066_01.csv'


def row(**updates):
    value = dict(zip(d.HEADER, ['1774514645409', '1774514645599', '1781991',
        'XiaomiM2102J20SG', '11', '24:e5:0f:37:db:88', '0', '-1200', '574',
        '8', '7', d.UNSUPPORTED, d.UNSUPPORTED, '-66']))
    value.update(updates)
    return value


def archive(tmp_path, rows=None, *, label='id,x,y,z\nS01P066,1,2,0.8\n', member=MEMBER):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=d.HEADER)
    writer.writeheader()
    writer.writerows([row()] if rows is None else rows)
    path = tmp_path / 'explicit-unit-fixture.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(member, stream.getvalue())
        z.writestr('00_locations.csv', label)
    return path, d.sha256(path)


def test_same_result_signed_range_and_unknown_frequency_preserved(tmp_path):
    path, digest = archive(tmp_path)
    summary = d.FilterSummary()
    values = list(d.iter_public_records(path, expected_sha256=digest, summary=summary))
    record = values[0]
    assert record.rssi_dbm == -66 and record.measured_range_m == -1.2
    assert record.reported_std_m == .574
    assert record.negative_measured_range and not record.frequency_known
    assert record.frequency_hz == 0 and not record.bandwidth_known
    assert record.request_start_unix_ms == 1774514645409
    assert record.capture_boot_ms == 1781991  # Not subtracted from the UNIX clock.
    assert summary.accepted == 1 and summary.negative_measured_ranges == 1
    assert record.key.source_row == 2 and record.partition == 'training'


def test_declared_status_and_zero_success_filter_only(tmp_path):
    failure = {name: '' for name in d.HEADER}
    failure['Status'] = '1'
    path, digest = archive(tmp_path, [failure, row(**{'Successful Measurements': '0'}),
                                    row(**{'Distance (mm)': '-255703'})])
    summary = d.FilterSummary()
    values = list(d.iter_public_records(path, expected_sha256=digest, summary=summary))
    assert len(values) == 1 and values[0].measured_range_m == -255.703
    assert values[0].large_absolute_measured_range
    assert (summary.source_rows, summary.excluded_status, summary.excluded_zero_success) == (3, 1, 1)


@pytest.mark.parametrize('updates', [
    {'Request Completed Unix Timestamp': '1774514645000'},
    {'Request Start Unix Timestamp': '1774514645'},
    {'Time Since System Boot (ms)': '-1'},
    {'Distance (mm)': 'nan'}, {'Std Dev (mm)': '-1'},
    {'Successful Measurements': '9'}, {'AP MAC Address': 'not-an-ap'},
    {'Frequency (MHz)': '5180000000'},
])
def test_corrupt_schema_numeric_or_clock_values_fail_closed(tmp_path, updates):
    path, digest = archive(tmp_path, [row(**updates)])
    with pytest.raises(ValueError):
        list(d.iter_public_records(path, expected_sha256=digest))


def test_batch_views_disjoint_label_free_and_truly_readonly(tmp_path):
    path, digest = archive(tmp_path, [row(), row(**{'Distance (mm)': '3400'})],
                           label='DELIBERATELY MALFORMED PRIVATE LABELS')
    batches = list(d.iter_public_batches(path, expected_sha256=digest, batch_size=1))
    assert len(batches) == 2
    batch = batches[0]
    np.testing.assert_array_equal(batch.statuses, [0])
    rssi, ftm = batch.view('rssi'), batch.view('ftm')
    assert rssi.values.shape == (1, 1) and ftm.values.shape == (1, 2)
    assert not set(rssi.value_names) & set(ftm.value_names)
    assert set(rssi.value_names) == {'rssi_dbm'}
    assert not any('xyz' in field.name or 'location' in field.name for field in fields(batch))
    with pytest.raises(ValueError):
        batch.measured_range_m.setflags(write=True)
    with pytest.raises(ValueError):
        rssi.values[0, 0] = 1
    with pytest.raises(ValueError):
        d.private_reference_labels(path, batch.keys, expected_sha256=digest)


def test_known_frequency_and_android_code_are_not_confused_with_bandwidth_hz(tmp_path):
    path, digest = archive(tmp_path, [row(**{'Frequency (MHz)': '5180', 'Bandwidth': '2'})])
    batch = next(d.iter_public_batches(path, expected_sha256=digest))
    np.testing.assert_array_equal(batch.frequency_hz, [5.18e9])
    assert batch.frequency_known[0] and batch.bandwidth_codes[0] == 2
    path, digest = archive(tmp_path, [row(**{'Frequency (MHz)': '0', 'Bandwidth': '-1'})])
    record = next(d.iter_public_records(path, expected_sha256=digest))
    assert not record.frequency_known and not record.bandwidth_known


def test_private_label_boundary_and_measured_row_membership(tmp_path):
    path, digest = archive(tmp_path)
    record = next(d.iter_public_records(path, expected_sha256=digest))
    labels = d.private_reference_labels(path, [record.key], expected_sha256=digest)
    np.testing.assert_array_equal(labels.xyz_m, [[1, 2, .8]])
    assert labels.keys == (record.key,) and labels.reference_ids == ('S01P066',)
    forged = d.SourceKey(digest, MEMBER, 99, d._identity(digest, MEMBER, 99))
    with pytest.raises(ValueError, match='missing'):
        d.private_reference_labels(path, [forged], expected_sha256=digest)


def test_archive_identity_and_member_selection_checked(tmp_path):
    path, digest = archive(tmp_path)
    with pytest.raises(ValueError, match='SHA256'):
        next(d.iter_public_records(path, expected_sha256='0' * 64))
    with pytest.raises(ValueError, match='member'):
        next(d.iter_public_records(path, expected_sha256=digest, members=['00_locations.csv']))
    with pytest.raises(ValueError):
        list(d.iter_public_batches(path, expected_sha256=digest, batch_size=True))
    records = list(d.iter_public_records(path, expected_sha256=digest))
    repeated = list(d.iter_public_records(path, expected_sha256=digest))
    assert records == repeated
