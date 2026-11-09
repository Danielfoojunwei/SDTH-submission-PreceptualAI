"""Explicit small software fixtures, never benchmark measurements/results."""
import io
import zipfile
from dataclasses import fields

import numpy as np
import pytest

from wirelessjev.device_transfer import data as d
from wirelessjev.device_transfer import benchmark as b


def fixture_archive(tmp_path):
    path = tmp_path / 'explicit-unit-fixture.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(d.MEMBERS['train_rssi'], '-60,-70,100,-80\n-80,-50,-70,100\n-50,-60,100,-70\n-70,-40,-60,100\n')
        archive.writestr(d.MEMBERS['train_labels'], '0,0,0,0,0,1\n1,0,0,0,0,1\n0,0,0,0,0,2\n1,0,0,0,0,2\n')
        archive.writestr(d.MEMBERS['test_rssi'], '-55,-65,100,-75\n')
        archive.writestr(d.MEMBERS['test_labels'], 'PRIVATE,XYZ,POISON,0,0,2\n')
    return path, d.digest(path)


def test_training_reader_does_not_open_test_and_never_invents_weak_rssi(tmp_path):
    path, digest = fixture_archive(tmp_path)
    value = d.load_training(path, expected=digest, expected_links=4)
    assert value.fingerprints.rssi_dbm.shape == (4, 4)
    assert not value.fingerprints.heard[0, 2]
    assert value.fingerprints.rssi_dbm[0, 2] == 0
    with pytest.raises(ValueError):
        value.fingerprints.rssi_dbm.setflags(write=True)
    with pytest.raises(RuntimeError, match='SHA256'):
        d.load_training(path, expected='0'*64, expected_links=4)


def test_test_features_parse_only_phone_metadata_not_xyz(tmp_path, monkeypatch):
    path, digest = fixture_archive(tmp_path)
    monkeypatch.setattr(d, '_authorized', lambda: None)
    monkeypatch.setattr(b, 'require_fits_before_test', lambda *args: None)
    value = d.load_test_features(path, expected=digest, expected_links=4)
    assert value.device_ids.tolist() == [2]
    assert 'xyz_m' not in {field.name for field in fields(value)}
    monkeypatch.setattr(b, 'require_predictions_before_labels', lambda *args: None)
    with pytest.raises(ValueError):
        d.private_test_xyz(path, expected=digest)


def test_test_guard_runs_before_any_heldout_archive_read(tmp_path, monkeypatch):
    path, digest = fixture_archive(tmp_path)
    touched = []
    def forbidden(*args, **kwargs):
        raise RuntimeError('Unpublished source fixture')
    monkeypatch.setattr(b, 'check_sources', forbidden)
    monkeypatch.setattr(d, 'verify_archive', lambda *args: touched.append(True))
    with pytest.raises(RuntimeError, match='Unpublished'):
        d.load_test_features(path, expected=digest, expected_links=4)
    assert touched == []


def test_distinct_real_support_positions_nested_and_source_only(tmp_path):
    path, digest = fixture_archive(tmp_path)
    value = d.load_training(path, expected=digest, expected_links=4)
    inventory = d.support_inventory(value, 2)
    assert inventory['source_rows'] == [0, 1]
    assert inventory['eligible_support_positions'] == 2
    small = d.support_order(value, 2, 0, 1)
    large = d.support_order(value, 2, 0, 2)
    np.testing.assert_array_equal(small, large[:1])
    assert set(large) == {2, 3}
    with pytest.raises(ValueError):
        d.support_order(value, 2, 0, 3)


def test_private_xyz_requires_persisted_predictions_even_after_publication(tmp_path, monkeypatch):
    path, digest = fixture_archive(tmp_path)
    monkeypatch.setattr(d, '_authorized', lambda: None)
    def forbidden(*args):
        raise RuntimeError('Predictions not durably persisted')
    monkeypatch.setattr(b, 'require_predictions_before_labels', forbidden)
    with pytest.raises(RuntimeError, match='persisted'):
        d.private_test_xyz(path, expected=digest)


def test_exact_hash_data_root_relocation(tmp_path, monkeypatch):
    path, digest = fixture_archive(tmp_path)
    destination = tmp_path / d.DEFAULT_ARCHIVE.name
    path.rename(destination)
    monkeypatch.setenv('DEVICE_TRANSFER_DATA_ROOT', str(tmp_path))
    value = d.load_training(expected=digest, expected_links=4)
    assert value.fingerprints.heard.shape == (4,4)
