"""Integrity/phase fixtures only; no real benchmark is fitted or evaluated."""
import copy
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.device_transfer import benchmark as b, data as d


def query_fixture():
    return d._fingerprints([[-60,-70,-80]], [2], links=3)


def test_checksum_consistent_wrong_phone_fit_rejected_before_loading_arrays(monkeypatch):
    monkeypatch.setattr(b, 'source_hashes', lambda: {'fixture': 'sha'})
    monkeypatch.setattr(b.c, 'digest', lambda path: 'inventory')
    loaded = []
    monkeypatch.setattr(b, '_load_npz', lambda ref: loaded.append(ref))
    fit = {'device': 1, 'source_sha256': {'fixture':'sha'}, 'inventory_sha256':'inventory',
           'target_test_rf_opened':False, 'target_test_xyz_opened':False,
           'state':{'path':str((b.OUT/'fits/device-2.npz').relative_to(b.ROOT)), 'sha256':'consistent'}}
    with pytest.raises(RuntimeError, match='device/source'):
        b._validate_fit_receipt(fit, 2, None, {}, {})
    assert loaded == []


@pytest.mark.parametrize('mutation', ['device','source','rows','scope','fit','path'])
def test_cached_prediction_rebinds_phone_source_fit_and_rows(monkeypatch, mutation):
    query = query_fixture()
    p = {'support_seeds':[0], 'support_sizes':[0,1]}
    source = {'fixture':'sha'}
    monkeypatch.setattr(b, 'source_hashes', lambda: source)
    monkeypatch.setattr(b.c, 'digest', lambda path: 'fit-sha')
    arrays = {'test_rows':query.row0,'base_xyz':np.zeros((1,3)), 'ridge_xyz':np.zeros((1,3)), 'base_supported':np.ones(1,bool)}
    for name in ('support','global','ap'):
        arrays[f'{name}_s0_n1_xyz'] = np.zeros((1,3))
        arrays[f'{name}_s0_n1_supported'] = np.ones(1,bool)
    monkeypatch.setattr(b, '_load_npz', lambda ref: arrays)
    value = {'device':2, 'private_xyz_opened':False, 'source_sha256':source,
        'test_rows':query.row0.tolist(),'fit_receipt_sha256':'fit-sha',
        'predictions':{'path':str((b.OUT/'predictions/device-2.npz').relative_to(b.ROOT)), 'sha256':'consistent'}}
    assert b._validate_prediction_receipt(value, 2, {}, query, p) is arrays
    if mutation == 'device': value['device'] = 1
    if mutation == 'source': value['source_sha256'] = {'old':'source'}
    if mutation == 'rows': value['test_rows'] = [999]
    if mutation == 'scope': value['private_xyz_opened'] = True
    if mutation == 'fit': value['fit_receipt_sha256'] = 'other-fit'
    if mutation == 'path': value['predictions']['path'] = str((b.OUT/'predictions/device-1.npz').relative_to(b.ROOT))
    with pytest.raises(RuntimeError, match='binding differs'):
        b._validate_prediction_receipt(value, 2, {}, query, p)


def test_numeric_evidence_tamper_fails_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setattr(b.c, 'ROOT', tmp_path)
    reference = b._npz(tmp_path/'predictions.npz', {'test_rows':np.array([0])})
    assert b._load_npz(reference)['test_rows'].tolist() == [0]
    (tmp_path/'predictions.npz').write_bytes(b'not the saved numeric archive')
    with pytest.raises(RuntimeError, match='changed'):
        b._load_npz(reference)


def test_missing_case_cannot_become_a_complete_report():
    p = {'target_devices':[1], 'models':['source_knn'], 'support_sizes':[0], 'support_seeds':[0]}
    with pytest.raises(RuntimeError, match='incomplete'):
        b.report_from_cases([], p)
