"""Evidence/resume contracts with analytic files and mocked study stages only.

These tests never open the measured archive, fit a model, or run RF inference.
"""
from copy import deepcopy
import io

import numpy as np
import pytest

from wirelessjev.measured_radio import study


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(study.c, 'ROOT', tmp_path)
    monkeypatch.setattr(study.c, 'OUT', tmp_path / 'artifacts' / 'analytic-integrity')
    study.c.write(study.c.OUT / 'protocol.json', {'analytic_software_fixture': True})
    receipt = {'protocol_sha256': study.c.digest(study.c.OUT / 'protocol.json'),
               'source_archive_sha256': study.c.SOURCE_SHA256,
               'source_metadata_sha256': study.c.METADATA_SHA256,
               'dataset_sha256': 'a' * 64}
    study.c.write(study.c.OUT / 'data/receipt.json', receipt)
    return receipt


def _replace(path, value):
    path.write_bytes(study.c.encoded(value))


def _group(partition='validation', budget=4):
    evidence = {'schema': 1, **study._binding(), 'partition': partition,
                'budget_locations': budget, 'methods': ['reference', 'radiojepa-seed0'],
                'queries': [{'id': 'analytic-Q1'}, {'id': 'analytic-Q2'}]}
    if partition == 'test':
        study.c.write(study.c.OUT / 'selection.json', {'analytic_selection': True})
        evidence['selection_sha256'] = study.c.digest(study.c.OUT / 'selection.json')
    arrays = {'target': np.array([-50., -60.]), 'mean': np.array([[-51., -59.], [-49., -61.]]),
              'std': np.full((2, 2), 2.), 'supported': np.ones((2, 2), dtype=bool),
              'std_known': np.ones((2, 2), dtype=bool)}
    study.save_group(evidence, arrays)
    return study.group_path(partition, budget), evidence, arrays


def _rewrite_arrays(stem, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    stem.with_suffix('.npz').write_bytes(buffer.getvalue())
    evidence = study.c.read(stem.with_suffix('.json'))
    evidence['arrays_sha256'] = study.c.digest(stem.with_suffix('.npz'))
    _replace(stem.with_suffix('.json'), evidence)


def test_prepared_receipt_is_bound_to_protocol_and_official_source(workspace):
    study._check_receipt(workspace)
    for key in ('protocol_sha256', 'source_archive_sha256', 'source_metadata_sha256'):
        altered = {**workspace, key: 'b' * 64}
        with pytest.raises(RuntimeError, match='another registration'):
            study._check_receipt(altered)


def test_evidence_replay_accepts_same_value_and_preserves_conflicting_existing_bytes(workspace):
    path = study.c.OUT / 'analytic.json'
    value = {'result': [1., 2.], 'scope': 'software-only'}
    assert study._verify_or_write(path, value) == value
    original = path.read_bytes()
    assert study._verify_or_write(path, deepcopy(value)) == value
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match='differs from deterministic replay'):
        study._verify_or_write(path, {**value, 'result': [1., 3.]})
    assert path.read_bytes() == original


@pytest.mark.parametrize('partition', ['validation', 'test'])
def test_registered_group_roundtrip_preserves_raw_axes_and_masks(workspace, partition):
    _, original, arrays = _group(partition)
    evidence, loaded = study.load_group(partition, 4)
    assert evidence['methods'] == original['methods']
    assert evidence['queries'] == original['queries']
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[key], value)


@pytest.mark.parametrize('change', ['protocol_sha256', 'dataset_sha256', 'data_receipt_sha256',
    'partition', 'budget_locations', 'duplicate_method', 'duplicate_query', 'path', 'hash'])
def test_group_resume_rejects_changed_identity_and_archive_redirects(workspace, change):
    stem, _, _ = _group()
    path = stem.with_suffix('.json')
    evidence = study.c.read(path)
    if change in {'protocol_sha256', 'dataset_sha256', 'data_receipt_sha256'}:
        evidence[change] = 'b' * 64
    elif change == 'partition':
        evidence[change] = 'test'
    elif change == 'budget_locations':
        evidence[change] = 16
    elif change == 'duplicate_method':
        evidence['methods'][1] = evidence['methods'][0]
    elif change == 'duplicate_query':
        evidence['queries'][1] = deepcopy(evidence['queries'][0])
    elif change == 'path':
        evidence['arrays_path'] = 'elsewhere/validation-locations4.npz'
    elif change == 'hash':
        evidence['arrays_sha256'] = 'c' * 64
    _replace(path, evidence)
    with pytest.raises(RuntimeError):
        study.load_group('validation', 4)


def test_test_archive_cannot_resume_after_validation_selection_changes(workspace):
    _group('test')
    _replace(study.c.OUT / 'selection.json', {'analytic_selection': 'changed'})
    with pytest.raises(RuntimeError, match='frozen validation selection'):
        study.load_group('test', 4)


@pytest.mark.parametrize('change', ['target_shape', 'prediction_shape', 'extra_array', 'mask_dtype',
    'nonfinite_target', 'nonfinite_mean', 'nonfinite_std', 'zero_std', 'negative_std',
    'known_std_without_support', 'method_coverage_mismatch'])
def test_group_resume_validates_values_after_authenticating_archive_bytes(workspace, change):
    stem, _, arrays = _group()
    if change == 'target_shape':
        arrays['target'] = arrays['target'][None]
    elif change == 'prediction_shape':
        arrays['mean'] = arrays['mean'][:, :1]
    elif change == 'extra_array':
        arrays['unregistered'] = np.zeros(1)
    elif change == 'mask_dtype':
        arrays['supported'] = arrays['supported'].astype('i1')
    elif change == 'nonfinite_target':
        arrays['target'][0] = np.nan
    elif change == 'nonfinite_mean':
        arrays['mean'][0, 0] = np.inf
    elif change == 'nonfinite_std':
        arrays['std'][0, 0] = np.nan
    elif change == 'zero_std':
        arrays['std'][0, 0] = 0.
    elif change == 'negative_std':
        arrays['std'][0, 0] = -1.
    elif change == 'known_std_without_support':
        arrays['supported'][:, 0] = False
    elif change == 'method_coverage_mismatch':
        arrays['supported'][0, 0] = False
        arrays['std_known'][0, 0] = False
    _rewrite_arrays(stem, arrays)
    with pytest.raises(RuntimeError, match='axes or known-value masks'):
        study.load_group('validation', 4)


def test_selection_replays_validation_and_refuses_overwriting_an_existing_choice(workspace, monkeypatch):
    monkeypatch.setattr(study.c, 'BUDGETS', (4,))
    configs = [{'id': 'reference', 'method': 'reference'}, {'id': 'distance', 'method': 'log_distance'}]
    monkeypatch.setattr(study, 'configuration_grid', lambda: configs)
    score = {'reference': 3., 'distance': 2., 'radiojepa': 4.}
    calls = []

    def validation_only(partition, budget):
        calls.append((partition, budget))
        assert (partition, budget) == ('validation', 4)
        return {'methods': [r['id'] for r in configs], 'arrays_sha256': 'd' * 64}, {}

    monkeypatch.setattr(study, 'load_group', validation_only)
    monkeypatch.setattr(study, 'metric', lambda evidence, arrays, method: score[method])
    initial = study.select()
    assert initial['budgets']['4']['best_numerical_id'] == 'distance'
    path = study.c.OUT / 'selection.json'
    original = path.read_bytes()
    assert study.select() == initial
    score['reference'] = 1.
    with pytest.raises(RuntimeError, match='differs from deterministic replay'):
        study.select()
    assert path.read_bytes() == original
    assert calls == [('validation', 4)] * 3


def test_resume_replays_exact_public_cohort_without_inference(workspace, monkeypatch):
    monkeypatch.setattr(study.c, 'SEEDS', (0,))
    episode = {'scenario': 'A', 'phone': 'P', 'planned_support_locations': ['T'],
        'actual_support_locations': ['T'], 'support_ids': ['analytic-support'],
        'public_queries': [{'id': 'analytic-Q'}], 'private_targets': [-51.],
        'prepared': {'statuses': [{'status': 'prediction'}]}}
    monkeypatch.setattr(study, 'episodes', lambda value, partition, budget: [deepcopy(episode)])
    evidence = {'partition': 'test', 'budget_locations': 4, 'queries': episode['public_queries'],
        'episodes': [{k: episode[k] for k in ('scenario', 'phone', 'planned_support_locations',
            'actual_support_locations', 'support_ids')}], 'methods': ['reference', 'radiojepa-seed0']}
    arrays = {'target': np.array([-51.]), 'supported': np.ones((2, 1), dtype=bool)}
    study.verify_cohort({}, evidence, arrays, [{'id': 'reference'}])
    for field in ('queries', 'episodes', 'methods', 'target', 'supported'):
        altered, changed = deepcopy(evidence), deepcopy(arrays)
        if field in {'queries', 'episodes', 'methods'}:
            altered[field] = []
        elif field == 'target':
            changed[field][0] -= 1.
        else:
            changed[field][0, 0] = False
        with pytest.raises(RuntimeError, match='exact prepared support/query cohort'):
            study.verify_cohort({}, altered, changed, [{'id': 'reference'}])


def test_run_recovers_after_report_written_before_seal_and_replays_identical_manifest(workspace, monkeypatch):
    events, crashes = [], [True]
    monkeypatch.setattr(study, 'guard', lambda: events.append('guard'))
    monkeypatch.setattr(study, 'prepare', lambda: events.append('prepare'))
    monkeypatch.setattr(study, 'dataset', lambda: {'analytic_only': True})
    monkeypatch.setattr(study, 'validation', lambda value: events.append('validation'))
    monkeypatch.setattr(study, 'select', lambda: {'analytic_selection': True})
    monkeypatch.setattr(study, 'testing', lambda value, selection: events.append('testing'))
    result = {'scope': 'analytic run control, no measured data or model calls'}
    monkeypatch.setattr(study, 'report', lambda value, selection:
        study._verify_or_write(study.c.OUT / 'report.json', result))
    real_seal = study.seal

    def interrupted_seal():
        if crashes:
            crashes.pop()
            raise RuntimeError('analytic interruption after report')
        return real_seal()

    monkeypatch.setattr(study, 'seal', interrupted_seal)
    with pytest.raises(RuntimeError, match='analytic interruption'):
        study.run()
    assert study.c.read(study.c.OUT / 'report.json') == result
    assert not (study.c.OUT / 'verification/manifest.json').exists()
    assert study.run() == result
    manifest = (study.c.OUT / 'verification/manifest.json').read_bytes()
    report = (study.c.OUT / 'report.json').read_bytes()
    assert study.run() == result
    assert (study.c.OUT / 'verification/manifest.json').read_bytes() == manifest
    assert (study.c.OUT / 'report.json').read_bytes() == report
    assert events.count('validation') == events.count('testing') == 3


def test_seal_replay_rejects_changed_evidence_without_overwriting_manifest(workspace, monkeypatch):
    monkeypatch.setattr(study, 'guard', lambda: None)
    study.c.write(study.c.OUT / 'report.json', {'analytic_result': 1.})
    study.seal()
    path = study.c.OUT / 'verification/manifest.json'
    original = path.read_bytes()
    _replace(study.c.OUT / 'report.json', {'analytic_result': 2.})
    with pytest.raises(RuntimeError, match='differs from deterministic replay'):
        study.seal()
    assert path.read_bytes() == original


@pytest.mark.parametrize('partition', ['validation', 'test'])
def test_unsealed_prediction_array_is_refused_before_repeating_inference(workspace, monkeypatch, partition):
    monkeypatch.setattr(study.c, 'BUDGETS', (4,))
    monkeypatch.setattr(study, 'configuration_grid', lambda: [{'id': 'reference', 'method': 'reference'}])
    stem = study.group_path(partition, 4)
    stem.parent.mkdir(parents=True)
    stem.with_suffix('.npz').write_bytes(b'analytic interrupted array write; not model output')
    original = stem.with_suffix('.npz').read_bytes()

    def forbidden_prediction(*args):
        pytest.fail('Unsealed evidence must be detected before repeating inference')

    monkeypatch.setattr(study, 'predictions', forbidden_prediction)
    with pytest.raises(RuntimeError, match='[Uu]nsealed'):
        if partition == 'validation':
            study.validation({})
        else:
            study.testing({}, {'budgets': {'4': {'family_best_ids': {'reference': 'reference'}}}})
    assert stem.with_suffix('.npz').read_bytes() == original
