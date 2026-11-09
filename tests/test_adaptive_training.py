"""Training-identity fixtures only: no registered data, fitting or RF solve.

Temporary checkpoints contain an explicitly named software-fixture tensor,
never learned model weights. Resume tests stop before any optimization update.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wirelessjev.adaptive_radio import training


def history_fixture(count=60):
    scores = [2. + epoch for epoch in range(count)]
    if count >= 3:
        scores[1] = scores[2] = 1.
    return [{'epoch': epoch, 'validation_mean_layout_delta_rmse_db': score,
             'updates': [{'loss': .25, 'measurement_nll': .2,
                          'measurement_mean_gradient': .1} for _ in range(8)]}
            for epoch, score in enumerate(scores)]


def fixture_identity():
    return {'source_receipt_sha256': '1' * 64, 'protocol_sha256': '2' * 64,
            'data_receipt_sha256': '3' * 64,
            'rt_stage_receipts_sha256': {'training': '4' * 64, 'validation': '5' * 64}}


@pytest.fixture
def software_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(training.c, 'OUT', tmp_path / 'software-only-output')
    path = training.c.OUT / 'checkpoints/joint-seed0.pt'
    path.parent.mkdir(parents=True)
    identity = fixture_identity()
    settings = {'cell': 'joint', 'width': 64, 'em_steps': 3}
    checkpoint = {'identity': deepcopy(identity), 'cell': 'joint', 'seed': 0,
                  'settings': deepcopy(settings), 'selected_epoch': 1,
                  'state': {'not_trained_software_fixture': torch.tensor([1.])}}
    torch.save(checkpoint, path)
    receipt = {'identity': deepcopy(identity), 'settings': deepcopy(settings),
               'checkpoint_sha256': training.c.digest(path), 'completed': True,
               'cell': 'joint', 'seed': 0, 'run_id': 'joint-seed0',
               'epochs': 60, 'updates': 480, 'history': history_fixture(),
               'selected_epoch': 1, 'validation_mean_layout_delta_rmse_db': 1.}
    return path, identity, checkpoint, receipt


def verify_fixture(fixture, receipt=None):
    _, identity, _, original = fixture
    return training.verify_completed(original if receipt is None else receipt,
                                     identity, 'joint', 0)


def test_completed_software_metadata_has_first_minimum_and_exact_finite_budget(software_checkpoint):
    value = verify_fixture(software_checkpoint)
    assert value['selected_epoch'] == 1
    assert set(value['state']) == {'not_trained_software_fixture'}
    training.validate_history(history_fixture(), complete=True)
    training.validate_history(history_fixture(7))
    training.validate_history([])


@pytest.mark.parametrize('kind', (
    'wrong_epoch', 'float_epoch', 'bool_epoch', 'too_many_epochs', 'missing_epoch',
    'too_few_updates', 'too_many_updates', 'nan_score', 'infinite_update',
    'empty_update', 'missing_loss', 'nonmapping_update', 'nonlist_history',
))
def test_history_cannot_certify_a_different_or_nonfinite_budget(kind):
    records = history_fixture()
    if kind == 'wrong_epoch': records[9]['epoch'] = 10
    elif kind == 'float_epoch': records[9]['epoch'] = 9.
    elif kind == 'bool_epoch': records[1]['epoch'] = True
    elif kind == 'too_many_epochs': records = history_fixture(61)
    elif kind == 'missing_epoch': records.pop(10)
    elif kind == 'too_few_updates': records[3]['updates'].pop()
    elif kind == 'too_many_updates': records[3]['updates'].append({'loss': .1})
    elif kind == 'nan_score': records[3]['validation_mean_layout_delta_rmse_db'] = np.nan
    elif kind == 'infinite_update': records[3]['updates'][2]['loss'] = np.inf
    elif kind == 'empty_update': records[3]['updates'][2] = {}
    elif kind == 'missing_loss': del records[3]['updates'][2]['loss']
    elif kind == 'nonmapping_update': records[3]['updates'][2] = ['loss', .25]
    else: records = tuple(records)
    with pytest.raises(RuntimeError):
        training.validate_history(records, complete=True)


@pytest.mark.parametrize(('key', 'changed'), (
    ('identity', {'source_receipt_sha256': 'different-software-source'}),
    ('cell', 'regression'), ('seed', 1), ('seed', False), ('seed', 0.),
    ('run_id', 'joint-seed1'), ('completed', False), ('completed', 1),
    ('epochs', 59), ('epochs', 60.), ('updates', 479), ('updates', 480.),
    ('settings', {'cell': 'joint', 'width': 128, 'em_steps': 3}),
    ('checkpoint_sha256', 'corrupt-checksum'), ('selected_epoch', True),
    ('selected_epoch', 1.),
))
def test_completed_receipt_cannot_relabel_source_cell_seed_settings_or_budget(software_checkpoint, key, changed):
    receipt = deepcopy(software_checkpoint[-1])
    receipt[key] = changed
    with pytest.raises(RuntimeError):
        verify_fixture(software_checkpoint, receipt)


@pytest.mark.parametrize('kind', ('later_tied_minimum', 'wrong_best_score', 'different_history_minimum'))
def test_completed_selection_is_recomputed_from_sparse_val_history(software_checkpoint, kind):
    receipt = deepcopy(software_checkpoint[-1])
    if kind == 'later_tied_minimum': receipt['selected_epoch'] = 2
    elif kind == 'wrong_best_score': receipt['validation_mean_layout_delta_rmse_db'] = 1.01
    else: receipt['history'][1]['validation_mean_layout_delta_rmse_db'] = .9
    with pytest.raises(RuntimeError, match='first-minimum'):
        verify_fixture(software_checkpoint, receipt)


@pytest.mark.parametrize(('key', 'changed'), (
    ('identity', {'source_receipt_sha256': 'different-software-source'}),
    ('cell', 'regression'), ('seed', 2), ('seed', False),
    ('settings', {'cell': 'joint', 'width': 64, 'em_steps': 4}),
    ('selected_epoch', 2), ('selected_epoch', True),
))
def test_checksum_valid_swapped_checkpoint_still_fails_metadata_binding(software_checkpoint, key, changed):
    path, _, original_checkpoint, original_receipt = software_checkpoint
    checkpoint = deepcopy(original_checkpoint)
    checkpoint[key] = changed
    torch.save(checkpoint, path)
    receipt = deepcopy(original_receipt)
    receipt['checkpoint_sha256'] = training.c.digest(path)
    with pytest.raises(RuntimeError, match='metadata'):
        verify_fixture(software_checkpoint, receipt)


@pytest.mark.parametrize('state', ({}, {'fixture': 'not-a-tensor'},
                                   {'fixture': torch.tensor([float('nan')])},
                                   {'fixture': torch.tensor([float('inf')])}))
def test_empty_nontensor_or_nonfinite_checkpoint_cannot_be_admitted(software_checkpoint, state):
    path, _, original_checkpoint, original_receipt = software_checkpoint
    checkpoint = deepcopy(original_checkpoint)
    checkpoint['state'] = state
    torch.save(checkpoint, path)
    receipt = deepcopy(original_receipt)
    receipt['checkpoint_sha256'] = training.c.digest(path)
    with pytest.raises(RuntimeError):
        verify_fixture(software_checkpoint, receipt)


def test_completed_cache_path_performs_metadata_checks_without_constructing_model(software_checkpoint, monkeypatch):
    _, identity, _, receipt = software_checkpoint
    training.c.write_once(training.c.OUT / 'fits/joint-seed0.json', receipt)
    monkeypatch.setattr(training, 'identity', lambda: deepcopy(identity))
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed metadata cache must not construct or fit a model')
    monkeypatch.setattr(training, 'AdaptiveScene', forbidden)
    monkeypatch.setattr(training.data, 'examples', forbidden)
    assert training.fit(('joint', 0)) == receipt


@pytest.mark.parametrize('seed', (True, False, 0., 1., -1, 3, '0'))
def test_fit_and_load_reject_noncanonical_seed_ids_before_source_or_data_access(monkeypatch, seed):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid member must be rejected before any source/data access')
    monkeypatch.setattr(training, 'identity', forbidden)
    monkeypatch.setattr(training.c, 'read_json', forbidden)
    with pytest.raises(ValueError, match='Unregistered'):
        training.fit(('joint', seed))
    with pytest.raises(ValueError, match='Unregistered'):
        training.load('joint', seed)


@pytest.fixture
def resume_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(training.c, 'OUT', tmp_path / 'software-only-output')
    monkeypatch.setattr(training.c, 'WORK', tmp_path / 'software-only-resume')
    identity = fixture_identity()
    monkeypatch.setattr(training, 'identity', lambda: deepcopy(identity))
    calls = []
    class SoftwareModel:
        config = {'cell': 'joint', 'width': 64, 'em_steps': 3}
        def __init__(self, cell):
            assert cell == 'joint'
        def parameters(self): return ()
        def load_state_dict(self, state): calls.append('model-state-read')
    class SoftwareOptimizer:
        def load_state_dict(self, state): calls.append('optimizer-state-read')
    monkeypatch.setattr(training, 'AdaptiveScene', SoftwareModel)
    monkeypatch.setattr(training.torch.optim, 'AdamW', lambda *a, **k: SoftwareOptimizer())
    monkeypatch.setattr(training.torch, 'set_rng_state', lambda state: calls.append('rng-state-read'))
    def fixture_cases(stage, *, regime):
        assert stage in ('training', 'validation'), 'No TEST API in resume fixture'
        count = 24 if stage == 'training' else 6
        return [SimpleNamespace(condition=k, layout_id=f'software-{stage}-{i}')
                for i in range(count) for k in (0, 1, 2)]
    monkeypatch.setattr(training.data, 'examples', fixture_cases)
    monkeypatch.setattr(training, 'batch', lambda examples: {})
    def no_update(*args, **kwargs): raise AssertionError('Resume negative fixture must never optimize')
    monkeypatch.setattr(training, 'objective', no_update)
    records = history_fixture(3)
    value = {'identity': deepcopy(identity), 'cell': 'joint', 'seed': 0,
             'run_id': 'joint-seed0', 'settings': deepcopy(SoftwareModel.config),
             'history': records, 'state': {'not_learned_fixture': torch.tensor([1.])},
             'optimizer': {}, 'torch_rng': torch.tensor([0], dtype=torch.uint8),
             'numpy_rng': np.random.default_rng(0).bit_generator.state,
             'best': 1., 'best_epoch': 1,
             'best_state': {'not_learned_fixture': torch.tensor([1.])}, 'elapsed_seconds': 0.}
    path = training.c.WORK / 'joint-seed0.last.pt'
    path.parent.mkdir(parents=True)
    return path, value, calls


@pytest.mark.parametrize(('key', 'changed'), (
    ('identity', {'source_receipt_sha256': 'other-software-source'}),
    ('cell', 'regression'), ('seed', 1), ('seed', False), ('seed', 0.),
    ('run_id', 'joint-seed1'),
    ('settings', {'cell': 'joint', 'width': 128, 'em_steps': 3}),
))
def test_resume_member_mismatch_is_rejected_before_loading_any_training_state(resume_fixture, key, changed):
    path, value, calls = resume_fixture
    value[key] = changed
    torch.save(value, path)
    with pytest.raises(RuntimeError, match='resume state'):
        training.fit(('joint', 0))
    assert calls == []


@pytest.mark.parametrize('kind', ('best_value', 'later_tied_epoch', 'malformed_history'))
def test_resume_cannot_change_val_selection_or_skip_optimization_budget(resume_fixture, kind):
    path, value, _ = resume_fixture
    if kind == 'best_value': value['best'] = .5
    elif kind == 'later_tied_epoch': value['best_epoch'] = 2
    else: value['history'][1]['updates'] = []
    torch.save(value, path)
    with pytest.raises(RuntimeError):
        training.fit(('joint', 0))


def test_identity_binds_training_validation_inputs_but_never_reads_test(tmp_path, monkeypatch):
    files = ('registration/source.json', 'protocol.json', 'data/prepared-training.json',
             'rt_data/training.json', 'rt_data/validation.json', 'rt_data/test.json')
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'software_fixture': name}))
    monkeypatch.setattr(training.c, 'OUT', tmp_path)
    monkeypatch.setattr(training.c, 'PROTOCOL', tmp_path / 'protocol.json')
    monkeypatch.setattr(training.c, 'check_sources', lambda: None)
    original_digest = training.c.digest
    opened = []
    def record_digest(path):
        opened.append(str(path))
        assert Path(path).name != 'test.json', 'TEST inputs cannot identify a fit'
        return original_digest(path)
    from pathlib import Path
    monkeypatch.setattr(training.c, 'digest', record_digest)
    before = training.identity()
    for stage in ('training', 'validation'):
        path = tmp_path / 'rt_data' / f'{stage}.json'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        after = training.identity()
        assert after['rt_stage_receipts_sha256'][stage] != before['rt_stage_receipts_sha256'][stage]
        other = 'validation' if stage == 'training' else 'training'
        assert after['rt_stage_receipts_sha256'][other] == before['rt_stage_receipts_sha256'][other]
        path.write_bytes(original)
    assert not any(p.endswith('/test.json') for p in opened)
