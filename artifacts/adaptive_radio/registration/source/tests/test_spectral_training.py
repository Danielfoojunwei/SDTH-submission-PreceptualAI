"""Training-identity software fixtures only; no RF data, RT solves or model fits.

Checkpoint fixtures contain metadata and an empty state dictionary. They never
enter a scientific output directory or represent a trained model.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from wirelessjev.spectral_scene import training


def history_fixture(count=60):
    values = [2.+epoch for epoch in range(count)]
    if count >= 3:
        values[1] = values[2] = 1.  # first equal minimum is the fixed choice.
    return [{'epoch': epoch, 'validation_mean_layout_rmse_db': value,
             'updates': [{'loss': .25, 'measurement_nll': .2,
                          'measurement_z_gradient': .1} for _ in range(8)]}
            for epoch, value in enumerate(values)]


@pytest.fixture
def software_checkpoint(tmp_path):
    identity = {'source_receipt_sha256': 'SOFTWARE-SOURCE-IDENTITY',
                'protocol_sha256': 'SOFTWARE-PROTOCOL-IDENTITY',
                'data_receipt_sha256': 'SOFTWARE-DATA-IDENTITY',
                'training_rt_receipt_sha256': 'SOFTWARE-TRAIN-IDENTITY',
                'validation_rt_receipt_sha256': 'SOFTWARE-VAL-IDENTITY'}
    settings = {'regularizer': 'sigreg', 'geometry_only': False}
    checkpoint = {'identity': deepcopy(identity), 'cell': 'joint_sigreg', 'seed': 0,
                  'settings': deepcopy(settings), 'selected_epoch': 1, 'state': {}}
    path = tmp_path/'metadata-only-not-a-trained-model.pt'
    torch.save(checkpoint, path)
    receipt = {'identity': deepcopy(identity), 'settings': deepcopy(settings),
               'checkpoint_sha256': training.c.digest(path), 'completed': True,
               'cell': 'joint_sigreg', 'seed': 0, 'run_id': 'joint_sigreg-seed0',
               'epochs': 60, 'updates': 480, 'history': history_fixture(),
               'selected_epoch': 1, 'validation_mean_layout_rmse_db': 1.}
    return path, identity, settings, checkpoint, receipt


def validate_fixture(fixture, receipt=None):
    path, identity, settings, _, original = fixture
    return training.validate_completed(original if receipt is None else receipt,
                                       identity, 'joint_sigreg', 0, settings, path)


def test_completed_metadata_fixture_obeys_first_minimum_and_exact_budget(software_checkpoint):
    saved = validate_fixture(software_checkpoint)
    assert saved['selected_epoch'] == 1 and saved['state'] == {}
    training.validate_history(history_fixture(), complete=True)
    training.validate_history(history_fixture(7))
    training.validate_history([])


@pytest.mark.parametrize('kind', ['wrong_epoch', 'too_many_epochs', 'missing_epoch',
                                 'too_few_updates', 'too_many_updates', 'nan_score',
                                 'infinite_update', 'nan_update', 'empty_update',
                                 'missing_loss', 'not_a_log_mapping'])
def test_invalid_training_records_cannot_certify_completed_budget(kind):
    records = history_fixture()
    if kind == 'wrong_epoch':
        records[9]['epoch'] = 10
    elif kind == 'too_many_epochs':
        records = history_fixture(61)
    elif kind == 'missing_epoch':
        records.pop(10)
    elif kind == 'too_few_updates':
        records[3]['updates'].pop()
    elif kind == 'too_many_updates':
        records[3]['updates'].append(deepcopy(records[3]['updates'][0]))
    elif kind == 'nan_score':
        records[3]['validation_mean_layout_rmse_db'] = float('nan')
    elif kind == 'infinite_update':
        records[3]['updates'][2]['loss'] = float('inf')
    elif kind == 'nan_update':
        records[3]['updates'][2]['measurement_z_gradient'] = float('nan')
    elif kind == 'empty_update':
        records[3]['updates'][2] = {}
    elif kind == 'missing_loss':
        del records[3]['updates'][2]['loss']
    elif kind == 'not_a_log_mapping':
        records[3]['updates'][2] = ['loss', .25]
    with pytest.raises(RuntimeError):
        training.validate_history(records, complete=True)


def test_incomplete_history_is_not_a_completed_sixty_epoch_fit():
    with pytest.raises(RuntimeError, match='length'):
        training.validate_history(history_fixture(59), complete=True)


@pytest.mark.parametrize(('key', 'changed'), [
    ('identity', {'source_receipt_sha256': 'OTHER-SOFTWARE-SOURCE'}),
    ('cell', 'matched_regression'),
    ('seed', 1),
    ('run_id', 'joint_sigreg-seed1'),
    ('settings', {'regularizer': 'vicreg', 'geometry_only': False}),
    ('checkpoint_sha256', 'CORRUPTED-CHECKPOINT-DIGEST'),
    ('completed', False),
    ('epochs', 59),
    ('updates', 479),
])
def test_completed_receipt_cannot_be_relabelled_across_source_cell_seed_or_budget(software_checkpoint, key, changed):
    receipt = deepcopy(software_checkpoint[-1]); receipt[key] = changed
    with pytest.raises(RuntimeError, match='identity'):
        validate_fixture(software_checkpoint, receipt)


@pytest.mark.parametrize('kind', ['later_equal_minimum', 'wrong_best_score', 'wrong_epoch_score'])
def test_checkpoint_selection_is_recomputed_from_history(software_checkpoint, kind):
    receipt = deepcopy(software_checkpoint[-1])
    if kind == 'later_equal_minimum':
        receipt['selected_epoch'] = 2
    elif kind == 'wrong_best_score':
        receipt['validation_mean_layout_rmse_db'] = 1.01
    else:
        receipt['history'][1]['validation_mean_layout_rmse_db'] = .9
    with pytest.raises(RuntimeError, match='selection'):
        validate_fixture(software_checkpoint, receipt)


@pytest.mark.parametrize(('key', 'changed'), [
    ('identity', {'source_receipt_sha256': 'OTHER-SOFTWARE-SOURCE'}),
    ('cell', 'matched_regression'),
    ('seed', 2),
    ('settings', {'regularizer': 'sigreg', 'geometry_only': True}),
    ('selected_epoch', 2),
])
def test_hash_valid_checkpoint_from_another_cell_or_seed_still_fails(software_checkpoint, key, changed):
    path, _, _, checkpoint, original = software_checkpoint
    checkpoint = deepcopy(checkpoint); checkpoint[key] = changed
    torch.save(checkpoint, path)
    receipt = deepcopy(original)
    # Update the digest deliberately: a consistent file checksum is insufficient
    # evidence that this is the requested independent cell/seed checkpoint.
    receipt['checkpoint_sha256'] = training.c.digest(path)
    with pytest.raises(RuntimeError, match='metadata'):
        validate_fixture(software_checkpoint, receipt)


def test_identity_binds_train_and_validation_raw_receipts(tmp_path, monkeypatch):
    files = ['registration/source.json', 'protocol.json', 'data/prepared-training.json',
             'rt_data/training.json', 'rt_data/validation.json']
    for name in files:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'software_fixture': name}))
    monkeypatch.setattr(training.c, 'OUT', tmp_path)
    monkeypatch.setattr(training.c, 'check_sources', lambda: None)
    before = training.identity()
    for name, key in [('rt_data/training.json', 'training_rt_receipt_sha256'),
                      ('rt_data/validation.json', 'validation_rt_receipt_sha256')]:
        path = tmp_path/name
        original = path.read_bytes(); path.write_bytes(original+b'\n')
        after = training.identity()
        assert after[key] != before[key]
        assert {k for k in after if after[k] != before[k]} == {key}
        path.write_bytes(original)


def context_fixture():
    context = torch.zeros(2, 8, 20)
    context[..., 11] = torch.tensor([0., 0., 0., 0., .7731899, .7731899, .7731899, .7731899])
    mask = torch.tensor([[True, False, True, True, True, True, False, True],
                         [False, True, True, False, True, False, True, True]])
    return context, mask


def test_dropout_cannot_remove_an_observed_band_or_restore_missing_sources():
    context, mask = context_fixture(); original = mask.clone()
    retained = training.dropout_context(context, mask, np.random.default_rng(12), probability=1.)
    assert retained.dtype == torch.bool and retained.shape == mask.shape
    assert not (retained & ~mask).any()
    torch.testing.assert_close(mask, original, rtol=0, atol=0)
    for row in range(len(mask)):
        for band in torch.unique(context[row, mask[row], 11]):
            assert retained[row, context[row, :, 11] == band].sum() == 1


def test_dropout_is_reproducible_and_zero_probability_preserves_observations():
    context, mask = context_fixture()
    a = training.dropout_context(context, mask, np.random.default_rng(13), probability=.5)
    b = training.dropout_context(context, mask, np.random.default_rng(13), probability=.5)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    all_observed = training.dropout_context(context, mask, np.random.default_rng(14), probability=0.)
    torch.testing.assert_close(all_observed, mask, rtol=0, atol=0)
