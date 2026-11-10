"""Production-shaped analytic software checks, never experimental RF evidence.

These deterministic toy fields exercise the real default-float32 neural model,
trainer tensor conversion and validation archive reducer. They do not collect
ray traces, enter a registered fit, or produce study artifacts.
"""
import io

import numpy as np
import pytest
import torch

from wirelessjev.hierarchical_radio import training as trainer
from wirelessjev.hierarchical_radio.model import CELLS, HierarchicalField, objective
from wirelessjev.spectral_scene.operators import (
    fspl_received_power_dbm, link_geometry_features, query_physics_basis,
)


def analytic_example(*, index=0, state=0, regime='full', stage='training'):
    """A public four-AP geometry and an explicitly analytic residual field."""
    sites = np.array(((-9., -7., 4.), (9., -7., 4.),
                      (-9., 7., 4.), (9., 7., 4.)), dtype='f4')
    sites[:, :2] += np.array((.04 * index, -.02 * index), dtype='f4')
    grid = np.stack(np.meshgrid(np.linspace(-7., 7., 6), np.linspace(-5., 5., 4)), -1).reshape(-1, 2)
    context_positions = np.column_stack((grid, np.full(24, 1.)))
    query_positions = context_positions + np.array((.19, .23, 0.))
    dimensions = np.array((20., 16., 5.))
    empty_boxes = np.empty((0, 2, 3))
    bands = np.array((2.4e9, 5.2e9))

    def links(positions, *, sparse=False, direction=0):
        pose, ap, band = np.indices((24, 4, 2)).reshape(3, -1)
        if sparse:
            keep = pose % 2 == 0
            pose, ap, band = pose[keep], ap[keep], band[keep]
        rx = positions[pose].astype('f4'); tx = sites[ap]
        q = link_geometry_features(rx, tx, bands[band], direction, dimensions,
            empty_boxes, calibration_pair_count=16, calibration_unknown=False,
            quantization_width_db=1., noise_floor_dbm=-100., tx_power_dbm=20.)
        fspl = fspl_received_power_dbm(rx, tx, bands[band], 20.).astype('f4')
        # A hand-written field, not a simulator label or trained prediction.
        residual = (-9. - .13 * rx[:, 0] + .08 * rx[:, 1]
                    + .03 * tx[:, 0] - .4 * state).astype('f4')
        # Alternate rank-censor slots retain finite real upper events. These
        # are software fixtures and do not assert a calibrated scan process.
        heard = (pose + ap + band) % 4 != 0
        return ap.astype('i8'), rx, q, fspl, residual, heard

    def view(direction):
        ap, rx, q, fspl, residual, heard = links(context_positions,
            sparse=regime == 'sparse', direction=direction)
        residual = residual + .2 * direction
        lower = np.where(heard, residual - .5, -np.inf).astype('f4')
        upper = (residual + .5).astype('f4')
        offset = np.full(len(ap), 2.5 * direction, dtype='f4')
        floor = (-90. - fspl - offset).astype('f4')
        context = np.column_stack((np.where(heard, residual / 30., 0.),
            np.where(~heard, upper / 30., 0.), heard, ~heard, q)).astype('f4')
        return {'context': context, 'known': np.ones(len(ap), bool),
            'scheduled': np.ones(len(ap), bool), 'ap': ap, 'rx_xyz': rx,
            'lower': lower, 'upper': upper, 'floor': floor,
            'fspl': fspl, 'offset': offset}

    ap, rx, q, fspl, residual, heard = links(query_positions)
    reported = fspl + residual
    query = {'q': q, 'ap': ap, 'rx_xyz': rx, 'basis': query_physics_basis(q),
             'fspl': fspl, 'offset': np.zeros(len(ap), dtype='f4')}
    target = {'center_dbm': np.where(heard, reported, 0.).astype('f4'),
        'lower_dbm': np.where(heard, reported - .5, 0.).astype('f4'),
        'upper_dbm': np.where(heard, reported + .5, 0.).astype('f4'),
        'heard': heard, 'censored': ~heard,
        'censor_upper_dbm': np.where(~heard, reported + .5, 0.).astype('f4'),
        'detection_threshold_dbm': np.full(len(ap), -90., dtype='f4')}
    case = f'analytic-{stage}-{index}-{state}'
    return {'a': view(0), 'b': view(1), 'query': query, 'target': target,
        'ap_xyz': sites, 'binding': {'stage': stage, 'index': index,
            'state': state, 'condition': state, 'layout_id': f'analytic-{stage}-{index}',
            'case_id': case, 'context_case_id': case, 'regime': regime,
            'held_band': False, 'stale_requested': False}}


@pytest.mark.parametrize('cell', CELLS)
def test_default_float32_model_optimizer_handles_full_and_sparse_production_shapes(cell):
    # Restrict this analytic smoke check to the same per-worker CPU threading
    # as scientific training, restoring the caller's thread budget afterwards.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(98117)
            model = HierarchicalField(cell)
        assert {p.dtype for p in model.parameters()} == {torch.float32}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1.e-4)
        before = {key: value.detach().clone() for key, value in model.named_parameters()}
        for regime, tokens in (('full', 192), ('sparse', 96)):
            examples = [analytic_example(index=i, state=i % 3, regime=regime) for i in range(4)]
            values = trainer.batch(examples)
            assert values['a']['context'].shape == values['b']['context'].shape == (4, tokens, 20)
            assert values['query']['q'].shape == (4, 192, 16)
            assert values['query']['ap'].dtype == torch.int64
            assert values['a']['known'].dtype == torch.bool
            optimizer.zero_grad(set_to_none=True)
            loss, details = objective(model, values['a'], values['b'],
                values['query'], values['target'], values['ap_xyz'])
            assert torch.isfinite(loss) and loss.ndim == 0
            assert all(value.ndim == 0 and torch.isfinite(value) for value in details.values())
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            assert torch.isfinite(norm)
            optimizer.step()
            assert all(torch.isfinite(value).all() for value in model.state_dict().values())
            assert all(torch.isfinite(value).all() for row in optimizer.state.values()
                       for value in row.values() if isinstance(value, torch.Tensor))
        assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    finally:
        torch.set_num_threads(previous_threads)


def test_real_float32_model_validation_arrays_survive_raw_archive_reduction(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        examples = [analytic_example(stage='validation', index=i, state=s, regime='sparse')
                    for i in range(6) for s in range(3)]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(98118)
            model = HierarchicalField('hier_joint')
        score, rows, mean, std = trainer.validation_score(model, examples, return_predictions=True)
        assert mean.shape == std.shape == (18, 192)
        assert model.training and np.isfinite(score)
        reconstructed, reconstructed_rows = trainer.reduce_validation_predictions(mean, std, examples)
        assert reconstructed == score and reconstructed_rows == rows

        # Only a pytest temporary archive is written; no registered scientific
        # dataset, attempt, fit, checkpoint or study receipt is touched.
        metadata = {'scope': 'analytic-software-fixture', 'source_view': 'a'}
        archive = tmp_path / 'analytic-validation.npz'
        stream = io.BytesIO()
        np.savez_compressed(stream, mean=mean, std=std,
            metadata_utf8=np.frombuffer(trainer._canonical(metadata), dtype=np.uint8))
        archive.write_bytes(stream.getvalue())
        archived, archived_rows, archived_mean, archived_std = trainer.read_validation_archive(
            archive, metadata, examples)
        assert archived == score and archived_rows == rows
        np.testing.assert_array_equal(archived_mean, mean)
        np.testing.assert_array_equal(archived_std, std)

        # Passive verification consumes the immutable arrays and cannot call
        # even the model accidentally. A changed archive identity is rejected.
        with pytest.raises(RuntimeError, match='exact source/data/member/epoch/weight'):
            trainer.read_validation_archive(archive, dict(metadata, source_view='b'), examples)
    finally:
        torch.set_num_threads(previous_threads)
