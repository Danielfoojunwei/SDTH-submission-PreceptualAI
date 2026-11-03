"""The auxiliary projection changes training only, with real native-label smoke."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from wirelessjev.packet_value.model import PacketValueModel, objective, update_target
from wirelessjev.representation_repair import (
    ProjectedPacketValueModel,
    batch,
    data_arrays,
    evaluate,
    latent_stats,
    service_errors,
    summarize,
)


def real_arrays():
    path = Path(__file__).resolve().parents[1]/'artifacts/packet_value/data/training-340000-0.03.npz'
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k][:8].copy() for k in z.files}


def paired_models():
    torch.manual_seed(299190)
    shared = ProjectedPacketValueModel('shared')
    torch.manual_seed(299190)
    projected = ProjectedPacketValueModel('projected')
    return shared, projected


def test_exact_identity_initialization_and_parameter_accounting():
    shared, projected = paired_models()
    assert shared.inference_parameters == projected.inference_parameters == 23860
    assert shared.extra_training_parameters == 0
    assert projected.extra_training_parameters == 2352
    for key, value in shared.state_dict().items():
        torch.testing.assert_close(value, projected.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(projected.future_projection.weight, torch.eye(48), rtol=0, atol=0)
    assert (projected.future_projection.bias == 0).all()
    a = batch(real_arrays(), slice(None))
    shared.train(); projected.train()
    for key, value in shared(a['context'], a['query']).items():
        torch.testing.assert_close(value, projected(a['context'], a['query'])[key], rtol=0, atol=0)


def test_eval_service_graph_never_calls_projection_even_with_nonzero_value_head():
    torch.manual_seed(299191)
    model = ProjectedPacketValueModel('projected')
    with torch.no_grad():
        model.head.weight.normal_(0, .1)
        model.head.bias.normal_(0, .1)
        model.future_projection.weight.normal_(10, 4)
        model.future_projection.bias.fill_(300.)
    base = PacketValueModel(**model.config).eval()
    base.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('future_projection.')})
    model.eval()
    b = batch(real_arrays(), slice(None))

    def forbidden(*_):
        raise AssertionError('Projection entered the service inference graph')

    hook = model.future_projection.register_forward_pre_hook(forbidden)
    with torch.no_grad():
        expected, actual = base(b['context'], b['query']), model(b['context'], b['query'])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual['mean'].std() > .1
    with pytest.raises(AssertionError, match='service inference graph'):
        model.projected_future_latent(b['context'], b['query'])
    hook.remove()


def test_training_projection_changes_only_auxiliary_output_and_keeps_service_head_before_it():
    shared, projected = paired_models()
    with torch.no_grad():
        shared.head.weight.normal_(0, .1)
        projected.head.load_state_dict(shared.head.state_dict())
        projected.future_projection.weight.mul_(2.)
        projected.future_projection.bias.fill_(1.)
    shared.train(); projected.train()
    b = batch(real_arrays(), slice(None))
    left, right = shared(b['context'], b['query']), projected(b['context'], b['query'])
    torch.testing.assert_close(left['mean'], right['mean'], rtol=0, atol=0)
    torch.testing.assert_close(left['std'], right['std'], rtol=0, atol=0)
    torch.testing.assert_close(right['latent'], left['latent']*2+1)
    torch.testing.assert_close(left['context_latent'], right['context_latent'], rtol=0, atol=0)


def test_real_finite_original_objective_step_updates_projection_but_never_teacher_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(299192)
    model = ProjectedPacketValueModel('projected')
    model.train()
    b = batch(real_arrays(), slice(None))
    old_teacher = [p.detach().clone() for p in model.target_encoder.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss, terms = objective(model, b)
    assert torch.isfinite(loss) and terms['valid_labels'] == 16
    optimizer.zero_grad(); loss.backward()
    assert model.future_projection.weight.grad.norm() > 0
    assert model.encoder.gru.weight_ih_l0.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.target_encoder.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step(); update_target(model)
    for old, target, online in zip(old_teacher, model.target_encoder.parameters(), model.encoder.parameters(), strict=True):
        torch.testing.assert_close(target, .99*old+.01*online)
    assert not model.target_encoder.training
    assert torch.isfinite(objective(model, b)[0])


def test_actual_data_metrics_reconcile_masks_constant_denominator_and_native_units():
    torch.set_num_threads(1)
    torch.manual_seed(299193)
    model = ProjectedPacketValueModel('projected')
    a = real_arrays()
    model.eval()
    metrics = evaluate(model, a, a)
    b = batch(a, slice(None))
    with torch.no_grad():
        p = model(b['context'], b['query'])
        future = model.projected_future_latent(b['context'], b['query'])
        teacher = model.target_encoder(b['future_context'], b['future_valid'])
    direct_miss_mae = np.abs(p['delta_misses'].numpy()-a['delta_misses']).mean()
    direct_gap_mae = np.abs(p['delta_gap'].numpy()-a['delta_gap']).mean()
    assert metrics['validation']['service']['all']['miss_mae'] == pytest.approx(direct_miss_mae)
    assert metrics['validation']['service']['all']['gap_mae_seconds'] == pytest.approx(direct_gap_mae)
    future_mse = (future-teacher).square().mean().item()
    constant_mse = (teacher-teacher.mean(0)).square().mean().item()
    assert metrics['validation']['future']['mse'] == pytest.approx(future_mse)
    assert metrics['validation']['future']['training_teacher_mean_constant_mse'] == pytest.approx(constant_mse)
    assert metrics['validation']['future']['mse_over_training_teacher_mean_constant'] == pytest.approx(future_mse/constant_mse)
    assert metrics['validation']['latents']['future_teacher']['centered_rms'] > 0
    assert metrics['validation']['latents']['future_teacher']['entropy_effective_rank'] > 1


def test_reserved_split_is_rejected_before_opening_any_files(tmp_path):
    with pytest.raises(ValueError, match='Only development'):
        data_arrays(tmp_path, {}, 'test', {})


def test_collapse_guard_blocks_apparent_future_improvement():
    metric = {'service': {'all': {'miss_mae': 10., 'gap_mae_seconds': 1.}},
              'future': {'mse_over_training_teacher_mean_constant': 2.},
              'latents': {'future_teacher': {'centered_rms': 1., 'entropy_effective_rank': 8.}}}
    shared = {'variant': 'shared', 'seed': 0, 'metrics': {'training': copy.deepcopy(metric), 'validation': copy.deepcopy(metric)}}
    projected = copy.deepcopy(shared)
    projected['variant'] = 'projected'
    projected['metrics']['validation']['future']['mse_over_training_teacher_mean_constant'] = .1
    for split in ('training', 'validation'):
        projected['metrics'][split]['latents']['future_teacher']['centered_rms'] = .01
    summary = summarize([shared, projected], {'training_seeds': [0], 'collapse_guard_ratio': .8})
    assert summary['paired_seed_results'][0]['future_normalized_mse_improves']
    assert not summary['all_seeds_future_improve_with_target_guard']
    assert summary['all_seeds_both_service_maes_nonincreasing']


def test_constant_stats_and_invalid_variant_are_explicit():
    assert latent_stats(np.zeros((8, 48)))['entropy_effective_rank'] == 0
    with pytest.raises(ValueError, match='shared or projected'):
        ProjectedPacketValueModel('something_else')
