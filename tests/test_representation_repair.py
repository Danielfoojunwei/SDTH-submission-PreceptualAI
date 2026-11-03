"""The auxiliary projection changes training only, with real native-label smoke."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

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


def archive_runner():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('current_representation_repair_archive_runner', root/'scripts/train_representation_repair.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


def test_completed_archive_verifies_exact_metrics_read_only_and_distinguishes_source_history():
    root, runner = archive_runner()
    out = root/'artifacts/representation_repair'
    before = {str(p.relative_to(out)): runner.digest(p) for p in out.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    result = runner.verify_archive(root)
    after = {str(p.relative_to(out)): runner.digest(p) for p in out.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    assert before == after
    assert result['verified_fits'] == 6 and result['fixed_final_epoch'] == 40
    assert result['exact_checkpoint_metric_recomputation']
    assert result['new_optimizer_steps'] == 0
    assert result['execution_source_copies_verified'] == 10
    assert result['exposed_input_and_receipt_files_verified'] == 80
    assert result['current_model_source_matches_execution']
    assert result['frozen_execution_source_copies_unchanged']
    assert result['current_support_differs_from_execution'] == [
        'scripts/train_representation_repair.py', 'tests/test_representation_repair.py']
    assert not result['reserved_holdout_inputs_opened']
    # The earlier verification is retained with its original point-in-time
    # meaning; current support changes are not called unchanged execution code.
    historical = json.loads((out/'verification_at_execution.json').read_text())
    assert historical['all_frozen_hashes_unchanged']
    assert runner.digest(out/'verification_at_execution.json') == result['verification_at_execution_sha256']


def test_nested_registration_preserves_original_bytes_and_declares_exact_exposed_data_reuse():
    root, runner = archive_runner()
    out = root/'artifacts/representation_repair'
    assert not (out/'protocol.json').exists()
    path = out/'registration/protocol.json'
    protocol = json.loads(path.read_text())
    manifest = json.loads((out/'preregistration.json').read_text())
    assert runner.digest(path) == manifest['source_sha256']['artifacts/representation_repair/protocol.json']
    assert path.read_bytes() == (out/'executed_source/artifacts/representation_repair/protocol.json').read_bytes()
    assert runner.digest(root/'src/wirelessjev/representation_repair.py') == manifest['source_sha256']['src/wirelessjev/representation_repair.py']
    original = json.loads((root/'artifacts/packet_value/protocol.json').read_text())
    runner.validate_reuse_registration(protocol, original)
    assert protocol['training_sites'] == list(range(340000, 340016))
    assert protocol['validation_sites'] == list(range(341000, 341004))


@pytest.mark.parametrize('relative', [
    'checkpoints/shared-0.pt', 'logs/projected-1.jsonl',
    'executed_source/scripts/train_representation_repair.py',
])
def test_archive_verifier_rejects_changed_historical_evidence_before_loading_data(tmp_path, relative):
    root, runner = archive_runner()
    original = root/'artifacts/representation_repair'
    copied = tmp_path/'artifacts/representation_repair'
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns('__pycache__'))
    target = copied/relative
    target.write_bytes(target.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='Archived evidence changed'):
        runner.verify_archive(tmp_path)


def test_reuse_registration_cannot_rename_or_substitute_reserved_scenes():
    root, runner = archive_runner()
    original = json.loads((root/'artifacts/packet_value/protocol.json').read_text())
    protocol = json.loads((root/'artifacts/representation_repair/registration/protocol.json').read_text())
    protocol['validation_sites'][0] = original['test_sites'][0]
    with pytest.raises(ValueError, match='original validation scene identities'):
        runner.validate_reuse_registration(protocol, original)


def test_current_runner_refuses_unregistered_new_training(tmp_path):
    _, runner = archive_runner()
    with pytest.raises(ValueError, match='Reproduce fresh training from preregistration commit'):
        runner.verify_archive(tmp_path)
