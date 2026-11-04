"""Matched inference, causal boundaries and physical structure of semantic JEPA."""
import copy
import pytest
import torch
from wirelessjev.semantic_jepa.model import SemanticJEPA, FAMILIES, objective, update_target, diagnostics


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def example(batch=4, links=6, horizons=9):
    generator = torch.Generator().manual_seed(190)
    def normal(*shape):
        return torch.randn(*shape, generator=generator)
    context = normal(batch, links, 6, 8) * .1
    context[..., 0] += .5
    context[..., 2:4] = 1.
    context[..., 7] = torch.tensor([0., .5, 1.]).repeat(links // 3)[None, :, None]
    base = normal(batch, horizons, links) * 2. + 12.
    query = normal(batch, horizons, links, 11) * .1
    query[..., 5] = torch.tensor([0., .5, 1.]).repeat(links // 3)
    query[..., 4] = torch.arange(horizons)[None, :, None] / 30.
    result = {'context': context.reshape(batch, links, 48), 'query': query,
              'memory': normal(batch, links, 13) * .1, 'map_mean': base,
              'map_std': torch.full_like(base, 2.), 'physics': base - 1.,
              'fleet_mean': base + 1., 'valid_link': torch.ones(batch, links),
              'truth': base + normal(batch, horizons, links) * 2.}
    result['future_context'] = result['context'][:, None].expand(-1, horizons, -1, -1).clone()
    return result


def nonzero_head(model):
    with torch.no_grad():
        model.head.weight.copy_(torch.linspace(-.02, .03, model.head.weight.numel()).reshape_as(model.head.weight))


def test_zero_initialized_anchor_and_residual_cap():
    b = example()
    model = SemanticJEPA()
    p = model(b)
    torch.testing.assert_close(p['mean'], b['map_mean'], rtol=0, atol=0)
    assert (p['std'] > .25).all()
    with torch.no_grad():
        model.head.bias[0] = 100.
    torch.testing.assert_close(model(b)['mean'], b['map_mean'] + 12., rtol=0, atol=2e-6)
    with torch.no_grad():
        model.head.bias[0] = -100.
    torch.testing.assert_close(model(b)['mean'], b['map_mean'] - 12., rtol=0, atol=2e-6)


def test_forward_never_reads_training_labels_or_teacher_modules():
    b = example()
    model = SemanticJEPA()
    nonzero_head(model)
    expected = model(b)['mean']
    b['truth'] = object()
    b['future_context'] = object()
    # Poison every training-only module: even accidental invocation must fail.
    def forbidden(*args, **kwargs):
        raise AssertionError('training-only module ran during inference')
    for module in (model.target_context_encoder, model.target_radio_encoder, model.projector, model.aux_head):
        module.forward = forbidden
    torch.testing.assert_close(model(b)['mean'], expected, rtol=0, atol=0)


def test_all_families_have_identical_inference_and_state_schema():
    b = example()
    predictions = []
    parameters = []
    state_keys = []
    for family in FAMILIES:
        torch.manual_seed(123)
        model = SemanticJEPA(family=family)
        nonzero_head(model)
        predictions.append(model(b)['mean'])
        parameters.append(model.inference_parameters)
        state_keys.append(set(model.state_dict()))
    assert len(set(parameters)) == 1
    assert all(keys == state_keys[0] for keys in state_keys)
    for actual in predictions[1:]:
        torch.testing.assert_close(actual, predictions[0], rtol=0, atol=0)
    total = sum(p.numel() for p in model.parameters())
    assert parameters[0] < total


def test_last_heard_semantics_and_explicit_unheard_imputation():
    b = example(batch=1)
    context = b['context'].reshape(1, 6, 6, 8)
    context[..., 0] = torch.arange(6) / 30.
    context[..., 2:4] = 0
    context[:, 0, 2, 2:4] = 1.
    context[:, 0, 4, 2:4] = 1.
    p = SemanticJEPA()(b)
    assert p['semantic_observed'][0, 0]
    assert not p['semantic_observed'][0, 1:].any()
    expected = torch.tensor([(4. - b['map_mean'][0, 0, 0]) / 10., -.1])
    torch.testing.assert_close(p['radio_semantic'][0, 0], expected)
    torch.testing.assert_close(p['radio_semantic'][0, 1:, 0], torch.zeros(5))
    torch.testing.assert_close(p['radio_semantic'][0, 1:, 1], (b['map_mean'][0, 0, 1:] - 5.) / 10.)


def test_horizons_are_direct_and_do_not_integrate_previous_predictions():
    b = example()
    model = SemanticJEPA()
    nonzero_head(model)
    original = model(b)['mean'].detach()
    changed = {k: v.clone() for k, v in b.items()}
    changed['query'][:, 3] += 2.
    altered = model(changed)['mean']
    keep = torch.tensor([0, 1, 2, 4, 5, 6, 7, 8])
    torch.testing.assert_close(altered[:, keep], original[:, keep], rtol=0, atol=0)
    assert not torch.allclose(altered[:, 3], original[:, 3])
    prefix = dict(b)
    for key in ('query', 'map_mean', 'map_std', 'physics', 'fleet_mean', 'truth', 'future_context'):
        prefix[key] = b[key][:, :3]
    torch.testing.assert_close(model(prefix)['mean'], original[:, :3], rtol=2e-6, atol=2e-6)


def test_ap_permutation_preserves_link_identity_and_band_specific_features():
    b = example(links=9)
    model = SemanticJEPA()
    nonzero_head(model)
    original = model(b)['mean']
    order = torch.tensor([6, 7, 8, 0, 1, 2, 3, 4, 5])
    permuted = {}
    for key, value in b.items():
        axis = 1 if key in ('context', 'memory', 'valid_link') else 2
        permuted[key] = value.index_select(axis, order)
    torch.testing.assert_close(model(permuted)['mean'], original[:, :, order], rtol=2e-6, atol=2e-6)
    changed = {k: v.clone() for k, v in b.items()}
    changed['query'][:, :, 0, 5] += 1.
    delta = model(changed)['mean'] - original
    assert delta[:, :, 0].abs().max() > 1e-5
    torch.testing.assert_close(delta[:, :, 1:], torch.zeros_like(delta[:, :, 1:]), rtol=0, atol=0)


def pad(batch):
    result = {}
    for key, value in batch.items():
        axis = 1 if key in ('context', 'memory', 'valid_link') else 2
        shape = list(value.shape)
        shape[axis] = 3
        fill = 0. if key == 'valid_link' else float('nan')
        result[key] = torch.cat([value, value.new_full(shape, fill)], axis)
    return result


@pytest.mark.parametrize('family', FAMILIES)
def test_padding_cannot_change_valid_forecasts_or_training_loss(family):
    b = example()
    model = SemanticJEPA(family=family)
    nonzero_head(model)
    p = model(b)
    loss, _ = objective(model, b)
    padded = pad(b)
    q = model(padded)
    other_loss, _ = objective(model, padded)
    torch.testing.assert_close(q['mean'][:, :, :6], p['mean'], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(other_loss, loss, rtol=3e-6, atol=3e-6)
    assert torch.isfinite(q['mean']).all()
    other_loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()


@pytest.mark.parametrize('family', FAMILIES)
def test_objectives_backpropagate_only_through_online_components(family):
    b = example()
    model = SemanticJEPA(family=family)
    nonzero_head(model)
    loss, metrics = objective(model, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert model.radio_encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.target_radio_encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.target_context_encoder.parameters())
    if family.endswith('jepa'):
        assert model.projector.weight.grad.abs().sum() > 0
    if family == 'semantic_aux':
        assert model.aux_head.weight.grad.abs().sum() > 0
    assert all(not value.requires_grad for value in metrics.values())


def test_target_encoder_stays_frozen_until_explicit_ema_update():
    model = SemanticJEPA()
    before = {k: v.clone() for k, v in model.target_radio_encoder.state_dict().items()}
    with torch.no_grad():
        for p in model.radio_encoder.parameters():
            p.add_(2.)
    assert all(torch.equal(model.target_radio_encoder.state_dict()[k], v) for k, v in before.items())
    update_target(model, momentum=.75)
    for key, old in before.items():
        expected = .75 * old + .25 * model.radio_encoder.state_dict()[key]
        torch.testing.assert_close(model.target_radio_encoder.state_dict()[key], expected)
    model.train()
    assert not model.target_radio_encoder.training and not model.target_context_encoder.training
    with pytest.raises(ValueError):
        update_target(model, float('nan'))


def test_semantic_teacher_never_reads_future_context_or_extra_geometry():
    b = example()
    for family in ('semantic_jepa', 'semantic_aux', 'supervised'):
        model = SemanticJEPA(family=family)
        initial, _ = objective(model, b)
        altered = dict(b, future_context=object(), actual_future_positions=object())
        other, _ = objective(model, altered)
        torch.testing.assert_close(initial, other, rtol=0, atol=0)
    with pytest.raises(ValueError):
        objective(SemanticJEPA(family='context_jepa'), dict(b, future_context=torch.full_like(b['future_context'], float('nan'))))


def test_semantic_jepa_weight_changes_training_loss_not_inference_graph():
    b = example()
    a = SemanticJEPA(jepa_weight=.03)
    c = SemanticJEPA(jepa_weight=.3)
    c.load_state_dict(a.state_dict())
    p, m = objective(a, b)
    q, _ = objective(c, b)
    torch.testing.assert_close(q - p, .27 * m['auxiliary'])
    torch.testing.assert_close(a(b)['mean'], c(b)['mean'], rtol=0, atol=0)


def test_diagnostics_separate_bands_and_rf_sensitivity_without_geometry_changes():
    b = example()
    saved = b['context'].clone()
    model = SemanticJEPA()
    result = diagnostics(model, b)
    assert set(result['online_radio_by_band']) == {'0', '1', '2'}
    assert set(result['teacher_radio_by_band']) == {'0', '1', '2'}
    assert all(row['std_mean'] > 0 for row in result['teacher_radio_by_band'].values())
    assert result['fixed_geometry_plus1db_radio_l2'] > 0
    assert result['fixed_geometry_plus1db_mean_l2'] == 0  # Fresh zero residual head.
    assert result['guarantee'] is False
    torch.testing.assert_close(b['context'], saved, rtol=0, atol=0)


def test_empty_or_constant_physical_groups_do_not_invent_variance_penalties():
    b = example(batch=1, links=3)
    _, metrics = objective(SemanticJEPA(), b)
    assert metrics['variance'] == 0  # One heard sample per physical band.
    b = example()
    context = b['context'].reshape(4, 6, 6, 8)
    context[..., 0] = .5
    b['map_mean'][:] = 10.
    _, metrics = objective(SemanticJEPA(), b)
    assert metrics['variance'] == 0


@pytest.mark.parametrize('field', ['context', 'query', 'memory', 'map_mean', 'map_std', 'physics'])
def test_nonfinite_valid_public_inputs_fail(field):
    b = example()
    b[field].reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError):
        SemanticJEPA()(b)


@pytest.mark.parametrize('change', ['link_count', 'query_width', 'mask_nonbinary', 'all_invalid', 'negative_std', 'truth_nan'])
def test_malformed_batches_fail(change):
    b = example()
    if change == 'link_count': b['context'] = b['context'][:, :5]
    elif change == 'query_width': b['query'] = b['query'][..., :10]
    elif change == 'mask_nonbinary': b['valid_link'][0, 0] = .5
    elif change == 'all_invalid': b['valid_link'][0] = 0
    elif change == 'negative_std': b['map_std'][0, 0, 0] = -1
    else: b['truth'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        objective(SemanticJEPA(), b)


@pytest.mark.parametrize('kwargs', [{'family': 'unknown'}, {'jepa_weight': float('inf')},
                                    {'variance_floor': -1}, {'hidden': 0}, {'ema_momentum': 1}])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        SemanticJEPA(**kwargs)


def test_config_and_state_reconstruct_the_same_inference():
    b = example()
    original = SemanticJEPA(family='semantic_aux', jepa_weight=.03)
    nonzero_head(original)
    restored = SemanticJEPA(**original.config)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    torch.testing.assert_close(restored(b)['mean'], original(b)['mean'], rtol=0, atol=0)
