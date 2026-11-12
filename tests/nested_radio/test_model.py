"""Analytic software checks; no registered RF data, model fit or artifact."""
from copy import deepcopy
import json
import math

import pytest
import torch

from wirelessjev.hierarchical_radio.model import (
    HierarchicalField, observation_nll,
)
from wirelessjev.nested_radio.model import CELLS, NestedField, objective
from wirelessjev.spectral_scene.operators import query_physics_basis


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fixture(*, batch=2, sites=4, tokens=24, queries=13, dtype=torch.float64):
    """A hand-written linear field with interval, censored and unknown events."""
    generator = torch.Generator().manual_seed(48191)
    aps = torch.rand((batch, sites, 3), generator=generator, dtype=dtype) * 16 - 8
    aps[..., 2] = 3.5

    def geometry(rx, routes):
        chosen = torch.gather(aps, 1, routes[..., None].expand(*routes.shape, 3))
        q = torch.zeros((*routes.shape, 16), dtype=dtype)
        q[..., :3] = rx / torch.tensor((20., 20., 5.), dtype=dtype)
        q[..., 3:6] = chosen / torch.tensor((20., 20., 5.), dtype=dtype)
        q[..., 6] = torch.linalg.vector_norm(rx-chosen, dim=-1).log10() / 2
        q[..., 7] = math.log(5.2/2.4)
        q[..., 8] = -1.
        q[..., 9] = .125
        q[..., 10] = .06
        q[..., 13] = .25
        q[..., 14] = -1.
        q[..., 15] = 2/3
        return q

    routes = torch.arange(tokens).remainder(sites)[None].expand(batch, -1).clone()
    rx = torch.rand((batch, tokens, 3), generator=generator, dtype=dtype) * 12 - 6
    rx[..., 2] = 1.
    q = geometry(rx, routes)
    residual = -12. - .3*rx[..., 0] + .2*rx[..., 1]
    known = torch.ones((batch, tokens), dtype=torch.bool)
    known[:, 2::7] = False
    censored = torch.zeros_like(known)
    censored[:, 1::4] = True
    censored &= known
    heard = known & ~censored
    lo = torch.where(heard, residual-.5, torch.full_like(residual, -torch.inf))
    hi = torch.where(known, residual+.5, torch.full_like(residual, torch.inf))
    context = torch.cat((torch.where(heard, residual/30, 0.)[..., None],
        torch.where(censored, hi/30, 0.)[..., None], heard[..., None], censored[..., None], q), -1)
    a = {'context': context, 'known': known, 'scheduled': torch.ones_like(known),
         'ap': routes, 'rx_xyz': rx, 'lower': lo, 'upper': hi,
         'floor': torch.full_like(residual, -50.)}
    b = {key: value.clone() for key, value in a.items()}
    b['lower'] = torch.where(heard, b['lower']+.8, b['lower'])
    b['upper'] = torch.where(known, b['upper']+.8, b['upper'])
    b['context'][..., 0] = torch.where(heard, (residual+.8)/30, 0.)
    b['context'][..., 1] = torch.where(censored, b['upper']/30, 0.)

    qr = torch.arange(queries).remainder(sites)[None].expand(batch, -1).clone()
    qrx = torch.rand((batch, queries, 3), generator=generator, dtype=dtype) * 12 - 6
    qrx[..., 2] = 1.
    qq = geometry(qrx, qr)
    query = {'q': qq, 'ap': qr, 'rx_xyz': qrx, 'basis': query_physics_basis(qq),
             'fspl': torch.full((batch, queries), -50., dtype=dtype),
             'offset': torch.zeros((batch, queries), dtype=dtype)}
    truth = query['fspl'] - 12. - .3*qrx[..., 0] + .2*qrx[..., 1]
    target_heard = torch.ones_like(truth, dtype=torch.bool)
    target_censored = torch.zeros_like(target_heard)
    target_censored[:, 3::7] = True
    target_heard &= ~target_censored
    target = {'heard': target_heard, 'censored': target_censored,
        'lower': torch.where(target_heard, truth-.5, torch.full_like(truth, -torch.inf)),
        'upper': truth+.5}
    return a, b, query, target, aps


def model(cell, *, dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(48192)
        result = NestedField(cell)
    return result.to(dtype=dtype)


def field_covariance(instance, belief, query):
    return instance.predictive_covariance(belief, query['q'], query['ap'],
        query['rx_xyz'], query['basis'])


@pytest.mark.parametrize('cell', CELLS)
def test_serializable_public_settings_preserve_variant_and_legacy_core(cell):
    instance = model(cell)
    expected = {'cell': cell, 'width': 32, 'inference_steps': 3}
    assert instance.settings == expected and instance.variant == cell
    restored = NestedField(**json.loads(json.dumps(instance.settings))).double()
    restored.load_state_dict(instance.state_dict(), strict=True)
    assert restored.settings == expected
    core = 'global8_reg' if cell.startswith('global8') else 'hier_reg'
    if cell == 'nested_joint':
        core = 'hier_joint'
    assert instance.config['cell'] == core


@pytest.mark.parametrize('cell', ['nested_reg', 'nested_joint'])
@pytest.mark.parametrize('events', ['interval', 'point', 'unknown'])
def test_exact_zero_local_design_recovers_global8_mean_std_and_field_covariance(cell, events):
    a, _, query, _, aps = fixture(sites=8)
    if events == 'point':
        heard = a['known'] & (a['context'][..., 2] > .5)
        centers = .5*(a['lower']+a['upper'])
        a['lower'] = torch.where(heard, centers, a['lower'])
        a['upper'] = torch.where(heard, centers, a['upper'])
    elif events == 'unknown':
        a['known'].zero_()
        a['context'][..., :4].zero_()
        a['lower'].fill_(-torch.inf)
        a['upper'].fill_(torch.inf)
    instance = model(cell)
    with torch.no_grad():
        instance.local_gates.zero_()
        instance.local_prior_mean.copy_(torch.linspace(-.4, .4, 16, dtype=torch.float64))
        instance.raw_local_std.fill_(2.)
    reference = HierarchicalField('global8_reg').double()
    reference.load_state_dict(instance.base_state_dict(), strict=True)
    state = instance.encode_view(a, aps)
    expected = reference.encode_view(a, aps)
    actual_mean, actual_std = instance.render_queries(state, query)
    expected_mean, expected_std = reference.render_queries(expected, query)
    # Exact submodel in real arithmetic; the pooled and block solvers have
    # different floating reduction orders, so allow float64 roundoff only.
    torch.testing.assert_close(actual_mean, expected_mean, rtol=0., atol=2e-11)
    torch.testing.assert_close(actual_std, expected_std, rtol=0., atol=2e-11)
    torch.testing.assert_close(field_covariance(instance, state, query),
        field_covariance(reference, expected, query), rtol=0., atol=2e-11)
    assert torch.count_nonzero(instance.local_features(a['context'][..., 4:],
        a['rx_xyz'], aps, a['ap'])) == 0


@pytest.mark.parametrize('cell', ['nested_reg', 'nested_joint'])
def test_one_local_gates_recover_legacy_hierarchy_predictions(cell):
    a, _, query, _, aps = fixture()
    instance = model(cell)
    with torch.no_grad():
        instance.local_gates.fill_(1.)
    reference = HierarchicalField('hier_reg').double()
    reference.load_state_dict(instance.base_state_dict(), strict=True)
    state = instance.encode_view(a, aps)
    expected = reference.encode_view(a, aps)
    for observed, wanted in zip(instance.render_queries(state, query), reference.render_queries(expected, query)):
        torch.testing.assert_close(observed, wanted, rtol=0., atol=2e-11)
    torch.testing.assert_close(field_covariance(instance, state, query),
        field_covariance(reference, expected, query), rtol=0., atol=2e-11)


def test_projection_is_finite_bounded_and_shrinkage_scales_effective_design():
    instance = model('nested_reg')
    with torch.no_grad():
        instance.local_gates.copy_(torch.linspace(-1., 2., 16, dtype=torch.float64))
        instance.local_prior_mean.fill_(.3)
        instance.raw_local_std.fill_(.7)
    instance.project_gates_()
    assert torch.equal(instance.local_gates, torch.linspace(-1., 2., 16, dtype=torch.float64).clamp(0., 1.))
    sigma = .2 + torch.nn.functional.softplus(instance.raw_local_std)
    rms = instance.gates()*torch.sqrt(instance.local_prior_mean.square()+sigma.square())
    torch.testing.assert_close(instance.shrinkage_penalty(), .01*(rms/.5).mean())
    diag = instance.diagnostics()
    assert isinstance(diag, dict) and diag
    json.dumps(diag, allow_nan=False)
    a, _, _, _, aps = fixture()
    reference = HierarchicalField('hier_reg').double()
    reference.load_state_dict(instance.base_state_dict(), strict=True)
    wanted = reference.local_features(a['context'][..., 4:], a['rx_xyz'], aps, a['ap']) * instance.gates()
    torch.testing.assert_close(instance.local_features(a['context'][..., 4:], a['rx_xyz'], aps, a['ap']), wanted)


def test_coefficient_rescaling_cannot_make_an_unchanged_effective_prior_cheaper():
    instance = model('nested_reg')
    with torch.no_grad():
        instance.local_gates.fill_(.4)
        instance.local_prior_mean.copy_(torch.linspace(-.3, .4, 16, dtype=torch.float64))
    prior_before = {name: value.detach().clone() for name, value in instance.effective_local_prior().items()}
    penalty_before = instance.shrinkage_penalty().detach().clone()
    gradients = torch.autograd.grad(instance.shrinkage_penalty(),
        (instance.local_gates, instance.local_prior_mean, instance.raw_local_std))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert (gradients[0] > 0.).all() and (gradients[2] > 0.).all()
    assert torch.equal(gradients[1].sign(), instance.local_prior_mean.sign())
    sigma = .2 + torch.nn.functional.softplus(instance.raw_local_std.detach())
    with torch.no_grad():
        instance.local_gates.div_(2.)
        instance.local_prior_mean.mul_(2.)
        instance.raw_local_std.copy_(torch.log(torch.expm1(2.*sigma-.2)))
    prior_after = instance.effective_local_prior()
    assert set(prior_after) == {'mean', 'std', 'rms'}
    for name in prior_before:
        torch.testing.assert_close(prior_after[name], prior_before[name], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(instance.shrinkage_penalty(), penalty_before, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('cell', ['nested_reg', 'nested_joint'])
def test_effective_rms_penalty_has_finite_backward_at_exact_zero_gates(cell):
    instance = model(cell)
    with torch.no_grad():
        instance.local_gates.zero_()
        instance.local_prior_mean.copy_(torch.linspace(-.3, .4, 16, dtype=torch.float64))
    penalty = instance.shrinkage_penalty()
    assert penalty == 0.
    penalty.backward()
    for parameter in (instance.local_gates, instance.local_prior_mean, instance.raw_local_std):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # Feasible projected parameters reach the objective directly, retaining
    # its ambient one-sided derivative without a second forward clamp.
    assert (instance.local_gates.grad > 0.).all()
    assert torch.count_nonzero(instance.local_prior_mean.grad) == 0
    assert torch.count_nonzero(instance.raw_local_std.grad) == 0
    sigma = .2 + torch.nn.functional.softplus(instance.raw_local_std)
    expected_right_derivative = .01*torch.sqrt(instance.local_prior_mean.square()+sigma.square()).mean()/.5
    with torch.no_grad():
        instance.local_gates.fill_(1e-6)
    right_derivative = (instance.shrinkage_penalty()-penalty.detach())/1e-6
    assert right_derivative > 0.
    torch.testing.assert_close(right_derivative, expected_right_derivative, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('bad', [-.01, 1.01])
def test_infeasible_live_gates_are_rejected_until_projected_before_forward(bad):
    instance = model('nested_reg')
    a, _, query, _, aps = fixture()
    with torch.no_grad():
        instance.local_gates[0] = bad
    with pytest.raises(ValueError, match='Project'):
        instance.gates()
    with pytest.raises(ValueError, match='Project'):
        instance(a, query, aps)
    instance.project_gates_()
    mean, std = instance(a, query, aps)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()


def test_invalid_checkpoint_gates_are_rejected_before_any_base_weight_is_copied():
    instance = model('nested_reg')
    original = {name: value.clone() for name, value in instance.state_dict().items()}
    invalid = [torch.full((16,), -.01), torch.full((16,), 1.01),
        torch.full((16,), float('nan')), torch.full((16,), float('inf')),
        torch.ones(15), torch.zeros(16, dtype=torch.int64),
        torch.zeros(16, dtype=torch.complex64), 'not a tensor', None]
    for gates in invalid:
        checkpoint = {name: value.clone() for name, value in original.items()}
        checkpoint['global_prior_mean'] += 7.
        if gates is None:
            del checkpoint['local_gates']
        else:
            checkpoint['local_gates'] = gates
        with pytest.raises(RuntimeError, match='[Gg]ates'):
            instance.load_state_dict(checkpoint, strict=True)
        for name, actual in instance.state_dict().items():
            assert torch.equal(actual, original[name]), name


@pytest.mark.parametrize('cell', ['global8_reg', 'global8_joint', 'hier_reg'])
def test_fixed_control_gate_tampering_is_rejected_live_and_before_checkpoint_copy(cell):
    instance = model(cell)
    with torch.no_grad():
        instance.local_gates.fill_(.5)
    with pytest.raises(ValueError, match='Fixed-control'):
        instance.gates()
    instance.project_gates_()
    expected = 1. if cell == 'hier_reg' else 0.
    assert torch.equal(instance.gates(), torch.full_like(instance.local_gates, expected))
    original = {name: value.clone() for name, value in instance.state_dict().items()}
    checkpoint = {name: value.clone() for name, value in original.items()}
    checkpoint['global_prior_mean'] += 7.
    checkpoint['local_gates'].fill_(.5)
    with pytest.raises(RuntimeError, match='fixed-control'):
        instance.load_state_dict(checkpoint)
    for name, actual in instance.state_dict().items():
        assert torch.equal(actual, original[name]), name


@pytest.mark.parametrize('cell', ['nested_reg', 'nested_joint'])
def test_direct_projected_gate_keeps_the_penalty_gradient_at_one(cell):
    instance = model(cell)
    with torch.no_grad():
        instance.local_gates.fill_(1.)
    instance.shrinkage_penalty().backward()
    sigma = .2 + torch.nn.functional.softplus(instance.raw_local_std)
    expected = .01*torch.sqrt(instance.local_prior_mean.square()+sigma.square())/(.5*16)
    torch.testing.assert_close(instance.local_gates.grad, expected, rtol=1e-12, atol=1e-12)
    assert (instance.local_gates.grad > 0.).all()


def test_effective_prior_diagnostics_and_penalty_share_declared_normalized_units():
    instance = model('nested_reg')
    with torch.no_grad():
        instance.local_gates.copy_(torch.linspace(0., 1., 16, dtype=torch.float64))
        instance.local_prior_mean.copy_(torch.linspace(-.3, .4, 16, dtype=torch.float64))
        sigma = torch.linspace(.25, 1., 16, dtype=torch.float64)
        instance.raw_local_std.copy_(torch.log(torch.expm1(sigma-.2)))
    alpha = instance.gates()
    expected = {'mean': alpha*instance.local_prior_mean, 'std': alpha*sigma,
        'rms': alpha*torch.sqrt(instance.local_prior_mean.square()+sigma.square())}
    prior = instance.effective_local_prior()
    diagnostics = instance.diagnostics()
    for name, value in expected.items():
        torch.testing.assert_close(prior[name], value, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(torch.tensor(diagnostics[f'effective_gated_prior_{name}_db'], dtype=torch.float64),
            10.*value, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(instance.shrinkage_penalty(), .01*(expected['rms']/.5).mean(), rtol=1e-12, atol=1e-12)


def test_prior_only_partial_gate_scales_mean_linearly_and_local_covariance_quadratically():
    a, _, query, _, aps = fixture(sites=8)
    a['known'].zero_()
    a['context'][..., :4].zero_()
    a['lower'].fill_(-torch.inf)
    a['upper'].fill_(torch.inf)
    instance = model('nested_reg')
    with torch.no_grad():
        instance.local_prior_mean.copy_(torch.linspace(-.4, .4, 16, dtype=torch.float64))
    snapshots = {}
    for gate in (0., .5, 1.):
        with torch.no_grad():
            instance.local_gates.fill_(gate)
        state = instance.encode_view(a, aps)
        mean, _ = instance.render_queries(state, query)
        snapshots[gate] = (mean, field_covariance(instance, state, query))
    zero, half, one = snapshots[0.], snapshots[.5], snapshots[1.]
    torch.testing.assert_close(half[0]-zero[0], .5*(one[0]-zero[0]), rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(half[1]-zero[1], .25*(one[1]-zero[1]), rtol=1e-10, atol=1e-11)
    assert (one[1].diagonal(dim1=-2, dim2=-1) > zero[1].diagonal(dim1=-2, dim2=-1)).all()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_gate_state_is_terminal_instead_of_silently_projected(bad):
    instance = model('nested_reg')
    with torch.no_grad():
        instance.local_gates[0] = bad
    with pytest.raises((ValueError, FloatingPointError)):
        instance.project_gates_()


@pytest.mark.parametrize('cell', CELLS)
def test_production_shapes_float32_objective_optimizer_and_projected_gates_are_finite(cell):
    instance = model(cell, dtype=torch.float32)
    optimizer = torch.optim.AdamW(instance.parameters(), lr=.003, weight_decay=1e-4)
    for tokens in (192, 96):
        a, b, query, target, aps = fixture(batch=4, sites=4, tokens=tokens,
                                          queries=192, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss, details = objective(instance, a, b, query, target, aps)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert details and all(torch.as_tensor(value).ndim == 0 and
            torch.isfinite(torch.as_tensor(value)) for value in details.values())
        loss.backward()
        gradients = [p.grad for p in instance.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        if cell.startswith('nested'):
            assert instance.local_gates.grad is not None
        torch.nn.utils.clip_grad_norm_(instance.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        instance.project_gates_()
        assert all(torch.isfinite(value).all() for value in instance.state_dict().values())
        assert ((instance.gates() >= 0) & (instance.gates() <= 1)).all()


@pytest.mark.parametrize('cell', CELLS)
def test_objective_is_same_two_view_likelihood_plus_declared_agreement_and_gate_penalty(cell):
    instance = model(cell)
    a, b, query, target, aps = fixture()
    ea, eb = instance.encode_view(a, aps), instance.encode_view(b, aps)
    pa, sa = instance.render_queries(ea, query)
    pb, sb = instance.render_queries(eb, query)
    likelihood = .5*(observation_nll(pa, sa, target) + observation_nll(pb, sb, target))
    supported = torch.gather(ea['observed_ap'] & eb['observed_ap'], 1, query['ap'])
    agreement = (((pa-pb)/10.).square() / ((sa.square()+sb.square())/100.+.01).detach())[supported].mean()
    expected = likelihood + instance.shrinkage_penalty()
    if cell.endswith('joint'):
        assert agreement > 0
        expected = expected + .1*agreement
    observed, _ = objective(instance, a, b, query, target, aps)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('cell', ['global8_joint', 'nested_joint'])
def test_joint_objective_gradient_matches_two_live_views_not_a_stopped_teacher(cell):
    instance = model(cell)
    a, b, query, target, aps = fixture()
    pa, sa = instance(a, query, aps)
    pb, sb = instance(b, query, aps)
    likelihood = .5*(observation_nll(pa, sa, target) + observation_nll(pb, sb, target))
    agreement = (((pa-pb)/10.).square() / ((sa.square()+sb.square())/100.+.01).detach()).mean()
    expected = likelihood + .1*agreement + instance.shrinkage_penalty()
    observed, _ = objective(instance, a, b, query, target, aps)
    parameters = tuple(instance.parameters())
    expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    observed_gradients = torch.autograd.grad(observed, parameters, allow_unused=True)
    compared = 0
    for wanted, actual in zip(expected_gradients, observed_gradients):
        if wanted is None or actual is None:
            assert wanted is actual
        else:
            compared += 1
            torch.testing.assert_close(actual, wanted, rtol=2e-10, atol=2e-11)
    assert compared > 0


@pytest.mark.parametrize('cell', ['global8_joint', 'nested_joint'])
def test_no_shared_observed_ap_produces_zero_agreement_without_invented_observations(cell):
    instance = model(cell)
    a, b, query, target, aps = fixture()
    for view in (a, b):
        view['known'].zero_()
        view['context'][..., :4].zero_()
        view['lower'].fill_(-torch.inf)
        view['upper'].fill_(torch.inf)
    loss, details = objective(instance, a, b, query, target, aps)
    assert torch.isfinite(loss)
    assert float(details['alignment']) == 0.
    assert float(details['supported_agreement_queries']) == 0.


@pytest.mark.parametrize('cell', ['global8_reg', 'nested_reg', 'nested_joint'])
def test_ap_renaming_changes_routes_without_changing_prediction(cell):
    instance = model(cell)
    a, _, query, _, aps = fixture(sites=8)
    mean, std = instance(a, query, aps)
    perm = torch.tensor([3, 0, 7, 1, 6, 2, 5, 4])
    inverse = perm.argsort()
    renamed_a, renamed_q = deepcopy(a), deepcopy(query)
    renamed_a['ap'] = inverse[a['ap']]
    renamed_q['ap'] = inverse[query['ap']]
    changed_mean, changed_std = instance(renamed_a, renamed_q, aps[:, perm])
    torch.testing.assert_close(changed_mean, mean, rtol=0., atol=2e-10)
    torch.testing.assert_close(changed_std, std, rtol=0., atol=2e-10)


def test_unknown_payloads_never_become_measurements():
    instance = model('nested_reg')
    a, _, query, _, aps = fixture()
    mean, std = instance(a, query, aps)
    unknown = ~a['known']
    altered = deepcopy(a)
    altered['context'][..., :4][unknown] = 123.
    altered['lower'][unknown] = 456.
    altered['upper'][unknown] = -456.
    changed_mean, changed_std = instance(altered, query, aps)
    torch.testing.assert_close(changed_mean, mean, rtol=0., atol=0.)
    torch.testing.assert_close(changed_std, std, rtol=0., atol=0.)


def test_large_forecast_uses_marginals_and_cannot_allocate_unbounded_full_covariance():
    instance = model('nested_reg')
    a, _, query, _, aps = fixture(queries=513)
    state = instance.encode_view(a, aps)
    mean, std = instance.render_queries(state, query)
    assert mean.shape == std.shape == (2, 513)
    assert torch.isfinite(mean).all() and torch.isfinite(std).all()
    with pytest.raises(ValueError, match='512'):
        field_covariance(instance, state, query)


@pytest.mark.parametrize('cell', ['unknown', 'hier_joint', '', None])
def test_unregistered_public_variants_are_rejected(cell):
    with pytest.raises(ValueError):
        NestedField(cell)
