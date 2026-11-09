"""Analytic model-boundary fixtures; no registered scene, fitting or trial."""
from inspect import signature
import math

import pytest
import torch

from wirelessjev.adaptive_radio.model import AdaptiveScene, objective, observation_nll
from wirelessjev.spectral_scene.operators import query_physics_basis


@pytest.fixture
def operands():
    torch.manual_seed(914)
    b, n, q = 3, 8, 5
    geometry = torch.rand(b, n, 16, dtype=torch.float64)
    geometry[..., 6] = .3 + .3 * geometry[..., 6]
    geometry[..., 7] = torch.tensor([0., .7731899] * 4)
    geometry[..., 8] = 1.
    geometry[..., 9:11] *= .2
    geometry[..., 13] = .25
    geometry[..., 14] = -.94
    geometry[..., 15] = -.1
    a = torch.cat((torch.randn(b, n, 2, dtype=torch.float64) * .1,
                   torch.ones(b, n, 1, dtype=torch.float64),
                   torch.zeros(b, n, 1, dtype=torch.float64), geometry), -1)
    a[:, -2:, 0] = 0.
    a[:, -2:, 2] = 0.
    a[:, -2:, 3] = 1.
    reverse = a.clone()
    reverse[..., 0] += .05 * a[..., 2]
    reverse[..., 12] *= -1.
    mask = torch.ones(b, n, dtype=torch.bool)
    scheduled = mask.clone()
    query = geometry[:, :q].clone()
    fspl = torch.full((b, q), -65., dtype=torch.float64)
    offset = torch.zeros_like(fspl)
    center = fspl - 3.
    heard = torch.ones_like(center, dtype=torch.bool)
    heard[:, 1] = False
    censored = torch.zeros_like(heard)
    censored[:, 1] = True
    heard[:, -1] = False
    target = dict(heard=heard, censored=censored,
                  center_dbm=torch.where(heard, center, 0.),
                  lower_dbm=torch.where(heard, center - .5, torch.nan),
                  upper_dbm=torch.where(heard, center + .5, torch.nan),
                  censor_upper_dbm=torch.where(censored, center + 2., torch.nan),
                  detection_threshold_dbm=torch.where(censored, center + 2., center - 10.))
    return a, reverse, mask, scheduled, query, query_physics_basis(query), fspl, offset, target


def same_belief(a, b, *, exact=True):
    for key in ('mean', 'covariance'):
        torch.testing.assert_close(a[key], b[key], rtol=0 if exact else 1e-8,
                                   atol=0 if exact else 1e-8)


def controlled_model():
    model = AdaptiveScene('regression', width=16).double().eval()
    with torch.no_grad():
        for module in (model.prior_mean, model.local):
            for parameter in module.parameters():
                parameter.zero_()
        model.log_prior_std.fill_(math.log(math.expm1(.8)))
        model.discrepancy[-1].bias.fill_(math.log(3. / 11.))
    return model


def intercept_context(*, heard=True, n=6, residual=-.2, bound=0.):
    context = torch.zeros(1, n, 20, dtype=torch.float64)
    context[..., 0] = residual
    context[..., 1] = bound
    context[..., 2 if heard else 3] = 1.
    context[..., 4 + 13] = .25  # one dB quantization interval
    context[..., 4 + 14] = -.94
    context[..., 4 + 15] = -.1
    mask = torch.ones(1, n, dtype=torch.bool)
    return context, mask, mask.clone()


def test_heard_update_has_exact_gaussian_mean_and_covariance_in_known_case():
    model = controlled_model()
    value = model.encode(*intercept_context())
    # Six identical intercept-only observations, prior N(0,I), field sigma4dB,
    # and one-dB quantization variance. Coefficient units are ten dB.
    variance = .4 ** 2 + 1. / 1200.
    expected_mean = torch.zeros(1, 8, dtype=torch.float64)
    expected_mean[0, 0] = 6. * (-.6) / (variance + 6.)
    expected_cov = torch.eye(8, dtype=torch.float64)[None].clone()
    expected_cov[0, 0, 0] = variance / (variance + 6.)
    torch.testing.assert_close(value['mean'], expected_mean, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(value['covariance'], expected_cov, rtol=1e-10, atol=1e-10)


def test_uninformative_censor_bound_retains_prior_covariance_not_em_precision():
    model = controlled_model()
    context, known, schedule = intercept_context(heard=False, bound=100.)
    value = model.encode(context, known, schedule)
    torch.testing.assert_close(value['mean'], torch.zeros_like(value['mean']), rtol=0, atol=1e-12)
    torch.testing.assert_close(value['covariance'], torch.eye(8, dtype=torch.float64)[None], rtol=0, atol=1e-12)


def test_censored_event_uses_bound_not_placeholder_as_a_point_measurement():
    model = controlled_model()
    context, known, schedule = intercept_context(heard=False, bound=-1. / 3.)
    first = model.encode(context, known, schedule)
    changed = context.clone()
    changed[..., 0] = 1e5
    same_belief(first, model.encode(changed, known, schedule))
    looser = context.clone()
    looser[..., 1] = 0.
    assert first['mean'][0, 0] < model.encode(looser, known, schedule)['mean'][0, 0]
    # A point at the bound and an upper-censored event have different updates.
    point = context.clone()
    point[..., 0] = point[..., 1]
    point[..., 2], point[..., 3] = 1., 0.
    assert first['mean'][0, 0] < model.encode(point, known, schedule)['mean'][0, 0]


def test_rf_erased_control_ignores_amplitudes_but_retains_event_status(operands):
    a, _, known, scheduled, *_ = operands
    model = AdaptiveScene('rf_erased', width=16).double().eval()
    base = model.encode(a, known, scheduled)
    changed = a.clone()
    changed[..., :2] = torch.randn_like(changed[..., :2]) * 1e3
    same_belief(base, model.encode(changed, known, scheduled))
    changed[..., 2:4] = changed[..., 2:4].flip(-1)
    assert not torch.equal(base['mean'], model.encode(changed, known, scheduled)['mean'])
    # Distinguish known-report availability from the public probe schedule.
    unavailable = a.clone()
    unavailable[:, :3, :4] = 0.
    some_known = known.clone()
    some_known[:, :3] = False
    assert not torch.equal(base['mean'], model.encode(unavailable, some_known, scheduled)['mean'])


def test_pure_geometry_control_is_invariant_to_rf_events_and_report_availability(operands):
    a, _, known, scheduled, *_ = operands
    model = AdaptiveScene('pure_geometry', width=16).double().eval()
    base = model.encode(a, known, scheduled)
    changed = a.clone()
    changed[..., :2] = 1e4
    changed[..., 2:4] = changed[..., 2:4].flip(-1)
    same_belief(base, model.encode(changed, known, scheduled))
    changed[..., :4] = 0.
    same_belief(base, model.encode(changed, torch.zeros_like(known), scheduled))


@pytest.mark.parametrize('cell', ('joint', 'regression', 'rf_erased', 'pure_geometry'))
def test_empty_known_view_returns_finite_geometry_belief_without_fake_readings(operands, cell):
    a, _, known, scheduled, *_ = operands
    model = AdaptiveScene(cell, width=16).double().eval()
    a = a.clone()
    a[..., :4] = 0.
    unknown = model.encode(a, torch.zeros_like(known), scheduled)
    assert torch.isfinite(unknown['mean']).all() and torch.isfinite(unknown['covariance']).all()
    assert (torch.linalg.eigvalsh(unknown['covariance']) > 0).all()


def test_unscheduled_padding_is_inert_and_cannot_acquire_a_fake_report(operands):
    a, _, known, scheduled, *_ = operands
    model = AdaptiveScene(width=16).double().eval()
    known, scheduled = known.clone(), scheduled.clone()
    known[:, -2:] = False
    scheduled[:, -2:] = False
    clean = a.clone()
    clean[:, -2:] = 0.
    base = model.encode(clean, known, scheduled)
    dirty = a.clone()
    dirty[:, -2:] = torch.nan
    same_belief(base, model.encode(dirty, known, scheduled))
    with pytest.raises(ValueError, match='schedule'):
        model.encode(clean, torch.ones_like(known), scheduled)
    with pytest.raises(ValueError, match='nonempty'):
        model.encode(clean, torch.zeros_like(known), torch.zeros_like(scheduled))


def test_render_has_no_context_or_hidden_identity_bypass_and_no_cached_rf_state(operands):
    a, b, known, scheduled, query, basis, fspl, offset, _ = operands
    model = AdaptiveScene(width=16).double().eval()
    assert tuple(signature(model.encode).parameters) == ('context', 'known_mask', 'scheduled_mask')
    assert tuple(signature(model.render).parameters) == ('belief', 'query', 'basis', 'fspl', 'offset')
    belief = model.encode(a, known, scheduled)
    before = model.render(belief, query, basis, fspl, offset)
    model.encode(b, known, scheduled)
    after = model.render(belief, query, basis, fspl, offset)
    for x, y in zip(before, after):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_calibrated_reverse_direction_has_same_physical_prediction(operands):
    a, _, known, scheduled, query, basis, fspl, offset, _ = operands
    model = AdaptiveScene(width=16).double().eval()
    belief = model.encode(a, known, scheduled)
    reverse = query.clone()
    reverse[..., 8] *= -1.
    after = model.render(belief, reverse, query_physics_basis(reverse), fspl, offset)
    for x, y in zip(model.render(belief, query, basis, fspl, offset), after):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    reverse_context = a.clone()
    reverse_context[..., 4 + 8] *= -1.
    same_belief(belief, model.encode(reverse_context, known, scheduled))


def test_interval_and_censor_likelihood_gradients_are_finite_and_unknown_is_inert(operands):
    *_, fspl, _, target = operands
    mean = (fspl + 20.).detach().requires_grad_()
    std = torch.full_like(mean, .3, requires_grad=True)
    loss = observation_nll(mean, std, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()
    assert mean.grad[target['heard']].abs().sum() > 0
    assert mean.grad[target['censored']].abs().sum() > 0
    assert torch.equal(mean.grad[~(target['heard'] | target['censored'])], torch.zeros_like(mean.grad[~(target['heard'] | target['censored'])]))


def test_heard_bin_straddling_detection_floor_scores_interval_event_intersection():
    target = dict(heard=torch.tensor([[True]]), censored=torch.tensor([[False]]),
                  center_dbm=torch.tensor([[-80.]], dtype=torch.float64),
                  lower_dbm=torch.tensor([[-80.5]], dtype=torch.float64),
                  upper_dbm=torch.tensor([[-79.5]], dtype=torch.float64),
                  censor_upper_dbm=torch.tensor([[0.]], dtype=torch.float64),
                  detection_threshold_dbm=torch.tensor([[-80.]], dtype=torch.float64))
    mean = torch.tensor([[-80.]], dtype=torch.float64, requires_grad=True)
    std = torch.ones_like(mean, requires_grad=True)
    expected = -torch.log(torch.special.ndtr(torch.tensor(.5, dtype=torch.float64)) - .5)
    actual = observation_nll(mean, std, target)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual.backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(std.grad).all()


def test_joint_alignment_reaches_both_view_means_and_regression_removes_only_alignment(operands):
    a, b, known, scheduled, query, basis, fspl, offset, target = operands
    joint = AdaptiveScene('joint', width=16).double()
    regression = AdaptiveScene('regression', width=16).double()
    regression.load_state_dict(joint.state_dict())
    captured = []
    original = joint.encode

    def capture(*args):
        value = original(*args)
        value['mean'].retain_grad()
        captured.append(value)
        return value

    joint.encode = capture
    loss, diagnostic = objective(joint, (a, known, scheduled), (b, known, scheduled),
                                 query, basis, fspl, offset, target)
    reference, rd = objective(regression, (a, known, scheduled), (b, known, scheduled),
                              query, basis, fspl, offset, target)
    torch.testing.assert_close(diagnostic['measurement_nll'], rd['measurement_nll'], rtol=0, atol=0)
    assert rd['alignment'] == 0 and rd['alignment_weight'] == 0
    torch.testing.assert_close(loss - reference, diagnostic['alignment_weight'] * diagnostic['alignment'], rtol=1e-8, atol=1e-10)
    assert diagnostic['measurement_mean_gradient'] > 0 and diagnostic['alignment_mean_gradient'] > 0
    assert not diagnostic['alignment_weight'].requires_grad
    difference = captured[0]['mean'] - captured[1]['mean']
    weight = (captured[0]['covariance'] + captured[1]['covariance'] + .01 * torch.eye(8)).detach()
    align = (difference * torch.linalg.solve(weight, difference[..., None]).squeeze(-1)).sum(-1).mean() / 8.
    ga, gb = torch.autograd.grad(align, (captured[0]['mean'], captured[1]['mean']), retain_graph=True)
    assert ga.abs().sum() > 0 and gb.abs().sum() > 0
    torch.testing.assert_close(ga, -gb, rtol=0, atol=0)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in joint.parameters())


@pytest.mark.parametrize('settings', ({'cell': 'unknown'}, {'width': 15}, {'em_steps': 4}))
def test_model_cannot_silently_add_unregistered_capacity_or_update_budget(settings):
    with pytest.raises(ValueError, match='Unregistered'):
        AdaptiveScene(**settings)
