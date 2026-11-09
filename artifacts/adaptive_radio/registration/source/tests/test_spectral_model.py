"""Analytic software fixtures only; no new RT scenes, data fits or results."""
from inspect import signature

import pytest
import torch

from wirelessjev.spectral_scene.model import (
    SpectralScene, balanced_objective, observation_nll,
)
from wirelessjev.spectral_scene.operators import query_physics_basis


@pytest.fixture
def operands():
    torch.manual_seed(811)
    b, c, q = 12, 16, 8
    geometry = torch.rand(b, c, 16)
    geometry[..., 6] = .3 + .4 * geometry[..., 6]
    geometry[..., 7] = torch.tensor([0., .7731899] * (c // 2))
    geometry[..., 8] = 1.
    geometry[..., 9] *= .5
    geometry[..., 10] *= .4
    geometry[..., 14] = -.94
    geometry[..., 15] = 2. / 3.
    a = torch.cat((torch.randn(b, c, 2) * .15,
                   torch.ones(b, c, 1), torch.zeros(b, c, 1), geometry), -1)
    a[:, -2:, 0] = 0.
    a[:, -2:, 2] = 0.
    a[:, -2:, 3] = 1.
    view_b = a.clone()
    view_b[..., 0] += torch.randn(b, c) * .05 * a[..., 2]
    view_b[..., 12] *= -1.  # calibrated reverse direction, same geometry.
    mask = torch.ones(b, c, dtype=torch.bool)
    query = geometry[:, :q].clone()
    basis = query_physics_basis(query)
    fspl = -60. + torch.randn(b, q)
    offset = torch.zeros(b, q)
    center = fspl - 4. + torch.randn(b, q)
    heard = torch.ones(b, q, dtype=torch.bool)
    censored = torch.zeros_like(heard)
    heard[:, 1::3] = False
    censored[:, 1::3] = True
    heard[:, -1] = False
    censored[:, -1] = False
    target = {'heard': heard, 'censored': censored,
              'center_dbm': center,
              'lower_dbm': torch.where(heard, center - .5, torch.nan),
              'upper_dbm': torch.where(heard, center + .5, torch.nan),
              'censor_upper_dbm': torch.where(censored, center + 2., torch.nan)}
    return a, view_b, mask, query, basis, fspl, offset, target


def test_decoder_has_no_rf_token_or_power_bypass(operands):
    a, b, mask, query, basis, fspl, offset, _ = operands
    model = SpectralScene('none').eval()
    assert tuple(signature(model.render).parameters) == ('z', 'query', 'basis', 'fspl', 'offset')
    z = model.encode(a, mask)
    before = model.render(z, query, basis, fspl, offset)
    # Another encoding cannot mutate hidden RF state used by an existing z.
    changed = a.clone()
    changed[..., :2] += 3.
    model.encode(changed, mask)
    after = model.render(z, query, basis, fspl, offset)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert not torch.equal(z, model.encode(changed, mask))


def test_reciprocal_query_direction_is_not_a_learned_power_offset(operands):
    a, _, mask, query, basis, fspl, offset, _ = operands
    model = SpectralScene('none').eval()
    z = model.encode(a, mask)
    reverse = query.clone()
    reverse[..., 8] *= -1.
    reverse_basis = query_physics_basis(reverse)
    torch.testing.assert_close(reverse_basis, basis, rtol=0, atol=0)
    torch.testing.assert_close(model.render(z, query, basis, fspl, offset),
                               model.render(z, reverse, reverse_basis, fspl, offset), rtol=0, atol=0)


def test_masked_nan_and_overflow_padding_are_inert(operands):
    a, _, mask, *_ = operands
    model = SpectralScene('none').eval()
    mask = mask.clone()
    mask[:, -3:] = False
    clean = a.clone()
    clean[:, -3:] = 0.
    expected = model.encode(clean, mask)
    nan = a.clone()
    nan[:, -3:] = torch.nan
    huge = a.clone()
    huge[:, -3:] = 1e38
    torch.testing.assert_close(expected, model.encode(nan, mask), rtol=0, atol=0)
    torch.testing.assert_close(expected, model.encode(huge, mask), rtol=0, atol=0)
    with pytest.raises(ValueError, match='finite'):
        model.encode(nan, torch.ones_like(mask))
    with pytest.raises(ValueError, match='nonempty'):
        model.encode(clean, torch.zeros_like(mask))


def test_inverse_uses_only_heard_points_and_correct_db_scale(operands):
    a, _, mask, *_ = operands
    model = SpectralScene('none').eval()
    with torch.no_grad():
        model.scene[-1].weight.zero_()
        model.scene[-1].bias.zero_()
    basis = query_physics_basis(a[..., 4:])
    weight = (mask & a[..., 2].bool()).float()
    lhs = basis.transpose(-1, -2) * weight[:, None]
    expected = torch.linalg.solve(lhs @ basis + .1 * torch.eye(8), lhs @ (a[..., :1] * 3.)).squeeze(-1)
    torch.testing.assert_close(model.encode(a, mask), expected, rtol=1e-6, atol=1e-6)
    changed = a.clone()
    changed[:, -2:, 1] -= 10.
    changed[:, -2:, 0] = 17.
    # Censored bound/placeholder values are not inserted as point observations.
    torch.testing.assert_close(model.encode(changed, mask), expected, rtol=1e-6, atol=1e-6)


def test_geometry_only_cannot_see_rf_values_or_report_status(operands):
    a, _, mask, *_ = operands
    model = SpectralScene('none', geometry_only=True).eval()
    changed = a.clone()
    changed[..., :2] = torch.randn_like(changed[..., :2]) * 20.
    changed[..., 2:4] = 1. - changed[..., 2:4]
    torch.testing.assert_close(model.encode(a, mask), model.encode(changed, mask), rtol=0, atol=0)


def test_observed_censored_and_missing_likelihood_gradients(operands):
    *_, fspl, offset, target = operands
    mean = (fspl + offset).detach().requires_grad_()
    score = observation_nll(mean, target)
    score.backward()
    assert torch.isfinite(score) and torch.isfinite(mean.grad).all()
    assert mean.grad[target['heard']].abs().sum() > 0
    assert mean.grad[target['censored']].abs().sum() > 0
    assert torch.equal(mean.grad[~(target['heard'] | target['censored'])],
                       torch.zeros_like(mean.grad[~(target['heard'] | target['censored'])]))


def test_balanced_auxiliary_gradients_match_stated_coordinate_scope(operands):
    a, b, mask, query, basis, fspl, offset, target = operands
    model = SpectralScene('vicreg')
    loss, d = balanced_objective(model, (a, mask), (b, mask), query, basis,
                                 fspl, offset, target, diagnose_backbone=True)
    assert d['measurement_z_gradient'] > 0
    assert .01 < d['prior_weight'] < 100.
    torch.testing.assert_close(d['prior_weight'] * d['prior_z_gradient'],
                               .25 * d['measurement_z_gradient'], rtol=1e-5, atol=1e-7)
    if .01 < d['alignment_weight'] < 100.:
        torch.testing.assert_close(d['alignment_weight'] * d['alignment_z_gradient'],
                                   .25 * d['measurement_z_gradient'], rtol=1e-5, atol=1e-7)
    assert not d['prior_weight'].requires_grad and not d['alignment_weight'].requires_grad
    for key in ('measurement_backbone_gradient', 'weighted_prior_backbone_gradient',
                'weighted_alignment_backbone_gradient'):
        assert torch.isfinite(d[key]) and d[key] >= 0
    for key in ('prior_measurement_backbone_cosine', 'alignment_measurement_backbone_cosine'):
        assert torch.isfinite(d[key]) and abs(float(d[key])) <= 1. + 1e-5
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_actual_sigreg_module_and_step_state_survive_objective(operands):
    a, b, mask, query, basis, fspl, offset, target = operands
    model = SpectralScene('sigreg')
    prior = model.prior
    before = int(prior.sigreg.global_step)
    loss, diagnostics = balanced_objective(model, (a, mask), (b, mask), query, basis,
                                           fspl, offset, target)
    assert model.prior is prior
    assert int(prior.sigreg.global_step) == before + 2
    assert torch.isfinite(loss) and torch.isfinite(diagnostics['prior'])
    restored = SpectralScene('sigreg')
    restored.load_state_dict(model.state_dict())
    assert int(restored.prior.sigreg.global_step) == before + 2


def test_exact_collapsed_prior_is_reported_not_silently_disabled(operands):
    a, b, mask, query, basis, fspl, offset, target = operands
    model = SpectralScene('sigreg')
    a, b = a.clone(), b.clone()
    for value in (a, b):
        value[..., 0] = 0.
        value[..., 2] = 0.
        value[..., 3] = 1.
    with torch.no_grad():
        model.scene[-1].weight.zero_()
        model.scene[-1].bias.zero_()
    assert torch.equal(model.encode(a, mask), torch.zeros(len(a), 8))
    loss, d = balanced_objective(model, (a, mask), (b, mask), query, basis,
                                 fspl, offset, target)
    assert d['prior'] > 0 and d['prior_z_gradient'] == 0
    assert d['prior_stationary_nonzero'] == 1 and d['prior_weight'] > 0
    assert loss > d['measurement_nll']


def test_matched_regression_removes_alignment_only(operands):
    a, b, mask, query, basis, fspl, offset, target = operands
    joint = SpectralScene('vicreg')
    regression = SpectralScene('vicreg')
    regression.load_state_dict(joint.state_dict())
    jloss, jd = balanced_objective(joint, (a, mask), (b, mask), query, basis,
                                  fspl, offset, target, alignment=True)
    rloss, rd = balanced_objective(regression, (a, mask), (b, mask), query, basis,
                                  fspl, offset, target, alignment=False)
    torch.testing.assert_close(jd['measurement_nll'], rd['measurement_nll'])
    torch.testing.assert_close(jd['prior'], rd['prior'])
    torch.testing.assert_close(jd['prior_weight'], rd['prior_weight'])
    torch.testing.assert_close(jloss - rloss, jd['alignment_weight'] * jd['alignment'], atol=1e-6, rtol=1e-5)
    assert rd['alignment'] == 0 and rd['alignment_weight'] == 0


def test_unidentified_prior_and_nonfinite_queries_fail_closed(operands):
    with pytest.raises(ValueError, match='ineligible'):
        SpectralScene('ppr')
    a, _, mask, query, basis, fspl, offset, _ = operands
    model = SpectralScene('none')
    z = model.encode(a, mask)
    invalid = query.clone()
    invalid[:, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        model.render(z, invalid, basis, fspl, offset)
