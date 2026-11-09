"""Analytic/unit fixtures only; no wireless training or performance claims."""
from dataclasses import replace
from pathlib import Path
import ast
import hashlib
import json

import pytest
import torch

from wirelessjev.radiojepa.priors import (
    EppsPulley, PhysicalFactor, PriorRegularizer, SlicedEppsPulley,
    physical_prior_loss, regularization,
)


def embeddings(n=128, d=6):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(1021), dtype=torch.float64)


def factor(values, semantics, target, **kwargs):
    return PhysicalFactor(values, semantics, target, identified=True, **kwargs)


def quantiles(n=4096):
    return (torch.arange(n, dtype=torch.float64) + .5) / n


def test_exact_vendored_source_segments_and_license():
    root = Path(__file__).resolve().parents[1] / 'src/wirelessjev/radiojepa/vendor'
    receipt = json.loads((root / 'provenance.json').read_text())
    assert receipt['commit'] == 'ab836bf699a2be712dfcf980a5eb70d36391e876'
    assert receipt['license'] == 'MIT'
    assert receipt['configured_SIGReg_num_slices'] == 64
    assert receipt['upstream_default_num_slices'] == 1024
    for name, expected in receipt['vendored_files_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    assert b'Copyright (c) 2024 rbalestr-lab' in (root / 'LICENSE').read_bytes()
    found = {}
    for file in ('sigreg.py', 'vicreg.py'):
        source = (root / file).read_text()
        for node in ast.parse(source).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in receipt['components']:
                found[node.name] = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    assert found == {name: row['exact_source_segment_sha256'] for name, row in receipt['components'].items()}


def test_actual_epps_pulley_matches_direct_characteristic_function_reference():
    x = embeddings(127, 3)
    model = EppsPulley().double()
    t = model.t
    real = torch.cos(x[..., None] * t).mean(0)
    imag = torch.sin(x[..., None] * t).mean(0)
    # phi was registered by the actual constructor in its original dtype;
    # moving the module to double preserves those already rounded constants.
    reference = (((real - model.phi).square() + imag.square()) * model.weights).sum(-1) * len(x)
    torch.testing.assert_close(model(x), reference, rtol=0, atol=1e-12)


def test_sigreg_penalizes_constant_representation_and_keeps_step_in_checkpoint():
    x = embeddings(1024, 8).requires_grad_()
    active = PriorRegularizer('sigreg', num_slices=64).double()
    actual = active(x)
    collapsed = PriorRegularizer('sigreg', num_slices=64).double()(torch.zeros_like(x))
    assert collapsed > actual * 20
    actual.backward()
    assert torch.isfinite(x.grad).all()
    assert active.sigreg.global_step.item() == 1
    restored = PriorRegularizer('sigreg', num_slices=64).double()
    restored.load_state_dict(active.state_dict())
    torch.testing.assert_close(active(x.detach()), restored(x.detach()), rtol=0, atol=0)
    assert restored.sigreg.global_step.item() == 2


def test_sigreg_module_does_not_consume_training_rng():
    model = PriorRegularizer('sigreg', num_slices=64).double()
    before = torch.get_rng_state().clone()
    model(embeddings())
    assert torch.equal(torch.get_rng_state(), before)


def test_vicreg_component_matches_actual_variance_covariance_formula():
    x = embeddings().requires_grad_()
    centered = x - x.mean(0)
    std = (centered.var(0) + 1e-4).sqrt()
    cov = centered.T @ centered / (len(x) - 1)
    reference = 25 * torch.relu(1 - std).mean() / 2 + (cov - torch.diag(cov.diag())).square().sum() / x.shape[1]
    actual = regularization('vicreg', x)
    torch.testing.assert_close(actual, reference)
    assert regularization('vicreg', torch.zeros_like(x)) > actual
    actual.backward()
    assert torch.isfinite(x.grad).all()


def test_none_is_exact_differentiable_zero():
    x = embeddings().requires_grad_()
    value = regularization('none', x)
    assert value.item() == 0.
    value.backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert regularization('none', torch.full((128, 3), 1e38)).item() == 0.


def test_uniform_circle_harmonics_reject_gaussian_angles_and_constant_phase():
    u = quantiles()
    angle = 2 * torch.pi * u
    good = torch.stack((angle.cos(), angle.sin()), -1)
    gaussian_angle = .2 * torch.special.ndtri(u)
    wrong = torch.stack((gaussian_angle.cos(), gaussian_angle.sin()), -1)
    constant = torch.zeros_like(good); constant[:, 0] = 1.
    z = embeddings(len(u))
    loss = lambda x: physical_prior_loss(z, {'phase': factor(x, 'calibrated_unit_phasor', 'uniform_circle')})
    assert loss(good) < 1e-20
    assert loss(wrong) > .5
    assert loss(constant) == 1.


def test_half_laplace_positive_amplitude_targets_fixed_conditional_scale():
    u = quantiles()
    scale = torch.linspace(.5, 2., len(u), dtype=torch.float64)
    good = -torch.log1p(-u) * scale
    z = embeddings(len(u))
    f = factor(good, 'fixed_delay_basis_magnitude', 'half_laplace', scale=scale)
    correct = physical_prior_loss(z, {'amplitude': f})
    wrong = physical_prior_loss(z, {'amplitude': replace(f, values=scale)})
    assert correct < 1e-7 and wrong > .02


def test_shadowing_cf_rejects_same_variance_wrong_uniform_marginal():
    u = quantiles()
    location, scale = 3., 2.
    good = location + scale * torch.special.ndtri(u)
    wrong = location + scale * (u * 2 - 1) * (3 ** .5)
    z = embeddings(len(u))
    f = factor(good, 'shadowing_db', 'normal', scale=scale, location=location)
    assert physical_prior_loss(z, {'shadowing': f}) < 1e-7
    assert physical_prior_loss(z, {'shadowing': replace(f, values=wrong)}) > 1e-3


def test_dirichlet_airtime_is_per_domain_with_idle_and_preserves_rng():
    alpha = torch.tensor((2., 3., 4.), dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(17291)
        x = torch.distributions.Dirichlet(alpha).sample((2048,))
    z = embeddings(len(x))
    f = factor(x, 'contention_domain_airtime_with_idle', 'dirichlet', concentration=alpha)
    before = torch.get_rng_state().clone()
    correct = physical_prior_loss(z, {'airtime': f})
    assert torch.equal(torch.get_rng_state(), before)
    constant = (alpha / alpha.sum()).expand_as(x)
    assert correct < 1e-20
    assert physical_prior_loss(z, {'airtime': replace(f, values=constant)}) > 1e-4
    with pytest.raises(ValueError, match='sum to one'):
        physical_prior_loss(z, {'airtime': replace(f, values=x * .5)})


@pytest.mark.parametrize('kind', ('sigreg', 'vicreg'))
def test_generic_regularizers_leave_absolute_power_channel_outside_contract(kind):
    # These regularizers consume only the separately chosen learned z. No
    # LayerNorm or physical-power normalization is silently applied here.
    x = embeddings()
    before = x.clone()
    PriorRegularizer(kind).double()(x)
    assert torch.equal(x, before)


def test_ppr_refuses_unidentified_power_only_cir_phase_or_arbitrary_latents():
    z = embeddings()
    with pytest.raises(ValueError, match='named, identified'):
        regularization('ppr', z)
    f = PhysicalFactor(z.abs(), 'fixed_delay_basis_magnitude', 'half_laplace', scale=1.)
    with pytest.raises(ValueError, match='not identified'):
        regularization('ppr', z, {'basis': f})
    with pytest.raises(ValueError, match='Unsupported factor'):
        regularization('ppr', z, {'position': replace(f, identified=True, semantics='latent_position', target='uniform')})
    with pytest.raises(TypeError, match='PhysicalFactor'):
        regularization('ppr', z, {'latent': z})


def test_physical_targets_cannot_be_trainable_or_invalid():
    z = embeddings()
    amplitude = z.abs()
    f = factor(amplitude, 'fixed_delay_basis_magnitude', 'half_laplace', scale=torch.ones(6, requires_grad=True))
    with pytest.raises(ValueError, match='not a learned'):
        physical_prior_loss(z, {'a': f})
    with pytest.raises(ValueError, match='Invalid amplitude scale'):
        physical_prior_loss(z, {'a': replace(f, scale=0.)})
    with pytest.raises(ValueError, match='cannot be negative'):
        physical_prior_loss(z, {'a': replace(f, values=-amplitude, scale=1.)})
    with pytest.raises(ValueError, match='unit phasors'):
        physical_prior_loss(z, {'p': factor(torch.zeros(len(z), 2, dtype=z.dtype), 'calibrated_unit_phasor', 'uniform_circle')})
    with pytest.raises(ValueError, match='real numeric'):
        physical_prior_loss(z, {'a': replace(f, scale=torch.tensor(1+2j))})
    with pytest.raises(ValueError, match='real numeric'):
        physical_prior_loss(z, {'a': replace(f, scale=torch.tensor(True))})
    with pytest.raises(ValueError, match='overflowed'):
        physical_prior_loss(z, {'a': replace(f, values=torch.full_like(z, 1e308), scale=1e-320)})


def test_ppr_gradients_are_finite_for_identified_nonnegative_amplitudes():
    z = embeddings().requires_grad_()
    amplitude = (z.square() + .01).sqrt()
    value = regularization('ppr', z, {'a': factor(amplitude, 'fixed_delay_basis_magnitude', 'half_laplace', scale=1.)})
    value.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_function_convenience_warns_about_projection_state_reset():
    with pytest.warns(RuntimeWarning, match='Persist'):
        value = regularization('sigreg', embeddings())
    assert torch.isfinite(value)
    module = PriorRegularizer('sigreg')
    regularization(module, embeddings())
    assert module.sigreg.global_step.item() == 1


@pytest.mark.parametrize('bad', (torch.zeros(1, 3), torch.zeros(2, 0), torch.full((2, 3), float('nan'))))
def test_invalid_embedding_inputs_fail_closed(bad):
    with pytest.raises(ValueError, match='finite real'):
        regularization('none', bad)
