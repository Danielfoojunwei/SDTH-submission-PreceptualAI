"""Numerical parity against the actual frozen Torch code; no RF/hardware data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import log_ndtr
import torch

from wirelessjev.hierarchical_radio.model import interval_moments as torch_moments
from wirelessjev.radiojepa.likelihoods import gaussian_interval_log_prob as torch_log_prob
from wirelessjev.smartcar_numpy.likelihoods import gaussian_interval_log_prob, interval_moments


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


@pytest.mark.parametrize("lower,upper", [
    (-np.inf, np.inf), (-np.inf, -40.), (40., np.inf), (-41., -40.), (40., 41.),
    (-8., -7.999), (7.999, 8.), (-.25, .75), (-.0005, .0005), (.123, .1231),
    (0., 0.), (-np.inf, -np.inf), (np.inf, np.inf)])
def test_interval_log_mass_matches_actual_frozen_torch(lower, upper):
    actual = gaussian_interval_log_prob(lower, upper, .2, 1.3)
    expected = torch_log_prob(tensor(lower), tensor(upper), tensor(.2), tensor(1.3)).numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert actual.dtype == np.float64 and actual.shape == ()


def test_broadcasted_interval_log_mass_matches_frozen_torch():
    lower = np.array([[-np.inf, -8., 1., 10.]])
    upper = np.array([[np.inf, -7.9, 1.001, 10.1]])
    mean, std = np.array([[-2.], [0.], [2.]]), np.array([[.5], [1.], [2.]])
    actual = gaussian_interval_log_prob(lower, upper, mean, std)
    expected = torch_log_prob(tensor(lower), tensor(upper), tensor(mean), tensor(std)).numpy()
    assert actual.shape == (3, 4)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("lower,upper", [
    (-np.inf, np.inf), (-np.inf, -40.), (40., np.inf), (-41., -40.), (40., 41.),
    (-8., -7.999), (7.999, 8.), (-.25, .75), (-.0005, .0005), (.123, .1231)])
def test_conditional_moments_match_actual_frozen_torch(lower, upper):
    actual = interval_moments(.2, 1.3, lower, upper)
    expected = torch_moments(tensor(.2), tensor(1.3), tensor(lower), tensor(upper))
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference.numpy(), rtol=2e-7, atol=2e-8)
        assert value.dtype == np.float64


def test_mixed_point_censor_interval_and_unknown_broadcast_parity():
    mean = np.array([[-2.], [0.], [2.]])
    std = np.array([[.5], [1.], [2.]])
    lower = np.broadcast_to(np.array([.123456789, -np.inf, 1., -np.inf]), (3, 4))
    upper = np.broadcast_to(np.array([.123456789, -.2, 1.2, np.inf]), (3, 4))
    mask = np.broadcast_to(np.array([True, False, False, False]), (3, 4))
    actual = interval_moments(mean, std, lower, upper, point_mask=mask)
    expected = torch_moments(tensor(mean), tensor(std), tensor(lower.copy()), tensor(upper.copy()),
                             point_mask=torch.as_tensor(mask.copy()))
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference.numpy(), rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(actual[0][:, 0], lower[:, 0])
    np.testing.assert_array_equal(actual[1][:, 0], np.zeros(3))
    np.testing.assert_array_equal(actual[0][:, -1], mean[:, 0])
    np.testing.assert_array_equal(actual[1][:, -1], np.square(std[:, 0]))


def test_f32_inputs_are_evaluated_as_f64_working_moments():
    mean, std = np.array([-.25], dtype="f4"), np.array([1.5], dtype="f4")
    lower, upper = np.array([8.], dtype="f4"), np.array([8.001], dtype="f4")
    actual = interval_moments(mean, std, lower, upper)
    expected = torch_moments(*(tensor(v) for v in (mean, std, lower, upper)))
    for value, reference in zip(actual, expected):
        assert value.dtype == np.float64
        np.testing.assert_allclose(value, reference.numpy(), rtol=2e-7, atol=2e-8)


def test_tail_mass_uses_log_cdf_without_underflow_or_fake_floor():
    actual = gaussian_interval_log_prob([-np.inf, 40.], [-40., np.inf], 0., 1.)
    assert np.isfinite(actual).all() and (actual < -800.).all()
    np.testing.assert_array_equal(actual, np.full(2, log_ndtr(-40.)))
    assert gaussian_interval_log_prob(1., 1., 0., 1.) == -np.inf


@pytest.mark.parametrize("lower,upper", [(-2., .7), (.2, .201), (-9., -8.)])
def test_conditional_moments_against_independent_quadrature(lower, upper):
    density = lambda x: np.exp(-.5 * x * x) / np.sqrt(2. * np.pi)
    mass = quad(density, lower, upper, epsabs=1e-30)[0]
    first = quad(lambda x: x * density(x), lower, upper, epsabs=1e-30)[0] / mass
    second = quad(lambda x: (x - first) ** 2 * density(x), lower, upper, epsabs=1e-30)[0] / mass
    actual = interval_moments(0., 1., lower, upper)
    np.testing.assert_allclose(actual[0], first, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual[1], second, rtol=1e-5, atol=2e-10)


@pytest.mark.parametrize("lower,upper,mean,std", [
    (1., 0., 0., 1.), (np.nan, 0., 0., 1.), (0., np.nan, 0., 1.),
    (0., 1., np.nan, 1.), (0., 1., np.inf, 1.), (0., 1., 0., 0.),
    (0., 1., 0., -1.), (0., 1., 0., np.inf), (0., 1., 0., np.nan),
    (0j, 1., 0., 1.)])
def test_invalid_gaussian_or_interval_is_rejected(lower, upper, mean, std):
    with pytest.raises(ValueError):
        gaussian_interval_log_prob(lower, upper, mean, std)
    with pytest.raises(ValueError):
        interval_moments(mean, std, lower, upper)


@pytest.mark.parametrize("lower,upper", [(0., 0.), (np.inf, np.inf), (-np.inf, -np.inf)])
def test_empty_intervals_cannot_silently_turn_into_point_observations(lower, upper):
    with pytest.raises(ValueError):
        interval_moments(0., 1., lower, upper)


@pytest.mark.parametrize("mask", [[1], [0.], [True, False], np.array(True)])
def test_point_mask_must_be_boolean_with_exact_broadcast_shape(mask):
    with pytest.raises(ValueError):
        interval_moments(np.array([0.]), np.array([1.]), np.array([.5]), np.array([.5]), point_mask=mask)


@pytest.mark.parametrize("lower,upper", [(.1, .2), (np.inf, np.inf), (-np.inf, -np.inf)])
def test_explicit_points_require_equal_finite_endpoints(lower, upper):
    with pytest.raises(ValueError):
        interval_moments(0., 1., lower, upper, point_mask=np.array(True))


def test_no_mutation_of_readonly_inputs_and_same_curvature_clip():
    values = [np.array(x, dtype="f8") for x in ([0., 0., 0.], [1., 2., .5], [-np.inf, 4., -.001], [np.inf, 4.1, .001])]
    before = [value.copy() for value in values]
    for value in values:
        value.setflags(write=False)
    expectation, variance = interval_moments(*values)
    assert np.isfinite(expectation).all() and np.isfinite(variance).all()
    assert (variance >= 0).all() and (variance <= np.square(values[1])).all()
    curvature = np.clip(1. - variance / np.square(values[1]), 0., 1.) / np.square(values[1])
    assert (curvature >= 0).all() and (curvature <= 1. / np.square(values[1])).all()
    for original, value in zip(before, values):
        np.testing.assert_array_equal(value, original)


def test_runtime_import_and_execution_with_torch_imports_blocked():
    root = Path(__file__).resolve().parents[2]
    script = r'''
import builtins,json
original = builtins.__import__
def blocked(name,*args,**kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("Torch deliberately unavailable for NumPy runtime check")
    return original(name,*args,**kwargs)
builtins.__import__ = blocked
from wirelessjev.smartcar_numpy.likelihoods import gaussian_interval_log_prob,interval_moments
mass=gaussian_interval_log_prob(-float("inf"),float("inf"),2.,3.)
mean,variance=interval_moments(2.,3.,-float("inf"),float("inf"))
print(json.dumps([float(mass),float(mean),float(variance)]))
'''
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=environment,
                            text=True, capture_output=True, timeout=30, check=True)
    assert json.loads(result.stdout) == [0., 2., 9.]
