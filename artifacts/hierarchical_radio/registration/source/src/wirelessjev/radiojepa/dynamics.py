"""Declared channel priors; none is an exact law for every radio scene."""
from __future__ import annotations
import math
import numpy as np
import torch
from .operators import _inputs, _return, _finite


def ou_transition(mean, variance, dt_s, tau_s, stationary_variance, *, equilibrium=0.):
    """Exact scalar OU moments for irregular temporal intervals."""
    (mean, var, dt, tau, stationary, eq), tensor = _inputs(
        mean, variance, dt_s, tau_s, stationary_variance, equilibrium)
    for value in (mean, var, dt, tau, stationary, eq): _finite(value, 'OU parameter')
    if (var < 0).any() or (dt < 0).any() or (tau <= 0).any() or (stationary < 0).any():
        raise ValueError('Invalid OU interval/variance/time constant')
    decay = torch.exp(-dt / tau)
    next_mean = eq + decay * (mean - eq)
    next_var = decay.square() * var + stationary * (-torch.expm1(-2 * dt / tau))
    return _return(next_mean, tensor), _return(next_var, tensor)


def spatial_shadow_covariance(positions_m, length_scale_m, variance_db2, *, jitter=0.):
    """Exponential Euclidean field covariance, preserving spatial loop closure.

    Using total distance travelled as a coordinate would not make revisits to
    the same static scene share the same random field.
    """
    (positions, length, var), tensor = _inputs(positions_m, length_scale_m, variance_db2)
    for value in (positions, length, var): _finite(value, 'Spatial covariance parameter')
    if positions.ndim != 2 or positions.shape[1] not in (2, 3) or length.ndim or var.ndim:
        raise ValueError('Use N physical XY/XYZ positions and scalar field parameters')
    if length <= 0 or var < 0 or not math.isfinite(jitter) or jitter < 0:
        raise ValueError('Invalid spatial covariance scale/variance/jitter')
    result = var * torch.exp(-torch.cdist(positions, positions) / length)
    if jitter:
        result = result + jitter * torch.eye(len(positions), dtype=result.dtype, device=result.device)
    return _return(result, tensor)


class _BesselJ0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return torch.special.bessel_j0(value)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        return -gradient * torch.special.bessel_j1(value)


def jakes_correlation(doppler_hz, dt_s):
    """J0(2*pi*fD*dt) under isotropic diffuse 2D scattering assumptions.

    The custom Bessel implementation supports first-order autograd.
    """
    (doppler, dt), tensor = _inputs(doppler_hz, dt_s)
    _finite(doppler, 'Doppler'); _finite(dt, 'Time lag')
    if (doppler < 0).any():
        raise ValueError('Maximum Doppler must be nonnegative')
    argument = 2 * math.pi * doppler * dt
    _finite(argument, 'Doppler phase')
    return _return(_BesselJ0.apply(argument), tensor)


def jakes_ar_transition(mean, variance, doppler_hz, dt_s, stationary_variance):
    """Single-lag AR approximation; not the full Jakes Doppler spectrum."""
    tensor = any(isinstance(v, torch.Tensor) for v in
                 (mean, variance, doppler_hz, dt_s, stationary_variance))
    like = next((v for v in (mean, variance, doppler_hz, dt_s, stationary_variance)
                 if isinstance(v, torch.Tensor)), None)
    dtype = (torch.float32 if like is not None and like.dtype in (torch.float32,torch.complex64)
             else torch.float64)
    device = like.device if like is not None else None
    mean = torch.as_tensor(mean, device=device)
    if not mean.is_complex(): mean = mean.to(dtype)
    real_parameters = (variance, doppler_hz, dt_s, stationary_variance)
    if any(v.is_complex() if isinstance(v,torch.Tensor) else np.iscomplexobj(v)
           for v in real_parameters):
        raise ValueError('Fading variance, Doppler and time must be real')
    converted = tuple(torch.as_tensor(v, dtype=dtype, device=device)
                      for v in real_parameters)
    var, doppler, dt, stationary = converted
    for value in (mean, var, dt, stationary): _finite(value, 'AR parameter')
    if (var < 0).any() or (dt < 0).any() or (stationary < 0).any():
        raise ValueError('Invalid AR time/variance')
    rho = jakes_correlation(doppler, dt).clamp(-1., 1.)
    return (_return(rho * mean, tensor),
            _return(rho.square() * var + (1 - rho.square()) * stationary, tensor))
