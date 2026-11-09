"""Auditable regularizer comparisons; physical labels require identified factors.

PPR is a candidate conditional-prior test, not a theorem that these marginals
describe arbitrary radio scenes. A power-only decoder does not identify CIR
phase or physical scatterers and must not pass those factors as identified.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import torch
from torch import nn

from .vendor.sigreg import EppsPulley, SlicedEppsPulley
from .vendor.vicreg import VCRegLoss

KINDS = ('none', 'sigreg', 'vicreg', 'ppr')
UPSTREAM_COMMIT = 'ab836bf699a2be712dfcf980a5eb70d36391e876'


@dataclass(frozen=True)
class PhysicalFactor:
    """An explicitly declared factor and fixed conditional target parameters.

    ``identified`` is an evidence/experiment contract, not inferred by this
    class. Uniform circle and Dirichlet are hypotheses that must be ablated.
    Target scale/location/concentration cannot require gradients.
    """
    values: torch.Tensor
    semantics: str
    target: str
    identified: bool = False
    scale: torch.Tensor | float | None = None
    location: torch.Tensor | float | None = None
    concentration: torch.Tensor | None = None


def _embeddings(z):
    if (not isinstance(z, torch.Tensor) or z.ndim != 2 or z.shape[0] < 2
            or z.shape[1] < 1 or z.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(z).all()):
        raise ValueError('Use at least two finite real embedding rows')
    return z


def _parameter(value, reference, name, *, positive=False):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'An explicit fixed {name} is required')
    if isinstance(value, torch.Tensor) and value.requires_grad:
        raise ValueError(f'{name} must be fixed, not a learned prior target')
    raw = torch.as_tensor(value, device=reference.device)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError(f'{name} must be a real numeric parameter')
    p = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if not torch.isfinite(p).all() or (positive and (p <= 0).any()):
        raise ValueError(f'Invalid {name}')
    try:
        return torch.broadcast_to(p, reference.shape)
    except RuntimeError as exc:
        raise ValueError(f'{name} does not broadcast to its factor') from exc


def _cf_distance(values, kind):
    """Finite-frequency, differentiable ECF distance; no universal GOF claim."""
    x = values.reshape(values.shape[0], -1)
    if not torch.isfinite(x).all():
        raise ValueError('Standardized factor overflowed its arithmetic dtype')
    frequency = x.new_tensor((.25, .5, 1., 2., 3.))
    projection = x[..., None] * frequency
    real, imag = projection.cos().mean(0), projection.sin().mean(0)
    if kind == 'normal':
        target_real, target_imag = (-.5 * frequency.square()).exp(), torch.zeros_like(frequency)
    elif kind == 'half_laplace':
        # abs(Laplace(0,b)) / b is Exponential(1).
        target_real = 1. / (1. + frequency.square())
        target_imag = frequency / (1. + frequency.square())
    else:
        raise ValueError('Unknown characteristic-function target')
    weight = (-.5 * frequency.square()).exp()
    return (((real - target_real).square() + (imag - target_imag).square()) * weight).mean()


def _phase_distance(x):
    if x.ndim < 2 or x.shape[-1] != 2:
        raise ValueError('A unit phase uses a final [cos,sin] axis')
    if not torch.allclose(x.square().sum(-1), torch.ones_like(x[..., 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('Phase coordinates must be unit phasors')
    phase = torch.complex(x[..., 0], x[..., 1]).reshape(x.shape[0], -1)
    # Uniform circular target: every nonzero integer harmonic has mean zero.
    return torch.stack([(phase.pow(k).mean(0).abs().square()).mean() for k in range(1, 5)]).mean()


def _airtime_distance(x, concentration):
    if x.ndim != 2 or x.shape[1] < 2 or (x < 0).any() or (x > 1).any():
        raise ValueError('Airtime must contain exclusive-domain shares and idle')
    if not torch.allclose(x.sum(-1), torch.ones_like(x[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('Airtime including idle must sum to one per domain')
    if (not isinstance(concentration, torch.Tensor) or concentration.requires_grad
            or concentration.is_complex() or concentration.dtype == torch.bool):
        raise ValueError('Use fixed Dirichlet concentrations')
    alpha = concentration.detach().to(device='cpu', dtype=torch.float64)
    if alpha.shape != (x.shape[1],) or not torch.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError('One positive concentration per exclusive-domain share is required')
    # Deterministic reference integration, isolated from training RNG. This
    # finite Monte Carlo CF test is not an exact Dirichlet likelihood.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(17291)
        reference = torch.distributions.Dirichlet(alpha).sample((2048,)).to(x)
    generator = torch.Generator(device=x.device).manual_seed(17303)
    directions = torch.randn(x.shape[1], 16, dtype=x.dtype, device=x.device, generator=generator)
    directions = directions / directions.norm(dim=0).clamp_min(torch.finfo(x.dtype).tiny)
    frequency = x.new_tensor((.5, 1., 2., 4.))
    measured = (x @ directions)[..., None] * frequency
    expected = (reference @ directions)[..., None] * frequency
    return ((measured.cos().mean(0) - expected.cos().mean(0)).square()
            + (measured.sin().mean(0) - expected.sin().mean(0)).square()).mean()


def physical_prior_loss(z, factors):
    """Reject unidentifiable or unlabeled physical-factor regularization."""
    _embeddings(z)
    if not isinstance(factors, Mapping) or not factors:
        raise ValueError('PPR requires named, identified physical factors')
    terms = []
    for name, factor in factors.items():
        if not isinstance(name, str) or not name or not isinstance(factor, PhysicalFactor):
            raise TypeError('Use named PhysicalFactor objects, not arbitrary latent coordinates')
        if factor.identified is not True:
            raise ValueError(f'Factor {name} is not identified; PPR is ineligible')
        x = factor.values
        if (not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[0] != z.shape[0]
                or not x.is_floating_point() or x.device != z.device or x.dtype != z.dtype
                or not torch.isfinite(x).all()):
            raise ValueError(f'Factor {name} must preserve finite sample/device/dtype axes')
        pair = factor.semantics, factor.target
        if pair == ('calibrated_unit_phasor', 'uniform_circle'):
            term = _phase_distance(x)
        elif pair == ('fixed_delay_basis_magnitude', 'half_laplace'):
            if (x < 0).any():
                raise ValueError('Basis magnitudes cannot be negative')
            term = _cf_distance(x / _parameter(factor.scale, x, 'amplitude scale', positive=True), 'half_laplace')
        elif pair == ('shadowing_db', 'normal'):
            location = _parameter(0. if factor.location is None else factor.location, x, 'shadowing location')
            scale = _parameter(factor.scale, x, 'shadowing scale', positive=True)
            term = _cf_distance((x - location) / scale, 'normal')
        elif pair == ('contention_domain_airtime_with_idle', 'dirichlet'):
            term = _airtime_distance(x, factor.concentration)
        else:
            raise ValueError(f'Unsupported factor semantics/target: {pair}')
        terms.append(term)
    # Deliberately no blind cross-factor decorrelation: geometry, K, shadowing,
    # velocities and airtime can be physically dependent.
    result = torch.stack(terms).mean() + z.mul(0.).sum()
    if not torch.isfinite(result):
        raise FloatingPointError('Nonfinite physical-prior arithmetic')
    return result


class PriorRegularizer(nn.Module):
    """Persistent actual SIGReg or VCReg component, or explicit PPR candidates.

    Keep this module in the model/checkpoint so SIGReg's projection step is
    preserved. The CPU comparison uses64 slices; upstream default is1024.
    Different regularizer scales are not asserted equivalent.
    """
    def __init__(self, kind='none', *, num_slices=64):
        super().__init__()
        if kind not in KINDS or type(num_slices) is not int or num_slices < 1:
            raise ValueError('Unknown regularizer or invalid projection count')
        self.kind, self.num_slices = kind, num_slices
        self.sigreg = SlicedEppsPulley(num_slices=num_slices) if kind == 'sigreg' else None
        self.vcreg = VCRegLoss() if kind == 'vicreg' else None

    def forward(self, z, factors=None):
        z = _embeddings(z)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            raise RuntimeError('Distributed regularizer behavior is not qualified by this CPU protocol')
        if self.kind == 'none':
            return z.mul(0.).sum()
        if self.kind == 'sigreg':
            # Upstream creates random projections in torch's default dtype,
            # without a dtype argument. Preserve those exact class bodies and
            # explicitly match their arithmetic dtype at this adapter boundary.
            dtype = torch.get_default_dtype()
            value = self.sigreg.to(device=z.device, dtype=dtype)(z.to(dtype=dtype))
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite SIGReg arithmetic')
            return value.to(dtype=z.dtype)
        if self.kind == 'vicreg':
            value = self.vcreg(z)
            if not torch.isfinite(value):
                raise FloatingPointError('Nonfinite VCReg arithmetic')
            return value
        return physical_prior_loss(z, factors)


def regularization(kind, z, factors=None):
    """One-shot convenience; pass a persistent module for training SIGReg.

    A string SIGReg call intentionally starts at projection step zero and emits
    a warning. It must not silently masquerade as stateful upstream training.
    """
    if isinstance(kind, PriorRegularizer):
        return kind(z, factors)
    if kind == 'sigreg':
        warnings.warn('Persist PriorRegularizer("sigreg") in the model to advance/checkpoint projection steps',
                      RuntimeWarning, stacklevel=2)
    return PriorRegularizer(kind)(z, factors)
