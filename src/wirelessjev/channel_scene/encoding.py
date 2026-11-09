"""Complex CSI encoders and causal Fourier prediction operators.

The default uses complex spectral phase curvature, not fabricated global
phase. Coherent and absolute-reference channels are explicitly evidence gated.
This is infrastructure; no training, promotion or model-loading side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .calibration import PhaseCoherenceProof

C = 299792458.


def _grid(offsets, count, device):
    f = torch.as_tensor(offsets, dtype=torch.float64, device=device)
    if f.shape != (count,) or count < 3 or not torch.isfinite(f).all():
        raise ValueError('Finite frequency offsets must preserve the measured grid')
    delta = f.diff()
    if not (delta > 0).all() or not torch.allclose(delta, delta[:1].expand_as(delta), rtol=1.e-9, atol=1.e-7):
        raise ValueError('Phase-curvature features require a uniform increasing grid')
    return f


def _clean(cfr, observed):
    if not isinstance(cfr, torch.Tensor) or cfr.ndim != 3 or cfr.dtype not in (torch.complex64, torch.complex128):
        raise ValueError('Complex CSI must have shape batch x link x frequency')
    if observed is None:
        mask = torch.ones_like(cfr.real, dtype=torch.bool)
    else:
        mask = torch.as_tensor(observed, device=cfr.device)
        if mask.dtype != torch.bool or mask.shape not in (cfr.shape, cfr.shape[:-1]):
            raise ValueError('Observation masks must be boolean link/bin masks')
        if mask.ndim == 2:
            mask = mask[..., None].expand_as(cfr)
    h = torch.where(mask, cfr, 0.)
    if not torch.isfinite(h).all():
        raise ValueError('Nonfinite observed complex CSI')
    return h, mask


def spectral_curvature(cfr, offsets, *, lags=(1, 2, 4), observed=None,
                       relative_amplitude_floor=1.e-6):
    """Differentiable per-link invariants to phase and sampling-origin shifts."""
    h, mask = _clean(cfr, observed)
    _grid(offsets, h.shape[-1], h.device)
    if not 0 <= relative_amplitude_floor < 1:
        raise ValueError('Invalid relative phase-validity threshold')
    amplitude = h.abs()
    valid = mask & (amplitude > relative_amplitude_floor * amplitude.amax(-1, keepdim=True))
    denominator = torch.where(valid, amplitude, torch.ones_like(amplitude))
    unit = torch.where(valid, h / denominator, 0.)
    values, validity = [], []
    for lag in lags:
        if type(lag) is not int or lag < 1 or 2 * lag >= h.shape[-1]:
            raise ValueError('Curvature lag is outside the measured frequency grid')
        keep = valid[..., 2 * lag:] & valid[..., :-2 * lag] & valid[..., lag:-lag]
        value = unit[..., 2 * lag:] * unit[..., :-2 * lag] * unit[..., lag:-lag].conj().square()
        values.append(torch.where(keep, value, 0.))
        validity.append(keep)
    return tuple(values), tuple(validity)


@dataclass(frozen=True)
class ComplexEncoding:
    scene: torch.Tensor
    link_tokens: torch.Tensor
    link_observed: torch.Tensor
    link_power_db: torch.Tensor
    mean_power_db: torch.Tensor
    phase_mode: str


class ComplexCSIEncoder(nn.Module):
    """Shared spectral processing and geometry-aware permutation-invariant sets.

    Group IDs are compared for equality only. Receiver/AP center distances
    provide a geometric prior, not an observed interference or material graph.
    The absolute referenced power coordinate bypasses shape normalization.
    """
    def __init__(self, hidden=48, heads=4, lags=(1, 2, 4), phase_mode='gauge_invariant',
                 power_floor=1.e-24):
        super().__init__()
        if (type(hidden) is not int or hidden < 8 or type(heads) is not int or heads < 1
                or hidden % heads or phase_mode not in ('gauge_invariant', 'relative_coherent', 'absolute_coherent')
                or not math.isfinite(power_floor) or power_floor <= 0):
            raise ValueError('Invalid complex-encoder configuration')
        if not lags or any(type(k) is not int or k < 1 for k in lags) or len(set(lags)) != len(lags):
            raise ValueError('Declare distinct positive curvature lags')
        self.hidden, self.heads, self.lags = hidden, heads, tuple(lags)
        self.phase_mode, self.power_floor = phase_mode, power_floor
        channels = 3 + 3 * len(lags) + (3 if phase_mode != 'gauge_invariant' else 0)
        self.spectral = nn.Sequential(nn.Conv1d(channels, 24, 5, padding=2), nn.SiLU(),
                                      nn.Conv1d(24, 24, 3, padding=1), nn.SiLU())
        self.link = nn.Sequential(nn.Linear(28, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.refine = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.pool_query = nn.Parameter(torch.zeros(hidden))
        self.distance_scale = nn.Parameter(torch.tensor(0.))
        self.same_group_bias = nn.Parameter(torch.tensor(0.))

    def forward(self, cfr, frequency_offsets_hz, *, group_ids, carrier_hz,
                infrastructure_xyz_m=None, observed=None, coherence_proof=None,
                coherence_budget=None, source_sha256=None, ideal_simulation_admitted=False,
                absolute_reference_admitted=False):
        h, mask = _clean(cfr, observed)
        b, links, bins = h.shape
        offsets = _grid(frequency_offsets_hz, bins, h.device)
        groups = torch.as_tensor(group_ids, device=h.device)
        if groups.shape == (links,): groups = groups[None].expand(b, -1)
        if groups.shape != (b, links) or groups.dtype not in (torch.int32, torch.int64):
            raise ValueError('Use integral physical group identities for each link')
        present = mask.any(-1)
        if (groups[present] < 0).any() or not present.any(-1).all():
            raise ValueError('No observed links or invalid identities; numerical fallback is required')
        carrier = torch.as_tensor(carrier_hz, dtype=torch.float64, device=h.device)
        if carrier.ndim == 0: carrier = carrier.expand(b)
        if carrier.shape != (b,) or not torch.isfinite(carrier).all() or (carrier <= 0).any():
            raise ValueError('One known positive carrier in Hz per scene is required')
        if self.phase_mode != 'gauge_invariant':
            if not isinstance(coherence_proof, PhaseCoherenceProof) or source_sha256 is None:
                raise ValueError('Coherent CSI encoding requires bound phase evidence')
            coherence_proof.require(source_sha256, groups[present].detach().cpu().numpy(),
                                    ideal_simulation_admitted=ideal_simulation_admitted,
                                    budget=coherence_budget)
            if self.phase_mode == 'absolute_coherent' and (
                    coherence_proof.scope != 'global' or absolute_reference_admitted is not True):
                raise ValueError('Absolute coherent features require an admitted absolute clock/reference')
        amplitude = h.abs()
        power = 20 * amplitude.clamp_min(math.sqrt(self.power_floor)).log10()
        count = mask.sum(-1).clamp_min(1)
        linear_power = amplitude.square()
        link_linear_power = (linear_power * mask).sum(-1) / count
        level = torch.where(present, 10 * link_linear_power.clamp_min(self.power_floor).log10(), 0.)
        shape = torch.where(mask, (power - level[..., None]) / 20., 0.)
        normalized_frequency = ((offsets - offsets.mean()) / (offsets[-1] - offsets[0])).to(shape.dtype)
        frequency = normalized_frequency.expand(b, links, -1)
        channels = [shape, frequency, mask.to(shape.dtype)]
        values, validity = spectral_curvature(h, offsets, lags=self.lags, observed=mask)
        for lag, value, valid in zip(self.lags, values, validity):
            channels.extend((F.pad(value.real, (0, 2 * lag)), F.pad(value.imag, (0, 2 * lag)),
                             F.pad(valid.to(shape.dtype), (0, 2 * lag))))
        if self.phase_mode != 'gauge_invariant':
            valid = mask & (amplitude > 1.e-6 * amplitude.amax(-1, keepdim=True))
            denominator = torch.where(valid, amplitude, torch.ones_like(amplitude))
            unit = torch.where(valid, h / denominator, 0.)
            if self.phase_mode == 'relative_coherent':
                reference = torch.zeros_like(h)
                for batch in range(b):
                    for group in groups[batch, present[batch]].unique():
                        chosen = groups[batch] == group
                        members = mask[batch, chosen].sum(0).clamp_min(1)
                        reference[batch, chosen] = h[batch, chosen].sum(0) / members
                refamp = reference.abs()
                valid &= refamp > 1.e-6 * refamp.amax(-1, keepdim=True)
                denominator = torch.where(valid, refamp, torch.ones_like(refamp))
                unit = unit * (reference / denominator).conj()
            unit = torch.where(valid, unit, 0.)
            channels.extend((unit.real, unit.imag, valid.to(shape.dtype)))
        dtype = self.link[0].weight.dtype
        spectral = torch.stack(channels, -2).reshape(b * links, len(channels), bins).to(dtype)
        encoded = self.spectral(spectral).mean(-1).reshape(b, links, 24)
        carrier_coordinate = (carrier / 1.e9).log().to(dtype)[:, None].expand(b, links)
        # Grid shape alone loses the physical delay scale. Retain actual bin
        # spacing and sampled bandwidth as Hz-derived continuous context.
        spacing_coordinate = ((offsets[1] - offsets[0]) / 1.e6).log().to(dtype).expand(b, links)
        bandwidth_coordinate = ((offsets[-1] - offsets[0]) / 1.e6).log().to(dtype).expand(b, links)
        tokens = self.link(torch.cat((encoded, (level / 30.).to(dtype)[..., None],
                                      carrier_coordinate[..., None],
                                      spacing_coordinate[..., None], bandwidth_coordinate[..., None]), -1))
        bias = self.same_group_bias * (groups[:, :, None] == groups[:, None, :]).to(dtype)
        if infrastructure_xyz_m is not None:
            xyz = torch.as_tensor(infrastructure_xyz_m, dtype=torch.float64, device=h.device)
            if xyz.shape == (links, 3): xyz = xyz[None].expand(b, -1, -1)
            if xyz.shape != (b, links, 3) or not torch.isfinite(xyz).all():
                raise ValueError('Public infrastructure centers must have shape link x XYZ')
            distance_wavelengths = torch.cdist(xyz, xyz) / (C / carrier)[:, None, None]
            bias = bias - F.softplus(self.distance_scale) * torch.log1p(distance_wavelengths).to(dtype)
        bias = bias.masked_fill(~present[:, None, :], -torch.inf)
        attention_mask = bias[:, None].expand(-1, self.heads, -1, -1).reshape(b * self.heads, links, links)
        coupled = self.attention(tokens, tokens, tokens, attn_mask=attention_mask, need_weights=False)[0]
        tokens = torch.where(present[..., None], tokens + self.refine(coupled), 0.)
        scores = (tokens * self.pool_query).sum(-1) / math.sqrt(self.hidden)
        weights = scores.masked_fill(~present, -torch.inf).softmax(-1)
        scene = (weights[..., None] * tokens).sum(1)
        mean_linear_power = (linear_power * mask).sum((-1, -2)) / mask.sum((-1, -2))
        mean_power = 10 * mean_linear_power.clamp_min(self.power_floor).log10()
        if not torch.isfinite(scene).all() or not torch.isfinite(tokens).all():
            raise FloatingPointError('Invalid complex-channel encoding')
        return ComplexEncoding(scene, tokens, present, level, mean_power, self.phase_mode)


class CausalFourierOperator(nn.Module):
    """Differentiable finite-path channel renderer, with no learned side effects.

    For measured channels, use a canonical relative reference and compare
    identifiable invariants. Nonnegative relative taps do not recover ToF.
    Absolute d/c bounds require explicitly public endpoint geometry and an
    admitted absolute reference; no hidden simulator ray labels are accepted.
    """
    def forward(self, coefficients, delays_s, frequency_offsets_hz, *,
                minimum_delay_s=0., reference='relative'):
        if (not isinstance(coefficients, torch.Tensor) or coefficients.dtype not in (torch.complex64, torch.complex128)
                or not isinstance(delays_s, torch.Tensor) or coefficients.shape != delays_s.shape
                or coefficients.ndim < 1 or not delays_s.is_floating_point()
                or not torch.isfinite(coefficients).all() or not torch.isfinite(delays_s).all()
                or reference not in ('relative', 'absolute_admitted')):
            raise ValueError('Use finite complex coefficients and real modeled delays')
        f = torch.as_tensor(frequency_offsets_hz, dtype=delays_s.dtype, device=delays_s.device)
        if f.ndim != 1 or not len(f) or not torch.isfinite(f).all() or coefficients.device != delays_s.device:
            raise ValueError('Invalid Fourier query frequency/device')
        minimum = torch.as_tensor(minimum_delay_s, dtype=delays_s.dtype, device=delays_s.device)
        if not torch.isfinite(minimum).all() or (minimum < 0).any():
            raise ValueError('Public/canonical minimum delays must be nonnegative')
        try:
            minimum = torch.broadcast_to(minimum, delays_s.shape[:-1])[..., None]
        except RuntimeError as exc:
            raise ValueError('Minimum delay must preserve query axes') from exc
        if (delays_s < minimum).any():
            raise ValueError('Modeled paths violate their nonnegative/public causal bound')
        if reference == 'relative' and (minimum > 0).any():
            raise ValueError('A physical d/c bound cannot be attached to an unknown relative time origin')
        phase = -2 * math.pi * delays_s[..., :, None] * f
        return (coefficients[..., :, None] * torch.polar(torch.ones_like(phase), phase)).sum(-2)
