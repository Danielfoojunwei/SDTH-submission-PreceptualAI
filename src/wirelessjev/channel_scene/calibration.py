"""Complex-channel calibration and identifiable phase/time gauges.

Receiver-specific STO/CPO correction is not a proof of absolute propagation
delay or inter-array coherence. Those claims need separate, bound evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


def immutable(a):
    a = np.asarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def frequency_spacing(frequency_offsets_hz):
    f = np.asarray(frequency_offsets_hz, dtype='f8')
    if f.ndim != 1 or len(f) < 3 or not np.isfinite(f).all():
        raise ValueError('Use at least three finite frequency offsets in Hz')
    delta = np.diff(f)
    if not (delta > 0).all() or not np.allclose(delta, delta[0], rtol=1.e-9, atol=1.e-7):
        raise ValueError('A uniform increasing frequency grid is required')
    return float(delta[0])


def sto_cpo_correct(cfr, sto_samples, cpo_rad, *, fft_size, shifted_bin_indices=None):
    """Official DICHASUS correction, AFTER FFT shift.

    Correction is exp(j*(2*pi*STO*k/fft_size+CPO)), where k is the original
    shifted-bin index. Subselected bins must retain their original indices.
    No gain, absolute-time or CFO/Doppler correction is invented.
    """
    h = np.asarray(cfr)
    if h.ndim < 2 or h.dtype.kind != 'c' or not np.isfinite(h).all():
        raise ValueError('Use finite complex link-by-frequency observations')
    if type(fft_size) is not int or fft_size < 1:
        raise ValueError('An explicit positive original FFT size is required')
    bins = np.arange(h.shape[-1]) if shifted_bin_indices is None else np.asarray(shifted_bin_indices)
    if (bins.shape != (h.shape[-1],) or bins.dtype.kind not in 'iu'
            or (bins < 0).any() or (bins >= fft_size).any()
            or len(np.unique(bins)) != len(bins)):
        raise ValueError('Invalid original shifted-frequency bin identities')
    if shifted_bin_indices is None and h.shape[-1] != fft_size:
        raise ValueError('Subselected CSI requires explicit original bin indices')
    sto, cpo = np.asarray(sto_samples, dtype='f8'), np.asarray(cpo_rad, dtype='f8')
    if sto.shape != (h.shape[-2],) or cpo.shape != sto.shape or not np.isfinite(sto).all() or not np.isfinite(cpo).all():
        raise ValueError('One finite actual STO/CPO estimate per link is required')
    phase = sto[:, None] * (2 * np.pi * bins / fft_size) + cpo[:, None]
    return immutable((h * np.exp(1j * phase)).astype(h.dtype))


@dataclass(frozen=True)
class PhaseCoherenceBudget:
    """Acceptance tolerances selected by a named protocol, not by its proof."""
    protocol_id: str
    maximum_phase_error_rad: float
    maximum_relative_timing_error_s: float

    def validate(self):
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise ValueError('A named protocol must select the coherence error budget')
        for value in (self.maximum_phase_error_rad, self.maximum_relative_timing_error_s):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Protocol coherence tolerances must be finite and nonnegative')
        return self


@dataclass(frozen=True)
class PhaseCoherenceProof:
    """Explicit evidence; calibration estimates alone cannot create it.

    Ideal-simulator zero bounds describe clock/reference assumptions, not
    total ray-tracer numerical error or agreement with real hardware.
    """
    evidence_id: str
    scope: str
    basis: str
    source_sha256: Mapping[str, str]
    groups: tuple[int, ...]
    maximum_phase_error_rad: float
    maximum_relative_timing_error_s: float

    def require(self, source_sha256, groups, *, ideal_simulation_admitted=False, budget=None):
        if (not self.evidence_id or self.scope not in ('per_group', 'global')
                or self.basis not in ('instrument_measurement', 'ideal_simulation')
                or not self.source_sha256 or not self.groups):
            raise ValueError('A concrete coherence evidence record is required')
        if any(source_sha256.get(k) != v for k, v in self.source_sha256.items()):
            raise ValueError('Coherence proof belongs to different source observations')
        if not set(np.asarray(groups).tolist()) <= set(self.groups):
            raise ValueError('Coherence proof does not cover every requested group')
        for value in (self.maximum_phase_error_rad, self.maximum_relative_timing_error_s):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Coherence error bounds must be explicit finite measurements/assumptions')
        if self.basis == 'ideal_simulation' and not ideal_simulation_admitted:
            raise ValueError('Ideal simulator phase must be explicitly admitted by the protocol')
        if not isinstance(budget, PhaseCoherenceBudget):
            raise ValueError('Coherent use requires an explicit protocol error budget')
        budget.validate()
        if (self.maximum_phase_error_rad > budget.maximum_phase_error_rad
                or self.maximum_relative_timing_error_s > budget.maximum_relative_timing_error_s):
            raise ValueError('Coherence evidence exceeds the protocol error budget')
        return self


def phase_curvature(cfr, frequency_offsets_hz, *, lags=(1, 2, 4), observed=None,
                    relative_amplitude_floor=1.e-6):
    """Per-link complex invariants to constant phase AND linear timing phase.

    u[k+2l]*u[k]*conj(u[k+l])**2 cancels exp(j*phi-j*2*pi*f*tau).
    It retains multipath phase curvature without assuming inter-link coherence.
    Unknown/deep-null triplets have a separate mask, never a fabricated phase.
    """
    h = np.asarray(cfr)
    frequency_spacing(frequency_offsets_hz)
    if h.dtype.kind != 'c' or h.shape[-1] != len(frequency_offsets_hz):
        raise ValueError('Invalid complex channel or frequency axes')
    mask = None
    if observed is not None:
        mask = np.asarray(observed)
        if mask.dtype.kind != 'b' or mask.shape != h.shape:
            raise ValueError('Observed mask must preserve complex observation axes')
        h = np.where(mask, h, 0.)
    if not np.isfinite(h).all():
        raise ValueError('Nonfinite observed complex channel')
    if not 0 <= relative_amplitude_floor < 1:
        raise ValueError('Use a relative amplitude threshold in[0,1)')
    amplitude = np.abs(h)
    valid = amplitude > relative_amplitude_floor * amplitude.max(-1, keepdims=True)
    if mask is not None:
        valid &= mask
    unit = np.divide(h, amplitude, out=np.zeros_like(h), where=valid)
    values, masks = [], []
    for lag in lags:
        if type(lag) is not int or lag < 1 or 2 * lag >= h.shape[-1]:
            raise ValueError('Curvature lag is outside the measured grid')
        keep = valid[..., 2 * lag:] & valid[..., :-2 * lag] & valid[..., lag:-lag]
        value = unit[..., 2 * lag:] * unit[..., :-2 * lag] * np.conj(unit[..., lag:-lag]) ** 2
        values.append(immutable(np.where(keep, value, 0.)))
        masks.append(immutable(keep))
    return tuple(values), tuple(masks)


@dataclass(frozen=True)
class CIRObservation:
    samples: np.ndarray
    delay_modulo_s: np.ndarray
    resolution_s: float
    ambiguity_period_s: float
    first_frequency_offset_hz: float
    absolute_reference_admitted: bool
    causality_certified_from_finite_band: bool = False


def spectrum_to_cir(cfr, frequency_offsets_hz, *, absolute_reference_admitted=False):
    """Finite-band DFT view, with circular delay ambiguity kept explicit.

    The complex taps use the FIRST sampled frequency as phase reference.
    Off-grid causal paths cause finite-window sidelobes around the circular
    boundary. Such sidelobes do not establish negative physical path delays.
    """
    h = np.asarray(cfr)
    spacing = frequency_spacing(frequency_offsets_hz)
    if h.dtype.kind != 'c' or h.shape[-1] != len(frequency_offsets_hz) or not np.isfinite(h).all():
        raise ValueError('Invalid complex spectrum')
    if type(absolute_reference_admitted) is not bool:
        raise ValueError('Absolute reference admission must be explicit')
    n = h.shape[-1]
    return CIRObservation(immutable(np.fft.ifft(h, axis=-1)),
        immutable(np.arange(n, dtype='f8') / (n * spacing)), 1 / (n * spacing),
        1 / spacing, float(np.asarray(frequency_offsets_hz)[0]), absolute_reference_admitted)


def render_cfr(coefficients, delays_s, frequency_offsets_hz, *, minimum_delay_s=0.):
    """Analytic causal Fourier sum. Delay bounds are inputs, never hidden labels."""
    a, tau = np.asarray(coefficients), np.asarray(delays_s, dtype='f8')
    f = np.asarray(frequency_offsets_hz, dtype='f8')
    minimum = np.asarray(minimum_delay_s, dtype='f8')
    if (a.dtype.kind != 'c' or a.shape != tau.shape or a.ndim < 1
            or f.ndim != 1 or not len(f) or not np.isfinite(f).all()
            or not np.isfinite(a).all() or not np.isfinite(tau).all()
            or not np.isfinite(minimum).all() or (minimum < 0).any()):
        raise ValueError('Malformed Fourier channel parameters')
    try:
        low = np.broadcast_to(minimum, tau.shape[:-1])[..., None]
    except ValueError as exc:
        raise ValueError('Minimum delay must match the public query axes') from exc
    if (tau < low).any():
        raise ValueError('Modeled path violates its nonnegative/public delay bound')
    return (a[..., :, None] * np.exp(-2j * np.pi * tau[..., :, None] * f)).sum(-2)
