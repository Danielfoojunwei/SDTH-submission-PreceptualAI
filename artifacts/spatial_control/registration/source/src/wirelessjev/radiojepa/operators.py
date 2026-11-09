"""Conditional physical operators, with explicit units and identifiability.

These are port/channel operations, not a universal indoor frequency transform.
NumPy inputs return NumPy arrays; Tensor inputs retain autograd and device.
"""
from __future__ import annotations

import math
import numpy as np
import torch

LIGHT_SPEED_M_S = 299_792_458.
BOLTZMANN_J_K = 1.380649e-23


def _tensor(value, *, like=None):
    if isinstance(value, torch.Tensor):
        if value.dtype.is_complex:
            return value
        return value if value.is_floating_point() else value.to(torch.float64)
    if isinstance(value,np.ndarray) and not value.flags.writeable:
        # Torch cannot safely share an immutable NumPy byte owner. Preserve
        # the public-data immutability contract instead of relying on an
        # unsupported writable Tensor alias of that owner.
        value=np.array(value,copy=True)
    return torch.as_tensor(value, dtype=like.dtype if like is not None else torch.float64,
                           device=like.device if like is not None else None)


def _inputs(*values):
    like = next((v for v in values if isinstance(v, torch.Tensor)), None)
    result = tuple(_tensor(v, like=like) for v in values)
    if any(v.is_complex() for v in result):
        raise ValueError('This physical quantity must be real')
    return result, like is not None


def _return(value, tensor):
    if tensor:
        return value
    return value.detach().cpu().numpy()


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(name + ' must be finite')


def dbm_to_mw(dbm):
    (x,), tensor = _inputs(dbm)
    _finite(x, 'dBm')
    power = torch.pow(10., x / 10.)
    if not torch.isfinite(power).all():
        raise ValueError('Power conversion overflowed the chosen floating-point dtype')
    return _return(power, tensor)


def mw_to_dbm(power_mw, *, floor_mw=None):
    """Zero power is -inf dBm unless a strictly positive floor is supplied."""
    (x,), tensor = _inputs(power_mw)
    _finite(x, 'mW')
    if (x < 0).any():
        raise ValueError('Power cannot be negative')
    if floor_mw is not None:
        if not math.isfinite(floor_mw) or floor_mw <= 0:
            raise ValueError('A power floor must be positive and finite')
        x = x.clamp_min(floor_mw)
    return _return(10. * torch.log10(x), tensor)


def thermal_noise_dbm(bandwidth_hz, *, temperature_k=290., noise_figure_db=0.):
    (bandwidth, temperature, nf), tensor = _inputs(bandwidth_hz, temperature_k, noise_figure_db)
    for value, name in ((bandwidth, 'Bandwidth'), (temperature, 'Temperature'), (nf, 'Noise figure')):
        _finite(value, name)
    if (bandwidth <= 0).any() or (temperature <= 0).any():
        raise ValueError('Bandwidth and temperature must be positive')
    # kTB is W; the factor 1000 converts to mW. NF is an explicitly supplied
    # receiver property, not inferred from an RSSI number.
    return _return(10. * (math.log10(1000. * BOLTZMANN_J_K)
                          + torch.log10(temperature) + torch.log10(bandwidth)) + nf, tensor)


def fspl_frequency_shift_db(reference_hz, target_hz):
    """Received-power change for fixed distance and fixed isotropic gains.

    Material dispersion and antenna/device response are deliberately absent;
    they require separate calibrated operators or learned conditional terms.
    """
    (reference, target), tensor = _inputs(reference_hz, target_hz)
    _finite(reference, 'Reference frequency'); _finite(target, 'Target frequency')
    if (reference <= 0).any() or (target <= 0).any():
        raise ValueError('Carrier frequencies must be positive')
    return _return(20. * (torch.log10(reference) - torch.log10(target)), tensor)


def bandwidth_noise_shift_db(reference_hz, target_hz):
    """kTB noise change only; this does not specify signal power or PER."""
    (reference, target), tensor = _inputs(reference_hz, target_hz)
    _finite(reference, 'Reference bandwidth'); _finite(target, 'Target bandwidth')
    if (reference <= 0).any() or (target <= 0).any():
        raise ValueError('Bandwidths must be positive')
    return _return(10. * (torch.log10(target) - torch.log10(reference)), tensor)


def snr_after_bandwidth_change(snr_db, reference_hz, target_hz, *, signal_assumption):
    """Conditional SNR transform, never an unconditional width/PER operator."""
    (snr, reference, target), tensor = _inputs(snr_db, reference_hz, target_hz)
    _finite(snr, 'SNR')
    shift = _tensor(bandwidth_noise_shift_db(reference, target), like=snr)
    if signal_assumption == 'fixed_total_received_power':
        result = snr - shift
    elif signal_assumption == 'flat_received_power_spectral_density':
        result = snr + shift * 0.
    else:
        raise ValueError('Signal-power convention must be explicitly specified')
    return _return(result, tensor)


def reciprocal_channel(channel, *, frequency_hz, reverse_frequency_hz,
                       delta_time_s, coherence_time_s, calibrated=False):
    """Transpose of a calibrated same-frequency reciprocal propagation matrix.

    Raw RF-chain responses and FDD links do not satisfy this interface. A
    coherence-time check is an operating assumption, not a guarantee.
    """
    if calibrated is not True:
        raise ValueError('Reciprocity requires explicit RF-chain calibration')
    if (not all(math.isfinite(v) for v in (frequency_hz, reverse_frequency_hz,
                                           delta_time_s, coherence_time_s))
            or frequency_hz <= 0 or frequency_hz != reverse_frequency_hz
            or delta_time_s < 0 or coherence_time_s <= 0 or delta_time_s > coherence_time_s):
        raise ValueError('Reciprocity requires the same carrier within the declared coherence interval')
    x = channel if isinstance(channel, torch.Tensor) else np.asarray(channel)
    if x.ndim < 2:
        raise ValueError('Use an explicit receive-by-transmit port matrix')
    if isinstance(x, torch.Tensor):
        _finite(x, 'Channel')
        return x.transpose(-1, -2)
    if not np.isfinite(x).all():
        raise ValueError('Channel must be finite')
    return x.swapaxes(-1, -2)


def render_cir(coefficients, delay_s, frequency_offsets_hz):
    """Render complex path/tap coefficients at known frequency offsets.

    ``coefficients`` and ``delay_s`` end in a common path axis; output ends in
    a frequency axis. Delays may be relative to a sanitized phase gauge. Such
    relative delays cannot be used to claim absolute time of flight.
    """
    tensor = any(isinstance(v, torch.Tensor) for v in (coefficients, delay_s, frequency_offsets_hz))
    like = next((v for v in (coefficients, delay_s, frequency_offsets_hz)
                 if isinstance(v, torch.Tensor)), None)
    real_dtype = (torch.float32 if like is not None and like.dtype in (torch.float32, torch.complex64)
                  else torch.float64)
    device = like.device if like is not None else None
    c = torch.as_tensor(coefficients, device=device)
    c = c.to(torch.complex64 if real_dtype == torch.float32 else torch.complex128)
    delay = torch.as_tensor(delay_s, dtype=real_dtype, device=device)
    frequency = torch.as_tensor(frequency_offsets_hz, dtype=real_dtype, device=device)
    for value, name in ((c, 'Path coefficient'), (delay, 'Delay'), (frequency, 'Frequency')):
        _finite(value, name)
    if c.ndim < 1 or delay.ndim < 1 or c.shape[-1] != delay.shape[-1] or frequency.ndim != 1:
        raise ValueError('Use compatible path axes and a one-dimensional frequency grid')
    if (delay < 0).any():
        raise ValueError('Relative or absolute path delays must be nonnegative')
    c, delay = torch.broadcast_tensors(c, delay)
    phase = -2. * math.pi * delay[..., :, None] * frequency
    result = (c[..., :, None] * torch.exp(1j * phase)).sum(-2)
    return _return(result, tensor)


def validate_absolute_delays(delay_s, distance_m, *, absolute_toa_identified=False, tolerance_s=1e-12):
    if absolute_toa_identified is not True:
        raise ValueError('Sanitized/relative CSI phase does not identify absolute time of flight')
    (delay, distance), _ = _inputs(delay_s, distance_m)
    _finite(delay, 'Delay'); _finite(distance, 'Distance')
    if (distance < 0).any() or (delay < 0).any() or not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError('Invalid absolute delay/distance/tolerance')
    if (delay + tolerance_s < distance / LIGHT_SPEED_M_S).any():
        raise ValueError('A path arrives earlier than the geometric light-travel bound')
    return True


def validate_normalized_port_power(channel, *, normalized_ports=False, tolerance=1e-6):
    """Passivity for a declared power-wave port operator (spectral norm <=1).

    Arbitrary CSI/RSSI with AGC and antenna gains is not such an operator.
    """
    if normalized_ports is not True:
        raise ValueError('Raw CSI does not establish normalized power-wave ports')
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid passivity tolerance')
    x = _tensor(channel)
    _finite(x, 'Port operator')
    if x.ndim < 2:
        raise ValueError('Use an explicit port matrix')
    if (torch.linalg.matrix_norm(x, ord=2) > 1. + tolerance).any():
        raise ValueError('Normalized port operator violates passivity')
    return True


def incoherent_power_sum(power_mw, *, dim=-1, incoherent=False):
    if incoherent is not True:
        raise ValueError('Adding powers requires an incoherent/uncorrelated averaging assumption')
    (x,), tensor = _inputs(power_mw)
    _finite(x, 'Power')
    if (x < 0).any():
        raise ValueError('Power cannot be negative')
    return _return(x.sum(dim), tensor)


def coherent_power_sum(amplitudes, *, dim=-1):
    tensor = isinstance(amplitudes, torch.Tensor)
    x = torch.as_tensor(amplitudes)
    _finite(x, 'Complex amplitude')
    return _return(x.sum(dim).abs().square(), tensor)
