"""Wireless measurement likelihoods with explicit units/support.

No family is assumed superior without held-out calibration. Missing readings
are not measurements; callers must exclude them or mark a real censor bound.
"""
from __future__ import annotations

import math
import torch
from .operators import _inputs, _return, _finite


def _gaussian(mean, std):
    _finite(mean, 'Mean'); _finite(std, 'Standard deviation')
    if (std <= 0).any():
        raise ValueError('Standard deviation must be positive')


def gaussian_tobit_log_prob(value, mean, std, *, censored, upper_bound):
    """Observed density or the genuine left-censor event Y<=upper_bound."""
    (value, mean, std, bound), tensor = _inputs(value, mean, std, upper_bound)
    _gaussian(mean, std)
    flag = torch.as_tensor(censored, device=mean.device)
    if flag.dtype != torch.bool:
        raise ValueError('Censor flags must be Boolean')
    value, mean, std, bound, flag = torch.broadcast_tensors(value, mean, std, bound, flag)
    if not torch.isfinite(value[~flag]).all() or not torch.isfinite(bound[flag]).all():
        raise ValueError('Observed values and actual censor bounds must be finite')
    observed = torch.where(flag, mean, value)
    z = (observed - mean) / std
    density = -.5 * z.square() - std.log() - .5 * math.log(2 * math.pi)
    # Unused placeholders cannot poison either the value or its gradient.
    safe_bound = torch.where(flag, bound, mean)
    result = torch.where(flag, torch.special.log_ndtr((safe_bound - mean) / std), density)
    return _return(result, tensor)


def _log_sub_exp(a, b):
    # a>=b. expm1 preserves very small interval probability differences.
    finite_a = torch.isfinite(a)
    delta = torch.where(finite_a, b - a, torch.full_like(a, -torch.inf))
    return torch.where(finite_a, a + torch.log(-torch.expm1(delta)), a)


def gaussian_interval_log_prob(lower, upper, mean, std):
    """Log P(lower<Y<=upper), supporting extreme tails and infinite endpoints."""
    (lower, upper, mean, std), tensor = _inputs(lower, upper, mean, std)
    _gaussian(mean, std)
    lower, upper, mean, std = torch.broadcast_tensors(lower, upper, mean, std)
    if torch.isnan(lower).any() or torch.isnan(upper).any() or (lower > upper).any():
        raise ValueError('Intervals must be ordered, with no NaN endpoints')
    empty = lower == upper
    # Standardizing infinity can give NaN gradients with respect to std even
    # when the limiting probability is well defined. Evaluate only finite
    # standardized endpoints, then substitute their exact CDF/SF limits.
    lower = torch.where(empty, mean, lower)
    upper = torch.where(empty, mean + std, upper)
    finite_lo, finite_hi = torch.isfinite(lower), torch.isfinite(upper)
    lo = (torch.where(finite_lo, lower, mean) - mean) / std
    hi = (torch.where(finite_hi, upper, mean) - mean) / std
    ninf = torch.full_like(lo, -torch.inf)
    zero = torch.zeros_like(lo)
    cdf_hi = torch.where(finite_hi, torch.special.log_ndtr(hi), zero)
    cdf_lo = torch.where(finite_lo, torch.special.log_ndtr(lo), ninf)
    sf_lo = torch.where(finite_lo, torch.special.log_ndtr(-lo), zero)
    sf_hi = torch.where(finite_hi, torch.special.log_ndtr(-hi), ninf)
    # Select stable tail BEFORE subtraction; computing both log differences
    # can create a 0*NaN gradient from an unused rounded-to-zero probability.
    positive = finite_lo & (lo > 0)
    result = _log_sub_exp(torch.where(positive, sf_lo, cdf_hi),
                          torch.where(positive, sf_hi, cdf_lo))
    result = torch.where(empty, ninf, result)
    return _return(result, tensor)


def rician_log_prob(amplitude, nu, sigma):
    """Rician amplitude: nu>=0 and per-quadrature Gaussian sigma>0."""
    (r, nu, sigma), tensor = _inputs(amplitude, nu, sigma)
    for x, name in ((r, 'Amplitude'), (nu, 'LOS amplitude'), (sigma, 'Quadrature sigma')):
        _finite(x, name)
    if (r < 0).any() or (nu < 0).any() or (sigma <= 0).any():
        raise ValueError('Invalid Rician support/parameters')
    z = r * nu / sigma.square()
    # i0e removes exp(z); combining that term with the exponent avoids large
    # cancelling squares when r and nu are close at high K.
    result = r.log() - 2 * sigma.log() - .5 * ((r - nu) / sigma).square() + torch.special.i0e(z).log()
    return _return(result, tensor)


def nakagami_log_prob(amplitude, m, omega):
    """Nakagami amplitude with m>=1/2 and mean-square amplitude omega>0."""
    (r, m, omega), tensor = _inputs(amplitude, m, omega)
    for x, name in ((r, 'Amplitude'), (m, 'Fading shape'), (omega, 'Mean-square amplitude')):
        _finite(x, name)
    if (r < 0).any() or (m < .5).any() or (omega <= 0).any():
        raise ValueError('Invalid radio Nakagami support/parameters')
    result = (math.log(2.) + m * m.log() - torch.lgamma(m) - m * omega.log()
              + torch.special.xlogy(2 * m - 1, r) - m * r.square() / omega)
    return _return(result, tensor)


def lognormal_log_prob(value, log_mean, log_std):
    """Positive quantity with natural-log mean/std (not dB parameters)."""
    (x, mean, std), tensor = _inputs(value, log_mean, log_std)
    _finite(x, 'Positive quantity'); _gaussian(mean, std)
    if (x < 0).any():
        raise ValueError('A log-normal quantity cannot be negative')
    safe = torch.where(x > 0, x, torch.ones_like(x))
    result = -.5 * ((safe.log() - mean) / std).square() - std.log() - safe.log() - .5 * math.log(2 * math.pi)
    return _return(torch.where(x > 0, result, torch.full_like(result, -torch.inf)), tensor)


def circular_loss(predicted_phase, observed_phase):
    (a, b), tensor = _inputs(predicted_phase, observed_phase)
    _finite(a, 'Predicted phase'); _finite(b, 'Observed phase')
    return _return(1 - torch.cos(a - b), tensor)


def gilbert_elliott_filter(losses, transition, loss_probability, initial, *, observed=None):
    """Bayesian filtering of one discrete-time ACK/loss trace.

    ``transition[i,j]`` is P(next=j|current=i). Missing events make no emission
    update. The model is not inferred from RSSI without actual packet labels.
    Returns Tensor arrays and a differentiable total log likelihood.
    """
    like = next((v for v in (transition, loss_probability, initial)
                 if isinstance(v, torch.Tensor)), torch.tensor(0., dtype=torch.float64))
    if like.is_complex():
        raise ValueError('Probabilities must be real')
    if not like.is_floating_point():
        like = like.to(torch.float64)
    trans = torch.as_tensor(transition, dtype=like.dtype, device=like.device)
    prob = torch.as_tensor(loss_probability, dtype=like.dtype, device=like.device)
    state = torch.as_tensor(initial, dtype=like.dtype, device=like.device)
    events = torch.as_tensor(losses, device=like.device)
    mask = torch.ones_like(events, dtype=torch.bool) if observed is None else torch.as_tensor(observed, device=like.device)
    if (trans.shape != (2, 2) or prob.shape != (2,) or state.shape != (2,)
            or events.ndim != 1 or mask.shape != events.shape or mask.dtype != torch.bool):
        raise ValueError('Use two states and one Boolean/integer loss trace')
    for x, name in ((trans, 'Transition'), (prob, 'Loss probability'), (state, 'Initial probability')):
        _finite(x, name)
    if ((trans < 0).any() or (trans > 1).any() or (state < 0).any()
            or (prob < 0).any() or (prob > 1).any()
            or not torch.allclose(trans.sum(-1), torch.ones(2, dtype=like.dtype, device=like.device))
            or not torch.allclose(state.sum(), state.new_tensor(1.))
            or not ((events[mask] == 0) | (events[mask] == 1)).all()):
        raise ValueError('Invalid transition/probability/event support')
    log_state, log_trans = state.log(), trans.log()
    filtered, predicted, terms = [], [], []
    for i in range(len(events)):
        # The initial distribution describes the first emission; transitions
        # occur between events, rather than silently before the first event.
        if i:
            log_state = torch.logsumexp(log_state[:, None] + log_trans, dim=0)
        predicted.append(log_state.exp())
        if mask[i]:
            emission = prob if events[i] == 1 else 1 - prob
            joint = log_state + emission.log()
            term = torch.logsumexp(joint, dim=0)
            if not torch.isfinite(term):
                raise ValueError('Observed event has zero probability under both states')
            log_state = joint - term
            terms.append(term)
        filtered.append(log_state.exp())
    shape = (0, 2)
    zero = trans.sum() * 0.
    return {'filtered': torch.stack(filtered) if filtered else trans.new_empty(shape),
            'predicted': torch.stack(predicted) if predicted else trans.new_empty(shape),
            'log_likelihood': torch.stack(terms).sum() if terms else zero}
