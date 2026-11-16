"""Float64 NumPy/SciPy counterpart of the frozen radio interval helpers.

These functions preserve the reference's event semantics and variance clipping.
An explicit point is a different event from an empty interval. An unobserved
whole-line interval contributes no information; no detector floor, artificial
bin width, or extra censor observation is introduced here.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import log_ndtr


_LOG_2PI = math.log(2.0 * math.pi)


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(name + " must be real numeric data")
    return np.asarray(array, dtype=np.float64)


def _inputs(mean, std, lower, upper):
    mean, std, lower, upper = np.broadcast_arrays(
        *(_real(value, name) for value, name in (
            (mean, "Mean"), (std, "Standard deviation"),
            (lower, "Lower endpoint"), (upper, "Upper endpoint"))))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Mean and standard deviation must be finite")
    if (std <= 0).any():
        raise ValueError("Standard deviation must be positive")
    if np.isnan(lower).any() or np.isnan(upper).any() or (lower > upper).any():
        raise ValueError("Intervals must be ordered, with no NaN endpoints")
    return mean, std, lower, upper


def _log_sub_exp(a, b):
    # Select the stable tail before subtraction, as in the reference. expm1
    # preserves small representable differences without inventing probability.
    finite_a = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        delta = np.where(finite_a, b - a, -np.inf)
        return np.where(finite_a, a + np.log(-np.expm1(delta)), a)


def gaussian_interval_log_prob(lower, upper, mean, std):
    """Return float64 log P(lower < Y <= upper), including infinite endpoints.

    Equal endpoints have zero interval mass and return negative infinity. Use
    an explicit point mask in :func:`interval_moments` for a point observation.
    """
    mean, std, lower, upper = _inputs(mean, std, lower, upper)
    empty = lower == upper
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower = np.where(empty, mean, lower)
        upper = np.where(empty, mean + std, upper)
        finite_lo, finite_hi = np.isfinite(lower), np.isfinite(upper)
        lo = (np.where(finite_lo, lower, mean) - mean) / std
        hi = (np.where(finite_hi, upper, mean) - mean) / std
        cdf_hi = np.where(finite_hi, log_ndtr(hi), 0.0)
        cdf_lo = np.where(finite_lo, log_ndtr(lo), -np.inf)
        sf_lo = np.where(finite_lo, log_ndtr(-lo), 0.0)
        sf_hi = np.where(finite_hi, log_ndtr(-hi), -np.inf)
        positive = finite_lo & (lo > 0)
        result = _log_sub_exp(np.where(positive, sf_lo, cdf_hi),
                              np.where(positive, sf_hi, cdf_lo))
    return np.asarray(np.where(empty, -np.inf, result), dtype=np.float64)


def interval_moments(mean, std, lower, upper, *, point_mask=None):
    """Return float64 conditional mean/variance with frozen-reference clipping.

    Nonpoint intervals must be strictly nonempty. Explicit points need equal
    finite endpoints and a Boolean mask of exactly the broadcast event shape.
    Conditional variance fractions are clipped to [0, 1], matching the existing
    working-Gaussian Newton helper; this is not a calibration certificate.
    """
    mean, std, lower, upper = _inputs(mean, std, lower, upper)
    points = np.zeros(mean.shape, dtype=bool) if point_mask is None else np.asarray(point_mask)
    if (points.shape != mean.shape or points.dtype != np.dtype(bool)
            or (points & (~np.isfinite(lower) | (lower != upper))).any()):
        raise ValueError("Point observations need equal finite endpoints and an explicit point mask")
    if ((lower >= upper) & ~points).any():
        raise ValueError("Use positive spread and nonempty ordered intervals")
    point_value = lower
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        lower = np.where(points, mean - std, lower)
        upper = np.where(points, mean + std, upper)
        lo_finite, hi_finite = np.isfinite(lower), np.isfinite(upper)
        a = (np.where(lo_finite, lower, mean) - mean) / std
        b = (np.where(hi_finite, upper, mean) - mean) / std
        log_mass = gaussian_interval_log_prob(lower, upper, mean, std)
        la = np.where(lo_finite, -0.5 * np.square(a) - 0.5 * _LOG_2PI, -np.inf)
        lb = np.where(hi_finite, -0.5 * np.square(b) - 0.5 * _LOG_2PI, -np.inf)
        ra = np.exp(np.minimum(la - log_mass, 300.0))
        rb = np.exp(np.minimum(lb - log_mass, 300.0))
        shift = ra - rb
        expectation = np.where(points, point_value, mean + std * shift)
        variance_fraction = np.clip(1.0 + a * ra - b * rb - np.square(shift), 0.0, 1.0)
        variance = np.where(points, 0.0, np.square(std) * variance_fraction)
    if not np.isfinite(expectation).all() or not np.isfinite(variance).all():
        raise FloatingPointError("Nonfinite interval moments")
    return np.asarray(expectation, dtype=np.float64), np.asarray(variance, dtype=np.float64)
