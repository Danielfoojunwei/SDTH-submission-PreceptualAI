"""Pilot-only complex-channel baselines with explicit TRAIN-only statistics.

These numerical estimators do not discover or open datasets. The Fourier
coordinate is a finite-band circular relative-delay basis; its nonnegative
indices do not identify physical path delays or certify propagation causality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .calibration import frequency_spacing, immutable


def _frequency_grid(frequency_offsets_hz):
    f = np.asarray(frequency_offsets_hz, dtype='f8')
    frequency_spacing(f)
    return f


def _pilots(pilot_cfr, pilot_indices, frequency_offsets_hz):
    f = _frequency_grid(frequency_offsets_hz)
    x, indices = np.asarray(pilot_cfr), np.asarray(pilot_indices)
    if (x.ndim != 2 or x.dtype.kind != 'c' or not np.isfinite(x).all()
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or len(indices) < 1 or len(indices) != x.shape[1]
            or (indices < 0).any() or (indices >= len(f)).any()
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Use finite batch-by-pilot complex samples and distinct actual bin indices')
    order = np.argsort(indices)
    return np.asarray(x[:, order], dtype='c16'), indices[order].astype('i8'), f


def linear_interpolate(pilot_cfr, pilot_indices, frequency_offsets_hz):
    """Interpolate complex I/Q in Hz, with explicit linear endpoint extrapolation.

    Endpoint extrapolation uses the two closest pilots, never held-out bins.
    A phase-unwrapping or target-dependent calibration assumption is not made.
    """
    x, indices, f = _pilots(pilot_cfr, pilot_indices, frequency_offsets_hz)
    if len(indices) < 2:
        raise ValueError('Linear interpolation requires at least two distinct pilots')
    xp = f[indices]
    output = np.empty((len(x), len(f)), dtype='c16')
    before, after = f < xp[0], f > xp[-1]
    for row, values in enumerate(x):
        output[row] = np.interp(f, xp, values.real) + 1j * np.interp(f, xp, values.imag)
        output[row, before] = values[0] + (f[before] - xp[0]) * (values[1] - values[0]) / (xp[1] - xp[0])
        output[row, after] = values[-1] + (f[after] - xp[-1]) * (values[-1] - values[-2]) / (xp[-1] - xp[-2])
    return output


def circular_fourier_basis(frequency_offsets_hz, *, n_taps):
    """Canonical circular DFT-delay dictionary, referred to the first bin.

    Its period is 1/delta_f and its tap step is 1/(F*delta_f). A truncated
    dictionary is a declared prior to tune on TRAIN/VAL, not ray-delay truth.
    """
    f = _frequency_grid(frequency_offsets_hz)
    if type(n_taps) is not int or not 1 <= n_taps <= len(f):
        raise ValueError('Declare a finite circular basis rank between one and the frequency count')
    spacing = frequency_spacing(f)
    tau = np.arange(n_taps, dtype='f8') / (len(f) * spacing)
    return np.exp(-2j * np.pi * (f - f[0])[:, None] * tau[None, :])


def fourier_ridge(pilot_cfr, pilot_indices, frequency_offsets_hz, *, n_taps=16, ridge=1.e-3):
    """Fit circular Fourier coefficients to each packet's observed pilots only.

    Objective: mean_pilot |A a-y|^2 + ridge * ||a||^2. For ridge=0,
    NumPy's deterministic minimum-norm least-squares solution is used.
    """
    x, indices, f = _pilots(pilot_cfr, pilot_indices, frequency_offsets_hz)
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError('Fourier ridge must be finite and nonnegative')
    basis = circular_fourier_basis(f, n_taps=n_taps)
    pilot_basis = basis[indices]
    if ridge == 0:
        coefficients = np.linalg.lstsq(pilot_basis, x.T, rcond=None)[0].T
    else:
        normal = pilot_basis.conj().T @ pilot_basis + ridge * len(indices) * np.eye(n_taps)
        coefficients = np.linalg.solve(normal, pilot_basis.conj().T @ x.T).T
    return coefficients @ basis.T


@dataclass(frozen=True)
class TrainLMMSE:
    """Zero-mean complex LMMSE using a covariance learned only from TRAIN.

    Prediction is linear and equivariant to a packet's scalar complex gain.
    No target-derived per-packet normalization, clock fit or phase alignment is
    used. ``fit`` may see complete TRAIN spectra; ``predict`` only sees pilots.
    """
    frequency_offsets_hz: np.ndarray
    covariance: np.ndarray
    training_count: int
    shrinkage: float
    noise_fraction: float

    @classmethod
    def fit(cls, training_cfr, frequency_offsets_hz, *, shrinkage=.05, noise_fraction=1.e-3):
        f = _frequency_grid(frequency_offsets_hz)
        x = np.asarray(training_cfr)
        if (x.ndim != 2 or x.shape[1] != len(f) or x.dtype.kind != 'c'
                or len(x) < 1 or not np.isfinite(x).all()):
            raise ValueError('Fit covariance from finite complete TRAIN complex spectra')
        if (isinstance(shrinkage, bool) or not math.isfinite(shrinkage) or not 0 <= shrinkage <= 1
                or isinstance(noise_fraction, bool) or not math.isfinite(noise_fraction) or noise_fraction < 0):
            raise ValueError('Declare finite covariance shrinkage and nonnegative noise fraction')
        x = np.asarray(x, dtype='c16')
        covariance = x.T @ x.conj() / len(x)
        covariance = (covariance + covariance.conj().T) / 2
        covariance = (1-shrinkage)*covariance + shrinkage*np.diag(covariance.diagonal().real)
        return cls(immutable(f), immutable(covariance), len(x), float(shrinkage), float(noise_fraction))

    def predict(self, pilot_cfr, pilot_indices):
        x, indices, f = _pilots(pilot_cfr, pilot_indices, self.frequency_offsets_hz)
        covariance = np.asarray(self.covariance)
        if (covariance.shape != (len(f), len(f)) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.conj().T, rtol=1.e-10, atol=1.e-12)):
            raise ValueError('Invalid fitted TRAIN covariance')
        pilot_covariance = covariance[np.ix_(indices, indices)]
        energy = float(pilot_covariance.diagonal().real.mean())
        if energy <= 0:
            return np.zeros((len(x), len(f)), dtype='c16')
        normal = pilot_covariance + self.noise_fraction * energy * np.eye(len(indices))
        cross_covariance = covariance[:, indices]
        if self.noise_fraction > 0:
            # Solving the transposed equation avoids a needless explicit inverse.
            weights = np.linalg.solve(normal.T, cross_covariance.T).T
        else:
            weights = cross_covariance @ np.linalg.pinv(normal, hermitian=True)
        return x @ weights.T
