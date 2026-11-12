"""Linear target-observability certificates and explicit Gaussian priors.

These statements concern the supplied linear operators. They do not prove
physical scatterer uniqueness, nonlinear power identifiability or clock gauges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _matrix(value,name):
    a=np.asarray(value)
    if a.ndim!=2 or a.dtype.kind not in 'fci' or not np.isfinite(a).all():
        raise ValueError(name+' must be a finite real/complex matrix')
    return np.asarray(a,dtype='c16')


def _readonly(a):
    a=np.asarray(a)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _psd(value):
    covariance=_matrix(value,'Covariance')
    if covariance.shape[0]!=covariance.shape[1] or not len(covariance):
        raise ValueError('Covariance must be a nonempty square matrix')
    scale=float(np.linalg.norm(covariance,2))
    if not np.isfinite(scale):raise FloatingPointError('Covariance scale overflow')
    tolerance=np.finfo(float).eps*max(len(covariance),1)*max(scale,np.finfo(float).tiny)*64
    if np.max(abs(covariance-covariance.conj().T))>tolerance:
        raise ValueError('Covariance must be Hermitian')
    covariance=(covariance+covariance.conj().T)/2
    eigenvalues,vectors=np.linalg.eigh(covariance)
    if eigenvalues.min() < -tolerance:raise ValueError('Covariance must be positive semidefinite')
    eigenvalues=np.maximum(eigenvalues,0.)
    cleaned=(vectors*eigenvalues[None])@vectors.conj().T
    return (cleaned+cleaned.conj().T)/2,eigenvalues,vectors,tolerance


@dataclass(frozen=True)
class GaussianPrior:
    """An explicitly named proper-complex covariance assumption; never fitted here."""
    label: str
    covariance: np.ndarray

    def __post_init__(self):
        if not isinstance(self.label,str) or not self.label.strip():
            raise ValueError('An explicit named prior assumption is required')
        covariance,_,_,_=_psd(self.covariance)
        object.__setattr__(self,'covariance',_readonly(covariance))


@dataclass(frozen=True)
class SubspaceCertificate:
    unknown_dimensions: int
    measurements: int
    rank: int
    nullity: int
    singular_values: tuple[float,...]
    rank_threshold: float
    target_threshold: float
    target_ambiguous_rank: int
    target_nullspace_operator_norm: float
    target_nullspace_frobenius_squared: float
    target_identifiable: bool
    observed_subspace_condition_number: float | None
    full_column_condition_number: float | None


@dataclass(frozen=True)
class IdentifiabilityCertificate:
    ambient: SubspaceCertificate
    prior_supported: SubspaceCertificate | None
    prior_label: str | None
    prior_support_eigenvalue_threshold: float | None
    omitted_prior_target_variance: float | None
    prior_support_truncated: bool
    scope: str = 'supplied linear operators at the stated numerical tolerances'


def _certificate(a,t,rtol,rank_atol,target_atol):
    _,singular,vh=np.linalg.svd(a,full_matrices=True)
    if not np.isfinite(singular).all():raise FloatingPointError('Measurement singular-value overflow')
    top=float(singular[0]) if len(singular) else 0.
    threshold=rank_atol+rtol*top
    rank=int((singular>threshold).sum())
    # FULL V is essential: thin SVD omits the unobserved right-nullspace.
    null=vh[rank:].conj().T
    ambiguity=t@null
    target_scale=float(np.linalg.norm(t,2)) if t.size else 0.
    target_threshold=target_atol+rtol*target_scale
    s=np.linalg.svd(ambiguity,compute_uv=False) if ambiguity.size else np.zeros(0)
    ambiguous=int((s>target_threshold).sum())
    with np.errstate(over='ignore',invalid='ignore'):
        ambiguity_energy=float(np.vdot(ambiguity,ambiguity).real)
        condition=float(np.divide(top,float(singular[rank-1]))) if rank else None
    if (not np.isfinite(threshold) or not np.isfinite(target_threshold) or not np.isfinite(s).all()
            or not np.isfinite(ambiguity_energy) or (condition is not None and not np.isfinite(condition))):
        raise FloatingPointError('Linear certificate arithmetic overflow')
    return SubspaceCertificate(a.shape[1],a.shape[0],rank,a.shape[1]-rank,
        tuple(float(v) for v in singular),float(threshold),float(target_threshold),ambiguous,
        float(s[0]) if len(s) else 0.,ambiguity_energy,ambiguous==0,
        condition,condition if rank==a.shape[1] and a.shape[1]>0 else None)


def certify_targets(measurement_operator,target_operator,*,prior:GaussianPrior|None=None,
                    rtol=None,rank_atol=0.,target_atol=0.,prior_support_rtol=0.):
    """Certify whether null(A) changes Tx, ambient and on an explicit prior range.

    The Frobenius ambiguity is tr(T P_null(A) Tᴴ); the operator norm is worst
    target perturbation per unit latent perturbation. Neither bounds actual
    nonlinear wireless errors. A full-rank stochastic prior does not remove
    ambient nonidentifiability; a low-rank prior restricts support by assumption.
    """
    a=_matrix(measurement_operator,'Measurement operator');t=_matrix(target_operator,'Target operator')
    if not a.shape[1] or t.shape[1]!=a.shape[1] or not len(t):
        raise ValueError('Measurement and nonempty target operators must share unknown dimensions')
    rtol=np.finfo(float).eps*max(a.shape[0],a.shape[1],t.shape[0],1)*16 if rtol is None else rtol
    for value in (rtol,rank_atol,target_atol,prior_support_rtol):
        if isinstance(value,bool) or not np.isfinite(value) or value<0:raise ValueError('Finite nonnegative rank tolerances required')
    ambient=_certificate(a,t,rtol,rank_atol,target_atol)
    supported=None;label=None;eigen_threshold=None;omitted=None;truncated=False
    if prior is not None:
        if not isinstance(prior,GaussianPrior) or prior.covariance.shape!=(a.shape[1],a.shape[1]):
            raise ValueError('The explicit prior must cover the same latent space')
        _,eigen,vectors,_=_psd(prior.covariance)
        # Preserve every positive supported direction by default. A tiny
        # eigenvalue can carry large target variance after applying T.
        eigen_threshold=float(eigen.max()*prior_support_rtol)
        keep=eigen>eigen_threshold
        loading=t@vectors[:,~keep]
        with np.errstate(over='ignore',invalid='ignore'):
            omitted=float(np.sum(abs(loading)**2*eigen[~keep][None]))
        if not np.isfinite(eigen_threshold) or not np.isfinite(omitted):
            raise FloatingPointError('Prior certificate arithmetic overflow')
        truncated=bool(((eigen>0)&~keep).any())
        factor=vectors[:,keep]*np.sqrt(eigen[keep])[None]
        supported=_certificate(a@factor,t@factor,rtol,rank_atol,target_atol)
        label=prior.label
    return IdentifiabilityCertificate(ambient,supported,label,eigen_threshold,omitted,truncated)


def expected_posterior_covariance(prior:GaussianPrior,measurement_operator,noise_variance):
    """Expected proper-complex Gaussian covariance, independent sensor noise.

    This API has no observation values. It cannot check impossible realized
    residuals in a singular/noiseless model, or certify posterior calibration.
    Correlated noise/nuisance clocks require an explicitly augmented model.
    """
    if not isinstance(prior,GaussianPrior):raise ValueError('An explicit GaussianPrior is required')
    a=_matrix(measurement_operator,'Measurement operator');covariance=prior.covariance
    if a.shape[1]!=len(covariance):raise ValueError('Operator/prior dimensions differ')
    noise=np.asarray(noise_variance,dtype='f8')
    if noise.ndim==0:noise=np.full(len(a),float(noise))
    if noise.shape!=(len(a),) or not np.isfinite(noise).all() or (noise<0).any():
        raise ValueError('Independent measurement noise variances must be nonnegative')
    if not len(a):return _readonly(covariance)
    # Independent-noise rank-one conditioning equals the joint Schur
    # complement. It avoids a global relative pseudoinverse cutoff silently
    # erasing a small positive direction that T may strongly amplify.
    result=np.array(covariance,copy=True)
    for row,variance in zip(a,noise):
        scale=max(float(np.linalg.norm(result,2)),np.finfo(float).tiny)
        with np.errstate(over='ignore',invalid='ignore'):
            v=result@row.conj();signal=float((row@v).real)
        if not np.isfinite(v).all() or not np.isfinite(signal):
            raise FloatingPointError('Innovation arithmetic overflow')
        tolerance=np.finfo(float).eps*scale*float(np.vdot(row,row).real)*64
        if signal < -tolerance:raise FloatingPointError('Negative innovation variance')
        denominator=max(signal,0.)+variance
        if not np.isfinite(denominator):raise FloatingPointError('Innovation variance overflow')
        if denominator==0:continue
        result=_posterior_psd(result-np.outer(v,v.conj())/denominator,scale,len(result))
    return _readonly(result)


def _posterior_psd(result,reference_scale,operation_size):
    """Clip only roundoff relative to the pre-update covariance, never a prior."""
    result=(result+result.conj().T)/2
    eigen,vectors=np.linalg.eigh(result)
    tolerance=np.finfo(float).eps*max(operation_size,1)*reference_scale*256
    if eigen.min() < -tolerance:raise FloatingPointError('Posterior lost positive semidefiniteness')
    result=(vectors*np.maximum(eigen,0.)[None])@vectors.conj().T
    return _readonly((result+result.conj().T)/2)


def target_variance(covariance,target_operator):
    """Trace target MSE in squared linear target units; not dB or an SLA bound."""
    covariance,_,_,_=_psd(covariance);t=_matrix(target_operator,'Target operator')
    if t.shape[1]!=len(covariance):raise ValueError('Target/covariance dimensions differ')
    with np.errstate(over='ignore',invalid='ignore'):
        value=float(np.trace(t@covariance@t.conj().T).real)
    if not np.isfinite(value):raise FloatingPointError('Target variance arithmetic overflow')
    return max(value,0.)
