"""Observation-driven, event-time field belief; no prediction-as-observation loop.

This is a linear eight-coefficient field filter, not a dynamic electromagnetic
simulator. Its working Gaussian discrepancy and Laplace covariance require
future calibration. Distance forgetting does not define a loop-consistent
spatial random field. No temporal experiment or safety claim is implied here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..spectral_scene.baselines import _truncated_moments, _log_interval_mass

DIMENSION = 8


def _array(value, shape, name):
    a = np.asarray(value, dtype='f8')
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Invalid finite filter '+name)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _covariance(value):
    a = _array(value, (DIMENSION, DIMENSION), 'covariance')
    if not np.allclose(a, a.T, atol=1e-10, rtol=1e-10):
        raise ValueError('Filter covariance must be symmetric')
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Filter covariance must be positive definite') from exc
    return a


@dataclass(frozen=True)
class Belief:
    timestamp: float
    mean: np.ndarray
    covariance: np.ndarray
    position_xyz: np.ndarray | None = None

    def __post_init__(self):
        if not math.isfinite(self.timestamp):
            raise ValueError('A finite event timestamp is required')
        object.__setattr__(self, 'mean', _array(self.mean, (DIMENSION,), 'mean'))
        object.__setattr__(self, 'covariance', _covariance(self.covariance))
        if self.position_xyz is not None:
            object.__setattr__(self, 'position_xyz', _array(self.position_xyz, (3,), 'position'))


@dataclass(frozen=True)
class FilterConfig:
    # Coefficient units are 10 dB per unit through the physical basis.
    time_variance_per_second: float = 0.
    distance_variance_per_metre: float = 0.
    mean_reversion_seconds: float = math.inf
    sensor_sigma_db: float = 0.
    em_steps: int = 12
    innovation_threshold: float = 3.5
    reset_on_change: bool = False
    reset_covariance_multiplier: float = 4.

    def __post_init__(self):
        for name in ('time_variance_per_second', 'distance_variance_per_metre', 'sensor_sigma_db'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Nonnegative finite filter scale required: '+name)
        if (self.mean_reversion_seconds <= 0 or math.isnan(self.mean_reversion_seconds)
                or type(self.em_steps) is not int or not 1 <= self.em_steps <= 100
                or not math.isfinite(self.innovation_threshold) or self.innovation_threshold <= 0
                or type(self.reset_on_change) is not bool
                or not math.isfinite(self.reset_covariance_multiplier) or self.reset_covariance_multiplier < 1):
            raise ValueError('Invalid declared filter dynamics/change policy')


@dataclass(frozen=True)
class ReportEvent:
    """One actual receiver report or missing attempt, with public link operands.

    ``discrepancy_sigma_db`` describes unresolved field error, not thermal noise.
    Heard readings supply their real quantization interval. A censored reading
    supplies only an upper bound. Missing events supply no RF value at all.
    ``measurement_id`` must identify a new receiver event, never a prediction.
    """
    measurement_id: str
    timestamp: float
    basis: np.ndarray
    fspl_dbm: float
    heard: bool = False
    censored: bool = False
    lower_dbm: float = math.nan
    upper_dbm: float = math.nan
    censor_upper_dbm: float = math.nan
    offset_db: float = 0.
    local_geometry_db: float = 0.
    discrepancy_sigma_db: float = 4.
    position_xyz: np.ndarray | None = None
    detection_threshold_dbm: float = -math.inf

    def __post_init__(self):
        if not isinstance(self.measurement_id, str) or not self.measurement_id:
            raise ValueError('Every filter event needs its actual receiver event identity')
        if (not math.isfinite(self.timestamp) or type(self.heard) is not bool
                or type(self.censored) is not bool or self.heard and self.censored):
            raise ValueError('Finite time and disjoint explicit report flags required')
        object.__setattr__(self, 'basis', _array(self.basis, (DIMENSION,), 'public basis'))
        if not all(math.isfinite(v) for v in (self.fspl_dbm, self.offset_db, self.local_geometry_db,
                                             self.discrepancy_sigma_db)) or self.discrepancy_sigma_db <= 0:
            raise ValueError('Finite physical anchor and positive working discrepancy required')
        if self.heard and (not math.isfinite(self.lower_dbm) or not math.isfinite(self.upper_dbm)
                           or self.lower_dbm >= self.upper_dbm):
            raise ValueError('Heard events require a real positive-width quantization interval')
        if self.censored and not math.isfinite(self.censor_upper_dbm):
            raise ValueError('Censored events require a true finite upper bound')
        if (math.isnan(self.detection_threshold_dbm) or self.detection_threshold_dbm == math.inf
                or self.heard and max(self.lower_dbm,self.detection_threshold_dbm)>=self.upper_dbm):
            raise ValueError('Heard quantization intervals must intersect their detector event')
        if self.position_xyz is not None:
            object.__setattr__(self, 'position_xyz', _array(self.position_xyz, (3,), 'position'))

    @property
    def anchor_dbm(self):
        return self.fspl_dbm+self.offset_db+self.local_geometry_db

    @property
    def interval(self):
        if self.heard:
            # -inf explicitly means a caller-supplied event interval with no
            # additional detector truncation; actual receiver events supply it.
            return max(self.lower_dbm,self.detection_threshold_dbm), self.upper_dbm
        if self.censored:
            return -math.inf, self.censor_upper_dbm
        return None


class EventTimeFilter:
    def __init__(self, initial: Belief, *, config: FilterConfig | None = None,
                 reference_mean=None, reference_covariance=None):
        if not isinstance(initial, Belief):
            raise TypeError('Initialize from an explicit field belief')
        self.config = FilterConfig() if config is None else config
        if not isinstance(self.config, FilterConfig):
            raise TypeError('Use an explicit FilterConfig')
        self.belief = initial
        self.reference_mean = _array(initial.mean if reference_mean is None else reference_mean,
                                     (DIMENSION,), 'reference mean')
        self.reference_covariance = _covariance(initial.covariance if reference_covariance is None
                                                else reference_covariance)
        self._seen_ids = set()

    @property
    def metadata(self):
        return {'state': 'eight reference-gauge field coefficients, not material or scatterer truth',
                'observation': 'received dBm = FSPL + estimated frontend offset + 10*basis@beta + geometry-only local field',
                'uncertainty_calibrated': False,
                'sensor_sigma_db': self.config.sensor_sigma_db,
                'bin_quantization': 'actual interval likelihood; no added bin Gaussian variance',
                'heard_detection': 'quantization bin intersected with explicit detector lower bound',
                'covariance': 'observed-information Laplace approximation, not complete-data EM precision',
                'prediction_feedback': False,
                'distance_forgetting': 'epistemic process variance, not a spatial OU/channel correlation law',
                'change_flags': 'conditional predictive-event innovation diagnostics, not a calibrated false-alarm or safety guarantee',
                'reset_enabled': self.config.reset_on_change,
                'reset_policy': 'restore declared reference mean and inflated reference covariance before processing the flagged actual event',
                'production_or_service_qualified': False}

    def _propagated(self, timestamp, position_xyz=None):
        if not math.isfinite(timestamp) or timestamp < self.belief.timestamp:
            raise ValueError('Out-of-order/nonfinite event timestamp')
        position = self.belief.position_xyz if position_xyz is None else _array(position_xyz, (3,), 'position')
        dt = timestamp-self.belief.timestamp
        distance = (0. if position is None or self.belief.position_xyz is None
                    else float(np.linalg.norm(position-self.belief.position_xyz)))
        rho = math.exp(-dt/self.config.mean_reversion_seconds)
        mean = self.reference_mean+rho*(self.belief.mean-self.reference_mean)
        covariance = rho*rho*self.belief.covariance+(1-rho*rho)*self.reference_covariance
        q = self.config.time_variance_per_second*dt+self.config.distance_variance_per_metre*distance
        covariance = covariance+q*np.eye(DIMENSION)
        return Belief(timestamp, mean, covariance, position)

    def advance(self, timestamp, position_xyz=None):
        """Propagate time/position without inventing any observation."""
        self.belief = self._propagated(timestamp, position_xyz)
        return self.belief

    @staticmethod
    def _moments(mean, lower, upper, sigma):
        expected, variance = _truncated_moments(np.array((mean,)), np.array((lower,)),
                                                np.array((upper,)), sigma)
        return float(expected[0]), float(variance[0])

    def observe(self, event: ReportEvent):
        if not isinstance(event, ReportEvent):
            raise TypeError('Only an explicit actual ReportEvent may update this filter')
        if event.measurement_id in self._seen_ids:
            raise ValueError('Duplicate receiver event cannot add information twice')
        prior = self._propagated(event.timestamp, event.position_xyz)
        interval = event.interval
        diagnostic = {'measurement_id': event.measurement_id, 'timestamp': event.timestamp,
                      'measurement_information_added': interval is not None,
                      'change_flag': False, 'reset_applied': False,
                      'normalized_innovation': None, 'event_log_probability': None}
        if interval is None:
            self.belief = prior
            self._seen_ids.add(event.measurement_id)
            return prior, diagnostic
        h = 10.*event.basis
        lower, upper = interval
        sigma = math.hypot(event.discrepancy_sigma_db, self.config.sensor_sigma_db)
        predicted = event.anchor_dbm+float(h@prior.mean)
        predictive_sigma = math.sqrt(float(h@prior.covariance@h)+sigma*sigma)
        expected, _ = self._moments(predicted, lower, upper, predictive_sigma)
        innovation = (expected-predicted)/predictive_sigma
        logp = float(_log_interval_mass(np.array(((lower-predicted)/predictive_sigma,)),
                                       np.array(((upper-predicted)/predictive_sigma,)))[0])
        diagnostic.update(normalized_innovation=innovation, event_log_probability=logp,
                          change_flag=abs(innovation) > self.config.innovation_threshold)
        if diagnostic['change_flag'] and self.config.reset_on_change:
            prior = Belief(prior.timestamp, self.reference_mean,
                           self.reference_covariance*self.config.reset_covariance_multiplier,
                           prior.position_xyz)
            diagnostic['reset_applied'] = True
        precision = np.linalg.inv(prior.covariance)
        complete = precision+np.outer(h, h)/(sigma*sigma)
        mean = prior.mean.copy()
        for _ in range(self.config.em_steps):
            mu = event.anchor_dbm+float(h@mean)
            expected, _ = self._moments(mu, lower, upper, sigma)
            updated = np.linalg.solve(complete, precision@prior.mean+h*(expected-event.anchor_dbm)/(sigma*sigma))
            if np.max(np.abs(updated-mean)) < 1e-10:
                mean = updated
                break
            mean = updated
        mu = event.anchor_dbm+float(h@mean)
        _, conditional_variance = self._moments(mu, lower, upper, sigma)
        # A weak upper bound has variance ~= sigma², hence near-zero information.
        information = float(np.clip(1.-conditional_variance/(sigma*sigma), 0., 1.))/(sigma*sigma)
        covariance = np.linalg.inv(precision+np.outer(h, h)*information)
        covariance = .5*(covariance+covariance.T)
        posterior = Belief(prior.timestamp, mean, covariance, prior.position_xyz)
        diagnostic['observed_information_db_inverse_squared'] = information
        self.belief = posterior
        self._seen_ids.add(event.measurement_id)
        return posterior, diagnostic

    def predict_power(self, basis, fspl_dbm, *, offset_db=0., local_geometry_db=0.,
                      discrepancy_sigma_db=4.):
        """Read-only forecast: prediction has no update/feedback side effect."""
        basis = np.asarray(basis, dtype='f8')
        if basis.ndim != 2 or basis.shape[1] != DIMENSION or not np.isfinite(basis).all():
            raise ValueError('Use finite N x 8 public prediction basis')
        n = len(basis)
        def vector(value, name):
            return _array(np.broadcast_to(np.asarray(value, dtype='f8'), (n,)), (n,), name)
        anchor = vector(fspl_dbm, 'FSPL')+vector(offset_db, 'offset')+vector(local_geometry_db, 'local geometry')
        discrepancy = vector(discrepancy_sigma_db, 'discrepancy')
        if np.any(discrepancy <= 0):
            raise ValueError('Positive working prediction discrepancy required')
        h = 10.*basis
        mean = anchor+h@self.belief.mean
        variance = np.einsum('nd,de,ne->n', h, self.belief.covariance, h)
        std = np.sqrt(np.maximum(variance, 0.)+discrepancy**2+self.config.sensor_sigma_db**2)
        return mean, std
