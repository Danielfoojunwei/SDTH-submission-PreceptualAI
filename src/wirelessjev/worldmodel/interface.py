"""Prediction-only boundary between radio models and mission planning.

SceneState contains a model's forecast, never simulator truth.  It does not
carry mission text, an environment object, credentials, or an actuation handle.
Calibration receipts are provenance references, not self-issued permission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

Timescale = Literal['fast', 'mid', 'slow']
BANDS = ('2g', '5g', '6g')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')


def immutable_array(value, *, dtype=float, ndim=None, shape=None, name='array'):
    result = np.array(value, dtype=dtype, copy=True)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f'{name}: wrong number of dimensions')
    if shape is not None and result.shape != shape:
        raise ValueError(f'{name}: wrong shape')
    if not np.isfinite(result).all():
        raise ValueError(f'{name}: finite values required')
    # An immutable byte owner prevents callers from toggling writeable=True.
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def finite_nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name}: a finite nonnegative number is required')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name}: a finite nonnegative number is required')
    return value


@dataclass(frozen=True)
class SceneState:
    """Predicted public RF field: N robots, H horizons, S APs, three bands.

    ``learned_outage_probability`` may be rendered by a future learned head.
    Its presence and ``qualified_head`` references alone grant no permission:
    an independent caller must approve that head before planning uses it.
    """
    times_s: np.ndarray
    current_ap: np.ndarray
    snr_db: np.ndarray
    source_id: str
    learned_outage_probability: np.ndarray | None = None
    qualified_head: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        times = immutable_array(self.times_s, ndim=1, name='times_s')
        if len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0):
            raise ValueError('times_s must increase strictly from zero')
        snr = immutable_array(self.snr_db, ndim=4, name='predicted snr_db')
        n, h, sites, bands = snr.shape
        if n < 1 or sites < 1 or bands != 3 or h != len(times):
            raise ValueError('Predicted SNR requires N x H x AP x three bands')
        raw = np.asarray(self.current_ap)
        if raw.shape != (n,) or raw.dtype.kind not in 'iu' or np.any((raw < 0) | (raw >= sites)):
            raise ValueError('current_ap must contain actual integral source associations')
        current = immutable_array(raw, dtype=np.int64, shape=(n,), name='current_ap')
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError('A prediction source identity is required')
        object.__setattr__(self, 'times_s', times)
        object.__setattr__(self, 'current_ap', current)
        object.__setattr__(self, 'snr_db', snr)
        if self.learned_outage_probability is not None:
            p = immutable_array(self.learned_outage_probability, shape=snr.shape,
                                name='learned outage probability')
            if np.any((p < 0) | (p > 1)):
                raise ValueError('Probabilities must lie in [0,1]')
            object.__setattr__(self, 'learned_outage_probability', p)
        receipts = dict(self.qualified_head)
        if any(not isinstance(k, str) or not k or not isinstance(v, str) or not _SHA256.fullmatch(v)
               for k, v in receipts.items()):
            raise ValueError('Head provenance requires explicit SHA256 receipt references')
        object.__setattr__(self, 'qualified_head', MappingProxyType(receipts))


@dataclass(frozen=True)
class CandidateAction:
    action_id: str
    ap: int
    band: str | None = None
    mcs: int | None = None
    handover_seconds: float = 4.
    mechanism: str = 'association'

    def __post_init__(self):
        if not isinstance(self.action_id, str) or not self.action_id:
            raise ValueError('An action identity is required')
        if type(self.ap) is not int or self.ap < 0:
            raise ValueError('Action AP must be a nonnegative integer')
        if self.band is not None and self.band not in BANDS:
            raise ValueError('Unsupported band')
        if self.mcs is not None and (type(self.mcs) is not int or self.mcs < 0):
            raise ValueError('MCS must be a nonnegative integer')
        if self.mcs is not None and self.band is None:
            raise ValueError('Explicit MCS requires an explicit band')
        if self.mechanism != 'association':
            raise ValueError('Only the measured/declared association mechanism is implemented; '
                             'MLO, power, width, routes and alternate handovers are unsupported')
        object.__setattr__(self, 'handover_seconds',
                           finite_nonnegative(self.handover_seconds, 'handover_seconds'))


@dataclass(frozen=True)
class OutcomeDistribution:
    """Action forecasts, not packet/SLA guarantees or complete MAC models.

    Pointwise values have N x A x H axes. Integrals and switching cost have
    N x A axes. Unknown silence/latency distributions stay None, not zero.
    """
    action_ids: tuple[str, ...]
    times_s: np.ndarray
    outage_probability: np.ndarray
    rate_mbps: np.ndarray
    outage_integral_seconds: np.ndarray
    switch_seconds: np.ndarray
    source_id: str
    metadata: Mapping[str, str]
    outage_upper: np.ndarray | None = None
    silence_upper_seconds: np.ndarray | None = None
    horizon_seconds: float | None = None

    def __post_init__(self):
        ids = tuple(self.action_ids)
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError('Unique action identities are required')
        times = immutable_array(self.times_s, ndim=1, name='outcome times')
        if len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0):
            raise ValueError('Outcome times must increase from zero')
        p = immutable_array(self.outage_probability, ndim=3, name='outage_probability')
        n, a, h = p.shape
        if n < 1 or a != len(ids) or h != len(times) or np.any((p < 0) | (p > 1)):
            raise ValueError('Invalid N x A x H outage probabilities')
        rate = immutable_array(self.rate_mbps, shape=p.shape, name='rate_mbps')
        cost = immutable_array(self.outage_integral_seconds, shape=(n, a), name='outage integral')
        switch = immutable_array(self.switch_seconds, shape=(n, a), name='switch seconds')
        horizon = times[-1] if self.horizon_seconds is None else finite_nonnegative(
            self.horizon_seconds, 'outcome horizon')
        if not 0 < horizon <= times[-1]:
            raise ValueError('Outcome horizon is outside prediction support')
        if (np.any(rate < 0) or np.any(cost < switch - 1e-12)
                or np.any(cost > horizon + 1e-12) or np.any(switch < 0)
                or np.any(switch > horizon)):
            raise ValueError('Negative rate/cost/switching time')
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError('A forecast source identity is required')
        for name, value in [('action_ids', ids), ('times_s', times), ('outage_probability', p),
                            ('rate_mbps', rate), ('outage_integral_seconds', cost), ('switch_seconds', switch)]:
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata)
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
            raise ValueError('Outcome metadata must be string provenance fields')
        object.__setattr__(self, 'metadata', MappingProxyType(metadata))
        object.__setattr__(self, 'horizon_seconds', float(horizon))
        if self.outage_upper is not None:
            upper = immutable_array(self.outage_upper, shape=p.shape, name='outage_upper')
            if np.any((upper < p) | (upper > 1)):
                raise ValueError('Outage upper bounds must contain the mean and stay <=1')
            object.__setattr__(self, 'outage_upper', upper)
        if self.silence_upper_seconds is not None:
            silence = immutable_array(self.silence_upper_seconds, shape=(n, a), name='silence upper')
            if np.any(silence < switch):
                raise ValueError('Silence bound cannot exclude the declared switch outage')
            object.__setattr__(self, 'silence_upper_seconds', silence)


@runtime_checkable
class WorldModel(Protocol):
    def query(self, scene_state: SceneState, candidate_actions: Sequence[CandidateAction],
              horizon: float, timescale: Timescale) -> OutcomeDistribution: ...
