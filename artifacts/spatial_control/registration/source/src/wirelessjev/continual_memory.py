"""Bounded differential-response memory; algebraic ID is not causal qualification."""
from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DifferentialPair:
    pair_id: str
    exposure_delta_mw: object
    received_delta_mw: float | None
    measured_at: float | None
    delivered_at: float | None
    controlled: bool = False
    provenance: str = ''


@dataclass(frozen=True)
class ResponseEstimate:
    identified: bool
    predicted_delta_mw: float | None
    rank: int
    observations: int
    reason: str
    evidence_kind: str = 'none'
    provenance: tuple = ()
    uncertainty_bound: None = None
    production_qualified: bool = False


def _finite(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + ' requires a real scalar')
    try: value = float(value)
    except (OverflowError, ValueError): raise ValueError('Invalid ' + name) from None
    if not math.isfinite(value) or (positive and value <= 0): raise ValueError('Invalid ' + name)
    return value


class ContinualResponseMemory:
    """One independently keyed receiver/context/radio-epoch linear response fit.

    Keys are (device_id, context_id, radio_epoch). Caller supplies signed known
    exposure changes in an unchanged transmitter order, never a hidden gain G.
    Retained pair IDs deduplicate within the bounded state/window; evicted IDs
    have no permanent exactly-once guarantee. All operations share one clock.
    """
    def __init__(self, dimension, *, max_states=16, max_pairs_per_state=64,
                 max_age_seconds=60., max_delivery_seconds=5., exposure_scale_mw=1.,
                 response_scale_mw=1., max_scaled_value=1e6, rcond=1e-10):
        for name, value in (('dimension', dimension), ('max_states', max_states),
                            ('max_pairs_per_state', max_pairs_per_state)):
            if type(value) is not int or value < 1: raise ValueError(name + ' must be a positive integer')
        if dimension > 64: raise ValueError('This small memory supports at most 64 exposure coordinates')
        self.dimension, self.max_states, self.max_pairs = dimension, max_states, max_pairs_per_state
        self.max_age = _finite(max_age_seconds, 'maximum evidence age', True)
        self.max_delivery = _finite(max_delivery_seconds, 'maximum delivery delay', True)
        self.xscale = _finite(exposure_scale_mw, 'exposure mW scale', True)
        self.yscale = _finite(response_scale_mw, 'response mW scale', True)
        self.limit = _finite(max_scaled_value, 'normalized magnitude limit', True)
        self.rcond = _finite(rcond, 'relative rank tolerance', True)
        if self.limit > 1e6 or self.rcond >= 1: raise ValueError('Unsafe magnitude/rank configuration')
        self._states = OrderedDict(); self._clock = None

    def _now(self, now):
        now = _finite(now, 'ingestion/query clock')
        if now < 0 or (self._clock is not None and now < self._clock): raise ValueError('Clock moved backwards')
        self._clock = now
        return now

    @staticmethod
    def _key(key):
        if (not isinstance(key, tuple) or len(key) != 3
                or any(not isinstance(v, str) or not 0 < len(v) <= 128 for v in key[:2])
                or not ((isinstance(key[2], str) and 0 < len(key[2]) <= 128)
                        or (type(key[2]) is int and 0 <= key[2] < 2**63))):
            raise ValueError('Use a bounded (device, context, radio_epoch) key')
        return key

    def _vector(self, values):
        if values is None: return None
        if (not isinstance(values, (list, tuple, np.ndarray)) or np.ndim(values) != 1
                or len(values) != self.dimension):
            raise ValueError('Exposure coordinates differ from the declared transmitter schema')
        raw = np.asarray(values)
        if raw.shape != (self.dimension,): raise ValueError('Exposure must be a vector')
        if any(v is None for v in raw): return None
        if raw.dtype.kind not in 'iuf': raise ValueError('Exposure requires signed numeric mW')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            x = raw.astype('f8') / self.xscale
        if not np.isfinite(x).all() or (np.abs(x) > self.limit).any(): raise ValueError('Exposure scaling overflow')
        if ((raw != 0) & (x == 0)).any(): raise ValueError('Exposure scaling underflow')
        peak = float(np.abs(x).max())
        if 0 < peak < 1e-150: raise ValueError('Rescale exposure to avoid Gram underflow')
        return x.copy()

    def _prune(self, now):
        for key, state in list(self._states.items()):
            state['rows'] = OrderedDict((i, r) for i, r in state['rows'].items() if now-r[2] <= self.max_age)
            if not state['rows']: del self._states[key]
            else: self._rebuild(state)

    def _rebuild(self, state):
        rows = list(state['rows'].values())
        x = np.stack([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        state['gram'], state['rhs'] = x.T @ x, x.T @ y

    def ingest(self, key, pair, *, now):
        now = self._now(now); key = self._key(key); self._prune(now)
        if not isinstance(pair, DifferentialPair): raise TypeError('Use a DifferentialPair')
        if not isinstance(pair.pair_id, str) or not 0 < len(pair.pair_id) <= 128: raise ValueError('Pair identity required')
        if type(pair.controlled) is not bool or not isinstance(pair.provenance, str) or len(pair.provenance) > 512:
            raise ValueError('Invalid declared control provenance')
        if pair.controlled and not pair.provenance.strip(): raise ValueError('Controlled pairs require declared experiment provenance')
        if pair.measured_at is None or pair.delivered_at is None or pair.received_delta_mw is None: return False
        measured = _finite(pair.measured_at, 'measurement time'); delivered = _finite(pair.delivered_at, 'delivery time')
        if not 0 <= measured <= delivered <= now: raise ValueError('Noncausal measurement/delivery clocks')
        if now-measured > self.max_age or delivered-measured > self.max_delivery: return False
        x = self._vector(pair.exposure_delta_mw)
        if x is None: return False
        raw_y = _finite(pair.received_delta_mw, 'received power difference mW')
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'): y = raw_y / self.yscale
        if not math.isfinite(y) or abs(y) > self.limit: raise ValueError('Response scaling overflow')
        if raw_y != 0 and (y == 0 or abs(y) < 1e-150): raise ValueError('Rescale response to avoid underflow')
        state = self._states.get(key)
        if state is not None and pair.pair_id in state['rows']: return False
        if state is None:
            if len(self._states) >= self.max_states: self._states.popitem(last=False)
            state = {'rows': OrderedDict()}; self._states[key] = state
        state['rows'][pair.pair_id] = (x, y, measured, pair.controlled, pair.provenance)
        if len(state['rows']) > self.max_pairs: state['rows'].popitem(last=False)
        self._rebuild(state); self._states.move_to_end(key)
        return True

    def query(self, key, exposure_delta_mw, *, now, query_observed_at=None, controlled_only=False):
        now = self._now(now); key = self._key(key); self._prune(now)
        unknown = lambda reason: ResponseEstimate(False, None, 0, 0, reason)
        if type(controlled_only) is not bool: raise ValueError('controlled_only must be boolean')
        if query_observed_at is None: return unknown('unknown_query_time')
        observed = _finite(query_observed_at, 'query exposure observation time')
        if not 0 <= observed <= now: raise ValueError('Query exposure is from the future')
        if now-observed > self.max_age: return unknown('stale_query')
        q = self._vector(exposure_delta_mw)
        if q is None: return unknown('missing_query')
        state = self._states.get(key)
        if state is None: return unknown('no_fresh_pairs')
        self._states.move_to_end(key)
        rows = [r for r in state['rows'].values() if not controlled_only or r[3]]
        if not rows: return unknown('no_controlled_pairs')
        if controlled_only:
            x = np.stack([r[0] for r in rows]); y = np.array([r[1] for r in rows]); gram, rhs = x.T@x, x.T@y
        else: gram, rhs = state['gram'], state['rhs']
        values, vectors = np.linalg.eigh(gram); keep = values > max(0., values[-1])*self.rcond
        basis = vectors[:, keep]; rank = int(keep.sum()); peak = float(np.abs(q).max())
        unit = q/peak if peak else q
        identified = bool(np.linalg.norm(unit-basis@(basis.T@unit)) <= 1e-9*np.linalg.norm(unit))
        kind = 'controlled_declared' if all(r[3] for r in rows) else 'associative_only'
        provenance = tuple(sorted({r[4] for r in rows if r[4]}))
        prediction = None
        if identified:
            with np.errstate(over='ignore', invalid='ignore'):
                coefficients = basis@((basis.T@rhs)/values[keep])  # same truncated Gram pseudoinverse
                normalized = np.dot(q.astype(np.longdouble), coefficients.astype(np.longdouble))
                physical = normalized*np.longdouble(self.yscale)
                prediction = float(physical)
            if not math.isfinite(prediction):
                return ResponseEstimate(True, None, rank, len(rows), 'numerical_overflow', kind, provenance)
            if physical != 0 and prediction == 0:
                return ResponseEstimate(True, None, rank, len(rows), 'numerical_underflow', kind, provenance)
        return ResponseEstimate(identified, prediction, rank, len(rows), 'identified' if identified else 'unidentified', kind, provenance)

    @property
    def size(self):
        return {'states': len(self._states), 'pairs': sum(len(s['rows']) for s in self._states.values())}
