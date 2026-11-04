"""Public-input forecasts and a FIFO that charges measured completion costs."""
import time
import numpy as np
import torch
from ..closed_loop.runtime import LiveForecasts
from ..fleetwm.data import FleetInputs, Geometry, Observations
from ..frontier.fast import FastMap, FleetMemory, fit_reference
from .common import protocol

FAMILIES = ('supervised', 'context_jepa', 'semantic_aux', 'semantic_jepa')


class Predictor:
    """No future-label argument; one independently trained member per request."""
    def __init__(self, geometry, family, training_seed=None, *, model=None):
        if not isinstance(geometry, Geometry):
            raise TypeError('Use public Geometry')
        if family not in (*FAMILIES, 'radiomap', 'frozen_fleet_jepa'):
            raise ValueError('Unknown forecast family')
        if family in FAMILIES and training_seed not in (0, 1, 2):
            raise ValueError('Use a preregistered training seed')
        if family not in FAMILIES and (training_seed is not None or model is not None):
            raise ValueError('Comparators have fixed identity')
        self.geometry, self.family, self.training_seed = geometry, family, training_seed
        self.rmap = self.refit_at = self.last_time = None
        self.memory = FleetMemory(geometry) if family in FAMILIES else None
        began = time.perf_counter()
        if family in FAMILIES:
            from .training import load
            self.model = load(family, training_seed) if model is None else model
            if self.model.config['family'] != family:
                raise ValueError('Checkpoint family mismatch')
            self.model.eval()
        elif family == 'frozen_fleet_jepa':
            from ..fleet_control import FleetWorldModel
            self.frozen = FleetWorldModel(geometry, family='hybrid_jepa')
        self.startup_seconds = time.perf_counter() - began

    @torch.no_grad()
    def step(self, past, plan):
        if not isinstance(past, Observations):
            raise TypeError('Use observable history, not a truth World')
        t = len(past.value)
        plan = np.asarray(plan)
        if t < 60 or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Need sixty past timestamps and increasing history')
        if plan.shape != (past.value.shape[1], 9, 2) or not np.isfinite(plan).all():
            raise ValueError('Expected finite N x nine horizons x XY route')
        began = time.perf_counter()
        if self.family == 'frozen_fleet_jepa':
            mean, std, meta = self.frozen.forecast(past, plan)
            kind = 'cold' if self.last_time is None else 'refit' if meta['map_refit_seconds'] > 0 else 'warm'
        else:
            kind = 'warm'
            if self.rmap is None or t - self.refit_at >= protocol()['map_refresh_seconds']:
                kind = 'cold' if self.rmap is None else 'refit'
                self.rmap = fit_reference(self.geometry, past, strong=True)
                self.refit_at = t
            if self.family == 'radiomap':
                mean, std = FastMap(self.rmap).predict(plan[..., 0], plan[..., 1])
                mean = mean - self.geometry.tx + 15 - self.geometry.noise
            else:
                from .training import predict
                public, _ = self.memory.inputs(past, plan, self.rmap)
                if not isinstance(public, FleetInputs):
                    raise TypeError('Only label-free fleet inputs cross model boundary')
                mean, std = predict(self.model, public)
                shape = (len(plan), 9, len(self.geometry.sites), 3)
                mean, std = mean.reshape(shape), std.reshape(shape)
        self.last_time = t
        expected = (len(plan), 9, len(self.geometry.sites), 3)
        if mean.shape != expected or std.shape != expected or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
            raise ValueError('Malformed forecast')
        return mean.astype('f4'), std.astype('f4'), kind, time.perf_counter() - began


class ChargedForecasts(LiveForecasts):
    """Preserve predecessor availability and record actual overrun vetoes."""
    def issue(self, decision, plan):
        mean, std = super().issue(decision, plan)
        row = self.records[-1]
        row['runtime_overrun'] = row['actual_prediction_seconds'] > self.profile[row['kind']]
        return mean, std
