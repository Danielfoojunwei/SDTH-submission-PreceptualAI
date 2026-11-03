"""Single-family causal predictions; the controller only sees completed jobs."""
import hashlib
import time
import numpy as np
import torch
from ..frontier.fast import FleetMemory, FastMap, fit_reference
from ..frontier.study import load_student, tensors
from ..frontier.timing import ForecastQueue


class SinglePredictor:
    def __init__(self, geometry, family):
        if family not in ('radiomap', 'distilled'):
            raise ValueError('Unsupported frozen family')
        self.geometry, self.family = geometry, family
        self.memory = FleetMemory(geometry) if family == 'distilled' else None
        self.model = load_student(family) if family == 'distilled' else None
        self.rmap = None
        self.refit_at = None

    @torch.no_grad()
    def step(self, past, plan):
        started = time.perf_counter()
        t = len(past.value)
        kind = 'cold' if self.refit_at is None else 'warm'
        g = self.geometry
        if self.family == 'radiomap':
            if self.rmap is None or t-self.refit_at >= 20:
                kind = 'cold' if self.rmap is None else 'refit'
                self.rmap = fit_reference(g, past, strong=True)
                self.refit_at = t
            mean, sd = FastMap(self.rmap).predict(plan[..., 0], plan[..., 1])
            mean = mean-g.tx+15-g.noise
        else:
            b, count = self.memory.inputs(past, plan)
            p = self.model(tensors(b, count))
            shape = (len(plan), 9, len(g.sites), 3)
            mean = p['mean'].numpy().reshape(shape)
            sd = p['std'].numpy().reshape(shape)
            self.refit_at = t
        return mean.astype('f4'), sd.astype('f4'), kind, time.perf_counter()-started


class LiveForecasts:
    def __init__(self, predictor, history, duration, profile, extra):
        self.predictor, self.history = predictor, history
        self.profile, self.extra = profile, extra
        self.ready = []
        self.last, self.age, self.previous_ready = -1, 0., 0.
        self.mean = np.zeros((duration, history.n, 9, len(predictor.geometry.sites), 3), dtype='f4')
        self.std = np.zeros_like(self.mean)
        self.records = []

    def issue(self, decision, plan):
        if decision != len(self.records):
            raise ValueError('Forecast jobs must be issued in time order')
        past = self.history.before(decision)
        m, s, kind, elapsed = self.predictor.step(past, plan)
        self.mean[decision], self.std[decision] = m, s
        service = self.profile[kind]+self.extra
        self.previous_ready = max(decision, self.previous_ready)+service
        self.ready.append(int(np.ceil(self.previous_ready*100-1e-10)))
        self.records.append({'decision': decision, 'history_cutoff_exclusive': 60+decision,
                             'ready_slot': self.ready[-1], 'charged_seconds': service,
                             'actual_prediction_seconds': elapsed, 'kind': kind,
                             'forecast_sha256': hashlib.sha256(m.tobytes()).hexdigest(),
                             'heard_entries': int(past.heard[-1].sum()),
                             'scans_last_bin': int(past.scanned[-1].sum())})
        return m, s

    def take(self, slot):
        index = int(np.searchsorted(self.ready, slot, side='right'))-1
        if index <= self.last:
            return None
        self.last, self.age = index, slot/100-index
        return index

    def advance(self, forecast):
        queue = ForecastQueue([0.])
        queue.age = self.age
        return queue.advance(forecast)
