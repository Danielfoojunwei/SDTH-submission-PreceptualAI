"""Integrate a PHY surrogate from the exact end of a handover outage."""
import numpy as np
from ..native.world import HORIZONS
from ..frontier.model import failure_table


def integral_from(probability, start, end):
    if not 0 <= start <= end <= HORIZONS[-1]:
        raise ValueError('Integration bounds outside forecast support')
    p = np.asarray(probability)
    h = np.asarray(HORIZONS, dtype=float)
    if p.shape[1] != len(h):
        raise ValueError('Wrong forecast horizon axis')
    if start == end:
        return np.zeros(p.shape[:1]+p.shape[2:])
    x = np.r_[start, h[(h > start) & (h < end)], end]
    right = np.searchsorted(h, x, side='right').clip(1, len(h)-1)
    left = right-1
    weights = ((x-h[left])/(h[right]-h[left])).reshape((1, len(x))+(1,)*(p.ndim-2))
    y = p[:, left]*(1-weights) + p[:, right]*weights
    return np.trapezoid(y, x, axis=1)


class OutageUtility:
    def __init__(self, forecasts, queue, cost, horizon):
        if not 0 <= cost <= horizon <= HORIZONS[-1]:
            raise ValueError('Invalid outage/horizon')
        self.forecasts, self.queue = forecasts, queue
        self.cost, self.horizon = cost, horizon
        self.table = failure_table()
        self.grid = np.linspace(-40, 60, 501)

    def loss(self, forecast):
        probability = np.stack([np.interp(forecast[..., b], self.grid, self.table[b]) for b in range(3)], -1).min(-1)
        stay = integral_from(probability, 0, self.horizon)
        move = self.cost + integral_from(probability, self.cost, self.horizon)
        return stay, move

    def scores(self, forecast, current):
        stay, move = self.loss(forecast)
        score = -move
        ar = np.arange(len(current))
        score[ar, current] = -stay[ar, current]
        return score

    def __call__(self, decision):
        forecast = self.queue.advance(self.forecasts[decision.decision])
        stay, move = self.loss(forecast)
        ar = np.arange(len(stay))
        return decision.eligible & (stay[ar, decision.current]-move[ar, decision.candidate] > 1e-7)
