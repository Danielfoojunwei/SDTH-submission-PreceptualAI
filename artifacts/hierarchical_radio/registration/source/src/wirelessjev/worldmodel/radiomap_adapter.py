"""Exact old outage-utility adapter; no model-supplied qualification accepted."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
from scipy.stats import gamma

from ..frontier.model import failure_table
from ..native.phy import LinkCurves
from .interface import BANDS, CandidateAction, OutcomeDistribution, SceneState, finite_nonnegative


def integrate(probability, times, start, end):
    """Piecewise-linear integral, matching old integral_from on its nine knots."""
    if not 0 <= start <= end <= times[-1]:
        raise ValueError('Integration bounds outside forecast support')
    if start == end:
        return np.zeros(probability.shape[:1] + probability.shape[2:])
    x = np.r_[start, times[(times > start) & (times < end)], end]
    right = np.searchsorted(times, x, side='right').clip(1, len(times) - 1)
    left = right - 1
    weights = ((x - times[left]) / (times[right] - times[left])).reshape(
        (1, len(x)) + (1,) * (probability.ndim - 2))
    y = probability[:, left] * (1 - weights) + probability[:, right] * weights
    return np.trapezoid(y, x, axis=1)


class RadioMapAdapter:
    """Render only supported AP/band/MCS candidates from predicted SNR.

    Default risk is exactly the current gamma-fade/three-retry/blockage
    surrogate, not calibrated real packet risk. Explicit MCS uses that same
    assumed fade/retry law and existing PHY curves. Rate is a PHY proxy; no
    airtime allocation, queue delay, burst silence or MLO model is invented.
    """
    def __init__(self):
        self.curves = LinkCurves()
        self.grid = np.linspace(-40, 60, 501)
        self.table = failure_table()

    @lru_cache(maxsize=64)
    def _mcs_failure(self, band, mcs):
        if mcs >= self.curves.n_mcs(band):
            raise ValueError('MCS is outside the actual band curve table')
        fades = 10 * np.log10(gamma.ppf((np.arange(32) + .5) / 32, 2, scale=.5))
        blocked = (1.5 / 60 / 4) / (1.5 / 60 / 4 + .5)
        index = np.full((501, 32), mcs, dtype=int)
        def fail(offset):
            return np.mean((1 - self.curves.success(
                band, index, self.grid[:, None] + fades + offset)) ** 3, axis=1)
        return ((1 - blocked) * fail(0) + blocked * fail(-15)).astype('f4')

    def query(self, scene_state: SceneState, candidate_actions: Sequence[CandidateAction],
              horizon: float, timescale='mid'):
        if not isinstance(scene_state, SceneState):
            raise TypeError('Only a predicted SceneState is accepted')
        if timescale not in ('fast', 'mid', 'slow'):
            raise ValueError('Unsupported timescale')
        horizon = finite_nonnegative(horizon, 'horizon')
        if not 0 < horizon <= scene_state.times_s[-1]:
            raise ValueError('Requested horizon is outside the prediction support')
        actions = tuple(candidate_actions)
        if not actions or any(not isinstance(a, CandidateAction) for a in actions):
            raise TypeError('Use explicit supported CandidateAction records')
        if len({a.action_id for a in actions}) != len(actions):
            raise ValueError('Duplicate action identities')
        snr, times = scene_state.snr_db, scene_state.times_s
        n, h, sites, _ = snr.shape
        risk, rate = np.empty((n, len(actions), h)), np.empty((n, len(actions), h))
        integral, switches = np.empty((n, len(actions))), np.empty((n, len(actions)))
        for j, action in enumerate(actions):
            if action.ap >= sites or action.handover_seconds > horizon:
                raise ValueError('Action AP or switching cost is outside forecast support')
            link = snr[:, :, action.ap]
            if action.band is None:
                all_risk = np.stack([np.interp(link[..., b], self.grid, self.table[b])
                                     for b in range(3)], -1)
                selected = all_risk.argmin(-1)
                p = all_risk.min(-1)
                r = np.stack([self.curves.rates(band)[0] * self.curves.success(
                    band, np.zeros((n, h), dtype=int), link[..., b]) for b, band in enumerate(BANDS)], -1)
                rate[:, j] = np.take_along_axis(r, selected[..., None], -1)[..., 0]
            else:
                b = BANDS.index(action.band)
                table = self.table[b] if action.mcs is None else self._mcs_failure(action.band, action.mcs)
                p = np.interp(link[..., b], self.grid, table)
                mcs = 0 if action.mcs is None else action.mcs
                rate[:, j] = self.curves.rates(action.band)[mcs] * self.curves.success(
                    action.band, np.full((n, h), mcs, dtype=int), link[..., b])
            switching = np.where(scene_state.current_ap == action.ap, 0., action.handover_seconds)
            during_switch = times[None] < switching[:, None]
            risk[:, j] = np.where(during_switch, 1., p)
            rate[:, j] = np.where(during_switch, 0., rate[:, j])
            stay = integrate(p, times, 0., horizon)
            move = action.handover_seconds + integrate(p, times, action.handover_seconds, horizon)
            integral[:, j] = np.where(scene_state.current_ap == action.ap, stay, move)
            switches[:, j] = switching
        return OutcomeDistribution(tuple(a.action_id for a in actions), times, risk, rate,
            integral, switches, scene_state.source_id,
            {'world_model': 'RadioMapAdapter', 'risk': 'legacy_gamma_fading_three_retry_blockage_surrogate',
             'calibration': 'uncalibrated', 'rate': 'PHY_success_times_MCS_rate_proxy_not_MAC_goodput',
             'silence': 'unknown', 'latency': 'unknown',
             '6g_curves': 'historical_5g_error_curve_and_doubled_rate_assumption',
             'intra_AP_band_switch': 'zero_cost_native_assumption',
             'integral': 'exact_declared_switch_boundary_plus_piecewise_linear_post_switch_risk',
             'qualification': 'independent_gate_required', 'requested_timescale': timescale},
            horizon_seconds=horizon)
