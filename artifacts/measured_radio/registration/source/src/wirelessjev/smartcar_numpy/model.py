"""NumPy inference for the released, zero-learned-feature nested RF fields.

This is a deployment backend, not another fit or a different likelihood. The
original float32 weights and features precede float64 interval/Newton/Schur
calculations. All sixteen local coordinates are retained in this first port;
only the exactly zero-gated learned-feature MLP evaluation is omitted. The
discrepancy MLP remains active. Source identity is enforced by ``load_export``.
Neither posterior spread nor field covariance is a hardware risk certificate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.special import expit

from .geometry import query_physics_basis
from .likelihoods import gaussian_interval_log_prob, interval_moments

GLOBAL_DIM = 8
LOCAL_DIM = 16
JITTER = 1.e-8
BACKTRACKING_STEPS = (1., .5, .25, .125, .0625, .03125)
OBJECTIVE_TOLERANCE = 1.e-10
MAX_CONTEXT = 192
MAX_COVARIANCE_QUERIES = 512

STATE_SHAPES = {
    'global_prior_mean': (8,), 'local_prior_mean': (16,),
    'raw_global_std': (8,), 'raw_local_std': (16,),
    'raw_discrepancy': (), 'local_gates': (16,),
    'rff_omega': (8, 3), 'rff_phase': (8,),
    'kernel.0.weight': (32, 13), 'kernel.0.bias': (32,),
    'kernel.2.weight': (16, 32), 'kernel.2.bias': (16,),
    'kernel.4.weight': (8, 16), 'kernel.4.bias': (8,),
    'discrepancy.0.weight': (32, 10), 'discrepancy.0.bias': (32,),
    'discrepancy.2.weight': (1, 32), 'discrepancy.2.bias': (1,),
}


def validate_arrays(arrays):
    """Copy the fixed released architecture's inert finite float32 state.

    The archive loader additionally pins the original checkpoint and exact
    member-specific gate identity. This structural validator does not claim
    that arbitrary arrays carrying these names have a published provenance.
    """
    if not isinstance(arrays, dict) or set(arrays) != set(STATE_SHAPES):
        raise ValueError('The complete released eighteen-array state is required')
    result = {}
    for name, shape in STATE_SHAPES.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32')
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError('Malformed finite float32 weight array: ' + name)
        item = np.array(value, dtype=np.float32, copy=True, order='C')
        item.setflags(write=False)
        result[name] = item
    gates = result['local_gates']
    if ((gates < 0).any() or (gates > 1).any() or np.any(gates[8:] != 0)
            or np.count_nonzero(gates[:8]) not in (5, 6)):
        raise ValueError('Only released zero-learned-feature gates with five or six Fourier columns are supported')
    return result


def _float(value, name, *, dtype=None):
    if (not isinstance(value, np.ndarray) or value.dtype.kind != 'f'
            or (dtype is not None and value.dtype != np.dtype(dtype))):
        raise ValueError(name + ' must be a floating NumPy array of the declared precision')
    return value


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(name + ' must be finite')


def _chol(matrix, name):
    try:
        result = cholesky(.5 * (matrix + np.swapaxes(matrix, -1, -2)), lower=True, check_finite=True)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise FloatingPointError(name + ' lost positive definiteness') from error
    if not np.isfinite(result).all():
        raise FloatingPointError(name + ' lost positive definiteness')
    return result


def _softplus_f32(value):
    # The frozen implementation applies softplus in parameter precision, then
    # promotes to float64 before adding its .2 prior-SD offset.
    value = np.asarray(value, dtype=np.float32)
    return np.where(value > np.float32(20.), value,
                    np.logaddexp(np.float32(0.), value)).astype(np.float32)


def _block_factor(g, local, groups, weights, global_precision, local_precision):
    sites = len(local_precision)
    ag = global_precision + g.T @ (weights[:, None] * g)
    local_cholesky = np.empty((sites, LOCAL_DIM, LOCAL_DIM), dtype=np.float64)
    cross = np.empty((sites, GLOBAL_DIM, LOCAL_DIM), dtype=np.float64)
    coupling = np.empty((sites, LOCAL_DIM, GLOBAL_DIM), dtype=np.float64)
    for site in range(sites):
        selected = groups == site
        routed = local[selected]
        weighted = weights[selected, None] * routed
        ca = local_precision[site] + routed.T @ weighted
        cross[site] = g[selected].T @ weighted
        local_cholesky[site] = _chol(ca, 'AP precision')
        coupling[site] = cho_solve((local_cholesky[site], True), cross[site].T)
    schur = ag - np.einsum('agl,alh->gh', cross, coupling)
    return {'global_cholesky': _chol(schur, 'global Schur complement'),
            'local_cholesky': local_cholesky, 'coupling': coupling,
            'cross_precision': cross}


def _block_natural_solution(g, local, groups, weights, natural, gm, gp, lm, lp):
    factor = _block_factor(g, local, groups, weights, gp, lp)
    rg = gp @ gm + g.T @ natural
    local_rhs = np.empty_like(lm)
    for site in range(len(lm)):
        selected = groups == site
        ra = lp[site] @ lm[site] + local[selected].T @ natural[selected]
        local_rhs[site] = cho_solve((factor['local_cholesky'][site], True), ra)
    rhs = rg - np.einsum('agl,al->g', factor['cross_precision'], local_rhs)
    posterior_global = cho_solve((factor['global_cholesky'], True), rhs)
    posterior_local = local_rhs - np.einsum('alg,g->al', factor['coupling'], posterior_global)
    return posterior_global, posterior_local


def _context_objective(predicted, sigma, lower, upper, points, weights,
                       states, priors, precisions):
    safe_lower = np.where(points, predicted - sigma, lower)
    safe_upper = np.where(points, predicted + sigma, upper)
    interval = -gaussian_interval_log_prob(safe_lower, safe_upper, predicted, sigma)
    value = np.where(points, lower, predicted)
    density = .5 * ((value - predicted) / sigma) ** 2 + np.log(sigma) + .5 * math.log(2. * math.pi)
    result = float(np.sum(weights * np.where(points, density, interval)))
    for state, prior, precision in zip(states, priors, precisions):
        difference = state - prior
        result += .5 * float(np.einsum('...d,...de,...e->...', difference, precision, difference).sum())
    return result


def _bounded_step(current, proposed, objective):
    if any(not np.isfinite(value).all() for value in proposed):
        raise FloatingPointError('Nonfinite Newton direction')
    before = objective(current)
    chosen, after, accepted = 0., before, False
    tolerance = OBJECTIVE_TOLERANCE * max(1., abs(before))
    for fraction in BACKTRACKING_STEPS:
        candidate = tuple(a + fraction * (b - a) for a, b in zip(current, proposed))
        value = objective(candidate)
        if np.isfinite(value) and value <= before + tolerance:
            chosen, after, accepted = fraction, value, True
            break
    updated = tuple(a + chosen * (b - a) for a, b in zip(current, proposed))
    return updated, {'objective_before': before, 'objective_after': after,
                     'step_size': chosen, 'null_step': not accepted}


class NumpyField:
    """Unbatched, read-only inference matching a released ``nested_joint`` member.

    ``arrays`` are the full unchanged float32 state arrays returned by the
    pinned archive loader. This class never imports Torch or fits parameters.
    """

    settings = {'cell': 'nested_joint', 'width': 32, 'inference_steps': 3}

    def __init__(self, arrays):
        self.arrays = validate_arrays(arrays)
        self.global_prior_mean = self.arrays['global_prior_mean'].astype(np.float64)
        self.local_prior_mean = self.arrays['local_prior_mean'].astype(np.float64)
        gs = .2 + _softplus_f32(self.arrays['raw_global_std']).astype(np.float64)
        ls = .2 + _softplus_f32(self.arrays['raw_local_std']).astype(np.float64)
        self.global_precision = np.diag(gs ** -2 + JITTER)
        self.local_precision = np.diag(ls ** -2 + JITTER)

    def discrepancy_db(self, query):
        query = _float(query, 'Public geometry', dtype='float32')
        x = np.concatenate((query[..., :8], query[..., 9:11]), axis=-1)
        hidden = x @ self.arrays['discrepancy.0.weight'].T + self.arrays['discrepancy.0.bias']
        hidden = hidden * expit(hidden)
        scalar = (hidden @ self.arrays['discrepancy.2.weight'].T + self.arrays['discrepancy.2.bias'])[..., 0]
        return (np.float32(1.) + np.float32(14.) * expit(self.arrays['raw_discrepancy']
                + np.float32(.25) * np.tanh(scalar))).astype(np.float32)

    def local_features(self, query, rx_xyz, ap_xyz, groups):
        relative = rx_xyz - ap_xyz[groups]
        fixed = np.float32(.5) * np.cos(relative @ self.arrays['rff_omega'].T + self.arrays['rff_phase'])
        # The loader and constructor require these last eight gates to equal
        # zero exactly. Their original MLP weights are still bound by export.
        features = np.concatenate((fixed, np.zeros((len(fixed), 8), dtype=np.float32)), axis=-1)
        return features * self.arrays['local_gates']

    def _geometry(self, q, groups, rx, aps, *, scheduled=None):
        for value, name in ((q, 'Public geometry'), (rx, 'Physical RX XYZ'), (aps, 'Physical AP XYZ')):
            _float(value, name, dtype='float32')
        if (q.ndim != 2 or q.shape[1] != 16 or rx.shape != (len(q), 3)
                or aps.ndim != 2 or aps.shape[1] != 3 or not 1 <= len(aps) <= 128
                or not isinstance(groups, np.ndarray) or groups.shape != (len(q),)
                or groups.dtype not in (np.dtype('int32'), np.dtype('int64'))):
            raise ValueError('Unbatched public geometry, AP routes and physical positions disagree')
        _finite(aps, 'Physical AP positions')
        mask = np.ones(len(q), dtype=bool) if scheduled is None else scheduled
        if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype('bool') or mask.shape != (len(q),):
            raise ValueError('Scheduled geometry mask disagrees')
        if ((groups[mask] < 0) | (groups[mask] >= len(aps))).any():
            raise ValueError('Scheduled AP route outside public AP set')
        _finite(q[mask], 'Scheduled geometry'); _finite(rx[mask], 'Scheduled RX positions')
        safe_groups = np.where(mask, groups, 0).astype(np.int64)
        safe_q = np.where(mask[:, None], q, np.float32(0.))
        safe_rx = np.where(mask[:, None], rx, np.float32(0.))
        distance = np.linalg.norm(safe_rx - aps[safe_groups], axis=-1)
        if (distance[mask] <= 0).any():
            raise ValueError('AP and receiver positions cannot coincide')
        if mask.any() and not np.allclose(np.float32(2.) * safe_q[mask, 6],
                np.log10(distance[mask]), atol=2.e-4, rtol=2.e-4):
            raise ValueError('Geometry distance disagrees with physical-metre positions')
        return safe_q, safe_groups, safe_rx, aps

    def encode_view(self, view, ap_xyz):
        if not isinstance(view, dict):
            raise ValueError('Explicit context event mapping is required')
        context = _float(view.get('context'), 'Context', dtype='float32')
        known, scheduled = view.get('known'), view.get('scheduled')
        if (context.ndim != 2 or context.shape[1] != 20 or not 1 <= len(context) <= MAX_CONTEXT
                or not isinstance(known, np.ndarray) or not isinstance(scheduled, np.ndarray)
                or known.dtype != np.dtype('bool') or scheduled.dtype != np.dtype('bool')
                or known.shape != (len(context),) or scheduled.shape != known.shape
                or (known & ~scheduled).any()):
            raise ValueError('Explicit unbatched context20/masks and at most192 observations required')
        _finite(context[scheduled], 'Scheduled context')
        context = np.where(scheduled[:, None], context, np.float32(0.))
        q, groups, rx, aps = self._geometry(context[:, 4:], view.get('ap'), view.get('rx_xyz'),
                                           ap_xyz, scheduled=scheduled)
        heard = known & (context[:, 2] > .5)
        censored = known & (context[:, 3] > .5)
        if (heard & censored).any() or not np.array_equal(heard | censored, known):
            raise ValueError('Known report semantics must be heard OR censored')
        lower, upper, floor = (view.get(name) for name in ('lower', 'upper', 'floor'))
        for value, name in ((lower, 'Residual lower'), (upper, 'Residual upper'), (floor, 'Public residual floor')):
            _float(value, name)
            if value.shape != known.shape:
                raise ValueError(name + ' axes differ')
        if (np.isnan(lower[known]).any() or np.isnan(upper[known]).any()
                or not np.isfinite(floor[scheduled]).all() or (lower[known] > upper[known]).any()
                or not np.isfinite(lower[heard]).all() or not np.isfinite(upper[heard]).all()
                or not np.isneginf(lower[censored]).all() or not np.isfinite(upper[censored]).all()
                or (lower[heard] + 1.e-4 < floor[heard]).any()):
            raise ValueError('Heard intervals/censor bounds/public floors are inconsistent')
        weights = known.astype(np.float64) if 'weights' not in view else _float(view['weights'], 'Weights').astype(np.float64)
        if (weights.shape != known.shape or not np.isfinite(weights).all()
                or (weights < 0).any() or (weights > 1).any() or (weights[~known] != 0).any()):
            raise ValueError('Explicit duplicate weights must be in0..1 and zero for unknown rows')
        lo, hi = lower.astype(np.float64) / 10., upper.astype(np.float64) / 10.
        points = heard & (lo == hi)
        lo = np.where(known, lo, -np.inf); hi = np.where(known, hi, np.inf)
        g = query_physics_basis(q, context=True).astype(np.float64)
        local = self.local_features(q, rx, aps, groups).astype(np.float64)
        sigma = self.discrepancy_db(q).astype(np.float64) / 10.
        gm = self.global_prior_mean
        lm = np.broadcast_to(self.local_prior_mean, (len(aps), LOCAL_DIM)).copy()
        gp = self.global_precision
        lp = np.broadcast_to(self.local_precision, (len(aps), LOCAL_DIM, LOCAL_DIM))
        state_g, state_l = gm.copy(), lm.copy()
        def predicted(states):
            return np.sum(g * states[0], axis=-1) + np.sum(local * states[1][groups], axis=-1)
        def objective(states):
            return _context_objective(predicted(states), sigma, lo, hi, points, weights,
                                      states, (gm, lm), (gp, lp))
        inference_log = []
        for _ in range(3):
            prediction = predicted((state_g, state_l))
            expected, conditional_var = interval_moments(prediction, sigma, lo, hi, point_mask=points)
            score = (expected - prediction) / sigma ** 2
            curvature = np.clip(1. - conditional_var / sigma ** 2, 0., 1.) / sigma ** 2
            information = weights * curvature
            natural = weights * (curvature * prediction + score)
            proposed = _block_natural_solution(g, local, groups, information, natural, gm, gp, lm, lp)
            (state_g, state_l), record = _bounded_step((state_g, state_l), proposed, objective)
            inference_log.append(record)
        prediction = predicted((state_g, state_l))
        _, conditional_var = interval_moments(prediction, sigma, lo, hi, point_mask=points)
        information = weights * np.clip(1. - conditional_var / sigma ** 2, 0., 1.) / sigma ** 2
        factor = _block_factor(g, local, groups, information, gp, lp)
        belief = {'global_mean': state_g, 'local_mean': state_l,
            'global_covariance': cho_solve((factor['global_cholesky'], True), np.eye(GLOBAL_DIM)),
            'local_cholesky': factor['local_cholesky'], 'local_coupling': factor['coupling'],
            'mode': 'hierarchy', 'ap_xyz': aps.copy(), 'ap_count': len(aps),
            'observed_ap': np.array([(known & (groups == site)).any() for site in range(len(aps))]),
            'field_covariance_qualified': False, 'packet_risk_qualified': False,
            'inference_log': inference_log}
        if not all(np.isfinite(value).all() for value in belief.values() if isinstance(value, np.ndarray)):
            raise FloatingPointError('Nonfinite NumPy hierarchical belief')
        return belief

    def _query_design(self, belief, query, query_ap, query_rx, basis):
        q, groups, rx, aps = self._geometry(query, query_ap, query_rx, belief['ap_xyz'])
        _float(basis, 'Explicit query basis')
        expected = query_physics_basis(q, context=True)
        if (basis.shape != (len(q), GLOBAL_DIM) or not np.isfinite(basis).all()
                or not np.allclose(basis.astype(expected.dtype), expected, atol=2.e-5, rtol=2.e-5)):
            raise ValueError('Query basis disagrees with public physical geometry')
        g = basis.astype(np.float64)
        local = self.local_features(q, rx, aps, groups).astype(np.float64)
        shared = np.empty_like(g)
        for site in range(belief['ap_count']):
            selected = groups == site
            shared[selected] = g[selected] - local[selected] @ belief['local_coupling'][site]
        return shared, local, groups, q

    def render_queries(self, belief, query):
        if not isinstance(query, dict):
            raise ValueError('Explicit unbatched query mapping required')
        shared, local, groups, q = self._query_design(belief, query['q'], query['ap'], query['rx_xyz'], query['basis'])
        fspl, offset = query['fspl'], query['offset']
        for value, name in ((fspl, 'Known FSPL'), (offset, 'Estimated offset')):
            _float(value, name)
            if value.shape != (len(q),) or not np.isfinite(value).all():
                raise ValueError(name + ' must match the finite query axes')
        residual = np.sum(query['basis'].astype(np.float64) * belief['global_mean'], axis=-1)
        residual += np.sum(local * belief['local_mean'][groups], axis=-1)
        variance = np.zeros(len(q), dtype=np.float64)
        for site in range(belief['ap_count']):
            selected = groups == site
            if selected.any():
                design = local[selected]
                solved = cho_solve((belief['local_cholesky'][site], True), design.T).T
                variance[selected] = np.sum(design * solved, axis=-1)
        variance += np.einsum('qd,de,qe->q', shared, belief['global_covariance'], shared)
        mean = fspl.astype(np.float64) + offset.astype(np.float64) + 10. * residual
        std = np.sqrt(100. * np.maximum(variance, 0.) + self.discrepancy_db(q).astype(np.float64) ** 2)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise FloatingPointError('Nonfinite NumPy RF marginal prediction')
        return mean, std

    def predictive_covariance(self, belief, query, query_ap=None, query_rx=None, basis=None):
        """Coefficient-field covariance in dB², excluding discrepancy/fading."""
        if isinstance(query, dict):
            if any(value is not None for value in (query_ap, query_rx, basis)):
                raise ValueError('Use either query mapping or expanded query arrays')
            query, query_ap, query_rx, basis = (query[k] for k in ('q', 'ap', 'rx_xyz', 'basis'))
        if not isinstance(query, np.ndarray) or query.ndim != 2 or len(query) > MAX_COVARIANCE_QUERIES:
            raise ValueError('Full covariance is restricted to at most512 public probe queries')
        shared, local, groups, _ = self._query_design(belief, query, query_ap, query_rx, basis)
        covariance = shared @ belief['global_covariance'] @ shared.T
        for site in range(belief['ap_count']):
            selected = groups == site
            if selected.any():
                design = local[selected]
                solved = cho_solve((belief['local_cholesky'][site], True), design.T)
                covariance[np.ix_(selected, selected)] += design @ solved
        return covariance * 100.
