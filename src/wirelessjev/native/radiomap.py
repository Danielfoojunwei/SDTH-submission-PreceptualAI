"""RadioJEV map v0: a learned per-BSS radio map with calibrated uncertainty.

Robots report what a real client scan gives: the eight strongest BSSIDs per
scan, measured RSSI with fast fading and occasional blockage, nothing below
-90 dBm. Each BSS gets a Bayesian linear model over known geometry (log
distance, rack rows crossed, from the site CAD) plus random Fourier features
for the unknown shadowing. The posterior gives a mean map and the standard
deviation of that mean; the trust gate uses the latter.

Two corrections make the uncertainty usable (an uncorrected posterior was
~1 dB wide against ~4.5 dB real error on development scene 400):
- non-reports are censored bounds (below the 8th strongest report, or below
  -90 dBm), fitted by Tobit EM, so lucky-fade selection does not lift the map;
- a model-discrepancy variance is estimated from held-out robots and added
  to the posterior variance.
Calibration is measured, never assumed.
"""
from __future__ import annotations

import math

import numpy as np

from ..twin.warehouse import RACK_DB, rack_obstruction

N_RFF = 48
RFF_SCALE_M = 10.0
SHADOW_PRIOR_DB = 4.0
NOISE_DB = 4.5            # fading + blockage residual, used in the likelihood
SCAN_TOP = 8
FLOOR_DBM = -90.0
CENSOR_THIN = 5           # keep non-report bounds from every 5th scan...
CENSOR_WEIGHT = 5.0       # ...and weight them back up
EM_ITERS = 12
DISC_SCALE_M = 8.0        # discrepancy grows with distance from reports


GENERIC_PRIOR = {b: {"mean": np.array([-40.0, -33.0, -3.5]),
                     "var": np.array([30.0 ** 2, 30.0 ** 2, 10.0 ** 2]),
                     "shadow_db": SHADOW_PRIOR_DB} for b in range(3)}


class RadioMap:
    def __init__(self, sites: np.ndarray, aps5, seed: int = 0, prior: dict | None = None):
        """prior: per band {"mean": (3,), "var": (3,), "shadow_db": float} for
        the geometry coefficients (intercept, log10 distance, rack rows) and
        the shadowing features. The generic default knows only free-space
        physics; `native.foundation.learn_prior` learns one across sites."""
        self.sites = sites
        self.aps5 = aps5
        self.prior = prior or GENERIC_PRIOR
        rng = np.random.default_rng(30_000 + seed)
        self.w = rng.normal(0, 1 / RFF_SCALE_M, size=(len(sites), 3, N_RFF, 2))
        self.ph = rng.uniform(0, 2 * math.pi, size=(len(sites), 3, N_RFF))
        self.post = {}

    def _features(self, s: int, b: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = np.maximum(1.0, np.hypot(x - self.sites[s, 0], y - self.sites[s, 1]))
        racks = rack_obstruction(self.aps5[s], x, y) / RACK_DB["5g"]
        arg = x[:, None] * self.w[s, b, :, 0] + y[:, None] * self.w[s, b, :, 1] + self.ph[s, b]
        rff = math.sqrt(2.0 / N_RFF) * np.cos(arg)
        return np.column_stack([np.ones_like(x), np.log10(d), racks, rff])

    def _prior_precision(self, b: int = 1) -> np.ndarray:
        pr = self.prior[b]
        var = np.concatenate([pr["var"], np.full(N_RFF, pr["shadow_db"] ** 2)])
        return np.diag(1 / var)

    def _fit_one(self, s, b, x, y, r, cx, cy, cb):
        P0 = self._prior_precision(b)
        m0 = np.zeros(P0.shape[0])
        m0[:3] = self.prior[b]["mean"]
        Phi = self._features(s, b, x, y) if len(x) else np.zeros((0, P0.shape[0]))
        Pc = self._features(s, b, cx, cy) if len(cx) else np.zeros((0, P0.shape[0]))
        A0 = P0 + Phi.T @ Phi / NOISE_DB ** 2
        w = np.linalg.solve(A0, P0 @ m0 + Phi.T @ r / NOISE_DB ** 2)   # reports-only start
        A = A0 + CENSOR_WEIGHT * Pc.T @ Pc / NOISE_DB ** 2
        Ainv = np.linalg.inv(A)
        rhs = P0 @ m0 + Phi.T @ r / NOISE_DB ** 2
        for _ in range(EM_ITERS if len(cx) else 0):
            # E-step: expected value of a censored reading given the current
            # mean, E[y | y < bound] for a Gaussian measurement.
            mu = Pc @ w
            a = (cb - mu) / NOISE_DB
            from scipy.stats import norm
            imputed = mu - NOISE_DB * norm.pdf(a) / np.maximum(norm.cdf(a), 1e-12)
            rhs = P0 @ m0 + Phi.T @ r / NOISE_DB ** 2 + CENSOR_WEIGHT * Pc.T @ imputed / NOISE_DB ** 2
            w = Ainv @ rhs
        return w, Ainv, A, rhs

    def fit(self, obs: dict, cens: dict | None = None, holdout: np.ndarray | None = None) -> "RadioMap":
        """obs[(s, b)] = (x, y, rssi, robot); cens[(s, b)] = (x, y, bound, robot).

        With `holdout` (robot ids), a discrepancy variance is estimated from
        those robots' reports and the final map is refitted on everyone."""
        cens = cens or {}
        empty = (np.zeros(0),) * 4
        self.seen = {}

        def fit_all(keep):
            for s in range(len(self.sites)):
                for b in range(3):
                    x, y, r, rob = obs.get((s, b), empty)
                    cx, cy, cb, crob = cens.get((s, b), empty)
                    k, kc = keep(rob), keep(crob)
                    self.post[(s, b)] = self._fit_one(s, b, x[k], y[k], r[k], cx[kc], cy[kc], cb[kc])
                    self.seen[(s, b)] = np.column_stack([x[k], y[k]]) if k.any() else np.zeros((0, 2))

        self.disc_var = 0.0
        self.disc_far = 0.0
        if isinstance(holdout, str) and holdout == "spatial":
            # Hold out whole quadrants: residual variance as a function of the
            # distance to the nearest training report of that BSS.
            from scipy.spatial import cKDTree
            mx, my = self.sites[:, 0].mean(), self.sites[:, 1].mean()
            rows = []
            for fold in range(4):
                def inside(x, y, fold=fold):
                    return ((x >= mx) == bool(fold & 1)) & ((y >= my) == bool(fold & 2))
                for s in range(len(self.sites)):
                    for b in range(3):
                        x, y, r, rob = obs.get((s, b), empty)
                        if len(x) == 0:
                            continue
                        cx, cy, cb, crob = cens.get((s, b), empty)
                        tr, te = ~inside(x, y), inside(x, y)
                        trc = ~inside(cx, cy) if len(cx) else np.zeros(0, dtype=bool)
                        if not te.any() or not tr.any():
                            continue
                        w, Ainv = self._fit_one(s, b, x[tr], y[tr], r[tr], cx[trc], cy[trc], cb[trc])[:2]
                        Phi = self._features(s, b, x[te], y[te])
                        d = cKDTree(np.column_stack([x[tr], y[tr]])).query(np.column_stack([x[te], y[te]]))[0]
                        pv = np.einsum("ij,jk,ik->i", Phi, Ainv, Phi)
                        rows.append(np.column_stack([(r[te] - Phi @ w) ** 2 - NOISE_DB ** 2 - pv,
                                                     1 - np.exp(-d / DISC_SCALE_M)]))
            if rows:
                R = np.concatenate(rows)
                X = np.column_stack([np.ones(len(R)), R[:, 1]])
                from scipy.optimize import nnls
                (self.disc_var, self.disc_far), _ = nnls(X, R[:, 0])
            holdout = None
        if holdout is not None and len(holdout):
            fit_all(lambda rob: ~np.isin(rob, holdout))
            res, pv = [], []
            for (s, b), (x, y, r, rob) in obs.items():
                k = np.isin(rob, holdout)
                if not k.any():
                    continue
                Phi = self._features(s, b, x[k], y[k])
                w, Ainv = self.post[(s, b)][:2]
                res.append(r[k] - Phi @ w)
                pv.append(np.einsum("ij,jk,ik->i", Phi, Ainv, Phi))
            res, pv = np.concatenate(res), np.concatenate(pv)
            # E[res^2] = measurement noise + posterior variance + discrepancy.
            self.disc_var = float(max(0.0, np.mean(res ** 2) - NOISE_DB ** 2 - np.mean(pv)))
        fit_all(lambda rob: np.ones(len(rob), dtype=bool))
        return self

    def update(self, s: int, b: int, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> None:
        """Add reports gathered during operation (Bayesian linear update)."""
        if len(x) == 0:
            return
        w, Ainv, A, rhs = self.post[(s, b)]
        Phi = self._features(s, b, x, y)
        A = A + Phi.T @ Phi / NOISE_DB ** 2
        rhs = rhs + Phi.T @ r / NOISE_DB ** 2
        Ainv = np.linalg.inv(A)
        self.post[(s, b)] = (Ainv @ rhs, Ainv, A, rhs)
        new = np.column_stack([x, y])
        self.seen[(s, b)] = np.vstack([self.seen.get((s, b), np.zeros((0, 2))), new])

    def predict(self, x: np.ndarray, y: np.ndarray):
        """Mean RSSI and posterior std of the mean, shape x.shape + (S, 3)."""
        fx, fy = x.ravel(), y.ravel()
        mean = np.empty(fx.shape + (len(self.sites), 3), dtype=np.float32)
        std = np.empty_like(mean)
        from scipy.spatial import cKDTree
        for (s, b), post in self.post.items():
            w, Ainv = post[:2]
            Phi = self._features(s, b, fx, fy)
            mean[:, s, b] = Phi @ w
            var = np.maximum(np.einsum("ij,jk,ik->i", Phi, Ainv, Phi), 1e-6) + self.disc_var
            seen = getattr(self, "seen", {}).get((s, b))
            if getattr(self, "disc_far", 0.0) > 0:
                if seen is not None and len(seen):
                    d = cKDTree(seen).query(np.column_stack([fx, fy]))[0]
                else:
                    d = np.full(len(fx), np.inf)
                var = var + self.disc_far * (1 - np.exp(-d / DISC_SCALE_M))
            std[:, s, b] = np.sqrt(var)
        return mean.reshape(x.shape + mean.shape[1:]), std.reshape(x.shape + std.shape[1:])


def survey(rssi_true: np.ndarray, pos: np.ndarray, rng):
    """Capped scans: rssi_true (T, N, S, 3) mean field, pos (T, N, 2).

    Each robot reports its 8 strongest measured BSSs per scan, >= -90 dBm.
    Returns (reports, censored bounds), each keyed by (site, band) with
    (x, y, value, robot) arrays."""
    t, n, s, b = rssi_true.shape
    fade = 10 * np.log10(rng.gamma(2, 0.5, size=rssi_true.shape))
    block = (rng.random((t, n, s)) < 0.05)[..., None] * 15.0
    meas = (rssi_true + fade - block).reshape(t, n, s * b)
    order = np.argsort(-meas, axis=2)
    obs, cens = {}, {}

    def add(store, key, x, y, v, i):
        store.setdefault(key, ([], [], [], []))
        for lst, val in zip(store[key], (x, y, v, i)):
            lst.append(val)

    for k in range(t):
        for i in range(n):
            top = [j for j in order[k, i, :SCAN_TOP] if meas[k, i, j] >= FLOOR_DBM]
            for j in top:
                add(obs, (int(j) // b, int(j) % b), pos[k, i, 0], pos[k, i, 1], meas[k, i, j], i)
            if k % CENSOR_THIN:
                continue
            bound = meas[k, i, top[-1]] if len(top) == SCAN_TOP else FLOOR_DBM
            heard = set(int(j) for j in top)
            for j in range(s * b):
                if j not in heard:
                    add(cens, (j // b, j % b), pos[k, i, 0], pos[k, i, 1], bound, i)
    pack = lambda d: {k: tuple(np.array(a) for a in v) for k, v in d.items()}
    return pack(obs), pack(cens)


def calibration(rssi_true: np.ndarray, mean: np.ndarray, std: np.ndarray,
                audible_dbm: float = -90.0) -> dict:
    """Error and interval coverage of the mean map on audible links."""
    m = rssi_true >= audible_dbm
    err = (mean - rssi_true)[m]
    z = err / std[m]
    return {"rmse_db": float(np.sqrt(np.mean(err ** 2))), "bias_db": float(err.mean()),
            "coverage_68": float(np.mean(np.abs(z) < 1.0)),
            "coverage_90": float(np.mean(np.abs(z) < 1.645)),
            "mean_predicted_std_db": float(std[m].mean()), "links": int(m.sum())}
