"""E7: planning against interference and attenuation in environments we do
not know, by learning the radio environment from the robots themselves.

A hidden ground truth is drawn per environment: path-loss exponent, shadowing
strength, AP mounting height, per-row rack materials and heights (3-D: a path
can pass over a low rack), concrete walls, non-Wi-Fi interferers, adjacent-
channel leakage, uplink interference from co-channel *robots*, and random
forklift blockage on future links. The planner never sees it.

What the planner does see is what the SmartCar scripts already report: every
5 s each robot's scan (RSSI of its 8 strongest BSSIDs, with fading), its link
bitrate, and channel busy time. From that it learns, per BSSID, a radio map
with uncertainty (random-feature Bayesian regression over position on top of a
textbook path-loss prior), plus each AP's external interference and each
channel's external occupancy. The fast planner then optimises on that belief,
optionally pessimistically (mean - k * std).

    uv run python -m wirelessjev.envlearn --envs 15
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from .fastplan import BatchEvaluator, FastPlanner
from .missions import WAREHOUSE_MISSIONS
from .oracle import horizon_snapshots
from .teachers.assign import legacy_policy
from .twin.radio import BANDS, CHANNELS, MAC_EFFICIENCY, noise_dbm, phy_rate_mbps
from .twin import warehouse as W

OUT = Path("artifacts/envlearn")
ROBOT_H = 0.3


# ------------------------------------------------------------------ truth
@dataclass
class EnvSpec:
    seed: int
    pl_coef: float
    shadow_db: float
    ap_height: float
    rack_db: np.ndarray          # per rack row, 5 GHz loss per crossing
    rack_h: np.ndarray           # per rack row, height in m
    walls: list                  # (x0, y0, x1, y1, dB)
    interferers: list            # (x, y, band, channel, dBm, duty)
    occupancy: dict
    fading_db: float
    blockage_p: float
    blockage_db: float
    leak_db: float = -28.0


def random_env(seed: int) -> EnvSpec:
    rng = np.random.default_rng(10_000 + seed)
    rows = len(W.RACK_ROWS)
    stock = rng.choice([0.5, 1.0, 1.8, 3.0], size=rows, p=[0.25, 0.35, 0.25, 0.15])
    walls = []
    for _ in range(int(rng.integers(0, 4))):
        x = float(rng.uniform(25, 95))
        gap = float(rng.uniform(10, 60))
        walls.append((x, 0.0, x, gap - 3.0, float(rng.uniform(8, 20))))
        walls.append((x, gap + 3.0, x, W.HEIGHT, float(rng.uniform(8, 20))))
    interferers = []
    for _ in range(int(rng.integers(0, 5))):
        band = "2g" if rng.random() < 0.6 else "5g"
        interferers.append((float(rng.uniform(0, W.WIDTH)), float(rng.uniform(0, W.HEIGHT)),
                            band, int(rng.choice(CHANNELS[band])), float(rng.uniform(5, 20)),
                            float(rng.uniform(0.1, 0.6))))
    occ = {}
    for band, chans in CHANNELS.items():
        for ch in chans:
            occ[(band, ch)] = float(rng.uniform(0.05, 0.5)) if rng.random() < 0.35 else 0.0
    for x, y, band, ch, p, duty in interferers:
        occ[(band, ch)] = min(0.9, occ[(band, ch)] + duty * 0.5)
    return EnvSpec(seed, pl_coef=float(rng.uniform(22, 38)), shadow_db=float(rng.uniform(2, 8)),
                   ap_height=float(rng.uniform(4, 10)), rack_db=stock * 3.5,
                   rack_h=rng.uniform(2.5, 9.0, size=rows), walls=walls,
                   interferers=interferers, occupancy=occ,
                   fading_db=float(rng.uniform(2, 5)), blockage_p=float(rng.uniform(0.0, 0.15)),
                   blockage_db=float(rng.uniform(5, 15)))


def obstruction_fn(spec: EnvSpec):
    def f(ap, xs, ys):
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        band_scale = 1.0 if ap.band == "5g" else 0.7
        loss = np.zeros_like(xs)
        dy = ys - ap.y
        lo, hi = np.minimum(ys, ap.y), np.maximum(ys, ap.y)
        for k, yr in enumerate(W.RACK_ROWS):
            spans = (lo < yr) & (hi > yr)
            if not spans.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(dy != 0, (yr - ap.y) / dy, 0.0)
            xc = ap.x + t * (xs - ap.x)
            z = spec.ap_height + t * (ROBOT_H - spec.ap_height)        # path height at the rack
            inside = np.zeros_like(xs, bool)
            for x0, x1 in W.RACK_SEGMENTS:
                inside |= (xc >= x0) & (xc <= x1)
            loss += np.where(spans & inside & (z < spec.rack_h[k]), spec.rack_db[k] * band_scale, 0.0)
        for x0, y0, x1, y1, db in spec.walls:
            # segment intersection between AP->point and the wall
            d1x, d1y = xs - ap.x, ys - ap.y
            d2x, d2y = x1 - x0, y1 - y0
            den = d1x * d2y - d1y * d2x
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = ((x0 - ap.x) * d2y - (y0 - ap.y) * d2x) / den
                tb = ((x0 - ap.x) * d1y - (y0 - ap.y) * d1x) / den
            hit = (den != 0) & (ta > 0) & (ta < 1) & (tb >= 0) & (tb <= 1)
            loss += np.where(hit, db * band_scale, 0.0)
        return np.minimum(loss, 45.0)
    return f


def truth_scene(seed: int, spec: EnvSpec):
    base = W.make_warehouse(seed)
    sc = replace(base, obstruction=obstruction_fn(spec), pl_coef=spec.pl_coef,
                 shadow_db=spec.shadow_db, ap_height_m=spec.ap_height,
                 external_occupancy=dict(spec.occupancy), _radio=None, _radio_key=None)
    return sc.with_associations(legacy_policy(sc))


def leak_matrix(scene, leak_db: float) -> np.ndarray:
    m = len(scene.aps)
    L = np.zeros((m, m))
    for j, a in enumerate(scene.aps):
        for k, b in enumerate(scene.aps):
            if j != k and a.band == b.band:
                chans = CHANNELS[a.band]
                if abs(chans.index(a.channel) - chans.index(b.channel)) == 1:
                    L[k, j] = 10 ** (leak_db / 10)
    return L


def external_noise(scene, spec: EnvSpec) -> np.ndarray:
    out = np.zeros(len(scene.aps))
    for j, ap in enumerate(scene.aps):
        for x, y, band, ch, p, duty in spec.interferers:
            if band == ap.band and ch == ap.channel:
                d = max(1.0, math.hypot(ap.x - x, ap.y - y))
                rx = p - (BANDS[band][0] + spec.pl_coef * math.log10(d))
                out[j] += duty * 10 ** (rx / 10)
    return out


def truth_evaluator(scene, spec: EnvSpec, rng) -> BatchEvaluator:
    snaps = horizon_snapshots(scene)
    ev = BatchEvaluator(scene, snaps)
    ev.client_itf = True
    ev.leak = leak_matrix(scene, spec.leak_db)
    ev.ext_noise_mw = external_noise(scene, spec)
    for s in ev.snaps[1:]:            # forklifts appear on some future links
        hit = rng.random(s.ul.shape) < spec.blockage_p
        s.ul = s.ul - hit * spec.blockage_db
        s.dl = s.dl - hit * spec.blockage_db
    return ev


# ------------------------------------------------------------------ survey
@dataclass
class Survey:
    x: list = field(default_factory=list)
    y: list = field(default_factory=list)
    ap: list = field(default_factory=list)
    rssi: list = field(default_factory=list)
    # "not heard" bounds: BSSIDs a robot did not report are weaker than the
    # weakest one it did report (or below -90 dBm). Censored, not missing.
    cx: list = field(default_factory=list)
    cy: list = field(default_factory=list)
    cap: list = field(default_factory=list)
    cbound: list = field(default_factory=list)
    link: list = field(default_factory=list)      # (x, y, ap, sinr_floor_db)
    busy: dict = field(default_factory=dict)      # (band, ch) -> [fractions]
    censor_weight: float = 1.0   # inverse sampling fraction for thinned non-reports


def drive_and_survey(scene, spec: EnvSpec, minutes: float, rng) -> tuple:
    """Advance the fleet `minutes` along its routes; record what robots report."""
    sv = Survey(censor_weight=3.0)
    steps = int(minutes * 60 / 5)
    leak = None
    for k in range(steps):
        r = scene.radio()
        for i, car in enumerate(scene.cars):
            row = r.dl[i]
            # Rank the measured scan, not the hidden noiseless signal. Only
            # then is an omitted BSSID censored below the weakest report.
            measured = row + rng.normal(0, spec.fading_db, size=len(row))
            measured -= (rng.random(len(row)) < spec.blockage_p) * spec.blockage_db
            top = np.argsort(-measured)[:8]
            heard = []
            for j in top:
                meas = measured[j]
                if meas >= -90:
                    sv.x.append(car.x); sv.y.append(car.y); sv.ap.append(int(j))
                    sv.rssi.append(float(np.round(meas)))
                    heard.append((int(j), meas))
            bound = min(m for _, m in heard) if len(heard) == 8 else -90.0
            heard_ids = {j for j, _ in heard}
            if k % 3 == 0:                      # thin the bounds; they are many
                for j in range(len(scene.aps)):
                    if j not in heard_ids:
                        sv.cx.append(car.x); sv.cy.append(car.y); sv.cap.append(j)
                        sv.cbound.append(float(bound))
        if k % 6 == 0:
            ev = BatchEvaluator(scene, [])
            ev.client_itf = True
            ev.ext_noise_mw = external_noise(scene, spec)
            if leak is None:
                leak = leak_matrix(scene, spec.leak_db)
            ev.leak = leak
            a = ev.current
            t = ev._snapshot_terms(a[None], ev.snaps[0], False, 30.0)
            for i, car in enumerate(scene.cars):
                s = float(t["sinr"][0, i])
                band = scene.aps[a[i]].band
                floor = max([th for th, mb in __import__("wirelessjev.twin.radio", fromlist=["x"]).rate_table(band)
                             if th <= s] or [s - 3.0])
                sv.link.append((car.x, car.y, int(a[i]), float(floor)))
            for (band, ch), occ in spec.occupancy.items():
                sv.busy.setdefault((band, ch), []).append(
                    float(np.clip(occ + rng.normal(0, 0.05), 0, 1)))
        scene = scene.advance(5.0)
        scene = scene.with_associations(legacy_policy(scene))   # robots roam as today
    return scene, sv


# ------------------------------------------------------------------ belief
class RadioMap:
    """Per-BSS RSSI map: textbook prior + random-feature Bayesian residual."""

    def __init__(self, scene, ap_height: float, D: int = 200, length_m: float = 8.0,
                 prior_std_db: float = 8.0, noise_std_db: float = 4.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.aps = scene.aps
        self.h = ap_height
        self.Wf = rng.normal(0, 1 / length_m, size=(D, 2))
        self.bf = rng.uniform(0, 2 * math.pi, size=D)
        self.D = D
        self.sw2, self.sn2 = prior_std_db ** 2, noise_std_db ** 2
        self.post = {}
        # Site-calibrated log-distance law per band: loss = a + b*log10(d).
        # Starts as the textbook law; fit() replaces it with this site's.
        self.law = {band: (BANDS[band][0], 30.0) for band in BANDS}

    def _dist(self, j, xs, ys):
        ap = self.aps[j]
        return np.maximum(1.0, np.sqrt((ap.x - xs) ** 2 + (ap.y - ys) ** 2 + self.h ** 2))

    def prior_dl(self, j, xs, ys):
        ap = self.aps[j]
        a, b = self.law[ap.band]
        return ap.tx_power_dbm - (a + b * np.log10(self._dist(j, xs, ys)))

    def calibrate(self, sv: "Survey"):
        """Least-squares fit of the site's path-loss law from every scan."""
        x, y, ap, v = map(np.asarray, (sv.x, sv.y, sv.ap, sv.rssi))
        for band in BANDS:
            sel = np.array([self.aps[j].band == band for j in ap])
            if sel.sum() < 20:
                continue
            ld = np.log10(np.array([self._dist(j, xx, yy) for j, xx, yy in
                                    zip(ap[sel], x[sel], y[sel])]))
            loss = np.array([self.aps[j].tx_power_dbm for j in ap[sel]]) - v[sel]
            X = np.stack([np.ones_like(ld), ld], 1)
            a, b = np.linalg.lstsq(X, loss, rcond=None)[0]
            self.law[band] = (float(a), float(np.clip(b, 15.0, 45.0)))

    def feats(self, xs, ys):
        Z = np.stack([xs, ys], 1) @ self.Wf.T + self.bf
        return np.hstack([np.sqrt(2 / self.D) * np.cos(Z), np.ones((len(xs), 1))])

    def fit(self, sv: Survey):
        self.cens = (np.asarray(sv.cx), np.asarray(sv.cy), np.asarray(sv.cap, int),
                     np.asarray(sv.cbound)) if sv.cx else None
        self.calibrate(sv)
        x, y, ap, v = map(np.asarray, (sv.x, sv.y, sv.ap, sv.rssi))
        for j in range(len(self.aps)):
            sel = ap == j
            if sel.sum() < 3:
                continue
            Phi = self.feats(x[sel], y[sel])
            r = v[sel] - self.prior_dl(j, x[sel], y[sel])
            A = Phi.T @ Phi / self.sn2 + np.eye(Phi.shape[1]) / self.sw2
            Ainv = np.linalg.inv(A)
            w = Ainv @ Phi.T @ r / self.sn2
            if self.cens is not None:
                # Censored Gaussian likelihood P(noisy RSSI <= bound). A
                # non-report is an inequality, not a sample at bound - 3 dB:
                # fabricated point samples bias the map as surveys grow.
                cx, cy, ca, cb = self.cens
                cs = ca == j
                if cs.any():
                    from scipy.linalg import solve_triangular
                    from scipy.optimize import minimize
                    from scipy.special import log_ndtr

                    Pc = self.feats(cx[cs], cy[cs])
                    prior_c = self.prior_dl(j, cx[cs], cy[cs])
                    sigma = math.sqrt(self.sn2)
                    # Whiten by the detected-data posterior for a well-scaled
                    # convex optimisation, including poorly covered features.
                    L = np.linalg.cholesky(A)
                    Q = solve_triangular(L, Pc.T, lower=True).T / sigma
                    z0 = (cb[cs] - prior_c - Pc @ w) / sigma

                    def objective(delta):
                        z = z0 - Q @ delta
                        logcdf = log_ndtr(z)
                        mills = np.exp(-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - logcdf)
                        return (0.5 * delta @ delta - sv.censor_weight * logcdf.sum(),
                                delta + sv.censor_weight * Q.T @ mills)

                    result = minimize(objective, np.zeros(len(w)), jac=True,
                                      method="L-BFGS-B",
                                      options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-6})
                    if not result.success:
                        raise RuntimeError(f"censored radio-map fit failed: {result.message}")
                    w += solve_triangular(L.T, result.x, lower=False)
                    z = z0 - Q @ result.x
                    mills = np.exp(-0.5 * z ** 2 - 0.5 * np.log(2 * np.pi) - log_ndtr(z))
                    curvature = sv.censor_weight * np.maximum(mills * (mills + z), 0.0)
                    precision = A + (Pc.T * curvature) @ Pc / sigma ** 2
                    Ainv = np.linalg.inv(precision)
            self.post[j] = (w, Ainv)

    def predict(self, xs, ys):
        n, m = len(xs), len(self.aps)
        mean = np.empty((n, m)); std = np.empty((n, m))
        Phi = self.feats(xs, ys)
        for j in range(m):
            p = self.prior_dl(j, xs, ys)
            if j in self.post:
                w, Ainv = self.post[j]
                mean[:, j] = p + Phi @ w
                std[:, j] = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", Phi, Ainv, Phi), 0))
            else:
                mean[:, j], std[:, j] = p, math.sqrt(self.sw2)
        return mean, std


def belief_evaluator(truth_ev: BatchEvaluator, scene, spec: EnvSpec, sv: Survey | None,
                     mode: str, kappa: float = 0.0) -> BatchEvaluator:
    """Build the planner's view. mode: 'naive' (textbook model, no data),
    'twin' (our hand-built warehouse twin), 'learned' (radio map from survey)."""
    ev = BatchEvaluator.__new__(BatchEvaluator)
    ev.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v)
                        for k, v in truth_ev.__dict__.items() if k != "snaps"})
    snaps = [scene] + horizon_snapshots(scene)
    from .fastplan import Snapshot
    car_tx = truth_ev.snaps[0].ul[0, 0] - truth_ev.snaps[0].dl[0, 0] + scene.aps[0].tx_power_dbm
    car_tx = scene.cars[0].tx_power_dbm
    ap_tx = np.array([a.tx_power_dbm for a in scene.aps])
    ev.snaps = []
    if mode == "learned":
        rm = RadioMap(scene, spec.ap_height, seed=spec.seed)
        rm.fit(sv)
    for s in snaps:
        xs = np.array([c.x for c in s.cars]); ys = np.array([c.y for c in s.cars])
        if mode == "naive":
            dl = np.stack([RadioMap(scene, spec.ap_height).prior_dl(j, xs, ys)
                           for j in range(len(scene.aps))], 1)
        elif mode == "twin":
            nominal = replace(s, obstruction=W.rack_obstruction, pl_coef=33.0, shadow_db=0.0,
                              ap_height_m=0.0, _radio=None, _radio_key=None)
            dl = nominal.radio().dl
        else:
            mean, std = rm.predict(xs, ys)
            dl = mean - kappa * std
        ul = dl - ap_tx[None, :] + car_tx
        ev.snaps.append(Snapshot(ul, dl))
    ev.leak = truth_ev.leak.copy()          # channel adjacency is protocol knowledge
    ev.client_itf = True
    # Channel busy time is measured the same way for every arm (cheap scan data).
    first = [int(np.flatnonzero(ev.dom_of_ap == d)[0]) for d in range(ev.nd)]
    if sv is not None and sv.busy:
        occ = {k: float(np.mean(v)) for k, v in sv.busy.items()}
        ev.avail = np.array([1.0 - occ.get((scene.aps[j].band, scene.aps[j].channel), 0.0)
                             for j in first])
    else:
        ev.avail = np.ones(ev.nd)
    if mode == "learned":
        # external interference per AP from link reports: I+N implied by the
        # observed MCS floor against the learned uplink signal
        est = np.zeros(len(scene.aps))
        lx = np.array([l[0] for l in sv.link]); ly = np.array([l[1] for l in sv.link])
        la = np.array([l[2] for l in sv.link]); lf = np.array([l[3] for l in sv.link])
        mean, _ = rm.predict(lx, ly)
        ul_hat = mean[np.arange(len(la)), la] - ap_tx[la] + car_tx
        implied = 10 ** ((ul_hat - lf) / 10) - ev.noise_mw[la]
        for j in range(len(scene.aps)):
            v = implied[la == j]
            est[j] = float(np.clip(np.median(v), 0, None)) * 0.5 if len(v) else 0.0
        ev.ext_noise_mw = est
    else:
        ev.ext_noise_mw = np.zeros(len(scene.aps))
    return ev


# ------------------------------------------------------------------ experiment
def map_error(truth_ev, belief_ev, *, subset="audible"):
    t, b = truth_ev.snaps[0].dl, belief_ev.snaps[0].dl
    mask = {"audible": t >= -90, "unheard": t < -90,
            "all": np.ones(t.shape, dtype=bool)}[subset]
    if not mask.any():
        return None
    return float(np.sqrt(np.mean((t[mask] - b[mask]) ** 2)))


def run_env(seed: int, budgets=(1, 5, 15), log=print) -> dict:
    spec = random_env(seed)
    rng = np.random.default_rng(seed)
    scene0 = truth_scene(30_000 + seed, spec)
    scene, sv_all = drive_and_survey(scene0, spec, max(budgets), rng)
    truth = truth_evaluator(scene, spec, np.random.default_rng(seed + 1))
    per_step = (len(sv_all.rssi) // (max(budgets) * 12))
    starts_fn = lambda ev: [ev.current, ev.from_assoc(legacy_policy(scene))]
    busy_only = Survey(busy=sv_all.busy)
    methods = {"naive": belief_evaluator(truth, scene, spec, busy_only, "naive"),
               "twin": belief_evaluator(truth, scene, spec, busy_only, "twin")}
    for b in budgets:
        n = b * 12 * per_step
        frac = b / max(budgets)
        nc = int(len(sv_all.cx) * frac)
        sub = Survey(sv_all.x[-n:], sv_all.y[-n:], sv_all.ap[-n:], sv_all.rssi[-n:],
                     sv_all.cx[-nc:] if nc else [], sv_all.cy[-nc:] if nc else [],
                     sv_all.cap[-nc:] if nc else [], sv_all.cbound[-nc:] if nc else [],
                     sv_all.link[-int(len(sv_all.link) * frac):],
                     {k: v[-max(1, int(len(v) * frac)):] for k, v in sv_all.busy.items()},
                     censor_weight=sv_all.censor_weight)
        methods[f"learned_{b}min"] = belief_evaluator(truth, scene, spec, sub, "learned")
        if b == 5:
            methods["robust_5min"] = belief_evaluator(truth, scene, spec, sub, "learned", kappa=1.0)
    out = {"seed": seed, "spec": {"pl_exponent": spec.pl_coef / 10, "shadow_db": spec.shadow_db,
                                  "ap_height": spec.ap_height, "walls": len(spec.walls) // 2,
                                  "interferers": len(spec.interferers), "blockage_p": spec.blockage_p},
           "map_rmse_db": {k: map_error(truth, ev) for k, ev in methods.items()},
           "map_rmse_all_db": {k: map_error(truth, ev, subset="all") for k, ev in methods.items()},
           "map_rmse_unheard_db": {k: map_error(truth, ev, subset="unheard") for k, ev in methods.items()},
           "unheard_link_fraction": float(np.mean(truth.snaps[0].dl < -90)),
           "missions": {}}
    for mid, m in WAREHOUSE_MISSIONS.items():
        focus = [truth.car_ids.index(c) for c in m.focus_cars if c in truth.car_ids]
        fp_t = FastPlanner(scene, evaluator=truth)
        a_or, u_or, _ = fp_t.search(m.weights, m.focus_cars, starts=starts_fn(truth), max_iter=40)
        u_leg = float(truth.utility(truth.from_assoc(legacy_policy(scene)), m.weights, focus)[0])
        row = {"oracle": u_or, "legacy": u_leg}
        for name, bev in methods.items():
            fp = FastPlanner(scene, evaluator=bev)
            a, _, _ = fp.search(m.weights, m.focus_cars, starts=starts_fn(bev), max_iter=40)
            row[name] = float(truth.utility(a, m.weights, focus)[0])
        out["missions"][mid] = row
    log(f"env {seed}: rmse { {k: round(v, 1) for k, v in out['map_rmse_db'].items()} } "
        f"gap { {k: round(np.mean([r['oracle'] - r[k] for r in out['missions'].values()]), 3) for k in ['legacy', *methods]} }")
    return out


def summarise(envs: list[dict]) -> dict:
    names = [k for k in envs[0]["missions"]["map_rollout"] if k != "oracle"]
    gaps = {k: [] for k in names}
    closed = {k: [] for k in names}
    for e in envs:
        for r in e["missions"].values():
            span = r["oracle"] - r["legacy"]
            for k in names:
                gaps[k].append(r["oracle"] - r[k])
                if span > 1e-6:
                    closed[k].append(1 - (r["oracle"] - r[k]) / span)
    return {
        "n_envs": len(envs),
        "regret_vs_truth_oracle": {k: float(np.mean(v)) for k, v in gaps.items()},
        "regret_ci95": {k: float(1.96 * np.std(v) / np.sqrt(len(v))) for k, v in gaps.items()},
        "share_of_legacy_gap_closed": {k: float(np.mean(v)) for k, v in closed.items()},
        "map_rmse_db": {k: float(np.mean([e["map_rmse_db"][k] for e in envs]))
                        for k in envs[0]["map_rmse_db"]},
        "map_rmse_all_db": {k: float(np.mean([e["map_rmse_all_db"][k] for e in envs]))
                            for k in envs[0]["map_rmse_db"]},
        "map_rmse_unheard_db": {
            k: float(np.mean(values)) if values else None
            for k in envs[0]["map_rmse_db"]
            for values in [[e["map_rmse_unheard_db"][k] for e in envs
                            if e["map_rmse_unheard_db"][k] is not None]]},
        "unheard_link_fraction": float(np.mean([e["unheard_link_fraction"] for e in envs])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=15)
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output directory; use a new directory to preserve previous evidence")
    ap.add_argument("--workers", type=int, default=1,
                    help="independent environment processes (limit BLAS threads when using several)")
    args = ap.parse_args()
    if args.envs < 1 or args.workers < 1:
        ap.error("--envs and --workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    if args.workers == 1:
        envs = []
        for seed in range(args.envs):
            result = run_env(seed, log=log)
            (args.out / f"environment_{seed}.json").write_text(json.dumps(result, indent=2))
            envs.append(result)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, args.envs)) as pool:
            futures = [pool.submit(run_env, seed) for seed in range(args.envs)]
            for future in as_completed(futures):
                result = future.result()
                seed = result["seed"]
                (args.out / f"environment_{seed}.json").write_text(json.dumps(result, indent=2))
                results[seed] = result
                log(f"completed {len(results)}/{args.envs} environments")
        envs = [results[seed] for seed in range(args.envs)]
    report = {"method": "measured-top8-censored-gaussian-v2", "budgets_min": [1, 5, 15],
              "summary": summarise(envs), "environments": envs}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
