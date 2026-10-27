"""Batched plan evaluation and search.

evaluate()/horizon_utility() score one plan at a time. Here a batch of B
plans (B x n BSS indices) is scored in one set of array operations with the
same model and the same mission utility, so a search can test every single
move of every robot at once. Works on numpy or, if given a torch device,
on the GPU (MPS/CUDA): this is data parallelism over candidate plans, which is
what speeds up planning; splitting the model across GPUs would not.

FastPlanner.search() is steepest-ascent hill climbing from several starting
plans, evaluating all (robot, BSS) moves per iteration in one batch.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from .oracle import _TABLES, HORIZON_OFFSETS_S, horizon_snapshots
from .twin.radio import DECISION_HORIZON_S, HANDOVER_OUTAGE_S, MAC_EFFICIENCY, Scene

WEIGHT_KEYS = ("B", "U", "V", "W", "fair", "ho", "focus", "viol", "prot")


@dataclass
class Snapshot:
    ul: np.ndarray        # (n, m) uplink RSSI
    dl: np.ndarray        # (n, m) downlink RSSI


class BatchEvaluator:
    """Precomputes everything that does not depend on the plan."""

    def __init__(self, scene: Scene, snaps: list[Scene] | None = None):
        r = scene.radio()
        self.scene = scene
        self.car_ids = r.car_ids
        self.ap_ids = r.ap_ids
        self.n, self.m = len(r.car_ids), len(r.ap_ids)
        channels = {ap.ap_id: ap.channel for ap in scene.aps}
        domains = scene.domains(channels)
        self.nd = len(domains)
        self.dom_of_ap = np.empty(self.m, dtype=np.int64)
        for d, members in enumerate(domains):
            for ap_id in members:
                self.dom_of_ap[r.ap_index[ap_id]] = d
        chan = np.array([channels[a] for a in self.ap_ids])
        same_bc = (r.band[:, None] == r.band[None, :]) & (chan[:, None] == chan[None, :])
        interferer = same_bc & (self.dom_of_ap[:, None] != self.dom_of_ap[None, :])
        self.apap_mw = np.where(interferer, 10 ** (r.apap / 10), 0.0)
        self.noise_mw = 10 ** (r.noise / 10)
        self.is5 = r.band == "5g"
        self.avail = np.array([1.0 - scene.occupancy(scene.ap(next(iter(dm))), channels)
                               for dm in domains])
        keys = {}
        self.bc_key = np.array([keys.setdefault((r.band[j], int(chan[j])), len(keys))
                                for j in range(self.m)])
        self.nkeys = len(keys)
        self.ext = np.array([scene.external_occupancy.get((r.band[j], int(chan[j])), 0.0) > 0
                             for j in range(self.m)])
        cars = scene.cars
        self.crit = np.array([sum(f.demand_mbps for f in c.flows if f.critical) for c in cars])
        self.be = np.array([sum(f.demand_mbps for f in c.flows if not f.critical) for c in cars])
        self.demand = self.crit + self.be
        K = max(1, max(sum(f.critical for f in c.flows) for c in cars))
        self.cf_dem = np.zeros((self.n, K))
        self.cf_floor = np.zeros((self.n, K))
        for i, c in enumerate(cars):
            for k, f in enumerate([f for f in c.flows if f.critical]):
                self.cf_dem[i, k], self.cf_floor[i, k] = f.demand_mbps, f.floor_mbps
        self.current = np.array([r.ap_index[c.current_ap] for c in cars])
        self.snaps = [Snapshot(r.ul, r.dl)]
        for s in snaps or []:
            rs = s.radio()
            self.snaps.append(Snapshot(rs.ul, rs.dl))
        self.visible = []
        for i, c in enumerate(cars):
            vis = [r.ap_index[ap.ap_id] for ap, _ in scene.visible_aps(c)]
            self.visible.append(np.array(vis, dtype=np.int64))
        self.t5, self.r5 = _TABLES["5g"]
        self.t2, self.r2 = _TABLES["2g"]
        # Optional richer interference model (off by default):
        #   ext_noise_mw  external (non-fleet) interference power at each AP
        #   client_itf    uplink interference from co-channel *robots* (what an
        #                 AP actually hears) instead of the AP-to-AP proxy
        #   leak          (m, m) linear leakage factor between adjacent channels
        self.ext_noise_mw = np.zeros(self.m)
        self.client_itf = False
        self.leak = np.zeros((self.m, self.m))
        self.interferer = interferer

    # ---------------------------------------------------------------------
    def _snapshot_terms(self, A: np.ndarray, snap: Snapshot, charge_handover: bool,
                        horizon_s: float, activity0: np.ndarray | None = None,
                        iters: int = 3) -> dict:
        B, n = A.shape
        rows = np.arange(n)[None, :]
        ul = snap.ul[rows, A]
        nz = self.noise_mw[A] + self.ext_noise_mw[A]
        is5 = self.is5[A]
        if self.client_itf:
            ul_lin = 10 ** (snap.ul / 10)                                   # (n, m)
            onehot = np.zeros((B, n, self.m))
            np.put_along_axis(onehot, A[..., None], 1.0, axis=2)
            per_ap = np.einsum("bnk,nj->bkj", onehot, ul_lin)              # sum over clients of AP k
            cnt = onehot.sum(1)[..., None]
            client_mw = per_ap / np.maximum(cnt, 1.0)                       # mean client power of k at j
            coupling = client_mw * (self.interferer.astype(float) + self.leak)[None]
        dcar = self.dom_of_ap[A]
        flat_d = (np.arange(B)[:, None] * self.nd + dcar).ravel()
        flat_a = (np.arange(B)[:, None] * self.m + A).ravel()
        activity = (np.full((B, self.m), 0.5) if activity0 is None
                    else np.broadcast_to(activity0, (B, self.m)))
        for _ in range(iters):
            if self.client_itf:
                interference = np.einsum("bk,bkj->bj", activity, coupling)
            else:
                interference = activity @ self.apap_mw
            itf = np.take_along_axis(interference, A, axis=1)
            sinr = ul - 10 * np.log10(nz + itf)
            i5 = np.searchsorted(self.t5, sinr, side="right") - 1
            i2 = np.searchsorted(self.t2, sinr, side="right") - 1
            r5 = np.where(i5 >= 0, self.r5[np.clip(i5, 0, None)], 0.0)
            r2 = np.where(i2 >= 0, self.r2[np.clip(i2, 0, None)], 0.0)
            rate = np.where(is5, r5, r2) * MAC_EFFICIENCY
            pos = rate > 0
            safe = np.where(pos, rate, 1.0)
            cn = np.where(pos, self.crit / safe, 0.0)
            bn = np.where(pos, self.be / safe, 0.0)
            crit_need = np.bincount(flat_d, cn.ravel(), B * self.nd).reshape(B, self.nd)
            be_need = np.bincount(flat_d, bn.ravel(), B * self.nd).reshape(B, self.nd)
            avail = self.avail[None, :]
            fits = crit_need <= avail
            rest = avail - crit_need
            be_frac = np.where(fits, np.where(be_need <= rest, 1.0,
                                              rest / np.where(be_need > 0, be_need, 1.0)), 0.0)
            crit_frac = np.where(fits, 1.0, np.where(crit_need > 0,
                                                     avail / np.where(crit_need > 0, crit_need, 1.0), 0.0))
            cf = np.take_along_axis(crit_frac, dcar, axis=1)
            bf = np.take_along_axis(be_frac, dcar, axis=1)
            per_car = np.where(pos, (cf * self.crit + bf * self.be) / safe, 0.0)
            activity = np.minimum(1.0, np.bincount(flat_a, per_car.ravel(),
                                                   B * self.m).reshape(B, self.m))
        if charge_handover:
            moved = A != self.current[None, :]
            keep = 1.0 - moved * (HANDOVER_OUTAGE_S / horizon_s)
            handovers = moved.sum(1)
        else:
            keep = np.ones_like(sinr)
            handovers = np.zeros(B)
        cf = np.where(pos, cf, 0.0)
        bf = np.where(pos, bf, 0.0)
        got_crit = self.cf_dem[None] * (cf * keep)[..., None]
        viol = ((got_crit < self.cf_floor[None] - 1e-9) & (self.cf_dem[None] > 0)).sum((1, 2))
        delivered = (cf * self.crit + bf * self.be) * keep
        sat = np.minimum(1.0, delivered / np.maximum(self.demand, 1e-9))
        U = delivered.sum(1) / max(self.demand.sum(), 1e-9)
        V = sat.mean(1)
        jain = np.where(sat.sum(1) > 0, sat.sum(1) ** 2 / (n * np.maximum((sat ** 2).sum(1), 1e-18)), 0.0)
        snr = ul - 10 * np.log10(nz)
        W = (np.maximum(0.0, snr - sinr) / 10.0).mean(1)
        key = self.bc_key[A]
        counts = np.bincount((np.arange(B)[:, None] * self.nkeys + key).ravel(),
                             minlength=B * self.nkeys).reshape(B, self.nkeys)
        shared = np.take_along_axis(counts, key, axis=1) > 1
        ext = self.ext[A]
        code_val = np.where(ext, -1.0, np.where(shared, -0.5, 1.0))
        return {"B": code_val.mean(1), "U": U, "V": V, "W": W, "jain": jain,
                "handovers": handovers, "viol": viol, "prot": ext.sum(1), "sat": sat,
                "activity_in": activity, "sinr": sinr, "pos": pos, "cf": cf, "bf": bf,
                "shared": shared, "ext": ext, "crit_need": crit_need, "be_need": be_need}

    def utility(self, A: np.ndarray, weights: dict, focus_idx: list[int] | None = None,
                horizon_s: float = DECISION_HORIZON_S, warm: list | None = None,
                iters: int = 3, return_activity: bool = False):
        """Mean mission utility over now + planned snapshots (handover charged
        once), identical to oracle.horizon_utility for each row of A when
        warm is None and iters == 3. warm/iters=1 is the fast screening mode:
        start the airtime fixed point from a known plan's activity."""
        A = np.atleast_2d(A)
        n = self.n
        total = np.zeros(A.shape[0])
        acts = []
        for k, snap in enumerate(self.snaps):
            t = self._snapshot_terms(A, snap, k == 0, horizon_s,
                                     None if warm is None else warm[k], iters)
            acts.append(t["activity_in"])
            focus = t["sat"][:, focus_idx].mean(1) if focus_idx else 0.0
            total += (weights["B"] * t["B"] + weights["U"] * t["U"] + weights["V"] * t["V"]
                      - weights["W"] * t["W"] + weights["fair"] * t["jain"]
                      - weights["ho"] * t["handovers"] / n + weights["focus"] * focus
                      - weights["viol"] * t["viol"] / n - weights["prot"] * t["prot"] / n)
        out = total / len(self.snaps)
        return (out, acts) if return_activity else out

    def exact(self, a: np.ndarray, weights: dict, focus_idx) -> tuple[float, list]:
        """Exact utility of one plan plus its converged per-snapshot activity
        (one more fixed-point step from the 3-step state, used as warm start)."""
        u, acts = self.utility(a, weights, focus_idx, return_activity=True)
        return float(u[0]), [x[0] for x in acts]

    def to_assoc(self, a: np.ndarray) -> dict[str, str]:
        return {cid: self.ap_ids[j] for cid, j in zip(self.car_ids, a)}

    def from_assoc(self, assoc: dict[str, str]) -> np.ndarray:
        index = {a: j for j, a in enumerate(self.ap_ids)}
        return np.array([index[assoc[c]] for c in self.car_ids])


class FastPlanner:
    """Steepest-ascent search over single-robot moves, all moves per
    iteration evaluated as one batch; several starting plans."""

    def __init__(self, scene: Scene, *, horizon: bool = True, pinned=frozenset(),
                 device: str | None = "cpu", evaluator: "BatchEvaluator | None" = None):
        """device: torch device for move screening ("cpu", "mps", "cuda"), or
        None to screen with numpy float64."""
        t0 = time.perf_counter()
        if evaluator is not None:
            self.snaps, self.ev = [], evaluator
        else:
            self.snaps = horizon_snapshots(scene) if horizon else []
            self.ev = BatchEvaluator(scene, self.snaps)
        # The torch screen mirrors the default interference model only.
        self.screen = (TorchScreen(self.ev, device)
                       if device and not self.ev.client_itf and not self.ev.ext_noise_mw.any()
                       else None)
        self.pinned = {self.ev.car_ids.index(c) for c in pinned if c in self.ev.car_ids}
        self.setup_ms = (time.perf_counter() - t0) * 1000
        moves = []
        for i, vis in enumerate(self.ev.visible):
            if i in self.pinned:
                continue
            for j in vis:
                moves.append((i, j))
        self.moves = np.array(moves, dtype=np.int64).reshape(-1, 2)

    def search(self, weights: dict, focus: tuple[str, ...] = (), starts=None,
               max_iter: int = 200, multi: bool = True,
               screen_iters: int = 1) -> tuple[np.ndarray, float, dict]:
        """Steepest ascent. Each iteration screens every (robot, BSS) move in
        one batch with a warm-started single fixed-point step, then verifies
        the chosen move set exactly; only exactly verified improvements are
        accepted, so the returned utility is exact."""
        ev = self.ev
        focus_idx = [ev.car_ids.index(c) for c in focus if c in ev.car_ids]
        t0 = time.perf_counter()
        if starts is None:
            starts = [ev.current]
        S = np.stack([np.asarray(x, dtype=np.int64) for x in starts])
        su = ev.utility(S, weights, focus_idx)
        a = S[int(np.argmax(su))].copy()
        u, warm = ev.exact(a, weights, focus_idx)
        iters = 0
        evals = len(S) + 1
        rows = np.arange(len(self.moves))
        for _ in range(max_iter):
            cand = np.repeat(a[None, :], len(self.moves), axis=0)
            cand[rows, self.moves[:, 0]] = self.moves[:, 1]
            if self.screen is not None:
                both = self.screen.utility(np.vstack([cand, a[None]]), weights, focus_idx, warm)
                both = both.cpu().numpy().astype(np.float64)
                gains = both[:-1] - both[-1]
            else:
                approx = ev.utility(cand, weights, focus_idx, warm=warm, iters=screen_iters)
                base = ev.utility(a, weights, focus_idx, warm=warm, iters=screen_iters)[0]
                gains = approx - base
            evals += len(cand) + 1
            iters += 1
            order = np.argsort(-gains)
            if gains[order[0]] <= 1e-9:
                break
            tried = []
            if multi:
                combo, used_cars, used_doms = a.copy(), set(), set()
                for k in order[:32]:
                    if gains[k] <= 1e-9:
                        break
                    i, j = self.moves[k]
                    d_new, d_old = ev.dom_of_ap[j], ev.dom_of_ap[a[i]]
                    if i in used_cars or d_new in used_doms or d_old in used_doms:
                        continue
                    combo[i] = j
                    used_cars.add(i)
                    used_doms.update((d_new, d_old))
                if len(used_cars) > 1:
                    tried.append(combo)
            for k in order[:4]:                       # exact check of top singles
                single = a.copy()
                single[self.moves[k, 0]] = self.moves[k, 1]
                tried.append(single)
            T = np.stack(tried)
            exact_u, acts = ev.utility(T, weights, focus_idx, return_activity=True)
            evals += len(T)
            best = int(np.argmax(exact_u))
            if exact_u[best] <= u + 1e-9:
                break
            a, u = T[best].copy(), float(exact_u[best])
            warm = [x[best] for x in acts]
        stats = {"iterations": iters, "plans_evaluated": evals,
                 "search_ms": (time.perf_counter() - t0) * 1000, "setup_ms": self.setup_ms}
        return a, u, stats


class TorchScreen:
    """GPU (MPS/CUDA) float32 copy of the screening path of BatchEvaluator.

    Only used to rank candidate moves; every accepted move is still verified
    with the float64 CPU evaluator, so decisions stay exact."""

    def __init__(self, ev: BatchEvaluator, device: str):
        import torch
        self.torch, self.ev, self.dev = torch, ev, torch.device(device)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev)
        i = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.int64, device=self.dev)
        self.apap, self.noise, self.is5 = f(ev.apap_mw), f(ev.noise_mw), torch.as_tensor(ev.is5, device=self.dev)
        self.dom, self.avail = i(ev.dom_of_ap), f(ev.avail)
        self.crit, self.be, self.demand = f(ev.crit), f(ev.be), f(ev.demand)
        self.cf_dem, self.cf_floor = f(ev.cf_dem), f(ev.cf_floor)
        self.current, self.key, self.ext = i(ev.current), i(ev.bc_key), torch.as_tensor(ev.ext, device=self.dev)
        self.t5, self.r5, self.t2, self.r2 = f(ev.t5), f(ev.r5), f(ev.t2), f(ev.r2)
        self.ul = [f(s.ul) for s in ev.snaps]

    def utility(self, A, weights: dict, focus_idx, warm, horizon_s=DECISION_HORIZON_S):
        torch = self.torch
        A = torch.as_tensor(A, device=self.dev)
        Bn, n = A.shape
        m, nd = self.apap.shape[0], self.avail.shape[0]
        rows = torch.arange(n, device=self.dev).expand(Bn, n)
        total = torch.zeros(Bn, device=self.dev)
        nz = self.noise[A]
        is5 = self.is5[A]
        dcar = self.dom[A]
        bidx = torch.arange(Bn, device=self.dev)[:, None]
        for k, ul_all in enumerate(self.ul):
            ul = ul_all[rows, A]
            act = torch.as_tensor(warm[k], dtype=torch.float32, device=self.dev).expand(Bn, m)
            itf = torch.gather(act @ self.apap, 1, A)
            sinr = ul - 10 * torch.log10(nz + itf)
            i5 = torch.searchsorted(self.t5, sinr.contiguous(), right=True) - 1
            i2 = torch.searchsorted(self.t2, sinr.contiguous(), right=True) - 1
            r5 = torch.where(i5 >= 0, self.r5[i5.clamp(min=0)], 0.0)
            r2 = torch.where(i2 >= 0, self.r2[i2.clamp(min=0)], 0.0)
            rate = torch.where(is5, r5, r2) * MAC_EFFICIENCY
            pos = rate > 0
            safe = torch.where(pos, rate, torch.ones_like(rate))
            cn = torch.where(pos, self.crit / safe, 0.0)
            bn = torch.where(pos, self.be / safe, 0.0)
            crit_need = torch.zeros(Bn, nd, device=self.dev).scatter_add_(1, dcar, cn)
            be_need = torch.zeros(Bn, nd, device=self.dev).scatter_add_(1, dcar, bn)
            fits = crit_need <= self.avail
            rest = self.avail - crit_need
            be_frac = torch.where(fits, torch.where(be_need <= rest, 1.0,
                                  rest / torch.where(be_need > 0, be_need, torch.ones_like(be_need))), 0.0)
            crit_frac = torch.where(fits, 1.0, torch.where(crit_need > 0,
                                    self.avail / torch.where(crit_need > 0, crit_need, torch.ones_like(crit_need)), 0.0))
            cf = torch.where(pos, torch.gather(crit_frac, 1, dcar), 0.0)
            bf = torch.where(pos, torch.gather(be_frac, 1, dcar), 0.0)
            if k == 0:
                moved = A != self.current
                keep = 1.0 - moved.float() * (HANDOVER_OUTAGE_S / horizon_s)
                ho = moved.sum(1).float()
            else:
                keep = torch.ones_like(cf)
                ho = torch.zeros(Bn, device=self.dev)
            got = self.cf_dem[None] * (cf * keep)[..., None]
            viol = ((got < self.cf_floor[None] - 1e-6) & (self.cf_dem[None] > 0)).sum((1, 2)).float()
            delivered = (cf * self.crit + bf * self.be) * keep
            sat = torch.clamp(delivered / self.demand.clamp(min=1e-9), max=1.0)
            U = delivered.sum(1) / self.demand.sum()
            V = sat.mean(1)
            jain = sat.sum(1) ** 2 / (n * (sat ** 2).sum(1).clamp(min=1e-18))
            W = (torch.clamp(10 * torch.log10(nz + itf) - 10 * torch.log10(nz), min=0) / 10).mean(1)
            key = self.key[A]
            counts = torch.zeros(Bn, int(self.key.max()) + 1, device=self.dev).scatter_add_(
                1, key, torch.ones_like(key, dtype=torch.float32))
            shared = torch.gather(counts, 1, key) > 1
            ext = self.ext[A]
            code = torch.where(ext, -1.0, torch.where(shared, -0.5, 1.0))
            focus = sat[:, focus_idx].mean(1) if focus_idx else 0.0
            total += (weights["B"] * code.mean(1) + weights["U"] * U + weights["V"] * V
                      - weights["W"] * W + weights["fair"] * jain - weights["ho"] * ho / n
                      + weights["focus"] * focus - weights["viol"] * viol / n
                      - weights["prot"] * ext.sum(1).float() / n)
        return total / len(self.ul)
