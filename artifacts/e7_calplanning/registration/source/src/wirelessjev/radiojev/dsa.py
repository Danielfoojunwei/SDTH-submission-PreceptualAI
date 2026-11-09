"""Dynamic spectrum access environment (Li / Shen / Lam setting).

Channels follow the papers' presets: some held by primary users (PUs) that
switch busy/idle as two-state Markov chains, some used by other secondary
users (SUs) at random, the rest idle. Each slot the agent (an SU robot)
picks one channel. Reward from Wang, Shen, Li & Lam (WCNC 2025):
+1 success on an idle channel, -0.5 collision with another SU, -1 collision
with a PU. The agent observes an H-slot sensing history per channel with
sensing errors (it never sees the Markov parameters).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRESETS = {"10/5/3/2": (10, 5, 3), "20/14/3/3": (20, 14, 3)}


@dataclass
class DSAEnv:
    preset: str = "10/5/3/2"
    history: int = 8
    sense_error: float = 0.1
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.C, n_pu, n_su = PRESETS[self.preset]
        kinds = np.array(["pu"] * n_pu + ["su"] * n_su + ["idle"] * (self.C - n_pu - n_su))
        self.kind = kinds[self.rng.permutation(self.C)]
        # PU chains: P(busy->busy), P(idle->busy); heterogeneous per channel
        self.p_bb = self.rng.uniform(0.6, 0.95, self.C)
        self.p_ib = self.rng.uniform(0.05, 0.4, self.C)
        self.p_su = self.rng.uniform(0.2, 0.6, self.C)
        self.busy = self.rng.random(self.C) < 0.5
        self.obs = np.zeros((self.history, self.C, 2), dtype=np.float32)
        for _ in range(self.history):
            self._step_world()

    def _step_world(self):
        u = self.rng.random(self.C)
        self.busy = np.where(self.busy, u < self.p_bb, u < self.p_ib) & (self.kind == "pu")
        self.su = (self.rng.random(self.C) < self.p_su) & (self.kind == "su")
        sensed_pu = self.busy ^ (self.rng.random(self.C) < self.sense_error)
        sensed_su = self.su ^ (self.rng.random(self.C) < self.sense_error)
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, 0] = sensed_pu
        self.obs[-1, :, 1] = sensed_su

    def state(self) -> np.ndarray:
        """(C, 2H) per-channel sensing history."""
        return self.obs.transpose(1, 0, 2).reshape(self.C, -1).copy()

    def step(self, channel: int) -> tuple[float, np.ndarray]:
        self._step_world()
        if self.busy[channel]:
            r = -1.0
        elif self.su[channel]:
            r = -0.5
        else:
            r = 1.0
        return r, self.state()

    def oracle_values(self) -> np.ndarray:
        """Expected reward of each channel next slot under the true model
        (for evaluation only)."""
        p_busy = np.where(self.busy, self.p_bb, self.p_ib) * (self.kind == "pu")
        p_su = self.p_su * (self.kind == "su")
        return -1.0 * p_busy + (1 - p_busy) * (-0.5 * p_su + 1.0 * (1 - p_su))
