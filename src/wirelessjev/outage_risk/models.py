"""Baselines and scoring for blockage detection. NumPy, with one small torch model."""
from __future__ import annotations

import numpy as np


class Logistic:
    """L2-regularised logistic regression fitted by Newton steps."""

    def __init__(self, l2: float = 1.0, steps: int = 25):
        self.l2, self.steps = l2, steps

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Logistic":
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        z = np.column_stack([(x - self.mu) / self.sd, np.ones(len(x))])
        w = np.zeros(z.shape[1])
        reg = self.l2 * np.eye(z.shape[1])
        reg[-1, -1] = 0.0
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))
            g = z.T @ (p - y) + reg @ w
            h = (z * (p * (1 - p))[:, None]).T @ z + reg + 1e-8 * np.eye(z.shape[1])
            step = np.linalg.solve(h, g)
            w = w - step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = np.column_stack([(x - self.mu) / self.sd, np.ones(len(x))])
        return 1 / (1 + np.exp(-np.clip(z @ self.w, -30, 30)))


class SmallNet:
    """Two hidden layers on the same windowed features: the matched supervised
    model any self-supervised (JEPA) detector has to beat."""

    def __init__(self, hidden: int = 32, epochs: int = 12, seed: int = 0, lr: float = 2e-3):
        self.hidden, self.epochs, self.seed, self.lr = hidden, epochs, seed, lr

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SmallNet":
        import torch
        torch.manual_seed(self.seed)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        xt = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], self.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden, self.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(xt), generator=gen).split(512):
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            z = self.net(torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)).squeeze(1)
            return torch.sigmoid(z).numpy().astype(float)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Probability that a blocked example scores above a clear one (ties count half)."""
    y = y.astype(bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ps = p[order]
    i = 0
    while i < len(ps):                               # average ranks over ties
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y.astype(float)) ** 2))


def at_false_alarm_rate(p: np.ndarray, y: np.ndarray, far: float) -> dict:
    """Share of blocked examples caught when the threshold is set so that the
    stated share of clear examples raises an alarm."""
    y = y.astype(bool)
    clear = np.sort(p[~y])
    if len(clear) == 0 or not y.any():
        return {"threshold": None, "caught": float("nan"), "false_alarm_rate": float("nan")}
    thr = clear[min(len(clear) - 1, int(np.ceil((1 - far) * len(clear))))] if far > 0 else clear[-1]
    alarm = p > thr
    return {"threshold": float(thr), "caught": float(alarm[y].mean()), "false_alarm_rate": float(alarm[~y].mean())}


def yes_no(alarm: np.ndarray, y: np.ndarray) -> dict:
    y = y.astype(bool)
    return {"caught": float(alarm[y].mean()) if y.any() else float("nan"),
            "false_alarm_rate": float(alarm[~y].mean()) if (~y).any() else float("nan"),
            "precision": float(y[alarm].mean()) if alarm.any() else float("nan")}
