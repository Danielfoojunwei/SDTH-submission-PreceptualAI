"""Two-tier sharing of channel-quality priors.

From Zhang, Lam, Shen, Wang & Li (Ad Hoc Networks 2023): split knowledge into
a shared "basic" part and a local "personalised" part (20% personalised worked
best), aggregate at the edge every k1 rounds and globally every k2 rounds
(5 and 10 in their text).

Here the knowledge is not network weights but per-BSS quality statistics
(mean uplink SNR and observed airtime load) that every car measures. Cars in
the same cluster (edge tier) average every k1 rounds; all cars (global tier)
every k2 rounds. Each car's working estimate is 80% shared + 20% its own,
which keeps local effects (antenna placement, a car's route) without
discarding fleet knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChannelPriors:
    personal_fraction: float = 0.2
    k_edge: int = 5
    k_global: int = 10
    local: dict[str, dict[str, float]] = field(default_factory=dict)    # car -> bss -> value
    edge: dict[str, dict[str, float]] = field(default_factory=dict)     # cluster -> bss -> value
    global_: dict[str, float] = field(default_factory=dict)
    rounds: int = 0

    def observe(self, car_id: str, bss: str, value: float, alpha: float = 0.3) -> None:
        row = self.local.setdefault(car_id, {})
        row[bss] = value if bss not in row else (1 - alpha) * row[bss] + alpha * value

    @staticmethod
    def _average(rows: list[dict[str, float]]) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for r in rows:
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in acc.items()}

    def step(self, clusters: dict[str, list[str]]) -> None:
        self.rounds += 1
        if self.rounds % self.k_edge == 0:
            for cid, members in clusters.items():
                self.edge[cid] = self._average([self.local.get(m, {}) for m in members])
        if self.rounds % self.k_global == 0:
            self.global_ = self._average(list(self.edge.values()) or
                                         list(self.local.values()))

    def estimate(self, car_id: str, cluster: str, bss: str) -> float | None:
        shared = self.edge.get(cluster, {}).get(bss, self.global_.get(bss))
        own = self.local.get(car_id, {}).get(bss)
        if shared is None:
            return own
        if own is None:
            return shared
        return (1 - self.personal_fraction) * shared + self.personal_fraction * own


def bilevel_kmeans(points: dict[str, list[tuple[float, float]]], k: int,
                   seed: int = 0) -> dict[str, list[str]]:
    """Bi-level k-means from Shen, Goh, Lam & Li (ICCT 2024): recursively split
    the cluster with the largest spread into two until k clusters exist,
    using each car's recent location history (centroid)."""
    import numpy as np
    ids = list(points)
    cents = {i: np.mean(np.array(points[i]), axis=0) for i in ids}
    clusters = [ids]
    rng = np.random.default_rng(seed)
    while len(clusters) < k:
        spreads = [np.var([cents[i] for i in c], axis=0).sum() if len(c) > 1 else -1
                   for c in clusters]
        idx = int(np.argmax(spreads))
        target = clusters.pop(idx)
        if len(target) < 2:
            clusters.append(target)
            break
        X = np.array([cents[i] for i in target])
        c = X[rng.choice(len(X), 2, replace=False)]
        for _ in range(20):
            lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j]
                            for j in range(2)])
            if np.allclose(new, c):
                break
            c = new
        a = [t for t, l in zip(target, lab) if l == 0]
        b = [t for t, l in zip(target, lab) if l == 1]
        clusters += [a, b] if a and b else [target]
        if not (a and b):
            break
    return {f"c{i}": c for i, c in enumerate(clusters)}
