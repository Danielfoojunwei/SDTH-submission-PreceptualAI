"""Count-based next-location predictor.

Follows the mobility feature construction of Li, Sun, Lam, Sun, Shen & Peng,
"Dynamic Spectrum Optimization for Internet-of-Vehicles with Deep-Learning-
Based Mobility Prediction" (Wireless Personal Communications 137, 2024):
the transition probability to a cell is proportional to its visit count
divided by squared distance, normalised; the top-2 candidates, previous
location and dwell time form the feature set. We stop at those counting
statistics and do NOT fit the paper's LSTM (no training in this project).
"""
from __future__ import annotations

from collections import Counter
import math


class MobilityPredictor:
    def __init__(self, cell_m: float = 6.0):
        self.cell_m = cell_m
        self.visits: dict[str, Counter] = {}
        self.last: dict[str, tuple[int, int]] = {}
        self.dwell: dict[str, int] = {}

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(x // self.cell_m), int(y // self.cell_m)

    def observe(self, car_id: str, x: float, y: float) -> None:
        c = self.cell(x, y)
        self.visits.setdefault(car_id, Counter())[c] += 1
        if self.last.get(car_id) == c:
            self.dwell[car_id] = self.dwell.get(car_id, 0) + 1
        else:
            self.dwell[car_id] = 1
        self.last[car_id] = c

    def predict(self, car_id: str, k: int = 2) -> list[tuple[tuple[int, int], float]]:
        """Top-k next cells with probabilities Pr_j ∝ r_j / d_j^2.

        The current cell is excluded (d = 0); neighbouring cells that were
        never visited get a pseudo-count of 1 so a new car still moves.
        """
        here = self.last.get(car_id)
        if here is None:
            return []
        counts = Counter(self.visits.get(car_id, Counter()))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    counts.setdefault((here[0] + dx, here[1] + dy), 1)
        scores = {}
        for cell, r in counts.items():
            if cell == here:
                continue
            d2 = (cell[0] - here[0]) ** 2 + (cell[1] - here[1]) ** 2
            scores[cell] = r / d2
        total = sum(scores.values())
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(cell, s / total) for cell, s in ranked]

    def predicted_xy(self, car_id: str) -> tuple[float, float] | None:
        top = self.predict(car_id, 1)
        if not top:
            return None
        (cx, cy), _ = top[0]
        return (cx + 0.5) * self.cell_m, (cy + 0.5) * self.cell_m

    def features(self, car_id: str) -> dict:
        top = self.predict(car_id, 2)
        return {
            "previous_cell": self.last.get(car_id),
            "dwell_steps": self.dwell.get(car_id, 0),
            "top2": [(list(c), round(p, 3)) for c, p in top],
        }


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
