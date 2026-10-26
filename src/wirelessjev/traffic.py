"""Trend-aware traffic labelling with a frozen decision model.

Adapts Chen, Lam & Li, "Large Language Models (LLMs) for Network Traffic
Prediction: A Trend-Aware Hybrid Framework" (IEEE IoT-J 2025). The paper
prompts a LoRA-fine-tuned LLaMA-2-7B with one line per time step, parses the
generated text into three label families, and feeds the label vector to an
XGBoost residual on top of a CNN-LSTM forecaster.

Here, with no training of our own:
- the three label families are typed choice questions answered by frozen
  d3-lite in one call (no generation, no parser):
    trend      {upward, downward, stable}
    event      {none, burst, anomaly, scheduled}
    volatility {low, medium, high}
- the numeric forecast comes from frozen Chronos-Bolt (Apache-2.0);
- a statistical rule labeller is the non-learned baseline;
- the paper's same-size ablation (semantic labels vs calendar vs PCA features)
  is reproduced with a closed-form ridge residual. That residual is the one
  fitted component in the project; it exists only to run the ablation and is
  never deployed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

CHRONOS_REPO = "amazon/chronos-bolt-small"
CHRONOS_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"

TREND = ("upward", "downward", "stable")
EVENT = ("none", "burst", "anomaly", "scheduled")
VOLATILITY = ("low", "medium", "high")
WINDOW = 48          # 48 x 5 s = 4 minutes of history
HORIZON = 6          # 30 s ahead
SCHEDULED_AT = 36    # the fleet's map sync starts at a fixed step


@dataclass
class TrafficWindow:
    car_id: str
    history: np.ndarray          # WINDOW samples, Mbps
    future: np.ndarray           # HORIZON samples, Mbps
    trend: str
    event: str
    volatility: str
    step_of_day: int


def synth_window(rng: np.random.Generator, car_id: str = "car-01") -> TrafficWindow:
    """Uplink throughput with known trend/event/volatility ground truth."""
    trend = str(rng.choice(TREND))
    event = str(rng.choice(EVENT, p=(0.4, 0.2, 0.2, 0.2)))
    vol = str(rng.choice(VOLATILITY))
    n = WINDOW + HORIZON
    t = np.arange(n, dtype=float)
    base = rng.uniform(8, 30)
    slope = {"upward": rng.uniform(0.15, 0.4), "downward": -rng.uniform(0.15, 0.4),
             "stable": 0.0}[trend]
    sigma = {"low": 0.03, "medium": 0.12, "high": 0.3}[vol] * base
    x = base + slope * t + rng.normal(0, sigma, n)
    if event == "burst":
        at = int(rng.integers(WINDOW - 14, WINDOW - 3))
        x[at:at + 4] += base * rng.uniform(1.0, 2.0)
    elif event == "anomaly":
        at = int(rng.integers(WINDOW - 14, WINDOW - 2))
        x[at:at + 2] = 0.0          # link outage / dropout
    elif event == "scheduled":
        x[SCHEDULED_AT:] += base * 0.6
    x = np.clip(x, 0.0, None)
    return TrafficWindow(car_id, x[:WINDOW], x[WINDOW:], trend, event, vol,
                         int(rng.integers(0, 17280)))


def prompt_lines(w: TrafficWindow, last: int = 24) -> str:
    """One line per time step, mirroring the paper's prompt layout."""
    rows = []
    start = WINDOW - last
    for i in range(start, WINDOW):
        sched = " sync_window=yes" if i >= SCHEDULED_AT else ""
        rows.append(f"T{i - start + 1}: uplink={w.history[i]:.1f} Mbps{sched}")
    return f"Car {w.car_id} uplink throughput, 5 s per step:\n" + "\n".join(rows)


def digest_lines(w: TrafficWindow) -> str:
    """Pre-digested statistics (numeric layer does arithmetic, the model
    reads consequences). Each line states a fact in words and numbers."""
    h = w.history
    level = max(float(np.median(h)), 1e-6)
    t = np.arange(len(h))
    change = np.polyfit(t, h, 1)[0] * len(h) / level
    recent = h[-16:]
    diffs = np.abs(np.diff(h)) / level
    before = float(np.median(h[SCHEDULED_AT - 12:SCHEDULED_AT]))
    after = float(np.median(h[SCHEDULED_AT:]))
    return "\n".join([
        f"Car {w.car_id} uplink over the last 4 minutes (median {level:.1f} Mbps).",
        f"Fitted change across the window: {100 * change:+.0f}% of the median.",
        f"Recent peak: {recent.max() / level:.1f}x the median; recent minimum: "
        f"{recent.min() / level:.2f}x the median.",
        f"Steps far above the median (>1.6x) in the recent minute: "
        f"{int((recent > 1.6 * level).sum())}.",
        f"Typical step-to-step change: {100 * float(np.median(diffs)):.0f}% of the median.",
        f"Since the scheduled sync window opened: median went from "
        f"{before:.1f} to {after:.1f} Mbps ({after / max(before, 1e-6):.2f}x).",
    ])


QUESTIONS = {
    "trend": {"type": "choice",
              "instructions": "What is the overall direction of this car's "
                              "uplink throughput across the window? Ignore single spikes.",
              "criteria": {"upward": "rising over the window",
                           "downward": "falling over the window",
                           "stable": "roughly flat over the window"}},
    "event": {"type": "choice",
              "instructions": "Which event best describes the recent "
                              "behaviour? A drop to zero is an anomaly; a short "
                              "surge is a burst; a step up when sync_window "
                              "starts is scheduled.",
              "criteria": {"none": "no notable event",
                           "burst": "short surge well above the usual level",
                           "anomaly": "throughput dropped to zero or near zero",
                           "scheduled": "sustained step up when the sync window starts"}},
    "volatility": {"type": "choice",
                   "instructions": "How variable is the throughput from step to step?",
                   "criteria": {"low": "smooth, small changes",
                                "medium": "noticeable step-to-step variation",
                                "high": "large, erratic swings"}},
}


def jev_labels(engine, w: TrafficWindow, digest: bool = True) -> tuple[dict, float]:
    """Three typed questions answered in one call by the frozen model.
    digest=False uses the paper's raw per-step prompt layout."""
    state = digest_lines(w) if digest else prompt_lines(w)
    answers, ms = engine.ask(state, QUESTIONS)
    out = {}
    for key, answer in answers.items():
        if "error" in answer:
            raise RuntimeError(f"MODEL_ERROR:{answer['error']}")
        out[key] = {"choice": answer["choice"], "probabilities": answer["probabilities"]}
    return out, ms


def rule_labels(w: TrafficWindow) -> dict:
    """Non-learned baseline labeller (fixed thresholds, no fitting)."""
    h = w.history
    t = np.arange(len(h))
    slope = np.polyfit(t, h, 1)[0]
    level = max(np.median(h), 1e-6)
    rel = slope * len(h) / level
    trend = "upward" if rel > 0.3 else "downward" if rel < -0.3 else "stable"
    recent = h[-16:]
    diffs = np.diff(h)
    mad = np.median(np.abs(diffs - np.median(diffs))) + 1e-6
    if recent.min() <= 0.05 * level:
        event = "anomaly"
    elif recent.max() > level + 6 * 1.4826 * mad and (recent > level * 1.6).sum() <= 6:
        event = "burst"
    elif np.median(h[SCHEDULED_AT:]) > 1.35 * np.median(h[SCHEDULED_AT - 12:SCHEDULED_AT]):
        event = "scheduled"
    else:
        event = "none"
    cov = 1.4826 * mad / level
    volatility = "low" if cov < 0.08 else "medium" if cov < 0.25 else "high"
    return {"trend": {"choice": trend}, "event": {"choice": event},
            "volatility": {"choice": volatility}}


class Forecaster:
    """Frozen Chronos-Bolt; zero-shot quantile forecasts."""

    def __init__(self, device: str = "cpu"):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        self.pipeline = BaseChronosPipeline.from_pretrained(
            CHRONOS_REPO, revision=CHRONOS_REVISION, device_map=device,
            torch_dtype=torch.float32)

    def forecast(self, histories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        ctx = [self.torch.tensor(h, dtype=self.torch.float32) for h in histories]
        q, mean = self.pipeline.predict_quantiles(
            ctx, prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9])
        return q.numpy(), mean.numpy()


def label_vector(labels: dict, soft: bool = True) -> np.ndarray:
    """8-dim semantic slot (as in the paper): trend(3)+volatility(3)+
    burst/anomaly flags(2)."""
    def probs(key, values):
        lab = labels[key]
        if soft and "probabilities" in lab:
            return [lab["probabilities"][v] for v in values]
        return [1.0 if lab["choice"] == v else 0.0 for v in values]
    ev = probs("event", EVENT)
    return np.array(probs("trend", TREND) + probs("volatility", VOLATILITY)
                    + [ev[1], ev[2]], dtype=float)


def calendar_vector(w: TrafficWindow) -> np.ndarray:
    """Same-size non-semantic control: time-of-day encodings."""
    phase = 2 * math.pi * w.step_of_day / 17280
    return np.array([math.sin(phase), math.cos(phase), math.sin(2 * phase),
                     math.cos(2 * phase), math.sin(3 * phase), math.cos(3 * phase),
                     (w.step_of_day % 720) / 720, float(w.step_of_day >= 8640)])


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ y)


def ridge_predict(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ W
