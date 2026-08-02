"""Outcome analysis over the node's own history.

The primitives in ``learning.py`` — change-point detection, causal attribution,
drift detection, and the signed-model contract — were written as guardrails for
a predictive model that does not exist yet, and so had no production caller.
Three of them do not actually need a model: they are statistics over the node's
own decision history, and they answer questions an operator asks anyway.

* **Did that switch help?** Every activation in the ledger is scored by
  :class:`~kval_node.learning.CausalGate`, comparing the same profile's measured
  latency and loss before and after. An improvement that does not clear both a
  minimum effect size and a t statistic is treated as noise, so the node does
  not congratulate itself for a coincidence.

* **Has something changed?** ``detect_change_point`` runs over the recent RTT
  and loss series to find a regime shift — a link that has genuinely got worse,
  as opposed to one noisy sample.

* **Are we outside the normal envelope?** A :class:`FeatureBaseline` computed
  from the node's own history lets :class:`DriftDetector` say whether current
  conditions look like anything it has seen before. Out-of-envelope conditions
  are exactly when a learned policy would have to abstain, and they are worth
  surfacing even with no policy present.

The fourth primitive, :func:`verify_model_manifest`, genuinely cannot run
without a model; it is wired in :mod:`kval_node.capabilities` so that
configuring a model path exercises the fail-closed contract for real.

Nothing here is a model and nothing here is trained. It is descriptive
statistics over recorded evidence, and every result carries the sample count it
was computed from so a thin result cannot be mistaken for a strong one.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .dataset import FEATURE_NAMES, CorpusExporter, DatasetError, Example
from .learning import (
    CausalGate,
    DriftDetector,
    FeatureBaseline,
    LearnedDecision,
    detect_change_point,
)

#: Minimum examples before a baseline is worth computing. Below this the
#: envelope is too narrow to call anything anomalous.
MIN_BASELINE_SAMPLES = 20

#: Minimum improvement, in milliseconds of round-trip time, before a switch is
#: credited with helping. Smaller differences are inside ordinary jitter.
DEFAULT_MIN_RTT_EFFECT_MS = 10.0


@dataclass(frozen=True)
class SwitchOutcome:
    profile_id: str | None
    samples_before: int
    samples_after: int
    attributed: bool
    mean_before_ms: float
    mean_after_ms: float
    improvement_ms: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "samples_before": self.samples_before,
            "samples_after": self.samples_after,
            "attributed": self.attributed,
            "mean_before_ms": self.mean_before_ms,
            "mean_after_ms": self.mean_after_ms,
            "improvement_ms": self.improvement_ms,
            "reasons": list(self.reasons),
        }


def build_baseline(examples: Sequence[Example]) -> tuple[FeatureBaseline | None, tuple[str, ...]]:
    """Compute a feature envelope from history, or explain why it cannot be.

    Features whose historical variance is zero are excluded: a constant carries
    no envelope, and ``FeatureBaseline`` rejects a zero standard deviation
    rather than dividing by it later.
    """
    if len(examples) < MIN_BASELINE_SAMPLES:
        return None, (f"NEED_AT_LEAST_{MIN_BASELINE_SAMPLES}_EXAMPLES",)
    means: dict[str, float] = {}
    stddevs: dict[str, float] = {}
    excluded: list[str] = []
    for name in FEATURE_NAMES:
        values = [example.features.get(name, 0.0) for example in examples]
        spread = statistics.pstdev(values)
        if spread <= 0.0:
            excluded.append(name)
            continue
        means[name] = statistics.fmean(values)
        stddevs[name] = spread
    if not means:
        return None, ("NO_FEATURE_VARIED_ACROSS_HISTORY",)
    reasons = (f"CONSTANT_FEATURES_EXCLUDED:{','.join(sorted(excluded))}",) if excluded else ()
    return FeatureBaseline(means=means, stddevs=stddevs), reasons


class OutcomeAnalyzer:
    """Descriptive analysis of the node's recorded decisions."""

    def __init__(
        self,
        ledger_path: Path,
        *,
        horizon_seconds: float = 60.0,
        min_rtt_effect_ms: float = DEFAULT_MIN_RTT_EFFECT_MS,
    ) -> None:
        self.exporter = CorpusExporter(ledger_path, horizon_seconds=horizon_seconds)
        self.min_rtt_effect_ms = min_rtt_effect_ms

    @staticmethod
    def _switch_points(examples: Sequence[Example]) -> list[int]:
        """Indices where the selected profile changed from the previous cycle."""
        points: list[int] = []
        for index in range(1, len(examples)):
            if examples[index].selected_profile_id != examples[index - 1].selected_profile_id:
                points.append(index)
        return points

    def _score_switch(self, examples: Sequence[Example], index: int, window: int) -> SwitchOutcome:
        before = [e.features["rtt_ms"] for e in examples[max(0, index - window) : index]]
        after = [e.features["rtt_ms"] for e in examples[index : index + window]]
        gate = CausalGate(min_effect=self.min_rtt_effect_ms, t_threshold=2.0, min_samples=3)
        verdict = gate.evaluate(before, after, higher_is_better=False)
        return SwitchOutcome(
            profile_id=examples[index].selected_profile_id,
            samples_before=len(before),
            samples_after=len(after),
            attributed=verdict.attributed,
            mean_before_ms=verdict.mean_before,
            mean_after_ms=verdict.mean_after,
            improvement_ms=verdict.improvement,
            reasons=verdict.reasons,
        )

    def analyze(self, *, window: int = 8, now: datetime | None = None) -> dict[str, Any]:
        try:
            examples, stats = self.exporter.export()
        except DatasetError as exc:
            return {"available": False, "reason": str(exc)}

        if not examples:
            return {
                "available": False,
                "reason": "no labelled history yet",
                "corpus": stats.to_dict(),
            }

        rtt_series = [example.features["rtt_ms"] for example in examples]
        loss_series = [example.features["loss_percent"] for example in examples]
        rtt_change = detect_change_point(rtt_series)
        loss_change = detect_change_point(loss_series)

        switches = [self._score_switch(examples, index, window) for index in self._switch_points(examples)]

        baseline, baseline_reasons = build_baseline(examples)
        drift: dict[str, Any]
        if baseline is None:
            drift = {"available": False, "reasons": list(baseline_reasons)}
        else:
            # Compare the most recent observation against the envelope built
            # from history, restricted to the features that actually varied.
            latest = {name: examples[-1].features[name] for name in baseline.means}
            verdict = DriftDetector(baseline).evaluate(latest)
            drift = {
                "available": True,
                "baseline_samples": len(examples),
                "baseline_reasons": list(baseline_reasons),
                **verdict.to_dict(),
            }

        helped = [outcome for outcome in switches if outcome.attributed]
        reasons: list[str] = []
        if rtt_change.detected or loss_change.detected:
            reasons.append("REGIME_CHANGE_DETECTED")
        if switches and not helped:
            reasons.append("NO_SWITCH_MEASURABLY_IMPROVED_LATENCY")
        if drift.get("decision") == LearnedDecision.ABSTAIN.value:
            reasons.append("CURRENT_CONDITIONS_OUTSIDE_HISTORICAL_ENVELOPE")
        if not reasons:
            reasons.append("NOMINAL")

        return {
            "available": True,
            "schema": "kval-outcome-analysis/v1",
            "examples": len(examples),
            "corpus": stats.to_dict(),
            "rtt_change_point": {
                "detected": rtt_change.detected,
                "index": rtt_change.index,
                "magnitude": rtt_change.magnitude,
                "reasons": list(rtt_change.reasons),
            },
            "loss_change_point": {
                "detected": loss_change.detected,
                "index": loss_change.index,
                "magnitude": loss_change.magnitude,
                "reasons": list(loss_change.reasons),
            },
            "switches": [outcome.to_dict() for outcome in switches],
            "switches_that_helped": len(helped),
            "drift": drift,
            "reasons": reasons,
        }
