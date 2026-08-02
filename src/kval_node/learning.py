"""Learning support for autonomous-network problems — fail closed.

Kval does not ship a trained predictive model. The problems that motivate one —
link-failure forecasting, energy/thermal forecasting, congestion and mobility
prediction, root-cause classification — are documented in
``docs/AUTONOMOUS_NETWORK_LEARNING.md`` with the standards that raise them. This
module provides the *guardrails* a predictive policy must pass through, all of
which work today without a model:

* a signed-model contract that refuses to load unless engine, feature schema,
  calibration report, and signature are all present and verified;
* an out-of-distribution / drift detector that forces ABSTAIN on novel inputs;
* a change-point detector for spotting regime shifts in a metric stream;
* a causal-attribution gate that only credits an action for an improvement when
  a pre/post comparison clears a minimum effect size with confidence.

Until a model is present and every guardrail passes, the learned path abstains
and the deterministic optimizer and Safety Shield remain in control.
"""

from __future__ import annotations

import base64
import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .signing import canonical_json, public_key_id
from .validation import require_finite, sha256_hex


class LearnedDecision(str, Enum):
    ABSTAIN = "ABSTAIN"
    ADVISE = "ADVISE"


@dataclass(frozen=True)
class ModelManifest:
    """Provenance for a candidate predictive model. Not a model itself."""

    model_id: str
    engine_sha256: str
    feature_schema_sha256: str
    calibration_report_sha256: str
    training_data_lineage: str
    created_at: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelManifest":
        return cls(
            model_id=str(raw["model_id"]).strip(),
            engine_sha256=sha256_hex(raw.get("engine_sha256"), "engine_sha256"),
            feature_schema_sha256=sha256_hex(raw.get("feature_schema_sha256"), "feature_schema_sha256"),
            calibration_report_sha256=sha256_hex(raw.get("calibration_report_sha256"), "calibration_report_sha256"),
            training_data_lineage=str(raw["training_data_lineage"]).strip(),
            created_at=str(raw["created_at"]).strip(),
        )

    def canonical(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "engine_sha256": self.engine_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "calibration_report_sha256": self.calibration_report_sha256,
            "training_data_lineage": self.training_data_lineage,
            "created_at": self.created_at,
        }


class ModelTrustError(RuntimeError):
    """Raised when a candidate model cannot be trusted; the node stays advisory."""


def verify_model_manifest(envelope: dict[str, Any], public_key: Ed25519PublicKey) -> ModelManifest:
    """Verify a signed model manifest. Missing evidence fails closed."""
    if envelope.get("algorithm") != "Ed25519":
        raise ModelTrustError("unsupported model manifest signature algorithm")
    if envelope.get("key_id") != public_key_id(public_key):
        raise ModelTrustError("model manifest key ID does not match the provisioned key")
    body = envelope.get("manifest")
    if not isinstance(body, dict):
        raise ModelTrustError("model manifest body must be an object")
    try:
        manifest = ModelManifest.from_dict(body)
    except (KeyError, ValueError, TypeError) as exc:
        raise ModelTrustError(f"model manifest is incomplete: {exc}") from exc
    if not manifest.model_id or not manifest.training_data_lineage:
        raise ModelTrustError("model manifest must name a model and its training-data lineage")
    # The signature covers the canonical projection, so any field the presented
    # manifest carries beyond it would be unauthenticated. Reject rather than
    # silently dropping it.
    if body != manifest.canonical():
        raise ModelTrustError("model manifest contains unauthenticated or non-canonical fields")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise ModelTrustError("model manifest signature is missing or malformed") from exc
    try:
        public_key.verify(signature, canonical_json(manifest.canonical()))
    except InvalidSignature as exc:
        raise ModelTrustError("model manifest signature verification failed") from exc
    return manifest


@dataclass(frozen=True)
class FeatureBaseline:
    """Signed per-feature training distribution used for OOD detection."""

    means: dict[str, float]
    stddevs: dict[str, float]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureBaseline":
        means = {str(k): require_finite(v, f"means.{k}") for k, v in raw.get("means", {}).items()}
        stddevs: dict[str, float] = {}
        for key, value in raw.get("stddevs", {}).items():
            number = require_finite(value, f"stddevs.{key}")
            if number <= 0:
                raise ValueError(f"stddev for {key} must be positive")
            stddevs[str(key)] = number
        if set(means) != set(stddevs):
            raise ValueError("baseline means and stddevs must cover the same features")
        if not means:
            raise ValueError("baseline must define at least one feature")
        return cls(means=means, stddevs=stddevs)


@dataclass(frozen=True)
class DriftVerdict:
    decision: LearnedDecision
    max_z: float
    out_of_range_features: tuple[str, ...]
    missing_features: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "max_z": self.max_z,
            "out_of_range_features": list(self.out_of_range_features),
            "missing_features": list(self.missing_features),
            "reasons": list(self.reasons),
        }


class DriftDetector:
    """Forces ABSTAIN when live features fall outside the trained distribution."""

    def __init__(self, baseline: FeatureBaseline, *, z_threshold: float = 4.0) -> None:
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        self.baseline = baseline
        self.z_threshold = z_threshold

    def evaluate(self, features: dict[str, float]) -> DriftVerdict:
        out_of_range: list[str] = []
        non_finite: list[str] = []
        max_z = 0.0
        provided = {str(k): float(v) for k, v in features.items()}
        missing = tuple(sorted(set(self.baseline.means) - set(provided)))
        unknown = tuple(sorted(set(provided) - set(self.baseline.means)))
        for name, mean in self.baseline.means.items():
            if name not in provided:
                continue
            value = provided[name]
            # A NaN comparison is False, so an unguarded z-test would silently
            # ADVISE on a non-finite feature. Treat it as out of distribution.
            if not math.isfinite(value):
                non_finite.append(name)
                continue
            z = abs(value - mean) / self.baseline.stddevs[name]
            max_z = max(max_z, z)
            if z > self.z_threshold:
                out_of_range.append(name)
        reasons: list[str] = []
        decision = LearnedDecision.ADVISE
        if missing:
            decision = LearnedDecision.ABSTAIN
            reasons.append("REQUIRED_FEATURES_MISSING")
        if non_finite:
            decision = LearnedDecision.ABSTAIN
            reasons.append("NON_FINITE_FEATURE_VALUES")
            out_of_range.extend(non_finite)
        if out_of_range:
            decision = LearnedDecision.ABSTAIN
            reasons.append("FEATURES_OUT_OF_TRAINED_DISTRIBUTION")
        if unknown:
            # Extra features mean the live schema does not match the trained
            # one; that is a schema mismatch, not a harmless addition.
            decision = LearnedDecision.ABSTAIN
            reasons.append("FEATURES_OUTSIDE_BASELINE_SCHEMA")
        if not reasons:
            reasons.append("WITHIN_TRAINED_DISTRIBUTION")
        return DriftVerdict(
            decision=decision,
            max_z=round(max_z, 6),
            out_of_range_features=tuple(sorted(out_of_range)),
            missing_features=missing,
            reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class ChangePoint:
    detected: bool
    index: int | None
    magnitude: float
    reasons: tuple[str, ...]


def detect_change_point(series: Sequence[float], *, threshold_sigmas: float = 5.0, min_samples: int = 8) -> ChangePoint:
    """Change-point detector over a metric stream.

    Splits the series at every interior index and returns the split whose
    before/after means are most separated, measured in pooled within-segment
    standard deviations. ``index`` is that best split — the first sample of the
    new regime — and ``magnitude`` is the separation there.

    ``magnitude`` is *standardized*, so it measures how cleanly the two regimes
    separate rather than the raw size of the shift: a noiseless 1→5 step and a
    noiseless 1→500 step are equally detectable and score alike. This is the
    intended semantic, and it is what a raw cumulative-sum peak fails to
    provide — that peak returns toward zero and scales with series length
    rather than with the strength of the change.

    Too-short, flat, or non-finite series report no change rather than a
    spurious one.
    """
    values = [require_finite(value, "series value") for value in series]
    if len(values) < max(min_samples, 4):
        return ChangePoint(False, None, 0.0, ("INSUFFICIENT_SAMPLES",))
    stddev = statistics.pstdev(values)
    if stddev == 0.0:
        return ChangePoint(False, None, 0.0, ("NO_VARIANCE",))

    # Floor the denominator relative to the series scale. Without it a
    # perfectly clean step divides by zero within-segment spread and would have
    # to fall back to the whole-series spread, which scores the true split
    # *lower* than its noisy neighbours and reports the wrong index.
    floor = stddev * 1e-6
    best = 0.0
    best_index: int | None = None
    for index in range(1, len(values)):
        before = values[:index]
        after = values[index:]
        spread = math.sqrt(
            (statistics.pvariance(before) + statistics.pvariance(after)) / 2.0
        )
        separation = abs(statistics.fmean(after) - statistics.fmean(before)) / max(spread, floor)
        if separation > best:
            best = separation
            best_index = index

    if best >= threshold_sigmas:
        return ChangePoint(True, best_index, round(best, 6), ("CHANGE_POINT_THRESHOLD_EXCEEDED",))
    return ChangePoint(False, None, round(best, 6), ("NO_CHANGE_POINT",))


@dataclass(frozen=True)
class CausalVerdict:
    attributed: bool
    mean_before: float
    mean_after: float
    improvement: float
    welch_t: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributed": self.attributed,
            "mean_before": self.mean_before,
            "mean_after": self.mean_after,
            "improvement": self.improvement,
            "welch_t": self.welch_t,
            "reasons": list(self.reasons),
        }


class CausalGate:
    """Credits an action for an improvement only if it clears an effect + t gate.

    Uses a Welch t statistic (unequal variances) between the before and after
    samples. ``higher_is_better`` flips the sign so the same gate serves both
    throughput-style and latency-style metrics. Without both a minimum effect
    size and a t statistic beyond ``t_threshold``, the improvement is treated as
    noise and the action is not learned from.
    """

    def __init__(self, *, min_effect: float, t_threshold: float = 2.0, min_samples: int = 5) -> None:
        if min_effect < 0:
            raise ValueError("min_effect must be non-negative")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        self.min_effect = min_effect
        self.t_threshold = t_threshold
        self.min_samples = min_samples

    def evaluate(
        self,
        before: Sequence[float],
        after: Sequence[float],
        *,
        higher_is_better: bool,
    ) -> CausalVerdict:
        before_values = [require_finite(v, "before sample") for v in before]
        after_values = [require_finite(v, "after sample") for v in after]
        if len(before_values) < self.min_samples or len(after_values) < self.min_samples:
            return CausalVerdict(False, 0.0, 0.0, 0.0, 0.0, ("INSUFFICIENT_SAMPLES",))
        mean_before = statistics.fmean(before_values)
        mean_after = statistics.fmean(after_values)
        improvement = (mean_after - mean_before) if higher_is_better else (mean_before - mean_after)
        var_before = statistics.variance(before_values)
        var_after = statistics.variance(after_values)
        denominator = math.sqrt(var_before / len(before_values) + var_after / len(after_values))
        reasons: list[str] = []
        if denominator == 0.0:
            # Both samples are perfectly flat. That is a degenerate measurement,
            # not infinite confidence, and an infinity would not survive JSON
            # serialization into the audit ledger. Refuse to attribute.
            return CausalVerdict(
                False,
                round(mean_before, 6),
                round(mean_after, 6),
                round(improvement, 6),
                0.0,
                ("ZERO_VARIANCE_IN_BOTH_SAMPLES",),
            )
        welch_t = (mean_after - mean_before) / denominator
        if not higher_is_better:
            welch_t = -welch_t
        attributed = improvement >= self.min_effect and welch_t >= self.t_threshold
        if improvement < self.min_effect:
            reasons.append("IMPROVEMENT_BELOW_MIN_EFFECT")
        if welch_t < self.t_threshold:
            reasons.append("EFFECT_NOT_STATISTICALLY_SEPARATED")
        if attributed:
            reasons.append("IMPROVEMENT_ATTRIBUTED_TO_ACTION")
        return CausalVerdict(
            attributed=attributed,
            mean_before=round(mean_before, 6),
            mean_after=round(mean_after, 6),
            improvement=round(improvement, 6),
            welch_t=round(welch_t, 6),
            reasons=tuple(reasons),
        )
