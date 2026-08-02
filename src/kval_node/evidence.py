"""Dempster-Shafer fusion of cross-node band evidence.

The fabric's first aggregator counted votes: N nodes reporting interference
above a quorum meant confirmed. Counting has a flaw this module fixes — it
cannot distinguish **"this node measured the band and found it clear"** from
**"this node could not measure the band at all"**. Both are simply "not a vote
for interference", so a fleet of blind nodes looks like a fleet of nodes
reporting quiet.

Evidence theory represents that difference natively. Each node's report becomes
a basic probability assignment over the frame {CLEAR, BUSY}:

* a confident measurement puts mass on ``clear`` or ``busy``;
* an unmeasurable band puts **all** its mass on Θ, the "either" set, which is
  ignorance rather than disbelief;
* a node's own confidence scales its committed mass, with the remainder going
  to Θ, so a low-quality observation contributes proportionally little.

Combination is Dempster's rule. Two further properties matter for a defensive
fleet:

**Credibility weighting.** Each node's mass is discounted by a credibility
factor before combination, so an unreliable, poorly sited, or
lower-reputation node cannot carry the same weight as a well-behaved one.
This is the standard defence against spectrum-sensing data falsification
(SSDF), where a compromised node reports fabricated occupancy.

**Conflict is reported, not hidden.** Dempster's rule normalises away the
conflict mass K, which is exactly how the rule produces notoriously
counter-intuitive results when sources disagree strongly. K is retained and
surfaced here: high conflict between nodes observing the same band is itself
an intelligence signal — localised jamming affects some nodes and not others,
and so does a lying node.

Method after Liu, Jia, Na, Lu and Li, "Multi-Modal Cooperative Spectrum Sensing
Based on Dempster-Shafer Fusion in 5G-Based Cognitive Radio", IEEE Access 6
(2018), 199-208, doi:10.1109/ACCESS.2017.2761910 — which combines
per-node multi-detector decisions and then fuses across nodes with
credibility-weighted DS, explicitly to down-weight malicious secondary users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .spectrum import BandStatus
from .validation import bounded_number

#: Statuses that constitute evidence the band is busy or contested.
_BUSY_STATUSES = frozenset(
    {BandStatus.CONGESTED, BandStatus.ANOMALOUS, BandStatus.JAMMING_SUSPECTED}
)
#: Statuses that constitute evidence the band is usable.
_CLEAR_STATUSES = frozenset({BandStatus.CLEAR, BandStatus.OCCUPIED})

#: Conflict mass above which the sources are treated as fundamentally
#: disagreeing, and the fused belief is reported as untrustworthy.
DEFAULT_MAX_CONFLICT = 0.85


@dataclass(frozen=True)
class BeliefMass:
    """Mass over {clear}, {busy}, and Θ={clear,busy}. Sums to 1."""

    clear: float
    busy: float
    unknown: float

    def __post_init__(self) -> None:
        total = self.clear + self.busy + self.unknown
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"belief mass must sum to 1, got {total}")

    @classmethod
    def from_status(cls, status: BandStatus, confidence: float, *, credibility: float = 1.0) -> "BeliefMass":
        """Convert one node's band assessment into a mass assignment.

        ``INSUFFICIENT_EVIDENCE`` yields total ignorance — all mass on Θ —
        which is the representation a vote count cannot express.
        """
        weight = bounded_number(confidence, "confidence", 0.0, 1.0) * bounded_number(
            credibility, "credibility", 0.0, 1.0
        )
        if status is BandStatus.INSUFFICIENT_EVIDENCE or weight <= 0.0:
            return cls(0.0, 0.0, 1.0)
        if status in _BUSY_STATUSES:
            # A merely congested band is weaker evidence of a problem than
            # suspected jamming, so it commits less mass.
            strength = weight if status is not BandStatus.CONGESTED else weight * 0.6
            return cls(0.0, strength, 1.0 - strength)
        if status in _CLEAR_STATUSES:
            strength = weight if status is BandStatus.CLEAR else weight * 0.6
            return cls(strength, 0.0, 1.0 - strength)
        return cls(0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "clear": round(self.clear, 6),
            "busy": round(self.busy, 6),
            "unknown": round(self.unknown, 6),
        }


@dataclass(frozen=True)
class FusedBelief:
    clear: float
    busy: float
    unknown: float
    conflict: float
    sources: int
    trustworthy: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "clear": round(self.clear, 6),
            "busy": round(self.busy, 6),
            "unknown": round(self.unknown, 6),
            "conflict": round(self.conflict, 6),
            "sources": self.sources,
            "trustworthy": self.trustworthy,
            "reasons": list(self.reasons),
        }


def combine_pair(first: BeliefMass, second: BeliefMass) -> tuple[BeliefMass, float]:
    """Dempster's rule for two sources, returning the mass and the conflict K."""
    conflict = first.clear * second.busy + first.busy * second.clear
    remainder = 1.0 - conflict
    if remainder <= 1e-12:
        # Total conflict: the sources are mutually exclusive and Dempster's
        # rule is undefined. Fall back to complete ignorance rather than a
        # division that would manufacture certainty from disagreement.
        return BeliefMass(0.0, 0.0, 1.0), 1.0
    clear = (
        first.clear * second.clear + first.clear * second.unknown + first.unknown * second.clear
    ) / remainder
    busy = (
        first.busy * second.busy + first.busy * second.unknown + first.unknown * second.busy
    ) / remainder
    unknown = (first.unknown * second.unknown) / remainder
    # Renormalise against accumulated floating-point drift.
    total = clear + busy + unknown
    return BeliefMass(clear / total, busy / total, unknown / total), conflict


def fuse(
    masses: Sequence[BeliefMass],
    *,
    max_conflict: float = DEFAULT_MAX_CONFLICT,
) -> FusedBelief:
    """Combine node evidence, retaining the conflict rather than hiding it."""
    contributing = [mass for mass in masses if mass.unknown < 1.0]
    reasons: list[str] = []
    if not contributing:
        return FusedBelief(
            clear=0.0,
            busy=0.0,
            unknown=1.0,
            conflict=0.0,
            sources=0,
            trustworthy=False,
            reasons=("NO_NODE_PRODUCED_COMMITTED_EVIDENCE",),
        )

    combined = contributing[0]
    peak_conflict = 0.0
    for mass in contributing[1:]:
        combined, conflict = combine_pair(combined, mass)
        peak_conflict = max(peak_conflict, conflict)

    trustworthy = peak_conflict <= max_conflict
    if not trustworthy:
        # Dempster's rule normalises conflict away, which is precisely how it
        # produces confident nonsense from irreconcilable sources.
        reasons.append("SOURCES_IRRECONCILABLY_CONFLICTED")
    if combined.unknown > 0.5:
        reasons.append("EVIDENCE_DOMINATED_BY_IGNORANCE")
    if not reasons:
        reasons.append("EVIDENCE_FUSED")
    return FusedBelief(
        clear=combined.clear,
        busy=combined.busy,
        unknown=combined.unknown,
        conflict=peak_conflict,
        sources=len(contributing),
        trustworthy=trustworthy,
        reasons=tuple(reasons),
    )
