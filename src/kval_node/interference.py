"""Interference reports a spectrum manager can actually ingest.

Kval detects and corroborates interference across nodes, which is the hard
part, and then emitted its own JSON — a report nobody's tooling reads is a
report that reaches nobody. Joint Electromagnetic Spectrum Operations
(CJCSM 3320.01D, 24 January 2025) defines Joint Spectrum Interference
Resolution reporting, and the fields below carry the information a JSIR report
needs: who observed it, when, where, on what frequency, what the victim system
was, and what the effect was.

Two things this deliberately will **not** do, both because the evidence does
not support them:

**It does not classify intent.** JSIR distinguishes intentional from
unintentional interference. Kval measures energy and correlation; nothing in a
noise floor says whether a microwave oven, a faulty amplifier or an adversary
put it there. The report carries the observation and leaves the attribution to
an analyst who has other sources.

**It does not name a source.** Emitter geolocation gives a position with an
uncertainty radius, which is a *place*, not an actor. The position is carried
with its uncertainty attached and never as a point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from .validation import bounded_number, non_empty_str

#: JSIR characterises the operational effect. These are the categories a
#: measurement can actually support; anything finer would be inference.
EFFECT_NONE = "none"
EFFECT_DEGRADED = "degraded"
EFFECT_INTERMITTENT_OUTAGE = "intermittent_outage"
EFFECT_LOSS_OF_SERVICE = "loss_of_service"


@dataclass(frozen=True)
class InterferenceReport:
    """One corroborated interference observation, in a transferable shape."""

    report_id: str
    observed_at: datetime
    reporting_node: str
    corroborating_nodes: tuple[str, ...]
    band_id: str
    centre_frequency_hz: int
    bandwidth_hz: int
    victim_role: str
    effect: str
    noise_floor_excess_db: float | None
    estimated_position: dict[str, Any] | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-interference-report/v1",
            "reporting_format": "JSIR-compatible fields per CJCSM 3320.01D",
            "report_id": self.report_id,
            "observed_at": self.observed_at.isoformat(),
            "reporting_node": self.reporting_node,
            "corroborating_nodes": list(self.corroborating_nodes),
            "band_id": self.band_id,
            "centre_frequency_hz": self.centre_frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "victim_role": self.victim_role,
            "effect": self.effect,
            "noise_floor_excess_db": self.noise_floor_excess_db,
            "estimated_position": self.estimated_position,
            "intent": None,
            "reasons": list(self.reasons)
            + [
                "INTENT_IS_NOT_CLASSIFIED_BECAUSE_A_MEASUREMENT_CANNOT_ESTABLISH_IT",
                "NO_SOURCE_IS_NAMED_A_POSITION_WITH_UNCERTAINTY_IS_A_PLACE_NOT_AN_ACTOR",
            ],
        }


def effect_from_evidence(
    *, corroborating_nodes: int, victim_role: str, link_available: bool
) -> str:
    """Characterise operational effect from what was actually observed."""
    if not link_available:
        return EFFECT_LOSS_OF_SERVICE
    if victim_role == "c2_cnpc" and corroborating_nodes >= 2:
        # A command link degraded across multiple nodes is the case a spectrum
        # manager needs on their desk first.
        return EFFECT_INTERMITTENT_OUTAGE
    return EFFECT_DEGRADED if corroborating_nodes >= 1 else EFFECT_NONE


def build_report(
    *,
    report_id: str,
    observed_at: datetime,
    reporting_node: str,
    corroborating_nodes: Sequence[str],
    band_id: str,
    centre_frequency_hz: int,
    bandwidth_hz: int,
    victim_role: str,
    link_available: bool,
    noise_floor_excess_db: float | None = None,
    estimated_position: dict[str, Any] | None = None,
) -> InterferenceReport:
    corroborators = tuple(sorted(set(corroborating_nodes) - {reporting_node}))
    effect = effect_from_evidence(
        corroborating_nodes=len(corroborators),
        victim_role=victim_role,
        link_available=link_available,
    )
    reasons: list[str] = []
    if not corroborators:
        reasons.append("SINGLE_NODE_OBSERVATION_UNCORROBORATED_BY_ANY_PEER")
    else:
        reasons.append(f"CORROBORATED_BY_{len(corroborators)}_PEERS")
    if estimated_position is None:
        reasons.append("NO_POSITION_ESTIMATE_SURVIVED_THE_LOCATOR_S_CHECKS")
    return InterferenceReport(
        report_id=non_empty_str(report_id, "report_id"),
        observed_at=observed_at,
        reporting_node=non_empty_str(reporting_node, "reporting_node"),
        corroborating_nodes=corroborators,
        band_id=non_empty_str(band_id, "band_id"),
        centre_frequency_hz=int(centre_frequency_hz),
        bandwidth_hz=int(bandwidth_hz),
        victim_role=victim_role,
        effect=effect,
        noise_floor_excess_db=(
            bounded_number(noise_floor_excess_db, "noise_floor_excess_db", -50.0, 200.0)
            if noise_floor_excess_db is not None
            else None
        ),
        estimated_position=estimated_position,
        reasons=tuple(reasons),
    )
