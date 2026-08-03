"""Drive the real node code with fabricated measurements.

The point of this harness is that **only the measurements are fabricated**.
Everything that makes a decision — ``plan_allocation``, the survivability
constraint, role segregation, the flow table's diversity floor — is the
production code from ``kval_node``, imported and called unmodified. If the
allocator is wrong, this catches it; if this is wrong, the allocator is
unaffected, because nothing in ``src`` can reach here.

A run answers one question: *given measurements shaped like this, what does the
node decide, and does it refuse when it should?* It answers nothing about radio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from kval_node.allocation import Demand, Slice, plan_allocation

from .vehicles import CONDITIONS, VEHICLE_TYPES, Condition, VehicleType, degraded_link

EPOCH = datetime(2026, 8, 3, 0, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class VehicleOutcome:
    vehicle_id: str
    type_id: str
    condition: str
    slices_offered: int
    slices_usable: int
    flows_carried: tuple[str, ...]
    flows_refused: tuple[str, ...]
    refusal_reasons: tuple[str, ...]
    c2_carried: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "type_id": self.type_id,
            "condition": self.condition,
            "slices_offered": self.slices_offered,
            "slices_usable": self.slices_usable,
            "flows_carried": list(self.flows_carried),
            "flows_refused": list(self.flows_refused),
            "refusal_reasons": list(self.refusal_reasons),
            "c2_carried": self.c2_carried,
        }


def slices_for(vehicle_type: VehicleType, condition: Condition) -> list[Slice]:
    """Turn each bearer's degraded measurements into a slice the allocator sees.

    Capacity is measured throughput and health is derived the way
    ``LinkObservation.confidence`` does it — a link with no probe result has no
    confidence, so it contributes a slice with zero usable capacity rather than
    silently vanishing. The allocator must be able to tell "bearer seen and
    empty" from "bearer never looked at".
    """
    slices: list[Slice] = []
    for bearer in vehicle_type.bearers:
        reading = degraded_link(bearer, condition)
        if not reading["up"]:
            slices.append(
                Slice(bearer.bearer_id, bearer.bearer_id, bearer.link_role, 0.0, 0.0)
            )
            continue
        throughput_mbps = (reading["rx_mbps"] or 0.0) + (reading["tx_mbps"] or 0.0)
        # Health mirrors ``LinkObservation.confidence`` exactly, and that method
        # measures **evidence completeness**, not signal strength: 0.7 weighted
        # on the three mandatory probe fields, 0.3 on the three optional ones.
        #
        # Feeding signal strength in here instead is wrong and was wrong in the
        # first version of this harness. Capacity is already the degraded
        # throughput, so discounting it again by signal applies the same
        # condition twice and makes every outcome pessimistic. That is a harness
        # defect, not an allocator one, but it would have been read as evidence
        # about the allocator.
        mandatory = (reading["rtt_ms"], reading["loss_percent"], reading["jitter_ms"])
        optional = (reading["rx_mbps"], reading["tx_mbps"], reading["signal_percent"])
        health = 0.7 * (sum(v is not None for v in mandatory) / 3.0) + 0.3 * (
            sum(v is not None for v in optional) / 3.0
        )
        slices.append(
            Slice(
                slice_id=bearer.bearer_id,
                profile_id=bearer.bearer_id,
                link_role=bearer.link_role,
                capacity_kbps=throughput_mbps * 1000.0,
                health=round(health, 6),
            )
        )
    return slices


def demands_for(vehicle_type: VehicleType) -> list[Demand]:
    return [
        Demand(flow_id, role, kbps, minimum_slices=slices)
        for flow_id, role, kbps, slices in vehicle_type.demands
    ]


def run_vehicle(
    vehicle_id: str, vehicle_type: VehicleType, condition: Condition, *, now: datetime
) -> tuple[VehicleOutcome, dict[str, Any]]:
    slices = slices_for(vehicle_type, condition)
    plan = plan_allocation(demands_for(vehicle_type), slices, now=now)
    usable = sum(1 for item in slices if item.usable_kbps() > 0.0)
    refused = plan.refused_flow_ids()
    reasons: list[str] = []
    for flow in plan.flows:
        if not flow.carried:
            reasons.extend(flow.reasons)
    c2_flows = [flow for flow in plan.flows if flow.flow_id.startswith("c2")]
    outcome = VehicleOutcome(
        vehicle_id=vehicle_id,
        type_id=vehicle_type.type_id,
        condition=condition.name,
        slices_offered=len(slices),
        slices_usable=usable,
        flows_carried=plan.carried_flow_ids(),
        flows_refused=refused,
        refusal_reasons=tuple(sorted(set(reasons))),
        c2_carried=all(flow.carried for flow in c2_flows) if c2_flows else False,
    )
    return outcome, plan.to_dict()


def run_fleet(
    *, cycles: int = 1, now: datetime | None = None
) -> tuple[list[VehicleOutcome], list[dict[str, Any]]]:
    """Every vehicle type against every condition, for the given cycles.

    Deterministic by construction — no randomness anywhere in this harness — so
    two runs of the same code produce byte-identical outcomes and a change in
    the results means a change in the decision logic.
    """
    start = now or EPOCH
    outcomes: list[VehicleOutcome] = []
    plans: list[dict[str, Any]] = []
    for cycle in range(cycles):
        stamp = start + timedelta(seconds=30 * cycle)
        for vehicle_type in VEHICLE_TYPES:
            for condition in CONDITIONS:
                vehicle_id = f"{vehicle_type.type_id}-{condition.name}"
                outcome, plan = run_vehicle(vehicle_id, vehicle_type, condition, now=stamp)
                outcomes.append(outcome)
                plan["vehicle_id"] = vehicle_id
                plan["cycle"] = cycle
                plans.append(plan)
    return outcomes, plans


def summarise(outcomes: list[VehicleOutcome]) -> dict[str, Any]:
    by_condition: dict[str, dict[str, int]] = {}
    for outcome in outcomes:
        bucket = by_condition.setdefault(
            outcome.condition, {"vehicles": 0, "c2_carried": 0, "flows_refused": 0}
        )
        bucket["vehicles"] += 1
        bucket["c2_carried"] += 1 if outcome.c2_carried else 0
        bucket["flows_refused"] += len(outcome.flows_refused)
    return {
        "note": (
            "Logic validation only. Measurements are fabricated by sim/; no number "
            "here is evidence about RF performance or real vehicle behaviour."
        ),
        "vehicles": len(outcomes),
        "by_condition": dict(sorted(by_condition.items())),
    }
