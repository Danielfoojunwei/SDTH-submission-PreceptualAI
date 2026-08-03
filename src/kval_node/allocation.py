"""Carry a flow on several slices at once, so losing one does not lose the flow.

Selecting a link is the wrong decision to be making. A node that picks the best
bearer and commits the whole flow to it has built a single point of failure out
of a measurement, and an adversary only has to take that one bearer down. Under
electronic attack, or a rain cell, or a terrain mask, that is exactly what
happens — and the node discovers it after the flow has already stopped.

So the decision here is an **allocation**, not a selection: given what each flow
needs and what each slice can actually carry, place the demand across several
slices in small units, and hold enough spread that the loss of any single slice
still leaves the flow carried.

Three properties do the work.

**Slices are small.** Demand is placed in quanta rather than whole-flow lumps,
so a flow can sit across many bearers and no single placement is large enough to
be worth attacking on its own.

**Survivability is a constraint, not a preference.** A plan is only valid if,
for every flow, removing its single largest placement still leaves the flow's
requirement met. That costs headroom — carrying a flow across ``n`` even slices
needs ``n/(n-1)`` times its rate — and that cost is the point. A plan that
cannot survive one loss is **refused**, never delivered as a degraded plan that
looks like it worked.

**Roles never mix.** A command flow is placed only on slices whose bearer is
declared ``c2_cnpc`` in the signed catalogue, for the same reason ranking never
crossed roles: a payload bearer cannot become safety-critical by being
convenient.

What this module does not do: it does not move packets. It computes a placement
and the reasons for it. Enforcement belongs to a data-plane component under an
operator-signed lease, and no plan here authorises anything on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import utc_now
from .validation import bounded_number, non_empty_str

#: Smallest unit of demand that may be placed on a slice. Placing in quanta is
#: what "spread thinly" means mechanically: a flow crosses more bearers, and no
#: single placement carries enough to be worth taking out by itself.
DEFAULT_QUANTUM_KBPS = 16.0

#: Fewest distinct slices a flow may be spread across. Two is the floor at
#: which losing one leaves anything at all; it is a floor, not a target, and a
#: flow that declares more gets more.
MINIMUM_DIVERSITY = 2


class AllocationError(RuntimeError):
    """Raised when a plan cannot be built from what was offered."""


@dataclass(frozen=True)
class Slice:
    """One addressable unit of carriage: a bearer, optionally a network slice.

    ``capacity_kbps`` is measured, never declared. A bearer that has not been
    probed has no capacity here, which keeps an unmeasured slice out of a plan
    rather than letting it absorb demand on a catalogue promise.
    """

    slice_id: str
    profile_id: str
    link_role: str
    capacity_kbps: float
    #: Live-probe confidence for the bearer, in ``[0, 1]``. Capacity is scaled
    #: by it, so a bearer measured through a weak observation contributes less
    #: than one measured cleanly rather than contributing equally.
    health: float
    s_nssai: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        non_empty_str(self.slice_id, "slice_id")
        non_empty_str(self.profile_id, "profile_id")
        non_empty_str(self.link_role, "link_role")
        bounded_number(self.capacity_kbps, "capacity_kbps", 0.0, 10_000_000.0)
        bounded_number(self.health, "health", 0.0, 1.0)

    def usable_kbps(self) -> float:
        """Capacity this slice can be relied on for, discounted by evidence."""
        return self.capacity_kbps * self.health

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "profile_id": self.profile_id,
            "link_role": self.link_role,
            "capacity_kbps": round(self.capacity_kbps, 6),
            "health": round(self.health, 6),
            "usable_kbps": round(self.usable_kbps(), 6),
            "s_nssai": self.s_nssai,
        }


@dataclass(frozen=True)
class Demand:
    """What one flow needs carried, and how hard it is to lose."""

    flow_id: str
    link_role: str
    required_kbps: float
    #: Distinct slices this flow must be spread across. Raising it buys
    #: survivability and costs headroom; the plan pays that cost explicitly
    #: rather than quietly under-delivering.
    minimum_slices: int = MINIMUM_DIVERSITY

    def __post_init__(self) -> None:
        non_empty_str(self.flow_id, "flow_id")
        non_empty_str(self.link_role, "link_role")
        bounded_number(self.required_kbps, "required_kbps", 0.0, 10_000_000.0)
        if self.minimum_slices < MINIMUM_DIVERSITY:
            raise ValueError(
                f"minimum_slices must be at least {MINIMUM_DIVERSITY}; "
                "a flow on one slice has no diversity to lose"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "link_role": self.link_role,
            "required_kbps": round(self.required_kbps, 6),
            "minimum_slices": self.minimum_slices,
        }


@dataclass(frozen=True)
class Placement:
    """A quantity of one flow riding one slice."""

    flow_id: str
    slice_id: str
    kbps: float

    def to_dict(self) -> dict[str, Any]:
        return {"flow_id": self.flow_id, "slice_id": self.slice_id, "kbps": round(self.kbps, 6)}


@dataclass(frozen=True)
class FlowPlan:
    """How one flow was placed, or why it could not be."""

    flow_id: str
    carried: bool
    placements: tuple[Placement, ...]
    #: Rate still carried after the single largest placement is removed. This
    #: is the number survivability is judged on, and it is reported whether the
    #: flow was carried or refused so an operator can see how far short it fell.
    survives_worst_loss_kbps: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "carried": self.carried,
            "placements": [item.to_dict() for item in self.placements],
            "slices_used": len(self.placements),
            "survives_worst_loss_kbps": round(self.survives_worst_loss_kbps, 6),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class AllocationPlan:
    created_at: datetime
    quantum_kbps: float
    flows: tuple[FlowPlan, ...]
    slice_load_kbps: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def carried_flow_ids(self) -> tuple[str, ...]:
        return tuple(flow.flow_id for flow in self.flows if flow.carried)

    def refused_flow_ids(self) -> tuple[str, ...]:
        return tuple(flow.flow_id for flow in self.flows if not flow.carried)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "quantum_kbps": round(self.quantum_kbps, 6),
            "flows": [flow.to_dict() for flow in self.flows],
            "slice_load_kbps": {key: round(value, 6) for key, value in sorted(self.slice_load_kbps.items())},
            "carried": list(self.carried_flow_ids()),
            "refused": list(self.refused_flow_ids()),
            "reasons": list(self.reasons),
        }


def _survivable_rate(placements: list[Placement]) -> float:
    """Rate still carried once the single largest placement is removed."""
    if not placements:
        return 0.0
    total = sum(item.kbps for item in placements)
    return total - max(item.kbps for item in placements)


def plan_allocation(
    demands: list[Demand],
    slices: list[Slice],
    *,
    quantum_kbps: float = DEFAULT_QUANTUM_KBPS,
    now: datetime | None = None,
) -> AllocationPlan:
    """Spread each flow across enough slices to survive losing any one of them.

    Flows are placed in priority order — command traffic first — because when
    capacity is short the question is not who gets the best slice but who gets
    carried at all, and that ordering is a safety property rather than a
    preference. Within a flow, demand goes to the slice with the most remaining
    usable capacity, one quantum at a time, so the spread is even and no single
    slice accumulates a placement large enough to matter on its own.
    """
    if quantum_kbps <= 0.0:
        raise AllocationError("quantum must be positive")
    created = now or utc_now()

    remaining: dict[str, float] = {item.slice_id: item.usable_kbps() for item in slices}
    by_id = {item.slice_id: item for item in slices}
    if len(by_id) != len(slices):
        raise AllocationError("slice IDs must be unique")

    # Command traffic is placed first. Everything else competes for what is left.
    ordered = sorted(demands, key=lambda d: (0 if d.link_role == "c2_cnpc" else 1, d.flow_id))

    flow_plans: list[FlowPlan] = []
    for demand in ordered:
        eligible = [item for item in slices if item.link_role == demand.link_role]
        if len(eligible) < demand.minimum_slices:
            flow_plans.append(
                FlowPlan(
                    demand.flow_id,
                    False,
                    (),
                    0.0,
                    (
                        "FEWER_SLICES_IN_THIS_ROLE_THAN_THE_FLOW_REQUIRES_DIVERSITY",
                    ),
                )
            )
            continue

        # Carrying a flow across n slices and still meeting its rate after
        # losing one costs n/(n-1) of the rate. Ask for that up front rather
        # than discovering the shortfall after the loss.
        target = demand.required_kbps * demand.minimum_slices / (demand.minimum_slices - 1)
        # Never let one slice take so much that losing it sinks the flow. The
        # binding constraint is that the largest placement must be no more than
        # what the rest still cover: ``total - required``, which for an even
        # spread over n slices is ``required / (n - 1)``. Capping at the flow
        # rate instead — the obvious reading of "no slice carries the whole
        # flow" — is far too loose: one fat bearer takes the entire rate, the
        # thin ones share the headroom, and losing the fat one drops the flow.
        per_slice_cap = target - demand.required_kbps
        placed: dict[str, float] = {}
        assigned = 0.0
        while assigned + 1e-9 < target:
            # The step a slice can take is bounded four ways: the quantum, what
            # is left of the target, the slice's own remaining capacity, and its
            # share of the cap. Bounding by the quantum alone strands the plan
            # short of target whenever the cap is not a whole number of quanta —
            # each slice stalls just under its cap with the remainder unplaceable.
            steps = []
            for item in eligible:
                headroom = per_slice_cap - placed.get(item.slice_id, 0.0)
                step = min(quantum_kbps, target - assigned, remaining[item.slice_id], headroom)
                if step > 1e-9:
                    steps.append((item, step))
            if not steps:
                break
            # Most remaining capacity first, ties broken by slice_id so the
            # plan is deterministic and two nodes reading the same evidence
            # produce the same placement.
            steps.sort(key=lambda pair: (-remaining[pair[0].slice_id], pair[0].slice_id))
            chosen, step = steps[0]
            placed[chosen.slice_id] = placed.get(chosen.slice_id, 0.0) + step
            remaining[chosen.slice_id] -= step
            assigned += step

        placements = [Placement(demand.flow_id, sid, kbps) for sid, kbps in sorted(placed.items())]
        survivable = _survivable_rate(placements)
        reasons: list[str] = []
        carried = True
        if len(placements) < demand.minimum_slices:
            carried = False
            reasons.append("PLACED_ON_FEWER_SLICES_THAN_THE_FLOW_REQUIRES")
        if survivable + 1e-9 < demand.required_kbps:
            carried = False
            reasons.append("LOSING_THE_LARGEST_SLICE_WOULD_DROP_THE_FLOW_BELOW_ITS_RATE")
        if carried:
            reasons.append("SPREAD_SURVIVES_THE_LOSS_OF_ANY_SINGLE_SLICE")
        else:
            # Refusing means handing the capacity back. A plan that half-carries
            # a command flow is worse than one that says it cannot: the operator
            # can act on a refusal and cannot act on a silent shortfall.
            for placement in placements:
                remaining[placement.slice_id] += placement.kbps
            placements = []

        flow_plans.append(
            FlowPlan(demand.flow_id, carried, tuple(placements), survivable, tuple(reasons))
        )

    load = {
        item.slice_id: round(item.usable_kbps() - remaining[item.slice_id], 9) for item in slices
    }
    plan_reasons: list[str] = []
    if any(not flow.carried for flow in flow_plans):
        plan_reasons.append("SOME_FLOWS_REFUSED_RATHER_THAN_PARTIALLY_CARRIED")
    if not slices:
        plan_reasons.append("NO_MEASURED_SLICES_WERE_OFFERED")
    return AllocationPlan(
        created_at=created,
        quantum_kbps=quantum_kbps,
        flows=tuple(flow_plans),
        slice_load_kbps=load,
        reasons=tuple(plan_reasons),
    )


@dataclass(frozen=True)
class FlowTable:
    """Operator-signed statement of what this node must carry.

    Signed for the same reason the spectrum policy and the catalogue are: a node
    that could declare its own demand could declare itself entitled to every
    slice in the fleet. Rates and diversity floors are an operator's decision.
    """

    table_id: str
    created_at: datetime
    expires_at: datetime
    demands: tuple[Demand, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, now: datetime | None = None) -> "FlowTable":
        from .models import parse_timestamp

        current = now or utc_now()
        table_id = non_empty_str(raw.get("table_id"), "table_id")
        created_at = parse_timestamp(str(raw["created_at"]))
        expires_at = parse_timestamp(str(raw["expires_at"]))
        if expires_at <= created_at:
            raise ValueError("flow table expiry must be after creation")
        if expires_at <= current:
            raise ValueError("flow table has expired")
        demands = tuple(
            Demand(
                flow_id=non_empty_str(item.get("flow_id"), "flow_id"),
                link_role=str(item.get("link_role", "payload")).strip().lower(),
                required_kbps=bounded_number(
                    item.get("required_kbps"), "required_kbps", 0.0, 10_000_000.0
                ),
                minimum_slices=int(item.get("minimum_slices", MINIMUM_DIVERSITY)),
            )
            for item in raw.get("demands", [])
        )
        if not demands:
            raise ValueError("flow table must declare at least one demand")
        flow_ids = [item.flow_id for item in demands]
        if len(flow_ids) != len(set(flow_ids)):
            raise ValueError("flow table contains duplicate flow IDs")
        return cls(table_id, created_at, expires_at, demands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "demands": [item.to_dict() for item in self.demands],
        }


def slices_from_telemetry(catalog: Any, snapshot: Any) -> list[Slice]:
    """Build the carriable slices from what the node actually measured.

    One slice per profile that returned a usable observation. Capacity is the
    measured throughput, not a catalogue figure, and an observation with zero
    confidence contributes a slice with zero usable capacity rather than being
    silently dropped — the plan should be able to say a bearer was seen and
    found empty, which is different from a bearer nobody looked at.
    """
    profiles = {profile.profile_id: profile for profile in catalog.profiles}
    slices: list[Slice] = []
    for observation in snapshot.observations:
        profile = profiles.get(observation.profile_id)
        if profile is None:
            continue
        throughput_mbps = (observation.rx_mbps or 0.0) + (observation.tx_mbps or 0.0)
        requirement = getattr(profile, "standards", None)
        s_nssai = getattr(requirement, "s_nssai", None) if requirement is not None else None
        slices.append(
            Slice(
                slice_id=observation.profile_id,
                profile_id=observation.profile_id,
                link_role=profile.link_role,
                capacity_kbps=throughput_mbps * 1000.0,
                health=observation.confidence(),
                s_nssai=s_nssai.to_dict() if s_nssai is not None else None,
            )
        )
    return slices
