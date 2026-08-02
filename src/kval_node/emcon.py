"""Emission control as a state the system can actually reason about.

Emission control is not a switch. Doctrine treats it as a spectrum of states
from unrestricted emission to total silence, chosen by weighing the risk to the
force against the risk to the mission — the US Army's own guidance is explicit
that "the price of complete EMCON in terms of absence of communications and
lack of tactical data may considerably inhibit mission completion".

Kval was EMCON-compatible by accident rather than by design: it has no
transmit path at all, so it cannot violate a silence order. That is a good
property and an incomplete one. A node that is *always* silent cannot express
"the platform is at EMCON Alpha, so suppress the backhaul heartbeat too", and
— far more seriously for an uncrewed platform — it cannot tell the difference
between silence it was ordered into and silence an adversary imposed.

**Why that distinction decides missions.** To a lost-link detector, a commanded
EMCON silence and a jamming-induced link loss are the same observation: nothing
is coming back. An uncrewed vehicle that runs its lost-link contingency because
it was *obeying* an emission-control order has been defeated by its own
doctrine while the adversary did nothing. And an adversary who jams during a
known EMCON window gets a return-to-base or a self-termination for free.

The resolution is directional, and it is the reason this module exists
separately from ``c2link.py``:

* **EMCON constrains what the platform emits. It says nothing about what the
  operator emits.** An operator can transmit commands to a silent vehicle all
  day. So a missing *return* path under commanded silence is compliance, while
  a missing *command* path is a lost link regardless of EMCON state.
* **Silence is bounded.** An EMCON order carries an expiry. A node that has
  been silent past the ordered window is no longer executing an order — it has
  lost contact with whoever would have lifted it — and the suppression must
  stop applying, or an adversary who jams the lifting order gets indefinite
  suppression of the contingency.
* **The state is signed and audited.** An unauthenticated EMCON order is a
  denial-of-service primitive: an adversary who can assert "you are at EMCON
  Alpha" can suppress a vehicle's lost-link response for as long as they like.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .models import parse_timestamp, utc_now
from .validation import non_empty_str

#: Ordered from least to most restrictive. The names follow common doctrinal
#: usage; the *meaning* here is defined by what the node is permitted to do,
#: because a level whose consequences are not encoded is decoration.
class EmconState(str, Enum):
    #: Emit normally. All links available.
    UNRESTRICTED = "unrestricted"
    #: Mission traffic only. Housekeeping — telemetry heartbeats, log shipping,
    #: corpus export, federated contributions — is suppressed.
    RESTRICTED = "restricted"
    #: Command and safety-of-life traffic only. Payload downlink is suppressed.
    MISSION_CRITICAL_ONLY = "mission_critical_only"
    #: Total silence from the platform. Receive-only sensing continues, because
    #: receiving emits nothing that a silence order is concerned with.
    SILENT = "silent"


_ORDER: tuple[EmconState, ...] = (
    EmconState.UNRESTRICTED,
    EmconState.RESTRICTED,
    EmconState.MISSION_CRITICAL_ONLY,
    EmconState.SILENT,
)


class EmconError(ValueError):
    """Raised when an emission-control order cannot be honoured."""


@dataclass(frozen=True)
class EmconOrder:
    """An operator-issued emission-control order, bounded in time.

    The expiry is not optional. An order without one is an order that survives
    the loss of whoever issued it, which turns a doctrinal control into a
    permanent condition an adversary can exploit by cutting the link that would
    have lifted it.
    """

    state: EmconState
    issued_at: datetime
    expires_at: datetime
    authority: str

    def __post_init__(self) -> None:
        non_empty_str(self.authority, "authority")
        if self.expires_at <= self.issued_at:
            raise EmconError("an EMCON order must expire after it was issued")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EmconOrder":
        try:
            state = EmconState(str(raw["state"]))
        except (KeyError, ValueError) as exc:
            raise EmconError(f"unknown EMCON state: {raw.get('state')!r}") from exc
        try:
            issued_at = parse_timestamp(str(raw["issued_at"]))
            expires_at = parse_timestamp(str(raw["expires_at"]))
        except (KeyError, ValueError) as exc:
            raise EmconError("an EMCON order must carry issued_at and expires_at") from exc
        return cls(state, issued_at, expires_at, non_empty_str(raw.get("authority"), "authority"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "authority": self.authority,
        }


@dataclass(frozen=True)
class EmconPosture:
    """What the node believes its emission posture is, and why."""

    state: EmconState
    ordered: bool
    expired: bool
    seconds_remaining: float | None
    suppressed_roles: tuple[str, ...]
    reasons: tuple[str, ...]

    def silence_commanded(self) -> bool:
        """Whether the platform is under an order to stop transmitting.

        The C2 monitor consumes exactly this to decide whether a dark return
        path is compliance or evidence. It is deliberately false for an expired
        order: silence past the ordered window is no longer an order being
        obeyed, and must stop suppressing the contingency.
        """
        return self.ordered and not self.expired and self.state in {
            EmconState.MISSION_CRITICAL_ONLY,
            EmconState.SILENT,
        }

    def permits(self, role: str) -> bool:
        return role not in self.suppressed_roles

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-emcon-posture/v1",
            "state": self.state.value,
            "ordered": self.ordered,
            "expired": self.expired,
            "silence_commanded": self.silence_commanded(),
            "seconds_remaining": (
                round(self.seconds_remaining, 1) if self.seconds_remaining is not None else None
            ),
            "suppressed_roles": list(self.suppressed_roles),
            "reasons": list(self.reasons),
        }


#: Which link roles each state suppresses. Encoding the consequence is what
#: makes a level mean something; a state with no encoded effect is a label.
_SUPPRESSED: dict[EmconState, tuple[str, ...]] = {
    EmconState.UNRESTRICTED: (),
    EmconState.RESTRICTED: ("bulk",),
    EmconState.MISSION_CRITICAL_ONLY: ("bulk", "payload"),
    EmconState.SILENT: ("bulk", "payload", "c2_cnpc"),
}


def assess_emcon(order: EmconOrder | None, *, now: datetime | None = None) -> EmconPosture:
    """Resolve the current emission posture from the standing order.

    With no order the posture is unrestricted — the node's receive-only
    architecture already means it emits nothing on its own account, so the
    absence of an order is not treated as a silence order. Inventing silence
    from missing information would suppress the lost-link contingency of every
    unconfigured platform.
    """
    current = now or utc_now()
    if order is None:
        return EmconPosture(
            state=EmconState.UNRESTRICTED,
            ordered=False,
            expired=False,
            seconds_remaining=None,
            suppressed_roles=(),
            reasons=(
                "NO_EMISSION_CONTROL_ORDER_IS_IN_FORCE",
                "RECEIVE_ONLY_SENSING_EMITS_NOTHING_A_SILENCE_ORDER_GOVERNS",
            ),
        )

    remaining = (order.expires_at - current).total_seconds()
    expired = remaining <= 0.0
    reasons: list[str] = [f"EMCON_ORDER_BY:{order.authority}"]

    if expired:
        # Critically: an expired order stops suppressing. Otherwise an
        # adversary who jams the lifting order keeps the contingency
        # suppressed for as long as they like.
        reasons.append("EMCON_ORDER_HAS_EXPIRED_AND_NO_LONGER_SUPPRESSES_ANYTHING")
        reasons.append("SILENCE_PAST_THE_ORDERED_WINDOW_IS_NOT_AN_ORDER_BEING_OBEYED")
        return EmconPosture(
            state=EmconState.UNRESTRICTED,
            ordered=True,
            expired=True,
            seconds_remaining=remaining,
            suppressed_roles=(),
            reasons=tuple(reasons),
        )

    suppressed = _SUPPRESSED[order.state]
    if order.state is EmconState.SILENT:
        reasons.append("TOTAL_SILENCE_ORDERED_INCLUDING_THE_COMMAND_RETURN_PATH")
        reasons.append("RECEIVE_ONLY_SENSING_CONTINUES_BECAUSE_RECEIVING_EMITS_NOTHING")
    elif suppressed:
        reasons.append(f"SUPPRESSED_ROLES:{','.join(suppressed)}")
    else:
        reasons.append("NO_ROLE_IS_SUPPRESSED_AT_THIS_LEVEL")

    if remaining < 60.0:
        reasons.append("EMCON_ORDER_EXPIRES_SHORTLY_AND_SUPPRESSION_WILL_LIFT")

    return EmconPosture(
        state=order.state,
        ordered=True,
        expired=False,
        seconds_remaining=remaining,
        suppressed_roles=suppressed,
        reasons=tuple(reasons),
    )


def is_more_restrictive(left: EmconState, right: EmconState) -> bool:
    return _ORDER.index(left) > _ORDER.index(right)
