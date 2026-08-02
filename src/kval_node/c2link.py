"""The command-and-control link is not one link among many.

Every ranking decision this system made until now treated links as
interchangeable: score them on latency, loss, jitter, throughput, cost and
energy, and take the best. For a fixed sensor that is right. For an autonomous
uncrewed system — air, ground, surface or subsurface — it is **wrong in a way
that is both unsafe and non-compliant**, and the reason is regulatory before it
is technical.

**CNPC is a protected, separate thing.** Control and Non-Payload
Communications — the C2 link — carries the commands that fly, drive or steer
the platform and the state that says whether it obeyed. ITU-R sets aside
dedicated spectrum for it: **5030–5091 MHz** for terrestrial line-of-sight
(WRC-12) and **960–977 MHz** in L-band, with fixed-satellite allocations for
beyond-line-of-sight, sized at 34 MHz terrestrial and 56 MHz satellite. It has
its own airworthiness standards — RTCA **DO-362** (terrestrial C2 MOPS),
**DO-377** (satellite), FAA **TSO-C213** for the radios themselves — and, since
August 2024, its own FCC rule part (**Part 88**). Payload video does not.

Three consequences follow, and this module enforces all three:

**A C2 link may never be traded for a better payload link.** Ranking a C2
candidate against a payload candidate on one utility scale means a high-value
sensor downlink can outscore the link that steers the vehicle. That is not a
tuning error; it is a category error. Candidates are ranked **within their
role**, never across.

**Losing C2 is an event with a mandated response, not a routing problem.**
ICAO and RTCA DO-400 require a *pre-planned* contingency — a lost-link route,
orbit points, altitudes, and a flight-termination point if recovery is
infeasible. The response must be declared before the mission and executed on
loss, not invented at the moment. ``ContingencyAction`` is therefore a closed
vocabulary chosen by the operator in advance; the node selects nothing.

**The link budget is a floor, not a preference.** A C2 link that no longer
meets its declared latency and continuity budget is degraded whether or not it
is still the best available option. "Best of a bad set" is how a vehicle flies
on a link that cannot command it.

**The insight this module exists for.** For a *military* uncrewed system there
is a fourth case that the civil standards do not cover, because civil aviation
has no reason to fall silent on purpose. Under **emission control**, a platform
is *ordered* to stop transmitting. To every lost-link detector ever built, a
commanded EMCON silence and a jamming-induced link loss look identical — and a
vehicle that runs its lost-link contingency because it was obeying an EMCON
order has been defeated by its own doctrine, without the adversary doing
anything. Worse, an adversary who can *induce* the appearance of EMCON, or who
jams during a known EMCON window, gets the vehicle to return to base or
self-terminate for free.

So EMCON state is an explicit input here (see ``emcon.py``). While silence is
commanded, loss of the *return* path is expected and does not arm the
contingency; loss of the *command* path still does, because nothing about
EMCON stops an operator transmitting to the vehicle. That asymmetry is the
whole point, and it is why this is modelled as two directions rather than one
link.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Sequence

from .validation import bounded_number, non_empty_str, positive_int

#: ITU-R / ICAO protected CNPC allocations, in hertz. A profile whose declared
#: spectrum falls inside one of these is operating in protected C2 spectrum and
#: is subject to the corresponding airworthiness regime.
CNPC_PROTECTED_BANDS: tuple[tuple[str, int, int], ...] = (
    ("l-band-cnpc", 960_000_000, 977_000_000),
    ("c-band-cnpc", 5_030_000_000, 5_091_000_000),
)


class LinkRole(str, Enum):
    """What a link is *for*. Ranking never crosses this boundary."""

    #: Control and Non-Payload Communications: commands to the platform and the
    #: state that says whether it obeyed. Safety-critical.
    C2_CNPC = "c2_cnpc"
    #: Mission data — imagery, detections, telemetry of interest.
    PAYLOAD = "payload"
    #: Everything that can wait: log shipping, model updates, corpus export.
    BULK = "bulk"


class C2State(str, Enum):
    """Where the command link stands, per RTCA DO-400's progression."""

    NOMINAL = "nominal"
    #: Still carrying commands, but outside its declared budget.
    DEGRADED = "degraded"
    #: No command path. The contingency timer is running.
    LOST = "lost"
    #: The pre-planned contingency has been declared.
    CONTINGENCY = "contingency"


class ContingencyAction(str, Enum):
    """What the platform does on confirmed C2 loss.

    Closed and operator-declared. The node never chooses between these at
    runtime — choosing a contingency in the moment is precisely what the
    pre-planning requirement exists to prevent.
    """

    #: Hold present position or orbit a declared point, awaiting recovery.
    HOLD = "hold"
    #: Navigate a declared lost-link route back to a recovery point.
    RETURN_TO_RECOVERY = "return_to_recovery"
    #: Continue the planned mission on the assumption the link returns.
    CONTINUE_MISSION = "continue_mission"
    #: Proceed to a declared flight-termination point. Last resort.
    TERMINATE = "terminate"


class C2LinkError(ValueError):
    """Raised when a C2 configuration cannot be honoured safely."""


@dataclass(frozen=True)
class C2Budget:
    """The declared floor a command link must meet to be considered usable.

    These are airworthiness figures from the mission plan, not preferences. A
    link outside them is degraded even if it is the best link available —
    "best of a bad set" is how a vehicle ends up flying on a link that cannot
    command it.
    """

    #: One-way latency ceiling. DO-362 style C2 links are budgeted in the
    #: hundreds of milliseconds; the operator sets the number.
    max_latency_ms: float
    #: Loss ceiling over the measurement window.
    max_loss_percent: float
    #: How long a command path may be absent before it counts as lost.
    loss_confirmation_seconds: float
    #: How long it must stay lost before the contingency is declared. Separate
    #: from confirmation so a brief outage does not commit the platform.
    contingency_after_seconds: float
    #: What happens then. Declared here, before the mission.
    action: ContingencyAction

    def __post_init__(self) -> None:
        bounded_number(self.max_latency_ms, "max_latency_ms", 1.0, 60_000.0)
        bounded_number(self.max_loss_percent, "max_loss_percent", 0.0, 100.0)
        bounded_number(self.loss_confirmation_seconds, "loss_confirmation_seconds", 0.1, 3600.0)
        bounded_number(self.contingency_after_seconds, "contingency_after_seconds", 0.1, 86_400.0)
        if self.contingency_after_seconds < self.loss_confirmation_seconds:
            raise C2LinkError(
                "contingency_after_seconds must not precede loss confirmation; a platform "
                "cannot commit to a contingency before the loss is confirmed"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "C2Budget":
        try:
            action = ContingencyAction(str(raw["contingency_action"]))
        except (KeyError, ValueError) as exc:
            raise C2LinkError(
                "a C2 budget must declare a contingency_action from the closed vocabulary; "
                "an undeclared contingency is one chosen at the moment of failure"
            ) from exc
        return cls(
            max_latency_ms=float(raw["max_latency_ms"]),
            max_loss_percent=float(raw["max_loss_percent"]),
            loss_confirmation_seconds=float(raw["loss_confirmation_seconds"]),
            contingency_after_seconds=float(raw["contingency_after_seconds"]),
            action=action,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_latency_ms": self.max_latency_ms,
            "max_loss_percent": self.max_loss_percent,
            "loss_confirmation_seconds": self.loss_confirmation_seconds,
            "contingency_after_seconds": self.contingency_after_seconds,
            "contingency_action": self.action.value,
        }


def protected_band_for(freq_low_hz: int, freq_high_hz: int) -> str | None:
    """Name the protected CNPC allocation a band sits in, if any."""
    for name, low, high in CNPC_PROTECTED_BANDS:
        if freq_low_hz >= low and freq_high_hz <= high:
            return name
    return None


@dataclass(frozen=True)
class DirectionalHealth:
    """Health of one direction of the C2 link.

    Two directions, because they fail differently and — under EMCON — they are
    *supposed* to fail differently. The command path is the operator reaching
    the vehicle; the return path is the vehicle reporting back. Silence on the
    return path while EMCON is commanded is compliance. Silence on the command
    path never is.
    """

    present: bool
    latency_ms: float | None
    loss_percent: float | None
    last_seen: datetime | None

    def within(self, budget: C2Budget) -> bool:
        if not self.present:
            return False
        if self.latency_ms is None or self.loss_percent is None:
            return False
        return (
            self.latency_ms <= budget.max_latency_ms
            and self.loss_percent <= budget.max_loss_percent
        )


@dataclass(frozen=True)
class C2Assessment:
    state: C2State
    command_path_ok: bool
    return_path_ok: bool
    contingency: ContingencyAction | None
    seconds_without_command: float
    emcon_suppressed_return_path: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-c2-assessment/v1",
            "state": self.state.value,
            "command_path_ok": self.command_path_ok,
            "return_path_ok": self.return_path_ok,
            "contingency": self.contingency.value if self.contingency else None,
            "seconds_without_command": round(self.seconds_without_command, 3),
            "emcon_suppressed_return_path": self.emcon_suppressed_return_path,
            "reasons": list(self.reasons),
        }


class C2LinkMonitor:
    """Tracks the command link and declares the pre-planned contingency.

    Stateful across cycles because the timers are the mechanism: a link that
    blinks is not a lost link, and a contingency committed on one bad sample is
    a contingency an adversary can trigger with one pulse.
    """

    def __init__(self, budget: C2Budget) -> None:
        self.budget = budget
        self._command_lost_since: datetime | None = None
        self._declared: ContingencyAction | None = None

    def declared_contingency(self) -> ContingencyAction | None:
        return self._declared

    def reset(self) -> None:
        """Clear a declared contingency. Only an operator action should do this."""
        self._command_lost_since = None
        self._declared = None

    def observe(
        self,
        *,
        command_path: DirectionalHealth,
        return_path: DirectionalHealth,
        now: datetime,
        emission_silence_commanded: bool = False,
    ) -> C2Assessment:
        reasons: list[str] = []
        command_ok = command_path.within(self.budget)
        return_ok = return_path.within(self.budget)

        # The asymmetry that makes EMCON survivable. Nothing about a silence
        # order stops an operator transmitting *to* the vehicle, so the command
        # path is judged normally; only the return path is expected to be dark.
        suppressed = False
        if emission_silence_commanded and not return_ok:
            suppressed = True
            reasons.append("RETURN_PATH_SILENT_UNDER_COMMANDED_EMISSION_CONTROL")
            reasons.append("EMCON_SILENCE_IS_NOT_EVIDENCE_OF_A_LOST_LINK")

        if command_ok:
            if self._command_lost_since is not None:
                reasons.append("COMMAND_PATH_RECOVERED")
            self._command_lost_since = None
        else:
            if self._command_lost_since is None:
                self._command_lost_since = now
            if not command_path.present:
                reasons.append("NO_COMMAND_PATH_PRESENT")
            else:
                reasons.append("COMMAND_PATH_OUTSIDE_ITS_DECLARED_BUDGET")

        elapsed = (
            (now - self._command_lost_since).total_seconds()
            if self._command_lost_since is not None
            else 0.0
        )

        if self._declared is not None:
            # A declared contingency is not withdrawn by the link coming back.
            # The platform is executing a plan; unwinding it automatically
            # would put it somewhere neither the plan nor the operator expects.
            reasons.append("CONTINGENCY_ALREADY_DECLARED_AND_ONLY_AN_OPERATOR_CLEARS_IT")
            state = C2State.CONTINGENCY
        elif command_ok and (return_ok or suppressed):
            state = C2State.NOMINAL
            reasons.append("COMMAND_LINK_WITHIN_ITS_DECLARED_BUDGET")
        elif command_ok:
            state = C2State.DEGRADED
            reasons.append("RETURN_PATH_UNAVAILABLE_WHILE_COMMANDS_STILL_ARRIVE")
        elif elapsed >= self.budget.contingency_after_seconds:
            state = C2State.CONTINGENCY
            self._declared = self.budget.action
            reasons.append("PRE_PLANNED_CONTINGENCY_DECLARED_AS_THE_OPERATOR_SPECIFIED")
        elif elapsed >= self.budget.loss_confirmation_seconds:
            state = C2State.LOST
            reasons.append("COMMAND_PATH_LOSS_CONFIRMED_CONTINGENCY_TIMER_RUNNING")
        else:
            state = C2State.DEGRADED
            reasons.append("COMMAND_PATH_INTERRUPTED_BUT_NOT_YET_CONFIRMED_LOST")

        return C2Assessment(
            state=state,
            command_path_ok=command_ok,
            return_path_ok=return_ok,
            contingency=self._declared,
            seconds_without_command=elapsed,
            emcon_suppressed_return_path=suppressed,
            reasons=tuple(dict.fromkeys(reasons)),
        )


@dataclass(frozen=True)
class RoleRanking:
    """Candidate profiles grouped by role, ranked only within each group."""

    by_role: dict[str, tuple[str, ...]]
    reasons: tuple[str, ...]

    def selection_for(self, role: LinkRole) -> str | None:
        candidates = self.by_role.get(role.value, ())
        return candidates[0] if candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-role-ranking/v1",
            "by_role": {role: list(items) for role, items in sorted(self.by_role.items())},
            "reasons": list(self.reasons),
        }


def rank_within_roles(
    scored: Sequence[tuple[str, LinkRole, float]],
) -> RoleRanking:
    """Rank candidates inside each role and never across roles.

    This is the safety property the whole module exists for. A single ranking
    over every link lets a high-value payload downlink outscore the link that
    steers the vehicle — and utility functions are exactly the kind of thing
    that gets retuned for throughput by someone who does not know that.
    Segregating by role makes the mistake unrepresentable rather than
    unlikely.
    """
    grouped: dict[str, list[tuple[str, float]]] = {}
    for profile_id, role, utility in scored:
        grouped.setdefault(role.value, []).append(
            (non_empty_str(profile_id, "profile_id"), float(utility))
        )
    ranked = {
        role: tuple(
            profile
            for profile, _utility in sorted(items, key=lambda pair: (-pair[1], pair[0]))
        )
        for role, items in grouped.items()
    }
    reasons = ["CANDIDATES_RANKED_WITHIN_ROLE_NEVER_ACROSS_ROLES"]
    if LinkRole.C2_CNPC.value not in ranked:
        reasons.append("NO_C2_CANDIDATE_OFFERED_SO_NO_COMMAND_LINK_CAN_BE_SELECTED")
    return RoleRanking(ranked, tuple(reasons))


@dataclass(frozen=True)
class InteroperabilityDeclaration:
    """What this node claims under STANAG 4586, stated rather than implied.

    STANAG 4586 defines five Levels of Interoperability, from indirect receipt
    of data (1) through payload control (3) and vehicle control (4) to full
    control including launch and recovery (5). A comms-layer component does not
    give a control station any of those by itself, and claiming otherwise in a
    capability statement is the kind of overclaim that gets found in
    integration rather than in a pitch.
    """

    #: The highest LOI this component contributes to. One: it moves data and
    #: metadata between systems, and controls nothing.
    level_of_interoperability: int = 1
    #: STANAG 4817 covers multi-domain UxV control across air, sea and
    #: subsurface. Kval is domain-agnostic because it selects links, not
    #: vehicle behaviour — a genuine strength, and not the same as compliance.
    domains: tuple[str, ...] = ("land", "sea", "air", "subsurface")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-interoperability/v1",
            "stanag_4586_loi": self.level_of_interoperability,
            "loi_meaning": "indirect receipt and transmission of UxS data and metadata",
            "domains": list(self.domains),
            "controls_vehicle": False,
            "controls_payload": False,
            "reasons": [
                "THIS_COMPONENT_SELECTS_AND_ASSURES_LINKS_IT_DOES_NOT_CONTROL_A_VEHICLE",
                "LOI_4_AND_5_REQUIRE_A_CONTROL_STATION_THIS_IS_NOT_ONE",
            ],
        }


def validate_c2_profile(
    *,
    profile_id: str,
    role: LinkRole,
    freq_low_hz: int | None,
    freq_high_hz: int | None,
) -> tuple[str | None, tuple[str, ...]]:
    """Check a profile declared as C2 against the protected allocations.

    A C2 profile outside protected CNPC spectrum is not automatically wrong —
    military systems operate C2 in allocations civil aviation never touches —
    but it is worth stating, because a C2 link sharing an ISM band with every
    consumer device in the area has a very different jamming and congestion
    exposure from one in 5030–5091 MHz.
    """
    reasons: list[str] = []
    if role is not LinkRole.C2_CNPC:
        return None, ()
    if freq_low_hz is None or freq_high_hz is None:
        reasons.append("C2_PROFILE_DECLARES_NO_SPECTRUM_SO_ITS_PROTECTION_IS_UNKNOWN")
        return None, tuple(reasons)
    band = protected_band_for(positive_int(freq_low_hz, "freq_low_hz"), positive_int(freq_high_hz, "freq_high_hz"))
    if band is None:
        reasons.append("C2_LINK_IS_OUTSIDE_PROTECTED_CNPC_SPECTRUM_AND_SHARES_ITS_BAND")
    else:
        reasons.append(f"C2_LINK_IS_IN_PROTECTED_CNPC_SPECTRUM:{band}")
    return band, tuple(reasons)
