"""Connectivity recovery state machine.

A deterministic link-health state machine that turns a stream of link
observations into a bounded next action: keep probing, reconnect with
exponential backoff, fail over to the next approved profile, or escalate to an
operator. It enforces dwell/backoff timing and refuses to flap: once a link has
switched too many times in the trailing hour it degrades to escalation instead
of thrashing the radio.

The machine only *recommends*. Executing a reconnect or failover still goes
through the Safety Shield and, for failover, a capability lease.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .models import LinkObservation


class LinkHealth(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


class RecoveryAction(str, Enum):
    HOLD = "HOLD"
    PROBE_MORE = "PROBE_MORE"
    RECONNECT = "RECONNECT"
    FAILOVER = "FAILOVER"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class RecoveryConfig:
    degraded_loss_percent: float = 15.0
    degraded_rtt_ms: float = 400.0
    failed_loss_percent: float = 80.0
    healthy_samples_to_recover: int = 3
    degraded_samples_to_fail: int = 3
    backoff_base_seconds: float = 2.0
    backoff_cap_seconds: float = 60.0
    max_flaps_per_hour: int = 6

    def __post_init__(self) -> None:
        if self.failed_loss_percent <= self.degraded_loss_percent:
            raise ValueError("failed_loss_percent must exceed degraded_loss_percent")
        if self.backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds must be positive")
        if self.backoff_cap_seconds < self.backoff_base_seconds:
            raise ValueError("backoff_cap_seconds must be >= backoff_base_seconds")
        if self.healthy_samples_to_recover < 1 or self.degraded_samples_to_fail < 1:
            raise ValueError("sample thresholds must be positive")
        if self.max_flaps_per_hour < 1:
            raise ValueError("max_flaps_per_hour must be positive")


@dataclass(frozen=True)
class RecoveryDecision:
    health: LinkHealth
    action: RecoveryAction
    backoff_seconds: float
    consecutive_bad: int
    consecutive_good: int
    flaps_last_hour: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "health": self.health.value,
            "action": self.action.value,
            "backoff_seconds": self.backoff_seconds,
            "consecutive_bad": self.consecutive_bad,
            "consecutive_good": self.consecutive_good,
            "flaps_last_hour": self.flaps_last_hour,
            "reasons": list(self.reasons),
        }


@dataclass
class LinkStateMachine:
    config: RecoveryConfig = field(default_factory=RecoveryConfig)
    health: LinkHealth = LinkHealth.HEALTHY
    consecutive_bad: int = 0
    consecutive_good: int = 0
    reconnect_attempts: int = 0
    _switch_times: list[datetime] = field(default_factory=list)

    def record_switch(self, now: datetime) -> None:
        """Record that a reconnect/failover was actually performed."""
        self._switch_times.append(now)
        self._prune_switches(now)

    def _prune_switches(self, now: datetime) -> None:
        cutoff = now - timedelta(hours=1)
        self._switch_times = [moment for moment in self._switch_times if moment >= cutoff]

    def flaps_last_hour(self, now: datetime) -> int:
        self._prune_switches(now)
        return len(self._switch_times)

    def _backoff(self) -> float:
        # Clamp the exponent before the shift: an unbounded attempt counter on a
        # permanently down link would otherwise overflow the float multiply.
        exponent = min(max(0, self.reconnect_attempts - 1), 32)
        delay = self.config.backoff_base_seconds * (2**exponent)
        return min(delay, self.config.backoff_cap_seconds)

    @staticmethod
    def _classify(observation: LinkObservation, config: RecoveryConfig) -> tuple[bool, bool, str]:
        """Return (is_bad, is_failed, reason) for one observation."""
        if not observation.interface_present:
            return True, True, "INTERFACE_MISSING"
        if not observation.link_up:
            return True, True, "LINK_DOWN"
        if not observation.connection_active:
            return True, True, "CONNECTION_INACTIVE"
        if observation.loss_percent is None or observation.rtt_ms is None:
            return True, False, "PROBE_EVIDENCE_INCOMPLETE"
        if observation.loss_percent >= config.failed_loss_percent:
            return True, True, "LOSS_AT_FAILURE_THRESHOLD"
        if observation.loss_percent >= config.degraded_loss_percent:
            return True, False, "LOSS_ABOVE_DEGRADED_THRESHOLD"
        if observation.rtt_ms >= config.degraded_rtt_ms:
            return True, False, "RTT_ABOVE_DEGRADED_THRESHOLD"
        return False, False, "WITHIN_HEALTHY_LIMITS"

    def observe(self, observation: LinkObservation, *, now: datetime) -> RecoveryDecision:
        is_bad, is_failed, reason = self._classify(observation, self.config)
        reasons: list[str] = [reason]

        if is_bad:
            self.consecutive_bad += 1
            self.consecutive_good = 0
        else:
            self.consecutive_good += 1
            self.consecutive_bad = 0

        if is_failed:
            self.health = LinkHealth.FAILED
        elif is_bad and self.consecutive_bad >= self.config.degraded_samples_to_fail:
            self.health = LinkHealth.FAILED
            reasons.append("SUSTAINED_DEGRADATION")
        elif is_bad:
            self.health = LinkHealth.DEGRADED
        elif self.health in {LinkHealth.FAILED, LinkHealth.RECOVERING}:
            if self.consecutive_good >= self.config.healthy_samples_to_recover:
                self.health = LinkHealth.HEALTHY
                self.reconnect_attempts = 0
                reasons.append("STABLE_RECOVERY_CONFIRMED")
            else:
                self.health = LinkHealth.RECOVERING
                reasons.append("AWAITING_STABLE_RECOVERY")
        else:
            self.health = LinkHealth.HEALTHY

        flaps = self.flaps_last_hour(now)
        action, backoff = self._decide_action(flaps, reasons)
        return RecoveryDecision(
            health=self.health,
            action=action,
            backoff_seconds=backoff,
            consecutive_bad=self.consecutive_bad,
            consecutive_good=self.consecutive_good,
            flaps_last_hour=flaps,
            reasons=tuple(reasons),
        )

    def _decide_action(self, flaps: int, reasons: list[str]) -> tuple[RecoveryAction, float]:
        if self.health is LinkHealth.HEALTHY:
            return RecoveryAction.HOLD, 0.0
        if self.health is LinkHealth.RECOVERING:
            return RecoveryAction.PROBE_MORE, 0.0
        if self.health is LinkHealth.DEGRADED:
            return RecoveryAction.PROBE_MORE, 0.0
        # FAILED
        if flaps >= self.config.max_flaps_per_hour:
            reasons.append("FLAP_LIMIT_REACHED")
            return RecoveryAction.ESCALATE, self.config.backoff_cap_seconds
        # Saturate rather than grow without bound; the value only selects a
        # backoff step and distinguishes reconnect from failover.
        self.reconnect_attempts = min(self.reconnect_attempts + 1, 1_000_000)
        backoff = self._backoff()
        if self.reconnect_attempts <= 2:
            return RecoveryAction.RECONNECT, backoff
        reasons.append("RECONNECT_ATTEMPTS_EXHAUSTED")
        return RecoveryAction.FAILOVER, backoff
