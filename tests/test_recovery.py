from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from kval_node.models import LinkObservation
from kval_node.recovery import LinkHealth, LinkStateMachine, RecoveryAction, RecoveryConfig

NOW = datetime.now(timezone.utc)


def obs(**overrides) -> LinkObservation:
    base = dict(
        profile_id="wifi",
        interface="wlan0",
        observed_at=NOW,
        interface_present=True,
        link_up=True,
        connection_active=True,
        rtt_ms=20.0,
        loss_percent=0.0,
        jitter_ms=2.0,
        rx_mbps=10.0,
        tx_mbps=1.0,
        signal_percent=90.0,
        access_technology=None,
        standards_verified=True,
        standards_evidence=(),
        standards_errors=(),
        evidence=("ip-link",),
    )
    base.update(overrides)
    return LinkObservation(**base)


class RecoveryConfigTests(unittest.TestCase):
    def test_thresholds_validated(self) -> None:
        with self.assertRaises(ValueError):
            RecoveryConfig(degraded_loss_percent=90.0, failed_loss_percent=80.0)


class LinkStateMachineTests(unittest.TestCase):
    def test_healthy_holds(self) -> None:
        machine = LinkStateMachine()
        decision = machine.observe(obs(), now=NOW)
        self.assertIs(decision.health, LinkHealth.HEALTHY)
        self.assertIs(decision.action, RecoveryAction.HOLD)

    def test_link_down_fails_immediately(self) -> None:
        machine = LinkStateMachine()
        decision = machine.observe(obs(link_up=False), now=NOW)
        self.assertIs(decision.health, LinkHealth.FAILED)
        self.assertIs(decision.action, RecoveryAction.RECONNECT)
        self.assertGreater(decision.backoff_seconds, 0.0)

    def test_backoff_grows_then_failover(self) -> None:
        machine = LinkStateMachine(RecoveryConfig(backoff_base_seconds=2.0, backoff_cap_seconds=60.0))
        first = machine.observe(obs(link_up=False), now=NOW)
        second = machine.observe(obs(link_up=False), now=NOW + timedelta(seconds=1))
        third = machine.observe(obs(link_up=False), now=NOW + timedelta(seconds=2))
        self.assertIs(first.action, RecoveryAction.RECONNECT)
        self.assertIs(second.action, RecoveryAction.RECONNECT)
        self.assertGreater(second.backoff_seconds, first.backoff_seconds)
        self.assertIs(third.action, RecoveryAction.FAILOVER)

    def test_flap_limit_escalates(self) -> None:
        machine = LinkStateMachine(RecoveryConfig(max_flaps_per_hour=2))
        for index in range(3):
            machine.record_switch(NOW + timedelta(seconds=index))
        decision = machine.observe(obs(interface_present=False), now=NOW + timedelta(seconds=4))
        self.assertIs(decision.action, RecoveryAction.ESCALATE)
        self.assertIn("FLAP_LIMIT_REACHED", decision.reasons)

    def test_recovery_requires_stable_samples(self) -> None:
        config = RecoveryConfig(healthy_samples_to_recover=2)
        machine = LinkStateMachine(config)
        machine.observe(obs(link_up=False), now=NOW)
        recovering = machine.observe(obs(), now=NOW + timedelta(seconds=1))
        self.assertIs(recovering.health, LinkHealth.RECOVERING)
        self.assertIs(recovering.action, RecoveryAction.PROBE_MORE)
        healthy = machine.observe(obs(), now=NOW + timedelta(seconds=2))
        self.assertIs(healthy.health, LinkHealth.HEALTHY)

    def test_sustained_degradation_fails(self) -> None:
        config = RecoveryConfig(degraded_samples_to_fail=2, degraded_loss_percent=15.0, failed_loss_percent=80.0)
        machine = LinkStateMachine(config)
        first = machine.observe(obs(loss_percent=30.0), now=NOW)
        self.assertIs(first.health, LinkHealth.DEGRADED)
        second = machine.observe(obs(loss_percent=30.0), now=NOW + timedelta(seconds=1))
        self.assertIs(second.health, LinkHealth.FAILED)


if __name__ == "__main__":
    unittest.main()
