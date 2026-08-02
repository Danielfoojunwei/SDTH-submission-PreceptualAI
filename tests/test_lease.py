from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.lease import (
    CapabilityLease,
    LeaseAction,
    LeaseError,
    LeaseNonceStore,
    redeem_lease,
    sign_lease,
    verify_lease,
)

NOW = datetime.now(timezone.utc)


def lease_body(**overrides) -> dict:
    body = {
        "lease_id": "lease-1",
        "node_id": "vault-node-01",
        "action": "set_power_mode",
        "bounds": {"nvpmodel_id": 1.0, "max_inference_duty": 0.25},
        "policy_hash": "b" * 64,
        "nonce": "nonce-0123456789abcdef",
        "issued_at": (NOW - timedelta(seconds=5)).isoformat(),
        "expires_at": (NOW + timedelta(minutes=10)).isoformat(),
        "rollback_action": "restore economy mode",
        "required_postcondition": "nvpmodel reports target id",
    }
    body.update(overrides)
    return body


class LeaseSigningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()

    def test_round_trip(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        lease = verify_lease(envelope, self.key.public_key(), node_id="vault-node-01", now=NOW)
        self.assertIsInstance(lease, CapabilityLease)
        self.assertIs(lease.action, LeaseAction.SET_POWER_MODE)
        self.assertEqual(lease.bound("nvpmodel_id"), 1.0)
        self.assertTrue(lease.within("nvpmodel_id", 1.0))
        self.assertFalse(lease.within("nvpmodel_id", 2.0))

    def test_tampered_bounds_fail(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        altered = copy.deepcopy(envelope)
        altered["body"]["bounds"]["nvpmodel_id"] = 3.0
        with self.assertRaisesRegex(LeaseError, "signature verification failed"):
            verify_lease(altered, self.key.public_key(), node_id="vault-node-01", now=NOW)

    def test_wrong_node_rejected(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        with self.assertRaisesRegex(LeaseError, "different node"):
            verify_lease(envelope, self.key.public_key(), node_id="other-node", now=NOW)

    def test_expired_rejected(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        later = NOW + timedelta(minutes=20)
        with self.assertRaisesRegex(LeaseError, "expired"):
            verify_lease(envelope, self.key.public_key(), node_id="vault-node-01", now=later)

    def test_wrong_key_rejected(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        other = Ed25519PrivateKey.generate()
        with self.assertRaisesRegex(LeaseError, "key ID"):
            verify_lease(envelope, other.public_key(), node_id="vault-node-01", now=NOW)


class LeaseRedemptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()

    def test_single_use_nonce_refuses_replay(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        with tempfile.TemporaryDirectory() as directory:
            store = LeaseNonceStore(Path(directory) / "nonces.db")
            try:
                redeem_lease(
                    envelope,
                    self.key.public_key(),
                    store,
                    node_id="vault-node-01",
                    expected_action=LeaseAction.SET_POWER_MODE,
                    now=NOW,
                )
                self.assertTrue(store.is_consumed("nonce-0123456789abcdef"))
                with self.assertRaisesRegex(LeaseError, "already been consumed"):
                    redeem_lease(
                        envelope,
                        self.key.public_key(),
                        store,
                        node_id="vault-node-01",
                        expected_action=LeaseAction.SET_POWER_MODE,
                        now=NOW,
                    )
            finally:
                store.close()

    def test_action_mismatch_rejected(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        with tempfile.TemporaryDirectory() as directory:
            store = LeaseNonceStore(Path(directory) / "nonces.db")
            try:
                with self.assertRaisesRegex(LeaseError, "authorises"):
                    redeem_lease(
                        envelope,
                        self.key.public_key(),
                        store,
                        node_id="vault-node-01",
                        expected_action=LeaseAction.FAILOVER_CONNECTIVITY,
                        now=NOW,
                    )
                # A rejected lease must not consume the nonce.
                self.assertFalse(store.is_consumed("nonce-0123456789abcdef"))
            finally:
                store.close()

    def test_policy_hash_mismatch_rejected(self) -> None:
        envelope = sign_lease(lease_body(), self.key)
        with tempfile.TemporaryDirectory() as directory:
            store = LeaseNonceStore(Path(directory) / "nonces.db")
            try:
                with self.assertRaisesRegex(LeaseError, "policy hash"):
                    redeem_lease(
                        envelope,
                        self.key.public_key(),
                        store,
                        node_id="vault-node-01",
                        expected_action=LeaseAction.SET_POWER_MODE,
                        expected_policy_hash="c" * 64,
                        now=NOW,
                    )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
