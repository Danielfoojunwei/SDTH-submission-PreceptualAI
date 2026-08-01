from __future__ import annotations

import base64
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.identity import verify_node_manifest
from kval_node.signing import canonical_json, public_key_id


def signed_manifest(private_key: Ed25519PrivateKey) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    payload = {
        "node_id": "air-node-01",
        "role": "air_relay",
        "deployment_id": "sdth-lab-2026",
        "learning_cohort_id": "connectivity-v1",
        "context_schema": "kval-runtime-context/v1",
        "allowed_model_families": ["deterministic-baseline-v1", "gnn-dqn-v1"],
        "allowed_service_classes": ["critical_data"],
        "expected_platform": "Jetson Orin Nano",
        "created_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(days=1)).isoformat(),
    }
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "payload": payload,
        "signature": base64.b64encode(private_key.sign(canonical_json(payload))).decode("ascii"),
    }


class IdentityTests(unittest.TestCase):
    def test_signed_identity_binds_role_and_learning_cohort(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        identity = verify_node_manifest(signed_manifest(private_key), private_key.public_key())
        self.assertEqual(identity.node_id, "air-node-01")
        self.assertEqual(identity.learning_cohort_id, "connectivity-v1")
        self.assertEqual(identity.role.value, "air_relay")

    def test_tampered_role_is_rejected(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        envelope = signed_manifest(private_key)
        envelope["payload"]["role"] = "ground_gateway"
        with self.assertRaisesRegex(ValueError, "signature"):
            verify_node_manifest(envelope, private_key.public_key())


if __name__ == "__main__":
    unittest.main()
