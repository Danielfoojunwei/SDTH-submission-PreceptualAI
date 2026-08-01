from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.signing import sign_catalog, verify_catalog


def valid_payload() -> dict[str, object]:
    now = datetime.now(timezone.utc)
    return {
        "catalog_version": "test-v1",
        "created_at": (now - timedelta(minutes=1)).isoformat(),
        "expires_at": (now + timedelta(days=1)).isoformat(),
        "profiles": [
            {
                "profile_id": "wifi",
                "access_type": "wifi",
                "connection_uuid": "11111111-1111-4111-8111-111111111111",
                "interface": "wlan0",
                "probe_target": "127.0.0.1",
                "monetary_cost": 0.1,
                "energy_cost": 0.2,
                "minimum_dwell_seconds": 30,
                "maximum_switches_per_hour": 8,
                "allowed_service_classes": ["critical_data", "telemetry"],
                "standards": {"required_identifiers": []},
            }
        ],
    }


class SigningTests(unittest.TestCase):
    def test_real_ed25519_round_trip(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        envelope = sign_catalog(valid_payload(), private_key)
        catalog = verify_catalog(envelope, private_key.public_key())
        self.assertEqual(catalog.version, "test-v1")
        self.assertEqual(len(catalog.profiles), 1)

    def test_tampered_payload_fails(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        envelope = sign_catalog(valid_payload(), private_key)
        altered = copy.deepcopy(envelope)
        altered["payload"]["profiles"][0]["monetary_cost"] = 0.9
        with self.assertRaisesRegex(ValueError, "signature verification failed"):
            verify_catalog(altered, private_key.public_key())

    def test_placeholder_profile_is_rejected(self) -> None:
        private_key = Ed25519PrivateKey.generate()
        payload = valid_payload()
        payload["profiles"][0]["connection_uuid"] = "REPLACE_WITH_REAL_NETWORKMANAGER_UUID"
        with self.assertRaisesRegex(ValueError, "real value"):
            sign_catalog(payload, private_key)


if __name__ == "__main__":
    unittest.main()

