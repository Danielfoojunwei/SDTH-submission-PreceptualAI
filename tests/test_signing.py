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


class CatalogTrustBoundaryTests(unittest.TestCase):
    """Every rejection the signed catalogue makes, exercised.

    The signature check had tests; none of the *content* checks did. Tracing
    the suite showed the whole rejection half of ``validate_catalog_payload``
    had never executed a single line — expiry, ordering, duplicate profile IDs,
    duplicate NetworkManager UUIDs, empty catalogues, missing fields. They all
    turned out to work. They are the boundary the entire trust model rests on,
    which is not a thing to be right about by luck.
    """

    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.public = self.key.public_key()
        self.now = datetime.now(timezone.utc)

    def _rejects(self, pattern: str, mutate) -> None:
        payload = valid_payload()
        mutate(payload)
        with self.assertRaisesRegex(ValueError, pattern):
            sign_catalog(payload, self.key)

    def test_a_catalogue_expiring_before_it_was_created_is_refused(self) -> None:
        self._rejects("expiry must be after creation", lambda p: p.update(
            created_at=(self.now - timedelta(days=1)).isoformat(),
            expires_at=(self.now - timedelta(days=2)).isoformat(),
        ))

    def test_a_catalogue_created_in_the_future_is_refused(self) -> None:
        # A clock ahead of the operator's is how a not-yet-valid catalogue would
        # be accepted early, which is why this is checked rather than assumed.
        self._rejects("creation time is in the future", lambda p: p.update(
            created_at=(self.now + timedelta(hours=1)).isoformat()))

    def test_an_empty_catalogue_is_refused(self) -> None:
        self._rejects("at least one profile", lambda p: p.update(profiles=[]))

    def test_duplicate_profile_ids_are_refused(self) -> None:
        def duplicate(payload):
            second = copy.deepcopy(payload["profiles"][0])
            second["connection_uuid"] = "22222222-2222-4222-8222-222222222222"
            payload["profiles"].append(second)
        self._rejects("duplicate profile IDs", duplicate)

    def test_two_profiles_sharing_one_networkmanager_uuid_are_refused(self) -> None:
        # Two profiles pointing at one connection means activating either one
        # silently activates the same link, so the node's record of what it
        # selected stops matching what it did.
        def duplicate(payload):
            second = copy.deepcopy(payload["profiles"][0])
            second["profile_id"] = "other"
            payload["profiles"].append(second)
        self._rejects("duplicate NetworkManager UUIDs", duplicate)

    def test_a_missing_required_field_names_the_field(self) -> None:
        self._rejects("missing fields: catalog_version", lambda p: p.pop("catalog_version"))

    def test_a_catalogue_valid_when_signed_stops_being_accepted_once_it_expires(self) -> None:
        # The case sign-time validation cannot cover: a catalogue that was
        # entirely legitimate when issued and has since run out. Rejecting it is
        # the whole point of carrying an expiry.
        envelope = sign_catalog(valid_payload(), self.key)
        self.assertIsNotNone(verify_catalog(envelope, self.public, now=self.now + timedelta(hours=23)))
        with self.assertRaisesRegex(ValueError, "has expired"):
            verify_catalog(envelope, self.public, now=self.now + timedelta(hours=25))


if __name__ == "__main__":
    unittest.main()

