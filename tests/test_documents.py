from __future__ import annotations

import copy
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.signing import sign_document, verify_document


class SignedDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.payload = {"policy_id": "p1", "value": 42}

    def test_round_trip_returns_payload_and_hash(self) -> None:
        envelope = sign_document("kval-power-policy/v1", self.payload, self.key)
        payload, payload_hash = verify_document(
            envelope, self.key.public_key(), expected_type="kval-power-policy/v1"
        )
        self.assertEqual(payload, self.payload)
        self.assertEqual(len(payload_hash), 64)

    def test_wrong_type_rejected(self) -> None:
        envelope = sign_document("kval-power-policy/v1", self.payload, self.key)
        with self.assertRaisesRegex(ValueError, "expected a"):
            verify_document(envelope, self.key.public_key(), expected_type="kval-spectrum-policy/v1")

    def test_tampered_payload_rejected(self) -> None:
        envelope = sign_document("kval-power-policy/v1", self.payload, self.key)
        altered = copy.deepcopy(envelope)
        altered["payload"]["value"] = 43
        with self.assertRaisesRegex(ValueError, "signature verification failed"):
            verify_document(altered, self.key.public_key(), expected_type="kval-power-policy/v1")

    def test_type_binding_prevents_swap(self) -> None:
        # A document signed as one type must not verify by relabelling the envelope.
        envelope = sign_document("kval-power-policy/v1", self.payload, self.key)
        altered = copy.deepcopy(envelope)
        altered["document_type"] = "kval-spectrum-policy/v1"
        with self.assertRaises(ValueError):
            verify_document(altered, self.key.public_key(), expected_type="kval-spectrum-policy/v1")


if __name__ == "__main__":
    unittest.main()
