from __future__ import annotations

import copy
import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.signing import sign_document, verify_document

import tempfile
from pathlib import Path

from kval_node.pqc import LMS_SHA256_M32_H5, LmsSigner, LmsStateStore, generate_lms_key
from kval_node.signing import (
    CNSA_2_0_SIGNATURE_ALGORITHMS,
    ED25519_ALGORITHM,
    SUPPORTED_SIGNATURE_ALGORITHMS,
    AlgorithmRefused,
    SignaturePolicy,
    sign_document_lms,
    verify_document_agile,
)



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


class CryptoAgilityTests(unittest.TestCase):
    """Documents must verify under whichever approved algorithm signed them.

    CNSA 2.0 retires classical asymmetric cryptography and puts software and
    firmware signing first — support and prefer from 2025, exclusive by 2030,
    with new NSS acquisitions gated on 1 January 2027. A signed-policy trust
    root is firmware signing in all but name, so it has to be able to move.
    See docs/DEFENCE_STANDARDS.md.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.ed_key = Ed25519PrivateKey.generate()
        self.lms_key = generate_lms_key(lms_type=LMS_SHA256_M32_H5)
        self.signer = LmsSigner(self.lms_key, LmsStateStore(self.root / "lms.json"))
        self.payload = {"policy_id": "sp-1", "passive_only": True}

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_an_lms_signed_document_verifies(self) -> None:
        envelope = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        payload, _hash = verify_document_agile(
            envelope,
            expected_type="kval-spectrum-policy/v1",
            lms_key=self.lms_key.public_key(),
        )
        self.assertEqual(payload, self.payload)

    def test_an_ed25519_signed_document_still_verifies(self) -> None:
        # The migration must not break artefacts already in the field.
        envelope = sign_document("kval-spectrum-policy/v1", self.payload, self.ed_key)
        payload, _hash = verify_document_agile(
            envelope,
            expected_type="kval-spectrum-policy/v1",
            ed25519_key=self.ed_key.public_key(),
        )
        self.assertEqual(payload, self.payload)

    def test_the_signed_body_is_identical_across_algorithms(self) -> None:
        # So an operator can re-sign an existing document under CNSA 2.0
        # without its payload changing meaning.
        classical = sign_document("kval-spectrum-policy/v1", self.payload, self.ed_key)
        quantum = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        self.assertEqual(classical["payload"], quantum["payload"])
        self.assertEqual(classical["document_type"], quantum["document_type"])

    def test_a_tampered_lms_payload_is_refused(self) -> None:
        envelope = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        envelope["payload"] = {"policy_id": "sp-1", "passive_only": False}
        with self.assertRaises(ValueError):
            verify_document_agile(
                envelope,
                expected_type="kval-spectrum-policy/v1",
                lms_key=self.lms_key.public_key(),
            )

    def test_an_algorithm_cannot_be_relabelled_to_one_the_attacker_controls(self) -> None:
        # The algorithm is read from the envelope but never trusted from it:
        # trust anchors come from the caller, so relabelling finds no key.
        attacker_key = Ed25519PrivateKey.generate()
        forged = sign_document("kval-spectrum-policy/v1", {"passive_only": False}, attacker_key)
        with self.assertRaises(ValueError):
            verify_document_agile(
                forged,
                expected_type="kval-spectrum-policy/v1",
                ed25519_key=self.ed_key.public_key(),
            )

    def test_a_document_whose_algorithm_has_no_anchor_is_refused_not_skipped(self) -> None:
        envelope = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        with self.assertRaises(AlgorithmRefused):
            verify_document_agile(envelope, expected_type="kval-spectrum-policy/v1")

    def test_an_unknown_algorithm_is_refused(self) -> None:
        envelope = sign_document("kval-spectrum-policy/v1", self.payload, self.ed_key)
        envelope["algorithm"] = "RSA-1024"
        with self.assertRaises(AlgorithmRefused):
            verify_document_agile(
                envelope,
                expected_type="kval-spectrum-policy/v1",
                ed25519_key=self.ed_key.public_key(),
            )

    def test_requiring_cnsa_2_0_refuses_ed25519(self) -> None:
        envelope = sign_document("kval-spectrum-policy/v1", self.payload, self.ed_key)
        with self.assertRaises(AlgorithmRefused):
            verify_document_agile(
                envelope,
                expected_type="kval-spectrum-policy/v1",
                ed25519_key=self.ed_key.public_key(),
                policy=SignaturePolicy(require_cnsa_2_0=True),
            )

    def test_requiring_cnsa_2_0_accepts_lms(self) -> None:
        envelope = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        payload, _hash = verify_document_agile(
            envelope,
            expected_type="kval-spectrum-policy/v1",
            lms_key=self.lms_key.public_key(),
            policy=SignaturePolicy(require_cnsa_2_0=True),
        )
        self.assertEqual(payload, self.payload)

    def test_ed25519_is_not_a_cnsa_2_0_algorithm(self) -> None:
        # The load-bearing fact behind this whole migration.
        self.assertNotIn(ED25519_ALGORITHM, CNSA_2_0_SIGNATURE_ALGORITHMS)
        self.assertIn(ED25519_ALGORITHM, SUPPORTED_SIGNATURE_ALGORITHMS)

    def test_a_document_of_the_wrong_type_is_refused(self) -> None:
        envelope = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        with self.assertRaises(ValueError):
            verify_document_agile(
                envelope,
                expected_type="kval-power-policy/v1",
                lms_key=self.lms_key.public_key(),
            )

    def test_a_document_signed_by_another_lms_key_is_refused(self) -> None:
        envelope = sign_document_lms("kval-spectrum-policy/v1", self.payload, self.signer)
        other = generate_lms_key(lms_type=LMS_SHA256_M32_H5).public_key()
        with self.assertRaises(ValueError):
            verify_document_agile(
                envelope, expected_type="kval-spectrum-policy/v1", lms_key=other
            )
