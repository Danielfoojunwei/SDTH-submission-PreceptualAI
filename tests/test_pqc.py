"""Quantum-resistant signing for the policy trust root.

Kval signs every trust decision with Ed25519, which is **not in CNSA 2.0** —
NSA's algorithm suite for National Security Systems retires all classical
asymmetric cryptography. Software and firmware signing is its most urgent
category: support and prefer from 2025, exclusive use by 2030, with every new
NSS acquisition gated on CNSA 2.0 support from 1 January 2027. See
docs/DEFENCE_STANDARDS.md.

LMS is implemented per RFC 8554 and NIST SP 800-208, and is checked against the
RFC's own published test vector rather than only against itself — a signature
scheme that agrees with its own bugs verifies nothing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kval_node.pqc import (
    D_INTR,
    D_LEAF,
    D_MESG,
    D_PBLC,
    LMOTS_SHA256_N32_W4,
    LMOTS_SHA256_N32_W8,
    LMS_SHA256_M32_H5,
    LmsPublicKey,
    LmsSigner,
    LmsStateStore,
    PqcError,
    StateExhausted,
    StateRollback,
    checksum,
    coef,
    generate_lms_key,
    lms_verify,
)

# --- RFC 8554 Appendix F, Test Case 1 ------------------------------------------
#
# The published vector is an HSS signature with two levels. Its sig[0] is a
# plain LMS signature by the level-0 key over the *serialised level-1 public
# key*, which is exactly the single-tree case CNSA 2.0 permits, so it is used
# here verbatim: same public key, same message bytes, same signature bytes as
# the RFC prints them.

RFC8554_TC1_PUBLIC_KEY = bytes.fromhex(
    """
    000000050000000461a5d57d37f5e46bfb7520806b07a1b850650e3b31fe4a773ea2
    9a07f09cf2ea30e579f0df58ef8e298da0434cb2b878
    """
)

RFC8554_TC1_MESSAGE = bytes.fromhex(
    """
    0000000500000004d2f14ff6346af964569f7d6cb880a1b66c5004917da6eafe4d9e
    f6c6407b3db0e5485b122d9ebe15cda93cfec582d7ab
    """
)

RFC8554_TC1_SIGNATURE = bytes.fromhex(
    """
    0000000500000004d32b56671d7eb98833c49b433c272586bc4a1c8a8970528ffa04
    b966f9426eb9965a25bfd37f196b9073f3d4a232feb69128ec45146f86292f9dff96
    10a7bf95a64c7f60f6261a62043f86c70324b7707f5b4a8a6e19c114c7be866d4887
    78a0e05fd5c6509a6e61d559cf1a77a970de927d60c70d3de31a7fa0100994e162a2
    582e8ff1b10cd99d4e8e413ef469559f7d7ed12c838342f9b9c96b83a4943d1681d8
    4b15357ff48ca579f19f5e71f18466f2bbef4bf660c2518eb20de2f66e3b14784269
    d7d876f5d35d3fbfc7039a462c716bb9f6891a7f41ad133e9e1f6d9560b960e7777c
    52f060492f2d7c660e1471e07e72655562035abc9a701b473ecbc3943c6b9c4f2405
    a3cb8bf8a691ca51d3f6ad2f428bab6f3a30f55dd9625563f0a75ee390e385e3ae0b
    906961ecf41ae073a0590c2eb6204f44831c26dd768c35b167b28ce8dc988a374825
    5230cef99ebf14e730632f27414489808afab1d1e783ed04516de012498682212b07
    810579b250365941bcc98142da13609e9768aaf65de7620dabec29eb82a17fde35af
    15ad238c73f81bdb8dec2fc0e7f932701099762b37f43c4a3c20010a3d72e2f606be
    108d310e639f09ce7286800d9ef8a1a40281cc5a7ea98d2adc7c7400c2fe5a101552
    df4e3cccfd0cbf2ddf5dc6779cbbc68fee0c3efe4ec22b83a2caa3e48e0809a0a750
    b73ccdcf3c79e6580c154f8a58f7f24335eec5c5eb5e0cf01dcf4439424095fceb07
    7f66ded5bec73b27c5b9f64a2a9af2f07c05e99e5cf80f00252e39db32f6c19674f1
    90c9fbc506d826857713afd2ca6bb85cd8c107347552f30575a5417816ab4db3f603
    f2df56fbc413e7d0acd8bdd81352b2471fc1bc4f1ef296fea1220403466b1afe78b9
    4f7ecf7cc62fb92be14f18c2192384ebceaf8801afdf947f698ce9c6ceb696ed70e9
    e87b0144417e8d7baf25eb5f70f09f016fc925b4db048ab8d8cb2a661ce3b57ada67
    571f5dd546fc22cb1f97e0ebd1a65926b1234fd04f171cf469c76b884cf3115cce6f
    792cc84e36da58960c5f1d760f32c12faef477e94c92eb75625b6a371efc72d60ca5
    e908b3a7dd69fef0249150e3eebdfed39cbdc3ce9704882a2072c75e13527b7a581a
    556168783dc1e97545e31865ddc46b3c957835da252bb7328d3ee2062445dfb85ef8
    c35f8e1f3371af34023cef626e0af1e0bc017351aae2ab8f5c612ead0b729a1d059d
    02bfe18efa971b7300e882360a93b025ff97e9e0eec0f3f3f13039a17f88b0cf808f
    488431606cb13f9241f40f44e537d302c64a4f1f4ab949b9feefadcb71ab50ef27d6
    d6ca8510f150c85fb525bf25703df7209b6066f09c37280d59128d2f0f637c7d7d7f
    ad4ed1c1ea04e628d221e3d8db77b7c878c9411cafc5071a34a00f4cf07738912753
    dfce48f07576f0d4f94f42c6d76f7ce973e9367095ba7e9a3649b7f461d9f9ac1332
    a4d1044c96aefee67676401b64457c54d65fef6500c59cdfb69af7b6dddfcb0f0862
    78dd8ad0686078dfb0f3f79cd893d314168648499898fbc0ced5f95b74e8ff14d735
    cdea968bee7400000005d8b8112f9200a5e50c4a262165bd342cd800b8496810bc71
    6277435ac376728d129ac6eda839a6f357b5a04387c5ce97382a78f2a4372917eefc
    bf93f63bb59112f5dbe400bd49e4501e859f885bf0736e90a509b30a26bfac8c17b5
    991c157eb5971115aa39efd8d564a6b90282c3168af2d30ef89d51bf14654510a12b
    8a144cca1848cf7da59cc2b3d9d0692dd2a20ba3863480e25b1b85ee860c62bf5136
    """
)


class Rfc8554KnownAnswerTests(unittest.TestCase):
    """The only test here that could catch a subtly wrong implementation.

    Round-tripping a signature through my own signer and verifier proves they
    agree with each other, which a consistently wrong pair does too. Agreeing
    with the RFC's printed bytes is the property that matters.
    """

    def test_the_rfc_s_own_signature_verifies(self) -> None:
        public_key = LmsPublicKey.from_bytes(RFC8554_TC1_PUBLIC_KEY)
        self.assertTrue(
            lms_verify(public_key, RFC8554_TC1_MESSAGE, RFC8554_TC1_SIGNATURE)
        )

    def test_the_published_parameters_are_what_we_parse(self) -> None:
        public_key = LmsPublicKey.from_bytes(RFC8554_TC1_PUBLIC_KEY)
        self.assertEqual(public_key.lms_type, LMS_SHA256_M32_H5)
        self.assertEqual(public_key.ots_type, LMOTS_SHA256_N32_W8)
        self.assertEqual(public_key.signatures_available(), 32)

    def test_a_single_altered_message_byte_is_rejected(self) -> None:
        public_key = LmsPublicKey.from_bytes(RFC8554_TC1_PUBLIC_KEY)
        altered = RFC8554_TC1_MESSAGE[:-1] + bytes([RFC8554_TC1_MESSAGE[-1] ^ 0x01])
        self.assertFalse(lms_verify(public_key, altered, RFC8554_TC1_SIGNATURE))

    def test_a_single_altered_signature_byte_is_rejected(self) -> None:
        public_key = LmsPublicKey.from_bytes(RFC8554_TC1_PUBLIC_KEY)
        for position in (0, 4, 40, 600, 1200, 1291):
            altered = (
                RFC8554_TC1_SIGNATURE[:position]
                + bytes([RFC8554_TC1_SIGNATURE[position] ^ 0x01])
                + RFC8554_TC1_SIGNATURE[position + 1 :]
            )
            self.assertFalse(
                lms_verify(public_key, RFC8554_TC1_MESSAGE, altered),
                f"a flipped bit at offset {position} was accepted",
            )

    def test_another_key_does_not_verify_it(self) -> None:
        other = generate_lms_key(lms_type=LMS_SHA256_M32_H5).public_key()
        self.assertFalse(lms_verify(other, RFC8554_TC1_MESSAGE, RFC8554_TC1_SIGNATURE))


class PrimitiveTests(unittest.TestCase):
    def test_domain_separators_match_the_rfc(self) -> None:
        # These stop a leaf hash ever being read as an interior node, which is
        # what a tree-shape forgery needs.
        self.assertEqual((D_PBLC, D_MESG, D_LEAF, D_INTR), (0x8080, 0x8181, 0x8282, 0x8383))

    def test_coef_reads_whole_bytes_at_w8(self) -> None:
        value = bytes([0x00, 0x7F, 0xFF, 0xAB])
        self.assertEqual([coef(value, i, 8) for i in range(4)], [0x00, 0x7F, 0xFF, 0xAB])

    def test_coef_reads_nibbles_at_w4(self) -> None:
        self.assertEqual([coef(bytes([0xAB, 0xCD]), i, 4) for i in range(4)], [0xA, 0xB, 0xC, 0xD])

    def test_checksum_is_maximal_for_an_all_zero_digest(self) -> None:
        # Every field is at the bottom of its chain, so the checksum is at its
        # maximum: 32 fields x 255.
        self.assertEqual(checksum(bytes(32), w=8, ls=0, n=32), 32 * 255)

    def test_checksum_is_zero_for_an_all_ones_digest(self) -> None:
        self.assertEqual(checksum(b"\xff" * 32, w=8, ls=0, n=32), 0)

    def test_checksum_falls_as_the_digest_rises(self) -> None:
        # The property the whole one-time construction rests on: you cannot
        # raise every field at once, because the checksum fields move the
        # other way and hashes do not run backwards.
        low = checksum(bytes(32), w=8, ls=0, n=32)
        high = checksum(b"\x80" * 32, w=8, ls=0, n=32)
        self.assertGreater(low, high)


class SigningTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.key = generate_lms_key(lms_type=LMS_SHA256_M32_H5)
        self.state_path = self.root / "lms-state.json"
        self.signer = LmsSigner(self.key, LmsStateStore(self.state_path))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_a_signature_verifies_under_its_own_key(self) -> None:
        message = b"kval-spectrum-policy/v1"
        self.assertTrue(lms_verify(self.key.public_key(), message, self.signer.sign(message)))

    def test_every_signature_spends_a_new_one_time_key(self) -> None:
        # Signing twice from one leaf hands an attacker that leaf's private
        # key (RFC 8554 section 9.1). The index must advance every time.
        leaves = [int.from_bytes(self.signer.sign(b"m")[:4], "big") for _ in range(5)]
        self.assertEqual(leaves, [0, 1, 2, 3, 4])

    def test_signing_the_same_message_twice_still_advances(self) -> None:
        first = self.signer.sign(b"identical")
        second = self.signer.sign(b"identical")
        self.assertNotEqual(first[:4], second[:4])
        self.assertTrue(lms_verify(self.key.public_key(), b"identical", first))
        self.assertTrue(lms_verify(self.key.public_key(), b"identical", second))

    def test_state_is_durable_across_a_restart(self) -> None:
        self.signer.sign(b"before")
        resumed = LmsSigner(self.key, LmsStateStore(self.state_path))
        self.assertEqual(int.from_bytes(resumed.sign(b"after")[:4], "big"), 1)

    def test_the_index_is_reserved_before_the_signature_exists(self) -> None:
        # A crash mid-sign must waste an index, never re-issue one. After a
        # signature the stored index is already past it.
        self.signer.sign(b"m")
        stored = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["next_index"], 1)

    def test_a_restored_backup_is_refused_rather_than_reusing_keys(self) -> None:
        # The ordinary, blameless way key reuse happens in the field.
        self.signer.sign(b"one")
        self.signer.sign(b"two")
        rolled_back = json.loads(self.state_path.read_text(encoding="utf-8"))
        rolled_back["next_index"] = 0
        self.state_path.write_text(json.dumps(rolled_back), encoding="utf-8")
        with self.assertRaises(StateRollback):
            self.signer.sign(b"three")

    def test_exhaustion_is_a_hard_stop_not_a_wrap_around(self) -> None:
        for _ in range(32):
            self.signer.sign(b"m")
        self.assertEqual(self.signer.remaining(), 0)
        with self.assertRaises(StateExhausted):
            self.signer.sign(b"one too many")

    def test_remaining_counts_down(self) -> None:
        self.assertEqual(self.signer.remaining(), 32)
        self.signer.sign(b"m")
        self.assertEqual(self.signer.remaining(), 31)

    def test_state_from_a_different_key_is_refused(self) -> None:
        # Pointing a new key at an old state file would start it mid-tree or,
        # worse, let two keys share an index space.
        self.signer.sign(b"m")
        other = generate_lms_key(lms_type=LMS_SHA256_M32_H5)
        with self.assertRaises(PqcError):
            LmsSigner(other, LmsStateStore(self.state_path)).sign(b"m")

    def test_unreadable_state_is_refused_not_treated_as_zero(self) -> None:
        # Assuming index zero is precisely what reuses every leaf in the tree.
        self.state_path.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(PqcError):
            LmsStateStore(self.state_path)

    def test_a_signature_does_not_verify_at_the_wrong_leaf(self) -> None:
        signature = self.signer.sign(b"m")
        moved = (1).to_bytes(4, "big") + signature[4:]
        self.assertFalse(lms_verify(self.key.public_key(), b"m", moved))

    def test_a_leaf_beyond_the_tree_is_rejected(self) -> None:
        signature = self.signer.sign(b"m")
        beyond = (999).to_bytes(4, "big") + signature[4:]
        self.assertFalse(lms_verify(self.key.public_key(), b"m", beyond))


class KeyHandlingTests(unittest.TestCase):
    def test_public_key_round_trips(self) -> None:
        public_key = generate_lms_key(lms_type=LMS_SHA256_M32_H5).public_key()
        self.assertEqual(LmsPublicKey.from_bytes(public_key.to_bytes()), public_key)

    def test_key_ids_differ_between_keys(self) -> None:
        first = generate_lms_key(lms_type=LMS_SHA256_M32_H5).public_key()
        second = generate_lms_key(lms_type=LMS_SHA256_M32_H5).public_key()
        self.assertNotEqual(first.key_id(), second.key_id())

    def test_a_truncated_public_key_is_refused(self) -> None:
        with self.assertRaises(PqcError):
            LmsPublicKey.from_bytes(b"\x00\x00\x00\x05")

    def test_an_unknown_algorithm_type_is_refused(self) -> None:
        with self.assertRaises(PqcError):
            LmsPublicKey.from_bytes(b"\xff\xff\xff\xff" + bytes(4 + 16 + 32))

    def test_an_unsupported_type_cannot_be_generated(self) -> None:
        with self.assertRaises(PqcError):
            generate_lms_key(lms_type=0xDEADBEEF)

    def test_a_narrower_winternitz_parameter_also_works(self) -> None:
        # w=4 gives longer signatures and faster signing; both are in the RFC
        # and neither should be silently unsupported.
        key = generate_lms_key(lms_type=LMS_SHA256_M32_H5, ots_type=LMOTS_SHA256_N32_W4)
        with tempfile.TemporaryDirectory() as directory:
            signer = LmsSigner(key, LmsStateStore(Path(directory) / "s.json"))
            self.assertTrue(lms_verify(key.public_key(), b"m", signer.sign(b"m")))

    def test_garbage_never_raises_out_of_verification(self) -> None:
        # A verifier that raises is a denial-of-service surface: an attacker
        # sends malformed bytes and takes the node down instead of being told
        # no.
        public_key = generate_lms_key(lms_type=LMS_SHA256_M32_H5).public_key()
        for junk in (b"", b"\x00", b"\xff" * 64, bytes(2000)):
            self.assertFalse(lms_verify(public_key, b"m", junk))


if __name__ == "__main__":
    unittest.main()
