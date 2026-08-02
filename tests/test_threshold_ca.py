"""Distributed certification and threshold custody for partitioned fleets.

From Lam's MANET distributed-CA work: "Self-initialized Distributed Certificate
Authority for Mobile Ad Hoc Network" (ISA 2009) and Lam, Gu, Chung et al., "A
robust certification service for highly dynamic MANET in emergency tasks",
Int. J. Communication Systems 22(9), 2009. See docs/DSA_RESEARCH_MAPPING.md.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.threshold_ca import (
    Attestation,
    CertificationError,
    CertifierRoster,
    Grant,
    QuorumRenewer,
    Share,
    ShareError,
    attest,
    scope_hash,
    shamir_combine,
    shamir_split,
)

NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
SECRET = b"corpus-encryption-key-0123456789"


class FieldArithmeticTests(unittest.TestCase):
    """The sharing scheme is only sound if the field underneath it is real.

    Round-trip tests alone would pass over a broken but self-consistent
    multiplication table, so the field is checked against published vectors and
    against its own axioms.
    """

    def test_matches_the_fips_197_worked_examples(self) -> None:
        from kval_node.threshold_ca import _mul

        # FIPS-197 section 4.2: 0x57 * 0x83 = 0xc1, 0x57 * 0x13 = 0xfe.
        self.assertEqual(_mul(0x57, 0x83), 0xC1)
        self.assertEqual(_mul(0x57, 0x13), 0xFE)

    def test_every_non_zero_element_has_an_inverse(self) -> None:
        from kval_node.threshold_ca import _div, _mul

        for value in range(1, 256):
            self.assertEqual(_mul(value, _div(1, value)), 1)

    def test_multiplication_is_commutative_and_absorbs_zero(self) -> None:
        from kval_node.threshold_ca import _mul

        for left in range(0, 256, 7):
            self.assertEqual(_mul(left, 0), 0)
            for right in range(0, 256, 11):
                self.assertEqual(_mul(left, right), _mul(right, left))

    def test_multiplication_distributes_over_addition(self) -> None:
        from kval_node.threshold_ca import _mul

        for left in range(0, 256, 13):
            for middle in range(0, 256, 17):
                for right in range(0, 256, 19):
                    self.assertEqual(
                        _mul(left, middle ^ right), _mul(left, middle) ^ _mul(left, right)
                    )

    def test_division_by_zero_is_an_error(self) -> None:
        from kval_node.threshold_ca import _div

        with self.assertRaises(ZeroDivisionError):
            _div(5, 0)


class ShamirTests(unittest.TestCase):
    def test_threshold_shares_reconstruct_the_secret(self) -> None:
        shares = shamir_split(SECRET, threshold=3, shares=5)
        self.assertEqual(shamir_combine(shares[:3]), SECRET)

    def test_any_threshold_sized_subset_works(self) -> None:
        shares = shamir_split(SECRET, threshold=3, shares=5)
        self.assertEqual(shamir_combine([shares[0], shares[2], shares[4]]), SECRET)
        self.assertEqual(shamir_combine([shares[1], shares[3], shares[4]]), SECRET)

    def test_more_than_the_threshold_also_works(self) -> None:
        shares = shamir_split(SECRET, threshold=3, shares=5)
        self.assertEqual(shamir_combine(shares), SECRET)

    def test_fewer_than_the_threshold_reveals_nothing(self) -> None:
        # A captured drone holds one share. Two below threshold must not leak
        # the secret, and must not accidentally produce it either.
        shares = shamir_split(SECRET, threshold=3, shares=5)
        recovered = shamir_combine(shares[:2])
        self.assertNotEqual(recovered, SECRET)
        self.assertEqual(len(recovered), len(SECRET))

    def test_a_single_share_carries_no_plaintext(self) -> None:
        shares = shamir_split(SECRET, threshold=3, shares=5)
        for share in shares:
            self.assertNotIn(SECRET[:8], share.payload)

    def test_splitting_twice_gives_different_shares(self) -> None:
        # A deterministic polynomial would let anyone who knows the generator
        # recover the secret from one share.
        first = shamir_split(SECRET, threshold=3, shares=5)
        second = shamir_split(SECRET, threshold=3, shares=5)
        self.assertNotEqual(first[0].payload, second[0].payload)
        self.assertEqual(shamir_combine(second[:3]), SECRET)

    def test_a_two_of_two_split_works(self) -> None:
        shares = shamir_split(SECRET, threshold=2, shares=2)
        self.assertEqual(shamir_combine(shares), SECRET)

    def test_a_secret_with_zero_bytes_survives(self) -> None:
        awkward = b"\x00\xff\x00\x01\x00"
        shares = shamir_split(awkward, threshold=2, shares=3)
        self.assertEqual(shamir_combine(shares[:2]), awkward)

    def test_a_long_secret_survives(self) -> None:
        long_secret = bytes(range(256)) * 4
        shares = shamir_split(long_secret, threshold=4, shares=6)
        self.assertEqual(shamir_combine(shares[:4]), long_secret)

    def test_an_empty_secret_is_refused(self) -> None:
        with self.assertRaises(ShareError):
            shamir_split(b"", threshold=2, shares=3)

    def test_a_threshold_of_one_is_refused(self) -> None:
        # k=1 means any node reconstructs alone, which is not sharing.
        with self.assertRaises(ShareError):
            shamir_split(SECRET, threshold=1, shares=3)

    def test_a_threshold_above_the_share_count_is_refused(self) -> None:
        with self.assertRaises(ShareError):
            shamir_split(SECRET, threshold=4, shares=3)

    def test_too_many_shares_are_refused(self) -> None:
        with self.assertRaises(ShareError):
            shamir_split(SECRET, threshold=2, shares=256)

    def test_duplicate_indices_are_refused(self) -> None:
        shares = shamir_split(SECRET, threshold=3, shares=5)
        with self.assertRaises(ShareError):
            shamir_combine([shares[0], shares[0], shares[1]])

    def test_mismatched_lengths_are_refused(self) -> None:
        shares = shamir_split(SECRET, threshold=2, shares=3)
        truncated = Share(index=shares[1].index, payload=shares[1].payload[:-1])
        with self.assertRaises(ShareError):
            shamir_combine([shares[0], truncated])

    def test_one_share_alone_is_refused(self) -> None:
        shares = shamir_split(SECRET, threshold=2, shares=3)
        with self.assertRaises(ShareError):
            shamir_combine(shares[:1])

    def test_share_round_trips_through_json(self) -> None:
        shares = shamir_split(SECRET, threshold=2, shares=3)
        encoded = json.dumps([share.to_dict() for share in shares])
        decoded = [Share.from_dict(raw) for raw in json.loads(encoded)]
        self.assertEqual(shamir_combine(decoded[:2]), SECRET)


def _fleet(count: int = 4) -> tuple[dict[str, Ed25519PrivateKey], CertifierRoster]:
    keys = {f"node-{index}": Ed25519PrivateKey.generate() for index in range(count)}
    roster = CertifierRoster(
        node_keys={node: key.public_key() for node, key in keys.items()}, threshold=2
    )
    return keys, roster


SCOPE = {"bands": ["band-2400"], "passive_only": True}


def _grant(subject: str = "node-0") -> Grant:
    return Grant(
        grant_id="grant-1",
        subject_node_id=subject,
        scope_hash=scope_hash(SCOPE),
        issued_at=NOW - timedelta(hours=12),
        expires_at=NOW + timedelta(minutes=5),
    )


class RosterTests(unittest.TestCase):
    def test_a_quorum_of_one_is_refused(self) -> None:
        keys, _ = _fleet()
        with self.assertRaises(CertificationError):
            CertifierRoster({k: v.public_key() for k, v in keys.items()}, threshold=1)

    def test_a_quorum_larger_than_the_fleet_is_refused(self) -> None:
        keys, _ = _fleet(3)
        with self.assertRaises(CertificationError):
            CertifierRoster({k: v.public_key() for k, v in keys.items()}, threshold=9)


class QuorumRenewalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keys, self.roster = _fleet()
        self.grant = _grant()
        self.new_expiry = NOW + timedelta(hours=2)

    def _attest(self, nodes: list[str], *, expiry: datetime | None = None, at: datetime = NOW):
        return [
            attest(
                self.grant,
                self.keys[node],
                certifier_node_id=node,
                new_expiry=expiry or self.new_expiry,
                now=at,
            )
            for node in nodes
        ]

    def test_a_quorum_extends_the_grant(self) -> None:
        result = QuorumRenewer(self.roster).renew(
            self.grant, self._attest(["node-1", "node-2"]), new_expiry=self.new_expiry, now=NOW
        )
        self.assertTrue(result.granted)
        self.assertEqual(result.grant.expires_at, self.new_expiry)
        self.assertEqual(result.valid_attestations, ("node-1", "node-2"))

    def test_renewal_never_widens_scope(self) -> None:
        result = QuorumRenewer(self.roster).renew(
            self.grant, self._attest(["node-1", "node-2"]), new_expiry=self.new_expiry, now=NOW
        )
        self.assertEqual(result.grant.scope_hash, self.grant.scope_hash)
        self.assertIn("SCOPE_IS_UNCHANGED_A_RENEWAL_CANNOT_WIDEN_AUTHORITY", result.reasons)

    def test_below_quorum_is_refused(self) -> None:
        result = QuorumRenewer(self.roster).renew(
            self.grant, self._attest(["node-1"]), new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn("ONLY_1_OF_2_REQUIRED_CERTIFIERS_ATTESTED", result.reasons)

    def test_a_node_cannot_vouch_for_itself(self) -> None:
        # Self-certification would make the quorum decorative.
        attestations = self._attest(["node-1"]) + [
            attest(
                self.grant,
                self.keys["node-0"],
                certifier_node_id="node-0",
                new_expiry=self.new_expiry,
                now=NOW,
            )
        ]
        result = QuorumRenewer(self.roster).renew(
            self.grant, attestations, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn(("node-0", "SUBJECT_CANNOT_ATTEST_TO_ITS_OWN_GRANT"), result.rejected)

    def test_one_node_signing_twice_is_not_two_peers(self) -> None:
        doubled = self._attest(["node-1"]) * 2
        result = QuorumRenewer(self.roster).renew(
            self.grant, doubled, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn(("node-1", "DUPLICATE_ATTESTATION_FROM_ONE_CERTIFIER"), result.rejected)

    def test_a_node_not_on_the_roster_is_ignored(self) -> None:
        outsider = Ed25519PrivateKey.generate()
        attestations = self._attest(["node-1"]) + [
            attest(
                self.grant,
                outsider,
                certifier_node_id="stranger",
                new_expiry=self.new_expiry,
                now=NOW,
            )
        ]
        result = QuorumRenewer(self.roster).renew(
            self.grant, attestations, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn(
            ("stranger", "CERTIFIER_IS_NOT_ON_THE_OPERATOR_SIGNED_ROSTER"), result.rejected
        )

    def test_a_roster_node_using_the_wrong_key_is_refused(self) -> None:
        impostor = Ed25519PrivateKey.generate()
        attestations = self._attest(["node-1"]) + [
            attest(
                self.grant,
                impostor,
                certifier_node_id="node-2",
                new_expiry=self.new_expiry,
                now=NOW,
            )
        ]
        result = QuorumRenewer(self.roster).renew(
            self.grant, attestations, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn(("node-2", "CERTIFIER_KEY_ID_DOES_NOT_MATCH_THE_ROSTER"), result.rejected)

    def test_an_attestation_for_a_different_expiry_does_not_verify(self) -> None:
        # Signatures are bound to the exact statement, so one collected for a
        # modest extension cannot be replayed for a longer one.
        collected = self._attest(["node-1", "node-2"], expiry=NOW + timedelta(minutes=30))
        result = QuorumRenewer(self.roster).renew(
            self.grant, collected, new_expiry=NOW + timedelta(hours=5), now=NOW
        )
        self.assertFalse(result.granted)
        self.assertTrue(
            all(reason == "SIGNATURE_DOES_NOT_VERIFY_FOR_THIS_STATEMENT" for _, reason in result.rejected)
        )

    def test_an_attestation_bound_to_another_scope_does_not_verify(self) -> None:
        wider = Grant(
            grant_id=self.grant.grant_id,
            subject_node_id=self.grant.subject_node_id,
            scope_hash=scope_hash({"bands": ["band-2400", "band-5800"], "passive_only": True}),
            issued_at=self.grant.issued_at,
            expires_at=self.grant.expires_at,
        )
        result = QuorumRenewer(self.roster).renew(
            wider, self._attest(["node-1", "node-2"]), new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)

    def test_stale_attestations_do_not_count(self) -> None:
        # Otherwise an adversary banks a quorum, then captures the nodes.
        old = self._attest(["node-1", "node-2"], at=NOW - timedelta(hours=3))
        result = QuorumRenewer(self.roster).renew(
            self.grant, old, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn(
            ("node-1", "ATTESTATION_IS_TOO_OLD_TO_SHOW_CURRENT_REACHABILITY"), result.rejected
        )

    def test_a_future_dated_attestation_is_refused(self) -> None:
        ahead = self._attest(["node-1", "node-2"], at=NOW + timedelta(hours=3))
        result = QuorumRenewer(self.roster).renew(
            self.grant, ahead, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)

    def test_revocation_cannot_be_undone_by_a_partition(self) -> None:
        renewer = QuorumRenewer(self.roster, revoked=frozenset({"node-0"}))
        result = renewer.renew(
            self.grant, self._attest(["node-1", "node-2"]), new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn("GRANT_OR_SUBJECT_IS_REVOKED_AND_CANNOT_BE_RENEWED", result.reasons)

    def test_a_revoked_certifier_does_not_count_toward_quorum(self) -> None:
        renewer = QuorumRenewer(self.roster, revoked=frozenset({"node-2"}))
        result = renewer.renew(
            self.grant, self._attest(["node-1", "node-2"]), new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn(("node-2", "CERTIFIER_IS_REVOKED"), result.rejected)

    def test_an_over_long_extension_is_refused(self) -> None:
        far = NOW + timedelta(days=30)
        result = QuorumRenewer(self.roster).renew(
            self.grant, self._attest(["node-1", "node-2"], expiry=far), new_expiry=far, now=NOW
        )
        self.assertFalse(result.granted)
        self.assertIn("REQUESTED_EXTENSION_EXCEEDS_THE_PER_RENEWAL_BOUND", result.reasons)

    def test_a_renewal_that_shortens_the_grant_is_refused(self) -> None:
        earlier = NOW + timedelta(minutes=1)
        result = QuorumRenewer(self.roster).renew(
            self.grant,
            self._attest(["node-1", "node-2"], expiry=earlier),
            new_expiry=earlier,
            now=NOW,
        )
        self.assertFalse(result.granted)
        self.assertIn("RENEWAL_MUST_EXTEND_THE_EXPIRY", result.reasons)

    def test_a_partition_eventually_runs_out_of_self_renewal(self) -> None:
        # An isolated fleet must wind down rather than run forever on its own
        # authority. Renew repeatedly until the total bound bites.
        renewer = QuorumRenewer(self.roster)
        grant = self.grant
        clock = NOW
        granted_rounds = 0
        for _ in range(10):
            target = grant.expires_at + timedelta(hours=5)
            attestations = [
                attest(
                    grant,
                    self.keys[node],
                    certifier_node_id=node,
                    new_expiry=target,
                    now=clock,
                )
                for node in ("node-1", "node-2")
            ]
            result = renewer.renew(grant, attestations, new_expiry=target, now=clock)
            if not result.granted:
                self.assertIn(
                    "PARTITION_HAS_EXHAUSTED_ITS_TOTAL_EXTENSION_AND_MUST_REACH_THE_OPERATOR",
                    result.reasons,
                )
                break
            granted_rounds += 1
            grant = result.grant
            clock = grant.expires_at - timedelta(minutes=5)
        else:
            self.fail("renewal never exhausted its total extension budget")
        self.assertGreater(granted_rounds, 0)
        self.assertLessEqual(grant.total_extension, timedelta(hours=24))

    def test_the_approaching_wind_down_is_announced_before_it_bites(self) -> None:
        renewer = QuorumRenewer(
            self.roster,
            max_extension=timedelta(hours=6),
            max_total_extension=timedelta(hours=7),
        )
        result = renewer.renew(
            self.grant,
            self._attest(["node-1", "node-2"], expiry=NOW + timedelta(hours=5)),
            new_expiry=NOW + timedelta(hours=5),
            now=NOW,
        )
        self.assertTrue(result.granted)
        self.assertIn(
            "THIS_PARTITION_IS_NEARLY_OUT_OF_SELF_RENEWAL_AND_WILL_FAIL_CLOSED", result.reasons
        )

    def test_a_forged_signature_is_refused(self) -> None:
        real = self._attest(["node-1", "node-2"])
        forged = [
            real[0],
            Attestation(
                certifier_node_id="node-2",
                signed_at=NOW,
                signature=real[0].signature,
                key_id=real[1].key_id,
            ),
        ]
        result = QuorumRenewer(self.roster).renew(
            self.grant, forged, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)

    def test_a_malformed_signature_is_refused_not_crashed_on(self) -> None:
        real = self._attest(["node-1", "node-2"])
        broken = [
            real[0],
            Attestation(
                certifier_node_id="node-2",
                signed_at=NOW,
                signature="not base64!!",
                key_id=real[1].key_id,
            ),
        ]
        result = QuorumRenewer(self.roster).renew(
            self.grant, broken, new_expiry=self.new_expiry, now=NOW
        )
        self.assertFalse(result.granted)

    def test_result_serializes(self) -> None:
        result = QuorumRenewer(self.roster).renew(
            self.grant, self._attest(["node-1", "node-2"]), new_expiry=self.new_expiry, now=NOW
        )
        json.dumps(result.to_dict())
        json.dumps(self._attest(["node-1"])[0].to_dict())


class ScopeHashTests(unittest.TestCase):
    def test_key_order_does_not_change_the_hash(self) -> None:
        self.assertEqual(
            scope_hash({"a": 1, "b": 2}),
            scope_hash({"b": 2, "a": 1}),
        )

    def test_a_wider_scope_hashes_differently(self) -> None:
        self.assertNotEqual(scope_hash(SCOPE), scope_hash({**SCOPE, "bands": ["band-2400", "x"]}))


if __name__ == "__main__":
    unittest.main()
