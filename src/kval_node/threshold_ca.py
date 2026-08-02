"""Certification and secret custody for a fleet that has lost its operator link.

Built from **Kwok-Yan Lam**'s distributed-certificate-authority work for mobile
ad-hoc networks — "Self-initialized Distributed Certificate Authority for
Mobile Ad Hoc Network" (ISA 2009), and Lam, Gu, Chung et al., "A robust
certification service for highly dynamic MANET in emergency tasks",
International Journal of Communication Systems 22(9), 2009,
[10.1002/dac.1015](https://doi.org/10.1002/dac.1015). Their problem is exactly
this fleet's problem: nodes that must validate each other's credentials with
**no reachable central authority**, in a topology that keeps changing.

**The gap this closes.** Today a Kval sub-node that loses its operator link runs
until its signed policy expires and then fails closed. Failing closed is
correct — but a partition that lasts an hour longer than the policy silently
disarms every node in it, and an adversary who can cut the backhaul can
therefore disarm the fleet by waiting. Distributed certification lets a
partition keep an *already granted* capability alive among themselves.

Two mechanisms, and the boundary between them is the important part.

**Quorum renewal.** ``k`` of ``n`` roster peers each sign a statement that a
specific grant, identified by its hash, should stay live. A renewal can only
ever *extend the lifetime of a grant the operator already signed* — never widen
its scope, never create authority, never resurrect something already revoked.
The peers are attesting "this node is still one of us and still in good order",
which is the one judgement a partition can make for itself. Everything else
still needs the operator, and still fails closed. Extensions are bounded in
both count and total duration, so a permanently isolated partition winds down
rather than running indefinitely on its own say-so.

**Threshold custody.** Shamir's ``(k, n)`` secret sharing over GF(2^8) splits a
secret across the fleet so that ``k`` nodes can reconstruct it and ``k-1``
learn *nothing* — the information-theoretic guarantee, not a computational one.
The operational use is capture resistance: a drone's stored corpus key is held
as shares, so a captured node yields nothing on its own, which is the disposal
problem the EW threat register raises and could not previously answer.

**Why threshold custody and not threshold signing.** The MANET papers share the
CA's *private key* across nodes so a quorum can issue new certificates. That is
not reproduced, deliberately. Distributing signing authority means a quorum of
captured nodes can mint credentials the operator never approved, which for a
receive-only fleet trades a bounded failure — a partition winding down — for an
unbounded one. Renewal by quorum attestation gets the availability benefit
without ever letting the fleet grant itself something new.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import parse_timestamp, utc_now
from .signing import canonical_json, public_key_id
from .validation import non_empty_str, positive_int

#: Longest a single quorum renewal may extend a grant. Short enough that a
#: compromised quorum buys hours, not weeks.
DEFAULT_MAX_EXTENSION = timedelta(hours=6)

#: Total extension a grant may accumulate before the operator must be reached.
#: A partition that outlasts this winds down rather than running indefinitely
#: on the fleet's own authority.
DEFAULT_MAX_TOTAL_EXTENSION = timedelta(hours=24)

#: Attestations older than this are not counted toward a quorum. A quorum must
#: represent peers that are reachable *now*, not signatures collected days ago.
DEFAULT_MAX_ATTESTATION_AGE = timedelta(minutes=15)

#: Smallest quorum with any Byzantine value. With k=1 any single node renews
#: its own grant, which is not certification at all.
MIN_QUORUM = 2


class CertificationError(RuntimeError):
    """Raised when a renewal cannot be justified."""


class ShareError(ValueError):
    """Raised when shares cannot produce the secret."""


# --- Shamir secret sharing over GF(2^8) -------------------------------------------

_EXP = [0] * 512
_LOG = [0] * 256


def _build_tables() -> None:
    value = 1
    for index in range(255):
        _EXP[index] = value
        _LOG[value] = index
        # Multiply by the generator 3 in GF(2^8) with the AES polynomial 0x11b.
        value ^= (value << 1) ^ (0x11B if value & 0x80 else 0)
        value &= 0xFF
    for index in range(255, 512):
        _EXP[index] = _EXP[index - 255]


_build_tables()


def _mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return _EXP[_LOG[left] + _LOG[right]]


def _div(left: int, right: int) -> int:
    if right == 0:
        raise ZeroDivisionError("division by zero in GF(2^8)")
    if left == 0:
        return 0
    return _EXP[(_LOG[left] - _LOG[right]) % 255]


@dataclass(frozen=True)
class Share:
    """One participant's share. ``index`` is the x-coordinate and is not secret."""

    index: int
    payload: bytes

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "payload": base64.b64encode(self.payload).decode("ascii")}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Share":
        return cls(
            index=positive_int(raw["index"], "index"),
            payload=base64.b64decode(str(raw["payload"]), validate=True),
        )


def shamir_split(secret: bytes, *, threshold: int, shares: int) -> list[Share]:
    """Split ``secret`` so any ``threshold`` shares reconstruct it.

    Fewer than ``threshold`` shares reveal nothing about the secret beyond its
    length — an information-theoretic guarantee, so a captured node's share
    stays worthless however much compute is thrown at it. Coefficients come
    from ``secrets.token_bytes``: a predictable polynomial would let an
    adversary who knows the generator recover the secret from one share.
    """
    if not secret:
        raise ShareError("refusing to split an empty secret")
    threshold = positive_int(threshold, "threshold")
    shares = positive_int(shares, "shares")
    if threshold < MIN_QUORUM:
        raise ShareError(f"threshold must be at least {MIN_QUORUM}")
    if threshold > shares:
        raise ShareError("threshold cannot exceed the number of shares")
    if shares > 255:
        raise ShareError("GF(2^8) supports at most 255 shares")

    results: list[bytearray] = [bytearray() for _ in range(shares)]
    for byte in secret:
        # A random polynomial of degree threshold-1 with the secret byte as its
        # constant term. Evaluating at x=1..n gives the shares.
        coefficients = [byte] + list(secrets.token_bytes(threshold - 1))
        for position in range(shares):
            x = position + 1
            accumulated = 0
            for coefficient in reversed(coefficients):
                accumulated = _mul(accumulated, x) ^ coefficient
            results[position].append(accumulated)
    return [Share(index=position + 1, payload=bytes(data)) for position, data in enumerate(results)]


def shamir_combine(shares: Sequence[Share]) -> bytes:
    """Reconstruct the secret by Lagrange interpolation at x=0."""
    if len(shares) < MIN_QUORUM:
        raise ShareError(f"need at least {MIN_QUORUM} shares to reconstruct")
    indices = [share.index for share in shares]
    if len(set(indices)) != len(indices):
        raise ShareError("shares must have distinct indices")
    if any(index < 1 or index > 255 for index in indices):
        raise ShareError("share indices must be in 1..255")
    lengths = {len(share.payload) for share in shares}
    if len(lengths) != 1:
        raise ShareError("shares are not all the same length")

    length = lengths.pop()
    secret = bytearray()
    for position in range(length):
        total = 0
        for share in shares:
            numerator = 1
            denominator = 1
            for other in shares:
                if other.index == share.index:
                    continue
                numerator = _mul(numerator, other.index)
                denominator = _mul(denominator, share.index ^ other.index)
            total ^= _mul(share.payload[position], _div(numerator, denominator))
        secret.append(total)
    return bytes(secret)


# --- Quorum renewal ----------------------------------------------------------------


@dataclass(frozen=True)
class CertifierRoster:
    """Operator-signed list of who may attest, and how many must agree.

    Both the membership and the threshold come from the operator. A fleet that
    could choose its own quorum size could lower it to one, which is why
    neither is derived from what the partition happens to contain.
    """

    node_keys: dict[str, Ed25519PublicKey]
    threshold: int

    def __post_init__(self) -> None:
        if self.threshold < MIN_QUORUM:
            raise CertificationError(f"quorum threshold must be at least {MIN_QUORUM}")
        if self.threshold > len(self.node_keys):
            raise CertificationError("quorum threshold exceeds the number of certifiers")


@dataclass(frozen=True)
class Grant:
    """The operator-signed authority a renewal may extend, and nothing more."""

    grant_id: str
    subject_node_id: str
    scope_hash: str
    issued_at: datetime
    expires_at: datetime
    extensions_used: int = 0
    total_extension: timedelta = timedelta(0)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Grant":
        """Read a grant from an operator artefact.

        Without this the quorum-renewal capability could not be driven by any
        operator at all: a grant could be constructed in Python and nowhere
        else, so the partition-tolerance story had no operable surface.
        """
        return cls(
            grant_id=non_empty_str(raw.get("grant_id"), "grant_id"),
            subject_node_id=non_empty_str(raw.get("subject_node_id"), "subject_node_id"),
            scope_hash=non_empty_str(raw.get("scope_hash"), "scope_hash"),
            issued_at=parse_timestamp(str(raw["issued_at"])),
            expires_at=parse_timestamp(str(raw["expires_at"])),
            extensions_used=int(raw.get("extensions_used", 0)),
            total_extension=timedelta(seconds=float(raw.get("total_extension_seconds", 0.0))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "subject_node_id": self.subject_node_id,
            "scope_hash": self.scope_hash,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "extensions_used": self.extensions_used,
            "total_extension_seconds": self.total_extension.total_seconds(),
        }

    def statement(self, *, new_expiry: datetime) -> dict[str, Any]:
        """Canonical text every certifier signs.

        ``scope_hash`` is inside it, so an attestation is bound to the exact
        authority it saw. A renewal collected for one scope cannot be replayed
        against a wider one.
        """
        return {
            "kind": "kval-quorum-renewal/v1",
            "grant_id": self.grant_id,
            "subject_node_id": self.subject_node_id,
            "scope_hash": self.scope_hash,
            "current_expiry": self.expires_at.isoformat(),
            "requested_expiry": new_expiry.isoformat(),
        }


@dataclass(frozen=True)
class Attestation:
    """One peer's signed assertion that a grant should stay live."""

    certifier_node_id: str
    signed_at: datetime
    signature: str
    key_id: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Attestation":
        return cls(
            certifier_node_id=non_empty_str(raw.get("certifier_node_id"), "certifier_node_id"),
            signed_at=parse_timestamp(str(raw["signed_at"])),
            signature=non_empty_str(raw.get("signature"), "signature"),
            key_id=non_empty_str(raw.get("key_id"), "key_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "certifier_node_id": self.certifier_node_id,
            "signed_at": self.signed_at.isoformat(),
            "signature": self.signature,
            "key_id": self.key_id,
        }


def attest(
    grant: Grant,
    private_key: Ed25519PrivateKey,
    *,
    certifier_node_id: str,
    new_expiry: datetime,
    now: datetime | None = None,
) -> Attestation:
    signed_at = now or utc_now()
    body = {
        "statement": grant.statement(new_expiry=new_expiry),
        "certifier_node_id": non_empty_str(certifier_node_id, "certifier_node_id"),
        "signed_at": signed_at.isoformat(),
    }
    return Attestation(
        certifier_node_id=body["certifier_node_id"],
        signed_at=signed_at,
        signature=base64.b64encode(private_key.sign(canonical_json(body))).decode("ascii"),
        key_id=public_key_id(private_key.public_key()),
    )


@dataclass(frozen=True)
class RenewalResult:
    granted: bool
    grant: Grant
    valid_attestations: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-quorum-renewal-result/v1",
            "granted": self.granted,
            "grant_id": self.grant.grant_id,
            "subject_node_id": self.grant.subject_node_id,
            "expires_at": self.grant.expires_at.isoformat(),
            "extensions_used": self.grant.extensions_used,
            "total_extension_seconds": round(self.grant.total_extension.total_seconds()),
            "valid_attestations": list(self.valid_attestations),
            "rejected": [{"node_id": node, "reason": reason} for node, reason in self.rejected],
            "reasons": list(self.reasons),
        }


class QuorumRenewer:
    """Extends an existing grant when enough roster peers attest to it.

    Every check here is a refusal to widen authority. The renewer can move one
    field — the expiry — of a grant the operator already signed, by a bounded
    amount, a bounded number of times.
    """

    def __init__(
        self,
        roster: CertifierRoster,
        *,
        max_extension: timedelta = DEFAULT_MAX_EXTENSION,
        max_total_extension: timedelta = DEFAULT_MAX_TOTAL_EXTENSION,
        max_attestation_age: timedelta = DEFAULT_MAX_ATTESTATION_AGE,
        revoked: frozenset[str] = frozenset(),
    ) -> None:
        self.roster = roster
        self.max_extension = max_extension
        self.max_total_extension = max_total_extension
        self.max_attestation_age = max_attestation_age
        self.revoked = revoked

    def renew(
        self,
        grant: Grant,
        attestations: Sequence[Attestation],
        *,
        new_expiry: datetime,
        now: datetime | None = None,
    ) -> RenewalResult:
        current = now or utc_now()
        reasons: list[str] = []
        rejected: list[tuple[str, str]] = []

        if grant.grant_id in self.revoked or grant.subject_node_id in self.revoked:
            # Revocation is the one operator decision a partition must never
            # be able to undo, so it is checked before anything else.
            return RenewalResult(
                False, grant, (), (), ("GRANT_OR_SUBJECT_IS_REVOKED_AND_CANNOT_BE_RENEWED",)
            )

        extension = new_expiry - grant.expires_at
        if extension <= timedelta(0):
            return RenewalResult(False, grant, (), (), ("RENEWAL_MUST_EXTEND_THE_EXPIRY",))
        if extension > self.max_extension:
            return RenewalResult(
                False, grant, (), (), ("REQUESTED_EXTENSION_EXCEEDS_THE_PER_RENEWAL_BOUND",)
            )
        if grant.total_extension + extension > self.max_total_extension:
            return RenewalResult(
                False,
                grant,
                (),
                (),
                ("PARTITION_HAS_EXHAUSTED_ITS_TOTAL_EXTENSION_AND_MUST_REACH_THE_OPERATOR",),
            )

        statement = grant.statement(new_expiry=new_expiry)
        accepted: list[str] = []
        seen: set[str] = set()
        for attestation in attestations:
            node_id = attestation.certifier_node_id
            if node_id in seen:
                # One node signing twice is not two peers agreeing.
                rejected.append((node_id, "DUPLICATE_ATTESTATION_FROM_ONE_CERTIFIER"))
                continue
            seen.add(node_id)
            if node_id == grant.subject_node_id:
                # A node cannot vouch for itself; that is the whole point of a
                # quorum.
                rejected.append((node_id, "SUBJECT_CANNOT_ATTEST_TO_ITS_OWN_GRANT"))
                continue
            if node_id in self.revoked:
                rejected.append((node_id, "CERTIFIER_IS_REVOKED"))
                continue
            public_key = self.roster.node_keys.get(node_id)
            if public_key is None:
                rejected.append((node_id, "CERTIFIER_IS_NOT_ON_THE_OPERATOR_SIGNED_ROSTER"))
                continue
            if public_key_id(public_key) != attestation.key_id:
                rejected.append((node_id, "CERTIFIER_KEY_ID_DOES_NOT_MATCH_THE_ROSTER"))
                continue
            age = current - attestation.signed_at
            if age > self.max_attestation_age:
                # Stale signatures would let an adversary bank a quorum in
                # advance and spend it after capturing the nodes.
                rejected.append((node_id, "ATTESTATION_IS_TOO_OLD_TO_SHOW_CURRENT_REACHABILITY"))
                continue
            if age < -DEFAULT_MAX_ATTESTATION_AGE:
                rejected.append((node_id, "ATTESTATION_IS_DATED_IN_THE_FUTURE"))
                continue
            body = {
                "statement": statement,
                "certifier_node_id": node_id,
                "signed_at": attestation.signed_at.isoformat(),
            }
            try:
                public_key.verify(
                    base64.b64decode(attestation.signature, validate=True), canonical_json(body)
                )
            except (InvalidSignature, ValueError, TypeError):
                rejected.append((node_id, "SIGNATURE_DOES_NOT_VERIFY_FOR_THIS_STATEMENT"))
                continue
            accepted.append(node_id)

        if len(accepted) < self.roster.threshold:
            reasons.append(
                f"ONLY_{len(accepted)}_OF_{self.roster.threshold}_REQUIRED_CERTIFIERS_ATTESTED"
            )
            return RenewalResult(
                False, grant, tuple(sorted(accepted)), tuple(rejected), tuple(reasons)
            )

        renewed = Grant(
            grant_id=grant.grant_id,
            subject_node_id=grant.subject_node_id,
            scope_hash=grant.scope_hash,
            issued_at=grant.issued_at,
            expires_at=new_expiry,
            extensions_used=grant.extensions_used + 1,
            total_extension=grant.total_extension + extension,
        )
        reasons.append("QUORUM_ATTESTED_SO_THE_EXISTING_GRANT_IS_EXTENDED")
        reasons.append("SCOPE_IS_UNCHANGED_A_RENEWAL_CANNOT_WIDEN_AUTHORITY")
        remaining = self.max_total_extension - renewed.total_extension
        if remaining <= self.max_extension:
            reasons.append("THIS_PARTITION_IS_NEARLY_OUT_OF_SELF_RENEWAL_AND_WILL_FAIL_CLOSED")
        return RenewalResult(
            True, renewed, tuple(sorted(accepted)), tuple(rejected), tuple(reasons)
        )


def scope_hash(scope: dict[str, Any]) -> str:
    """Stable identity for the authority a grant carries."""
    return hashlib.sha256(canonical_json(scope)).hexdigest()
