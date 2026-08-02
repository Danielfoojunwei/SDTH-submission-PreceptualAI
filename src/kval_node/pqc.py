"""Quantum-resistant signing for the policy trust root.

**Why this exists.** Every trust decision this node makes rests on an Ed25519
signature — the catalogue, the spectrum and power policies, capability leases,
peer reports, federated contributions, audit records. Ed25519 is **not in
CNSA 2.0**, the NSA's algorithm suite for National Security Systems, and
neither is any other classical asymmetric algorithm: RSA, ECDH and ECDSA are
all retired by it.

The dates are the point. NSA's transition table puts **software and firmware
signing first** — "support and prefer" from **2025**, exclusive use by
**2030** — because a verification algorithm baked into fielded hardware is
locked in for that hardware's operational life. And **from 1 January 2027 every
new National Security System acquisition must support CNSA 2.0** or carry a
documented exception. A signed-policy trust root is firmware signing in
everything but name: it decides what the node is permitted to do.

* NSA, *Announcing the Commercial National Security Algorithm Suite 2.0*,
  CNSA 2.0 (see ``docs/DEFENCE_STANDARDS.md`` for the full citation set).
* NIST SP 800-208, *Recommendation for Stateful Hash-Based Signature Schemes*.
* Leighton-Micali Hash-Based Signatures, RFC 8554.

**Why LMS rather than ML-DSA.** CNSA 2.0 approves both for this purpose, and
NSA names LMS as the near-term recommendation for firmware precisely because
its security rests on nothing but a hash function — no lattice assumption, no
new mathematics, and no dependency. ML-DSA needs a vetted implementation this
repository has no way to supply without taking on a runtime dependency it has
deliberately avoided (``dependencies = []``). LMS needs SHA-256 and care.

**The care is the hard part, and it is not optional.** LMS is *stateful*. Each
leaf of the tree is a one-time signature: sign two different messages with the
same leaf and an attacker recovers that leaf's private key and forges freely
(RFC 8554 §9.1, SP 800-208 §6). This module therefore treats the signature
index as the asset:

* state is written to durable storage and ``fsync``ed **before** a signature is
  produced, never after — a crash then wastes an index, which costs nothing,
  instead of reusing one, which is fatal;
* an index that has been used is never issued again, and a state file that goes
  *backwards* is refused rather than trusted, because a restored backup is the
  ordinary way key reuse happens in practice;
* exhaustion is a hard stop, not a wrap-around.

**What this does not achieve, stated plainly.** SP 800-208 §5 requires key
generation and state management inside a FIPS 140 Level 3 hardware module that
cannot export private key material. This implementation keeps state in a file.
That is materially weaker: an attacker with filesystem access and the ability
to roll the state back can still induce reuse, and this module can only detect
rollback it can see. Using it satisfies the *algorithm* requirement of
CNSA 2.0 and not the *implementation profile* of SP 800-208. An operator
deploying against a real accreditation needs an HSM, and the signing interface
here is shaped so that swapping the state store for one is a contained change.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- RFC 8554 constants ------------------------------------------------------------

#: Domain separators. These keep a leaf hash from ever being confused with an
#: interior node or a message hash, which is what stops a tree-shape attack.
D_PBLC = 0x8080
D_MESG = 0x8181
D_LEAF = 0x8282
D_INTR = 0x8383

#: LM-OTS type codes (RFC 8554 §4.1). Only the SHA-256/n=32 family is offered:
#: CNSA 2.0 requires SHA-384 or better for general hashing, but the LMS
#: construction's security rests on second-preimage resistance of the tree
#: hash, for which SHA-256 at n=32 is the parameter set NIST approves here.
LMOTS_SHA256_N32_W1 = 0x00000001
LMOTS_SHA256_N32_W2 = 0x00000002
LMOTS_SHA256_N32_W4 = 0x00000003
LMOTS_SHA256_N32_W8 = 0x00000004

#: (n, w, p, ls) per RFC 8554 §4.1.
_LMOTS_PARAMS: dict[int, tuple[int, int, int, int]] = {
    LMOTS_SHA256_N32_W1: (32, 1, 265, 7),
    LMOTS_SHA256_N32_W2: (32, 2, 133, 6),
    LMOTS_SHA256_N32_W4: (32, 4, 67, 4),
    LMOTS_SHA256_N32_W8: (32, 8, 34, 0),
}

#: LMS type codes (RFC 8554 §5.1), giving tree height h and hash length m.
LMS_SHA256_M32_H5 = 0x00000005
LMS_SHA256_M32_H10 = 0x00000006
LMS_SHA256_M32_H15 = 0x00000007
LMS_SHA256_M32_H20 = 0x00000008
LMS_SHA256_M32_H25 = 0x00000009

_LMS_PARAMS: dict[int, tuple[int, int]] = {
    LMS_SHA256_M32_H5: (32, 5),
    LMS_SHA256_M32_H10: (32, 10),
    LMS_SHA256_M32_H15: (32, 15),
    LMS_SHA256_M32_H20: (32, 20),
    LMS_SHA256_M32_H25: (32, 25),
}

#: Height 10 gives 1,024 signatures. Policy documents are signed rarely — a
#: catalogue revision, a policy change, a lease — so this is years of headroom
#: while keeping key generation fast enough to run on the node.
#:
#: The choice is **irrevocable**: a tree's height is fixed at generation, and
#: running out means generating and distributing a new public key. An operator
#: signing far more often should generate a taller tree up front rather than
#: discover the ceiling in the field.
DEFAULT_LMS_TYPE = LMS_SHA256_M32_H10
DEFAULT_LMOTS_TYPE = LMOTS_SHA256_N32_W8

#: Algorithm label carried in signed envelopes.
LMS_ALGORITHM = "LMS_SHA256_M32"


class PqcError(RuntimeError):
    """Raised when a signature cannot be produced or trusted."""


class StateExhausted(PqcError):
    """Raised when every one-time key in the tree has been spent."""


class StateRollback(PqcError):
    """Raised when stored state moved backwards — the reuse precondition."""


# --- primitives --------------------------------------------------------------------


def u32(value: int) -> bytes:
    return value.to_bytes(4, "big")


def u16(value: int) -> bytes:
    return value.to_bytes(2, "big")


def u8(value: int) -> bytes:
    return value.to_bytes(1, "big")


def _H(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def coef(value: bytes, index: int, w: int) -> int:
    """The ``index``-th ``w``-bit field of ``value`` (RFC 8554 §3.1.3)."""
    per_byte = 8 // w
    byte = value[(index * w) // 8]
    shift = 8 - (w * (index % per_byte) + w)
    return ((1 << w) - 1) & (byte >> shift)


def checksum(value: bytes, *, w: int, ls: int, n: int) -> int:
    """RFC 8554 §4.4.

    The checksum is what makes the Winternitz chain one-time-safe: forging a
    message whose fields are all *higher* up the chains forces the checksum
    fields *lower*, and those cannot be walked backwards through a hash.
    """
    total = 0
    for index in range((n * 8) // w):
        total += ((1 << w) - 1) - coef(value, index, w)
    return total << ls


# --- LM-OTS ------------------------------------------------------------------------


def _ots_private_chain(seed: bytes, identifier: bytes, leaf: int, index: int) -> bytes:
    """Pseudorandom LM-OTS chain start (RFC 8554 Appendix A).

    Deriving every chain from one seed means the private key is a single secret
    rather than thousands, which is what makes tree generation and state
    handling tractable. The derivation is domain-separated by leaf and chain
    index so no two chains ever start from the same value.
    """
    return _H(identifier + u32(leaf) + u16(index) + u8(0xFF) + seed)


def _ots_public_key(seed: bytes, identifier: bytes, leaf: int, ots_type: int) -> bytes:
    n, w, p, _ls = _LMOTS_PARAMS[ots_type]
    end = (1 << w) - 1
    accumulator = b""
    for index in range(p):
        value = _ots_private_chain(seed, identifier, leaf, index)
        for step in range(end):
            value = _H(identifier + u32(leaf) + u16(index) + u8(step) + value)
        accumulator += value
    return _H(identifier + u32(leaf) + u16(D_PBLC) + accumulator)


def _ots_sign(
    seed: bytes, identifier: bytes, leaf: int, message: bytes, ots_type: int, randomiser: bytes
) -> bytes:
    n, w, p, ls = _LMOTS_PARAMS[ots_type]
    digest = _H(identifier + u32(leaf) + u16(D_MESG) + randomiser + message)
    expanded = digest + u16(checksum(digest, w=w, ls=ls, n=n))
    signature = u32(ots_type) + randomiser
    for index in range(p):
        steps = coef(expanded, index, w)
        value = _ots_private_chain(seed, identifier, leaf, index)
        for step in range(steps):
            value = _H(identifier + u32(leaf) + u16(index) + u8(step) + value)
        signature += value
    return signature


def _ots_public_key_candidate(
    signature: bytes, identifier: bytes, leaf: int, message: bytes
) -> bytes:
    """Recover the public key a signature implies (RFC 8554 §4.6).

    Verification never compares signatures. It walks each Winternitz chain the
    rest of the way to its end and rebuilds the public key; only a signature
    made with the matching private key rebuilds the right one.
    """
    if len(signature) < 4:
        raise PqcError("LM-OTS signature is truncated")
    ots_type = int.from_bytes(signature[:4], "big")
    params = _LMOTS_PARAMS.get(ots_type)
    if params is None:
        raise PqcError(f"unsupported LM-OTS type 0x{ots_type:08x}")
    n, w, p, ls = params
    if len(signature) != 4 + n + p * n:
        raise PqcError("LM-OTS signature has the wrong length for its type")

    randomiser = signature[4 : 4 + n]
    digest = _H(identifier + u32(leaf) + u16(D_MESG) + randomiser + message)
    expanded = digest + u16(checksum(digest, w=w, ls=ls, n=n))
    end = (1 << w) - 1
    accumulator = b""
    for index in range(p):
        start = coef(expanded, index, w)
        value = signature[4 + n + index * n : 4 + n + (index + 1) * n]
        for step in range(start, end):
            value = _H(identifier + u32(leaf) + u16(index) + u8(step) + value)
        accumulator += value
    return _H(identifier + u32(leaf) + u16(D_PBLC) + accumulator)


# --- LMS tree ----------------------------------------------------------------------


@dataclass(frozen=True)
class LmsPublicKey:
    """What a verifier needs, and nothing that could help a forger."""

    lms_type: int
    ots_type: int
    identifier: bytes
    root: bytes

    def to_bytes(self) -> bytes:
        return u32(self.lms_type) + u32(self.ots_type) + self.identifier + self.root

    @classmethod
    def from_bytes(cls, raw: bytes) -> "LmsPublicKey":
        if len(raw) < 8 + 16:
            raise PqcError("LMS public key is truncated")
        lms_type = int.from_bytes(raw[0:4], "big")
        ots_type = int.from_bytes(raw[4:8], "big")
        if lms_type not in _LMS_PARAMS:
            raise PqcError(f"unsupported LMS type 0x{lms_type:08x}")
        if ots_type not in _LMOTS_PARAMS:
            raise PqcError(f"unsupported LM-OTS type 0x{ots_type:08x}")
        m, _h = _LMS_PARAMS[lms_type]
        if len(raw) != 8 + 16 + m:
            raise PqcError("LMS public key has the wrong length for its type")
        return cls(lms_type, ots_type, raw[8:24], raw[24 : 24 + m])

    def key_id(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()[:16]

    def signatures_available(self) -> int:
        _m, h = _LMS_PARAMS[self.lms_type]
        return 1 << h


def _build_tree(
    seed: bytes, identifier: bytes, lms_type: int, ots_type: int
) -> list[bytes]:
    """All ``2^(h+1)`` tree nodes, one-indexed as RFC 8554 numbers them.

    Held in memory because the authentication path for any leaf must be
    available at signing time. At the default height this is 2,048 hashes —
    trivial. A height-25 tree would not fit this way and would need the
    standard incremental traversal; that is a deliberate reason the default
    is modest.
    """
    _m, h = _LMS_PARAMS[lms_type]
    leaves = 1 << h
    nodes: list[bytes] = [b""] * (2 * leaves)
    for leaf in range(leaves):
        public = _ots_public_key(seed, identifier, leaf, ots_type)
        nodes[leaves + leaf] = _H(
            identifier + u32(leaves + leaf) + u16(D_LEAF) + public
        )
    for node in range(leaves - 1, 0, -1):
        nodes[node] = _H(
            identifier + u32(node) + u16(D_INTR) + nodes[2 * node] + nodes[2 * node + 1]
        )
    return nodes


def _authentication_path(nodes: list[bytes], leaf: int, h: int) -> list[bytes]:
    path: list[bytes] = []
    node = (1 << h) + leaf
    while node > 1:
        path.append(nodes[node ^ 1])
        node //= 2
    return path


def lms_verify(public_key: LmsPublicKey, message: bytes, signature: bytes) -> bool:
    """Verify an LMS signature (RFC 8554 §5.4.2). Never raises on bad input."""
    try:
        m, h = _LMS_PARAMS[public_key.lms_type]
        n, _w, p, _ls = _LMOTS_PARAMS[public_key.ots_type]
        ots_length = 4 + n + p * n
        expected = 4 + ots_length + 4 + h * m
        if len(signature) != expected:
            return False
        leaf = int.from_bytes(signature[0:4], "big")
        if leaf >= (1 << h):
            return False
        ots_signature = signature[4 : 4 + ots_length]
        offset = 4 + ots_length
        if int.from_bytes(signature[offset : offset + 4], "big") != public_key.lms_type:
            return False
        offset += 4
        path = [signature[offset + i * m : offset + (i + 1) * m] for i in range(h)]

        candidate = _ots_public_key_candidate(
            ots_signature, public_key.identifier, leaf, message
        )
        node = (1 << h) + leaf
        accumulator = _H(
            public_key.identifier + u32(node) + u16(D_LEAF) + candidate
        )
        for sibling in path:
            if node % 2 == 1:
                accumulator = _H(
                    public_key.identifier + u32(node // 2) + u16(D_INTR) + sibling + accumulator
                )
            else:
                accumulator = _H(
                    public_key.identifier + u32(node // 2) + u16(D_INTR) + accumulator + sibling
                )
            node //= 2
        return secrets.compare_digest(accumulator, public_key.root)
    except (PqcError, IndexError, ValueError):
        return False


# --- stateful signer ---------------------------------------------------------------


class LmsStateStore:
    """Durable custody of the next unused leaf index.

    The whole security of LMS reduces to this file being honest. Two rules are
    enforced and both matter more than they look:

    **Reserve before signing.** The index is advanced and ``fsync``ed to disk
    *before* the signature exists. A crash between the two wastes one index out
    of a thousand. Advancing afterwards would mean a crash leaves the index
    unspent on disk and spent in the world — and the next signature reuses it,
    which hands over the private key.

    **Never go backwards.** A state file whose index is lower than one already
    seen is refused. Restoring a machine from backup is the ordinary,
    blameless way key reuse happens, and it must fail loudly rather than
    silently re-issue spent leaves.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._high_water = self._read().get("next_index", 0)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # Unreadable state cannot be assumed to be index zero; that
            # assumption is exactly what reuses every leaf in the tree.
            raise PqcError(f"LMS state at {self.path} is unreadable: {exc}") from exc

    def next_index(self) -> int:
        return int(self._read().get("next_index", 0))

    def reserve(self, *, key_id: str, capacity: int) -> int:
        stored = self._read()
        if stored and stored.get("key_id") not in (None, key_id):
            raise PqcError("LMS state belongs to a different key")
        index = int(stored.get("next_index", 0))
        if index < self._high_water:
            raise StateRollback(
                f"LMS state went backwards ({index} < {self._high_water}); a restored "
                "backup would reuse one-time keys and expose the private key"
            )
        if index >= capacity:
            raise StateExhausted(
                f"every one of the {capacity} one-time keys in this tree has been used; "
                "generate and distribute a new key rather than wrapping around"
            )
        self._write({"key_id": key_id, "next_index": index + 1, "capacity": capacity})
        self._high_water = index + 1
        return index

    def _write(self, payload: dict[str, Any]) -> None:
        """Atomically replace the state file, durably.

        Written to a temporary file, ``fsync``ed, renamed, then the *directory*
        is ``fsync``ed. Without the directory sync the rename itself can be
        lost on power failure, leaving the old index in place — which is the
        reuse case this whole class exists to prevent.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


@dataclass(frozen=True)
class LmsPrivateKey:
    """Seed, identifier and the precomputed tree.

    The seed is the secret. It is deliberately not serialisable through this
    dataclass: writing a private key to disk is an operator decision made with
    file modes and, ideally, an HSM — not something a ``to_dict`` should make
    convenient.
    """

    seed: bytes
    identifier: bytes
    lms_type: int
    ots_type: int
    nodes: list[bytes]

    def public_key(self) -> LmsPublicKey:
        return LmsPublicKey(self.lms_type, self.ots_type, self.identifier, self.nodes[1])


def generate_lms_key(
    *, lms_type: int = DEFAULT_LMS_TYPE, ots_type: int = DEFAULT_LMOTS_TYPE
) -> LmsPrivateKey:
    """Generate a fresh LMS key. Cost grows with tree height, so choose once."""
    if lms_type not in _LMS_PARAMS:
        raise PqcError(f"unsupported LMS type 0x{lms_type:08x}")
    if ots_type not in _LMOTS_PARAMS:
        raise PqcError(f"unsupported LM-OTS type 0x{ots_type:08x}")
    seed = secrets.token_bytes(32)
    identifier = secrets.token_bytes(16)
    nodes = _build_tree(seed, identifier, lms_type, ots_type)
    return LmsPrivateKey(seed, identifier, lms_type, ots_type, nodes)


class LmsSigner:
    """Signs with a fresh one-time key each time, or refuses."""

    def __init__(self, private_key: LmsPrivateKey, state: LmsStateStore) -> None:
        self.private_key = private_key
        self.state = state

    def remaining(self) -> int:
        capacity = self.private_key.public_key().signatures_available()
        return max(0, capacity - self.state.next_index())

    def sign(self, message: bytes) -> bytes:
        public = self.private_key.public_key()
        _m, h = _LMS_PARAMS[self.private_key.lms_type]
        # Reserve first. If the process dies here the index is spent and
        # unused, which costs one signature out of the tree and nothing else.
        leaf = self.state.reserve(
            key_id=public.key_id(), capacity=public.signatures_available()
        )
        ots_signature = _ots_sign(
            self.private_key.seed,
            self.private_key.identifier,
            leaf,
            message,
            self.private_key.ots_type,
            secrets.token_bytes(32),
        )
        path = _authentication_path(self.private_key.nodes, leaf, h)
        return u32(leaf) + ots_signature + u32(self.private_key.lms_type) + b"".join(path)
