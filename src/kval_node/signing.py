from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from dataclasses import dataclass

from .models import Catalog, Profile, parse_timestamp
from .pqc import LMS_ALGORITHM, LmsPublicKey, LmsSigner, lms_verify


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def public_key_id(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_operator_keypair(private_path: Path, public_path: Path) -> None:
    if private_path.exists() or public_path.exists():
        raise FileExistsError("refusing to overwrite an existing operator key")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    # Create the file 0600 *before* any key bytes are written; writing first and
    # chmod-ing after leaves a window where the private key is world-readable.
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    public_path.write_bytes(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("operator private key is not Ed25519")
    return key


def load_public_key(path: Path) -> Ed25519PublicKey:
    key = serialization.load_pem_public_key(path.read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("catalog public key is not Ed25519")
    return key


def validate_catalog_payload(payload: dict[str, Any], *, now: datetime | None = None) -> tuple[Profile, ...]:
    required = {"catalog_version", "created_at", "expires_at", "profiles"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"catalog payload is missing fields: {', '.join(missing)}")
    created_at = parse_timestamp(str(payload["created_at"]))
    expires_at = parse_timestamp(str(payload["expires_at"]))
    current = now or datetime.now(timezone.utc)
    if expires_at <= created_at:
        raise ValueError("catalog expiry must be after creation")
    if created_at > current:
        raise ValueError("catalog creation time is in the future")
    if expires_at <= current:
        raise ValueError("catalog has expired")
    profiles = tuple(Profile.from_dict(raw) for raw in payload["profiles"])
    if not profiles:
        raise ValueError("catalog must contain at least one profile")
    profile_ids = [profile.profile_id for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("catalog contains duplicate profile IDs")
    connection_uuids = [profile.connection_uuid for profile in profiles]
    if len(connection_uuids) != len(set(connection_uuids)):
        raise ValueError("catalog contains duplicate NetworkManager UUIDs")
    return profiles


def sign_catalog(payload: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    validate_catalog_payload(payload)
    encoded = canonical_json(payload)
    signature = private_key.sign(encoded)
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "payload": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_catalog(
    envelope: dict[str, Any],
    public_key: Ed25519PublicKey,
    *,
    now: datetime | None = None,
) -> Catalog:
    if envelope.get("algorithm") != "Ed25519":
        raise ValueError("unsupported catalog signature algorithm")
    expected_key_id = public_key_id(public_key)
    if envelope.get("key_id") != expected_key_id:
        raise ValueError("catalog key ID does not match the provisioned public key")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("catalog payload must be an object")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise ValueError("catalog signature is missing or malformed") from exc
    encoded = canonical_json(payload)
    try:
        public_key.verify(signature, encoded)
    except InvalidSignature as exc:
        raise ValueError("catalog signature verification failed") from exc
    profiles = validate_catalog_payload(payload, now=now)
    return Catalog(
        version=str(payload["catalog_version"]),
        created_at=parse_timestamp(str(payload["created_at"])),
        expires_at=parse_timestamp(str(payload["expires_at"])),
        profiles=profiles,
        payload_hash=hashlib.sha256(encoded).hexdigest(),
        key_id=expected_key_id,
    )


def load_verified_catalog(catalog_path: Path, public_key_path: Path) -> Catalog:
    envelope = json.loads(catalog_path.read_text(encoding="utf-8"))
    return verify_catalog(envelope, load_public_key(public_key_path))


def sign_document(document_type: str, payload: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Sign an arbitrary policy document (spectrum, power, transport) offline."""
    if not document_type:
        raise ValueError("document_type is required")
    body = {"document_type": document_type, "payload": payload}
    signature = private_key.sign(canonical_json(body))
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "document_type": document_type,
        "payload": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_document(
    envelope: dict[str, Any],
    public_key: Ed25519PublicKey,
    *,
    expected_type: str,
) -> tuple[dict[str, Any], str]:
    """Verify a signed policy document and return ``(payload, payload_hash)``."""
    if envelope.get("algorithm") != "Ed25519":
        raise ValueError("unsupported document signature algorithm")
    if envelope.get("key_id") != public_key_id(public_key):
        raise ValueError("document key ID does not match the provisioned public key")
    if envelope.get("document_type") != expected_type:
        raise ValueError(f"expected a {expected_type} document")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("document payload must be an object")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise ValueError("document signature is missing or malformed") from exc
    body = {"document_type": expected_type, "payload": payload}
    encoded = canonical_json(body)
    try:
        public_key.verify(signature, encoded)
    except InvalidSignature as exc:
        raise ValueError("document signature verification failed") from exc
    return payload, hashlib.sha256(canonical_json(payload)).hexdigest()


def load_verified_document(path: Path, public_key_path: Path, *, expected_type: str) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    return verify_document(envelope, load_public_key(public_key_path), expected_type=expected_type)


# --- algorithm agility and CNSA 2.0 -------------------------------------------------

#: Signature algorithms this node will accept, and whether each is approved for
#: National Security Systems under CNSA 2.0. Ed25519 is not: CNSA 2.0 retires
#: all classical asymmetric cryptography, and NSA puts software and firmware
#: signing first in the transition because a verification algorithm baked into
#: fielded hardware is locked in for that hardware's life.
#: See ``docs/DEFENCE_STANDARDS.md``.
ED25519_ALGORITHM = "Ed25519"

CNSA_2_0_SIGNATURE_ALGORITHMS = frozenset({LMS_ALGORITHM})
SUPPORTED_SIGNATURE_ALGORITHMS = frozenset({ED25519_ALGORITHM, LMS_ALGORITHM})


class AlgorithmRefused(ValueError):
    """Raised when a signature uses an algorithm this node will not accept."""


@dataclass(frozen=True)
class SignaturePolicy:
    """Which signature algorithms this node will trust.

    ``require_cnsa_2_0`` is the switch an operator flips to enforce the NSS
    requirement. It is off by default because turning it on makes every
    Ed25519-signed artefact already in the field unverifiable, which is a
    migration to be planned rather than a flag to be surprised by.
    """

    require_cnsa_2_0: bool = False

    def accepted(self) -> frozenset[str]:
        return CNSA_2_0_SIGNATURE_ALGORITHMS if self.require_cnsa_2_0 else SUPPORTED_SIGNATURE_ALGORITHMS

    def check(self, algorithm: str) -> None:
        if algorithm not in SUPPORTED_SIGNATURE_ALGORITHMS:
            raise AlgorithmRefused(f"unsupported signature algorithm {algorithm!r}")
        if algorithm not in self.accepted():
            raise AlgorithmRefused(
                f"{algorithm} is not approved under CNSA 2.0 and this node is configured "
                "to require CNSA 2.0 signatures"
            )


def sign_document_lms(
    document_type: str, payload: dict[str, Any], signer: LmsSigner
) -> dict[str, Any]:
    """Sign a policy document with a stateful hash-based key.

    The body signed is byte-identical to the Ed25519 path's, so an operator can
    re-sign an existing document under the new algorithm without the payload
    changing meaning.
    """
    if not document_type:
        raise ValueError("document_type is required")
    body = {"document_type": document_type, "payload": payload}
    public_key = signer.private_key.public_key()
    signature = signer.sign(canonical_json(body))
    return {
        "algorithm": LMS_ALGORITHM,
        "key_id": public_key.key_id(),
        "document_type": document_type,
        "payload": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
        "public_key": base64.b64encode(public_key.to_bytes()).decode("ascii"),
    }


def verify_document_agile(
    envelope: dict[str, Any],
    *,
    expected_type: str,
    ed25519_key: Ed25519PublicKey | None = None,
    lms_key: LmsPublicKey | None = None,
    policy: SignaturePolicy | None = None,
) -> tuple[dict[str, Any], str]:
    """Verify a document under whichever approved algorithm signed it.

    The algorithm is read from the envelope but never *trusted* from it: the
    caller supplies the trust anchors, so an attacker cannot downgrade a
    document by relabelling it as an algorithm whose key they control. An
    envelope naming an algorithm this node has no anchor for is refused, not
    skipped.
    """
    active = policy or SignaturePolicy()
    algorithm = str(envelope.get("algorithm", ""))
    active.check(algorithm)

    if algorithm == ED25519_ALGORITHM:
        if ed25519_key is None:
            raise AlgorithmRefused("document is Ed25519-signed but no Ed25519 anchor is provisioned")
        return verify_document(envelope, ed25519_key, expected_type=expected_type)

    if lms_key is None:
        raise AlgorithmRefused("document is LMS-signed but no LMS anchor is provisioned")
    if envelope.get("document_type") != expected_type:
        raise ValueError(f"expected a {expected_type} document")
    if envelope.get("key_id") != lms_key.key_id():
        raise ValueError("document key ID does not match the provisioned LMS public key")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("document payload must be an object")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise ValueError("document signature is missing or malformed") from exc
    body = {"document_type": expected_type, "payload": payload}
    if not lms_verify(lms_key, canonical_json(body), signature):
        raise ValueError("document signature verification failed")
    return payload, hashlib.sha256(canonical_json(payload)).hexdigest()


def load_lms_public_key(path: Path) -> LmsPublicKey:
    """Read a provisioned LMS trust anchor."""
    return LmsPublicKey.from_bytes(base64.b64decode(path.read_text(encoding="utf-8").strip(), validate=True))
