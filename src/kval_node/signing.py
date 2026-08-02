from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import Catalog, Profile, parse_timestamp


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
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    private_path.chmod(0o600)
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

