from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .models import TelemetrySnapshot, parse_timestamp
from .signing import canonical_json, load_public_key, public_key_id


class NodeRole(str, Enum):
    AIR_RELAY = "air_relay"
    GROUND_GATEWAY = "ground_gateway"
    MARITIME_GATEWAY = "maritime_gateway"
    FIXED_INFRASTRUCTURE = "fixed_infrastructure"
    LAB_VALIDATION = "lab_validation"


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str
    role: NodeRole
    deployment_id: str
    learning_cohort_id: str
    context_schema: str
    allowed_model_families: tuple[str, ...]
    allowed_service_classes: tuple[str, ...]
    expected_platform: str
    created_at: datetime
    expires_at: datetime
    manifest_hash: str
    key_id: str

    def to_public_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["role"] = self.role.value
        value["created_at"] = self.created_at.isoformat()
        value["expires_at"] = self.expires_at.isoformat()
        return value


def _bounded_identifier(value: object, field: str) -> str:
    result = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", result):
        raise ValueError(f"node manifest {field} is invalid")
    return result


def _context_schema(value: object) -> str:
    result = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", result):
        raise ValueError("node manifest context_schema is invalid")
    return result


def verify_node_manifest(
    envelope: dict[str, Any],
    public_key: Ed25519PublicKey,
    *,
    now: datetime | None = None,
) -> NodeIdentity:
    if envelope.get("algorithm") != "Ed25519":
        raise ValueError("unsupported node manifest signature algorithm")
    expected_key_id = public_key_id(public_key)
    if envelope.get("key_id") != expected_key_id:
        raise ValueError("node manifest key ID does not match the provisioned public key")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("node manifest payload must be an object")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
        public_key.verify(signature, canonical_json(payload))
    except (KeyError, ValueError, InvalidSignature) as exc:
        raise ValueError("node manifest signature verification failed") from exc

    required = {
        "node_id",
        "role",
        "deployment_id",
        "learning_cohort_id",
        "context_schema",
        "allowed_model_families",
        "allowed_service_classes",
        "expected_platform",
        "created_at",
        "expires_at",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"node manifest is missing fields: {', '.join(missing)}")
    created_at = parse_timestamp(str(payload["created_at"]))
    expires_at = parse_timestamp(str(payload["expires_at"]))
    current = now or datetime.now(timezone.utc)
    if created_at > current or expires_at <= current or expires_at <= created_at:
        raise ValueError("node manifest is not currently valid")
    model_families = tuple(_bounded_identifier(value, "allowed_model_families") for value in payload["allowed_model_families"])
    service_classes = tuple(_bounded_identifier(value, "allowed_service_classes") for value in payload["allowed_service_classes"])
    if not model_families or not service_classes:
        raise ValueError("node manifest must allow at least one model family and service class")
    if len(model_families) != len(set(model_families)) or len(service_classes) != len(set(service_classes)):
        raise ValueError("node manifest contains duplicate allowances")
    encoded = canonical_json(payload)
    return NodeIdentity(
        node_id=_bounded_identifier(payload["node_id"], "node_id"),
        role=NodeRole(payload["role"]),
        deployment_id=_bounded_identifier(payload["deployment_id"], "deployment_id"),
        learning_cohort_id=_bounded_identifier(payload["learning_cohort_id"], "learning_cohort_id"),
        context_schema=_context_schema(payload["context_schema"]),
        allowed_model_families=model_families,
        allowed_service_classes=service_classes,
        expected_platform=str(payload["expected_platform"]).strip(),
        created_at=created_at,
        expires_at=expires_at,
        manifest_hash=hashlib.sha256(encoded).hexdigest(),
        key_id=expected_key_id,
    )


def load_verified_node_manifest(path: Path, public_key_path: Path) -> NodeIdentity:
    return verify_node_manifest(json.loads(path.read_text(encoding="utf-8")), load_public_key(public_key_path))


@dataclass(frozen=True)
class RuntimeContext:
    schema: str
    observed_at: str
    node: dict[str, object]
    platform_hash: str
    catalog_hash: str
    current_profile_id: str | None
    observable_profile_ids: tuple[str, ...]
    service_class: str
    decision_engine: str
    context_hash: str

    @classmethod
    def build(
        cls,
        identity: NodeIdentity,
        *,
        platform: dict[str, object],
        catalog_hash: str,
        snapshot: TelemetrySnapshot,
        service_class: str,
        decision_engine: str,
    ) -> "RuntimeContext":
        if service_class not in identity.allowed_service_classes:
            raise ValueError("service class is not permitted by the signed node manifest")
        platform_model = str(platform.get("model", ""))
        if identity.expected_platform not in platform_model:
            raise ValueError("live platform does not match the signed node manifest")
        observable = tuple(sorted(item.profile_id for item in snapshot.observations if item.interface_present))
        body = {
            "schema": identity.context_schema,
            "observed_at": snapshot.observed_at.isoformat(),
            "node": identity.to_public_dict(),
            "platform_hash": hashlib.sha256(canonical_json(platform)).hexdigest(),
            "catalog_hash": catalog_hash,
            "current_profile_id": snapshot.current_profile_id,
            "observable_profile_ids": observable,
            "service_class": service_class,
            "decision_engine": decision_engine,
        }
        return cls(**body, context_hash=hashlib.sha256(canonical_json(body)).hexdigest())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
