"""Capability leases.

Every bounded action a node may take beyond passive observation is expressed as
an expiring, single-use capability lease that an operator has signed offline.
A lease names exactly one approved action, the numeric bounds of that action,
the node it is valid for, the hash of the policy that authorises it, a nonce,
a validity window, a rollback action, and a required post-condition.

The node holds only the operator public key. It cannot mint, widen, or renew a
lease, and it cannot replay one: a consumed nonce is refused forever.
"""

from __future__ import annotations

import base64
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import parse_timestamp, utc_now
from .signing import canonical_json, public_key_id
from .validation import non_empty_str, require_finite, sha256_hex


class LeaseAction(str, Enum):
    """The closed set of actions a lease can authorise.

    Notably absent: anything that transmits, changes frequency/gain/EIRP, edits
    trust anchors, deletes audit records, or grants the node authority over its
    own promotion. Those are never leasable.
    """

    SELECT_CONNECTIVITY_PROFILE = "select_connectivity_profile"
    FAILOVER_CONNECTIVITY = "failover_connectivity"
    SET_POWER_MODE = "set_power_mode"
    SET_INFERENCE_DUTY = "set_inference_duty"
    SET_SCAN_INTERVAL = "set_scan_interval"
    SET_TELEMETRY_INTERVAL = "set_telemetry_interval"
    SET_QOS_BUDGET = "set_qos_budget"
    ACTIVATE_APPROVED_SPECTRUM_PROFILE = "activate_approved_spectrum_profile"
    QUARANTINE_UPDATE = "quarantine_update"


NONCE_MIN_LENGTH = 16
NONCE_MAX_LENGTH = 128


@dataclass(frozen=True)
class CapabilityLease:
    lease_id: str
    node_id: str
    action: LeaseAction
    bounds: dict[str, float]
    policy_hash: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    rollback_action: str
    required_postcondition: str

    @classmethod
    def body_from_dict(cls, raw: dict[str, Any], *, now: datetime | None = None) -> "CapabilityLease":
        current = now or utc_now()
        lease_id = non_empty_str(raw.get("lease_id"), "lease_id")
        node_id = non_empty_str(raw.get("node_id"), "node_id")
        action = LeaseAction(str(raw["action"]))
        bounds_raw = raw.get("bounds", {})
        if not isinstance(bounds_raw, dict) or not bounds_raw:
            raise ValueError("lease must declare at least one numeric bound")
        bounds = {str(key): require_finite(value, f"bounds.{key}") for key, value in bounds_raw.items()}
        policy_hash = sha256_hex(raw.get("policy_hash"), "policy_hash")
        nonce = non_empty_str(raw.get("nonce"), "nonce")
        if not NONCE_MIN_LENGTH <= len(nonce) <= NONCE_MAX_LENGTH:
            raise ValueError("lease nonce length is out of range")
        issued_at = parse_timestamp(str(raw["issued_at"]))
        expires_at = parse_timestamp(str(raw["expires_at"]))
        if expires_at <= issued_at:
            raise ValueError("lease expiry must be after issuance")
        if issued_at > current:
            raise ValueError("lease issuance time is in the future")
        rollback_action = non_empty_str(raw.get("rollback_action"), "rollback_action")
        required_postcondition = non_empty_str(raw.get("required_postcondition"), "required_postcondition")
        return cls(
            lease_id=lease_id,
            node_id=node_id,
            action=action,
            bounds=bounds,
            policy_hash=policy_hash,
            nonce=nonce,
            issued_at=issued_at,
            expires_at=expires_at,
            rollback_action=rollback_action,
            required_postcondition=required_postcondition,
        )

    def to_body(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "node_id": self.node_id,
            "action": self.action.value,
            "bounds": {key: self.bounds[key] for key in sorted(self.bounds)},
            "policy_hash": self.policy_hash,
            "nonce": self.nonce,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "rollback_action": self.rollback_action,
            "required_postcondition": self.required_postcondition,
        }

    def bound(self, name: str) -> float:
        if name not in self.bounds:
            raise KeyError(f"lease does not declare bound {name}")
        return self.bounds[name]

    def within(self, name: str, value: float) -> bool:
        """A leased value is valid only up to its declared numeric bound."""
        return require_finite(value, name) <= self.bounds.get(name, float("-inf"))


class LeaseError(RuntimeError):
    """Raised when a capability lease is invalid, expired, or replayed."""


def sign_lease(body: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    lease = CapabilityLease.body_from_dict(body)
    encoded = canonical_json(lease.to_body())
    signature = private_key.sign(encoded)
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "body": lease.to_body(),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_lease(
    envelope: dict[str, Any],
    public_key: Ed25519PublicKey,
    *,
    node_id: str,
    now: datetime | None = None,
) -> CapabilityLease:
    current = now or utc_now()
    if envelope.get("algorithm") != "Ed25519":
        raise LeaseError("unsupported lease signature algorithm")
    if envelope.get("key_id") != public_key_id(public_key):
        raise LeaseError("lease key ID does not match the provisioned operator key")
    body = envelope.get("body")
    if not isinstance(body, dict):
        raise LeaseError("lease body must be an object")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise LeaseError("lease signature is missing or malformed") from exc
    try:
        lease = CapabilityLease.body_from_dict(body, now=current)
    except ValueError as exc:
        raise LeaseError(str(exc)) from exc
    encoded = canonical_json(lease.to_body())
    try:
        public_key.verify(signature, encoded)
    except InvalidSignature as exc:
        raise LeaseError("lease signature verification failed") from exc
    if lease.node_id != node_id:
        raise LeaseError("lease is bound to a different node")
    if current >= lease.expires_at:
        raise LeaseError("lease has expired")
    return lease


class LeaseNonceStore:
    """Durable single-use nonce store; a consumed lease can never be replayed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consumed_leases (
                nonce TEXT PRIMARY KEY,
                lease_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                action TEXT NOT NULL,
                consumed_at TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def consume(self, lease: CapabilityLease, *, now: datetime | None = None) -> None:
        """Record a lease nonce as used, or raise if it was already consumed."""
        consumed_at = (now or utc_now()).isoformat()
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO consumed_leases (nonce, lease_id, node_id, action, consumed_at) VALUES (?, ?, ?, ?, ?)",
                    (lease.nonce, lease.lease_id, lease.node_id, lease.action.value, consumed_at),
                )
        except sqlite3.IntegrityError as exc:
            raise LeaseError("lease nonce has already been consumed (replay refused)") from exc

    def is_consumed(self, nonce: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM consumed_leases WHERE nonce = ? LIMIT 1", (nonce,)
        ).fetchone()
        return row is not None


def redeem_lease(
    envelope: dict[str, Any],
    public_key: Ed25519PublicKey,
    store: LeaseNonceStore,
    *,
    node_id: str,
    expected_action: LeaseAction,
    expected_policy_hash: str | None = None,
    now: datetime | None = None,
) -> CapabilityLease:
    """Verify, scope-check, and atomically consume a lease in one step."""
    current = now or utc_now()
    lease = verify_lease(envelope, public_key, node_id=node_id, now=current)
    if lease.action is not expected_action:
        raise LeaseError(f"lease authorises {lease.action.value}, not {expected_action.value}")
    if expected_policy_hash is not None and lease.policy_hash != expected_policy_hash.lower():
        raise LeaseError("lease policy hash does not match the active signed policy")
    store.consume(lease, now=current)
    return lease
