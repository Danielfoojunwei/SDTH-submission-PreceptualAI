"""The operator-signed set of nodes whose statements this node will accept.

This is the identity half of what used to be ``fabric.py``. The spectrum half —
band reports, fleet views, corroboration quorums — was removed along with
passive RF sensing, but peer identity outlived it: the federated learning tier
still has to answer "is this update from a node the operator vouched for, signed
by the key the operator recorded for it?" before it will admit a contribution.

Two properties are load-bearing and both are inherited unchanged:

The roster is **operator-signed and expiring**. A node cannot add a peer, and a
roster that has run out is not a roster. Trust is granted offline, not
negotiated in the field.

A peer's position is **operator-declared**, never self-reported. A node that
could claim to be anywhere could drag every estimate that used it, so the claim
lives in the document the operator signed rather than in anything a node emits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .models import parse_timestamp, utc_now
from .validation import bounded_number, non_empty_str


class RosterError(RuntimeError):
    """Raised when a peer's statement cannot be trusted."""


@dataclass(frozen=True)
class PeerIdentity:
    node_id: str
    public_key: Ed25519PublicKey
    role: str
    #: Operator-declared position. It lives in the signed roster rather than in
    #: anything a node emits on purpose: a self-reported position is a spoofing
    #: surface, and a node that could claim to be anywhere could drag every
    #: estimate that trusted it.
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    altitude_m: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PeerIdentity":
        node_id = non_empty_str(raw.get("node_id"), "node_id")
        role = str(raw.get("role", "sub-node")).strip() or "sub-node"
        pem = str(raw.get("public_key_pem", "")).encode("utf-8")
        if not pem:
            raise ValueError(f"peer {node_id} has no public_key_pem")
        key = serialization.load_pem_public_key(pem)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError(f"peer {node_id} key is not Ed25519")
        position = raw.get("position") or {}
        latitude = longitude = None
        altitude = 0.0
        if position:
            latitude = bounded_number(position["latitude_deg"], "latitude_deg", -90.0, 90.0)
            longitude = bounded_number(position["longitude_deg"], "longitude_deg", -180.0, 180.0)
            altitude = bounded_number(position.get("altitude_m", 0.0), "altitude_m", -500.0, 20000.0)
        return cls(
            node_id=node_id,
            public_key=key,
            role=role,
            latitude_deg=latitude,
            longitude_deg=longitude,
            altitude_m=altitude,
        )

    def has_position(self) -> bool:
        return self.latitude_deg is not None and self.longitude_deg is not None


@dataclass(frozen=True)
class PeerRoster:
    """Operator-signed set of nodes whose statements this node will accept."""

    roster_id: str
    created_at: datetime
    expires_at: datetime
    peers: tuple[PeerIdentity, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, now: datetime | None = None) -> "PeerRoster":
        current = now or utc_now()
        roster_id = non_empty_str(raw.get("roster_id"), "roster_id")
        created_at = parse_timestamp(str(raw["created_at"]))
        expires_at = parse_timestamp(str(raw["expires_at"]))
        if expires_at <= created_at:
            raise ValueError("roster expiry must be after creation")
        if expires_at <= current:
            raise ValueError("peer roster has expired")
        peers = tuple(PeerIdentity.from_dict(item) for item in raw.get("peers", []))
        if not peers:
            raise ValueError("peer roster must list at least one node")
        ids = [peer.node_id for peer in peers]
        if len(ids) != len(set(ids)):
            raise ValueError("peer roster contains duplicate node IDs")
        return cls(roster_id=roster_id, created_at=created_at, expires_at=expires_at, peers=peers)

    def by_id(self) -> dict[str, PeerIdentity]:
        return {peer.node_id: peer for peer in self.peers}
