"""Cross-node spectrum fabric.

Until now each node sensed alone: the vault/sub-node topology was documented but
no code moved an observation between nodes. This module is that path.

A sub-node publishes a **signed report** of what its radio measured. A vault
node verifies each peer's signature against a signed roster, discards reports
that are stale, replayed, or from unknown peers, and fuses what survives into a
fleet view: which bands are clear across the area, where interference is
corroborated by more than one node, and which peers have gone silent.

Three properties matter and are enforced here:

**Every report is signed and node-bound.** A report carries its author's node
ID and is verified against that node's public key from an operator-signed
roster. An unknown author is refused — a fabric that accepts anonymous
observations is a poisoning surface, and threat 14 (poisoned telemetry) and
threat 17 (Sybil nodes) both live here.

**Correlation requires comparable clocks.** A report carries the author's own
time-integrity verdict. Reports from an undisciplined clock are accepted for
*presence* but excluded from corroboration, because two observations that cannot
be aligned in time cannot confirm each other.

**Corroboration is counted, not assumed.** A band is only reported as
fleet-confirmed interference when independent nodes agree within a time window.
One node shouting is an observation; three nodes agreeing is evidence.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import parse_timestamp, utc_now
from .signing import canonical_json, public_key_id
from .spectrum import BandStatus, SpectrumIntelligence
from .validation import bounded_number, non_empty_str, sha256_hex

REPORT_SCHEMA = "kval-fabric-report/v1"

#: A report older than this cannot describe the current RF environment.
DEFAULT_REPORT_MAX_AGE_SECONDS = 120.0

#: Independent nodes that must agree before interference is fleet-confirmed.
DEFAULT_CORROBORATION_QUORUM = 2


class FabricError(RuntimeError):
    """Raised when a peer report cannot be trusted."""


@dataclass(frozen=True)
class PeerIdentity:
    node_id: str
    public_key: Ed25519PublicKey
    role: str
    #: Operator-declared position, used for emitter localisation. It lives in
    #: the signed roster rather than in a node's own report on purpose: a
    #: self-reported position is a spoofing surface, and a node that could
    #: claim to be anywhere could drag every estimate.
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
    """Operator-signed set of nodes whose reports this node will accept."""

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


@dataclass(frozen=True)
class BandReport:
    band_id: str
    status: BandStatus
    occupancy_percent: float | None
    noise_floor_excess_db: float | None
    detected_emitters: int | None
    confidence: float
    #: Strongest received power this node measured in the band, in dBm. Carried
    #: across the fabric because emitter localisation needs a real power figure
    #: from each node; absent when no backend measured one.
    peak_power_dbm: float | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BandReport":
        return cls(
            band_id=non_empty_str(raw.get("band_id"), "band_id"),
            status=BandStatus(str(raw["status"])),
            occupancy_percent=(float(raw["occupancy_percent"]) if raw.get("occupancy_percent") is not None else None),
            noise_floor_excess_db=(
                float(raw["noise_floor_excess_db"]) if raw.get("noise_floor_excess_db") is not None else None
            ),
            detected_emitters=(int(raw["detected_emitters"]) if raw.get("detected_emitters") is not None else None),
            confidence=float(raw.get("confidence", 0.0)),
            peak_power_dbm=(float(raw["peak_power_dbm"]) if raw.get("peak_power_dbm") is not None else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "status": self.status.value,
            "occupancy_percent": self.occupancy_percent,
            "noise_floor_excess_db": self.noise_floor_excess_db,
            "detected_emitters": self.detected_emitters,
            "confidence": self.confidence,
            "peak_power_dbm": self.peak_power_dbm,
        }


@dataclass(frozen=True)
class FabricReport:
    """One node's signed statement of what it measured."""

    node_id: str
    produced_at: datetime
    policy_id: str
    policy_hash: str
    clock_trusted: bool
    bands: tuple[BandReport, ...]

    @classmethod
    def from_intelligence(
        cls,
        intelligence: SpectrumIntelligence,
        *,
        node_id: str,
        policy_hash: str,
        clock_trusted: bool,
    ) -> "FabricReport":
        return cls(
            node_id=non_empty_str(node_id, "node_id"),
            produced_at=intelligence.produced_at,
            policy_id=intelligence.policy_id,
            policy_hash=sha256_hex(policy_hash, "policy_hash"),
            clock_trusted=clock_trusted,
            bands=tuple(
                BandReport(
                    band_id=band.band_id,
                    status=band.status,
                    occupancy_percent=band.occupancy_percent,
                    noise_floor_excess_db=band.noise_floor_excess_db,
                    detected_emitters=band.detected_emitters,
                    confidence=band.confidence,
                    peak_power_dbm=band.peak_power_dbm,
                )
                for band in intelligence.bands
            ),
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FabricReport":
        return cls(
            node_id=non_empty_str(raw.get("node_id"), "node_id"),
            produced_at=parse_timestamp(str(raw["produced_at"])),
            policy_id=non_empty_str(raw.get("policy_id"), "policy_id"),
            policy_hash=sha256_hex(raw.get("policy_hash"), "policy_hash"),
            clock_trusted=bool(raw.get("clock_trusted", False)),
            bands=tuple(BandReport.from_dict(item) for item in raw.get("bands", [])),
        )

    def to_body(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "node_id": self.node_id,
            "produced_at": self.produced_at.isoformat(),
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "clock_trusted": self.clock_trusted,
            "bands": [band.to_dict() for band in self.bands],
        }


def sign_report(report: FabricReport, private_key: Ed25519PrivateKey) -> dict[str, Any]:
    body = report.to_body()
    signature = private_key.sign(canonical_json(body))
    return {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "body": body,
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_report(
    envelope: dict[str, Any],
    roster: PeerRoster,
    *,
    now: datetime | None = None,
    max_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> FabricReport:
    """Verify a peer report against the signed roster, or raise ``FabricError``."""
    current = now or utc_now()
    if envelope.get("algorithm") != "Ed25519":
        raise FabricError("unsupported report signature algorithm")
    body = envelope.get("body")
    if not isinstance(body, dict):
        raise FabricError("report body must be an object")
    if body.get("schema") != REPORT_SCHEMA:
        raise FabricError("unsupported report schema")
    try:
        report = FabricReport.from_dict(body)
    except (ValueError, KeyError, TypeError) as exc:
        raise FabricError(f"report body is malformed: {exc}") from exc
    # Reject anything outside the signed projection rather than dropping it.
    if body != report.to_body():
        raise FabricError("report body contains unauthenticated or non-canonical fields")

    peer = roster.by_id().get(report.node_id)
    if peer is None:
        raise FabricError(f"report author {report.node_id!r} is not on the signed roster")
    if envelope.get("key_id") != public_key_id(peer.public_key):
        raise FabricError("report key ID does not match the roster entry for this node")
    try:
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
    except (KeyError, ValueError) as exc:
        raise FabricError("report signature is missing or malformed") from exc
    try:
        peer.public_key.verify(signature, canonical_json(report.to_body()))
    except InvalidSignature as exc:
        raise FabricError("report signature verification failed") from exc

    age = (current - report.produced_at).total_seconds()
    if age > max_age_seconds:
        raise FabricError("report is older than the correlation window")
    if age < -1.0:
        raise FabricError("report is dated in the future")
    return report


@dataclass(frozen=True)
class FleetBandView:
    band_id: str
    reporting_nodes: tuple[str, ...]
    corroborating_nodes: tuple[str, ...]
    interference_confirmed: bool
    clear_everywhere: bool
    max_noise_excess_db: float | None
    total_emitters: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "reporting_nodes": list(self.reporting_nodes),
            "corroborating_nodes": list(self.corroborating_nodes),
            "interference_confirmed": self.interference_confirmed,
            "clear_everywhere": self.clear_everywhere,
            "max_noise_excess_db": self.max_noise_excess_db,
            "total_emitters": self.total_emitters,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FleetView:
    produced_at: datetime
    bands: tuple[FleetBandView, ...]
    contributing_nodes: tuple[str, ...]
    excluded_nodes: tuple[str, ...]
    silent_nodes: tuple[str, ...]
    fleet_clear_band_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-fleet-view/v1",
            "produced_at": self.produced_at.isoformat(),
            "bands": [band.to_dict() for band in self.bands],
            "contributing_nodes": list(self.contributing_nodes),
            "excluded_nodes": list(self.excluded_nodes),
            "silent_nodes": list(self.silent_nodes),
            "fleet_clear_band_ids": list(self.fleet_clear_band_ids),
            "reasons": list(self.reasons),
        }


#: Band statuses that indicate a node is seeing interference rather than traffic.
_INTERFERENCE_STATUSES = frozenset({BandStatus.JAMMING_SUSPECTED, BandStatus.ANOMALOUS})
_USABLE_STATUSES = frozenset({BandStatus.CLEAR, BandStatus.OCCUPIED})


class FabricAggregator:
    """Fuses verified peer reports into a fleet view."""

    def __init__(
        self,
        *,
        quorum: int = DEFAULT_CORROBORATION_QUORUM,
        max_age_seconds: float = DEFAULT_REPORT_MAX_AGE_SECONDS,
    ) -> None:
        if quorum < 1:
            raise ValueError("quorum must be at least 1")
        self.quorum = quorum
        self.max_age_seconds = max_age_seconds

    def aggregate(
        self,
        reports: list[FabricReport],
        *,
        expected_nodes: frozenset[str] | None = None,
        now: datetime | None = None,
    ) -> FleetView:
        current = now or utc_now()
        cutoff = current - timedelta(seconds=self.max_age_seconds)

        fresh: list[FabricReport] = []
        excluded: set[str] = set()
        for report in reports:
            if report.produced_at < cutoff:
                excluded.add(report.node_id)
                continue
            fresh.append(report)

        # A node whose clock is not disciplined still tells us something is
        # there, but its observations cannot be aligned with a peer's, so it
        # never counts toward corroboration.
        correlatable = [report for report in fresh if report.clock_trusted]
        uncorrelatable = sorted({r.node_id for r in fresh if not r.clock_trusted})

        band_ids: list[str] = []
        for report in fresh:
            for band in report.bands:
                if band.band_id not in band_ids:
                    band_ids.append(band.band_id)

        views: list[FleetBandView] = []
        for band_id in band_ids:
            reporting = sorted({r.node_id for r in fresh if any(b.band_id == band_id for b in r.bands)})
            corroborating = sorted(
                {
                    r.node_id
                    for r in correlatable
                    for b in r.bands
                    if b.band_id == band_id and b.status in _INTERFERENCE_STATUSES and b.confidence > 0.0
                }
            )
            usable_votes = [
                b
                for r in fresh
                for b in r.bands
                if b.band_id == band_id and b.confidence > 0.0
            ]
            clear_everywhere = bool(usable_votes) and all(b.status in _USABLE_STATUSES for b in usable_votes)
            excesses = [
                b.noise_floor_excess_db
                for r in fresh
                for b in r.bands
                if b.band_id == band_id and b.noise_floor_excess_db is not None
            ]
            emitters = sum(
                b.detected_emitters or 0
                for r in fresh
                for b in r.bands
                if b.band_id == band_id
            )
            confirmed = len(corroborating) >= self.quorum

            reasons: list[str] = []
            if confirmed:
                reasons.append("INTERFERENCE_CORROBORATED_BY_QUORUM")
            elif corroborating:
                reasons.append("INTERFERENCE_REPORTED_BELOW_QUORUM")
            if clear_everywhere:
                reasons.append("USABLE_AT_EVERY_REPORTING_NODE")
            if not usable_votes:
                reasons.append("NO_NODE_PRODUCED_USABLE_EVIDENCE")

            views.append(
                FleetBandView(
                    band_id=band_id,
                    reporting_nodes=tuple(reporting),
                    corroborating_nodes=tuple(corroborating),
                    interference_confirmed=confirmed,
                    clear_everywhere=clear_everywhere,
                    max_noise_excess_db=(max(excesses) if excesses else None),
                    total_emitters=emitters,
                    reasons=tuple(reasons),
                )
            )

        contributing = sorted({report.node_id for report in fresh})
        silent = sorted(expected_nodes - set(contributing)) if expected_nodes else []

        reasons: list[str] = []
        if any(view.interference_confirmed for view in views):
            reasons.append("FLEET_CONFIRMED_INTERFERENCE")
        if silent:
            reasons.append("EXPECTED_NODES_SILENT")
        if uncorrelatable:
            reasons.append("REPORTS_EXCLUDED_FROM_CORROBORATION_FOR_UNTRUSTED_CLOCK")
        if excluded:
            reasons.append("STALE_REPORTS_DISCARDED")
        if not reasons:
            reasons.append("NOMINAL")

        return FleetView(
            produced_at=current,
            bands=tuple(views),
            contributing_nodes=tuple(contributing),
            excluded_nodes=tuple(sorted(excluded | set(uncorrelatable))),
            silent_nodes=tuple(silent),
            fleet_clear_band_ids=tuple(
                view.band_id for view in views if view.clear_everywhere and not view.interference_confirmed
            ),
            reasons=tuple(reasons),
        )
