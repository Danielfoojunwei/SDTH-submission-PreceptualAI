from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ServiceClass


#: Roles a node may be configured to select for. Kept in sync with
#: ``c2link.LinkRole`` by the test suite rather than imported, so the config
#: layer does not depend on the uncrewed-platform module.
VALID_LINK_ROLES = frozenset({"c2_cnpc", "payload", "bulk"})


def _link_role(raw: dict[str, Any]) -> str:
    value = str(raw.get("link_role", "payload")).strip().lower()
    if value not in VALID_LINK_ROLES:
        raise ValueError(
            f"link_role must be one of {sorted(VALID_LINK_ROLES)}, got {value!r}"
        )
    return value


def _optional_path(raw: dict[str, Any], key: str) -> Path | None:
    value = raw.get(key)
    if value in (None, ""):
        return None
    return Path(str(value))


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    catalog_path: Path
    catalog_public_key_path: Path
    audit_database_path: Path
    latest_result_path: Path
    service_class: ServiceClass
    poll_interval_seconds: float
    observation_max_age_seconds: float
    minimum_confidence: float
    apply_network_changes: bool
    # Optional capability policies (absent = capability not enabled on this node).
    power_policy_path: Path | None = None
    transport_policy_path: Path | None = None
    transport_wifi_connection_uuid: str | None = None
    lease_public_key_path: Path | None = None
    lease_nonce_database_path: Path | None = None
    # Cross-node fabric. The roster is operator-signed; the node key signs this
    # node's own reports; the inbox is where verified peer reports are read from.
    fabric_roster_path: Path | None = None
    fabric_node_key_path: Path | None = None
    #: Signed manifest for a candidate predictive model. Absent means the
    #: learned path is not present at all and the node stays deterministic.
    model_manifest_path: Path | None = None
    federated_inbox_path: Path | None = None
    #: Range within which two positioned peers are treated as linked when
    #: building the fabric graph. An operator-declared planning figure: it
    #: assumes line of sight and says nothing about terrain, so a link that
    #: does not carry traffic in practice belongs out of the roster rather
    #: than inferred away here.
    fabric_link_range_m: float = 2000.0
    #: Per-release privacy budget for federated contributions. ``None`` means
    #: exact means are released, which is the historical behaviour and leaks
    #: what each sensor hears. Setting it trades aggregate accuracy for that
    #: exposure, and the trade is reported on every release rather than assumed.
    federated_privacy_epsilon: float | None = None
    #: Roster peers that must attest before an expiring grant is extended under
    #: partition. Operator-declared: a fleet that could pick its own quorum
    #: could pick one.
    quorum_threshold: int = 2
    #: Refuse signatures from algorithms CNSA 2.0 does not approve. Off by
    #: default because turning it on makes every Ed25519-signed artefact
    #: already in the field unverifiable — a migration to plan, not a flag to
    #: be surprised by.
    require_cnsa_2_0: bool = False
    #: Operator-signed emission-control order and command-link budget. Absent
    #: means this node is not on an uncrewed platform, and the C2 assessment
    #: reports unavailable rather than inventing a budget.
    emcon_order_path: Path | None = None
    c2_budget_path: Path | None = None
    #: Operator-signed statement of what this node must carry, and how many
    #: distinct slices each flow must survive on. Absent means the node plans no
    #: allocation: it will not invent its own entitlement.
    flow_table_path: Path | None = None
    #: Which link role this node's autonomous loop selects for. Ranking never
    #: crosses roles, so a node whose mission is to hold the command link and
    #: one whose mission is to move payload are different deployments — and
    #: without this the loop could only ever pick a payload link, which made
    #: a declared c2_cnpc profile permanently unselectable.
    link_role: str = "payload"
    #: CNSA 2.0 trust anchor. Provisioned alongside the Ed25519 one so a fleet
    #: can migrate document by document rather than in a flag day.
    lms_public_key_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "NodeConfig":
        config_path = Path(path)
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        required = {
            "node_id",
            "catalog_path",
            "catalog_public_key_path",
            "audit_database_path",
            "latest_result_path",
            "service_class",
            "poll_interval_seconds",
            "observation_max_age_seconds",
            "minimum_confidence",
            "apply_network_changes",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"node configuration is missing fields: {', '.join(missing)}")
        poll = float(raw["poll_interval_seconds"])
        age = float(raw["observation_max_age_seconds"])
        confidence = float(raw["minimum_confidence"])
        if poll < 0.2:
            raise ValueError("poll_interval_seconds must be at least 0.2")
        if age <= poll:
            raise ValueError("observation_max_age_seconds must exceed the poll interval")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        node_id = str(raw["node_id"]).strip()
        if not node_id:
            raise ValueError("node_id cannot be empty")
        transport_wifi_uuid_raw = raw.get("transport_wifi_connection_uuid")
        transport_wifi_uuid = str(transport_wifi_uuid_raw).strip() if transport_wifi_uuid_raw else ""
        return cls(
            node_id=node_id,
            catalog_path=Path(raw["catalog_path"]),
            catalog_public_key_path=Path(raw["catalog_public_key_path"]),
            audit_database_path=Path(raw["audit_database_path"]),
            latest_result_path=Path(raw["latest_result_path"]),
            service_class=ServiceClass(raw["service_class"]),
            poll_interval_seconds=poll,
            observation_max_age_seconds=age,
            minimum_confidence=confidence,
            apply_network_changes=bool(raw["apply_network_changes"]),
            power_policy_path=_optional_path(raw, "power_policy_path"),
            transport_policy_path=_optional_path(raw, "transport_policy_path"),
            transport_wifi_connection_uuid=transport_wifi_uuid or None,
            lease_public_key_path=_optional_path(raw, "lease_public_key_path"),
            lease_nonce_database_path=_optional_path(raw, "lease_nonce_database_path"),
            fabric_roster_path=_optional_path(raw, "fabric_roster_path"),
            fabric_node_key_path=_optional_path(raw, "fabric_node_key_path"),
            model_manifest_path=_optional_path(raw, "model_manifest_path"),
            federated_inbox_path=_optional_path(raw, "federated_inbox_path"),
            fabric_link_range_m=float(raw.get("fabric_link_range_m", 2000.0)),
            federated_privacy_epsilon=(
                float(raw["federated_privacy_epsilon"])
                if raw.get("federated_privacy_epsilon") is not None
                else None
            ),
            quorum_threshold=int(raw.get("quorum_threshold", 2)),
            require_cnsa_2_0=bool(raw.get("require_cnsa_2_0", False)),
            emcon_order_path=_optional_path(raw, "emcon_order_path"),
            c2_budget_path=_optional_path(raw, "c2_budget_path"),
            flow_table_path=_optional_path(raw, "flow_table_path"),
            link_role=_link_role(raw),
            lms_public_key_path=_optional_path(raw, "lms_public_key_path"),
        )

