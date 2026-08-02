from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import ServiceClass


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
    spectrum_policy_path: Path | None = None
    spectrum_sensor_socket: Path | None = None
    power_policy_path: Path | None = None
    transport_policy_path: Path | None = None
    transport_wifi_connection_uuid: str | None = None
    lease_public_key_path: Path | None = None
    lease_nonce_database_path: Path | None = None

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
            spectrum_policy_path=_optional_path(raw, "spectrum_policy_path"),
            spectrum_sensor_socket=_optional_path(raw, "spectrum_sensor_socket"),
            power_policy_path=_optional_path(raw, "power_policy_path"),
            transport_policy_path=_optional_path(raw, "transport_policy_path"),
            transport_wifi_connection_uuid=transport_wifi_uuid or None,
            lease_public_key_path=_optional_path(raw, "lease_public_key_path"),
            lease_nonce_database_path=_optional_path(raw, "lease_nonce_database_path"),
        )

