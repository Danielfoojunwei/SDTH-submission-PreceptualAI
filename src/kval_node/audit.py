from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .signing import canonical_json


GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    records: int
    error: str | None


class AuditLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                node_id TEXT NOT NULL,
                record_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                record_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _latest_hash(self) -> str:
        row = self._connection.execute(
            "SELECT record_hash FROM audit_records ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return str(row[0]) if row else GENESIS_HASH

    @staticmethod
    def _hash(previous_hash: str, created_at: str, node_id: str, record_type: str, payload_json: str) -> str:
        material = b"\x1f".join(
            (
                previous_hash.encode("ascii"),
                created_at.encode("utf-8"),
                node_id.encode("utf-8"),
                record_type.encode("utf-8"),
                payload_json.encode("utf-8"),
            )
        )
        return hashlib.sha256(material).hexdigest()

    def append(self, node_id: str, record_type: str, payload: dict[str, Any]) -> str:
        created_at = datetime.now(timezone.utc).isoformat()
        payload_json = canonical_json(payload).decode("utf-8")
        with self._connection:
            previous_hash = self._latest_hash()
            record_hash = self._hash(previous_hash, created_at, node_id, record_type, payload_json)
            self._connection.execute(
                """
                INSERT INTO audit_records
                    (created_at, node_id, record_type, payload_json, previous_hash, record_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (created_at, node_id, record_type, payload_json, previous_hash, record_hash),
            )
        return record_hash

    def recent_activations(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or datetime.now(timezone.utc)
        cutoff = (current - timedelta(hours=1)).isoformat()
        rows = self._connection.execute(
            """
            SELECT created_at, payload_json
            FROM audit_records
            WHERE record_type = 'activation' AND created_at >= ?
            ORDER BY id ASC
            """,
            (cutoff,),
        ).fetchall()
        activations: list[dict[str, Any]] = []
        for created_at, payload_json in rows:
            payload = json.loads(payload_json)
            if payload.get("success") is True:
                payload["created_at"] = created_at
                activations.append(payload)
        return activations

    def verify(self) -> AuditVerification:
        rows = self._connection.execute(
            """
            SELECT created_at, node_id, record_type, payload_json, previous_hash, record_hash
            FROM audit_records ORDER BY id ASC
            """
        ).fetchall()
        expected_previous = GENESIS_HASH
        for index, (created_at, node_id, record_type, payload_json, previous_hash, record_hash) in enumerate(rows, 1):
            if previous_hash != expected_previous:
                return AuditVerification(False, len(rows), f"record {index} previous hash mismatch")
            expected = self._hash(previous_hash, created_at, node_id, record_type, payload_json)
            if record_hash != expected:
                return AuditVerification(False, len(rows), f"record {index} content hash mismatch")
            expected_previous = record_hash
        return AuditVerification(True, len(rows), None)

