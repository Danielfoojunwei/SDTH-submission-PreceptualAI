from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .signing import canonical_json, public_key_id


GENESIS_HASH = "0" * 64


@dataclass(frozen=True)
class AuditVerification:
    valid: bool
    records: int
    error: str | None


class AuditLedger:
    """Hash-chained decision history, optionally bound to its author.

    The chain proves the sequence has not been *modified*. It cannot, on its
    own, stop plausible new history being *appended* — which is the memory-
    poisoning attack ``memory_integrity`` exists to defend. Supplying a signing
    key binds each record to the node that wrote it, so an attacker with
    database write access but no node key cannot manufacture accepted history.

    The signature is stored alongside the chain rather than inside it, so
    existing ledgers keep verifying unchanged. Stripping a signature therefore
    does not break the chain — it makes the record unauthenticated, which the
    memory auditor reports.
    """

    def __init__(self, path: Path, *, signing_key: Ed25519PrivateKey | None = None) -> None:
        self.path = path
        self._signing_key = signing_key
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
        self._migrate_authorship()
        self._connection.commit()

    def _migrate_authorship(self) -> None:
        """Add the authorship columns to a ledger written before they existed."""
        existing = {
            str(row[1]) for row in self._connection.execute("PRAGMA table_info(audit_records)")
        }
        for column in ("signature", "key_id"):
            if column not in existing:
                self._connection.execute(
                    f"ALTER TABLE audit_records ADD COLUMN {column} TEXT NOT NULL DEFAULT ''"
                )

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
        signature = ""
        key_id = ""
        if self._signing_key is not None:
            body = {
                "node_id": node_id,
                "recorded_at": created_at,
                "record_type": record_type,
                "payload": payload,
            }
            signature = base64.b64encode(self._signing_key.sign(canonical_json(body))).decode(
                "ascii"
            )
            key_id = public_key_id(self._signing_key.public_key())
        with self._connection:
            previous_hash = self._latest_hash()
            record_hash = self._hash(previous_hash, created_at, node_id, record_type, payload_json)
            self._connection.execute(
                """
                INSERT INTO audit_records
                    (created_at, node_id, record_type, payload_json, previous_hash, record_hash,
                     signature, key_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    node_id,
                    record_type,
                    payload_json,
                    previous_hash,
                    record_hash,
                    signature,
                    key_id,
                ),
            )
        return record_hash

    def signed_records(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """History in the shape the memory auditor consumes.

        Returned newest-last, matching write order, because the injection
        detector's burst test reads timing directly from the records.
        """
        query = (
            "SELECT created_at, node_id, record_type, payload_json, signature, key_id "
            "FROM audit_records ORDER BY id ASC"
        )
        rows = self._connection.execute(query).fetchall()
        if limit is not None and limit > 0:
            rows = rows[-limit:]
        return [
            {
                "recorded_at": created_at,
                "node_id": node_id,
                "record_type": record_type,
                "payload": json.loads(payload_json),
                "signature": signature,
                "key_id": key_id,
            }
            for created_at, node_id, record_type, payload_json, signature, key_id in rows
        ]

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

