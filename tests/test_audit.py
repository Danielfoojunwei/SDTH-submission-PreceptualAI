from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.audit import AuditLedger
from kval_node.memory_integrity import (
    MemoryAuditor,
    MemoryIntegrityError,
    SignedRecord,
    verify_record,
)
from kval_node.models import parse_timestamp


class AuditTests(unittest.TestCase):
    def test_hash_chain_verifies_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.db"
            ledger = AuditLedger(path)
            ledger.append("node-1", "decision", {"selected": "wifi"})
            ledger.append("node-1", "activation", {"profile_id": "wifi", "success": True})
            self.assertTrue(ledger.verify().valid)
            ledger.close()

            connection = sqlite3.connect(path)
            connection.execute("UPDATE audit_records SET payload_json = ? WHERE id = 1", ('{"selected":"altered"}',))
            connection.commit()
            connection.close()

            reopened = AuditLedger(path)
            try:
                result = reopened.verify()
                self.assertFalse(result.valid)
                self.assertIn("content hash mismatch", result.error or "")
            finally:
                reopened.close()


class AuthorshipTests(unittest.TestCase):
    """Records bound to the node that wrote them.

    The hash chain proves history was not rewritten. It cannot show whether
    plausible history was *appended*, which is the memory-poisoning attack
    ``memory_integrity`` defends against and this signing exists to answer.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "audit.db"
        self.key = Ed25519PrivateKey.generate()

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_signed_records_verify_as_this_node_s_own(self) -> None:
        ledger = AuditLedger(self.path, signing_key=self.key)
        ledger.append("node-1", "decision", {"selected": "wifi"})
        rows = ledger.signed_records()
        ledger.close()
        record = SignedRecord(
            node_id=rows[0]["node_id"],
            recorded_at=parse_timestamp(rows[0]["recorded_at"]),
            record_type=rows[0]["record_type"],
            payload=rows[0]["payload"],
            signature=rows[0]["signature"],
            key_id=rows[0]["key_id"],
        )
        verify_record(record, self.key.public_key(), node_id="node-1")

    def test_a_row_inserted_behind_the_ledger_s_back_is_unauthenticated(self) -> None:
        # The attacker with database write access but no node key. The chain
        # would need repairing too, but the point is that the signature check
        # fails independently.
        ledger = AuditLedger(self.path, signing_key=self.key)
        ledger.append("node-1", "decision", {"selected": "wifi"})
        ledger.close()

        connection = sqlite3.connect(self.path)
        connection.execute(
            "INSERT INTO audit_records (created_at, node_id, record_type, payload_json,"
            " previous_hash, record_hash, signature, key_id)"
            " VALUES ('2026-01-01T00:00:00+00:00','node-1','decision','{\"selected\":\"forged\"}',"
            " 'x','y','','')"
        )
        connection.commit()
        connection.close()

        reopened = AuditLedger(self.path)
        rows = reopened.signed_records()
        reopened.close()
        forged = SignedRecord(
            node_id=rows[-1]["node_id"],
            recorded_at=parse_timestamp(rows[-1]["recorded_at"]),
            record_type=rows[-1]["record_type"],
            payload=rows[-1]["payload"],
            signature=rows[-1]["signature"],
            key_id=rows[-1]["key_id"],
        )
        report = MemoryAuditor().audit([forged], self.key.public_key(), node_id="node-1")
        self.assertFalse(report.trustworthy)
        self.assertIn("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE", report.reasons)

    def test_relabelling_a_signed_record_s_type_is_detected(self) -> None:
        # record_type is inside the signed body, so a signed observation cannot
        # be re-presented as an activation.
        ledger = AuditLedger(self.path, signing_key=self.key)
        ledger.append("node-1", "decision", {"selected": "wifi"})
        rows = ledger.signed_records()
        ledger.close()
        relabelled = SignedRecord(
            node_id=rows[0]["node_id"],
            recorded_at=parse_timestamp(rows[0]["recorded_at"]),
            record_type="activation",
            payload=rows[0]["payload"],
            signature=rows[0]["signature"],
            key_id=rows[0]["key_id"],
        )
        with self.assertRaises(MemoryIntegrityError):
            verify_record(relabelled, self.key.public_key(), node_id="node-1")

    def test_an_unsigned_ledger_still_works(self) -> None:
        # Signing is opt-in; a node without a fabric key must keep operating.
        ledger = AuditLedger(self.path)
        ledger.append("node-1", "decision", {"selected": "wifi"})
        rows = ledger.signed_records()
        self.assertTrue(ledger.verify().valid)
        ledger.close()
        self.assertEqual(rows[0]["signature"], "")

    def test_a_ledger_written_before_signing_existed_still_opens(self) -> None:
        # Migration: the columns are added to an existing table rather than
        # requiring the operator to discard recorded history.
        connection = sqlite3.connect(self.path)
        connection.execute(
            """
            CREATE TABLE audit_records (
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
        connection.commit()
        connection.close()

        ledger = AuditLedger(self.path, signing_key=self.key)
        try:
            ledger.append("node-1", "decision", {"selected": "wifi"})
            self.assertTrue(ledger.verify().valid)
            self.assertNotEqual(ledger.signed_records()[0]["signature"], "")
        finally:
            ledger.close()

    def test_the_chain_still_verifies_for_pre_signing_records(self) -> None:
        # The signature lives alongside the chain, not inside it, so adding it
        # cannot invalidate history recorded before it existed.
        unsigned = AuditLedger(self.path)
        unsigned.append("node-1", "decision", {"selected": "wifi"})
        unsigned.close()
        signed = AuditLedger(self.path, signing_key=self.key)
        try:
            signed.append("node-1", "decision", {"selected": "cell"})
            self.assertTrue(signed.verify().valid)
        finally:
            signed.close()

    def test_records_come_back_in_write_order(self) -> None:
        # The burst detector reads timing from the sequence, so order matters.
        ledger = AuditLedger(self.path, signing_key=self.key)
        for index in range(5):
            ledger.append("node-1", "decision", {"n": index})
        rows = ledger.signed_records()
        ledger.close()
        self.assertEqual([row["payload"]["n"] for row in rows], [0, 1, 2, 3, 4])

    def test_the_limit_returns_the_most_recent_records(self) -> None:
        ledger = AuditLedger(self.path, signing_key=self.key)
        for index in range(10):
            ledger.append("node-1", "decision", {"n": index})
        rows = ledger.signed_records(limit=3)
        ledger.close()
        self.assertEqual([row["payload"]["n"] for row in rows], [7, 8, 9])


if __name__ == "__main__":
    unittest.main()

