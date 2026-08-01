from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from kval_node.audit import AuditLedger


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


if __name__ == "__main__":
    unittest.main()

