from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kval_node.radio_agent import ApprovedRadioProfile, RadioAgent, RadioAgentConfig, SignedEvidenceStore, _canonical_json


UUID = "11111111-1111-4111-8111-111111111111"


class RadioAgentTests(unittest.TestCase):
    def test_signed_evidence_binds_profile_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch("kval_node.radio_agent._secure_regular_file"):
            root = Path(directory)
            report = root / "report.pdf"
            report.write_bytes(b"laboratory conformance result")
            private_key = Ed25519PrivateKey.generate()
            public_key = root / "public.pem"
            public_key.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            capability = {
                "connection_uuid": UUID,
                "rat": "nr-ntn",
                "implementation_id": "vendor-radio-1.2.3",
                "standard_identifiers": ["3gpp-release-18"],
            }
            payload = {
                "profile_id": "approved-ntn",
                "connection_uuid": UUID,
                "conformance_report_path": str(report),
                "conformance_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "capability": capability,
            }
            signature = private_key.sign(_canonical_json(payload))
            evidence = root / "evidence.json"
            evidence.write_text(json.dumps({"payload": payload, "signature": base64.b64encode(signature).decode()}))
            loaded = SignedEvidenceStore(public_key).load(evidence, profile_id="approved-ntn", connection_uuid=UUID)
            self.assertEqual(loaded["implementation_id"], "vendor-radio-1.2.3")
            report.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "hash"):
                SignedEvidenceStore(public_key).load(evidence, profile_id="approved-ntn", connection_uuid=UUID)

    def test_unapproved_profile_fails_before_driver_execution(self) -> None:
        profile = ApprovedRadioProfile("ntn-primary", "approved-ntn", UUID, Path("/evidence"), ("/driver", "activate"), ("/driver", "status"))
        config = RadioAgentConfig(Path("/run/kval-radio/test.sock"), Path("/public.pem"), (profile,))
        with mock.patch.object(SignedEvidenceStore, "__init__", return_value=None):
            agent = RadioAgent(config)
        with self.assertRaisesRegex(ValueError, "not approved"):
            agent.handle({"operation": "activate", "profile_id": "other", "sdr_profile_id": "approved-ntn", "connection_uuid": UUID, "catalog_hash": "a" * 64})


if __name__ == "__main__":
    unittest.main()
