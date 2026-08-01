from __future__ import annotations

import unittest

from kval_node.standards import (
    RadioAccessTechnology,
    SliceSelection,
    StandardsRequirement,
    StandardsVerifier,
)
from kval_node.models import Profile


class StandardsTests(unittest.TestCase):
    def test_snssai_is_strictly_validated(self) -> None:
        selection = SliceSelection.from_dict({"sst": 1, "sd": "a1b2c3"})
        self.assertEqual(selection.to_dict(), {"sst": 1, "sd": "A1B2C3"})
        with self.assertRaises(ValueError):
            SliceSelection.from_dict({"sst": 300, "sd": "A1B2C3"})

    def test_advanced_standard_requires_live_agent(self) -> None:
        with self.assertRaisesRegex(ValueError, "live capability socket"):
            StandardsRequirement.from_dict(
                {
                    "required_identifiers": ["3gpp-release-18"],
                    "minimum_3gpp_release": 18,
                }
            )

    def test_rat_mapping_uses_modemmanager_vocabulary(self) -> None:
        self.assertTrue(StandardsVerifier._rat_matches(RadioAccessTechnology.EUTRAN, "lte"))
        self.assertTrue(StandardsVerifier._rat_matches(RadioAccessTechnology.NR, "lte,5gnr"))
        self.assertFalse(StandardsVerifier._rat_matches(RadioAccessTechnology.NR, "umts"))

    def test_3gpp_access_type_cannot_claim_wrong_rat(self) -> None:
        raw = {
            "profile_id": "bad-5g",
            "access_type": "5g",
            "connection_uuid": "11111111-1111-4111-8111-111111111111",
            "interface": "wwan0",
            "probe_target": "127.0.0.1",
            "monetary_cost": 0.1,
            "energy_cost": 0.2,
            "minimum_dwell_seconds": 30,
            "maximum_switches_per_hour": 5,
            "allowed_service_classes": ["telemetry"],
            "standards": {
                "required_identifiers": ["3gpp-ts-23.003"],
                "required_rat": "eutran",
                "dnn": "operator.example",
            },
        }
        with self.assertRaisesRegex(ValueError, "require RAT nr"):
            Profile.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
