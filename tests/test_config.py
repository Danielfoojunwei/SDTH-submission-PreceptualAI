from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kval_node.config import NodeConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def base_config() -> dict:
    return {
        "node_id": "n1",
        "catalog_path": "/etc/kval/catalog.signed.json",
        "catalog_public_key_path": "/etc/kval/operator.public.pem",
        "audit_database_path": "/var/lib/kval/audit.db",
        "latest_result_path": "/run/kval/latest.json",
        "service_class": "critical_data",
        "poll_interval_seconds": 2.0,
        "observation_max_age_seconds": 8.0,
        "minimum_confidence": 0.75,
        "apply_network_changes": False,
    }


class NodeConfigTests(unittest.TestCase):
    def test_minimal_config_has_no_optional_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.json"
            path.write_text(json.dumps(base_config()), encoding="utf-8")
            config = NodeConfig.load(path)
            self.assertIsNone(config.spectrum_policy_path)
            self.assertIsNone(config.power_policy_path)
            self.assertIsNone(config.transport_policy_path)
            self.assertIsNone(config.lease_public_key_path)

    def test_optional_capabilities_load(self) -> None:
        raw = base_config()
        raw.update(
            {
                "spectrum_policy_path": "/etc/kval/spectrum.signed.json",
                "spectrum_sensor_socket": "/run/kval-spectrum/sensor.sock",
                "power_policy_path": "/etc/kval/power.signed.json",
                "transport_policy_path": "/etc/kval/transport.signed.json",
                "lease_public_key_path": "/etc/kval/operator.public.pem",
                "lease_nonce_database_path": "/var/lib/kval/nonces.db",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = NodeConfig.load(path)
            self.assertEqual(config.spectrum_policy_path, Path("/etc/kval/spectrum.signed.json"))
            self.assertEqual(config.spectrum_sensor_socket, Path("/run/kval-spectrum/sensor.sock"))
            self.assertEqual(config.lease_nonce_database_path, Path("/var/lib/kval/nonces.db"))

    def test_repository_example_configs_load(self) -> None:
        for name in ("node.example.json", "node.full-example.json"):
            config = NodeConfig.load(PROJECT_ROOT / "config" / name)
            self.assertTrue(config.node_id)


if __name__ == "__main__":
    unittest.main()
