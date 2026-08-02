"""End-to-end wire test: sensor daemon socket <-> node's SpectrumSensorClient.

Exercises the real ``kval-spectrum-agent/v1`` socket path — the daemon's request
handler, the node's strict client, and the analyzer — so the contract is proven
against the actual implementations rather than assumed.

The measurement backend is supplied by the test because this host has no radio.
Everything downstream of it is production code: the same adapter, the same
handler, the same client, the same policy checks. Backends that read real
hardware live in ``kval_node.sensor_adapter`` and are covered by the parser
tests in ``test_sensor_adapter.py``.
"""

from __future__ import annotations

import os
import socketserver
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kval_node.models import utc_now
from kval_node.sensor_adapter import BandMeasurement, SpectrumSensorAdapter
from kval_node.sensor_cli import _Handler, _Server
from kval_node.spectrum import BandStatus, SpectrumAnalyzer, SpectrumPolicy, SpectrumSensorClient

NOW = datetime.now(timezone.utc)


def policy() -> SpectrumPolicy:
    return SpectrumPolicy.from_dict(
        {
            "policy_id": "wire-test",
            "region": "US",
            "created_at": (NOW - timedelta(minutes=1)).isoformat(),
            "expires_at": (NOW + timedelta(days=1)).isoformat(),
            "min_scan_interval_seconds": 0.05,
            "bands": [
                {
                    "band_id": "ism-2g4",
                    "label": "ISM 2.4 GHz",
                    "freq_low_hz": 2400000000,
                    "freq_high_hz": 2483500000,
                    "max_resolution_bandwidth_hz": 1000000,
                    "occupancy_baseline_percent": 20.0,
                    "occupancy_alert_percent": 75.0,
                    "noise_floor_baseline_dbm": -95.0,
                    "comms_candidate": True,
                },
                {
                    "band_id": "unii-1",
                    "label": "U-NII-1",
                    "freq_low_hz": 5150000000,
                    "freq_high_hz": 5250000000,
                    "max_resolution_bandwidth_hz": 2000000,
                    "occupancy_baseline_percent": 10.0,
                    "occupancy_alert_percent": 70.0,
                    "noise_floor_baseline_dbm": -98.0,
                    "comms_candidate": True,
                },
            ],
        },
        now=NOW,
    )


class _FixedBackend:
    """Supplies pre-set BandMeasurements so the wire path can be exercised.

    A stand-in for radio hardware, because the socket contract cannot be tested
    without something on the far end of it. It declares itself passive: it
    stands in for a receive-only backend, and a backend that declares nothing is
    counted as emitting.
    """

    name = "wire-test-backend"
    passive = True

    def __init__(self, table: dict[str, BandMeasurement]) -> None:
        self._table = table

    def available(self) -> bool:
        return True

    def measure(self, bands):
        return {band.band_id: self._table[band.band_id] for band in bands if band.band_id in self._table}


@unittest.skipUnless(os.geteuid() == 0, "client requires a root-owned socket")
class SensorSocketTests(unittest.TestCase):
    def _serve(self, measurements: dict[str, BandMeasurement]) -> tuple[Path, "_Server"]:
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: None)
        path = Path(directory) / "sensor.sock"
        adapter = SpectrumSensorAdapter([_FixedBackend(measurements)], adapter_id="wire-test-adapter")
        previous = os.umask(0o077)
        try:
            server = _Server(str(path), _Handler)
        finally:
            os.umask(previous)
        server.adapter = adapter
        server.policy = self.policy
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return path, server

    def setUp(self) -> None:
        self.policy = policy()

    def test_full_path_from_socket_to_intelligence(self) -> None:
        measurements = {
            "ism-2g4": BandMeasurement("ism-2g4", 92.0, -80.0, -40.0, 5, 1.0, ("iw-survey",), ()),
            "unii-1": BandMeasurement("unii-1", 4.0, -98.0, -70.0, 1, 1.0, ("iw-survey",), ()),
        }
        path, _server = self._serve(measurements)

        observations = SpectrumSensorClient().scan(path, self.policy)
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(item.passive for item in observations))

        # Analyse at the time the scan was actually taken. The adapter stamps
        # observations with the real clock, so analysing at a fixed module-level
        # timestamp would (correctly) have them discarded as future-dated.
        intel = SpectrumAnalyzer().analyze(self.policy, observations, now=utc_now())
        by_band = {band.band_id: band for band in intel.bands}
        # 92% occupancy with a 15 dB noise-floor rise on 2.4 GHz.
        self.assertEqual(by_band["ism-2g4"].status, BandStatus.JAMMING_SUSPECTED)
        self.assertTrue(intel.jamming_suspected)
        # 5 GHz is quiet and stays a usable comms candidate.
        self.assertEqual(by_band["unii-1"].status, BandStatus.CLEAR)
        self.assertEqual(intel.clear_comms_band_ids, ("unii-1",))
        self.assertTrue(intel.presence_detected)

    def test_unmeasured_band_reaches_the_node_as_insufficient(self) -> None:
        measurements = {
            "ism-2g4": BandMeasurement("ism-2g4", None, None, None, 0, 0.0, ("iw-scan",), ()),
            "unii-1": BandMeasurement("unii-1", 4.0, -98.0, None, 0, 1.0, ("iw-survey",), ()),
        }
        path, _server = self._serve(measurements)
        observations = SpectrumSensorClient().scan(path, self.policy)
        intel = SpectrumAnalyzer().analyze(self.policy, observations, now=utc_now())
        by_band = {band.band_id: band for band in intel.bands}
        self.assertEqual(by_band["ism-2g4"].status, BandStatus.INSUFFICIENT_EVIDENCE)
        self.assertNotIn("ism-2g4", intel.clear_comms_band_ids)

    def test_client_rejects_a_socket_that_is_world_writable(self) -> None:
        measurements = {
            "ism-2g4": BandMeasurement("ism-2g4", 10.0, -95.0, None, 0, 1.0, ("iw-survey",), ()),
            "unii-1": BandMeasurement("unii-1", 4.0, -98.0, None, 0, 1.0, ("iw-survey",), ()),
        }
        path, _server = self._serve(measurements)
        os.chmod(path, 0o777)
        with self.assertRaisesRegex(RuntimeError, "group/world writable"):
            SpectrumSensorClient().scan(path, self.policy)


if __name__ == "__main__":
    unittest.main()
