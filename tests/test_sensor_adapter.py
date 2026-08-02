"""Parser tests for the real spectrum sensor adapter.

The fixtures below are verbatim-format samples of what ``iw``, ``nmcli``, and
``rtl_power`` actually print. They exercise the parsers only — there is no
simulated measurement backend, and nothing here is wired into the production
path. A node still obtains observations exclusively from a real adapter reading
real hardware.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from kval_node.sensor_adapter import (
    BandMeasurement,
    SpectrumSensorAdapter,
    adapter_conformance_hash,
    merge_measurements,
    occupancy_from_sweep,
    parse_iw_scan,
    parse_iw_survey,
    parse_rtl_power_csv,
)
from kval_node.spectrum import SpectrumBand, SpectrumObservation, SpectrumPolicy

NOW = datetime.now(timezone.utc)

IW_SURVEY = """Survey data from wlan0
\tfrequency:\t\t\t2412 MHz [in use]
\tnoise:\t\t\t\t-92 dBm
\tchannel active time:\t\t60000 ms
\tchannel busy time:\t\t18000 ms
\tchannel receive time:\t\t9000 ms
\tchannel transmit time:\t\t1200 ms
Survey data from wlan0
\tfrequency:\t\t\t2437 MHz
\tnoise:\t\t\t\t-95 dBm
\tchannel active time:\t\t60000 ms
\tchannel busy time:\t\t3000 ms
Survey data from wlan0
\tfrequency:\t\t\t5180 MHz
\tnoise:\t\t\t\t-104 dBm
\tchannel active time:\t\t60000 ms
\tchannel busy time:\t\t600 ms
"""

IW_SCAN = """BSS aa:bb:cc:dd:ee:01(on wlan0)
\tTSF: 1234567890 usec
\tfreq: 2412
\tbeacon interval: 100 TUs
\tsignal: -42.00 dBm
\tSSID: example-net
\tBSS Load:
\t\t * station count: 4
\t\t * channel utilisation: 51/255
\t\t * available admission capacity: 0 [*32us]
BSS aa:bb:cc:dd:ee:02(on wlan0)
\tfreq: 2412
\tsignal: -71.00 dBm
\tSSID: other-net
BSS aa:bb:cc:dd:ee:03(on wlan0)
\tfreq: 5180
\tsignal: -60.00 dBm
\tSSID: fast-net
"""

RTL_POWER_CSV = (
    "2026-08-02, 02:30:00, 2400000000, 2402000000, 500000.00, 12, "
    "-70.10, -69.80, -45.20, -70.30\n"
    "2026-08-02, 02:30:01, 2402000000, 2404000000, 500000.00, 12, "
    "-70.00, -71.10, -70.40, -69.90\n"
)


def band(band_id: str, low: int, high: int, **overrides) -> SpectrumBand:
    raw = {
        "band_id": band_id,
        "label": band_id,
        "freq_low_hz": low,
        "freq_high_hz": high,
        "max_resolution_bandwidth_hz": 1_000_000,
        "occupancy_baseline_percent": 20.0,
        "occupancy_alert_percent": 75.0,
        "noise_floor_baseline_dbm": -95.0,
        "comms_candidate": True,
    }
    raw.update(overrides)
    return SpectrumBand.from_dict(raw)


class IwSurveyParserTests(unittest.TestCase):
    def test_parses_every_channel_block(self) -> None:
        channels = parse_iw_survey(IW_SURVEY)
        self.assertEqual(len(channels), 3)
        self.assertEqual(channels[0].freq_hz, 2_412_000_000)
        self.assertEqual(channels[0].noise_dbm, -92.0)
        self.assertEqual(channels[0].active_ms, 60000)
        self.assertEqual(channels[0].busy_ms, 18000)

    def test_occupancy_is_measured_busy_over_active(self) -> None:
        channels = parse_iw_survey(IW_SURVEY)
        self.assertAlmostEqual(channels[0].occupancy_percent(), 30.0)
        self.assertAlmostEqual(channels[1].occupancy_percent(), 5.0)
        self.assertAlmostEqual(channels[2].occupancy_percent(), 1.0)

    def test_channel_without_counters_reports_no_occupancy(self) -> None:
        channels = parse_iw_survey("Survey data from wlan0\n\tfrequency:\t2412 MHz\n\tnoise:\t-90 dBm\n")
        self.assertEqual(len(channels), 1)
        self.assertIsNone(channels[0].occupancy_percent())

    def test_busy_exceeding_active_is_clamped(self) -> None:
        text = (
            "Survey data from wlan0\n\tfrequency:\t2412 MHz\n"
            "\tchannel active time:\t100 ms\n\tchannel busy time:\t150 ms\n"
        )
        self.assertEqual(parse_iw_survey(text)[0].occupancy_percent(), 100.0)

    def test_empty_output_yields_nothing(self) -> None:
        self.assertEqual(parse_iw_survey(""), [])


class IwScanParserTests(unittest.TestCase):
    def test_parses_bssids_freq_and_signal(self) -> None:
        stations = parse_iw_scan(IW_SCAN)
        self.assertEqual(len(stations), 3)
        self.assertEqual(stations[0].bssid, "aa:bb:cc:dd:ee:01")
        self.assertEqual(stations[0].freq_hz, 2_412_000_000)
        self.assertEqual(stations[0].signal_dbm, -42.0)

    def test_bss_load_channel_utilisation_is_scaled_from_255(self) -> None:
        stations = parse_iw_scan(IW_SCAN)
        self.assertAlmostEqual(stations[0].channel_utilisation_percent, 51 / 255 * 100.0)
        self.assertIsNone(stations[1].channel_utilisation_percent)

    def test_empty_output_yields_nothing(self) -> None:
        self.assertEqual(parse_iw_scan(""), [])


class RtlPowerParserTests(unittest.TestCase):
    def test_parses_bins_across_rows(self) -> None:
        bins = parse_rtl_power_csv(RTL_POWER_CSV)
        self.assertEqual(len(bins), 8)
        self.assertEqual(bins[0].freq_hz, 2_400_000_000)
        self.assertEqual(bins[1].freq_hz, 2_400_500_000)

    def test_occupancy_derives_from_the_sweep_itself(self) -> None:
        bins = parse_rtl_power_csv(RTL_POWER_CSV)
        occupancy, noise_floor, peak = occupancy_from_sweep(bins)
        self.assertEqual(peak, -45.2)
        # One bin of eight sits far above the floor.
        self.assertAlmostEqual(occupancy, 12.5)
        self.assertLess(noise_floor, -60.0)

    def test_empty_sweep_is_an_error_not_a_zero(self) -> None:
        with self.assertRaises(ValueError):
            occupancy_from_sweep([])

    def test_malformed_rows_are_skipped(self) -> None:
        self.assertEqual(parse_rtl_power_csv("not,a,valid,row\n"), [])


class MergeTests(unittest.TestCase):
    def test_first_backend_with_a_real_figure_wins(self) -> None:
        target = band("b", 2_400_000_000, 2_483_500_000)
        survey = BandMeasurement("b", 30.0, -92.0, None, 0, 1.0, ("iw-survey",), ())
        scan = BandMeasurement("b", 20.0, None, -42.0, 2, 1.0, ("iw-scan",), ())
        merged = merge_measurements(target, [survey, scan])
        self.assertEqual(merged.occupancy_percent, 30.0)
        self.assertEqual(merged.noise_floor_dbm, -92.0)
        self.assertEqual(merged.peak_power_dbm, -42.0)
        self.assertEqual(merged.detected_emitters, 2)
        self.assertEqual(merged.errors, ())

    def test_unmeasured_band_has_zero_quality_and_says_why(self) -> None:
        target = band("b", 2_400_000_000, 2_483_500_000)
        nothing = BandMeasurement("b", None, None, None, 0, 0.0, ("iw-scan",), ())
        merged = merge_measurements(target, [nothing])
        self.assertEqual(merged.quality, 0.0)
        self.assertTrue(any("occupancy" in e for e in merged.errors))
        self.assertTrue(any("noise floor" in e for e in merged.errors))


class AdapterContractTests(unittest.TestCase):
    """The adapter's output must satisfy the node's own strict parser."""

    class _Backend:
        name = "test-backend"

        def __init__(self, table):
            self._table = table

        def available(self) -> bool:
            return True

        def measure(self, bands):
            return self._table

    def _policy(self) -> SpectrumPolicy:
        return SpectrumPolicy.from_dict(
            {
                "policy_id": "p",
                "region": "US",
                "created_at": (NOW - timedelta(minutes=1)).isoformat(),
                "expires_at": (NOW + timedelta(days=1)).isoformat(),
                "bands": [
                    {
                        "band_id": "ism-2g4",
                        "label": "ISM 2.4",
                        "freq_low_hz": 2_400_000_000,
                        "freq_high_hz": 2_483_500_000,
                        "max_resolution_bandwidth_hz": 1_000_000,
                        "comms_candidate": True,
                    }
                ],
            },
            now=NOW,
        )

    def test_output_parses_as_a_spectrum_observation(self) -> None:
        measured = {"ism-2g4": BandMeasurement("ism-2g4", 30.0, -92.0, -42.0, 3, 1.0, ("iw-survey",), ())}
        adapter = SpectrumSensorAdapter([self._Backend(measured)], adapter_id="test-adapter")
        observations = adapter.observe(self._policy(), now=NOW)
        self.assertEqual(len(observations), 1)
        parsed = SpectrumObservation.from_dict(observations[0])
        self.assertTrue(parsed.passive)
        self.assertEqual(parsed.occupancy_percent, 30.0)
        self.assertEqual(parsed.noise_floor_dbm, -92.0)
        self.assertEqual(parsed.detected_emitters, 3)
        self.assertGreater(parsed.confidence(), 0.0)

    def test_unmeasured_band_yields_zero_confidence(self) -> None:
        measured = {"ism-2g4": BandMeasurement("ism-2g4", None, None, None, 0, 0.0, ("iw-scan",), ())}
        adapter = SpectrumSensorAdapter([self._Backend(measured)], adapter_id="test-adapter")
        parsed = SpectrumObservation.from_dict(adapter.observe(self._policy(), now=NOW)[0])
        self.assertEqual(parsed.confidence(), 0.0)

    def test_handle_rejects_non_passive_and_foreign_bands(self) -> None:
        measured = {"ism-2g4": BandMeasurement("ism-2g4", 10.0, -95.0, None, 1, 1.0, ("iw-survey",), ())}
        adapter = SpectrumSensorAdapter([self._Backend(measured)], adapter_id="test-adapter")
        policy = self._policy()
        base = {"operation": "scan", "mode": "passive", "policy_id": "p",
                "region": "US", "band_ids": ["ism-2g4"]}
        self.assertTrue(adapter.handle(base, policy)["success"])
        self.assertFalse(adapter.handle({**base, "mode": "active"}, policy)["success"])
        self.assertFalse(adapter.handle({**base, "band_ids": ["not-approved"]}, policy)["success"])
        self.assertFalse(adapter.handle({**base, "policy_id": "other"}, policy)["success"])
        self.assertFalse(adapter.handle({**base, "operation": "transmit"}, policy)["success"])

    def test_response_is_json_serializable(self) -> None:
        measured = {"ism-2g4": BandMeasurement("ism-2g4", 10.0, -95.0, None, 1, 1.0, ("iw-survey",), ())}
        adapter = SpectrumSensorAdapter([self._Backend(measured)], adapter_id="test-adapter")
        response = adapter.handle(
            {"operation": "scan", "mode": "passive", "policy_id": "p", "region": "US",
             "band_ids": ["ism-2g4"]},
            self._policy(),
        )
        json.dumps(response)

    def test_failing_backend_does_not_abort_the_scan(self) -> None:
        class Broken:
            name = "broken"

            def available(self) -> bool:
                return True

            def measure(self, bands):
                raise RuntimeError("radio disappeared")

        adapter = SpectrumSensorAdapter([Broken()], adapter_id="test-adapter")
        parsed = SpectrumObservation.from_dict(adapter.observe(self._policy(), now=NOW)[0])
        self.assertEqual(parsed.confidence(), 0.0)
        self.assertTrue(any("radio disappeared" in e for e in parsed.errors))

    def test_conformance_hash_is_a_real_digest(self) -> None:
        digest = adapter_conformance_hash()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(digest, adapter_conformance_hash())


class BandPlanTests(unittest.TestCase):
    def test_shipped_band_plans_parse_and_carry_sources(self) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "config" / "band-plans.json").read_text(encoding="utf-8"))
        sources = data["_sources"]
        self.assertTrue(sources)
        for region, entry in data["regions"].items():
            with self.subTest(region=region):
                self.assertTrue(entry["bands"])
                for raw in entry["bands"]:
                    parsed = SpectrumBand.from_dict(
                        {**raw, "occupancy_baseline_percent": 20.0,
                         "occupancy_alert_percent": 75.0,
                         "noise_floor_baseline_dbm": -95.0}
                    )
                    self.assertLess(parsed.freq_low_hz, parsed.freq_high_hz)
                    self.assertIn(raw["source"], sources)


if __name__ == "__main__":
    unittest.main()
