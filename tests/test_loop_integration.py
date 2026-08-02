"""The node's decision loop must actually use the capability modules.

Before this, spectrum, power, recovery, multimodal and clock integrity were
libraries with tests and no production caller: ``service.py`` imported none of
them and the running node behaved exactly as it had before they existed. These
tests pin the integration so that cannot silently regress.

``KvalNode.__init__`` runs the Jetson platform gate, so the tests build an
instance without ``__init__`` and attach only what the method under test needs.
Every method exercised here is the real production implementation.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

from kval_node.models import (
    AccessType,
    Catalog,
    LinkObservation,
    Profile,
    ServiceClass,
    TelemetrySnapshot,
)
from kval_node.multimodal import Modality, MultimodalFuser
from kval_node.optimizer import SPECTRUM_DEGRADED_MULTIPLIER, ProfileOptimizer
from kval_node.recovery import RecoveryAction
from kval_node.service import KvalNode
from kval_node.spectrum import BandStatus
from kval_node.standards import StandardsRequirement

NOW = datetime.now(timezone.utc)


def profile(profile_id: str, uuid: str, interface: str, bands: list[str] | None = None) -> Profile:
    return Profile.from_dict(
        {
            "profile_id": profile_id,
            "access_type": "wifi",
            "connection_uuid": uuid,
            "interface": interface,
            "probe_target": "127.0.0.1",
            "monetary_cost": 0.1,
            "energy_cost": 0.1,
            "minimum_dwell_seconds": 10,
            "maximum_switches_per_hour": 8,
            "allowed_service_classes": ["critical_data"],
            "standards": {"required_identifiers": []},
            "spectrum_band_ids": bands or [],
        }
    )


def observation(profile_id: str, interface: str, rtt: float = 20.0) -> LinkObservation:
    return LinkObservation(
        profile_id=profile_id,
        interface=interface,
        observed_at=NOW,
        interface_present=True,
        link_up=True,
        connection_active=True,
        rtt_ms=rtt,
        loss_percent=0.0,
        jitter_ms=1.0,
        rx_mbps=10.0,
        tx_mbps=1.0,
        signal_percent=90.0,
        access_technology=None,
        standards_verified=True,
        standards_evidence=(),
        standards_errors=(),
        evidence=("ip-link",),
    )


def bare_node(observation_max_age_seconds: float = 30.0) -> KvalNode:
    """A KvalNode without the Jetson platform gate, for helper-level tests."""
    node = object.__new__(KvalNode)
    node._link_machines = {}
    node.fuser = MultimodalFuser(window_seconds=observation_max_age_seconds)
    # Spectrum evidence is now carried across cycles where sensing is
    # unavailable, so the helpers need the retention state and the age bound.
    node._last_degraded_bands = frozenset()
    node._degraded_until = None
    node.config = SimpleNamespace(
        node_id="test-node",
        observation_max_age_seconds=observation_max_age_seconds,
        link_role="payload",
    )
    return node


class SpectrumInfluencesSelectionTests(unittest.TestCase):
    """Sensing must demote a contested band — and must never promote one."""

    def setUp(self) -> None:
        self.clean = profile("clean", "11111111-1111-4111-8111-111111111111", "wlan0", ["unii-1"])
        self.contested = profile("contested", "22222222-2222-4222-8222-222222222222", "wlan1", ["ism-2g4"])
        self.catalog = Catalog(
            "v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1),
            (self.contested, self.clean), "a" * 64, "key",
        )
        # Make the contested profile the better link on pure network evidence,
        # so any change in ranking is attributable to spectrum.
        self.snapshot = TelemetrySnapshot(
            NOW, None,
            (observation("contested", "wlan1", rtt=10.0), observation("clean", "wlan0", rtt=30.0)),
        )

    def test_without_spectrum_the_faster_link_wins(self) -> None:
        result = ProfileOptimizer().recommend(self.catalog, self.snapshot, ServiceClass.CRITICAL_DATA)
        self.assertEqual(result.selected_profile_id, "contested")

    def test_contested_band_demotes_its_profile(self) -> None:
        result = ProfileOptimizer().recommend(
            self.catalog, self.snapshot, ServiceClass.CRITICAL_DATA,
            degraded_band_ids=frozenset({"ism-2g4"}),
        )
        self.assertEqual(result.selected_profile_id, "clean")
        demoted = next(score for score in result.scores if score.profile_id == "contested")
        self.assertIn("SPECTRUM_BAND_CONTESTED", demoted.reasons)
        self.assertEqual(demoted.components["spectrum_penalty"], SPECTRUM_DEGRADED_MULTIPLIER)

    def test_a_profile_declaring_no_bands_is_unaffected(self) -> None:
        plain = profile("plain", "33333333-3333-4333-8333-333333333333", "wlan2")
        catalog = Catalog("v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1), (plain,), "a" * 64, "key")
        snapshot = TelemetrySnapshot(NOW, None, (observation("plain", "wlan2"),))
        with_bands = ProfileOptimizer().recommend(
            catalog, snapshot, ServiceClass.CRITICAL_DATA, degraded_band_ids=frozenset({"ism-2g4"})
        )
        without = ProfileOptimizer().recommend(catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertEqual(with_bands.scores[0].utility, without.scores[0].utility)

    def test_spectrum_cannot_promote_a_profile_without_probe_evidence(self) -> None:
        # A link with no probe evidence stays ineligible no matter how quiet its
        # band is: sensing may warn us off a medium, never onto one.
        blind = observation("clean", "wlan0")
        blind = LinkObservation(**{**blind.__dict__, "rtt_ms": None, "loss_percent": None, "jitter_ms": None})
        snapshot = TelemetrySnapshot(NOW, None, (blind,))
        catalog = Catalog("v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1), (self.clean,), "a" * 64, "key")
        result = ProfileOptimizer().recommend(catalog, snapshot, ServiceClass.CRITICAL_DATA)
        self.assertIsNone(result.selected_profile_id)


class DegradedBandExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node = bare_node()

    def test_contested_statuses_are_extracted(self) -> None:
        spectrum = {
            "available": True,
            "bands": [
                {"band_id": "a", "status": BandStatus.JAMMING_SUSPECTED.value, "confidence": 1.0},
                {"band_id": "b", "status": BandStatus.ANOMALOUS.value, "confidence": 0.8},
                {"band_id": "c", "status": BandStatus.CLEAR.value, "confidence": 1.0},
                {"band_id": "d", "status": BandStatus.CONGESTED.value, "confidence": 1.0},
            ],
        }
        self.assertEqual(self.node._degraded_band_ids(spectrum), frozenset({"a", "b"}))

    def test_zero_confidence_is_not_evidence_of_interference(self) -> None:
        spectrum = {
            "available": True,
            "bands": [{"band_id": "a", "status": BandStatus.JAMMING_SUSPECTED.value, "confidence": 0.0}],
        }
        self.assertEqual(self.node._degraded_band_ids(spectrum), frozenset())

    def test_unavailable_sensing_with_no_history_degrades_nothing(self) -> None:
        self.assertEqual(self.node._degraded_band_ids({"available": False, "reason": "x"}), frozenset())

    def test_a_refused_scan_does_not_erase_what_the_last_scan_saw(self) -> None:
        # The rate limiter refusing a scan is not evidence the band is clear.
        # Treating them alike made the node re-select a link it had just judged
        # jammed: with a 2 s poll against a 5 s minimum scan interval — the
        # shipped example config — that happened two cycles in every three.
        contested = {
            "available": True,
            "bands": [
                {"band_id": "ism-2g4", "status": BandStatus.JAMMING_SUSPECTED.value,
                 "confidence": 1.0}
            ],
        }
        self.assertEqual(self.node._degraded_band_ids(contested), frozenset({"ism-2g4"}))
        refused = {"available": False, "reason": "scan refused: below policy minimum"}
        self.assertEqual(self.node._degraded_band_ids(refused), frozenset({"ism-2g4"}))

    def test_retained_evidence_expires_rather_than_pinning_a_profile_forever(self) -> None:
        node = bare_node(observation_max_age_seconds=0.05)
        node._degraded_band_ids(
            {
                "available": True,
                "bands": [
                    {"band_id": "ism-2g4", "status": BandStatus.JAMMING_SUSPECTED.value,
                     "confidence": 1.0}
                ],
            }
        )
        time.sleep(0.1)
        self.assertEqual(
            node._degraded_band_ids({"available": False, "reason": "x"}), frozenset()
        )

    def test_a_clear_scan_clears_the_retained_evidence(self) -> None:
        self.node._degraded_band_ids(
            {
                "available": True,
                "bands": [
                    {"band_id": "ism-2g4", "status": BandStatus.JAMMING_SUSPECTED.value,
                     "confidence": 1.0}
                ],
            }
        )
        self.node._degraded_band_ids({"available": True, "bands": []})
        self.assertEqual(
            self.node._degraded_band_ids({"available": False, "reason": "x"}), frozenset()
        )


class RecoveryLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node = bare_node()

    def test_healthy_current_link_holds(self) -> None:
        snapshot = TelemetrySnapshot(NOW, "p", (observation("p", "wlan0"),))
        result = self.node._run_recovery(snapshot)
        self.assertTrue(result["available"])
        self.assertEqual(result["action"], RecoveryAction.HOLD.value)

    def test_down_link_drives_reconnect(self) -> None:
        down = observation("p", "wlan0")
        down = LinkObservation(**{**down.__dict__, "link_up": False})
        snapshot = TelemetrySnapshot(NOW, "p", (down,))
        result = self.node._run_recovery(snapshot)
        self.assertEqual(result["action"], RecoveryAction.RECONNECT.value)

    def test_state_persists_across_cycles(self) -> None:
        down = observation("p", "wlan0")
        down = LinkObservation(**{**down.__dict__, "link_up": False})
        first = self.node._run_recovery(TelemetrySnapshot(NOW, "p", (down,)))
        second = self.node._run_recovery(TelemetrySnapshot(NOW, "p", (down,)))
        self.assertGreater(second["backoff_seconds"], first["backoff_seconds"])

    def test_no_current_profile_is_unavailable_not_an_error(self) -> None:
        snapshot = TelemetrySnapshot(NOW, None, (observation("p", "wlan0"),))
        self.assertFalse(self.node._run_recovery(snapshot)["available"])


class FusedStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node = bare_node()
        self.snapshot = TelemetrySnapshot(NOW, "p", (observation("p", "wlan0"),))

    def test_network_and_security_modalities_are_always_present(self) -> None:
        state = self.node._fuse_state(
            self.snapshot, {"available": False}, {"available": False}, {"trusted": True}
        )
        self.assertIn("network", state["present_modalities"])
        self.assertIn("security", state["present_modalities"])

    def test_spectrum_and_power_join_when_available(self) -> None:
        spectrum = {
            "available": True,
            "policy_id": "sp",
            "policy_hash": "a" * 64,
            "bands": [{"band_id": "a", "status": BandStatus.CLEAR.value, "confidence": 1.0}],
            "clear_comms_band_ids": ["a"],
        }
        power = {
            "available": True,
            "policy_id": "pp",
            "policy_hash": "b" * 64,
            "reading": {"cpu_temp_c": 55.0, "module_watts": 8.0},
            "recommendation": {"max_inference_duty": 1.0},
        }
        state = self.node._fuse_state(self.snapshot, spectrum, power, {"trusted": True})
        self.assertIn("spectrum", state["present_modalities"])
        self.assertIn("power", state["present_modalities"])
        self.assertTrue(state["decision_ready"])

    def test_untrusted_clock_drops_the_security_modality_quality_to_zero(self) -> None:
        state = self.node._fuse_state(
            self.snapshot, {"available": False}, {"available": False},
            {"trusted": False, "observed_step_seconds": 900.0},
        )
        security = next(o for o in state["observations"] if o["modality"] == Modality.SECURITY.value)
        self.assertEqual(security["quality"], 0.0)
        self.assertEqual(security["features"]["clock_trusted"], 0.0)


class SubNodeBoundaryTests(unittest.TestCase):
    """A low-compute sub-node must be able to run the thin path off-Jetson.

    The boundary is real only if the thin modules and the capability entry point
    never import the Jetson platform gate. If that regresses, a drone build
    would start refusing to run on its own hardware.
    """

    THIN_MODULES = (
        "kval_node.validation",
        "kval_node.models",
        "kval_node.signing",
        "kval_node.lease",
        "kval_node.spectrum",
        "kval_node.power",
        "kval_node.recovery",
        "kval_node.optimizer",
        "kval_node.policy",
        "kval_node.audit",
        "kval_node.fabric",
        "kval_node.timesync",
        "kval_node.multimodal",
        "kval_node.dataset",
        "kval_node.sensor_adapter",
        "kval_node.capabilities",
    )

    def test_thin_path_does_not_import_the_jetson_gate(self) -> None:
        import importlib
        import subprocess
        import sys

        # A subprocess so an unrelated test that already imported service.py
        # cannot make this pass by accident.
        script = (
            "import sys; sys.path.insert(0, 'src');"
            "import importlib;"
            f"[importlib.import_module(m) for m in {list(self.THIN_MODULES)!r}];"
            "print('kval_node.hardware' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False", "thin path pulled in the Jetson platform gate")
        # Sanity: the modules really do import, so a typo cannot make this pass.
        for module in self.THIN_MODULES:
            importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
