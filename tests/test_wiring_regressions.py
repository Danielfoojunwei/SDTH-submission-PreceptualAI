"""Regressions for defects found by running the system, not by reading it.

Five end-to-end audits drove every CLI command and capability against real
signed fixtures. The suite was green throughout and the module-level
reachability guard passed. Everything below shipped anyway, which is the point:
these tests exist because unit coverage and import-graph analysis both missed
them.

Each test names the defect it pins and why it was invisible.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from kval_node.c2link import LinkRole
from kval_node.config import VALID_LINK_ROLES
from kval_node.dataset import Example
from kval_node.models import (
    AccessType,
    Catalog,
    Profile,
    Recommendation,
    ServiceClass,
    TelemetrySnapshot,
)
from kval_node.optimizer import ProfileOptimizer

NOW = datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)


def _profile(profile_id: str, uuid: str, interface: str, role: str) -> Profile:
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
            "link_role": role,
        }
    )


class ForecastCorpusContractTests(unittest.TestCase):
    """`forecast_links` read fields `Example` does not have.

    Two adjacent bugs — `example.profile_id` (the field is
    `selected_profile_id`) and `dict(zip(FEATURE_NAMES, example.features))` over
    a value that is already a dict. Invisible because the only test called it
    against an *empty* corpus, so the loop body never ran, and its assertions
    were guarded by `if result["available"]`. Fatal in production the moment one
    decision record gained a successor — and the ledger is durable, so every
    restart died in the same place.
    """

    def test_example_exposes_the_fields_the_forecaster_reads(self) -> None:
        example = Example(
            node_id="n",
            decided_at=NOW,
            selected_profile_id="wifi",
            features={"rtt_ms": 42.0},
            outcome_rtt_ms=42.0,
            outcome_loss_percent=0.0,
            outcome_link_up=True,
            label_improved=True,
            horizon_seconds=60.0,
        )
        self.assertFalse(hasattr(example, "profile_id"))
        self.assertEqual(example.selected_profile_id, "wifi")

    def test_features_is_a_mapping_not_a_sequence(self) -> None:
        # zip() over this walks the keys and yields the field *names* as values,
        # so float() raised on the string 'rtt_ms'.
        example = Example("n", NOW, "wifi", {"rtt_ms": 42.0}, 42.0, 0.0, True, True, 60.0)
        self.assertIsInstance(example.features, dict)
        self.assertEqual(example.features.get("rtt_ms"), 42.0)

    def test_a_populated_corpus_actually_produces_a_forecast(self) -> None:
        from kval_node.orchestration import forecast_advice

        rising = [20.0, 22.0, 25.0, 29.0, 34.0, 40.0, 47.0, 55.0]
        advice = forecast_advice({"wifi": rising})
        self.assertIn("wifi", advice.degrading_profiles)


class SuppressedRoleBranchTests(unittest.TestCase):
    """The emission-control branch had never executed and crashed when it did.

    It constructed `Recommendation(reasons=...)` — the field is `reason_codes` —
    and omitted five required fields. Nothing passed `suppressed_roles`, so the
    branch was unreachable and the `TypeError` invisible. Wiring EMCON in would
    have crashed the cycle at exactly the moment the control exists for.
    """

    def setUp(self) -> None:
        self.catalog = Catalog(
            "v1", NOW - timedelta(minutes=1), NOW + timedelta(days=1),
            (
                _profile("c2", "11111111-1111-4111-8111-111111111111", "wlan0", "c2_cnpc"),
                _profile("pay", "22222222-2222-4222-8222-222222222222", "wlan1", "payload"),
            ),
            "a" * 64, "key",
        )
        self.snapshot = TelemetrySnapshot(NOW, None, ())

    def test_a_suppressed_role_returns_a_valid_recommendation(self) -> None:
        result = ProfileOptimizer().recommend(
            self.catalog,
            self.snapshot,
            ServiceClass.CRITICAL_DATA,
            role="payload",
            suppressed_roles=frozenset({"payload"}),
        )
        self.assertIsInstance(result, Recommendation)
        self.assertIsNone(result.selected_profile_id)
        self.assertIn(
            "EMISSION_CONTROL_SUPPRESSES_THIS_ROLE_SO_NO_LINK_IS_SELECTED", result.reason_codes
        )

    def test_the_suppressed_recommendation_serialises(self) -> None:
        result = ProfileOptimizer().recommend(
            self.catalog, self.snapshot, ServiceClass.CRITICAL_DATA,
            role="payload", suppressed_roles=frozenset({"payload"}),
        )
        json.dumps(result.to_dict(), default=str)

    def test_an_unsuppressed_role_is_unaffected(self) -> None:
        result = ProfileOptimizer().recommend(
            self.catalog, self.snapshot, ServiceClass.CRITICAL_DATA,
            role="payload", suppressed_roles=frozenset({"bulk"}),
        )
        self.assertNotIn(
            "EMISSION_CONTROL_SUPPRESSES_THIS_ROLE_SO_NO_LINK_IS_SELECTED", result.reason_codes
        )


class LinkRoleReachabilityTests(unittest.TestCase):
    """A `c2_cnpc` profile was permanently unselectable by the autonomous loop.

    `role` defaulted to `"payload"` and the service never passed it, so every
    command link scored -1.0 / DIFFERENT_LINK_ROLE forever. There was also no
    config field to set it, so it was unreachable by configuration rather than
    merely unset.
    """

    def test_the_config_vocabulary_matches_the_link_role_enum(self) -> None:
        self.assertEqual(VALID_LINK_ROLES, {role.value for role in LinkRole})

    def test_a_node_can_be_configured_to_hold_the_command_link(self) -> None:
        from kval_node.config import _link_role

        self.assertEqual(_link_role({"link_role": "c2_cnpc"}), "c2_cnpc")

    def test_an_unknown_role_is_refused_rather_than_defaulted(self) -> None:
        from kval_node.config import _link_role

        with self.assertRaises(ValueError):
            _link_role({"link_role": "whatever"})

    def test_the_default_is_payload_so_nothing_becomes_safety_critical_by_omission(self) -> None:
        from kval_node.config import _link_role

        self.assertEqual(_link_role({}), "payload")

    def test_the_public_catalogue_view_shows_which_link_is_the_command_link(self) -> None:
        profile = _profile("c2", "11111111-1111-4111-8111-111111111111", "wlan0", "c2_cnpc")
        self.assertEqual(profile.to_public_dict()["link_role"], "c2_cnpc")





class JsonValidityTests(unittest.TestCase):
    """Refusals emitted bare `Infinity`, which RFC 8259 does not permit.

    Python and jq tolerate it; a Go or JavaScript consumer rejects the whole
    document. Every declined track prediction produced one.
    """

    def test_an_infinite_uncertainty_is_not_emitted_as_bare_infinity(self) -> None:
        with self.assertRaises(ValueError):
            json.dumps({"uncertainty_radius_m": math.inf}, allow_nan=False)

    def test_the_cli_printer_refuses_non_finite_values(self) -> None:
        from kval_node import cli

        with self.assertRaises(ValueError):
            cli._print({"uncertainty_radius_m": math.inf})



class ExitCodeContractTests(unittest.TestCase):
    """`_print_capability` only gated on `verified` and `valid`.

    `audit-memory` reports under `trustworthy` and exited 0 on a ledger it had
    just proven poisoned — the exact opposite of its own docstring's promise.
    """

    def _exit_code(self, payload: dict) -> int:
        from kval_node import cli

        try:
            cli._print_capability(payload)
        except SystemExit as exc:
            return int(exc.code)
        return 0

    def test_a_poisoned_ledger_does_not_exit_zero(self) -> None:
        self.assertEqual(self._exit_code({"available": True, "trustworthy": False}), 4)

    def test_a_clean_ledger_exits_zero(self) -> None:
        self.assertEqual(self._exit_code({"available": True, "trustworthy": True}), 0)

    def test_a_refused_renewal_does_not_exit_zero(self) -> None:
        self.assertEqual(self._exit_code({"available": True, "granted": False}), 4)

    def test_unavailable_still_exits_three(self) -> None:
        self.assertEqual(self._exit_code({"available": False, "reason": "x"}), 3)


class ReputationBlastRadiusTests(unittest.TestCase):
    """Excising a node that never contributed used to cost it its standing.

    A typo'd node ID silently degraded a healthy node, and the certificate said
    NODE_WAS_NOT_A_CONTRIBUTOR at the same time.
    """

    def setUp(self) -> None:
        from kval_node.dataset import FEATURE_NAMES
        from kval_node.federated import FeatureStatistics, FederatedUpdate

        self._directory = tempfile.TemporaryDirectory()
        self.path = Path(self._directory.name) / "reputation.json"
        self.fleet = [
            FederatedUpdate(
                f"n{index}",
                NOW,
                50,
                {name: FeatureStatistics(50, 0.0, 50.0) for name in FEATURE_NAMES},
            )
            for index in range(4)
        ]

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_a_typo_does_not_degrade_an_innocent_node(self) -> None:
        from kval_node.orchestration import FleetLearning

        learning = FleetLearning(self.path)
        learning.aggregate(self.fleet)
        before = learning.reputation.score("n0")
        _remaining, certificate = learning.excise(self.fleet, "nonexistent")
        self.assertFalse(certificate.was_present)
        self.assertEqual(learning.reputation.score("n0"), before)
        self.assertEqual(learning.reputation.score("nonexistent"), 1.0)

    def test_a_real_excision_still_costs_standing(self) -> None:
        from kval_node.orchestration import FleetLearning

        learning = FleetLearning(self.path)
        learning.aggregate(self.fleet)
        before = learning.reputation.score("n0")
        _remaining, certificate = learning.excise(self.fleet, "n0")
        self.assertTrue(certificate.was_present)
        self.assertLess(learning.reputation.score("n0"), before)


class LmsProvisioningTests(unittest.TestCase):
    """The LMS key generator had no loader, so its output was unusable.

    `generate_lms_key.py` wrote `seed || identifier` with neither type code, and
    nothing could rebuild the tree. The whole CNSA 2.0 signing half was
    therefore unreachable from production regardless of how correct it was.
    """

    def test_a_generated_key_can_be_loaded_back(self) -> None:
        from kval_node.pqc import (
            LMS_SHA256_M32_H5,
            deserialise_private_key,
            generate_lms_key,
            serialise_private_key,
        )

        key = generate_lms_key(lms_type=LMS_SHA256_M32_H5)
        restored = deserialise_private_key(serialise_private_key(key))
        self.assertEqual(restored.public_key(), key.public_key())

    def test_a_truncated_blob_is_refused(self) -> None:
        from kval_node.pqc import PqcError, deserialise_private_key

        with self.assertRaises(PqcError):
            deserialise_private_key(b"\x00" * 10)

    def test_state_rollback_is_caught_across_a_restart(self) -> None:
        # The class promised to catch a restored backup. The high-water mark
        # was read from the state file itself, so it was per-process and a
        # restart reissued spent leaves in silence.
        from kval_node.pqc import (
            LMS_SHA256_M32_H5,
            LmsSigner,
            LmsStateStore,
            StateRollback,
            generate_lms_key,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            key = generate_lms_key(lms_type=LMS_SHA256_M32_H5)
            first = LmsSigner(key, LmsStateStore(path))
            first.sign(b"a")
            first.sign(b"b")
            backup = path.read_text(encoding="utf-8")
            LmsSigner(key, LmsStateStore(path)).sign(b"c")
            path.write_text(backup, encoding="utf-8")  # restore the backup
            with self.assertRaises(StateRollback):
                LmsSigner(key, LmsStateStore(path)).sign(b"d")


class AnchorIdentificationTests(unittest.TestCase):
    """The CNSA verdict was derived from the filename and never opened the file.

    `path.suffix == ".lms"` meant `cp operator.pem decoy.lms` produced
    EVERY_CONFIGURED_TRUST_ANCHOR_USES_A_CNSA_2_0_APPROVED_ALGORITHM on a wholly
    classical node — a compliance report changeable with `mv`.
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _ed25519_anchor(self, name: str) -> Path:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        path = self.root / name
        path.write_bytes(
            Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        return path

    def _lms_anchor(self, name: str) -> Path:
        import base64

        from kval_node.pqc import LMS_SHA256_M32_H5, generate_lms_key

        path = self.root / name
        key = generate_lms_key(lms_type=LMS_SHA256_M32_H5)
        path.write_text(
            base64.b64encode(key.public_key().to_bytes()).decode("ascii"), encoding="utf-8"
        )
        return path

    def test_renaming_an_ed25519_anchor_does_not_make_it_compliant(self) -> None:
        from kval_node.capabilities import _anchor_algorithm

        algorithm, _reason = _anchor_algorithm(self._ed25519_anchor("decoy.lms"))
        self.assertEqual(algorithm, "Ed25519")

    def test_a_real_lms_anchor_is_recognised_whatever_it_is_called(self) -> None:
        from kval_node.capabilities import _anchor_algorithm

        for name in ("anchor.pub", "anchor.pem", "anchor.LMS", "anchor"):
            algorithm, _reason = _anchor_algorithm(self._lms_anchor(name))
            self.assertEqual(algorithm, "LMS_SHA256_M32", f"{name} was misidentified")

    def test_an_unloadable_anchor_is_not_reported_as_approved(self) -> None:
        from kval_node.capabilities import _anchor_algorithm

        path = self.root / "junk.lms"
        path.write_text("not a key at all", encoding="utf-8")
        algorithm, reason = _anchor_algorithm(path)
        self.assertIsNone(algorithm)
        self.assertIn("CANNOT_BE_USED", reason)


class GrantSerialisationTests(unittest.TestCase):
    """A grant could be built in Python and nowhere else.

    With no `from_dict`, quorum renewal had no operator-drivable surface, so the
    partition-tolerance story was not operable by anyone.
    """

    def test_a_grant_round_trips(self) -> None:
        from kval_node.threshold_ca import Grant, scope_hash

        grant = Grant(
            grant_id="g1",
            subject_node_id="node-0",
            scope_hash=scope_hash({"bands": ["b"]}),
            issued_at=NOW - timedelta(hours=1),
            expires_at=NOW + timedelta(minutes=5),
        )
        self.assertEqual(Grant.from_dict(grant.to_dict()), grant)

    def test_an_attestation_round_trips(self) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        from kval_node.threshold_ca import Attestation, Grant, attest, scope_hash

        grant = Grant("g1", "node-0", scope_hash({"a": 1}), NOW - timedelta(hours=1), NOW)
        attestation = attest(
            grant,
            Ed25519PrivateKey.generate(),
            certifier_node_id="node-1",
            new_expiry=NOW + timedelta(hours=1),
            now=NOW,
        )
        self.assertEqual(Attestation.from_dict(attestation.to_dict()), attestation)


if __name__ == "__main__":
    unittest.main()


class FlightTierSelfContainmentTests(unittest.TestCase):
    """The loop constructs ``FlightServices``; its methods must work on it.

    The flight/operator split moved ``emcon_posture``, ``plan_power`` and
    ``c2_status_from_telemetry`` into ``FlightServices`` but left four members
    they call behind in ``CapabilityServices``. A base class cannot borrow from
    its subclass, so all three broke in flight while every static check stayed
    green:

    * ``emcon_posture`` caught its own ``AttributeError`` and returned
      ``available: False``, which ``_suppressed_roles`` treats as "no order" —
      a signed EMCON SILENT order stopped suppressing any link role.
    * ``plan_power`` raised ``NameError`` every cycle, so power governance was
      permanently unavailable and ``power_duty_limit`` recorded 1.0.
    * ``c2_status_from_telemetry`` raised ``AttributeError``.

    Invisible because no test had ever instantiated ``FlightServices`` — the
    suite exercised only the subclass the CLI builds — and because the
    reachability guard checks that names have callers, never that a body
    resolves.
    """

    def test_flight_services_defines_every_member_its_methods_call(self) -> None:
        from kval_node.flight import FlightServices

        for member in (
            "load_emcon_order",
            "_emcon_or_none",
            "c2_status",
            "_catalog_profiles",
            "_c2_monitor",
        ):
            with self.subTest(member=member):
                self.assertTrue(
                    hasattr(FlightServices, member),
                    f"FlightServices.{member} is missing: the loop constructs "
                    "FlightServices, so calling it raises AttributeError in flight",
                )

    def test_tegrastats_reader_resolves_in_the_flight_module(self) -> None:
        import kval_node.flight as flight

        self.assertTrue(
            hasattr(flight, "read_tegrastats_sample"),
            "plan_power calls read_tegrastats_sample() as a bare global; if it is "
            "not in flight.py's namespace the loop raises NameError every cycle",
        )

    def test_emcon_posture_reaches_the_order_loader_on_the_flight_tier(self) -> None:
        from kval_node.flight import FlightServices

        services = FlightServices.__new__(FlightServices)
        services.config = SimpleNamespace(emcon_order_path=None)
        posture = services.emcon_posture(now=NOW)
        # No order configured is UNRESTRICTED and available. The regression
        # returned available=False with an AttributeError in the reason, which
        # _suppressed_roles cannot distinguish from "no order provisioned".
        self.assertTrue(
            posture.get("available"),
            f"emcon_posture degraded on the flight tier: {posture.get('reason')!r}",
        )
        self.assertNotIn("attribute", str(posture.get("reason", "")).lower())
