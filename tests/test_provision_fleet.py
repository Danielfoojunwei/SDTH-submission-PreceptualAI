"""Fleet provisioning: node keys and an operator-signed roster.

Every cross-node capability needs a roster naming each node and carrying its
public key, and until now that file had to be hand-authored with PEM blocks
pasted into JSON. This tests the tool that removes that step, because a
provisioning error does not fail loudly — it produces a fleet that comes up,
runs, and silently ignores the node whose key was mistyped.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from kval_node.capabilities import ROSTER_DOCUMENT_TYPE
from kval_node.roster import PeerRoster
from kval_node.models import utc_now
from kval_node.signing import load_public_key, verify_document_agile

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "provision_fleet.py"
_spec = importlib.util.spec_from_file_location("provision_fleet", _SCRIPT)
assert _spec and _spec.loader
provision_fleet = importlib.util.module_from_spec(_spec)
sys.modules["provision_fleet"] = provision_fleet
_spec.loader.exec_module(provision_fleet)


class PositionParsingTests(unittest.TestCase):
    def test_absent_position_stays_absent(self) -> None:
        # A node that does not know where it is must not be given a default,
        # which would place the whole fleet on one point and drag every
        # emitter estimate toward it.
        self.assertIsNone(provision_fleet._parse_position(None, "n1"))

    def test_lat_lon_parses(self) -> None:
        self.assertEqual(
            provision_fleet._parse_position("1.2966,103.7764", "n1"),
            {"latitude_deg": 1.2966, "longitude_deg": 103.7764},
        )

    def test_altitude_is_optional_and_kept_when_given(self) -> None:
        self.assertEqual(
            provision_fleet._parse_position("1.3,103.8,15", "n1"),
            {"latitude_deg": 1.3, "longitude_deg": 103.8, "altitude_m": 15.0},
        )

    def test_a_malformed_position_is_refused_not_guessed(self) -> None:
        for bad in ("1.3", "1.3,103.8,15,9", "north,east", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                provision_fleet._parse_position(bad, "n1")


class AssignmentParsingTests(unittest.TestCase):
    def test_node_id_value_pairs_parse(self) -> None:
        self.assertEqual(
            provision_fleet._parse_assignments(["a=sub-node", "b=node"], "--role"),
            {"a": "sub-node", "b": "node"},
        )

    def test_a_pair_without_an_equals_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            provision_fleet._parse_assignments(["sub-node"], "--role")


class RosterBodyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pems = {"n1": "-----BEGIN PUBLIC KEY-----\nAAA\n-----END PUBLIC KEY-----\n",
                     "n2": "-----BEGIN PUBLIC KEY-----\nBBB\n-----END PUBLIC KEY-----\n"}

    def _body(self, **kwargs):
        defaults = dict(
            roster_id="fleet-test", roles={}, positions={"n1": None, "n2": None}, valid_days=30,
        )
        defaults.update(kwargs)
        return provision_fleet.build_roster_body(["n1", "n2"], self.pems, **defaults)

    def test_every_node_appears_exactly_once(self) -> None:
        peers = self._body()["peers"]
        self.assertEqual([peer["node_id"] for peer in peers], ["n1", "n2"])

    def test_each_peer_carries_its_own_key(self) -> None:
        # The transposition this tool exists to prevent: two nodes sharing one
        # key means either can sign as the other, and corroboration between
        # them is one node agreeing with itself.
        peers = {peer["node_id"]: peer["public_key_pem"] for peer in self._body()["peers"]}
        self.assertEqual(peers["n1"], self.pems["n1"])
        self.assertEqual(peers["n2"], self.pems["n2"])
        self.assertNotEqual(peers["n1"], peers["n2"])

    def test_a_node_without_a_position_carries_no_position_field(self) -> None:
        peers = {peer["node_id"]: peer for peer in self._body()["peers"]}
        self.assertNotIn("position", peers["n1"])

    def test_a_declared_position_survives_into_the_roster(self) -> None:
        body = self._body(positions={"n1": {"latitude_deg": 1.3, "longitude_deg": 103.8}, "n2": None})
        peers = {peer["node_id"]: peer for peer in body["peers"]}
        self.assertEqual(peers["n1"]["position"]["latitude_deg"], 1.3)

    def test_the_role_defaults_to_node_and_can_be_overridden(self) -> None:
        body = self._body(roles={"n2": "sub-node"})
        peers = {peer["node_id"]: peer["role"] for peer in body["peers"]}
        self.assertEqual(peers, {"n1": "node", "n2": "sub-node"})

    def test_expiry_is_after_creation_by_the_requested_span(self) -> None:
        body = self._body(valid_days=7)
        from kval_node.models import parse_timestamp
        span = parse_timestamp(body["expires_at"]) - parse_timestamp(body["created_at"])
        self.assertEqual(span, timedelta(days=7))


class EndToEndProvisioningTests(unittest.TestCase):
    """The whole tool, against the node's own verification path.

    Generating a roster the fleet would reject is the failure worth catching,
    so this verifies with the same calls a node makes rather than re-reading
    the JSON it just wrote.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.operator_private = self.root / "operator.private.pem"
        self.operator_public = self.root / "operator.public.pem"
        self.fleet = self.root / "fleet"

    def _run(self, *extra: str) -> int:
        argv = [
            "provision_fleet.py",
            "--node", "kval-node-01", "--node", "kval-node-02",
            "--operator-private-key", str(self.operator_private),
            "--operator-public-key", str(self.operator_public),
            "--generate-operator-key",
            "--output-dir", str(self.fleet),
            *extra,
        ]
        original, sys.argv = sys.argv, argv
        try:
            return provision_fleet.main()
        finally:
            sys.argv = original

    def _roster(self) -> PeerRoster:
        envelope = json.loads((self.fleet / "peer-roster.signed.json").read_text(encoding="utf-8"))
        payload, _ = verify_document_agile(
            envelope,
            expected_type=ROSTER_DOCUMENT_TYPE,
            ed25519_key=load_public_key(self.operator_public),
        )
        return PeerRoster.from_dict(payload)

    def test_the_roster_it_writes_verifies_against_the_operator_key(self) -> None:
        with _quiet():
            self._run()
        roster = self._roster()
        self.assertEqual(sorted(roster.by_id()), ["kval-node-01", "kval-node-02"])

    def test_each_node_gets_a_distinct_key_pair(self) -> None:
        with _quiet():
            self._run()
        first = (self.fleet / "kval-node-01" / "node.private.pem").read_bytes()
        second = (self.fleet / "kval-node-02" / "node.private.pem").read_bytes()
        self.assertNotEqual(first, second)

    def test_node_private_keys_are_not_world_readable(self) -> None:
        with _quiet():
            self._run()
        mode = (self.fleet / "kval-node-01" / "node.private.pem").stat().st_mode
        self.assertEqual(mode & 0o077, 0, "a node private key was group- or world-accessible")

    def test_a_declared_position_reaches_the_verified_roster(self) -> None:
        with _quiet():
            self._run("--position", "kval-node-01=1.2966,103.7764,12")
        peer = self._roster().by_id()["kval-node-01"]
        self.assertTrue(peer.has_position())
        self.assertAlmostEqual(peer.latitude_deg, 1.2966)
        self.assertAlmostEqual(peer.altitude_m, 12.0)

    def test_an_undeclared_position_stays_undeclared(self) -> None:
        with _quiet():
            self._run()
        self.assertFalse(self._roster().by_id()["kval-node-02"].has_position())

    def test_a_single_node_fleet_is_refused(self) -> None:
        # A roster of one cannot corroborate: the quorum it needs is itself.
        argv = [
            "provision_fleet.py", "--node", "only-one",
            "--operator-private-key", str(self.operator_private),
            "--operator-public-key", str(self.operator_public),
            "--generate-operator-key", "--output-dir", str(self.fleet),
        ]
        original, sys.argv = sys.argv, argv
        try:
            with self.assertRaises(SystemExit):
                provision_fleet.main()
        finally:
            sys.argv = original

    def test_duplicate_node_ids_are_refused(self) -> None:
        argv = [
            "provision_fleet.py", "--node", "same", "--node", "same",
            "--operator-private-key", str(self.operator_private),
            "--operator-public-key", str(self.operator_public),
            "--generate-operator-key", "--output-dir", str(self.fleet),
        ]
        original, sys.argv = sys.argv, argv
        try:
            with self.assertRaises(SystemExit):
                provision_fleet.main()
        finally:
            sys.argv = original

    def test_it_refuses_to_overwrite_an_existing_roster(self) -> None:
        # Re-running provisioning over a live fleet would mint new node keys
        # while the nodes still hold the old ones, and every report would
        # start failing its roster check.
        with _quiet():
            self._run()
        with self.assertRaises(SystemExit):
            self._run("--roster-id", "second-attempt")

    def test_a_position_naming_an_unknown_node_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._run("--position", "kval-node-99=1.0,2.0")

    def test_an_unknown_role_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self._run("--role", "kval-node-01=commander")


class _quiet:
    """Silence the tool's JSON summary during tests."""

    def __enter__(self):
        import io
        self._stdout, sys.stdout = sys.stdout, io.StringIO()
        return self

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        return False


if __name__ == "__main__":
    unittest.main()
