from __future__ import annotations

import base64
import unittest
from datetime import datetime, timedelta, timezone

from kval_node.transport_security import (
    NftablesPolicy,
    TransportPolicy,
    WifiSecurityPolicy,
    WireguardPolicy,
    assert_no_secret_material,
    hash_nft_ruleset,
    output_chain_defaults_drop,
    parse_wg_show,
    parse_handshake_age,
    verify_default_routes,
    verify_nftables,
    verify_wifi_security,
    verify_wireguard,
)

NOW = datetime.now(timezone.utc)
KEY_A = base64.b64encode(bytes([1]) * 32).decode("ascii")
KEY_B = base64.b64encode(bytes([2]) * 32).decode("ascii")


def wg_show(key: str = KEY_A, handshake: str = "30 seconds ago") -> str:
    return f"""interface: wg0
  public key: {base64.b64encode(bytes([9]) * 32).decode()}
  private key: (hidden)
  listening port: 51820

peer: {key}
  endpoint: 203.0.113.1:51820
  allowed ips: 10.66.0.1/32, fd66::1/128
  latest handshake: {handshake}
  transfer: 1.00 KiB received, 2.00 KiB sent
"""


def wg_policy() -> WireguardPolicy:
    return WireguardPolicy.from_dict(
        {
            "interface": "wg0",
            "require_default_route": True,
            "peers": [{"public_key": KEY_A, "allowed_ips": ["10.66.0.1/32", "fd66::1/128"]}],
        }
    )


def nft_ruleset(policy: str = "drop", handle: int = 5, packets: int = 10) -> dict:
    return {
        "nftables": [
            {"metainfo": {"version": "1.0.9"}},
            {"table": {"family": "inet", "name": "filter", "handle": 1}},
            {
                "chain": {
                    "family": "inet",
                    "table": "filter",
                    "name": "output",
                    "handle": 2,
                    "type": "filter",
                    "hook": "output",
                    "prio": 0,
                    "policy": policy,
                }
            },
            {
                "rule": {
                    "family": "inet",
                    "table": "filter",
                    "chain": "output",
                    "handle": handle,
                    "expr": [{"counter": {"packets": packets, "bytes": packets * 84}}],
                }
            },
        ]
    }


class TransportPolicyTests(unittest.TestCase):
    def test_at_least_one_section_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one section"):
            TransportPolicy.from_dict(
                {
                    "policy_id": "t",
                    "created_at": (NOW - timedelta(minutes=1)).isoformat(),
                    "expires_at": (NOW + timedelta(days=1)).isoformat(),
                },
                now=NOW,
            )

    def test_enterprise_192_requires_eap(self) -> None:
        with self.assertRaisesRegex(ValueError, "WPA3-Enterprise-192"):
            WifiSecurityPolicy.from_dict({"required_key_mgmt": ["wpa-eap-suite-b-192"]})


class WireguardParseTests(unittest.TestCase):
    def test_parse_and_verify_match(self) -> None:
        parsed = parse_wg_show(wg_show())
        ok, evidence, errors = verify_wireguard(wg_policy(), parsed)
        self.assertTrue(ok, errors)
        self.assertIn("PEER_ALLOWED_IPS_MATCH", evidence)
        self.assertIn("PEER_HANDSHAKE_RECENT", evidence)

    def test_unexpected_peer_fails(self) -> None:
        parsed = parse_wg_show(wg_show(key=KEY_B))
        ok, _evidence, errors = verify_wireguard(wg_policy(), parsed)
        self.assertFalse(ok)
        self.assertTrue(any("missing" in e or "unexpected" in e for e in errors))

    def test_no_handshake_fails(self) -> None:
        parsed = parse_wg_show(wg_show(handshake="never"))
        ok, _evidence, errors = verify_wireguard(wg_policy(), parsed)
        self.assertFalse(ok)
        self.assertTrue(any("handshake" in e for e in errors))

    def test_stale_handshake_fails_closed(self) -> None:
        # A days-old handshake means the tunnel is not carrying live traffic.
        parsed = parse_wg_show(wg_show(handshake="5 days, 3 hours, 2 minutes ago"))
        ok, _evidence, errors = verify_wireguard(wg_policy(), parsed)
        self.assertFalse(ok)
        self.assertTrue(any("older than the freshness limit" in e for e in errors))

    def test_unparseable_handshake_fails_closed(self) -> None:
        parsed = parse_wg_show(wg_show(handshake="(unknown)"))
        ok, _evidence, _errors = verify_wireguard(wg_policy(), parsed)
        self.assertFalse(ok)

    def test_handshake_age_parsing(self) -> None:
        self.assertEqual(parse_handshake_age("30 seconds ago"), 30.0)
        self.assertEqual(parse_handshake_age("1 minute, 30 seconds ago"), 90.0)
        self.assertEqual(parse_handshake_age("2 days ago"), 172800.0)
        self.assertIsNone(parse_handshake_age("never"))
        self.assertIsNone(parse_handshake_age(""))

    def test_multiple_interfaces_do_not_bleed(self) -> None:
        output = wg_show() + "\n" + wg_show(key=KEY_B).replace("interface: wg0", "interface: wg1")
        parsed = parse_wg_show(output)
        self.assertEqual(len(parsed["wg0"]), 1)
        self.assertEqual(len(parsed["wg1"]), 1)
        self.assertEqual(parsed["wg0"][0].public_key, KEY_A)
        self.assertEqual(parsed["wg1"][0].public_key, KEY_B)


class NftablesTests(unittest.TestCase):
    def test_hash_is_stable_across_counters_and_handles(self) -> None:
        a = hash_nft_ruleset(nft_ruleset(handle=5, packets=10))
        b = hash_nft_ruleset(nft_ruleset(handle=99, packets=9999))
        self.assertEqual(a, b)

    def test_output_default_drop_detected(self) -> None:
        self.assertTrue(output_chain_defaults_drop(nft_ruleset(policy="drop")))
        self.assertFalse(output_chain_defaults_drop(nft_ruleset(policy="accept")))

    def test_verify_hash_and_drop(self) -> None:
        ruleset = nft_ruleset(policy="drop")
        expected = hash_nft_ruleset(ruleset)
        policy = NftablesPolicy.from_dict(
            {"expected_ruleset_sha256": expected, "require_default_drop_output": True}
        )
        ok, evidence, errors = verify_nftables(policy, ruleset)
        self.assertTrue(ok, errors)
        self.assertIn("NFT_OUTPUT_DEFAULT_DROP", evidence)
        self.assertIn("NFT_RULESET_HASH_MATCH", evidence)

    def test_accept_output_fails(self) -> None:
        policy = NftablesPolicy.from_dict({"require_default_drop_output": True})
        ok, _evidence, errors = verify_nftables(policy, nft_ruleset(policy="accept"))
        self.assertFalse(ok)

    def test_ipv6_only_drop_chain_does_not_cover_ipv4(self) -> None:
        ruleset = nft_ruleset(policy="drop")
        for item in ruleset["nftables"]:
            if isinstance(item.get("chain"), dict):
                item["chain"]["family"] = "ip6"
        self.assertFalse(output_chain_defaults_drop(ruleset))

    def test_separate_ip_and_ip6_chains_cover_both(self) -> None:
        v4 = nft_ruleset(policy="drop")["nftables"]
        for item in v4:
            if isinstance(item.get("chain"), dict):
                item["chain"]["family"] = "ip"
        v6 = nft_ruleset(policy="drop")["nftables"]
        for item in v6:
            if isinstance(item.get("chain"), dict):
                item["chain"]["family"] = "ip6"
        self.assertTrue(output_chain_defaults_drop({"nftables": v4 + v6}))

    def test_no_output_chain_fails(self) -> None:
        self.assertFalse(output_chain_defaults_drop({"nftables": [{"table": {"name": "filter"}}]}))

    def test_empty_nftables_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pin a ruleset hash or require default-drop"):
            NftablesPolicy.from_dict({"require_default_drop_output": False})


class WifiSecurityTests(unittest.TestCase):
    def test_sae_with_pmf_passes(self) -> None:
        policy = WifiSecurityPolicy.from_dict({"required_key_mgmt": ["sae"], "require_pmf": True})
        observed = {"802-11-wireless-security.key-mgmt": "sae", "802-11-wireless-security.pmf": "3"}
        ok, evidence, errors = verify_wifi_security(policy, observed)
        self.assertTrue(ok, errors)
        self.assertIn("PMF_REQUIRED", evidence)

    def test_missing_pmf_fails(self) -> None:
        policy = WifiSecurityPolicy.from_dict({"required_key_mgmt": ["sae"], "require_pmf": True})
        observed = {"802-11-wireless-security.key-mgmt": "sae", "802-11-wireless-security.pmf": "2"}
        ok, _evidence, errors = verify_wifi_security(policy, observed)
        self.assertFalse(ok)

    def test_wrong_key_mgmt_fails(self) -> None:
        policy = WifiSecurityPolicy.from_dict({"required_key_mgmt": ["sae"], "require_pmf": False})
        observed = {"802-11-wireless-security.key-mgmt": "wpa-psk"}
        ok, _evidence, errors = verify_wifi_security(policy, observed)
        self.assertFalse(ok)


class RouteTests(unittest.TestCase):
    def test_both_families_via_tunnel(self) -> None:
        ok, evidence, errors = verify_default_routes("wg0", [{"dev": "wg0"}], [{"dev": "wg0"}])
        self.assertTrue(ok, errors)
        self.assertIn("IPv4_DEFAULT_ROUTE_VIA_TUNNEL", evidence)
        self.assertIn("IPv6_DEFAULT_ROUTE_VIA_TUNNEL", evidence)

    def test_leak_via_other_device_fails(self) -> None:
        ok, _evidence, errors = verify_default_routes("wg0", [{"dev": "eth0"}], [{"dev": "wg0"}])
        self.assertFalse(ok)

    def test_missing_route_fails(self) -> None:
        ok, _evidence, errors = verify_default_routes("wg0", [], [{"dev": "wg0"}])
        self.assertFalse(ok)


class RedactionTests(unittest.TestCase):
    def test_flags_private_key_material(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_no_secret_material({"leak": f"private key: {KEY_A}"})

    def test_flags_wireguard_config_private_key_form(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_no_secret_material({"conf": f"PrivateKey = {KEY_A}"})

    def test_flags_preshared_key_and_nmcli_psk(self) -> None:
        for leak in (
            f"PresharedKey = {KEY_A}",
            "802-11-wireless-security.psk:hunter2",
            "psk = hunter2",
            "802-1x.password:secret",
        ):
            with self.subTest(leak=leak):
                with self.assertRaises(RuntimeError):
                    assert_no_secret_material({"leak": leak})

    def test_public_evidence_is_allowed(self) -> None:
        assert_no_secret_material({"evidence": ["PEER_ALLOWED_IPS_MATCH"], "hash": "a" * 64})
        assert_no_secret_material({"key-mgmt": "sae", "public_key": KEY_A})


if __name__ == "__main__":
    unittest.main()
