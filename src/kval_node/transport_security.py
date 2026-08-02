"""Secure-transport boundary (defensive).

This module verifies that a node's over-the-air traffic is carried inside an
operator-approved encrypted tunnel and that plaintext cannot leak around it. It
checks, against a signed policy:

* a WireGuard interface whose peers and AllowedIPs match exactly;
* the effective IPv4 and IPv6 default routes leaving through that tunnel;
* an nftables ruleset whose normalized hash matches and whose output chain
  defaults to drop;
* WPA3 (SAE or Enterprise-192) with management-frame protection required.

Two honesty constraints are built in:

* **Encryption is not invisibility.** Confidentiality and integrity are
  enforced; the *existence*, timing, size, direction, and endpoints of RF
  traffic remain observable, and encryption does not stop jamming or a
  compromised endpoint. See ``docs/SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md``.
* **Evidence carries no secrets.** Only public keys, AllowedIPs, and hashes are
  retained. Private keys and pre-shared keys are never read into evidence, and
  :func:`assert_no_secret_material` guards against regressions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .commands import CommandError, run_command, run_json
from .models import parse_timestamp, utc_now
from .signing import canonical_json
from .validation import non_empty_str, sha256_hex

_WG_KEY = re.compile(r"^[A-Za-z0-9+/]{43}=$")
_PSK_HINT = re.compile(r"preshared|private key\s*[:=]\s*[A-Za-z0-9+/]{43}=", re.IGNORECASE)

WPA3_SAE = "sae"
WPA3_ENTERPRISE_192 = "wpa-eap-suite-b-192"
VALID_KEY_MGMT = frozenset({WPA3_SAE, WPA3_ENTERPRISE_192})


def _validate_wg_key(value: str, name: str) -> str:
    text = str(value).strip()
    if not _WG_KEY.fullmatch(text):
        raise ValueError(f"{name} must be a 44-character base64 WireGuard key")
    return text


@dataclass(frozen=True)
class WireguardPeerPolicy:
    public_key: str
    allowed_ips: frozenset[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WireguardPeerPolicy":
        public_key = _validate_wg_key(raw.get("public_key", ""), "peer public_key")
        allowed = frozenset(str(item).strip() for item in raw.get("allowed_ips", []))
        if not allowed or any(not item for item in allowed):
            raise ValueError("each WireGuard peer must declare explicit allowed_ips")
        return cls(public_key=public_key, allowed_ips=allowed)


@dataclass(frozen=True)
class WireguardPolicy:
    interface: str
    peers: tuple[WireguardPeerPolicy, ...]
    require_default_route: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WireguardPolicy":
        interface = non_empty_str(raw.get("interface"), "wireguard.interface")
        peers = tuple(WireguardPeerPolicy.from_dict(item) for item in raw.get("peers", []))
        if not peers:
            raise ValueError("wireguard policy must define at least one peer")
        keys = [peer.public_key for peer in peers]
        if len(keys) != len(set(keys)):
            raise ValueError("wireguard policy contains duplicate peer keys")
        return cls(
            interface=interface,
            peers=peers,
            require_default_route=bool(raw.get("require_default_route", True)),
        )


@dataclass(frozen=True)
class WifiSecurityPolicy:
    required_key_mgmt: frozenset[str]
    require_pmf: bool
    allowed_bssids: frozenset[str]
    eap_ca_cert_sha256: str | None
    eap_server_domain: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "WifiSecurityPolicy":
        key_mgmt = frozenset(str(item).strip().lower() for item in raw.get("required_key_mgmt", []))
        if not key_mgmt or not key_mgmt <= VALID_KEY_MGMT:
            raise ValueError(f"required_key_mgmt must be a non-empty subset of {sorted(VALID_KEY_MGMT)}")
        allowed_bssids = frozenset(str(item).strip().upper() for item in raw.get("allowed_bssids", []))
        eap = raw.get("eap") or {}
        ca_hash = sha256_hex(eap["ca_cert_sha256"], "eap.ca_cert_sha256") if eap.get("ca_cert_sha256") else None
        server_domain = non_empty_str(eap["server_domain"], "eap.server_domain") if eap.get("server_domain") else None
        if WPA3_ENTERPRISE_192 in key_mgmt and (ca_hash is None or server_domain is None):
            raise ValueError("WPA3-Enterprise-192 requires eap.ca_cert_sha256 and eap.server_domain")
        return cls(
            required_key_mgmt=key_mgmt,
            require_pmf=bool(raw.get("require_pmf", True)),
            allowed_bssids=allowed_bssids,
            eap_ca_cert_sha256=ca_hash,
            eap_server_domain=server_domain,
        )


@dataclass(frozen=True)
class NftablesPolicy:
    expected_ruleset_sha256: str | None
    require_default_drop_output: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NftablesPolicy":
        expected = sha256_hex(raw["expected_ruleset_sha256"], "expected_ruleset_sha256") if raw.get("expected_ruleset_sha256") else None
        require_drop = bool(raw.get("require_default_drop_output", True))
        if expected is None and not require_drop:
            raise ValueError("nftables policy must pin a ruleset hash or require default-drop output")
        return cls(expected_ruleset_sha256=expected, require_default_drop_output=require_drop)


@dataclass(frozen=True)
class TransportPolicy:
    policy_id: str
    created_at: datetime
    expires_at: datetime
    wireguard: WireguardPolicy | None
    wifi: WifiSecurityPolicy | None
    nftables: NftablesPolicy | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, now: datetime | None = None) -> "TransportPolicy":
        current = now or utc_now()
        policy_id = non_empty_str(raw.get("policy_id"), "policy_id")
        created_at = parse_timestamp(str(raw["created_at"]))
        expires_at = parse_timestamp(str(raw["expires_at"]))
        if expires_at <= created_at:
            raise ValueError("transport policy expiry must be after creation")
        if expires_at <= current:
            raise ValueError("transport policy has expired")
        wireguard = WireguardPolicy.from_dict(raw["wireguard"]) if raw.get("wireguard") else None
        wifi = WifiSecurityPolicy.from_dict(raw["wifi"]) if raw.get("wifi") else None
        nftables = NftablesPolicy.from_dict(raw["nftables"]) if raw.get("nftables") else None
        if wireguard is None and wifi is None and nftables is None:
            raise ValueError("transport policy must configure at least one section")
        return cls(
            policy_id=policy_id,
            created_at=created_at,
            expires_at=expires_at,
            wireguard=wireguard,
            wifi=wifi,
            nftables=nftables,
        )


def assert_no_secret_material(evidence: object) -> None:
    """Raise if any string in ``evidence`` looks like private key or PSK material."""
    if isinstance(evidence, dict):
        for value in evidence.values():
            assert_no_secret_material(value)
    elif isinstance(evidence, (list, tuple)):
        for value in evidence:
            assert_no_secret_material(value)
    elif isinstance(evidence, str):
        if _PSK_HINT.search(evidence):
            raise RuntimeError("transport evidence contains suspected secret material")


# --- Pure parsers (testable without hardware) -----------------------------------


@dataclass(frozen=True)
class WireguardPeerState:
    public_key: str
    allowed_ips: frozenset[str]
    has_recent_handshake: bool


def parse_wg_show(output: str) -> dict[str, list[WireguardPeerState]]:
    """Parse human-readable ``wg show`` output into peer state per interface.

    The human format hides the private key as ``(hidden)`` and never prints a
    pre-shared key, so it is safe to parse into evidence.
    """
    interfaces: dict[str, list[WireguardPeerState]] = {}
    current_iface: str | None = None
    current_peer: dict[str, Any] | None = None

    def flush_peer() -> None:
        nonlocal current_peer
        if current_iface is not None and current_peer is not None:
            interfaces[current_iface].append(
                WireguardPeerState(
                    public_key=current_peer["public_key"],
                    allowed_ips=frozenset(current_peer.get("allowed_ips", [])),
                    has_recent_handshake=current_peer.get("has_recent_handshake", False),
                )
            )
        current_peer = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("interface:"):
            flush_peer()
            current_iface = line.split(":", 1)[1].strip()
            interfaces.setdefault(current_iface, [])
        elif line.startswith("peer:"):
            flush_peer()
            current_peer = {"public_key": line.split(":", 1)[1].strip()}
        elif current_peer is not None and line.startswith("allowed ips:"):
            addresses = line.split(":", 1)[1].strip()
            current_peer["allowed_ips"] = [a.strip() for a in addresses.split(",") if a.strip() and a.strip() != "(none)"]
        elif current_peer is not None and line.startswith("latest handshake:"):
            current_peer["has_recent_handshake"] = "never" not in line.lower()
    flush_peer()
    return interfaces


def verify_wireguard(policy: WireguardPolicy, parsed: dict[str, list[WireguardPeerState]]) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    evidence: list[str] = []
    errors: list[str] = []
    peers = parsed.get(policy.interface)
    if peers is None:
        return False, (), (f"WireGuard interface {policy.interface} is not present",)
    observed = {peer.public_key: peer for peer in peers}
    expected_keys = {peer.public_key for peer in policy.peers}
    unexpected = sorted(set(observed) - expected_keys)
    if unexpected:
        errors.append("unexpected WireGuard peer present")
    for peer in policy.peers:
        state = observed.get(peer.public_key)
        if state is None:
            errors.append("required WireGuard peer is missing")
            continue
        if state.allowed_ips != peer.allowed_ips:
            errors.append("WireGuard peer AllowedIPs do not match policy exactly")
        else:
            evidence.append("PEER_ALLOWED_IPS_MATCH")
        if not state.has_recent_handshake:
            errors.append("WireGuard peer has no recent handshake")
        else:
            evidence.append("PEER_HANDSHAKE_RECENT")
    return not errors, tuple(sorted(set(evidence))), tuple(sorted(set(errors)))


def _strip_volatile_nft(node: Any) -> Any:
    """Remove handles and counters so a ruleset hashes stably across reloads."""
    if isinstance(node, dict):
        cleaned = {}
        for key, value in node.items():
            if key == "handle":
                continue
            if key == "counter" and isinstance(value, dict):
                cleaned[key] = {k: 0 for k in value}
                continue
            cleaned[key] = _strip_volatile_nft(value)
        return cleaned
    if isinstance(node, list):
        return [_strip_volatile_nft(item) for item in node]
    return node


def normalize_nft_ruleset(ruleset_json: dict[str, Any]) -> str:
    return canonical_json(_strip_volatile_nft(ruleset_json)).decode("utf-8")


def hash_nft_ruleset(ruleset_json: dict[str, Any]) -> str:
    return hashlib.sha256(normalize_nft_ruleset(ruleset_json).encode("utf-8")).hexdigest()


def output_chain_defaults_drop(ruleset_json: dict[str, Any]) -> bool:
    items = ruleset_json.get("nftables", [])
    for item in items:
        chain = item.get("chain") if isinstance(item, dict) else None
        if isinstance(chain, dict) and chain.get("hook") == "output":
            if str(chain.get("policy", "")).lower() != "drop":
                return False
    return any(
        isinstance(item, dict)
        and isinstance(item.get("chain"), dict)
        and item["chain"].get("hook") == "output"
        for item in items
    )


def verify_nftables(policy: NftablesPolicy, ruleset_json: dict[str, Any]) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    evidence: list[str] = []
    errors: list[str] = []
    if policy.expected_ruleset_sha256 is not None:
        observed = hash_nft_ruleset(ruleset_json)
        if observed == policy.expected_ruleset_sha256:
            evidence.append("NFT_RULESET_HASH_MATCH")
        else:
            errors.append("nftables ruleset hash does not match policy")
    if policy.require_default_drop_output:
        if output_chain_defaults_drop(ruleset_json):
            evidence.append("NFT_OUTPUT_DEFAULT_DROP")
        else:
            errors.append("nftables output chain does not default to drop")
    return not errors, tuple(sorted(set(evidence))), tuple(sorted(set(errors)))


def verify_wifi_security(policy: WifiSecurityPolicy, observed: dict[str, str]) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Verify parsed nmcli security key/values against the policy."""
    evidence: list[str] = []
    errors: list[str] = []
    key_mgmt = str(observed.get("802-11-wireless-security.key-mgmt", "")).strip().lower()
    if key_mgmt in policy.required_key_mgmt:
        evidence.append(f"WPA3_KEY_MGMT_{key_mgmt.upper()}")
    else:
        errors.append("Wi-Fi key management is not an approved WPA3 mode")
    if policy.require_pmf:
        pmf = str(observed.get("802-11-wireless-security.pmf", "")).strip().lower()
        if pmf in {"3", "required"}:
            evidence.append("PMF_REQUIRED")
        else:
            errors.append("management-frame protection is not set to required")
    if policy.allowed_bssids:
        bssid = str(observed.get("802-11-wireless.bssid", "")).strip().upper()
        if bssid and bssid in policy.allowed_bssids:
            evidence.append("BSSID_ALLOWLISTED")
        else:
            errors.append("connected BSSID is not on the allowlist")
    if policy.eap_server_domain is not None:
        domain = str(observed.get("802-1x.domain-suffix-match", "")).strip()
        if domain == policy.eap_server_domain:
            evidence.append("EAP_SERVER_DOMAIN_MATCH")
        else:
            errors.append("EAP server domain-suffix-match does not equal policy")
    return not errors, tuple(sorted(set(evidence))), tuple(sorted(set(errors)))


def verify_default_routes(interface: str, v4_routes: Any, v6_routes: Any) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    """Confirm both IPv4 and IPv6 default routes leave via ``interface``."""
    evidence: list[str] = []
    errors: list[str] = []
    for family, routes, label in (("v4", v4_routes, "IPv4"), ("v6", v6_routes, "IPv6")):
        if not isinstance(routes, list) or not routes:
            errors.append(f"{label} default route is absent")
            continue
        devices = {str(route.get("dev")) for route in routes if isinstance(route, dict) and route.get("dev")}
        if devices == {interface}:
            evidence.append(f"{label}_DEFAULT_ROUTE_VIA_TUNNEL")
        else:
            errors.append(f"{label} default route does not leave exclusively via {interface}")
    return not errors, tuple(sorted(set(evidence))), tuple(sorted(set(errors)))


# --- Fail-closed collectors -----------------------------------------------------


class TransportVerifier:
    """Aggregates live transport evidence against a signed policy, fail-closed."""

    def __init__(self, policy: TransportPolicy) -> None:
        self.policy = policy

    @staticmethod
    def _wifi_security(uuid: str) -> dict[str, str]:
        fields = [
            "802-11-wireless-security.key-mgmt",
            "802-11-wireless-security.pmf",
            "802-11-wireless.bssid",
            "802-1x.domain-suffix-match",
        ]
        result = run_command(
            ["nmcli", "--terse", "--get-values", ",".join(fields), "connection", "show", "uuid", uuid],
            timeout_seconds=10.0,
        )
        if result.returncode != 0:
            raise CommandError(result.stderr.strip() or "nmcli security query failed")
        values = result.stdout.splitlines()
        return {field: (values[index] if index < len(values) else "") for index, field in enumerate(fields)}

    def preflight(self, *, wifi_connection_uuid: str | None = None) -> dict[str, Any]:
        sections: dict[str, Any] = {}
        overall_ok = True

        if self.policy.wireguard is not None:
            try:
                output = run_command(["wg", "show", self.policy.wireguard.interface], timeout_seconds=10.0)
                if output.returncode != 0:
                    raise CommandError(output.stderr.strip() or "wg show failed")
                parsed = parse_wg_show(output.stdout)
                ok, evidence, errors = verify_wireguard(self.policy.wireguard, parsed)
                v4 = run_json(["ip", "-j", "route", "show", "default"], timeout_seconds=10.0) if self.policy.wireguard.require_default_route else []
                v6 = run_json(["ip", "-j", "-6", "route", "show", "default"], timeout_seconds=10.0) if self.policy.wireguard.require_default_route else []
                route_ok, route_evidence, route_errors = (
                    verify_default_routes(self.policy.wireguard.interface, v4, v6)
                    if self.policy.wireguard.require_default_route
                    else (True, (), ())
                )
                ok = ok and route_ok
                sections["wireguard"] = {
                    "verified": ok,
                    "evidence": list(evidence) + list(route_evidence),
                    "errors": list(errors) + list(route_errors),
                }
            except Exception as exc:  # fail closed
                ok = False
                sections["wireguard"] = {"verified": False, "evidence": [], "errors": [str(exc)]}
            overall_ok = overall_ok and ok

        if self.policy.nftables is not None:
            try:
                ruleset = run_json(["nft", "-j", "list", "ruleset"], timeout_seconds=10.0)
                if not isinstance(ruleset, dict):
                    raise CommandError("nft did not return a JSON object")
                ok, evidence, errors = verify_nftables(self.policy.nftables, ruleset)
                sections["nftables"] = {"verified": ok, "evidence": list(evidence), "errors": list(errors)}
            except Exception as exc:  # fail closed
                ok = False
                sections["nftables"] = {"verified": False, "evidence": [], "errors": [str(exc)]}
            overall_ok = overall_ok and ok

        if self.policy.wifi is not None:
            if wifi_connection_uuid is None:
                sections["wifi"] = {"verified": False, "evidence": [], "errors": ["no Wi-Fi connection UUID supplied for verification"]}
                overall_ok = False
            else:
                try:
                    observed = self._wifi_security(wifi_connection_uuid)
                    ok, evidence, errors = verify_wifi_security(self.policy.wifi, observed)
                    sections["wifi"] = {"verified": ok, "evidence": list(evidence), "errors": list(errors)}
                except Exception as exc:  # fail closed
                    ok = False
                    sections["wifi"] = {"verified": False, "evidence": [], "errors": [str(exc)]}
                overall_ok = overall_ok and ok

        report = {
            "schema": "kval-transport-preflight/v1",
            "policy_id": self.policy.policy_id,
            "verified": overall_ok,
            "sections": sections,
        }
        assert_no_secret_material(report)
        return report
