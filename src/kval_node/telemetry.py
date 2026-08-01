from __future__ import annotations

import concurrent.futures
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .commands import CommandError, run_command, run_json
from .models import Catalog, LinkObservation, Profile, TelemetrySnapshot
from .standards import StandardsVerifier


PING_LOSS = re.compile(r"(?P<loss>[0-9]+(?:\.[0-9]+)?)% packet loss")
PING_RTT = re.compile(
    r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = "
    r"(?P<minimum>[0-9.]+)/(?P<average>[0-9.]+)/(?P<maximum>[0-9.]+)/(?P<jitter>[0-9.]+) ms"
)


@dataclass(frozen=True)
class ProbeResult:
    rtt_ms: float | None
    loss_percent: float | None
    jitter_ms: float | None
    error: str | None


def parse_ping_output(output: str) -> ProbeResult:
    loss_match = PING_LOSS.search(output)
    rtt_match = PING_RTT.search(output)
    loss = float(loss_match.group("loss")) if loss_match else None
    if rtt_match:
        return ProbeResult(
            rtt_ms=float(rtt_match.group("average")),
            loss_percent=loss,
            jitter_ms=float(rtt_match.group("jitter")),
            error=None,
        )
    if loss is not None:
        return ProbeResult(
            rtt_ms=None,
            loss_percent=loss,
            jitter_ms=None,
            error="probe returned no successful RTT samples",
        )
    return ProbeResult(None, None, None, "ping output did not contain packet-loss evidence")


class TelemetryCollector:
    """Collects live operating-system and link evidence; no simulation backend exists."""

    def __init__(self) -> None:
        self._previous_counters: dict[str, tuple[float, int, int]] = {}
        self._standards = StandardsVerifier()

    @staticmethod
    def _active_connections() -> tuple[dict[str, str], dict[str, str]]:
        result = run_command(
            [
                "nmcli",
                "--terse",
                "--escape",
                "no",
                "--fields",
                "UUID,DEVICE,STATE",
                "connection",
                "show",
                "--active",
            ],
            timeout_seconds=10.0,
        )
        if result.returncode != 0:
            raise CommandError(result.stderr.strip() or "NetworkManager active-connection query failed")
        by_uuid: dict[str, str] = {}
        by_device: dict[str, str] = {}
        for line in result.stdout.splitlines():
            fields = line.split(":", 2)
            if len(fields) != 3:
                continue
            connection_uuid, device, state = fields
            if state in {"activated", "activating"} and connection_uuid and device:
                by_uuid[connection_uuid] = device
                by_device[device] = connection_uuid
        return by_uuid, by_device

    @staticmethod
    def _link_inventory() -> dict[str, dict[str, Any]]:
        raw = run_json(["ip", "-j", "-s", "link", "show"], timeout_seconds=10.0)
        if not isinstance(raw, list):
            raise CommandError("ip link JSON response was not a list")
        return {str(item["ifname"]): item for item in raw if isinstance(item, dict) and "ifname" in item}

    @staticmethod
    def _is_link_up(link: dict[str, Any]) -> bool:
        state = str(link.get("operstate", "")).upper()
        flags = {str(value).upper() for value in link.get("flags", [])}
        return "UP" in flags and state not in {"DOWN", "LOWERLAYERDOWN", "NOTPRESENT"}

    @staticmethod
    def _default_device() -> str | None:
        raw = run_json(["ip", "-j", "route", "show", "default"], timeout_seconds=10.0)
        if not isinstance(raw, list) or not raw:
            return None
        routes = [route for route in raw if isinstance(route, dict) and route.get("dev")]
        if not routes:
            return None
        routes.sort(key=lambda route: int(route.get("metric", 0)))
        return str(routes[0]["dev"])

    @staticmethod
    def _wifi_signals() -> dict[str, float]:
        result = run_command(
            ["nmcli", "--terse", "--escape", "no", "--fields", "DEVICE,IN-USE,SIGNAL", "device", "wifi", "list", "--rescan", "no"],
            timeout_seconds=10.0,
        )
        if result.returncode != 0:
            return {}
        signals: dict[str, float] = {}
        for line in result.stdout.splitlines():
            fields = line.split(":", 2)
            if len(fields) == 3 and fields[1] == "*":
                try:
                    signals[fields[0]] = float(fields[2])
                except ValueError:
                    continue
        return signals

    @staticmethod
    def _cellular_evidence() -> dict[str, tuple[float | None, str | None]]:
        result = run_command(["mmcli", "-L", "-J"], timeout_seconds=10.0)
        if result.returncode != 0:
            return {}
        try:
            listing = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        modem_paths = listing.get("modem-list", []) if isinstance(listing, dict) else []
        evidence: dict[str, tuple[float | None, str | None]] = {}
        for modem_path in modem_paths:
            modem_result = run_command(["mmcli", "-m", str(modem_path), "-J"], timeout_seconds=10.0)
            if modem_result.returncode != 0:
                continue
            try:
                modem = json.loads(modem_result.stdout).get("modem", {})
            except (json.JSONDecodeError, AttributeError):
                continue
            generic = modem.get("generic", {}) if isinstance(modem, dict) else {}
            signal_quality = generic.get("signal-quality", {})
            if isinstance(signal_quality, dict):
                signal = signal_quality.get("value")
            elif isinstance(signal_quality, (list, tuple)) and signal_quality:
                signal = signal_quality[0]
            else:
                signal = None
            technologies = generic.get("access-technologies", [])
            if isinstance(technologies, str):
                technology = technologies
            elif technologies:
                technology = ",".join(str(value) for value in technologies)
            else:
                technology = None
            ports = generic.get("ports", [])
            for port in ports:
                if isinstance(port, (list, tuple)) and len(port) >= 2 and str(port[1]) == "net":
                    evidence[str(port[0])] = (
                        float(signal) if signal is not None else None,
                        technology,
                    )
        return evidence

    @staticmethod
    def _probe(profile: Profile) -> ProbeResult:
        result = run_command(
            [
                "ping",
                "-n",
                "-q",
                "-I",
                profile.interface,
                "-c",
                "3",
                "-W",
                "1",
                profile.probe_target,
            ],
            timeout_seconds=5.0,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        parsed = parse_ping_output(combined)
        if result.returncode not in {0, 1}:
            return ProbeResult(parsed.rtt_ms, parsed.loss_percent, parsed.jitter_ms, result.stderr.strip() or parsed.error)
        return parsed

    def _rates(self, interface: str, link: dict[str, Any], sampled_at: float) -> tuple[float | None, float | None]:
        stats = link.get("stats64") or link.get("stats") or {}
        rx_bytes = int(stats.get("rx", {}).get("bytes", 0))
        tx_bytes = int(stats.get("tx", {}).get("bytes", 0))
        previous = self._previous_counters.get(interface)
        self._previous_counters[interface] = (sampled_at, rx_bytes, tx_bytes)
        if previous is None:
            return None, None
        previous_time, previous_rx, previous_tx = previous
        elapsed = sampled_at - previous_time
        if elapsed <= 0 or rx_bytes < previous_rx or tx_bytes < previous_tx:
            return None, None
        rx_mbps = (rx_bytes - previous_rx) * 8.0 / elapsed / 1_000_000.0
        tx_mbps = (tx_bytes - previous_tx) * 8.0 / elapsed / 1_000_000.0
        return rx_mbps, tx_mbps

    def collect(self, catalog: Catalog) -> TelemetrySnapshot:
        sampled_at_monotonic = time.monotonic()
        observed_at = datetime.now(timezone.utc)
        links = self._link_inventory()
        active_by_uuid, active_by_device = self._active_connections()
        default_device = self._default_device()
        wifi_signals = self._wifi_signals()
        cellular = self._cellular_evidence()

        probe_profiles = [
            profile
            for profile in catalog.profiles
            if profile.interface in links
            and self._is_link_up(links[profile.interface])
            and active_by_uuid.get(profile.connection_uuid) == profile.interface
        ]
        probes: dict[str, ProbeResult] = {}
        if probe_profiles:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(probe_profiles))) as pool:
                futures = {pool.submit(self._probe, profile): profile.profile_id for profile in probe_profiles}
                for future, profile_id in futures.items():
                    try:
                        probes[profile_id] = future.result()
                    except Exception as exc:
                        probes[profile_id] = ProbeResult(None, None, None, f"probe failed: {exc}")

        observations: list[LinkObservation] = []
        for profile in catalog.profiles:
            link = links.get(profile.interface)
            interface_present = link is not None
            link_up = bool(link and self._is_link_up(link))
            active = active_by_uuid.get(profile.connection_uuid) == profile.interface
            rx_mbps: float | None = None
            tx_mbps: float | None = None
            if link is not None:
                rx_mbps, tx_mbps = self._rates(profile.interface, link, sampled_at_monotonic)

            signal = wifi_signals.get(profile.interface)
            access_technology: str | None = None
            if profile.interface in cellular:
                cellular_signal, access_technology = cellular[profile.interface]
                signal = cellular_signal if cellular_signal is not None else signal

            probe = probes.get(profile.profile_id)
            errors: list[str] = []
            evidence = ["ip-link", "networkmanager"]
            if signal is not None:
                evidence.append("wifi-or-modem-signal")
            if rx_mbps is not None and tx_mbps is not None:
                evidence.append("kernel-byte-counters")
            if probe is not None:
                evidence.append("bound-icmp-probe")
                if probe.error:
                    errors.append(probe.error)
            else:
                errors.append("probe not run because the approved connection is not active and up")

            standards = self._standards.verify(
                profile.standards,
                profile_id=profile.profile_id,
                connection_uuid=profile.connection_uuid,
                observed_access_technology=access_technology,
            )

            observations.append(
                LinkObservation(
                    profile_id=profile.profile_id,
                    interface=profile.interface,
                    observed_at=observed_at,
                    interface_present=interface_present,
                    link_up=link_up,
                    connection_active=active,
                    rtt_ms=probe.rtt_ms if probe else None,
                    loss_percent=probe.loss_percent if probe else None,
                    jitter_ms=probe.jitter_ms if probe else None,
                    rx_mbps=rx_mbps,
                    tx_mbps=tx_mbps,
                    signal_percent=signal,
                    access_technology=access_technology,
                    standards_verified=standards.verified,
                    standards_evidence=standards.evidence,
                    standards_errors=standards.errors,
                    evidence=tuple(evidence),
                    errors=tuple(errors),
                )
            )

        current_profile_id = None
        if default_device is not None:
            current_uuid = active_by_device.get(default_device)
            for profile in catalog.profiles:
                if profile.connection_uuid == current_uuid:
                    current_profile_id = profile.profile_id
                    break
        return TelemetrySnapshot(observed_at, current_profile_id, tuple(observations))
