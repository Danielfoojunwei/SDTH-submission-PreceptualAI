"""Handover measurement and the fast-roam path on one Raspbot.

Why a switch takes seconds today: SmartCarCmd.py makes one NetworkManager
connection profile *per BSSID* and runs `nmcli connection up` on it, so every
switch tears the old connection down and does a fresh scan, association,
4-way handshake and DHCP lease.

Fast roam stays inside the active connection: wpa_supplicant's D-Bus
`Roam(bssid)` re-associates to another BSSID of the same SSID, NetworkManager
keeps the connection and IP, and no password is sent.

Everything here is either read-only (fixed command table) or the single
operator-confirmed roam request, with SmartCar's outcome rules: a roam is
sent once; timeouts and dropped channels are UNKNOWN and only status may be
re-read. wlan0 (the management hotspot) is never addressed.
"""
from __future__ import annotations

import ipaddress
import re
import time

from .carlink import BSSID_RE

WPA = "fi.w1.wpa_supplicant1"
_IFACE_PATH = (f"$(sudo -n busctl call {WPA} /fi/w1/wpa_supplicant1 {WPA} GetInterface s wlan1"
               " | awk '{print $2}' | tr -d '\"')")

# Read-only commands for the handover probe (no state change).
HANDOVER_READONLY = {
    "wpa_version": "wpa_supplicant -v 2>&1 | head -n 1",
    "nm_version": "nmcli --version",
    "wpa_interface_methods": f"sudo -n busctl introspect {WPA} \"{_IFACE_PATH}\" {WPA}.Interface",
    "active_connection": "nmcli -g GENERAL.CONNECTION device show wlan1",
    "active_profile": ("nmcli -g 802-11-wireless.ssid,802-11-wireless.bssid,"
                       "802-11-wireless-security.key-mgmt,ipv4.method connection show "
                       "\"$(nmcli -g GENERAL.CONNECTION device show wlan1)\""),
    "wlan1_addr": "ip -4 -o addr show dev wlan1",
    "wlan1_route": "ip -4 route show dev wlan1",
    "car_clock": "date +%s.%N",
    "scan_cache": "sudo -n iw dev wlan1 scan dump",
}


def run_handover_readonly(link, key: str, timeout: float = 15.0) -> str:
    if key not in HANDOVER_READONLY:
        raise ValueError(f"not a read-only handover probe command: {key}")
    if link.client is None:
        raise ConnectionError("not connected")
    stdin, stdout, _ = link.client.exec_command(HANDOVER_READONLY[key], timeout=timeout)
    stdin.close()
    return stdout.read().decode("utf-8", errors="replace")


def summarise_handover(raw: dict[str, str]) -> dict:
    prof = [line.strip() for line in raw.get("active_profile", "").splitlines()]
    prof += [""] * (4 - len(prof))
    methods = raw.get("wpa_interface_methods", "")
    scan = raw.get("scan_cache", "")
    blocks = re.split(r"\nBSS ", "\n" + scan)[1:]
    ft_aps = sum(bool(re.search(r"Mobility Domain|FT/PSK|FT/IEEE 802.1X|FT/SAE", b)) for b in blocks)
    addr = re.search(r"inet (\d+\.\d+\.\d+\.\d+)/(\d+)", raw.get("wlan1_addr", ""))
    gw = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", raw.get("wlan1_route", ""))
    ver = re.search(r"v(\d+)\.(\d+)", raw.get("wpa_version", ""))
    bssid_lock = prof[1].replace("\\", "").lower()
    return {
        "wpa_supplicant_version": f"{ver.group(1)}.{ver.group(2)}" if ver else None,
        "networkmanager_version": raw.get("nm_version", "").strip() or None,
        "dbus_roam_method": bool(re.search(r"^\.Roam\s+method", methods, re.M)),
        "active_connection": raw.get("active_connection", "").strip() or None,
        "profile_ssid": prof[0] or None,
        "profile_bssid_locked": bool(bssid_lock),
        "key_mgmt": prof[2] or None,
        "ipv4_method": prof[3] or None,
        "wlan1_ipv4": addr.group(1) if addr else None,
        "gateway": gw.group(1) if gw else None,
        "aps_in_scan_cache": len(blocks),
        "aps_advertising_fast_transition": ft_aps,
    }


def fast_roam_ready(summary: dict) -> dict:
    """Which preconditions for the fast-roam path hold, and why not if not."""
    checks = {
        "wpa_supplicant exposes D-Bus Roam": summary["dbus_roam_method"],
        "active profile is not BSSID-locked (else NetworkManager fights the roam)":
            not summary["profile_bssid_locked"],
        "wlan1 has an IPv4 address and gateway (to measure the first packet)":
            bool(summary["wlan1_ipv4"] and summary["gateway"]),
    }
    return {"ready": all(checks.values()), "checks": checks}


def scan_target(scan: str, bssid: str) -> dict | None:
    """SSID and frequency of `bssid` in an `iw scan dump`, or None."""
    for block in re.split(r"\nBSS ", "\n" + scan)[1:]:
        if block[:17].lower() == bssid:
            ssid = re.search(r"\n\s*SSID: ?(.*)", block)
            freq = re.search(r"freq: (\d+(?:\.\d+)?)", block)
            sig = re.search(r"signal: (-?\d+(?:\.\d+)?) dBm", block)
            return {"ssid": ssid.group(1).strip() if ssid else None,
                    "freq_mhz": float(freq.group(1)) if freq else None,
                    "signal_dbm": float(sig.group(1)) if sig else None}
    return None


# --------------------------------------------------------------------------
# Timing capture. All stage times are on the car's clock (journal and ping -D
# both print car epoch seconds); the Mac-side send time is mapped with a
# clock offset measured by bracketing `date` on the car.
def roam_command(bssid: str) -> str:
    bssid = bssid.lower()
    if not BSSID_RE.match(bssid):
        raise ValueError("invalid BSSID")
    return f"date +%s.%N && sudo -n busctl call {WPA} \"{_IFACE_PATH}\" {WPA}.Interface Roam s {bssid}"


def ping_command(gateway: str, seconds: int) -> str:
    ip = ipaddress.IPv4Address(gateway)                      # raises if not an IPv4 literal
    if not 3 <= int(seconds) <= 120:
        raise ValueError("ping window must be 3-120 s")
    return f"sudo -n ping -I wlan1 -D -n -i 0.05 -w {int(seconds)} {ip}"


def journal_command(since_epoch: float) -> str:
    since = float(since_epoch)
    if not 1e9 < since < 1e11:
        raise ValueError("implausible epoch")
    return (f"sudo -n journalctl -u NetworkManager -u wpa_supplicant -o short-unix "
            f"--no-pager --since @{since:.3f}")


_STAGE_PATTERNS = [
    ("association", r"Associated with|state change: associating -> associated|"
                    r"supplicant interface state: \w+ -> (?:associated|4-way handshake|4way_handshake)"),
    ("authentication", r"Key negotiation completed|CTRL-EVENT-CONNECTED|"
                       r"supplicant interface state: \w+ -> completed"),
    ("dhcp", r"dhcp4 \(wlan1\):.*(?:new lease|state changed (?:bound|extended))"),
    ("ip_ready", r"device \(wlan1\): (?:state change: ip-config -> ip-check|Activation: successful)"),
]
_TEARDOWN = r"device \(wlan1\): state change: activated -> (?:deactivating|disconnected)|CTRL-EVENT-DISCONNECTED"


def parse_journal(text: str, after: float) -> dict:
    """First time each stage appears after `after` (car epoch seconds)."""
    first, teardown = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*(\d{9,11}\.\d+)\s", line)
        if not m:
            continue
        t = float(m.group(1))
        if t < after:
            continue
        if "wlan0" in line:                                  # never attribute hotspot events
            continue
        if teardown is None and re.search(_TEARDOWN, line):
            teardown = t
        for stage, pat in _STAGE_PATTERNS:
            if stage not in first and re.search(pat, line):
                first[stage] = t
    return {"stages": first, "teardown": teardown}


def parse_ping(text: str) -> list[float]:
    """Car epoch times of ping replies (`ping -D` lines)."""
    return [float(m.group(1)) for m in re.finditer(r"\[(\d{9,11}\.\d+)\] \d+ bytes from", text)]


def gap_around(replies: list[float], t0: float) -> dict:
    before = [t for t in replies if t <= t0]
    after = [t for t in replies if t > t0]
    if not before or not after:
        return {"last_reply_before_s": None, "first_reply_after_s": None, "outage_s": None}
    return {"last_reply_before_s": round(t0 - before[-1], 3),
            "first_reply_after_s": round(after[0] - t0, 3),
            "outage_s": round(after[0] - before[-1], 3)}


def stage_events(submitted: float, journal: dict, first_reply: float | None, gateway: str) -> list[dict]:
    """Events in measurement.switch_timing's schema, on the car's clock."""
    def ev(stage, t, **extra):
        return {"stage": stage, "source": "hardware", "clock_id": "car_epoch",
                "monotonic_ns": int(round(t * 1e9)), **extra}
    out = [ev("request_submitted", submitted)]
    for stage, t in journal["stages"].items():
        if t >= submitted:
            out.append(ev(stage, t))
    if first_reply is not None:
        out.append(ev("first_packet", first_reply, endpoint=gateway, received_bytes=64))
    return out


def clock_offset(link) -> float:
    """car_epoch - mac_epoch, from a `date` read bracketed by Mac timestamps."""
    t0 = time.time()
    car = float(run_handover_readonly(link, "car_clock").strip())
    t1 = time.time()
    return car - (t0 + t1) / 2


class PingMonitor:
    """Background `ping -D` to the wlan1 gateway on its own SSH channel."""

    def __init__(self, link, gateway: str, seconds: int):
        self.link, self.cmd = link, ping_command(gateway, seconds)
        self.buf = b""
        self.channel = None

    def start(self):
        _, stdout, _ = self.link.client.exec_command(self.cmd)
        self.channel = stdout.channel
        return self

    def drain(self):
        while self.channel.recv_ready():
            self.buf += self.channel.recv(65536)

    def finish(self, wait_s: float) -> str:
        end = time.monotonic() + wait_s
        while time.monotonic() < end and not self.channel.exit_status_ready():
            self.drain()
            time.sleep(0.05)
        self.drain()
        self.channel.close()
        return self.buf.decode("utf-8", errors="replace")


def read_journal(link, since_epoch: float, timeout: float = 20.0) -> str:
    stdin, stdout, _ = link.client.exec_command(journal_command(since_epoch), timeout=timeout)
    stdin.close()
    return stdout.read().decode("utf-8", errors="replace")


def send_roam(link, bssid: str, timeout: float = 15.0) -> float:
    """Send one D-Bus Roam. Returns the car-side submission epoch.

    Raises RemoteRefused-like ValueError on a D-Bus error before any roam
    (NOT_APPLIED), and carlink.OutcomeUnknown if the channel is lost after
    the call was issued."""
    from .carlink import OutcomeUnknown
    stdin, stdout, stderr = link.client.exec_command(roam_command(bssid), timeout=timeout)
    stdin.close()
    channel = stdout.channel
    try:
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = channel.recv_exit_status()
    except Exception as exc:
        raise OutcomeUnknown(f"channel lost after the roam was sent: {exc}") from None
    stamp = re.match(r"\s*(\d{9,11}\.\d+)", out)
    if stamp is None:
        raise OutcomeUnknown("no car-side submission time in the reply")
    if code != 0:
        # Only errors that prove nothing started count as NOT_APPLIED: sudo
        # refusing, or wpa_supplicant rejecting the call synchronously.
        if re.search(r"sudo|password is required|Unknown method|UnknownMethod|InvalidArgs|"
                     r"Invalid|not found|No such|Access denied", err, re.I):
            raise RoamRefused(err.strip() or f"busctl exit {code}", float(stamp.group(1)))
        raise OutcomeUnknown(f"roam call failed ambiguously (exit {code}): {err.strip()[:200]}")
    return float(stamp.group(1))


class RoamRefused(RuntimeError):
    """wpa_supplicant rejected the Roam call; no roam was started."""

    def __init__(self, message: str, submitted: float):
        super().__init__(message)
        self.submitted = submitted
