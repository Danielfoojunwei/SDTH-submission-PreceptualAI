"""Real link to one Raspbot: SSH over the management hotspot to the existing
SmartCar.zip on-car scripts. Nothing is deployed or changed on the Pi.

Same invocation as the SmartCar PC app (PC/core/ssh_manager.py, CONFIG
[Command] Request), with these differences:
- the host key is pinned: known_hosts must contain the car's key and unknown
  keys are rejected (the PC app auto-adds);
- the SSH password comes from a getpass prompt or WJ_SSH_PASSWORD and is
  never stored;
- SmartCarCmd's `connect` needs the Wi-Fi password in the request (that is
  how the PC app works). It is injected from the operator's WiFi.json only
  at send time and is never part of a plan, receipt or log.

Outcome semantics follow SmartCar's own: an SSH timeout or a dropped channel
after a request was written means the result is UNKNOWN, and a mutation is
never retried; only status may be re-read.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shlex
import threading
import time
from typing import Callable

MANAGEMENT_HOST = "192.168.1.11"
CMD = ('sudo -n python3 -u "$HOME"/SmartCarCmd.py '
       '--interface wlan1 --management-interface wlan0')
SCAN = ('sudo -n python3 -u "$HOME"/SmartCatScan.py --interface wlan1 '
        '--management-interface wlan0 --ssid={ssid} --ssids-json {ssids} '
        '--interval 0.5 --request-interval 2 --max-ap-age 15 '
        '--status-interval 0.5 --ping-interval 0.5 --ping-target= '
        '--heartbeat-timeout 15')
BSSID_RE = re.compile(r"^(?:[0-9a-f]{2}:){5}[0-9a-f]{2}$")
SECURITIES = ("wpa-psk", "sae", "open")


class OutcomeUnknown(RuntimeError):
    """The request may or may not have taken effect; read status, never resend."""


class RemoteRefused(RuntimeError):
    def __init__(self, message: str, code: str | None, activation_attempted):
        super().__init__(message)
        self.code = code
        self.activation_attempted = activation_attempted


@dataclass(frozen=True)
class WifiProfile:
    ssid: str
    security: str
    priority: int
    password: str          # kept in memory only; repr hides it

    def __repr__(self):
        return f"WifiProfile(ssid={self.ssid!r}, security={self.security!r}, priority={self.priority})"


def load_wifi_json(path: str | Path) -> dict[str, WifiProfile]:
    """Parse SmartCar PI/config/WiFi.json (version 1, enabled networks only)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("networks"), list):
        raise ValueError("WiFi.json must have version 1 and a networks list")
    out = {}
    for net in data["networks"]:
        if not net.get("enabled", False):
            continue
        ssid, sec = net.get("ssid"), net.get("security", "wpa-psk")
        if not isinstance(ssid, str) or not 1 <= len(ssid.encode()) <= 32:
            raise ValueError("invalid SSID in WiFi.json")
        if sec not in SECURITIES:
            raise ValueError(f"unsupported security {sec!r}")
        pw = net.get("password", "")
        if sec != "open" and not pw:
            raise ValueError(f"{ssid}: encrypted network without password")
        out[ssid] = WifiProfile(ssid, sec, int(net.get("priority", 0)), pw)
    if not out:
        raise ValueError("WiFi.json has no enabled networks")
    return out


def known_hosts_has(path: str | Path, host: str = MANAGEMENT_HOST) -> bool:
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and host in parts[0].split(","):
            return True
    return False


class CarLink:
    def __init__(self, known_hosts: str | Path, password: Callable[[], str], *,
                 host: str = MANAGEMENT_HOST, port: int = 22, username: str = "pi",
                 timeout: float = 10.0, client_factory=None):
        if not known_hosts_has(known_hosts, host):
            raise ValueError(f"known_hosts has no key for {host}; refusing to connect")
        self.known_hosts = str(known_hosts)
        self.password = password
        self.host, self.port, self.username, self.timeout = host, port, username, timeout
        self._factory = client_factory
        self.client = None
        self._lock = threading.Lock()

    # --- connection -----------------------------------------------------------
    def connect(self) -> None:
        import paramiko
        client = (self._factory or paramiko.SSHClient)()
        client.load_host_keys(self.known_hosts)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect(hostname=self.host, port=self.port, username=self.username,
                       password=self.password(), timeout=self.timeout,
                       auth_timeout=self.timeout, banner_timeout=self.timeout,
                       look_for_keys=False, allow_agent=False)
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(10)
        self.client = client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    # --- one JSON request to SmartCarCmd.py -----------------------------------
    def request(self, payload: dict, timeout: float) -> dict:
        if self.client is None:
            raise ConnectionError("not connected")
        with self._lock:
            try:
                stdin, stdout, _ = self.client.exec_command(CMD, timeout=timeout)
            except Exception as exc:          # nothing was sent yet
                raise ConnectionError(f"could not open channel: {exc}") from None
            channel = stdout.channel
            written = False
            try:
                stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
                stdin.flush()
                written = True
                channel.shutdown_write()
                out = bytearray()
                deadline = time.monotonic() + timeout
                while True:
                    if time.monotonic() >= deadline:
                        raise OutcomeUnknown("timed out after the request was sent")
                    if channel.recv_ready():
                        out.extend(channel.recv(65536))
                    if channel.recv_stderr_ready():
                        channel.recv_stderr(65536)
                    if (channel.exit_status_ready() and not channel.recv_ready()
                            and not channel.recv_stderr_ready()):
                        break
                    time.sleep(0.01)
                code = channel.recv_exit_status()
            except OutcomeUnknown:
                raise
            except Exception as exc:
                if written:
                    raise OutcomeUnknown(f"channel lost after the request was sent: {exc}") from None
                raise ConnectionError(f"channel lost before sending: {exc}") from None
            finally:
                channel.close()
        if code == -1:
            raise OutcomeUnknown("no exit status from the car")
        try:
            response = json.loads(out.decode("utf-8", errors="replace"))
        except ValueError:
            raise OutcomeUnknown(f"non-JSON reply (exit {code})") from None
        if response.get("ok") is True:
            return response["result"]
        raise RemoteRefused(str(response.get("error", "refused")),
                            response.get("code"), response.get("activation_attempted"))

    def status(self, timeout: float = 10.0) -> dict:
        return self.request({"action": "status"}, timeout)

    def connect_bssid(self, profile: WifiProfile, bssid: str, timeout: float = 150.0) -> dict:
        bssid = bssid.lower()
        if not BSSID_RE.match(bssid):
            raise ValueError("invalid BSSID")
        payload = {"action": "connect", "ssid": profile.ssid, "bssid": bssid,
                   "security": profile.security, "password": profile.password}
        return self.request(payload, timeout)

    # --- read-only scan stream (existing SmartCatScan.py) ---------------------
    def scan(self, ssids: list[str], seconds: float) -> list[dict]:
        """Stream SmartCatScan for `seconds`, sending heartbeats; when we stop
        heartbeating the script exits by itself (heartbeat timeout 15 s)."""
        if self.client is None:
            raise ConnectionError("not connected")
        cmd = SCAN.format(ssid=shlex.quote(ssids[0]),
                          ssids=shlex.quote(json.dumps(ssids, ensure_ascii=False)))
        stdin, stdout, _ = self.client.exec_command(cmd)
        channel = stdout.channel
        events, buf = [], b""
        end = time.monotonic() + seconds
        last_beat = 0.0
        try:
            while time.monotonic() < end:
                if time.monotonic() - last_beat >= 5:
                    stdin.write(b"heartbeat\n")
                    stdin.flush()
                    last_beat = time.monotonic()
                if channel.recv_ready():
                    buf += channel.recv(65536)
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        try:
                            events.append(json.loads(line))
                        except ValueError:
                            continue
                elif channel.exit_status_ready():
                    break
                else:
                    time.sleep(0.05)
        finally:
            channel.close()
        return events


# --------------------------------------------------------------------------
# Read-only capability probe. Only these fixed commands can run; nothing here
# changes radio, network or system state. wlan0 (the HOTPOT management radio)
# is never queried beyond what is needed to confirm roles.
READONLY_COMMANDS = {
    "kernel": "uname -r",
    "iw_version": "iw --version",
    "wlan1_info": "iw dev wlan1 info",
    "wlan1_link": "iw dev wlan1 link",
    "wlan1_driver": "readlink -f /sys/class/net/wlan1/device/driver",
    "wlan1_pci": "cat /sys/class/net/wlan1/device/vendor /sys/class/net/wlan1/device/device",
    "phy_info": "iw phy \"$(iw dev wlan1 info | awk '/wiphy/{print \"phy\"$2}')\" info",
    "survey": "iw dev wlan1 survey dump",
    "scan_cache": "sudo -n iw dev wlan1 scan dump",
    "bluetooth": "bluetoothctl show 2>/dev/null || hciconfig -a 2>/dev/null || echo none",
    "bt_list": "bluetoothctl list 2>/dev/null || echo none",
}


def run_readonly(link: "CarLink", key: str, timeout: float = 15.0) -> str:
    if key not in READONLY_COMMANDS:
        raise ValueError(f"not a read-only probe command: {key}")
    if link.client is None:
        raise ConnectionError("not connected")
    stdin, stdout, stderr = link.client.exec_command(READONLY_COMMANDS[key], timeout=timeout)
    stdin.close()
    out = stdout.read().decode("utf-8", errors="replace")
    return out


def summarise_capabilities(raw: dict[str, str]) -> dict:
    """Turn probe output into the facts that decide which ideas the car supports."""
    phy = raw.get("phy_info", "")
    scan = raw.get("scan_cache", "")
    survey = raw.get("survey", "")
    bt = raw.get("bluetooth", "")
    freqs = [float(f) for f in re.findall(r"\*\s+(\d{4}(?:\.\d+)?)\s+MHz", phy)]
    disabled = set(float(f) for f in re.findall(r"\*\s+(\d{4}(?:\.\d+)?)\s+MHz[^\n]*disabled", phy))
    usable = [f for f in freqs if f not in disabled]
    ap_blocks = re.split(r"\nBSS ", "\n" + scan)[1:]
    ap_ftm = sum("FTM Responder" in b or "FTM responder" in b for b in ap_blocks)
    ap_he = sum("HE capabilities" in b for b in ap_blocks)
    ap_eht = sum("EHT capabilities" in b for b in ap_blocks)
    ap_6g = sum(any(5925 <= float(f) <= 7125 for f in re.findall(r"freq: (\d+(?:\.\d+)?)", b))
                for b in ap_blocks)
    busy_entries = len(re.findall(r"channel busy time", survey))
    return {
        "wifi7_eht_client": "EHT" in phy,
        "wifi6_he_client": "HE PHY" in phy or "HE Iftypes" in phy,
        "band_6ghz_usable": any(5925 <= f <= 7125 for f in usable),
        "multi_link_mlo": bool(re.search(r"\bMLO\b|multi-link|MLD", phy, re.I)),
        "ftm_initiator": bool(re.search(r"FTM_INITIATOR|ftm initiator|FTM initiator|"
                                        r"Peer measurement", phy, re.I)),
        "ftm_responder_feature": "FTM_RESPONDER" in phy or "ENABLE_FTM_RESPONDER" in phy,
        "survey_busy_time": busy_entries > 0,
        "survey_channels_reported": busy_entries,
        "aps_in_scan_cache": len(ap_blocks),
        "aps_advertising_ftm_responder": ap_ftm,
        "aps_wifi6": ap_he, "aps_wifi7": ap_eht, "aps_on_6ghz": ap_6g,
        "bluetooth_controller": "Controller" in bt or "BD Address" in bt,
        "bluetooth_le": bool(re.search(r"\bLE\b|Low Energy|BR/EDR/LE|le-only", bt)),
        "driver": raw.get("wlan1_driver", "").strip().rsplit("/", 1)[-1],
        "pci_id": " ".join(raw.get("wlan1_pci", "").split()),
        "kernel": raw.get("kernel", "").strip(),
    }


IDEA_REQUIREMENTS = {
    "attenuation from ranging (FTM)": ("ftm_initiator", "aps_advertising_ftm_responder"),
    "robots as spectrum sensors (busy time)": ("survey_busy_time",),
    "Bluetooth heartbeat relay during reconnects": ("bluetooth_controller", "bluetooth_le"),
    "no-outage switching (Wi-Fi 7 multi-link)": ("multi_link_mlo", "aps_wifi7"),
    "6 GHz channels": ("band_6ghz_usable", "aps_on_6ghz"),
}


def ideas_enabled(summary: dict) -> dict:
    return {idea: all(bool(summary.get(k)) for k in keys)
            for idea, keys in IDEA_REQUIREMENTS.items()}
