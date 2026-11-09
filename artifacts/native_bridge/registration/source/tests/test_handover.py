"""Handover probe, timing capture and fast roam against a fake car. No network."""
import json
import re
import time as _time

import pytest

import wirelessjev.handover as ho
import wirelessjev.onecar as onecar
from wirelessjev.carlink import CarLink

KEY_LINE = "192.168.1.11 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestsOnly\n"
WIFI_PW = "test-only-wifi-passphrase"
OLD, NEW = "e0:b6:68:db:11:2a", "e0:b6:68:db:17:c2"
T0 = 1793000000.0
_real_sleep = _time.sleep

SCAN = (f"BSS {NEW}(on wlan1)\n\tfreq: 5200\n\tsignal: -55.00 dBm\n\tSSID: fleet\n"
        f"\tMobility Domain:\n\t\t * MDID: 0x1234\n"
        f"BSS {OLD}(on wlan1) -- associated\n\tfreq: 5180\n\tsignal: -71.00 dBm\n\tSSID: fleet\n")


def ping_text(gap_from, gap_to):
    out, t = [], T0 + 0.6
    while t < T0 + 12:
        if not gap_from <= t < gap_to:
            out.append(f"[{t:.6f}] 64 bytes from 10.0.0.1: icmp_seq=1 ttl=64 time=3.1 ms")
        t += 0.05
    return "\n".join(out) + "\n"


class Stream:
    def __init__(self, data=b"", code=0):
        self.data, self.channel = data, self
        self.code, self._sent = code, False

    def read(self):
        return self.data

    def close(self):
        pass

    def recv_exit_status(self):
        return self.code

    # streaming interface (ping monitor)
    def recv_ready(self):
        return not self._sent and bool(self.data)

    def recv(self, n):
        self._sent = True
        return self.data

    def exit_status_ready(self):
        return True


class SmartCarChannel:
    """SmartCarCmd.py: status / connect (as in test_onecar)."""

    def __init__(self, car):
        self.car, self.reply, self._sent = car, None, False

    def shutdown_write(self):
        self.reply = self.car.handle(self.stdin_data)

    def recv_ready(self):
        return self.reply is not None and not self._sent

    def recv(self, n):
        self._sent = True
        return self.reply

    def recv_stderr_ready(self):
        return False

    def exit_status_ready(self):
        return self._sent

    def recv_exit_status(self):
        return 0

    def close(self):
        pass


class Stdin:
    def __init__(self, ch):
        self.ch = ch
        ch.stdin_data = b""

    def write(self, data):
        self.ch.stdin_data += data

    def flush(self):
        pass

    def close(self):
        pass


class Out:
    def __init__(self, ch):
        self.channel = ch


class FakeCar:
    def __init__(self, locked=False, roam_error=None, legacy=False):
        self.bssid, self.locked, self.roam_error, self.legacy = OLD, locked, roam_error, legacy
        self.commands, self.requests = [], []

    def handle(self, raw):
        req = json.loads(raw.decode())
        self.requests.append(req)
        if req["action"] == "connect":
            assert req["password"] == WIFI_PW
            self.bssid = req.get("bssid", self.bssid)
            res = {"ok": True, "result": {"connected": True, "bssid": self.bssid, "ssid": "fleet"}}
        else:
            res = {"ok": True, "result": {"connected": True, "bssid": self.bssid, "ssid": "fleet",
                                          "signal_dbm": -60.0, "tx_bitrate_mbps": 300.0,
                                          "interface": "wlan1", "role": "WIFI"}}
        return (json.dumps(res) + "\n").encode()

    def journal(self):
        if self.legacy:
            lines = [f"{T0 + 2.00:.6f} pi NetworkManager[500]: <info> device (wlan1): state change: activated -> deactivating",
                     f"{T0 + 3.10:.6f} pi wpa_supplicant[612]: wlan1: Associated with {NEW}",
                     f"{T0 + 3.15:.6f} pi wpa_supplicant[612]: wlan1: WPA: Key negotiation completed with {NEW}",
                     f"{T0 + 5.40:.6f} pi NetworkManager[500]: <info> dhcp4 (wlan1): state changed new lease, address=10.0.0.23",
                     f"{T0 + 5.60:.6f} pi NetworkManager[500]: <info> device (wlan1): Activation: successful"]
        else:
            lines = [f"{T0 + 2.03:.6f} pi wpa_supplicant[612]: wlan1: Associated with {NEW}",
                     f"{T0 + 2.06:.6f} pi wpa_supplicant[612]: wlan1: WPA: Key negotiation completed with {NEW}"]
        lines.append(f"{T0 + 2.05:.6f} pi NetworkManager[500]: <info> device (wlan0): state change: activated -> deactivating")
        lines.insert(0, f"{T0 - 5:.6f} pi wpa_supplicant[612]: wlan1: Associated with {OLD}")
        return "\n".join(lines) + "\n"

    def exec(self, cmd):
        self.commands.append(cmd)
        if "SmartCarCmd.py" in cmd:
            ch = SmartCarChannel(self)
            return Stdin(ch), Out(ch), None
        R = ho.HANDOVER_READONLY
        table = {R["wpa_version"]: "wpa_supplicant v2.10\n", R["nm_version"]: "nmcli tool, version 1.42.4\n",
                 R["wpa_interface_methods"]: ".Roam   method   s   -   -\n.Scan   method   a{sv} - -\n",
                 R["active_connection"]: "fleet\n",
                 R["active_profile"]: f"fleet\n{OLD if self.locked else ''}\nwpa-psk\nauto\n",
                 R["wlan1_addr"]: "3: wlan1    inet 10.0.0.23/24 brd 10.0.0.255 scope global wlan1\n",
                 R["wlan1_route"]: "default via 10.0.0.1 proto dhcp metric 600\n",
                 R["car_clock"]: f"{T0 + 0.5:.9f}\n", R["scan_cache"]: SCAN}
        if cmd in table:
            return Stream(), Stream(table[cmd].encode()), Stream()
        if "Roam s" in cmd:
            if self.roam_error:
                return Stream(), Stream(f"{T0 + 2:.9f}\n".encode(), code=1), Stream(self.roam_error.encode())
            self.bssid = cmd.rsplit(" ", 1)[1]
            return Stream(), Stream(f"{T0 + 2:.9f}\n".encode()), Stream()
        if cmd.startswith("sudo -n ping"):
            gap = (T0 + 2.0, T0 + 5.7) if self.legacy else (T0 + 2.0, T0 + 2.12)
            return Stream(), Stream(ping_text(*gap).encode()), Stream()
        if "journalctl" in cmd:
            return Stream(), Stream(self.journal().encode()), Stream()
        raise AssertionError(f"unexpected command: {cmd}")


class FakeClient:
    def __init__(self, car):
        self.car = car

    def load_host_keys(self, path):
        pass

    def set_missing_host_key_policy(self, policy):
        pass

    def connect(self, **kw):
        pass

    def get_transport(self):
        return None

    def exec_command(self, cmd, timeout=None):
        return self.car.exec(cmd)

    def close(self):
        pass


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    kh = tmp_path / "known_hosts"
    kh.write_text(KEY_LINE)
    wifi = tmp_path / "WiFi.json"
    wifi.write_text(json.dumps({"version": 1, "networks": [
        {"ssid": "fleet", "password": WIFI_PW, "security": "wpa-psk", "enabled": True, "priority": 100}]}))
    monkeypatch.setattr(onecar, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(onecar.time, "sleep", lambda s: None)
    return kh, wifi, tmp_path


def _link(kh, car):
    link = CarLink(kh, lambda: "ssh-test", client_factory=lambda: FakeClient(car))
    link.connect()
    return link


def _records(tmp):
    out = []
    for p in sorted((tmp / "logs").glob("*.jsonl")):
        out += [json.loads(line) for line in p.read_text().splitlines()]
    return out


def test_probe_commands_are_read_only_and_never_touch_wlan0():
    for cmd in ho.HANDOVER_READONLY.values():
        assert "wlan0" not in cmd
        for verb in (r"\bconnect\b", r"\bup\b", r"\bdown\b", "rescan", "trigger", "modify",
                     "delete", r"Roam s", "reapply", r"\bset\b"):
            assert not re.search(verb, cmd), (verb, cmd)
        if "nmcli" in cmd:
            assert "show" in cmd or "--version" in cmd
        if "busctl" in cmd:
            assert "introspect" in cmd and "GetInterface" in cmd


def test_command_builders_reject_injection():
    with pytest.raises(ValueError):
        ho.roam_command("e0:b6:68:db:17:c2; reboot")
    with pytest.raises(ValueError):
        ho.ping_command("10.0.0.1; reboot", 10)
    with pytest.raises(ValueError):
        ho.ping_command("10.0.0.1", 1000)
    with pytest.raises(ValueError):
        ho.journal_command(12.0)
    assert ho.roam_command(NEW.upper()).endswith(f"Roam s {NEW}")


def test_summary_and_readiness():
    car = FakeCar()
    raw = {k: car.exec(v)[1].read().decode() for k, v in ho.HANDOVER_READONLY.items()}
    s = ho.summarise_handover(raw)
    assert s["dbus_roam_method"] and not s["profile_bssid_locked"]
    assert s["gateway"] == "10.0.0.1" and s["wlan1_ipv4"] == "10.0.0.23"
    assert s["aps_advertising_fast_transition"] == 1 and s["wpa_supplicant_version"] == "2.10"
    assert ho.fast_roam_ready(s)["ready"]
    locked = FakeCar(locked=True)
    raw["active_profile"] = locked.exec(ho.HANDOVER_READONLY["active_profile"])[1].read().decode()
    assert not ho.fast_roam_ready(ho.summarise_handover(raw))["ready"]
    assert ho.scan_target(SCAN, NEW) == {"ssid": "fleet", "freq_mhz": 5200.0, "signal_dbm": -55.0}
    assert ho.scan_target(SCAN, "aa:bb:cc:dd:ee:ff") is None


def test_journal_and_ping_parsing_ignore_wlan0_and_old_lines():
    j = ho.parse_journal(FakeCar(legacy=True).journal(), T0 + 2)
    assert j["teardown"] == pytest.approx(T0 + 2.0)
    assert j["stages"]["association"] == pytest.approx(T0 + 3.1)
    assert j["stages"]["dhcp"] == pytest.approx(T0 + 5.4)
    gap = ho.gap_around(ho.parse_ping(ping_text(T0 + 2.0, T0 + 5.7)), T0 + 2.0)
    assert 3.6 < gap["outage_s"] < 3.8


def test_fast_roam_applies_once_times_stages_and_keeps_ip(env):
    kh, _, tmp = env
    car = FakeCar()
    rc = onecar.cmd_roam(Args(bssid=NEW, verify_reads=2, window=5), link=_link(kh, car),
                         ask=lambda _: NEW)
    assert rc == 0
    assert sum("Roam s" in c for c in car.commands) == 1
    assert not any(r["action"] == "connect" for r in car.requests)
    recs = _records(tmp)
    timing = next(r for r in recs if r["event"] == "handover_timing")
    assert timing["submit_clock"] == "car" and timing["validation"] is None
    assert timing["stages_s"]["association"] == pytest.approx(0.03, abs=1e-3)
    assert timing["stages_s"]["first_packet"] < 0.2
    assert timing["ip_retained"] and not timing["teardown_seen"]
    assert next(r for r in recs if r["event"] == "roam_result")["outcome"] == "APPLIED"
    text = "".join(p.read_text() for p in (tmp / "logs").glob("*.jsonl"))
    assert WIFI_PW not in text and "password" not in text


def test_roam_cancelled_without_sending_when_unconfirmed_or_locked(env):
    kh, _, tmp = env
    car = FakeCar()
    assert onecar.cmd_roam(Args(bssid=NEW, verify_reads=1, window=5), link=_link(kh, car),
                           ask=lambda _: OLD) == 1
    locked = FakeCar(locked=True)
    assert onecar.cmd_roam(Args(bssid=NEW, verify_reads=1, window=5), link=_link(kh, locked),
                           ask=lambda _: NEW) == 1
    for c in car.commands + locked.commands:
        assert "Roam s" not in c and "ping" not in c


def test_roam_refused_by_supplicant_is_not_applied(env):
    kh, _, tmp = env
    car = FakeCar(roam_error="Call failed: fi.w1.wpa_supplicant1.InvalidArgs")
    rc = onecar.cmd_roam(Args(bssid=NEW, verify_reads=1, window=5), link=_link(kh, car),
                         ask=lambda _: NEW)
    assert rc == 0
    assert next(r for r in _records(tmp) if r["event"] == "roam_result")["outcome"] == "NOT_APPLIED"
    assert sum("Roam s" in c for c in car.commands) == 1


def test_legacy_switch_timing_shows_teardown_and_dhcp(env, monkeypatch):
    kh, wifi, tmp = env
    # The Mac-side send time maps through the measured clock offset, so the
    # 1.5 s baseline wait must really elapse here.
    monkeypatch.setattr(onecar.time, "sleep", _real_sleep)
    car = FakeCar(legacy=True)
    rc = onecar.cmd_switch(Args(bssid=NEW, ssid="fleet", wifi_json=wifi, verify_reads=1,
                                timing=True, window=8), link=_link(kh, car), ask=lambda _: NEW)
    assert rc == 0
    timing = next(r for r in _records(tmp) if r["event"] == "handover_timing")
    assert timing["teardown_seen"] and timing["submit_clock"] == "mac+offset"
    assert timing["stages_s"]["dhcp"] is not None
    assert 3.6 < timing["outage_s"] < 3.8
    assert sum(r["action"] == "connect" for r in car.requests) == 1


def test_ssid_auto_connect_sends_no_bssid_and_needs_ssid_confirmation(env):
    kh, wifi, tmp = env
    car = FakeCar()
    rc = onecar.cmd_switch(Args(bssid="any", ssid="fleet", wifi_json=wifi, verify_reads=1),
                           link=_link(kh, car), ask=lambda _: "fleet")
    assert rc == 0
    connects = [r for r in car.requests if r["action"] == "connect"]
    assert len(connects) == 1 and "bssid" not in connects[0]
    car2 = FakeCar()
    assert onecar.cmd_switch(Args(bssid="any", ssid="fleet", wifi_json=wifi, verify_reads=1),
                             link=_link(kh, car2), ask=lambda _: "any") == 1
    assert not any(r["action"] == "connect" for r in car2.requests)
