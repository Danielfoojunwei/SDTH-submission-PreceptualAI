"""One-car tooling against a fake SSH client that imitates SmartCarCmd.py.
No network, no car."""
import json

import pytest

import wirelessjev.onecar as onecar
from wirelessjev.carlink import CarLink, OutcomeUnknown, load_wifi_json

KEY_LINE = "192.168.1.11 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestsOnly\n"
WIFI_PW = "test-only-wifi-passphrase"


class FakeChannel:
    def __init__(self, car):
        self.car, self.reply, self.closed = car, None, False

    def shutdown_write(self):
        self.reply = self.car.handle(self.stdin_data)

    def recv_ready(self):
        if self.car.mode == "drop" and self.car.last_action == "connect":
            raise EOFError("channel dropped")
        return self.reply is not None and not getattr(self, "_sent", False)

    def recv(self, n):
        self._sent = True
        return self.reply

    def recv_stderr_ready(self):
        return False

    def exit_status_ready(self):
        return getattr(self, "_sent", False)

    def recv_exit_status(self):
        return 0

    def close(self):
        self.closed = True


class FakeStdin:
    def __init__(self, ch):
        self.ch = ch
        ch.stdin_data = b""

    def write(self, data):
        self.ch.stdin_data += data

    def flush(self):
        pass


class FakeOut:
    def __init__(self, ch):
        self.channel = ch


class FakeCar:
    """Imitates SmartCarCmd.py replies for status/connect."""

    def __init__(self, bssid="e0:b6:68:db:11:2a", mode="ok"):
        self.bssid, self.mode = bssid, mode
        self.requests, self.last_action = [], None

    def handle(self, raw):
        req = json.loads(raw.decode())
        self.requests.append(req)
        self.last_action = req["action"]
        if req["action"] == "status":
            res = {"ok": True, "result": {"connected": True, "bssid": self.bssid,
                                          "ssid": "fleet", "signal_dbm": -50.0,
                                          "tx_bitrate_mbps": 234.0,
                                          "interface": "wlan1", "role": "WIFI"}}
        elif self.mode == "invisible":
            res = {"ok": False, "code": "AP_NOT_VISIBLE", "activation_attempted": False,
                   "error": "not visible"}
        else:
            assert req["password"] == WIFI_PW      # SmartCarCmd requires it
            self.bssid = req["bssid"]
            res = {"ok": True, "result": {"connected": True, "bssid": self.bssid,
                                          "ssid": req["ssid"]}}
        return (json.dumps(res) + "\n").encode()


class FakeClient:
    instances = []

    def __init__(self, car):
        self.car, self.policy, self.kwargs, self.host_keys = car, None, None, None
        FakeClient.instances.append(self)

    def load_host_keys(self, path):
        self.host_keys = path

    def set_missing_host_key_policy(self, policy):
        self.policy = policy

    def connect(self, **kw):
        self.kwargs = kw

    def get_transport(self):
        return None

    def exec_command(self, cmd, timeout=None):
        assert "SmartCarCmd.py" in cmd and "--interface wlan1" in cmd
        ch = FakeChannel(self.car)
        return FakeStdin(ch), FakeOut(ch), None

    def close(self):
        pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    kh = tmp_path / "known_hosts"
    kh.write_text(KEY_LINE)
    wifi = tmp_path / "WiFi.json"
    wifi.write_text(json.dumps({"version": 1, "networks": [
        {"ssid": "fleet", "password": WIFI_PW, "security": "wpa-psk",
         "enabled": True, "priority": 100}]}))
    monkeypatch.setattr(onecar, "LOG_DIR", tmp_path / "logs")
    return kh, wifi, tmp_path


def _link(kh, car):
    link = CarLink(kh, lambda: "ssh-test", client_factory=lambda: FakeClient(car))
    link.connect()
    return link


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _logs(tmp):
    return "".join(p.read_text() for p in (tmp / "logs").glob("*.jsonl"))


def test_host_key_is_pinned_and_unknown_hosts_refused(env, tmp_path):
    import paramiko
    kh, _, _ = env
    link = _link(kh, FakeCar())
    client = FakeClient.instances[-1]
    assert isinstance(client.policy, paramiko.RejectPolicy)
    assert client.host_keys == str(kh)
    assert client.kwargs["look_for_keys"] is False and client.kwargs["allow_agent"] is False
    empty = tmp_path / "empty"
    empty.write_text("")
    with pytest.raises(ValueError):
        CarLink(empty, lambda: "x")
    link.close()


def test_status_is_read_only(env):
    kh, _, tmp = env
    car = FakeCar()
    rc = onecar.cmd_status(Args(repeat=3, interval=0.0), link=_link(kh, car))
    assert rc == 0
    assert [r["action"] for r in car.requests] == ["status"] * 3


def test_switch_requires_typed_confirmation(env):
    kh, wifi, tmp = env
    car = FakeCar()
    args = Args(bssid="E0:B6:68:DB:17:C2", ssid="fleet", wifi_json=wifi, verify_reads=2)
    rc = onecar.cmd_switch(args, link=_link(kh, car), ask=lambda _: "e0:b6:68:db:17:c3")
    assert rc == 1
    assert all(r["action"] == "status" for r in car.requests)


def test_switch_applies_once_verifies_and_never_logs_password(env):
    kh, wifi, tmp = env
    car = FakeCar()
    args = Args(bssid="e0:b6:68:db:17:c2", ssid="fleet", wifi_json=wifi, verify_reads=3)
    rc = onecar.cmd_switch(args, link=_link(kh, car), ask=lambda _: "e0:b6:68:db:17:c2")
    assert rc == 0
    assert sum(r["action"] == "connect" for r in car.requests) == 1
    logs = _logs(tmp)
    assert '"outcome": "APPLIED"' in logs and WIFI_PW not in logs
    assert "password" not in logs


def test_not_visible_is_not_applied(env):
    kh, wifi, tmp = env
    car = FakeCar(mode="invisible")
    args = Args(bssid="e0:b6:68:db:17:c2", ssid="fleet", wifi_json=wifi, verify_reads=2)
    rc = onecar.cmd_switch(args, link=_link(kh, car), ask=lambda _: "e0:b6:68:db:17:c2")
    assert rc == 0 and '"outcome": "NOT_APPLIED"' in _logs(tmp)


def test_dropped_channel_is_unknown_and_never_resent(env):
    kh, wifi, tmp = env
    car = FakeCar(mode="drop")
    link = _link(kh, car)
    profile = load_wifi_json(wifi)["fleet"]
    with pytest.raises(OutcomeUnknown):
        link.connect_bssid(profile, "e0:b6:68:db:17:c2", timeout=2)
    assert sum(r["action"] == "connect" for r in car.requests) == 1


def test_wifi_profile_repr_hides_password(env):
    _, wifi, _ = env
    assert WIFI_PW not in repr(load_wifi_json(wifi))


PHY_OUT = """Wiphy phy1
	Band 1:
		Frequencies:
			* 2412.0 MHz [1] (22.0 dBm)
	Band 2:
		HE PHY Capabilities: (0x0c3f)
		Frequencies:
			* 5180.0 MHz [36] (22.0 dBm)
	Band 4:
		EHT Iftypes: managed
		Frequencies:
			* 5955.0 MHz [1] (disabled)
			* 6035.0 MHz [17] (12.0 dBm)
	Supported extended features:
		* [ FTM_RESPONDER ]: FTM responder
	Peer measurement:
		FTM: initiator
"""
SCAN_OUT = """BSS aa:bb:cc:dd:ee:01(on wlan1)
	freq: 5200
	HE capabilities:
	Extended capabilities:
		 * FTM Responder
BSS aa:bb:cc:dd:ee:02(on wlan1)
	freq: 2437
"""
SURVEY_OUT = """Survey data from wlan1
	frequency:			5200 MHz [in use]
	channel active time:		100 ms
	channel busy time:		23 ms
Survey data from wlan1
	frequency:			5180 MHz
	channel busy time:		5 ms
"""
BT_OUT = "Controller 2C:CF:67:00:00:01 (public)\n\tPowered: yes\n\tRoles: central\n\tSupportedSettings: le bredr\n\tLE\n"


class ProbeClient:
    def __init__(self):
        self.commands = []

    def exec_command(self, cmd, timeout=None):
        self.commands.append(cmd)
        from wirelessjev.carlink import READONLY_COMMANDS as R
        table = {R["phy_info"]: PHY_OUT, R["scan_cache"]: SCAN_OUT, R["survey"]: SURVEY_OUT,
                 R["bluetooth"]: BT_OUT, R["wlan1_driver"]: "/sys/bus/pci/drivers/iwlwifi\n",
                 R["wlan1_pci"]: "0x8086\n0x272b\n", R["kernel"]: "6.6.51+rpt-rpi-2712\n"}
        out = table.get(cmd, "")

        class S:
            def __init__(self, data): self.data = data.encode()
            def read(self): return self.data
            def close(self): pass
        return S(""), S(out), S("")

    def close(self):
        pass


def test_capability_probe_is_whitelisted_and_parses(env):
    from wirelessjev.carlink import (READONLY_COMMANDS, ideas_enabled, run_readonly,
                                     summarise_capabilities)
    kh, _, tmp = env
    client = ProbeClient()
    link = CarLink(kh, lambda: "x", client_factory=lambda: client)
    link.client = client
    with pytest.raises(ValueError):
        run_readonly(link, "reboot")
    raw = {k: run_readonly(link, k) for k in READONLY_COMMANDS}
    assert set(client.commands) == set(READONLY_COMMANDS.values())
    for cmd in client.commands:            # nothing that changes state
        assert not any(w in cmd for w in ("connect", "set ", " up", " down", "rescan", "trigger", "nmcli"))
        assert "wlan0" not in cmd
    s = summarise_capabilities(raw)
    assert s["wifi7_eht_client"] and s["wifi6_he_client"] and s["band_6ghz_usable"]
    assert s["ftm_initiator"] and s["aps_advertising_ftm_responder"] == 1
    assert s["survey_busy_time"] and s["survey_channels_reported"] == 2
    assert s["bluetooth_controller"] and s["bluetooth_le"]
    assert s["driver"] == "iwlwifi" and s["pci_id"] == "0x8086 0x272b"
    ideas = ideas_enabled(s)
    assert ideas["attenuation from ranging (FTM)"] and not ideas["no-outage switching (Wi-Fi 7 multi-link)"]


def test_capabilities_command_logs_without_secrets(env):
    kh, _, tmp = env
    client = ProbeClient()
    link = CarLink(kh, lambda: "x", client_factory=lambda: client)
    link.client = client
    assert onecar.cmd_capabilities(Args(), link=link) == 0
    logs = _logs(tmp)
    assert '"ideas_enabled"' in logs and "password" not in logs
