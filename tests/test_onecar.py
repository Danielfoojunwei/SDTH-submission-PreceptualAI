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
