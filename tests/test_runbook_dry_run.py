"""Dry run of docs/TUESDAY_RUNBOOK.md against a fake car. No network, no car.

Every command in the runbook's code blocks is parsed by the real CLI parsers,
then the blocks are walked in order: probes, survey laps, handover timing,
ten route laps, a lap under ghost load, a blockage lap, and the two offline
checks. The fake car answers the fixed command tables and SmartCarCmd
requests; lap streams are generated from the telemetry command it receives.
"""
import functools
import json
import math
from pathlib import Path
import re
import shlex
import sys
import threading
import time

import pytest

import wirelessjev.handover as ho
import wirelessjev.onecar as onecar
from wirelessjev.carlink import CMD, MANAGEMENT_HOST, READONLY_COMMANDS, CarLink
from wirelessjev.hil import analyse, ghosts, telemetry as tel, trace_twin

RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "TUESDAY_RUNBOOK.md"
KEY_LINE = "192.168.1.11 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFakeKeyForTestsOnly\n"
WIFI_PW = "test-only-wifi-passphrase"
SSH_PW = "test-only-ssh-passphrase"
SSID = "fleet"
APS = [("aa:bb:cc:00:00:01", 2.0, 2.0, 5180), ("aa:bb:cc:00:00:02", 18.0, 2.0, 5260),
       ("aa:bb:cc:00:00:03", 10.0, 10.0, 5500)]
SITE = {"schema": 1, "site_id": "test-hall", "dimensions_m": [20.0, 12.0, 4.0],
        "aps": [{"bssid": b, "ssid": SSID, "xyz_m": [x, y, 3.0], "frequency_mhz": f, "tx_power_dbm": 20.0}
                for b, x, y, f in APS],
        "cad_boxes_m": [], "receiver": {"noise_floor_dbm": -92.0, "offset_db": 0.0, "quantization_db": 1.0,
                                        "reference_gauge_assumed": True}}
WAYPOINTS = {"A": [1.0, 6.0], "B": [19.0, 6.0]}
SKEW = 1234.5                       # car clock minus laptop clock
LAP_S = 30                          # the runbook says 180; shortened so the test stays quick
PHY = ("Wiphy phy1\n\tRetry short limit: 7\n\tRetry long limit: 4\n\tRTS threshold: off\n"
       "\tBand 2:\n\t\tHE PHY Capabilities: (0x0c3f)\n\t\tEHT Iftypes: managed\n\t\tFrequencies:\n"
       "\t\t\t* 5180.0 MHz [36] (22.0 dBm)\n\tvalid interface combinations:\n"
       "\t\t * #{ managed } <= 2, #{ AP, P2P-client, P2P-GO } <= 1,\n\t\t   total <= 3, #channels <= 2\n")
BT = "Controller 2C:CF:67:00:00:01 (public)\n\tPowered: yes\n\tSupportedSettings: le bredr\n\tLE\n"


def rssi(x, y, ax, ay):
    return 20.0 - (46.7 + 33 * math.log10(max(1.0, math.hypot(x - ax, y - ay))))


def runbook_commands():
    text = RUNBOOK.read_text(encoding="utf-8")
    return [line.strip() for block in re.findall(r"```bash\n(.*?)```", text, re.S)
            for line in block.splitlines() if line.strip() and not line.strip().startswith("cd ")]


def argv_of(line, kh, wifi, bssid=APS[1][0], target="10.0.0.5"):
    for old, new in (("$KH", str(kh)), ("~/WiFi.json", str(wifi)), ("<SSID>", SSID), ("<bssid>", bssid),
                     ("<AP or sink IP>", target)):
        line = line.replace(old, new)
    argv = shlex.split(line)
    # The runbook uses the project's own Python, so no command depends on `uv` being on the path.
    assert argv[:2] == [".venv/bin/python", "-m"], line
    return argv[2], argv[3:]


# --------------------------------------------------------------------------
class Stream:
    """stdout of a one-shot command; also its own channel. `data` may be a callable (late text)."""

    def __init__(self, data=b"", code=0, done=None):
        self._data, self.channel, self.code, self._sent, self._done = data, self, code, False, done

    def _bytes(self):
        if callable(self._data):
            self._data = self._data()
        return self._data

    def read(self):
        return self._bytes()

    def close(self):
        pass

    def recv_exit_status(self):
        return self.code

    def recv_ready(self):
        return not self._sent

    def recv(self, n):
        self._sent = True
        return self._bytes()

    def exit_status_ready(self):
        return self._sent and (self._done is None or self._done.is_set())


class CmdChannel:
    """SmartCarCmd.py: one JSON request on stdin, one JSON reply."""

    def __init__(self, car):
        self.car, self.reply, self._sent, self.stdin_data = car, None, False, b""

    def shutdown_write(self):
        self.reply = self.car.handle(self.stdin_data)

    def recv_ready(self):
        if self.car.drop_connect and self.car.last_action == "connect":
            raise EOFError("channel dropped")
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


class CmdStdin:
    def __init__(self, ch):
        self.ch = ch

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
    def __init__(self):
        self.bssid, self.locked = APS[0][0], False
        self.commands, self.requests, self.last_action = [], [], None
        self.drop_connect = False
        self.event_t = None                       # car time of the last connect or roam
        self.legacy = True                        # the last change was a full reconnect
        self.route_y, self.block, self.marks_done = 6.0, False, None

    def now(self):
        return time.time() + SKEW

    # --- SmartCarCmd ------------------------------------------------------
    def handle(self, raw):
        req = json.loads(raw.decode())
        self.requests.append(req)
        self.last_action = req["action"]
        if req["action"] == "connect":
            assert req["password"] == WIFI_PW and req["ssid"] == SSID
            if self.drop_connect:
                return b""
            self.event_t, self.legacy = self.now(), True
            self.locked = "bssid" in req           # SmartCarCmd makes one profile per BSSID
            self.bssid = req.get("bssid", self.bssid)
            res = {"ok": True, "result": {"connected": True, "bssid": self.bssid, "ssid": SSID}}
        else:
            res = {"ok": True, "result": {"connected": True, "bssid": self.bssid, "ssid": SSID,
                                          "signal_dbm": -60.0, "tx_bitrate_mbps": 300.0,
                                          "interface": "wlan1", "role": "WIFI"}}
        return (json.dumps(res) + "\n").encode()

    # --- text the car would print -----------------------------------------
    def scan_dump(self, x=10.0, y=6.0):
        return "".join(f"BSS {b}(on wlan1){' -- associated' if b == self.bssid else ''}\n\tfreq: {f}\n"
                       f"\tsignal: {rssi(x, y, ax, ay):.2f} dBm\n\tlast seen: 200 ms ago\n\tSSID: {SSID}\n"
                       f"\tMobility Domain:\n\t\t * MDID: 0x1234\n" for b, ax, ay, f in APS)

    def journal(self):
        t = self.event_t
        if self.legacy:
            lines = [f"{t + 0.28:.6f} pi NetworkManager[500]: <info> device (wlan1): state change: activated -> deactivating",
                     f"{t + 1.10:.6f} pi wpa_supplicant[612]: wlan1: Associated with {self.bssid}",
                     f"{t + 1.15:.6f} pi wpa_supplicant[612]: wlan1: WPA: Key negotiation completed with {self.bssid}",
                     f"{t + 3.40:.6f} pi NetworkManager[500]: <info> dhcp4 (wlan1): state changed new lease, address=10.0.0.23",
                     f"{t + 3.60:.6f} pi NetworkManager[500]: <info> device (wlan1): Activation: successful"]
        else:
            lines = [f"{t + 0.03:.6f} pi wpa_supplicant[612]: wlan1: Associated with {self.bssid}",
                     f"{t + 0.06:.6f} pi wpa_supplicant[612]: wlan1: WPA: Key negotiation completed with {self.bssid}"]
        return ("\n".join(lines) + "\n").encode()

    def ping_around_change(self, start, seconds):
        """Replies every 50 ms, silent while the last connect or roam was in progress.

        The test skips the real 1.5 s baseline wait, so the replies are back-dated
        by that much: on the car the ping would already have been running."""
        start -= 1.5
        gap = (0.0, 0.0) if self.event_t is None else (
            (self.event_t + 0.30, self.event_t + 3.7) if self.legacy else (self.event_t + 0.01, self.event_t + 0.12))
        out = [f"[{start + 0.05 * i:.6f}] 64 bytes from 10.0.0.1: icmp_seq={i} ttl=64 time=3.1 ms"
               for i in range(int((seconds + 1.5) / 0.05)) if not gap[0] <= start + 0.05 * i < gap[1]]
        return ("\n".join(out) + "\n").encode()

    def lap_stream(self, cmd, start):
        """What the telemetry loop prints, built from the numbers in the command it was sent."""
        n = int(re.search(r"-lt (\d+)", cmd).group(1))
        scan_every = int(re.search(r"i % (\d+)", cmd).group(1))
        period = float(re.search(r"sleep ([\d.]+); done", cmd).group(1))
        out = []
        for i in range(n):
            t = start + i * period
            x = 1.0 + 18.0 * i / (n - 1)
            best = max(APS, key=lambda a: rssi(x, self.route_y, a[1], a[2]))
            sig = rssi(x, self.route_y, best[1], best[2]) - (18.0 if self.block and 10.0 <= t - start < 12.0 else 0.0)
            out.append(f"@T {t:.6f}")
            out.append(f"Connected to {best[0]} (on wlan1)\n\tSSID: {SSID}\n\tfreq: {best[3]}\n"
                       f"\tsignal: {sig:.0f} dBm\n\ttx bitrate: 234.0 MBit/s")
            out.append(f"Station {best[0]} (on wlan1)\n\ttx packets:\t{100 * i}\n\ttx retries:\t{3 * i}\n"
                       f"\ttx failed:\t0\n\tbeacon loss:\t0\n\tsignal avg:\t-60 dBm")
            if i % scan_every == 0:
                out.append("@SCAN\n" + self.scan_dump(x, self.route_y).rstrip("\n") + "\n@ENDSCAN")
        return ("\n".join(out) + "\n").encode()

    def lap_ping(self, start, seconds):
        out = [f"[{start + 0.05 * i:.6f}] 64 bytes from 10.0.0.1: icmp_seq={i} ttl=64 time=2.5 ms"
               for i in range(int(seconds / 0.05))
               if not (self.block and 10.3 <= 0.05 * i < 11.8)]
        return ("\n".join(out) + "\n").encode()

    # --- command dispatch ---------------------------------------------------
    def exec(self, cmd):
        self.commands.append(cmd)
        if cmd == CMD:
            ch = CmdChannel(self)
            return CmdStdin(ch), Out(ch), None
        R, P = ho.HANDOVER_READONLY, READONLY_COMMANDS
        table = {R["wpa_version"]: "wpa_supplicant v2.10\n", R["nm_version"]: "nmcli tool, version 1.42.4\n",
                 R["wpa_interface_methods"]: ".Roam   method   s   -   -\n.Scan   method   a{sv} - -\n",
                 R["active_connection"]: f"{SSID}\n",
                 R["active_profile"]: f"{SSID}\n{self.bssid if self.locked else ''}\nwpa-psk\nauto\n",
                 R["wlan1_addr"]: "3: wlan1    inet 10.0.0.23/24 brd 10.0.0.255 scope global wlan1\n",
                 R["wlan1_route"]: "default via 10.0.0.1 proto dhcp metric 600\n",
                 R["car_clock"]: f"{self.now():.9f}\n", R["scan_cache"]: self.scan_dump(),
                 P["phy_info"]: PHY, P["bluetooth"]: BT, P["kernel"]: "6.6.51+rpt-rpi-2712\n",
                 P["wlan1_driver"]: "/sys/bus/pci/drivers/iwlwifi\n", P["wlan1_pci"]: "0x8086\n0x272b\n",
                 P["wlan1_info"]: "Interface wlan1\n\twiphy 1\n\ttxpower 22.00 dBm\n",
                 P["power_save"]: "Power save: on\n"}
        if cmd in table:
            return Stream(), Stream(table[cmd].encode()), Stream()
        if cmd in P.values():                       # probes this fake has nothing to say about
            return Stream(), Stream(b""), Stream()
        if "Roam s" in cmd:
            assert cmd == ho.roam_command(cmd.rsplit(" ", 1)[1])
            self.event_t, self.legacy, self.bssid = self.now(), False, cmd.rsplit(" ", 1)[1]
            return Stream(), Stream(f"{self.event_t:.9f}\n".encode()), Stream()
        m = re.fullmatch(r"sudo -n ping -I wlan1 -D -n -i 0\.05 -w (\d+) 10\.0\.0\.1", cmd)
        if m:
            start, seconds = self.now(), int(m.group(1))
            if self.marks_done is not None:         # a lap is being recorded
                return Stream(), Stream(self.lap_ping(start, seconds), done=self.marks_done), Stream()
            return Stream(), Stream(lambda: self.ping_around_change(start, seconds)), Stream()
        if "journalctl" in cmd:
            return Stream(), Stream(self.journal()), Stream()
        if cmd.startswith("i=0; while"):
            return Stream(), Stream(self.lap_stream(cmd, self.now()), done=self.marks_done), Stream()
        raise AssertionError(f"command not in any fixed table: {cmd}")


class FakeClient:
    def __init__(self, car):
        self.car = car

    def load_host_keys(self, path):
        pass

    def set_missing_host_key_policy(self, policy):
        import paramiko
        assert isinstance(policy, paramiko.RejectPolicy)       # unknown host keys are refused

    def connect(self, **kw):
        assert kw["password"] == SSH_PW and kw["look_for_keys"] is False

    def get_transport(self):
        return None

    def exec_command(self, cmd, timeout=None):
        return self.car.exec(cmd)

    def close(self):
        pass


class Keyboard:
    """Stands in for the operator's terminal during a lap: types A at once and B 29.5 s later."""

    def __init__(self, clock_jump, done, labels=("A", "b"), gap_s=29.5):      # "b": typed in the wrong case
        self.clock_jump, self.done, self.labels, self.gap_s = clock_jump, done, labels, gap_s

    def isatty(self):
        return True

    def __iter__(self):
        try:
            for i, label in enumerate(self.labels):
                self.clock_jump[0] = self.gap_s * i / max(len(self.labels) - 1, 1)
                yield label + "\n"
        finally:
            self.done.set()


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# --------------------------------------------------------------------------
def test_every_runbook_command_parses_with_the_real_parsers(tmp_path):
    lines = runbook_commands()
    assert len(lines) >= 14
    seen = set()
    for line in lines:
        if line.startswith("export "):
            continue
        module, rest = argv_of(line, tmp_path / "kh", tmp_path / "WiFi.json")
        seen.add((module, rest[0] if module.endswith("onecar") or module.endswith("ghosts") else ""))
        if module == "wirelessjev.onecar":
            onecar.build_parser().parse_args(rest)             # SystemExit here means a typo in the runbook
        elif module == "wirelessjev.hil.ghosts":
            args = ghosts.build_parser().parse_args(rest)
            assert args.seconds <= ghosts.MAX_SECONDS
            assert sum(g.mbps for g in ghosts.fleet(args.ghosts, args.video, args.bulk)) <= ghosts.MAX_TOTAL_MBPS
        else:
            assert module in ("wirelessjev.hil.analyse", "wirelessjev.hil.trace_twin", "wirelessjev.native_ext.coverage"), module
    for sub in ("preflight", "status", "capabilities", "handover-probe", "lap", "switch", "roam"):
        assert ("wirelessjev.onecar", sub) in seen, sub
    # The offered ghost load the runbook quotes must be the real default.
    quoted = re.search(r"nine ghost robots \(about (\d+) Mb/s\)", RUNBOOK.read_text(encoding="utf-8"))
    assert quoted and abs(int(quoted.group(1)) - sum(g.mbps for g in ghosts.fleet())) < 1.0


def test_runbook_end_to_end_against_a_fake_car(tmp_path, monkeypatch, capsys):
    kh, wifi, logs = tmp_path / "known_hosts", tmp_path / "WiFi.json", tmp_path / "onecar"
    kh.write_text(KEY_LINE)
    wifi.write_text(json.dumps({"version": 1, "networks": [
        {"ssid": SSID, "password": WIFI_PW, "security": "wpa-psk", "enabled": True, "priority": 100}]}))
    monkeypatch.setattr(onecar, "LOG_DIR", logs)
    car = FakeCar()

    def link(args=None):
        if args is not None:
            assert args.host == MANAGEMENT_HOST and Path(args.known_hosts) == kh
        fresh = CarLink(kh, lambda: SSH_PW, client_factory=lambda: FakeClient(car))
        fresh.connect()
        return fresh

    monkeypatch.setattr(onecar, "_link", link)
    by_sub = {}
    for line in runbook_commands():
        if not line.startswith("export "):
            module, rest = argv_of(line, kh, wifi)
            by_sub.setdefault((module.rsplit(".", 1)[1], rest[0] if module.endswith(("onecar", "ghosts")) else ""),
                              []).append(rest)

    def records(step):
        out = []
        for p in sorted(logs.glob(f"*-{step}.jsonl")):
            out += [json.loads(row) for row in p.read_text().splitlines()]
        return out

    # A missing known_hosts or WiFi.json is a failed preflight with a reason, not a traceback.
    assert onecar.main(["preflight", "--known-hosts", str(tmp_path / "absent"), "--wifi-json", str(tmp_path / "absent.json")]) == 1
    failed = records("preflight")
    assert {r["event"] for r in failed} == {"known_hosts_error", "wifi_json_error", "preflight_result"}
    for p in logs.glob("*-preflight.jsonl"):
        p.unlink()

    # ---- Block 1: preflight and probes, exactly as written -----------------
    with monkeypatch.context() as m:
        m.setattr(onecar.time, "sleep", lambda s: None)
        for sub in ("preflight", "status", "capabilities", "handover-probe"):
            (argv,) = by_sub[("onecar", sub)]
            assert onecar.main(argv) == 0, sub
    assert records("preflight")[-1] == {**records("preflight")[-1], "event": "preflight_result", "ok": True}
    assert records("status")[-1]["passed"] and [r["action"] for r in car.requests] == ["status"] * 10
    block1 = json.dumps(records("capabilities") + records("handover-probe"))
    text = RUNBOOK.read_text(encoding="utf-8")
    told_to_read = re.findall(r"^- `(\w+)`", text[text.index("**Read in the output:**"):text.index("### Block 2")], re.M)
    assert len(told_to_read) == 4
    for key in told_to_read:                                    # every key the operator is told to read exists
        assert f'"{key}"' in block1, key
    caps = next(r for r in records("capabilities") if r["event"] == "capabilities")
    assert caps["two_clients_two_channels"] is True and caps["bluetooth_le"] is True
    probe = next(r for r in records("handover-probe") if r["event"] == "handover_capabilities")
    assert probe["dbus_roam_method"] and probe["aps_advertising_fast_transition"] == 3 and probe["gateway"] == "10.0.0.1"

    # ---- Laps (blocks 2, 4, 5, 6): the runbook's flags, a shorter lap -------
    clock_jump = [0.0]
    monkeypatch.setattr(tel, "record_lap",
                        functools.partial(tel.record_lap, clock=lambda: time.time() + clock_jump[0]))

    def lap(template, name, y, block=False):
        argv = [LAP_S if prev == "--seconds" else a for prev, a in zip([None] + template, template)]
        argv = [str(a) for a in argv]
        argv[argv.index("--name") + 1] = name
        car.route_y, car.block, car.marks_done = y, block, threading.Event()
        clock_jump[0] = 0.0
        monkeypatch.setattr(sys, "stdin", Keyboard(clock_jump, car.marks_done))
        assert onecar.main(argv) == 0, name
        car.marks_done = None
        clock_jump[0] = 0.0

    survey, route, load, blocked = (by_sub[("onecar", "lap")][i] for i in range(4))
    assert "--note" in blocked and survey[survey.index("--seconds") + 1] == "180"
    for i, y in enumerate((5.0, 6.0, 7.0)):
        lap(survey, f"survey{i + 1}", y)

    # ---- Block 3: handover timing ------------------------------------------
    target = APS[1][0]
    switch_bssid, switch_any = by_sub[("onecar", "switch")]
    (roam_argv,) = by_sub[("onecar", "roam")]
    parse = onecar.build_parser().parse_args
    with monkeypatch.context() as m:
        m.setattr(onecar.time, "sleep", lambda s: None)
        # A wrong confirmation sends nothing.
        assert onecar.cmd_switch(parse(switch_bssid), link=link(), ask=lambda _: "aa:bb:cc:00:00:09") == 1
        assert not any(r["action"] == "connect" for r in car.requests)
        # Today's switch, timed.
        assert onecar.cmd_switch(parse(switch_bssid), link=link(), ask=lambda _: target) == 0
        assert car.bssid == target and car.locked
        # The profile is BSSID-locked now, so a fast roam is refused before anything is sent.
        back = APS[0][0]
        assert onecar.cmd_roam(parse(roam_argv[:-1] + [back]), link=link(), ask=lambda _: back) == 1
        assert not any("Roam s" in c for c in car.commands)
        # Unlock the profile once (SSID only), then roam.
        assert onecar.cmd_switch(parse(switch_any), link=link(), ask=lambda _: SSID) == 0
        assert not car.locked
        assert onecar.cmd_roam(parse(roam_argv[:-1] + [back]), link=link(), ask=lambda _: back) == 0
        assert car.bssid == back
        # A connect whose channel drops is UNKNOWN: status is re-read, the connect is never resent.
        sent_before = sum(r["action"] == "connect" for r in car.requests)
        car.drop_connect = True
        assert onecar.cmd_switch(parse(switch_bssid), link=link(), ask=lambda _: target) == 1
        car.drop_connect = False
        assert sum(r["action"] == "connect" for r in car.requests) == sent_before + 1
    connects = [r for r in car.requests if r["action"] == "connect"]
    assert [("bssid" in r) for r in connects] == [True, False, True]
    assert sum("Roam s" in c for c in car.commands) == 1
    timings = [r for r in records("switch") + records("roam") if r["event"] == "handover_timing"]
    legacy, fast = timings[0], next(r for r in records("roam") if r["event"] == "handover_timing")
    assert legacy["teardown_seen"] and 3.3 < legacy["outage_s"] < 3.5 and legacy["stages_s"]["dhcp"] is not None
    assert 0.2 < legacy["silence_started_after_s"] < 0.35 and legacy["validation"] is None   # replies continued after submission
    assert 3.6 < legacy["stages_s"]["first_packet"] < 3.8
    assert fast["submit_clock"] == "car" and not fast["teardown_seen"] and fast["outage_s"] < 0.3
    assert fast["ip_retained"] and fast["stages_s"]["association"] == pytest.approx(0.03, abs=5e-3)
    results = [r["outcome"] for r in records("switch") if r["event"] == "switch_result"]
    assert results == ["APPLIED", "APPLIED", "UNKNOWN"]
    assert [r["outcome"] for r in records("roam") if r["event"] == "roam_result"] == ["APPLIED"]

    # ---- Block 4: ten route laps --------------------------------------------
    for i in range(10):
        lap(route, f"r{i + 1:02d}", 3.0 + 0.6 * i)

    # ---- Block 5: ghost load, then a lap under it ---------------------------
    sent = []

    class Socket:
        def __init__(self, *a):
            pass

        def sendto(self, payload, addr):
            sent.append((addr, len(payload)))

        def close(self):
            pass

    monkeypatch.setattr(ghosts.socket, "socket", Socket)
    (ghost_argv,) = by_sub[("ghosts", "send")]
    g = ghosts.build_parser().parse_args(ghost_argv)
    report = ghosts.send(g.target, g.port, ghosts.fleet(g.ghosts, g.video, g.bulk), g.seconds,
                         sleep=lambda s: time.sleep(0.3))
    assert len(report["ghosts"]) == 9 and {a for a, _ in sent} == {("10.0.0.5", 47000)}
    assert all(v["sent_mbps"] > 0 and v["send_errors"] == 0 for v in report["ghosts"].values())
    for bad in ("127.0.0.1", "8.8.8.8", MANAGEMENT_HOST):
        with pytest.raises(ValueError):
            ghosts.send(bad, 47000, ghosts.fleet(), 5, sleep=lambda s: None)
    lap(load, "load1", 6.0)

    # ---- Block 6: a blockage lap --------------------------------------------
    lap(blocked, "block1", 6.0, block=True)

    # ---- Block 7: check the data --------------------------------------------
    lap_files = sorted(logs.glob("*-lap-*.jsonl"))
    assert len(lap_files) == 15
    for p in lap_files:
        data = tel.load_lap(p)
        n = int(LAP_S / 0.1)
        assert len(data["samples"]) == n and len(data["scans"]) == n // 30 and len(data["marks"]) == 2
        assert len(data["pings"]) >= 500 and data["meta"]["gateway"] == "10.0.0.1"
        assert abs(data["meta"]["car_minus_mac_s"] - SKEW) < 0.5
        a, b = (mk["t_mac"] for mk in data["marks"])
        assert 29.0 < b - a < 30.0                             # the marks span the lap on the laptop clock
        xy = tel.positions([s["t_car"] for s in data["samples"]], data["marks"], WAYPOINTS,
                           data["meta"]["car_minus_mac_s"])
        assert (~(xy != xy).any(axis=1)).sum() > 0.9 * n        # nearly every sample gets a position
    assert tel.load_lap(next(p for p in lap_files if p.name.endswith("lap-block1.jsonl")))["meta"]["note"] == "blocker at C"

    capsys.readouterr()
    (analyse_argv,) = by_sub[("analyse", "")]
    assert analyse_argv == ["artifacts/onecar/*-lap-*.jsonl"]
    assert analyse.main([str(p) for p in lap_files]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert len(rep["laps"]) == 15 and all(r["samples"] and r["pings"] and r["marks"] >= 2 for r in rep["laps"])
    by_name = {r["name"]: r for r in rep["laps"]}
    assert by_name["block1"]["outages"] == 1 and 1.4 < by_name["block1"]["longest_gap_s"] < 1.7
    assert by_name["r05"]["outages"] == 0 and by_name["r05"]["roams_seen"] >= 1
    replay = rep["blockage_rule_replay"]
    assert replay["total"]["outages"] == 1 and replay["total"]["caught"] == 1 and replay["total"]["false_alarms"] == 0
    assert set(rep["outage_prediction_baseline"]) == {"0.1", "0.5", "1.0"}

    site, waypoints = tmp_path / "site.json", tmp_path / "waypoints.json"
    site.write_text(json.dumps(SITE))
    waypoints.write_text(json.dumps(WAYPOINTS))
    (twin_argv,) = by_sub[("trace_twin", "")]
    (coverage_argv,) = by_sub[("coverage", "")]
    assert twin_argv == ["artifacts/onecar/*-lap-r*.jsonl", "--site", "site.json", "--waypoints", "waypoints.json"]
    r_laps = sorted(logs.glob("*-lap-r*.jsonl"))
    assert len(r_laps) == 10                                    # the glob picks the ten route laps and nothing else
    assert trace_twin.main([str(p) for p in r_laps] + ["--site", str(site), "--waypoints", str(waypoints)]) == 0
    twin = json.loads(capsys.readouterr().out)
    assert twin["robots"] == 10 and 25 < twin["duration_s"] <= 30
    assert set(twin["arms"]) == {"wifi7_fixed_rule", "single_radio", "second_path_same_card"}
    for arm in twin["arms"].values():
        assert arm["control_sent"] > 0 and 0.0 <= arm["control_miss"] <= 1.0

    # ---- Coverage report from the same laps ----------------------------------
    from wirelessjev.native_ext import coverage
    assert coverage_argv[:2] == ["--laps", "artifacts/onecar/*-lap-*.jsonl"] and coverage_argv[-2:] == ["--out", "artifacts/onecar/coverage"]
    out_prefix = tmp_path / "coverage_out" / "coverage"
    all_laps = sorted(logs.glob("*-lap-*.jsonl"))
    assert coverage.main(["--laps", *map(str, all_laps), "--site", str(site), "--waypoints", str(waypoints), "--out", str(out_prefix)]) == 0
    cov = json.loads(capsys.readouterr().out)["report"]
    assert cov["source"]["laps"] == len(all_laps) and cov["area_m2"] > 0
    assert out_prefix.with_suffix(".png").exists() and out_prefix.with_suffix(".json").exists()

    # ---- Safety rules, over everything the dry run did -----------------------
    for cmd in car.commands:
        assert MANAGEMENT_HOST not in cmd
        # wlan0 may only ever appear as the guard that tells SmartCarCmd which radio to leave alone.
        assert "wlan0" not in cmd.replace("--management-interface wlan0", "")
        if cmd != CMD:
            assert not re.search(r"\bnmcli\b.*\b(up|down|modify|delete|connect)\b|\biw\b.*\b(connect|disconnect|set)\b|"
                                 r"scan trigger|rescan|\breboot\b|\bip\b.*\b(add|del|set)\b", cmd), cmd
    written = list(logs.rglob("*"))
    assert len(written) >= 15 + 7
    for p in written:
        body = p.read_text()
        assert WIFI_PW not in body and SSH_PW not in body and "password" not in body.lower(), p.name
