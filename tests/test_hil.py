"""Hardware-in-the-loop tools: telemetry parsing, ghosts and the trace twin.
No network and no car: the stream is a recorded-format fixture."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from wirelessjev.hil import ghosts, telemetry as tel
from wirelessjev.hil.trace_twin import build_trace_world, site_index
from wirelessjev.native.sim import policies, simulate

APS = [("aa:bb:cc:00:00:01", 2.0, 2.0, 5180), ("aa:bb:cc:00:00:02", 18.0, 2.0, 5260),
       ("aa:bb:cc:00:00:03", 10.0, 10.0, 5500)]
SITE = {"schema": 1, "site_id": "test-hall", "dimensions_m": [20.0, 12.0, 4.0],
        "aps": [{"bssid": b, "ssid": "fleet", "xyz_m": [x, y, 3.0], "frequency_mhz": f, "tx_power_dbm": 20.0}
                for b, x, y, f in APS],
        "cad_boxes_m": [], "receiver": {"noise_floor_dbm": -92.0, "offset_db": 0.0, "quantization_db": 1.0,
                                        "reference_gauge_assumed": True}}
WAYPOINTS = {"A": [1.0, 6.0], "B": [19.0, 6.0]}
T0 = 1794000000.0


def rssi(x, y, ax, ay):
    return 20.0 - (46.7 + 33 * math.log10(max(1.0, math.hypot(x - ax, y - ay))))


def stream_text(seconds=30, period=0.5, y=6.0):
    out, n = [], int(seconds / period)
    for i in range(n):
        t = T0 + i * period
        x = 1.0 + 18.0 * i / (n - 1)
        best = max(APS, key=lambda a: rssi(x, y, a[1], a[2]))
        out.append(f"@T {t:.6f}")
        out.append(f"Connected to {best[0]} (on wlan1)\n\tSSID: fleet\n\tfreq: {best[3]}\n"
                   f"\tsignal: {rssi(x, y, best[1], best[2]):.0f} dBm\n\ttx bitrate: 234.0 MBit/s")
        out.append(f"Station {best[0]} (on wlan1)\n\ttx packets:\t{100 * i}\n\ttx retries:\t{3 * i}\n"
                   f"\ttx failed:\t0\n\tbeacon loss:\t0\n\tsignal avg:\t-60 dBm")
        if i % 4 == 0:
            out.append("@SCAN")
            for b, ax, ay, f in APS:
                out.append(f"BSS {b}(on wlan1)\n\tfreq: {f}\n\tsignal: {rssi(x, y, ax, ay):.2f} dBm\n"
                           f"\tlast seen: 200 ms ago\n\tSSID: fleet")
            out.append("@ENDSCAN")
    return "\n".join(out) + "\n"


def make_lap(name, y):
    p = tel.StreamParser()
    events = []
    text = stream_text(y=y)
    for k in range(0, len(text), 257):                     # split reads mid-line on purpose
        events += p.feed(text[k:k + 257])
    events += p.close()
    return {"meta": {"name": name, "car_minus_mac_s": 0.0},
            "samples": [e for e in events if e["event"] == "sample"],
            "scans": [e for e in events if e["event"] == "scan"], "pings": [],
            "marks": [{"t_mac": T0, "label": "A"}, {"t_mac": T0 + 29.5, "label": "B"}]}


def test_commands_are_fixed_and_validated():
    cmd = tel.stream_command(0.1, 30, 60)
    assert "wlan0" not in cmd and "scan dump" in cmd and "trigger" not in cmd and "connect" not in cmd
    for bad in ((0.3, 30, 60), (0.1, 1, 60), (0.1, 30, 5000)):
        with pytest.raises(ValueError):
            tel.stream_command(*bad)
    with pytest.raises(ValueError):
        tel.ping_command("10.0.0.1; reboot", 30)


def test_stream_parser_handles_split_reads_and_scans():
    lap = make_lap("a", 6.0)
    assert len(lap["samples"]) == 60 and len(lap["scans"]) == 15
    s = lap["samples"][10]
    assert s["connected"] and s["bssid"] in {a[0] for a in APS} and s["tx_retries"] == 30.0
    assert len(lap["scans"][0]["aps"]) == 3 and lap["scans"][0]["aps"][0]["age_ms"] == 200


def test_positions_interpolate_and_never_extrapolate():
    marks = [{"t_mac": 100.0, "label": "A"}, {"t_mac": 110.0, "label": "B"}]
    xy = tel.positions([95.0, 100.0, 105.0, 110.0, 120.0], marks, WAYPOINTS, 0.0)
    assert np.isnan(xy[0]).all() and np.isnan(xy[4]).all()
    assert np.allclose(xy[2], [10.0, 6.0])
    # A label typed in the other case still finds its waypoint; an unknown or ambiguous one does not.
    typed = [{"t_mac": 100.0, "label": "a"}, {"t_mac": 105.0, "label": "zz"}, {"t_mac": 110.0, "label": "b"}]
    assert np.allclose(tel.positions([105.0], typed, WAYPOINTS, 0.0), [[10.0, 6.0]])
    assert tel.waypoint_of("a", {"A": [0, 0], "a": [1, 1]}) == "a" and tel.waypoint_of("B", {"b": [0, 0]}) == "b"
    assert tel.waypoint_of("c", {"C": [0, 0], "c ": [1, 1]}) == "C" and tel.waypoint_of("x", WAYPOINTS) is None
    assert tel.waypoint_of("q", {"Q": [0, 0], "q": [1, 1]}) == "q" and tel.waypoint_of("Q", {"q": 1, "Q": 2}) == "Q"


def test_ghost_targets_and_caps():
    for bad in ("8.8.8.8", "127.0.0.1", "224.0.0.1"):
        with pytest.raises(ValueError):
            ghosts.check_target(bad)
    assert ghosts.check_target("192.168.50.1") == "192.168.50.1"
    f = ghosts.fleet(9)
    assert len(f) == 9 and sum(g.video_mbps > 0 for g in f) == 1
    with pytest.raises(ValueError):
        ghosts.send("192.168.50.1", 47000, [ghosts.Ghost("g", bulk_mbps=500.0)], 1)


def test_trace_world_runs_the_unchanged_simulator():
    laps = [make_lap(f"lap{i}", y) for i, y in enumerate((5.0, 6.0, 7.0))]
    sites, index = site_index(SITE)
    assert len(sites) == 3 and len(index) == 3
    w = build_trace_world(laps, SITE, WAYPOINTS)
    assert w.n == 3 and w.snr.shape[1:] == (3, 3, 3) and np.isfinite(w.forecast).all()
    # 5 GHz is the only band these APs have; the others are marked absent.
    assert (w.rssi_dl[..., 0] < -140).all() and (w.rssi_dl[..., 1] > -100).any()
    # Measured signal is strongest near each AP's end of the route.
    assert w.rssi_dl[0, 1, 0, 1] > w.rssi_dl[0, 1, 1, 1]
    pol = {p.name: p for p in policies()}
    nat = replace(pol["native"], gate=True)
    for p in (pol["mlo_split"], nat):
        r = simulate(w, p)
        assert 0.0 <= r["control_miss"] <= 1.0 and r["control_sent"] > 0
    with pytest.raises(ValueError):
        build_trace_world(laps[:1], SITE, WAYPOINTS)


def test_lap_command_records_and_redacts(tmp_path, monkeypatch):
    import wirelessjev.onecar as onecar
    from wirelessjev import handover as ho
    monkeypatch.setattr(onecar, "LOG_DIR", tmp_path)

    class Chan:
        def __init__(self, text):
            self.data, self.sent = text.encode(), False
        def recv_ready(self): return not self.sent
        def recv(self, n):
            self.sent = True
            return self.data
        def exit_status_ready(self): return True
        def close(self): pass

    class Out:
        def __init__(self, text=""):
            self.channel, self.text = Chan(text), text
        def read(self): return self.text.encode()
        def close(self): pass

    class Client:
        def __init__(self): self.commands = []
        def exec_command(self, cmd, timeout=None):
            self.commands.append(cmd)
            R = ho.HANDOVER_READONLY
            table = {R["wlan1_addr"]: "3: wlan1    inet 10.0.0.23/24 brd 10.0.0.255\n",
                     R["wlan1_route"]: "default via 10.0.0.1 proto dhcp\n", R["car_clock"]: f"{T0:.6f}\n"}
            if cmd in table:
                return Out(), Out(table[cmd]), Out()
            if cmd.startswith("sudo -n ping"):
                return Out(), Out(f"[{T0 + 1:.6f}] 64 bytes from 10.0.0.1: icmp_seq=1 ttl=64 time=2.5 ms\n"), Out()
            assert cmd == tel.stream_command(0.5, 30, 30)
            return Out(), Out(stream_text()), Out()
        def close(self): pass

    class Link:
        client = Client()
        def close(self): pass

    args = type("A", (), dict(name="lap1", seconds=30, period=0.5, scan_every=30, note=""))()
    assert onecar.cmd_lap(args, link=Link(), marks=iter(["A", "bad label!", "B"])) == 0
    log = next(tmp_path.glob("*-lap-lap1.jsonl"))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "lap_start" and kinds[-1] == "lap_end" and kinds.count("sample") == 60
    assert kinds.count("ping") == 1 and [e["label"] for e in events if e["event"] == "mark"] == ["A", "B"]
    assert all("wlan0" not in c for c in Link.client.commands)
    assert tel.load_lap(log)["meta"]["gateway"] == "10.0.0.1"


def test_outage_analysis_from_ping_gaps():
    from wirelessjev.hil import analyse
    lap = make_lap("a", 6.0)
    t = [T0 + 0.05 * i for i in range(600) if not 200 <= i < 230]      # one 1.5 s outage at t = 10 s
    lap["pings"] = [{"t_car": x, "rtt_ms": 3.0} for x in t]
    s = analyse.lap_summary(lap)
    assert s["outages"] == 1 and 1.4 < s["longest_gap_s"] < 1.6
    assert s["stops_if_plan_s"] == {"0.2": 1, "0.5": 1, "1.0": 1, "2.0": 0}
    X, Y = analyse.outage_dataset(lap)
    assert X.shape == (60, 4) and Y.shape == (60, 3)
    times = np.array([e["t_car"] for e in lap["samples"]]) - T0
    assert Y[(times >= 10.0) & (times < 11.4), 0].all() and not Y[times < 9.0].any()
    assert Y[np.isclose(times, 9.5), 2][0] and not Y[np.isclose(times, 9.5), 0][0]   # 1 s warning, not 0.1 s
    b = analyse.persistence_brier(X, Y, Y.mean(0))
    assert b["0.1"]["brier_persistence"] <= b["0.1"]["brier_base_rate"]


def test_blockage_rule_replay_scores_against_real_outages():
    from wirelessjev.hil import analyse
    lap = make_lap("a", 6.0)
    assert analyse.detector_replay(lap) == []                # a smooth drive never fires the rule
    # A blocker at t = 10 s: 18 dB down for 2 s, and replies stop 0.5 s after it starts.
    for s in lap["samples"]:
        if 10.0 <= s["t_car"] - T0 < 12.0:
            s["signal_dbm"] -= 18.0
    lap["pings"] = [{"t_car": T0 + 0.05 * i, "rtt_ms": 3.0} for i in range(600) if not 210 <= i < 240]
    fired = analyse.detector_replay(lap)
    assert len(fired) >= 1 and 10.4 < fired[0] - T0 < 10.6   # second low reading at 0.5 s samples
    s = analyse.detector_score(lap)
    assert s["outages"] == 1 and s["caught"] == 1 and s["false_alarms"] == 0
    assert s["median_lead_s"] is not None and -0.1 < s["median_lead_s"] < 0.1
    # The same dip with no outage behind it is a false alarm.
    lap["pings"] = [{"t_car": T0 + 0.05 * i, "rtt_ms": 3.0} for i in range(600)]
    s = analyse.detector_score(lap)
    assert s["outages"] == 0 and s["caught"] == 0 and s["false_alarms"] == s["fired"] >= 1
    # A single low reading is not enough.
    lap2 = make_lap("b", 6.0)
    lap2["samples"][20]["signal_dbm"] -= 30.0
    assert analyse.detector_replay(lap2) == []


def _fake_car(stream, pings, stream_err=""):
    """A car that prints `stream` for the telemetry loop and `pings` for the gateway ping."""
    from wirelessjev import handover as ho

    class Chan:
        def __init__(self, text, err=""):
            self.data, self.err, self.sent, self.err_sent = text.encode(), err.encode(), False, False
        def recv_ready(self): return not self.sent
        def recv(self, n):
            self.sent = True
            return self.data
        def recv_stderr_ready(self): return bool(self.err) and not self.err_sent
        def recv_stderr(self, n):
            self.err_sent = True
            return self.err
        def exit_status_ready(self): return True
        def close(self): pass

    class Out:
        def __init__(self, text="", err=""):
            self.channel, self.text = Chan(text, err), text
        def read(self): return self.text.encode()
        def close(self): pass

    class Client:
        def __init__(self): self.commands = []
        def exec_command(self, cmd, timeout=None):
            self.commands.append(cmd)
            R = ho.HANDOVER_READONLY
            table = {R["wlan1_addr"]: "3: wlan1    inet 10.0.0.23/24 brd 10.0.0.255\n",
                     R["wlan1_route"]: "default via 10.0.0.1 proto dhcp\n", R["car_clock"]: f"{T0:.6f}\n"}
            if cmd in table:
                return Out(), Out(table[cmd]), Out()
            if cmd.startswith("sudo -n ping"):
                return Out(), Out(pings), Out()
            return Out(), Out(stream, stream_err), Out()
        def close(self): pass

    class Link:
        def __init__(self): self.client = Client()
        def close(self): pass

    return Link()


def test_site_check_passes_on_expected_output_and_keeps_the_raw_stream(tmp_path, monkeypatch, capsys):
    import argparse
    import wirelessjev.onecar as onecar
    monkeypatch.setattr(onecar, "LOG_DIR", tmp_path)
    pings = "".join(f"[{T0 + 0.05 * i:.6f}] 64 bytes from 10.0.0.1: icmp_seq={i} ttl=64 time=2.5 ms\n" for i in range(200))
    link = _fake_car(stream_text(seconds=10, period=0.1), pings)
    assert onecar.cmd_check(argparse.Namespace(seconds=10), link=link) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out.split("{")[0] and json.loads(out.strip().splitlines()[-1])["check"] == "PASS"
    assert link.client.commands[-2] == tel.stream_command(0.1, 30, 10) or tel.stream_command(0.1, 30, 10) in link.client.commands
    log = next(tmp_path.glob("*-lap-check.jsonl"))
    raws = tel.raw_paths(log)
    assert raws["stream"].read_text().count("@T ") == 100 and raws["ping"].read_text().count("bytes from") == 200
    for f in tmp_path.iterdir():
        assert "password" not in f.read_text().lower() and "psk" not in f.read_text().lower()
    assert all("192.168.1.11" not in c for c in link.client.commands)
    # Parsing the raw files again gives the same lap.
    before = tel.load_lap(log)
    counts = tel.reparse_lap(log)
    after = tel.load_lap(log)
    assert counts == {"sample": 100, "scan": len(before["scans"]), "ping": 200}
    assert [s["signal_dbm"] for s in after["samples"]] == [s["signal_dbm"] for s in before["samples"]]
    assert after["meta"]["gateway"] == "10.0.0.1" and log.with_name(log.name + ".orig").exists()


def test_site_check_fails_clearly_and_a_parser_fix_recovers_the_lap(tmp_path, monkeypatch, capsys):
    import argparse
    import wirelessjev.onecar as onecar
    monkeypatch.setattr(onecar, "LOG_DIR", tmp_path)
    # A car whose `iw` prints the signal differently, with no scan rights and no ping replies.
    odd = stream_text(seconds=10, period=0.1).replace("signal: ", "rssi = ")
    odd = "\n".join(line for line in odd.split("\n") if not line.startswith(("BSS ", "\tlast seen", "\tSSID: fleet\n")))
    link = _fake_car(odd, "", stream_err="sudo: a password is required\n")
    assert onecar.cmd_check(argparse.Namespace(seconds=10), link=link) == 1
    out = capsys.readouterr().out
    verdict = json.loads(out.strip().splitlines()[-1])
    assert verdict["check"] == "FAIL" and "field signal_dbm" in verdict["failed"] and "gateway pings" in verdict["failed"]
    assert "sudo: a password is required" in out and "field bssid" not in verdict["failed"]
    log = next(tmp_path.glob("*-lap-check.jsonl"))
    assert all(s["signal_dbm"] is None for s in tel.load_lap(log)["samples"])
    # Fix the parser for this car's format, then rebuild the lap from the raw output: nothing was lost.
    monkeypatch.setitem(tel._NUM, "signal_dbm", r"rssi =\s+(-?\d+)")
    assert tel.reparse_lap(log)["sample"] == 100
    fixed = tel.load_lap(log)
    assert all(s["signal_dbm"] is not None for s in fixed["samples"])
    rows = {r["item"]: r["status"] for r in tel.check_lap(fixed, 10, 0.1)}
    assert rows["field signal_dbm"] == "PASS" and rows["wlan1 default route"] == "PASS"
    with pytest.raises(FileNotFoundError):
        tel.reparse_lap(tmp_path / "missing-lap-x.jsonl")

