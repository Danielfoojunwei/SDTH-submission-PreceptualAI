"""Field Platform control and oversight: typed confirmations and holds, settings, the map,
the router check, the local-model assistant and the activity log. No car, no network."""
import io
import json
import sys
import time
from pathlib import Path

import pytest

from wirelessjev import onecar
from wirelessjev.hil import viewer
from wirelessjev.platform import agent as agent_mod
from wirelessjev.platform import demo, mapview, params, sitecheck
from wirelessjev.platform.audit import Audit
from wirelessjev.platform.server import Platform, result_of
from wirelessjev.platform import steps as st


def make(tmp_path, monkeypatch, password=True):
    kh, wj, log = tmp_path / "kh", tmp_path / "WiFi.json", tmp_path / "onecar"
    kh.write_text("192.168.1.11 ssh-ed25519 AAAA\n")
    wj.write_text("{}")
    log.mkdir()
    if password:
        monkeypatch.setenv("WJ_SSH_PASSWORD", "x")
    else:
        monkeypatch.delenv("WJ_SSH_PASSWORD", raising=False)
    paths = st.Paths(Path.cwd(), log, kh, wj)
    demo.write_site(log)
    return Platform(paths, assistant=agent_mod.Assistant(client=None))


def fake_car(outcome):
    """A stand-in for `onecar switch/roam --confirm-stdin`: prompts, checks the typing, reports an outcome."""
    code = ("import json,sys\n"
            "print('CONFIRM> Switch wlan1 to aa:bb:cc:00:00:02. Re-type the target BSSID to confirm: ', flush=True)\n"
            "typed = sys.stdin.readline().strip()\n"
            "if typed != 'aa:bb:cc:00:00:02':\n"
            "    print(json.dumps({'event': 'cancelled', 'reason': 'confirmation did not match; nothing sent'})); sys.exit(1)\n"
            f"print(json.dumps({{'event': 'switch_result', 'outcome': {outcome!r}}}))\n"
            f"sys.exit(0 if {outcome!r} in ('APPLIED', 'NOT_APPLIED') else 1)\n")
    return [sys.executable, "-c", code]


def wait(plat, cond, n=200):
    for _ in range(n):
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_car_change_waits_for_typed_confirmation_and_unknown_outcome_holds(tmp_path, monkeypatch):
    plat = make(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr("wirelessjev.platform.server.command",
                        lambda step, paths, extra=None, cfg=None, confirm_stdin=False: calls.append(confirm_stdin) or
                        (fake_car(outcome["v"]) if step.kind == "car_change" else [sys.executable, "-c", "print('ok')"]))
    outcome = {"v": "APPLIED"}
    with pytest.raises(ValueError):
        plat.run("switch", {"ssid": "fleet", "bssid": "not-a-bssid"})          # target checked before anything runs
    job = plat.run("switch", {"ssid": "fleet", "bssid": "aa:bb:cc:00:00:02"})
    assert calls[-1] is True                                                  # the page path reads the confirm from stdin
    assert wait(plat, lambda: plat.runner.current() and plat.runner.current().summary()["confirm"])
    assert "Re-type the target BSSID" in plat.snapshot()["running"]["confirm"]
    assert plat.snapshot()["next"]["action"] == "confirm"
    plat.confirm(job["id"], "aa:bb:cc:00:00:02")
    with pytest.raises(RuntimeError):
        plat.confirm(job["id"], "aa:bb:cc:00:00:02")                          # one answer only
    assert wait(plat, lambda: plat.runner.current() is None)
    assert plat.done.get("switch") == 1 and plat.hold is None

    # A wrong answer cancels: nothing counted, no hold.
    job = plat.run("switch", {"ssid": "fleet", "bssid": "aa:bb:cc:00:00:02"})
    assert wait(plat, lambda: plat.runner.current() and plat.runner.current().summary()["confirm"])
    plat.confirm(job["id"], "aa:bb:cc:00:00:09")
    assert wait(plat, lambda: plat.runner.current() is None)
    assert plat.done.get("switch") == 1 and plat.hold is None

    # UNKNOWN holds every further change until a status read succeeds.
    outcome["v"] = "UNKNOWN"
    job = plat.run("switch", {"ssid": "fleet", "bssid": "aa:bb:cc:00:00:02"})
    assert wait(plat, lambda: plat.runner.current() and plat.runner.current().summary()["confirm"])
    plat.confirm(job["id"], "aa:bb:cc:00:00:02")
    assert wait(plat, lambda: plat.runner.current() is None)
    assert "UNKNOWN" in plat.hold and json.loads(plat.state_file.read_text())["hold"]
    with pytest.raises(PermissionError):
        plat.run("roam", {"bssid": "aa:bb:cc:00:00:01"})
    assert plat.snapshot()["next"]["step"] == "status"
    plat.run("status", {})
    assert wait(plat, lambda: plat.runner.current() is None)
    assert plat.hold is None
    acts = [a["action"] for a in plat.audit.tail()]
    assert {"command_start", "confirmation", "command_end", "hold_cleared"} <= set(acts)


def test_demo_and_missing_password_refuse_car_changes(tmp_path, monkeypatch):
    plat = make(tmp_path, monkeypatch, password=False)
    with pytest.raises(PermissionError):
        plat.run("switch", {"ssid": "fleet", "bssid": "aa:bb:cc:00:00:02"})
    plat.demo = True
    monkeypatch.setenv("WJ_SSH_PASSWORD", "x")
    with pytest.raises(PermissionError):
        plat.run("roam", {"bssid": "aa:bb:cc:00:00:02"})
    assert result_of(['{"event": "roam_result", "outcome": "SUBMITTED_NOT_VERIFIED"}']) == "SUBMITTED_NOT_VERIFIED"
    assert result_of(['{"event": "cancelled", "reason": "confirmation did not match; nothing sent"}']) == "CANCELLED"


def test_onecar_confirm_stdin_prompt_is_marked_and_terminal_rule_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("aa:bb:cc:00:00:02\n"))
    assert onecar._stdin_ask("Re-type the target BSSID to confirm: ") == "aa:bb:cc:00:00:02"
    assert capsys.readouterr().out.startswith(onecar.CONFIRM_MARK)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))                       # not a terminal, no flag: refused
    assert onecar.main(["roam", "--known-hosts", "kh", "--bssid", "aa:bb:cc:00:00:02"]) == 1
    args = onecar.build_parser().parse_args(["switch", "--known-hosts", "k", "--wifi-json", "w", "--ssid", "s",
                                             "--bssid", "any", "--confirm-stdin"])
    assert args.confirm_stdin


def test_settings_validate_persist_and_drive_commands(tmp_path, monkeypatch):
    plat = make(tmp_path, monkeypatch)
    bad = [{"lap_seconds": 5000}, {"lap_period": 0.3}, {"local_url": "http://example.com:11434"},
           {"local_url": "http://192.168.0.5:11434"}, {"ghost_target": "192.168.1.20"}, {"ghost_target": "8.8.8.8"},
           {"ghost_bulk": 9, "ghost_video": 9}, {"nope": 1}, {"mission": "x" * 500}]
    for change in bad:
        with pytest.raises(ValueError):
            plat.set_config(change)
    assert plat.config.overrides == {}                                       # nothing half-saved
    plat.set_config({"lap_seconds": 240, "lap_period": "0.2", "gap_s": 0.5, "switch_timing": False, "ghost_target": "10.0.0.1"})
    again = params.Config(plat.paths.log_dir / "platform_config.json")
    assert again.get("lap_seconds") == 240 and again.get("lap_period") == 0.2
    cfg = plat.cfg()
    lap = st.command(st.BY_ID["route"], plat.paths, {}, cfg)
    assert lap[lap.index("--seconds") + 1] == "240" and lap[lap.index("--period") + 1] == "0.2"
    assert st.command(st.BY_ID["analyse"], plat.paths, {}, cfg)[-2:] == ["--gap", "0.5"]
    sw = st.command(st.BY_ID["switch"], plat.paths, {"ssid": "fleet", "bssid": "aa:bb:cc:00:00:02"}, cfg, confirm_stdin=True)
    assert "--timing" not in sw and sw[-1] == "--confirm-stdin"
    assert "--target 10.0.0.1" in st.ghost_command(cfg)
    plat.set_config({"lap_seconds": 180})                                     # back to default: override dropped
    assert "lap_seconds" not in plat.config.overrides
    assert [a for a in plat.audit.tail() if a["action"] == "setting" and a["id"] == "lap_seconds"][0]["new"] == 180


def test_map_positions_roams_outages_and_health_from_a_demo_lap(tmp_path):
    demo.write_site(tmp_path)
    path = demo.demo_lap(tmp_path, seconds=200, realtime=False)
    cfg = {p.id: p.default for p in params.PARAMS}
    m = mapview.lap_map(viewer.read_events(path), demo.SITE, demo.WAYPOINTS, cfg, now_mac=None)
    assert m["lap"]["finished"] and len(m["points"]) > 1000 and m["health"]["positioned_share"] > 0.9
    W, H = demo.SITE["dimensions_m"][:2]
    assert all(0 <= x <= W and 0 <= y <= H for x, y, *_ in m["points"])
    placed = [r for r in m["roams"] if r["x"] is not None]
    assert len(placed) >= len(m["roams"]) - 1 and placed                      # one after the last mark may have no place
    assert len(m["outages"]) >= 4                                              # a person steps in every 40 s
    assert 0 <= m["health"]["score"] <= 100 and "Score =" in m["health"]["definition"]
    assert {a["bssid"] for a in m["aps"]} == {a["bssid"] for a in demo.SITE["aps"]}
    assert abs(sum(a["serving_share"] for a in m["aps"]) - 1.0) < 0.01
    stricter = mapview.lap_map(viewer.read_events(path), demo.SITE, demo.WAYPOINTS, cfg | {"good_dbm": -50}, now_mac=None)
    assert stricter["health"]["signal_good_share"] < m["health"]["signal_good_share"]
    # No marks, no positions: nothing is invented.
    events = [e for e in viewer.read_events(path) if e.get("event") != "mark"]
    assert mapview.lap_map(events, demo.SITE, demo.WAYPOINTS, cfg, now_mac=None)["points"] == []
    merged = mapview.merged_map(tmp_path, demo.SITE, demo.WAYPOINTS, cfg)
    assert merged["laps"] == 1 and merged["readings"] >= len(merged["points"])


def test_router_check_compares_the_cars_scan_with_the_sheet():
    site = json.loads(json.dumps(demo.SITE))
    scan = {"source": "scan step", "aps": [
        {"bssid": "aa:bb:cc:00:00:01", "ssid": "fleet", "freq_mhz": 5180, "signal_dbm": -60},
        {"bssid": "aa:bb:cc:00:00:02", "ssid": "fleet", "freq_mhz": 5300, "signal_dbm": -62},     # wrong channel
        {"bssid": "aa:bb:cc:00:00:09", "ssid": "fleet", "freq_mhz": 5745, "signal_dbm": -70}]}    # not on the sheet
    rows = {r["item"]: r["status"] for r in sitecheck.check(site, scan)}
    assert rows["AP aa:bb:cc:00:00:01"] == "PASS"
    assert rows["AP aa:bb:cc:00:00:02 channel"] == "FAIL"
    assert rows["AP aa:bb:cc:00:00:03"] == "FAIL"                                              # not heard
    assert rows["Unknown AP aa:bb:cc:00:00:09"] == "WARN"
    site["aps"][1]["frequency_mhz"] = 5180
    assert any(r["item"].startswith("Channel 36") and r["status"] == "WARN" for r in sitecheck.check(site, None))
    assert sitecheck.check(None, None)[0]["status"] == "FAIL"


def test_assistant_uses_a_local_model_on_this_laptop_only():
    guide = {"step": "check", "action": "run", "text": "Run the format check."}
    base = {p.id: p.default for p in params.PARAMS} | {"local_model": "small-instruct"}
    sent = []

    def ollama(url, body, timeout):
        sent.append((url, body))
        return {"message": {"content": json.dumps({"answer": "Run the format check first.", "suggested_step": "check"})}}
    r = agent_mod.Assistant(local_post=ollama).ask("what next?", {"steps": []}, guide, base)
    assert r["source"] == "local" and r["suggested_step"] == "check"
    url, body = sent[0]
    assert url == "http://127.0.0.1:11434/api/chat" and body["format"] == agent_mod.SCHEMA and body["stream"] is False
    assert "How the platform works" in body["messages"][0]["content"]

    def openai(url, body, timeout):
        assert url.endswith("/v1/chat/completions") and body["response_format"]["type"] == "json_schema"
        return {"choices": [{"message": {"content": 'Sure: {"answer": "Save the site sheet.", "suggested_step": "nope"}'}}]}
    r = agent_mod.Assistant(local_post=openai).ask("?", {}, guide, base | {"local_kind": "openai-compatible",
                                                                           "local_url": "http://localhost:8080"})
    assert r["source"] == "local" and r["answer"] == "Save the site sheet." and r["suggested_step"] == "none"

    def down(url, body, timeout):
        raise ConnectionRefusedError()
    r = agent_mod.Assistant(local_post=down).ask("How do I set the router channels?", {}, guide, base | {"assistant_order": "local, then guide"})
    assert r["source"] == "guide" and "Router setup" in r["answer"] and "local model unavailable" in r["note"]
    r = agent_mod.Assistant(local_post=ollama).ask("?", {}, guide, base | {"local_url": "http://10.0.0.5:11434"})
    assert r["source"] != "local"                                              # never sends site data off the laptop
    st_ = agent_mod.local_status(base, probe=lambda url, t: {"models": [{"name": "small-instruct:latest"}]})
    assert st_["available"]
    assert not agent_mod.local_status(base | {"local_model": "other"}, probe=lambda url, t: {"models": [{"name": "x"}]})["available"]


def test_audit_never_records_password_fields(tmp_path):
    a = Audit(tmp_path / "audit.jsonl")
    a.write("setting", id="lap_seconds", old=180, new=240)
    with pytest.raises(ValueError):
        a.write("oops", ssh_password="x")
    assert [r["action"] for r in a.tail()] == ["setting"]


def test_spectrum_map_covers_the_floor_and_matches_the_synthetic_truth(tmp_path):
    from wirelessjev.platform import spectrum
    import numpy as np
    demo.write_site(tmp_path)
    demo.demo_lap(tmp_path, seconds=300, realtime=False, name="survey1")
    cfg = {p.id: p.default for p in params.PARAMS}
    cache = spectrum.Cache()
    out = cache.get(tmp_path, demo.SITE, demo.WAYPOINTS, cfg, time.time())
    g = out["grid"]
    n = g["nx"] * g["ny"]
    assert all(len(out["layers"][k]) == n for k in ("best", "best_ap", "backup", "snr", "sinr", "cochan", "uncertainty", "near_m"))
    assert len(out["layers"]["per_ap"]) == 3 and out["fit"]["readings"] > 1000 and out["fit"]["aps_measured"] == 3
    assert set(out["layers"]["best_ap"]) == {0, 1, 2}                    # every AP serves part of the floor
    # The map is a fit to the car's readings: compare it with the generator's own physics near the route.
    near = np.array(out["layers"]["near_m"]).reshape(g["ny"], g["nx"])
    err = []
    for i, ap in enumerate(demo.SITE["aps"]):
        P = np.array(out["layers"]["per_ap"][i], dtype=float).reshape(g["ny"], g["nx"])
        for j, k in zip(*np.nonzero(near <= 2.0)):
            err.append(P[j, k] - demo.rssi(((k + 0.5) * g["cell"], (j + 0.5) * g["cell"]), ap))
    assert np.sqrt(np.mean(np.square(err))) < 3.0
    # Backup comes from another AP location, so it is never stronger than the best AP.
    b = np.array([v if v is not None else -999 for v in out["layers"]["best"]])
    k = np.array([v if v is not None else -999 for v in out["layers"]["backup"]])
    assert (k <= b + 1).all()
    assert cache.get(tmp_path, demo.SITE, demo.WAYPOINTS, cfg, time.time()) is out   # cached until a lap changes
    # Without any lap: the site sheet's physics only, said so.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert spectrum.Cache().get(empty, demo.SITE, demo.WAYPOINTS, cfg, time.time())["fit"]["source"] == "site sheet only"
    with pytest.raises(ValueError):
        spectrum.Cache().get(empty, None, None, cfg, time.time())


def test_obstruction_counts_boxes_on_the_path():
    from wirelessjev.platform.spectrum import obstruction
    import numpy as np
    boxes = [[[6.0, 6.6], [18.0, 7.4]]]
    n = obstruction(12.0, 2.0, np.array([12.0, 12.0, 2.0]), np.array([12.0, 4.0, 2.5]), boxes)
    assert n.tolist() == [1.0, 0.0, 0.0]


def test_help_shown_command_and_field_kit():
    from wirelessjev.platform.help import page_help
    h = page_help()
    assert any("local" in b["title"].lower() for b in h["before"]) and h["whats_new"][0]["items"]
    assert any("Fixed" in x for x in h["whats_new"][0]["items"])
    shown = st.shell_text([st.PY, "-m", "wirelessjev.onecar", "status"], Path.cwd())
    assert shown.split()[0] in (".venv/bin/python", st.PY) and shown.endswith("-m wirelessjev.onecar status")
    import importlib.util
    spec = importlib.util.spec_from_file_location("kit", Path(__file__).resolve().parents[1] / "scripts" / "make_field_kit.py")
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    names = [f.name.lower() for f in kit.files()]
    assert "field_platform.sh" in names and not any(r in n for n in names for r in kit.REFUSE)


def test_full_kit_carries_runtime_data_and_proposal_explains_a_missing_model(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("kit", Path(__file__).resolve().parents[1] / "scripts" / "make_field_kit.py")
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    names = {str(f.relative_to(kit.ROOT)) for f in kit.code_files()}
    assert "artifacts/warehouse_v2/studies/wifi_link_curves.json" in names       # the 10-robot replay reads it
    assert {"requirements-field.txt", "requirements-wirelessjev.txt", "field_platform.sh"} <= names
    plat = make(tmp_path, monkeypatch)
    demo.write_scan(plat.paths.log_dir)
    cmd = st.command(st.BY_ID["propose"], plat.paths, {}, plat.cfg())
    assert cmd[cmd.index("--scan-log") + 1].endswith("-scan.jsonl")
    monkeypatch.setattr("wirelessjev.platform.steps.jev_status",
                        lambda: {"ready": False, "reason": "not installed (torch missing)", "fix": "Start it with ./field_platform.sh"})
    with pytest.raises(PermissionError, match="field_platform.sh"):
        plat.run("propose", {})
    status = st._jev_check()
    assert set(status) == {"ready", "reason", "fix"}
