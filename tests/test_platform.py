"""Field Platform: steps, guide, safety rules, jobs, assistant and HTTP surface. No car, no network."""
import json
import sys
import time
from pathlib import Path

import pytest

from wirelessjev.platform import agent as agent_mod
from wirelessjev.platform import steps as st
from wirelessjev.platform.guide import next_step
from wirelessjev.platform.server import Platform

SITE = {"schema": 1, "site_id": "test-hall", "dimensions_m": [20.0, 12.0, 4.0],
        "aps": [{"bssid": "aa:bb:cc:00:00:01", "ssid": "fleet", "xyz_m": [2.0, 2.0, 3.0], "frequency_mhz": 5180, "tx_power_dbm": 20.0},
                {"bssid": "aa:bb:cc:00:00:02", "ssid": "fleet", "xyz_m": [18.0, 2.0, 3.0], "frequency_mhz": 5260, "tx_power_dbm": 20.0}],
        "cad_boxes_m": [], "receiver": {"noise_floor_dbm": -92.0, "offset_db": 0.0, "quantization_db": 1.0,
                                        "reference_gauge_assumed": True}}


def make_paths(tmp_path, files=True):
    kh, wj = tmp_path / "kh", tmp_path / "WiFi.json"
    if files:
        kh.write_text("192.168.1.11 ssh-ed25519 AAAA\n")
        wj.write_text("{}")
    log = tmp_path / "onecar"
    log.mkdir()
    return st.Paths(Path.cwd(), log, kh, wj)


def touch_lap(log, name, t="20261118-101500"):
    (log / f"{t}-lap-{name}.jsonl").write_text(json.dumps({"event": "lap_start", "name": name}) + "\n")


def test_step_commands_are_the_tested_cli_and_car_changes_are_never_run(tmp_path, monkeypatch):
    paths = make_paths(tmp_path)
    cmd = st.command(st.BY_ID["check"], paths)
    assert cmd[1:4] == ["-m", "wirelessjev.onecar", "check"] and str(paths.known_hosts) in cmd
    touch_lap(paths.log_dir, "survey1")
    touch_lap(paths.log_dir, "r01")
    touch_lap(paths.log_dir, "r02", "20261118-101600")
    assert st.next_lap_name(paths.log_dir, "survey") == "survey2" and st.next_lap_name(paths.log_dir, "r") == "r03"
    lap = st.command(st.BY_ID["route"], paths, {"seconds": 60, "note": "blocker at C"})
    assert lap[lap.index("--name") + 1] == "r03" and lap[lap.index("--seconds") + 1] == "60" and "blocker at C" in lap
    assert st.progress(st.BY_ID["route"], paths, {}) == {"done": 2, "wanted": 10, "complete": False}
    shown = st.shell_text(st.command(st.BY_ID["roam"], paths, {"bssid": "aa:bb:cc:00:00:02"}), paths.repo)
    assert shown.startswith(".venv/bin/python -m wirelessjev.onecar roam") and "aa:bb:cc:00:00:02" in shown
    plat = Platform(paths, assistant=agent_mod.Assistant(client=None))
    monkeypatch.delenv("WJ_SSH_PASSWORD", raising=False)
    for sid in ("join", "switch", "roam"):
        with pytest.raises(PermissionError):
            plat.run(sid, {})                                     # car changes need the SSH password too
    with pytest.raises(PermissionError):
        plat.run("check", {})                                     # no SSH password: refuse, never prompt
    with pytest.raises(ValueError):
        plat.run("files", {})
    monkeypatch.setenv("WJ_SSH_PASSWORD", "x")
    with pytest.raises(ValueError):
        plat.run("route", {"name": "r 01; rm -rf"})               # names are checked before anything runs


def test_jobs_run_one_at_a_time_feed_marks_and_record_completion(tmp_path, monkeypatch):
    paths = make_paths(tmp_path)
    plat = Platform(paths, assistant=agent_mod.Assistant(client=None))
    echo = [sys.executable, "-c", "import sys\nfor line in sys.stdin:\n    print('got', line.strip(), flush=True)\n    if line.strip() == 'B': break"]
    monkeypatch.setattr("wirelessjev.platform.server.command", lambda step, paths, extra=None, cfg=None, **kw: echo + ["lap"])
    job = plat.run("analyse", {})
    with pytest.raises(RuntimeError):
        plat.run("twin", {})                                      # one command at a time
    plat.runner.mark("A")
    plat.runner.mark("B")
    for _ in range(100):
        if not plat.runner.current():
            break
        time.sleep(0.05)
    out = plat.runner.get(job["id"]).summary()
    assert out["returncode"] == 0 and "got A" in out["output"] and "got B" in out["output"]
    assert plat.done.get("analyse") == 1 and json.loads(plat.state_file.read_text())["done"]["analyse"] == 1
    with pytest.raises(RuntimeError):
        plat.runner.mark("C")                                     # nothing recording


def test_guide_orders_steps_and_reacts_to_a_failed_format_check(tmp_path):
    bare = make_paths(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, files=False)
    assert next_step(bare, {}, None)["step"] == "files"
    paths = make_paths(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    n = next_step(paths, {}, None)
    assert n["step"] == "site"                                    # measure the site first, as in the runbook
    Platform(paths, assistant=agent_mod.Assistant(client=None)).save_site({"site": SITE, "waypoints": {"A": [1, 6], "B": [19, 6]}})
    n = next_step(paths, {}, None)
    assert n["step"] == "routers"                                 # then the router checklist
    for item, _ in __import__("wirelessjev.platform.sitecheck", fromlist=["CHECKLIST"]).CHECKLIST:
        Platform(paths, assistant=agent_mod.Assistant(client=None)).tick(item, True)
    n = next_step(paths, {}, None)
    assert n["step"] == "preflight" and n["action"] == "run"
    n = next_step(paths, {"preflight": 1, "status": 1, "capabilities": 1, "probe": 1}, None)
    assert n["step"] == "join" and "onecar switch" in n["command"] and "--bssid any" in n["command"]
    # A format check whose lap has no gateway and no samples: the guide says fix, not carry on.
    (paths.log_dir / "20261118-101500-lap-check.jsonl").write_text(json.dumps({"event": "lap_start", "name": "check", "gateway": None}) + "\n")
    n = next_step(paths, {"preflight": 1, "status": 1, "capabilities": 1, "probe": 1, "join": 1}, None)
    assert n["step"] == "check" and n["action"] == "fix" and "gateway" in n["text"]
    assert next_step(paths, {}, {"step": "route"})["action"] == "wait"


def test_site_sheet_is_validated_before_saving(tmp_path):
    paths = make_paths(tmp_path)
    plat = Platform(paths, assistant=agent_mod.Assistant(client=None))
    with pytest.raises(ValueError):
        plat.save_site({"site": SITE, "waypoints": {"A": [1.0, 6.0]}})          # one waypoint is not a route
    with pytest.raises(Exception):
        plat.save_site({"site": {**SITE, "aps": [{"bssid": "nope"}]}, "waypoints": {"A": [1, 6], "B": [19, 6]}})
    assert not paths.site_json.exists()
    out = plat.save_site({"site": SITE, "waypoints": {"A": [1, 6], "B": [19, 6]}})
    assert out["aps"] == 2 and json.loads(paths.waypoints_json.read_text()) == {"A": [1.0, 6.0], "B": [19.0, 6.0]}
    assert st.progress(st.BY_ID["site"], paths, {})["complete"]


class FakeBlock:
    def __init__(self, text):
        self.type, self.text = "text", text


class FakeResponse:
    def __init__(self, text, stop="end_turn"):
        self.content, self.stop_reason = [FakeBlock(text)], stop


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.beta = self
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return self.reply


def test_assistant_uses_claude_when_it_can_and_the_guide_when_it_cannot():
    guide = {"step": "check", "action": "run", "text": "Run the format check."}
    ok = FakeClient(FakeResponse(json.dumps({"answer": "Run the format check now.", "suggested_step": "check"})))
    a = agent_mod.Assistant(client=ok)
    r = a.ask("what next?", {"steps": []}, guide)
    assert r["source"] == "claude" and r["suggested_step"] == "check"
    kw = ok.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"]["format"]["type"] == "json_schema"
    assert "password" not in json.dumps(kw["messages"]).lower() or "never ask" in agent_mod.SYSTEM.lower()
    bad = agent_mod.Assistant(client=FakeClient(FakeResponse(json.dumps({"answer": "x", "suggested_step": "reboot"}))))
    assert bad.ask("?", {}, guide)["suggested_step"] == "none"
    offline = agent_mod.Assistant(client=FakeClient(error=ConnectionError("no route to host")))
    r = offline.ask("?", {}, guide)
    assert r["source"] == "guide" and r["answer"].startswith("Run the format check") and "unavailable" in r["note"]
    refused = agent_mod.Assistant(client=FakeClient(FakeResponse("", stop="refusal")))
    assert refused.ask("?", {}, guide)["source"] == "guide"


def test_http_surface_refuses_other_origins(tmp_path, monkeypatch):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from wirelessjev.platform.server import make_handler
    paths = make_paths(tmp_path)
    monkeypatch.delenv("WJ_SSH_PASSWORD", raising=False)
    plat = Platform(paths, assistant=agent_mod.Assistant(client=None))
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    except PermissionError:
        pytest.skip("local port binding is not allowed here")
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(plat, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Field Platform" in page and "https://" not in page.split("<script>")[1]
        assert "alert(" not in page                                # errors show on the page, not in popups
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert {s["id"] for s in state["steps"]} >= {"check", "route", "coverage"} and state["next"]["step"] == "site"

        def post(path, body, origin):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", **({"Origin": origin} if origin else {})})
            try:
                return urllib.request.urlopen(req).status, None
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
        assert post("/api/run", {"step": "preflight"}, "https://evil.example")[0] == 403
        assert post("/api/run", {"step": "preflight"}, None)[0] == 403
        code, err = post("/api/run", {"step": "roam"}, base)
        assert code == 409 and "terminal" in err["error"]
        assert urllib.request.urlopen(base + "/live").status == 200
    finally:
        srv.shutdown()
        plat.runner.stop()
