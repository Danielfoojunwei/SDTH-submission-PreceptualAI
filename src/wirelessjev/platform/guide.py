"""What to do next, worked out from the platform's state with no network. The assistant uses
this as its starting point, and it is the answer on site when there is no internet."""
from __future__ import annotations

import os

from ..hil import telemetry as tel
from .steps import BY_ID, STEPS, Paths, command, progress, shell_text


def last_check(paths: Paths) -> list[dict] | None:
    logs = sorted(paths.log_dir.glob("*-lap-check.jsonl"))
    if not logs:
        return None
    lap = tel.load_lap(logs[-1])
    return tel.check_lap(lap, 10, 0.1)


def environment(paths: Paths, agent_status: dict) -> list[dict]:
    rows = [
        {"item": "Car host key", "ok": paths.known_hosts.exists(), "seen": str(paths.known_hosts),
         "fix": "cp ~/Downloads/SmartCar/PC/data/deploy_known_hosts ~/smartcar_known_hosts"},
        {"item": "Car Wi-Fi list", "ok": paths.wifi_json.exists(), "seen": str(paths.wifi_json),
         "fix": "cp ~/Downloads/SmartCar/PI/config/WiFi.json ~/WiFi.json, then add the test network to it"},
        {"item": "SSH password in this terminal", "ok": bool(os.environ.get("WJ_SSH_PASSWORD")),
         "seen": "set" if os.environ.get("WJ_SSH_PASSWORD") else "not set",
         "fix": "Stop the platform, run: read -s WJ_SSH_PASSWORD; export WJ_SSH_PASSWORD  then start it again"},
        {"item": "Site sheet", "ok": paths.site_json.exists() and paths.waypoints_json.exists(),
         "seen": "saved" if paths.site_json.exists() else "not saved", "fix": "Fill the Setup tab and press Save"},
        {"item": "Assistant (Claude)", "ok": agent_status.get("available", False), "seen": agent_status.get("reason", ""),
         "fix": agent_status.get("fix", "")},
    ]
    return rows


def next_step(paths: Paths, done_ids: dict, running: dict | None, extra: dict | None = None) -> dict:
    """The one thing to do now, with the reason."""
    if running:
        step = BY_ID.get(running["step"])
        tip = " Press the waypoint letter as the car passes each marker." if step and step.lap_prefix else ""
        return {"step": running["step"], "action": "wait",
                "text": f"Wait: {step.title if step else running['step']} is running.{tip}"}
    if not (paths.known_hosts.exists() and paths.wifi_json.exists()):
        return {"step": "files", "action": "setup", "text": "Put the car's host key and Wi-Fi list on this laptop (Setup tab shows the commands)."}
    check = last_check(paths)
    fails = [r for r in (check or []) if r["status"] == "FAIL"]
    if fails:
        blocking = [r for r in fails if r["item"] in ("wlan1 default route", "associated on wlan1", "link samples", "gateway pings")]
        r = (blocking or fails)[0]
        return {"step": "check", "action": "fix",
                "text": f"The last format check failed on '{r['item']}' ({r['seen']}). {r['fix']} Then run the format check again."}
    for step in STEPS:
        if step.block == "Extra":
            continue
        p = progress(step, paths, done_ids)
        if p["complete"]:
            continue
        unmet = [n for n in step.needs if not progress(BY_ID[n], paths, done_ids)["complete"]]
        if unmet:
            continue
        out = {"step": step.id, "action": "run" if step.kind in ("read", "analysis") else step.kind,
               "text": f"{step.title} ({p['done']} of {p['wanted']} done). {step.why}"}
        if step.kind == "car_change":
            out["command"] = shell_text(command(step, paths, extra), paths.repo)
            out["text"] += " Run the command below in a terminal; it asks you to retype the target before sending."
        if step.kind == "setup" and step.id == "site":
            out["text"] = "Fill the site sheet (Setup tab): each AP's BSSID, channel and position, and the taped waypoints."
        return out
    return {"step": "analyse", "action": "done",
            "text": "Every planned step has data. Run the analysis, replay and coverage report, then back up artifacts/onecar."}
