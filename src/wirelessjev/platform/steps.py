"""The field day as structured steps: what each one does, how it runs, and when it counts as done."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
import sys

PY = sys.executable

# Kinds: "setup" (done on the page), "read" (read-only on the car, runs from the page),
# "car_change" (changes the car's radio: terminal only, typed confirmation), "analysis" (offline).
KINDS = ("setup", "read", "car_change", "analysis")


@dataclass(frozen=True)
class Step:
    id: str
    block: str
    title: str
    kind: str
    why: str
    repeat: int = 1                    # laps: how many are wanted
    lap_prefix: str = ""               # laps: name prefix (survey -> survey1, survey2, ...)
    needs: tuple = field(default_factory=tuple)

    def as_dict(self):
        return {"id": self.id, "block": self.block, "title": self.title, "kind": self.kind, "why": self.why,
                "repeat": self.repeat, "lap_prefix": self.lap_prefix, "needs": list(self.needs)}


STEPS = [
    Step("files", "Setup", "Car files on this laptop", "setup",
         "The car's host key and Wi-Fi list must exist before anything can connect."),
    Step("site", "Setup", "Site sheet: APs and waypoints", "setup",
         "AP positions and taped waypoints turn laps into positioned measurements."),
    Step("preflight", "1", "Preflight", "read", "Checks the host key and Wi-Fi list parse; touches nothing on the car.",
         needs=("files",)),
    Step("status", "1", "Car status (10 reads)", "read", "Confirms the management link and reads the test radio.",
         needs=("preflight",)),
    Step("capabilities", "1", "Capabilities probe", "read",
         "Answers whether the card can hold two AP connections (two_clients_two_channels).", needs=("status",)),
    Step("probe", "1", "Handover probe", "read", "Finds which handover methods the car offers.",
         needs=("status",)),
    Step("join", "3", "Join the car to the test network", "car_change",
         "First connection of the test radio to your APs. Changes the car: terminal only.", needs=("probe",)),
    Step("check", "1b", "Format check", "read",
         "A 10 s lap; PASS/WARN/FAIL for everything the analysis needs. Never skip.", needs=("join",)),
    Step("survey", "2", "Survey laps", "read", "Three laps along the route: the training data for the radio map.",
         repeat=3, lap_prefix="survey", needs=("check", "site")),
    Step("switch", "3", "Handover timing: timed switch", "car_change",
         "Measures today's switch outage, 5 or more times. Changes the car: terminal only.", repeat=5,
         needs=("survey",)),
    Step("roam", "3", "Handover timing: fast roam", "car_change",
         "Measures 802.11r roam outage, 5 or more times. Changes the car: terminal only.", repeat=5,
         needs=("switch",)),
    Step("route", "4", "Ten route laps", "read", "r01-r10: replayed afterwards as a 10-robot fleet.", repeat=10,
         lap_prefix="r", needs=("check", "site")),
    Step("block", "6", "Blockage trials", "read",
         "Laps with a person or cart in the path: measures how long real blockages last.", repeat=2,
         lap_prefix="block", needs=("check",)),
    Step("load", "5", "Lap under ghost load", "read",
         "Start ghost traffic in a terminal (runbook block 5), then record this lap.", repeat=1, lap_prefix="load",
         needs=("check",)),
    Step("scan", "Extra", "Scan the APs", "read", "A scan the WirelessJEV proposal reads.", needs=("join",)),
    Step("propose", "Extra", "WirelessJEV proposal (shadow)", "analysis",
         "WirelessJEV picks the AP it would use for the mission. Nothing is sent to the car.", needs=("scan",)),
    Step("analyse", "7", "Lap analysis", "analysis", "Outages, plan-length stops, blockage-rule replay.",
         needs=("check",)),
    Step("twin", "7", "10-robot replay", "analysis", "The route laps replayed as a fleet in the simulator.",
         needs=("route", "site")),
    Step("coverage", "7", "Coverage report", "analysis", "A coverage map of the site from all laps.",
         needs=("survey", "site")),
]
BY_ID = {s.id: s for s in STEPS}


@dataclass
class Paths:
    repo: Path
    log_dir: Path
    known_hosts: Path
    wifi_json: Path

    @property
    def site_json(self):
        return self.log_dir / "site.json"

    @property
    def waypoints_json(self):
        return self.log_dir / "waypoints.json"


def lap_files(log_dir: Path, prefix: str) -> list[Path]:
    pat = re.compile(rf"\d{{8}}-\d{{6}}-lap-{re.escape(prefix)}\d+\.jsonl$")
    return sorted(p for p in log_dir.glob("*-lap-*.jsonl") if pat.search(p.name))


def next_lap_name(log_dir: Path, prefix: str) -> str:
    used = [int(re.search(r"-lap-[A-Za-z]+(\d+)\.jsonl$", p.name).group(1)) for p in lap_files(log_dir, prefix)]
    n = max(used, default=0) + 1
    return f"{prefix}{n:02d}" if prefix == "r" else f"{prefix}{n}"


def command(step: Step, paths: Paths, extra: dict | None = None) -> list[str]:
    """The exact command for a step, as an argument list. car_change steps return the command
    to show the user; the platform never runs them."""
    extra = extra or {}
    kh, wj = str(paths.known_hosts), str(paths.wifi_json)
    onecar = [PY, "-m", "wirelessjev.onecar"]
    if step.id == "preflight":
        return onecar + ["preflight", "--known-hosts", kh, "--wifi-json", wj]
    if step.id == "status":
        return onecar + ["status", "--known-hosts", kh, "--repeat", "10"]
    if step.id == "capabilities":
        return onecar + ["capabilities", "--known-hosts", kh]
    if step.id == "probe":
        return onecar + ["handover-probe", "--known-hosts", kh]
    if step.id == "check":
        return onecar + ["check", "--known-hosts", kh]
    if step.lap_prefix:
        name = extra.get("name") or next_lap_name(paths.log_dir, step.lap_prefix)
        cmd = onecar + ["lap", "--known-hosts", kh, "--name", name, "--seconds", str(int(extra.get("seconds", 180)))]
        if extra.get("note"):
            cmd += ["--note", str(extra["note"])]
        return cmd
    if step.id == "scan":
        return onecar + ["scan", "--known-hosts", kh, "--wifi-json", wj, "--seconds", "20"]
    if step.id == "propose":
        scans = sorted(paths.log_dir.glob("*-scan.jsonl"))
        scan_log = str(scans[-1]) if scans else "<no scan log yet>"
        return onecar + ["propose", "--scan-log", scan_log, "--mission", str(extra.get("mission") or
                                                                               "Keep the robot's control link reliable")]
    if step.id == "join":
        return onecar + ["switch", "--timing", "--known-hosts", kh, "--wifi-json", wj, "--ssid",
                         str(extra.get("ssid") or "<SSID>"), "--bssid", "any"]
    if step.id == "switch":
        return onecar + ["switch", "--timing", "--known-hosts", kh, "--wifi-json", wj, "--ssid",
                         str(extra.get("ssid") or "<SSID>"), "--bssid", str(extra.get("bssid") or "<bssid>")]
    if step.id == "roam":
        return onecar + ["roam", "--known-hosts", kh, "--bssid", str(extra.get("bssid") or "<bssid>")]
    if step.id == "analyse":
        return [PY, "-m", "wirelessjev.hil.analyse"] + [str(p) for p in sorted(paths.log_dir.glob("*-lap-*.jsonl"))]
    if step.id == "twin":
        return [PY, "-m", "wirelessjev.hil.trace_twin"] + [str(p) for p in lap_files(paths.log_dir, "r")] + \
               ["--site", str(paths.site_json), "--waypoints", str(paths.waypoints_json)]
    if step.id == "coverage":
        return [PY, "-m", "wirelessjev.native_ext.coverage", "--laps"] + \
               [str(p) for p in sorted(paths.log_dir.glob("*-lap-*.jsonl"))] + \
               ["--site", str(paths.site_json), "--waypoints", str(paths.waypoints_json),
                "--out", str(paths.log_dir / "coverage")]
    raise ValueError(f"step {step.id!r} has no command")


def shell_text(cmd: list[str], repo: Path) -> str:
    """A command as the user would type it from the repo folder."""
    shown = [".venv/bin/python" if c == PY else c for c in cmd]
    return " ".join(shlex.quote(c) for c in shown)


def progress(step: Step, paths: Paths, done_ids: dict) -> dict:
    """How far a step has got: count done, wanted, and whether it is complete."""
    if step.id == "files":
        n = int(paths.known_hosts.exists() and paths.wifi_json.exists())
    elif step.id == "site":
        n = int(paths.site_json.exists() and paths.waypoints_json.exists())
    elif step.lap_prefix:
        n = len(lap_files(paths.log_dir, step.lap_prefix))
    else:
        n = int(done_ids.get(step.id, 0))
    return {"done": n, "wanted": step.repeat, "complete": n >= step.repeat}
