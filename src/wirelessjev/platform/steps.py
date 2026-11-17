"""The field day as structured steps: what each one does, how it runs, and when it counts as done."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import shlex
import sys

PY = sys.executable

# Kinds: "setup" (done on the page), "read" (read-only on the car, runs from the page),
# "car_change" (changes the car's radio: typed confirmation of the target), "analysis" (offline).
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
    Step("routers", "Setup", "Router checklist", "setup",
         "Consumer routers are set up in their own app: tick each item once it is done on every router."),
    Step("preflight", "1", "Preflight", "read", "Checks the host key and Wi-Fi list parse; touches nothing on the car.",
         needs=("files",)),
    Step("status", "1", "Car status (10 reads)", "read", "Confirms the management link and reads the test radio.",
         needs=("preflight",)),
    Step("capabilities", "1", "Capabilities probe", "read",
         "Answers whether the card can hold two AP connections (two_clients_two_channels).", needs=("status",)),
    Step("probe", "1", "Handover probe", "read", "Finds which handover methods the car offers.",
         needs=("status",)),
    Step("join", "3", "Join the car to the test network", "car_change",
         "First connection of the test radio to your APs. Changes the car: you confirm by typing the SSID.", needs=("probe",)),
    Step("check", "1b", "Format check", "read",
         "A 10 s lap; PASS/WARN/FAIL for everything the analysis needs. Never skip.", needs=("join",)),
    Step("survey", "2", "Survey laps", "read", "Three laps along the route: the training data for the radio map.",
         repeat=3, lap_prefix="survey", needs=("check", "site")),
    Step("switch", "3", "Handover timing: timed switch", "car_change",
         "Measures today's switch outage, 5 or more times. Changes the car: you confirm by typing the BSSID.", repeat=5,
         needs=("survey",)),
    Step("roam", "3", "Handover timing: fast roam", "car_change",
         "Measures 802.11r roam outage, 5 or more times. Changes the car: you confirm by typing the BSSID.", repeat=5,
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

    @property
    def checklist_json(self):
        return self.log_dir / "router_checklist.json"


def lap_files(log_dir: Path, prefix: str) -> list[Path]:
    pat = re.compile(rf"\d{{8}}-\d{{6}}-lap-{re.escape(prefix)}\d+\.jsonl$")
    return sorted(p for p in log_dir.glob("*-lap-*.jsonl") if pat.search(p.name))


def next_lap_name(log_dir: Path, prefix: str) -> str:
    used = [int(re.search(r"-lap-[A-Za-z]+(\d+)\.jsonl$", p.name).group(1)) for p in lap_files(log_dir, prefix)]
    n = max(used, default=0) + 1
    return f"{prefix}{n:02d}" if prefix == "r" else f"{prefix}{n}"


def _defaults() -> dict:
    from .params import PARAMS
    return {p.id: p.default for p in PARAMS}


def command(step: Step, paths: Paths, extra: dict | None = None, cfg: dict | None = None,
            confirm_stdin: bool = False) -> list[str]:
    """The exact command for a step, as an argument list, using the Settings values in `cfg`.
    With `confirm_stdin`, car_change steps read the typed confirmation from the page's confirm box."""
    extra, c = extra or {}, (cfg or _defaults())
    kh, wj = str(paths.known_hosts), str(paths.wifi_json)
    onecar = [PY, "-m", "wirelessjev.onecar"]
    if step.id == "preflight":
        return onecar + ["preflight", "--known-hosts", kh, "--wifi-json", wj]
    if step.id == "status":
        return onecar + ["status", "--known-hosts", kh, "--repeat", str(c["status_repeat"])]
    if step.id == "capabilities":
        return onecar + ["capabilities", "--known-hosts", kh]
    if step.id == "probe":
        return onecar + ["handover-probe", "--known-hosts", kh]
    if step.id == "check":
        return onecar + ["check", "--known-hosts", kh, "--seconds", str(c["check_seconds"])]
    if step.lap_prefix:
        name = extra.get("name") or next_lap_name(paths.log_dir, step.lap_prefix)
        cmd = onecar + ["lap", "--known-hosts", kh, "--name", name, "--seconds", str(int(extra.get("seconds") or c["lap_seconds"])),
                        "--period", str(c["lap_period"]), "--scan-every", str(c["scan_every"])]
        if extra.get("note"):
            cmd += ["--note", str(extra["note"])]
        return cmd
    if step.id == "scan":
        return onecar + ["scan", "--known-hosts", kh, "--wifi-json", wj, "--seconds", str(c["scan_seconds"])]
    if step.id == "propose":
        scans = sorted(paths.log_dir.glob("*-scan.jsonl"))
        scan_log = str(scans[-1]) if scans else "<no scan log yet>"
        return onecar + ["propose", "--scan-log", scan_log, "--mission", str(extra.get("mission") or c["mission"])]
    tail = ["--confirm-stdin"] if confirm_stdin else []
    if step.id in ("join", "switch"):
        bssid = "any" if step.id == "join" else str(extra.get("bssid") or "<bssid>")
        return onecar + ["switch"] + (["--timing"] if c["switch_timing"] else []) + \
            ["--known-hosts", kh, "--wifi-json", wj, "--ssid", str(extra.get("ssid") or "<SSID>"), "--bssid", bssid,
             "--verify-reads", str(c["switch_verify_reads"]), "--window", str(c["switch_window"])] + tail
    if step.id == "roam":
        return onecar + ["roam", "--known-hosts", kh, "--bssid", str(extra.get("bssid") or "<bssid>"),
                         "--verify-reads", str(c["roam_verify_reads"]), "--window", str(c["roam_window"])] + tail
    if step.id == "analyse":
        return [PY, "-m", "wirelessjev.hil.analyse"] + [str(p) for p in sorted(paths.log_dir.glob("*-lap-*.jsonl"))] + \
            ["--gap", str(c["gap_s"])]
    if step.id == "twin":
        return [PY, "-m", "wirelessjev.hil.trace_twin"] + [str(p) for p in lap_files(paths.log_dir, "r")] + \
               ["--site", str(paths.site_json), "--waypoints", str(paths.waypoints_json), "--collisions", str(c["collisions"])]
    if step.id == "coverage":
        return [PY, "-m", "wirelessjev.native_ext.coverage", "--laps"] + \
               [str(p) for p in sorted(paths.log_dir.glob("*-lap-*.jsonl"))] + \
               ["--site", str(paths.site_json), "--waypoints", str(paths.waypoints_json),
                "--out", str(paths.log_dir / "coverage"), "--cell-m", str(c["cov_cell_m"]), "--knn", str(c["cov_knn"]),
                "--radius-m", str(c["cov_radius_m"]), "--near-m", str(c["cov_near_m"])]
    raise ValueError(f"step {step.id!r} has no command")


def ghost_command(cfg: dict | None = None, repo: Path | None = None) -> str:
    """The ghost-traffic command for a second terminal (it runs alongside a lap)."""
    c = cfg or _defaults()
    target = c["ghost_target"] or "<AP or sink IP>"
    return (f"{shell_text([PY], repo or Path.cwd())} -m wirelessjev.hil.ghosts send --target {target} --ghosts {c['ghost_count']} "
            f"--video {c['ghost_video']} --bulk {c['ghost_bulk']} --seconds {c['ghost_seconds']}")


def shell_text(cmd: list[str], repo: Path) -> str:
    """A command as the user would type it from the platform's folder, with the Python actually in use."""
    try:
        py = str(Path(PY).relative_to(Path(repo).resolve()))
    except ValueError:
        py = PY
    shown = [py if c == PY else c for c in cmd]
    return " ".join(shlex.quote(c) for c in shown)


def progress(step: Step, paths: Paths, done_ids: dict) -> dict:
    """How far a step has got: count done, wanted, and whether it is complete."""
    if step.id == "files":
        n = int(paths.known_hosts.exists() and paths.wifi_json.exists())
    elif step.id == "site":
        n = int(paths.site_json.exists() and paths.waypoints_json.exists())
    elif step.id == "routers":
        from .sitecheck import CHECKLIST
        try:
            ticked = json.loads(paths.checklist_json.read_text())
        except (OSError, ValueError):
            ticked = {}
        n = sum(bool(ticked.get(k)) for k, _ in CHECKLIST)
        return {"done": n, "wanted": len(CHECKLIST), "complete": n >= len(CHECKLIST)}
    elif step.lap_prefix:
        n = len(lap_files(paths.log_dir, step.lap_prefix))
    else:
        n = int(done_ids.get(step.id, 0))
    return {"done": n, "wanted": step.repeat, "complete": n >= step.repeat}


_JEV_CACHE: dict = {}


def jev_status() -> dict:
    """Whether WirelessJEV's d3-lite model can run here (cached for a minute; the page asks often)."""
    import time as _time
    hit = _JEV_CACHE.get("v")
    if hit and _time.time() - hit[0] < 60:
        return hit[1]
    out = _jev_check()
    _JEV_CACHE["v"] = (_time.time(), out)
    return out


def _jev_check() -> dict:
    """Checked without loading the model: the packages are installed and the pinned model revision
    is in the local Hugging Face cache."""
    import importlib.util
    fix = ("Start the platform with ./field_platform.sh from the full field kit: it installs the packages and uses the "
           "d3-lite model bundled in hf-cache/. From the repository: the .venv has the packages, and the model downloads with "
           "huggingface_hub.snapshot_download.")
    missing = [m for m in ("torch", "transformers", "huggingface_hub") if importlib.util.find_spec(m) is None]
    if missing:
        return {"ready": False, "reason": f"not installed ({', '.join(missing)} missing)", "fix": fix}
    try:
        from huggingface_hub import try_to_load_from_cache
        from ..policy import MODEL_REPO, MODEL_REVISION
        hit = try_to_load_from_cache(MODEL_REPO, "model-00001-of-00001.safetensors", revision=MODEL_REVISION)
    except Exception as exc:                                        # an unexpected cache layout: say so, do not guess
        return {"ready": False, "reason": f"model cache not readable ({type(exc).__name__})", "fix": fix}
    if not isinstance(hit, str):
        return {"ready": False, "reason": "packages installed, model not downloaded", "fix": fix}
    return {"ready": True, "reason": f"d3-lite {MODEL_REVISION[:8]} downloaded; runs offline", "fix": ""}
