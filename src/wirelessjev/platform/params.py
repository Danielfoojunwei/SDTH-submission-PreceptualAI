"""Every setting the operator can change, in one place: default, allowed range, unit, what it does
and where it is used. Overrides are saved in the log folder as platform_config.json; anything not
overridden uses the default, so an empty file means "the runbook's values"."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse

from ..hil import ghosts


@dataclass(frozen=True)
class Param:
    id: str
    group: str
    label: str
    kind: str                      # int | float | bool | choice | text | ip | url
    default: object
    why: str
    used_by: str
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    choices: tuple = ()
    max_len: int = 200

    def as_dict(self):
        return {"id": self.id, "group": self.group, "label": self.label, "kind": self.kind, "default": self.default,
                "why": self.why, "used_by": self.used_by, "lo": self.lo, "hi": self.hi, "unit": self.unit,
                "choices": list(self.choices)}


P = Param
PARAMS = [
    # ---- recording laps -------------------------------------------------------------------
    P("lap_seconds", "Laps", "Lap length", "int", 180, "How long each lap records. Long enough to drive the whole route once.",
      "every lap step (survey, route, blockage, load)", 10, 1800, "s"),
    P("lap_period", "Laps", "Link sample period", "choice", 0.1, "How often the car's link state is read. 0.1 s gives 10 readings a second.",
      "every lap step", choices=(0.1, 0.2, 0.5, 1.0), unit="s"),
    P("scan_every", "Laps", "Scan every N samples", "int", 30, "A scan-cache snapshot every N link samples (30 at 0.1 s is every 3 s).",
      "every lap step", 5, 600, "samples"),
    P("check_seconds", "Laps", "Format check length", "int", 10, "Length of the short read-only lap that checks the car's output format.",
      "format check", 5, 120, "s"),
    # ---- reading the car ------------------------------------------------------------------
    P("status_repeat", "Car reads", "Status reads", "int", 10, "How many status reads the status step makes.", "car status", 1, 100),
    P("scan_seconds", "Car reads", "Scan length", "int", 20, "How long the AP scan streams.", "scan the APs", 5, 120, "s"),
    # ---- changing the car's radio ---------------------------------------------------------
    P("switch_timing", "Radio changes", "Time each switch", "bool", True,
      "Ping the gateway and read the car's journal around each switch, to time every stage of the outage.", "join, timed switch"),
    P("switch_verify_reads", "Radio changes", "Switch verify reads", "int", 10,
      "Status reads after a switch to confirm it took. The connect itself is never resent.", "join, timed switch", 1, 60),
    P("switch_window", "Radio changes", "Switch timing window", "int", 20, "Seconds of ping and journal kept around a timed switch.",
      "join, timed switch", 5, 120, "s"),
    P("roam_verify_reads", "Radio changes", "Roam verify reads", "int", 5, "Status reads after a fast roam to confirm it took.",
      "fast roam", 1, 60),
    P("roam_window", "Radio changes", "Roam timing window", "int", 10, "Seconds of ping and journal kept around a roam.",
      "fast roam", 5, 120, "s"),
    # ---- analysis -------------------------------------------------------------------------
    P("gap_s", "Analysis", "Outage threshold", "float", 0.2,
      "A gap in gateway replies longer than this counts as an outage. 0.2 s is the fleet's stop rule.",
      "lap analysis, map, health", 0.05, 5.0, "s"),
    P("collisions", "Analysis", "Replay collision rate", "float", 0.01,
      "Packet collision rate the 10-robot replay assumes between the virtual robots.", "10-robot replay", 0.0, 0.2),
    P("cov_cell_m", "Analysis", "Coverage cell size", "float", 1.0, "Grid cell size of the coverage report.", "coverage report",
      0.25, 5.0, "m"),
    P("cov_knn", "Analysis", "Coverage neighbours", "int", 5, "Readings averaged for each cell (nearest first).", "coverage report", 1, 50),
    P("cov_radius_m", "Analysis", "Coverage reach", "float", 3.0, "Readings further than this from a cell are not used for it.",
      "coverage report", 0.5, 20.0, "m"),
    P("cov_near_m", "Analysis", "Coverage route margin", "float", 1.5, "The report covers floor this close to where the car drove.",
      "coverage report", 0.5, 10.0, "m"),
    P("mission", "Analysis", "WirelessJEV mission", "text", "Keep the robot's control link reliable",
      "The intent WirelessJEV scores the visible APs against (shadow only; nothing is sent).", "WirelessJEV proposal", max_len=200),
    # ---- map and health -------------------------------------------------------------------
    P("good_dbm", "Map and health", "Good signal", "int", -67, "Signal at or above this is good (a common target for real-time traffic).",
      "map colours, health score", -90, -30, "dBm"),
    P("fair_dbm", "Map and health", "Fair signal", "int", -75, "Signal between this and Good is fair; below it is weak.",
      "map colours, health score", -95, -40, "dBm"),
    P("bad_roam_s", "Map and health", "Bad roam", "float", 0.5, "A roam with a reply gap longer than this counts as a bad roam.",
      "health, roam list", 0.05, 10.0, "s"),
    P("latency_good_ms", "Map and health", "Good latency", "float", 50.0, "Gateway round trip at or below this is good.",
      "health score", 1.0, 1000.0, "ms"),
    P("heat_cell_m", "Map and health", "Heatmap cell", "float", 1.0, "Cell size of the measured signal heatmap on the map.",
      "map", 0.25, 5.0, "m"),
    # ---- ghost traffic (runs in a terminal) ----------------------------------------------
    P("ghost_target", "Ghost traffic", "Ghost target IP", "ip", "", "Where ghost traffic goes: the AP or a sink on the test network. "
      "Never 192.168.1.x.", "ghost traffic command"),
    P("ghost_count", "Ghost traffic", "Ghost robots", "int", 9, "Virtual robots sending control and telemetry.", "ghost traffic command",
      0, 30),
    P("ghost_video", "Ghost traffic", "Video ghosts", "int", 1, "Ghosts that also send a video stream.", "ghost traffic command", 0, 9),
    P("ghost_bulk", "Ghost traffic", "Bulk ghosts", "int", 4, "Ghosts that also send bulk uploads.", "ghost traffic command", 0, 9),
    P("ghost_seconds", "Ghost traffic", "Ghost duration", "int", 120, "How long the ghost traffic runs.", "ghost traffic command",
      10, ghosts.MAX_SECONDS, "s"),
    # ---- assistant ------------------------------------------------------------------------
    P("assistant_order", "Assistant", "Who answers", "choice", "local, then Claude, then guide",
      "The assistant tries these in order and says which one answered. The built-in guide always works offline.",
      "assistant", choices=("local, then Claude, then guide", "local, then guide", "Claude, then guide", "guide only")),
    P("local_kind", "Assistant", "Local model server", "choice", "ollama",
      "Ollama, or any OpenAI-compatible server on this laptop (llama.cpp server, LM Studio).", "assistant",
      choices=("ollama", "openai-compatible")),
    P("local_url", "Assistant", "Local server address", "url", "http://127.0.0.1:11434",
      "Must be on this laptop (127.0.0.1 or localhost), so site data never leaves it. "
      "Ollama uses port 11434, llama.cpp 8080, LM Studio 1234.", "assistant"),
    P("local_model", "Assistant", "Local model name", "text", "", "The model to ask, as the server names it (for example a small "
      "instruct model you pulled in Ollama). Empty means the local model is off.", "assistant", max_len=120),
    P("local_timeout_s", "Assistant", "Local model timeout", "int", 60, "How long to wait for the local model before using the guide.",
      "assistant", 5, 600, "s"),
]
BY_ID = {p.id: p for p in PARAMS}
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def coerce(p: Param, value):
    """The value as the right type, or ValueError with a readable reason."""
    if p.kind == "bool":
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "1", "yes", "on"):
            return True
        if str(value).lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{p.label}: yes or no")
    if p.kind in ("int", "float"):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{p.label}: a number is needed") from None
        if p.kind == "int":
            if v != int(v):
                raise ValueError(f"{p.label}: a whole number is needed")
            v = int(v)
        if (p.lo is not None and v < p.lo) or (p.hi is not None and v > p.hi):
            raise ValueError(f"{p.label}: between {p.lo:g} and {p.hi:g} {p.unit}".rstrip())
        return v
    if p.kind == "choice":
        for c in p.choices:
            if str(c) == str(value) or (isinstance(c, float) and _num(value) == c):
                return c
        raise ValueError(f"{p.label}: one of {', '.join(map(str, p.choices))}")
    text = str(value).strip()
    if len(text) > p.max_len:
        raise ValueError(f"{p.label}: at most {p.max_len} characters")
    if any(ord(ch) < 32 for ch in text):
        raise ValueError(f"{p.label}: no control characters")
    if p.kind == "ip" and text:
        try:
            text = ghosts.check_target(text)                     # the ghost tool's own rule: private, never 192.168.1.x
        except ValueError as exc:
            raise ValueError(f"{p.label}: {exc}") from None
    if p.kind == "url":
        u = urlparse(text)
        if u.scheme != "http" or u.hostname not in LOOPBACK or not u.port:
            raise ValueError(f"{p.label}: must be http://127.0.0.1:PORT or http://localhost:PORT, so data stays on this laptop")
    return text


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class Config:
    """Current values: defaults plus saved overrides."""

    def __init__(self, path: Path):
        self.path = path
        self.overrides = {}
        try:
            raw = json.loads(path.read_text())
            for k, v in raw.items():
                if k in BY_ID:
                    self.overrides[k] = coerce(BY_ID[k], v)
        except (OSError, json.JSONDecodeError, ValueError):
            self.overrides = {}

    def get(self, key):
        return self.overrides.get(key, BY_ID[key].default)

    def values(self) -> dict:
        return {p.id: self.get(p.id) for p in PARAMS}

    def update(self, changes: dict) -> dict:
        """Validate every change first; save only if all are valid. Returns {id: (old, new)}."""
        clean = {}
        for k, v in changes.items():
            if k not in BY_ID:
                raise ValueError(f"unknown setting {k!r}")
            clean[k] = coerce(BY_ID[k], v)
        merged = self.values() | clean
        if merged["ghost_video"] + merged["ghost_bulk"] > merged["ghost_count"]:
            raise ValueError("Ghost traffic: video ghosts plus bulk ghosts cannot be more than the ghost robots")
        try:
            load = sum(g.mbps for g in ghosts.fleet(merged["ghost_count"], merged["ghost_video"], merged["ghost_bulk"]))
        except ValueError as exc:
            raise ValueError(f"Ghost traffic: {exc}") from None
        if load > ghosts.MAX_TOTAL_MBPS:
            raise ValueError(f"Ghost traffic: {load:.0f} Mb/s is over the {ghosts.MAX_TOTAL_MBPS:.0f} Mb/s limit; "
                             "use fewer video or bulk ghosts")
        done = {}
        for k, v in clean.items():
            old = self.get(k)
            if v == BY_ID[k].default:
                self.overrides.pop(k, None)
            else:
                self.overrides[k] = v
            if old != v:
                done[k] = (old, v)
        self.path.write_text(json.dumps(self.overrides, indent=1))
        return done

    def view(self) -> list[dict]:
        return [p.as_dict() | {"value": self.get(p.id), "changed": p.id in self.overrides} for p in PARAMS]
