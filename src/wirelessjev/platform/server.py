"""HTTP server for the Field Platform. Standard library only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

from ..carlink import BSSID_RE
from ..hil import viewer
from . import agent as agent_mod
from . import mapview, sitecheck, spectrum
from .audit import Audit
from .guide import environment, last_check, next_step
from .page import PAGE
from .params import Config
from .steps import BY_ID, STEPS, Paths, command, ghost_command, progress, shell_text

FILES = {"coverage.png": "image/png", "coverage.json": "application/json"}
MAX_BODY = 200_000
LABEL_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
SSID_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-. ")
SETTLED = {"APPLIED", "NOT_APPLIED", "CANCELLED"}


def result_of(lines) -> str | None:
    """The outcome a car-change command reported ('APPLIED', 'UNKNOWN', ...), or None."""
    for line in reversed(list(lines)):
        if line.startswith("{") and ('_result"' in line or '"cancelled"' in line):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("event", "")).endswith("_result"):
                return rec.get("outcome")
            if rec.get("event") == "cancelled":                  # onecar refused or the typing did not match: nothing sent
                return "CANCELLED"
    return None


class Platform:
    """State shared by every request."""

    def __init__(self, paths: Paths, assistant=None, demo: bool = False):
        from .jobs import Runner
        self.paths, self.demo = paths, demo
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = paths.log_dir / "platform_state.json"
        state = self._load_state()
        self.done, self.hold = state.get("done", {}), state.get("hold")
        self.lock = threading.Lock()
        self.config = Config(paths.log_dir / "platform_config.json")
        self.audit = Audit(paths.log_dir / "platform_audit.jsonl")
        self.runner = Runner(cwd=str(paths.repo), on_finish=self._finished)
        self.assistant = assistant or agent_mod.Assistant()
        self.spectrum = spectrum.Cache()

    # ---- persistence -----------------------------------------------------
    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self) -> None:
        self.state_file.write_text(json.dumps({"done": self.done, "hold": self.hold}, indent=1))

    def _finished(self, job) -> None:
        step = BY_ID[job.step_id]
        with self.lock:
            outcome = None
            if step.kind == "car_change":
                outcome = result_of(job.lines)
                if outcome == "APPLIED":
                    self.done[job.step_id] = self.done.get(job.step_id, 0) + 1
                elif outcome is not None and outcome not in SETTLED:
                    self.hold = f"the last {step.title.lower()} ended with outcome {outcome}."
                elif outcome is None and job.answered:
                    self.hold = f"the last {step.title.lower()} stopped after the confirmation, before reporting an outcome."
            elif job.returncode == 0 and not step.lap_prefix:
                self.done[job.step_id] = self.done.get(job.step_id, 0) + 1
                if job.step_id == "status" and self.hold:
                    self.audit.write("hold_cleared", by="status read", was=self.hold)
                    self.hold = None
            self._save_state()
        self.audit.write("command_end", job=job.id, step=job.step_id, returncode=job.returncode, outcome=outcome,
                         seconds=round((job.ended or 0) - job.started, 1))

    # ---- views -----------------------------------------------------------
    def cfg(self) -> dict:
        return self.config.values()

    def snapshot(self, extra: dict | None = None) -> dict:
        cur = self.runner.current()
        running = cur.summary(tail=0) if cur else None
        cfg = self.cfg()
        steps = []
        for s in STEPS:
            d = s.as_dict() | progress(s, self.paths, self.done)
            try:
                d["command"] = shell_text(command(s, self.paths, extra, cfg), self.paths.repo) if s.kind != "setup" else ""
            except ValueError:
                d["command"] = ""
            steps.append(d)
        site = self.site()
        return {"steps": steps, "running": running,
                "next": next_step(self.paths, self.done, running, extra, cfg, self.hold),
                "environment": environment(self.paths, agent_mod.status(cfg)), "check": last_check(self.paths),
                "jobs": [j.summary(tail=0) for j in self.runner.jobs[-12:]], "site": site,
                "targets": sorted({(a["ssid"], a["bssid"]) for a in (site["site"] or {}).get("aps", []) if a.get("ssid")}),
                "hold": self.hold, "ghost_command": ghost_command(cfg, self.paths.repo), "lap_seconds": cfg["lap_seconds"],
                "log_dir": str(self.paths.log_dir), "demo": self.demo}

    def site(self) -> dict:
        out = {}
        for key, path in (("site", self.paths.site_json), ("waypoints", self.paths.waypoints_json)):
            try:
                out[key] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                out[key] = None
        return out

    def map(self, which: str) -> dict:
        site = self.site()
        cfg = self.cfg()
        if which == "all":
            return {"mode": "all"} | mapview.merged_map(self.paths.log_dir, site["site"], site["waypoints"], cfg) | \
                   {"site": {"dimensions_m": (site["site"] or {}).get("dimensions_m"), "aps": (site["site"] or {}).get("aps", [])},
                    "waypoints": site["waypoints"] or {}, "thresholds": {"good_dbm": cfg["good_dbm"], "fair_dbm": cfg["fair_dbm"]}}
        laps = viewer.list_laps(self.paths.log_dir)
        names = {lap["file"] for lap in laps}
        if which in ("", "newest"):
            if not laps:
                return {"mode": "lap", "empty": True, "site": {"dimensions_m": (site["site"] or {}).get("dimensions_m"),
                                                               "aps": (site["site"] or {}).get("aps", [])},
                        "waypoints": site["waypoints"] or {}}
            which = laps[0]["file"]
        if which not in names:
            raise ValueError("no such lap")
        return {"mode": "lap", "file": which} | mapview.lap_map(
            viewer.read_events(self.paths.log_dir / which), site["site"], site["waypoints"], cfg)

    def router_check(self) -> dict:
        try:
            ticked = json.loads(self.paths.checklist_json.read_text())
        except (OSError, json.JSONDecodeError):
            ticked = {}
        scan = sitecheck.newest_scan(self.paths.log_dir)
        return {"checklist": [{"id": k, "text": t, "done": bool(ticked.get(k))} for k, t in sitecheck.CHECKLIST],
                "scan": {"source": scan["source"], "aps": len(scan["aps"])} if scan else None,
                "rows": sitecheck.check(self.site()["site"], scan)}

    def tick(self, item: str, value: bool) -> dict:
        if item not in dict(sitecheck.CHECKLIST):
            raise ValueError("unknown checklist item")
        try:
            ticked = json.loads(self.paths.checklist_json.read_text())
        except (OSError, json.JSONDecodeError):
            ticked = {}
        ticked[item] = bool(value)
        self.paths.checklist_json.write_text(json.dumps(ticked, indent=1))
        self.audit.write("checklist", item=item, done=bool(value))
        return {"ok": True}

    def assistant_state(self) -> dict:
        snap = self.snapshot()
        lap = None
        laps = viewer.list_laps(self.paths.log_dir)
        if laps:
            m = mapview.lap_map(viewer.read_events(self.paths.log_dir / laps[0]["file"]), snap["site"]["site"],
                                snap["site"]["waypoints"], self.cfg())
            lap = {"lap": m["lap"], "health": {k: v for k, v in m["health"].items() if k != "definition"},
                   "recent_events": m["events"][-8:]}
        jobs = [j.summary(tail=25) for j in self.runner.jobs[-3:]]
        changed = {p["id"]: p["value"] for p in self.config.view() if p["changed"]}
        return {"steps": [{k: st[k] for k in ("id", "block", "title", "kind", "done", "wanted", "complete")} for st in snap["steps"]],
                "environment": [{k: r[k] for k in ("item", "ok", "seen")} for r in snap["environment"]],
                "last_format_check": snap["check"], "latest_lap": lap, "recent_commands": jobs,
                "site_saved": snap["site"]["site"] is not None, "hold": self.hold, "settings_changed": changed,
                "demo": self.demo}

    # ---- actions ---------------------------------------------------------
    def run(self, step_id: str, extra: dict) -> dict:
        step = BY_ID.get(step_id)
        if step is None:
            raise ValueError("unknown step")
        if step.kind == "setup":
            raise ValueError("this step is done on the Setup tab")
        if self.demo and step.kind in ("read", "car_change"):
            raise PermissionError("demo mode: commands that talk to the car are switched off")
        needs_ssh = step.kind in ("read", "car_change") and step.id != "preflight"
        if needs_ssh and not os.environ.get("WJ_SSH_PASSWORD"):
            raise PermissionError("WJ_SSH_PASSWORD is not set in the terminal that started the platform. "
                                  "Stop it, run: read -s WJ_SSH_PASSWORD; export WJ_SSH_PASSWORD  and start it again")
        clean = {k: str(v)[:80] for k, v in (extra or {}).items() if k in ("name", "seconds", "note", "mission", "bssid", "ssid")}
        if "seconds" in clean and not clean["seconds"].isdigit():
            raise ValueError("seconds must be a whole number")
        if "name" in clean and not set(clean["name"]) <= LABEL_OK:
            raise ValueError("lap names use letters, digits, _ or -")
        if step.id == "propose":
            from .steps import jev_status
            st = jev_status()
            if not st["ready"]:
                raise PermissionError(f"WirelessJEV's d3-lite model is {st['reason']}. {st['fix']}")
        if step.kind == "car_change":
            if self.hold:
                raise PermissionError(f"Hold: {self.hold} Run Car status first; never repeat a change whose outcome is unknown.")
            if step.id in ("switch", "roam"):
                clean["bssid"] = clean.get("bssid", "").lower()
                if not BSSID_RE.match(clean["bssid"]):
                    raise ValueError("pick the target BSSID (aa:bb:cc:dd:ee:ff)")
            if step.id in ("join", "switch"):
                if not clean.get("ssid") or not set(clean["ssid"]) <= SSID_OK:
                    raise ValueError("pick the test network (SSID) from the site sheet")
        argv = command(step, self.paths, clean, self.cfg(), confirm_stdin=step.kind == "car_change")
        if any(a.startswith("<") for a in argv):
            raise ValueError("this step needs data that does not exist yet (for example a scan log)")
        job = self.runner.start(step.id, argv, shell_text(argv, self.paths.repo))
        self.audit.write("command_start", job=job.id, step=step.id, kind=step.kind, command=job.shown)
        return job.summary(tail=0)

    def confirm(self, job_id: int, text: str) -> dict:
        out = self.runner.confirm(job_id, text)
        self.audit.write("confirmation", job=job_id, typed=text.strip()[:80], cancelled=out["cancelled"])
        return out

    def save_site(self, body: dict) -> dict:
        from ..smartcar_runtime.contracts import validate_site
        site, waypoints = body.get("site"), body.get("waypoints")
        if not isinstance(waypoints, dict) or len(waypoints) < 2:
            raise ValueError("at least two waypoints are needed, as {\"A\": [x, y], ...}")
        for k, v in waypoints.items():
            if not (set(k) <= LABEL_OK and 0 < len(k) <= 16) or not (isinstance(v, list) and len(v) == 2):
                raise ValueError(f"waypoint {k!r} must be a short label with [x, y] in metres")
            float(v[0]), float(v[1])
        site = validate_site(site)                                  # raises with a readable message
        self.paths.site_json.write_text(json.dumps(site, indent=1))
        self.paths.waypoints_json.write_text(json.dumps({k: [float(v[0]), float(v[1])] for k, v in waypoints.items()}, indent=1))
        self.audit.write("site_saved", aps=len(site["aps"]), waypoints=len(waypoints), site_id=site.get("site_id"))
        return {"saved": [str(self.paths.site_json), str(self.paths.waypoints_json)], "aps": len(site["aps"])}

    def set_config(self, changes: dict) -> dict:
        done = self.config.update(changes)
        for k, (old, new) in done.items():
            self.audit.write("setting", id=k, old=old, new=new)
        return {"changed": list(done), "settings": self.config.view()}

    def ask(self, question: str) -> dict:
        snap = self.snapshot()
        out = self.assistant.ask(question, self.assistant_state(), snap["next"], self.cfg())
        self.audit.write("assistant", question=question[:200], source=out["source"], suggested=out["suggested_step"])
        return out


def make_handler(platform: Platform, port: int):
    allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, ctype, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):        # the browser or a probe hung up: nothing to do
                pass

        def _json(self, status, obj):
            self._send(status, "application/json", json.dumps(obj, default=str).encode())

        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            p, q = url.path, parse_qs(url.query)
            try:
                if p == "/":
                    return self._send(200, "text/html; charset=utf-8", PAGE.encode())
                if p == "/live":
                    return self._send(200, "text/html; charset=utf-8", viewer.PAGE.encode())
                if p in ("/data", "/laps"):                         # the live view's own endpoints
                    st, ct, body = viewer.route(self.path, platform.paths.log_dir)
                    return self._send(st, ct, body)
                if p == "/api/state":
                    return self._json(200, platform.snapshot())
                if p == "/api/map":
                    return self._json(200, platform.map((q.get("lap") or ["newest"])[0]))
                if p == "/api/config":
                    return self._json(200, {"settings": platform.config.view()})
                if p == "/api/spectrum":
                    site = platform.site()
                    import time as _t
                    return self._json(200, platform.spectrum.get(platform.paths.log_dir, site["site"], site["waypoints"],
                                                                 platform.cfg(), _t.time()))
                if p == "/api/help":
                    from .help import page_help
                    return self._json(200, page_help())
                if p == "/api/activity":
                    return self._json(200, {"activity": platform.audit.tail(300)})
                if p == "/api/routers":
                    return self._json(200, platform.router_check())
                if p.startswith("/api/job/"):
                    job = platform.runner.get(int(p.rsplit("/", 1)[1])) if p.rsplit("/", 1)[1].isdigit() else None
                    return self._json(200, job.summary()) if job else self._json(404, {"error": "no such job"})
                if p.startswith("/files/") and p[7:] in FILES:
                    f = platform.paths.log_dir / p[7:]
                    if f.exists():
                        return self._send(200, FILES[p[7:]], f.read_bytes())
            except ValueError as e:
                return self._json(400, {"error": str(e)})
            return self._send(404, "text/plain", b"not found")

        def do_POST(self):
            # Only this page may call the API: refuse other origins and non-JSON bodies.
            if self.headers.get("Origin") not in allowed or not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(403, {"error": "refused: requests must come from the platform page"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad JSON"})
            p = urlparse(self.path).path
            try:
                if p == "/api/run":
                    return self._json(200, platform.run(str(body.get("step")), body.get("extra") or {}))
                if p == "/api/confirm":
                    return self._json(200, platform.confirm(int(body.get("job")), str(body.get("text", ""))))
                if p == "/api/mark":
                    label = str(body.get("label", ""))
                    if not (0 < len(label) <= 16 and set(label) <= LABEL_OK):
                        raise ValueError("bad waypoint label")
                    rec = platform.runner.mark(label)
                    platform.audit.write("mark", label=label)
                    return self._json(200, rec)
                if p == "/api/stop":
                    stopped = platform.runner.stop()
                    platform.audit.write("stop", stopped=stopped)
                    return self._json(200, {"stopped": stopped})
                if p == "/api/site":
                    return self._json(200, platform.save_site(body))
                if p == "/api/config":
                    return self._json(200, platform.set_config(body.get("changes") or {}))
                if p == "/api/config/reset":
                    from .params import BY_ID as PB
                    ids = [i for i in body.get("ids") or [] if i in PB]
                    return self._json(200, platform.set_config({i: PB[i].default for i in ids}))
                if p == "/api/checklist":
                    return self._json(200, platform.tick(str(body.get("item")), bool(body.get("done"))))
                if p == "/api/ask":
                    q = str(body.get("question", "")).strip()[:2000] or "What should I do next?"
                    return self._json(200, platform.ask(q))
            except PermissionError as e:
                return self._json(409, {"error": str(e)})
            except (ValueError, RuntimeError, TypeError, KeyError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(404, {"error": "not found"})

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--repo", default=".", help="the WirelessJEV repo folder")
    ap.add_argument("--log-dir", default="artifacts/onecar")
    ap.add_argument("--known-hosts", default=str(Path.home() / "smartcar_known_hosts"))
    ap.add_argument("--wifi-json", default=str(Path.home() / "WiFi.json"))
    ap.add_argument("--demo", action="store_true", help="rehearsal: a synthetic site and lap in a temporary folder; car commands off")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    if args.demo:
        import tempfile
        from . import demo
        tmp = Path(tempfile.mkdtemp(prefix="wj-platform-demo-"))
        paths = Paths(repo, tmp / "onecar", tmp / "known_hosts (demo)", tmp / "WiFi.json (demo)")
        paths.log_dir.mkdir(parents=True)
        paths.known_hosts.write_text("# demo placeholder, not a real host key\n")
        paths.wifi_json.write_text("{}\n")
        demo.write_site(paths.log_dir)
        demo.demo_lap(paths.log_dir, seconds=300, realtime=False, name="survey1")     # one finished lap for "All laps"
        for i in (1, 2):                                                              # route laps for the 10-robot replay
            demo.demo_lap(paths.log_dir, seconds=150, realtime=False, name=f"r0{i}")
        demo.write_scan(paths.log_dir)                                                # a scan for the WirelessJEV proposal
        threading.Thread(target=demo.demo_lap, args=(paths.log_dir,), daemon=True).start()
    else:
        paths = Paths(repo, (repo / args.log_dir).resolve(), Path(args.known_hosts).expanduser(), Path(args.wifi_json).expanduser())
    platform = Platform(paths, demo=args.demo)
    platform.audit.write("platform_start", demo=args.demo, log_dir=str(paths.log_dir),
                         ssh_ready=bool(os.environ.get("WJ_SSH_PASSWORD")))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(platform, args.port))
    print(f"WirelessJEV Field Platform on http://127.0.0.1:{args.port}   logs: {paths.log_dir}\n"
          f"SSH password {'is set' if os.environ.get('WJ_SSH_PASSWORD') else 'is NOT set (car commands will refuse)'}; "
          f"assistant: {agent_mod.status(platform.cfg())['reason']}\nPress Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        platform.runner.stop()
        server.server_close()
    return 0
