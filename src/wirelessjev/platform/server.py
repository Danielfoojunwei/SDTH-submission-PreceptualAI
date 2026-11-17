"""HTTP server for the Field Platform. Standard library only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlparse

from ..hil import telemetry as tel
from ..hil import viewer
from . import agent as agent_mod
from .guide import environment, last_check, next_step
from .page import PAGE
from .steps import BY_ID, STEPS, Paths, command, progress, shell_text

FILES = {"coverage.png": "image/png", "coverage.json": "application/json"}
MAX_BODY = 200_000
LABEL_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


class Platform:
    """State shared by every request."""

    def __init__(self, paths: Paths, assistant=None, demo: bool = False):
        from .jobs import Runner
        self.paths, self.demo = paths, demo
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = paths.log_dir / "platform_state.json"
        self.done = self._load_done()
        self.lock = threading.Lock()
        self.runner = Runner(cwd=str(paths.repo), on_finish=self._finished)
        self.assistant = assistant or agent_mod.Assistant()

    # ---- persistence -----------------------------------------------------
    def _load_done(self) -> dict:
        try:
            return json.loads(self.state_file.read_text()).get("done", {})
        except (OSError, json.JSONDecodeError):
            return {}

    def _finished(self, job) -> None:
        if job.returncode == 0 and not BY_ID[job.step_id].lap_prefix:
            with self.lock:
                self.done[job.step_id] = self.done.get(job.step_id, 0) + 1
                self.state_file.write_text(json.dumps({"done": self.done}, indent=1))

    # ---- views -----------------------------------------------------------
    def snapshot(self, extra: dict | None = None) -> dict:
        cur = self.runner.current()
        running = cur.summary(tail=0) if cur else None
        steps = []
        for s in STEPS:
            d = s.as_dict() | progress(s, self.paths, self.done)
            try:
                d["command"] = shell_text(command(s, self.paths, extra), self.paths.repo) if s.kind != "setup" else ""
            except ValueError:
                d["command"] = ""
            steps.append(d)
        return {"steps": steps, "running": running, "next": next_step(self.paths, self.done, running, extra),
                "environment": environment(self.paths, agent_mod.status()), "check": last_check(self.paths),
                "jobs": [j.summary(tail=0) for j in self.runner.jobs[-12:]], "site": self.site(),
                "log_dir": str(self.paths.log_dir), "demo": self.demo}

    def site(self) -> dict:
        out = {}
        for key, path in (("site", self.paths.site_json), ("waypoints", self.paths.waypoints_json)):
            try:
                out[key] = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                out[key] = None
        return out

    def assistant_state(self) -> dict:
        snap = self.snapshot()
        lap = None
        laps = viewer.list_laps(self.paths.log_dir)
        if laps:
            s = viewer.summarise(viewer.read_events(self.paths.log_dir / laps[0]["file"]))
            lap = {k: s[k] for k in ("lap", "now", "counts", "outages", "rate_hz")}
        jobs = [j.summary(tail=25) for j in self.runner.jobs[-3:]]
        return {"steps": [{k: st[k] for k in ("id", "block", "title", "kind", "done", "wanted", "complete")} for st in snap["steps"]],
                "environment": [{k: r[k] for k in ("item", "ok", "seen")} for r in snap["environment"]],
                "last_format_check": snap["check"], "latest_lap": lap, "recent_commands": jobs,
                "site_saved": snap["site"]["site"] is not None}

    # ---- actions ---------------------------------------------------------
    def run(self, step_id: str, extra: dict) -> dict:
        step = BY_ID.get(step_id)
        if step is None:
            raise ValueError("unknown step")
        if step.kind == "car_change":
            raise PermissionError("this step changes the car: run the shown command in a terminal, "
                                  "where it asks you to retype the target")
        if step.kind == "setup":
            raise ValueError("this step is done on the Setup tab")
        if self.demo and step.kind == "read":
            raise PermissionError("demo mode: commands that talk to the car are switched off")
        needs_ssh = step.kind == "read" and step.id != "preflight"
        if needs_ssh and not os.environ.get("WJ_SSH_PASSWORD"):
            raise PermissionError("WJ_SSH_PASSWORD is not set in the terminal that started the platform. "
                                  "Stop it, run: read -s WJ_SSH_PASSWORD; export WJ_SSH_PASSWORD  and start it again")
        clean = {k: str(v)[:80] for k, v in (extra or {}).items() if k in ("name", "seconds", "note", "mission")}
        if "seconds" in clean and not clean["seconds"].isdigit():
            raise ValueError("seconds must be a whole number")
        if "name" in clean and not set(clean["name"]) <= LABEL_OK:
            raise ValueError("lap names use letters, digits, _ or -")
        argv = command(step, self.paths, clean)
        if any(a.startswith("<") for a in argv):
            raise ValueError("this step needs data that does not exist yet (for example a scan log)")
        job = self.runner.start(step.id, argv, shell_text(argv, self.paths.repo))
        return job.summary(tail=0)

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
        return {"saved": [str(self.paths.site_json), str(self.paths.waypoints_json)], "aps": len(site["aps"])}


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
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, "application/json", json.dumps(obj, default=str).encode())

        def log_message(self, *args):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            p = url.path
            if p == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if p == "/live":
                return self._send(200, "text/html; charset=utf-8", viewer.PAGE.encode())
            if p in ("/data", "/laps"):                         # the live view's own endpoints
                st, ct, body = viewer.route(self.path, platform.paths.log_dir)
                return self._send(st, ct, body)
            if p == "/api/state":
                return self._json(200, platform.snapshot())
            if p.startswith("/api/job/"):
                job = platform.runner.get(int(p.rsplit("/", 1)[1])) if p.rsplit("/", 1)[1].isdigit() else None
                return self._json(200, job.summary()) if job else self._json(404, {"error": "no such job"})
            if p.startswith("/files/") and p[7:] in FILES:
                f = platform.paths.log_dir / p[7:]
                if f.exists():
                    return self._send(200, FILES[p[7:]], f.read_bytes())
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
                if p == "/api/mark":
                    label = str(body.get("label", ""))
                    if not (0 < len(label) <= 16 and set(label) <= LABEL_OK):
                        raise ValueError("bad waypoint label")
                    return self._json(200, platform.runner.mark(label))
                if p == "/api/stop":
                    return self._json(200, {"stopped": platform.runner.stop()})
                if p == "/api/site":
                    return self._json(200, platform.save_site(body))
                if p == "/api/ask":
                    q = str(body.get("question", "")).strip()[:2000] or "What should I do next?"
                    snap = platform.snapshot()
                    return self._json(200, platform.assistant.ask(q, platform.assistant_state(), snap["next"]))
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
    ap.add_argument("--demo", action="store_true", help="rehearsal: a synthetic lap in a temporary folder; car commands off")
    args = ap.parse_args(argv)
    repo = Path(args.repo).resolve()
    if args.demo:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="wj-platform-demo-"))
        paths = Paths(repo, tmp / "onecar", tmp / "known_hosts (demo)", tmp / "WiFi.json (demo)")
        paths.log_dir.mkdir(parents=True)
        paths.known_hosts.write_text("# demo placeholder, not a real host key\n")
        paths.wifi_json.write_text("{}\n")
        threading.Thread(target=viewer.demo_lap, args=(paths.log_dir,), daemon=True).start()
    else:
        paths = Paths(repo, (repo / args.log_dir).resolve(), Path(args.known_hosts).expanduser(), Path(args.wifi_json).expanduser())
    platform = Platform(paths, demo=args.demo)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(platform, args.port))
    print(f"WirelessJEV Field Platform on http://127.0.0.1:{args.port}   logs: {paths.log_dir}\n"
          f"SSH password {'is set' if os.environ.get('WJ_SSH_PASSWORD') else 'is NOT set (car commands will refuse)'}; "
          f"assistant: {agent_mod.status()['reason']}\nPress Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        platform.runner.stop()
        server.server_close()
    return 0
