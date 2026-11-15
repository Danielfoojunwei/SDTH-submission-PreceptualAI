"""Driving telemetry for one real SmartCar: a read-only stream of link state,
gateway pings and scan-cache snapshots, with operator waypoint marks.

Codex's deployment bundle captures position-labelled scans from a *parked*
car. This module records what a moving car experiences, which is what the
outage-risk model and the trace-driven twin need:

- every `period` seconds: `iw dev wlan1 link` and `iw dev wlan1 station dump`
  (signal, bitrates, retries, failures, beacon loss);
- every `scan_every` samples: the existing scan cache (no new scan is
  triggered, so client airtime is not interrupted);
- a 20 Hz gateway ping on its own channel;
- operator marks: the operator types a waypoint label as the car passes a
  taped floor marker; positions are interpolated between marks afterwards.

Everything sent to the car is one of two fixed command templates with
validated numeric arguments. Nothing here changes radio or network state, and
wlan0 (the management hotspot) is never addressed.
"""
from __future__ import annotations

import ipaddress
import json
import re
import threading
import time

PERIODS = (0.1, 0.2, 0.5, 1.0)
MAX_SECONDS = 900


def stream_command(period: float, scan_every: int, seconds: int) -> str:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    if not 5 <= int(scan_every) <= 600 or not 5 <= int(seconds) <= MAX_SECONDS:
        raise ValueError("scan_every must be 5-600 samples and seconds 5-900")
    n = int(round(int(seconds) / period))
    return ("i=0; while [ $i -lt %d ]; do echo \"@T $(date +%%s.%%N)\"; iw dev wlan1 link; "
            "iw dev wlan1 station dump; if [ $((i %% %d)) -eq 0 ]; then echo '@SCAN'; "
            "sudo -n iw dev wlan1 scan dump; echo '@ENDSCAN'; fi; i=$((i+1)); sleep %s; done"
            % (n, int(scan_every), period))


def ping_command(gateway: str, seconds: int) -> str:
    ip = ipaddress.IPv4Address(gateway)
    if not 5 <= int(seconds) <= MAX_SECONDS:
        raise ValueError("ping window must be 5-900 s")
    return f"sudo -n ping -I wlan1 -D -n -i 0.05 -w {int(seconds)} {ip}"


_NUM = {"signal_dbm": r"signal:\s+(-?\d+)", "signal_avg_dbm": r"signal avg:\s+(-?\d+)",
        "tx_bitrate_mbps": r"tx bitrate:\s+([\d.]+)", "rx_bitrate_mbps": r"rx bitrate:\s+([\d.]+)",
        "tx_packets": r"tx packets:\s+(\d+)", "tx_retries": r"tx retries:\s+(\d+)",
        "tx_failed": r"tx failed:\s+(\d+)", "beacon_loss": r"beacon loss:\s+(\d+)",
        "freq_mhz": r"freq:\s+([\d.]+)"}


def parse_sample(block: str) -> dict:
    """One `@T` block (link + station dump) -> a flat record."""
    out = {}
    m = re.search(r"Connected to ((?:[0-9a-f]{2}:){5}[0-9a-f]{2})", block)
    out["connected"] = m is not None
    out["bssid"] = m.group(1) if m else None
    s = re.search(r"SSID: (.*)", block)
    out["ssid"] = s.group(1).strip() if s else None
    for key, pat in _NUM.items():
        m = re.search(pat, block)
        out[key] = float(m.group(1)) if m else None
    return out


def parse_scan(text: str) -> list[dict]:
    aps = []
    for block in re.split(r"\nBSS ", "\n" + text)[1:]:
        if not re.match(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", block):
            continue
        freq = re.search(r"freq: (\d+(?:\.\d+)?)", block)
        sig = re.search(r"signal: (-?\d+(?:\.\d+)?) dBm", block)
        age = re.search(r"last seen: (\d+) ms ago", block)
        if freq and sig:
            aps.append({"bssid": block[:17], "freq_mhz": float(freq.group(1)),
                        "signal_dbm": float(sig.group(1)),
                        "age_ms": int(age.group(1)) if age else None})
    return aps


class StreamParser:
    """Incremental parser for the telemetry stream; tolerant of split reads."""

    def __init__(self):
        self.buf = ""
        self.t = None
        self.lines: list[str] = []
        self.scan: list[str] | None = None

    def feed(self, data: str) -> list[dict]:
        self.buf += data
        *lines, self.buf = self.buf.split("\n")
        events = []
        for line in lines:
            if line.startswith("@T "):
                events += self._flush()
                try:
                    self.t = float(line[3:])
                except ValueError:
                    self.t = None
            elif line == "@SCAN":
                self.scan = []
            elif line == "@ENDSCAN":
                if self.scan is not None and self.t is not None:
                    events.append({"event": "scan", "t_car": self.t,
                                   "aps": parse_scan("\n".join(self.scan))})
                self.scan = None
            elif self.scan is not None:
                self.scan.append(line)
            else:
                self.lines.append(line)
        return events

    def _flush(self) -> list[dict]:
        if self.t is None or not self.lines:
            self.lines = []
            return []
        rec = {"event": "sample", "t_car": self.t, **parse_sample("\n".join(self.lines))}
        self.lines = []
        return [rec]

    def close(self) -> list[dict]:
        return self._flush()


def parse_ping_lines(text: str) -> list[dict]:
    return [{"event": "ping", "t_car": float(m.group(1)), "rtt_ms": float(m.group(2))}
            for m in re.finditer(r"\[(\d{9,11}\.\d+)\] \d+ bytes from \S+ .*?time=([\d.]+) ms", text)]


def record_lap(link, write, *, gateway: str | None, seconds: int, period: float = 0.1,
               scan_every: int = 30, marks=None, clock=time.time, sleep=time.sleep, raw=None) -> dict:
    """Stream telemetry for `seconds`; `write(event_dict)` persists each event.

    `raw(channel, text)`, if given, receives everything the car printed
    ("stream", "ping", "stream_err", "ping_err") before parsing. Keeping it
    means a lap is never lost to a parser that does not match the car's real
    output: it can be parsed again later (`reparse_lap`).

    `marks` is an iterator of waypoint labels typed by the operator (blocking,
    read on a helper thread); each is stamped with the Mac clock. Returns
    counts. Stops early if the stream ends."""
    _, stdout, _ = link.client.exec_command(stream_command(period, scan_every, seconds))
    channel = stdout.channel
    pchan = None
    if gateway:
        _, pout, _ = link.client.exec_command(ping_command(gateway, seconds))
        pchan = pout.channel
    counts = {"sample": 0, "scan": 0, "ping": 0, "mark": 0}
    lock = threading.Lock()

    def emit(ev):
        with lock:
            counts[ev["event"]] += 1
            write(ev)

    stop = threading.Event()
    if marks is not None:
        def reader():
            for label in marks:
                if stop.is_set():
                    break
                label = str(label).strip()
                if re.fullmatch(r"[A-Za-z0-9_-]{1,16}", label):
                    emit({"event": "mark", "t_mac": clock(), "label": label})
        threading.Thread(target=reader, daemon=True).start()

    parser, ping_buf = StreamParser(), ""
    end = clock() + seconds + 3
    errors = {"stream_err": "", "ping_err": ""}

    def drain_errors(chan, key):
        # Error text (for example "sudo: a password is required") comes on a separate stream.
        ready = getattr(chan, "recv_stderr_ready", None)
        if ready is None or not ready():
            return
        text = chan.recv_stderr(65536).decode("utf-8", errors="replace")
        errors[key] = (errors[key] + text)[-2000:]
        if raw is not None:
            raw(key, text)
    try:
        while clock() < end:
            idle = True
            if channel.recv_ready():
                idle = False
                text = channel.recv(65536).decode("utf-8", errors="replace")
                if raw is not None:
                    raw("stream", text)
                for ev in parser.feed(text):
                    emit(ev)
            drain_errors(channel, "stream_err")
            if pchan is not None and pchan.recv_ready():
                idle = False
                text = pchan.recv(65536).decode("utf-8", errors="replace")
                if raw is not None:
                    raw("ping", text)
                ping_buf += text
                *done, ping_buf = ping_buf.split("\n")
                for ev in parse_ping_lines("\n".join(done) + "\n"):
                    emit(ev)
            if pchan is not None:
                drain_errors(pchan, "ping_err")
            if idle:
                if channel.exit_status_ready() and not channel.recv_ready():
                    break
                sleep(0.02)
        for ev in parser.close():
            emit(ev)
    finally:
        stop.set()
        channel.close()
        if pchan is not None:
            pchan.close()
    counts.update({k: v.strip() for k, v in errors.items() if v.strip()})
    return counts


def raw_paths(lap_path):
    """Where a lap's unparsed car output is kept, next to its .jsonl."""
    from pathlib import Path
    lap_path = Path(lap_path)
    return {k: lap_path.with_name(lap_path.stem + f".{k}.raw") for k in ("stream", "ping")}


def reparse_lap(lap_path) -> dict:
    """Rebuild a lap's samples, scans and pings from its raw files with the
    current parsers, keeping its start, marks and end. The original .jsonl is
    kept as .jsonl.orig. Use after fixing a parser that did not match the car."""
    from pathlib import Path
    lap_path = Path(lap_path)
    raws = raw_paths(lap_path)
    if not raws["stream"].exists():
        raise FileNotFoundError(f"no raw stream kept for {lap_path.name}")
    old = [json.loads(line) for line in lap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [e for e in old if e.get("event") in ("lap_start", "mark")]
    end = [e for e in old if e.get("event") == "lap_end"]
    parser = StreamParser()
    events = parser.feed(raws["stream"].read_text(encoding="utf-8", errors="replace")) + parser.close()
    if raws["ping"].exists():
        events += parse_ping_lines(raws["ping"].read_text(encoding="utf-8", errors="replace"))
    counts = {k: sum(e["event"] == k for e in events) for k in ("sample", "scan", "ping")}
    backup = lap_path.with_name(lap_path.name + ".orig")
    if not backup.exists():
        backup.write_text(lap_path.read_text(encoding="utf-8"), encoding="utf-8")
    out = keep[:1] + events + keep[1:] + [{**(end[0] if end else {"event": "lap_end"}), **counts, "reparsed": True}]
    lap_path.write_text("".join(json.dumps(e) + "\n" for e in out), encoding="utf-8")
    return counts


def check_lap(lap: dict, seconds: float, period: float) -> list[dict]:
    """Does a short lap from the real car contain what the analysis needs?
    One row per item: status PASS / WARN / FAIL, what was seen, what to do."""
    rows = []

    def add(item, status, seen, fix=""):
        rows.append({"item": item, "status": status, "seen": seen, "fix": fix})

    meta, samples, scans, pings = lap["meta"], lap["samples"], lap["scans"], lap["pings"]
    add("wlan1 default route", "PASS" if meta.get("gateway") else "FAIL", meta.get("gateway") or "none",
        "" if meta.get("gateway") else "wlan1 has no gateway, so no pings and no outage data: check DHCP on the test APs")
    rate = len(samples) / seconds if seconds else 0.0
    add("link samples", "FAIL" if not samples else "WARN" if rate < 0.5 / period else "PASS", f"{len(samples)} ({rate:.1f} per second)",
        "" if samples and rate >= 0.5 / period else
        ("nothing parsed: read the .stream.raw file, fix StreamParser, then reparse" if not samples else
         "slower than asked: fine for positions, coarser for outages; note the real rate"))
    conn = [s for s in samples if s.get("connected")]
    add("associated on wlan1", "PASS" if conn else "FAIL", f"{len(conn)} of {len(samples)} samples",
        "" if conn else "the car is not on the test network, or `iw dev wlan1 link` prints a different format")
    for key, need in (("bssid", "FAIL"), ("signal_dbm", "FAIL"), ("freq_mhz", "WARN"), ("tx_bitrate_mbps", "WARN"),
                      ("tx_retries", "WARN"), ("tx_failed", "WARN")):
        got = sum(s.get(key) is not None for s in conn)
        ok = bool(conn) and got >= 0.9 * len(conn)
        add(f"field {key}", "PASS" if ok else need, f"{got} of {len(conn)}",
            "" if ok else "format differs from the parser: compare .stream.raw with _NUM in hil/telemetry.py")
    aps = max((len(s["aps"]) for s in scans), default=0)
    add("scan snapshots", "PASS" if aps else "FAIL", f"{len(scans)} snapshots, up to {aps} APs",
        "" if aps else "`sudo -n iw dev wlan1 scan dump` returned nothing: sudo rights, or the format differs")
    want = 20.0 * seconds
    add("gateway pings", "PASS" if len(pings) >= 0.5 * want else "WARN" if pings else "FAIL", f"{len(pings)} (expected about {want:.0f})",
        "" if len(pings) >= 0.5 * want else
        ("no replies parsed: sudo rights for ping, `ping -D` format, or no route" if not pings else
         "fewer replies than sent: a lossy link, or ping is rate-limited without sudo"))
    for key in ("stream_err", "ping_err"):
        if meta.get(key) or lap.get("end", {}).get(key):
            add(f"car error output ({key})", "WARN", (meta.get(key) or lap["end"][key])[:200], "read it: it usually names the cause")
    return rows


# --------------------------------------------------------------------------
# Offline: positions from waypoint marks.
def load_lap(path) -> dict:
    events = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    head = next((e for e in events if e.get("event") == "lap_start"), {})
    return {"meta": head, "end": next((e for e in events if e.get("event") == "lap_end"), {}),
            "samples": [e for e in events if e.get("event") == "sample"],
            "scans": [e for e in events if e.get("event") == "scan"],
            "pings": [e for e in events if e.get("event") == "ping"],
            "marks": [e for e in events if e.get("event") == "mark"]}


def waypoint_of(label, waypoints: dict):
    """The waypoints.json key a typed label refers to: an exact match, else the
    one key that matches ignoring case (an operator typing "a" for "A")."""
    if label in waypoints:
        return label
    hits = [k for k in waypoints if k.lower() == str(label).lower()]
    return hits[0] if len(hits) == 1 else None


def positions(times_car, marks, waypoints: dict, offset_s: float):
    """Linear interpolation of (x, y) between consecutive waypoint marks.

    `offset_s` is car clock minus Mac clock. Times outside the marked span
    return NaN: no position is invented before the first or after the last mark."""
    import numpy as np
    pts = [(m["t_mac"] + offset_s, waypoints[key]) for m in marks
           if (key := waypoint_of(m["label"], waypoints)) is not None]
    pts.sort(key=lambda p: p[0])
    t = np.asarray(times_car, dtype=float)
    out = np.full((len(t), 2), np.nan)
    if len(pts) < 2:
        return out
    tm = np.array([p[0] for p in pts])
    xy = np.array([p[1][:2] for p in pts], dtype=float)
    inside = (t >= tm[0]) & (t <= tm[-1])
    out[inside, 0] = np.interp(t[inside], tm, xy[:, 0])
    out[inside, 1] = np.interp(t[inside], tm, xy[:, 1])
    return out
