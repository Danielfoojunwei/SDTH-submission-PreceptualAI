"""Read-only live lap viewer: a browser page that shows the lap being recorded.

    .venv/bin/python -m wirelessjev.hil.viewer            # watches artifacts/onecar
    .venv/bin/python -m wirelessjev.hil.viewer --demo     # a synthetic lap, for rehearsal

Then open http://127.0.0.1:8765 in a browser on this laptop.

It only reads the `*-lap-*.jsonl` files that `onecar lap` and `onecar check`
write. It never connects to the car, never sends a command and never writes a
file in the log folder. It listens on 127.0.0.1 only, and the page loads
nothing from the internet, so it works on the car's hotspot.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse

from .telemetry import check_lap

WINDOW_S = 60.0                 # how much history the charts show
GAP_S = 0.2                     # a ping-reply gap longer than this is an outage (the fleet's stop rule)
LAP_GLOB = "*-lap-*.jsonl"
NAME_RE = re.compile(r"[0-9]{8}-[0-9]{6}-lap-[A-Za-z0-9_-]{1,32}\.jsonl")


def read_events(path: Path) -> list[dict]:
    """All complete events in a lap file. The last line may be half-written; it is skipped."""
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def list_laps(log_dir: Path) -> list[dict]:
    files = [p for p in log_dir.glob(LAP_GLOB) if NAME_RE.fullmatch(p.name)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"file": p.name, "name": p.name.split("-lap-", 1)[1][:-6], "modified": p.stat().st_mtime} for p in files]


def _round(v, n=2):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), n)


def summarise(events: list[dict], now_mac: float | None = None) -> dict:
    """Everything the page shows, from the events of one lap."""
    head = next((e for e in events if e.get("event") == "lap_start"), {})
    end = next((e for e in events if e.get("event") == "lap_end"), None)
    samples = [e for e in events if e.get("event") == "sample"]
    pings = sorted((e for e in events if e.get("event") == "ping"), key=lambda e: e["t_car"])
    scans = [e for e in events if e.get("event") == "scan"]
    marks = [e for e in events if e.get("event") == "mark"]
    offset = float(head.get("car_minus_mac_s") or 0.0)
    times = [e["t_car"] for e in samples] + [e["t_car"] for e in pings]
    t_last = max(times) if times else None
    t0 = (head.get("t_mac", 0.0) + offset) if head else (min(times) if times else 0.0)
    now_mac = time.time() if now_mac is None else now_mac
    lo = (t_last - WINDOW_S) if t_last is not None else 0.0

    connected = [s for s in samples if s.get("connected")]
    last = samples[-1] if samples else {}
    span = (samples[-1]["t_car"] - samples[0]["t_car"]) if len(samples) > 1 else 0.0
    roams = [b for a, b in zip(connected, connected[1:]) if a.get("bssid") and b.get("bssid") and a["bssid"] != b["bssid"]]

    gaps_all = [(a["t_car"], b["t_car"] - a["t_car"]) for a, b in zip(pings, pings[1:])]
    outages = [g for g in gaps_all if g[1] > GAP_S]
    # A ping silence that is still going on (the lap is live and no reply for a while) counts too.
    live_gap = None
    if pings and end is None:
        silent = (now_mac + offset) - pings[-1]["t_car"]
        if silent > GAP_S:
            live_gap = silent

    elapsed = (t_last - t0) if t_last is not None else 0.0
    lap = {"name": head.get("name"), "note": head.get("note") or "", "gateway": head.get("gateway"),
           "planned_s": head.get("seconds"), "period_s": head.get("period_s"), "finished": end is not None,
           "elapsed_s": _round(elapsed, 1),
           "last_data_age_s": _round((now_mac + offset) - t_last, 1) if t_last is not None else None}
    checks = check_lap({"meta": head, "end": end or {}, "samples": samples, "scans": scans, "pings": pings},
                       max(elapsed, 1.0), float(head.get("period_s") or 0.1)) if samples or pings else []
    return {
        "lap": lap,
        "now": {"connected": bool(last.get("connected")), "bssid": last.get("bssid"), "ssid": last.get("ssid"),
                "signal_dbm": last.get("signal_dbm"), "freq_mhz": last.get("freq_mhz"),
                "tx_bitrate_mbps": last.get("tx_bitrate_mbps"),
                "rtt_ms": pings[-1]["rtt_ms"] if pings else None},
        "counts": {"samples": len(samples), "pings": len(pings), "scans": len(scans), "marks": len(marks), "roams": len(roams)},
        "rate_hz": _round((len(samples) - 1) / span, 1) if span > 0 else None,
        "window_s": WINDOW_S,
        "signal": [[_round(s["t_car"] - lo), s.get("signal_dbm"), s.get("bssid")] for s in samples if s["t_car"] >= lo],
        "rtt": [[_round(p["t_car"] - lo), p["rtt_ms"]] for p in pings if p["t_car"] >= lo],
        "gaps": [[_round(t - lo), _round(d)] for t, d in outages if t + d >= lo],
        "live_gap_s": _round(live_gap),
        "outages": {"count": len(outages) + (1 if live_gap else 0), "longest_s": _round(max([d for _, d in outages] + [live_gap or 0.0])),
                    "total_s": _round(sum(d for _, d in outages))},
        "marks": [{"label": m["label"], "t_s": _round(m["t_mac"] + offset - t0, 1),
                   "in_window_s": _round(m["t_mac"] + offset - lo) if m["t_mac"] + offset >= lo else None} for m in marks],
        "scan": sorted(scans[-1]["aps"], key=lambda a: -a["signal_dbm"])[:12] if scans else [],
        "scan_age_s": _round(t_last - scans[-1]["t_car"], 1) if scans and t_last is not None else None,
        "checks": checks,
    }


def route(path: str, log_dir: Path, now_mac: float | None = None):
    """(status, content type, body bytes) for one GET request. Only reads files."""
    url = urlparse(path)
    if url.path == "/":
        return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")
    if url.path == "/laps":
        return 200, "application/json", json.dumps(list_laps(log_dir)).encode()
    if url.path == "/data":
        laps = list_laps(log_dir)
        want = parse_qs(url.query).get("lap", [None])[0]
        if want is not None and want not in {l["file"] for l in laps}:
            return 404, "application/json", json.dumps({"error": "no such lap"}).encode()
        if not laps:
            return 200, "application/json", json.dumps({"lap": None, "laps": [], "dir": str(log_dir)}).encode()
        chosen = want or laps[0]["file"]
        out = summarise(read_events(log_dir / chosen), now_mac)
        out["file"], out["laps"], out["dir"] = chosen, laps[:30], str(log_dir)
        return 200, "application/json", json.dumps(out, default=str).encode()
    return 404, "text/plain; charset=utf-8", b"not found"


def serve(log_dir: Path, port: int = 8765, host: str = "127.0.0.1"):
    log_dir = Path(log_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status, ctype, body = route(self.path, log_dir)
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


# --------------------------------------------------------------------------
def demo_lap(log_dir: Path, seconds: float = 600.0, speed: float = 1.0, stop: threading.Event | None = None):
    """Write a synthetic lap into `log_dir` as if the car were driving: 10 Hz link samples,
    20 Hz pings with an outage every 25 s, a scan every 3 s, a waypoint mark every 12 s."""
    stop = stop or threading.Event()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / time.strftime("%Y%m%d-%H%M%S-lap-demo.jsonl")
    aps = [("aa:bb:cc:00:00:01", 5180, 2.0), ("aa:bb:cc:00:00:02", 5260, 18.0), ("aa:bb:cc:00:00:03", 5500, 10.0)]
    start = time.time()

    def rssi(x, ax):
        return 20.0 - (46.7 + 33 * math.log10(max(1.0, abs(x - ax) + 2.0)))

    with path.open("a", encoding="utf-8") as f:
        def w(ev):
            f.write(json.dumps(ev) + "\n")
            f.flush()
        w({"event": "lap_start", "name": "demo", "t_mac": start, "car_minus_mac_s": 0.0, "gateway": "10.0.0.1",
           "period_s": 0.1, "seconds": int(seconds), "note": "synthetic demo lap"})
        k, labels, serving = 0, "ABCDEFGH", aps[0]
        while not stop.is_set() and k * 0.1 < seconds:
            t = start + k * 0.1
            x = 1.0 + 18.0 * (0.5 - 0.5 * math.cos(2 * math.pi * (k * 0.1) / 90.0))
            levels = {a[0]: rssi(x, a[2]) for a in aps}
            best = max(aps, key=lambda a: levels[a[0]])
            if levels[best[0]] > levels[serving[0]] + 6:
                serving = best
            phase = (k * 0.1) % 25.0
            dip = 12.0 if phase < 2.5 else 0.0                     # a blocker: the signal drops for 2.5 s...
            silent = 0.5 <= phase < 2.0                            # ...and replies stop for 1.5 s of it
            w({"event": "sample", "t_car": t, "connected": True, "bssid": serving[0], "ssid": "fleet", "freq_mhz": serving[1],
               "signal_dbm": round(levels[serving[0]] - dip), "tx_bitrate_mbps": 234.0 if not dip else 52.0,
               "tx_retries": 3.0 * k, "tx_failed": 0.0})
            if not silent:
                for j in (0, 1):
                    w({"event": "ping", "t_car": t + 0.05 * j, "rtt_ms": round(2.5 + (8.0 if dip else 0.0), 1)})
            if k % 30 == 0:
                w({"event": "scan", "t_car": t, "aps": [{"bssid": a[0], "freq_mhz": float(a[1]), "signal_dbm": round(levels[a[0]], 1),
                                                         "age_ms": 200} for a in aps]})
            if k % 120 == 60:
                w({"event": "mark", "t_mac": t, "label": labels[(k // 120) % len(labels)]})
            k += 1
            target = start + k * 0.1 / speed
            time.sleep(max(0.0, target - time.time()))
        w({"event": "lap_end", "t_mac": time.time()})
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="artifacts/onecar", help="folder with the lap logs (read only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--demo", action="store_true", help="show a synthetic lap written to a temporary folder")
    args = ap.parse_args(argv)
    log_dir = Path(args.dir)
    stop = threading.Event()
    if args.demo:
        log_dir = Path(tempfile.mkdtemp(prefix="wj-viewer-demo-"))
        threading.Thread(target=demo_lap, args=(log_dir,), kwargs={"stop": stop}, daemon=True).start()
    server = serve(log_dir, args.port)
    print(f"Live lap viewer (read only) on http://127.0.0.1:{args.port}  watching {log_dir.resolve()}\n"
          "Press Ctrl+C to stop. It never connects to the car.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
    return 0


# --------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live lap</title>
<style>
:root{--bg:#F4F6F3;--panel:#FFFFFF;--ink:#13212C;--muted:#5A6A75;--line:#D9E0E4;--accent:#D9661F;--good:#2E7D5B;--warn:#B7791F;--bad:#C0392B;--blue:#2D6CB5;--grid:#E7ECEF}
@media (prefers-color-scheme: dark){:root{--bg:#0F1A22;--panel:#16242F;--ink:#E8EEF1;--muted:#9DB0BC;--line:#2A3B47;--accent:#F08A4B;--good:#4CBF8C;--warn:#E3B04B;--bad:#F06C5E;--blue:#6FA8E8;--grid:#21323E}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
h1{font-size:18px;margin:0;font-weight:650}
.sub{color:var(--muted);font-size:13px}
select{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink)}
main{padding:16px 20px;display:grid;gap:16px;grid-template-columns:repeat(12,1fr)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}.span8{grid-column:span 8}.span12{grid-column:span 12}
@media (max-width:900px){.span3,.span4,.span6,.span8{grid-column:span 12}}
.k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.big{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1.1;margin-top:4px}
.mid{font-size:15px;font-variant-numeric:tabular-nums;margin-top:4px;color:var(--muted);overflow-wrap:anywhere}
.ok{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
canvas{width:100%;height:190px;display:block}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}
.pill{display:inline-block;min-width:44px;text-align:center;padding:1px 6px;border-radius:999px;font-size:12px;font-weight:700;color:#fff}
.PASS{background:var(--good)}.WARN{background:var(--warn)}.FAIL{background:var(--bad)}
.banner{grid-column:span 12;padding:10px 14px;border-radius:8px;font-weight:600;display:none}
.banner.show{display:block}.banner.bad{background:var(--bad);color:#fff}.banner.warn{background:var(--warn);color:#13212C}
.marks{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}
.mark{border:1px solid var(--line);border-radius:6px;padding:2px 8px;font-variant-numeric:tabular-nums}
.empty{color:var(--muted);padding:40px;text-align:center}
</style>
</head>
<body>
<header>
  <div><h1>Live lap <span id="lapname"></span></h1><div class="sub" id="lapsub">Read only: this page never connects to the car.</div></div>
  <div><label class="sub" for="lapsel">Lap&nbsp;</label><select id="lapsel"><option value="">newest</option></select></div>
</header>
<main id="main">
  <div class="banner" id="banner"></div>
  <div class="card span3"><div class="k">Signal</div><div class="big" id="sig">–</div><div class="mid" id="ap">–</div></div>
  <div class="card span3"><div class="k">Gateway ping</div><div class="big" id="rtt">–</div><div class="mid" id="outnow">–</div></div>
  <div class="card span3"><div class="k">Outages this lap (gap &gt; 0.2 s)</div><div class="big" id="out">–</div><div class="mid" id="outlong">–</div></div>
  <div class="card span3"><div class="k">Data</div><div class="big" id="rate">–</div><div class="mid" id="counts">–</div></div>
  <div class="card span8"><div class="k">Signal, last 60 s (dBm; dashed: −67 dBm; colour changes at each roam)</div><canvas id="csig"></canvas></div>
  <div class="card span4"><div class="k">Waypoint marks</div><div class="big" id="lastmark">–</div><div class="marks" id="marks"></div></div>
  <div class="card span8"><div class="k">Ping round trip, last 60 s (ms; red: no reply for more than 0.2 s)</div><canvas id="crtt"></canvas></div>
  <div class="card span4"><div class="k">Last scan <span id="scanage"></span></div><table id="scan"></table></div>
  <div class="card span12"><div class="k">Format check, live (same items as <code>onecar check</code>)</div><table id="checks"></table></div>
</main>
<script>
const $ = id => document.getElementById(id);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let chosen = "";
$("lapsel").addEventListener("change", e => { chosen = e.target.value; tick(); });
function esc(s){return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function setup(cv){const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1; cv.width = r.width*d; cv.height = r.height*d;
  const g = cv.getContext("2d"); g.setTransform(d,0,0,d,0,0); return [g, r.width, r.height];}
function axes(g, w, h, lo, hi, win, unit, ticks){
  const L = 44, B = 18, T = 6, R = 8, X = t => L + (w-L-R)*t/win, Y = v => T + (h-T-B)*(1-(v-lo)/(hi-lo));
  g.font = "11px sans-serif"; g.fillStyle = css("--muted"); g.strokeStyle = css("--grid"); g.lineWidth = 1;
  for (const v of ticks){ g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w-R, Y(v)); g.stroke(); g.fillText(v + unit, 2, Y(v)+4); }
  for (let s = 0; s <= win; s += 10){ g.fillText((s-win) + " s", X(s)-12, h-4); }
  return [X, Y, L, R, T, B];
}
const palette = () => [css("--blue"), css("--accent"), css("--good"), css("--warn")];
function drawSignal(d){
  const [g, w, h] = setup($("csig")); g.clearRect(0,0,w,h);
  const [X, Y] = axes(g, w, h, -90, -30, d.window_s, "", [-90,-80,-70,-60,-50,-40,-30]);
  g.setLineDash([5,4]); g.strokeStyle = css("--muted"); g.beginPath(); g.moveTo(X(0), Y(-67)); g.lineTo(X(d.window_s), Y(-67)); g.stroke(); g.setLineDash([]);
  const cols = palette(); let ci = 0, prev = null, bss = null; g.lineWidth = 2;
  for (const [t, v, b] of d.signal){ if (v === null){ prev = null; continue; }
    const y = Y(Math.max(-90, Math.min(-30, v)));
    if (b !== bss){ if (bss !== null) ci++; bss = b; prev = null; }
    g.strokeStyle = cols[ci % cols.length]; if (prev){ g.beginPath(); g.moveTo(prev[0], prev[1]); g.lineTo(X(t), y); g.stroke(); }
    prev = [X(t), y]; }
  g.fillStyle = css("--ink"); for (const m of d.marks){ if (m.in_window_s === null) continue; const x = X(m.in_window_s);
    g.fillRect(x-0.5, 6, 1, h-24); g.fillText(m.label, x+3, 16); }
}
function drawRtt(d){
  const [g, w, h] = setup($("crtt")); g.clearRect(0,0,w,h);
  const top = Math.max(20, ...d.rtt.map(p => p[1])) * 1.15, step = top > 100 ? 50 : top > 40 ? 20 : 5, ticks = [];
  for (let v = 0; v <= top; v += step) ticks.push(v);
  const [X, Y] = axes(g, w, h, 0, top, d.window_s, "", ticks);
  g.fillStyle = css("--bad"); g.globalAlpha = .28;
  for (const [t, dur] of d.gaps){ g.fillRect(X(Math.max(0,t)), 6, Math.max(2, X(Math.min(d.window_s, t+dur)) - X(Math.max(0,t))), h-24); }
  if (d.live_gap_s){ const x0 = X(Math.max(0, d.window_s - d.live_gap_s)); g.fillRect(x0, 6, Math.max(2, X(d.window_s) - x0), h-24); }
  g.globalAlpha = 1; g.fillStyle = css("--blue");
  for (const [t, v] of d.rtt){ g.fillRect(X(t)-1, Y(v)-1, 2.5, 2.5); }
}
function render(d){
  if (!d.lap){ const b = $("banner"); b.className = "banner show warn";
    b.textContent = "No lap files yet in " + d.dir + ". Start onecar lap or onecar check and this page will follow it."; return; }
  const sel = $("lapsel"), keep = sel.value; sel.innerHTML = '<option value="">newest</option>' + d.laps.map(l => '<option value="' + esc(l.file) + '">' + esc(l.name) + ' · ' + new Date(l.modified*1000).toLocaleTimeString() + '</option>').join(""); sel.value = keep;
  const L = d.lap, n = d.now;
  $("lapname").textContent = "· " + (L.name || "?") + (L.finished ? " (finished)" : "");
  $("lapsub").textContent = "Read only · " + L.elapsed_s + " s of " + (L.planned_s ?? "?") + " s" + (L.note ? " · " + L.note : "") + " · gateway " + (L.gateway || "none") + " · " + d.file;
  const s = n.signal_dbm; $("sig").textContent = s === null || s === undefined ? "–" : s + " dBm";
  $("sig").className = "big " + (s === null || s === undefined ? "" : s >= -67 ? "ok" : s >= -75 ? "warn" : "bad");
  $("ap").textContent = n.connected ? (n.bssid + " · " + (n.freq_mhz ?? "?") + " MHz · " + (n.tx_bitrate_mbps ?? "?") + " Mb/s") : "not associated";
  $("rtt").textContent = n.rtt_ms === null || n.rtt_ms === undefined ? "–" : n.rtt_ms + " ms";
  $("outnow").innerHTML = d.live_gap_s ? '<span class="bad">no reply for ' + d.live_gap_s + ' s</span>' : (d.counts.pings ? "replying" : "no pings yet");
  $("out").textContent = d.outages.count; $("out").className = "big " + (d.outages.count ? "warn" : "ok");
  $("outlong").textContent = "longest " + (d.outages.longest_s ?? 0) + " s · total " + (d.outages.total_s ?? 0) + " s · roams " + d.counts.roams;
  $("rate").textContent = (d.rate_hz ?? "–") + " /s";
  $("counts").textContent = d.counts.samples + " samples · " + d.counts.pings + " pings · " + d.counts.scans + " scans";
  const ms = d.marks; $("lastmark").textContent = ms.length ? ms[ms.length-1].label + " at " + ms[ms.length-1].t_s + " s" : "none yet";
  $("marks").innerHTML = ms.slice(-24).map(m => '<span class="mark">' + esc(m.label) + ' ' + m.t_s + ' s</span>').join("");
  $("scanage").textContent = d.scan_age_s === null ? "" : "(" + d.scan_age_s + " s ago)";
  $("scan").innerHTML = '<tr><th>AP</th><th>MHz</th><th>dBm</th></tr>' + d.scan.map(a => '<tr><td>' + esc(a.bssid) + (a.bssid === n.bssid ? " ◀" : "") + '</td><td>' + a.freq_mhz + '</td><td>' + a.signal_dbm + '</td></tr>').join("");
  $("checks").innerHTML = '<tr><th></th><th>Item</th><th>Seen</th><th>What to do</th></tr>' + d.checks.map(c => '<tr><td><span class="pill ' + c.status + '">' + c.status + '</span></td><td>' + esc(c.item) + '</td><td>' + esc(c.seen) + '</td><td>' + esc(c.fix) + '</td></tr>').join("");
  const b = $("banner"), stale = !L.finished && L.last_data_age_s !== null && L.last_data_age_s > 5;
  if (stale){ b.className = "banner show warn"; b.textContent = "No new data for " + L.last_data_age_s + " s. Is the lap still recording?"; }
  else if (d.live_gap_s && d.live_gap_s > 1){ b.className = "banner show bad"; b.textContent = "Outage now: no gateway reply for " + d.live_gap_s + " s."; }
  else { b.className = "banner"; }
  drawSignal(d); drawRtt(d);
}
let busy = false;
async function tick(){ if (busy) return; busy = true;
  try { const r = await fetch("/data" + (chosen ? "?lap=" + encodeURIComponent(chosen) : ""), {cache: "no-store"}); render(await r.json()); }
  catch (e) { const b = $("banner"); if (b){ b.className = "banner show warn"; b.textContent = "The viewer process is not answering. Is it still running in the terminal?"; } }
  finally { busy = false; } }
tick(); setInterval(tick, 500); window.addEventListener("resize", tick);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
