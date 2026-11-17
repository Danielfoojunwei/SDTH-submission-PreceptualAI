"""Router setup for consumer routers configured in their own app.

The platform cannot configure such routers, so it gives a checklist of what to set in each app
(the runbook's AP setup sheet), then checks the result from what the car actually hears:
the newest scan compared with the site sheet."""
from __future__ import annotations

import json
from pathlib import Path

from .mapview import band_of, channel_of

CHECKLIST = [
    ("ssid", "Same network name (SSID) and the same password on every router or AP."),
    ("security", "Security WPA2-Personal or WPA2/WPA3-Personal (PSK) on every one."),
    ("mode", "Extra routers in Access Point (or Bridge) mode, so one router hands out addresses and the "
             "car keeps its IP when it moves between them."),
    ("subnet", "That network is not 192.168.1.x (the car's management hotspot uses it). Change the main "
               "router's address range in its app if needed."),
    ("channels", "Fixed channels, different on each AP, not Auto. 5 GHz example: 36, 52, 100. Turn off "
                 "'smart connect' or band steering if it merges the bands under one name you cannot pin."),
    ("ft", "Fast roaming (802.11r, sometimes called 'fast transition' or 'seamless roaming') on, if the app offers it."),
    ("power", "Transmit power fixed (not Auto), and noted on the site sheet."),
    ("placement", "Placed so two APs reach most of the route; positions measured and on the site sheet."),
    ("wifi_json", "The test network's name and password added to ~/WiFi.json on this laptop (never in the repo)."),
]


def newest_scan(log_dir: Path) -> dict | None:
    """The newest scan the car made: a scan step's snapshot, or a lap's scan-cache snapshot."""
    found = []
    for p in log_dir.glob("*-scan.jsonl"):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "snapshot":
                found.append((e.get("t", 0), "scan step", [{"bssid": (a.get("bssid") or "").lower(), "ssid": a.get("ssid"),
                                                             "freq_mhz": a.get("frequency"), "signal_dbm": a.get("signal")}
                                                            for a in e.get("aps", [])]))
    for p in log_dir.glob("*-lap-*.jsonl"):
        last = None
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"scan"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "scan":
                last = e
        if last:
            found.append((last.get("t_car", 0), f"lap {p.name.split('-lap-')[1][:-6]}",
                          [{"bssid": (a.get("bssid") or "").lower(), "ssid": a.get("ssid"), "freq_mhz": a.get("freq_mhz"),
                            "signal_dbm": a.get("signal_dbm")} for a in last.get("aps", [])]))
    if not found:
        return None
    t, source, aps = max(found, key=lambda f: f[0])
    return {"t": t, "source": source, "aps": aps}


def check(site: dict | None, scan: dict | None) -> list[dict]:
    rows = []

    def row(status, item, seen, fix=""):
        rows.append({"status": status, "item": item, "seen": seen, "fix": fix})

    if not site:
        row("FAIL", "Site sheet", "not saved", "Fill the AP rows on the Setup tab and save.")
        return rows
    aps = [a | {"bssid": a["bssid"].lower()} for a in site.get("aps", [])]
    ssids = {a.get("ssid") for a in aps if a.get("ssid")}
    if len(ssids) > 1:
        row("WARN", "One network name", f"site sheet has {', '.join(sorted(ssids))}",
            "All APs should share one SSID so the car can roam between them.")
    by_chan = {}
    for a in aps:
        by_chan.setdefault((band_of(a.get("frequency_mhz")), channel_of(a.get("frequency_mhz"))), []).append(a["bssid"])
    for (band, ch), b in by_chan.items():
        if len(b) > 1:
            row("WARN", f"Channel {ch} ({band})", f"used by {len(b)} APs on the sheet",
                "Give each AP its own channel in its app, then update the sheet.")
    two4 = sorted(ch for (band, ch) in by_chan if band == "2.4 GHz" and ch)
    if any(b - a < 5 for a, b in zip(two4, two4[1:])):
        row("WARN", "2.4 GHz overlap", f"channels {two4}", "On 2.4 GHz use only 1, 6 and 11.")
    if scan is None:
        row("WARN", "Car's view of the APs", "no scan yet",
            "Run 'Scan the APs' (or any lap) once the car is on the test network; this check then compares it with the sheet.")
        return rows
    heard = {a["bssid"]: a for a in scan["aps"] if a["bssid"]}
    for a in aps:
        h = heard.get(a["bssid"])
        if not h:
            row("FAIL", f"AP {a['bssid']}", f"not heard in the newest scan ({scan['source']})",
                "Check it is powered, on the sheet's channel and broadcasting the SSID; or fix the BSSID on the sheet "
                "(each band of a router has its own BSSID).")
            continue
        if h.get("freq_mhz") and a.get("frequency_mhz") and abs(float(h["freq_mhz"]) - float(a["frequency_mhz"])) > 1:
            row("FAIL", f"AP {a['bssid']} channel", f"heard on {h['freq_mhz']:.0f} MHz (channel {channel_of(h['freq_mhz'])}), "
                f"sheet says {a['frequency_mhz']} MHz", "Fix the channel in the router's app (not Auto), or correct the sheet.")
        elif h.get("ssid") and a.get("ssid") and h["ssid"] != a["ssid"]:
            row("FAIL", f"AP {a['bssid']} name", f"broadcasts {h['ssid']!r}, sheet says {a['ssid']!r}",
                "Use the same SSID on every AP.")
        else:
            sig = h.get("signal_dbm")
            row("PASS" if sig is None or sig >= -80 else "WARN", f"AP {a['bssid']}",
                f"heard at {sig} dBm on channel {channel_of(h.get('freq_mhz') or a.get('frequency_mhz'))}",
                "" if sig is None or sig >= -80 else "Very weak where the car stood; check its position and power.")
    sheet = {a["bssid"] for a in aps}
    test_ssids = ssids or set()
    for b, h in heard.items():
        if b not in sheet and h.get("ssid") in test_ssids:
            row("WARN", f"Unknown AP {b}", f"broadcasts {h['ssid']!r} but is not on the sheet",
                "Add it to the site sheet (probably another band of one of your routers), or turn that band off.")
    return rows
