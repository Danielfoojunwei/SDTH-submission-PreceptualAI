"""How to use the platform, as short topics. The built-in guide answers "how do I ..." questions
from these with no network, and they are the manual a local model is given."""
from __future__ import annotations

import re

TOPICS = [
    ("Start of the day", "start begin first morning order plan day steps",
     "Follow the Plan tab top to bottom; the Next step box always names the one thing to do now. Setup first (car files, "
     "site sheet, router checklist), then preflight and probes, join the car to the test network, the format check, and "
     "only then laps."),
    ("Site sheet", "site sheet ap aps bssid position waypoint waypoints save coordinates metres",
     "Setup tab: one row per BSSID (each band of each router has its own), with frequency, x/y/z in metres from one corner "
     "and transmit power. Add the taped waypoints A, B, C... with their x/y. Save checks everything before writing site.json "
     "and waypoints.json."),
    ("Router setup", "router routers app consumer tp-link asus channel channels ssid mesh 802.11r roaming bridge access point mode",
     "Consumer routers are set up in their own app; the platform cannot change them. Work through the Router checklist on "
     "the Setup tab (same SSID and password, AP or bridge mode, not 192.168.1.x, fixed different channels, fast roaming on "
     "if offered). Then run Scan the APs: the Router check compares what the car hears with the site sheet."),
    ("Running a step", "run button command read only output stop running",
     "Read-only steps have a Run button. One command runs at a time; its output streams to the Command output tab and Stop "
     "ends it. Every step shows the exact command, so the runbook in a terminal does the same thing."),
    ("Recording a lap", "lap laps record recording survey route drive marks mark letter key waypoint press",
     "Press Run on a lap step (set its length and a note), drive the route, and press the waypoint's letter (button on the "
     "Live tab, or the key on the keyboard) as the car passes each taped marker. Marks are how readings get positions on "
     "the map; a lap with fewer than two marks cannot be mapped."),
    ("Changing the car's radio", "join switch roam change confirm confirmation bssid type retype target car_change handover",
     "Join, timed switch and fast roam change the car's radio. Pick the target and press Start: the car's current state is "
     "read first, then a confirm box shows exactly what will change. Type the target BSSID (or the SSID for join) yourself; "
     "a wrong or empty answer cancels with nothing sent. One change at a time."),
    ("UNKNOWN outcome", "unknown outcome hold repeat resend failed timeout",
     "If a change's outcome is UNKNOWN, the platform holds further changes until a Car status read shows the real state. "
     "Never repeat the command: run Car status, read it, then decide."),
    ("Format check", "format check fail pass warn check failed gateway route samples",
     "A 10 s read-only lap that checks the car's real output against what the analysis needs. A FAIL blocks the plan and "
     "the Next step box names the fix. Run it again after fixing."),
    ("Map", "map floor plan heatmap position car where layers points cells roams outages coverage visual",
     "The Map tab draws the site: APs, waypoints and the route, the car (hollow when its place is estimated after the last "
     "mark), the measured signal (dots or heatmap cells), roams (diamonds, red when bad) and outages (red rings). Choose one "
     "lap or All laps to see the radio map built so far. Colours use the Good and Fair signal settings."),
    ("Health", "health score latency loss signal roam roams bad good fair kpi",
     "Health shows the share of samples with good signal, gateway latency (median and 95th percentile), outages, roams and "
     "bad roams (reply gap longer than the Bad roam setting). Score = good-signal share x share of the lap with replies."),
    ("Settings", "settings setting parameter parameters tune tuning configure configuration change default reset",
     "Settings tab: every parameter the commands use, grouped, with default, range and unit. Changes are checked before "
     "saving, apply to the next command and are recorded in Activity. Reset puts a setting back to the runbook's value."),
    ("Activity", "activity log audit history oversight who what when record",
     "Activity lists every action: commands started and how they ended, confirmations, setting changes, site saves, "
     "checklist ticks and assistant questions. It is saved as platform_audit.jsonl next to the laps."),
    ("Analysis", "analysis analyse replay twin coverage report proposal wirelessjev shadow robots fleet",
     "Analysis runs on the laptop from recorded laps: lap analysis (outages, stops, blockage rule), the 10-robot replay "
     "(the ten route laps as a fleet), the coverage report (map image) and the WirelessJEV proposal, which is shadow only."),
    ("Ghost traffic", "ghost ghosts load traffic fleet load nine robots",
     "Ghost traffic is nine virtual robots' traffic sent from this laptop to the test network, so the car's lap runs under "
     "fleet load. Set the target on Settings; the Plan's load step shows the command to run in a second terminal."),
    ("Assistant", "assistant ask agent model local ollama llama claude offline question help",
     "Ask anything about the day or the platform. It answers from a local model on this laptop if one is set up "
     "(Settings: Assistant), else Claude when online, else the built-in guide. It only advises; it never runs commands."),
    ("Demo", "demo rehearse rehearsal practice synthetic fake",
     "Start with --demo to rehearse: a synthetic site and a car driving the loop. Commands that talk to the car are off."),
    ("Safety", "safety safe wlan0 hotspot 192.168.1 password ssh security",
     "Only the test radio wlan1 is used; the management hotspot (192.168.1.x) is never touched. The SSH password is set in "
     "the terminal that started the platform and never typed into the page. The platform listens only on this laptop."),
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def best_topics(question: str, n: int = 2) -> list[tuple]:
    q = _words(question)
    scored = []
    for title, keys, text in TOPICS:
        k = _words(keys) | _words(title)
        hits = len(q & k)
        if hits:
            scored.append((hits, title, text))
    scored.sort(key=lambda s: -s[0])
    return [(t, x) for _, t, x in scored[:n]]


def is_next_question(question: str) -> bool:
    q = question.lower().strip()
    return not q or any(p in q for p in ("what next", "what should i do", "next step", "what now", "what do i do"))


def manual() -> str:
    return "\n".join(f"- {t}: {x}" for t, _, x in TOPICS)


# ---- shown on the Help tab --------------------------------------------------------------------
BEFORE_SITE = [
    ("Get the platform", "Done if this page is open. To update it later: unzip the newer field kit over this folder (the laps "
     "in artifacts/onecar are kept), or git pull if you run it from the repository, then start it again.", ""),
    ("Car files on this laptop", "The car's host key and Wi-Fi list from the SmartCar package (ask whoever holds it; it contains "
     "passwords, so it never goes in the repo, a chat or a screenshot).",
     "cp <SmartCar>/PC/data/deploy_known_hosts ~/smartcar_known_hosts\ncp <SmartCar>/PI/config/WiFi.json ~/WiFi.json"),
    ("Local assistant model (needs internet, do it at home)", "Install Ollama from ollama.com (or LM Studio), pull a small instruct "
     "model, then set its name on Settings > Assistant. On site the laptop has no internet, so without this the assistant "
     "answers from the built-in guide only. Choosing: an instruct (chat) model of about 3 to 8 billion parameters is a good "
     "start; smaller answers faster on a laptop, larger follows instructions better. Try it in demo mode before the day.",
     "ollama pull <model name>\nollama serve"),
    ("Rehearse", "Start the platform with --demo and try every tab: a synthetic site and a car driving the loop.",
     "./field_platform.sh --demo"),
    ("Pack", "Charged SmartCar, its charger and a power bank; the vendor remote or the phone with the driving app; 2-4 routers or "
     "APs, a switch, cables, power strips; this laptop and its charger; tape and a tape measure; a USB stick to back up "
     "artifacts/onecar before leaving; a cart or a person for blockage trials; a second person to type marks.", ""),
    ("SSH password on the day", "Typed into the terminal that starts the platform; the starter script asks for it. Never type it "
     "into the page.", ""),
]

WHATS_NEW = [
    ("17 Nov 2026", [
        "Site radio map: RadioMap fitted to every positioned reading and drawn over the whole floor, with views for signal, "
        "best AP, backup signal from a second AP location, SNR, SINR, co-channel APs and uncertainty; faded where nothing was "
        "measured; hover for the values at any spot.",
        "Car changes (join, switch, roam) start from the page; you type the target in a confirm box. An UNKNOWN outcome holds "
        "further changes until a status read.",
        "Settings for every parameter, an Activity log of every action, a router checklist and router check.",
        "Assistant: a small model on this laptop first (Ollama or LM Studio), then Claude when online, then the built-in guide.",
        "Fixed: the Plan's lap box overrode the lap length from Settings.",
        "Fixed: a wrong confirmation (which sends nothing) held further changes as if the outcome were unknown.",
        "Fixed: typed lap notes were wiped every 1.5 s.",
        "Fixed: error messages and the stop button relied on browser pop-ups that some browsers hide.",
    ]),
]


def page_help() -> dict:
    return {"before": [{"title": t, "text": x, "commands": c} for t, x, c in BEFORE_SITE],
            "topics": [{"title": t, "text": x, "keys": k} for t, k, x in TOPICS],
            "whats_new": [{"date": d, "items": items} for d, items in WHATS_NEW]}
