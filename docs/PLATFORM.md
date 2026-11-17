# WirelessJEV Field Platform

The Field Platform is the production tool for testing on a real site. It is one local web page that runs the whole field day: setting up the site, running the read-only car commands, recording laps with waypoint marks, watching the lap live, analysing it, and asking an assistant what to do next.

It wraps the same tested commands as the runbook ([TUESDAY_RUNBOOK.md](TUESDAY_RUNBOOK.md)): `onecar`, `hil.analyse`, `hil.trace_twin` and `native_ext.coverage`. Everything the page runs, it shows as the exact command line, so the runbook and the page stay interchangeable.

For the simulation evidence (every canonical campaign, its parameters and results), use the Studio instead. The platform is for real hardware only.

## Start it

From the repo folder, in a terminal you will keep open all day:

```bash
read -s WJ_SSH_PASSWORD; export WJ_SSH_PASSWORD
```

```bash
.venv/bin/python -m wirelessjev.platform
```

Then open http://127.0.0.1:8770 on this laptop. The terminal prints whether the SSH password is set and whether the assistant can use Claude.

Options: `--log-dir` (default `artifacts/onecar`), `--known-hosts` (default `~/smartcar_known_hosts`), `--wifi-json` (default `~/WiFi.json`), `--port` (default 8770).

To rehearse without the car:

```bash
.venv/bin/python -m wirelessjev.platform --demo
```

Demo mode uses a temporary folder, placeholder car files and a synthetic lap. Commands that talk to the car are switched off; the Setup, Live lap and Analysis tabs work.

## What is on the page

| Tab | What it does |
|---|---|
| Plan | The day as steps in runbook order, with progress (for example route laps 3 of 10). Read-only steps have a Run button; lap steps take seconds and a note. |
| Setup | Checks this laptop (host key, Wi-Fi list, SSH password, site sheet, assistant). The site sheet: AP rows (BSSID, SSID, frequency, x/y/z, transmit power) and taped waypoints, validated before saving as `site.json` and `waypoints.json` in the log folder. |
| Live lap | Waypoint buttons while a lap records (the letter key works too), and the live view of the newest lap. |
| Analysis | Lap analysis, the 10-robot replay, the coverage report (with the map image) and the WirelessJEV proposal in shadow. All run on the laptop from recorded laps. |
| Command output | The running command's output, a stop button, and recent commands. |

The side panel shows the next step, the assistant and the result of the last format check.

## The assistant

"What next?" and free questions are answered from the platform's live state: which steps are done, the environment checks, the last format check, the newest lap and the last commands' output.

- **With internet and credentials**, it uses Claude (`claude-opus-5-5`) and answers with a short explanation and at most one suggested step, which you can run if it is read-only. To enable it: `.venv/bin/python -m pip install anthropic`, and `export ANTHROPIC_API_KEY=...` in the terminal before starting the platform.
- **Offline, which is normal on site** (the laptop is on the car's hotspot), every answer comes from the built-in guide. The guide follows the runbook order, stops on a failed format check and names the fix. The answer says which source it came from.

The assistant only advises. It cannot run anything, and it is told never to suggest running a car-changing step from the page.

## Safety model

- **Read-only steps** (preflight, status, probes, format check, laps, scan) run from the page. They need `WJ_SSH_PASSWORD` set in the terminal that started the platform; the page never asks for a password.
- **Steps that change the car** (join, switch, roam) are never run from the page. It shows the command; run it in a terminal, where it asks you to retype the target before sending.
- **One command at a time.** A second Run is refused until the first finishes or is stopped.
- **Local only.** It listens on 127.0.0.1. Every action request must come from the page itself (checked by origin and content type), so another website cannot drive it.
- The existing rules still hold: `wlan1` only, never `wlan0` or the 192.168.1.x hotspot; the pinned host key; an UNKNOWN outcome is never resent; the learned model runs in shadow only.

## Code

`src/wirelessjev/platform/`:

- `steps.py`: the step list and the exact command for each step.
- `guide.py`: the offline next-step logic and environment checks.
- `agent.py`: the assistant (Claude when available, otherwise the guide).
- `jobs.py`: runs one command at a time and feeds waypoint marks to a recording lap.
- `server.py`: the local web server and its safety checks.
- `page.py`: the page itself (one file, no internet needed).

Tests: `tests/test_platform.py` (no car, no network).
