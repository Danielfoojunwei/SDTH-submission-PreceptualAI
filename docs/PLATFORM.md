# WirelessJEV Field Platform

The Field Platform is the production tool for testing on a real site. It is one local web page that runs the whole field day:

- **Setup:** the site sheet and the router checklist.
- **The car:** read-only commands, and radio changes behind a typed confirmation.
- **Laps:** recording them with waypoint marks, and watching each one live.
- **Map:** how the car is mapping the site.
- **Analysis.**
- **Control:** every setting, with a record of everything done.
- **Assistant:** ask it what to do next and how to use the platform.

It wraps the same tested commands as the runbook ([TUESDAY_RUNBOOK.md](TUESDAY_RUNBOOK.md)): `onecar`, `hil.analyse`, `hil.trace_twin` and `native_ext.coverage`. Every command the page runs is shown in full, so the runbook in a terminal and the page are interchangeable.

For the simulation evidence (every canonical campaign, its parameters and results), use the Studio. The platform is for real hardware only.

## Getting it to someone else (the field kit)

The whole repository is too large to hand over (about 600 MB of files and a 4.8 GB history). The field kit carries the whole field system instead. Build it with:

```bash
.venv/bin/python scripts/make_field_kit.py
```

This writes `artifacts/field_kit/wirelessjev-field-kit-<date>.zip`, about 1.6 GB, containing:

- **The code**, the starter script, the guides, and the one data file the field code reads at run time: `artifacts/warehouse_v2/studies/wifi_link_curves.json`, used by the 10-robot replay. That file was found by tracing every field command's file opens.
- **WirelessJEV's d3-lite model.** `vllm-sr/d3-lite` at the pinned revision, Apache-2.0, with its LICENSE and NOTICE, as a ready Hugging Face cache including the saved file listing (`trees/`), so it loads with no internet. Every weight file is checked against its SHA-256 when the kit is built and again on first run.
- **Every Python package as wheels** for Apple-Silicon Macs with Python 3.12 (macOS 14+): the pinned `requirements-field.txt` and `requirements-wirelessjev.txt` and their dependencies, 58 wheels, 204 MB. Build them once into `artifacts/field_kit/wheels-macos-arm64-py312` with `pip download --only-binary=:all: --platform macosx_14_0_arm64 --python-version 3.12 --implementation cp -r requirements-field.txt -r requirements-wirelessjev.txt`.

`--code-only` builds the small kit without the model and the wheels, for updates. Builds refuse anything that looks like car credentials, lap data or local settings.

The recipient unzips the kit and follows `FIELD_KIT.md`. On first run, `./field_platform.sh`:

- checks the model files;
- creates `.field-venv` and installs every package, from the bundled wheels on an Apple-Silicon Mac with Python 3.12 (no internet), or otherwise by downloading the same pinned versions once;
- makes the kit's code importable.

On every run it points the system at the bundled model and keeps it offline (`HF_HOME` is the kit's cache, with Hugging Face access switched off). `--setup-only` sets up and stops.

Tested from an empty folder on this Mac: setup ran inside a sandbox with no internet, then lap analysis, the 10-robot replay, the coverage report and the WirelessJEV proposal all ran through the platform. A stock Mac's Python (3.9) gets a clear message. Windows and Intel Macs are untested; Intel Macs and Linux download the packages.

## Start it

From the repo folder, in a terminal you will keep open all day:

```bash
read -s WJ_SSH_PASSWORD; export WJ_SSH_PASSWORD
```

```bash
.venv/bin/python -m wirelessjev.platform
```

Then open http://127.0.0.1:8770 on this laptop. The terminal prints whether the SSH password is set and who will answer the assistant.

Options:

| Option | Default |
|---|---|
| `--log-dir` | `artifacts/onecar` |
| `--known-hosts` | `~/smartcar_known_hosts` |
| `--wifi-json` | `~/WiFi.json` |
| `--port` | 8770 |

To rehearse without the car:

```bash
.venv/bin/python -m wirelessjev.platform --demo
```

Demo mode makes a synthetic site with three APs and a shelf row, plus one finished survey lap and a car driving the loop live. All of it is made-up data and labelled so. Commands that talk to the car are switched off; everything else works.

## Tabs

| Tab | What it does |
|---|---|
| Plan | The day as steps in runbook order, with progress (for example, route laps 3 of 10). Read-only steps have a Run button; lap steps take a length and a note. Car-changing steps have a target picker and a Start button (see Changing the car's radio). |
| Map | The site's radio map over the whole floor, fitted by RadioMap to the car's readings, with views and layers (listed below), plus link health, a signal timeline, an AP table and an events list. |
| Setup | This laptop's checks; the site sheet (AP rows and taped waypoints, validated before saving); the router checklist; the router check. |
| Live lap | Waypoint buttons while a lap records (the letter key works too), and the live view of the newest lap. |
| Analysis | Lap analysis, the 10-robot replay, the coverage report (with its map image) and the WirelessJEV proposal in shadow. All run on the laptop from recorded laps. |
| Settings | Every parameter (see below). |
| Activity | Every action, newest first (see Oversight). |
| Command output | The running command's output, a stop button (press twice) and recent commands. |

The side panel shows the next step (red-edged during a hold), the assistant and the last format check.

### The radio map

The map is the site's radio map, the kind a site-survey tool shows. Every positioned reading is used: the serving AP's link samples, and every AP in each scan snapshot. These are fitted with RadioMap, the system's own map model, separately for each AP and band. The model is Bayesian: log distance, the obstructions drawn as boxes on the site sheet, and random features for shadowing nobody knows, with each AP's real transmit power from the sheet. It is then evaluated over the whole floor.

The Show list picks the view:

| View | What it shows |
|---|---|
| Signal strength (best AP) | The strongest AP's signal at each spot |
| Best AP | Which AP serves where |
| Backup signal | The strongest AP at a different location from the best one: the second path WirelessJEV relies on |
| SNR | Best signal against the noise floor |
| SINR | Best signal against noise plus every AP on the same channel |
| Co-channel APs | Other APs on the serving channel heard at or above −82 dBm, which share airtime |
| Uncertainty | RadioMap's own error estimate |
| Measured readings only | The chosen lap's readings averaged per cell, no model |
| Signal: one AP | Each AP on its own |

Colours use a perceptually uniform scale. Lines mark the Good and Fair signal thresholds. Hovering anywhere shows every value for that spot, plus the distance to the nearest real reading.

**Fade where not measured** (on by default) fades everything more than 3 m from any reading, because there the map is mostly the model's guess. With no laps yet, the map is the site sheet's physics alone, and the page says so.

On the demo, the fitted map is within about 1 dB of the synthetic truth on the route and 2 to 5 m from it. That is the easy case: the demo's physics are the model's own family, with no fading. It says nothing yet about a real site. RadioMap's uncertainty was somewhat optimistic there, which is why the fade uses distance to real readings rather than the model's confidence.

The Map tab's layers:

- APs with their channels;
- waypoints and the route;
- the car (hollow and dashed when its place is estimated after the last mark);
- obstructions from the site sheet, the Good and Fair lines, and the fade;
- the car's own readings, coloured by signal grade;
- roams (diamonds, red when bad);
- outages (red rings).

Pick one lap, or "All laps" for the radio map built so far. The rest of the Map tab:

- **Link health:** a score, the share of good signal, gateway replies, latency (median and 95th percentile), roams and bad roams, and how many readings have a position.
- **Signal timeline:** coloured by serving AP, with outage bands, roam lines and waypoint ticks.
- **AP table:** for each AP, its share of serving time, signal, roams in and out, and its last scan reading.
- **Events list:** marks, roams and outages.

### How the map gets positions

The car has no positioning of its own. Each reading is placed by interpolating in time between the operator's waypoint marks, the same rule the coverage report and the 10-robot replay use. Readings before the first mark or after the last one get no position: nothing is invented. While a lap records, the car's place after its last mark is an estimate (progress towards the next waypoint at the speed driven so far), drawn dashed. A lap with fewer than two marks cannot be mapped.

### Link health

- **Score:** the share of link samples at or above the Good signal setting, multiplied by the share of the lap with gateway replies. 100 means good signal and no reply gaps all lap.
- **Outage:** a reply gap longer than the outage threshold.
- **Bad roam:** a roam whose reply gap is longer than the Bad roam setting.

The page shows these definitions next to the numbers.

The layout follows patterns from Cisco Meraki's wireless health and roaming analytics:

- connection and performance health;
- a roaming timeline;
- APs placed on a floor plan.

Everything is scaled down to one car on one site.

## Changing the car's radio

Join, timed switch and fast roam start from the Plan tab:

1. Pick the target from the site sheet and press **Start**.
2. `onecar` reads the car's current state and prints exactly what it will change.
3. The page shows that in a confirm box. You type the target yourself (the BSSID, or the SSID for join). The box is never pre-filled.
4. A wrong or empty answer cancels with nothing sent, exactly as in a terminal.

Rules:

- **One change at a time.**
- **No resending.** The connect is never resent.
- **UNKNOWN hold.** If the outcome is UNKNOWN, unverified, or the command stops after you confirmed, further changes are held until a Car status read succeeds. The hold survives a restart.

The terminal command is still shown under each step. Behind the scenes the page runs `onecar ... --confirm-stdin`. Without that flag, `onecar` still refuses to run a change outside a real terminal.

## Routers set up in their own app

Consumer routers (configured in a phone or web app) cannot be configured by the platform, so it does two things instead:

- **Router checklist (Setup):** what to set in each app:
  - the same SSID and password, with WPA2-Personal;
  - extra routers in AP or bridge mode;
  - not 192.168.1.x;
  - fixed, different channels, not Auto;
  - fast roaming (802.11r) on if offered;
  - fixed transmit power;
  - placement;
  - the test network in `~/WiFi.json`.

  Tick each item when it is done on every router. It is a plan step.
- **Router check (Setup):** compares the newest scan the car made with the site sheet:
  - an AP not heard;
  - a different channel than the sheet says;
  - a different SSID;
  - an unknown AP broadcasting your SSID (usually another band of one router);
  - two sheet APs on one channel;
  - overlapping 2.4 GHz channels.

## Settings

Every value the commands, the map and the assistant use, in groups:

- **Laps:** length, sample period, scan interval, format-check length.
- **Car reads:** status reads, scan length.
- **Radio changes:** timing on or off, verify reads, timing windows.
- **Analysis:** outage threshold, replay collision rate, coverage grid (cell size, neighbours, reach, route margin), the WirelessJEV mission.
- **Map and health:** Good and Fair signal, bad roam, good latency, heatmap cell.
- **Ghost traffic:** target, count, video, bulk, duration. Settings builds the command for the second terminal.
- **Assistant.**

How changes work:

- Each setting shows its default, its range and unit, what it does and which steps use it.
- A save checks every change first and saves only if all of them are valid.
- Saved values apply to the next command. They are kept in `platform_config.json` in the log folder; only differences from the default are stored.
- Reset puts a setting back to the runbook's value.
- The ghost target uses the ghost tool's own rule: a private address, never 192.168.1.x.
- The total ghost load is capped.

## Oversight

Activity records, one line each:

- every command started and how it ended (including the car's reported outcome);
- confirmations (what was typed, and whether it cancelled);
- holds and when they cleared;
- setting changes (old and new);
- site saves, checklist ticks and waypoint marks;
- assistant questions (and who answered);
- platform starts.

It is saved as `platform_audit.jsonl` next to the laps. The log refuses any field that looks like a password.

## The assistant

"What next?" and free questions ("How do I set the router channels?") are answered from the platform's live state:

- steps done;
- environment checks;
- the last format check;
- the newest lap's health and events;
- the last commands' output;
- any hold;
- changed settings.

Who answers, in the order set on Settings:

1. **A small model on this laptop.** It works with no internet, the normal case on site, and the address must be on this laptop (127.0.0.1 or localhost), so site data never leaves it. The model is given the platform's manual (the built-in help topics) and a compact state, and replies in a fixed JSON shape. Supported servers:
   - Ollama (`http://127.0.0.1:11434`);
   - any OpenAI-compatible server (llama.cpp server on port 8080, LM Studio on port 1234).
2. **Claude**, when the `anthropic` package, an API key and internet are available.
3. **The built-in guide**, which always works. It follows the runbook order, stops on a failed format check or a hold, and answers "how do I" questions from the help topics.

Every answer says which source it came from. The assistant only advises: it cannot run anything and never answers a confirmation.

### Setting up the local model

Do this at home, with internet, before the site day. The platform does not install anything itself:

1. Install Ollama (ollama.com) or LM Studio.
2. Pull a small instruct model, for example one of 3 to 8 billion parameters from a family your server lists. Ollama: `ollama pull <model>`. Smaller answers faster on a laptop CPU; larger follows instructions better.
3. Start the server (`ollama serve`, or LM Studio's local server).
4. On Settings > Assistant, set the server type, the address and the model name exactly as the server lists it.

The Setup tab's Assistant row turns green when the model answers.

This was tested against stand-in servers only. No real local model has been run with the platform yet.

## Safety model

- **Read-only steps** run from the page. They need `WJ_SSH_PASSWORD` set in the terminal that started the platform; the page never asks for a password.
- **Car-changing steps** need the same password, plus a target from the site sheet and your typed confirmation. They are refused in demo mode and during a hold.
- **One command at a time.**
- **Local only.** It listens on 127.0.0.1. Every action request must come from the page itself (checked by origin and content type). The local model address must be on this laptop.
- **The existing rules still hold:**
  - `wlan1` only, never `wlan0` or the 192.168.1.x hotspot;
  - the pinned host key;
  - an UNKNOWN outcome is never resent;
  - the learned model runs in shadow only.

## Code

`src/wirelessjev/platform/`:

- `steps.py`: the step list and the exact command for each step, built from the settings.
- `params.py`: every setting, with its validation.
- `guide.py`: the offline next-step logic and environment checks.
- `help.py`: the help topics (offline answers and the local model's manual).
- `agent.py`: the assistant (local model, Claude, guide).
- `mapview.py`: positions, roams, outages, per-AP figures, health and the measured cells.
- `spectrum.py`: the radio map (RadioMap with the site sheet's obstructions) and its views.
- `sitecheck.py`: the router checklist and the router check.
- `jobs.py`: runs one command at a time; waypoint marks; typed confirmations.
- `audit.py`: the activity log.
- `demo.py`: the rehearsal site and lap.
- `server.py`: the local web server and its safety checks.
- `page.py`: the page itself (one file, no internet needed).

Tests (no car, no network):

- `tests/test_platform.py`
- `tests/test_platform_control.py`
