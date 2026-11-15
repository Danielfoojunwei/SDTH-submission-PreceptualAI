# Tuesday 17 Nov 2026: one SmartCar, your own APs, one day

Everything below runs from the laptop over the car's management hotspot.
Nothing drives the car and nothing changes the radio without your typed
confirmation. The learned radio model runs in shadow only.

## Safety rules (unchanged)
- Client radio `wlan1` only. Never `wlan0` or 192.168.1.11's hotspot.
- Pinned host key; SSH password typed at the prompt or in `WJ_SSH_PASSWORD`, never stored.
- Every change needs the target re-typed. One request at a time. An UNKNOWN outcome is never resent.
- No Wi-Fi or SSH passwords in the repo, logs or chat.

## Bring
| Item | Why |
|---|---|
| SmartCar, charged, plus a power bank | The day is long |
| 2–4 access points, one switch, cables, power strips | Overlapping cells; all on one subnet |
| This laptop (management + ghosts) | Runs every command below |
| A second laptop or phone (optional) | Ghost-traffic sink, or a second ghost station |
| Tape and a tape measure | Waypoints and AP positions in metres |
| A cart or a person | Blockage trials |

## AP setup sheet
Fill one row per BSSID and copy it into `site.json` (Codex's format: `python PC/run.py init-site`).
- **SSID and security:** the same SSID and WPA2/3-PSK on every AP.
- **Roaming:** 802.11r on if the AP has it.
- **Channels:** fixed and non-overlapping. Example for 5 GHz: 36, 52, 100.
- **Network:** the same subnet and one DHCP server, so the IP survives a roam. **Not 192.168.1.x:** the car's management hotspot uses that range, and the car and the laptop would each have two radios on one subnet.
- **Placement:** so two APs are usable along most of the route.
- **Car credentials:** add the test SSID to your local `~/WiFi.json` (kept out of the repo); the first `switch --bssid any` joins the car to it.

| AP | BSSID | Band / channel | x, y, z (m) | Tx power (dBm) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**Waypoints:** tape markers A, B, C, … along the route, measure each, and save `waypoints.json` as `{"A": [x, y], "B": [x, y]}`.

## Commands
**Before leaving home (needs internet):** run `uv sync` once in the repo. On site the laptop is on the car's hotspot and has no internet.

**Two files must exist on the laptop** (neither is in the repo; on 15 Nov neither was at these paths):
- `~/smartcar_known_hosts`: the car's host key, copied from SmartCar.zip `PC/data/deploy_known_hosts` (see [ONE_CAR_TEST.md](ONE_CAR_TEST.md)).
- `~/WiFi.json`: the car's Wi-Fi list, with the test SSID added.

Set once per terminal:
```bash
export KH=~/smartcar_known_hosts
```
```bash
export UV_OFFLINE=1
```
`UV_OFFLINE=1` stops `uv` from trying the network. If preflight fails, it names the missing file.

### Block 1. Preflight and probes (read-only)
```bash
uv run python -m wirelessjev.onecar preflight --known-hosts $KH --wifi-json ~/WiFi.json
```
```bash
uv run python -m wirelessjev.onecar status --known-hosts $KH --repeat 10
```
```bash
uv run python -m wirelessjev.onecar capabilities --known-hosts $KH
```
```bash
uv run python -m wirelessjev.onecar handover-probe --known-hosts $KH
```
**Read in the output:**
- `two_clients_two_channels`: can one card hold two AP connections?
- `dbus_roam_method`: is fast roam possible?
- `bluetooth_le`.
- `aps_advertising_fast_transition`.

### Block 2. Survey laps (read-only; 3 or more)
Start the lap, then drive. Type the waypoint letter and Enter as the car passes each marker. Upper or lower case both work; a letter that is not in `waypoints.json` is ignored. Each lap needs at least two marks to get positions.
```bash
uv run python -m wirelessjev.onecar lap --known-hosts $KH --name survey1 --seconds 180
```
Also take Codex's parked, position-labelled captures at 10–20 spots for its map and shadow model (see `docs/SMARTCAR_NUMPY_DEPLOYMENT.md`, `PC/run.py capture`).

### Block 3. Handover timing (supervised; 5 or more of each)
Today's switch:
```bash
uv run python -m wirelessjev.onecar switch --timing --known-hosts $KH --wifi-json ~/WiFi.json --ssid "<SSID>" --bssid <bssid>
```
Unlock the profile once:
```bash
uv run python -m wirelessjev.onecar switch --timing --known-hosts $KH --wifi-json ~/WiFi.json --ssid "<SSID>" --bssid any
```
Fast roam (refused, with nothing sent, until the profile has been unlocked by the step above):
```bash
uv run python -m wirelessjev.onecar roam --known-hosts $KH --bssid <bssid>
```
**Read in the `handover_timing` line:** `outage_s` (the longest silence in the gateway ping), `stages_s`, `ip_retained`. An outcome of `UNKNOWN` means read `status` again; do not repeat the command blindly.

### Block 4. Ten laps for the 10-robot fleet (read-only)
Drive ten *different* routes, one lap each, named `r01` … `r10`:
```bash
uv run python -m wirelessjev.onecar lap --known-hosts $KH --name r01 --seconds 180
```

### Block 5. Laps under fleet load (read-only on the car)
On the laptop, joined to the test SSID, start nine ghost robots (about 50 Mb/s), then record a lap:
```bash
uv run python -m wirelessjev.hil.ghosts send --target <AP or sink IP> --seconds 200
```
```bash
uv run python -m wirelessjev.onecar lap --known-hosts $KH --name load1 --seconds 180
```
The laptop needs a second network interface for this (USB Wi-Fi or the second laptop), because its built-in Wi-Fi is on the car's hotspot.

### Block 6. Blockage trials (read-only)
Record laps while a person or cart stands between the car and its AP at marked spots. Note the times; add `--note "blocker at C 14:32"`.
```bash
uv run python -m wirelessjev.onecar lap --known-hosts $KH --name block1 --seconds 180 --note "blocker at C"
```

### Block 7. Check the data before leaving
```bash
uv run python -m wirelessjev.hil.analyse artifacts/onecar/*-lap-*.jsonl
```
Every lap should show samples, pings and at least two marks. The report's `blockage_rule_replay` section shows how many real outages the simulator's blockage rule would have caught, and its false alarms (see [BLOCKAGE_DETECTOR.md](BLOCKAGE_DETECTOR.md)). Block 6 is what tests it, so try to keep block 6.
```bash
uv run python -m wirelessjev.hil.trace_twin artifacts/onecar/*-lap-r*.jsonl --site site.json --waypoints waypoints.json
```
This already gives a first 10-robot result on the measured signals. With ten 180 s laps it takes about a minute.

## If time runs short
Keep, in this order: blocks 1, 2, 3, 4. Drop block 6 first, then block 5.

## Not built for Tuesday
A Bluetooth heartbeat test and a laptop-as-relay test (directions 2 and 3) are simulated only. They need the Bluetooth probe result and a second device before they can be tried on site.

## Dry run, 15 Nov 2026
Every command above was run against a fake car on the laptop (`tests/test_runbook_dry_run.py`): no network, no car.

**What was exercised**
- Every command in this file is parsed by the real command-line parsers, so a mistyped flag fails the test.
- Blocks 1–7 in order: preflight, status, both probes, three survey laps, a timed switch, the unlock, a fast roam, ten route laps, ghost traffic, a lap under load, a blockage lap, then both offline checks on the files produced.
- Safety rules: nothing sent to the car names the management address; `wlan0` appears only as the guard that tells SmartCarCmd which radio to leave alone; no Wi-Fi or SSH password appears in any file written; a connect whose channel drops is recorded as UNKNOWN and is not sent again; a wrong confirmation sends nothing.

**What it found, now fixed**
| Problem | Fix |
|---|---|
| Handover outage was measured from the first ping reply after the command. Replies keep arriving for a moment before the link drops, so a 3.4 s outage could be reported as 0.05 s. | The outage is now the longest silence after the command. |
| The blockage-rule replay counted one alarm for every two low readings, so a single 2 s dip gave several "false alarms". | One alarm per dip. |
| This file told you to read `two_clients_two_channels`, which the probe did not print. | The probe prints it. |
| This file said the ghost load was about 56 Mb/s. | It is about 50 Mb/s. |
| A waypoint typed in lower case was dropped, leaving the lap without positions. | Either case is accepted. |
| Preflight crashed if the host-key file was missing. | It reports the missing file and fails. |
| Ghost traffic could be aimed at the car's management address. | Refused. |

**What a fake car cannot show**
- **Real radio behaviour:** whether roams succeed, how long they take, whether the card holds two connections.
- **The car's real output formats:** the fake prints `iw`, `ping -D`, `journalctl` and `nmcli` text in the shape the parsers expect. If the car's versions print something else, laps or timings come back empty. Check the first survey lap with block 7 before recording more.
- **`sudo -n` permissions:** the scan dump, the ping and the journal need passwordless sudo on the car.
- **A gateway on `wlan1`:** with no default route on `wlan1` a lap records no pings, and the outage analysis has nothing to use.
- **The real sample rate:** the lap loop sleeps 0.1 s between readings, so the car will log somewhat fewer than 10 readings a second.
- **Clock offset:** measured once per lap; drift within a lap is not corrected.
- **The second network interface for ghosts**, and whether the access point isolates clients.

