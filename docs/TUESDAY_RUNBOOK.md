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
- **Network:** the same subnet and one DHCP server, so the IP survives a roam.
- **Placement:** so two APs are usable along most of the route.
- **Car credentials:** add the test SSID to your local `~/WiFi.json` (kept out of the repo); the first `switch --bssid any` joins the car to it.

| AP | BSSID | Band / channel | x, y, z (m) | Tx power (dBm) |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |

**Waypoints:** tape markers A, B, C, … along the route, measure each, and save `waypoints.json` as `{"A": [x, y], "B": [x, y]}`.

## Commands
Set once:
```bash
export KH=~/smartcar_known_hosts
```

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
Start the lap, then drive. Type the waypoint letter and Enter as the car passes each marker.
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
Fast roam:
```bash
uv run python -m wirelessjev.onecar roam --known-hosts $KH --bssid <bssid>
```

### Block 4. Ten laps for the 10-robot fleet (read-only)
Drive ten *different* routes, one lap each, named `r01` … `r10`:
```bash
uv run python -m wirelessjev.onecar lap --known-hosts $KH --name r01 --seconds 180
```

### Block 5. Laps under fleet load (read-only on the car)
On the laptop, joined to the test SSID, start nine ghost robots (about 56 Mb/s), then record a lap:
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
This already gives a first 10-robot result on the measured signals.

## If time runs short
Keep, in this order: blocks 1, 2, 3, 4. Drop block 6 first, then block 5.

## Not built for Tuesday
A Bluetooth heartbeat test and a laptop-as-relay test (directions 2 and 3) are simulated only. They need the Bluetooth probe result and a second device before they can be tried on site.
