# One-car real test: read-only status, then one supervised BSSID switch

**Goal:** prove WirelessJEV's execution path on the one existing Raspbot, with the same hardware and on-car software as SmartCar.zip, before any autonomous control.
- **Pass:** status reads are correct and repeatable, and exactly one operator-confirmed BSSID switch is applied and verified (or cleanly reported as not applied), with the management link never touched.
- **Fail:** anything else, logged as-is.

## What is on the car already (unchanged by this test)
- `~/SmartCarCmd.py` and `~/SmartCatScan.py`, deployed by SmartCarDeploy. This test deploys nothing and edits nothing on the Pi.
- SmartCarCmd's own guards:
  - It re-discovers radio roles from the bus: the SDIO radio is HOTPOT and the PCIe radio is WIFI.
  - It refuses to act on the HOTPOT radio or the interface holding 192.168.1.11.
  - It serialises radio access with a lock, refuses a target BSSID that isn't visible (`AP_NOT_VISIBLE`, activation not attempted), and verifies SSID/BSSID after the nmcli activation.
- SmartCarGuard keeps the hotspot alive; it doesn't touch `wlan1`.
- The management SSH session runs over `wlan0`/hotspot, a different radio from the one being switched. The switch therefore can't cut the control connection; only a Pi or firmware fault could.

## How WirelessJEV talks to it (`src/wirelessjev/carlink.py`)
- Same invocation as the PC app: SSH to `192.168.1.11:22` as `pi`, then
  `sudo -n python3 -u "$HOME"/SmartCarCmd.py --interface wlan1 --management-interface wlan0`,
  with one JSON request on stdin.
- **Host key pinned:** loads the `known_hosts` file you provide (SmartCar.zip ships `PC/data/deploy_known_hosts` with the car's ed25519 key) and uses `RejectPolicy`. The PC app's `AutoAddPolicy` is not used.
- **SSH password:** typed by you at a `getpass` prompt, or taken from the `WJ_SSH_PASSWORD` environment variable. It is never stored or logged.
- **Wi-Fi password:** SmartCarCmd needs it inside the `connect` request (that is how the PC app works today). It is read from the `WiFi.json` path you pass, only at the moment of sending. It never appears in plans, receipts or logs; the log writer refuses any record with a password field.
- **Timeouts as unknowns:** timeouts and dropped channels are recorded as UNKNOWN. A connect is never resent; the tool only reads status to reconcile.

## Prerequisites (you, before starting)
1. The car is powered, on the bench or parked, with motors idle. This test sends no motion commands.
2. This Mac is joined to the Raspbot hotspot, so `192.168.1.11` is reachable. This takes the Mac off its normal Wi-Fi.
3. The car's `known_hosts` file is copied somewhere readable, e.g. `~/smartcar_known_hosts` (from SmartCar.zip `PC/data/deploy_known_hosts`).
4. `WiFi.json` (SmartCar.zip `PI/config/WiFi.json`) is saved locally. Keep it out of this repo.
5. You know the customer SSID and two BSSIDs of it that the car can see. The SmartCar notes list two 5 GHz BSSIDs on channels 40 and 48.

## Steps
Each step writes a JSON-lines log to `artifacts/onecar/<timestamp>.jsonl`.

### 0. Preflight (offline, no network)
```bash
~/.local/bin/uv run python -m wirelessjev.onecar preflight --known-hosts ~/smartcar_known_hosts --wifi-json ~/WiFi.json
```
This checks that the known_hosts file has a key for 192.168.1.11 and that WiFi.json parses (version 1, enabled networks, valid security types). It prints only the SSIDs and priorities, never the passwords.

### 1. Read-only status (no radio change)
```bash
~/.local/bin/uv run python -m wirelessjev.onecar status --known-hosts ~/smartcar_known_hosts --repeat 10
```
It runs 10 `{"action":"status"}` calls about 1 s apart.

**Pass:**
- every reply has `ok: true`, `interface: wlan1` and `role: WIFI`;
- `connected` is stable;
- BSSID and SSID match what `iw dev wlan1 link` shows on the car;
- the signal varies by less than 6 dB while stationary.

**Abort** on any `INTERFACE_UNSAFE`, a host-key mismatch, or a sudo/permission error.

### 1b. Read-only capability probe (no radio change)
```bash
~/.local/bin/uv run python -m wirelessjev.onecar capabilities --known-hosts ~/smartcar_known_hosts
```
This runs only a fixed whitelist of query commands over the pinned SSH session (`iw` phy/info/link/survey, the existing scan *cache* rather than a new scan, the driver/PCI ID, `bluetoothctl show`). It never queries the `wlan0` hotspot radio and changes nothing.

It reports:
- **Client radio:** Wi-Fi 6/7 support, usable 6 GHz channels, multi-link support, and FTM ranging (initiator).
- **Scanning:** whether per-channel busy time is available.
- **Nearby APs:** how many advertise FTM responder, Wi-Fi 6/7 or 6 GHz.
- **Bluetooth:** whether a controller with Low Energy is present.

It then lists which ideas the hardware enables: attenuation from ranging, spectrum sensing, a Bluetooth heartbeat relay during reconnects, no-outage multi-link switching, and 6 GHz. Raw output is saved next to the log for checking.

### 2. Read-only scan (no radio change)
```bash
~/.local/bin/uv run python -m wirelessjev.onecar scan --known-hosts ~/smartcar_known_hosts --wifi-json ~/WiFi.json --seconds 20
```
This streams the existing `SmartCatScan.py` for 20 s, the same scan the PC app runs continuously, and then stops it. It records the visible BSSIDs of the enabled SSIDs (band, frequency, signal).

**Pass:** at least two BSSIDs of the target SSID are visible, and the current BSSID is among them.

### 3. Proposal (local only)
```bash
~/.local/bin/uv run python -m wirelessjev.onecar propose --scan-log artifacts/onecar/<scan>.jsonl --mission "Keep the camera uplink stable; prefer 5 GHz."
```
WirelessJEV (frozen d3-lite) scores the visible BSSIDs, described by band, signal and SSID priority, and prints its choice and probabilities. **Nothing is sent.** Proposing *away* from the current BSSID is the useful case. If the model proposes staying, pick the alternative BSSID yourself for step 4: this test is about execution, not model quality.

### 4. One supervised switch (the only mutation)
```bash
~/.local/bin/uv run python -m wirelessjev.onecar switch --known-hosts ~/smartcar_known_hosts --wifi-json ~/WiFi.json --ssid "<SSID>" --bssid <aa:bb:cc:dd:ee:ff>
```
1. A fresh status read (precondition). It aborts if the car is already on the target, or the radio roles are wrong.
2. You must re-type the target BSSID to confirm. Any mismatch cancels the switch, and nothing is sent.
3. Exactly one `connect` request is sent, with a 150 s timeout (the PC app's `Command` timeout).
4. Outcomes:
   - `ok: true` → **APPLIED**. Then 10 status reads confirm the BSSID and log signal and tx bitrate.
   - `AP_NOT_VISIBLE` or `INTERFACE_UNSAFE` (activation not attempted) → **NOT_APPLIED**. Safe to stop.
   - Timeout, channel loss or any other error → **UNKNOWN**. Up to 10 status reads reconcile it; the connect is **not** resent.
5. Wall-clock time from send to verified is recorded. This is the real handover cost, compared with the twin's assumed ~4 s.

**Pass:**
- the outcome is APPLIED and verified, or NOT_APPLIED with a clear code;
- the management SSH session survives the switch;
- exactly one connect appears in the log.

### 5. Return to the original BSSID (also supervised)
This is the same `switch` command with the original BSSID from step 1, again with typed confirmation. It is a second, separate single request, not an automatic rollback.

## Abort and recovery
- At any point, Ctrl-C stops the tool. A connect already sent is logged as UNKNOWN; run `status` to see the truth.
- If the car loses its customer Wi-Fi, the management hotspot stays up, because it is a different radio. Re-run `switch` back to the original BSSID, or use the existing SmartCar PC app.
- If SSH to 192.168.1.11 is lost, SmartCarGuard recovers the hotspot on the car (3 consecutive failures, at most 3 attempts per 10 min). Wait about a minute and reconnect. Don't power-cycle mid-connect unless the car is unresponsive.

## What this test does and does not establish
- **Does:** that the WirelessJEV transport, confirmation gate and outcome semantics work with the real SmartCarCmd on the real BE200; the real switch latency; real RSSI and bitrate readings to calibrate the twin.
- **Does not:** fleet coordination, model quality on real traffic, motion, or anything about the other nine cars.
