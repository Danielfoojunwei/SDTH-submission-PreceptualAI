# WirelessJEV for one real SmartCar

This bundle starts with measurements from your own car and site. It collects
position-labelled Wi-Fi observations, fits a numerical RadioMap, compares an
experimental RadioJEPA forecast, and produces a typed association proposal.
Changing the client connection is a separate, interactive, one-request action.
It does not drive the car, discover its position automatically, or authorize
autonomous radio control.

**RadioMap is the default.** The supplied RadioJEPA weights come from a simulated
RF experiment, not a physical-car qualification or a SOTA result. Missing
observations and unsupported radios remain missing; they are not filled with
invented measurements or service guarantees.

## What to copy

Extract `smartcar-wirelessJEV.zip` on both machines. On the car, use
`~/smartcar-wirelessJEV`; on your computer, use any writable directory. Keep
`PC`, `PI`, `src`, `models`, the site template and manifests together.
The archive contains no SSH passwords, Wi-Fi credentials, private host keys or
private SmartCar vendor archive. The guide is included at
`docs/SMARTCAR_DEPLOYMENT.md`.

After copying the archive to the car's home directory:

```bash
cd "$HOME"
python3 -m zipfile -e smartcar-wirelessJEV.zip "$HOME"
cd ~/smartcar-wirelessJEV
python3 verify_bundle.py
printf '%s\n' '{"action":"doctor"}' | python3 PI/wj_car_agent.py
```

The Pi collector uses Python's standard library. It queries the actual Linux
radio tooling already installed on the car; it does not install a driver,
create a hotspot, change a NetworkManager profile or start the motors.
The optional local doctor above queries actual hardware and reports missing
prerequisites. The agent accepts JSON on stdin; the human-facing command-line
interface is `PC/run.py`. The computer runs fitting and optional neural inference.

## Prepare the computer before joining the hotspot

Use a Linux/macOS computer, Python 3.12 and a separate virtual environment.
The launcher uses POSIX process/file primitives and does not support native
Windows; a configured Linux/WSL environment needs its own working network route.

```bash
python3 -m zipfile -e smartcar-wirelessJEV.zip .
cd smartcar-wirelessJEV
python3 verify_bundle.py
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r PC/requirements-map.txt
python PC/run.py --help
```

The map path needs NumPy/SciPy. The optional shadow path additionally needs
Torch from `PC/requirements-radiojepa.txt`; install it while internet is
available. Sionna, Hugging Face access, d3-lite and a GPU are not required for
this workflow. Do not install the research repository's full dependency set
on the Pi.

Connect the computer to the car's management network. Provision the car's
actual SSH host key through a trusted route and configure ordinary SSH access.
The example below uses the documented management address; change it when
your approved setup differs.

```bash
export CAR_HOST=192.168.1.11
export CAR_KNOWN_HOSTS="$HOME/.ssh/smartcar_known_hosts"
python PC/run.py doctor --host "$CAR_HOST" --known-hosts "$CAR_KNOWN_HOSTS"
```

A successful software self-check is not a hardware trial. Read the doctor's
actual interface, bus/driver, command and permission findings before scanning.
`--host` accepts a hostname or IP; the SSH user defaults to `pi` and the port to
22. Use `--user` or `--port` when your approved installation differs.
Fresh scans need root/CAP_NET_ADMIN. For an ordinary user, the collector makes
one fixed noninteractive `sudo -n <resolved iw> dev wlan1 scan flush` attempt;
the operator must already permit that exact operation, or run the collector
under an approved root session. Copying the ZIP or running `doctor` grants no
privileges and does not configure sudoers or setcap. Tool presence is not proof
that a scan is authorized.

## Survey, fit and compare

Create a site file, edit its real site/AP geometry and declared radio metadata,
and validate it:

```bash
python PC/run.py init-site --out site.json
python PC/run.py validate-site --site site.json
```

Do not treat template coordinates, assumed transmit powers or example identities
as measurements. The generated template has no APs: enter the actual AP
coordinates, carriers, TX references and declared receiver metadata. Coordinates
are metres from the floor origin, within `[0,width] × [0,depth] × [0,height]`.
Move or place the parked car at a measured position and enter it yourself.
This is an operator-positioned survey, not SLAM or inferred navigation.

```bash
python PC/run.py capture --site site.json --survey captures.jsonl \
  --xyz 2 3 0.3 --host "$CAR_HOST" --known-hosts "$CAR_KNOWN_HOSTS" --active-scan
```

An active scan requests real radio work and may interrupt customer traffic.
Run it only when a scan is appropriate. Repeat at several separated, measured
positions, changing `--xyz` each time; reserve some positions for checking
predictions. A single scan is not an environment map. Without `--active-scan`,
capture is refused; this workflow admits only explicitly requested fresh scans.
The collector flushes the transient BSS cache as part of that active scan.

```bash
python PC/run.py fit-map --site site.json --survey captures.jsonl --out map.json
python PC/run.py predict --site site.json --map map.json --xyz 4 3 0.3
python PC/run.py capture --site site.json --survey captures.jsonl \
  --xyz 4 3 0.3 --host "$CAR_HOST" --known-hosts "$CAR_KNOWN_HOSTS" --active-scan
python PC/run.py fit-map --site site.json --survey captures.jsonl --out map-v2.json
python PC/run.py propose --site site.json --survey captures.jsonl \
  --map map-v2.json --xyz 4 3 0.3 --out proposal.json
```

Inspect the raw captures, map support and proposal reasons. A proposal never
sends a switch. The latest scan must be recent and within 0.5 m of the requested
position, and the map must be freshly refitted; otherwise it defers or has no
eligible predictions. The deployable proposal is a bounded same-SSID
RSSI rule, not the research mission-language/MPC controller, and returns no
invented success probability. The map must bind the exact current normalized
survey; a switch proposal requires both its mean-minus-SD score and the latest actual
same-SSID RSSI to improve by at least 6 dB, otherwise disagreement defers.
For an optional comparison using the actual
bundled checkpoints:

```bash
python -m pip install -r PC/requirements-radiojepa.txt
python PC/run.py shadow --site site.json --survey captures.jsonl \
  --xyz 4 3 0.3 --models models/manifest.json --out shadow.json
```

Shadow results remain experimental. They do not replace the default map or
establish packet-loss, latency, handover-risk or physical-transfer accuracy.
Real carriers outside the admitted 2.4–5.2 GHz source range remain unknown in
the neural shadow, even though the research assay separately evaluated 6 GHz.

## Optional supervised switch

The real switch path additionally requires the existing, trusted
`~/SmartCarCmd.py` installation from your SmartCar setup, its guards, appropriate
noninteractive sudo permissions and a separate working management radio.
This bundle does not contain or install that private vendor software.
If it is absent or the radio roles cannot be verified, stop at survey/proposal.

Read status, record the original connection, then use a separately supplied,
private Wi-Fi configuration:

```bash
python PC/run.py status --host "$CAR_HOST" --known-hosts "$CAR_KNOWN_HOSTS"
python PC/run.py switch --host "$CAR_HOST" --known-hosts "$CAR_KNOWN_HOSTS" \
  --wifi-json /private/path/WiFi.json --ssid "YOUR_SSID" --bssid aa:bb:cc:dd:ee:ff \
  --journal switch-events.jsonl
```

Replace the example target with a currently visible, approved BSSID. Re-type
the target when prompted. The journaled command path sends one request and
reconciles ambiguous outcomes through status; it does not automatically retry
a mutation. After an interruption or lost reply, read status before considering
any new action. Returning to the original BSSID is another separately confirmed
request, not an automatic rollback. Do not depend on the hotspot surviving a
driver, routing or power fault.

See `docs/SMARTCAR_DEPLOYMENT.md` for prerequisites, interpretation and recovery.
