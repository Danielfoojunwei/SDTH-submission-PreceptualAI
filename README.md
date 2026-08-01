# Kval SPECTRAVAULT Node

Kval is the Jetson Orin Nano edge node for SPECTRAVAULT. It observes real Linux
network interfaces, measures real reachability, ranks only operator-approved
connectivity profiles, applies deterministic safety rules, and can ask
NetworkManager to activate an approved connection UUID.

The supported access labels are WiFi, 3G, 4G, 5G, NTN, and Ethernet. Kval does
not generate radio parameters, inspect packet contents, control a vehicle, or
implement electronic-warfare functions.

## Honest implementation status

Implemented production paths:

- Jetson Orin Nano and JetPack 7.2 / Jetson Linux 39.2 hardware gate.
- Live `ip`, NetworkManager, WiFi, ModemManager, and ICMP telemetry.
- Multi-objective profile optimization from measured latency, loss, jitter,
  throughput, signal, monetary cost, energy cost, and switching cost.
- Ed25519-signed profile catalogue with expiry checks.
- Safety Shield with freshness, confidence, dwell-time, and switch-rate limits.
- Real NetworkManager activation followed by UUID verification.
- 3GPP DNN, S-NSSAI, RAT, Release 17/18 and NR-NTN policy vocabulary.
- Fail-closed ETSI RRS, IEEE 1900.5, and ANSI/VITA 49.2 capability contract for
  standards-based SDR integrations.
- Root-only radio-agent service with signed evidence, conformance-report hash
  verification, exact profile/UUID pinning, and fixed no-shell driver commands.
- Offline-signed per-node identity and learning-cohort manifest, with every
  decision bound to a fresh runtime context digest.
- SQLite WAL audit ledger with a SHA-256 hash chain.
- A systemd service, dependency installer, dependency cache, provisioning tool,
  and hardware-in-loop validation script.

Not claimed yet:

- Jetson latency numbers. They must be produced by the included HIL script on
  the physical board.
- A trained GNN-DQN. No public author implementation or real Kval field trace
  corpus is currently present. The runtime will not substitute random weights,
  a synthetic model, or a pretend TensorRT result.
- FHE on the Jetson. Private fleet coordination remains a separate service.

There are no simulated telemetry providers or permissive production fallbacks.
Missing hardware, unsigned policy, missing probes, or failed activation cause a
clear rejection.

## Hardware baseline

- NVIDIA Jetson Orin Nano 8 GB developer kit for prototyping.
- JetPack 7.2 / Jetson Linux 39.2.
- NVMe storage recommended for logs and future model artifacts.
- One or more NetworkManager-controlled adapters. Cellular profiles require a
  ModemManager-compatible modem. NTN can be an approved Ethernet or modem
  connection; Kval treats the local connection UUID as the authority.

NVIDIA developer kits are development hardware, not production modules. A
production product must use a commercial Orin module and validated carrier.

## Install on the physical Jetson

```bash
git clone https://github.com/Danielfoojunwei/SDTH-submission-PreceptualAI.git
cd SDTH-submission-PreceptualAI
sudo ./scripts/install_jetson.sh
```

The installer refuses non-Jetson machines and unsupported Jetson Linux
releases. It installs dependencies from the signed Ubuntu/NVIDIA APT
repositories and verifies them before installing Kval.

## Provision real profiles

1. Create the actual NetworkManager connections and record their UUIDs:

```bash
nmcli -f NAME,UUID,TYPE,DEVICE connection show
```

2. Copy `config/catalog.template.json`, replace every placeholder with an
   actual connection UUID, interface, and reachable probe target.
3. Sign the catalogue on an operator/admin machine:

```bash
python3 scripts/provision_catalog.py \
  --catalog catalog.json \
  --private-key operator.private.pem \
  --public-key operator.public.pem \
  --generate-key \
  --output catalog.signed.json
```

4. Copy only `catalog.signed.json` and `operator.public.pem` to the node. Keep
   `operator.private.pem` off the node.
5. Copy and edit `config/node.example.json` as `/etc/kval/node.json`.
6. Create one node manifest from `config/node-manifest.template.json`, sign it
   offline with `scripts/provision_node_manifest.py`, and install the signed
   envelope as `/etc/kval/node-manifest.signed.json`.

## Validate before enabling switching

```bash
sudo ./scripts/run_hil_validation.sh
```

This verifies the board, installed packages, signed catalogue, live interfaces,
real probes, optimizer latency, and audit chain without changing connections.

To include a real NetworkManager activation test, set an explicit approved
profile and opt in:

```bash
sudo KVAL_HIL_APPLY_PROFILE=<approved-profile-id> \
  ./scripts/run_hil_validation.sh
```

Only after validation should `apply_network_changes` be set to `true` in the
node configuration. The process also requires `KVAL_ENABLE_APPLY=YES` in the
systemd environment, preventing configuration drift from silently enabling
network changes.

## Development verification

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests scripts
```

These tests exercise the pure optimizer, real Ed25519 implementation, catalogue
validation, and real SQLite audit chain. They do not claim to be Jetson or radio
hardware tests.

See [Jetson validation](docs/JETSON_VALIDATION.md) and
[architecture](docs/ARCHITECTURE.md), plus the
[standards interoperability contract](docs/STANDARDS_INTEROPERABILITY.md) and
[dependency policy](docs/DEPENDENCIES.md).
The full learning and fleet evidence ladder is in
[the end-to-end proof plan](docs/END_TO_END_PROOF_PLAN.md).
