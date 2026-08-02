# Kval SPECTRAVAULT Node

Kval is the Jetson Orin Nano edge node for SPECTRAVAULT. It observes real Linux
network interfaces, measures real reachability, ranks only operator-approved
connectivity profiles, applies deterministic safety rules, and can ask
NetworkManager to activate an approved connection UUID. It also senses the radio
spectrum **passively** and governs its own power and secure-transport boundary
inside operator-signed limits.

The supported access labels are WiFi, 3G, 4G, 5G, NTN, and Ethernet. Kval does
not generate radio parameters, transmit, jam, inspect packet contents, control a
vehicle, or implement electronic-attack functions.

## Honest implementation status

Implemented production paths:

- Jetson Orin Nano and JetPack 7.2 / Jetson Linux 39.2 hardware gate.
- Live `ip`, NetworkManager, WiFi, ModemManager, and ICMP telemetry.
- Multi-objective profile optimization from measured latency, loss, jitter,
  throughput, signal, monetary cost, energy cost, and switching cost.
- Ed25519-signed profile catalogue with expiry checks, and a generic signed
  policy-document envelope for spectrum, power, and transport policies.
- Safety Shield with freshness, confidence, dwell-time, and switch-rate limits.
- Real NetworkManager activation followed by UUID verification.
- 3GPP DNN, S-NSSAI, RAT, Release 17/18 and NR-NTN policy vocabulary.
- Fail-closed ETSI RRS, IEEE 1900.5, and ANSI/VITA 49.2 capability contract for
  standards-based SDR integrations.
- **Passive spectrum-sensing fabric**: signed band allowlist, root-owned
  receive-only sensor-adapter contract, and a deterministic analyzer producing
  band availability, jamming *suspicion*, and emitter presence.
- **Capability leases**: operator-signed, node-bound, single-use, expiring
  authorization for each bounded action, with rollback and post-condition.
- **Power governance**: signed thermal/battery/watt/duty policy, a real
  `tegrastats` parser, and a governor with thermal hysteresis.
- **Connectivity recovery**: link-health state machine with exponential backoff,
  failover, and a flap ceiling that escalates instead of thrashing the radio.
- **Multimodal state fusion** with per-modality provenance, quality, and
  missingness, failing closed when a mandatory modality is stale or absent.
- **Learning guardrails**: signed model-manifest contract, OOD/drift abstention,
  CUSUM change-point detection, and a causal-attribution gate.
- **Secure-transport verification**: WireGuard peer/AllowedIPs/handshake checks,
  IPv4 **and** IPv6 default-route confinement, pinned nftables ruleset hash with
  default-drop output, and WPA3-SAE / WPA3-Enterprise-192 with PMF required.
- SQLite WAL audit ledger with a SHA-256 hash chain.
- A systemd service, dependency installer, dependency cache, provisioning tools,
  and hardware-in-loop validation script.

Not claimed yet:

- Jetson latency numbers. They must be produced by the included HIL script on
  the physical board.
- A trained GNN-DQN or any predictive model. No public author implementation or
  real Kval field trace corpus is currently present. The runtime will not
  substitute random weights, a synthetic model, or a pretend TensorRT result;
  the learned path abstains until a signed model manifest verifies.
- Spectrum sensitivity, false-alarm rate, or bearing accuracy. These belong to
  the sensor adapter and must be measured with real hardware.
- Transport preflight passes. They require real keys, UUIDs, BSSIDs, and a pinned
  ruleset on the physical node.
- Power, energy-per-bit, or battery-life figures for any node or sub-node.
- FHE on the Jetson. Private fleet coordination remains a separate service.
- A separately packaged reduced wheel for low-compute sub-nodes. The supported
  module boundary is documented; the packaging work is not done.

There are no simulated telemetry providers or permissive production fallbacks.
Missing hardware, unsigned policy, missing probes, an absent sensor adapter, or
failed activation cause a clear rejection.

## Scope boundary

Kval is defensive. It has no transmit path, no jamming, no spoofing, no covert
signalling, and no arbitrary RF control — and no capability lease can authorise
one. Spectrum sensing is receive-only by construction: an observation that is not
marked passive is refused. The offensive threats Kval is designed to *withstand*
are catalogued in
[the secure-transport and EW threat model](docs/SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md).

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
git clone https://github.com/Danielfoojunwei/Kval-.git
cd Kval-
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

## Provision the optional capability policies

Spectrum sensing, power governance, secure transport, and capability leases are
each optional and off unless configured. Sign them on the operator machine:

```bash
python3 scripts/provision_policy.py spectrum \
  --input config/spectrum-policy.template.json \
  --private-key operator.private.pem --public-key operator.public.pem \
  --output spectrum-policy.signed.json

python3 scripts/provision_policy.py power    --input config/power-policy.template.json    ...
python3 scripts/provision_policy.py transport --input config/transport-policy.template.json ...
python3 scripts/provision_policy.py lease    --input lease.body.json                       ...
```

Then point `/etc/kval/node.json` at them — see `config/node.full-example.json`.
Each capability fails closed when its policy is absent, expired, unsigned, or
unverifiable:

```bash
kval-node spectrum-scan       --config /etc/kval/node.json
kval-node power-plan          --config /etc/kval/node.json
kval-node transport-preflight --config /etc/kval/node.json
kval-node verify-lease        --config /etc/kval/node.json --lease lease.signed.json \
                              --action set_power_mode --redeem
```

These commands exit `0` only on a genuine pass. A capability that is not
configured or whose evidence is unavailable exits `3`; one that is configured
but reports `verified: false` or `valid: false` exits `4`.

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
and policy-document validation, capability leases and nonce replay refusal, the
spectrum analyzer, the power governor, the recovery state machine, multimodal
fusion, the learning guardrails, the transport-security parsers/verifiers, and
the real SQLite audit chain. They do not claim to be Jetson or radio hardware
tests.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Spectrum-sensing fabric](docs/SPECTRUM_SENSING_FABRIC.md)
- [Low-compute sub-nodes](docs/LOW_COMPUTE_SUBNODE.md)
- [Learning in autonomous networks — problems first](docs/AUTONOMOUS_NETWORK_LEARNING.md)
- [Secure transport and the EW threat model](docs/SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md)
- [Standards interoperability contract](docs/STANDARDS_INTEROPERABILITY.md)
- [Jetson validation](docs/JETSON_VALIDATION.md)
- [Dependency policy](docs/DEPENDENCIES.md)
