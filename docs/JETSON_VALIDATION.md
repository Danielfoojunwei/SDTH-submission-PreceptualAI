# Physical Jetson validation

No Jetson performance result is recorded until this procedure runs on the
target device and its JSON evidence is retained.

## Required device

- Jetson Orin Nano device-tree model.
- Jetson Linux 39.2 from JetPack 7.2.
- Installed `nvidia-jetpack`, TensorRT, NetworkManager, ModemManager, `ip`,
  `ping`, `iw`, `ethtool`, SQLite, and Python cryptography packages.
- Signed non-expired catalogue using real NetworkManager UUIDs.
- A reachable, operator-controlled probe target for each active profile.

## Gates

| Gate | Passing condition |
|---|---|
| Platform | Device tree identifies Jetson Orin Nano |
| BSP | `nvidia-l4t-core` is release 39.2 |
| Dependencies | Every required binary and package is present |
| Policy | Ed25519 signature and expiry validate |
| Telemetry | At least one approved profile has live link and probe evidence |
| Local latency | 100 repeated decisions; P95 at or below 100 ms |
| Safety | Unsigned, expired, stale, unavailable, and rate-limited choices rejected |
| Advisory continuity | Decision and audit record produced with upstream services absent |
| Activation | Optional explicit HIL profile becomes the active NM UUID |
| Audit | Full SQLite hash chain verifies |

## Optional capability gates

Each is skipped when the capability is not configured, and never faked.

| Gate | Passing condition |
|---|---|
| Power governance | `power-plan` reports `available: true` with a real `tegrastats` reading and a target `nvpmodel` id from the signed allowlist |
| Spectrum sensing | `spectrum-scan` reports `available: true`, at least one band is not `INSUFFICIENT_EVIDENCE`, and **every** returned observation is `passive: true` |
| Secure transport | `transport-preflight` reports `available: true`; with `KVAL_HIL_REQUIRE_TRANSPORT=YES` it must also report `verified: true` |
| Capability lease | A signed lease redeems once and is refused on replay |

Spectrum sensitivity, false-alarm rate, bearing accuracy, and any power or
energy-per-bit figure are **not** produced by this script. They require the real
sensor adapter and instrumented measurement, and no such number is claimed
anywhere in this repository.

The transport gate additionally needs adversarial runs before it can be called
proven: repeat `transport-preflight` during tunnel loss, during reconnection, and
against a rogue BSSID, confirming each fails closed with the expected reason, and
verify with an independent packet capture that no plaintext underlay egress
occurs while the tunnel is down.

The initial PR can prove software correctness on generic Linux. It cannot prove
the platform, telemetry, latency, activation, spectrum, power, or transport gates
without the physical Jetson and connected adapters.

