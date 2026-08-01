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

The initial PR can prove software correctness on generic Linux. It cannot prove
the platform, telemetry, latency, or activation gates without the physical
Jetson and connected adapters.

