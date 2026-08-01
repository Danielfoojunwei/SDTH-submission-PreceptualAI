# Dependency policy

Kval deliberately avoids a generic PyPI runtime stack on Jetson. Runtime
dependencies come from the signed Ubuntu and NVIDIA repositories associated
with JetPack 7.2 / Jetson Linux 39.2:

- `nvidia-jetpack` for CUDA, TensorRT, and the Jetson platform stack.
- NetworkManager and ModemManager for real interface and modem control.
- `iproute2`, `iputils-ping`, `iw`, and `ethtool` for live measurements.
- SQLite for the durable audit chain.
- Ubuntu's `python3-cryptography` for Ed25519.
- Python 3.12 and standard-library runtime code.

`scripts/install_jetson.sh` downloads, installs, and verifies these packages on
the physical aarch64 target. `scripts/cache_dependencies.sh` downloads the
resolved `.deb` files into an offline cache and creates `MANIFEST.sha256`.

The repository cannot honestly include the final resolved aarch64 package set
without running APT against the target JetPack image: repository revisions,
carrier BSPs, and architecture determine the selected packages. The physical
HIL gate records exact versions from `dpkg-query` in every result.

The Kval Python wheel has no PyPI runtime dependency resolution. Installation
uses `--no-deps --no-build-isolation`; a missing system dependency fails the
preflight.

