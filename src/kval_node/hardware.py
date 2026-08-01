from __future__ import annotations

import importlib.util
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .commands import CommandError, run_checked


REQUIRED_BINARIES = ("ip", "nmcli", "mmcli", "ping", "iw", "ethtool", "sqlite3", "tegrastats")
REQUIRED_DEBIAN_PACKAGES = (
    "nvidia-l4t-core",
    "nvidia-jetpack",
    "network-manager",
    "modemmanager",
    "iproute2",
    "iputils-ping",
    "iw",
    "ethtool",
    "sqlite3",
    "python3-cryptography",
)


@dataclass(frozen=True)
class PlatformReport:
    model: str
    l4t_version: str
    jetpack_version: str
    binaries: dict[str, str]
    packages: dict[str, str]
    python_cryptography: bool
    tensorrt_python: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_device_model() -> str:
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        raise RuntimeError("device tree model is unavailable; this is not a supported Jetson boot")
    return model_path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="strict").strip()


def _package_version(package: str) -> str:
    return run_checked(
        ["dpkg-query", "-W", "-f=${Status}\t${Version}", package],
        timeout_seconds=10.0,
    ).strip()


def verify_jetson_platform() -> PlatformReport:
    model = _read_device_model()
    if "Jetson Orin Nano" not in model:
        raise RuntimeError(f"unsupported device: {model}; Jetson Orin Nano is required")

    binaries: dict[str, str] = {}
    for binary in REQUIRED_BINARIES:
        resolved = shutil.which(binary)
        if resolved is None:
            raise RuntimeError(f"required executable is missing: {binary}")
        binaries[binary] = resolved

    packages: dict[str, str] = {}
    for package in REQUIRED_DEBIAN_PACKAGES:
        try:
            status = _package_version(package)
        except CommandError as exc:
            raise RuntimeError(f"required Debian package is not installed: {package}") from exc
        if not status.startswith("install ok installed\t"):
            raise RuntimeError(f"required Debian package is not installed: {package}")
        packages[package] = status.split("\t", 1)[1]

    l4t_version = packages["nvidia-l4t-core"]
    release = re.match(r"(?P<major>\d+)\.(?P<minor>\d+)", l4t_version)
    if release is None or (release.group("major"), release.group("minor")) != ("39", "2"):
        raise RuntimeError(f"Jetson Linux 39.2 is required; found {l4t_version}")

    cryptography_available = importlib.util.find_spec("cryptography") is not None
    tensorrt_available = importlib.util.find_spec("tensorrt") is not None
    if not cryptography_available:
        raise RuntimeError("Python cryptography package is unavailable")
    if not tensorrt_available:
        raise RuntimeError("TensorRT Python bindings are unavailable from JetPack")

    return PlatformReport(
        model=model,
        l4t_version=l4t_version,
        jetpack_version=packages["nvidia-jetpack"],
        binaries=binaries,
        packages=packages,
        python_cryptography=True,
        tensorrt_python=True,
    )

