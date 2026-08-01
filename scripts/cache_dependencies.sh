#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "cache_dependencies.sh must run as root on the target Jetson" >&2
  exit 1
fi

if [[ ! -r /proc/device-tree/model ]] || [[ $(tr -d '\0' </proc/device-tree/model) != *"Jetson Orin Nano"* ]]; then
  echo "A Jetson Orin Nano is required so APT resolves aarch64 JetPack packages" >&2
  exit 1
fi

output=${1:-/var/lib/kval/dependency-cache}
install -d -m 0750 "${output}" "${output}/partial"

packages=(
  nvidia-jetpack
  network-manager
  modemmanager
  iproute2
  iputils-ping
  iw
  ethtool
  sqlite3
  python3
  python3-venv
  python3-pip
  python3-setuptools
  python3-cryptography
  jq
)

apt-get update
apt-get --yes --download-only --reinstall \
  -o "Dir::Cache::archives=${output}" \
  install "${packages[@]}"

find "${output}" -maxdepth 1 -type f -name '*.deb' -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${output}/MANIFEST.sha256"

echo "Dependency cache written to ${output}; verify with sha256sum -c MANIFEST.sha256"

