#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "install_jetson.sh must run as root" >&2
  exit 1
fi

if [[ ! -r /proc/device-tree/model ]]; then
  echo "Jetson device tree is unavailable" >&2
  exit 1
fi

device_model=$(tr -d '\0' </proc/device-tree/model)
if [[ ${device_model} != *"Jetson Orin Nano"* ]]; then
  echo "Unsupported device: ${device_model}" >&2
  exit 1
fi

l4t_version=$(dpkg-query -W -f='${Version}' nvidia-l4t-core 2>/dev/null || true)
if [[ ${l4t_version} != 39.2* ]]; then
  echo "Jetson Linux 39.2 from JetPack 7.2 is required; found ${l4t_version:-none}" >&2
  exit 1
fi

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

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes --no-install-recommends "${packages[@]}"

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
install -d -m 0750 /etc/kval /var/lib/kval /run/kval /opt/kval
install -d -m 0700 /etc/kval/radio-evidence /run/kval-radio
python3 -m venv --system-site-packages /opt/kval/venv
/opt/kval/venv/bin/python -m pip install --no-deps --no-build-isolation "${project_root}"

if [[ ! -e /etc/kval/node.json ]]; then
  install -m 0640 "${project_root}/config/node.example.json" /etc/kval/node.json
fi
if [[ ! -e /etc/kval/node-manifest.template.json ]]; then
  install -m 0600 "${project_root}/config/node-manifest.template.json" /etc/kval/node-manifest.template.json
fi
install -m 0644 "${project_root}/systemd/kval-node.service" /etc/systemd/system/kval-node.service
if [[ ! -e /etc/kval/radio-agent.json ]]; then
  install -m 0600 "${project_root}/config/radio-agent.example.json" /etc/kval/radio-agent.json
fi
install -m 0644 "${project_root}/systemd/kval-radio-agent.service" /etc/systemd/system/kval-radio-agent.service
systemctl daemon-reload
systemctl enable kval-node.service
systemctl enable kval-radio-agent.service

/opt/kval/venv/bin/python -c 'import cryptography, tensorrt, kval_node; print(kval_node.__version__)'
/opt/kval/venv/bin/kval-node preflight

echo "Kval installed. Provision /etc/kval/catalog.signed.json and operator.public.pem before starting the service."
