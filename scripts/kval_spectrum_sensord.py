#!/usr/bin/env python3
"""Run the receive-only spectrum sensor daemon from a source checkout.

Installed nodes get the ``kval-spectrum-sensord`` console script instead; this
wrapper exists so the daemon can be run before installation.

    sudo python3 scripts/kval_spectrum_sensord.py \
        --socket /run/kval-spectrum/sensor.sock \
        --policy /etc/kval/spectrum-policy.signed.json \
        --public-key /etc/kval/operator.public.pem \
        --interface wlan0
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kval_node.sensor_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
