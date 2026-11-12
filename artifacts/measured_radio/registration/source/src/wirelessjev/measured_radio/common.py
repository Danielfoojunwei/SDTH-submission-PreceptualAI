"""Finite study registration and immutable evidence for measured RSSI transfer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/measured_radio'
SOURCE = Path('/workspace/upstream/scene-radio/rssi/downloads/20051265-UVIndoorLoc-RTT&RSSI.zip')
SOURCE_METADATA = Path('/workspace/upstream/scene-radio/rssi/zenodo-20051265.json')
SOURCE_SHA256 = '0ff469b8bd9d58a7d37c77423fc40450916bf26bbe208010bfc2bdf075ccf9f5'
METADATA_SHA256 = '36539227b9ede5413b974de1e7fb91be1e17df4b841936df208706018414c931'
NAMESPACE = 'uv-measured-v1'
BUDGETS = (4, 8, 16)
PHONES = ('GP10', 'S24U')
SCENARIOS = ('S01', 'S02', 'S03')
SEEDS = (0, 1, 2)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value))


def read(path):
    return json.loads(Path(path).read_bytes())
