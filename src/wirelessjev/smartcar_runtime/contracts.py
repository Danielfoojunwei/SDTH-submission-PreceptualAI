"""Strict real-survey boundaries, independent of training and simulator code."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import time

BSSID = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\Z")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
SECRET_KEYS = frozenset(("password", "psk", "passphrase", "private_key", "token", "ssh_password"))


def reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SECRET_KEYS:
                raise ValueError("Credentials cannot enter survey, map or journal files")
            reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_secrets(item)


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    return float(value)


def xyz(value, name="xyz_m"):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(name + " must contain three metre coordinates")
    return [finite(x, name) for x in value]


def bssid(value):
    if not isinstance(value, str) or not BSSID.fullmatch(value.lower()):
        raise ValueError("A literal BSSID is required")
    normalized = value.lower()
    first = int(normalized[:2], 16)
    if first & 1 or normalized == "00:00:00:00:00:00":
        raise ValueError("A unicast nonzero BSSID is required")
    return normalized


def _identifier(value, name):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(name + " must be an explicit short identifier")


def _frequency(value):
    frequency = finite(value, "frequency_mhz")
    if not (2400 <= frequency <= 2500 or 4900 <= frequency <= 5900 or 5925 <= frequency <= 7125):
        raise ValueError("An actual supported Wi-Fi carrier in MHz is required")
    return frequency


def _inside(point, dims):
    return all(0 <= x <= width for x, width in zip(point, dims))


def validate_site(value):
    reject_secrets(value)
    site = copy.deepcopy(value)
    if not isinstance(site, dict) or type(site.get("schema")) is not int or site["schema"] != 1:
        raise ValueError("Site schema 1 is required")
    _identifier(site.get("site_id"), "site_id")
    dims = xyz(site.get("dimensions_m"), "dimensions_m")
    if min(dims) <= 0 or max(dims) > 10000:
        raise ValueError("Positive finite site extents are required")
    site["dimensions_m"] = dims
    aps = site.get("aps")
    if not isinstance(aps, list) or not 1 <= len(aps) <= 32:
        raise ValueError("Declare between one and 32 actual BSSIDs")
    seen = set()
    for ap in aps:
        if not isinstance(ap, dict):
            raise ValueError("AP records must be objects")
        ap["bssid"] = bssid(ap.get("bssid"))
        if ap["bssid"] in seen:
            raise ValueError("Duplicate site BSSID")
        seen.add(ap["bssid"])
        if not isinstance(ap.get("ssid"), str) or not 1 <= len(ap["ssid"].encode()) <= 32:
            raise ValueError("Declare the actual enabled SSID")
        ap["xyz_m"] = xyz(ap.get("xyz_m"), "AP xyz_m")
        if not _inside(ap["xyz_m"], dims):
            raise ValueError("AP coordinates must use the declared floor-origin site frame")
        ap["frequency_mhz"] = _frequency(ap.get("frequency_mhz"))
        ap["tx_power_dbm"] = finite(ap.get("tx_power_dbm"), "tx_power_dbm")
        if not -30 <= ap["tx_power_dbm"] <= 60:
            raise ValueError("Declare a plausible AP transmit-power assumption")
    boxes = site.get("cad_boxes_m")
    if not isinstance(boxes, list) or len(boxes) > 4096:
        raise ValueError("Declare public CAD boxes, using [] when unavailable")
    for i, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 2:
            raise ValueError("CAD boxes require lower and upper XYZ corners")
        low, high = xyz(box[0]), xyz(box[1])
        if not _inside(low, dims) or not _inside(high, dims) or any(a >= b for a, b in zip(low, high)):
            raise ValueError("CAD boxes must be ordered and inside the site")
        boxes[i] = [low, high]
    receiver = site.get("receiver")
    if not isinstance(receiver, dict):
        raise ValueError("Declare receiver assumptions separately from observed RSSI")
    for key in ("noise_floor_dbm", "offset_db", "quantization_db"):
        receiver[key] = finite(receiver.get(key), "receiver " + key)
    if not -150 <= receiver["noise_floor_dbm"] <= -20 or not -40 <= receiver["offset_db"] <= 40:
        raise ValueError("Receiver reference assumptions are out of bounds")
    if not 0 <= receiver["quantization_db"] <= 20 or type(receiver.get("reference_gauge_assumed")) is not bool:
        raise ValueError("Explicit quantization and reference-gauge assumptions are required")
    return site


def site_sha256(site):
    return canonical_sha256(validate_site(site))


def validate_query(site, point):
    point = xyz(point)
    if not _inside(point, site["dimensions_m"]):
        raise ValueError("Query coordinates are outside the declared site")
    return point


def validate_record(site, value, now_ns=None):
    site = validate_site(site)
    reject_secrets(value)
    row = copy.deepcopy(value)
    if not isinstance(row, dict) or type(row.get("schema")) is not int or row["schema"] != 1:
        raise ValueError("Capture schema 1 is required")
    if row.get("site_id") != site["site_id"] or row.get("site_sha256") != site_sha256(site):
        raise ValueError("Capture belongs to another site configuration")
    if row.get("source") != "hardware" or row.get("interface") != "wlan1" or row.get("fresh_scan") is not True:
        raise ValueError("Only explicitly requested fresh client-radio hardware captures are admitted")
    _identifier(row.get("capture_id"), "capture_id")
    pose = row.get("pose")
    if not isinstance(pose, dict) or pose.get("source") not in ("operator", "external"):
        raise ValueError("Capture requires an explicit operator or external pose source")
    pose["xyz_m"] = validate_query(site, pose.get("xyz_m"))
    for key in ("started_unix_ns", "completed_unix_ns", "pc_request_started_unix_ns", "pc_received_unix_ns"):
        if type(row.get(key)) is not int or row[key] <= 0:
            raise ValueError("Capture requires positive integer timestamps and a desktop availability window")
    if row["completed_unix_ns"] < row["started_unix_ns"]:
        raise ValueError("Car-local capture timestamps are reversed")
    now_ns = time.time_ns() if now_ns is None else now_ns
    if row["pc_request_started_unix_ns"] > row["pc_received_unix_ns"] or row["pc_received_unix_ns"] > now_ns:
        raise ValueError("Capture is not causally available at the desktop query time")
    # Car and desktop wall clocks need not agree. Only same-clock ordering is checked.
    aps = row.get("aps")
    if not isinstance(aps, list) or not 1 <= len(aps) <= 512:
        raise ValueError("An actual nonempty bounded scan is required")
    configured = {ap["bssid"]: ap for ap in site["aps"]}
    seen = set()
    for ap in aps:
        if not isinstance(ap, dict):
            raise ValueError("Scan AP records must be objects")
        ap["bssid"] = bssid(ap.get("bssid"))
        if ap["bssid"] in seen:
            raise ValueError("Duplicate BSSID in one capture")
        seen.add(ap["bssid"])
        ap["frequency_mhz"] = _frequency(ap.get("frequency_mhz"))
        ap["signal_dbm"] = finite(ap.get("signal_dbm"), "signal_dbm")
        if not -150 <= ap["signal_dbm"] <= 0 or not isinstance(ap.get("ssid"), str):
            raise ValueError("Scan RSSI or SSID is malformed")
        expected = configured.get(ap["bssid"])
        if expected and (ap["ssid"] != expected["ssid"] or abs(ap["frequency_mhz"] - expected["frequency_mhz"]) > 0.01):
            raise ValueError("An observed configured BSSID changed SSID or carrier; update site metadata explicitly")
    if row.get("current_bssid") is not None:
        row["current_bssid"] = bssid(row["current_bssid"])
    return row


def records_sha256(records):
    return canonical_sha256(records)


def load_records(path, site):
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Survey exceeds the 64 MiB desktop intake limit")
    values, seen = [], set()
    now = time.time_ns()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = validate_record(site, json.loads(line), now_ns=now)
        if row["capture_id"] in seen:
            raise ValueError("A capture was replayed into the survey")
        seen.add(row["capture_id"])
        values.append(row)
    if not values:
        raise ValueError("Collect genuine position-tagged survey records first")
    if [r["pc_received_unix_ns"] for r in values] != sorted(r["pc_received_unix_ns"] for r in values):
        raise ValueError("Survey records must preserve desktop delivery order")
    return values
