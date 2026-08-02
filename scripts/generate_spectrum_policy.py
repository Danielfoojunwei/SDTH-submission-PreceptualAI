#!/usr/bin/env python3
"""Generate a spectrum policy from real regulatory band plans.

Band edges come from ``config/band-plans.json``, which records published
allocations (FCC Part 15, ETSI harmonised standards, ITU ISM designations) with
the source cited per band. Nothing here is invented frequency data.

Occupancy and noise-floor thresholds are *not* regulatory data — they are
operator policy. They can be supplied explicitly, or measured from this host's
own radio with ``--calibrate-from-host``, which takes a real survey and derives
each band's baseline from what the environment actually looks like right now
rather than from a guessed constant.

    python3 scripts/generate_spectrum_policy.py --region US \
        --output spectrum-policy.json
    python3 scripts/generate_spectrum_policy.py --region EU \
        --calibrate-from-host --interface wlan0 --output spectrum-policy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kval_node.sensor_adapter import IwSurveyBackend  # noqa: E402
from kval_node.spectrum import SpectrumBand, SpectrumPolicy  # noqa: E402

BAND_PLANS = PROJECT_ROOT / "config" / "band-plans.json"

#: Conservative starting thresholds used when the host is not calibrated. They
#: are operator policy, not measurements, and the tool says so in its output.
DEFAULT_OCCUPANCY_BASELINE = 20.0
DEFAULT_OCCUPANCY_ALERT = 75.0
DEFAULT_NOISE_FLOOR_DBM = -95.0

#: Headroom above the measured quiet-state noise floor before a rise is treated
#: as anomalous, applied when calibrating from the host.
CALIBRATION_NOISE_HEADROOM_DB = 3.0
CALIBRATION_OCCUPANCY_HEADROOM = 15.0


def load_band_plan(region: str) -> list[dict[str, object]]:
    data = json.loads(BAND_PLANS.read_text(encoding="utf-8"))
    regions = data["regions"]
    if region not in regions:
        raise SystemExit(f"unknown region {region!r}; available: {', '.join(sorted(regions))}")
    return regions[region]["bands"]


def calibrate(bands: list[dict[str, object]], interface: str) -> dict[str, dict[str, float]]:
    """Measure this host's current noise floor and occupancy per band."""
    backend = IwSurveyBackend(interface)
    if not backend.available():
        raise SystemExit(f"cannot calibrate: no usable wireless interface {interface!r}")
    spectrum_bands = [
        SpectrumBand.from_dict({**band, "occupancy_baseline_percent": 0.0,
                                "occupancy_alert_percent": 100.0,
                                "noise_floor_baseline_dbm": DEFAULT_NOISE_FLOOR_DBM})
        for band in bands
    ]
    measurements = backend.measure(spectrum_bands)
    calibrated: dict[str, dict[str, float]] = {}
    for band in spectrum_bands:
        measurement = measurements.get(band.band_id)
        if measurement is None or measurement.noise_floor_dbm is None or measurement.occupancy_percent is None:
            continue
        calibrated[band.band_id] = {
            "noise_floor_baseline_dbm": round(measurement.noise_floor_dbm, 2),
            "occupancy_baseline_percent": round(
                min(100.0, measurement.occupancy_percent + CALIBRATION_OCCUPANCY_HEADROOM), 2
            ),
        }
    return calibrated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a spectrum policy from real band plans")
    parser.add_argument("--region", required=True, help="regulatory region key from config/band-plans.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", default=None, help="defaults to spectrum-<region>-<timestamp>")
    parser.add_argument("--valid-days", type=int, default=30)
    parser.add_argument("--max-scans-per-hour", type=int, default=120)
    parser.add_argument("--min-scan-interval-seconds", type=float, default=5.0)
    parser.add_argument("--only", default=None, help="comma-separated band_ids to include")
    parser.add_argument("--calibrate-from-host", action="store_true",
                        help="measure this host's real noise floor and occupancy per band")
    parser.add_argument("--interface", default=None, help="wireless interface used for calibration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")

    bands = load_band_plan(args.region)
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        bands = [band for band in bands if band["band_id"] in wanted]
        missing = wanted - {str(band["band_id"]) for band in bands}
        if missing:
            raise SystemExit(f"region {args.region} has no band(s): {', '.join(sorted(missing))}")
    if not bands:
        raise SystemExit("no bands selected")

    calibrated: dict[str, dict[str, float]] = {}
    if args.calibrate_from_host:
        if not args.interface:
            raise SystemExit("--calibrate-from-host requires --interface")
        calibrated = calibrate(bands, args.interface)
        if not calibrated:
            raise SystemExit(
                "calibration produced no measurements; the interface may not cover these bands"
            )

    now = datetime.now(timezone.utc)
    policy_bands = []
    for band in bands:
        measured = calibrated.get(str(band["band_id"]), {})
        baseline = measured.get("occupancy_baseline_percent", DEFAULT_OCCUPANCY_BASELINE)
        noise = measured.get("noise_floor_baseline_dbm", DEFAULT_NOISE_FLOOR_DBM)
        policy_bands.append(
            {
                "band_id": band["band_id"],
                "label": band["label"],
                "freq_low_hz": band["freq_low_hz"],
                "freq_high_hz": band["freq_high_hz"],
                "max_resolution_bandwidth_hz": band["max_resolution_bandwidth_hz"],
                "occupancy_baseline_percent": baseline,
                "occupancy_alert_percent": max(baseline, DEFAULT_OCCUPANCY_ALERT),
                "noise_floor_baseline_dbm": noise,
                "comms_candidate": bool(band.get("comms_candidate", False)),
            }
        )

    payload = {
        "policy_id": args.policy_id or f"spectrum-{args.region.lower()}-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "region": args.region,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=args.valid_days)).isoformat(),
        "passive_only": True,
        "max_scans_per_hour": args.max_scans_per_hour,
        "min_scan_interval_seconds": args.min_scan_interval_seconds,
        "bands": policy_bands,
    }
    # Validate against the real parser before writing, so a policy that the node
    # would reject never reaches an operator's signing step.
    SpectrumPolicy.from_dict(payload)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(args.output),
                "region": args.region,
                "bands": [band["band_id"] for band in policy_bands],
                "band_edges": "published regulatory allocations (see config/band-plans.json _sources)",
                "thresholds": (
                    f"measured on this host via iw survey ({len(calibrated)} band(s))"
                    if calibrated
                    else "operator defaults, NOT measurements - review before signing"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
