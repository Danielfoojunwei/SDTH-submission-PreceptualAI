#!/usr/bin/env python3
"""Generate a Kval profile catalogue from this host's real connections.

The shipped ``config/catalog.template.json`` carries ``REPLACE_*`` markers
because NetworkManager UUIDs, interfaces, and probe targets are unique to each
node and cannot be known in advance. This tool removes the manual editing step:
it reads the machine's actual NetworkManager connections via ``nmcli``, maps
each to its real access type and interface, and writes a catalogue containing
only genuine values.

    sudo kval-generate-catalog --probe-target 192.0.2.10 --output catalog.json
    python3 scripts/provision_catalog.py --catalog catalog.json ...

Nothing is invented. A connection whose type Kval does not support, or that has
no bound interface, is skipped and reported. If no usable connection exists the
tool exits non-zero rather than emitting a placeholder catalogue.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kval_node.commands import CommandError, run_command  # noqa: E402

#: nmcli connection type -> Kval access label. Cellular connections are refined
#: further using the modem's live access technology.
_TYPE_MAP = {
    "802-11-wireless": "wifi",
    "wifi": "wifi",
    "802-3-ethernet": "ethernet",
    "ethernet": "ethernet",
}

#: ModemManager access-technology token -> Kval access label.
_MODEM_MAP = {
    "gsm": "3g", "umts": "3g", "hsdpa": "3g", "hsupa": "3g", "hspa": "3g", "hspa-plus": "3g",
    "lte": "4g", "lte-cat-m": "4g", "lte-nb-iot": "4g",
    "5gnr": "5g", "nr": "5g", "nr5g": "5g", "5g": "5g",
}

#: Default cost weights per access type. These are catalogue *policy* inputs an
#: operator is expected to review, not measurements; they are emitted as
#: conventional starting values and flagged in the tool's output.
_DEFAULT_COSTS = {
    "ethernet": (0.02, 0.10),
    "wifi": (0.10, 0.20),
    "3g": (0.40, 0.40),
    "4g": (0.45, 0.45),
    "5g": (0.50, 0.50),
}

_DEFAULT_DWELL = {"ethernet": 15, "wifi": 30, "3g": 60, "4g": 60, "5g": 60}
_DEFAULT_SWITCHES = {"ethernet": 10, "wifi": 8, "3g": 6, "4g": 6, "5g": 6}


def _nmcli_connections() -> list[dict[str, str]]:
    try:
        result = run_command(
            ["nmcli", "--terse", "--escape", "no", "--fields",
             "UUID,NAME,TYPE,DEVICE", "connection", "show"],
            timeout_seconds=15.0,
        )
    except CommandError as exc:
        # NetworkManager is the authority on connections; without it there is
        # nothing real to build a catalogue from.
        raise SystemExit(f"cannot read host connections: {exc}") from exc
    if result.returncode != 0:
        raise SystemExit(
            f"nmcli connection show failed: {result.stderr.strip() or 'no diagnostic output'}"
        )
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":")
        if len(parts) < 4:
            continue
        uuid, name, conn_type, device = parts[0], parts[1], parts[2], parts[3]
        rows.append({"uuid": uuid, "name": name, "type": conn_type, "device": device})
    return rows


def _connection_apn(uuid: str) -> str | None:
    try:
        result = run_command(
            ["nmcli", "--get-values", "gsm.apn", "connection", "show", "uuid", uuid],
            timeout_seconds=10.0,
        )
    except CommandError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _modem_access_technologies() -> dict[str, str]:
    """Map network interface -> live access technology token, via ModemManager.

    ModemManager is optional: a node with no cellular hardware simply has no
    modem profiles, so an absent ``mmcli`` yields an empty map rather than an
    error.
    """
    try:
        listing = run_command(["mmcli", "-L", "-J"], timeout_seconds=10.0)
    except CommandError:
        return {}
    if listing.returncode != 0:
        return {}
    try:
        modems = json.loads(listing.stdout).get("modem-list", [])
    except json.JSONDecodeError:
        return {}
    mapping: dict[str, str] = {}
    for path in modems:
        detail = run_command(["mmcli", "-m", str(path), "-J"], timeout_seconds=10.0)
        if detail.returncode != 0:
            continue
        try:
            generic = json.loads(detail.stdout).get("modem", {}).get("generic", {})
        except (json.JSONDecodeError, AttributeError):
            continue
        technologies = generic.get("access-technologies", [])
        token = (technologies[0] if isinstance(technologies, list) and technologies else technologies) or ""
        for port in generic.get("ports", []):
            if isinstance(port, (list, tuple)) and len(port) >= 2 and str(port[1]) == "net":
                mapping[str(port[0])] = str(token).lower()
    return mapping


def _access_type(row: dict[str, str], modem_tech: dict[str, str]) -> str | None:
    conn_type = row["type"].lower()
    if conn_type in {"gsm", "cdma", "modem"}:
        token = modem_tech.get(row["device"], "")
        return _MODEM_MAP.get(token)
    return _TYPE_MAP.get(conn_type)


def _standards_for(access_type: str, apn: str | None) -> dict[str, object]:
    if access_type in {"3g", "4g", "5g"}:
        rat = {"3g": "utran", "4g": "eutran", "5g": "nr"}[access_type]
        if not apn:
            raise ValueError("cellular connection has no configured APN/DNN")
        return {"required_identifiers": ["3gpp-ts-23.003"], "required_rat": rat, "dnn": apn}
    return {"required_identifiers": []}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Kval catalogue from real host connections")
    parser.add_argument(
        "--link-role",
        choices=["c2_cnpc", "payload", "bulk"],
        default="payload",
        help=(
            "role for the generated profiles. c2_cnpc marks a command link, "
            "which is ranked separately and never traded against payload."
        ),
    )
    parser.add_argument("--probe-target", required=True,
                        help="reachable operator-controlled host each profile probes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-version", default=None, help="defaults to a UTC timestamp")
    parser.add_argument("--valid-days", type=int, default=30)
    parser.add_argument("--service-classes", default="critical_data,telemetry,bulk_data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    if "REPLACE_" in args.probe_target:
        raise SystemExit("--probe-target must be a real reachable host")

    modem_tech = _modem_access_technologies()
    profiles: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    classes = [item.strip() for item in args.service_classes.split(",") if item.strip()]

    for row in _nmcli_connections():
        if not row["device"] or row["device"] == "--":
            skipped.append({**row, "reason": "connection is not bound to an interface"})
            continue
        access_type = _access_type(row, modem_tech)
        if access_type is None:
            skipped.append({**row, "reason": f"unsupported connection type {row['type']!r}"})
            continue
        apn = _connection_apn(row["uuid"]) if access_type in {"3g", "4g", "5g"} else None
        try:
            standards = _standards_for(access_type, apn)
        except ValueError as exc:
            skipped.append({**row, "reason": str(exc)})
            continue
        monetary, energy = _DEFAULT_COSTS[access_type]
        profiles.append(
            {
                "profile_id": f"{access_type}-{row['device']}",
                "access_type": access_type,
                "connection_uuid": row["uuid"],
                "interface": row["device"],
                "probe_target": args.probe_target,
                "monetary_cost": monetary,
                "energy_cost": energy,
                "minimum_dwell_seconds": _DEFAULT_DWELL[access_type],
                "maximum_switches_per_hour": _DEFAULT_SWITCHES[access_type],
                "allowed_service_classes": classes,
                "standards": standards,
                # Emitted explicitly rather than left to the schema default, so
                # an operator can see the field and change it. Without this in
                # the generated catalogue the command-link role was undeclarable
                # by any shipped tool, which made role segregation unusable in
                # practice however correct the ranking code was.
                "link_role": args.link_role,
            }
        )

    if not profiles:
        raise SystemExit(
            "no usable NetworkManager connection was found; "
            f"skipped {len(skipped)} candidate(s): {json.dumps(skipped)}"
        )

    now = datetime.now(timezone.utc)
    payload = {
        "catalog_version": args.catalog_version or f"host-{now.strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=args.valid_days)).isoformat(),
        "profiles": profiles,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profiles": [p["profile_id"] for p in profiles],
                "skipped": skipped,
                "review_before_signing": [
                    "monetary_cost and energy_cost are conventional defaults, not measurements",
                    "allowed_service_classes applies the same set to every profile",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
