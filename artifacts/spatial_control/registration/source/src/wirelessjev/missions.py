"""Missions: a natural-language intent (what WirelessJEV sees) paired with a
hidden weight vector (what the oracle scores with).

The decision model never sees the weights. The intent-flip test checks that
changing only the text changes the choice in the direction the weights imply.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mission:
    mission_id: str
    intent: str
    weights: dict = field(default_factory=dict)
    focus_cars: tuple[str, ...] = ()


# Weight keys (see oracle.mission_utility):
#   B channel status, U demand utilisation, V satisfaction, W interference,
#   fair Jain fairness, ho handover/retune cost, focus focus-car service,
#   viol critical-floor violations, prot protected-channel hits.
_BASE = dict(B=0.3, U=1.0, V=1.0, W=0.3, fair=0.3, ho=0.5, focus=0.0,
             viol=2.0, prot=1.0)


def _w(**over) -> dict:
    out = dict(_BASE)
    out.update(over)
    return out


MISSIONS = {
    "protect_video": Mission(
        "protect_video",
        "Car-03 is streaming the live inspection video. Keep car-03's video "
        "uplink at full rate even if other cars lose throughput.",
        _w(focus=4.0, U=0.3, fair=0.0, ho=0.2),
        ("car-03",),
    ),
    "convoy_continuity": Mission(
        "convoy_continuity",
        "The fleet is in a tight convoy manoeuvre. Avoid reconnections and "
        "channel changes; uninterrupted links matter more than speed.",
        _w(ho=4.0, U=0.4, V=0.6),
    ),
    "bulk_upload": Mission(
        "bulk_upload",
        "All cars are uploading survey maps. Maximise the total data the "
        "fleet delivers; brief reconnections are acceptable.",
        _w(U=3.0, V=0.5, ho=0.1, fair=0.0),
    ),
    "fair_share": Mission(
        "fair_share",
        "Every car runs the same patrol task. No car may be starved; share "
        "capacity evenly across the fleet.",
        _w(fair=3.0, V=1.5, U=0.3),
    ),
    "spectrum_etiquette": Mission(
        "spectrum_etiquette",
        "A neighbouring lab's protected equipment uses some channels. Stay "
        "off protected channels and keep co-channel interference low.",
        _w(prot=5.0, W=2.0, B=1.5, U=0.5),
    ),
}

# Pairs whose optimal choices should usually differ (intent-flip test).
INTENT_FLIP_PAIRS = (
    ("convoy_continuity", "bulk_upload"),
    ("protect_video", "fair_share"),
    ("spectrum_etiquette", "bulk_upload"),
)


# Warehouse (AGV/AMR) missions. amr-017 streams teleoperation video.
WAREHOUSE_MISSIONS = {
    "teleop_assist": Mission(
        "teleop_assist",
        "Robot amr-017 is under remote teleoperation. Keep amr-017's video "
        "uplink at full rate even if other robots lose throughput.",
        _w(focus=4.0, U=0.3, fair=0.0, ho=0.2), ("amr-017",)),
    "peak_picking": Mission(
        "peak_picking",
        "Peak picking wave: robots are moving fast between aisles. Keep every "
        "safety link up and avoid reconnections during the wave.",
        _w(ho=4.0, viol=4.0, U=0.4, V=0.6)),
    "map_rollout": Mission(
        "map_rollout",
        "A new site map is being pushed to all robots. Maximise the total data "
        "the fleet delivers; brief reconnections are acceptable.",
        _w(U=3.0, V=0.5, ho=0.1, fair=0.0)),
    "shift_fairness": Mission(
        "shift_fairness",
        "Every robot has the same task load this shift. No robot may be "
        "starved; share capacity evenly across the fleet.",
        _w(fair=3.0, V=1.5, U=0.3)),
    "scanner_coexistence": Mission(
        "scanner_coexistence",
        "Handheld barcode scanners share some channels. Keep robots off "
        "scanner-occupied channels and keep co-channel interference low.",
        _w(prot=5.0, W=2.0, B=1.5, U=0.5)),
}

WAREHOUSE_FLIP_PAIRS = (
    ("peak_picking", "map_rollout"),
    ("teleop_assist", "shift_fairness"),
    ("scanner_coexistence", "map_rollout"),
)
