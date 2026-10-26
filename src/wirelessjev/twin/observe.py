"""Per-car, per-BSS interference state.

Encoding from Zhang, Lam, Shen, Wang & Li, "Dynamic spectrum access for
Internet-of-Things with hierarchical federated deep reinforcement learning"
(Ad Hoc Networks 149, 2023):

    2  = associated, no interference
    1  = associated, shares the channel with other fleet traffic
    0  = no access on this BSS
   -1  = associated on a channel that collides with protected (external/PU)
         traffic
"""
from __future__ import annotations

from .radio import Scene


def interference_state(scene: Scene, assoc: dict[str, str],
                       channels: dict[str, int]) -> dict[str, dict[str, int]]:
    cars_on_channel: dict[tuple[str, int], int] = {}
    for ap_id in assoc.values():
        ap = scene.ap(ap_id)
        key = (ap.band, channels[ap_id])
        cars_on_channel[key] = cars_on_channel.get(key, 0) + 1

    state: dict[str, dict[str, int]] = {}
    for car in scene.cars:
        row = {ap.ap_id: 0 for ap in scene.aps}
        ap_id = assoc.get(car.car_id)
        if ap_id is not None:
            ap = scene.ap(ap_id)
            key = (ap.band, channels[ap_id])
            if scene.external_occupancy.get(key, 0.0) > 0.0:
                row[ap_id] = -1
            elif cars_on_channel.get(key, 0) > 1:
                row[ap_id] = 1
            else:
                row[ap_id] = 2
        state[car.car_id] = row
    return state


def channel_status_term(code: int) -> float:
    """B term of Li, Shui, Lam, Shen & Wang (ICAIIC 2025):
    +1 idle/clean, -0.5 occupied by other fleet users, -1 busy (protected)."""
    return {2: 1.0, 1: -0.5, -1: -1.0, 0: 0.0}[code]
