"""Slice admission "price" oracle.

Adapted from Shen, Lam, Li (& Wang), "Privacy-Aware Spectrum Pricing and Power
Control Optimization for LEO Satellite Internet-of-Things" (arXiv 2407.00814,
IEEE TWC 2025): revenue = price * #{users whose budget >= price}, with budget
proportional to SINR, and a DDQN that steps the price one level up/down with
reward +1/0/-1.

Here the "price" is the admission threshold of a congested contention domain:
a best-effort flow is admitted only if priority * (SINR / 10) >= price. Instead
of training a DDQN we compute the exact best price level and label the
one-step direction (raise / hold / lower) that the frozen decision model is
asked to choose.
"""
from __future__ import annotations

from ..twin.radio import Scene

PRICE_LEVELS = tuple(round(0.5 * i, 1) for i in range(0, 13))   # 0 .. 6
PRIORITY = {"control": 5.0, "video_uplink": 3.0, "telemetry": 2.0,
            "map_upload": 1.0}


def admitted_value(scene: Scene, assoc: dict[str, str], sinr: dict[str, float],
                   domain: set[str], price: float, capacity: float) -> float:
    """Value delivered in one domain at a given admission price."""
    flows = []
    for car in scene.cars:
        if assoc[car.car_id] not in domain:
            continue
        budget_scale = max(sinr[car.car_id], 0.0) / 10.0
        for f in car.flows:
            if f.critical:
                continue
            bid = PRIORITY.get(f.name, 1.0) * budget_scale
            if bid >= price:
                flows.append((PRIORITY.get(f.name, 1.0), f.demand_mbps))
    total = sum(d for _, d in flows)
    share = 1.0 if total <= capacity else capacity / total
    return sum(p * d * share for p, d in flows)


def admission_oracle(scene: Scene, assoc: dict[str, str], sinr: dict[str, float],
                     domain: set[str], capacity_mbps: float,
                     current_level: int) -> dict:
    values = [admitted_value(scene, assoc, sinr, domain, p, capacity_mbps)
              - 0.05 * p for p in PRICE_LEVELS]
    best = max(range(len(values)), key=lambda i: values[i])
    nbr = {"lower": current_level - 1, "hold": current_level, "raise": current_level + 1}
    nbr = {k: v for k, v in nbr.items() if 0 <= v < len(PRICE_LEVELS)}
    local = max(nbr, key=lambda k: values[nbr[k]])
    if best > current_level:
        direction = "raise"
    elif best < current_level:
        direction = "lower"
    else:
        direction = "hold"
    return {"best_level": best, "direction": direction, "local_direction": local,
            "neighbour_levels": nbr, "values": [round(v, 3) for v in values]}


def admitted_flows(scene: Scene, assoc, sinr, domain, price) -> int:
    n = 0
    for car in scene.cars:
        if assoc[car.car_id] not in domain:
            continue
        for f in car.flows:
            if not f.critical and PRIORITY.get(f.name, 1.0) * max(sinr[car.car_id], 0) / 10 >= price:
                n += 1
    return n
