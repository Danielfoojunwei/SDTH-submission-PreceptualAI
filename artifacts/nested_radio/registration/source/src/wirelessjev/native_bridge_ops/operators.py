"""Reuse frequency-independent native rack intersections across three bands.

This standalone operator consumes only public AP sites, known transmit/noise
values and planned robot positions. It does not receive a World, observations,
model, private RF, material labels or packet simulator. The frozen native runner
continues to use its original implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..native_bridge.predictor import (
    AP_HEIGHT_M,
    HALL_HEIGHT_M,
    HEIGHT,
    HORIZONS,
    NATIVE_CARRIERS_HZ,
    RX_HEIGHT_M,
    WIDTH,
    rack_boxes,
    vectorized_box_features,
)
from ..spectral_scene.operators import fspl_received_power_dbm, query_physics_basis


@dataclass(frozen=True)
class PublicQueryGeometry:
    """Explicit public-only snapshot; no simulator or observation handle."""

    sites: np.ndarray
    tx: np.ndarray
    noise: np.ndarray

    def __post_init__(self):
        sites, tx, noise = (np.asarray(value, dtype=float) for value in (self.sites, self.tx, self.noise))
        if (sites.ndim != 2 or sites.shape[1:] != (2,) or len(sites) < 1
                or tx.shape != (3,) or noise.shape != (3,)
                or not all(np.isfinite(value).all() for value in (sites, tx, noise))):
            raise ValueError("Finite public AP XY sites and three known transmit/noise values required")
        for name, value in (("sites", sites), ("tx", tx), ("noise", noise)):
            frozen = np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
            object.__setattr__(self, name, frozen)


def query_operands(geometry: PublicQueryGeometry, plan):
    """Return frozen-query byte-equivalent operands with one AABB pass per link.

    Output order remains robot × horizon × AP × band, with the exact frozen
    carrier, power, noise, float32 feature/basis and FSPL arithmetic. Only the
    frequency-independent rack geometry is shared across bands. No result is
    cached across calls, so changes in a plan or AP coordinates take effect.
    """
    if not isinstance(geometry, PublicQueryGeometry):
        raise TypeError("Supply an explicit PublicQueryGeometry snapshot, not a simulator")
    plan = np.asarray(plan, dtype=float)
    if (plan.ndim != 3 or plan.shape[1:] != (len(HORIZONS), 2)
            or not np.isfinite(plan).all()):
        raise ValueError("Only a finite public native robot×nine-horizon×XY plan is accepted")
    n, h = plan.shape[:2]
    sites = geometry.sites
    s = len(sites)
    link_shape, shape = (n, h, s), (n, h, s, 3)
    rx_links = np.broadcast_to(
        np.concatenate((plan, np.full((n, h, 1), RX_HEIGHT_M)), axis=-1)[:, :, None, :],
        link_shape + (3,),
    ).reshape(-1, 3)
    ap_links = np.broadcast_to(
        np.column_stack((sites, np.full(s, AP_HEIGHT_M)))[None, None, :, :],
        link_shape + (3,),
    ).reshape(-1, 3)
    # AABB geometry and distance are band-independent; compute them once.
    distance = np.linalg.norm(rx_links - ap_links, axis=-1)
    crossings, fraction = vectorized_box_features(rx_links, ap_links, rack_boxes())
    links = len(rx_links)
    base = np.column_stack((
        rx_links / np.array((WIDTH, HEIGHT, HALL_HEIGHT_M)),
        ap_links / np.array((WIDTH, HEIGHT, HALL_HEIGHT_M)),
        np.log10(distance) / 2,
        np.zeros(links), -np.ones(links), crossings / 8, fraction,
        np.zeros(links), np.zeros(links), np.zeros(links), np.zeros(links), np.zeros(links),
    )).astype("f4")
    query = np.broadcast_to(base[:, None, :], (links, 3, 16)).copy()
    query[:, :, 7] = np.log(np.asarray(NATIVE_CARRIERS_HZ, dtype=float) / 2.4e9)
    query[:, :, 14] = geometry.noise / 100
    query[:, :, 15] = geometry.tx / 30
    query = query.reshape(-1, 16)
    # Preserve the frozen physical-operator execution and rounding exactly.
    # This FSPL call still sees all carrier-specific queries; it is not an
    # approximation that substitutes a nominal carrier or shared band loss.
    rx = np.broadcast_to(rx_links[:, None, :], (links, 3, 3)).reshape(-1, 3)
    ap = np.broadcast_to(ap_links[:, None, :], (links, 3, 3)).reshape(-1, 3)
    carrier = np.broadcast_to(np.asarray(NATIVE_CARRIERS_HZ), shape).reshape(-1)
    tx = np.broadcast_to(geometry.tx, shape).reshape(-1)
    return {
        "query": query,
        "basis": query_physics_basis(query),
        "fspl": fspl_received_power_dbm(rx, ap, carrier, tx).astype("f4"),
        "offset": np.zeros(len(query), dtype="f4"),
        "shape": shape,
    }
