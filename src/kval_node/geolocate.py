"""Emitter localisation from cross-node signal reports.

The fabric can now move signed observations between nodes, so several nodes'
views of the same emitter can be combined. This module does that geometry — and
is deliberate about what it refuses to claim.

Two estimators, selected by what evidence actually exists:

**Weighted centroid** (always available with three positioned nodes). The
estimate is the signal-power-weighted centroid of the reporting nodes. This
needs no propagation model and no calibration, and is a standard sensor-network
technique. Its honest limit is structural: a centroid can only ever place an
emitter *inside the convex hull of the sensing nodes*. An emitter outside that
hull will be reported at the hull edge, with the uncertainty radius reflecting
that, and the estimate is flagged accordingly.

**Trilateration** (only when the operator has supplied calibrated path-loss
parameters in signed policy). Distances come from the log-distance path-loss
model, then a linearised least-squares solve. Without a calibrated reference
power and path-loss exponent for the actual deployment environment, the
distances would be arbitrary, so this estimator is simply unavailable rather
than run on guessed constants.

Both refuse rather than guess when the geometry cannot support an answer:
fewer than three positioned nodes, near-collinear nodes (the position
perpendicular to the line is unconstrained), or nodes whose clocks are not
disciplined enough for their reports to describe the same moment.

Every result carries an uncertainty radius and the method that produced it.
**No accuracy figure is claimed here.** Real accuracy depends on node geometry,
multipath, antenna patterns, and the calibration quality, and must be measured
in the deployment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .validation import bounded_number, non_empty_str, require_finite

#: Earth radius used for the local equirectangular projection. Baselines in a
#: sensing fleet are short (metres to kilometres), where this is accurate to
#: well under the uncertainty the estimators report.
EARTH_RADIUS_M = 6_371_000.0

#: Minimum positioned, correlatable nodes for any estimate. Two nodes leave the
#: solution ambiguous across the line between them.
MIN_NODES = 3

#: Ratio of the smaller to larger principal spread below which the nodes are
#: treated as collinear and the estimate is refused.
MIN_GEOMETRY_RATIO = 0.05


class GeolocationError(RuntimeError):
    """Raised when an estimate cannot honestly be produced."""


@dataclass(frozen=True)
class NodePosition:
    """Operator-declared position for a node.

    Positions come from the signed roster, never from the node's own report: a
    self-reported position is a spoofing surface, and an attacker who could
    claim to be anywhere could drag every estimate.
    """

    node_id: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NodePosition":
        return cls(
            node_id=non_empty_str(raw.get("node_id"), "node_id"),
            latitude_deg=bounded_number(raw["latitude_deg"], "latitude_deg", -90.0, 90.0),
            longitude_deg=bounded_number(raw["longitude_deg"], "longitude_deg", -180.0, 180.0),
            altitude_m=bounded_number(raw.get("altitude_m", 0.0), "altitude_m", -500.0, 20000.0),
        )


@dataclass(frozen=True)
class PathLossModel:
    """Operator-calibrated log-distance path-loss parameters.

    ``reference_power_dbm`` is the received power at ``reference_distance_m``
    from an emitter of interest, and ``path_loss_exponent`` describes how
    quickly power falls with distance (2.0 is free space; 2.7-4.0 typical
    outdoors/urban). Both are properties of a real environment and a real
    antenna, so they must be measured, not assumed.
    """

    reference_power_dbm: float
    path_loss_exponent: float
    reference_distance_m: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PathLossModel":
        exponent = bounded_number(raw["path_loss_exponent"], "path_loss_exponent", 1.5, 6.0)
        reference = bounded_number(raw.get("reference_distance_m", 1.0), "reference_distance_m", 0.1, 100.0)
        return cls(
            reference_power_dbm=bounded_number(raw["reference_power_dbm"], "reference_power_dbm", -120.0, 30.0),
            path_loss_exponent=exponent,
            reference_distance_m=reference,
        )

    def distance_m(self, received_dbm: float) -> float:
        """Invert the log-distance model for a received power."""
        exponent = (self.reference_power_dbm - require_finite(received_dbm, "received_dbm")) / (
            10.0 * self.path_loss_exponent
        )
        return self.reference_distance_m * (10.0**exponent)


@dataclass(frozen=True)
class SignalSighting:
    """One node's measured power for the emitter being located."""

    node_id: str
    received_dbm: float
    clock_trusted: bool


@dataclass(frozen=True)
class Estimate:
    method: str
    latitude_deg: float
    longitude_deg: float
    uncertainty_radius_m: float
    contributing_nodes: tuple[str, ...]
    inside_hull: bool
    geometry_ratio: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-emitter-estimate/v1",
            "method": self.method,
            "latitude_deg": round(self.latitude_deg, 7),
            "longitude_deg": round(self.longitude_deg, 7),
            "uncertainty_radius_m": round(self.uncertainty_radius_m, 2),
            "contributing_nodes": list(self.contributing_nodes),
            "inside_hull": self.inside_hull,
            "geometry_ratio": round(self.geometry_ratio, 6),
            "reasons": list(self.reasons),
        }


# --- local projection ------------------------------------------------------------


def to_local_xy(
    positions: Sequence[NodePosition],
) -> tuple[list[tuple[float, float]], float, float]:
    """Project to local east/north metres about the mean position."""
    mean_lat = sum(p.latitude_deg for p in positions) / len(positions)
    mean_lon = sum(p.longitude_deg for p in positions) / len(positions)
    cos_lat = math.cos(math.radians(mean_lat))
    points = [
        (
            math.radians(p.longitude_deg - mean_lon) * EARTH_RADIUS_M * cos_lat,
            math.radians(p.latitude_deg - mean_lat) * EARTH_RADIUS_M,
        )
        for p in positions
    ]
    return points, mean_lat, mean_lon


def to_lat_lon(x: float, y: float, mean_lat: float, mean_lon: float) -> tuple[float, float]:
    cos_lat = math.cos(math.radians(mean_lat))
    latitude = mean_lat + math.degrees(y / EARTH_RADIUS_M)
    longitude = mean_lon + math.degrees(x / (EARTH_RADIUS_M * cos_lat)) if cos_lat else mean_lon
    return latitude, longitude


def geometry_ratio(points: Sequence[tuple[float, float]]) -> float:
    """Ratio of the minor to major principal spread of the node positions.

    Near zero means the nodes are effectively collinear, so position
    perpendicular to that line is unconstrained and no estimate is honest.
    """
    count = len(points)
    mean_x = sum(p[0] for p in points) / count
    mean_y = sum(p[1] for p in points) / count
    cxx = sum((p[0] - mean_x) ** 2 for p in points) / count
    cyy = sum((p[1] - mean_y) ** 2 for p in points) / count
    cxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / count
    trace = cxx + cyy
    if trace <= 0.0:
        return 0.0
    # Analytic eigenvalues of a symmetric 2x2 covariance.
    discriminant = math.sqrt(max(0.0, ((cxx - cyy) / 2.0) ** 2 + cxy**2))
    major = trace / 2.0 + discriminant
    minor = trace / 2.0 - discriminant
    if major <= 0.0:
        return 0.0
    return math.sqrt(max(0.0, minor) / major)


def _point_in_hull(point: tuple[float, float], points: Sequence[tuple[float, float]]) -> bool:
    """Convex-hull containment via cross-product sign consistency.

    Uses the hull of the node positions; for the small node counts here an
    O(n log n) hull followed by a sign test is more than fast enough.
    """
    ordered = sorted(set(points))
    if len(ordered) < 3:
        return False

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for candidate in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], candidate) <= 0:
            lower.pop()
        lower.append(candidate)
    upper: list[tuple[float, float]] = []
    for candidate in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], candidate) <= 0:
            upper.pop()
        upper.append(candidate)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return False
    sign = 0
    for index in range(len(hull)):
        edge = cross(hull[index], hull[(index + 1) % len(hull)], point)
        if edge == 0:
            continue
        current = 1 if edge > 0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return True


# --- estimators --------------------------------------------------------------------


class EmitterLocator:
    def __init__(self, *, min_nodes: int = MIN_NODES, min_geometry_ratio: float = MIN_GEOMETRY_RATIO) -> None:
        self.min_nodes = max(3, min_nodes)
        self.min_geometry_ratio = min_geometry_ratio

    def _usable(
        self,
        sightings: Sequence[SignalSighting],
        positions: dict[str, NodePosition],
    ) -> tuple[list[SignalSighting], list[NodePosition], list[str]]:
        reasons: list[str] = []
        usable: list[SignalSighting] = []
        for sighting in sightings:
            if sighting.node_id not in positions:
                reasons.append("NODE_WITHOUT_DECLARED_POSITION_EXCLUDED")
                continue
            if not sighting.clock_trusted:
                # Sightings that cannot be placed at the same moment cannot be
                # combined into one emitter's position.
                reasons.append("NODE_WITH_UNTRUSTED_CLOCK_EXCLUDED")
                continue
            usable.append(sighting)
        node_positions = [positions[s.node_id] for s in usable]
        return usable, node_positions, reasons

    def locate(
        self,
        sightings: Sequence[SignalSighting],
        positions: Sequence[NodePosition],
        *,
        path_loss: PathLossModel | None = None,
    ) -> Estimate:
        by_id = {position.node_id: position for position in positions}
        usable, node_positions, reasons = self._usable(sightings, by_id)

        if len(usable) < self.min_nodes:
            raise GeolocationError(
                f"need at least {self.min_nodes} positioned nodes with trusted clocks; got {len(usable)}"
            )

        points, mean_lat, mean_lon = to_local_xy(node_positions)
        ratio = geometry_ratio(points)
        if ratio < self.min_geometry_ratio:
            raise GeolocationError(
                "node geometry is effectively collinear; position perpendicular to that line is unconstrained"
            )

        if path_loss is not None:
            estimate_xy, radius, method = self._trilaterate(usable, points, path_loss)
        else:
            estimate_xy, radius, method = self._weighted_centroid(usable, points)
            reasons.append("NO_CALIBRATED_PATH_LOSS_MODEL_SO_CENTROID_ONLY")

        inside = _point_in_hull(estimate_xy, points)
        if not inside:
            reasons.append("ESTIMATE_OUTSIDE_NODE_HULL")
        if method == "weighted_centroid":
            # Structural limit worth stating on every centroid result.
            reasons.append("CENTROID_CANNOT_PLACE_AN_EMITTER_OUTSIDE_THE_NODE_HULL")

        latitude, longitude = to_lat_lon(estimate_xy[0], estimate_xy[1], mean_lat, mean_lon)
        return Estimate(
            method=method,
            latitude_deg=latitude,
            longitude_deg=longitude,
            uncertainty_radius_m=radius,
            contributing_nodes=tuple(sorted(s.node_id for s in usable)),
            inside_hull=inside,
            geometry_ratio=ratio,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _weighted_centroid(
        sightings: Sequence[SignalSighting], points: Sequence[tuple[float, float]]
    ) -> tuple[tuple[float, float], float, str]:
        # Weight by linear received power: a stronger signal implies a closer
        # node. This needs no propagation model at all.
        weights = [10.0 ** (s.received_dbm / 10.0) for s in sightings]
        total = sum(weights)
        if total <= 0.0:
            raise GeolocationError("received powers produced no usable weighting")
        x = sum(w * p[0] for w, p in zip(weights, points)) / total
        y = sum(w * p[1] for w, p in zip(weights, points)) / total
        # Uncertainty: the weighted spread of contributing nodes about the
        # estimate. It is a spread, not a calibrated error bound, and is
        # reported as such.
        variance = sum(w * ((p[0] - x) ** 2 + (p[1] - y) ** 2) for w, p in zip(weights, points)) / total
        return (x, y), math.sqrt(max(0.0, variance)), "weighted_centroid"

    @staticmethod
    def _trilaterate(
        sightings: Sequence[SignalSighting],
        points: Sequence[tuple[float, float]],
        path_loss: PathLossModel,
    ) -> tuple[tuple[float, float], float, str]:
        distances = [path_loss.distance_m(s.received_dbm) for s in sightings]
        # Linearise by subtracting the first circle equation from the rest,
        # then solve the 2x2 normal equations directly.
        x0, y0 = points[0]
        d0 = distances[0]
        a11 = a12 = a22 = b1 = b2 = 0.0
        for (xi, yi), di in zip(points[1:], distances[1:]):
            ax = 2.0 * (xi - x0)
            ay = 2.0 * (yi - y0)
            bz = (d0**2 - di**2) + (xi**2 - x0**2) + (yi**2 - y0**2)
            a11 += ax * ax
            a12 += ax * ay
            a22 += ay * ay
            b1 += ax * bz
            b2 += ay * bz
        determinant = a11 * a22 - a12 * a12
        if abs(determinant) < 1e-9:
            raise GeolocationError("trilateration normal equations are singular for this geometry")
        x = (b1 * a22 - b2 * a12) / determinant
        y = (a11 * b2 - a12 * b1) / determinant
        # Uncertainty from residual disagreement between the measured distances
        # and the distances the solution implies.
        residuals = [
            abs(math.dist((x, y), point) - distance) for point, distance in zip(points, distances)
        ]
        radius = math.sqrt(sum(r**2 for r in residuals) / len(residuals))
        return (x, y), radius, "trilateration"
