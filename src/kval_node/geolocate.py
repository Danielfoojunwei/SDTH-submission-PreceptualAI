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

**Non-line-of-sight rejection.** Trilateration additionally applies the
residual-consistency test of Hua, Yin, Lu, Zhang and **Feng Li**, "NLOS
Identification and Positioning Algorithm Based on Localization Residual in
Wireless Sensor Networks", Sensors 18(9):2991, 2018
([10.3390/s18092991](https://doi.org/10.3390/s18092991)). Their insight is that
an obstructed path produces a range measurement that is inconsistent with the
solution the *other* anchors agree on, and that the inconsistency has a known
sign: an obstruction can only make a path look **longer**, never shorter. So a
node whose measured range greatly exceeds what the rest of the fleet implies is
plausibly blocked and is excluded; a node whose measured range is much *shorter*
cannot be explained by obstruction at all, and is excluded under a separate,
louder reason — a signal arriving stronger than physics allows for that distance
is the signature of a nearby repeater, a reflection, or a fabricated report.

Every result carries an uncertainty radius and the method that produced it.
**No accuracy figure is claimed here.** Real accuracy depends on node geometry,
multipath, antenna patterns, and the calibration quality, and must be measured
in the deployment.
"""

from __future__ import annotations

import math
import statistics
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

#: Consistency factor converting the median absolute deviation into a standard
#: deviation estimate for a normal distribution.
MAD_TO_SIGMA = 1.4826

#: Residual, in units of the robust residual scale, beyond which a sighting is
#: treated as inconsistent with what the rest of the fleet agrees on. 3.0 is the
#: conventional robust-statistics cut; it is a threshold on *disagreement*, not
#: an accuracy claim.
NLOS_RESIDUAL_SIGMAS = 3.0

#: Residual, in metres, below which a solution is treated as already consistent
#: and nothing is excluded. Also floors the robust scale, since a set of
#: perfectly consistent measurements otherwise gives a scale of ~0 and every
#: sighting looks like an infinite-sigma outlier.
MIN_RESIDUAL_SCALE_M = 1.0

#: Anchors that must remain for the residual test to mean anything. **Three
#: anchors always fit perfectly** — the linearised system has two equations and
#: two unknowns — so a three-anchor solution has zero residual whether or not
#: one of them is blocked. Four is the smallest set where a residual carries
#: information, which caps how many blocked nodes a fleet of ``n`` can survive
#: at ``n - 4``.
MIN_NODES_FOR_RESIDUAL_TEST = 4

#: Residual RMS, as a multiple of the array's own aperture, beyond which the
#: measurements carry no positional information at that scale and no estimate is
#: returned. When the anchors disagree by more than the distance between them,
#: the solution is an artefact of least squares, not a location.
MAX_RESIDUAL_TO_APERTURE_RATIO = 1.0

#: Factor by which dropping one anchor must improve the residual before that
#: anchor is blamed for the inconsistency. Removing an anchor from a clean set
#: improves the fit slightly, just by removing a constraint; removing a blocked
#: one improves it by an order of magnitude. Requiring a large ratio keeps the
#: test from quietly trimming good anchors to flatter the result.
RESIDUAL_IMPROVEMENT_FACTOR = 3.0


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
class ResidualFinding:
    """One node's disagreement with the solution the others support."""

    node_id: str
    residual_m: float
    normalised: float
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "residual_m": round(self.residual_m, 2),
            "normalised": round(self.normalised, 3),
            "classification": self.classification,
        }


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
    excluded: tuple[ResidualFinding, ...] = ()
    residuals: tuple[ResidualFinding, ...] = ()

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
            "excluded": [finding.to_dict() for finding in self.excluded],
            "residuals": [finding.to_dict() for finding in self.residuals],
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


def array_aperture(points: Sequence[tuple[float, float]]) -> float:
    """RMS distance of the anchors from their centroid, in metres.

    The natural length scale of the sensing array, used to judge whether a
    residual is small enough for the solution to mean anything.
    """
    if not points:
        return 0.0
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    return math.sqrt(
        sum((p[0] - mean_x) ** 2 + (p[1] - mean_y) ** 2 for p in points) / len(points)
    )


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


# --- NLOS identification via localisation residual --------------------------------


def robust_residual_scale(residuals: Sequence[float]) -> float:
    """Scale of the residual distribution, resistant to the outliers being sought.

    A plain standard deviation is inflated by the very anchors this test is
    meant to find, so a blocked node raises the bar it is measured against and
    hides. The median absolute deviation does not have that feedback.
    """
    if not residuals:
        return MIN_RESIDUAL_SCALE_M
    ordered = sorted(residuals)
    median = statistics.median(ordered)
    deviation = statistics.median([abs(value - median) for value in ordered])
    return max(MIN_RESIDUAL_SCALE_M, deviation * MAD_TO_SIGMA)


def classify_residuals(
    node_ids: Sequence[str],
    residuals: Sequence[float],
    *,
    threshold_sigmas: float = NLOS_RESIDUAL_SIGMAS,
) -> list[ResidualFinding]:
    """Label each sighting by how far it sits from the fleet's agreed solution.

    ``residuals`` are ``implied_distance - measured_distance`` in metres, so:

    * **negative** — the node measured a *longer* range than the solution
      implies. That is what an obstruction does: extra attenuation reads as
      extra distance. Classified ``NLOS_SUSPECTED``.
    * **positive** — the node measured a *shorter* range than the solution
      implies, meaning it heard the emitter more strongly than the geometry
      allows. Obstruction cannot cause this. Classified
      ``IMPLAUSIBLY_STRONG`` and treated as the more serious finding, because a
      repeater, a reflection or a fabricated report all look like this and none
      of them are propagation.
    """
    if len(node_ids) != len(residuals):
        raise ValueError("node_ids and residuals must be the same length")
    scale = robust_residual_scale(residuals)
    median = statistics.median(residuals) if residuals else 0.0
    findings: list[ResidualFinding] = []
    for node_id, residual in zip(node_ids, residuals):
        # Centre on the median: a constant offset shared by every node is a
        # calibration error in the path-loss model, not one node being blocked.
        normalised = (residual - median) / scale
        if abs(normalised) <= threshold_sigmas:
            classification = "CONSISTENT"
        elif normalised < 0:
            classification = "NLOS_SUSPECTED"
        else:
            classification = "IMPLAUSIBLY_STRONG"
        findings.append(
            ResidualFinding(
                node_id=node_id,
                residual_m=residual,
                normalised=normalised,
                classification=classification,
            )
        )
    return findings


# --- estimators --------------------------------------------------------------------


class EmitterLocator:
    def __init__(
        self,
        *,
        min_nodes: int = MIN_NODES,
        min_geometry_ratio: float = MIN_GEOMETRY_RATIO,
        reject_nlos: bool = True,
        nlos_threshold_sigmas: float = NLOS_RESIDUAL_SIGMAS,
        max_residual_ratio: float = MAX_RESIDUAL_TO_APERTURE_RATIO,
    ) -> None:
        self.min_nodes = max(3, min_nodes)
        self.min_geometry_ratio = min_geometry_ratio
        self.reject_nlos = reject_nlos
        self.nlos_threshold_sigmas = nlos_threshold_sigmas
        self.max_residual_ratio = max_residual_ratio

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

        excluded: tuple[ResidualFinding, ...] = ()
        residuals: tuple[ResidualFinding, ...] = ()
        if path_loss is not None:
            if self.reject_nlos:
                usable, points, excluded, drop_reasons = self._reject_inconsistent(
                    usable, points, path_loss
                )
                reasons.extend(drop_reasons)
                # Excluding a node changes the geometry, so the collinearity
                # guard has to hold for the set actually used, not the set
                # offered.
                node_positions = [by_id[s.node_id] for s in usable]
                points, mean_lat, mean_lon = to_local_xy(node_positions)
                ratio = geometry_ratio(points)
                if ratio < self.min_geometry_ratio:
                    raise GeolocationError(
                        "after excluding inconsistent sightings the remaining nodes are effectively collinear"
                    )
            else:
                reasons.append("NLOS_REJECTION_DISABLED_BY_CONFIGURATION")
            estimate_xy, radius, method = self._trilaterate(usable, points, path_loss)
            aperture = array_aperture(points)
            if radius > self.max_residual_ratio * aperture:
                # Reached when several anchors are blocked at once: no single
                # removal explains the disagreement, so nothing was excluded and
                # the surviving set still cannot be reconciled. Returning the
                # least-squares point here would hand back a position that can
                # be kilometres wrong with only a reason string to warn of it.
                raise GeolocationError(
                    "anchors disagree by more than the array aperture; no position is supported "
                    f"by these measurements (residual {radius:.0f} m vs aperture {aperture:.0f} m)"
                )
            # Report how well each *surviving* anchor agrees, so a marginal one
            # that was not excluded is still visible to an operator.
            final_distances = [path_loss.distance_m(s.received_dbm) for s in usable]
            residuals = tuple(
                classify_residuals(
                    [s.node_id for s in usable],
                    [
                        math.dist(estimate_xy, point) - distance
                        for point, distance in zip(points, final_distances)
                    ],
                    threshold_sigmas=self.nlos_threshold_sigmas,
                )
            )
        else:
            estimate_xy, radius, method = self._weighted_centroid(usable, points)
            reasons.append("NO_CALIBRATED_PATH_LOSS_MODEL_SO_CENTROID_ONLY")
            reasons.append("NLOS_REJECTION_NEEDS_RANGES_SO_IS_UNAVAILABLE_FOR_CENTROID")

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
            excluded=excluded,
            residuals=residuals,
        )

    def _fit(
        self,
        sightings: Sequence[SignalSighting],
        points: Sequence[tuple[float, float]],
        path_loss: PathLossModel,
    ) -> tuple[tuple[float, float], float, list[float]]:
        """Solve, and report how badly the anchors disagree with that solution."""
        estimate_xy, _radius, _method = self._trilaterate(sightings, points, path_loss)
        distances = [path_loss.distance_m(s.received_dbm) for s in sightings]
        residuals = [
            math.dist(estimate_xy, point) - distance
            for point, distance in zip(points, distances)
        ]
        rms = math.sqrt(sum(r**2 for r in residuals) / len(residuals))
        return estimate_xy, rms, residuals

    def _rejection_floor(self) -> int:
        return max(self.min_nodes, MIN_NODES_FOR_RESIDUAL_TEST)

    def _reject_inconsistent(
        self,
        sightings: Sequence[SignalSighting],
        points: Sequence[tuple[float, float]],
        path_loss: PathLossModel,
    ) -> tuple[list[SignalSighting], list[tuple[float, float]], tuple[ResidualFinding, ...], list[str]]:
        """Exclude anchors that cannot be reconciled with the rest.

        This is the localisation-residual test of Hua et al.: a blocked anchor
        is found not by looking at its own residual under the full solve — the
        full solve is already dragged by it, which is why every anchor then
        looks mildly wrong and none looks decisively wrong — but by asking
        **which single anchor, if removed, makes the remaining ones agree**.
        Removal happens one at a time with a full re-solve after each, and only
        when the improvement is large enough that no clean anchor could produce
        it by chance.
        """
        remaining = list(sightings)
        remaining_points = list(points)
        dropped: list[ResidualFinding] = []
        reasons: list[str] = []
        floor = self._rejection_floor()

        if len(remaining) <= floor:
            reasons.append("TOO_FEW_ANCHORS_TO_IDENTIFY_A_BLOCKED_ONE")
            return remaining, remaining_points, (), reasons

        while len(remaining) > floor:
            try:
                _xy, rms, _residuals = self._fit(remaining, remaining_points, path_loss)
            except GeolocationError:
                # A geometry this solve cannot handle is not evidence about any
                # single node, so nothing is excluded on the strength of it.
                break
            if rms <= MIN_RESIDUAL_SCALE_M:
                break

            best_index: int | None = None
            best_rms = math.inf
            best_residual = 0.0
            for index in range(len(remaining)):
                trial = remaining[:index] + remaining[index + 1 :]
                trial_points = remaining_points[:index] + remaining_points[index + 1 :]
                try:
                    trial_xy, trial_rms, _ = self._fit(trial, trial_points, path_loss)
                except GeolocationError:
                    continue
                if trial_rms < best_rms:
                    best_index, best_rms = index, trial_rms
                    # Judge the candidate against the solution the *others*
                    # support, not the one it helped produce.
                    best_residual = math.dist(trial_xy, remaining_points[index]) - path_loss.distance_m(
                        remaining[index].received_dbm
                    )

            if best_index is None or best_rms * RESIDUAL_IMPROVEMENT_FACTOR >= rms:
                # No single anchor explains the disagreement. That is a fleet-
                # wide problem — a mis-calibrated path-loss model, or several
                # blocked anchors at once — and blaming one node would be wrong.
                if rms > MIN_RESIDUAL_SCALE_M:
                    reasons.append("ANCHORS_DISAGREE_BUT_NO_SINGLE_ONE_EXPLAINS_IT")
                break

            classification = "NLOS_SUSPECTED" if best_residual < 0 else "IMPLAUSIBLY_STRONG"
            dropped.append(
                ResidualFinding(
                    node_id=remaining[best_index].node_id,
                    residual_m=best_residual,
                    normalised=rms / max(best_rms, MIN_RESIDUAL_SCALE_M),
                    classification=classification,
                )
            )
            reasons.append(
                "EXCLUDED_NLOS_SUSPECTED_SIGHTING"
                if classification == "NLOS_SUSPECTED"
                else "EXCLUDED_SIGHTING_STRONGER_THAN_PROPAGATION_ALLOWS"
            )
            del remaining[best_index]
            del remaining_points[best_index]

        if len(remaining) == floor and len(sightings) > floor:
            # Worth saying out loud: from here the test cannot run again, so a
            # further blocked anchor would survive silently into the solution.
            reasons.append("AT_MINIMUM_ANCHOR_COUNT_NO_FURTHER_REJECTION_POSSIBLE")
        return remaining, remaining_points, tuple(dropped), reasons

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
