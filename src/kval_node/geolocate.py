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

**Trilateration** (only when calibrated path-loss parameters exist for this
environment). Distances come from the log-distance path-loss model, then a
linearised least-squares solve. Without a calibrated reference power and
path-loss exponent for the actual deployment environment, the distances would be
arbitrary, so this estimator is simply unavailable rather than run on guessed
constants.

**Where those parameters come from.** Two routes, and only two. An operator can
measure them by hand and put them in a signed document, which is what
:class:`PathLossModel` has always described. Or the node can *fit* them from
measurements of emitters the operator has surveyed, which is what
:class:`PathLossCalibrator` adds. The fit is ordinary least squares on
``log10(d / d0)`` against received power, and its slope is the environment: an
exponent near 2 is free space, 2.7-3.5 is outdoor clutter, 4-6 is inside a
structure. A node parked in a metal hangar therefore *learns* that it is in one,
as a number, instead of being told.

The fit accepts surveyed reference emitters and nothing else. Fitting needs
known distances, which needs known emitter positions — and emitter positions are
precisely what this module is trying to find, so fitting against estimated ones
would fit the model to its own errors and then use that model to produce the
next estimate. :class:`SurveyedReference` exists to make that loop impossible to
close by accident; see its docstring for how.

All three refuse rather than guess when the evidence cannot support an answer:
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

from .validation import bounded_number, non_empty_str, positive_int, require_finite

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


# --- constants governing the path-loss fit ----------------------------------------

#: Provenance values that describe a position somebody *surveyed*. A fit accepts
#: these and nothing else. The vocabulary is closed rather than free text so that
#: "where did this position come from" is a decision the caller has to make
#: explicitly, and so that no spelling of "estimated" can pass for a survey.
SURVEY_PROVENANCE = frozenset(
    {"OPERATOR_SURVEYED", "SURVEY_CONTROL_POINT", "GNSS_SURVEYED_MARKER"}
)

#: Keys that appear only on a position this software *derived* — an emitter
#: estimate or a track prediction. A mapping offered as a survey that carries one
#: of them is this module's own output piped back into its own input. ``method``
#: is deliberately not in the set: a real survey record may well name its method,
#: and every derived document this node produces already carries several of the
#: keys below regardless.
DERIVED_POSITION_KEYS = frozenset(
    {
        "uncertainty_radius_m",
        "contributing_nodes",
        "inside_hull",
        "geometry_ratio",
        "residuals",
        "excluded",
        "exploration_probability",
        "distinct_places",
        "predicted",
    }
)

#: Schema identifiers of the derived-position documents, checked alongside the
#: keys so that renaming a field cannot quietly open the loop the keys close.
DERIVED_POSITION_SCHEMAS = frozenset(
    {"kval-emitter-estimate/v1", "kval-track-prediction/v1"}
)

#: Measurements below which a two-parameter fit has no residual worth trusting.
#: Two points define a line exactly, so their residual is zero whatever they say;
#: three leave one degree of freedom, where one bad measurement and a wrong model
#: are indistinguishable. Five leaves three, the smallest set where the residual
#: is a property of the data rather than of a single point. This is stricter than
#: the four-anchor floor the NLOS test uses, and deliberately so: a wrong
#: decision there drops one anchor, while a wrong exponent here mis-scales every
#: range this node will compute until it is recalibrated.
MIN_CALIBRATION_POINTS = 5

#: Decades of distance the measurements must span. A slope cannot be fitted to
#: points clustered at one radius at all — that is an exact degeneracy — and near
#: it the slope is fitted to noise instead. With the near and far measurements
#: half a decade apart, a 5 dB anomaly at one end moves the fitted exponent by a
#: full 1.0, which is a whole environment band, so half a decade is the point
#: below which the fit stops being evidence about anything.
MIN_DISTANCE_SPAN_DECADES = 0.5

#: Span below which the exponent is reported with a caveat rather than refused.
#: A decade — a factor of ten between the nearest and furthest measurement —
#: halves the sensitivity above: the same 5 dB anomaly then moves the exponent by
#: 0.5, which stays inside a band rather than crossing one.
WEAK_DISTANCE_SPAN_DECADES = 1.0

#: Residual standard error, in dB, above which the log-distance law is not
#: describing these measurements at all. Reported log-normal shadow fading
#: reaches roughly 8 dB outdoors, so scatter beyond that is a *different*
#: mechanism — waveguiding, a reflective structure, two environments averaged
#: together — rather than shadowing about a valid law. The same figure is the
#: practical ranging limit: at an exponent of 2.5, 8 dB of residual already means
#: every derived range is out by a factor of two.
MAX_RESIDUAL_RMS_DB = 8.0

#: Implied range error factor above which a fit that passed every gate is still
#: reported as weak. Ranges good only to +/-50% will place an emitter inside a
#: very large annulus, and an operator should see that before trusting a fix.
WEAK_RANGE_ERROR_FACTOR = 1.5

#: Standard error of the fitted exponent above which the fit is refused. The
#: descriptive bands below are about one unit wide, so an exponent whose own
#: error bar is wider than a band has not identified an environment, and its
#: slope is not distinguishable from the scatter it was fitted through. Least
#: squares always returns a number; this is the test of whether that number is
#: evidence.
MAX_EXPONENT_STANDARD_ERROR = 1.0

#: Physically plausible bounds on a fitted path-loss exponent. Corridor
#: waveguiding is the shallowest commonly reported case, around 1.6-1.8, and
#: heavily obstructed indoor propagation the steepest, around 6. A fit landing
#: outside this window means the *measurements* are wrong — a mislabelled
#: reference, a distance in the wrong unit, an emitter whose power changed
#: mid-survey — so it is refused rather than clamped, because a clamped value is
#: a fabricated one wearing a plausible number. The bounds match
#: ``PathLossModel.from_dict`` exactly, so any model this fit accepts is also one
#: the operator-calibrated loader would accept from a signed document.
MIN_PLAUSIBLE_PATH_LOSS_EXPONENT = 1.5
MAX_PLAUSIBLE_PATH_LOSS_EXPONENT = 6.0

#: Bounds on the fitted reference power, also matching ``PathLossModel``. The
#: reference distance is typically 1 m while the measurements are hundreds of
#: metres out, so this parameter is an extrapolation well outside the data and
#: can run away on a poor fit. A value no receiver could report is the symptom.
MIN_PLAUSIBLE_REFERENCE_POWER_DBM = -120.0
MAX_PLAUSIBLE_REFERENCE_POWER_DBM = 30.0

#: Fraction of a measurement's distance that the reference's own survey
#: uncertainty may consume before that measurement carries no distance
#: information. At a tenth, the position error contributes at most log10(1.1) =
#: 0.041 decades of error on the fit's x axis, which biases the exponent by less
#: than the width of an environment band. It is also what rejects an *estimated*
#: position offered as a survey: estimates come with radii of tens to hundreds of
#: metres, and one declared honestly fails this on its own numbers.
MAX_SURVEY_UNCERTAINTY_FRACTION = 0.1

#: Exponent boundaries between the descriptive environment bands. Both sit in the
#: *gaps* between the ranges the literature reports — free space at 2.0, urban
#: and foliage at 2.7-3.5, inside structures at 4-6 — because an edge placed
#: inside a reported range would split one environment across two names.
FREE_SPACE_BAND_MAX_EXPONENT = 2.5
OUTDOOR_CLUTTER_BAND_MAX_EXPONENT = 3.8


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

    Measured by hand into a signed document is one way, and the one this class
    was written for; nothing about it has changed. :class:`PathLossCalibrator`
    is the other, fitting the same two numbers from measurements of surveyed
    reference emitters. A deployment with no surveyed references behaves exactly
    as it always has.
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


# --- path-loss calibration from surveyed references --------------------------------


@dataclass(frozen=True)
class SurveyedReference:
    """An emitter at a position somebody surveyed, not one this node estimated.

    This type exists to make a single mistake impossible. Fitting a propagation
    model needs *known* distances, which needs known emitter positions — and
    emitter positions are exactly what :class:`EmitterLocator` produces rather
    than consumes. Fitting against an estimated position would tune the model on
    its own errors, and the model would then be used to produce the next
    estimate, so the error compounds instead of averaging away. Nothing
    downstream shows a symptom: every position stays confident and every one is
    wrong. So a fit accepts this type and no other.

    ``survey_uncertainty_m`` has no default deliberately. A caller must state how
    well the position is actually known, and an estimate declared honestly — tens
    or hundreds of metres — is then rejected by :class:`PathLossCalibrator` on
    its own numbers rather than on anyone's good intentions.

    Altitude is carried because the platforms this runs on fly, and a slant range
    is not a ground range.
    """

    reference_id: str
    latitude_deg: float
    longitude_deg: float
    survey_uncertainty_m: float
    survey_provenance: str
    altitude_m: float = 0.0

    def __post_init__(self) -> None:
        non_empty_str(self.reference_id, "reference_id")
        bounded_number(self.latitude_deg, "latitude_deg", -90.0, 90.0)
        bounded_number(self.longitude_deg, "longitude_deg", -180.0, 180.0)
        bounded_number(self.altitude_m, "altitude_m", -500.0, 20000.0)
        bounded_number(self.survey_uncertainty_m, "survey_uncertainty_m", 0.0, 100_000.0)
        if self.survey_provenance not in SURVEY_PROVENANCE:
            raise ValueError(
                "survey_provenance must be one of "
                f"{sorted(SURVEY_PROVENANCE)}; a position this node derived is not a survey"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SurveyedReference":
        """Load a surveyed reference, refusing anything shaped like an estimate.

        The realistic accident is not somebody typing an estimate's coordinates
        in by hand. It is ``SurveyedReference.from_dict(estimate.to_dict())``, or
        the same mapping read back out of the audit ledger, or a peer's
        localisation result forwarded over the fabric — all of which carry the
        fields a derived position has and a survey does not. Those fields are
        what this refuses on, by name, so the caller is told which one gave it
        away rather than being handed a silently poisoned model.
        """
        derived = sorted(DERIVED_POSITION_KEYS & set(raw))
        schema = raw.get("schema")
        if derived or schema in DERIVED_POSITION_SCHEMAS:
            evidence = derived or [str(schema)]
            raise ValueError(
                f"this mapping is a position this node derived, not a survey: it carries "
                f"{evidence}. Fitting propagation against an estimated emitter position "
                "would fit the model to its own errors"
            )
        return cls(
            reference_id=non_empty_str(raw.get("reference_id"), "reference_id"),
            latitude_deg=bounded_number(raw["latitude_deg"], "latitude_deg", -90.0, 90.0),
            longitude_deg=bounded_number(raw["longitude_deg"], "longitude_deg", -180.0, 180.0),
            survey_uncertainty_m=bounded_number(
                raw["survey_uncertainty_m"], "survey_uncertainty_m", 0.0, 100_000.0
            ),
            survey_provenance=str(raw.get("survey_provenance", "")),
            altitude_m=bounded_number(raw.get("altitude_m", 0.0), "altitude_m", -500.0, 20000.0),
        )


@dataclass(frozen=True)
class CalibrationObservation:
    """One node's measured power for one surveyed reference emitter."""

    reference: SurveyedReference
    observer: NodePosition
    received_dbm: float

    def __post_init__(self) -> None:
        # Checked at runtime rather than left to the annotations, because Python
        # does not enforce annotations and the entire value of the surveyed-only
        # rule is that it cannot be bypassed. An ``Estimate`` has
        # ``latitude_deg`` and ``longitude_deg`` too, so duck typing would let
        # the circular case through the front door.
        if not isinstance(self.reference, SurveyedReference):
            raise TypeError(
                "reference must be a SurveyedReference; an estimated or duck-typed "
                "position cannot ground a propagation fit"
            )
        if not isinstance(self.observer, NodePosition):
            raise TypeError(
                "observer must be a NodePosition, which comes from the signed roster; "
                "a self-reported observer position is a spoofing surface"
            )


def slant_range_m(reference: SurveyedReference, observer: NodePosition) -> float:
    """True three-dimensional distance between a reference and an observing node.

    Altitude is included where the rest of this module works in two dimensions,
    because here it changes the answer. A node 300 m above a reference 100 m away
    horizontally is 316 m from it; using the horizontal 100 m would place that
    measurement half a decade to the left on the fit's x axis, which is enough on
    its own to move the fitted exponent by a whole environment band.
    """
    cos_lat = math.cos(math.radians((reference.latitude_deg + observer.latitude_deg) / 2.0))
    east = (
        math.radians(observer.longitude_deg - reference.longitude_deg)
        * EARTH_RADIUS_M
        * cos_lat
    )
    north = math.radians(observer.latitude_deg - reference.latitude_deg) * EARTH_RADIUS_M
    up = observer.altitude_m - reference.altitude_m
    return math.sqrt(east * east + north * north + up * up)


def _band_of(exponent: float) -> str:
    if exponent < FREE_SPACE_BAND_MAX_EXPONENT:
        return "FREE_SPACE_LIKE"
    if exponent < OUTDOOR_CLUTTER_BAND_MAX_EXPONENT:
        return "OUTDOOR_CLUTTER_LIKE"
    return "HEAVY_OBSTRUCTION_LIKE"


def environment_band(exponent: float, standard_error: float) -> str:
    """Descriptive band for a fitted exponent, or ``UNCLASSIFIED``.

    The mapping from exponent to environment is indicative and nothing more.
    Published ranges overlap, they depend on frequency and antenna height, and
    two different environments can produce the same exponent — a clear path
    through foliage and a partly obstructed urban street both sit near 3. The
    band is a description of the number, not an identification of the place.

    The band is claimed only when the fitted exponent's own error bar lies wholly
    inside one band. An exponent of 2.6 +/- 0.4 is consistent with free space and
    with outdoor clutter alike, and naming either would be a claim the data does
    not support.
    """
    if not math.isfinite(exponent) or not math.isfinite(standard_error):
        return "UNCLASSIFIED"
    spread = max(0.0, standard_error)
    low = _band_of(exponent - spread)
    high = _band_of(exponent + spread)
    return low if low == high else "UNCLASSIFIED"


def _rounded(value: float | None, digits: int) -> float | None:
    """Round for serialisation, emitting ``None`` for anything not finite.

    RFC 8259 has no ``Infinity`` or ``NaN`` literal, so emitting one makes the
    whole document unparseable to a strict consumer.
    """
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


@dataclass(frozen=True)
class PathLossFit:
    """The outcome of a path-loss fit, whether or not it produced a model.

    ``samples`` counts the measurements offered and ``excluded_samples`` those
    dropped because the reference's own survey uncertainty did not resolve the
    distance, so the fit used the difference between them.

    A refused fit still reports every number it managed to compute. An operator
    who is told the exponent came out at 0.8 can find the mislabelled reference;
    an operator told only ``fitted: false`` cannot.
    """

    fitted: bool
    model: PathLossModel | None
    reference_id: str | None
    samples: int
    excluded_samples: int
    path_loss_exponent: float | None
    exponent_standard_error: float | None
    reference_power_dbm: float | None
    reference_power_standard_error_db: float | None
    residual_rms_db: float | None
    distance_span_decades: float | None
    range_error_factor: float | None
    environment_band: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-path-loss-fit/v1",
            "fitted": self.fitted,
            "reference_id": self.reference_id,
            "samples": self.samples,
            "excluded_samples": self.excluded_samples,
            "path_loss_exponent": _rounded(self.path_loss_exponent, 4),
            "exponent_standard_error": _rounded(self.exponent_standard_error, 4),
            "reference_power_dbm": _rounded(self.reference_power_dbm, 3),
            "reference_power_standard_error_db": _rounded(
                self.reference_power_standard_error_db, 3
            ),
            "residual_rms_db": _rounded(self.residual_rms_db, 3),
            "distance_span_decades": _rounded(self.distance_span_decades, 4),
            "range_error_factor": _rounded(self.range_error_factor, 3),
            "environment_band": self.environment_band,
            # Keyed exactly as ``PathLossModel.from_dict`` expects, so a fit an
            # operator accepts can be signed into the same path-loss document the
            # hand-measured route already uses, with no translation step that
            # could transpose the two numbers.
            "model": (
                {
                    "reference_power_dbm": _rounded(self.model.reference_power_dbm, 3),
                    "path_loss_exponent": _rounded(self.model.path_loss_exponent, 4),
                    "reference_distance_m": _rounded(self.model.reference_distance_m, 3),
                }
                if self.model is not None
                else None
            ),
            "reasons": list(self.reasons),
        }


class PathLossCalibrator:
    """Fits ``P(d) = P0 - 10 n log10(d / d0)`` from surveyed reference emitters.

    Ordinary least squares on ``x = log10(d / d0)`` against ``y = P``, in two
    accumulator loops. The slope gives the path-loss exponent and the intercept
    the reference power, which are the two numbers :class:`PathLossModel` has
    always required an operator to measure by hand.

    **The exponent is the environment measurement.** A node that fits ``n = 4.6``
    has established that it is behind heavy obstruction — inside a hull, under a
    deck, in a building — as a number rather than as an assumption written into a
    config file before deployment. :func:`environment_band` names that, with the
    error bar taken into account and with the mapping's indicativeness stated on
    every result.

    **Why the fit lives on the node.** EPFL-REMNet (arXiv:2511.05238) splits a
    radio-map model into a shared backbone and a *local personalised head*, the
    head staying on the client because it captures that client's own propagation.
    Their ablation is the argument for the split: removing personalisation was
    the most damaging change they measured, roughly tripling RMSE. The fitted
    exponent here is that head, reached without the network their design needs,
    which this repository's zero-runtime-dependency policy forbids. It is
    per-environment, it stays local, and it is not shared, because an exponent
    measured inside a metal hull describes nothing about the node parked outside
    it.

    **What it refuses.** Too few measurements; measurements clustered at one
    radius, where the slope is a degeneracy rather than a fit; non-finite inputs;
    a mixture of reference emitters, whose differing radiated powers cannot share
    one intercept; a residual too large for a log-distance law to be what these
    measurements follow; an exponent whose own standard error is wider than an
    environment band; and any fitted parameter outside physical plausibility,
    which is refused rather than clamped.
    """

    def __init__(
        self,
        *,
        reference_distance_m: float = 1.0,
        min_points: int = MIN_CALIBRATION_POINTS,
        min_span_decades: float = MIN_DISTANCE_SPAN_DECADES,
        max_residual_rms_db: float = MAX_RESIDUAL_RMS_DB,
        max_exponent_standard_error: float = MAX_EXPONENT_STANDARD_ERROR,
    ) -> None:
        self.reference_distance_m = bounded_number(
            reference_distance_m, "reference_distance_m", 0.1, 100.0
        )
        # Floored, not merely defaulted: below five points the residual is a
        # property of one measurement rather than of the data, so a caller
        # cannot configure the evidence away.
        self.min_points = max(MIN_CALIBRATION_POINTS, positive_int(min_points, "min_points"))
        self.min_span_decades = min_span_decades
        self.max_residual_rms_db = max_residual_rms_db
        self.max_exponent_standard_error = max_exponent_standard_error

    def fit(self, observations: Sequence[CalibrationObservation]) -> PathLossFit:
        """Fit the model, or return a refusal that says why.

        Data conditions return ``fitted=False`` with reasons, because a caller
        wants to record and act on them. Type violations raise instead: offering
        an estimated position where a survey is required is a defect in the
        calling code, not a property of the measurements, and swallowing it into
        a reason string would let it recur unnoticed.
        """
        items = list(observations)
        for item in items:
            if not isinstance(item, CalibrationObservation):
                raise TypeError(
                    "path-loss calibration accepts CalibrationObservation only, so that "
                    "every distance it fits is grounded in a surveyed position"
                )

        reference_ids = sorted({item.reference.reference_id for item in items})
        reference_id = reference_ids[0] if len(reference_ids) == 1 else None
        if len(reference_ids) > 1:
            # The intercept is the emitter's own radiated power seen at the
            # reference distance. Two emitters have two intercepts, and a single
            # fit through both would trade slope against that difference: a
            # weaker beacon further out reads exactly like a steeper environment.
            return self._unfitted(
                ["OBSERVATIONS_MIX_MORE_THAN_ONE_SURVEYED_REFERENCE_EMITTER"],
                reference_id=None,
                samples=len(items),
                excluded=0,
            )

        distances: list[float] = []
        powers: list[float] = []
        excluded = 0
        reasons: list[str] = []
        for item in items:
            if not math.isfinite(item.received_dbm):
                # Dropping it and fitting the rest would be the flattering
                # choice: a receiver reporting a non-finite power is broken, and
                # nothing else it reported in that batch is trustworthy either.
                return self._unfitted(
                    reasons + ["NON_FINITE_RECEIVED_POWER_IN_THE_CALIBRATION_SET"],
                    reference_id=reference_id,
                    samples=len(items),
                    excluded=excluded,
                )
            distance = slant_range_m(item.reference, item.observer)
            if not math.isfinite(distance) or distance <= 0.0:
                return self._unfitted(
                    reasons + ["AN_OBSERVER_IS_AT_THE_SURVEYED_REFERENCE_SO_NO_DISTANCE_EXISTS"],
                    reference_id=reference_id,
                    samples=len(items),
                    excluded=excluded,
                )
            if distance * MAX_SURVEY_UNCERTAINTY_FRACTION <= item.reference.survey_uncertainty_m:
                # The survey does not resolve this distance, so the point says
                # more about the survey than about propagation. Recorded as it
                # happens rather than after the loop, so a refusal raised
                # part-way through still reports the exclusions already made.
                excluded += 1
                reasons.append("MEASUREMENT_TOO_CLOSE_FOR_THE_SURVEY_TO_RESOLVE_EXCLUDED")
                continue
            distances.append(distance)
            powers.append(item.received_dbm)

        count = len(distances)
        if count < self.min_points:
            return self._unfitted(
                reasons + [f"NEED_AT_LEAST_{self.min_points}_SURVEYED_MEASUREMENTS_TO_FIT"],
                reference_id=reference_id,
                samples=len(items),
                excluded=excluded,
            )

        span = math.log10(max(distances) / min(distances))
        if span < self.min_span_decades:
            return self._unfitted(
                reasons + ["DISTANCE_SPAN_TOO_NARROW_TO_SEPARATE_A_SLOPE_FROM_NOISE"],
                reference_id=reference_id,
                samples=len(items),
                excluded=excluded,
                span=span,
            )

        xs = [math.log10(distance / self.reference_distance_m) for distance in distances]
        mean_x = sum(xs) / count
        mean_y = sum(powers) / count
        # Centred accumulators rather than raw sums of squares. With a 1 m
        # reference distance the x values sit around 2-3 and differ in the second
        # decimal, where ``sum(x*x) - count*mean_x**2`` loses most of its
        # significant figures to cancellation. The span gate above guarantees the
        # denominator is non-zero, since two distinct distances give two distinct
        # x and therefore a positive centred sum of squares.
        sxx = sum((x - mean_x) ** 2 for x in xs)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, powers))
        slope = sxy / sxx
        intercept = mean_y - slope * mean_x
        exponent = -slope / 10.0

        residuals = [y - (intercept + slope * x) for x, y in zip(xs, powers)]
        # Divided by the residual degrees of freedom, not by the sample count.
        # The fit spends two of them, and dividing by the count would report a
        # two-point fit as perfect — which it is, trivially, and which is exactly
        # the false confidence this number exists to prevent.
        residual_rms = math.sqrt(sum(r * r for r in residuals) / (count - 2))
        exponent_error = residual_rms / (10.0 * math.sqrt(sxx))
        power_error = residual_rms * math.sqrt(1.0 / count + mean_x * mean_x / sxx)
        # What the residual costs in metres: a residual of R dB read back through
        # the model is a range wrong by 10**(R / 10n). This is the number an
        # operator can act on, where a residual in dB is not.
        range_factor: float | None = None
        if exponent > 0.0:
            decades = residual_rms / (10.0 * exponent)
            # Capped rather than evaluated. Flat, noisy measurements fit a
            # near-zero exponent, and dividing by it sends this past the largest
            # float, where ``10.0 ** x`` raises rather than saturating — so the
            # arithmetic that describes how useless a fit is would crash before
            # the fit could be refused for being useless. Three decades is
            # already "no range information whatsoever"; the exact figure beyond
            # it carries nothing, and ``to_dict`` reports it as null.
            range_factor = 10.0**decades if decades <= 3.0 else math.inf

        refusals: list[str] = []
        if slope >= 0.0:
            refusals.append("FITTED_POWER_RISES_WITH_DISTANCE_WHICH_IS_NOT_PROPAGATION")
        if not (
            MIN_PLAUSIBLE_PATH_LOSS_EXPONENT
            <= exponent
            <= MAX_PLAUSIBLE_PATH_LOSS_EXPONENT
        ):
            refusals.append("FITTED_EXPONENT_OUTSIDE_PHYSICALLY_PLAUSIBLE_RANGE")
        if not (
            MIN_PLAUSIBLE_REFERENCE_POWER_DBM
            <= intercept
            <= MAX_PLAUSIBLE_REFERENCE_POWER_DBM
        ):
            refusals.append("FITTED_REFERENCE_POWER_OUTSIDE_ANY_PLAUSIBLE_RECEIVER_RANGE")
        if residual_rms > self.max_residual_rms_db:
            refusals.append("RESIDUAL_TOO_LARGE_FOR_A_LOG_DISTANCE_LAW_TO_DESCRIBE_THESE_POINTS")
        if exponent_error > self.max_exponent_standard_error:
            refusals.append("FITTED_EXPONENT_IS_NOT_DISTINGUISHABLE_FROM_THE_SCATTER")

        band = "UNCLASSIFIED" if refusals else environment_band(exponent, exponent_error)
        if refusals:
            reasons.extend(refusals)
            reasons.append("NO_ENVIRONMENT_BAND_IS_CLAIMED_FOR_A_REFUSED_FIT")
        else:
            reasons.append("PATH_LOSS_FITTED_FROM_SURVEYED_REFERENCE_MEASUREMENTS")
            reasons.append("ENVIRONMENT_BAND_IS_INDICATIVE_NOT_AN_IDENTIFICATION_OF_THE_PLACE")
            reasons.append("FIT_DESCRIBES_THIS_ENVIRONMENT_ONLY_AND_DOES_NOT_TRANSFER_TO_A_PEER")
            if band == "UNCLASSIFIED":
                reasons.append("EXPONENT_UNCERTAINTY_SPANS_MORE_THAN_ONE_ENVIRONMENT_BAND")
            if span < WEAK_DISTANCE_SPAN_DECADES:
                reasons.append("DISTANCE_SPAN_IS_NARROW_SO_THE_EXPONENT_IS_WEAKLY_CONSTRAINED")
            if count == self.min_points:
                reasons.append("AT_THE_MINIMUM_SAMPLE_COUNT_SO_ONE_MEASUREMENT_MOVES_THE_RESIDUAL")
            if range_factor is not None and range_factor > WEAK_RANGE_ERROR_FACTOR:
                reasons.append("RESIDUAL_IMPLIES_RANGE_ERRORS_OF_MORE_THAN_FIFTY_PERCENT")

        return PathLossFit(
            fitted=not refusals,
            model=(
                None
                if refusals
                else PathLossModel(
                    reference_power_dbm=intercept,
                    path_loss_exponent=exponent,
                    reference_distance_m=self.reference_distance_m,
                )
            ),
            reference_id=reference_id,
            samples=len(items),
            excluded_samples=excluded,
            path_loss_exponent=exponent,
            exponent_standard_error=exponent_error,
            reference_power_dbm=intercept,
            reference_power_standard_error_db=power_error,
            residual_rms_db=residual_rms,
            distance_span_decades=span,
            range_error_factor=range_factor,
            environment_band=band,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _unfitted(
        reasons: Sequence[str],
        *,
        reference_id: str | None,
        samples: int,
        excluded: int,
        span: float | None = None,
    ) -> PathLossFit:
        """A refusal from before a fit was computable, carrying what is known."""
        return PathLossFit(
            fitted=False,
            model=None,
            reference_id=reference_id,
            samples=samples,
            excluded_samples=excluded,
            path_loss_exponent=None,
            exponent_standard_error=None,
            reference_power_dbm=None,
            reference_power_standard_error_db=None,
            residual_rms_db=None,
            distance_span_decades=span,
            range_error_factor=None,
            environment_band="UNCLASSIFIED",
            reasons=tuple(dict.fromkeys(reasons)),
        )
