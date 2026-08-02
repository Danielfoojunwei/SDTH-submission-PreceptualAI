"""Emitter track prediction from repeated fleet localisations.

Adapted from **Shen** and co-authors, "Dynamic Spectrum Optimization for
Internet-of-Vehicles with Deep-Learning-Based Mobility Prediction", Wireless
Personal Communications 137, 2024
([10.1007/s11277-024-11449-w](https://doi.org/10.1007/s11277-024-11449-w)).
Their pipeline: extend the **Exploration and Preferential Return** model to
D-EPR, use it to synthesise vehicle traces, train an LSTM on those traces to
predict where nodes will be, and feed the predictions to a graph-theoretic
spectrum-reuse optimiser.

**What is taken, and what is not.** The optimiser is transmit-side and is not
built. The LSTM is not built either, and the reason is worth stating plainly:
their network learns from traces the EPR model generates, so in their pipeline
EPR *is* the ground truth the network approximates. Shipping the model that
approximates it, rather than the model itself, would add a training dependency
and a fitted-weights artefact in exchange for a worse predictor of the very
process being modelled. Kval implements EPR directly.

**What EPR says.** From Song, Koren, Wang and Barabási, "Modelling the scaling
properties of human mobility", Nature Physics 6:818-823, 2010 — the empirical
basis the IoV paper builds on. An agent that has visited ``S`` distinct
locations either

* **explores** a new location, with probability ``p = rho * S**(-gamma)``, or
* **returns** to a previously visited one, chosen in proportion to how often it
  has been visited there — the rich get richer.

Both regularities are measured, not assumed: the exploration probability decays
as an agent's territory grows, and revisits follow a preferential-attachment
law. ``rho ~ 0.6`` and ``gamma ~ 0.21`` are the values reported in that paper.

**Why an emitter is a fair target for it.** EPR describes *purposeful* movement
with habitual places — patrols, depots, checkpoints, routes. That is what a
ground emitter of operational interest usually does, and it is exactly what
straight-line extrapolation gets wrong: kinematics predicts a vehicle continues
through the turn it is about to make back to somewhere it has already been.

**What it refuses.** EPR is a statistical model of a *population*, applied here
to a single track. Two consequences are enforced rather than glossed:

* With too little history there is no visit distribution to prefer from, and
  the model declines instead of returning the last position with false
  confidence.
* Every prediction reports its exploration probability. A high value means the
  model's own answer is "this emitter is likely to go somewhere it has never
  been", and no prediction can be made about *where* — the honest output is the
  probability itself, not a coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from .geolocate import EARTH_RADIUS_M
from .validation import bounded_number, require_finite

#: Exploration scale from Song et al., Nature Physics 6:818-823, 2010.
DEFAULT_RHO = 0.6

#: Exploration decay exponent from the same paper. Territory growth slows as an
#: agent accumulates familiar places.
DEFAULT_GAMMA = 0.21

#: Sightings below which no visit distribution exists worth preferring from.
MIN_TRACK_LENGTH = 4

#: Radius within which two fixes are treated as the same place. Set from the
#: locator's own uncertainty in practice; this default suits a fleet with
#: hundred-metre baselines.
DEFAULT_PLACE_RADIUS_M = 150.0

#: Exploration probability above which no location is predicted. At this point
#: the model is saying the emitter will most likely go somewhere new, and
#: naming a place would contradict the model producing it.
DEFAULT_MAX_EXPLORATION = 0.5

#: Gap beyond which consecutive fixes are not one continuous track. Bridging a
#: long silence would invent movement that was never observed.
DEFAULT_MAX_GAP_SECONDS = 900.0


class MobilityError(ValueError):
    """Raised when a track cannot support a prediction."""


@dataclass(frozen=True)
class Fix:
    """One localisation of an emitter at a moment in time."""

    observed_at: datetime
    latitude_deg: float
    longitude_deg: float
    uncertainty_radius_m: float = 0.0

    def __post_init__(self) -> None:
        bounded_number(self.latitude_deg, "latitude_deg", -90.0, 90.0)
        bounded_number(self.longitude_deg, "longitude_deg", -180.0, 180.0)
        require_finite(self.uncertainty_radius_m, "uncertainty_radius_m")


@dataclass(frozen=True)
class Place:
    """A location the emitter has been seen at, and how often."""

    latitude_deg: float
    longitude_deg: float
    visits: int
    last_seen: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude_deg": round(self.latitude_deg, 7),
            "longitude_deg": round(self.longitude_deg, 7),
            "visits": self.visits,
            "last_seen": self.last_seen.isoformat(),
        }


def metres_between(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Local equirectangular distance, matching the locator's own projection."""
    cos_lat = math.cos(math.radians((lat1 + lat2) / 2.0))
    east = math.radians(lon2 - lon1) * EARTH_RADIUS_M * cos_lat
    north = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    return math.hypot(east, north)


def cluster_places(
    fixes: Sequence[Fix], *, place_radius_m: float = DEFAULT_PLACE_RADIUS_M
) -> list[Place]:
    """Collapse fixes into distinct places with visit counts.

    A visit is counted per *arrival*, not per fix: an emitter that sits still
    while the fleet takes ten readings has visited once, and counting ten would
    make a stationary emitter look like a habitual returner.
    """
    radius = bounded_number(place_radius_m, "place_radius_m", 1.0, 100_000.0)
    ordered = sorted(fixes, key=lambda fix: fix.observed_at)
    centres: list[tuple[float, float]] = []
    visits: list[int] = []
    last_seen: list[datetime] = []
    previous_index: int | None = None

    for fix in ordered:
        match: int | None = None
        for index, (lat, lon) in enumerate(centres):
            if metres_between(lat, lon, fix.latitude_deg, fix.longitude_deg) <= radius:
                match = index
                break
        if match is None:
            centres.append((fix.latitude_deg, fix.longitude_deg))
            visits.append(1)
            last_seen.append(fix.observed_at)
            match = len(centres) - 1
        else:
            if match != previous_index:
                visits[match] += 1
            last_seen[match] = fix.observed_at
        previous_index = match

    return [
        Place(latitude_deg=lat, longitude_deg=lon, visits=count, last_seen=seen)
        for (lat, lon), count, seen in zip(centres, visits, last_seen)
    ]


def exploration_probability(
    distinct_places: int, *, rho: float = DEFAULT_RHO, gamma: float = DEFAULT_GAMMA
) -> float:
    """``p = rho * S**(-gamma)`` — the EPR exploration law.

    Falls as the emitter's known territory grows: an emitter seen in two places
    is far more likely to appear somewhere new than one seen in twenty.
    """
    if distinct_places < 1:
        return 1.0
    scale = bounded_number(rho, "rho", 0.0, 1.0)
    decay = bounded_number(gamma, "gamma", 0.0, 2.0)
    return min(1.0, scale * distinct_places ** (-decay))


@dataclass(frozen=True)
class TrackPrediction:
    predicted: bool
    latitude_deg: float | None
    longitude_deg: float | None
    uncertainty_radius_m: float
    exploration_probability: float
    distinct_places: int
    method: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-track-prediction/v1",
            "predicted": self.predicted,
            "latitude_deg": round(self.latitude_deg, 7) if self.latitude_deg is not None else None,
            "longitude_deg": (
                round(self.longitude_deg, 7) if self.longitude_deg is not None else None
            ),
            # ``null`` rather than ``Infinity``: a refusal has no radius, and
            # RFC 8259 has no infinity literal, so emitting one made the whole
            # document unparseable to a strict consumer.
            "uncertainty_radius_m": (
                round(self.uncertainty_radius_m, 2)
                if math.isfinite(self.uncertainty_radius_m)
                else None
            ),
            "exploration_probability": round(self.exploration_probability, 4),
            "distinct_places": self.distinct_places,
            "method": self.method,
            "reasons": list(self.reasons),
        }


class TrackPredictor:
    """Predicts where a tracked emitter goes next, or declines to.

    Two mechanisms, chosen by what the history supports rather than blended:

    **Preferential return** when the emitter has habitual places and is not
    presently in motion between them. The prediction is the visit-weighted
    centroid of known places, which is EPR's own return rule.

    **Kinematic continuation** when the recent fixes show consistent movement.
    An emitter mid-transit will finish the leg it is on before any return rule
    applies, and predicting a depot it is currently driving away from would be
    worse than useless.

    Preferring one over the other, rather than averaging, is deliberate:
    averaging a "carry on north" with a "go back to the depot" yields a point
    the emitter will not be at under either hypothesis.
    """

    def __init__(
        self,
        *,
        place_radius_m: float = DEFAULT_PLACE_RADIUS_M,
        rho: float = DEFAULT_RHO,
        gamma: float = DEFAULT_GAMMA,
        max_exploration: float = DEFAULT_MAX_EXPLORATION,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> None:
        self.place_radius_m = place_radius_m
        self.rho = rho
        self.gamma = gamma
        self.max_exploration = bounded_number(max_exploration, "max_exploration", 0.0, 1.0)
        self.max_gap = timedelta(seconds=max(1.0, max_gap_seconds))

    def predict(self, fixes: Sequence[Fix], *, horizon: timedelta) -> TrackPrediction:
        if horizon.total_seconds() <= 0:
            raise MobilityError("prediction horizon must be positive")
        ordered = sorted(fixes, key=lambda fix: fix.observed_at)
        if len(ordered) < MIN_TRACK_LENGTH:
            return TrackPrediction(
                predicted=False,
                latitude_deg=None,
                longitude_deg=None,
                uncertainty_radius_m=math.inf,
                exploration_probability=1.0,
                distinct_places=len(cluster_places(ordered, place_radius_m=self.place_radius_m)),
                method="none",
                reasons=(f"NEED_AT_LEAST_{MIN_TRACK_LENGTH}_FIXES_TO_MODEL_A_TRACK",),
            )

        reasons: list[str] = []
        if ordered[-1].observed_at - ordered[-2].observed_at > self.max_gap:
            # A long silence means the emitter's recent movement is unobserved,
            # so neither mechanism has the evidence it needs.
            return TrackPrediction(
                predicted=False,
                latitude_deg=None,
                longitude_deg=None,
                uncertainty_radius_m=math.inf,
                exploration_probability=1.0,
                distinct_places=len(cluster_places(ordered, place_radius_m=self.place_radius_m)),
                method="none",
                reasons=("TRACK_HAS_A_GAP_TOO_LONG_TO_BRIDGE",),
            )

        places = cluster_places(ordered, place_radius_m=self.place_radius_m)
        explore = exploration_probability(len(places), rho=self.rho, gamma=self.gamma)

        # The exploration gate decides between *known places*. With only one
        # known place there is nothing to choose between, so applying it there
        # was a category error: it refused hardest exactly when the evidence was
        # cleanest. An emitter observed repeatedly at one location has
        # demonstrably not moved, and "it stays" is the honest prediction —
        # p = rho * S^-gamma at S=1 is 0.600, above any sane threshold, so the
        # obvious operator flow (locate a stationary emitter a few times, then
        # track) returned nothing at all.
        #
        # The probability is still reported, because a single known place is
        # also the situation in which an emitter is *most* likely to strike out
        # somewhere new. The caller gets the prediction and the caveat.
        if len(places) > 1 and explore > self.max_exploration:
            # With a real choice available, the model's own answer is
            # "somewhere new". Naming one of the known places would contradict
            # the model that produced the probability.
            return TrackPrediction(
                predicted=False,
                latitude_deg=None,
                longitude_deg=None,
                uncertainty_radius_m=math.inf,
                exploration_probability=explore,
                distinct_places=len(places),
                method="none",
                reasons=("EXPLORATION_MORE_LIKELY_THAN_RETURN_SO_NO_PLACE_IS_PREDICTED",),
            )

        velocity = self._recent_velocity(ordered)
        if velocity is not None:
            east, north, speed = velocity
            seconds = horizon.total_seconds()
            latitude, longitude = _offset(
                ordered[-1].latitude_deg,
                ordered[-1].longitude_deg,
                east * seconds,
                north * seconds,
            )
            reasons.append("EMITTER_IS_IN_TRANSIT_SO_THE_CURRENT_LEG_CONTINUES")
            reasons.append(f"PREFERENTIAL_RETURN_DEFERRED_WITH_{len(places)}_KNOWN_PLACES")
            return TrackPrediction(
                predicted=True,
                latitude_deg=latitude,
                longitude_deg=longitude,
                uncertainty_radius_m=self._kinematic_uncertainty(ordered, speed, seconds),
                exploration_probability=explore,
                distinct_places=len(places),
                method="kinematic_continuation",
                reasons=tuple(reasons),
            )

        total_visits = sum(place.visits for place in places)
        if total_visits <= 0:
            raise MobilityError("clustered places carry no visits")
        latitude = sum(p.latitude_deg * p.visits for p in places) / total_visits
        longitude = sum(p.longitude_deg * p.visits for p in places) / total_visits
        spread = math.sqrt(
            sum(
                p.visits
                * metres_between(latitude, longitude, p.latitude_deg, p.longitude_deg) ** 2
                for p in places
            )
            / total_visits
        )
        reasons.append("PREFERENTIAL_RETURN_TO_A_VISIT_WEIGHTED_PLACE")
        if len(places) == 1:
            reasons.append("ONLY_ONE_KNOWN_PLACE_SO_THE_PREDICTION_IS_THAT_IT_STAYS")
            if explore > self.max_exploration:
                # Reported rather than acted on: one known place is also the
                # case where striking out somewhere new is most likely, and an
                # operator should see that alongside the prediction.
                reasons.append(
                    "A_SINGLE_KNOWN_PLACE_MAKES_EXPLORATION_LIKELY_TREAT_THIS_AS_WEAK"
                )
        return TrackPrediction(
            predicted=True,
            latitude_deg=latitude,
            longitude_deg=longitude,
            uncertainty_radius_m=max(spread, self.place_radius_m),
            exploration_probability=explore,
            distinct_places=len(places),
            method="preferential_return",
            reasons=tuple(reasons),
        )

    def _recent_velocity(
        self, ordered: Sequence[Fix]
    ) -> tuple[float, float, float] | None:
        """East/north metres per second over the last leg, if the emitter is moving.

        "Moving" means the last two legs agree in direction. One leg alone
        cannot distinguish transit from localisation noise, and disagreeing
        legs mean the emitter is manoeuvring, where extrapolation is worse than
        the return rule.
        """
        legs: list[tuple[float, float, float]] = []
        for earlier, later in zip(ordered[-3:], ordered[-2:]):
            seconds = (later.observed_at - earlier.observed_at).total_seconds()
            if seconds <= 0:
                return None
            east = (
                math.radians(later.longitude_deg - earlier.longitude_deg)
                * EARTH_RADIUS_M
                * math.cos(math.radians((later.latitude_deg + earlier.latitude_deg) / 2.0))
            )
            north = math.radians(later.latitude_deg - earlier.latitude_deg) * EARTH_RADIUS_M
            legs.append((east / seconds, north / seconds, math.hypot(east, north) / seconds))
        if len(legs) < 2:
            return None

        # Movement must exceed the fixes' own uncertainty, or "motion" is noise.
        noise = max((fix.uncertainty_radius_m for fix in ordered[-3:]), default=0.0)
        threshold = max(noise, self.place_radius_m) / max(
            1.0, (ordered[-1].observed_at - ordered[-3].observed_at).total_seconds()
        )
        if any(leg[2] <= threshold for leg in legs):
            return None

        # Direction agreement, via the cosine between successive leg vectors.
        first, second = legs[-2], legs[-1]
        dot = first[0] * second[0] + first[1] * second[1]
        magnitude = math.hypot(first[0], first[1]) * math.hypot(second[0], second[1])
        if magnitude <= 0 or dot / magnitude < 0.8:
            return None
        return second

    @staticmethod
    def _kinematic_uncertainty(
        ordered: Sequence[Fix], speed: float, seconds: float
    ) -> float:
        """Uncertainty grows with how far ahead the prediction reaches.

        A quarter of the distance travelled is a blunt but honest widening: an
        extrapolation twice as far ahead is roughly twice as uncertain, and the
        localisation error at the last fix floors it.
        """
        floor = max((fix.uncertainty_radius_m for fix in ordered[-3:]), default=0.0)
        return max(floor, 0.25 * speed * seconds)


def _offset(
    latitude: float, longitude: float, east_m: float, north_m: float
) -> tuple[float, float]:
    cos_lat = math.cos(math.radians(latitude))
    new_lat = latitude + math.degrees(north_m / EARTH_RADIUS_M)
    new_lon = (
        longitude + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat)) if cos_lat else longitude
    )
    return new_lat, new_lon
