"""What is *typical* here, and why that is never what the weather *is*.

:mod:`kval_node.propagation` can say that a fade's shape is consistent with
hydrometeor attenuation. That is a statement about physics, and it is true in
Singapore, in the Atacama and over the Barents Sea alike. What it cannot say is
whether the mechanism it named is plausible **here, in this month**. Twenty-five
millimetres of rain an hour is an unremarkable November afternoon on the
equator and very nearly impossible in an Atacama July; wet snow is not a
candidate at sea level on the equator in any month at all. Without that context
every fade in every climate is scored identically, and the node cannot tell an
operator whether the sky is a likely explanation or a surprising one.

**Why the context is declared and signed rather than fetched.** This node
deploys disconnected. There is no runtime weather service and there will not be
one: a receive-only defence sensor that reaches out for a forecast has, at that
instant, stopped being receive-only. Nor can a third-party climate grid be
vendored into this repository — its provenance and licensing cannot be verified
from inside the repository, and data whose origin cannot be checked is not
evidence here. So the operator, who knows where the platform is going, declares
the climate of the area of operations, signs it with the same operator key that
signs the spectrum policy and the emission-control order, and the node verifies
and reasons from it. The document is the evidence, and it carries an author.

**What this module refuses to do.**

*It never reports weather.* Every figure here is climatology — a statement about
a distribution over years, not about today. Nothing in the output is named in a
way that could be read as an observation, and every serialised result carries
that caveat in its own record rather than trusting a downstream consumer to
remember it.

*It never overrides the physics.* Climatology combines with a measurement as
evidence in one direction only. When the two disagree the conflict is reported
and the measurement is kept whole. The declaration can be stale, the platform
can have flown out of the area it describes, and a climate that produces a
mechanism 0.01% of the time produces it: a node that discards a real
observation because a file disagreed with it has been defeated by its own
paperwork. Equally, a typical season is never allowed to *create* a mechanism
the measurement did not support — an implausible-looking direction that matters
just as much, because "it is the monsoon" is not evidence that anything is
falling on this particular path.

*It never fills a gap with a default.* A declaration that does not say whether
the area is maritime, or that leaves part of the calendar unaccounted for, is
refused rather than completed. A default climate is a claim about the world that
nobody made and nobody signed.

**What the document deliberately does not carry.** No temperature, humidity or
pressure: the node has no instrument to compare them against and no model that
consumes them, and monthly means invite exactly the misreading this module
exists to prevent. No vertical structure or rain height: those are inputs to an
attenuation calculation, which belongs with the physics, not with the question
of whether a mechanism is credible. No altitude band: a platform's height is
known to the platform, and asking an operator for a vertical climate profile
they do not have is how documents acquire invented numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from .models import parse_timestamp, utc_now
from .validation import bounded_number, non_empty_str

CLIMATE_DECLARATION_SCHEMA = "kval-climate-declaration/v1"

#: The mechanisms an operator can credibly speak to. The vocabulary is closed on
#: purpose. An open set lets a declaration name ``monsoon_effects`` while the
#: physics side asks about ``rain``, and the two never meet — the node would
#: report "not assessed" forever and nothing would look wrong. A closed set
#: turns that mismatch into a refusal at load time.
#:
#: Liquid hydrometeors. The dominant path attenuation above roughly 10 GHz and
#: the only mechanism here for which an operator can obtain a published
#: intensity climatology (ITU-R P.837 rain-rate maps).
MECHANISM_RAIN = "rain"
#: Melting snow. A water-coated ice particle is electrically much closer to a
#: raindrop than to a snowflake, so the melting layer attenuates far more
#: strongly than either the snow above it or, per unit water, the rain below.
#: Credible only where the freezing level intersects the path.
MECHANISM_WET_SNOW = "wet_snow"
#: Dry snow and ice crystals. Weak attenuators individually, but a declaration
#: that cannot distinguish them from wet snow cannot express the one climate
#: fact that matters here — whether the melting layer is in reach at all.
MECHANISM_DRY_SNOW = "dry_snow"
#: Cloud and fog liquid water (ITU-R P.840 territory). Small compared with rain
#: but persistent, and in a coastal or maritime climate it is present for a far
#: larger fraction of the year than rain is.
MECHANISM_FOG_OR_CLOUD = "fog_or_cloud"
#: Lofted dust and sand. Attenuation is modest at microwave frequencies, which
#: is exactly why it belongs in a *plausibility* document rather than an
#: attenuation model: in an arid area of operations it is the mechanism an
#: operator will reach for first, and elsewhere it should not be reachable.
MECHANISM_DUST_OR_SAND = "dust_or_sand"
#: Evaporation ducting. Trapping of energy in the humidity gradient immediately
#: above a water surface. Not attenuation at all — it enhances propagation
#: inside the duct and deepens the shadow outside it — and it is the one
#: mechanism that is a property of the surface rather than of the sky, which is
#: why the declaration has to say whether there is a sea underneath.
MECHANISM_EVAPORATION_DUCT = "evaporation_duct"
#: Refractive-index turbulence: rapid, shallow scintillation. Strong in humid
#: convective climates and in the dry, deeply mixed boundary layer over a
#: desert, for opposite physical reasons, which is why "arid" does not imply
#: "quiet".
MECHANISM_TROPOSPHERIC_SCINTILLATION = "tropospheric_scintillation"

DECLARABLE_MECHANISMS: frozenset[str] = frozenset(
    {
        MECHANISM_RAIN,
        MECHANISM_WET_SNOW,
        MECHANISM_DRY_SNOW,
        MECHANISM_FOG_OR_CLOUD,
        MECHANISM_DUST_OR_SAND,
        MECHANISM_EVAPORATION_DUCT,
        MECHANISM_TROPOSPHERIC_SCINTILLATION,
    }
)

#: Mechanisms that cannot exist without a water surface under the path. Declared
#: here so that a document claiming evaporation ducting over a land-locked area
#: of operations is rejected rather than believed — one of the two statements is
#: wrong, and the node is not entitled to choose which.
MARITIME_ONLY_MECHANISMS: frozenset[str] = frozenset({MECHANISM_EVAPORATION_DUCT})

#: How often the operator expects the mechanism in that season. Three levels,
#: not a probability: an operator can honestly rank "the usual state of this
#: season", "happens most years" and "has been known to happen", and cannot
#: honestly produce a percentage for wet snow in a coastal desert.
PREVALENCE_TYPICAL = "typical"
PREVALENCE_OCCASIONAL = "occasional"
PREVALENCE_RARE = "rare"

DECLARABLE_PREVALENCES: frozenset[str] = frozenset(
    {PREVALENCE_TYPICAL, PREVALENCE_OCCASIONAL, PREVALENCE_RARE}
)

#: What the declaration says about the mechanism on this date. Every value is
#: prefixed ``climatologically_`` so that no field, log line or dashboard can
#: quote one of these and be read as a report of present weather.
PLAUSIBILITY_TYPICAL = "climatologically_typical"
PLAUSIBILITY_OCCASIONAL = "climatologically_occasional"
PLAUSIBILITY_RARE = "climatologically_rare"
PLAUSIBILITY_OUT_OF_SEASON = "climatologically_out_of_season"
PLAUSIBILITY_NOT_CREDIBLE = "climatologically_not_credible"
#: A refusal, not a verdict. The declaration was missing, lapsed, about
#: somewhere else, or silent on this mechanism's vocabulary.
PLAUSIBILITY_NOT_ASSESSED = "not_assessed"

#: Plausibilities under which the declaration agrees with a measurement.
SUPPORTING_PLAUSIBILITIES: frozenset[str] = frozenset(
    {PLAUSIBILITY_TYPICAL, PLAUSIBILITY_OCCASIONAL}
)
#: Plausibilities under which it does not. ``rare`` is included: a rare
#: mechanism is still a real one, so this marks the case for an operator's
#: attention rather than deciding it.
CONFLICTING_PLAUSIBILITIES: frozenset[str] = frozenset(
    {PLAUSIBILITY_RARE, PLAUSIBILITY_OUT_OF_SEASON, PLAUSIBILITY_NOT_CREDIBLE}
)

#: How the measurement and the declaration ended up standing to each other.
CORROBORATION_SUPPORTED = "climate_supports_the_physics"
CORROBORATION_CONFLICT = "climate_conflicts_with_the_physics"
CORROBORATION_NOT_ASSESSED = "climate_not_assessed"
CORROBORATION_NO_MECHANISM = "physics_did_not_establish_the_mechanism"

#: Mean Earth radius, in kilometres (IUGG). Containment is computed on the
#: sphere rather than with the local flat projection used elsewhere in this
#: repository: those distances are metres to a few kilometres, these run to a
#: thousand, and at that range the flat approximation is wrong by percent —
#: which at the boundary is wrong about which climate applies.
EARTH_RADIUS_KM = 6371.0088

#: Largest area of operations a single declaration may claim. A circle wider
#: than two thousand kilometres crosses, at any latitude on Earth, from one
#: climate regime into another — tropical convection into subtropical
#: subsidence, maritime into continental. One mechanism set cannot be true
#: across it, so a document that claims it is not describing a climate.
MAX_AREA_RADIUS_KM = 1000.0

#: Smallest declared area. Climatology does not vary on a scale below this, so a
#: tighter radius buys no fidelity and only makes the containment test fail for
#: a platform that has not left the climate at all.
MIN_AREA_RADIUS_KM = 1.0

#: Bounds on the declared rain-rate figure, in mm/h. The upper bound sits above
#: the highest reliably measured one-hour rainfall on record — around 300 mm —
#: with headroom, and still catches an operator who has entered a monthly or
#: annual total in a field asking for an hourly rate.
MIN_RAIN_RATE_MM_PER_HOUR = 0.1
MAX_RAIN_RATE_MM_PER_HOUR = 500.0

#: Bounds on the exceedance percentage the rain rate is quoted against. ITU-R
#: publishes its reference rain maps at 0.01% of an average year; below 0.001%
#: the underlying statistics have nothing left to say, and above 5% the figure
#: describes drizzle and carries no information about fade-producing rain.
MIN_RAIN_EXCEEDANCE_PERCENT = 0.001
MAX_RAIN_EXCEEDANCE_PERCENT = 5.0

#: How far above the declared exceedance rate an inferred rain rate must sit
#: before the *intensity* is called atypical, even where rain itself is typical.
#: A fade deep enough to notice is by construction a tail event — exceeding the
#: 0.01% figure is what 0.01% of the time means — so the test must not fire on
#: the tail the declaration already describes. Across ITU-R rain climates the
#: 0.001% rate runs roughly one and a half to two times the 0.01% rate, so three
#: times the declared figure is past the far end of the distribution the
#: operator characterised.
RAIN_RATE_EXTREME_FACTOR = 3.0

#: Notice given before a declaration lapses. Seven days is a resupply cycle: it
#: is long enough for an operator to push a replacement before the node is
#: reasoning without context, and short enough that the warning still means
#: something when it appears.
EXPIRY_WARNING_SECONDS = 7.0 * 86400.0

#: Clock skew tolerated on the creation timestamp. A declaration created in the
#: node's future is a provenance or clock fault, and the node's clock is the
#: less trustworthy of the two on a cold start, so a few minutes are allowed
#: before the document is refused.
CREATION_SKEW_TOLERANCE_SECONDS = 300.0

#: Month lengths on a leap year, so that 29 February is expressible as a season
#: boundary. Seasons are stored as calendar dates rather than ordinal day
#: numbers precisely because the ordinal for a given date moves by one after
#: February, and a monsoon does not.
_MONTH_LENGTHS: tuple[int, ...] = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

#: Every calendar date a season can be asked about, including 29 February.
_CALENDAR_DAYS: tuple[tuple[int, int], ...] = tuple(
    (month, day)
    for month, length in enumerate(_MONTH_LENGTHS, start=1)
    for day in range(1, length + 1)
)

#: Appended to every serialised result. These are the two claims this module is
#: structurally incapable of making, and putting them in the record costs less
#: than trusting every consumer of the record to remember them.
_STANDING_CAVEATS: tuple[str, ...] = (
    "THIS_IS_CLIMATOLOGY_NOT_AN_OBSERVATION_OF_PRESENT_WEATHER",
    "NO_ATMOSPHERIC_QUANTITY_WAS_MEASURED_BY_THIS_NODE",
)


class ClimateError(ValueError):
    """Raised when a declaration cannot be trusted, or a caller wired this wrongly.

    Reserved for document faults and caller mistakes — an unknown mechanism in
    the signed file, a calendar with a hole in it, a timestamp with no timezone.
    Weakness in the *context* never raises: a lapsed declaration, an unknown
    position and an area that does not contain the platform all produce a result
    carrying its own refusal, because "we have no climatology for here" is an
    answer the operator needs and an exception is an answer they never see.
    """


def _bounded(value: Any, name: str, minimum: float, maximum: float) -> float:
    """:func:`bounded_number`, with the failure reported as a document fault.

    The shared validators raise bare ``ValueError``. Re-raising as
    :class:`ClimateError` gives a caller one exception type to catch for "this
    declaration cannot be trusted", instead of two that mean the same thing and
    are easy to half-handle.
    """
    try:
        return bounded_number(value, name, minimum, maximum)
    except (TypeError, ValueError) as exc:
        raise ClimateError(str(exc)) from exc


def _text(value: Any, name: str) -> str:
    """:func:`non_empty_str`, reported as a document fault for the same reason."""
    try:
        return non_empty_str(value, name)
    except (TypeError, ValueError) as exc:
        raise ClimateError(str(exc)) from exc


def _rounded(value: float | None, digits: int = 3) -> float | None:
    """Round for serialisation, mapping anything non-finite to ``None``.

    RFC 8259 has no encoding for infinity or NaN. Emitting one produces a
    document that some parsers accept with a surprising value and others reject
    outright, and both outcomes are worse than an explicit absence.
    """
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _deduplicate(reasons: Sequence[str]) -> tuple[str, ...]:
    """Preserve first-appearance order while dropping repeats."""
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return tuple(ordered)


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres, by the haversine formula.

    Chosen over the equirectangular projection used for emitter localisation
    because it holds at any separation and across the antimeridian and the
    poles. A declaration is a worldwide artefact; a containment test that
    degrades at high latitude would quietly stop applying in exactly the
    climates where the mechanism set differs most.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    inner = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, inner)))


def _month_day(name: str, raw: Mapping[str, Any], month_key: str, day_key: str) -> tuple[int, int]:
    try:
        month = int(raw[month_key])
        day = int(raw[day_key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClimateError(f"{name} must carry integer {month_key} and {day_key}") from exc
    if not 1 <= month <= 12:
        raise ClimateError(f"{name}: {month_key} must be between 1 and 12")
    if not 1 <= day <= _MONTH_LENGTHS[month - 1]:
        raise ClimateError(
            f"{name}: {day_key} {day} does not exist in month {month}"
        )
    return month, day


@dataclass(frozen=True)
class ClimateSeason:
    """One stretch of the calendar over which the operator's answer is constant.

    Seasons are *named by the operator* and bounded by calendar dates, and both
    choices are load-bearing. A fixed vocabulary of winter/spring/summer/autumn
    is hemisphere-relative — a Southern Hemisphere "winter" declaration read by
    a node that assumes the Northern one is wrong by six months — and it is
    simply absent in the tropics, where the year divides into wet and dry, or
    into a northeast and a southwest monsoon, on dates that match no
    meteorological quarter. So the node stores an interval of the calendar and
    whatever the operator calls it, and interprets neither.

    A range whose end precedes its start wraps the year end, which is what a
    November-to-March monsoon requires and what a hemisphere-neutral scheme
    cannot do without.
    """

    season_id: str
    starts_month: int
    starts_day: int
    ends_month: int
    ends_day: int
    #: Sorted ``(mechanism, prevalence)`` pairs. A tuple rather than a mapping so
    #: the season stays hashable and genuinely immutable; a frozen dataclass
    #: holding a dict is neither.
    mechanism_prevalence: tuple[tuple[str, str], ...]
    rain_rate_mm_per_hour: float | None
    rain_rate_exceedance_percent: float | None

    def __post_init__(self) -> None:
        _text(self.season_id, "season_id")
        for month, day, label in (
            (self.starts_month, self.starts_day, "start"),
            (self.ends_month, self.ends_day, "end"),
        ):
            if not 1 <= month <= 12 or not 1 <= day <= _MONTH_LENGTHS[month - 1]:
                raise ClimateError(f"season {self.season_id} has an impossible {label} date")
        declared = [mechanism for mechanism, _ in self.mechanism_prevalence]
        if len(declared) != len(set(declared)):
            raise ClimateError(f"season {self.season_id} names a mechanism twice")
        for mechanism, prevalence in self.mechanism_prevalence:
            if mechanism not in DECLARABLE_MECHANISMS:
                raise ClimateError(
                    f"season {self.season_id} names an unknown mechanism {mechanism!r}"
                )
            if prevalence not in DECLARABLE_PREVALENCES:
                raise ClimateError(
                    f"season {self.season_id} gives {mechanism} an unknown prevalence "
                    f"{prevalence!r}"
                )
        if not declared:
            # A season in which not one of the seven mechanisms is even rare is a
            # season in which the atmosphere does nothing. That is not true
            # anywhere; it is what a truncated document looks like.
            raise ClimateError(
                f"season {self.season_id} declares no credible mechanism at all"
            )
        rain_declared = MECHANISM_RAIN in set(declared)
        has_rate = self.rain_rate_mm_per_hour is not None
        has_exceedance = self.rain_rate_exceedance_percent is not None
        if has_rate != has_exceedance:
            # A rate without its exceedance percentage is a bare number wearing
            # the costume of a measurement. The pair is the claim.
            raise ClimateError(
                f"season {self.season_id} must give a rain rate and its exceedance "
                "percentage together or give neither"
            )
        if rain_declared and not has_rate:
            raise ClimateError(
                f"season {self.season_id} declares rain credible but states no rain rate"
            )
        if has_rate and not rain_declared:
            # One of the two statements is wrong and the node cannot tell which,
            # so it refuses the document rather than silently preferring one.
            raise ClimateError(
                f"season {self.season_id} states a rain rate while declaring rain "
                "not credible"
            )
        if has_rate:
            _bounded(
                self.rain_rate_mm_per_hour,
                "rain_rate_mm_per_hour",
                MIN_RAIN_RATE_MM_PER_HOUR,
                MAX_RAIN_RATE_MM_PER_HOUR,
            )
            _bounded(
                self.rain_rate_exceedance_percent,
                "rain_rate_exceedance_percent",
                MIN_RAIN_EXCEEDANCE_PERCENT,
                MAX_RAIN_EXCEEDANCE_PERCENT,
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ClimateSeason":
        season_id = _text(raw.get("season_id"), "season_id")
        starts = _month_day(f"season {season_id}", raw, "starts_month", "starts_day")
        ends = _month_day(f"season {season_id}", raw, "ends_month", "ends_day")
        mechanisms = raw.get("mechanisms")
        if not isinstance(mechanisms, Mapping):
            raise ClimateError(
                f"season {season_id} must carry a mechanisms map of mechanism to prevalence"
            )
        pairs = tuple(
            sorted((str(key), str(value)) for key, value in mechanisms.items())
        )
        rain_rate = raw.get("rain_rate")
        rate: float | None = None
        exceedance: float | None = None
        if rain_rate is not None:
            if not isinstance(rain_rate, Mapping):
                raise ClimateError(f"season {season_id}: rain_rate must be an object")
            missing = sorted(
                {"mm_per_hour", "exceeded_percent_of_season_hours"} - set(rain_rate.keys())
            )
            if missing:
                raise ClimateError(
                    f"season {season_id}: rain_rate is missing {', '.join(missing)}"
                )
            rate = float(rain_rate["mm_per_hour"])
            exceedance = float(rain_rate["exceeded_percent_of_season_hours"])
        return cls(
            season_id=season_id,
            starts_month=starts[0],
            starts_day=starts[1],
            ends_month=ends[0],
            ends_day=ends[1],
            mechanism_prevalence=pairs,
            rain_rate_mm_per_hour=rate,
            rain_rate_exceedance_percent=exceedance,
        )

    def contains(self, month: int, day: int) -> bool:
        """Whether this season covers the given calendar date.

        Comparison is on the ``(month, day)`` pair rather than an ordinal day
        number so that 29 February falls where it belongs — after the 28th and
        before 1 March — without the whole calendar shifting in leap years.
        """
        probe = (month, day)
        start = (self.starts_month, self.starts_day)
        end = (self.ends_month, self.ends_day)
        if start <= end:
            return start <= probe <= end
        # An end before a start wraps the year boundary: a northeast monsoon
        # running 1 November to 15 March is one season, not two.
        return probe >= start or probe <= end

    def mechanisms(self) -> frozenset[str]:
        return frozenset(mechanism for mechanism, _ in self.mechanism_prevalence)

    def prevalence(self, mechanism: str) -> str | None:
        """The declared prevalence, or ``None`` where the season is silent.

        Silence is a claim: the operator has said this mechanism is not credible
        in this season. It is reported as such rather than as missing data.
        """
        for declared, prevalence in self.mechanism_prevalence:
            if declared == mechanism:
                return prevalence
        return None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "season_id": self.season_id,
            "starts_month": self.starts_month,
            "starts_day": self.starts_day,
            "ends_month": self.ends_month,
            "ends_day": self.ends_day,
            "mechanisms": {
                mechanism: prevalence for mechanism, prevalence in self.mechanism_prevalence
            },
        }
        if self.rain_rate_mm_per_hour is not None:
            result["rain_rate"] = {
                "mm_per_hour": _rounded(self.rain_rate_mm_per_hour),
                "exceeded_percent_of_season_hours": _rounded(
                    self.rain_rate_exceedance_percent, 6
                ),
            }
        return result


@dataclass(frozen=True)
class ClimateDeclaration:
    """The operator's signed statement of what is typical in the area of operations.

    Verified the same way the spectrum policy and the capability lease are: the
    envelope is checked against the provisioned operator key elsewhere, and this
    class validates the payload it was given. Nothing here reaches the network,
    and nothing here is inferred from anything but the document.

    The area is a centre and a radius rather than a bounding box. A box needs
    special cases at the antimeridian and at the poles, and a special case in a
    containment test is a place where a declaration silently stops applying; a
    single spherical distance has none.
    """

    declaration_id: str
    authority: str
    #: Where the figures came from, in the operator's own words — a published
    #: climate map and the point it was read at, a station record, a fleet's own
    #: history. Never parsed and never dereferenced by the node, which has no
    #: network. It exists so that a claim in the record has someone to check it
    #: against, and so a superseding declaration can be argued with.
    basis: str
    area_label: str
    centre_latitude_deg: float
    centre_longitude_deg: float
    radius_km: float
    #: Whether there is water under the path. Mandatory and never defaulted:
    #: defaulting it false deletes evaporation ducting from every maritime
    #: deployment whose document omitted the field, and defaulting it true
    #: invents a sea.
    maritime: bool
    created_at: datetime
    expires_at: datetime
    seasons: tuple[ClimateSeason, ...]

    def __post_init__(self) -> None:
        _text(self.declaration_id, "declaration_id")
        _text(self.authority, "authority")
        _text(self.basis, "basis")
        _text(self.area_label, "area_label")
        _bounded(self.centre_latitude_deg, "centre_latitude_deg", -90.0, 90.0)
        _bounded(self.centre_longitude_deg, "centre_longitude_deg", -180.0, 180.0)
        _bounded(self.radius_km, "radius_km", MIN_AREA_RADIUS_KM, MAX_AREA_RADIUS_KM)
        if self.maritime is not True and self.maritime is not False:
            raise ClimateError("maritime must be an explicit boolean")
        for moment, name in ((self.created_at, "created_at"), (self.expires_at, "expires_at")):
            if moment.tzinfo is None:
                raise ClimateError(f"{name} must carry a timezone")
        if self.expires_at <= self.created_at:
            raise ClimateError("a climate declaration must expire after it was created")
        if not self.seasons:
            raise ClimateError("a climate declaration must declare at least one season")
        ids = [season.season_id for season in self.seasons]
        if len(ids) != len(set(ids)):
            raise ClimateError("a climate declaration contains duplicate season IDs")
        self._require_total_calendar_coverage()
        declared = self.declared_mechanisms()
        if not declared:
            raise ClimateError(
                "a climate declaration must name at least one credible mechanism"
            )
        if not self.maritime:
            over_water = sorted(declared & MARITIME_ONLY_MECHANISMS)
            if over_water:
                raise ClimateError(
                    "a non-maritime declaration cannot claim "
                    f"{', '.join(over_water)}"
                )

    def _require_total_calendar_coverage(self) -> None:
        """Every date must fall in exactly one season.

        A gap would leave dates on which the node has no answer, and a silent
        "not assessed" for half of a year is indistinguishable from a document
        that was never provisioned. An overlap would leave two different
        mechanism sets both true, and picking between them is the node
        inventing a season. Both are refused at the document, where an operator
        can still fix them, rather than in flight.
        """
        for month, day in _CALENDAR_DAYS:
            covering = [season.season_id for season in self.seasons if season.contains(month, day)]
            if not covering:
                raise ClimateError(
                    f"no declared season covers {month:02d}-{day:02d}; seasons must "
                    "cover the whole calendar"
                )
            if len(covering) > 1:
                raise ClimateError(
                    f"seasons {', '.join(sorted(covering))} overlap on {month:02d}-{day:02d}"
                )

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, now: datetime | None = None
    ) -> "ClimateDeclaration":
        """Validate a verified declaration payload.

        Note the asymmetry in the time checks, which is deliberate. A creation
        timestamp in the node's future is a provenance or clock fault and the
        document is refused outright. An *expiry* in the past is not: the
        document still loads, and it is the assessment that refuses to reason
        from it. Refusing to load a lapsed declaration would collapse "this
        node was never given a climate" and "the climate it was given lapsed
        three days ago" into the same silence, and those two call for different
        actions from the operator — provision one, or push a fresh one.
        """
        current = now or utc_now()
        schema = raw.get("schema")
        if schema is not None and str(schema) != CLIMATE_DECLARATION_SCHEMA:
            raise ClimateError(
                f"unsupported climate declaration schema {schema!r}; expected "
                f"{CLIMATE_DECLARATION_SCHEMA}"
            )
        required = {
            "declaration_id",
            "authority",
            "basis",
            "area_label",
            "centre_latitude_deg",
            "centre_longitude_deg",
            "radius_km",
            "maritime",
            "created_at",
            "expires_at",
            "seasons",
        }
        missing = sorted(required - set(raw.keys()))
        if missing:
            raise ClimateError(
                f"climate declaration is missing fields: {', '.join(missing)}"
            )
        try:
            created_at = parse_timestamp(str(raw["created_at"]))
            expires_at = parse_timestamp(str(raw["expires_at"]))
        except ValueError as exc:
            raise ClimateError(f"climate declaration timestamps are unusable: {exc}") from exc
        if (created_at - current).total_seconds() > CREATION_SKEW_TOLERANCE_SECONDS:
            raise ClimateError("climate declaration was created in the future")
        raw_seasons = raw["seasons"]
        if not isinstance(raw_seasons, Sequence) or isinstance(raw_seasons, (str, bytes)):
            raise ClimateError("climate declaration seasons must be a list")
        maritime = raw["maritime"]
        if maritime is not True and maritime is not False:
            # Identity, not truthiness: the string "false" is truthy, and a
            # declaration that accidentally asserted a sea would make
            # evaporation ducting a candidate over a desert.
            raise ClimateError("maritime must be declared as an explicit true or false")
        return cls(
            declaration_id=_text(raw["declaration_id"], "declaration_id"),
            authority=_text(raw["authority"], "authority"),
            basis=_text(raw["basis"], "basis"),
            area_label=_text(raw["area_label"], "area_label"),
            centre_latitude_deg=_bounded(
                raw["centre_latitude_deg"], "centre_latitude_deg", -90.0, 90.0
            ),
            centre_longitude_deg=_bounded(
                raw["centre_longitude_deg"], "centre_longitude_deg", -180.0, 180.0
            ),
            radius_km=_bounded(
                raw["radius_km"], "radius_km", MIN_AREA_RADIUS_KM, MAX_AREA_RADIUS_KM
            ),
            maritime=maritime,
            created_at=created_at,
            expires_at=expires_at,
            seasons=tuple(ClimateSeason.from_dict(item) for item in raw_seasons),
        )

    def declared_mechanisms(self) -> frozenset[str]:
        """Every mechanism credible in this area of operations in any season."""
        return frozenset().union(*(season.mechanisms() for season in self.seasons))

    def seasons_declaring(self, mechanism: str) -> tuple[str, ...]:
        return tuple(
            season.season_id for season in self.seasons if mechanism in season.mechanisms()
        )

    def season_for(self, moment: datetime) -> ClimateSeason | None:
        """The season covering ``moment``, read in UTC.

        The date is taken in UTC rather than in an unknown local zone. At a
        season boundary that can put a platform on the far side of the world in
        the neighbouring season for part of a day — which is accepted knowingly:
        a monsoon onset is a climatological convention with weeks of real
        scatter around it, and inventing a timezone for the area of operations
        would be a fabricated fact in exchange for a false precision.
        """
        if moment.tzinfo is None:
            raise ClimateError("the moment being assessed must carry a timezone")
        utc = moment.astimezone(timezone.utc)
        for season in self.seasons:
            if season.contains(utc.month, utc.day):
                return season
        return None

    def contains(self, latitude_deg: float, longitude_deg: float) -> tuple[bool, float]:
        """Whether a position is inside the declared area, and how far off centre.

        No tolerance is added to the declared radius. An operator who wants
        margin declares a larger radius; a margin invented here would extend
        every declaration in the fleet into terrain nobody described.
        """
        distance = great_circle_km(
            self.centre_latitude_deg,
            self.centre_longitude_deg,
            _bounded(latitude_deg, "latitude_deg", -90.0, 90.0),
            _bounded(longitude_deg, "longitude_deg", -180.0, 180.0),
        )
        return distance <= self.radius_km, distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CLIMATE_DECLARATION_SCHEMA,
            "declaration_id": self.declaration_id,
            "authority": self.authority,
            "basis": self.basis,
            "area_label": self.area_label,
            "centre_latitude_deg": _rounded(self.centre_latitude_deg, 7),
            "centre_longitude_deg": _rounded(self.centre_longitude_deg, 7),
            "radius_km": _rounded(self.radius_km),
            "maritime": self.maritime,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "seasons": [season.to_dict() for season in self.seasons],
        }


@dataclass(frozen=True)
class ClimatePlausibility:
    """What the declaration says about one mechanism on one date.

    Every field name states that this is climatology. There is no field here
    whose value could be quoted as the weather, because the single most likely
    way for this module to cause harm is for one of its outputs to be read as an
    observation somewhere downstream.
    """

    mechanism: str
    #: False for every refusal — no declaration, lapsed, outside the area,
    #: outside the vocabulary. A consumer that checks nothing else still cannot
    #: mistake a refusal for a finding.
    assessed: bool
    plausibility: str
    declaration_id: str | None
    area_label: str | None
    season_id: str | None
    season_prevalence: str | None
    #: True only when a position was supplied *and* fell inside the declared
    #: area. Unknown position and outside the area are distinguished by
    #: ``assessed``, because they are different failures: one is a sensor gap,
    #: the other is a document that is about somewhere else.
    area_confirmed: bool
    distance_from_area_centre_km: float | None
    declared_radius_km: float | None
    declaration_expired: bool
    seconds_until_expiry: float | None
    maritime_declared: bool | None
    climatological_rain_rate_mm_per_hour: float | None
    rain_rate_exceedance_percent_of_season_hours: float | None
    inferred_rain_rate_mm_per_hour: float | None
    reasons: tuple[str, ...]

    def supports(self) -> bool:
        return self.assessed and self.plausibility in SUPPORTING_PLAUSIBILITIES

    def conflicts(self) -> bool:
        return self.assessed and self.plausibility in CONFLICTING_PLAUSIBILITIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-climate-plausibility/v1",
            "mechanism": self.mechanism,
            "assessed": self.assessed,
            "climatological_plausibility": self.plausibility,
            "declaration_id": self.declaration_id,
            "area_label": self.area_label,
            "season_id": self.season_id,
            "season_prevalence": self.season_prevalence,
            "area_confirmed": self.area_confirmed,
            "distance_from_area_centre_km": _rounded(self.distance_from_area_centre_km, 3),
            "declared_radius_km": _rounded(self.declared_radius_km, 3),
            "declaration_expired": self.declaration_expired,
            "seconds_until_expiry": _rounded(self.seconds_until_expiry, 1),
            "maritime_declared": self.maritime_declared,
            "climatological_rain_rate_mm_per_hour": _rounded(
                self.climatological_rain_rate_mm_per_hour, 3
            ),
            "rain_rate_exceedance_percent_of_season_hours": _rounded(
                self.rain_rate_exceedance_percent_of_season_hours, 6
            ),
            "inferred_rain_rate_mm_per_hour": _rounded(self.inferred_rain_rate_mm_per_hour, 3),
            # Held at ``null`` structurally. This module has no instrument and
            # reads no feed, so there is no present state for it to report.
            "observed_weather": None,
            "reasons": list(self.reasons) + list(_STANDING_CAVEATS),
        }


@dataclass(frozen=True)
class MechanismCorroboration:
    """A physics finding and its climatological context, side by side.

    The two are never merged into a single verdict. ``physics_consistent`` is
    carried through untouched and ``physics_retained`` restates it, so the
    record itself proves the measurement was not suppressed by the document.
    """

    mechanism: str
    physics_consistent: bool
    #: The physics finding as this record carries it. Identical to
    #: ``physics_consistent`` in every case by construction, and deliberately
    #: not derived at read time: the invariant this module has to keep is that
    #: no climatological outcome can ever change it, and an invariant nothing
    #: states is an invariant nothing tests.
    physics_retained: bool
    corroboration: str
    conflict: bool
    plausibility: ClimatePlausibility
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-mechanism-corroboration/v1",
            "mechanism": self.mechanism,
            "physics_consistent": self.physics_consistent,
            "physics_retained": self.physics_retained,
            "corroboration": self.corroboration,
            "conflict": self.conflict,
            "climate": self.plausibility.to_dict(),
            # The caveats ride on the outer record too. A consumer that reads
            # only the top level would otherwise see a mechanism, a season and
            # an agreement, with nothing left saying none of it was measured.
            "reasons": list(self.reasons) + list(_STANDING_CAVEATS),
        }


def _refusal(
    mechanism: str,
    reasons: Iterable[str],
    *,
    declaration: ClimateDeclaration | None = None,
    expired: bool = False,
    seconds_until_expiry: float | None = None,
    distance_km: float | None = None,
) -> ClimatePlausibility:
    """Build a "no climatological answer" result that cannot read as one.

    Every numeric field is left at ``None``. A zero distance or a blank season
    would read as measured-and-nothing-found, which is the one thing a refusal
    is not.
    """
    return ClimatePlausibility(
        mechanism=mechanism,
        assessed=False,
        plausibility=PLAUSIBILITY_NOT_ASSESSED,
        declaration_id=declaration.declaration_id if declaration else None,
        area_label=declaration.area_label if declaration else None,
        season_id=None,
        season_prevalence=None,
        area_confirmed=False,
        distance_from_area_centre_km=distance_km,
        declared_radius_km=declaration.radius_km if declaration else None,
        declaration_expired=expired,
        seconds_until_expiry=seconds_until_expiry,
        maritime_declared=declaration.maritime if declaration else None,
        climatological_rain_rate_mm_per_hour=None,
        rain_rate_exceedance_percent_of_season_hours=None,
        inferred_rain_rate_mm_per_hour=None,
        reasons=_deduplicate(tuple(reasons)),
    )


def assess_plausibility(
    declaration: ClimateDeclaration | None,
    mechanism: str,
    *,
    at: datetime,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
    inferred_rain_rate_mm_per_hour: float | None = None,
) -> ClimatePlausibility:
    """Is this mechanism typical *here*, in the season ``at`` falls in?

    ``at`` is both the calendar date whose season is looked up and the instant
    the declaration's validity is checked at. One parameter, because assessing a
    fade that happened last Tuesday against a document that was still valid last
    Tuesday is the honest comparison, and checking the season at one time and
    the expiry at another would let a lapsed document answer for a live event.

    A date *earlier* than the declaration's creation is not refused. The
    document was written later; the climate it describes was not.
    """
    if at.tzinfo is None:
        raise ClimateError("the moment being assessed must carry a timezone")
    if (latitude_deg is None) != (longitude_deg is None):
        # Half a fix is not a position. Treating the missing half as zero would
        # place the platform in the Gulf of Guinea, which is inside somebody's
        # declared area.
        raise ClimateError("a position needs both latitude_deg and longitude_deg")
    inferred_rate = (
        _bounded(
            inferred_rain_rate_mm_per_hour,
            "inferred_rain_rate_mm_per_hour",
            0.0,
            MAX_RAIN_RATE_MM_PER_HOUR * 10.0,
        )
        if inferred_rain_rate_mm_per_hour is not None
        else None
    )

    if mechanism not in DECLARABLE_MECHANISMS:
        # A refusal rather than an exception: the physics module names its own
        # mechanisms, and a vocabulary mismatch at the seam between two modules
        # must surface as an unanswered question in the record, not as a crash
        # in the sensing loop.
        return _refusal(
            mechanism,
            (f"MECHANISM_IS_OUTSIDE_THE_DECLARATION_VOCABULARY:{mechanism}",),
        )

    if declaration is None:
        return _refusal(
            mechanism,
            (
                "NO_CLIMATE_DECLARATION_IS_CONFIGURED_FOR_THIS_NODE",
                "PHYSICS_STANDS_ALONE_WITHOUT_CLIMATOLOGICAL_CONTEXT",
            ),
        )

    seconds_until_expiry = (declaration.expires_at - at).total_seconds()
    if seconds_until_expiry <= 0.0:
        # A lapsed declaration is not applied at all. The climate it described
        # may well still hold, but nobody is standing behind the document any
        # more, and an unsupervised statement about the world is exactly what
        # this module refuses to manufacture.
        return _refusal(
            mechanism,
            (
                f"CLIMATE_DECLARATION_LAPSED_AT:{declaration.expires_at.isoformat()}",
                "A_LAPSED_DECLARATION_IS_NOT_APPLIED_AND_THE_MEASUREMENT_STANDS_ALONE",
                f"DECLARATION_AUTHORITY:{declaration.authority}",
            ),
            declaration=declaration,
            expired=True,
            seconds_until_expiry=seconds_until_expiry,
        )

    reasons: list[str] = [
        f"CLIMATE_DECLARATION:{declaration.declaration_id}",
        f"DECLARATION_AUTHORITY:{declaration.authority}",
        f"AREA_OF_OPERATIONS:{declaration.area_label}",
    ]
    if seconds_until_expiry <= EXPIRY_WARNING_SECONDS:
        reasons.append("CLIMATE_DECLARATION_LAPSES_WITHIN_THE_WEEK")

    distance_km: float | None = None
    area_confirmed = False
    if latitude_deg is not None and longitude_deg is not None:
        inside, distance_km = declaration.contains(latitude_deg, longitude_deg)
        if not inside:
            # The document is about somewhere else. Applying it here would be
            # the paperwork following the platform out of the area it covers.
            return _refusal(
                mechanism,
                reasons
                + [
                    "PLATFORM_IS_OUTSIDE_THE_DECLARED_AREA_OF_OPERATIONS",
                    "A_DECLARATION_FOR_SOMEWHERE_ELSE_IS_NOT_APPLIED_HERE",
                ],
                declaration=declaration,
                seconds_until_expiry=seconds_until_expiry,
                distance_km=distance_km,
            )
        area_confirmed = True
        reasons.append("PLATFORM_POSITION_CONFIRMED_INSIDE_THE_DECLARED_AREA")
    else:
        # Contested spectrum is where satellite navigation is least available,
        # so an unknown position has to remain workable. The declaration is
        # still applied — it is the only climatology the node has — but the
        # record says the platform was never confirmed to be under it.
        reasons.append("PLATFORM_POSITION_UNKNOWN_SO_AREA_CONTAINMENT_WAS_NOT_CONFIRMED")

    season = declaration.season_for(at)
    if season is None:
        # Total calendar coverage is enforced when the declaration is built, so
        # nothing can reach this today. It is kept because a refusal is the
        # right answer if that rule is ever relaxed, and a fabricated season is
        # not.
        return _refusal(
            mechanism,
            reasons + ["NO_DECLARED_SEASON_COVERS_THIS_DATE"],
            declaration=declaration,
            seconds_until_expiry=seconds_until_expiry,
            distance_km=distance_km,
        )

    reasons.append(f"SEASON:{season.season_id}")
    prevalence = season.prevalence(mechanism)
    elsewhere = declaration.seasons_declaring(mechanism)

    if prevalence is not None:
        plausibility = {
            PREVALENCE_TYPICAL: PLAUSIBILITY_TYPICAL,
            PREVALENCE_OCCASIONAL: PLAUSIBILITY_OCCASIONAL,
            PREVALENCE_RARE: PLAUSIBILITY_RARE,
        }[prevalence]
        reasons.append(f"DECLARED_PREVALENCE_IN_SEASON:{prevalence}")
    elif elsewhere:
        plausibility = PLAUSIBILITY_OUT_OF_SEASON
        reasons.append(f"MECHANISM_IS_DECLARED_ONLY_IN_SEASONS:{','.join(sorted(elsewhere))}")
    else:
        plausibility = PLAUSIBILITY_NOT_CREDIBLE
        reasons.append("MECHANISM_IS_NOT_DECLARED_CREDIBLE_ANYWHERE_IN_THIS_AREA")
        if mechanism in MARITIME_ONLY_MECHANISMS and not declaration.maritime:
            reasons.append("AREA_IS_NOT_DECLARED_MARITIME")

    rain_rate = season.rain_rate_mm_per_hour
    exceedance = season.rain_rate_exceedance_percent
    if mechanism == MECHANISM_RAIN and inferred_rate is not None:
        plausibility, rain_reasons = _weigh_rain_rate(
            plausibility, inferred_rate, rain_rate, exceedance
        )
        reasons.extend(rain_reasons)
    elif inferred_rate is not None:
        reasons.append("A_RAIN_RATE_WAS_SUPPLIED_FOR_A_MECHANISM_THAT_IS_NOT_RAIN_AND_WAS_IGNORED")
        # Dropped from the record rather than echoed. The field sits beside the
        # declared climatological rate, and a number in both would read as a
        # comparison that was made when none was.
        inferred_rate = None

    return ClimatePlausibility(
        mechanism=mechanism,
        assessed=True,
        plausibility=plausibility,
        declaration_id=declaration.declaration_id,
        area_label=declaration.area_label,
        season_id=season.season_id,
        season_prevalence=prevalence,
        area_confirmed=area_confirmed,
        distance_from_area_centre_km=distance_km,
        declared_radius_km=declaration.radius_km,
        declaration_expired=False,
        seconds_until_expiry=seconds_until_expiry,
        maritime_declared=declaration.maritime,
        climatological_rain_rate_mm_per_hour=rain_rate,
        rain_rate_exceedance_percent_of_season_hours=exceedance,
        inferred_rain_rate_mm_per_hour=inferred_rate,
        reasons=_deduplicate(reasons),
    )


def _weigh_rain_rate(
    plausibility: str,
    inferred: float,
    declared: float | None,
    exceedance: float | None,
) -> tuple[str, list[str]]:
    """Compare an inferred rain rate against the declared exceedance figure.

    The comparison has to be generous in one specific way, and getting this
    backwards would make the module wrong on precisely the events it exists for.
    The declared figure is a rate *exceeded* some small percentage of the
    season's hours — and a fade deep enough for a receiver to notice is, by
    construction, drawn from that tail. Exceeding the figure is therefore
    expected during a rain fade and is reported as such, not flagged. Only a
    rate far past the tail the operator characterised says anything, and even
    then it downgrades the plausibility to rare and never to not-credible: rain
    that falls here fell here.
    """
    reasons: list[str] = []
    if declared is None or exceedance is None:
        reasons.append("NO_RAIN_RATE_IS_DECLARED_FOR_THIS_SEASON_SO_INTENSITY_WAS_NOT_WEIGHED")
        return plausibility, reasons

    reasons.append(f"RAIN_RATE_EXCEEDANCE_BASIS_PERCENT_OF_SEASON_HOURS:{exceedance}")
    if inferred <= declared:
        reasons.append("INFERRED_RAIN_RATE_IS_WITHIN_THE_DECLARED_CLIMATOLOGICAL_EXCEEDANCE")
        return plausibility, reasons

    if inferred <= declared * RAIN_RATE_EXTREME_FACTOR:
        reasons.append("INFERRED_RAIN_RATE_SITS_IN_THE_TAIL_THE_DECLARATION_DESCRIBES")
        reasons.append("A_FADE_DEEP_ENOUGH_TO_DETECT_IS_EXPECTED_TO_COME_FROM_THAT_TAIL")
        return plausibility, reasons

    reasons.append("INFERRED_RAIN_RATE_IS_FAR_ABOVE_THE_DECLARED_CLIMATOLOGICAL_EXCEEDANCE")
    reasons.append("RAIN_REMAINS_CREDIBLE_HERE_BUT_NOT_AT_THIS_INTENSITY")
    if plausibility in SUPPORTING_PLAUSIBILITIES:
        return PLAUSIBILITY_RARE, reasons
    return plausibility, reasons


def corroborate(
    mechanism: str,
    *,
    physics_consistent: bool,
    plausibility: ClimatePlausibility,
) -> MechanismCorroboration:
    """Stand a physics finding and its climatological context next to each other.

    Four cases, and the two that matter are the ones where the sides disagree:

    * **Consistent and typical.** The strongest statement available — and it is
      still only "consistent with", because two pieces of circumstantial
      evidence agreeing is not an observation of rain.
    * **Consistent and implausible.** Reported as a conflict, with the physics
      carried through whole. The declaration can be stale, the platform can have
      left the area it describes, and rare events occur; the node says all three
      and suppresses nothing. This is the case the module was written for.
    * **Not consistent.** Climatology cannot supply a mechanism the measurement
      does not support. A monsoon season is not evidence that anything is
      falling on this path, and allowing the declaration to promote a mechanism
      here would be the same failure as suppression, pointing the other way.
    * **Not assessed.** Physics alone, and the record says why the context was
      unavailable rather than leaving it blank.
    """
    if physics_consistent is not True and physics_consistent is not False:
        # Identity, not truthiness: a caller passing a verdict object rather
        # than its boolean would otherwise corroborate everything forever.
        raise ClimateError("physics_consistent must be an explicit boolean")
    if plausibility.mechanism != mechanism:
        raise ClimateError(
            f"plausibility describes {plausibility.mechanism!r}, not {mechanism!r}"
        )

    if not physics_consistent:
        return MechanismCorroboration(
            mechanism=mechanism,
            physics_consistent=False,
            physics_retained=False,
            corroboration=CORROBORATION_NO_MECHANISM,
            conflict=False,
            plausibility=plausibility,
            reasons=_deduplicate(
                (
                    "THE_MEASUREMENT_DID_NOT_SUPPORT_THIS_MECHANISM",
                    "CLIMATOLOGY_CANNOT_ESTABLISH_A_MECHANISM_THE_MEASUREMENT_DOES_NOT_SUPPORT",
                    "A_TYPICAL_SEASON_IS_NOT_EVIDENCE_THAT_ANYTHING_IS_FALLING_ON_THIS_PATH",
                )
            ),
        )

    if not plausibility.assessed:
        return MechanismCorroboration(
            mechanism=mechanism,
            physics_consistent=True,
            physics_retained=True,
            corroboration=CORROBORATION_NOT_ASSESSED,
            conflict=False,
            plausibility=plausibility,
            reasons=_deduplicate(
                ("PHYSICS_STANDS_ALONE_NO_CLIMATOLOGICAL_CONTEXT_WAS_AVAILABLE",)
                + plausibility.reasons
            ),
        )

    shared: tuple[str, ...] = (
        f"CLIMATOLOGICAL_PLAUSIBILITY:{plausibility.plausibility}",
        f"SEASON:{plausibility.season_id}",
    )
    if not plausibility.area_confirmed:
        shared = shared + ("PLATFORM_POSITION_WAS_NEVER_CONFIRMED_INSIDE_THE_DECLARED_AREA",)

    if plausibility.supports():
        return MechanismCorroboration(
            mechanism=mechanism,
            physics_consistent=True,
            physics_retained=True,
            corroboration=CORROBORATION_SUPPORTED,
            conflict=False,
            plausibility=plausibility,
            reasons=_deduplicate(
                (
                    "THE_MEASUREMENT_IS_CONSISTENT_WITH_A_MECHANISM_TYPICAL_OF_THIS_AREA_AND_SEASON",
                    "TWO_STRANDS_OF_CIRCUMSTANTIAL_EVIDENCE_AGREEING_IS_NOT_AN_OBSERVATION",
                )
                + shared
                + plausibility.reasons
            ),
        )

    return MechanismCorroboration(
        mechanism=mechanism,
        physics_consistent=True,
        physics_retained=True,
        corroboration=CORROBORATION_CONFLICT,
        conflict=True,
        plausibility=plausibility,
        reasons=_deduplicate(
            (
                "THE_MEASUREMENT_IS_CONSISTENT_WITH_A_MECHANISM_THE_DECLARATION_CALLS_UNLIKELY",
                "THE_MEASUREMENT_IS_RETAINED_IN_FULL_THE_DECLARATION_IS_NOT_AN_OVERRIDE",
                "THE_DECLARATION_MAY_BE_STALE_OR_THE_PLATFORM_MAY_HAVE_LEFT_THE_DECLARED_AREA",
                "CLIMATOLOGY_DESCRIBES_A_DISTRIBUTION_AND_A_RARE_EVENT_IS_A_MEMBER_OF_IT",
            )
            + shared
            + plausibility.reasons
        ),
    )


def corroborate_mechanism(
    declaration: ClimateDeclaration | None,
    mechanism: str,
    *,
    physics_consistent: bool,
    at: datetime,
    latitude_deg: float | None = None,
    longitude_deg: float | None = None,
    inferred_rain_rate_mm_per_hour: float | None = None,
) -> MechanismCorroboration:
    """Assess and corroborate in one call — the entry point for the physics side.

    A module that has established a mechanism passes what it knows: the
    mechanism, whether its own evidence supported it, when the fade happened,
    where the platform was if that is known, and an inferred rain rate if it
    derived one. Everything else is read from the signed declaration, and a
    missing declaration is answered rather than raised.
    """
    plausibility = assess_plausibility(
        declaration,
        mechanism,
        at=at,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        inferred_rain_rate_mm_per_hour=inferred_rain_rate_mm_per_hour,
    )
    return corroborate(
        mechanism, physics_consistent=physics_consistent, plausibility=plausibility
    )
