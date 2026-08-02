"""Whether a mechanism is plausible here and now, and never whether it is happening.

Two failures would make this module worse than not having it. The first is
suppression: letting a signed file overrule a real measurement, so a node that
actually saw a rain fade reports nothing because the document said the season is
dry. The second is invention: letting a typical season promote a mechanism the
measurement never supported, so every fade in the monsoon becomes rain. These
tests push on both, and on the refusals — a lapsed declaration, an area the
platform has left, a calendar with a hole in it — at least as hard as on the
agreements, because a context module that always answers is one whose answers
carry no information.

See ``src/kval_node/climate.py``.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import pathlib
import sys
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from kval_node.climate import (
    CLIMATE_DECLARATION_SCHEMA,
    CORROBORATION_CONFLICT,
    CORROBORATION_NO_MECHANISM,
    CORROBORATION_NOT_ASSESSED,
    CORROBORATION_SUPPORTED,
    DECLARABLE_MECHANISMS,
    MARITIME_ONLY_MECHANISMS,
    MAX_AREA_RADIUS_KM,
    MECHANISM_DRY_SNOW,
    MECHANISM_DUST_OR_SAND,
    MECHANISM_EVAPORATION_DUCT,
    MECHANISM_FOG_OR_CLOUD,
    MECHANISM_RAIN,
    MECHANISM_TROPOSPHERIC_SCINTILLATION,
    MECHANISM_WET_SNOW,
    PLAUSIBILITY_NOT_ASSESSED,
    PLAUSIBILITY_NOT_CREDIBLE,
    PLAUSIBILITY_OCCASIONAL,
    PLAUSIBILITY_OUT_OF_SEASON,
    PLAUSIBILITY_RARE,
    PLAUSIBILITY_TYPICAL,
    RAIN_RATE_EXTREME_FACTOR,
    ClimateDeclaration,
    ClimateError,
    ClimateSeason,
    assess_plausibility,
    corroborate,
    corroborate_mechanism,
    great_circle_km,
)

#: A November afternoon in the Singapore Strait: inside the northeast monsoon,
#: inside every declaration below that covers it.
NOVEMBER = datetime(2026, 11, 20, 6, 30, tzinfo=timezone.utc)
#: A July afternoon. The same instant is deep winter in one hemisphere, the
#: southwest monsoon in another, and the middle of the dry season in a third —
#: which is the whole reason seasons are declared rather than assumed.
JULY = datetime(2026, 7, 14, 6, 30, tzinfo=timezone.utc)
#: Leap day, which exists in the calendar a season is written against and not in
#: the year the node happens to be running.
LEAP_DAY = datetime(2028, 2, 29, 12, 0, tzinfo=timezone.utc)

CREATED = "2026-08-01T00:00:00+00:00"
EXPIRES = "2027-08-01T00:00:00+00:00"

#: Roughly one degree of latitude in kilometres, for placing a position a known
#: distance from a declared centre without importing a projection.
KM_PER_DEGREE_LAT = math.pi * 6371.0088 / 180.0


def tropical_maritime() -> dict[str, Any]:
    """A wet, warm, maritime area of operations with a monsoon calendar."""
    return {
        "schema": CLIMATE_DECLARATION_SCHEMA,
        "declaration_id": "ao-singapore-strait-2026",
        "authority": "J6-SPECTRUM-CELL",
        "basis": "ITU-R P.837-7 read at 1.25N 103.85E; ten-year station record",
        "area_label": "Singapore Strait",
        "centre_latitude_deg": 1.25,
        "centre_longitude_deg": 103.85,
        "radius_km": 120.0,
        "maritime": True,
        "created_at": CREATED,
        "expires_at": EXPIRES,
        "seasons": [
            {
                "season_id": "northeast_monsoon",
                "starts_month": 11,
                "starts_day": 1,
                "ends_month": 3,
                "ends_day": 15,
                "mechanisms": {
                    MECHANISM_RAIN: "typical",
                    MECHANISM_FOG_OR_CLOUD: "typical",
                    MECHANISM_EVAPORATION_DUCT: "typical",
                    MECHANISM_TROPOSPHERIC_SCINTILLATION: "occasional",
                },
                "rain_rate": {
                    "mm_per_hour": 95.0,
                    "exceeded_percent_of_season_hours": 0.01,
                },
            },
            {
                "season_id": "inter_monsoon",
                "starts_month": 3,
                "starts_day": 16,
                "ends_month": 5,
                "ends_day": 31,
                "mechanisms": {
                    MECHANISM_RAIN: "typical",
                    MECHANISM_EVAPORATION_DUCT: "typical",
                },
                "rain_rate": {
                    "mm_per_hour": 80.0,
                    "exceeded_percent_of_season_hours": 0.01,
                },
            },
            {
                "season_id": "southwest_monsoon",
                "starts_month": 6,
                "starts_day": 1,
                "ends_month": 10,
                "ends_day": 31,
                "mechanisms": {
                    MECHANISM_RAIN: "occasional",
                    MECHANISM_EVAPORATION_DUCT: "typical",
                    MECHANISM_FOG_OR_CLOUD: "occasional",
                },
                "rain_rate": {
                    "mm_per_hour": 70.0,
                    "exceeded_percent_of_season_hours": 0.01,
                },
            },
        ],
    }


def arid_continental() -> dict[str, Any]:
    """An inland desert: dust and dry-air scintillation, essentially no rain."""
    return {
        "schema": CLIMATE_DECLARATION_SCHEMA,
        "declaration_id": "ao-atacama-2026",
        "authority": "J6-SPECTRUM-CELL",
        "basis": "ITU-R P.837-7 read at 23.5S 68.0W; regional dust-storm record",
        "area_label": "Atacama interior",
        "centre_latitude_deg": -23.5,
        "centre_longitude_deg": -68.0,
        "radius_km": 200.0,
        "maritime": False,
        "created_at": CREATED,
        "expires_at": EXPIRES,
        "seasons": [
            {
                # The altiplanic summer, when what little rain there is falls.
                "season_id": "altiplanic_summer",
                "starts_month": 12,
                "starts_day": 1,
                "ends_month": 3,
                "ends_day": 31,
                "mechanisms": {
                    MECHANISM_RAIN: "rare",
                    MECHANISM_DUST_OR_SAND: "occasional",
                    MECHANISM_TROPOSPHERIC_SCINTILLATION: "typical",
                },
                "rain_rate": {
                    "mm_per_hour": 5.0,
                    "exceeded_percent_of_season_hours": 0.01,
                },
            },
            {
                "season_id": "dry_remainder",
                "starts_month": 4,
                "starts_day": 1,
                "ends_month": 11,
                "ends_day": 30,
                "mechanisms": {
                    MECHANISM_DUST_OR_SAND: "typical",
                    MECHANISM_TROPOSPHERIC_SCINTILLATION: "typical",
                },
            },
        ],
    }


def southern_maritime() -> dict[str, Any]:
    """A Southern Hemisphere maritime area, where July is the cold month."""
    return {
        "schema": CLIMATE_DECLARATION_SCHEMA,
        "declaration_id": "ao-south-atlantic-2026",
        "authority": "J6-SPECTRUM-CELL",
        "basis": "ITU-R P.837-7 read at 51.7S 57.85W; fleet observation log",
        "area_label": "South Atlantic approaches",
        "centre_latitude_deg": -51.7,
        "centre_longitude_deg": -57.85,
        "radius_km": 150.0,
        "maritime": True,
        "created_at": CREATED,
        "expires_at": EXPIRES,
        "seasons": [
            {
                "season_id": "austral_winter",
                "starts_month": 5,
                "starts_day": 1,
                "ends_month": 9,
                "ends_day": 30,
                "mechanisms": {
                    MECHANISM_RAIN: "typical",
                    MECHANISM_WET_SNOW: "occasional",
                    MECHANISM_DRY_SNOW: "occasional",
                    MECHANISM_FOG_OR_CLOUD: "typical",
                    MECHANISM_EVAPORATION_DUCT: "occasional",
                },
                "rain_rate": {
                    "mm_per_hour": 18.0,
                    "exceeded_percent_of_season_hours": 0.01,
                },
            },
            {
                "season_id": "austral_summer",
                "starts_month": 10,
                "starts_day": 1,
                "ends_month": 4,
                "ends_day": 30,
                "mechanisms": {
                    MECHANISM_RAIN: "typical",
                    MECHANISM_FOG_OR_CLOUD: "typical",
                    MECHANISM_EVAPORATION_DUCT: "typical",
                },
                "rain_rate": {
                    "mm_per_hour": 24.0,
                    "exceeded_percent_of_season_hours": 0.01,
                },
            },
        ],
    }


def declaration(raw: dict[str, Any] | None = None, *, now: datetime = NOVEMBER) -> ClimateDeclaration:
    return ClimateDeclaration.from_dict(raw or tropical_maritime(), now=now)


def inside_tropics() -> dict[str, float]:
    return {"latitude_deg": 1.30, "longitude_deg": 103.80}


class CalendarCoverageTests(unittest.TestCase):
    def test_a_calendar_gap_is_refused(self) -> None:
        """A hole in the year answers "not assessed" on dates nobody noticed.

        An unprovisioned node and a node whose document forgot April look
        identical to every consumer, and only one of them is a bug an operator
        would go looking for.
        """
        raw = tropical_maritime()
        raw["seasons"][1]["starts_day"] = 20  # leaves 16-19 March uncovered
        with self.assertRaises(ClimateError) as caught:
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)
        self.assertIn("03-16", str(caught.exception))

    def test_overlapping_seasons_are_refused(self) -> None:
        """Two mechanism sets both true on one date means the node picks one."""
        raw = tropical_maritime()
        raw["seasons"][1]["starts_day"] = 10  # overlaps the monsoon's 15 March end
        with self.assertRaises(ClimateError) as caught:
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)
        self.assertIn("overlap", str(caught.exception))

    def test_a_season_may_wrap_the_year_end(self) -> None:
        """A November-to-March monsoon is one season; splitting it is a fiction."""
        declared = declaration()
        monsoon = declared.season_for(NOVEMBER)
        assert monsoon is not None
        self.assertEqual(monsoon.season_id, "northeast_monsoon")
        january = declared.season_for(NOVEMBER.replace(month=1, day=20))
        assert january is not None
        self.assertEqual(january.season_id, "northeast_monsoon")

    def test_every_calendar_date_resolves_to_exactly_one_season(self) -> None:
        """Including 29 February, which exists in the calendar seasons are written against."""
        declared = declaration()
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    moment = datetime(2028, month, day, tzinfo=timezone.utc)
                except ValueError:
                    continue
                self.assertIsNotNone(
                    declared.season_for(moment), f"{month:02d}-{day:02d} has no season"
                )
        self.assertIsNotNone(declared.season_for(LEAP_DAY))

    def test_a_season_boundary_may_fall_on_29_february(self) -> None:
        raw = tropical_maritime()
        raw["seasons"][0]["ends_month"] = 2
        raw["seasons"][0]["ends_day"] = 29
        raw["seasons"][1]["starts_month"] = 3
        raw["seasons"][1]["starts_day"] = 1
        declared = ClimateDeclaration.from_dict(raw, now=NOVEMBER)
        season = declared.season_for(LEAP_DAY)
        assert season is not None
        self.assertEqual(season.season_id, "northeast_monsoon")

    def test_duplicate_season_ids_are_refused(self) -> None:
        raw = tropical_maritime()
        raw["seasons"][1]["season_id"] = raw["seasons"][0]["season_id"]
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_southern_hemisphere_year_needs_no_hemisphere_to_be_named(self) -> None:
        """Wet snow in July and not in January is the Southern Hemisphere.

        A scheme with a fixed winter/summer vocabulary would have to know which
        hemisphere it was in to read this document, and would be six months
        wrong when it guessed. Calendar intervals carry it with nothing assumed.
        """
        declared = declaration(southern_maritime())
        winter = assess_plausibility(declared, MECHANISM_WET_SNOW, at=JULY)
        summer = assess_plausibility(
            declared, MECHANISM_WET_SNOW, at=JULY.replace(month=1)
        )
        self.assertEqual(winter.plausibility, PLAUSIBILITY_OCCASIONAL)
        self.assertEqual(summer.plausibility, PLAUSIBILITY_OUT_OF_SEASON)
        self.assertEqual(winter.season_id, "austral_winter")
        self.assertEqual(summer.season_id, "austral_summer")


class SeasonDocumentTests(unittest.TestCase):
    def test_an_unknown_mechanism_is_refused(self) -> None:
        """An open vocabulary lets a document and the physics never meet.

        A declaration naming ``monsoon_effects`` while the physics asks about
        ``rain`` would answer "not assessed" forever, and nothing would look
        broken.
        """
        raw = tropical_maritime()
        raw["seasons"][0]["mechanisms"]["monsoon_effects"] = "typical"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_an_unknown_prevalence_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["seasons"][0]["mechanisms"][MECHANISM_RAIN] = "very likely"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_season_with_no_credible_mechanism_is_refused(self) -> None:
        """A season where the atmosphere does nothing describes nowhere on Earth."""
        raw = tropical_maritime()
        raw["seasons"][2]["mechanisms"] = {}
        raw["seasons"][2].pop("rain_rate")
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_rain_declared_credible_without_a_rain_rate_is_refused(self) -> None:
        """"It rains here, I won't say how hard" gives the physics nothing to use."""
        raw = tropical_maritime()
        raw["seasons"][0].pop("rain_rate")
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_rain_rate_for_a_season_that_declares_no_rain_is_refused(self) -> None:
        """One of the two statements is wrong; the node may not choose which."""
        raw = arid_continental()
        raw["seasons"][1]["rain_rate"] = {
            "mm_per_hour": 5.0,
            "exceeded_percent_of_season_hours": 0.01,
        }
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_rain_rate_without_its_exceedance_percentage_is_refused(self) -> None:
        """A bare number wearing the costume of a measurement.

        Twenty-five millimetres an hour means nothing until it says whether it
        is a monthly mean or the rate exceeded four hours a year.
        """
        raw = tropical_maritime()
        raw["seasons"][0]["rain_rate"] = {"mm_per_hour": 95.0}
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_rain_rate_that_is_really_a_monthly_total_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["seasons"][0]["rain_rate"]["mm_per_hour"] = 2400.0
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_an_exceedance_percentage_outside_the_useful_range_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["seasons"][0]["rain_rate"]["exceeded_percent_of_season_hours"] = 60.0
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_an_impossible_calendar_date_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["seasons"][0]["starts_month"] = 2
        raw["seasons"][0]["starts_day"] = 30
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_season_cannot_name_one_mechanism_twice(self) -> None:
        """Built directly, because a JSON object cannot hold a duplicate key."""
        with self.assertRaises(ClimateError):
            ClimateSeason(
                season_id="all_year",
                starts_month=1,
                starts_day=1,
                ends_month=12,
                ends_day=31,
                mechanism_prevalence=(
                    (MECHANISM_DUST_OR_SAND, "typical"),
                    (MECHANISM_DUST_OR_SAND, "rare"),
                ),
                rain_rate_mm_per_hour=None,
                rain_rate_exceedance_percent=None,
            )


class DeclarationDocumentTests(unittest.TestCase):
    def test_every_required_field_is_refused_when_absent_rather_than_defaulted(self) -> None:
        """A default climate is a claim about the world that nobody signed."""
        for field in (
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
        ):
            raw = tropical_maritime()
            raw.pop(field)
            with self.subTest(field=field):
                with self.assertRaises(ClimateError) as caught:
                    ClimateDeclaration.from_dict(raw, now=NOVEMBER)
                self.assertIn(field, str(caught.exception))

    def test_maritime_must_be_an_explicit_boolean(self) -> None:
        """The string "false" is truthy, and would put a sea under a desert."""
        for value in ("false", "true", 0, 1, None):
            raw = tropical_maritime()
            raw["maritime"] = value
            with self.subTest(value=value):
                with self.assertRaises(ClimateError):
                    ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_evaporation_ducting_declared_over_land_is_refused(self) -> None:
        """A duct needs a water surface; the document contradicts itself."""
        raw = arid_continental()
        raw["seasons"][1]["mechanisms"][MECHANISM_EVAPORATION_DUCT] = "occasional"
        with self.assertRaises(ClimateError) as caught:
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)
        self.assertIn(MECHANISM_EVAPORATION_DUCT, str(caught.exception))

    def test_the_maritime_only_set_is_not_empty(self) -> None:
        """If nothing needed a sea, the maritime flag would be decoration."""
        self.assertTrue(MARITIME_ONLY_MECHANISMS)
        self.assertTrue(MARITIME_ONLY_MECHANISMS <= DECLARABLE_MECHANISMS)

    def test_a_declaration_that_expires_before_it_was_created_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["expires_at"] = "2026-07-01T00:00:00+00:00"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_declaration_created_in_the_future_is_refused(self) -> None:
        """A document from the node's future is a provenance or clock fault."""
        raw = tropical_maritime()
        raw["created_at"] = "2027-01-01T00:00:00+00:00"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_naive_timestamp_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["created_at"] = "2026-08-01T00:00:00"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_a_lapsed_declaration_still_loads_so_the_lapse_can_be_reported(self) -> None:
        """Refusing to load it would hide a stale document behind "none configured".

        Those two states call for different actions — provision one, or push a
        fresh one — and an operator cannot tell them apart from silence.
        """
        declared = ClimateDeclaration.from_dict(
            tropical_maritime(), now=datetime(2027, 12, 1, tzinfo=timezone.utc)
        )
        self.assertEqual(declared.declaration_id, "ao-singapore-strait-2026")

    def test_an_unsupported_schema_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["schema"] = "kval-climate-declaration/v2"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_an_area_too_large_to_be_one_climate_is_refused(self) -> None:
        """A circle wider than two thousand kilometres crosses climate regimes."""
        raw = tropical_maritime()
        raw["radius_km"] = MAX_AREA_RADIUS_KM + 1.0
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_an_unprovisioned_identifier_is_refused(self) -> None:
        raw = tropical_maritime()
        raw["authority"] = "REPLACE_WITH_ISSUING_AUTHORITY"
        with self.assertRaises(ClimateError):
            ClimateDeclaration.from_dict(raw, now=NOVEMBER)

    def test_the_document_round_trips_through_to_dict(self) -> None:
        """An audit record that cannot be reloaded is not a record of the document."""
        original = declaration()
        reloaded = ClimateDeclaration.from_dict(original.to_dict(), now=NOVEMBER)
        self.assertEqual(reloaded, original)
        json.dumps(original.to_dict())


class AreaTests(unittest.TestCase):
    def test_a_degree_of_latitude_is_about_111_kilometres(self) -> None:
        self.assertAlmostEqual(great_circle_km(0.0, 0.0, 1.0, 0.0), 111.19, places=1)

    def test_the_antimeridian_is_not_a_wall(self) -> None:
        """A bounding box needs a special case here; a spherical distance does not.

        Two positions a tenth of a degree apart across 180 degrees of longitude
        are eleven kilometres apart, not most of the way round the planet.
        """
        self.assertLess(great_circle_km(0.0, 179.95, 0.0, -179.95), 12.0)

    def test_distance_holds_at_high_latitude(self) -> None:
        """Ten degrees of longitude at 80 south is a short hop, not 1100 km."""
        self.assertLess(great_circle_km(-80.0, 0.0, -80.0, 10.0), 200.0)

    def test_containment_adds_no_margin_to_the_declared_radius(self) -> None:
        """Margin invented here extends every declaration into terrain nobody described."""
        declared = declaration()
        just_inside = declared.centre_latitude_deg + 119.0 / KM_PER_DEGREE_LAT
        just_outside = declared.centre_latitude_deg + 121.0 / KM_PER_DEGREE_LAT
        self.assertTrue(declared.contains(just_inside, declared.centre_longitude_deg)[0])
        self.assertFalse(declared.contains(just_outside, declared.centre_longitude_deg)[0])

    def test_a_platform_outside_the_area_gets_no_climatological_answer(self) -> None:
        """The document is about somewhere else, and following the platform out is the failure."""
        result = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=NOVEMBER, latitude_deg=35.0, longitude_deg=139.0
        )
        self.assertFalse(result.assessed)
        self.assertEqual(result.plausibility, PLAUSIBILITY_NOT_ASSESSED)
        self.assertIn("PLATFORM_IS_OUTSIDE_THE_DECLARED_AREA_OF_OPERATIONS", result.reasons)
        self.assertGreater(result.distance_from_area_centre_km, 120.0)
        self.assertEqual(result.declared_radius_km, 120.0)

    def test_an_unknown_position_still_assesses_but_is_never_marked_confirmed(self) -> None:
        """Contested spectrum is where satellite navigation is least available.

        Refusing without a fix would disable the module exactly where it is
        needed; claiming containment without one would be a fabricated fact.
        """
        result = assess_plausibility(declaration(), MECHANISM_RAIN, at=NOVEMBER)
        self.assertTrue(result.assessed)
        self.assertFalse(result.area_confirmed)
        self.assertIsNone(result.distance_from_area_centre_km)
        self.assertIn(
            "PLATFORM_POSITION_UNKNOWN_SO_AREA_CONTAINMENT_WAS_NOT_CONFIRMED", result.reasons
        )

    def test_half_a_position_is_refused(self) -> None:
        """A missing longitude read as zero puts the platform in the Gulf of Guinea."""
        with self.assertRaises(ClimateError):
            assess_plausibility(declaration(), MECHANISM_RAIN, at=NOVEMBER, latitude_deg=1.3)
        with self.assertRaises(ClimateError):
            assess_plausibility(declaration(), MECHANISM_RAIN, at=NOVEMBER, longitude_deg=103.8)

    def test_a_position_off_the_globe_is_refused(self) -> None:
        with self.assertRaises(ClimateError):
            assess_plausibility(
                declaration(), MECHANISM_RAIN, at=NOVEMBER, latitude_deg=91.0, longitude_deg=0.0
            )


class PlausibilityTests(unittest.TestCase):
    def test_a_typical_mechanism_in_its_season_is_typical(self) -> None:
        result = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=NOVEMBER, **inside_tropics()
        )
        self.assertTrue(result.assessed)
        self.assertEqual(result.plausibility, PLAUSIBILITY_TYPICAL)
        self.assertEqual(result.season_id, "northeast_monsoon")
        self.assertTrue(result.supports())
        self.assertFalse(result.conflicts())

    def test_a_mechanism_the_operator_called_occasional_still_supports(self) -> None:
        result = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=JULY, **inside_tropics()
        )
        self.assertEqual(result.plausibility, PLAUSIBILITY_OCCASIONAL)
        self.assertTrue(result.supports())

    def test_a_mechanism_declared_only_in_other_seasons_is_out_of_season(self) -> None:
        """Credible here, not now — and the record names the seasons that do declare it."""
        result = assess_plausibility(
            declaration(), MECHANISM_FOG_OR_CLOUD, at=NOVEMBER.replace(month=4), **inside_tropics()
        )
        self.assertEqual(result.plausibility, PLAUSIBILITY_OUT_OF_SEASON)
        self.assertTrue(result.conflicts())
        self.assertTrue(
            any(reason.startswith("MECHANISM_IS_DECLARED_ONLY_IN_SEASONS:") for reason in result.reasons)
        )

    def test_wet_snow_is_not_credible_on_the_equator_in_any_month(self) -> None:
        for moment in (NOVEMBER, JULY, NOVEMBER.replace(month=4)):
            with self.subTest(month=moment.month):
                result = assess_plausibility(
                    declaration(), MECHANISM_WET_SNOW, at=moment, **inside_tropics()
                )
                self.assertEqual(result.plausibility, PLAUSIBILITY_NOT_CREDIBLE)
                self.assertIn(
                    "MECHANISM_IS_NOT_DECLARED_CREDIBLE_ANYWHERE_IN_THIS_AREA", result.reasons
                )

    def test_evaporation_ducting_over_a_desert_says_why_it_is_not_credible(self) -> None:
        """"Not declared" and "there is no sea here" are different facts."""
        result = assess_plausibility(
            declaration(arid_continental()),
            MECHANISM_EVAPORATION_DUCT,
            at=JULY,
            latitude_deg=-23.5,
            longitude_deg=-68.0,
        )
        self.assertEqual(result.plausibility, PLAUSIBILITY_NOT_CREDIBLE)
        self.assertIn("AREA_IS_NOT_DECLARED_MARITIME", result.reasons)
        self.assertIs(result.maritime_declared, False)

    def test_rain_in_an_arid_area_is_rare_rather_than_impossible(self) -> None:
        """An operator who knows it rains twice a decade should be able to say so.

        Forcing that into "not credible" would make the one real event of the
        deployment a contradiction of the document.
        """
        result = assess_plausibility(
            declaration(arid_continental()),
            MECHANISM_RAIN,
            at=NOVEMBER.replace(month=1),
            latitude_deg=-23.5,
            longitude_deg=-68.0,
        )
        self.assertEqual(result.plausibility, PLAUSIBILITY_RARE)
        self.assertTrue(result.conflicts())

    def test_a_lapsed_declaration_is_not_applied(self) -> None:
        """Nobody is standing behind the document any more."""
        result = assess_plausibility(
            declaration(),
            MECHANISM_RAIN,
            at=datetime(2027, 12, 1, tzinfo=timezone.utc),
            **inside_tropics(),
        )
        self.assertFalse(result.assessed)
        self.assertTrue(result.declaration_expired)
        self.assertLess(result.seconds_until_expiry, 0.0)
        self.assertIn(
            "A_LAPSED_DECLARATION_IS_NOT_APPLIED_AND_THE_MEASUREMENT_STANDS_ALONE",
            result.reasons,
        )

    def test_an_impending_lapse_is_announced_while_the_document_still_applies(self) -> None:
        """A warning that only arrives after the lapse is not a warning."""
        expiry = datetime.fromisoformat(EXPIRES)
        result = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=expiry - timedelta(days=2), **inside_tropics()
        )
        self.assertTrue(result.assessed)
        self.assertIn("CLIMATE_DECLARATION_LAPSES_WITHIN_THE_WEEK", result.reasons)

    def test_no_declaration_leaves_the_physics_alone_and_says_so(self) -> None:
        result = assess_plausibility(None, MECHANISM_RAIN, at=NOVEMBER, **inside_tropics())
        self.assertFalse(result.assessed)
        self.assertIn("NO_CLIMATE_DECLARATION_IS_CONFIGURED_FOR_THIS_NODE", result.reasons)
        self.assertIsNone(result.declaration_id)

    def test_a_mechanism_outside_the_vocabulary_is_refused_not_raised(self) -> None:
        """A vocabulary mismatch between two modules must not stop the sensing loop."""
        result = assess_plausibility(
            declaration(), "monsoon_effects", at=NOVEMBER, **inside_tropics()
        )
        self.assertFalse(result.assessed)
        self.assertIn(
            "MECHANISM_IS_OUTSIDE_THE_DECLARATION_VOCABULARY:monsoon_effects", result.reasons
        )

    def test_a_refusal_reports_nothing_rather_than_zero(self) -> None:
        """A zero rain rate reads as "looked it up, found none" — what a refusal is not."""
        result = assess_plausibility(None, MECHANISM_RAIN, at=NOVEMBER)
        self.assertIsNone(result.season_id)
        self.assertIsNone(result.season_prevalence)
        self.assertIsNone(result.climatological_rain_rate_mm_per_hour)
        self.assertIsNone(result.rain_rate_exceedance_percent_of_season_hours)
        self.assertIsNone(result.distance_from_area_centre_km)
        self.assertFalse(result.area_confirmed)

    def test_a_naive_assessment_time_is_refused(self) -> None:
        with self.assertRaises(ClimateError):
            assess_plausibility(declaration(), MECHANISM_RAIN, at=datetime(2026, 11, 20))

    def test_a_date_before_the_declaration_was_written_is_still_assessed(self) -> None:
        """The document was written later; the climate it describes was not."""
        result = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=datetime(2026, 2, 1, tzinfo=timezone.utc)
        )
        self.assertTrue(result.assessed)

    def test_no_output_field_can_be_read_as_present_weather(self) -> None:
        """The single most likely harm is one of these values quoted as an observation."""
        payload = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=NOVEMBER, **inside_tropics()
        ).to_dict()
        self.assertIsNone(payload["observed_weather"])
        self.assertTrue(payload["climatological_plausibility"].startswith("climatologically_"))
        self.assertIn("THIS_IS_CLIMATOLOGY_NOT_AN_OBSERVATION_OF_PRESENT_WEATHER", payload["reasons"])
        self.assertIn("NO_ATMOSPHERIC_QUANTITY_WAS_MEASURED_BY_THIS_NODE", payload["reasons"])
        for key in payload:
            self.assertNotIn("current", key)
            self.assertNotIn("forecast", key)

    def test_every_refusal_still_carries_the_caveats(self) -> None:
        payload = assess_plausibility(None, MECHANISM_RAIN, at=NOVEMBER).to_dict()
        self.assertIn("THIS_IS_CLIMATOLOGY_NOT_AN_OBSERVATION_OF_PRESENT_WEATHER", payload["reasons"])

    def test_the_result_serialises_to_json_without_infinity(self) -> None:
        payload = assess_plausibility(
            declaration(), MECHANISM_RAIN, at=NOVEMBER, **inside_tropics()
        ).to_dict()
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("Infinity", encoded)
        self.assertNotIn("NaN", encoded)


class RainRateTests(unittest.TestCase):
    def rate(self, value: float, *, at: datetime = NOVEMBER):
        return assess_plausibility(
            declaration(),
            MECHANISM_RAIN,
            at=at,
            inferred_rain_rate_mm_per_hour=value,
            **inside_tropics(),
        )

    def test_a_rate_within_the_declared_exceedance_is_unremarkable(self) -> None:
        result = self.rate(25.0)
        self.assertEqual(result.plausibility, PLAUSIBILITY_TYPICAL)
        self.assertIn(
            "INFERRED_RAIN_RATE_IS_WITHIN_THE_DECLARED_CLIMATOLOGICAL_EXCEEDANCE", result.reasons
        )

    def test_a_rate_in_the_tail_is_expected_rather_than_flagged(self) -> None:
        """The fade you can detect is drawn from the tail; flagging it inverts the test.

        The declared figure is exceeded 0.01% of the season's hours, so a rain
        fade deep enough to measure is *supposed* to sit above it. Treating that
        as a contradiction would make the module wrong on exactly the events it
        exists for.
        """
        result = self.rate(150.0)  # above the declared 95 mm/h, inside 3x
        self.assertEqual(result.plausibility, PLAUSIBILITY_TYPICAL)
        self.assertIn("INFERRED_RAIN_RATE_SITS_IN_THE_TAIL_THE_DECLARATION_DESCRIBES", result.reasons)

    def test_a_rate_far_past_the_tail_is_rare_but_never_impossible(self) -> None:
        result = self.rate(95.0 * RAIN_RATE_EXTREME_FACTOR + 1.0)
        self.assertEqual(result.plausibility, PLAUSIBILITY_RARE)
        self.assertIn("RAIN_REMAINS_CREDIBLE_HERE_BUT_NOT_AT_THIS_INTENSITY", result.reasons)

    def test_an_extreme_rate_never_pushes_a_mechanism_below_rare(self) -> None:
        """Rain that fell here fell here; intensity cannot make it not credible."""
        result = assess_plausibility(
            declaration(arid_continental()),
            MECHANISM_RAIN,
            at=JULY,  # rain is out of season in the arid declaration
            inferred_rain_rate_mm_per_hour=200.0,
            latitude_deg=-23.5,
            longitude_deg=-68.0,
        )
        self.assertEqual(result.plausibility, PLAUSIBILITY_OUT_OF_SEASON)
        self.assertIn(
            "NO_RAIN_RATE_IS_DECLARED_FOR_THIS_SEASON_SO_INTENSITY_WAS_NOT_WEIGHED",
            result.reasons,
        )

    def test_the_exceedance_basis_travels_with_the_comparison(self) -> None:
        """A comparison against an unstated basis is not reviewable."""
        result = self.rate(150.0)
        self.assertTrue(
            any(
                reason.startswith("RAIN_RATE_EXCEEDANCE_BASIS_PERCENT_OF_SEASON_HOURS:")
                for reason in result.reasons
            )
        )
        self.assertEqual(result.rain_rate_exceedance_percent_of_season_hours, 0.01)

    def test_a_rain_rate_supplied_for_another_mechanism_is_ignored_and_said_so(self) -> None:
        result = assess_plausibility(
            declaration(),
            MECHANISM_FOG_OR_CLOUD,
            at=NOVEMBER,
            inferred_rain_rate_mm_per_hour=40.0,
            **inside_tropics(),
        )
        self.assertEqual(result.plausibility, PLAUSIBILITY_TYPICAL)
        self.assertIsNone(result.inferred_rain_rate_mm_per_hour)
        self.assertIn(
            "A_RAIN_RATE_WAS_SUPPLIED_FOR_A_MECHANISM_THAT_IS_NOT_RAIN_AND_WAS_IGNORED",
            result.reasons,
        )

    def test_a_non_finite_rate_is_refused(self) -> None:
        """A NaN comparing false against every threshold would silently pass every test."""
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ClimateError):
                    self.rate(value)


class CorroborationTests(unittest.TestCase):
    def support(self) -> Any:
        return corroborate_mechanism(
            declaration(),
            MECHANISM_RAIN,
            physics_consistent=True,
            at=NOVEMBER,
            **inside_tropics(),
        )

    def test_agreement_is_reported_as_agreement_and_still_not_an_observation(self) -> None:
        result = self.support()
        self.assertEqual(result.corroboration, CORROBORATION_SUPPORTED)
        self.assertFalse(result.conflict)
        self.assertIn(
            "TWO_STRANDS_OF_CIRCUMSTANTIAL_EVIDENCE_AGREEING_IS_NOT_AN_OBSERVATION",
            result.reasons,
        )

    def test_a_conflict_is_reported_and_the_measurement_is_kept_whole(self) -> None:
        """A node that discards a real observation because a file disagreed has lost.

        The declaration can be stale, the platform can have left the area it
        describes, and rare events occur. The record says all three and
        suppresses nothing.
        """
        result = corroborate_mechanism(
            declaration(),
            MECHANISM_WET_SNOW,
            physics_consistent=True,
            at=NOVEMBER,
            **inside_tropics(),
        )
        self.assertEqual(result.corroboration, CORROBORATION_CONFLICT)
        self.assertTrue(result.conflict)
        self.assertTrue(result.physics_consistent)
        self.assertTrue(result.physics_retained)
        self.assertIn(
            "THE_MEASUREMENT_IS_RETAINED_IN_FULL_THE_DECLARATION_IS_NOT_AN_OVERRIDE",
            result.reasons,
        )
        self.assertIn(
            "THE_DECLARATION_MAY_BE_STALE_OR_THE_PLATFORM_MAY_HAVE_LEFT_THE_DECLARED_AREA",
            result.reasons,
        )

    def test_climatology_cannot_create_a_mechanism_the_measurement_excluded(self) -> None:
        """"It is the monsoon" is not evidence that anything is falling on this path.

        Invention is the same failure as suppression, pointing the other way,
        and it is the easier one to ship by accident.
        """
        result = corroborate_mechanism(
            declaration(),
            MECHANISM_RAIN,
            physics_consistent=False,
            at=NOVEMBER,
            **inside_tropics(),
        )
        self.assertEqual(result.corroboration, CORROBORATION_NO_MECHANISM)
        self.assertFalse(result.conflict)
        self.assertIn(
            "CLIMATOLOGY_CANNOT_ESTABLISH_A_MECHANISM_THE_MEASUREMENT_DOES_NOT_SUPPORT",
            result.reasons,
        )
        # The climatology is still carried, so an operator can see it was typical
        # and that it changed nothing.
        self.assertEqual(result.plausibility.plausibility, PLAUSIBILITY_TYPICAL)

    def test_no_declaration_leaves_physics_standing_alone(self) -> None:
        result = corroborate_mechanism(
            None, MECHANISM_RAIN, physics_consistent=True, at=NOVEMBER
        )
        self.assertEqual(result.corroboration, CORROBORATION_NOT_ASSESSED)
        self.assertFalse(result.conflict)
        self.assertTrue(result.physics_retained)
        self.assertIn(
            "PHYSICS_STANDS_ALONE_NO_CLIMATOLOGICAL_CONTEXT_WAS_AVAILABLE", result.reasons
        )

    def test_a_lapsed_declaration_leaves_physics_standing_alone(self) -> None:
        result = corroborate_mechanism(
            declaration(),
            MECHANISM_WET_SNOW,
            physics_consistent=True,
            at=datetime(2027, 12, 1, tzinfo=timezone.utc),
            **inside_tropics(),
        )
        self.assertEqual(result.corroboration, CORROBORATION_NOT_ASSESSED)
        self.assertFalse(result.conflict)

    def test_an_unconfirmed_position_is_stated_on_the_corroboration_itself(self) -> None:
        """A conflict from a document that may be about elsewhere must say so."""
        result = corroborate_mechanism(
            declaration(), MECHANISM_WET_SNOW, physics_consistent=True, at=NOVEMBER
        )
        self.assertIn(
            "PLATFORM_POSITION_WAS_NEVER_CONFIRMED_INSIDE_THE_DECLARED_AREA", result.reasons
        )

    def test_physics_is_retained_across_every_outcome(self) -> None:
        """The invariant the module exists to keep, checked over the whole matrix."""
        declared = declaration()
        cases = [
            (declared, MECHANISM_RAIN, NOVEMBER),
            (declared, MECHANISM_WET_SNOW, NOVEMBER),
            (declared, MECHANISM_FOG_OR_CLOUD, NOVEMBER.replace(month=4)),
            (None, MECHANISM_RAIN, NOVEMBER),
            (declared, MECHANISM_RAIN, datetime(2027, 12, 1, tzinfo=timezone.utc)),
        ]
        for declared_doc, mechanism, moment in cases:
            for consistent in (True, False):
                with self.subTest(mechanism=mechanism, consistent=consistent):
                    result = corroborate_mechanism(
                        declared_doc, mechanism, physics_consistent=consistent, at=moment
                    )
                    self.assertEqual(result.physics_retained, consistent)
                    self.assertEqual(result.physics_consistent, consistent)

    def test_a_conflict_is_never_reported_without_a_physics_finding(self) -> None:
        """Climatology alone can never raise a conflict; there is nothing to conflict with."""
        for mechanism in sorted(DECLARABLE_MECHANISMS):
            with self.subTest(mechanism=mechanism):
                result = corroborate_mechanism(
                    declaration(), mechanism, physics_consistent=False, at=NOVEMBER
                )
                self.assertFalse(result.conflict)

    def test_a_plausibility_for_another_mechanism_is_refused(self) -> None:
        """Mismatched wiring would attribute one mechanism's climate to another."""
        rain = assess_plausibility(declaration(), MECHANISM_RAIN, at=NOVEMBER)
        with self.assertRaises(ClimateError):
            corroborate(MECHANISM_WET_SNOW, physics_consistent=True, plausibility=rain)

    def test_a_non_boolean_physics_finding_is_refused(self) -> None:
        """A verdict object passed instead of its boolean would corroborate everything."""
        rain = assess_plausibility(declaration(), MECHANISM_RAIN, at=NOVEMBER)
        for value in ("true", 1, None, object()):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ClimateError):
                    corroborate(MECHANISM_RAIN, physics_consistent=value, plausibility=rain)

    def test_the_record_serialises_to_json_without_infinity(self) -> None:
        payload = self.support().to_dict()
        encoded = json.dumps(payload, allow_nan=False)
        self.assertNotIn("Infinity", encoded)
        self.assertNotIn("NaN", encoded)
        self.assertEqual(payload["climate"]["schema"], "kval-climate-plausibility/v1")
        self.assertIn("THIS_IS_CLIMATOLOGY_NOT_AN_OBSERVATION_OF_PRESENT_WEATHER", payload["reasons"])


class ImportDisciplineTests(unittest.TestCase):
    def test_the_module_imports_only_the_standard_library_and_kval_node(self) -> None:
        """A runtime dependency here would be one the node cannot install offline."""
        source = pathlib.Path(__file__).resolve().parent.parent / "src" / "kval_node" / "climate.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside kval_node
                    continue
                if node.module:
                    imported.add(node.module.split(".")[0])
        outside = sorted(imported - sys.stdlib_module_names - {"kval_node"})
        self.assertEqual(outside, [])

    def test_nothing_in_the_module_reaches_the_network(self) -> None:
        """A sensor that phones out for a forecast has stopped being receive-only."""
        source = pathlib.Path(__file__).resolve().parent.parent / "src" / "kval_node" / "climate.py"
        text = source.read_text(encoding="utf-8")
        for token in ("socket", "urllib", "http.client", "requests"):
            self.assertNotIn(f"import {token}", text)


class DocumentIndependenceTests(unittest.TestCase):
    def test_two_declarations_do_not_share_state(self) -> None:
        """Season tuples leaking between documents would move one area's monsoon."""
        first = declaration()
        second = declaration(arid_continental())
        self.assertNotEqual(first.declared_mechanisms(), second.declared_mechanisms())
        self.assertNotEqual(
            {season.season_id for season in first.seasons},
            {season.season_id for season in second.seasons},
        )

    def test_the_source_mapping_is_not_mutated_by_parsing(self) -> None:
        """A parser that edits its input corrupts the audit copy of the document."""
        raw = tropical_maritime()
        before = copy.deepcopy(raw)
        ClimateDeclaration.from_dict(raw, now=NOVEMBER)
        self.assertEqual(raw, before)


if __name__ == "__main__":
    unittest.main()
