"""Attenuation or arriving energy: what made a band worse.

The discriminator rests on one asymmetry — something on the path removes signal
and cannot add noise, another emitter adds noise and cannot remove signal — and
on refusing everywhere that asymmetry does not hold. These tests exercise the
refusals at least as hard as the verdicts, because a discriminator that always
answers is one whose answers carry no information.

See ``src/kval_node/propagation.py`` and docs/SPECTRUM_SENSING_FABRIC.md.
"""

from __future__ import annotations

import json
import math
import unittest
from datetime import datetime, timedelta, timezone
from typing import Sequence

from kval_node.mobility import Fix
from kval_node.propagation import (
    CAUSE_AMBIGUOUS,
    CAUSE_CONCURRENT,
    CAUSE_ENERGY_IN_BAND,
    CAUSE_INSUFFICIENT_EVIDENCE,
    CAUSE_NO_DEGRADATION,
    CAUSE_PATH_ATTENUATION,
    MECHANISM_ABRUPT_OBSTRUCTION,
    MECHANISM_FREQUENCY_SELECTIVE_FADE,
    MECHANISM_NOT_APPLICABLE,
    MECHANISM_UNDETERMINED,
    MIN_FREQUENCY_SPAN_DECADES,
    BandSeries,
    PropagationDiscriminator,
    PropagationError,
    PropagationVerdict,
    centre_frequency_hz,
    fade_slope_db_per_decade,
)
from kval_node.spectrum import SpectrumBand, SpectrumObservation

START = datetime(2026, 5, 12, 9, 0, 0, tzinfo=timezone.utc)

#: A fixed, repeatable wobble in dB. Every window needs non-zero scatter or the
#: sigma gate is being exercised against an unrealistically silent instrument.
_JITTER_DB = (0.0, 0.3, -0.2, 0.1, -0.3, 0.2, -0.1, 0.25, -0.25, 0.15, -0.15, 0.05)

#: Roughly the length of a degree of latitude, for placing a track a known
#: number of metres apart without importing a projection.
_METRES_PER_DEGREE_LAT = 111_320.0

BASE_LAT = 1.3000
BASE_LON = 103.8000


def wobbled(values: Sequence[float]) -> list[float]:
    return [value + _JITTER_DB[index % len(_JITTER_DB)] for index, value in enumerate(values)]


def band(band_id: str, low_hz: int, high_hz: int) -> SpectrumBand:
    return SpectrumBand.from_dict(
        {
            "band_id": band_id,
            "label": band_id.upper(),
            "freq_low_hz": low_hz,
            "freq_high_hz": high_hz,
            "max_resolution_bandwidth_hz": 1_000_000,
            "occupancy_baseline_percent": 20.0,
            "occupancy_alert_percent": 75.0,
            "noise_floor_baseline_dbm": -95.0,
            "comms_candidate": True,
        }
    )


ISM_2G4 = band("ism-2g4", 2_400_000_000, 2_483_500_000)
LINK_5G8 = band("link-5g8", 5_725_000_000, 5_875_000_000)
KA_20G = band("ka-20g", 19_700_000_000, 20_200_000_000)


def moments(count: int, *, cadence_seconds: float = 30.0) -> list[datetime]:
    return [START + timedelta(seconds=cadence_seconds * index) for index in range(count)]


def readings(
    definition: SpectrumBand,
    noise_dbm: Sequence[float],
    peak_dbm: Sequence[float | None],
    *,
    cadence_seconds: float = 30.0,
    quality: Sequence[float] | None = None,
    times: Sequence[datetime] | None = None,
) -> tuple[SpectrumObservation, ...]:
    """Build a band's observation series from parallel dBm lists."""
    if len(noise_dbm) != len(peak_dbm):
        raise AssertionError("noise and peak series must be the same length")
    stamps = list(times) if times is not None else moments(len(noise_dbm), cadence_seconds=cadence_seconds)
    built: list[SpectrumObservation] = []
    for index, moment in enumerate(stamps):
        raw = {
            "band_id": definition.band_id,
            "observed_at": moment.isoformat(),
            "source_adapter_id": "sensor-01",
            "conformance_hash": "b" * 64,
            "passive": True,
            "occupancy_percent": 30.0,
            "noise_floor_dbm": noise_dbm[index],
            "detected_emitters": 0,
            "quality": 1.0 if quality is None else quality[index],
        }
        if peak_dbm[index] is not None:
            raw["peak_power_dbm"] = peak_dbm[index]
        built.append(SpectrumObservation.from_dict(raw))
    return tuple(built)


def track(
    stamps: Sequence[datetime], *, metres_per_step: float = 0.0, uncertainty_m: float = 2.0
) -> tuple[Fix, ...]:
    return tuple(
        Fix(
            moment,
            BASE_LAT + index * metres_per_step / _METRES_PER_DEGREE_LAT,
            BASE_LON,
            uncertainty_m,
        )
        for index, moment in enumerate(stamps)
    )


def flat_series(definition: SpectrumBand, *, noise: float = -95.0, peak: float = -60.0, count: int = 10):
    return BandSeries(definition, readings(definition, wobbled([noise] * count), wobbled([peak] * count)))


class CentreFrequencyTests(unittest.TestCase):
    def test_the_centre_is_the_geometric_mean(self) -> None:
        """A power law in frequency puts a band's leverage at its geometric centre."""
        self.assertAlmostEqual(
            centre_frequency_hz(KA_20G), math.sqrt(19_700_000_000 * 20_200_000_000), places=1
        )

    def test_a_narrow_band_agrees_with_the_arithmetic_centre(self) -> None:
        arithmetic = (ISM_2G4.freq_low_hz + ISM_2G4.freq_high_hz) / 2.0
        self.assertLess(abs(centre_frequency_hz(ISM_2G4) - arithmetic) / arithmetic, 0.001)


class SlopeTests(unittest.TestCase):
    def test_slope_is_decibels_per_decade(self) -> None:
        """A 10 dB deepening across exactly one decade is 10 dB per decade."""
        slope, refusal = fade_slope_db_per_decade([(1e9, 0.0), (1e10, 10.0)])
        self.assertIsNone(refusal)
        self.assertAlmostEqual(slope, 10.0, places=6)

    def test_one_band_cannot_show_frequency_dependence(self) -> None:
        slope, refusal = fade_slope_db_per_decade([(2e10, 8.0)])
        self.assertIsNone(slope)
        self.assertEqual(
            refusal, "FEWER_THAN_TWO_BANDS_MEASURED_SO_FREQUENCY_DEPENDENCE_WAS_NOT_TESTED"
        )

    def test_bands_too_close_in_frequency_are_refused(self) -> None:
        """Dividing a decibel difference by a tiny log span invents a slope.

        Two adjacent bands differing only by calibration would otherwise
        register as steep frequency selectivity, which is how an instrument
        error becomes a confident claim about the weather.
        """
        slope, refusal = fade_slope_db_per_decade([(2.40e9, 0.0), (2.45e9, 9.0)])
        self.assertIsNone(slope)
        self.assertEqual(refusal, "BANDS_SPAN_TOO_LITTLE_FREQUENCY_TO_RESOLVE_A_SLOPE")

    def test_the_span_requirement_is_derived_from_the_other_two_thresholds(self) -> None:
        # The shift floor divided by the span must not on its own reach the
        # slope threshold, or scatter alone would clear the test.
        self.assertAlmostEqual(MIN_FREQUENCY_SPAN_DECADES, 0.5, places=9)


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()

    def test_a_short_series_cannot_be_split(self) -> None:
        """Nine samples cannot give six of baseline and four of recent."""
        verdict = self.discriminator.discriminate(flat_series(LINK_5G8, count=9))
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertFalse(verdict.discriminated)
        self.assertTrue(any("NEED_AT_LEAST_10" in reason for reason in verdict.reasons))

    def test_a_refusal_reports_no_measurement_rather_than_a_zero(self) -> None:
        """A zero shift reads as "measured, nothing moved" — the one thing a refusal is not."""
        verdict = self.discriminator.discriminate(flat_series(LINK_5G8, count=9))
        self.assertIsNone(verdict.noise_floor_shift_db)
        self.assertIsNone(verdict.received_power_shift_db)
        self.assertIsNone(verdict.baseline_noise_spread_db)
        self.assertIsNone(verdict.onset_seconds)
        self.assertEqual(verdict.mechanism, MECHANISM_NOT_APPLICABLE)

    def test_readings_below_the_confidence_floor_are_not_evidence(self) -> None:
        series = BandSeries(
            LINK_5G8,
            readings(
                LINK_5G8,
                wobbled([-95.0] * 12),
                wobbled([-60.0] * 12),
                quality=[0.05] * 12,
            ),
        )
        verdict = self.discriminator.discriminate(series)
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertIn("NO_OBSERVATION_CLEARED_THE_CONFIDENCE_FLOOR", verdict.reasons)

    def test_dropped_readings_are_stated_not_absorbed(self) -> None:
        """A series that lost readings is thinner than its sample count suggests."""
        quality = [1.0] * 12
        quality[3] = 0.05
        quality[8] = 0.05
        series = BandSeries(
            LINK_5G8,
            readings(LINK_5G8, wobbled([-95.0] * 12), wobbled([-60.0] * 12), quality=quality),
        )
        verdict = self.discriminator.discriminate(series)
        self.assertIn("OBSERVATIONS_BELOW_THE_CONFIDENCE_FLOOR_WERE_DROPPED", verdict.reasons)
        self.assertEqual(verdict.samples, 10)

    def test_duplicate_timestamps_are_refused(self) -> None:
        """Two readings of one instant disagree; averaging them invents a third."""
        stamps = moments(11)
        stamps[5] = stamps[4]
        series = BandSeries(
            LINK_5G8,
            readings(LINK_5G8, wobbled([-95.0] * 11), wobbled([-60.0] * 11), times=stamps),
        )
        verdict = self.discriminator.discriminate(series)
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertIn(
            "SERIES_CONTAINS_DUPLICATE_TIMESTAMPS_AND_NO_RULE_PICKS_A_WINNER", verdict.reasons
        )

    def test_an_unobserved_interval_is_refused(self) -> None:
        """A silence longer than the shortest slow fade hides whether it was a step."""
        stamps = moments(11)
        stamps = stamps[:6] + [moment + timedelta(seconds=600) for moment in stamps[6:]]
        series = BandSeries(
            LINK_5G8,
            readings(LINK_5G8, wobbled([-95.0] * 11), wobbled([-60.0] * 11), times=stamps),
        )
        verdict = self.discriminator.discriminate(series)
        self.assertIn("SERIES_HAS_AN_UNOBSERVED_INTERVAL_TOO_LONG_TO_BRIDGE", verdict.reasons)

    def test_a_baseline_that_swings_is_not_a_baseline(self) -> None:
        """At the sigma gate a 7 dB baseline needs a 21 dB shift — an outage, not a fade."""
        noise = [-88.0, -102.0, -88.0, -102.0, -88.0, -102.0] + [-95.0] * 4
        series = BandSeries(LINK_5G8, readings(LINK_5G8, noise, wobbled([-60.0] * 10)))
        verdict = self.discriminator.discriminate(series)
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertIn("BASELINE_NOISE_FLOOR_IS_TOO_VARIABLE_TO_SERVE_AS_A_BASELINE", verdict.reasons)

    def test_a_baseline_containing_the_event_is_refused(self) -> None:
        """Comparing a change's tail against its own head reports it as quiet."""
        noise = wobbled([-95.0, -95.0, -95.0, -85.0, -85.0, -85.0, -85.0, -85.0, -85.0, -85.0])
        series = BandSeries(LINK_5G8, readings(LINK_5G8, noise, wobbled([-60.0] * 10)))
        verdict = self.discriminator.discriminate(series)
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertIn("BASELINE_NOISE_FLOOR_IS_NOT_STABLE_ITS_TWO_HALVES_DISAGREE", verdict.reasons)

    def test_a_drifting_power_baseline_is_refused_on_its_own_channel(self) -> None:
        peak = wobbled([-60.0, -60.0, -60.0, -50.0, -50.0, -50.0] + [-50.0] * 4)
        series = BandSeries(LINK_5G8, readings(LINK_5G8, wobbled([-95.0] * 10), peak))
        verdict = self.discriminator.discriminate(series)
        self.assertIn(
            "BASELINE_RECEIVED_POWER_IS_NOT_STABLE_ITS_TWO_HALVES_DISAGREE", verdict.reasons
        )

    def test_no_power_series_and_a_quiet_floor_cannot_exclude_attenuation(self) -> None:
        """A quiet noise floor is not evidence that the path is clear."""
        series = BandSeries(
            LINK_5G8, readings(LINK_5G8, wobbled([-95.0] * 10), [None] * 10)
        )
        verdict = self.discriminator.discriminate(series)
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertIn("NO_RECEIVED_POWER_SERIES_SO_ATTENUATION_COULD_NOT_BE_TESTED", verdict.reasons)

    def test_a_partial_power_series_is_discarded_whole(self) -> None:
        """An adapter omits the peak when it cannot find one — that is, when the
        signal is weakest. Averaging the readings that survive biases the answer
        towards "no drop" in exactly the case this module exists to catch.
        """
        peak: list[float | None] = list(wobbled([-60.0] * 10))
        peak[7] = None
        series = BandSeries(LINK_5G8, readings(LINK_5G8, wobbled([-95.0] * 10), peak))
        verdict = self.discriminator.discriminate(series)
        self.assertEqual(verdict.cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertIn(
            "RECEIVED_POWER_MISSING_FROM_SOME_OBSERVATIONS_SO_THE_WHOLE_SERIES_IS_UNUSED",
            verdict.reasons,
        )


class CauseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()

    def _series(self, noise: Sequence[float], peak: Sequence[float | None]) -> BandSeries:
        return BandSeries(LINK_5G8, readings(LINK_5G8, noise, peak))

    def test_signal_falls_with_a_steady_floor_is_attenuation(self) -> None:
        """Nothing on the path can add energy to the band, so the floor holding still
        means the loss happened between the emitter and the receiver."""
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 10), wobbled([-60.0] * 6 + [-70.0] * 4))
        )
        self.assertEqual(verdict.cause, CAUSE_PATH_ATTENUATION)
        self.assertTrue(verdict.discriminated)
        self.assertIn("RECEIVED_POWER_FELL_WHILE_THE_NOISE_FLOOR_HELD", verdict.reasons)
        self.assertLess(verdict.received_power_shift_db, -9.0)
        self.assertEqual(verdict.baseline_samples, 6)
        self.assertEqual(verdict.recent_samples, 4)

    def test_a_rising_floor_with_steady_signal_is_energy_in_the_band(self) -> None:
        """Attenuation cannot raise a receiver's own noise floor."""
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-85.0] * 4), wobbled([-60.0] * 10))
        )
        self.assertEqual(verdict.cause, CAUSE_ENERGY_IN_BAND)
        self.assertTrue(verdict.discriminated)
        self.assertIn("NOISE_FLOOR_ROSE_WHILE_RECEIVED_POWER_HELD", verdict.reasons)

    def test_a_rising_floor_that_lifts_the_peak_with_it_is_still_energy_in_the_band(self) -> None:
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-85.0] * 4), wobbled([-60.0] * 6 + [-50.0] * 4))
        )
        self.assertEqual(verdict.cause, CAUSE_ENERGY_IN_BAND)
        self.assertIn("NOISE_FLOOR_ROSE_AND_RECEIVED_POWER_ROSE_WITH_IT", verdict.reasons)

    def test_both_moving_is_not_resolved_by_picking_the_larger(self) -> None:
        """Two causes at once and one uncomposable cause look identical here.

        In-band energy strong enough to compress a receiver front end lowers
        apparent signal and raises the floor by itself, so a bigger number on
        one side is not evidence about which happened.
        """
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-80.0] * 4), wobbled([-60.0] * 6 + [-70.0] * 4))
        )
        self.assertEqual(verdict.cause, CAUSE_CONCURRENT)
        self.assertFalse(verdict.discriminated)
        self.assertIn(
            "EITHER_TWO_CAUSES_AT_ONCE_OR_ONE_CAUSE_THIS_MODULE_CANNOT_SEPARATE", verdict.reasons
        )
        self.assertIn(
            "IN_BAND_ENERGY_STRONG_ENOUGH_TO_COMPRESS_A_RECEIVER_FRONT_END"
            "_REPRODUCES_BOTH_SIGNATURES",
            verdict.reasons,
        )
        self.assertEqual(verdict.mechanism, MECHANISM_NOT_APPLICABLE)

    def test_signal_and_floor_falling_together_is_the_instrument_not_the_path(self) -> None:
        """No path change lowers the receiver's own floor; a gain step lowers both."""
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-105.0] * 4), wobbled([-60.0] * 6 + [-70.0] * 4))
        )
        self.assertEqual(verdict.cause, CAUSE_AMBIGUOUS)
        self.assertFalse(verdict.discriminated)
        self.assertIn(
            "RECEIVED_POWER_AND_NOISE_FLOOR_FELL_TOGETHER_WHICH_IS_A_GAIN_CHANGE"
            "_NOT_A_PATH_CHANGE",
            verdict.reasons,
        )

    def test_a_quiet_band_is_reported_as_undegraded(self) -> None:
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 10), wobbled([-60.0] * 10))
        )
        self.assertEqual(verdict.cause, CAUSE_NO_DEGRADATION)
        self.assertTrue(verdict.discriminated)
        self.assertIn("NEITHER_CHANNEL_MOVED_BEYOND_THE_MEASUREMENT_FLOOR", verdict.reasons)

    def test_an_improving_floor_is_not_a_degradation(self) -> None:
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-107.0] * 4), wobbled([-60.0] * 10))
        )
        self.assertEqual(verdict.cause, CAUSE_NO_DEGRADATION)
        self.assertIn("NOISE_FLOOR_FELL_WHICH_IS_AN_IMPROVEMENT_NOT_A_DEGRADATION", verdict.reasons)

    def test_a_shift_inside_the_measurement_floor_is_not_a_shift(self) -> None:
        """Two decibels is inside a cheap front end's own calibration budget."""
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 10), wobbled([-60.0] * 6 + [-62.0] * 4))
        )
        self.assertEqual(verdict.cause, CAUSE_NO_DEGRADATION)

    def test_a_band_already_degraded_before_the_series_shows_no_transition(self) -> None:
        """The module compares two windows, so it detects transitions only.

        Reporting a degradation with no transition in view would be inventing
        history the series does not contain.
        """
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 10), wobbled([-78.0] * 10))
        )
        self.assertEqual(verdict.cause, CAUSE_NO_DEGRADATION)

    def test_a_floor_rise_without_a_power_series_says_what_it_did_not_test(self) -> None:
        """Energy arriving is measured; concurrent attenuation is not, and says so."""
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-85.0] * 4), [None] * 10)
        )
        self.assertEqual(verdict.cause, CAUSE_ENERGY_IN_BAND)
        self.assertIn(
            "RECEIVED_POWER_UNAVAILABLE_SO_CONCURRENT_ATTENUATION_WAS_NOT_TESTED", verdict.reasons
        )
        self.assertIsNone(verdict.received_power_shift_db)

    def test_an_absent_power_series_is_never_reported_as_a_steady_one(self) -> None:
        """"Received power held" asserts a comparison. With no peak figures at all
        that comparison never ran, and claiming its result would be exactly the
        "could not measure, reported as measured and fine" failure.
        """
        verdict = self.discriminator.discriminate(
            self._series(wobbled([-95.0] * 6 + [-85.0] * 4), [None] * 10)
        )
        self.assertNotIn("NOISE_FLOOR_ROSE_WHILE_RECEIVED_POWER_HELD", verdict.reasons)
        self.assertIn("NOISE_FLOOR_ROSE_AND_NO_RECEIVED_POWER_FIGURE_EXISTED_TO_COMPARE", verdict.reasons)
        self.assertIn("NO_OBSERVATION_CARRIED_A_RECEIVED_POWER_FIGURE", verdict.reasons)


class ObstructionMechanismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()
        self.stamps = moments(10)
        self.series = BandSeries(
            LINK_5G8,
            readings(
                LINK_5G8,
                wobbled([-95.0] * 10),
                wobbled([-60.0] * 6 + [-72.0] * 4),
                times=self.stamps,
            ),
        )

    def test_a_step_inside_one_interval_is_an_abrupt_obstruction(self) -> None:
        verdict = self.discriminator.discriminate(self.series)
        self.assertEqual(verdict.cause, CAUSE_PATH_ATTENUATION)
        self.assertEqual(verdict.mechanism, MECHANISM_ABRUPT_OBSTRUCTION)
        self.assertIn("FADE_ARRIVED_AS_A_STEP_WITHIN_ONE_SAMPLE_INTERVAL", verdict.reasons)
        self.assertGreaterEqual(verdict.step_concentration, 0.7)

    def test_without_a_track_the_movement_correlation_is_declared_untested(self) -> None:
        """A caveat, not a silence: the operator must see which test did not run."""
        verdict = self.discriminator.discriminate(self.series)
        self.assertIn("PLATFORM_MOVEMENT_CORRELATION_WAS_NOT_TESTED", verdict.reasons)
        self.assertIn("PLATFORM_TRACK_ABSENT_SO_GEOMETRY_COULD_NOT_BE_CHECKED", verdict.reasons)
        self.assertIsNone(verdict.platform_displacement_m)

    def test_a_step_while_the_platform_moved_is_correlated_with_movement(self) -> None:
        verdict = self.discriminator.discriminate(
            self.series, platform_track=track(self.stamps, metres_per_step=25.0)
        )
        self.assertEqual(verdict.mechanism, MECHANISM_ABRUPT_OBSTRUCTION)
        self.assertIn("STEP_COINCIDES_WITH_PLATFORM_MOVEMENT", verdict.reasons)
        self.assertGreater(verdict.platform_displacement_m, 6.0)

    def test_a_step_while_the_platform_held_station_means_something_else_moved(self) -> None:
        """One clears by moving, the other clears by waiting — worth separating."""
        verdict = self.discriminator.discriminate(
            self.series, platform_track=track(self.stamps, metres_per_step=0.0)
        )
        self.assertIn("PLATFORM_HELD_STATION_SO_SOMETHING_ELSE_ENTERED_THE_PATH", verdict.reasons)

    def test_loose_fixes_cannot_establish_stillness(self) -> None:
        """A fix good to twenty-five metres cannot show the platform stayed within six."""
        verdict = self.discriminator.discriminate(
            self.series, platform_track=track(self.stamps, metres_per_step=0.0, uncertainty_m=25.0)
        )
        self.assertIsNone(verdict.platform_displacement_m)
        self.assertIn("PLATFORM_FIX_UNCERTAINTY_EXCEEDS_THE_STILLNESS_THRESHOLD", verdict.reasons)

    def test_a_track_that_misses_the_transition_is_not_used(self) -> None:
        elsewhere = [moment + timedelta(days=1) for moment in self.stamps]
        verdict = self.discriminator.discriminate(self.series, platform_track=track(elsewhere))
        self.assertIn("PLATFORM_TRACK_DOES_NOT_COVER_THE_TRANSITION_INTERVAL", verdict.reasons)

    def test_a_track_outward_and_back_still_counts_as_movement(self) -> None:
        """Endpoint comparison would report a platform that returned as stationary."""
        stamps = self.stamps
        offsets = [0.0, 40.0, 80.0, 120.0, 80.0, 40.0, 0.0, 0.0, 0.0, 0.0]
        excursion = tuple(
            Fix(moment, BASE_LAT + offset / _METRES_PER_DEGREE_LAT, BASE_LON, 2.0)
            for moment, offset in zip(stamps, offsets)
        )
        verdict = self.discriminator.discriminate(self.series, platform_track=excursion)
        self.assertGreater(verdict.platform_displacement_m, 6.0)


def _ramped(baseline: int, start_db: float, step_db: float, count: int) -> list[float]:
    return wobbled([start_db] * baseline + [start_db - step_db * (k + 1) for k in range(count)])


class FrequencySelectiveFadeTests(unittest.TestCase):
    """The rain-fade branch, and the four things it insists on before firing."""

    CADENCE = 60.0

    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()
        self.stamps = moments(18, cadence_seconds=self.CADENCE)

    def _fading(self, definition: SpectrumBand, step_db: float) -> BandSeries:
        return BandSeries(
            definition,
            readings(
                definition,
                wobbled([-95.0] * 18),
                _ramped(12, -60.0, step_db, 6),
                times=self.stamps,
            ),
        )

    def test_a_slow_fade_deepening_with_frequency_over_a_still_platform(self) -> None:
        """All four gates pass: high band, slow ramp, positive slope, static geometry."""
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(ISM_2G4, 0.0), self._fading(KA_20G, 2.4)],
            platform_track=track(self.stamps, metres_per_step=0.0),
        )
        high = verdicts[1]
        self.assertEqual(high.cause, CAUSE_PATH_ATTENUATION)
        self.assertEqual(high.mechanism, MECHANISM_FREQUENCY_SELECTIVE_FADE)
        self.assertGreaterEqual(high.fade_slope_db_per_decade, 6.0)
        self.assertGreaterEqual(high.onset_seconds, 120.0)
        self.assertLess(high.step_concentration, 0.7)

    def test_the_fade_verdict_never_asserts_weather(self) -> None:
        """The module has no met feed, so it reports the signature and stops."""
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(ISM_2G4, 0.0), self._fading(KA_20G, 2.4)],
            platform_track=track(self.stamps, metres_per_step=0.0),
        )
        self.assertIn(
            "CONSISTENT_WITH_HYDROMETEOR_ATTENUATION_PER_ITU_R_P838"
            "_WHICH_IS_NOT_A_WEATHER_MEASUREMENT",
            verdicts[1].reasons,
        )

    def test_a_moving_platform_cannot_exclude_a_diffracting_edge(self) -> None:
        """Knife-edge diffraction also deepens with frequency, so slope alone
        proves nothing about the sky once the geometry is changing."""
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(ISM_2G4, 0.0), self._fading(KA_20G, 2.4)],
            platform_track=track(self.stamps, metres_per_step=40.0),
        )
        self.assertEqual(verdicts[1].mechanism, MECHANISM_UNDETERMINED)
        self.assertIn(
            "PLATFORM_MOVED_DURING_THE_FADE_SO_A_DIFFRACTING_EDGE_CANNOT_BE_EXCLUDED",
            verdicts[1].reasons,
        )

    def test_an_unknown_geometry_also_declines(self) -> None:
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(ISM_2G4, 0.0), self._fading(KA_20G, 2.4)]
        )
        self.assertEqual(verdicts[1].mechanism, MECHANISM_UNDETERMINED)
        self.assertIn(
            "PATH_GEOMETRY_UNCONFIRMED_SO_A_DIFFRACTING_EDGE_CANNOT_BE_EXCLUDED",
            verdicts[1].reasons,
        )

    def test_below_ten_gigahertz_a_slow_fade_is_not_hydrometeor_attenuation(self) -> None:
        """Rain attenuation at 5.8 GHz is a fraction of a dB per kilometre.

        Labelling a slow fade there as rain would be attributing a mechanism
        that physically could not produce the measured depth.
        """
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(ISM_2G4, 0.0), self._fading(LINK_5G8, 2.4)],
            platform_track=track(self.stamps, metres_per_step=0.0),
        )
        self.assertEqual(verdicts[1].mechanism, MECHANISM_UNDETERMINED)
        self.assertIn(
            "BAND_IS_BELOW_THE_FREQUENCY_AT_WHICH_HYDROMETEOR_ATTENUATION_IS_MEASURABLE",
            verdicts[1].reasons,
        )

    def test_one_band_cannot_reach_the_fade_verdict(self) -> None:
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(KA_20G, 2.4)], platform_track=track(self.stamps, metres_per_step=0.0)
        )
        self.assertEqual(verdicts[0].mechanism, MECHANISM_UNDETERMINED)
        self.assertIn(
            "FEWER_THAN_TWO_BANDS_MEASURED_SO_FREQUENCY_DEPENDENCE_WAS_NOT_TESTED",
            verdicts[0].reasons,
        )

    def test_a_frequency_flat_fade_is_not_called_frequency_selective(self) -> None:
        """A wall of fixed loss fades both bands equally; rain does not."""
        verdicts = self.discriminator.discriminate_bands(
            [self._fading(ISM_2G4, 2.4), self._fading(KA_20G, 2.4)],
            platform_track=track(self.stamps, metres_per_step=0.0),
        )
        self.assertEqual(verdicts[1].mechanism, MECHANISM_UNDETERMINED)
        self.assertIn(
            "FADE_IS_NOT_FREQUENCY_SELECTIVE_ENOUGH_FOR_HYDROMETEOR_ATTENUATION",
            verdicts[1].reasons,
        )

    def test_a_fast_ramp_is_neither_a_step_nor_a_weather_event(self) -> None:
        """Between the two shapes there is no honest mechanism to name."""
        quick = PropagationDiscriminator()
        stamps = moments(18, cadence_seconds=5.0)
        series = [
            BandSeries(
                definition,
                readings(
                    definition,
                    wobbled([-95.0] * 18),
                    _ramped(12, -60.0, step_db, 6),
                    times=stamps,
                ),
            )
            for definition, step_db in ((ISM_2G4, 0.0), (KA_20G, 2.4))
        ]
        verdicts = quick.discriminate_bands(
            series, platform_track=track(stamps, metres_per_step=0.0)
        )
        self.assertEqual(verdicts[1].mechanism, MECHANISM_UNDETERMINED)
        self.assertIn(
            "FADE_ONSET_TOO_FAST_FOR_A_HYDROMETEOR_EVENT_AND_TOO_GRADUAL_FOR_A_STEP",
            verdicts[1].reasons,
        )


class ReceiverWideRiseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()

    def _rising(self, definition: SpectrumBand, rise_db: float = 10.0) -> BandSeries:
        return BandSeries(
            definition,
            readings(
                definition,
                wobbled([-95.0] * 6 + [-95.0 + rise_db] * 4),
                wobbled([-60.0] * 10),
            ),
        )

    def test_the_same_rise_in_three_spread_bands_is_not_attributable(self) -> None:
        """A gain step or a warming LNA reproduces this exactly, and nothing in a
        noise floor separates the receiver from the environment."""
        verdicts = self.discriminator.discriminate_bands(
            [self._rising(ISM_2G4), self._rising(LINK_5G8), self._rising(KA_20G)]
        )
        for verdict in verdicts:
            self.assertEqual(verdict.cause, CAUSE_AMBIGUOUS)
            self.assertFalse(verdict.discriminated)
            self.assertIn(
                "THE_SAME_NOISE_FLOOR_RISE_APPEARED_IN_EVERY_BAND_A_RECEIVER_GAIN"
                "_CHANGE_REPRODUCES_THIS",
                verdict.reasons,
            )

    def test_two_bands_can_sit_inside_one_emitters_bandwidth(self) -> None:
        verdicts = self.discriminator.discriminate_bands(
            [self._rising(ISM_2G4), self._rising(KA_20G)]
        )
        for verdict in verdicts:
            self.assertEqual(verdict.cause, CAUSE_ENERGY_IN_BAND)

    def test_three_bands_rising_by_different_amounts_stay_attributable(self) -> None:
        """Unequal rises are what an emitter does; a gain step is equal everywhere."""
        verdicts = self.discriminator.discriminate_bands(
            [
                self._rising(ISM_2G4, 20.0),
                self._rising(LINK_5G8, 10.0),
                self._rising(KA_20G, 4.0),
            ]
        )
        for verdict in verdicts:
            self.assertEqual(verdict.cause, CAUSE_ENERGY_IN_BAND)


class MultiBandBookkeepingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()

    def test_a_band_that_refuses_still_appears_in_input_order(self) -> None:
        """Dropping it would let a caller read "two bands, one measurable" as "two measured"."""
        good = flat_series(ISM_2G4, count=10)
        thin = flat_series(KA_20G, count=5)
        verdicts = self.discriminator.discriminate_bands([thin, good])
        self.assertEqual(len(verdicts), 2)
        self.assertEqual(verdicts[0].band_id, KA_20G.band_id)
        self.assertEqual(verdicts[0].cause, CAUSE_INSUFFICIENT_EVIDENCE)
        self.assertEqual(verdicts[1].band_id, ISM_2G4.band_id)


class InputHygieneTests(unittest.TestCase):
    def test_observations_from_another_band_are_a_wiring_error(self) -> None:
        """Frequency is what makes the rain test possible; the wrong band poisons it."""
        with self.assertRaises(PropagationError):
            BandSeries(KA_20G, readings(ISM_2G4, wobbled([-95.0] * 10), wobbled([-60.0] * 10)))

    def test_a_recent_window_that_starves_the_baseline_is_refused(self) -> None:
        with self.assertRaises(PropagationError):
            PropagationDiscriminator().discriminate(
                flat_series(LINK_5G8, count=10), recent_samples=8
            )

    def test_an_explicit_recent_window_is_honoured(self) -> None:
        verdict = PropagationDiscriminator().discriminate(
            flat_series(LINK_5G8, count=14), recent_samples=5
        )
        self.assertEqual(verdict.recent_samples, 5)
        self.assertEqual(verdict.baseline_samples, 9)

    def test_the_default_split_keeps_the_baseline_larger_than_the_window_it_judges(self) -> None:
        verdict = PropagationDiscriminator().discriminate(flat_series(LINK_5G8, count=30))
        self.assertGreaterEqual(verdict.baseline_samples, 2 * verdict.recent_samples)


class SerialisationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.discriminator = PropagationDiscriminator()

    def _verdict(self) -> PropagationVerdict:
        return self.discriminator.discriminate(
            BandSeries(
                LINK_5G8,
                readings(
                    LINK_5G8, wobbled([-95.0] * 10), wobbled([-60.0] * 6 + [-72.0] * 4)
                ),
            )
        )

    def test_a_verdict_serialises(self) -> None:
        json.dumps(self._verdict().to_dict())

    def test_a_refusal_serialises_without_an_infinity(self) -> None:
        """RFC 8259 has no infinity literal; emitting one breaks a strict parser."""
        encoded = json.dumps(
            self.discriminator.discriminate(flat_series(LINK_5G8, count=5)).to_dict()
        )
        self.assertNotIn("Infinity", encoded)
        self.assertNotIn("NaN", encoded)

    def test_no_source_and_no_intent_are_ever_carried(self) -> None:
        """A rising floor says energy arrived. It does not say who, or why."""
        payload = self._verdict().to_dict()
        self.assertIsNone(payload["source"])
        self.assertIsNone(payload["intent"])
        self.assertIn(
            "NO_SOURCE_AND_NO_INTENT_ARE_CLAIMED_ENERGY_IN_A_BAND_IS_AN_OBSERVATION",
            payload["reasons"],
        )

    def test_a_consistent_mechanism_is_never_reported_as_an_observed_one(self) -> None:
        payload = self._verdict().to_dict()
        self.assertIn(
            "A_MECHANISM_CONSISTENT_WITH_THE_SIGNATURE_IS_NOT_A_MECHANISM_OBSERVED",
            payload["reasons"],
        )

    def test_every_verdict_carries_the_sample_count_it_rests_on(self) -> None:
        payload = self._verdict().to_dict()
        self.assertEqual(payload["samples"], 10)
        self.assertEqual(payload["baseline_samples"] + payload["recent_samples"], 10)

    def test_reasons_are_screaming_snake_case_and_never_repeat(self) -> None:
        reasons = self._verdict().to_dict()["reasons"]
        self.assertEqual(len(reasons), len(set(reasons)))
        for reason in reasons:
            self.assertRegex(reason, r"^[A-Z0-9_]+$")


if __name__ == "__main__":
    unittest.main()
