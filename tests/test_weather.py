"""Which weather mechanism a fade is consistent with, and which it is not.

Two kinds of test live here, and the first kind is the reason the module exists.

**The coefficients are checked against the published tables**, not against
themselves. A fabricated ``k`` or ``alpha`` is a plausible number nobody can
check, and it would pass any test written from the same fabrication. So the
ITU-R P.838-3 curve fits are evaluated and compared against that
recommendation's *own* Table 5, which is a separate artefact derived from the
same scattering calculations by a different route — if a coefficient had been
transcribed wrongly, the two would disagree.

**The refusals are exercised at least as hard as the findings.** A
discriminator that always names a mechanism is one whose names carry no
information, and most of this module is the machinery that declines: too few
bands, too narrow a span, too shallow a fade, a model asked outside its
published range, a mechanism with no verifiable frequency law at all.

See ``src/kval_node/weather.py``.
"""

from __future__ import annotations

import json
import math
import unittest

from kval_node.weather import (
    DEFAULT_CLOUD_TEMPERATURE_K,
    DUCTING_TRAPPING_GRADIENT_N_PER_KM,
    FINDING_CONSISTENT,
    FINDING_EXCLUDED,
    FINDING_INDETERMINATE,
    MAX_EXPONENT_STANDARD_ERROR,
    MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H,
    MECHANISM_CLOUD_LIQUID_WATER,
    MECHANISM_DRY_SNOW,
    MECHANISM_DUCTING,
    MECHANISM_MELTING_LAYER,
    MECHANISM_OXYGEN,
    MECHANISM_RAIN,
    MECHANISM_SAND_AND_DUST,
    MECHANISM_SCINTILLATION,
    MECHANISM_WATER_VAPOUR,
    MECHANISM_WET_FOLIAGE,
    MECHANISMS,
    MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H,
    OXYGEN_COMPLEX_HIGH_HZ,
    OXYGEN_COMPLEX_LOW_HZ,
    OXYGEN_ISOLATED_LINE_HZ,
    SCINTILLATION_SIGMA_EXPONENT,
    SENSE_ATTENUATION,
    SENSE_ENHANCEMENT,
    SENSE_MIXED,
    SENSE_NONE,
    WATER_VAPOUR_PRINCIPAL_LINE_HZ,
    BandExcess,
    DeclaredConditions,
    WeatherDiscriminator,
    WeatherModelError,
    WeatherSignature,
    cloud_attenuation_coefficient,
    cloud_exponent,
    cloud_specific_attenuation,
    fit_scintillation_exponent,
    fit_spectral_exponent,
    implied_rain_rate,
    rain_coefficients,
    rain_exponent,
    rain_specific_attenuation,
    scintillation_aperture_factor,
)

#: ITU-R P.838-3 Table 5, "Frequency-dependent coefficients for estimating
#: specific rain attenuation", as ``(f GHz, k_H, alpha_H, k_V, alpha_V)``. A
#: representative span of the published table rather than all of it — enough
#: frequencies that a wrong constant in any of the four Gaussian sets shows up,
#: including the 4-6 GHz region where ``alpha_H`` peaks above 1.6 and a
#: transcription error would be least obvious.
P838_TABLE_5 = (
    (1.0, 0.0000259, 0.9691, 0.0000308, 0.8592),
    (2.0, 0.0000847, 1.0664, 0.0000998, 0.9490),
    (4.0, 0.0001071, 1.6009, 0.0002461, 1.2476),
    (6.0, 0.0007056, 1.5900, 0.0004878, 1.5728),
    (8.0, 0.004115, 1.3905, 0.003450, 1.3797),
    (10.0, 0.01217, 1.2571, 0.01129, 1.2156),
    (12.0, 0.02386, 1.1825, 0.02455, 1.1216),
    (15.0, 0.04481, 1.1233, 0.05008, 1.0440),
    (20.0, 0.09164, 1.0568, 0.09611, 0.9847),
    (25.0, 0.1571, 0.9991, 0.1533, 0.9491),
    (30.0, 0.2403, 0.9485, 0.2291, 0.9129),
    (35.0, 0.3374, 0.9047, 0.3224, 0.8761),
    (40.0, 0.4431, 0.8673, 0.4274, 0.8421),
    (50.0, 0.6600, 0.8084, 0.6472, 0.7871),
    (60.0, 0.8606, 0.7656, 0.8515, 0.7486),
    (80.0, 1.1704, 0.7115, 1.1668, 0.7021),
)

#: Polarisation tilts at which ITU-R P.838-3 eq. (4) and (5) collapse to the
#: pure horizontal and pure vertical cases: ``cos(2 tau)`` is +1 and -1.
HORIZONTAL_TILT_DEG = 0.0
VERTICAL_TILT_DEG = 90.0


def rain_bands(
    frequencies_hz, rain_rate_mm_per_h: float, path_length_km: float
) -> tuple[BandExcess, ...]:
    """Band excesses a path of known length would show under a known rain rate.

    Built from the published model rather than from numbers chosen to pass, so
    a test that recovers the rain rate is testing the inversion against the
    forward model and not against a hand-tuned constant.
    """
    return tuple(
        BandExcess(
            frequency_hz=frequency,
            excess_db=rain_specific_attenuation(
                frequency, rain_rate_mm_per_h, path_length_km=path_length_km
            ).path_attenuation_db,
        )
        for frequency in frequencies_hz
    )


def cloud_bands(
    frequencies_hz, liquid_water_g_per_m3: float, path_length_km: float
) -> tuple[BandExcess, ...]:
    """Band excesses a path of known length would show under known cloud water."""
    return tuple(
        BandExcess(
            frequency_hz=frequency,
            excess_db=cloud_specific_attenuation(
                frequency, liquid_water_g_per_m3, path_length_km=path_length_km
            ).path_attenuation_db,
        )
        for frequency in frequencies_hz
    )


def finding_for(assessment, mechanism: str) -> str:
    for item in assessment.findings:
        if item.mechanism == mechanism:
            return item.finding
    raise AssertionError(f"{mechanism} was not assessed")


class PublishedCoefficientTests(unittest.TestCase):
    """The constants are the product. Every one is checked against its source."""

    def test_rain_coefficients_reproduce_the_itu_r_p838_3_published_table(self) -> None:
        """Catches a mistyped Gaussian constant in any of the four coefficient sets.

        The recommendation gives both the curve-fit constants (Tables 1-4) and a
        table of the resulting ``k`` and ``alpha`` at listed frequencies
        (Table 5). Evaluating the fits and comparing against Table 5 is the only
        check available offline that a transcription error would fail — a test
        written from the same transcription would agree with itself no matter
        what was typed.
        """
        for frequency_ghz, k_h, alpha_h, k_v, alpha_v in P838_TABLE_5:
            with self.subTest(frequency_ghz=frequency_ghz):
                got_k_h, got_alpha_h = rain_coefficients(
                    frequency_ghz * 1e9,
                    polarisation_tilt_deg=HORIZONTAL_TILT_DEG,
                    path_elevation_deg=0.0,
                )
                got_k_v, got_alpha_v = rain_coefficients(
                    frequency_ghz * 1e9,
                    polarisation_tilt_deg=VERTICAL_TILT_DEG,
                    path_elevation_deg=0.0,
                )
                # Table 5 is printed to four significant figures, so agreement
                # to a part in a thousand is agreement to the table's own
                # precision.
                self.assertAlmostEqual(got_k_h / k_h, 1.0, delta=1e-3)
                self.assertAlmostEqual(got_k_v / k_v, 1.0, delta=1e-3)
                self.assertAlmostEqual(got_alpha_h, alpha_h, delta=2e-4)
                self.assertAlmostEqual(got_alpha_v, alpha_v, delta=2e-4)

    def test_circular_polarisation_is_the_mean_of_the_two_linear_coefficients(
        self,
    ) -> None:
        """Catches a sign error in the ITU-R P.838-3 eq. (4) geometry term.

        At 45 degrees ``cos(2 tau)`` is zero, so the tilt term must vanish and
        the result must be the plain mean of horizontal and vertical, at every
        elevation. A module that got the factor of two wrong would still look
        reasonable at one tilt and would be wrong everywhere else.
        """
        horizontal, _ = rain_coefficients(
            20e9, polarisation_tilt_deg=HORIZONTAL_TILT_DEG
        )
        vertical, _ = rain_coefficients(20e9, polarisation_tilt_deg=VERTICAL_TILT_DEG)
        for elevation in (0.0, 15.0, 45.0, 90.0):
            with self.subTest(elevation=elevation):
                circular, _ = rain_coefficients(
                    20e9, polarisation_tilt_deg=45.0, path_elevation_deg=elevation
                )
                self.assertAlmostEqual(
                    circular, (horizontal + vertical) / 2.0, places=12
                )

    def test_rain_specific_attenuation_matches_the_figures_the_sibling_module_quotes(
        self,
    ) -> None:
        """Keeps this module and ``propagation.py`` telling operators the same story.

        ``propagation.py`` justifies its 10 GHz hydrometeor floor with two
        numbers: 25 mm/h gives of the order of 0.7 dB/km at 10 GHz and of the
        order of 0.003 dB/km at 2 GHz. If this module's coefficients disagreed,
        one of the two modules would be lying about the same physics.
        """
        # The sibling module's figures are the horizontal-polarisation case,
        # which is what a terrestrial link usually is. Circular gives 0.63 dB/km
        # at 10 GHz, and stating which is which is the point of the tilt term.
        at_ten = rain_specific_attenuation(
            10e9, 25.0, polarisation_tilt_deg=HORIZONTAL_TILT_DEG
        )
        at_two = rain_specific_attenuation(
            2e9, 25.0, polarisation_tilt_deg=HORIZONTAL_TILT_DEG
        )
        self.assertTrue(at_ten.computed)
        self.assertTrue(at_two.computed)
        self.assertAlmostEqual(
            at_ten.specific_attenuation_db_per_km, 0.7, delta=0.05
        )
        self.assertAlmostEqual(
            at_two.specific_attenuation_db_per_km, 0.003, delta=0.0005
        )

    def test_cloud_coefficient_rises_as_frequency_squared_in_the_rayleigh_regime(
        self,
    ) -> None:
        """Catches a wrong Debye constant, which would break the f-squared law.

        The whole rain-versus-cloud discrimination rests on cloud following a
        Rayleigh ``f**2``. Well below the principal relaxation frequency the
        ITU-R P.840 model must produce an exponent of two to several decimal
        places; a mistyped permittivity or relaxation constant would bend it.
        """
        for temperature in (263.15, 273.15, 293.15):
            with self.subTest(temperature=temperature):
                exponent = cloud_exponent(
                    (1e9, 3e9), temperature_k=temperature
                )
                self.assertAlmostEqual(exponent, 2.0, delta=0.01)

    def test_cloud_exponent_does_not_depend_on_liquid_water_density(self) -> None:
        """Guards the property that makes the cloud test usable without a met feed.

        ``gamma_c = K_l M`` is linear in ``M``, so ``M`` is a constant offset in
        the logarithm and cannot move a slope. If it ever did, the cloud test
        would silently need a quantity this node has no way to obtain.
        """
        # Both densities are chosen so that every band clears the measurable
        # floor: a band dropped from one fit and kept in the other would make
        # this test compare two different band sets and pass for the wrong
        # reason.
        frequencies = (10e9, 20e9, 40e9)
        thin = cloud_bands(frequencies, 1.0, 50.0)
        thick = cloud_bands(frequencies, 3.0, 50.0)
        self.assertAlmostEqual(
            fit_spectral_exponent(thin).exponent,
            fit_spectral_exponent(thick).exponent,
            places=9,
        )

    def test_cloud_attenuation_is_far_weaker_than_rain_per_unit_path(self) -> None:
        """Records the ordering the module's prose claims, from the models themselves.

        The docstrings tell operators that cloud is much weaker per kilometre
        than rain. That is a claim about two published models and it should
        break the build if it ever stops being true of the code.
        """
        thick_fog = cloud_specific_attenuation(20e9, 0.5)
        moderate_rain = rain_specific_attenuation(20e9, 25.0)
        self.assertLess(
            thick_fog.specific_attenuation_db_per_km,
            moderate_rain.specific_attenuation_db_per_km / 10.0,
        )

    def test_ducting_trapping_gradient_equals_the_earth_curvature_it_is_derived_from(
        self,
    ) -> None:
        """Catches the constant drifting away from the derivation in its comment.

        The trapping gradient is not a tabulated measurement — it is
        ``-1e6 / R_e``. Writing the round number without the derivation would
        make it unfalsifiable, so the derivation is asserted.
        """
        earth_radius_km = 6371.0
        self.assertAlmostEqual(
            DUCTING_TRAPPING_GRADIENT_N_PER_KM, -1e6 / earth_radius_km, delta=0.1
        )

    def test_scintillation_aperture_factor_is_unity_for_a_point_antenna(self) -> None:
        """Underpins the claim that a small node aperture leaves 7/12 intact.

        The module tests a measured sigma exponent against 7/12 and justifies it
        by ``g(x)`` being flat for a small aperture. If ``g(0)`` were not one,
        or if the ``x >= 7`` cut-off were missing, that justification would be
        wrong and the exponent test would be biased without saying so.
        """
        self.assertAlmostEqual(
            scintillation_aperture_factor(10e9, 0.0, 30.0), 1.0, delta=0.001
        )
        # ITU-R P.618: at x >= 7 the aperture has averaged the turbulence away.
        self.assertEqual(scintillation_aperture_factor(20e9, 100.0, 30.0), 0.0)

    def test_gaseous_line_centres_sit_where_itu_r_p676_puts_them(self) -> None:
        """Catches a transposed digit in a line frequency.

        These centres decide whether a band is treated as sitting on an
        absorption feature. A digit wrong by a GHz would silently move a real
        band out of the window and a clear band into it.
        """
        self.assertAlmostEqual(WATER_VAPOUR_PRINCIPAL_LINE_HZ, 22.235080e9, places=1)
        self.assertAlmostEqual(OXYGEN_COMPLEX_LOW_HZ, 50.474214e9, places=1)
        self.assertAlmostEqual(OXYGEN_COMPLEX_HIGH_HZ, 68.960312e9, places=1)
        self.assertAlmostEqual(OXYGEN_ISOLATED_LINE_HZ, 118.750334e9, places=1)


class SpectralExponentTests(unittest.TestCase):
    """The exponent is the discriminator, so its refusals are load-bearing."""

    def test_exponent_is_unchanged_by_path_length(self) -> None:
        """The property that lets a node discriminate without knowing the geometry.

        Path length multiplies every band's excess by the same factor, which is
        an additive constant in the logarithm. If the fit ever became sensitive
        to it, every mechanism test would silently start depending on a distance
        this node cannot measure.
        """
        # Both paths are long enough that every band clears the measurable
        # floor. A shorter path would drop the lowest band, and the fits would
        # then differ because they used different bands rather than because the
        # exponent moved.
        short = fit_spectral_exponent(rain_bands((10e9, 20e9, 40e9), 50.0, 5.0))
        long = fit_spectral_exponent(rain_bands((10e9, 20e9, 40e9), 50.0, 20.0))
        self.assertEqual(short.bands_used, long.bands_used)
        self.assertAlmostEqual(short.exponent, long.exponent, places=9)
        # The depths differ by a factor of four, so this is not two fits of the
        # same numbers. The exponent is identical while its uncertainty falls in
        # exact proportion to the depth, which is the weighting working: a
        # deeper fade pins the same slope more tightly.
        self.assertAlmostEqual(
            short.standard_error / long.standard_error, 4.0, places=6
        )

    def test_a_single_moved_band_yields_no_frequency_law(self) -> None:
        """Prevents a one-band observation being reported as a frequency slope.

        One point defines no slope. Returning any number here would be inventing
        the one measurement the mechanism tests most depend on.
        """
        fit = fit_spectral_exponent(
            (BandExcess(20e9, 12.0), BandExcess(10e9, 0.4))
        )
        self.assertIsNone(fit.exponent)
        self.assertIn("FEWER_THAN_2_BANDS_MOVED_MEASURABLY", fit.refusal)
        self.assertEqual(fit.bands_supplied, 2)
        self.assertEqual(fit.bands_used, 1)

    def test_bands_too_close_in_frequency_yield_no_frequency_law(self) -> None:
        """Prevents calibration scatter between adjacent bands becoming an exponent.

        Dividing a difference in decibels by a small log span manufactures an
        enormous exponent from an instrument error that has nothing to do with
        the atmosphere.
        """
        fit = fit_spectral_exponent(
            (BandExcess(20.0e9, 10.0), BandExcess(20.4e9, 14.0))
        )
        self.assertIsNone(fit.exponent)
        self.assertEqual(
            fit.refusal, "BANDS_SPAN_TOO_LITTLE_FREQUENCY_TO_RESOLVE_A_POWER_LAW"
        )

    def test_bands_moving_in_opposite_directions_fit_no_single_power_law(self) -> None:
        """Prevents two mechanisms being averaged into a third that is neither.

        A band that got worse and a band that got better are not one event, and
        a power law fitted through both describes nothing that happened.
        """
        fit = fit_spectral_exponent(
            (BandExcess(10e9, 12.0), BandExcess(20e9, -9.0))
        )
        self.assertIsNone(fit.exponent)
        self.assertEqual(fit.sense, SENSE_MIXED)
        self.assertEqual(
            fit.refusal, "BANDS_MOVED_IN_OPPOSITE_DIRECTIONS_SO_NO_SINGLE_POWER_LAW_FITS"
        )

    def test_a_shallow_fade_is_reported_as_too_uncertain_rather_than_declined(
        self,
    ) -> None:
        """Keeps an imprecise exponent visible without letting it decide anything.

        A fade only a few decibels deep carries a real but weak exponent. Hiding
        it would lose information; acting on it would let a one-decibel
        calibration error pick a mechanism. Both the number and its unusability
        are reported.
        """
        fit = fit_spectral_exponent(
            (BandExcess(12e9, 3.5), BandExcess(18e9, 4.6))
        )
        self.assertIsNotNone(fit.exponent)
        self.assertGreater(fit.standard_error, MAX_EXPONENT_STANDARD_ERROR)
        self.assertFalse(fit.usable)
        self.assertIsNone(fit.refusal)

    def test_a_deeper_fade_produces_a_tighter_exponent_than_a_shallow_one(self) -> None:
        """Guards the per-band weighting against regressing to a uniform error.

        A fade is measured to about a decibel whatever its depth, so a 40 dB
        band pins its logarithm far more tightly than a 4 dB band. Assigning
        every band the same log-space error — or worse, the shallowest band's —
        throws away the deep bands that carry the slope and makes the module
        decline on data that was adequate.
        """
        shallow = fit_spectral_exponent(
            (BandExcess(10e9, 4.0), BandExcess(40e9, 16.0))
        )
        deep = fit_spectral_exponent(
            (BandExcess(10e9, 40.0), BandExcess(40e9, 160.0))
        )
        self.assertAlmostEqual(shallow.exponent, deep.exponent, places=9)
        self.assertLess(deep.standard_error, shallow.standard_error)

    def test_scintillation_fit_declines_when_no_band_reported_a_fluctuation(
        self,
    ) -> None:
        """Prevents a missing standard deviation being read as a quiet band.

        A band that did not report its scatter is not a band that had none, and
        substituting zero would make every quiet-looking observation into
        evidence against scintillation.
        """
        fit = fit_scintillation_exponent(
            (BandExcess(10e9, 1.0), BandExcess(20e9, 1.0))
        )
        self.assertIsNone(fit.exponent)
        self.assertIn("FLUCTUATION_STANDARD_DEVIATION", fit.refusal)


class RainInversionTests(unittest.TestCase):
    """The closed-form inversion of ITU-R P.838-3, and where it must refuse."""

    def test_the_rain_rate_that_built_a_fade_is_recovered_from_its_shape_alone(
        self,
    ) -> None:
        """The core claim: shape gives rain rate with no knowledge of the path.

        Fades are synthesised from the forward model at a known rate over a
        known path, then the rate is recovered from the exponent alone. If the
        inversion and the forward model ever disagreed, every rain finding would
        be reporting a rainfall figure that the model it cites does not support.
        """
        cases = (
            (1.0, (20e9, 30e9, 40e9), 60.0),
            (5.0, (10e9, 20e9, 40e9), 40.0),
            (25.0, (10e9, 20e9, 40e9), 6.0),
            (60.0, (10e9, 20e9, 40e9), 4.0),
            (120.0, (10e9, 20e9, 40e9), 3.0),
        )
        for rate, frequencies, path_km in cases:
            with self.subTest(rate=rate, path_km=path_km):
                signature = WeatherSignature(
                    rain_bands(frequencies, rate, path_km), onset_seconds=900.0
                )
                assessment = WeatherDiscriminator().assess(signature)
                implied = assessment.implied_rain
                self.assertIsNotNone(implied)
                self.assertAlmostEqual(
                    implied.rain_rate_mm_per_h / rate, 1.0, delta=1e-6
                )
                self.assertTrue(implied.plausible)
                self.assertEqual(
                    finding_for(assessment, MECHANISM_RAIN), FINDING_CONSISTENT
                )

    def test_an_exponent_no_rain_rate_reaches_is_reported_as_implausible(self) -> None:
        """The exclusion test: rain is ruled out by the model, not by assertion.

        A slope far steeper than drizzle produces implies a rainfall figure
        below anything a gauge registers, and saying so is a stronger statement
        than "the slope looked wrong".
        """
        steep = implied_rain_rate(5.0, (10e9, 40e9))
        self.assertFalse(steep.plausible)
        self.assertLess(steep.rain_rate_mm_per_h, MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H)

    def test_the_inversion_declines_where_alpha_barely_changes(self) -> None:
        """Prevents an arbitrary rainfall figure from a near-zero denominator.

        The inversion divides by the change in ``alpha`` across the bands. Over
        a narrow pair that change approaches zero, and dividing measurement
        scatter by it would produce a confident number from nothing.
        """
        # ``alpha_H`` peaks near 5 GHz, so a C-band pair straddling that peak
        # has almost the same exponent at both ends whatever the rain rate. The
        # whole plausible rainfall range then maps into a slope difference
        # smaller than the measurement's own uncertainty.
        straddling_the_alpha_peak = (4.8e9, 6.0e9)
        narrow = implied_rain_rate(1.5, straddling_the_alpha_peak)
        self.assertIsNone(narrow.rain_rate_mm_per_h)
        self.assertFalse(narrow.plausible)
        self.assertLess(
            abs(narrow.sensitivity_per_decade), MAX_EXPONENT_STANDARD_ERROR
        )
        self.assertIn("CARRIES_NO_RAIN_RATE_INFORMATION", narrow.reasons[0])

    def test_inverting_a_refused_fit_raises_rather_than_coercing_none(self) -> None:
        """A refusal has no exponent, and turning one into a number is the bug.

        Coercing ``None`` would raise a TypeError from inside a validator, which
        tells a caller nothing about what they did wrong.
        """
        with self.assertRaises(WeatherModelError):
            implied_rain_rate(None, (10e9, 20e9))

    def test_the_model_exponent_is_linear_in_the_logarithm_of_rain_rate(self) -> None:
        """Guards the identity the closed-form inversion is built on.

        ``n = intercept + sensitivity * log10 R`` is exact, not an
        approximation. If it ever stopped holding, the inversion would need a
        search and its answers would quietly become approximate.
        """
        frequencies = (10e9, 30e9)
        first = rain_exponent(frequencies, 1.0)
        second = rain_exponent(frequencies, 10.0)
        third = rain_exponent(frequencies, 100.0)
        self.assertAlmostEqual(second - first, third - second, places=12)


class SignatureMatchingTests(unittest.TestCase):
    """Which mechanisms survive a signature, and which the physics rules out."""

    def setUp(self) -> None:
        self.discriminator = WeatherDiscriminator()

    def test_a_heavy_rain_fade_excludes_cloud_because_the_slope_is_too_shallow(
        self,
    ) -> None:
        """The clean separation the module exists to make.

        At a tropical rain rate the ITU-R P.838-3 exponent falls near 1.2, well
        below the Rayleigh f-squared law cloud is confined to. This is the case
        where slope alone genuinely decides, and it must decide.
        """
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 100.0, 5.0), onset_seconds=900.0
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(finding_for(assessment, MECHANISM_RAIN), FINDING_CONSISTENT)
        self.assertEqual(
            finding_for(assessment, MECHANISM_CLOUD_LIQUID_WATER), FINDING_EXCLUDED
        )

    def test_a_moderate_rain_fade_leaves_both_rain_and_cloud_consistent(self) -> None:
        """The honest non-separation, which a naive module would get wrong.

        The textbook shorthand says rain is f-to-the-one and cloud f-squared, so
        they always separate. Computed from the published models they do not: at
        around 25 mm/h between 10 and 40 GHz the rain exponent runs straight
        through the cloud window. Naming one mechanism here would be a guess
        dressed as a measurement, so both are returned.
        """
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 25.0, 5.0), onset_seconds=900.0
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(finding_for(assessment, MECHANISM_RAIN), FINDING_CONSISTENT)
        self.assertEqual(
            finding_for(assessment, MECHANISM_CLOUD_LIQUID_WATER), FINDING_CONSISTENT
        )
        self.assertGreater(len(assessment.consistent), 1)

    def test_declaring_a_rain_rate_separates_what_slope_alone_could_not(self) -> None:
        """Shows the declared-conditions path narrowing an honest ambiguity.

        A fog signature is consistent with rain only because the undeclared rain
        rate is free to be whatever fits. Once an operator asserts a rate, the
        model predicts one exponent and the measurement can disagree with it.
        """
        signature = WeatherSignature(
            cloud_bands((10e9, 20e9, 40e9), 1.0, 40.0), onset_seconds=5400.0
        )
        without = self.discriminator.assess(signature)
        self.assertEqual(finding_for(without, MECHANISM_RAIN), FINDING_CONSISTENT)

        with_rate = self.discriminator.assess(
            signature, DeclaredConditions(rain_rate_mm_per_h=25.0)
        )
        self.assertEqual(finding_for(with_rate, MECHANISM_RAIN), FINDING_EXCLUDED)
        self.assertEqual(
            finding_for(with_rate, MECHANISM_CLOUD_LIQUID_WATER), FINDING_CONSISTENT
        )

    def test_a_fade_faster_than_a_rain_cell_excludes_rain(self) -> None:
        """Keeps a change of geometry from being attributed to the sky.

        Rain cells are kilometres across and their rates change over minutes. A
        fade that completes in seconds is a mast crossing the path, and calling
        it weather would send an operator to wait out something that will not
        clear on its own.
        """
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 60.0, 5.0), onset_seconds=8.0
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(finding_for(assessment, MECHANISM_RAIN), FINDING_EXCLUDED)
        self.assertEqual(
            finding_for(assessment, MECHANISM_CLOUD_LIQUID_WATER), FINDING_EXCLUDED
        )

    def test_an_untimed_onset_leaves_time_dependent_mechanisms_undecided_by_time(
        self,
    ) -> None:
        """Prevents a missing onset being read as an onset that passed every test.

        A fade whose onset could not be timed has not shown it is slow. The
        assessment says so rather than quietly skipping the check.
        """
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 100.0, 5.0), onset_seconds=None
        )
        assessment = self.discriminator.assess(signature)
        self.assertIn(
            "ONSET_WAS_NOT_TIMED_SO_EVERY_TIME_CONSTANT_TEST_DECLINED",
            assessment.reasons,
        )

    def test_an_enhancement_excludes_every_absorptive_mechanism(self) -> None:
        """Stops a duct being described as a kind of loss.

        Nothing on the path that absorbs energy can raise received power. A
        module that only modelled loss would have to force this observation into
        an absorptive mechanism, which is the failure the signed excess exists
        to prevent.
        """
        signature = WeatherSignature(
            tuple(BandExcess(frequency, -8.0) for frequency in (5e9, 10e9, 18e9)),
            onset_seconds=3600.0,
        )
        assessment = self.discriminator.assess(
            signature, DeclaredConditions(refractivity_gradient_n_per_km=-220.0)
        )
        self.assertEqual(assessment.sense, SENSE_ENHANCEMENT)
        for mechanism in (
            MECHANISM_RAIN,
            MECHANISM_CLOUD_LIQUID_WATER,
            MECHANISM_MELTING_LAYER,
            MECHANISM_WATER_VAPOUR,
            MECHANISM_OXYGEN,
            MECHANISM_SAND_AND_DUST,
            MECHANISM_WET_FOLIAGE,
        ):
            with self.subTest(mechanism=mechanism):
                self.assertEqual(finding_for(assessment, mechanism), FINDING_EXCLUDED)
        self.assertEqual(finding_for(assessment, MECHANISM_DUCTING), FINDING_CONSISTENT)
        self.assertTrue(assessment.resolved)

    def test_a_declared_refractivity_gradient_above_trapping_excludes_ducting(
        self,
    ) -> None:
        """Uses the one quantitative ducting criterion there is.

        A duct needs the refractivity gradient at or below -157 N-units per km.
        Above that no ray is trapped, whatever the received power did, and the
        enhancement has some other cause.
        """
        signature = WeatherSignature(
            tuple(BandExcess(frequency, -8.0) for frequency in (5e9, 10e9, 18e9)),
            onset_seconds=3600.0,
        )
        assessment = self.discriminator.assess(
            signature, DeclaredConditions(refractivity_gradient_n_per_km=-100.0)
        )
        self.assertEqual(finding_for(assessment, MECHANISM_DUCTING), FINDING_EXCLUDED)
        self.assertEqual(assessment.consistent, ())
        self.assertIn(
            "NO_MODELLED_MECHANISM_IS_CONSISTENT_WITH_THIS_SIGNATURE",
            assessment.reasons,
        )

    def test_a_strong_frequency_law_excludes_ducting_because_refraction_has_none(
        self,
    ) -> None:
        """Separates a refractive mechanism from an absorptive one by shape.

        Refraction redirects energy rather than absorbing it, so it does not
        produce the power-law slope every absorptive mechanism here shows. An
        enhancement with a steep slope is not a duct.
        """
        signature = WeatherSignature(
            (
                BandExcess(10e9, -4.0),
                BandExcess(20e9, -16.0),
                BandExcess(40e9, -64.0),
            ),
            onset_seconds=3600.0,
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(finding_for(assessment, MECHANISM_DUCTING), FINDING_EXCLUDED)

    def test_ducting_is_undetermined_rather_than_consistent_when_flatness_is_untested(
        self,
    ) -> None:
        """Stops every unfittable fade from looking like a duct.

        Flatness in frequency is the only test this module has for a duct. When
        no frequency law could be established there is nothing to test, and
        reporting consistency on an untested property would make ducting the
        default answer for weak evidence.
        """
        signature = WeatherSignature(
            (BandExcess(10e9, -12.0),), onset_seconds=3600.0
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(
            finding_for(assessment, MECHANISM_DUCTING), FINDING_INDETERMINATE
        )

    def test_precipitation_with_no_fade_is_consistent_with_dry_snow(self) -> None:
        """The informative negative: absence of a fade is itself evidence.

        Ice has too little dielectric loss to attenuate measurably, so a winter
        deployment reporting precipitation and a quiet band has learned
        something real. A module that only looked for fades would discard it.
        """
        signature = WeatherSignature(
            (BandExcess(8e9, 0.5), BandExcess(12e9, 0.8)), onset_seconds=None
        )
        assessment = self.discriminator.assess(
            signature, DeclaredConditions(precipitation_observed=True)
        )
        self.assertEqual(assessment.sense, SENSE_NONE)
        self.assertEqual(finding_for(assessment, MECHANISM_DRY_SNOW), FINDING_CONSISTENT)

    def test_a_measurable_fade_excludes_dry_snow(self) -> None:
        """Keeps the qualitative dry-snow argument from being applied backwards."""
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 25.0, 5.0), onset_seconds=900.0
        )
        assessment = self.discriminator.assess(
            signature, DeclaredConditions(precipitation_observed=True)
        )
        self.assertEqual(finding_for(assessment, MECHANISM_DRY_SNOW), FINDING_EXCLUDED)

    def test_a_quiet_band_without_declared_precipitation_leaves_dry_snow_undecided(
        self,
    ) -> None:
        """Prevents a dry sky and a snowfall that does not attenuate looking alike.

        Both produce a quiet band. Without a declaration the node cannot tell
        them apart, and calling it dry snow would be inventing weather.
        """
        signature = WeatherSignature((BandExcess(8e9, 0.5), BandExcess(12e9, 0.8)))
        assessment = self.discriminator.assess(signature)
        self.assertEqual(
            finding_for(assessment, MECHANISM_DRY_SNOW), FINDING_INDETERMINATE
        )

    def test_the_melting_layer_is_never_separated_from_rain_by_slope(self) -> None:
        """Records that this mechanism has no slope test, rather than faking one.

        Melting-layer excess attenuation scales rain attenuation, so it shares
        rain's frequency law. Whatever the exponent says, it says the same about
        both. Only a declared 0 C isotherm crossing separates them, and this
        node cannot observe an isotherm.
        """
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 25.0, 5.0), onset_seconds=900.0
        )
        undeclared = self.discriminator.assess(signature)
        self.assertEqual(
            finding_for(undeclared, MECHANISM_MELTING_LAYER), FINDING_INDETERMINATE
        )

        declared = self.discriminator.assess(
            signature, DeclaredConditions(melting_layer_on_path=True)
        )
        self.assertEqual(
            finding_for(declared, MECHANISM_MELTING_LAYER), FINDING_CONSISTENT
        )
        absent = self.discriminator.assess(
            signature, DeclaredConditions(melting_layer_on_path=False)
        )
        self.assertEqual(
            finding_for(absent, MECHANISM_MELTING_LAYER), FINDING_EXCLUDED
        )

    def test_wet_foliage_is_always_undetermined_when_a_band_faded(self) -> None:
        """The refusal the task of this module most depends on getting right.

        Wet vegetation loss has no published frequency law that could be carried
        with a citation, so there is no test to run. Reporting it as excluded
        would be claiming a measurement that was never made; reporting it as
        consistent would be claiming a mechanism that was never shown.
        """
        for rate in (5.0, 100.0):
            with self.subTest(rate=rate):
                signature = WeatherSignature(
                    rain_bands((10e9, 20e9, 40e9), rate, 5.0), onset_seconds=900.0
                )
                assessment = self.discriminator.assess(signature)
                self.assertEqual(
                    finding_for(assessment, MECHANISM_WET_FOLIAGE),
                    FINDING_INDETERMINATE,
                )
                self.assertIn(MECHANISM_WET_FOLIAGE, assessment.indeterminate)

    def test_sand_and_dust_is_undetermined_because_no_coefficient_was_verifiable(
        self,
    ) -> None:
        """States the absence of a source instead of substituting a guessed one."""
        signature = WeatherSignature(
            rain_bands((10e9, 20e9, 40e9), 25.0, 5.0), onset_seconds=900.0
        )
        arid = self.discriminator.assess(
            signature, DeclaredConditions(arid_environment=True)
        )
        self.assertEqual(
            finding_for(arid, MECHANISM_SAND_AND_DUST), FINDING_INDETERMINATE
        )
        temperate = self.discriminator.assess(
            signature, DeclaredConditions(arid_environment=False)
        )
        self.assertEqual(
            finding_for(temperate, MECHANISM_SAND_AND_DUST), FINDING_EXCLUDED
        )

    def test_a_transition_excludes_both_gaseous_mechanisms(self) -> None:
        """Stops a constant loss being offered as the reason a band just changed.

        Oxygen concentration is constant and water vapour varies over hours to
        seasons. Neither can produce a transition in minutes, so neither can
        explain the thing ``propagation.py`` detects.
        """
        signature = WeatherSignature(
            (BandExcess(22.2e9, 9.0), BandExcess(60e9, 30.0)), onset_seconds=600.0
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(
            finding_for(assessment, MECHANISM_WATER_VAPOUR), FINDING_EXCLUDED
        )
        self.assertEqual(finding_for(assessment, MECHANISM_OXYGEN), FINDING_EXCLUDED)

    def test_a_standing_loss_on_a_resonance_is_consistent_with_that_gas(self) -> None:
        """The case where a gaseous mechanism is the right answer.

        A band sitting on the 22 GHz vapour line or inside the 60 GHz oxygen
        complex carries a loss that will not clear. Saying so is useful to a
        node choosing bands, and it is a different claim from explaining a fade.
        """
        vapour = self.discriminator.assess(
            WeatherSignature((BandExcess(22.2e9, 6.0),), onset_seconds=None)
        )
        self.assertEqual(
            finding_for(vapour, MECHANISM_WATER_VAPOUR), FINDING_CONSISTENT
        )

        oxygen = self.discriminator.assess(
            WeatherSignature((BandExcess(60e9, 30.0),), onset_seconds=None)
        )
        self.assertEqual(finding_for(oxygen, MECHANISM_OXYGEN), FINDING_CONSISTENT)
        self.assertEqual(
            finding_for(oxygen, MECHANISM_WATER_VAPOUR), FINDING_EXCLUDED
        )

    def test_a_variance_rise_with_a_steady_mean_is_consistent_with_scintillation(
        self,
    ) -> None:
        """The one mechanism that is a variance signature rather than a mean shift.

        Fitting scintillation's depth would be fitting noise. What grows with
        frequency is sigma, as the ITU-R P.618 seven-twelfths power, and the
        test has to be run on the standard deviations or it tests nothing.
        """
        frequencies = (4e9, 8e9, 16e9, 20e9)
        bands = tuple(
            BandExcess(
                frequency_hz=frequency,
                excess_db=0.3,
                fluctuation_std_db=0.5
                * (frequency / 4e9) ** SCINTILLATION_SIGMA_EXPONENT,
            )
            for frequency in frequencies
        )
        signature = WeatherSignature(
            bands, onset_seconds=6.0, baseline_fluctuation_std_db=0.15
        )
        assessment = self.discriminator.assess(
            signature,
            DeclaredConditions(
                path_elevation_deg=25.0, effective_antenna_diameter_m=0.3
            ),
        )
        self.assertEqual(assessment.sense, SENSE_NONE)
        self.assertEqual(
            finding_for(assessment, MECHANISM_SCINTILLATION), FINDING_CONSISTENT
        )
        self.assertAlmostEqual(
            assessment.scintillation_fit.exponent,
            SCINTILLATION_SIGMA_EXPONENT,
            places=9,
        )

    def test_a_sigma_growing_on_the_wrong_power_excludes_scintillation(self) -> None:
        """Keeps the seven-twelfths law from being decorative.

        A fluctuation that rises far faster with frequency than ITU-R P.618
        predicts is some other frequency-selective process, and accepting it
        would make the scintillation finding mean only "something fluctuated".
        """
        frequencies = (4e9, 8e9, 16e9, 20e9)
        bands = tuple(
            BandExcess(
                frequency_hz=frequency,
                excess_db=0.3,
                fluctuation_std_db=0.5 * (frequency / 4e9) ** 2.0,
            )
            for frequency in frequencies
        )
        signature = WeatherSignature(
            bands, onset_seconds=6.0, baseline_fluctuation_std_db=0.15
        )
        assessment = self.discriminator.assess(signature)
        self.assertEqual(
            finding_for(assessment, MECHANISM_SCINTILLATION), FINDING_EXCLUDED
        )

    def test_a_mean_shift_excludes_scintillation_however_large_the_variance(
        self,
    ) -> None:
        """Holds the definition of the mechanism against a tempting near-miss.

        Scintillation is a fluctuation about a mean. A signature dominated by a
        shift of the mean is something else, with scintillation possibly on top
        of it, and the module must not absorb the shift into the fluctuation.
        """
        bands = tuple(
            BandExcess(frequency, 14.0, fluctuation_std_db=2.0)
            for frequency in (5e9, 10e9, 18e9)
        )
        assessment = self.discriminator.assess(
            WeatherSignature(bands, onset_seconds=6.0, baseline_fluctuation_std_db=0.2)
        )
        self.assertEqual(
            finding_for(assessment, MECHANISM_SCINTILLATION), FINDING_EXCLUDED
        )

    def test_scintillation_declines_outside_the_range_itu_r_p618_is_offered_for(
        self,
    ) -> None:
        """Prevents extrapolating a prediction method past its published range."""
        bands = tuple(
            BandExcess(frequency, 0.3, fluctuation_std_db=0.8)
            for frequency in (1e9, 2e9, 3e9)
        )
        assessment = self.discriminator.assess(
            WeatherSignature(bands, baseline_fluctuation_std_db=0.2)
        )
        self.assertEqual(
            finding_for(assessment, MECHANISM_SCINTILLATION), FINDING_INDETERMINATE
        )

    def test_a_low_elevation_path_declines_rather_than_extrapolating(self) -> None:
        """Refuses hardest exactly where the mechanism is worst, and says so.

        ITU-R P.618 states its scintillation method for elevations above five
        degrees, which is also where scintillation is most severe. Declining
        with that stated is more useful than a number the recommendation does
        not support.
        """
        bands = tuple(
            BandExcess(frequency, 0.3, fluctuation_std_db=1.2)
            for frequency in (5e9, 10e9, 18e9)
        )
        assessment = self.discriminator.assess(
            WeatherSignature(bands, baseline_fluctuation_std_db=0.2),
            DeclaredConditions(path_elevation_deg=2.0),
        )
        self.assertEqual(
            finding_for(assessment, MECHANISM_SCINTILLATION), FINDING_INDETERMINATE
        )

    def test_every_mechanism_is_reported_on_every_assessment(self) -> None:
        """Stops a mechanism disappearing when its test happens not to run.

        Omitting an untested mechanism would let a caller read a shorter list as
        a narrower field, when in fact nothing was learned about the ones that
        went missing.
        """
        assessment = self.discriminator.assess(WeatherSignature(()))
        self.assertEqual(
            tuple(item.mechanism for item in assessment.findings), MECHANISMS
        )
        self.assertIn(
            "SIGNATURE_CARRIED_NO_BANDS_SO_NOTHING_COULD_BE_TESTED",
            assessment.reasons,
        )

    def test_a_shortlist_of_one_is_not_a_resolution_while_anything_is_untested(
        self,
    ) -> None:
        """Keeps the strongest flag in the module honest about what it means.

        A fade almost never clears this bar, because wet foliage has no test and
        therefore never leaves ``indeterminate``. That is not a defect: it is
        the module declining to report "only rain" when "rain, or something with
        no published frequency law" is what the evidence supports.
        """
        assessment = self.discriminator.assess(
            WeatherSignature(
                rain_bands((10e9, 20e9, 40e9), 100.0, 5.0), onset_seconds=900.0
            )
        )
        self.assertEqual(assessment.consistent, (MECHANISM_RAIN,))
        self.assertIn(MECHANISM_WET_FOLIAGE, assessment.indeterminate)
        self.assertFalse(assessment.resolved)


class ValidityRangeTests(unittest.TestCase):
    """Every model states its range, and refuses outside it rather than extrapolating."""

    def test_rain_attenuation_refuses_above_the_published_frequency_range(self) -> None:
        """Prevents Gaussian curve fits being evaluated where nothing supported them.

        ITU-R P.838-3 defines its coefficients from 1 to 1000 GHz. Beyond that
        the fits still produce a number, and that number has no scattering
        calculation behind it.
        """
        estimate = rain_specific_attenuation(2e12, 25.0)
        self.assertFalse(estimate.computed)
        self.assertIsNone(estimate.specific_attenuation_db_per_km)
        self.assertIn("VALIDITY_RANGE", estimate.reasons[0])

    def test_rain_attenuation_refuses_a_rate_rain_does_not_come_at(self) -> None:
        """Keeps an implausible input from producing a confident output."""
        for rate in (0.01, 500.0):
            with self.subTest(rate=rate):
                estimate = rain_specific_attenuation(20e9, rate)
                self.assertFalse(estimate.computed)
                self.assertIsNone(estimate.path_attenuation_db)

    def test_cloud_attenuation_refuses_above_the_rayleigh_limit(self) -> None:
        """ITU-R P.840-8 states the Rayleigh approximation holds to 200 GHz.

        Above it the droplets are no longer small against a wavelength and the
        whole basis of the model has gone, however well the arithmetic behaves.
        """
        estimate = cloud_specific_attenuation(300e9, 0.5)
        self.assertFalse(estimate.computed)
        self.assertIn("200_GHZ", estimate.reasons[0])

    def test_cloud_attenuation_refuses_a_water_density_no_cloud_holds(self) -> None:
        """Prevents a mistyped input producing a plausible-looking fade."""
        estimate = cloud_specific_attenuation(20e9, 40.0)
        self.assertFalse(estimate.computed)
        self.assertIsNone(estimate.specific_attenuation_db_per_km)

    def test_every_estimate_states_the_range_it_is_valid_over(self) -> None:
        """An operator reading a number should be able to see where it stops holding."""
        for estimate in (
            rain_specific_attenuation(20e9, 25.0),
            rain_specific_attenuation(2e12, 25.0),
            cloud_specific_attenuation(20e9, 0.5),
            cloud_specific_attenuation(300e9, 0.5),
        ):
            with self.subTest(model=estimate.model, computed=estimate.computed):
                self.assertTrue(estimate.validity)
                self.assertIn("ITU-R", estimate.validity)

    def test_every_finding_states_the_range_its_mechanism_is_valid_over(self) -> None:
        """The same obligation, for the mechanisms with no computable figure at all."""
        assessment = WeatherDiscriminator().assess(
            WeatherSignature(
                rain_bands((10e9, 20e9, 40e9), 25.0, 5.0), onset_seconds=900.0
            )
        )
        for item in assessment.findings:
            with self.subTest(mechanism=item.mechanism):
                self.assertTrue(item.validity)
                self.assertTrue(item.reasons)

    def test_a_missing_path_length_yields_no_path_attenuation(self) -> None:
        """Prevents a per-kilometre figure being read as a path loss.

        The specific attenuation is computable without a path; the loss on the
        link is not. Returning zero, or the per-kilometre figure under the
        path's name, would be inventing a distance.
        """
        estimate = rain_specific_attenuation(20e9, 25.0)
        self.assertTrue(estimate.computed)
        self.assertIsNotNone(estimate.specific_attenuation_db_per_km)
        self.assertIsNone(estimate.path_attenuation_db)
        self.assertIn(
            "PATH_LENGTH_UNDECLARED_SO_ONLY_A_PER_KILOMETRE_FIGURE_EXISTS",
            estimate.reasons,
        )


class SerialisationTests(unittest.TestCase):
    """Nothing this module emits may be unparseable to a strict JSON consumer."""

    def test_serialised_output_is_json_and_carries_no_infinity_or_nan(self) -> None:
        """RFC 8259 has no literal for either, so one would break the whole document.

        A quantity that was not computed must serialise as ``null``, which is
        also what it means.
        """
        signatures = (
            WeatherSignature(()),
            WeatherSignature((BandExcess(10e9, -12.0),), onset_seconds=3600.0),
            WeatherSignature(
                rain_bands((10e9, 20e9, 40e9), 25.0, 5.0), onset_seconds=900.0
            ),
            WeatherSignature(
                (BandExcess(10e9, 12.0), BandExcess(20e9, -9.0)), onset_seconds=900.0
            ),
        )
        for index, signature in enumerate(signatures):
            with self.subTest(signature=index):
                payload = WeatherDiscriminator().assess(signature).to_dict()
                text = json.dumps(payload, allow_nan=False)
                self.assertNotIn("Infinity", text)
                self.assertNotIn("NaN", text)
                self.assertEqual(payload["schema"], "kval-weather-assessment/v1")

    def test_every_assessment_carries_the_standing_caveats(self) -> None:
        """Two of these are properties of the hardware and one of the vocabulary.

        Restating them on every record is cheaper than trusting every downstream
        consumer to remember that a consistent mechanism was never observed and
        that an untested one was never ruled out.
        """
        payload = WeatherDiscriminator().assess(WeatherSignature(())).to_dict()
        self.assertIn(
            "A_MECHANISM_CONSISTENT_WITH_A_SIGNATURE_IS_NOT_A_MECHANISM_OBSERVED",
            payload["reasons"],
        )
        self.assertIn(
            "INDETERMINATE_MEANS_THE_TEST_COULD_NOT_RUN_IT_DOES_NOT_MEAN_RULED_OUT",
            payload["reasons"],
        )

    def test_a_refused_estimate_serialises_its_figures_as_null_not_zero(self) -> None:
        """A zero would read as "computed, and negligible", which is the wrong story."""
        payload = rain_specific_attenuation(2e12, 25.0).to_dict()
        json.dumps(payload, allow_nan=False)
        self.assertFalse(payload["computed"])
        self.assertIsNone(payload["specific_attenuation_db_per_km"])
        self.assertIsNone(payload["path_attenuation_db"])


class CallerErrorTests(unittest.TestCase):
    """Wiring mistakes raise. Weak evidence never does."""

    def test_a_coefficient_outside_the_published_range_raises(self) -> None:
        """A caller asking where the recommendation defines nothing has a bug."""
        with self.assertRaises(WeatherModelError):
            rain_coefficients(0.5e9)
        with self.assertRaises(WeatherModelError):
            rain_coefficients(2e12)

    def test_a_negative_rain_rate_raises(self) -> None:
        """No refusal reason would make sense of this, so it is a wiring error."""
        with self.assertRaises(WeatherModelError):
            rain_specific_attenuation(20e9, -5.0)

    def test_two_bands_at_one_frequency_raise(self) -> None:
        """Two excesses at one frequency disagree about a fact only one can hold.

        Averaging them would invent a third measurement that neither instrument
        reported.
        """
        with self.assertRaises(WeatherModelError):
            WeatherSignature((BandExcess(10e9, 5.0), BandExcess(10e9, 9.0)))

    def test_an_exponent_from_a_single_frequency_raises(self) -> None:
        """One frequency defines no slope, and no default would be honest."""
        with self.assertRaises(WeatherModelError):
            rain_exponent((20e9,), 25.0)
        with self.assertRaises(WeatherModelError):
            cloud_exponent((20e9,))

    def test_mismatched_weights_raise(self) -> None:
        """Silently recycling or truncating weights would misweight the model."""
        with self.assertRaises(WeatherModelError):
            rain_exponent((10e9, 20e9), 25.0, weights=(1.0,))

    def test_weak_evidence_never_raises(self) -> None:
        """The central rule: a refusal is a result, and an exception is not one.

        An operator sees results. Every one of these inputs is unusable, and
        every one must come back as an assessment carrying its reasons.
        """
        for signature in (
            WeatherSignature(()),
            WeatherSignature((BandExcess(10e9, 0.1),)),
            WeatherSignature(
                (BandExcess(20.0e9, 9.0), BandExcess(20.2e9, 9.4)), onset_seconds=1.0
            ),
            WeatherSignature(
                (BandExcess(10e9, 12.0), BandExcess(20e9, -9.0)),
                onset_seconds=100_000.0,
            ),
        ):
            with self.subTest(bands=len(signature.bands)):
                assessment = WeatherDiscriminator().assess(signature)
                self.assertTrue(assessment.reasons)
                self.assertEqual(len(assessment.findings), len(MECHANISMS))


class BoundaryTests(unittest.TestCase):
    """The bounds are engineering envelopes, so their edges are exercised."""

    def test_the_rain_window_reaches_both_ends_of_the_plausible_rain_rate_range(
        self,
    ) -> None:
        """Guards the envelope against silently narrowing.

        Rain is excluded only when the published model cannot reach the
        measurement at *any* rate rain comes at. If the window stopped spanning
        the declared range, rain would start being excluded on rates that are
        inside it.
        """
        frequencies = (10e9, 40e9)
        drizzle = rain_exponent(frequencies, MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H)
        downpour = rain_exponent(frequencies, MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H)
        self.assertGreater(drizzle, downpour)
        for rate in (1.0, 10.0, 100.0):
            with self.subTest(rate=rate):
                middle = rain_exponent(frequencies, rate)
                self.assertLessEqual(downpour, middle)
                self.assertLessEqual(middle, drizzle)

    def test_the_cloud_window_is_far_narrower_than_the_rain_window(self) -> None:
        """The quantitative basis of the module's central discrimination claim.

        Cloud's exponent is pinned near two across every temperature liquid
        water exists at, while rain's runs across more than a unit. That
        asymmetry is why an exponent well away from two rules cloud out, and it
        should break the build if the models ever stopped supporting it.
        """
        frequencies = (10e9, 20e9)
        cloud_span = abs(
            cloud_exponent(frequencies, temperature_k=253.15)
            - cloud_exponent(frequencies, temperature_k=313.15)
        )
        rain_span = abs(
            rain_exponent(frequencies, MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H)
            - rain_exponent(frequencies, MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H)
        )
        self.assertLess(cloud_span, 0.25)
        self.assertGreater(rain_span, 1.5)
        self.assertGreater(rain_span, 5.0 * cloud_span)

    def test_a_default_cloud_temperature_sits_inside_the_supported_range(self) -> None:
        """Keeps the default from being one the model would refuse."""
        self.assertTrue(
            math.isfinite(
                cloud_attenuation_coefficient(
                    20e9, temperature_k=DEFAULT_CLOUD_TEMPERATURE_K
                )
            )
        )

    def test_the_sense_of_a_signature_matches_which_way_its_bands_moved(self) -> None:
        """A one-line property that several exclusions are built on top of."""
        self.assertEqual(
            WeatherSignature((BandExcess(10e9, 9.0),)).sense, SENSE_ATTENUATION
        )
        self.assertEqual(
            WeatherSignature((BandExcess(10e9, -9.0),)).sense, SENSE_ENHANCEMENT
        )
        self.assertEqual(
            WeatherSignature((BandExcess(10e9, 0.4),)).sense, SENSE_NONE
        )
        self.assertEqual(
            WeatherSignature(
                (BandExcess(10e9, 9.0), BandExcess(20e9, -9.0))
            ).sense,
            SENSE_MIXED,
        )


if __name__ == "__main__":
    unittest.main()
