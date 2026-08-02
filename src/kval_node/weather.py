"""Which atmospheric mechanism a fade's shape is consistent with, and which it is not.

:mod:`kval_node.propagation` can already separate *attenuation on the path* from
*energy arriving in the band*, and inside attenuation it can recognise one
weather mechanism: rain. That is one mechanism, and this node deploys in
tropical monsoon, temperate winter, arid and maritime conditions in the same
year. A module that can only name rain will name rain.

**The discriminating principle.** Every mechanism here has a frequency law and a
time constant, and those two are enough to separate most of them without any
meteorological instrument at all. A node observing three bands measures how a
fade deepens with frequency; that slope is a property of the mechanism, not of
the path length, because path length is a constant offset in the logarithm and
a constant offset does not change a slope. So the node can ask "what shape is
this?" without knowing how far away the emitter is.

**Why the exponent and not decibels per decade.** The sibling module fits fade
depth in dB against ``log10 f``, which is the right test for "is this
frequency-selective at all". It is the wrong test for "which mechanism", because
its answer scales with the depth of the fade, and the depth depends on the path
length and the rain rate — neither of which this node knows. Fitting
``log10(fade_dB)`` against ``log10 f`` instead yields the **spectral exponent**
``n`` in ``fade ∝ f**n``, which is scale-free: doubling the path length moves
every point by the same constant and leaves ``n`` untouched. Everything in this
module keys off ``n``.

**What the textbook summary gets wrong, and this module does not.** The usual
shorthand is that rain goes as ``f`` and cloud as ``f**2``, so the two separate
cleanly. Computed from the published models rather than from the shorthand, that
is only true at the edges. ITU-R P.838-3 makes the rain exponent between two
bands an exactly linear function of ``log10 R``:

    ``n = [log10(k2/k1) + (alpha2 - alpha1) * log10 R] / log10(f2/f1)``

so between 10 and 20 GHz the rain exponent runs from about 3.4 at drizzle to
about 1.4 at a tropical downpour, and passes through 1.99 at roughly 25 mm/h.
The ITU-R P.840 cloud exponent over the same pair sits between 1.82 and 1.99
across the whole plausible liquid-water temperature range. **They overlap.** A
25 mm/h rain and a fog bank produce the same slope between those two bands, and
a module that reported "cloud" there would be guessing. This one reports both,
says they are not separable at that span, and names what would separate them.

That linearity cuts the other way too, and it is the strongest single result
here: the relation inverts in closed form, so a measured exponent across two
bands yields **the rain rate that would have produced it**, with no knowledge of
the path length. If that rate is not a rate rain comes at, rain is excluded —
not by assertion, but by the published model refusing to reach the measurement.

**Three findings, not two.** Each mechanism comes back ``consistent``,
``excluded``, or ``indeterminate``. The third is the one that matters: wet
foliage has no published frequency law, and sand and dust has no coefficient
table this module could verify, so neither can be confirmed *or* ruled out. A
binary verdict would have to lie about one of those, and the lie an operator
cannot see is that "not consistent" quietly became "ruled out".

**Enhancement, not only loss.** Ducting raises received power for a node inside
the duct and lowers it for one outside, and it is **refractive rather than
absorptive** — it redirects energy instead of absorbing it. A module that could
only describe loss could not describe a duct at all, so the signature carries a
signed excess and the absorptive mechanisms are excluded outright when the sign
says the band got better.

**What it will not do.** It will not measure weather. There is no barometer, no
rain gauge, no hygrometer and no met feed on this node, so every meteorological
quantity here is *declared* by an operator or is absent, and absent means the
tests needing it return ``indeterminate`` rather than a default. It will not
extrapolate a model past its published validity range. And it will not carry a
coefficient nobody can check: every number below is either transcribed from a
named ITU-R recommendation, computed from one, or labelled a bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .validation import bounded_number, require_finite

# ---------------------------------------------------------------------------
# ITU-R P.838-3 — specific attenuation for rain
# ---------------------------------------------------------------------------
#
# ``gamma_R = k * R**alpha`` in dB/km for a rain rate ``R`` in mm/h, with ``k``
# and ``alpha`` obtained from curve fits to scattering calculations:
#
#     log10 k = sum_j a_j * exp(-((log10 f - b_j)/c_j)**2) + m_k * log10 f + c_k
#     alpha   = sum_j a_j * exp(-((log10 f - b_j)/c_j)**2) + m_a * log10 f + c_a
#
# with ``f`` in GHz. The four tables below are Tables 1 to 4 of the
# recommendation, transcribed and then verified two ways: they reproduce every
# entry of the recommendation's own Table 5 to five significant figures across
# 1-40 GHz (for example 10 GHz gives k_H = 0.012167 against a tabulated 0.01217,
# alpha_H = 1.25710 against 1.2571), and the resulting specific attenuation
# matches the figures quoted in :mod:`kval_node.propagation` (0.70 dB/km for
# 25 mm/h at 10 GHz, 0.0026 dB/km for the same rain at 2 GHz).

#: ITU-R P.838-3 Table 1 — constants for ``k_H``, horizontal polarisation.
#: Dimensionless; ``f`` enters in GHz.
P838_K_HORIZONTAL = (
    (-5.33980, -0.10008, 1.13098),
    (-0.35351, 1.26970, 0.45400),
    (-0.23789, 0.86036, 0.15354),
    (-0.94158, 0.64552, 0.16817),
)
#: ITU-R P.838-3 Table 1 — ``m_k`` and ``c_k`` for ``k_H``.
P838_K_HORIZONTAL_LINEAR = (-0.18961, 0.71147)

#: ITU-R P.838-3 Table 2 — constants for ``k_V``, vertical polarisation.
P838_K_VERTICAL = (
    (-3.80595, 0.56934, 0.81061),
    (-3.44965, -0.22911, 0.51059),
    (-0.39902, 0.73042, 0.11899),
    (0.50167, 1.07319, 0.27195),
)
#: ITU-R P.838-3 Table 2 — ``m_k`` and ``c_k`` for ``k_V``.
P838_K_VERTICAL_LINEAR = (-0.16398, 0.63297)

#: ITU-R P.838-3 Table 3 — constants for ``alpha_H``, horizontal polarisation.
P838_ALPHA_HORIZONTAL = (
    (-0.14318, 1.82442, -0.55187),
    (0.29591, 0.77564, 0.19822),
    (0.32177, 0.63773, 0.13164),
    (-5.37610, -0.96230, 1.47828),
    (16.1721, -3.29980, 3.43990),
)
#: ITU-R P.838-3 Table 3 — ``m_alpha`` and ``c_alpha`` for ``alpha_H``.
P838_ALPHA_HORIZONTAL_LINEAR = (0.67849, -1.95537)

#: ITU-R P.838-3 Table 4 — constants for ``alpha_V``, vertical polarisation.
P838_ALPHA_VERTICAL = (
    (-0.07771, 2.33840, -0.76284),
    (0.56727, 0.95545, 0.54039),
    (-0.20238, 1.14520, 0.26809),
    (-48.2991, 0.791669, 0.116226),
    (48.5833, 0.791459, 0.116479),
)
#: ITU-R P.838-3 Table 4 — ``m_alpha`` and ``c_alpha`` for ``alpha_V``.
P838_ALPHA_VERTICAL_LINEAR = (-0.053739, 0.83433)

#: Frequency range over which ITU-R P.838-3 states its coefficients are defined:
#: "in the range from 1 to 1 000 GHz". Outside it the Gaussian fits are being
#: evaluated where no scattering calculation supported them, and this module
#: refuses rather than returning the number the arithmetic happens to produce.
P838_MIN_FREQUENCY_HZ = 1.0e9
P838_MAX_FREQUENCY_HZ = 1.0e12

# ---------------------------------------------------------------------------
# ITU-R P.840-8 — specific attenuation for cloud and fog liquid water
# ---------------------------------------------------------------------------
#
# ``gamma_c = K_l(f, T) * M`` in dB/km for a liquid water density ``M`` in
# g/m**3, with ``K_l`` from a Rayleigh-scattering model over a double-Debye
# permittivity for water. Annex 1 equations (2) to (11), transcribed from the
# recommendation itself.

#: ITU-R P.840-8 eq. (6) — static permittivity, ``eps_0 = 77.66 + 103.3(θ - 1)``.
P840_EPSILON_STATIC = (77.66, 103.3)
#: ITU-R P.840-8 eq. (7) — first intermediate permittivity, ``eps_1 = 0.0671 eps_0``.
P840_EPSILON_FIRST_RATIO = 0.0671
#: ITU-R P.840-8 eq. (8) — high-frequency permittivity, dimensionless.
P840_EPSILON_HIGH = 3.52
#: ITU-R P.840-8 eq. (10) — principal relaxation frequency in GHz,
#: ``f_p = 20.20 - 146(θ - 1) + 316(θ - 1)**2``.
P840_PRINCIPAL_RELAXATION_GHZ = (20.20, -146.0, 316.0)
#: ITU-R P.840-8 eq. (11) — secondary relaxation frequency, ``f_s = 39.8 f_p``.
P840_SECONDARY_RELAXATION_RATIO = 39.8
#: ITU-R P.840-8 eq. (2) — the leading factor in
#: ``K_l = 0.819 f / (eps'' (1 + eta**2))``, units ((dB/km)/(g/m**3)) per GHz.
P840_ATTENUATION_FACTOR = 0.819
#: ITU-R P.840-8 eq. (9) — the reference temperature in ``θ = 300 / T``, in K.
P840_REFERENCE_TEMPERATURE_K = 300.0

#: Upper frequency at which ITU-R P.840-8 states the Rayleigh approximation
#: holds: "for clouds or fog consisting entirely of small droplets, generally
#: less than 0.01 cm, the Rayleigh approximation is valid for frequencies up to
#: 200 GHz". Above it the droplets are no longer small against a wavelength and
#: the model does not apply.
P840_MAX_FREQUENCY_HZ = 2.0e11
#: Lower frequency below which ITU-R P.840 is not offered as a design tool: the
#: recommendation's own scope is stated for "frequencies higher than 10 GHz",
#: and the model remains mathematically defined below that. The bound is
#: retained as a caveat rather than a refusal because the f**2 law it predicts
#: is exactly what makes a low band useful as the *anchor* of a slope.
P840_STATED_SCOPE_MIN_FREQUENCY_HZ = 1.0e10

# ---------------------------------------------------------------------------
# ITU-R P.676-13 — gaseous absorption line centres
# ---------------------------------------------------------------------------
#
# Only line *centres* are carried. Evaluating the line-by-line model needs the
# full spectroscopic tables plus the pressure, temperature and vapour density at
# the path, and this node measures none of those. What the centres support is
# the question actually worth asking: is this band sitting on an absorption
# feature, in which case its loss is a standing property rather than an event?

#: ITU-R P.676-13 Table 2, first entry — the principal water vapour resonance.
WATER_VAPOUR_PRINCIPAL_LINE_HZ = 22.235_080e9
#: ITU-R P.676-13 Table 2 — the next strong water vapour line.
WATER_VAPOUR_SECOND_LINE_HZ = 183.310_087e9
#: ITU-R P.676-13 Table 1, first entry — the lowest line of the oxygen complex.
OXYGEN_COMPLEX_LOW_HZ = 50.474_214e9
#: ITU-R P.676-13 Table 1 — the highest line of the complex that merges, at
#: sea-level pressure, into the single broad absorption band near 60 GHz.
OXYGEN_COMPLEX_HIGH_HZ = 68.960_312e9
#: ITU-R P.676-13 Table 1 — the isolated oxygen line above the 60 GHz complex.
OXYGEN_ISOLATED_LINE_HZ = 118.750_334e9

#: Half-width used to decide whether a band sits *on* the 22 GHz water vapour
#: line. Computed rather than chosen: ITU-R P.676-13 gives the line width as
#: ``b3 * 1e-4 * (p * θ**b4 + b5 * e * θ**b6)``, and Table 2 gives the 22.235 GHz
#: line ``b3 = 26.38``, ``b5 = 5.087``, ``b6 = 1.00``. At sea level (1013 hPa)
#: with a 10 hPa partial vapour pressure that is ``26.38e-4 * (1013 + 50.9)``,
#: about 2.8 GHz. Three half-widths — where a Lorentzian has fallen to a tenth
#: of its peak — gives the window below.
WATER_VAPOUR_LINE_HALF_WIDTH_HZ = 2.8e9
#: Multiple of the half-width taken as the line's zone of influence.
LINE_INFLUENCE_HALF_WIDTHS = 3.0

# ---------------------------------------------------------------------------
# ITU-R P.618-14 — tropospheric scintillation
# ---------------------------------------------------------------------------

#: Exponent by which the scintillation standard deviation grows with frequency
#: in ITU-R P.618: ``sigma = sigma_ref * f**(7/12) * g(x) / sin(θ)**1.2``. The
#: exponent is carried; ``sigma_ref`` is not, because it is
#: ``3.6e-3 + 1e-4 * N_wet`` and ``N_wet`` is a wet-refractivity figure this node
#: has no instrument to obtain. The exponent alone is what discriminates, and it
#: needs nothing declared.
SCINTILLATION_SIGMA_EXPONENT = 7.0 / 12.0

#: ITU-R P.618 aperture-averaging factor, ``x = 1.22 * D_eff**2 * f / L``, where
#: ``g(x) = sqrt(3.86 (x**2+1)**(11/12) sin(11/6 arctan(1/x)) - 7.08 x**(5/6))``.
#: ``g`` is frequency-dependent through ``x``, so a large aperture biases the
#: measured exponent away from 7/12. Below this value of ``x`` the bias is not
#: material: ``g(0.03)/g(0) = 0.907``, and across a full decade of frequency the
#: resulting exponent bias is under 0.05, an order below the tolerance any
#: mechanism test here uses. A bound, derived from the published ``g``, not a
#: tabulated threshold.
SCINTILLATION_NEGLIGIBLE_APERTURE_X = 0.03
#: ITU-R P.618 effective path length constant, the turbulent layer height in
#: metres, used in ``L = 2 h_L / (sqrt(sin(θ)**2 + 2.35e-4) + sin(θ))``.
SCINTILLATION_TURBULENT_LAYER_HEIGHT_M = 1000.0
#: The ``2.35e-4`` in that expression, which is ``2 h_L / R_eff`` for an
#: effective Earth radius of 8 500 km.
SCINTILLATION_EARTH_CURVATURE_TERM = 2.35e-4

#: Frequency range over which the ITU-R P.618 scintillation method is offered.
#: The upper limit is the recommendation's own: the prediction method is stated
#: for frequencies up to 20 GHz. The lower limit is a bound rather than a
#: tabulated value — below a few GHz ionospheric rather than tropospheric
#: scintillation dominates, and this module will not attribute a tropospheric
#: mechanism where a different layer of the atmosphere is the likelier cause.
P618_SCINTILLATION_MIN_FREQUENCY_HZ = 4.0e9
P618_SCINTILLATION_MAX_FREQUENCY_HZ = 2.0e10
#: Path elevation below which the ITU-R P.618 scintillation method is not
#: offered: the recommendation states it for elevation angles above 5 degrees.
P618_SCINTILLATION_MIN_ELEVATION_DEG = 5.0

# ---------------------------------------------------------------------------
# ITU-R P.453 — refractivity, and the gradient at which trapping begins
# ---------------------------------------------------------------------------

#: Refractivity gradient, in N-units per km, at or below which a ray curves at
#: least as fast as the Earth and is trapped. Derived, not tabulated: modified
#: refractivity is ``M = N + 1e6 * h / R_e``, so ``dM/dh`` reaches zero when
#: ``dN/dh = -1e6 / R_e``, which for a 6 371 km Earth is -156.97 N-units/km. The
#: conventional round figure is -157, and it is the standard trapping criterion
#: in the ITU-R P.453 refractivity framework.
DUCTING_TRAPPING_GRADIENT_N_PER_KM = -157.0

# ---------------------------------------------------------------------------
# Engineering bounds. These are NOT tabulated values.
# ---------------------------------------------------------------------------
#
# Everything below is an envelope chosen so that a mechanism is excluded only
# when the published model cannot reach the measurement anywhere inside it. They
# are deliberately generous: a wide envelope makes exclusion rarer and harder,
# which is the safe direction for a module whose exclusions are load-bearing.

#: Lowest rain rate treated as rain. A bound: 0.25 mm/h is the conventional
#: floor at which a tipping-bucket gauge registers at all, and below it the
#: hydrometeors are drizzle or cloud droplets, which the cloud model covers.
MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H = 0.25
#: Highest rain rate treated as rain. A bound: ITU-R P.837 rain-rate maps put
#: the 0.01% exceeded rate below about 145 mm/h even in the wettest tropical
#: zones, so 150 sits above the wettest place this node deploys.
MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H = 150.0

#: Liquid water temperature range over which the double-Debye permittivity is
#: exercised here. A bound: the lower end covers supercooled cloud liquid water,
#: which persists to roughly -20 C, and the upper end covers warm maritime fog.
#: Widening it further only widens the cloud exponent window, which makes cloud
#: harder to exclude, not easier to claim.
MIN_CLOUD_TEMPERATURE_K = 253.15
MAX_CLOUD_TEMPERATURE_K = 313.15
#: Default liquid water temperature when none is declared: 0 C, the middle of
#: the range above and the temperature at which most attenuating cloud sits.
DEFAULT_CLOUD_TEMPERATURE_K = 273.15

#: Highest cloud liquid water density treated as cloud. A bound: ITU-R P.840-8
#: quotes 0.05 g/m**3 for medium fog and 0.5 g/m**3 for thick fog, and vigorous
#: convective cloud reaches a few g/m**3. 5 sits above anything a path crosses.
MAX_PLAUSIBLE_LIQUID_WATER_G_PER_M3 = 5.0

#: Default polarisation tilt, in degrees. 45 degrees is circular polarisation,
#: for which ``cos(2 tau)`` vanishes and ITU-R P.838-3 eq. (4) and (5) collapse
#: to the mean of the horizontal and vertical coefficients. That is the
#: least-committed choice when the link's polarisation has not been declared,
#: and it is stated in every result that used it.
DEFAULT_POLARISATION_TILT_DEG = 45.0

#: Excess, in dB, below which a band is treated as not having moved. Taken from
#: :mod:`kval_node.propagation`, which justifies 3 dB as the smallest change
#: separable from a low-cost SDR front end whose absolute calibration is good to
#: about a decibel. It bounds the *instrument*, not any weather mechanism.
MIN_MEASURABLE_EXCESS_DB = 3.0

#: Uncertainty, in dB, assumed on each band's measured excess when propagating
#: it into the spectral exponent. One decibel, again from the sibling module's
#: statement of what a low-cost front end's absolute calibration is worth.
DEFAULT_EXCESS_UNCERTAINTY_DB = 1.0

#: Fractional uncertainty assumed on a measured fluctuation standard deviation.
#: Derived, not chosen: the sampling error of ``sigma`` estimated from ``N``
#: readings is about ``sigma / sqrt(2(N - 1))``, so a window of nine readings
#: gives a quarter. Expressed as a fraction because that is the shape of the
#: error — a standard deviation is uncertain in proportion to itself, which
#: makes its logarithm uncertain by a constant.
DEFAULT_SIGMA_FRACTIONAL_UNCERTAINTY = 0.25

#: Largest standard error on the spectral exponent that still permits a verdict.
#: Derived: the mechanisms this module most needs to tell apart sit about one
#: unit apart in exponent in the cases where they separate at all, so a two-sigma
#: interval of +/-0.5 is the widest that leaves them distinguishable. Above this
#: the exponent is reported and every exponent-dependent test declines.
MAX_EXPONENT_STANDARD_ERROR = 0.25

#: Multiples of the exponent's standard error taken as its interval when testing
#: against a model window. Two sigma, so a mechanism is excluded only when the
#: measurement misses its window by more than a 95% interval.
EXPONENT_CONFIDENCE_SIGMA = 2.0

#: Frequency span, in decades, below which no exponent is fitted. A bound: below
#: about a third of an octave two bands share antenna, front end and calibration
#: to the point that their difference is instrument rather than atmosphere, and
#: dividing that difference by a small log span manufactures a large exponent.
MIN_FREQUENCY_SPAN_DECADES = 0.1

#: Bands required before a frequency law can be fitted at all.
MIN_BANDS_FOR_EXPONENT = 2

#: Onset bounds, in seconds, for each mechanism's time constant. Every one is a
#: bound derived from the physical scale of the phenomenon, not a tabulated
#: figure, and each is stated in the reason that uses it.
#:
#: Rain: cells are kilometres across and rates change over minutes to hours, so
#: a fade completing inside two minutes is a change of geometry, not of sky —
#: the same two minutes :mod:`kval_node.propagation` already uses. The upper
#: bound is the passage of a convective cell or a frontal rain band; longer than
#: six hours is a diurnal process, not a cell.
RAIN_MIN_ONSET_SECONDS = 120.0
RAIN_MAX_ONSET_SECONDS = 21_600.0
#: Cloud and fog: fog forms and burns off over hours and is often dawn-peaked.
#: Ten minutes is the fastest a fog bank thickens across a path; half a day is
#: the longest a single episode lasts before it is a change of air mass.
CLOUD_MIN_ONSET_SECONDS = 600.0
CLOUD_MAX_ONSET_SECONDS = 43_200.0
#: Water vapour and oxygen: quasi-static. Anything that arrived faster than six
#: hours is an event, and neither of these is an event.
QUASI_STATIC_MIN_ONSET_SECONDS = 21_600.0

#: Ratio by which the in-band fluctuation standard deviation must exceed the
#: baseline's before the variance is treated as having risen. A bound: a factor
#: of two in sigma is a factor of four in variance, which no ordinary drift in a
#: quiet window reaches.
SCINTILLATION_MIN_VARIANCE_RATIO = 2.0

#: Largest mean excess, in dB, still consistent with scintillation. A bound, and
#: the definition of the mechanism rather than an arbitrary number:
#: scintillation is a fluctuation about a mean, so a signature dominated by a
#: mean shift is something else with scintillation on top of it. Set equal to
#: the measurable-excess floor so that "the mean did not measurably move" and
#: "consistent with scintillation" are the same statement.
SCINTILLATION_MAX_MEAN_EXCESS_DB = MIN_MEASURABLE_EXCESS_DB

#: Frequency below which sand and dust attenuation is negligible. A bound, and
#: a qualitative one: dust particles are tens of microns, so they are deep in
#: the Rayleigh regime through the microwave bands and their absorption falls
#: away steeply towards lower frequency. No coefficient table for it was
#: verifiable here, so this bound gates a caveat and never an attenuation
#: figure.
SAND_AND_DUST_MIN_SIGNIFICANT_FREQUENCY_HZ = 1.0e10

#: Largest spectral exponent magnitude still consistent with a refractive
#: mechanism. A bound: refraction redirects energy rather than absorbing it, so
#: over a microwave band its frequency dependence is weak and not a power law,
#: whereas every absorptive mechanism here has an exponent of order one or more.
DUCTING_MAX_EXPONENT_MAGNITUDE = 0.5

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Mechanisms this module can reason about.
MECHANISM_RAIN = "rain"
MECHANISM_CLOUD_LIQUID_WATER = "cloud_liquid_water"
MECHANISM_MELTING_LAYER = "melting_layer"
MECHANISM_DRY_SNOW = "dry_snow"
MECHANISM_WATER_VAPOUR = "water_vapour"
MECHANISM_OXYGEN = "oxygen"
MECHANISM_SCINTILLATION = "scintillation"
MECHANISM_SAND_AND_DUST = "sand_and_dust"
MECHANISM_WET_FOLIAGE = "wet_foliage"
MECHANISM_DUCTING = "ducting"

#: Order in which mechanisms are reported, so a serialised assessment is stable.
MECHANISMS = (
    MECHANISM_RAIN,
    MECHANISM_CLOUD_LIQUID_WATER,
    MECHANISM_MELTING_LAYER,
    MECHANISM_DRY_SNOW,
    MECHANISM_WATER_VAPOUR,
    MECHANISM_OXYGEN,
    MECHANISM_SCINTILLATION,
    MECHANISM_SAND_AND_DUST,
    MECHANISM_WET_FOLIAGE,
    MECHANISM_DUCTING,
)

#: What was concluded about one mechanism. ``INDETERMINATE`` is the load-bearing
#: one: it says the test could not be run, which is not the same as saying the
#: mechanism is ruled out, and collapsing the two is how a module starts lying.
FINDING_CONSISTENT = "consistent"
FINDING_EXCLUDED = "excluded"
FINDING_INDETERMINATE = "indeterminate"

#: Which way the band moved. ``mixed`` is not a mechanism — it is several.
SENSE_ATTENUATION = "attenuation"
SENSE_ENHANCEMENT = "enhancement"
SENSE_MIXED = "mixed"
SENSE_NONE = "none"

#: Appended to every serialised assessment. Two of these are properties of the
#: node's hardware and one is a property of this module's own vocabulary, and
#: all three are cheaper to state than to trust a downstream consumer to recall.
_STANDING_CAVEATS = (
    "A_MECHANISM_CONSISTENT_WITH_A_SIGNATURE_IS_NOT_A_MECHANISM_OBSERVED",
    "THIS_NODE_CARRIES_NO_METEOROLOGICAL_INSTRUMENT_EVERY_WEATHER_INPUT_IS_DECLARED",
    "INDETERMINATE_MEANS_THE_TEST_COULD_NOT_RUN_IT_DOES_NOT_MEAN_RULED_OUT",
)


class WeatherModelError(ValueError):
    """Raised when a caller wires a model up wrongly.

    Reserved for caller mistakes — a negative frequency, a rain rate below zero,
    a temperature outside anything liquid water can be. Weakness in the
    *evidence* never raises: it produces a result carrying its refusal, because
    an operator sees a result and never sees an exception.
    """


def _rounded(value: float | None, digits: int = 4) -> float | None:
    """Round for serialisation, mapping anything non-finite to ``None``.

    RFC 8259 has no literal for infinity or NaN, so emitting one makes the whole
    document unparseable to a strict consumer. A quantity that was not computed
    is ``null``, which is what it means.
    """
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _deduplicate(reasons: Sequence[str]) -> tuple[str, ...]:
    """Preserve order while removing repeats, so a reason reads once."""
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# ITU-R P.838-3: rain
# ---------------------------------------------------------------------------


def _gaussian_sum(
    table: Sequence[tuple[float, float, float]],
    linear: tuple[float, float],
    log_frequency_ghz: float,
) -> float:
    """Evaluate ITU-R P.838-3 eq. (2) or (3) at ``log10 f``."""
    total = sum(
        a * math.exp(-(((log_frequency_ghz - b) / c) ** 2)) for a, b, c in table
    )
    slope, intercept = linear
    return total + slope * log_frequency_ghz + intercept


def rain_coefficients(
    frequency_hz: float,
    *,
    polarisation_tilt_deg: float = DEFAULT_POLARISATION_TILT_DEG,
    path_elevation_deg: float = 0.0,
) -> tuple[float, float]:
    """``(k, alpha)`` for ``gamma = k * R**alpha`` at one frequency.

    Implements ITU-R P.838-3 equations (2) and (3) for the two linear
    polarisations, then combines them for the path's geometry with equations (4)
    and (5). At the default 45-degree tilt — circular polarisation —
    ``cos(2 tau)`` is zero and the result is the plain mean of the two linear
    cases, independent of elevation.

    Raises rather than refuses outside 1-1000 GHz: a caller asking for a
    coefficient where the recommendation defines none has made a wiring mistake,
    and the signature matcher checks the range before it ever gets here.
    """
    frequency = require_finite(frequency_hz, "frequency_hz")
    if not P838_MIN_FREQUENCY_HZ <= frequency <= P838_MAX_FREQUENCY_HZ:
        raise WeatherModelError(
            "ITU-R P.838-3 defines k and alpha from 1 to 1000 GHz; "
            f"{frequency / 1e9:.6g} GHz is outside that range"
        )
    tilt = bounded_number(polarisation_tilt_deg, "polarisation_tilt_deg", -90.0, 90.0)
    elevation = bounded_number(path_elevation_deg, "path_elevation_deg", -90.0, 90.0)

    log_f = math.log10(frequency / 1e9)
    k_h = 10.0 ** _gaussian_sum(P838_K_HORIZONTAL, P838_K_HORIZONTAL_LINEAR, log_f)
    k_v = 10.0 ** _gaussian_sum(P838_K_VERTICAL, P838_K_VERTICAL_LINEAR, log_f)
    alpha_h = _gaussian_sum(P838_ALPHA_HORIZONTAL, P838_ALPHA_HORIZONTAL_LINEAR, log_f)
    alpha_v = _gaussian_sum(P838_ALPHA_VERTICAL, P838_ALPHA_VERTICAL_LINEAR, log_f)

    geometry = math.cos(math.radians(elevation)) ** 2 * math.cos(
        2.0 * math.radians(tilt)
    )
    k = (k_h + k_v + (k_h - k_v) * geometry) / 2.0
    if k <= 0.0:
        # Defensive: the fits are positive throughout the published range, so a
        # non-positive k would mean the tables were edited. Dividing by it in
        # eq. (5) would return a plausible alpha from a corrupt k.
        raise WeatherModelError("ITU-R P.838-3 produced a non-positive k coefficient")
    alpha = (
        k_h * alpha_h
        + k_v * alpha_v
        + (k_h * alpha_h - k_v * alpha_v) * geometry
    ) / (2.0 * k)
    return k, alpha


def _rain_exponent_terms(
    frequencies: Sequence[float],
    weights: Sequence[float] | None,
    polarisation_tilt_deg: float,
    path_elevation_deg: float,
) -> tuple[float, float]:
    """``(intercept, sensitivity)`` in ``n = intercept + sensitivity * log10 R``.

    ITU-R P.838-3 makes ``log10 gamma = log10 k(f) + alpha(f) log10 R``, and a
    least-squares slope is a *linear* functional of the fitted values, so the
    slope of the model's own attenuation against ``log10 f`` decomposes into the
    slope of ``log10 k`` plus ``log10 R`` times the slope of ``alpha``. Both
    terms depend only on the frequencies and the weights, so the entire rain
    family across every possible rain rate is two numbers.

    That is what makes the rain test exact rather than a search, and it is what
    makes the inversion closed form.
    """
    if len(frequencies) < 2:
        raise WeatherModelError("at least two frequencies are needed for an exponent")
    if weights is None:
        weights = [1.0] * len(frequencies)
    elif len(weights) != len(frequencies):
        raise WeatherModelError("weights and frequencies must be the same length")
    xs = [math.log10(frequency) for frequency in frequencies]
    coefficients = [
        rain_coefficients(
            frequency,
            polarisation_tilt_deg=polarisation_tilt_deg,
            path_elevation_deg=path_elevation_deg,
        )
        for frequency in frequencies
    ]
    intercept, _ = weighted_slope(
        xs, [math.log10(k) for k, _ in coefficients], weights
    )
    sensitivity, _ = weighted_slope(xs, [alpha for _, alpha in coefficients], weights)
    if intercept is None or sensitivity is None:
        raise WeatherModelError(
            "the supplied frequencies carry no leverage on a frequency law"
        )
    return intercept, sensitivity


def rain_exponent(
    frequencies: Sequence[float],
    rain_rate_mm_per_h: float,
    *,
    weights: Sequence[float] | None = None,
    polarisation_tilt_deg: float = DEFAULT_POLARISATION_TILT_DEG,
    path_elevation_deg: float = 0.0,
) -> float:
    """Spectral exponent ITU-R P.838-3 predicts across a set of bands.

    ``weights`` are the measurement's own weights, so that the model is reduced
    to a single exponent exactly the way the measurement was. With two
    frequencies the weights cancel and this is the chord exponent

        ``n = [log10(k2/k1) + (alpha2 - alpha1) log10 R] / log10(f2/f1)``

    which is *linear in* ``log10 R``. That linearity is why the rain test is
    sharp rather than a guess: one measured exponent picks out one rain rate,
    and a rain rate outside anything rain comes at excludes rain.
    """
    rate = bounded_number(rain_rate_mm_per_h, "rain_rate_mm_per_h", 1e-6, 10_000.0)
    intercept, sensitivity = _rain_exponent_terms(
        frequencies, weights, polarisation_tilt_deg, path_elevation_deg
    )
    return intercept + sensitivity * math.log10(rate)


@dataclass(frozen=True)
class ImpliedRainRate:
    """The rain rate that would have produced a measured exponent, if it is rain.

    Conditional by construction. It does not say it is raining; it says what
    rain rate the published model would need in order to reproduce the shape
    that was measured, and whether that is a rate rain comes at.
    """

    rain_rate_mm_per_h: float | None
    plausible: bool
    sensitivity_per_decade: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-implied-rain-rate/v1",
            "rain_rate_mm_per_h": _rounded(self.rain_rate_mm_per_h, 3),
            "plausible": self.plausible,
            "sensitivity_per_decade": _rounded(self.sensitivity_per_decade, 4),
            "reasons": list(self.reasons),
        }


def implied_rain_rate(
    exponent: float,
    frequencies: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    polarisation_tilt_deg: float = DEFAULT_POLARISATION_TILT_DEG,
    path_elevation_deg: float = 0.0,
) -> ImpliedRainRate:
    """Invert ITU-R P.838-3 for the rain rate implied by a measured exponent.

    Conditional on the excess being rain at all, and independent of path length,
    because path length is an additive constant in the logarithm and vanishes
    from a slope. That is the unusual part: a passive receiver that knows
    neither how far away the emitter is nor how much of the path is in the cell
    can still say what rain rate the shape corresponds to.

    Refuses when ``alpha`` barely changes across the supplied bands. The
    inversion divides by that change, and near zero the division turns ordinary
    measurement scatter into an arbitrary rainfall figure. The refusal bar is
    the exponent tolerance itself: a decade of rain rate must move the model's
    exponent by more than the largest standard error a verdict tolerates, or the
    whole plausible rainfall range maps into less than the measurement's own
    uncertainty and the answer would be arithmetic rather than evidence.
    """
    if exponent is None:
        # A fit that refused has no exponent to invert, and the refusal is the
        # answer. Coercing ``None`` here would raise a TypeError from deep
        # inside a validator, which tells a caller nothing about what went
        # wrong.
        raise WeatherModelError(
            "an exponent is required; a fit that refused has none to invert"
        )
    measured = require_finite(exponent, "exponent")
    intercept, sensitivity = _rain_exponent_terms(
        frequencies, weights, polarisation_tilt_deg, path_elevation_deg
    )
    if abs(sensitivity) < MAX_EXPONENT_STANDARD_ERROR:
        return ImpliedRainRate(
            rain_rate_mm_per_h=None,
            plausible=False,
            sensitivity_per_decade=sensitivity,
            reasons=(
                "ALPHA_BARELY_CHANGES_ACROSS_THESE_BANDS_SO_THE_EXPONENT_"
                "CARRIES_NO_RAIN_RATE_INFORMATION",
            ),
        )

    log_rate = (measured - intercept) / sensitivity
    if log_rate > 12.0 or log_rate < -12.0:
        # Exponentiating this would overflow or underflow to a number that looks
        # like a rainfall figure. It is not one.
        return ImpliedRainRate(
            rain_rate_mm_per_h=None,
            plausible=False,
            sensitivity_per_decade=sensitivity,
            reasons=("IMPLIED_RAIN_RATE_IS_OUTSIDE_ANY_REPRESENTABLE_RANGE",),
        )
    rate = 10.0**log_rate
    reasons: list[str] = []
    plausible = (
        MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H <= rate <= MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H
    )
    if rate < MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H:
        reasons.append("IMPLIED_RAIN_RATE_IS_BELOW_THE_RATE_AT_WHICH_A_GAUGE_REGISTERS")
    elif rate > MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H:
        reasons.append("IMPLIED_RAIN_RATE_EXCEEDS_ANY_RATE_IN_THE_ITU_R_P837_MAPS")
    else:
        reasons.append("IMPLIED_RAIN_RATE_IS_A_RATE_RAIN_COMES_AT")
    return ImpliedRainRate(
        rain_rate_mm_per_h=rate,
        plausible=plausible,
        sensitivity_per_decade=sensitivity,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# ITU-R P.840-8: cloud and fog
# ---------------------------------------------------------------------------


def cloud_attenuation_coefficient(
    frequency_hz: float, *, temperature_k: float = DEFAULT_CLOUD_TEMPERATURE_K
) -> float:
    """``K_l`` in ((dB/km)/(g/m**3)), ITU-R P.840-8 Annex 1 eq. (2) to (11).

    The Rayleigh regime this sits in is exactly why cloud and fog rise as
    ``f**2``: for droplets small against a wavelength the absorption cross
    section scales with frequency and with the imaginary permittivity, and below
    the principal relaxation the latter is itself linear in frequency.
    """
    frequency = require_finite(frequency_hz, "frequency_hz")
    if not 0.0 < frequency <= P840_MAX_FREQUENCY_HZ:
        raise WeatherModelError(
            "ITU-R P.840-8 states the Rayleigh approximation to 200 GHz; "
            f"{frequency / 1e9:.6g} GHz is outside that range"
        )
    temperature = bounded_number(
        temperature_k, "temperature_k", MIN_CLOUD_TEMPERATURE_K, MAX_CLOUD_TEMPERATURE_K
    )
    frequency_ghz = frequency / 1e9

    theta = P840_REFERENCE_TEMPERATURE_K / temperature
    static, slope = P840_EPSILON_STATIC
    epsilon_0 = static + slope * (theta - 1.0)
    epsilon_1 = P840_EPSILON_FIRST_RATIO * epsilon_0
    epsilon_2 = P840_EPSILON_HIGH
    constant, linear, quadratic = P840_PRINCIPAL_RELAXATION_GHZ
    principal = constant + linear * (theta - 1.0) + quadratic * (theta - 1.0) ** 2
    if principal <= 0.0:
        # Guarded because the quadratic in eq. (10) turns over: a temperature
        # cold enough to drive it non-positive would make the Debye terms below
        # divide by zero and return a large, entirely fictitious coefficient.
        raise WeatherModelError(
            "ITU-R P.840-8 eq. (10) gives a non-positive relaxation frequency at "
            f"{temperature:.2f} K"
        )
    secondary = P840_SECONDARY_RELAXATION_RATIO * principal

    imaginary = frequency_ghz * (epsilon_0 - epsilon_1) / (
        principal * (1.0 + (frequency_ghz / principal) ** 2)
    ) + frequency_ghz * (epsilon_1 - epsilon_2) / (
        secondary * (1.0 + (frequency_ghz / secondary) ** 2)
    )
    real = (
        (epsilon_0 - epsilon_1) / (1.0 + (frequency_ghz / principal) ** 2)
        + (epsilon_1 - epsilon_2) / (1.0 + (frequency_ghz / secondary) ** 2)
        + epsilon_2
    )
    eta = (2.0 + real) / imaginary
    return P840_ATTENUATION_FACTOR * frequency_ghz / (imaginary * (1.0 + eta * eta))


def cloud_exponent(
    frequencies: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    temperature_k: float = DEFAULT_CLOUD_TEMPERATURE_K,
) -> float:
    """Spectral exponent ITU-R P.840-8 predicts across a set of bands.

    Independent of liquid water density, because ``gamma_c = K_l M`` is linear
    in ``M`` and a linear factor is a constant offset in the logarithm. That is
    what makes the cloud test strong: it needs no declared water content, only a
    temperature, and the exponent barely moves across the whole temperature
    range anyway — between 10 and 20 GHz it runs from 1.82 at -20 C to 1.99 at
    +40 C, against a rain window more than ten times as wide.

    As with rain, ``weights`` are the measurement's own, so that the model is
    reduced to one exponent the same way the measurement was.
    """
    if len(frequencies) < 2:
        raise WeatherModelError("at least two frequencies are needed for an exponent")
    if weights is None:
        weights = [1.0] * len(frequencies)
    elif len(weights) != len(frequencies):
        raise WeatherModelError("weights and frequencies must be the same length")
    slope, _ = weighted_slope(
        [math.log10(frequency) for frequency in frequencies],
        [
            math.log10(
                cloud_attenuation_coefficient(frequency, temperature_k=temperature_k)
            )
            for frequency in frequencies
        ],
        weights,
    )
    if slope is None:
        raise WeatherModelError(
            "the supplied frequencies carry no leverage on a frequency law"
        )
    return slope


# ---------------------------------------------------------------------------
# Computable attenuation, kept apart from signature matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttenuationEstimate:
    """A specific attenuation computed from a published model, or a refusal.

    Deliberately a separate return type from the signature findings. Matching a
    shape says "this is the kind of thing that happened"; this says "and if it
    is, here is how much loss it puts on a kilometre of path". Conflating them
    would let a computed figure be read as a measurement.
    """

    model: str
    computed: bool
    frequency_hz: float
    specific_attenuation_db_per_km: float | None
    path_attenuation_db: float | None
    validity: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-attenuation-estimate/v1",
            "model": self.model,
            "computed": self.computed,
            "frequency_hz": _rounded(self.frequency_hz, 1),
            "specific_attenuation_db_per_km": _rounded(
                self.specific_attenuation_db_per_km, 6
            ),
            "path_attenuation_db": _rounded(self.path_attenuation_db, 4),
            "validity": self.validity,
            "reasons": list(self.reasons),
        }


_RAIN_VALIDITY = "ITU-R P.838-3, 1 GHz to 1000 GHz, rain rate in mm/h"
_CLOUD_VALIDITY = (
    "ITU-R P.840-8, Rayleigh regime to 200 GHz, liquid water density in g/m**3"
)


def _refused_estimate(
    model: str, frequency_hz: float, validity: str, reason: str
) -> AttenuationEstimate:
    """An estimate that carries why no estimate was made.

    Every computed field is ``None`` rather than zero. A zero specific
    attenuation would read as "computed, and the loss is negligible", which is
    the one thing a refusal must never be mistaken for.
    """
    return AttenuationEstimate(
        model=model,
        computed=False,
        frequency_hz=frequency_hz,
        specific_attenuation_db_per_km=None,
        path_attenuation_db=None,
        validity=validity,
        reasons=(reason,),
    )


def rain_specific_attenuation(
    frequency_hz: float,
    rain_rate_mm_per_h: float,
    *,
    polarisation_tilt_deg: float = DEFAULT_POLARISATION_TILT_DEG,
    path_elevation_deg: float = 0.0,
    path_length_km: float | None = None,
) -> AttenuationEstimate:
    """``gamma_R = k R**alpha`` in dB/km, ITU-R P.838-3 eq. (1).

    Refuses outside the recommendation's 1-1000 GHz range and outside the rain
    rates rain comes at, rather than extrapolating a curve fit past the
    scattering calculations that produced it.
    """
    frequency = require_finite(frequency_hz, "frequency_hz")
    rate = require_finite(rain_rate_mm_per_h, "rain_rate_mm_per_h")
    if rate < 0.0:
        raise WeatherModelError("rain_rate_mm_per_h cannot be negative")
    if not P838_MIN_FREQUENCY_HZ <= frequency <= P838_MAX_FREQUENCY_HZ:
        return _refused_estimate(
            "itu-r-p838-3",
            frequency,
            _RAIN_VALIDITY,
            "FREQUENCY_IS_OUTSIDE_THE_ITU_R_P838_3_VALIDITY_RANGE_OF_1_TO_1000_GHZ",
        )
    if rate < MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H:
        return _refused_estimate(
            "itu-r-p838-3",
            frequency,
            _RAIN_VALIDITY,
            "RAIN_RATE_IS_BELOW_THE_RATE_AT_WHICH_A_GAUGE_REGISTERS_SO_THIS_IS_NOT_RAIN",
        )
    if rate > MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H:
        return _refused_estimate(
            "itu-r-p838-3",
            frequency,
            _RAIN_VALIDITY,
            "RAIN_RATE_EXCEEDS_ANY_RATE_IN_THE_ITU_R_P837_MAPS",
        )

    k, alpha = rain_coefficients(
        frequency,
        polarisation_tilt_deg=polarisation_tilt_deg,
        path_elevation_deg=path_elevation_deg,
    )
    specific = k * rate**alpha
    reasons = [f"SPECIFIC_ATTENUATION_FROM_ITU_R_P838_3_WITH_K_{k:.6g}_ALPHA_{alpha:.5g}"]
    if abs(polarisation_tilt_deg - DEFAULT_POLARISATION_TILT_DEG) < 1e-9:
        reasons.append(
            "POLARISATION_TILT_WAS_NOT_DECLARED_SO_THE_CIRCULAR_45_DEGREE_MEAN_WAS_USED"
        )
    path: float | None = None
    if path_length_km is None:
        reasons.append("PATH_LENGTH_UNDECLARED_SO_ONLY_A_PER_KILOMETRE_FIGURE_EXISTS")
    else:
        path = specific * bounded_number(path_length_km, "path_length_km", 0.0, 10_000.0)
    return AttenuationEstimate(
        model="itu-r-p838-3",
        computed=True,
        frequency_hz=frequency,
        specific_attenuation_db_per_km=specific,
        path_attenuation_db=path,
        validity=_RAIN_VALIDITY,
        reasons=tuple(reasons),
    )


def cloud_specific_attenuation(
    frequency_hz: float,
    liquid_water_g_per_m3: float,
    *,
    temperature_k: float = DEFAULT_CLOUD_TEMPERATURE_K,
    path_length_km: float | None = None,
) -> AttenuationEstimate:
    """``gamma_c = K_l(f, T) M`` in dB/km, ITU-R P.840-8 Annex 1 eq. (1).

    Refuses above 200 GHz, where the recommendation states the Rayleigh
    approximation no longer holds, and above a liquid water density no cloud
    reaches.
    """
    frequency = require_finite(frequency_hz, "frequency_hz")
    water = require_finite(liquid_water_g_per_m3, "liquid_water_g_per_m3")
    if water < 0.0:
        raise WeatherModelError("liquid_water_g_per_m3 cannot be negative")
    if not 0.0 < frequency <= P840_MAX_FREQUENCY_HZ:
        return _refused_estimate(
            "itu-r-p840-8",
            frequency,
            _CLOUD_VALIDITY,
            "FREQUENCY_IS_ABOVE_THE_200_GHZ_LIMIT_OF_THE_RAYLEIGH_APPROXIMATION",
        )
    if water > MAX_PLAUSIBLE_LIQUID_WATER_G_PER_M3:
        return _refused_estimate(
            "itu-r-p840-8",
            frequency,
            _CLOUD_VALIDITY,
            "LIQUID_WATER_DENSITY_EXCEEDS_ANYTHING_A_CLOUD_OR_FOG_HOLDS",
        )
    if not MIN_CLOUD_TEMPERATURE_K <= temperature_k <= MAX_CLOUD_TEMPERATURE_K:
        return _refused_estimate(
            "itu-r-p840-8",
            frequency,
            _CLOUD_VALIDITY,
            "TEMPERATURE_IS_OUTSIDE_THE_RANGE_OVER_WHICH_CLOUD_LIQUID_WATER_EXISTS",
        )

    coefficient = cloud_attenuation_coefficient(frequency, temperature_k=temperature_k)
    specific = coefficient * water
    reasons = [f"SPECIFIC_ATTENUATION_FROM_ITU_R_P840_8_WITH_KL_{coefficient:.6g}"]
    if frequency < P840_STATED_SCOPE_MIN_FREQUENCY_HZ:
        reasons.append("BELOW_10_GHZ_ITU_R_P840_IS_OUTSIDE_ITS_OWN_STATED_SCOPE")
    path: float | None = None
    if path_length_km is None:
        reasons.append("PATH_LENGTH_UNDECLARED_SO_ONLY_A_PER_KILOMETRE_FIGURE_EXISTS")
    else:
        path = specific * bounded_number(path_length_km, "path_length_km", 0.0, 10_000.0)
    return AttenuationEstimate(
        model="itu-r-p840-8",
        computed=True,
        frequency_hz=frequency,
        specific_attenuation_db_per_km=specific,
        path_attenuation_db=path,
        validity=_CLOUD_VALIDITY,
        reasons=tuple(reasons),
    )


def scintillation_aperture_factor(
    frequency_hz: float,
    effective_antenna_diameter_m: float,
    path_elevation_deg: float,
) -> float:
    """ITU-R P.618 antenna averaging factor ``g(x)``.

    Carried only so the exponent test can check its own validity: ``g`` depends
    on frequency through ``x``, so a large aperture bends the measured
    scintillation exponent away from the 7/12 this module tests against. It is
    not used to predict a fade depth, because that needs ``sigma_ref`` and
    therefore a wet-refractivity figure this node cannot obtain.
    """
    frequency = bounded_number(frequency_hz, "frequency_hz", 1.0, 1e15)
    diameter = bounded_number(
        effective_antenna_diameter_m, "effective_antenna_diameter_m", 0.0, 1_000.0
    )
    elevation = bounded_number(
        path_elevation_deg, "path_elevation_deg", 0.0001, 90.0
    )
    sine = math.sin(math.radians(elevation))
    path_length_m = (
        2.0
        * SCINTILLATION_TURBULENT_LAYER_HEIGHT_M
        / (math.sqrt(sine * sine + SCINTILLATION_EARTH_CURVATURE_TERM) + sine)
    )
    x = 1.22 * diameter * diameter * (frequency / 1e9) / path_length_m
    if x >= 7.0:
        # ITU-R P.618 states the predicted scintillation fade is zero here: the
        # aperture has averaged the turbulence away entirely.
        return 0.0
    inner = 3.86 * (x * x + 1.0) ** (11.0 / 12.0) * math.sin(
        11.0 / 6.0 * math.atan2(1.0, x)
    ) - 7.08 * x ** (5.0 / 6.0)
    return math.sqrt(inner) if inner > 0.0 else 0.0


# ---------------------------------------------------------------------------
# The signature: what the node measured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandExcess:
    """One band's departure from its own baseline, in dB.

    ``excess_db`` is signed: positive is loss, negative is enhancement. Both
    directions are carried because a duct is a real thing that happens to a real
    link, and a type that can only hold loss cannot describe one.
    """

    frequency_hz: float
    excess_db: float
    fluctuation_std_db: float | None = None

    def __post_init__(self) -> None:
        bounded_number(self.frequency_hz, "frequency_hz", 1.0, 1e15)
        bounded_number(self.excess_db, "excess_db", -300.0, 300.0)
        if self.fluctuation_std_db is not None:
            bounded_number(self.fluctuation_std_db, "fluctuation_std_db", 0.0, 300.0)


@dataclass(frozen=True)
class WeatherSignature:
    """The four things that separate mechanisms, and nothing else.

    A frequency law, a time constant, a fluctuation, and the bands they were
    measured in. No rain gauge reading, no humidity, no sky camera — those live
    in :class:`DeclaredConditions` where their provenance is visible.
    """

    bands: tuple[BandExcess, ...]
    onset_seconds: float | None = None
    baseline_fluctuation_std_db: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "bands", tuple(self.bands))
        if self.onset_seconds is not None:
            bounded_number(self.onset_seconds, "onset_seconds", 0.0, 31_536_000.0)
        if self.baseline_fluctuation_std_db is not None:
            bounded_number(
                self.baseline_fluctuation_std_db,
                "baseline_fluctuation_std_db",
                0.0,
                300.0,
            )
        seen = {band.frequency_hz for band in self.bands}
        if len(seen) != len(self.bands):
            # Two excesses at one frequency disagree about a fact only one of
            # them can hold, and averaging them would invent a third.
            raise WeatherModelError(
                "a signature carries two bands at the same frequency"
            )

    @property
    def sense(self) -> str:
        """Which way the bands moved, taken together."""
        rising = [
            band for band in self.bands if band.excess_db >= MIN_MEASURABLE_EXCESS_DB
        ]
        falling = [
            band for band in self.bands if band.excess_db <= -MIN_MEASURABLE_EXCESS_DB
        ]
        if rising and falling:
            return SENSE_MIXED
        if rising:
            return SENSE_ATTENUATION
        if falling:
            return SENSE_ENHANCEMENT
        return SENSE_NONE

    @property
    def deepest_excess_db(self) -> float:
        """Largest signed departure, by magnitude. Zero for an empty signature."""
        if not self.bands:
            return 0.0
        return max((band.excess_db for band in self.bands), key=abs)


@dataclass(frozen=True)
class DeclaredConditions:
    """Everything an operator asserted, and nothing the node measured.

    Kept a separate type from the signature so that the boundary is visible in
    the code as well as the docs: no field here was obtained by this node, and
    a test that depends on one returns ``indeterminate`` when it is absent
    rather than assuming a default.
    """

    rain_rate_mm_per_h: float | None = None
    liquid_water_g_per_m3: float | None = None
    cloud_temperature_k: float | None = None
    path_length_km: float | None = None
    polarisation_tilt_deg: float | None = None
    path_elevation_deg: float | None = None
    melting_layer_on_path: bool | None = None
    precipitation_observed: bool | None = None
    arid_environment: bool | None = None
    refractivity_gradient_n_per_km: float | None = None
    effective_antenna_diameter_m: float | None = None

    def __post_init__(self) -> None:
        if self.rain_rate_mm_per_h is not None:
            bounded_number(self.rain_rate_mm_per_h, "rain_rate_mm_per_h", 0.0, 10_000.0)
        if self.liquid_water_g_per_m3 is not None:
            bounded_number(
                self.liquid_water_g_per_m3, "liquid_water_g_per_m3", 0.0, 100.0
            )
        if self.cloud_temperature_k is not None:
            bounded_number(self.cloud_temperature_k, "cloud_temperature_k", 150.0, 400.0)
        if self.path_length_km is not None:
            bounded_number(self.path_length_km, "path_length_km", 0.0, 10_000.0)
        if self.polarisation_tilt_deg is not None:
            bounded_number(
                self.polarisation_tilt_deg, "polarisation_tilt_deg", -90.0, 90.0
            )
        if self.path_elevation_deg is not None:
            bounded_number(self.path_elevation_deg, "path_elevation_deg", -90.0, 90.0)
        if self.refractivity_gradient_n_per_km is not None:
            bounded_number(
                self.refractivity_gradient_n_per_km,
                "refractivity_gradient_n_per_km",
                -2_000.0,
                2_000.0,
            )
        if self.effective_antenna_diameter_m is not None:
            bounded_number(
                self.effective_antenna_diameter_m,
                "effective_antenna_diameter_m",
                0.0,
                1_000.0,
            )

    @property
    def tilt_deg(self) -> float:
        """Declared polarisation tilt, or the least-committed circular default."""
        if self.polarisation_tilt_deg is None:
            return DEFAULT_POLARISATION_TILT_DEG
        return self.polarisation_tilt_deg

    @property
    def elevation_deg(self) -> float:
        """Declared path elevation, or zero for a terrestrial path."""
        return 0.0 if self.path_elevation_deg is None else self.path_elevation_deg


# ---------------------------------------------------------------------------
# The frequency law
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectralFit:
    """``n`` in ``excess ∝ f**n``, with the uncertainty that decides its worth.

    ``exponent`` is ``None`` whenever the fit refused, and ``refusal`` then says
    which test failed. A fit that succeeded but is too uncertain to separate
    mechanisms reports both the exponent and a ``standard_error`` above
    :data:`MAX_EXPONENT_STANDARD_ERROR`; the mechanism tests read that and
    decline, so the number is visible without being load-bearing.
    """

    exponent: float | None
    standard_error: float | None
    span_decades: float
    bands_used: int
    bands_supplied: int
    sense: str
    refusal: str | None

    @property
    def usable(self) -> bool:
        """True when the exponent exists and is precise enough to discriminate."""
        return (
            self.exponent is not None
            and self.standard_error is not None
            and self.standard_error <= MAX_EXPONENT_STANDARD_ERROR
        )

    def interval(self) -> tuple[float, float] | None:
        """Two-sigma interval on the exponent, or ``None`` if there is no fit."""
        if self.exponent is None or self.standard_error is None:
            return None
        margin = EXPONENT_CONFIDENCE_SIGMA * self.standard_error
        return self.exponent - margin, self.exponent + margin

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-spectral-fit/v1",
            "exponent": _rounded(self.exponent),
            "standard_error": _rounded(self.standard_error),
            "span_decades": _rounded(self.span_decades),
            "bands_used": self.bands_used,
            "bands_supplied": self.bands_supplied,
            "sense": self.sense,
            "usable": self.usable,
            "refusal": self.refusal,
        }


def _log_log_slope(
    points: Sequence[tuple[float, float, float]],
    *,
    min_span_decades: float,
) -> tuple[float | None, float | None, float, str | None]:
    """``(exponent, standard error, span, refusal)`` for a weighted log-log fit.

    ``points`` are ``(frequency, value, sigma_y)`` with ``sigma_y`` the point's
    uncertainty *after* the logarithm, so a caller working in decibels converts
    before it gets here and a caller working in a fractional uncertainty
    converts differently. Keeping the conversion outside is what lets one fitter
    serve a fade depth and a standard deviation without either pretending to be
    the other.

    The fit is weighted by ``1/sigma_y**2`` rather than treating every band
    alike, and that is not a refinement — it is the difference between using the
    evidence and discarding it. A fade is measured to about a decibel whatever
    its depth, so a 34 dB band pins its logarithm ten times more tightly than a
    3 dB band does. Weighting them equally, or worse assigning every band the
    shallowest band's error, throws away the deep bands that carry the slope and
    makes the module decline on data that was perfectly adequate.
    """
    xs = [math.log10(frequency) for frequency, _, _ in points]
    ys = [math.log10(value) for _, value, _ in points]
    weights = [1.0 / (sigma * sigma) for _, _, sigma in points]
    span = max(xs) - min(xs)
    if span < min_span_decades:
        return None, None, span, "BANDS_SPAN_TOO_LITTLE_FREQUENCY_TO_RESOLVE_A_POWER_LAW"
    slope, standard_error = weighted_slope(xs, ys, weights)
    if slope is None:
        # Every band's leverage collapsed onto one frequency. Dividing by that
        # would turn calibration scatter into an arbitrarily large exponent.
        return None, None, span, "BANDS_SPAN_TOO_LITTLE_FREQUENCY_TO_RESOLVE_A_POWER_LAW"
    return slope, standard_error, span, None


def weighted_slope(
    xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]
) -> tuple[float | None, float | None]:
    """Weighted least-squares slope of ``ys`` against ``xs``, and its standard error.

    Exposed rather than hidden because a model has to be compared to a
    measurement *the same way the measurement was made*. Fitting a curved law
    over a wide span and then comparing the answer against the law's endpoint
    chord compares two different numbers: over 10-40 GHz the ITU-R P.838-3 rain
    exponent runs from 2.6 to 1.1, so a fit that leans on the deepest bands lands
    well away from the chord and the disagreement looks like evidence when it is
    arithmetic. Passing the measurement's own weights through the model removes
    that entirely.
    """
    total = sum(weights)
    sum_x = sum(w * x for w, x in zip(weights, xs))
    sum_y = sum(w * y for w, y in zip(weights, ys))
    sum_xx = sum(w * x * x for w, x in zip(weights, xs))
    sum_xy = sum(w * x * y for w, x, y in zip(weights, xs, ys))
    determinant = total * sum_xx - sum_x * sum_x
    if determinant <= 0.0 or total <= 0.0:
        return None, None
    return (
        (total * sum_xy - sum_x * sum_y) / determinant,
        math.sqrt(total / determinant),
    )


def fit_spectral_exponent(
    bands: Sequence[BandExcess],
    *,
    excess_uncertainty_db: float = DEFAULT_EXCESS_UNCERTAINTY_DB,
    min_span_decades: float = MIN_FREQUENCY_SPAN_DECADES,
    min_magnitude_db: float = MIN_MEASURABLE_EXCESS_DB,
) -> SpectralFit:
    """Least-squares fit of ``log10 |excess|`` against ``log10 f``.

    The logarithm on both axes is the point. Path length multiplies every band's
    excess by the same factor, which is an additive constant in ``log10`` and
    therefore invisible to a slope — so the exponent describes the *mechanism*
    while a slope in dB per decade describes the mechanism and the path
    together, and this node knows the path length of nothing.

    Bands that did not move by ``min_magnitude_db`` are dropped: the logarithm
    of a number near zero is dominated by the noise on it, and one such band can
    rotate a two-band fit arbitrarily.

    The standard error is a *propagated* uncertainty, not a residual scatter —
    with two bands the fit is exact and has no residual to scatter. Each band
    carries its own uncertainty, ``excess_uncertainty_db / (ln 10 * |excess|)``,
    which is the calibration error of a decibel measurement transformed into log
    space and is therefore smaller for a deeper fade.
    """
    uncertainty = bounded_number(
        excess_uncertainty_db, "excess_uncertainty_db", 1e-6, 100.0
    )
    span_floor = bounded_number(min_span_decades, "min_span_decades", 0.0, 10.0)
    floor = bounded_number(min_magnitude_db, "min_magnitude_db", 1e-6, 100.0)
    supplied = len(bands)

    rising = [band for band in bands if band.excess_db >= floor]
    falling = [band for band in bands if band.excess_db <= -floor]
    if rising and falling:
        # Some bands worse, some better. That is not one power law, and fitting
        # one to it would average two mechanisms into a third that is neither.
        return SpectralFit(
            exponent=None,
            standard_error=None,
            span_decades=0.0,
            bands_used=0,
            bands_supplied=supplied,
            sense=SENSE_MIXED,
            refusal="BANDS_MOVED_IN_OPPOSITE_DIRECTIONS_SO_NO_SINGLE_POWER_LAW_FITS",
        )
    usable = rising or falling
    sense = (
        SENSE_ATTENUATION if rising else (SENSE_ENHANCEMENT if falling else SENSE_NONE)
    )
    if len(usable) < MIN_BANDS_FOR_EXPONENT:
        return SpectralFit(
            exponent=None,
            standard_error=None,
            span_decades=0.0,
            bands_used=len(usable),
            bands_supplied=supplied,
            sense=sense,
            refusal=(
                f"FEWER_THAN_{MIN_BANDS_FOR_EXPONENT}_BANDS_MOVED_MEASURABLY_SO_NO_"
                "FREQUENCY_LAW_WAS_FITTED"
            ),
        )

    exponent, standard_error, span, refusal = _log_log_slope(
        [
            (
                band.frequency_hz,
                abs(band.excess_db),
                uncertainty / (math.log(10.0) * abs(band.excess_db)),
            )
            for band in usable
        ],
        min_span_decades=span_floor,
    )
    return SpectralFit(
        exponent=exponent,
        standard_error=standard_error,
        span_decades=span,
        bands_used=len(usable),
        bands_supplied=supplied,
        sense=sense,
        refusal=refusal,
    )


def fit_scintillation_exponent(
    bands: Sequence[BandExcess],
    *,
    fractional_sigma_uncertainty: float = DEFAULT_SIGMA_FRACTIONAL_UNCERTAINTY,
    min_span_decades: float = MIN_FREQUENCY_SPAN_DECADES,
) -> SpectralFit:
    """The same fit applied to the per-band fluctuation standard deviations.

    Scintillation does not shift a mean, so fitting its *depth* would be fitting
    noise. What grows with frequency is ``sigma``, as ``f**(7/12)`` in
    ITU-R P.618, and that is what this fits. Bands with no reported standard
    deviation are dropped rather than assumed quiet.

    The uncertainty is fractional rather than absolute, which is the right shape
    for a standard deviation: the sampling error of ``sigma`` estimated from
    ``N`` readings is about ``sigma / sqrt(2(N-1))``, so it scales with ``sigma``
    itself and its logarithm has a constant uncertainty. The default corresponds
    to roughly nine readings.
    """
    fraction = bounded_number(
        fractional_sigma_uncertainty, "fractional_sigma_uncertainty", 1e-6, 10.0
    )
    span_floor = bounded_number(min_span_decades, "min_span_decades", 0.0, 10.0)
    carried = [
        band
        for band in bands
        if band.fluctuation_std_db is not None and band.fluctuation_std_db > 0.0
    ]
    if len(carried) < MIN_BANDS_FOR_EXPONENT:
        return SpectralFit(
            exponent=None,
            standard_error=None,
            span_decades=0.0,
            bands_used=len(carried),
            bands_supplied=len(bands),
            sense=SENSE_NONE,
            refusal=(
                f"FEWER_THAN_{MIN_BANDS_FOR_EXPONENT}_BANDS_REPORTED_A_FLUCTUATION_"
                "STANDARD_DEVIATION"
            ),
        )
    exponent, standard_error, span, refusal = _log_log_slope(
        [
            (
                band.frequency_hz,
                float(band.fluctuation_std_db or 0.0),
                fraction / math.log(10.0),
            )
            for band in carried
        ],
        min_span_decades=span_floor,
    )
    return SpectralFit(
        exponent=exponent,
        standard_error=standard_error,
        span_decades=span,
        bands_used=len(carried),
        bands_supplied=len(bands),
        # A standard deviation has no direction, so the sense of the mean shift
        # is deliberately not carried over from a fit that never saw one.
        sense=SENSE_NONE,
        refusal=refusal,
    )


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanismFinding:
    """What was concluded about one mechanism, and on what test."""

    mechanism: str
    finding: str
    validity: str
    reasons: tuple[str, ...]
    model_exponent_low: float | None = None
    model_exponent_high: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "finding": self.finding,
            "validity": self.validity,
            "model_exponent_low": _rounded(self.model_exponent_low),
            "model_exponent_high": _rounded(self.model_exponent_high),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class SignatureAssessment:
    """Every mechanism, its finding, and what could not be tested.

    ``resolved`` is true only when exactly one mechanism came back consistent
    *and* nothing was left indeterminate. Both halves matter: one consistent
    mechanism alongside three untested ones is not a resolution, it is a
    shortlist with most of the field unexamined.
    """

    findings: tuple[MechanismFinding, ...]
    fit: SpectralFit
    scintillation_fit: SpectralFit
    sense: str
    onset_seconds: float | None
    bands: int
    implied_rain: ImpliedRainRate | None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def _named(self, finding: str) -> tuple[str, ...]:
        return tuple(
            item.mechanism for item in self.findings if item.finding == finding
        )

    @property
    def consistent(self) -> tuple[str, ...]:
        return self._named(FINDING_CONSISTENT)

    @property
    def excluded(self) -> tuple[str, ...]:
        return self._named(FINDING_EXCLUDED)

    @property
    def indeterminate(self) -> tuple[str, ...]:
        return self._named(FINDING_INDETERMINATE)

    @property
    def resolved(self) -> bool:
        return len(self.consistent) == 1 and not self.indeterminate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-weather-assessment/v1",
            "sense": self.sense,
            "bands": self.bands,
            "onset_seconds": _rounded(self.onset_seconds, 1),
            "resolved": self.resolved,
            "consistent": list(self.consistent),
            "excluded": list(self.excluded),
            "indeterminate": list(self.indeterminate),
            "spectral_fit": self.fit.to_dict(),
            "scintillation_fit": self.scintillation_fit.to_dict(),
            "implied_rain_rate": (
                self.implied_rain.to_dict() if self.implied_rain is not None else None
            ),
            "findings": [item.to_dict() for item in self.findings],
            "reasons": list(self.reasons) + list(_STANDING_CAVEATS),
        }


_VALIDITY = {
    MECHANISM_RAIN: _RAIN_VALIDITY,
    MECHANISM_CLOUD_LIQUID_WATER: _CLOUD_VALIDITY,
    MECHANISM_MELTING_LAYER: (
        "no verified coefficient table; identified only by a declared 0 C "
        "isotherm crossing the path, and never separated from rain by slope"
    ),
    MECHANISM_DRY_SNOW: (
        "no verified coefficient table; ice has too little dielectric loss to "
        "produce a measurable fade, so the test is qualitative"
    ),
    MECHANISM_WATER_VAPOUR: (
        "ITU-R P.676-13 line centres only; quasi-static, so it explains a "
        "standing offset and never a transition"
    ),
    MECHANISM_OXYGEN: (
        "ITU-R P.676-13 line centres only; 50.474-68.960 GHz complex and the "
        "118.750 GHz line, constant in time"
    ),
    MECHANISM_SCINTILLATION: (
        "ITU-R P.618-14, 4 GHz to 20 GHz, path elevation at or above 5 degrees"
    ),
    MECHANISM_SAND_AND_DUST: (
        "no verified coefficient table; negligible below 10 GHz and strongly "
        "moisture-dependent, so no frequency law is asserted"
    ),
    MECHANISM_WET_FOLIAGE: (
        "no published frequency law this module could verify; it can be "
        "excluded by sign alone and never confirmed"
    ),
    MECHANISM_DUCTING: (
        "refractive rather than absorptive; ITU-R P.453 trapping gradient of "
        "-157 N-units/km, no attenuation figure is computed"
    ),
}


class WeatherDiscriminator:
    """Names every mechanism a signature is consistent with, and every one it is not.

    Every threshold is a constructor argument so a deployment that has
    characterised its own front end can tighten them, but the defaults are the
    ones this module's constants justify and are what the tests exercise.
    """

    def __init__(
        self,
        *,
        excess_uncertainty_db: float = DEFAULT_EXCESS_UNCERTAINTY_DB,
        min_span_decades: float = MIN_FREQUENCY_SPAN_DECADES,
        max_exponent_standard_error: float = MAX_EXPONENT_STANDARD_ERROR,
        min_measurable_excess_db: float = MIN_MEASURABLE_EXCESS_DB,
    ) -> None:
        self.excess_uncertainty_db = bounded_number(
            excess_uncertainty_db, "excess_uncertainty_db", 1e-6, 100.0
        )
        self.min_span_decades = bounded_number(
            min_span_decades, "min_span_decades", 0.0, 10.0
        )
        self.max_exponent_standard_error = bounded_number(
            max_exponent_standard_error, "max_exponent_standard_error", 1e-6, 10.0
        )
        self.min_measurable_excess_db = bounded_number(
            min_measurable_excess_db, "min_measurable_excess_db", 0.1, 100.0
        )

    # -- public API ---------------------------------------------------------

    def assess(
        self,
        signature: WeatherSignature,
        conditions: DeclaredConditions | None = None,
    ) -> SignatureAssessment:
        """Test one signature against every mechanism.

        Returns all consistent mechanisms rather than a winner. Several usually
        are, and picking one would be the guessing the rest of this repository
        exists to refuse.
        """
        declared = DeclaredConditions() if conditions is None else conditions
        reasons: list[str] = []

        if not signature.bands:
            reasons.append("SIGNATURE_CARRIED_NO_BANDS_SO_NOTHING_COULD_BE_TESTED")
        fit = fit_spectral_exponent(
            signature.bands,
            excess_uncertainty_db=self.excess_uncertainty_db,
            min_span_decades=self.min_span_decades,
            min_magnitude_db=self.min_measurable_excess_db,
        )
        scintillation = fit_scintillation_exponent(
            signature.bands, min_span_decades=self.min_span_decades
        )
        if fit.refusal is not None:
            reasons.append(fit.refusal)
        elif not fit.usable:
            reasons.append(
                "EXPONENT_STANDARD_ERROR_EXCEEDS_THE_LIMIT_AT_WHICH_MECHANISMS_SEPARATE"
            )
        if signature.onset_seconds is None:
            reasons.append("ONSET_WAS_NOT_TIMED_SO_EVERY_TIME_CONSTANT_TEST_DECLINED")

        moved = self._moved(signature)
        implied = self._implied_rain(fit, moved, declared)

        findings = tuple(
            self._finding(mechanism, signature, declared, fit, scintillation, moved, implied)
            for mechanism in MECHANISMS
        )
        if not any(item.finding == FINDING_CONSISTENT for item in findings):
            # Every mechanism ruled out is a result, not a failure to answer:
            # it says the shape belongs to something outside this module's
            # vocabulary, which is exactly when an operator should stop
            # attributing it to weather.
            reasons.append("NO_MODELLED_MECHANISM_IS_CONSISTENT_WITH_THIS_SIGNATURE")
        return SignatureAssessment(
            findings=findings,
            fit=fit,
            scintillation_fit=scintillation,
            sense=signature.sense,
            onset_seconds=signature.onset_seconds,
            bands=len(signature.bands),
            implied_rain=implied,
            reasons=_deduplicate(reasons),
        )

    # -- shared helpers -----------------------------------------------------

    def _moved(
        self, signature: WeatherSignature
    ) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """Frequencies of the measurably moved bands, and the weight each carries.

        The weights are the same ones the exponent fit used —
        ``1 / sigma_y**2`` with ``sigma_y = uncertainty / (ln 10 * |excess|)``,
        which reduces to a weight proportional to the square of the excess. They
        travel with the frequencies so that every model window is collapsed to a
        single exponent exactly the way the measurement was.
        """
        moved = [
            band
            for band in signature.bands
            if abs(band.excess_db) >= self.min_measurable_excess_db
        ]
        if len(moved) < MIN_BANDS_FOR_EXPONENT:
            return None
        frequencies = tuple(band.frequency_hz for band in moved)
        if max(frequencies) <= min(frequencies):
            return None
        weights = tuple(
            (math.log(10.0) * abs(band.excess_db) / self.excess_uncertainty_db) ** 2
            for band in moved
        )
        return frequencies, weights

    def _implied_rain(
        self,
        fit: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        declared: DeclaredConditions,
    ) -> ImpliedRainRate | None:
        """Rain rate the measured exponent implies, when that is computable."""
        if moved is None or not fit.usable or fit.sense != SENSE_ATTENUATION:
            return None
        if fit.exponent is None:
            return None
        frequencies, weights = moved
        if (
            min(frequencies) < P838_MIN_FREQUENCY_HZ
            or max(frequencies) > P838_MAX_FREQUENCY_HZ
        ):
            return None
        return implied_rain_rate(
            fit.exponent,
            frequencies,
            weights=weights,
            polarisation_tilt_deg=declared.tilt_deg,
            path_elevation_deg=declared.elevation_deg,
        )

    def _exponent_overlaps(
        self, fit: SpectralFit, low: float, high: float
    ) -> bool:
        """True when the measurement's two-sigma interval reaches a model window."""
        interval = fit.interval()
        if interval is None:
            return False
        measured_low, measured_high = interval
        return measured_low <= high and measured_high >= low

    def _absorptive_sense_refusal(self, signature: WeatherSignature) -> str | None:
        """Why an absorptive mechanism is impossible, given which way the band moved."""
        sense = signature.sense
        if sense == SENSE_ENHANCEMENT:
            return "BAND_IMPROVED_AND_AN_ABSORPTIVE_MECHANISM_CANNOT_ADD_SIGNAL"
        if sense == SENSE_MIXED:
            return "SOME_BANDS_IMPROVED_AND_SOME_WORSENED_WHICH_IS_NOT_ONE_MECHANISM"
        return None

    def _finding(
        self,
        mechanism: str,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        handler = {
            MECHANISM_RAIN: self._rain,
            MECHANISM_CLOUD_LIQUID_WATER: self._cloud,
            MECHANISM_MELTING_LAYER: self._melting_layer,
            MECHANISM_DRY_SNOW: self._dry_snow,
            MECHANISM_WATER_VAPOUR: self._water_vapour,
            MECHANISM_OXYGEN: self._oxygen,
            MECHANISM_SCINTILLATION: self._scintillation,
            MECHANISM_SAND_AND_DUST: self._sand_and_dust,
            MECHANISM_WET_FOLIAGE: self._wet_foliage,
            MECHANISM_DUCTING: self._ducting,
        }[mechanism]
        return handler(signature, declared, fit, scintillation, moved, implied)

    def _result(
        self,
        mechanism: str,
        finding: str,
        reasons: Sequence[str],
        *,
        window: tuple[float, float] | None = None,
    ) -> MechanismFinding:
        return MechanismFinding(
            mechanism=mechanism,
            finding=finding,
            validity=_VALIDITY[mechanism],
            reasons=_deduplicate(reasons),
            model_exponent_low=None if window is None else window[0],
            model_exponent_high=None if window is None else window[1],
        )

    # -- rain ---------------------------------------------------------------

    def _rain(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Rain, tested by inverting ITU-R P.838-3 for the rate it would need."""
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(MECHANISM_RAIN, FINDING_EXCLUDED, [wrong_sense])
        if signature.sense == SENSE_NONE:
            return self._result(
                MECHANISM_RAIN,
                FINDING_EXCLUDED,
                ["NO_BAND_MOVED_MEASURABLY_AND_RAIN_HEAVY_ENOUGH_TO_MATTER_WOULD_SHOW"],
            )

        reasons: list[str] = []
        onset = signature.onset_seconds
        if onset is not None:
            if onset < RAIN_MIN_ONSET_SECONDS:
                return self._result(
                    MECHANISM_RAIN,
                    FINDING_EXCLUDED,
                    [
                        "FADE_ARRIVED_FASTER_THAN_A_RAIN_CELL_CROSSES_A_PATH_SO_THE_"
                        "GEOMETRY_CHANGED_NOT_THE_SKY"
                    ],
                )
            if onset > RAIN_MAX_ONSET_SECONDS:
                return self._result(
                    MECHANISM_RAIN,
                    FINDING_EXCLUDED,
                    ["FADE_TOOK_LONGER_THAN_A_RAIN_EVENT_LASTS_SO_IT_IS_A_SLOWER_PROCESS"],
                )
            reasons.append("ONSET_IS_WITHIN_THE_MINUTES_TO_HOURS_RAIN_OCCUPIES")

        if moved is None:
            return self._result(
                MECHANISM_RAIN,
                FINDING_INDETERMINATE,
                reasons + ["TOO_FEW_MOVED_BANDS_TO_TEST_A_FREQUENCY_LAW"],
            )
        frequencies, weights = moved
        low, high = min(frequencies), max(frequencies)
        if low < P838_MIN_FREQUENCY_HZ or high > P838_MAX_FREQUENCY_HZ:
            return self._result(
                MECHANISM_RAIN,
                FINDING_INDETERMINATE,
                reasons
                + ["BANDS_REACH_OUTSIDE_THE_ITU_R_P838_3_VALIDITY_RANGE_OF_1_TO_1000_GHZ"],
            )
        if not fit.usable:
            return self._result(
                MECHANISM_RAIN,
                FINDING_INDETERMINATE,
                reasons
                + [
                    fit.refusal
                    or "EXPONENT_IS_TOO_UNCERTAIN_TO_TEST_AGAINST_THE_RAIN_MODEL"
                ],
            )

        if declared.rain_rate_mm_per_h is not None:
            return self._declared_rate_rain(frequencies, weights, declared, fit, reasons)

        # The model exponent is linear in ``log10 R``, so the whole plausible
        # rainfall range maps onto the interval between its two endpoints and no
        # search is needed. Which endpoint is which depends on the bands: above
        # about 5 GHz ``alpha`` falls with frequency and heavier rain flattens
        # the slope, below it the sign reverses.
        endpoints = tuple(
            rain_exponent(
                frequencies,
                rate,
                weights=weights,
                polarisation_tilt_deg=declared.tilt_deg,
                path_elevation_deg=declared.elevation_deg,
            )
            for rate in (
                MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H,
                MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H,
            )
        )
        window = (min(endpoints), max(endpoints))
        reasons.append(
            "RAIN_RATE_UNDECLARED_SO_THE_MODEL_WINDOW_SPANS_DRIZZLE_TO_TROPICAL_DOWNPOUR"
        )
        if high < P840_STATED_SCOPE_MIN_FREQUENCY_HZ:
            reasons.append(
                "EVERY_BAND_IS_BELOW_10_GHZ_WHERE_RAIN_NEEDS_AN_EXTREME_RATE_TO_MATTER"
            )
        if not self._exponent_overlaps(fit, *window):
            if implied is not None:
                reasons.extend(implied.reasons)
            return self._result(
                MECHANISM_RAIN,
                FINDING_EXCLUDED,
                reasons
                + ["MEASURED_EXPONENT_LIES_OUTSIDE_EVERY_RAIN_RATE_THE_MODEL_ADMITS"],
                window=window,
            )
        if implied is not None and implied.rain_rate_mm_per_h is not None:
            reasons.append(
                f"EXPONENT_IMPLIES_ABOUT_{implied.rain_rate_mm_per_h:.3g}_MM_PER_HOUR"
            )
            if not implied.plausible:
                # The point estimate fell outside the range while its two-sigma
                # interval still reaches inside. Saying so is the whole finding:
                # rain survives on the strength of the measurement's own
                # uncertainty and not on the strength of the measurement.
                reasons.append(
                    "RAIN_SURVIVES_ONLY_INSIDE_THE_EXPONENT_UNCERTAINTY_THE_POINT_"
                    "ESTIMATE_IS_OUTSIDE_THE_PLAUSIBLE_RANGE"
                )
        reasons.append("MEASURED_EXPONENT_IS_REACHABLE_BY_A_PLAUSIBLE_RAIN_RATE")
        return self._result(MECHANISM_RAIN, FINDING_CONSISTENT, reasons, window=window)

    def _declared_rate_rain(
        self,
        frequencies: Sequence[float],
        weights: Sequence[float],
        declared: DeclaredConditions,
        fit: SpectralFit,
        reasons: list[str],
    ) -> MechanismFinding:
        """Rain tested against a rate the operator asserted, which is the sharp case."""
        rate = declared.rain_rate_mm_per_h
        if rate is None or not (
            MIN_PLAUSIBLE_RAIN_RATE_MM_PER_H <= rate <= MAX_PLAUSIBLE_RAIN_RATE_MM_PER_H
        ):
            return self._result(
                MECHANISM_RAIN,
                FINDING_EXCLUDED,
                reasons + ["DECLARED_RAIN_RATE_IS_OUTSIDE_THE_RATES_RAIN_COMES_AT"],
            )
        predicted = rain_exponent(
            frequencies,
            rate,
            weights=weights,
            polarisation_tilt_deg=declared.tilt_deg,
            path_elevation_deg=declared.elevation_deg,
        )
        reasons.append(f"DECLARED_RAIN_RATE_PREDICTS_AN_EXPONENT_OF_{predicted:.3f}")
        window = (predicted, predicted)
        if not self._exponent_overlaps(fit, predicted, predicted):
            return self._result(
                MECHANISM_RAIN,
                FINDING_EXCLUDED,
                reasons
                + ["MEASURED_EXPONENT_DISAGREES_WITH_THE_DECLARED_RAIN_RATE"],
                window=window,
            )
        reasons.append("MEASURED_EXPONENT_MATCHES_THE_DECLARED_RAIN_RATE")
        return self._result(MECHANISM_RAIN, FINDING_CONSISTENT, reasons, window=window)

    # -- cloud and fog ------------------------------------------------------

    def _cloud(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Cloud and fog liquid water, tested against the ITU-R P.840 Rayleigh law."""
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(
                MECHANISM_CLOUD_LIQUID_WATER, FINDING_EXCLUDED, [wrong_sense]
            )
        if signature.sense == SENSE_NONE:
            return self._result(
                MECHANISM_CLOUD_LIQUID_WATER,
                FINDING_EXCLUDED,
                ["NO_BAND_MOVED_MEASURABLY_SO_THERE_IS_NO_ATTENUATION_TO_ATTRIBUTE"],
            )

        reasons: list[str] = []
        onset = signature.onset_seconds
        if onset is not None:
            if onset < CLOUD_MIN_ONSET_SECONDS:
                return self._result(
                    MECHANISM_CLOUD_LIQUID_WATER,
                    FINDING_EXCLUDED,
                    [
                        "FADE_ARRIVED_FASTER_THAN_FOG_THICKENS_ACROSS_A_PATH"
                    ],
                )
            if onset > CLOUD_MAX_ONSET_SECONDS:
                return self._result(
                    MECHANISM_CLOUD_LIQUID_WATER,
                    FINDING_EXCLUDED,
                    ["FADE_OUTLASTED_A_SINGLE_FOG_OR_CLOUD_EPISODE"],
                )
            reasons.append("ONSET_IS_WITHIN_THE_HOURS_FOG_AND_CLOUD_OCCUPY")

        if moved is None:
            return self._result(
                MECHANISM_CLOUD_LIQUID_WATER,
                FINDING_INDETERMINATE,
                reasons + ["TOO_FEW_MOVED_BANDS_TO_TEST_A_FREQUENCY_LAW"],
            )
        frequencies, weights = moved
        low, high = min(frequencies), max(frequencies)
        if high > P840_MAX_FREQUENCY_HZ:
            return self._result(
                MECHANISM_CLOUD_LIQUID_WATER,
                FINDING_INDETERMINATE,
                reasons
                + ["BANDS_REACH_ABOVE_THE_200_GHZ_LIMIT_OF_THE_RAYLEIGH_APPROXIMATION"],
            )
        if not fit.usable:
            return self._result(
                MECHANISM_CLOUD_LIQUID_WATER,
                FINDING_INDETERMINATE,
                reasons
                + [
                    fit.refusal
                    or "EXPONENT_IS_TOO_UNCERTAIN_TO_TEST_AGAINST_THE_CLOUD_MODEL"
                ],
            )

        if declared.cloud_temperature_k is None:
            cold = cloud_exponent(
                frequencies, weights=weights, temperature_k=MIN_CLOUD_TEMPERATURE_K
            )
            warm = cloud_exponent(
                frequencies, weights=weights, temperature_k=MAX_CLOUD_TEMPERATURE_K
            )
            window = (min(cold, warm), max(cold, warm))
            reasons.append(
                "CLOUD_TEMPERATURE_UNDECLARED_SO_THE_WINDOW_SPANS_SUPERCOOLED_TO_WARM"
            )
        else:
            predicted = cloud_exponent(
                frequencies,
                weights=weights,
                temperature_k=declared.cloud_temperature_k,
            )
            window = (predicted, predicted)
            reasons.append(
                f"DECLARED_TEMPERATURE_PREDICTS_AN_EXPONENT_OF_{predicted:.3f}"
            )
        if low < P840_STATED_SCOPE_MIN_FREQUENCY_HZ:
            reasons.append("A_BAND_SITS_BELOW_ITU_R_P840_OWN_STATED_SCOPE_OF_10_GHZ")

        if not self._exponent_overlaps(fit, *window):
            return self._result(
                MECHANISM_CLOUD_LIQUID_WATER,
                FINDING_EXCLUDED,
                reasons + ["MEASURED_EXPONENT_IS_NOT_THE_RAYLEIGH_F_SQUARED_LAW"],
                window=window,
            )
        reasons.append("MEASURED_EXPONENT_MATCHES_THE_RAYLEIGH_LAW_FOR_LIQUID_WATER")
        reasons.append(self._cloud_depth_reason(signature, declared, low, high))
        return self._result(
            MECHANISM_CLOUD_LIQUID_WATER, FINDING_CONSISTENT, reasons, window=window
        )

    def _cloud_depth_reason(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        low: float,
        high: float,
    ) -> str:
        """State whether the fade's depth is one cloud liquid water can reach.

        Reported as a reason rather than acted on. Turning it into an exclusion
        would make the finding depend on a declared path length, and a wrong
        path length would then silently rule out the right mechanism.
        """
        if declared.path_length_km is None or declared.path_length_km <= 0.0:
            return "PATH_LENGTH_UNDECLARED_SO_THE_FADE_DEPTH_WAS_NOT_CHECKED"
        temperature = (
            DEFAULT_CLOUD_TEMPERATURE_K
            if declared.cloud_temperature_k is None
            else declared.cloud_temperature_k
        )
        deepest = max(signature.bands, key=lambda band: abs(band.excess_db))
        coefficient = cloud_attenuation_coefficient(
            deepest.frequency_hz, temperature_k=temperature
        )
        required = abs(deepest.excess_db) / (coefficient * declared.path_length_km)
        if required > MAX_PLAUSIBLE_LIQUID_WATER_G_PER_M3:
            return (
                f"FADE_DEPTH_WOULD_NEED_{required:.3g}_G_PER_M3_OF_LIQUID_WATER_"
                "WHICH_NO_CLOUD_HOLDS"
            )
        return f"FADE_DEPTH_IMPLIES_ABOUT_{required:.3g}_G_PER_M3_OF_LIQUID_WATER"

    # -- melting layer ------------------------------------------------------

    def _melting_layer(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Wet snow in the melting layer — never separated from rain by slope.

        Melting-layer excess attenuation is a *multiple* of the rain attenuation
        at the same water content, so it shares rain's frequency law almost
        exactly. That is the honest finding and the reason this is not a slope
        test: whatever the exponent says, it says the same thing about both. The
        only thing that separates them is whether the 0 C isotherm crosses the
        path, and this node cannot see an isotherm.
        """
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(MECHANISM_MELTING_LAYER, FINDING_EXCLUDED, [wrong_sense])
        if signature.sense == SENSE_NONE:
            return self._result(
                MECHANISM_MELTING_LAYER,
                FINDING_EXCLUDED,
                ["NO_BAND_MOVED_MEASURABLY_SO_THERE_IS_NO_ATTENUATION_TO_ATTRIBUTE"],
            )
        if declared.melting_layer_on_path is False:
            return self._result(
                MECHANISM_MELTING_LAYER,
                FINDING_EXCLUDED,
                ["OPERATOR_DECLARED_NO_ZERO_DEGREE_ISOTHERM_ON_THE_PATH"],
            )
        base = [
            "MELTING_LAYER_ATTENUATION_SCALES_RAIN_ATTENUATION_SO_SLOPE_CANNOT_SEPARATE_THEM"
        ]
        if declared.melting_layer_on_path is None:
            return self._result(
                MECHANISM_MELTING_LAYER,
                FINDING_INDETERMINATE,
                base
                + [
                    "NO_DECLARED_ZERO_DEGREE_ISOTHERM_CROSSING_AND_THIS_NODE_CANNOT_"
                    "OBSERVE_ONE"
                ],
            )
        onset = signature.onset_seconds
        if onset is not None and onset > RAIN_MAX_ONSET_SECONDS:
            return self._result(
                MECHANISM_MELTING_LAYER,
                FINDING_EXCLUDED,
                base + ["FADE_OUTLASTED_THE_MINUTES_A_BRIGHT_BAND_TAKES_TO_CROSS_A_PATH"],
            )
        return self._result(
            MECHANISM_MELTING_LAYER,
            FINDING_CONSISTENT,
            base
            + [
                "OPERATOR_DECLARED_A_ZERO_DEGREE_ISOTHERM_ON_THE_PATH",
                "EXCESS_LOSS_OVER_RAIN_AT_THE_SAME_WATER_CONTENT_WAS_NOT_QUANTIFIED",
            ],
        )

    # -- dry snow -----------------------------------------------------------

    def _dry_snow(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Dry snow — identified by the *absence* of a fade during precipitation.

        Ice has a far smaller dielectric loss than liquid water, so dry snow at a
        given water-equivalent rate attenuates orders of magnitude less than rain
        at the same rate. That makes the informative case the negative one: a
        winter deployment reporting precipitation and *no* measurable fade has
        learned something real about the band, and a module that only looked for
        fades would have thrown it away.
        """
        deepest = abs(signature.deepest_excess_db)
        if deepest >= self.min_measurable_excess_db:
            return self._result(
                MECHANISM_DRY_SNOW,
                FINDING_EXCLUDED,
                [
                    "A_MEASURABLE_EXCESS_EXISTS_AND_ICE_HAS_TOO_LITTLE_DIELECTRIC_LOSS_"
                    "TO_PRODUCE_ONE",
                    "THIS_EXCLUSION_IS_QUALITATIVE_NO_VERIFIED_DRY_SNOW_COEFFICIENT_EXISTS",
                ],
            )
        if declared.precipitation_observed is None:
            return self._result(
                MECHANISM_DRY_SNOW,
                FINDING_INDETERMINATE,
                [
                    "NO_BAND_MOVED_BUT_NO_PRECIPITATION_WAS_DECLARED_SO_A_QUIET_BAND_"
                    "AND_A_DRY_SKY_ARE_THE_SAME_MEASUREMENT"
                ],
            )
        if not declared.precipitation_observed:
            return self._result(
                MECHANISM_DRY_SNOW,
                FINDING_EXCLUDED,
                ["OPERATOR_DECLARED_NO_PRECIPITATION"],
            )
        return self._result(
            MECHANISM_DRY_SNOW,
            FINDING_CONSISTENT,
            [
                "PRECIPITATION_WAS_DECLARED_AND_NO_BAND_FADED_WHICH_IS_THE_DRY_SNOW_"
                "SIGNATURE",
                "THIS_FINDING_IS_QUALITATIVE_NO_VERIFIED_DRY_SNOW_COEFFICIENT_EXISTS",
            ],
        )

    # -- gases --------------------------------------------------------------

    def _near_line(self, frequency_hz: float, line_hz: float) -> bool:
        """True when a frequency sits inside a line's zone of influence."""
        reach = LINE_INFLUENCE_HALF_WIDTHS * WATER_VAPOUR_LINE_HALF_WIDTH_HZ
        return abs(frequency_hz - line_hz) <= reach

    def _water_vapour(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Water vapour — a standing offset, never a transition."""
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(MECHANISM_WATER_VAPOUR, FINDING_EXCLUDED, [wrong_sense])
        onset = signature.onset_seconds
        if onset is not None and onset < QUASI_STATIC_MIN_ONSET_SECONDS:
            return self._result(
                MECHANISM_WATER_VAPOUR,
                FINDING_EXCLUDED,
                [
                    "VAPOUR_ABSORPTION_VARIES_DIURNALLY_AND_SEASONALLY_SO_IT_CANNOT_"
                    "PRODUCE_A_TRANSITION_THIS_FAST"
                ],
            )
        on_line = [
            band
            for band in signature.bands
            if self._near_line(band.frequency_hz, WATER_VAPOUR_PRINCIPAL_LINE_HZ)
            or self._near_line(band.frequency_hz, WATER_VAPOUR_SECOND_LINE_HZ)
        ]
        if not on_line:
            return self._result(
                MECHANISM_WATER_VAPOUR,
                FINDING_EXCLUDED,
                [
                    "NO_BAND_SITS_WITHIN_THREE_PRESSURE_BROADENED_HALF_WIDTHS_OF_A_"
                    "WATER_VAPOUR_LINE"
                ],
            )
        return self._result(
            MECHANISM_WATER_VAPOUR,
            FINDING_CONSISTENT,
            [
                "A_BAND_SITS_ON_THE_ITU_R_P676_WATER_VAPOUR_RESONANCE",
                "VAPOUR_LOSS_IS_A_STANDING_OFFSET_ON_THIS_BAND_NOT_AN_EXPLANATION_FOR_"
                "A_CHANGE",
                "NO_VAPOUR_ATTENUATION_FIGURE_WAS_COMPUTED_THIS_NODE_HAS_NO_HYGROMETER",
            ],
        )

    def _oxygen(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Oxygen — a property of the band, and not weather at all."""
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(MECHANISM_OXYGEN, FINDING_EXCLUDED, [wrong_sense])
        onset = signature.onset_seconds
        if onset is not None and onset < QUASI_STATIC_MIN_ONSET_SECONDS:
            return self._result(
                MECHANISM_OXYGEN,
                FINDING_EXCLUDED,
                [
                    "OXYGEN_CONCENTRATION_IS_CONSTANT_SO_IT_CANNOT_PRODUCE_A_TRANSITION"
                ],
            )
        in_band = [
            band
            for band in signature.bands
            if OXYGEN_COMPLEX_LOW_HZ <= band.frequency_hz <= OXYGEN_COMPLEX_HIGH_HZ
            or self._near_line(band.frequency_hz, OXYGEN_ISOLATED_LINE_HZ)
        ]
        if not in_band:
            return self._result(
                MECHANISM_OXYGEN,
                FINDING_EXCLUDED,
                ["NO_BAND_SITS_IN_THE_60_GHZ_COMPLEX_OR_ON_THE_118_GHZ_LINE"],
            )
        return self._result(
            MECHANISM_OXYGEN,
            FINDING_CONSISTENT,
            [
                "A_BAND_SITS_IN_AN_ITU_R_P676_OXYGEN_ABSORPTION_BAND",
                "OXYGEN_ABSORPTION_IS_A_PROPERTY_OF_THE_BAND_NOT_A_WEATHER_EVENT",
                "CHOOSING_THIS_BAND_ACCEPTS_A_STANDING_LOSS_THAT_WILL_NOT_CLEAR",
            ],
        )

    # -- scintillation ------------------------------------------------------

    def _scintillation(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Scintillation — a variance signature, tested on sigma and not on depth."""
        reasons: list[str] = []
        frequencies = [band.frequency_hz for band in signature.bands]
        if not frequencies:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_INDETERMINATE,
                ["SIGNATURE_CARRIED_NO_BANDS"],
            )
        if (
            min(frequencies) < P618_SCINTILLATION_MIN_FREQUENCY_HZ
            or max(frequencies) > P618_SCINTILLATION_MAX_FREQUENCY_HZ
        ):
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_INDETERMINATE,
                [
                    "BANDS_REACH_OUTSIDE_THE_4_TO_20_GHZ_RANGE_THE_ITU_R_P618_METHOD_"
                    "IS_OFFERED_FOR"
                ],
            )
        elevation = declared.path_elevation_deg
        if elevation is not None and elevation < P618_SCINTILLATION_MIN_ELEVATION_DEG:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_INDETERMINATE,
                [
                    "PATH_ELEVATION_IS_BELOW_THE_5_DEGREES_THE_ITU_R_P618_METHOD_IS_"
                    "OFFERED_FOR_AND_SCINTILLATION_IS_WORST_THERE"
                ],
            )

        if abs(signature.deepest_excess_db) >= SCINTILLATION_MAX_MEAN_EXCESS_DB:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_EXCLUDED,
                [
                    "THE_MEAN_MOVED_AND_SCINTILLATION_IS_A_FLUCTUATION_ABOUT_A_MEAN_"
                    "NOT_A_SHIFT_OF_ONE"
                ],
            )

        stds = [
            band.fluctuation_std_db
            for band in signature.bands
            if band.fluctuation_std_db is not None
        ]
        if not stds:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_INDETERMINATE,
                ["NO_BAND_REPORTED_A_FLUCTUATION_STANDARD_DEVIATION"],
            )
        baseline = signature.baseline_fluctuation_std_db
        if baseline is None:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_INDETERMINATE,
                [
                    "NO_BASELINE_FLUCTUATION_WAS_SUPPLIED_SO_A_RISE_IN_VARIANCE_COULD_"
                    "NOT_BE_ESTABLISHED"
                ],
            )
        if baseline <= 0.0:
            # A baseline standard deviation of exactly zero is a quantised
            # series, not a silent one, and dividing by it would let any
            # fluctuation at all clear the gate.
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_INDETERMINATE,
                ["BASELINE_FLUCTUATION_OF_ZERO_IS_A_QUANTISED_SERIES_NOT_A_QUIET_ONE"],
            )
        if max(stds) < SCINTILLATION_MIN_VARIANCE_RATIO * baseline:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_EXCLUDED,
                ["FLUCTUATION_DID_NOT_RISE_ABOVE_THE_BASELINE_SCATTER"],
            )
        reasons.append("FLUCTUATION_ROSE_WELL_ABOVE_THE_BASELINE_SCATTER")
        reasons.append("MEAN_RECEIVED_POWER_DID_NOT_MEASURABLY_MOVE")

        aperture = self._aperture_reason(declared, max(frequencies))
        reasons.append(aperture)

        if not scintillation.usable:
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_CONSISTENT,
                reasons
                + [
                    scintillation.refusal
                    or "SIGMA_EXPONENT_TOO_UNCERTAIN_TO_TEST_AGAINST_THE_7_OVER_12_LAW",
                    "CONSISTENT_ON_THE_VARIANCE_SIGNATURE_ALONE_THE_FREQUENCY_LAW_WAS_"
                    "NOT_TESTED",
                ],
            )
        window = (SCINTILLATION_SIGMA_EXPONENT, SCINTILLATION_SIGMA_EXPONENT)
        if not self._exponent_overlaps(scintillation, *window):
            return self._result(
                MECHANISM_SCINTILLATION,
                FINDING_EXCLUDED,
                reasons
                + ["SIGMA_DOES_NOT_GROW_AS_THE_ITU_R_P618_SEVEN_TWELFTHS_POWER"],
                window=window,
            )
        reasons.append("SIGMA_GROWS_AS_THE_ITU_R_P618_SEVEN_TWELFTHS_POWER_OF_FREQUENCY")
        return self._result(
            MECHANISM_SCINTILLATION, FINDING_CONSISTENT, reasons, window=window
        )

    def _aperture_reason(
        self, declared: DeclaredConditions, frequency_hz: float
    ) -> str:
        """Whether aperture averaging could have bent the measured sigma exponent."""
        if declared.effective_antenna_diameter_m is None:
            return "APERTURE_AVERAGING_WAS_NOT_TESTED_NO_ANTENNA_DIAMETER_WAS_DECLARED"
        elevation = declared.path_elevation_deg
        if elevation is None or elevation <= 0.0:
            return "APERTURE_AVERAGING_WAS_NOT_TESTED_NO_PATH_ELEVATION_WAS_DECLARED"
        sine = math.sin(math.radians(elevation))
        path_length_m = (
            2.0
            * SCINTILLATION_TURBULENT_LAYER_HEIGHT_M
            / (math.sqrt(sine * sine + SCINTILLATION_EARTH_CURVATURE_TERM) + sine)
        )
        diameter = declared.effective_antenna_diameter_m
        x = 1.22 * diameter * diameter * (frequency_hz / 1e9) / path_length_m
        if x > SCINTILLATION_NEGLIGIBLE_APERTURE_X:
            return (
                f"APERTURE_AVERAGING_IS_NOT_NEGLIGIBLE_X_IS_{x:.4g}_SO_THE_MEASURED_"
                "SIGMA_EXPONENT_IS_BIASED_BELOW_SEVEN_TWELFTHS"
            )
        return "APERTURE_AVERAGING_IS_NEGLIGIBLE_SO_THE_SEVEN_TWELFTHS_LAW_APPLIES"

    # -- sand and dust ------------------------------------------------------

    def _sand_and_dust(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Sand and dust — no coefficient table was verifiable, so no law is asserted."""
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(MECHANISM_SAND_AND_DUST, FINDING_EXCLUDED, [wrong_sense])
        if declared.arid_environment is False:
            return self._result(
                MECHANISM_SAND_AND_DUST,
                FINDING_EXCLUDED,
                ["OPERATOR_DECLARED_A_NON_ARID_ENVIRONMENT"],
            )
        if signature.sense == SENSE_NONE:
            return self._result(
                MECHANISM_SAND_AND_DUST,
                FINDING_EXCLUDED,
                ["NO_BAND_MOVED_MEASURABLY_SO_THERE_IS_NO_ATTENUATION_TO_ATTRIBUTE"],
            )
        reasons = [
            "NO_VERIFIED_COEFFICIENT_TABLE_FOR_SAND_AND_DUST_SO_NO_FREQUENCY_LAW_WAS_TESTED",
            "SAND_AND_DUST_ATTENUATION_DEPENDS_STRONGLY_ON_PARTICLE_MOISTURE_WHICH_IS_"
            "NOT_OBSERVABLE_HERE",
        ]
        if declared.arid_environment is None:
            reasons.append("NO_ARID_ENVIRONMENT_WAS_DECLARED")
        else:
            reasons.append("OPERATOR_DECLARED_AN_ARID_ENVIRONMENT")
        moved = [
            band
            for band in signature.bands
            if abs(band.excess_db) >= self.min_measurable_excess_db
        ]
        if moved and all(
            band.frequency_hz < SAND_AND_DUST_MIN_SIGNIFICANT_FREQUENCY_HZ
            for band in moved
        ):
            reasons.append(
                "EVERY_MOVED_BAND_IS_BELOW_10_GHZ_WHERE_DUST_ATTENUATION_IS_NEGLIGIBLE"
            )
        return self._result(MECHANISM_SAND_AND_DUST, FINDING_INDETERMINATE, reasons)

    # -- wet foliage --------------------------------------------------------

    def _wet_foliage(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Wet foliage — undetermined by construction.

        Vegetation loss on wetting is a step change with no clean frequency law:
        it depends on species, leaf area, water film thickness and the geometry
        of which branches sit in the first Fresnel zone. There is no coefficient
        this module could carry with a citation, so there is no test it could
        run, and inventing a slope window here would be exactly the defect this
        repository exists to prevent. The one thing that *can* be said is that
        wet vegetation absorbs, so it cannot make a band better.
        """
        wrong_sense = self._absorptive_sense_refusal(signature)
        if wrong_sense is not None:
            return self._result(MECHANISM_WET_FOLIAGE, FINDING_EXCLUDED, [wrong_sense])
        if signature.sense == SENSE_NONE:
            return self._result(
                MECHANISM_WET_FOLIAGE,
                FINDING_EXCLUDED,
                ["NO_BAND_MOVED_MEASURABLY_SO_THERE_IS_NO_ATTENUATION_TO_ATTRIBUTE"],
            )
        return self._result(
            MECHANISM_WET_FOLIAGE,
            FINDING_INDETERMINATE,
            [
                "WET_FOLIAGE_HAS_NO_PUBLISHED_FREQUENCY_LAW_THIS_MODULE_WILL_NOT_INVENT_ONE",
                "IT_CAN_BE_NEITHER_CONFIRMED_NOR_EXCLUDED_FROM_SLOPE_AND_ONSET_ALONE",
            ],
        )

    # -- ducting ------------------------------------------------------------

    def _ducting(
        self,
        signature: WeatherSignature,
        declared: DeclaredConditions,
        fit: SpectralFit,
        scintillation: SpectralFit,
        moved: tuple[tuple[float, ...], tuple[float, ...]] | None,
        implied: ImpliedRainRate | None,
    ) -> MechanismFinding:
        """Ducting — refractive, not absorptive, and the one mechanism that can enhance.

        A trapping layer bends rays back towards the surface. A node inside the
        duct sees received power *above* free space; a node outside one that the
        emitter is inside sees energy that never reaches it. So the sign of the
        excess does not settle it — but the frequency law does. Refraction
        redirects rather than absorbs, so it has no power-law slope of the kind
        every absorptive mechanism here shows.

        No enhancement figure is computed. A duct's gain depends on the layer's
        thickness and strength and on where both ends of the link sit inside it,
        and this node knows none of those.
        """
        gradient = declared.refractivity_gradient_n_per_km
        if gradient is not None and gradient > DUCTING_TRAPPING_GRADIENT_N_PER_KM:
            return self._result(
                MECHANISM_DUCTING,
                FINDING_EXCLUDED,
                [
                    "DECLARED_REFRACTIVITY_GRADIENT_IS_ABOVE_THE_MINUS_157_N_UNITS_PER_"
                    "KM_AT_WHICH_TRAPPING_BEGINS"
                ],
            )
        reasons = ["DUCTING_IS_REFRACTIVE_NOT_ABSORPTIVE_IT_REDIRECTS_ENERGY"]
        if gradient is None:
            reasons.append(
                "NO_REFRACTIVITY_GRADIENT_WAS_DECLARED_SO_TRAPPING_WAS_NOT_CONFIRMED"
            )
        else:
            reasons.append("DECLARED_REFRACTIVITY_GRADIENT_SUPPORTS_TRAPPING")

        if signature.sense == SENSE_NONE:
            return self._result(
                MECHANISM_DUCTING,
                FINDING_EXCLUDED,
                reasons + ["NO_BAND_MOVED_MEASURABLY_IN_EITHER_DIRECTION"],
            )
        if fit.usable and fit.exponent is not None:
            if abs(fit.exponent) > DUCTING_MAX_EXPONENT_MAGNITUDE:
                return self._result(
                    MECHANISM_DUCTING,
                    FINDING_EXCLUDED,
                    reasons
                    + [
                        "EXCESS_FOLLOWS_A_POWER_LAW_IN_FREQUENCY_WHICH_A_REFRACTIVE_"
                        "MECHANISM_DOES_NOT_PRODUCE"
                    ],
                    window=(
                        -DUCTING_MAX_EXPONENT_MAGNITUDE,
                        DUCTING_MAX_EXPONENT_MAGNITUDE,
                    ),
                )
            reasons.append("EXCESS_IS_NEARLY_FLAT_IN_FREQUENCY_AS_REFRACTION_IS")
        else:
            # Flatness is the whole test. Without it a duct is neither shown nor
            # ruled out, and reporting "consistent" on the strength of an
            # untested property would let every unfitted fade look like a duct.
            return self._result(
                MECHANISM_DUCTING,
                FINDING_INDETERMINATE,
                reasons
                + [
                    fit.refusal
                    or "EXPONENT_IS_TOO_UNCERTAIN_TO_ESTABLISH_THAT_THE_EXCESS_IS_FLAT",
                    "FLATNESS_IN_FREQUENCY_IS_THE_ONLY_TEST_THIS_MODULE_HAS_FOR_A_DUCT",
                ],
            )

        if signature.sense == SENSE_ENHANCEMENT:
            reasons.append("RECEIVED_POWER_ROSE_WHICH_IS_THE_SIGNATURE_OF_BEING_IN_A_DUCT")
        else:
            reasons.append(
                "RECEIVED_POWER_FELL_WHICH_A_DUCT_ALSO_PRODUCES_FOR_A_NODE_OUTSIDE_IT"
            )
        reasons.append("NO_ENHANCEMENT_FIGURE_WAS_COMPUTED_DUCT_GEOMETRY_IS_UNKNOWN_HERE")
        return self._result(MECHANISM_DUCTING, FINDING_CONSISTENT, reasons)
