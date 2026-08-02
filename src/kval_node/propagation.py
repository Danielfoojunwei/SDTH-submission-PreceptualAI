"""Why a band got worse: something in the path, or something in the band.

When a link degrades, the operator's first question is not *how bad* but *what
kind of bad*, because the two kinds have opposite remedies. Something between
the node and the emitter — a hangar wall, a ridge line, a treeline, a rain cell
— is answered by moving, waiting, or climbing. Energy arriving in the band from
another emitter is answered by leaving the band. Treating one as the other
wastes the only thing a contested-spectrum node has, which is time.

The physics separates them cleanly, and the separation is visible in the two
figures a passive receiver already reports:

* **Attenuation** sits on the path. It cuts the wanted signal and leaves the
  receiver's own noise floor exactly where it was, because nothing has been
  added to the band — something has been taken out of it.
* **Energy arriving in the band** raises the measured noise floor. It cannot
  lower the wanted signal, because the path is untouched.

So received power down with the floor flat is attenuation; the floor up is
arriving energy. That is the whole discriminator, and everything else in this
module exists to stop it being applied where it does not hold.

**What it will not do.**

*It will not name an actor or an intent.* A rising noise floor says energy
arrived. It does not say who emitted it, whether they meant to, or whether they
know this node exists. A microwave oven, a harmonic from a failing amplifier, a
partner unit's own radio and a hostile emitter produce the same measurement.
:mod:`kval_node.interference` already draws that line for reporting; this module
draws it for detection, and the two must agree.

*It will not resolve the concurrent case by picking the larger effect.* A signal
that falls while the floor rises is either two things happening at once or one
thing this module cannot decompose — in-band energy strong enough to compress a
receiver's front end reduces apparent signal *and* raises the floor by itself.
The verdict says exactly that and attributes nothing.

*It will not claim weather.* A slow, frequency-selective fade is the hydrometeor
signature, and the module reports the signature. It has no barometer, no
rain gauge and no met feed, so it says "consistent with", carries the evidence,
and leaves the meteorology to something that measured it.

*It will not describe an absolute state.* A baseline window and a recent window
are compared, so what is detected is a **transition**. A band that was already
degraded before the series began yields "no degradation": there is no
transition in view, and reporting one would be inventing history.

**Frequency dependence, and why one band is rarely enough.** Hydrometeor
attenuation is steeply frequency-selective — ITU-R P.838-3 models specific rain
attenuation as ``gamma = k * R**alpha`` with both coefficients rising sharply
with frequency — so a fade that deepens with frequency across simultaneously
observed bands is the rain-fade signature. It is not a *unique* signature:
knife-edge diffraction loss also grows with frequency, at roughly the same
order. What separates them is time and geometry. Rain changes over minutes to
hours with the path geometry unchanged; a diffracting edge changes when the
platform moves. The fade branch therefore requires a slow onset, a
frequency slope, a band high enough for rain to matter at all, *and* a platform
track showing the geometry held still. On a manoeuvring platform this branch
will usually decline, which is the correct answer rather than a defect.

**Research grounding.** The framing comes from the radio-environment-map
literature — EPFL's REMNet work (arXiv:2511.05238) attributes propagation
differences across a map to "building densities and material compositions",
that is, to structure on the path rather than to occupancy of the spectrum.
Their estimator is a neural network trained on surveyed maps. Neither the
network nor the survey exists here: this repository ships no PyPI runtime
dependency and no training corpus, and a fitted map of somewhere else is not
evidence about here. What transfers is the physical distinction their features
encode, implemented directly from the measurements a receive-only node already
holds.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from .mobility import Fix, metres_between
from .spectrum import SpectrumBand, SpectrumObservation
from .validation import bounded_number, positive_int

#: What the evidence supported. ``insufficient_evidence`` and ``ambiguous`` are
#: refusals; ``concurrent`` is a refusal that names both possibilities.
CAUSE_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CAUSE_NO_DEGRADATION = "no_degradation"
CAUSE_PATH_ATTENUATION = "path_attenuation"
CAUSE_ENERGY_IN_BAND = "energy_in_band"
CAUSE_CONCURRENT = "concurrent_attenuation_and_energy_in_band"
CAUSE_AMBIGUOUS = "ambiguous"

#: Which attenuation mechanism the fade's shape is consistent with. Reported
#: only under ``CAUSE_PATH_ATTENUATION``; a cause that was not established has
#: no mechanism to describe.
MECHANISM_NOT_APPLICABLE = "not_applicable"
MECHANISM_UNDETERMINED = "undetermined"
MECHANISM_ABRUPT_OBSTRUCTION = "abrupt_obstruction"
MECHANISM_FREQUENCY_SELECTIVE_FADE = "frequency_selective_fade"

#: Shift, in dB, below which a change is not separable from the instrument that
#: measured it. A low-cost SDR front end drifts with temperature and its
#: absolute calibration is good to about a decibel; 3 dB — a factor of two in
#: power — clears that budget with margin and is the smallest change this module
#: will attribute to the environment rather than to itself.
MIN_SHIFT_DB = 3.0

#: Multiples of the baseline's own standard deviation a shift must also clear.
#: For an approximately normal baseline, 3 sigma is a 0.27% two-sided
#: false-alarm rate per test, and two tests run per verdict — noise floor and
#: received power — so a verdict false-alarms at roughly half a percent. That
#: arithmetic assumes independent, normally distributed samples. Real RF
#: measurements are autocorrelated and are neither, so the true rate is higher:
#: this is a design floor, not a guarantee, and it is why corroboration across
#: nodes remains the thing that turns a verdict into a decision.
MIN_SHIFT_SIGMA = 3.0

#: Baseline standard deviation above which there is no baseline to compare
#: against. A band whose readings already swing by 6 dB has something happening
#: inside the reference window, and at the 3-sigma gate a shift would have to
#: reach 18 dB before it registered — which is an outage, not a degradation. A
#: bar nothing can clear tells an operator less than a refusal does.
MAX_BASELINE_SPREAD_DB = 6.0

#: Observations required in the reference window. Six is the point at which the
#: window's own scatter is estimable and the standard error of its mean has
#: fallen to about 0.4 of the per-sample spread, so a shift of a few spreads is
#: not a sampling artefact.
MIN_BASELINE_SAMPLES = 6

#: Observations required in the window under test. Four is the fewest that can
#: distinguish a step from a ramp: three intervals, which a linear ramp divides
#: evenly and a step does not.
MIN_RECENT_SAMPLES = 4

#: Fraction of the whole fade that must land inside a single sample interval
#: before the onset is called a step. A linear ramp across the four-sample
#: minimum concentrates at most a third of its change in any one interval, so
#: 0.7 separates the two with room for scatter on a genuine step.
STEP_CONCENTRATION_RATIO = 0.7

#: Ten-to-ninety-percent onset, in seconds, below which a fade is too fast to be
#: a hydrometeor event. Rain cells are kilometres across and their rates change
#: over minutes to hours. A fade that completes inside two minutes is a change
#: of geometry — a platform rounding a building, a mast crossing the path — not
#: a change in what is falling out of the sky.
MIN_SLOW_ONSET_SECONDS = 120.0

#: Below this frequency a slow fade is not hydrometeor attenuation, so the
#: module will not call it one. ITU-R P.838-3 tabulates ``k`` and ``alpha``
#: against frequency for ``gamma = k * R**alpha``: at 10 GHz a 25 mm/h rain
#: gives on the order of 0.7 dB/km, while the same rain at 2 GHz gives on the
#: order of 0.003 dB/km — a factor of a few hundred. 10 GHz is where a path of a
#: few kilometres first produces a fade deep enough for this module's own 3 dB
#: floor to see at all.
RAIN_FADE_MIN_FREQUENCY_HZ = 10_000_000_000.0

#: Deepening of the fade per decade of frequency required before it counts as
#: frequency-selective. A mechanism that is flat in frequency — a wall of fixed
#: loss, terrain shadowing across an octave — produces about zero. 6 dB per
#: decade is a factor of four in power across a decade, which no flat mechanism
#: reaches. It deliberately does *not* exclude knife-edge diffraction, whose
#: loss grows as roughly ``10 * log10(f)``, about 10 dB per decade: frequency
#: slope alone cannot separate rain from a diffracting edge, which is why the
#: fade branch also requires the platform's geometry to have held still.
RAIN_FADE_MIN_SLOPE_DB_PER_DECADE = 6.0

#: Frequency span the analysed bands must cover before a slope means anything.
#: Derived rather than chosen: the shift floor divided by the span must not on
#: its own exceed the slope threshold, or ordinary calibration scatter between
#: two adjacent bands would register as frequency selectivity. 3.0 / 6.0 = 0.5
#: decades, a little over one and a half octaves.
MIN_FREQUENCY_SPAN_DECADES = MIN_SHIFT_DB / RAIN_FADE_MIN_SLOPE_DB_PER_DECADE

#: Platform displacement, in metres, below which the path geometry is treated as
#: unchanged. The first Fresnel zone radius at this module's 10 GHz floor, taken
#: at the midpoint of a 5 km path, is ``sqrt(lambda * d1 * d2 / d)`` =
#: ``sqrt(0.03 * 2500 * 2500 / 5000)``, about 6 m. A platform that moves less
#: than one Fresnel radius has not appreciably changed which objects clip the
#: path. Consumer GNSS uncertainty is of the same order, which is why a track
#: whose fixes are looser than this cannot establish stillness at all.
STATIC_PLATFORM_DISPLACEMENT_M = 6.0

#: Bands that must all show the same noise-floor rise before the rise stops
#: being attributable to the environment. Two bands can sit inside one emitter's
#: occupied bandwidth. Three bands spread across more than half a decade of
#: frequency, all rising by the same amount, is explained at least as
#: economically by the receiver — a gain step, an LNA warming up, an antenna
#: connector going high-resistance — as by anything outside it, and nothing in a
#: noise floor separates those two.
RECEIVER_WIDE_RISE_MIN_BANDS = 3

#: Longest unobserved interval that may sit inside an analysed series. Tied to
#: the slow-onset bar: a silence longer than the shortest fade this module will
#: call gradual makes a step indistinguishable from a ramp that ran during the
#: silence, and the onset classification would be reporting the sampling
#: schedule rather than the weather.
MAX_SAMPLE_GAP_SECONDS = MIN_SLOW_ONSET_SECONDS

#: Samples of pre-transition context prepended to the window under test when an
#: onset is timed. Three is enough to catch a fade that began just before the
#: window opened, without reaching so far back that the baseline's own scatter
#: reads as the start of the fade.
TRANSITION_LEAD_SAMPLES = 3

#: Smallest baseline scatter the sigma gate will divide by. A reported power
#: series with a standard deviation of exactly zero is quantised, not noiseless,
#: and treating it as noiseless would let any change at all clear the gate.
MIN_EFFECTIVE_SPREAD_DB = 0.5

#: Observation confidence below which a reading is not evidence. The same floor
#: :class:`~kval_node.spectrum.SpectrumAnalyzer` applies before it will call a
#: band clear: readings too poor to support "this band is usable" are too poor
#: to support "this is why it stopped being usable".
MIN_OBSERVATION_CONFIDENCE = 0.25

#: Appended to every serialised verdict. These are the two claims the module is
#: structurally incapable of making, and stating them in the record is cheaper
#: than trusting every downstream consumer to remember.
_STANDING_CAVEATS = (
    "NO_SOURCE_AND_NO_INTENT_ARE_CLAIMED_ENERGY_IN_A_BAND_IS_AN_OBSERVATION",
    "A_MECHANISM_CONSISTENT_WITH_THE_SIGNATURE_IS_NOT_A_MECHANISM_OBSERVED",
)


class PropagationError(ValueError):
    """Raised when a caller wires the discriminator up wrongly.

    Reserved for caller mistakes — observations from the wrong band, a window
    size that cannot exist. Weakness in the *evidence* never raises; it produces
    a verdict carrying its refusal, because "we could not tell" is a result an
    operator needs and an exception is a result they never see.
    """


@dataclass(frozen=True)
class BandSeries:
    """One band's observations, with the band definition that gives it a frequency.

    The band is required rather than inferred because frequency is what makes
    the rain-fade test possible, and :class:`SpectrumObservation` does not carry
    it. Observations belonging to another band are a wiring error and raise.
    """

    band: SpectrumBand
    observations: tuple[SpectrumObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        wrong = {
            observation.band_id
            for observation in self.observations
            if observation.band_id != self.band.band_id
        }
        if wrong:
            raise PropagationError(
                f"observations for {sorted(wrong)} were supplied under band "
                f"{self.band.band_id!r}"
            )


def centre_frequency_hz(band: SpectrumBand) -> float:
    """The band's representative frequency, as a geometric mean.

    Geometric rather than arithmetic because the dependence being tested is a
    power law in frequency, so a band's leverage on a log axis is set by its
    geometric centre. For a narrow band the two agree to well under a percent;
    for a wide one the arithmetic centre would overstate the band's frequency
    and flatten a slope that is really there.
    """
    return math.sqrt(float(band.freq_low_hz) * float(band.freq_high_hz))


def fade_slope_db_per_decade(
    points: Sequence[tuple[float, float]],
    *,
    min_span_decades: float = MIN_FREQUENCY_SPAN_DECADES,
) -> tuple[float | None, str | None]:
    """Least-squares slope of fade depth against ``log10`` frequency.

    ``points`` are ``(centre_frequency_hz, fade_db)`` pairs, fade positive for a
    deeper fade. Returns ``(slope, None)`` or ``(None, reason)``.

    Refuses on fewer than two bands, and on bands whose frequencies sit too
    close together: dividing a decibel difference by a small log span
    manufactures an enormous slope out of ordinary calibration scatter, which is
    exactly how a frequency-independent measurement error becomes a confident
    claim about weather.
    """
    usable = [
        (math.log10(frequency), depth)
        for frequency, depth in points
        if frequency > 0.0 and math.isfinite(depth)
    ]
    if len(usable) < 2:
        return None, "FEWER_THAN_TWO_BANDS_MEASURED_SO_FREQUENCY_DEPENDENCE_WAS_NOT_TESTED"
    decades = [x for x, _ in usable]
    if max(decades) - min(decades) < min_span_decades:
        return None, "BANDS_SPAN_TOO_LITTLE_FREQUENCY_TO_RESOLVE_A_SLOPE"
    mean_x = statistics.fmean(decades)
    mean_y = statistics.fmean(depth for _, depth in usable)
    variance = sum((x - mean_x) ** 2 for x in decades)
    if variance <= 0.0:
        return None, "BANDS_SPAN_TOO_LITTLE_FREQUENCY_TO_RESOLVE_A_SLOPE"
    covariance = sum((x - mean_x) * (depth - mean_y) for x, depth in usable)
    return covariance / variance, None


def _rounded(value: float | None, digits: int = 3) -> float | None:
    """Round for serialisation, mapping anything non-finite to ``None``.

    RFC 8259 has no literal for infinity or NaN, so emitting one makes the whole
    document unparseable to a strict consumer. A quantity that was not measured
    is ``null``, which is what it means.
    """
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


@dataclass(frozen=True)
class PropagationVerdict:
    """What the two windows said, and how far it can honestly be pushed.

    ``discriminated`` is true only when a single cause was established, which
    includes establishing that nothing degraded. It is false for every refusal:
    too little evidence, an unusable baseline, and the concurrent and ambiguous
    cases where more than one cause fits the same measurement.
    """

    band_id: str
    centre_frequency_hz: float
    cause: str
    mechanism: str
    discriminated: bool
    samples: int
    baseline_samples: int
    recent_samples: int
    noise_floor_shift_db: float | None
    received_power_shift_db: float | None
    baseline_noise_spread_db: float | None
    baseline_power_spread_db: float | None
    onset_seconds: float | None
    step_concentration: float | None
    fade_slope_db_per_decade: float | None
    platform_displacement_m: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-propagation-verdict/v1",
            "band_id": self.band_id,
            "centre_frequency_hz": _rounded(self.centre_frequency_hz, 1),
            "cause": self.cause,
            "mechanism": self.mechanism,
            "discriminated": self.discriminated,
            "samples": self.samples,
            "baseline_samples": self.baseline_samples,
            "recent_samples": self.recent_samples,
            "noise_floor_shift_db": _rounded(self.noise_floor_shift_db),
            "received_power_shift_db": _rounded(self.received_power_shift_db),
            "baseline_noise_spread_db": _rounded(self.baseline_noise_spread_db),
            "baseline_power_spread_db": _rounded(self.baseline_power_spread_db),
            "onset_seconds": _rounded(self.onset_seconds, 1),
            "step_concentration": _rounded(self.step_concentration, 4),
            "fade_slope_db_per_decade": _rounded(self.fade_slope_db_per_decade, 3),
            "platform_displacement_m": _rounded(self.platform_displacement_m, 2),
            # Held at ``null`` structurally. A measurement establishes neither.
            "source": None,
            "intent": None,
            "reasons": list(self.reasons) + list(_STANDING_CAVEATS),
        }


@dataclass(frozen=True)
class _Measurement:
    """The two windows reduced to numbers, before any cause is assigned."""

    band_id: str
    centre_frequency_hz: float
    samples: int
    baseline_samples: int
    recent_samples: int
    noise_shift_db: float
    noise_spread_db: float
    power_shift_db: float | None
    power_spread_db: float | None
    noise_rose: bool
    noise_fell: bool
    power_rose: bool
    power_fell: bool
    onset_seconds: float | None
    step_concentration: float | None
    transition_start: datetime
    transition_end: datetime
    caveats: tuple[str, ...]

    @property
    def fade_db(self) -> float:
        """Depth of the fade in dB, positive, or zero if power did not fall."""
        if self.power_shift_db is None or self.power_shift_db >= 0.0:
            return 0.0
        return -self.power_shift_db


class PropagationDiscriminator:
    """Separates attenuation from arriving energy, or declines to.

    Every threshold is a constructor argument so a deployment with a better
    characterised front end can tighten them, but the defaults are the ones
    justified in this module's constants and are what the tests exercise.
    """

    def __init__(
        self,
        *,
        min_shift_db: float = MIN_SHIFT_DB,
        min_shift_sigma: float = MIN_SHIFT_SIGMA,
        max_baseline_spread_db: float = MAX_BASELINE_SPREAD_DB,
        min_baseline_samples: int = MIN_BASELINE_SAMPLES,
        min_recent_samples: int = MIN_RECENT_SAMPLES,
        max_sample_gap_seconds: float = MAX_SAMPLE_GAP_SECONDS,
        min_observation_confidence: float = MIN_OBSERVATION_CONFIDENCE,
        min_slow_onset_seconds: float = MIN_SLOW_ONSET_SECONDS,
        rain_fade_min_frequency_hz: float = RAIN_FADE_MIN_FREQUENCY_HZ,
        rain_fade_min_slope_db_per_decade: float = RAIN_FADE_MIN_SLOPE_DB_PER_DECADE,
        static_platform_displacement_m: float = STATIC_PLATFORM_DISPLACEMENT_M,
    ) -> None:
        self.min_shift_db = bounded_number(min_shift_db, "min_shift_db", 0.1, 100.0)
        self.min_shift_sigma = bounded_number(min_shift_sigma, "min_shift_sigma", 0.0, 20.0)
        self.max_baseline_spread_db = bounded_number(
            max_baseline_spread_db, "max_baseline_spread_db", 0.1, 100.0
        )
        self.min_baseline_samples = positive_int(
            min_baseline_samples, "min_baseline_samples", minimum=2
        )
        self.min_recent_samples = positive_int(min_recent_samples, "min_recent_samples", minimum=2)
        self.max_sample_gap_seconds = bounded_number(
            max_sample_gap_seconds, "max_sample_gap_seconds", 1.0, 86_400.0
        )
        self.min_observation_confidence = bounded_number(
            min_observation_confidence, "min_observation_confidence", 0.0, 1.0
        )
        self.min_slow_onset_seconds = bounded_number(
            min_slow_onset_seconds, "min_slow_onset_seconds", 1.0, 86_400.0
        )
        self.rain_fade_min_frequency_hz = bounded_number(
            rain_fade_min_frequency_hz, "rain_fade_min_frequency_hz", 1.0, 1e15
        )
        self.rain_fade_min_slope_db_per_decade = bounded_number(
            rain_fade_min_slope_db_per_decade, "rain_fade_min_slope_db_per_decade", 0.0, 1000.0
        )
        self.static_platform_displacement_m = bounded_number(
            static_platform_displacement_m, "static_platform_displacement_m", 0.1, 100_000.0
        )

    # -- public API ---------------------------------------------------------

    def discriminate(
        self,
        series: BandSeries,
        *,
        platform_track: Sequence[Fix] | None = None,
        recent_samples: int | None = None,
        slope_db_per_decade: float | None = None,
        slope_refusal: str | None = None,
    ) -> PropagationVerdict:
        """Attribute one band's degradation to a cause, or refuse.

        Refuses — returning :data:`CAUSE_INSUFFICIENT_EVIDENCE` — when the
        series is too short to split, when every reading fell below the
        confidence floor, when timestamps repeat, when an unobserved gap makes
        the onset unreadable, when the reference window is too variable or is
        not self-consistent, and when no received-power series exists to test
        attenuation against. Refuses to attribute — :data:`CAUSE_CONCURRENT` or
        :data:`CAUSE_AMBIGUOUS` — when more than one cause fits.

        ``slope_db_per_decade`` is supplied by :meth:`discriminate_bands`; a
        single band cannot test frequency dependence on its own, and passing a
        slope measured elsewhere is how the multi-band test reaches the
        per-band mechanism.
        """
        outcome = self._measure(series, recent_samples=recent_samples)
        if isinstance(outcome, PropagationVerdict):
            return outcome
        return self._classify(
            outcome,
            platform_track=platform_track,
            slope_db_per_decade=slope_db_per_decade,
            slope_refusal=slope_refusal,
            receiver_wide_noise_rise=False,
        )

    def discriminate_bands(
        self,
        series: Sequence[BandSeries],
        *,
        platform_track: Sequence[Fix] | None = None,
        recent_samples: int | None = None,
    ) -> tuple[PropagationVerdict, ...]:
        """Attribute several simultaneously observed bands, sharing what only they can see.

        Two tests exist only across bands and are folded into each verdict:

        **Frequency dependence.** A fade that deepens with frequency is the
        hydrometeor signature. One band cannot show it.

        **Receiver-wide rise.** The same noise-floor rise in three or more
        bands spread across frequency is as well explained by the receiver as by
        anything outside it, so the verdict becomes ambiguous rather than
        confidently blaming the environment.

        Bands that individually refuse still appear in the output, in input
        order, carrying their own reasons. Dropping them would let a caller
        mistake "four bands, three measurable" for "four bands measured".
        """
        measured: list[tuple[int, _Measurement]] = []
        refusals: dict[int, PropagationVerdict] = {}
        for index, item in enumerate(series):
            outcome = self._measure(item, recent_samples=recent_samples)
            if isinstance(outcome, PropagationVerdict):
                refusals[index] = outcome
            else:
                measured.append((index, outcome))

        slope, slope_refusal = fade_slope_db_per_decade(
            [(m.centre_frequency_hz, m.fade_db) for _, m in measured if m.power_shift_db is not None]
        )
        receiver_wide = self._receiver_wide_rise([m for _, m in measured])

        verdicts: list[PropagationVerdict] = []
        by_index = {
            index: self._classify(
                measurement,
                platform_track=platform_track,
                slope_db_per_decade=slope,
                slope_refusal=slope_refusal,
                receiver_wide_noise_rise=receiver_wide,
            )
            for index, measurement in measured
        }
        by_index.update(refusals)
        for index in range(len(series)):
            verdicts.append(by_index[index])
        return tuple(verdicts)

    # -- measurement --------------------------------------------------------

    def _measure(
        self, series: BandSeries, *, recent_samples: int | None
    ) -> _Measurement | PropagationVerdict:
        """Reduce a series to numbers, or to the verdict explaining why not.

        Returns either kind, never both and never neither, so no caller has to
        decide which of a pair to trust.
        """
        band_id = series.band.band_id
        centre = centre_frequency_hz(series.band)
        caveats: list[str] = []

        usable = [
            observation
            for observation in series.observations
            if observation.confidence() >= self.min_observation_confidence
        ]
        dropped = len(series.observations) - len(usable)
        if dropped:
            # Stated rather than silently absorbed: a series that lost half its
            # readings to a failing front end is thinner evidence than its
            # sample count alone suggests.
            caveats.append("OBSERVATIONS_BELOW_THE_CONFIDENCE_FLOOR_WERE_DROPPED")
        if not usable:
            return self._refuse(
                band_id, centre, 0, ("NO_OBSERVATION_CLEARED_THE_CONFIDENCE_FLOOR",) + tuple(caveats)
            )

        ordered = sorted(usable, key=lambda observation: observation.observed_at)
        minimum = self.min_baseline_samples + self.min_recent_samples
        if len(ordered) < minimum:
            return self._refuse(
                band_id,
                centre,
                len(ordered),
                (
                    f"NEED_AT_LEAST_{minimum}_USABLE_OBSERVATIONS_TO_SPLIT_A_BASELINE"
                    "_FROM_A_RECENT_WINDOW",
                )
                + tuple(caveats),
            )

        gaps = [
            (later.observed_at - earlier.observed_at).total_seconds()
            for earlier, later in zip(ordered, ordered[1:])
        ]
        if any(gap <= 0.0 for gap in gaps):
            # Two readings claiming the same instant disagree about a fact only
            # one of them can hold. Averaging them would be inventing a third.
            return self._refuse(
                band_id,
                centre,
                len(ordered),
                ("SERIES_CONTAINS_DUPLICATE_TIMESTAMPS_AND_NO_RULE_PICKS_A_WINNER",)
                + tuple(caveats),
            )
        if any(gap > self.max_sample_gap_seconds for gap in gaps):
            return self._refuse(
                band_id,
                centre,
                len(ordered),
                ("SERIES_HAS_AN_UNOBSERVED_INTERVAL_TOO_LONG_TO_BRIDGE",) + tuple(caveats),
            )

        recent_count = self._recent_count(len(ordered), recent_samples)
        baseline = ordered[: len(ordered) - recent_count]
        recent = ordered[len(ordered) - recent_count :]

        # Received power is used only when *every* usable reading carries it.
        # A partial series is worse than none: an adapter that omits the peak
        # when it cannot find one omits it exactly when the signal is weakest,
        # so averaging the readings that survive biases the answer towards "no
        # drop" precisely in the case the module exists to catch.
        missing_power = sum(1 for observation in ordered if observation.peak_power_dbm is None)
        power_available = missing_power == 0
        if missing_power == len(ordered):
            caveats.append("NO_OBSERVATION_CARRIED_A_RECEIVED_POWER_FIGURE")
        elif missing_power:
            caveats.append(
                "RECEIVED_POWER_MISSING_FROM_SOME_OBSERVATIONS_SO_THE_WHOLE_SERIES_IS_UNUSED"
            )

        noise_shift, noise_spread, noise_refusal = self._channel(
            [observation.noise_floor_dbm for observation in baseline],
            [observation.noise_floor_dbm for observation in recent],
            "NOISE_FLOOR",
        )
        if noise_refusal is not None:
            return self._refuse(
                band_id, centre, len(ordered), (noise_refusal,) + tuple(caveats)
            )

        power_shift: float | None = None
        power_spread: float | None = None
        if power_available:
            power_shift, power_spread, power_refusal = self._channel(
                [observation.peak_power_dbm for observation in baseline],
                [observation.peak_power_dbm for observation in recent],
                "RECEIVED_POWER",
            )
            if power_refusal is not None:
                return self._refuse(
                    band_id, centre, len(ordered), (power_refusal,) + tuple(caveats)
                )

        noise_rose = self._moved(noise_shift, noise_spread, rising=True)
        noise_fell = self._moved(noise_shift, noise_spread, rising=False)
        power_rose = (
            self._moved(power_shift, power_spread, rising=True)
            if power_shift is not None and power_spread is not None
            else False
        )
        power_fell = (
            self._moved(power_shift, power_spread, rising=False)
            if power_shift is not None and power_spread is not None
            else False
        )

        # Nothing can be said about attenuation without a power series, and a
        # quiet noise floor is not evidence that the path is clear.
        if not power_available and not noise_rose:
            return self._refuse(
                band_id,
                centre,
                len(ordered),
                ("NO_RECEIVED_POWER_SERIES_SO_ATTENUATION_COULD_NOT_BE_TESTED",) + tuple(caveats),
            )

        onset_seconds: float | None = None
        concentration: float | None = None
        if power_fell and power_shift is not None:
            onset_seconds, concentration = self._onset(
                baseline,
                recent,
                statistics.fmean(
                    observation.peak_power_dbm
                    for observation in baseline
                    if observation.peak_power_dbm is not None
                ),
                -power_shift,
            )

        lead = baseline[-TRANSITION_LEAD_SAMPLES:] if baseline else []
        transition_start = lead[0].observed_at if lead else recent[0].observed_at
        return _Measurement(
            band_id=band_id,
            centre_frequency_hz=centre,
            samples=len(ordered),
            baseline_samples=len(baseline),
            recent_samples=len(recent),
            noise_shift_db=noise_shift,
            noise_spread_db=noise_spread,
            power_shift_db=power_shift,
            power_spread_db=power_spread,
            noise_rose=noise_rose,
            noise_fell=noise_fell,
            power_rose=power_rose,
            power_fell=power_fell,
            onset_seconds=onset_seconds,
            step_concentration=concentration,
            transition_start=transition_start,
            transition_end=recent[-1].observed_at,
            caveats=tuple(caveats),
        )

    def _recent_count(self, total: int, requested: int | None) -> int:
        """Size of the window under test, defaulting to the last third.

        A third keeps the reference window at least twice the size of the
        window being judged against it, so the comparison is never a coin toss
        between two equally thin halves.
        """
        if requested is None:
            return max(self.min_recent_samples, total // 3)
        count = positive_int(requested, "recent_samples", minimum=self.min_recent_samples)
        if total - count < self.min_baseline_samples:
            raise PropagationError(
                f"recent_samples={count} leaves fewer than {self.min_baseline_samples} "
                f"baseline observations out of {total}"
            )
        return count

    def _channel(
        self, baseline: Sequence[float], recent: Sequence[float], label: str
    ) -> tuple[float, float, str | None]:
        """Reduce one measured channel to ``(shift, baseline spread, refusal)``.

        The spread is the *sample* standard deviation, not the population one:
        understating the reference window's own scatter would make the sigma
        gate easier to clear, which is the wrong direction for a module whose
        job is declining to over-claim.
        """
        spread = statistics.stdev(baseline) if len(baseline) >= 2 else 0.0
        if spread > self.max_baseline_spread_db:
            return 0.0, spread, f"BASELINE_{label}_IS_TOO_VARIABLE_TO_SERVE_AS_A_BASELINE"
        half = len(baseline) // 2
        if half >= 1:
            divergence = abs(statistics.fmean(baseline[:half]) - statistics.fmean(baseline[half:]))
            if divergence > self.min_shift_db:
                # The reference window already contains the event, so comparing
                # against it would measure the tail of a change against its own
                # head and call the result quiet.
                return (
                    0.0,
                    spread,
                    f"BASELINE_{label}_IS_NOT_STABLE_ITS_TWO_HALVES_DISAGREE",
                )
        return statistics.fmean(recent) - statistics.fmean(baseline), spread, None

    def _moved(self, shift: float | None, spread: float | None, *, rising: bool) -> bool:
        """True when a shift clears both the absolute floor and the sigma gate."""
        if shift is None or spread is None:
            return False
        magnitude = shift if rising else -shift
        if magnitude < self.min_shift_db:
            return False
        effective = max(spread, MIN_EFFECTIVE_SPREAD_DB)
        return magnitude >= self.min_shift_sigma * effective

    def _onset(
        self,
        baseline: Sequence[SpectrumObservation],
        recent: Sequence[SpectrumObservation],
        baseline_mean: float,
        fade_db: float,
    ) -> tuple[float | None, float | None]:
        """Ten-to-ninety-percent onset time, and how much of the fade fell in one interval.

        Returns ``(None, None)`` when the fade never crosses the ninety-percent
        level inside the observed window, or when it had already crossed it
        before the window opened: in both cases the onset that would be reported
        is the length of the window rather than the length of the fade.
        """
        if fade_db <= 0.0:
            return None, None
        lead = list(baseline[-TRANSITION_LEAD_SAMPLES:])
        transition = lead + list(recent)
        values = [
            observation.peak_power_dbm
            for observation in transition
            if observation.peak_power_dbm is not None
        ]
        if len(values) != len(transition) or len(values) < 2:
            return None, None
        times = [observation.observed_at for observation in transition]

        high = baseline_mean - 0.1 * fade_db
        low = baseline_mean - 0.9 * fade_db
        end = next((index for index, value in enumerate(values) if value <= low), None)
        if end is None or end == 0:
            return None, None
        start = 0
        for index in range(end):
            if values[index] >= high:
                start = index
        onset = (times[end] - times[start]).total_seconds()

        drops = [values[index] - values[index + 1] for index in range(len(values) - 1)]
        concentration = max(0.0, min(1.0, max(drops) / fade_db)) if drops else None
        return onset, concentration

    def _receiver_wide_rise(self, measurements: Sequence[_Measurement]) -> bool:
        """True when every analysed band rose by the same amount across frequency."""
        if len(measurements) < RECEIVER_WIDE_RISE_MIN_BANDS:
            return False
        if not all(measurement.noise_rose for measurement in measurements):
            return False
        rises = [measurement.noise_shift_db for measurement in measurements]
        if max(rises) - min(rises) > self.min_shift_db:
            return False
        decades = [math.log10(measurement.centre_frequency_hz) for measurement in measurements]
        return max(decades) - min(decades) >= MIN_FREQUENCY_SPAN_DECADES

    # -- classification -----------------------------------------------------

    def _classify(
        self,
        measurement: _Measurement,
        *,
        platform_track: Sequence[Fix] | None,
        slope_db_per_decade: float | None,
        slope_refusal: str | None,
        receiver_wide_noise_rise: bool,
    ) -> PropagationVerdict:
        reasons: list[str] = list(measurement.caveats)
        displacement: float | None = None
        displacement_refusal: str | None = None
        mechanism = MECHANISM_NOT_APPLICABLE

        if measurement.power_fell and measurement.noise_rose:
            cause = CAUSE_CONCURRENT
            discriminated = False
            reasons.append("RECEIVED_POWER_FELL_AND_THE_NOISE_FLOOR_ROSE_IN_THE_SAME_WINDOW")
            reasons.append("EITHER_TWO_CAUSES_AT_ONCE_OR_ONE_CAUSE_THIS_MODULE_CANNOT_SEPARATE")
            # The single-cause reading that reproduces both signatures at once,
            # named so nobody has to rediscover it under pressure.
            reasons.append(
                "IN_BAND_ENERGY_STRONG_ENOUGH_TO_COMPRESS_A_RECEIVER_FRONT_END"
                "_REPRODUCES_BOTH_SIGNATURES"
            )
        elif measurement.power_fell and measurement.noise_fell:
            cause = CAUSE_AMBIGUOUS
            discriminated = False
            reasons.append(
                "RECEIVED_POWER_AND_NOISE_FLOOR_FELL_TOGETHER_WHICH_IS_A_GAIN_CHANGE"
                "_NOT_A_PATH_CHANGE"
            )
        elif measurement.power_fell:
            cause = CAUSE_PATH_ATTENUATION
            discriminated = True
            reasons.append("RECEIVED_POWER_FELL_WHILE_THE_NOISE_FLOOR_HELD")
            mechanism, mechanism_reasons, displacement, displacement_refusal = self._mechanism(
                measurement,
                platform_track=platform_track,
                slope_db_per_decade=slope_db_per_decade,
                slope_refusal=slope_refusal,
            )
            reasons.extend(mechanism_reasons)
        elif measurement.noise_rose and receiver_wide_noise_rise:
            cause = CAUSE_AMBIGUOUS
            discriminated = False
            reasons.append(
                "THE_SAME_NOISE_FLOOR_RISE_APPEARED_IN_EVERY_BAND_A_RECEIVER_GAIN"
                "_CHANGE_REPRODUCES_THIS"
            )
        elif measurement.noise_rose:
            cause = CAUSE_ENERGY_IN_BAND
            discriminated = True
            if measurement.power_shift_db is None:
                # The floor rising is positive evidence on its own — nothing on
                # the path can add energy to the band — but whether attenuation
                # is *also* happening was not tested. Saying "the noise floor
                # rose while received power held" here would be reporting a
                # comparison that never ran as one that came out clean.
                reasons.append("NOISE_FLOOR_ROSE_AND_NO_RECEIVED_POWER_FIGURE_EXISTED_TO_COMPARE")
                reasons.append("RECEIVED_POWER_UNAVAILABLE_SO_CONCURRENT_ATTENUATION_WAS_NOT_TESTED")
            elif measurement.power_rose:
                reasons.append("NOISE_FLOOR_ROSE_AND_RECEIVED_POWER_ROSE_WITH_IT")
            else:
                reasons.append("NOISE_FLOOR_ROSE_WHILE_RECEIVED_POWER_HELD")
        else:
            cause = CAUSE_NO_DEGRADATION
            discriminated = True
            if measurement.noise_fell:
                reasons.append("NOISE_FLOOR_FELL_WHICH_IS_AN_IMPROVEMENT_NOT_A_DEGRADATION")
            if measurement.power_rose:
                reasons.append("RECEIVED_POWER_ROSE_WHICH_IS_AN_IMPROVEMENT_NOT_A_DEGRADATION")
            if not measurement.noise_fell and not measurement.power_rose:
                reasons.append("NEITHER_CHANNEL_MOVED_BEYOND_THE_MEASUREMENT_FLOOR")

        if displacement_refusal is not None:
            reasons.append(displacement_refusal)

        return PropagationVerdict(
            band_id=measurement.band_id,
            centre_frequency_hz=measurement.centre_frequency_hz,
            cause=cause,
            mechanism=mechanism,
            discriminated=discriminated,
            samples=measurement.samples,
            baseline_samples=measurement.baseline_samples,
            recent_samples=measurement.recent_samples,
            noise_floor_shift_db=measurement.noise_shift_db,
            received_power_shift_db=measurement.power_shift_db,
            baseline_noise_spread_db=measurement.noise_spread_db,
            baseline_power_spread_db=measurement.power_spread_db,
            onset_seconds=measurement.onset_seconds,
            step_concentration=measurement.step_concentration,
            fade_slope_db_per_decade=slope_db_per_decade,
            platform_displacement_m=displacement,
            reasons=_deduplicate(reasons),
        )

    def _mechanism(
        self,
        measurement: _Measurement,
        *,
        platform_track: Sequence[Fix] | None,
        slope_db_per_decade: float | None,
        slope_refusal: str | None,
    ) -> tuple[str, list[str], float | None, str | None]:
        """Shape the fade, once attenuation is established.

        A step is an obstruction: the path was clear and then it was not, inside
        one sample interval. A slow ramp is only called a frequency-selective
        fade when everything that could reproduce it has been excluded — the
        band is high enough for hydrometeors to matter, the fade deepens with
        frequency, and the platform held station so a diffracting edge cannot
        be the explanation. Anything short of all four leaves the mechanism
        undetermined and says which test was not passed.
        """
        reasons: list[str] = []
        displacement, displacement_refusal = self._displacement(
            platform_track, measurement.transition_start, measurement.transition_end
        )

        if measurement.step_concentration is None or measurement.onset_seconds is None:
            reasons.append("FADE_ONSET_COULD_NOT_BE_TIMED_WITHIN_THE_OBSERVED_WINDOW")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        if measurement.step_concentration >= STEP_CONCENTRATION_RATIO:
            reasons.append("FADE_ARRIVED_AS_A_STEP_WITHIN_ONE_SAMPLE_INTERVAL")
            if displacement is None:
                reasons.append("PLATFORM_MOVEMENT_CORRELATION_WAS_NOT_TESTED")
            elif displacement > self.static_platform_displacement_m:
                reasons.append("STEP_COINCIDES_WITH_PLATFORM_MOVEMENT")
            else:
                # A structure or vehicle crossing a stationary path produces the
                # same step, and the distinction matters: one clears by moving,
                # the other clears by waiting.
                reasons.append("PLATFORM_HELD_STATION_SO_SOMETHING_ELSE_ENTERED_THE_PATH")
            return MECHANISM_ABRUPT_OBSTRUCTION, reasons, displacement, displacement_refusal

        if measurement.onset_seconds < self.min_slow_onset_seconds:
            reasons.append("FADE_ONSET_TOO_FAST_FOR_A_HYDROMETEOR_EVENT_AND_TOO_GRADUAL_FOR_A_STEP")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        if measurement.centre_frequency_hz < self.rain_fade_min_frequency_hz:
            reasons.append("BAND_IS_BELOW_THE_FREQUENCY_AT_WHICH_HYDROMETEOR_ATTENUATION_IS_MEASURABLE")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        if slope_db_per_decade is None:
            reasons.append(slope_refusal or "FREQUENCY_DEPENDENCE_WAS_NOT_TESTED")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        if slope_db_per_decade < self.rain_fade_min_slope_db_per_decade:
            reasons.append("FADE_IS_NOT_FREQUENCY_SELECTIVE_ENOUGH_FOR_HYDROMETEOR_ATTENUATION")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        if displacement is None:
            reasons.append("PATH_GEOMETRY_UNCONFIRMED_SO_A_DIFFRACTING_EDGE_CANNOT_BE_EXCLUDED")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        if displacement > self.static_platform_displacement_m:
            # Knife-edge diffraction loss also rises with frequency, so on a
            # moving platform the slope proves nothing about the sky.
            reasons.append("PLATFORM_MOVED_DURING_THE_FADE_SO_A_DIFFRACTING_EDGE_CANNOT_BE_EXCLUDED")
            return MECHANISM_UNDETERMINED, reasons, displacement, displacement_refusal

        reasons.append("SLOW_FREQUENCY_SELECTIVE_FADE_WITH_THE_PATH_GEOMETRY_HELD_STILL")
        reasons.append(
            "CONSISTENT_WITH_HYDROMETEOR_ATTENUATION_PER_ITU_R_P838"
            "_WHICH_IS_NOT_A_WEATHER_MEASUREMENT"
        )
        return MECHANISM_FREQUENCY_SELECTIVE_FADE, reasons, displacement, displacement_refusal

    def _displacement(
        self, track: Sequence[Fix] | None, start: datetime, end: datetime
    ) -> tuple[float | None, str | None]:
        """How far the platform moved across the transition, if that is knowable.

        Widest separation between any two fixes in the interval, not first-to-
        last: a platform that leaves and returns has changed the path geometry
        in between, and an endpoint comparison would report that it never moved.

        Returns ``None`` with a reason when the track does not cover the
        interval, and when the fixes' own uncertainty is looser than the
        stillness threshold — a fix that could be anywhere within ten metres
        cannot establish that the platform stayed within six.
        """
        if not track:
            return None, "PLATFORM_TRACK_ABSENT_SO_GEOMETRY_COULD_NOT_BE_CHECKED"
        covering = [fix for fix in track if start <= fix.observed_at <= end]
        if len(covering) < 2:
            return None, "PLATFORM_TRACK_DOES_NOT_COVER_THE_TRANSITION_INTERVAL"
        displacement = max(
            metres_between(
                first.latitude_deg, first.longitude_deg, second.latitude_deg, second.longitude_deg
            )
            for index, first in enumerate(covering)
            for second in covering[index + 1 :]
        )
        if displacement <= self.static_platform_displacement_m:
            worst = max(fix.uncertainty_radius_m for fix in covering)
            if worst > self.static_platform_displacement_m:
                return None, "PLATFORM_FIX_UNCERTAINTY_EXCEEDS_THE_STILLNESS_THRESHOLD"
        return displacement, None

    # -- refusals -----------------------------------------------------------

    def _refuse(
        self, band_id: str, centre: float, samples: int, reasons: Sequence[str]
    ) -> PropagationVerdict:
        """A verdict that carries why no verdict was reached.

        Every measured field is ``None`` rather than a default: a zero shift
        would read as "measured, and nothing moved", which is the one thing a
        refusal must never be mistaken for.
        """
        return PropagationVerdict(
            band_id=band_id,
            centre_frequency_hz=centre,
            cause=CAUSE_INSUFFICIENT_EVIDENCE,
            mechanism=MECHANISM_NOT_APPLICABLE,
            discriminated=False,
            samples=samples,
            baseline_samples=0,
            recent_samples=0,
            noise_floor_shift_db=None,
            received_power_shift_db=None,
            baseline_noise_spread_db=None,
            baseline_power_spread_db=None,
            onset_seconds=None,
            step_concentration=None,
            fade_slope_db_per_decade=None,
            platform_displacement_m=None,
            reasons=_deduplicate(reasons),
        )


def _deduplicate(reasons: Sequence[str]) -> tuple[str, ...]:
    """Preserve order while removing repeats, so a reason reads once."""
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            ordered.append(reason)
    return tuple(ordered)
