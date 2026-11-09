# Numerical field controls

The primary `radiomap_operator` control predicts the same reference-gauge
reported RSSI as the neural field. It sees the same public CAD boxes, AP/RX
positions, carriers, known transmit power and estimated frontend calibration.
Its global prior uses all permitted reports in the 48 TRAIN scenes: bands
2.4/5.2 GHz, both directions, and devices 0–2. It never fits a 6 GHz report,
device 3 report, development query or field query. At inference it conditions
on A's 24 reports only: both observed bands, two APs, six even poses,
direction 0, device 0. It does not pool A with B.

The baseline subtracts the exact per-link free-space operator and the common
TRAIN paired-report frontend estimate before learning a residual field. The
global residual prior is a Gaussian-prior linear model with quantization
interval likelihood and left-censored Gaussian likelihood. Its regressors are
the eight shared physics bases, the ten propagation-only geometry columns
0–7/9/10, and 48 fixed random Fourier features over AP-relative physical XY.
Normalization is fitted on TRAIN only, and constant regressors retain scale
one. Direction, receiver noise floor, quantization width and calibration
status cannot enter the physical mean field. Unknown calibration widens the
reported predictive spread instead.

Local conditioning calls the unchanged native
`wirelessjev.native.radiomap.RadioMap._fit_one` implementation through an
explicit feature-matrix adapter, separately for each AP and observed carrier.
The adapter replaces the original warehouse-specific rack regressors with
these public CAD/spectral regressors. This is a declared geometry/spectral
adaptation of RadioMap, not a claim that its old fixed warehouse coordinates
describe a new scene. It retains the native Tobit refinement and censor
weight 5. Global learning respects reported quantization intervals; the
native local fitter consumes their report centers plus actual censor bounds.
Targets and coefficient priors are rescaled together so native measurement
noise units equal the working noise inferred from TRAIN same-link calibrated
co-report spread, after subtracting quantization variance. That conditioning
noise is distinct from the larger TRAIN field-discrepancy estimate.

Both observed carriers contribute to a query using interpolation or
extrapolation in log frequency. The shared known free-space operator and
TRAIN-learned frequency/geometry coefficients remain available at 6 GHz;
RadioMap is never handicapped with an empty isolated 6 GHz map. Absolute
spectral weights combine local posterior standard deviations without an
independence assumption. The final working spread also includes TRAIN field
discrepancy. These spreads have no production calibration or risk guarantee.

The descriptive `bayesian_operator` control uses only the eight shared
physics bases, with the same global interval/censor likelihood, TRAIN data,
calibration and A-only local adaptation. It is a separately named numerical
algorithm. The `operator_only` control adds estimated frontend offset to
per-link free-space received power without a learned material correction.

The supplementary `direct_same_link_frequency` control is eligible only
when the query is genuinely co-located with a heard 5.2 GHz A report from the
same AP and both frontend estimates are identified. It applies the known
20 log10(5.2/6) shift to that report. Reverse-direction use explicitly assumes
calibrated reciprocity; fixed antenna/material response across carriers is
also an assumption. Its quantization-only spread omits material dispersion.
Unavailable pairs return NaN and an explicit false eligibility mask. Its
paired subset and sample count must be reported separately from full-cohort
Map/model comparisons.

`prepare()` binds the TRAIN prior archive to the published source receipt,
protocol and common frontend receipt. `load_prior()` runs outside timed
adaptation/prediction. The fast path reads only query geometry, known FSPL
and estimated offset; tests use query objects that raise on RF label access.
`verify_prior()` checks the stored identities and fitting scope passively,
without a new fit or model forward pass. Analytic software tests validate
this machinery; they are not evidence of wireless accuracy or real hardware.
