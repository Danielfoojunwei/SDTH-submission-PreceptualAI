# Public RF factors and calibrated visual geometry

The visual-radio physics module separates two evidence paths: calibrated object
geometry can describe a potential obstruction; RF observations and configuration
describe received power, transmitter activity and interference. Neither an image
embedding nor a visible object identifies its dielectric properties, RF loss,
channel occupancy or an unseen transmitter's state. These helpers implement
established relationships with explicit assumptions. They do not establish a
learned improvement, novelty, SOTA or readiness for physical deployment.

## Typed inputs and reference planes

| Input | Meaning and units | Missing evidence |
|---|---|---|
| `ReceiverState` | Physical receiver identity, optional metric 3-D position, receiver passband and noise in mW | `None`; `observed=False` masks all numeric fields |
| `TransmitterState` | One physical radio identity, optional position, passband, transmit power in dBm and activity fraction | A missing activity or power is unknown, rather than zero |
| `RFCoupling` | Linear received/transmitted **power** ratio at a stated frequency | Invalid or missing gain remains unknown |
| `VisualSilhouette` | Optional calibrated metric centre and radius of a circular silhouette cross-section | An image mask without metric calibration cannot provide clearance |

Use a separate identity for each physical transmitter radio. Multiple graph
edges referring to the same radio are not additional transmitters. Equal
duplicate radio/measurement records are deduplicated; conflicting duplicates
and ambiguous compatible-frequency measurements are rejected. Measurements for
another receiver or frequency are not reused.

Transmitter power and coupling must use matching RF reference planes. For
example, adding antenna gain into both transmit EIRP and a coupling that already
contains that gain would count it twice. Measurements, activity and channel
configuration must also describe the same relevant time window. This numerical
helper does not infer freshness; the ingestion/fusion layer must enforce it.

`coupling_from_isolated_power_dbm` converts a calibrated, isolated received-power
measurement and matching transmit power to a linear ratio. Aggregate RSSI,
channel energy or SINR is not automatically such an isolated measurement.
Missing measurements stay unknown; masked NaN placeholders are ignored.

## Receiver-centric interference

For desired transmitter `s`, receiver `r`, and distinct interferers `j`, the
declared power model is

\[
S_r=P_sG_{sr}\eta_{sr},\qquad
\bar I_r=\sum_jP_jG_{jr}a_j\eta_{jr},\qquad
\gamma_r=\frac{S_r}{N_r+\bar I_r}.
\]

All sums and products use **linear mW**. Conversion from dBm happens before
addition. `G` is a dimensionless power coupling; `a` is an explicitly observed
or configured activity fraction in `[0,1]`. The desired signal is conditional
on an actual or hypothetical desired transmission, so its population duty cycle
does not attenuate that packet's power.

With binary activity and valid instantaneous measurements, this is the scalar
link-budget SINR model. With fractional activity, it is `S/(N+E[I])`; it is not
expected instantaneous SINR, a delivery probability or a calibrated outage
prediction. Incoherent power addition assumes cross terms average out. Coherent
joint transmissions require a complex-channel/waveform treatment instead.

The default spectral factor uses flat transmitter power spectral density and
an ideal rectangular receiver filter:

\[
\eta_{jr}=
\frac{\left|B_j\cap B_r\right|}{\left|B_j\right|}.
\]

The denominator is the **transmitter's** bandwidth. A 20-MHz receiver admits
half the total power of an aligned 40-MHz flat-spectrum transmitter. Actual
Wi-Fi/5G spectral masks, filter leakage, puncturing and subcarrier loading need
their own measurements/models; disjoint ideal rectangles do not prove zero
real adjacent-channel interference.

A measured gain already integrated through the receiver filter must set
`includes_receiver_filter=True` and specify **both** the transmitter and
receiver channels used for that measurement. The code then applies no second
overlap factor. Changing either bandwidth invalidates its applicability.
Such a measurement can represent adjacent-channel leakage that the ideal
rectangle would omit.

## Unknown interference is preserved

`receiver_link_budget` returns each transmitter's contribution and the sum of
known contributions. Its default `inventory_complete=False` preserves unknown
unlisted interferers. An explicitly listed transmitter with unknown coupling,
power or required activity also makes the total incomplete. These cases return
`total_interference_mw=None` and `sinr_linear=None`.

An observed inactive transmitter can contribute exactly zero without a known
coupling. That is different from unobserved activity. A declared complete empty
inventory produces zero modelled interference; the caller must justify that
completeness declaration with RF evidence or a controlled scenario.

When desired power and noise are known, the ratio using only known interference
is exposed as `sinr_upper_bound_linear`. It is an upper bound only if those
signal/noise values and the declared filter/coupling model are correct and
missing interference is nonnegative. It is not a real-world risk certificate
when measurements or propagation assumptions are wrong. No positive lower SINR
bound is invented for unbounded unknown interference.

Positions contribute public link distances. They do not automatically substitute
for an RF coupling. `free_space_coupling` requires the explicit
`assume_isotropic_far_field=True` declaration and labels the result
`origin='declared_free_space'`:

\[
G_{\rm FS}=\left(\frac{\lambda}{4\pi d}\right)^2,
\quad\lambda=c/f.
\]

This Friis reference assumes unobstructed isotropic far-field propagation. The
caller must check antenna aperture/far-field validity. It does not model
warehouse multipath, polarization, reflection, diffraction, absorption or an
object's material. Measured and assumed coupling origins remain distinguishable.

## Calibrated Fresnel geometry

For axial distances `d1` and `d2` between an obstruction plane and the two link
endpoints, the first Fresnel radius is

\[
r_1=\sqrt{\frac{\lambda d_1d_2}{d_1+d_2}}.
\]

`fresnel_silhouette_clearance` projects a calibrated silhouette centre onto the
finite line-of-sight segment. At that axial location, it compares perpendicular
distance `rho` with silhouette radius `R` and an explicitly requested fraction
`q` of the first Fresnel radius:

\[
\text{clearance}=\rho-R-q r_1,\qquad 0\le q\le1.
\]

This is a circular cross-section screen approximation at the projected centre,
rather than exact intersection of a 3-D object volume with a Fresnel ellipsoid.
The output distinguishes line-of-sight intersection from Fresnel intersection.
A screen plane beyond either endpoint does not intersect this finite segment.
Geometric intersection does not assign RF attenuation or assert packet loss.

Metric depth/scale, camera calibration, extrinsics and an agreed world coordinate
frame must come from calibrated sensors, multi-view reconstruction or known CAD.
DINO features alone provide none of those guarantees. Unobserved or uncalibrated
geometry produces an unknown result. Additional uncertain geometry requires its
own uncertainty treatment; a single deterministic clearance is not a confidence
interval. See [ITU-R P.526](https://www.itu.int/rec/R-REC-P.526/en) for established
diffraction/Fresnel context; this helper is not a full diffraction model.

## Verification scope

The bounded unit tests exercise linear power arithmetic, spectral overlap,
filter provenance, unknown/masked inputs, duplicate handling, radio permutation,
joint SE(2) rotation/translation, monotonic known SINR effects and Fresnel
geometry. They use deterministic arithmetic examples, rather than fabricated
hardware evidence. They neither change the frozen service simulator nor claim
an experiment on real RF traffic, vision accuracy or trained model performance.
