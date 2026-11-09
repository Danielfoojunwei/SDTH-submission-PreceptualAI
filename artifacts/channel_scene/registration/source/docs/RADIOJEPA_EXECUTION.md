# RadioJEPA execution

The [user plan](plans/RADIOJEPA_USER_PLAN.md) supersedes the unlaunched temporal
observation-frontier experiment. Its source remains available, but no training
or confirmation from that campaign is being launched. Completed prior evidence
is preserved. RadioMap remains the application default and fallback.

The implementation separates three responsibilities:

- **RadioJEPA** learns radio-scene representations from observed physical views.
- **WirelessJEV** prices typed candidate actions and applies mission constraints.
- The independent gate qualifies individual heads and controls fallback.

Public data and dependency acquisition precede model experiments. Original
licenses, hashes, units, device identities, timestamps and split semantics are
recorded. Code and data licenses are reviewed separately. Missing rights or
clock alignment are blockers to a particular use, not grounds for fabricated
views or labels. No downloaded dataset is automatically warehouse service data.

The first implemented neural slice uses simultaneous receiver-array channel
power from measured DICHASUS records. It has receiver-set attention, a separate
absolute power coordinate, a projector for view alignment, and a relative-delay
CIR basis renderer. Antennas in each physical array are averaged in linear
power before taking dB: only array-center geometry is verified for every domain,
so this endpoint makes no per-element fading or angle-recovery claim.
It does not add an unbounded residual to RadioMap and uses
no EMA target or future-window pretext. The basis coefficients are not identified
scatterer locations. Reported channel coefficients have reference gains, so
they are not automatically dBm or normalized passive-port gains.

Independent runs compare simultaneous-view alignment, temporal alignment, a
matched measurement-regression control, an unstructured decoder, and the
none/SIGReg/VICReg ladder. PPR is implemented and analytically tested, but is
ineligible for the power-only trial because its phase/path factors are not
identified. No regularizer is presumed superior. Frozen linear probes, constant
and linear baselines, latent interventions and cross-view/run alignment test
whether the representation is useful. Effective rank alone is not a cure for
collapse or a measure of scene sufficiency.

Actual Sionna RT generates a separate multiband/reciprocal data track with
declared simulated front ends. Twenty-four confirmation geometries are reserved
as specifications; generating their measurements requires a separate gate.
Real paired RSSI/FTM and cross-device datasets have separate adapters and
eligibility records. Unpaired camera images or different-day phones are not
silently treated as simultaneous scenes.

The new planner interface reproduces existing RadioMap outage utility before
learning changes it. Its first proposal planner holds one action over multiple
horizons. Multi-step learned dynamics, qualified outage/silence distributions,
joint MLO/fleet/route control and physical execution are further requirements,
not features established by interface or unit tests.

The [physics audit](RADIOJEPA_PHYSICS_AUDIT.md) corrects several premises in the
proposal: simultaneity does not by itself satisfy the paper's identifiability
assumptions; reciprocity requires calibration and matching frequency/time;
kTB describes noise rather than a universal cross-band signal transform; and
conformal marginal coverage does not establish safety under arbitrary drift.
Simulator-truth training supervision is privileged supervision, and is leakage
only when prohibited future/evaluation information reaches inference or fitting.

The plan's bracketed paper results remain unfilled. A new world model may be
useful without winning every endpoint; promotion still requires the registered
service, tail, compute, calibration and raw-evidence checks under the same
planner. Current software tests and public-data experiments cannot establish
physical SmartCar qualification or global SOTA.
