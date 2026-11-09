# Reported radio power to service decisions

This bridge connects an explicitly described RSSI forecast to the existing
prediction-only `WorldModel` interface and native packet simulator. RadioMap
remains the default model. A learned field or dynamic state is an opt-in
candidate using the same outcome adapter, mission compiler, planner and
independent admission gates. This document specifies the bridge and its
validation limits; it reports no experiment or hardware result.

## Inputs and physical reference

`RadioAssumptions` records the physical interpretation needed before power
can become SNR or SINR. `RSSIForecast` carries finite predictions with robot,
horizon, AP and three-band axes, strictly increasing forecast times beginning
at zero, a model/source identity, and uncertainty whose semantics remain
explicit. The reference must include exact carriers, link direction, reference
transmit power, receiver gauge and noise assumptions. A band name alone does
not identify an interchangeable measurement.

Paired receiver reports identify offsets relative to device 0. They do not
identify device 0's absolute bias. Converting a reference-gauge forecast to
absolute received power therefore requires a stated device-0 gauge assumption,
or receiver noise measured in the same reference gauge so that the common
offset cancels. Converting downlink measurements to uplink service also
requires a declared transmit-power correction and reciprocity assumption.
Unknown antenna response, calibration bias or direction-dependent hardware
gain is not inferred from RSSI alone.

Noise and interference combine as linear powers before conversion back to dB.
An omitted interference process must be an explicit scenario assumption; it
is not evidence of zero interference in a warehouse. The bridge cannot use
private simulator fading, blockage, true receiver offsets, material labels or
future packet outcomes as prediction inputs. Missions never enter the radio
encoder or power-to-outcome conversion.

## Typed outcomes and supported actions

`RSSIOutcomeAdapter` produces the existing typed action outcomes through the
native PHY packet-error tables. Those outcomes preserve their actual scope:
the unqualified risk is a declared fading/retry/blockage surrogate, and the
rate is a PHY success-times-rate proxy. It is not measured MAC goodput,
application availability or a packet-deadline guarantee. Unknown silence and
latency distributions remain unavailable.

Only the association, band and MCS actions implemented by the existing
interface may be proposed. Power control, channel width, MLO, routes, new
handover mechanisms and unimplemented outcome heads must be rejected rather
than assigned invented counterfactual values. Switching cost is explicit.
Typed planning proposals still require the existing independent gate and do
not authorize hardware actuation.

The archived `native.phy.LinkCurves` source is the ns-3 NIST model for HT20 and
VHT80, two streams and 1,200-byte payloads. The native simulator reuses these
curves for its 600-byte control packets. Its 6 GHz table uses the 5 GHz
per-MCS error behavior with doubled nominal rate for a 160 MHz proxy. These
are declared modeling approximations, not measured SmartCar PHY curves.
Native propagation's extra 6 GHz loss uses **6.5 GHz**; the spectral field
study's held carrier is **6.0 GHz**. A bridge must match the configured carrier
and endpoint, or reject the mismatch. Relabeling one as the other would not
validate transfer.

## Calibration is specific to the labeled quantity

`HeldoutCalibration` identifies the admitted TRAIN/VAL calibration unit,
source bytes, split and endpoint. `ResidualBound` concerns RF residuals under
that scope. A fitted residual quantile is neither a universal coverage
guarantee nor a packet-risk certificate, particularly after drift, frontend
changes or action-dependent selection. Unsupported conditions require refusal
or the independent fallback to RadioMap.

`PacketRiskCalibration` requires actual packet labels for the head being
calibrated. For the native control workload this means the declared 600-byte,
20 Hz stream and 50 ms deadline, together with retry behavior, scheduling,
contention, traffic, interference, scan unavailability and handover cost.
Delivery, deadline misses and silence are different endpoints. A model
calibrated for one does not automatically qualify the others. Repeated
packets from a robot or geometry are not independent calibration units;
the receipt must state the unit, grouping, sample counts and empirical
coverage or error being reported.

Calibration fits may use only the explicitly admitted TRAIN/VAL sources and
must remain separate from their evaluation. Fresh FIELD labels and the
original 24 SERVICE confirmation geometries are not calibration data.
A receipt or SHA256 reference records provenance; it does not self-approve a
head for a chance constraint. Independent qualification remains necessary.

## A matched native development probe

`BridgedPredictor` is the public-input integration boundary. Its forecast
callback receives only the delivered, strictly past report history and public
planned positions. The existing completion-safe queue expects
`step(past, plan)` to return mean, standard deviation, timing regime and actual
elapsed time. Forecasts have shape `N × 9 × AP × 3` on the existing native
horizons. They replace forecasts only; the simulator's physical world and
hidden channel draws remain unchanged.

`ServiceProbeConfig` and `run_development_pair` bound a development comparison
between a candidate and a strong RadioMap control. Both arms must use the
same 100-robot physical scenario, seed, traffic, packet curves, action set,
planner, mission, scan cadence and cost, delivered-report capture capacity,
memory/history allowance, proposal lifetime and independent gates. Each arm
receives its own copied world and causal telemetry state. Actual native
packet execution supplies labels; predicted success is never substituted for
a delivered packet.

Completion accounting must charge at least the maximum of the common sealed
compute allowance, predictor-reported elapsed time and measured issue-path
time, plus the common transport delay. FIFO completion releases only finished
forecasts and records their source time, age, hashes and consumption. A budget
overrun remains visible. This synchronous simulator accounting does not
establish asynchronous hard real-time execution on a robot.

Keep native planning enabled with `policy.gate=True`, so both existing
independent gate hooks execute for each arm. A shared typed outcome scorer
can supply association scores without changing those gates. The simulator's
`external_primary` path bypasses native planned-roam admission and is not an
equivalent drop-in for this comparison. Exact outage integration must include
the fractional handover boundary; an older knot-truncated integral is not the
matched reference.

Evidence should retain forecast and calibration identities, paid completion
records, delivered-report provenance, decision and admission masks, executed
moves, and native packet counters. Detailed counters include per-robot sent
packets, misses and gaps, cumulative time windows, delivered report events,
and the declared workload/configuration. These permit checking a service
result independently of its radio prediction error.

The finite source, protocol, cases, metrics and admissible calibration inputs
must be registered and published before official development execution.
Software fixtures check shape, units, refusal, causality and native parity;
they are not scientific cases. No development probe opens the original 24
SERVICE holdouts, deploys a learned head by default or establishes production
or real-hardware reliability.
