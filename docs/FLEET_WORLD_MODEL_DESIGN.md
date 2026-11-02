# Causal fleet world-model design

The warehouse model predicts mean uplink SNR at nine positions on a robot's
planned route. It is a forecast of the radio field, not a packet-success
certificate. AP choice, retry rate, band, duplication, roaming, recovery and
bulk timing remain explicit native-controller decisions.

The model state has three parts:

1. Six past capped device scans, with separate heard/missing/censor-bound
   fields and robot-to-AP geometry. Unknown RSSI never becomes a zero-dBm
   observation. Only reports strictly before the current decision are used.
2. Sixty seconds of fleet memory. Each AP/band has report counts, residual
   statistics, temporal change, spatial coverage and a regularised geometric
   fit. Residuals are relative to known geometric physics, so the standalone
   model does not depend on per-BSS RadioMap predictions.
3. Route queries with distance, rack crossings, band, position features and
   horizon. They encode a robot's plan, not future observations.

An encoder embeds each link's scan context and pools a robot-level context.
The predictor combines local context, pooled context, fleet memory and route
features. A residual head predicts mean SNR and positive uncertainty. The
hybrid adds its correction to a censored rolling RadioMap; the standalone
adds it to a small reports-only geometric estimate. This distinction is
exposed in both results and the runtime API.

Training combines robust mean-field prediction, Gaussian error scoring,
soft AP-regret and relative AP-score losses. JEPA models also predict an EMA
encoder's future scan representation, with a variance regulariser. Future
scan embeddings and simulator mean-field labels are training targets only.
The supervised ablation removes the JEPA objective while keeping the same
architecture and information. All three families have three independent
training initialisations.

RadioMap baselines get the same fleet scans and CAD geometry. Rolling models
refit every twenty seconds from the trailing minute, including proper top-k
censor bounds. A stronger reference uses the existing cross-site prior.
Each reference and learned family gets the same development gate choices;
the model and strongest reference are selected before fresh tests.

## Runtime use

```python
from wirelessjev.fleet_control import FleetWorldModel

# geometry contains known AP positions, band powers/noise and rack CAD.
# history is an Observations prefix ending BEFORE this decision.
model = FleetWorldModel(geometry, family="development_best")
mean, std, timing = model.forecast(history, planned_positions)
policy, roam_gate = model.controller()
```

`planned_positions` has shape `(robots, 9, 2)`. The outputs have shape
`(robots, 9, access_points, 3)`, matching the native controller. The helper
verifies checkpoint and execution-source identities against the frozen
development selection. It never consults test scores to choose a model.
`family="fleet_jepa"` runs without fitting or evaluating a per-BSS RadioMap.
`family="confirmed"` exposes the separately fixed confirmation candidate only
if its independent primary control-improvement criterion passes. Otherwise
the helper refuses that profile. The development-selected research profile
remains a distinct choice; it is never silently relabelled as confirmation.

The reusable policy is scoped to the tested bulk-upload mission. The native
default remains RadioMap. This is an opt-in research interface; the results
do not qualify physical-car inference, scanning or roaming.

## What the experiment can and cannot resolve

The fresh paired warehouse runs test whether richer learned forecasts reduce
control misses beyond a separately tuned RadioMap. Forecast RMSE and AP-rank
error test a different property. A learned forecast can improve one and lose
the other, so both are reported alongside video, bulk and stop time.

JEPA's necessity is a separate question, answered by the supervised ablation.
Hybrid superiority would establish the value of a learned correction to
RadioMap, not replacement of RadioMap or lower total compute. Standalone
superiority must be supported by its own results. Three-seed ensembles do
not provide confidence intervals over independent retraining campaigns.

This study assumes known CAD, accurate localisation, a synchronised 1 Hz
scan stream and a fixed robot route. Fast fading and independent blockage
remain hidden. Drift tests alter the mean field with an explicitly declared
smooth attenuation intervention. Real scan scheduling cost, 4-second stock
handover behaviour, on-car inference and measured interference remain field
validation tasks. No synthetic result is presented as a hardware result.

The warehouse seeds share the canonical rack/AP layout while randomising
propagation, routes and traffic. The outdoor yard supplies a different AP
topology, but this is not a test of arbitrary warehouse layouts or general
Earth environments. A broader foundation-model claim needs independent CAD
layouts and device/channel measurements beyond this simulation family.
Stress, yard and drift examples are included in training. Their test seeds
are fresh, but those tests do not hold out the entire environment family.

Forecast labels use the realised future mean SNR, while the query supplies
planned positions. A plan can end before the realised next task, so route
mismatch also contributes to forecast error. The reported RMSE is therefore
future-SNR prediction error, not a pure map-reconstruction error at a fixed
query coordinate. Every controller follows the same precomputed trajectory;
the safety-stop metric does not resimulate motion after a radio interruption.
Forecasts are computed from causal histories, then supplied to the native
simulation offline. The measured inference and map-refit times are not
inserted as extra delays in the packet simulator. A live asynchronous
forecast service and delayed-input evaluation are needed before interpreting
these results as a real-time robot-control improvement.
