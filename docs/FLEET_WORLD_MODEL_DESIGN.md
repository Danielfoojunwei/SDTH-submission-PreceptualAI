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
