# Fleet-memory world model: prospective protocol

The previous JEPA used six scans from one robot; RadioMap used sixty seconds
of fleet reports plus rack geometry. This study closes that information gap
and tests whether learned forecasts improve beyond a tuned RadioMap.

Two models are built: a RadioMap-assisted JEPA residual predictor, and a
fleet-only geometric-memory JEPA without the per-BSS RadioMap. A third model
removes the JEPA objective from the assisted predictor. All receive causal
device reports, known AP/CAD geometry and planned routes. Simulator mean-field
labels supervise training only. The assisted model is explicitly a hybrid;
it cannot establish that neural inference replaces RadioMap's fitting cost.

The rolling reference refits the same censored RadioMap on the same trailing
sixty seconds of fleet observations, every twenty seconds. Forecasts between
refreshes use that frozen posterior with current planned positions. All
methods use the same native controller, physical assumptions and reactive
measurement model. Controller penalties are tuned separately for each
method on development scenes. Static RadioMap, the previous JEPA, a fixed
split and reactive-rescue choices remain explicit comparisons.

Training uses 32 new sites and three independent seeds per model. Checkpoints
are selected on six other sites. Closed-loop policy selection uses twelve
separate sites. Only after the choice is sealed are 32 fresh test sites
opened: eight primary warehouses and six each under stress, outdoors, with
ten robots, or with changing propagation. The drift condition applies smooth
AP attenuation changes to the mean field; the schedule is hidden from all
forecasters. It is a declared simulation intervention, not hardware evidence.

The primary metric is missed control commands per million, with paired-site
intervals. Forecast RMSE, positive/nonpositive mean-SNR errors, AP-choice regret,
traffic delivery and compute are reported too. Failures and the no-JEPA
ablation are part of the result. All original studies remain immutable.

Protocol: [protocol.json](../artifacts/fleet_world_model/protocol.json).

Before selection or fresh-test access, the reference set was expanded to
include the existing forty-site learned RadioMap prior. It gets the same
past scans, rolling refits and independent controller tuning. The strongest
RadioMap reference is chosen on development scenes only. This adds sixty
selection runs and forty test runs without changing the core model training.
See [the amendment](../artifacts/fleet_world_model/strong_reference_amendment.json).

The raw kernel names its SNR>0 split `audible` and its SNR<=0 split `unheard`;
the final report uses the precise SNR labels. These are not measurements of
whether a BSSID appeared in a capped scan.
