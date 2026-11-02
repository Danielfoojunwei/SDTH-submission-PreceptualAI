# Diagnose and repair the JEPA gate

The study is complete. [Results and causal diagnosis](JEPA_GATE_REPAIR_RESULTS.md)
include all 270 study runs, six exact historical replays, raw traces and the
sealed development choice. This file retains the prospective protocol.

This follow-up is designed after the first study. Its old test scenes remain
historical evidence; no threshold or model is selected on them. Five existing
JEPA checkpoints remain frozen. New fitting, calibration, closed-loop selection
and fresh test sites are disjoint and specified in
`artifacts/gate_repair/protocol.json` before their evaluation.

The suspected mismatch is between absolute link uncertainty and uncertainty of
a relative, route-averaged AP decision. The legacy gate takes the maximum spread
across all bands at each AP, then assumes independent errors across the two APs.
Its residual scaling was fitted to all-link SNR coverage, not control quality.
Decision traces will measure beneficial predicted moves rejected without rescue,
pairwise error, active-band versus maximum-band spread and uncertainty ranking.

The repair keeps the JEPA mean forecasts identical across arms. It derives
ensemble uncertainty from candidate-minus-current AP scores, retaining error
covariance, and fits a small conditional model for one-sided gain overprediction
on development traces. Separate development sites calibrate its residual bound.
The gate asks whether a switch still has positive estimated advantage after the
bound. It returns control to the reactive rule when no planned move is eligible.
No true SNR, future scan or evaluation label reaches the deployed gate.

Closed-loop development chooses a gain-bound quantile and, independently, a
simple multiplier for the old gate. Selection balances the four conditions,
using each condition's smoothed miss-count ratio to ungated JEPA. A gain gate
must reject at least 1% of eligible development opportunities; otherwise it is
not represented as a meaningful uncertainty filter. A separate always-eligible
planner plus reactive-rescue arm distinguishes fallback routing from uncertainty
estimation. Fresh tests compare seven arms over six scenes in each condition.

The gain score is still the native AP score, not complete packet utility.
Residual quantiles are empirical and correlated, so no distribution-free safety
guarantee is claimed. Real radios and hardware deployment remain out of scope.
Original study artifacts stay immutable. The simulator's default gate behavior
is preserved; the new interface is optional and tested against the old path.
