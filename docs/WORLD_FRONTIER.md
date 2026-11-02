# Warehouse world-model optimisation protocol

This follow-up targets remaining weaknesses: preprocessing/ensemble cost,
route-change error, weak drift adaptation, and the gap between offline
forecasts and a controller paying for scans and inference.

The candidate set includes a compact distilled model, a route/reliability-loss
model and a temporal model. Each has three independent fits on new sites;
validation chooses a single checkpoint before controller selection. Training
varies fleet density and scan cadence. A packet-failure loss is explicitly a
PHY surrogate derived from the archived ns-3 curves, not measured end-to-end
packet-loss supervision. Native packet results remain the control criterion.

All references receive the same optimisation of RadioMap covariance
prediction. Candidate forecasts must wait in a measured-service-time FIFO
before the planner can consume them. Their horizons are advanced by their
age; fast control continues during computation. A cost-free replay isolates
the effect of the old timing assumptions. Scan outages and sparse scan
reports are charged consistently at the declared cadence. The simulator
still idealises attached-link measurements and scanner phases; the assumed
30/90 ms scan durations are not physical measurements.

Eight development sites choose models/planners before 24 fresh sites are
opened: eight normal warehouses, six drift, six sparse-scanning and four
with stock-length 4-second handovers plus slower sensing/computation.
The scoring alternatives are the existing SNR planner and a saturated
reliability-aware planner, independently selected for each reference too.

The existing nine fits, 760 runs and independent confirmation stay unchanged.
This remains simulation of the canonical warehouse geometry. It does not
claim arbitrary-layout transfer or qualification of physical smartcars.

[Machine-readable protocol](../artifacts/world_frontier/protocol.json).
