# Shared and AP-local physical field: finite registered experiment

The experiment trains a new spatial conditional RF model, rather than changing
only its gate. All earlier evidence remains frozen. The executable protocol
and complete geometry registry are written and remotely published before any
fresh physical solve, neural fitting or experimental prediction.

The new model has eight shared physical coefficients and sixteen AP-local
spatial coefficients. Shared functions use physical coordinates; AP integers
only route blocks and never enter an ordinal embedding. Gaussian conditioning
uses block Schur complements instead of a dense state whose cost grows
cubically with AP count. Three observed interval-likelihood Newton/Laplace
steps and covariance are approximations. Each step uses a fixed six-candidate
backtracking rule on the context negative log posterior, retaining the current
iterate if none improves it. Uninformative bounds contribute no curvature;
this is bounded inference, not a new fit or post-result numerical retry.
The coefficients do not identify scatterers,
materials, independent latent factors or packet risk. Priors are learned from
TRAIN or public geometry, not fitted from current reports and then counted
again as independent evidence. Predictions never become measurements.

Fresh actual Sionna RT data contains 24 TRAIN, six validation, four technical
DEV and 24 TEST geometries, plus four separate eight-AP transfer geometries.
Each has three static states: base material assignment, obstacle zero changed,
and obstacle one changed. CAD, TX, receiver/AP positions and solver seeds stay
identical within a geometry. Independent deterministic streams choose geometry,
material assignments and TX, preventing a built-in geometry/material shortcut.
There are 186 physical cases; these are static counterfactual fields, not
chronological environment learning or measured hardware.

Main scenes have four APs and 48 receiver poses, split into 24 context and 24
held spatial query poses. Full/sparse source views contain 192/96 reports in
total, identically admitted for learned and numerical methods. Eight-AP transfer
retains those total budgets, reducing observations per AP. Each source band
ranks actual noisy receiver reports separately and returns top two APs. A
ranked-out report has the kth upper interval bound when full, otherwise the
receiver floor; an unresolved numerical path stays unknown. Query cutoff values
remain target data and never become input features. Source bands are 2.4/5.2
GHz; the separate 6.0 GHz target assay supplies no training or selection input.

View A is device-zero downlink; B is device-one uplink on the same context
physical links. Device-one calibration uses only paired TRAIN context reports
from the same physical downlink; device zero fixes an explicitly assumed gauge.
Targets are reported device-zero downlink values on disjoint query poses.
Directions, receivers, APs and material states remain correlated rows nested
within geometry. A Gaussian above-floor probability is not a top-K reporting
probability, so no such heard-label Brier claim is made.

Five cells each receive three independent training seeds and the same 480
committed optimizer updates: hierarchical regression, hierarchical joint-view
agreement, a parameter-matched global 24-state model using the same learned
features, a global-eight capacity control, and a detection-only value-erased
hierarchy. Erasure retains genuine heard lower-detection events; it cannot
invent below-floor values for ranked-out observations. Both matched hierarchy
branches use the same two-view target likelihood. Only the joint cell adds
fixed variance-weighted predictive agreement at supported common query probes;
this is not a stopped-teacher/EMA equivalence claim.

Checkpoints use first-minimum sparse validation composite interval NLL at
epochs 10/20/30/40. The hierarchy branch uses the mean of all three members'
validation scores; no TEST/native result chooses a branch. Three declared
RadioMap-operator spatial scales use TRAIN-only interval/censor priors and
validation selection. A smaller fixed-feature hierarchical Bayesian control
is retained. The operator is not claimed bit-identical to native RadioMap.
Continuous native context reports use genuine point-Gaussian updates, not
invented quantization intervals.

Four primary sparse heard-RSSI RMSE contrasts are selected hierarchy versus
global 24-state, versus the selected numerical map, joint versus regression,
and regression versus value-erased hierarchy. Each member metric is computed
first, then members and material states are averaged within each of 24 TEST
geometries. A fixed whole-geometry 20,000-draw bootstrap gives four-family
adjusted central 98.75% intervals. These are conditional on the three trained
members, not uncertainty over arbitrary retraining. Full context, localized
state deltas, tails, empirical heard coverage, stale context, held band and
eight-AP transfer remain descriptive. DEV checks technical completeness and
finite outputs only; it does not screen for efficacy or tune TEST behavior.

A separate fixed warehouse diagnostic follows RF evaluation: four fresh
20-second, 100-robot episodes, two handover costs and four methods (32 runs),
plus two numerical round-trip parity runs. Full-history native RadioMap is a
strong reference; matched numerical, global and hierarchy share 192 admitted
source reports and the same support guard. They use the unchanged causal
kernel and paid compute/transport from the completed control study. Local
recovery is off. Native 16-AP geometry/scale/6.5 GHz and PHY assumptions differ
from RT; this is descriptive transfer, not RF or service qualification.

Every physical/native case has a durable once-only attempt. Training resumes
from the last atomically committed state/RNG; repeating an interrupted
uncommitted computation does not add an optimizer update or independent trial.
Nonfinite failures do not trigger new seeds or numerical retries. Raw inputs,
targets, predictions, weights and reductions are retained for canonical and
independent passive checks. Original SERVICE layouts remain unopened; no
physical radio or application default changes, and no universal foundation
model, JEPA superiority, novelty or SOTA outcome is assumed.
