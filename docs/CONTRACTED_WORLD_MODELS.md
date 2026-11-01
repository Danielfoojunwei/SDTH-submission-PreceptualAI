# Contracted device-side radio world models

This is a prospective, simulation-only research protocol for the two supplied
paper plans. It does not assume either hypothesis succeeds. The existing
Bayesian reports remain immutable. The new run is separate from the slide deck.

## Frozen design

- Training sites: 260000–260023; development: 261000–261005; primary test:
  262000–262005. Generalisation: 263000–263005 (10 robots), 264000–264005
  (stress), 265000–265005 (open logistics yard). Development pilot: 250000.
- Warehouse physics are sampled with `native.foundation.random_site`. The yard
  has different AP coordinates, perimeter routes, no warehouse racks and a
  declared analytic propagation field; it is another simulation, not field data.
- Six one-second scans form the device context. Every scan has at most eight
  reported BSSs above −90 dBm; unheard BSSs retain the top-eight/floor censoring
  bound. An absent scan has its own mask and is not a non-detection.
- A shared per-link encoder, EMA target encoder and future-route predictor form
  the JEPA. The target contains future **censored scans**, not complete channel
  truth. A censored Gaussian link head predicts all nine existing planner
  horizons (0–30 seconds); primary forecast analysis uses 0–5 seconds.
- Four equal-sized families: JEPA; decision-regret-weighted JEPA; the same
  regret family without future routes; and censored-supervised scratch. Five
  independently initialised fits per family, seeds 0–4, twelve epochs each.
- Two actual cached d3 readings (`bulk_uploads`, `scanner_quiet`) define training
  missions. `protect_teleop` and `protect_worst` are held-out mission readings.
  No new language inference or general semantic transfer is implied.
- Regret weights use twin truth **only as training supervision**, evaluating
  the finite candidate AP choice in the native planner's SNR/load/switch-cost
  score. Leave-one-AP correction measures decision sensitivity. This is AP-score
  regret, not an exact end-to-end packet-loss gradient. Physics inputs and JEPA
  targets remain device observations. The regret family also minimises a soft
  finite-choice regret term.
- Model choice/checkpoint selection uses development censored likelihood only.
  A development-only multiplicative residual calibration is fitted per family.
  Test truth is confined to evaluation and is never a predictor input.
- Ensemble spread and the link head's variance supply uncertainty. The adapter
  fills `World.forecast` and `sigma_belief`; the existing planner is unchanged.
  Native fallback/associated-link measurements retain their existing idealised
  measurement model; the device-only restriction applies to world-model input.

## Executed comparisons to report

Primary scenes: 100 robots, 120 seconds, 60-second initial survey; generalisation:
60 seconds per scene. Report every paired scene and raw sent/missed counts.
These durations are a new declared protocol, not a reproduction of 300-second
native studies. Packets within a scene are correlated.

1. Fixed split; Bayesian ungated/gated; JEPA ungated/gated; scratch ungated/gated.
   Report the five individual JEPA fits as well as the five-member ensemble.
   Paper 1's proposed success requires gated JEPA to beat fixed split on at least
   five of six fresh scenes. Report whether ungated JEPA already does so.
2. Hold JEPA forecasts bit-identical and change uncertainty only: ×0.5, ×2,
   independent lognormal noise, and link shuffling. Report coverage, error/spread
   Spearman correlation and control outcomes. Scaling preserves ranks but can
   alter a fixed gate; a difference refutes a ranking-only explanation.
3. JEPA versus regret JEPA versus Bayesian behind the same mission-specific
   contract; report control misses, forecast RMSE and finite-choice regret.
4. Held-out missions, regret-route ablation and 10-robot/stress/yard comparisons.
   A control gain is an empirical result, not a requirement imposed on outputs.

## Theory acceptance boundary

No distribution-free bound based only on rank correlation exists for this fixed
threshold gate. A perfect ordering can have arbitrarily wrong scale. We instead
derive a gain-error-envelope bound and give an executable counterexample to the
rank-only claim. The theorem concerns the declared one-decision AP score. It
does not certify cumulative deadline misses, a physical robot, or an unverified
fallback. Empirical coverage is reported rather than assumed.

## Prior art and paper claims

The uploaded titles/identifiers are search leads, not verified citations.
The prior-art audit retrieves identifier metadata/full texts where accessible,
records hashes, searches decision-focused learning, selective prediction and
network-control terms, and distinguishes inspected full text from metadata and
access failures. Missing indexed/paywalled full texts prevent an exhaustive
novelty claim. Publication readiness requires results **and** verified literature;
completing software does not make a proposed paper claim true.
