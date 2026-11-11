# What the hierarchy experiment does and does not explain

The completed [RF report](../../artifacts/hierarchical_radio/field_report.json)
does not support the selected AP-local hierarchy over its global or numerical
references. It supports a modest matched predictive-agreement benefit and a
large benefit from retaining RSSI amplitudes. These findings do not identify
representation collapse, convergence, or the cause of the hierarchy's higher
error. The [results](RESULTS.md) retain the registered contrasts and their
limits; this interpretation introduces no new TEST scores or model selection.

## The hidden physical changes are shared across links

The [generator](../../src/wirelessjev/hierarchical_radio/simulation.py) changes
the materials of two public obstacles. Within each geometry, state one changes
obstacle zero to metal and state two changes obstacle one to metal. AP and
receiver positions, CAD, known transmit power, solver seeds and receiver-noise
draws are shared across the three states. No AP-specific transmitter activity,
moving obstacle, receiver drift or chronological transition is introduced.

An obstacle's electromagnetic response affects multiple AP–receiver paths.
It is therefore plausible that shared world factors are more useful here than
many independently adjustable per-AP residual coefficients. The eight-state
control already uses geometry-conditioned regressors for distance, obstacle
intersection, path length and frequency interactions. Its lower descriptive
error is consistent with that hypothesis.

This does **not** prove that two material assignments require only two linear
RF coefficients. Reflections, frequency dependence and coherent path sums make
the map from material to reported power nonlinear. Nor does it establish that
AP-local state is unnecessary in a moving, heterogeneous warehouse. The tested
hierarchy retains cross-AP coupling through its shared global state; its local
priors are separate conditional on that state. Calling the entire posterior
independent across APs would be incorrect.

## A concrete constraint: the local channel cannot be switched off exactly

In the [model](../../src/wirelessjev/hierarchical_radio/model.py), the local
coefficient prior standard deviation is `0.2 + softplus(raw_local_std)` in the
normalized residual units. Residuals are multiplied by ten when rendered in
dB. Each local coefficient therefore has a prior SD floor of 2 dB before
conditioning. Eight fixed, nonzero Fourier features remain in the local
design even if all eight learned features become negligible.

Consequently the hierarchy cannot turn its AP-local **prior channel** into the
exact global-eight submodel by shrinking its variance to zero. The
`global8_reg` control explicitly zeros that local design. This is not a uniform
2 dB floor on the final predictive SD: conditioning reduces coefficient
uncertainty, and field variance also depends on the query features.

For a fixed Gaussian working model, write the observation as
`y = G g + L l + noise`. Marginalizing a local prior with covariance `S_l`
changes the effective noise seen by the shared state from `R` to
`R + L S_l L^T`. If that local variation is unnecessary or too large, it can
weaken identification of the shared state. Useful local variation can instead
improve prediction. This algebra supplies a specific mechanism to test; it
does not establish that it caused the observed error. The interval
Newton/Laplace likelihood, learned discrepancies and optimization add further
approximations.

Equal allocated trainable-parameter counts do not remove this issue. A
four-AP hierarchy infers 72 scene coefficients, compared with 24 for the pooled
control and eight for the small control. More state can introduce uncertainty
and estimation cost as well as expressive capacity.

## Collapse and optimization remain separate questions

The retained [fit ledgers](../../artifacts/hierarchical_radio/fits) show that all
15 members choose epoch 40 and improve at every fixed validation check
(10/20/30/40). That is selection at the budget boundary, not a convergence
certificate or proof that longer training would reverse the ranking.

Across all 7,200 committed training updates, the recorded mean inference step
is one and the null-step count is zero. Thus the logged training inference was
not repeatedly halted by the backtracking fallback. This does not prove that
three Newton steps reach a mode or that neural optimization has converged.

The amplitude-erasure contrast demonstrates prediction dependence on RF
values. It does not establish that the eight learned local features have high
effective rank or carry information beyond the fixed Fourier and physical
bases. No activation-rank, conditional feature-variation or feature-ablation
diagnosis was performed here. Poor error alone diagnoses neither collapse nor
its absence. This model also does not learn temporal transitions or feed
predictions back as observations; static conditioning cannot validate stable
long rollouts.

## Three bounded next checks

1. **Separate budget limits from representation failure on TRAIN/validation.**
   Predeclare a larger training-budget comparison, retain its full learning
   curves, and inspect feature variation and observation sensitivity separately
   from posterior-state dimension. Check effective rank conditional on geometry
   and compare learned-feature ablations with the fixed bases. Choose any
   configuration using TRAIN/validation only, then freeze it before a new
   reserved evaluation. Do not extend a current run and reuse this TEST as a
   fresh confirmation.

2. **Test a genuinely nested shared-world model.** Keep the global-eight
   reference and compare the current hierarchy with a shared object/path-factor
   model and an AP-residual channel that admits an explicit zero/off case or
   true shrinkage toward it. Share object influence across APs using public
   geometry, while allowing residuals where observations support them. Include
   an equally informed numerical hierarchical or GP control, the same admitted
   reports and training budget, and fresh held layouts. Abstract shared factors
   must not be named identified materials or scatterers without an observability
   argument; any privileged TRAIN labels must be disclosed and matched.

3. **Evaluate the capability missing from the static assay.** Use actual
   event-time sequences with shared moving blockers or interferer activity,
   causal observations, paid sensing and longer unseen intervals. First test
   whether the available observations distinguish future-relevant states;
   retain ambiguity or abstention when they do not. Compare fresh RadioMap,
   last-observation and time-filter baselines with filtering and prediction
   evaluated separately. Hold whole sequences out and report accuracy,
   uncertainty, compute, packet tails and both handover costs. Such a study
   could establish an advantage in prediction or sensing efficiency; the
   current static results cannot supply it.

These checks are proposals, not extra experiments or a rescue claim for this
batch. They give physical sharing, shrinkage and observability testable roles
without assuming that a larger latent, a JEPA name, diffusion, or additional
consistency loss must outperform a numerical map.
