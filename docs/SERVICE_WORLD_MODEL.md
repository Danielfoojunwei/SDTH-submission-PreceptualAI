# Physics-anchored service model: implemented contract and limits

The implementation is
[`service_world/model.py`](../src/wirelessjev/service_world/model.py). It is a
**deterministic recurrent radio-history encoder with an action-conditioned
Gaussian latent transition and physics-residual service heads**. JEPA is an
optional training objective. The default configuration has 68,021 inference
parameters; its supervised ablation has the identical inference graph.

This terminology is deliberate. The encoder's `belief` is an amortized summary
of observed history. It is not an explicit Bayesian posterior over physical
channel, obstruction and queue states. The implementation has no physical
observation likelihood, Bayes normalization, variational posterior/prior KL,
or Dreamer actor. A single latent/decoder standard-deviation head does not
separate epistemic uncertainty from irreducible channel randomness. Calling
this a calibrated Bayesian radio filter, a complete RSSM or a new foundation
architecture would exceed what was implemented.

The components—shared link encoders, graph messages, recurrent state,
action-conditioned latent prediction, Gaussian heads, numerical residuals and
EMA future targets—are established methods. Their combination here is a
concrete method adaptation, with no novelty, SOTA or control-superiority claim
implied by the model class name.

## Wireless structure represented

Each batch item represents one client's history against an AP list, retaining
separate AP and three-band axes. The batch axis is not a learned robot-to-robot
graph. Fleet interactions can enter through supplied global features and known
edges; explicit cross-client message passing is not implemented.

The same link encoder and recurrent weights process every AP/band link. There
is no learned opaque AP-ID embedding. A known, nonnegative adjacency can encode
same-band co-channel or CAD relationships. Without one, the encoder uses a
symmetric fully connected same-band message average; that default must not be
described as measured interference topology.

Relabeling the AP list, relabeling adjacency on both AP axes and remapping the
candidate AP indices leaves global predictions unchanged and relabels link
latents accordingly. Tests check this with nonzero decoder weights, so the
property is not hidden by the initial zero residual. This permutation property
is a useful inductive bias; it does not prove transfer to unseen CAD, AP counts,
hardware or protocols.

Actual values, validity masks and elapsed physical seconds enter the encoder.
An unknown RSSI can use an arbitrary masked placeholder without becoming an
invented zero-dBm or below-threshold measurement. Known geometry, map priors,
censoring information and observation-age features can remain valid while
individual receiver features are missing. The caller must supply that distinction
correctly; the model does not infer whether a scan genuinely occurred.

Time awareness is explicit, but this is one recurrent observation sequence,
not separate fast/slow hierarchical encoders. Genuine multi-rate behavior
depends on the caller providing receiver/packet events at their available
rates, per-link age/missingness and correct elapsed times. A one-Hz input does
not become a 100-ms receiver trace merely by choosing a smaller time feature.

The encoder preserves band-specific link state. The service decoder currently
pools the three embeddings of the selected AP and conditions on the supplied
action features. Detailed band actions therefore need explicit physical action
features and validation. Subcarrier CSI/CIR, complex I/Q, phase/gauge symmetry,
reciprocity and electromagnetic propagation constraints are not implemented.

## Inputs, outputs and units

```python
model = ServiceWorldModel(
    family="jepa", link_dim=12, global_dim=8, action_dim=12, hidden=48
)
prediction = model(
    links, globals, observed, elapsed,
    actions, ap_index, physics_mean,
    hazard_prior=None,
    adjacency=None,
    action_elapsed=None,
)
```

| Argument | Shape | Meaning |
|---|---|---|
| `links` | `B × T × S × 3 × F` | Public link observations and physical features |
| `globals` | `B × T × G` | Public client, service and aggregate context |
| `observed` | link shape or `B × T × S × 3` | Valid features or whole-token validity |
| `elapsed` | `B × T` | Seconds since the preceding observation |
| `actions` | `B × H × Q` | Candidate action/route features known at decision time |
| `ap_index` | `B` or `B × H` | Candidate indices in the supplied public AP list |
| `physics_mean` | `B × H × 2` | Numerical miss-count/gap values or paired differences |
| `hazard_prior` | optional `B × H` | Explicit conditional interval-event prior |
| `action_elapsed` | optional `B × H` | Seconds advanced by each candidate transition |

`mean`, `miss_value` and `gap_value` retain the caller's native count/second
units. If the task is an intervention comparison, they represent paired
differences and can legitimately be negative. The `delta_misses`/`delta_gap`
aliases are appropriate only with that paired-value task. Neither signed
service values nor their residuals are probabilities.

The numerical decoder is

\[
\widehat c_{k}=c_{\mathrm{physics},k}+r_\theta(h_k,a_k).
\]

Both residual heads start at zero. Supplied numerical means and hazard priors
are then preserved exactly in tests. This is a **constructor identity**, not
a learned-accuracy result, an improvement over RadioMap, or a guarantee that
training cannot harm the numerical estimate. Residuals can become nonzero
after fitting. Known physical floors and feasible action constraints belong
in a separately validated projection/controller; generic paired values must
not be clamped to an absolute-service floor.

If no hazard prior is supplied, `hazard` and `survival` are `None`. The model
does not invent a small probability or label an unprovided prior calibrated.
Supplying values in `[0, 1]` establishes numerical range only, not calibration.

For the product `survival = cumprod(1-hazard)` to describe first-event survival,
each hazard must be conditional on survival through preceding intervals.
Training must supply a correct at-risk/censoring mask via `hazard_valid`.
Multiplying arbitrary correlated interval marginal risks does not establish
a valid joint survival probability. A single-horizon probe avoids that
multi-interval interpretation but still needs held-out calibration.

## State and stochastic rollout semantics

`encode(...)` returns per-link state, pooled history state and a streaming
`state` dictionary. Passing that dictionary is valid only for **new,
nonoverlapping** observation chunks. Feeding a sliding window together with
its previous state assimilates old evidence twice. Chunk/full-history
equivalence is tested under the documented streaming contract.

`rollout(encoded, ...)` reuses one encoding across candidate action sequences.
Each transition conditions on the selected AP representation, action features,
elapsed action time and preceding latent state. It predicts a diagonal
Gaussian distribution in learned latent coordinates. Those coordinates are
not identified physical channel variables.

Default rollout propagates latent means. This is not the exact expectation of
a nonlinear stochastic future. An explicit `latent_noise[B,H,D]` supplies
reproducible samples and allows estimating effects of multiple future paths.
Sampling, integration error, cross-step covariance and additional compute must
be measured before interpreting sampled tails as service-risk guarantees.

The action transition updates a global latent; it does not evolve a separate
future channel tensor for every AP/subcarrier. The selected AP representations
come from the encoded history. This is a compact decision-oriented latent
model, not a full differentiable PHY or packet simulator.

## Training and the required ablations

`objective(model, batch, phase="pretrain")` uses future observable windows
through a frozen EMA target encoder and needs no native service labels. It is
defined for the JEPA family. `phase="joint"` combines that objective with
native value regression; `phase="finetune"` uses the native labels without
reading future observations for either family. `update_target(model)` performs
an explicit EMA update. Future inputs never enter `forward`.

The supervised and JEPA models share the same encoder, transition, residual
decoder, stored teacher and inference parameter count. Their training objective
is the controlled difference. Separate pretraining experiments must also report
their additional data and optimization budget; equal inference parameters
alone do not establish equal training resources.

Before any improvement claim, compare the numerical zero-residual baseline,
a simpler temporal/filter baseline, the same supervised model and the JEPA
variant using identical observations, native action labels, controller
continuation and charged timing. Preserve the baseline's recovery behavior.
Measure representation probes, label efficiency, selected-action calibration,
actual packet loss, outage tails and cumulative excess independently. A
single-horizon offline prediction win is not an active-controller win; a
model-mechanics test is neither benchmark evidence nor evidence that no prior
work exists.

The implemented tests cover exact numerical anchoring, AP relabeling,
streaming equivalence, causal action rollouts, missing-data poisoning,
EMA isolation, label-free pretraining, supervised future-input isolation and
explicit stochastic reproducibility. Physical hardware validation, broad
wireless SOTA comparisons and publication novelty remain separate work.
