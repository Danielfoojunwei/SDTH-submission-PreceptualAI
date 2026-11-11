# Scope and remaining validation

The shared/AP-local field model and its finite experiment are implemented and
registered. All 186 RT cases, 15 neural fits and 34 warehouse simulations have
completed. The canonical seal binds 567 source and 4,900 evidence files, and
the [first independent actual-data audit](../../artifacts/hierarchical_radio_review/independent.json)
passes without scientific reruns or changed evidence.
[PLAN.md](PLAN.md) defines the acceptance rules;
[ARCHITECTURE.md](ARCHITECTURE.md) describes the actual model, and
[RESULTS.md](RESULTS.md) retains its primary harms and limited gains. Software
success and completed execution do not establish an experimental improvement.

The selected hierarchy has higher primary RF error than the global control
and numerical map, and higher native mean command misses than global-24 and
full-history RadioMap. Joint agreement has a small matched RF benefit, and
the hierarchy improves native mean misses over the matched numerical map;
these do not overturn the main negative comparisons. The retained 8.89 s
simulated gap also prevents a dependable-connectivity claim for 100 cars.

The model supplies a conditional **spatial RSSI field**. It combines a shared
physical state with AP-local spatial state, using shared learned feature
functions and admitted observations. The joint cell adds predictive agreement;
it has no EMA teacher. A supported gain over a global model would establish a
benefit for this representation under the registered conditions. A joint-cell
gain would need its own matched contrast. Neither follows from the model name,
latent size or an aggregate win over a numerical map.
Predictions are never reinserted as observations, removing one source of
self-reinforcing belief error. That design choice alone does not bound field
error, trajectory error or long-horizon control losses.
The [interpretation](INTERPRETATION.md) also identifies a testable limitation:
the current local prior has a positive coefficient-variance floor and fixed
active Fourier features, so it cannot recover the exact global-eight submodel
by zeroing the local channel. A nested zero/off or shrinkage control belongs
in a new protocol. The source constraint does not prove why this batch lost.

| Built scope | Validation still required beyond this batch |
| --- | --- |
| Shared and AP-local posterior, with new AP counts handled by shared functions | Larger and independently held warehouses, diverse AP/receiver geometry, adequate training-seed replication, memory and complete inference-cost scaling. More per-scene inferred coefficients must remain explicit when comparing global state. |
| Context-conditioned static material counterfactuals | Event-time transitions, revisits and moving blockers/interference, with actual measurement, arrival, computation and action clocks. Static state changes do not establish temporal learning or nonaccumulating rollout error. |
| Quantization/censor likelihood and approximate posterior spread | Empirical coverage under new sites, correlated reports, selective queries and drift; separately calibrated packet/deadline/silence risks. Both interval Newton/Laplace and numerical interval EM/Laplace are approximations. |
| A fixed admitted-report budget and causal native selector | Separate matched retrieval ablations, changing observation budgets, obstruction/age support and sensing value per paid cost. No representation gain should be credited to extra information. |
| Three trained members per RF cell and a fixed seed-zero native diagnostic | Replicated training for service transfer, longer independently reserved episodes and adequate power for rare tails. Whole-layout intervals conditional on three members do not cover arbitrary retraining. |
| Separate 100-robot simulated warehouse control transfer | Real protocol observations, verified driver capabilities, measured handover stages, qualified command authority, supervised rollback and physical fleet trials. The simulator's differing scale/carrier and PHY assumptions remain extrapolation. |
| RSSI-specific features and a typed downstream interface | Validated frontends and observation/action semantics for each new radio protocol. A common interface alone does not make the trained model protocol agnostic or establish executable PHY control. |

The next model change should follow the retained contrasts. Preserve the strong
numerical maps, parameter-matched supervision, value-erased and stale-context
controls. A field-prediction gain may coexist with worse control outcomes;
the full-history native RadioMap and matched-context numerical reference answer
different information-budget questions. Preserve that distinction and report
handover costs separately.

In RF evaluation, learned and numerical predictors receive the same archived
view-A context for each example; the value-erased control deliberately removes
amplitudes internally. In the native experiment, the three matched methods
share the 192-report budget, selector and support-guard rules. Their realized
scans and delivered history can diverge as their actions change connectivity.
The native comparison therefore does not hold context bytes identical across
arms. It tests the complete closed loop under the declared information rules.

Representation-collapse diagnosis still needs evidence such as context
sensitivity, conditional feature/state variation and held-out prediction
dependence. Bounded features, shared functions and positive covariance do not
prove collapse absent. Conversely, an unfavorable metric does not diagnose
collapse by itself. Any architecture or loss chosen after these outcomes needs
a new protocol and fresh evaluation cases.

Complex-channel observability, physical frontend inversion and RF–vision
alignment remain separate tasks. The retained
[CSI analysis](../research_status/CSI_INTERPRETATION.md) and
[broader research requirements](../research_status/REMAINING.md) explain their
evidence limits. The latter is an earlier status snapshot; this directory
defines the new shared/AP-local implementation and its registered scope.

RadioMap remains the application default. The original SERVICE holdouts stay
closed. No universal foundation-model, JEPA-superiority, specialist-SOTA,
production, hardware or arbitrary-environment claim follows from completing
this finite batch. The [one-car procedure](../ONE_CAR_TEST.md) still requires
its real hardware prerequisites.
