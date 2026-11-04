# Independent confirmation of the selected compact JEPA

The completed semantic-target study did not pass its development gate. Its twelve fits, 112 evaluations, source capsule, negative results and unused confirmation sites remain intact. This separate experiment selects a stronger JEPA profile from those already observed development results and tests it on eight new sites. It neither refits the models nor reinterprets the rejected study as a success.

## Selection before new test data

The two eligible JEPA families share the same compact inference graph. The rule averages each family's three independently acting training members, computes its command-loss ratio to RadioMap at each handover cost, and averages the two ratios equally. Exact command-count fractions determine the selection; no best-member selection occurs.

| Prior development method | Command loss: 30 ms | Command loss: 4 s |
|---|---:|---:|
| Strong RadioMap | 1.022292% | 4.406667% |
| Matched supervised | **1.019931%** | **4.138056%** |
| Compact context JEPA | 1.020069% | 4.162361% |
| Semantic regression | 1.020139% | 4.164306% |
| Semantic JEPA | 1.020278% | 4.169722% |
| Original frozen JEPA ensemble | 1.021458% | 4.189792% |

These are descriptive four-site development results. **Supervised is the strongest overall point estimate.** Context JEPA is selected only as the stronger of the two new JEPA families under the declared rule. Its architecture, actual delivered-observation training, twelve-fit source and checkpoint provenance are documented in [the model design](SEMANTIC_JEPA_DESIGN.md). The clean semantic teacher did not establish an advantage.

The [selection record](../artifacts/selected_jepa/selection.json) preserves both eligible scores, the formula, every selected/supervised checkpoint hash and the reused timing profiles. All three members are fixed before the new sites are opened. The recorded outcomes are averages of separately acting single members; this is not an ensemble policy.

## Fresh comparison

Eight previously unused procedural sites, 100 simulated robots per run, sixty seconds and both 30 ms/four-second handover assumptions produce 128 main evaluations. Each site/cost runs the three selected JEPA members, three matched supervised members, strong RadioMap and the original frozen three-member JEPA ensemble. The first site supplies eight registered exact replays. No new neural fitting, controller selection or timing calibration occurs.

The wrapper composes the frozen runtime, delivered-report collector, PHY curves and instrumented native packet kernel. It does not alter their globals, constants or output paths. RadioMap retains its optimized measured cost; all new neural members share the already sealed envelope. The original ensemble retains its original API/map prior and measured profile, making it a practical package comparator. Its difference cannot be attributed solely to architecture or JEPA loss.

## Three different claims

All six reference/cost comparisons share the registered multiplicity budget. Whole sites are the independent units after averaging paired training members. Both observed counters and conservative unresolved-command assignments must support superiority; per-reference engineering margins, source, causality, timing and replay checks apply.

| Reference | What a passing comparison establishes |
|---|---|
| RadioMap | A cost-specific numerical-reference improvement in this simulation distribution |
| Original frozen JEPA | An improvement of the complete model/data/runtime package |
| Matched supervised | An advantage for this auxiliary-training package; it does not isolate EMA necessity |

Every comparison is reported. A broad best-method claim requires all references at both costs; a slow-cost RadioMap win cannot imply that. Supervised results remain visible even if they outperform JEPA. Engineering margins are empirical research thresholds, not a safety certificate.

The [protocol](../artifacts/selected_jepa/protocol.json) explicitly states that source training and development data were already observed. Only the new confirmation is prospective. The first experiment's reserved sites are not consumed. Application RadioMap remains the default, and hardware, arbitrary layouts, production risk and SOTA remain unqualified.
