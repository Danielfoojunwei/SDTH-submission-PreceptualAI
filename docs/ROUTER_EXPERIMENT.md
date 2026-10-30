# Proposed mission-capability router experiment

Status: design only; no router has been trained or evaluated. The current
E1/E2 results motivate this experiment but do not establish a router benefit.

## What the existing comparison establishes

E1 intent + planning has lower observed regret than native-radio LoRA on
both known and new missions. E2 radio-Q has lower observed normalized regret
than the existing intent baseline, with substantially greater CPU latency.
Both comparisons rank a shared candidate set within the experiment. Their
regret definitions and scene sets differ, so do not pool their raw scores.

`IntentReader` emits nine fixed objective weights and extracts explicitly
named robots. It does not emit numeric constraints, coordinate predicates,
thresholds or conditional objectives. E2 therefore tests a particular mission
interface, not the general ability of optimisation to handle constraints.

The actual utilities in `src/wirelessjev/radiojev/nonlinear.py` are:

| Mission family | Implemented utility | Implication |
|---|---|---|
| Video floor then fairness | Compare the horizon mean of the worst video-robot service with a threshold; use fairness if it passes, otherwise service minus one | A conditional objective; does not guarantee the floor at every snapshot |
| Move budget | Total service minus 0.2 times excess reconnections | A soft hinge penalty, not a hard cap on switches |
| Front/back/left zone | Mean service in a coordinate-selected subset plus 0.1 times fleet service | Expressible using the existing focus-set utility if a parser supplies the subset and suitable weights |
| Worst case | Horizon mean of the least-served robot's service | Requires a minimum-service objective beyond the current fixed-weight interface |
| Safety then data | Fleet service minus 10 times the violation fraction | A linear penalty in existing utility terms, not a hard or lexicographic safety guarantee |

For zones, the current reader recognises named robot IDs but does not resolve
the spatial predicate. These distinctions mean a binary "linear versus
nonlinear" explanation is incomplete. Keep historical E2 utilities unchanged;
evaluate any stricter semantics as a separately versioned experiment.

## Required comparison

Use a mixed mission set, with held-out scenes, held-out paraphrases and unseen
numeric thresholds or spatial predicates. Fix the data split, router rule or
training procedure, and latency/quality trade-off before evaluating.

Compare:

1. The existing fixed-weight intent reader + planner.
2. RadioJEV with the appropriate independently trained checkpoint.
3. A structured mission reader + deterministic constrained/conditional
   candidate evaluator. This is the stronger baseline needed to distinguish
   a semantic-interface limitation from a need for learned plan scoring.
4. An automatic router using mission text and declared evaluator capabilities.
   It must not read test utility, mission IDs, source-experiment labels or
   post-hoc regrets to choose a branch.
5. A perfect-information per-case branch choice as an explicitly unattainable
   ceiling, never the implemented routing policy.

Use the same plans and radio observations across arms. Persist per-case
predictions, route, ground-truth utility, chosen plan, constraint violations
and latency. Keep ground-truth mission fields available only to evaluation
and a clearly labelled gold-parser ablation. For mixed E1/E2 cases report
each family separately and a predeclared common normalised metric; do not
average E1's raw regret with E2's normalised regret.

A mission-capability check should select a supported evaluator, rather than
assume every threshold or zone requires RadioJEV. Specify handling for
unsupported or ambiguous missions before testing. Neither routing nor a low
regret score substitutes for the separate safety/actuation checks.

## Evidence needed before a paper claim

Train at least three independent model seeds, including independent upstream
E1/E2 checkpoints for each run. Merely evaluating the same weights under
three seeds is not independent training. Keep scene splits fixed for paired
comparisons and record every seed and checkpoint dependency.

Report per-seed results, paired differences and scene-cluster uncertainty:
multiple missions share each scene, so the current case-level intervals do
not capture training variation or scene clustering. Report accuracy of routing,
failure cases, quality/latency trade-offs and safety violations. Include all
branches actually invoked in end-to-end timing; distinguish first-use cost,
cached mission reads and per-decision computation on the same hardware.

The current E7 geometry-aware calibration also needs a paired planning sweep
on matched scenes before promotion. Its lower inaudible-path RMSE alone does
not establish lower plan regret or safe distant-interference prediction.
