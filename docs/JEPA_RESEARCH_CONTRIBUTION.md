# Contribution, novelty and the competitive bar

A research contribution can be a useful new combination, formulation,
capability or trade-off built from existing methods. It does not require a
previously unseen building block. The combination should deliver a meaningful
advance over closely related systems and explain why it works.

This is an interpretive companion to the [sealed results](PILOT_FASTLOOP_RESULTS.md).
It changes no registered hypothesis, numerical result or experimental evidence.

## The contribution to lead with

**Predictive fleet connectivity under genuinely delivered evidence and paid
computation:** combine a learned wireless world model, typed partial pilot
observations, measurement/forecast clock contracts and cost-aware control;
evaluate forecast quality and delivered robot commands separately.

JEPA, RadioMap, measurement-anchored prediction and multirate control can all
come from prior work. The candidate system contribution is their co-design
for the warehouse's actual information and action constraints:

| Constraint | Formulation | Capability or insight |
|---|---|---|
| Robots report only some measurements | Positive-only typed evidence; other links remain unknown | Learn from delivered pilots without inventing weak-link labels |
| Capture, delivery and forecast clocks differ | Explicit clock/availability contracts | Reuse predictions against the correct measurement and reject wrong-AP evidence |
| Neural forecasts arrive more slowly than local feedback | Slow AP planning plus cached local prediction feedback | Test predictive control without a neural forward at every local update |
| Scanning, handover and inference have costs | Actual-paid-cost control experiments | Test whether predictive improvements benefit command service under those costs |
| Prediction and decision quality can diverge | Common-input RF tests plus native packet tests | Identify model capability and ranking/control weaknesses separately |

The fast-control mechanism is a built capability, not a demonstrated independent
packet improvement: it did not improve the recorded point means. A useful
systems paper retains that distinction.

## Current evidence and trade-offs

The preselected combined JEPA package has **4.31% fewer missed commands** than
equally augmented RadioMap, passing registered adjusted packet and relative
engineering gates. Matched short-route RMSE is **3.34% lower**, lower at all
eight sites and passing its separate adjusted primary test. These are measured
advantages of the configured packages.

Neither combined-versus-own-base comparison qualifies. AP-ranking regret
favors RadioMap. JEPA uses **2.81× more summed forecast elapsed work**, and
unscheduled helper timing has multi-second outliers. Thus this study establishes
neither an inference-speed advantage nor real-time execution. Different map
priors remain, and fixed neural weights coexist with causal map fitting.

The contribution can be a useful system combination and a new empirical
perspective without claiming every component is new or universally superior.

## Establishing competitiveness with SOTA

SOTA is task and resource dependent. A valuable advance could offer better
quality at comparable cost, comparable quality with lower latency/energy/data
requirements, stronger tail behavior, or a capability existing systems lack.
It needs direct relevant competitors and matched evaluation budgets.

The next comparison should include a strong classical baseline selected on
development data, supervised temporal/graph predictors with the same inputs
and numerical prior, and closely related learned wireless field/world models
where their task definitions permit a faithful comparison. The present
RadioMap comparison does not establish superiority to all published methods.

Measure command loss, worst-robot behavior, silence, route/AP-ranking errors,
calibration, inference and tail latency, energy/compute, and sensing budgets.
Keep hardware, training data, adaptation history and allowed actions explicit.
Use independent training runs and held-out layouts for broader representation
claims. Predeclare a practically meaningful gain and tail constraints.

## Concrete next advances

1. **Wireless-native decisions:** AP-relative contrasts, calibrated uncertainty
   and cost-sensitive ranking, with continuity and worst-robot objectives.
2. **Observation-native learning:** partial masks, capture ages and
   association/delivery selection during training; a matched supervised control
   and identical numerical prior to test representation value.
3. **Execution-native inference:** shared/batched context computation,
   distillation/quantization and bounded control/logging work, followed by actual
   latency-tail measurements. Validation-cache speedups are orchestration
   improvements and must not be called faster model inference.

Lead the paper with the demonstrated capability and trade-off, explain the
useful combination, and attach each broader claim to a direct test. Strong
novelty does not require a new primitive.
