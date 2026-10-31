# Compositional missions and radio representation audit

This campaign implements the remaining cloud work in the user's six-direction
research list. It preserves all original E1–E5/E7, v2 and horizon evidence.
The experiments use actual pretrained inference, actual compact-model training
and actual native simulator events. All radio environments are simulated.

## WirelessJEV contracts

`warehouse_v3/missions.py` compiles complete controlled-language contracts with
explicit switch budgets, declared teleoperation/video/ID groups, spatial
predicates, per-offset service floors, conditional priorities and ordered
objectives. For example:

> Switch at most 12 robots; require service >= 30% for teleoperated robots at every offset; minimise switches; then maximise throughput.

Every clause must match the grammar; unsupported text is rejected. Priorities
are lexicographic, not a hidden weighted sum. A condition uses observed current
service. Groups are resolved from application declarations and current poses,
then held fixed across the commitment. The tail objective averages the lowest
fraction at each offset; reported worst-decile metrics additionally examine
each robot's mean service across the evaluated interval.

`Reader` optionally executes pinned d3 to choose the contract family, checks it
against the complete literal compiler and abstains on disagreement. Its actual
abstentions are retained. Controlled-language results must not be reported as
successful d3 interpretations when that model abstained. Caller-provided typed
contracts bypass language inference and are labelled separately.

A follow-up clause reader verifies each instruction separately using actual d3
decisions, with a fresh canonical-statement audit. The whole-family failure is
preserved. The CLI defaults to the complete controlled grammar; neural clause
verification is explicit (`--reader clauses`) and its first-use cost is recorded.
This is not unrestricted-language support. Typed JSON accepts declared schema
defaults while rejecting unknown or conflicting fields. A separate audit applies
those defaults to unchanged Qwen outputs; it does not repair missing constraints
using labels or replace malformed generation.

`scripts/plan_warehouse_contract.py` performs local search with independent
visibility/budget/floor checks and writes proposals only. If no searched plan
is admissible it reports `infeasible_within_candidates`; this is not a proof of
global physical infeasibility. Teleoperation membership must be supplied by the
caller. Self-reported input origin is not authenticated. Lower forecasts use a
validation-derived marginal residual margin; this does not prove simultaneous
service floors or physical safety. No car command is sent.

## RadioJEV representation experiments

Four compact arms each have three independent training seeds: MLP, graph from
scratch, graph with masked observed-link pretraining and graph with JEPA-style
latent prediction. The last two use four pretraining and eight supervised
epochs; scratch controls use twelve supervised epochs. Actual update counts and
CPU training time are recorded. JEPA omits final episode transitions without a
next observation, so equal epoch totals do not mean identical update counts.

The masked target contains only actually reported simulated scan values. The
JEPA target is a subsequent observed state, encoded by an EMA teacher, with the
recorded intervening association as action context. Neither pretext objective
uses simulator reward labels. Supervised adaptation predicts six route-offset
service quantiles, critical-rate-proxy events, signal and a cached-semantic plan
score. Switch cost remains a declared four-second assumption. It is not a
learned or measured authentication/roaming distribution.

The loss and EMA utility adapts MIT-licensed V-JEPA 2 training code. Exact
upstream revision, hashes and licence are in `third_party/vjepa2/`. These are
new small radio encoders, not video weights or a reproduced V-JEPA 2 system.
LWM's model metadata and file listing at the reviewed revision contain no reuse
licence; its channel representation also differs from RSSI. No LWM code or
weights are copied or evaluated. Original masked-link reconstruction tests the
research idea without claiming LWM transfer.

Training reuses the verified 18-environment horizon observation dataset;
validation uses six environments. New tests hold out three environments in
each of ordinary, shifted AP/propagation layout, 200-robot and tripled bulk-demand
regimes. Warehouse dimensions and nominal rack CAD remain 120 by 80 metres;
this is not arbitrary-building generalisation. Every model seed is evaluated
on every test environment in its domain. Calibration and routing are fitted
only on validation and sealed before that study's test evaluation.

The frontier study separately compares the trained cached-semantic score heads,
actual dynamic-int8 operators and validation-selected routing at three latency
weights. It reports coverage, false feasibility and ordinal regret conditional
on feasible proposals. Abstention is charged during routing. It does not
establish a quality win merely by reducing coverage. End-to-end d3 coverage is
reported separately from the controlled-language numerical evaluation.

The original value-head arm only replaced the last soft priority. A separately
versioned full-priority scorer repairs that limitation with explicit group-aware
pooling and complete lexicographic rank targets. It compares identical text/no-text
heads across three frozen graph checkpoint chains on fresh environments.
Predicted floors, switch counts and visible targets remain independently checked.
Neither a correct guard nor an offline rank score qualifies a learned controller.

## Sensing and mapping

No-scan and scanned-but-unheard are distinct states. A scan-availability mask
prevents uncollected APs being fitted as below-floor measurements. Initial
history contains four capped scans from twelve robots. Uniform and
decision-aware policies receive the same extra current-scan budgets of 10 or
20; passive uses no extra scans. Query selection sees public beliefs only.
Scan costs of 20/100/500 milliseconds are assumptions, charged against realised
service and per-flow violations. The acquisition is an uncertainty/decision
potential heuristic, not exact information gain. Inaudible-path error is
reported alongside downstream outcomes, rather than declared solved.

An initial pilot used reset-valued previous-service features. Its report is
retained with a supersession receipt. The corrected run uses actually simulated
previous-interval service and supplies the reported sensing results.

## Packet-level audit and measurement

The adapted GPL ns-3 program follows committed routes, uses the same public
geometry, demands, assignments and fixed primary channels, and runs MinstrelHt
with NIST errors, VHT80/HT20, two spatial streams, seven-dB noise figure and
declared AP/client transmit powers. Critical packets use WMM priority six;
bulk uses best effort. The numerical evaluator's strict priority abstraction
is still different from Wi-Fi contention. The eight-second route prefix is a
separate packet audit, not a full thirty-second closed-loop qualification.

Native events provide retries, queues, busy intervals, generated/received bytes,
endpoint delays, 50-ms critical and 250-ms bulk deadline events, critical silence
and first critical packet arrival. RTS/CTS is compared under paired seeds.
External busy metadata parameterises independent burst traffic; it is not
measured airtime calibration. A separate two-client constructed geometry tests
hidden-transmitter sensitivity without calling it a 100-robot result.

Propagation still lacks rack/material calibration. Four-second application
muting models the cost assumption; it does not emulate real association,
authentication or DHCP. Muted demand is charged in service. Packets never
generated while muted are not invented in deadline denominators. Deadline
results and critical-rate proxies must remain separate.

A preliminary packet run retained the earlier 20-MHz/one-stream profile. It was
stopped, preserved and explicitly superseded by the matched-profile audit;
partial preliminary comparisons are never labelled a completed matched study.

The packet-telemetry models train on actual native counters with explicit
missing-denominator masks. They forecast next-second throughput and the event
that a received critical packet is late. This is not lost-packet probability.
Train/validation/test split by entire scene, with three test scenes and three
training seeds. Report class coverage before interpreting discrimination.
Every next-second bin is included. A bin with no critical arrival has no late
arrival event; it is not treated as healthy service. An earlier conditional
pilot omitted those bins and is retained separately, not presented as an
unconditional event probability.

`measurement.py` and `scripts/analyze_switch_timing.py` validate provided stage
events and clock domains. Missing authentication, DHCP or first-packet events
stay missing. No on-car event collector is claimed installed. Physical switch
distributions, hardware calibration and one-/small-/100-car qualification still
require actual access and measurements; see `WAREHOUSE_HARDWARE_DATA.md`.

## E3–E5 and compute

The completed v2 E3 dataset has all admission labels and actual Chronos/rule
comparisons; its negative results remain valid. The new four-site study adds
actual pooled shared training and actual FedAvg alongside initial and local
models. A release must improve on initial/local means and avoid per-site RMSE
and Brier regressions. Gates are not loosened to force acceptance.

Removal targets a shared research model whose weights actually incorporated
the forgotten site's examples. Compare approximate gradient-ascent removal
against retained-site retraining and three diagnostic attacks. Pooled training
is labelled central training, not federation. Rejected shared candidates are
not deployed. Clean retraining is a reference, not efficient unlearning or
erasure of retained audit copies.

Timing covers input decoding/validation, public belief, candidate construction,
numeric/neural prediction and execution checks on this CPU. First-use mission
interpretation is separated from cached control cost. Physical sensing and
network actuation are unmeasured, explicitly excluded. A real pinned same-size
Qwen baseline generates the complete contract; invalid/truncated output counts
as failure without label-guided repairs. These protocols do not establish a
foundation model, orders-of-magnitude compute win at matched capability or SOTA.

## Reproduction

```bash
uv sync --frozen
uv run --frozen python scripts/setup_warehouse_v3.py --build
uv run --frozen python scripts/run_warehouse_v3.py --out /workspace/training/new-v3
uv run --frozen python scripts/run_warehouse_packet_audit.py --v3 /workspace/training/new-v3 --out /workspace/training/new-packets
uv run --frozen python scripts/run_warehouse_v3_studies.py --v3 /workspace/training/new-v3 --packets /workspace/training/new-packets --out /workspace/training/new-studies
uv run --frozen python scripts/benchmark_warehouse_v3.py --run /workspace/training/new-v3 --out /workspace/training/new-generative --generative-only
uv run --frozen python scripts/benchmark_warehouse_v3.py --run /workspace/training/new-v3 --out /workspace/training/new-frontier
```

Use fresh directories for changed protocols. Preserve the exact executed-source
capsules, provenance, immutable report hashes and checkpoints. Completed studies
must not be restarted just to make the working tree match their older sources.
The implementation remains Wi-Fi-specific; see `PROTOCOL_BOUNDARY.md`.
