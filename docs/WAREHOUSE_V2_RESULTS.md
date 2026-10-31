# Warehouse JEV v2 findings

Evidence: simulation, real pretrained-model inference, native ns-3 packet simulation and Sionna coded-PHY execution. No physical car test is claimed.

Original E1–E5/E7 reports are preserved. These are new protocols, not interchangeable reruns of the original experiments. Three independent compact-model seeds do not turn the historical large-model runs into multi-seed evidence.

## Structured missions

- controlled_language, test: 120/120 answered; normalized candidate regret 0.0000 on answered cases.
- controlled_language, novel: 120/120 answered; normalized candidate regret 0.0000 on answered cases.
- d3_typed_family, test: 120/120 answered; normalized candidate regret 0.0000 on answered cases.
- d3_typed_family, novel: 100/120 answered; normalized candidate regret 0.0000 on answered cases.

The exact simulator outcomes and common candidate set are available here. This establishes a stronger semantic-interface baseline; it does not establish prediction from real radio measurements. Historical E2 does not establish a learned-scorer advantage over this structured evaluator.

## Sensed-input models

Training uses 24 environments, validation 6, ordinary test 12, shifted-layout test 6, and shifted-layout 200-robot test 6. Inputs are noisy capped scan histories, public route/demand/AP metadata and reported current service. Future truth is confined to intervention labels.

| Test split | Model | Mean normalized regret across three seeds | Per-seed regret | Mean service RMSE |
|---|---|---|---|---|
| test | instant_mlp | 0.0504 | 0.0622, 0.0386, 0.0505 | 0.2547 |
| test | temporal_graph | 0.0654 | 0.0602, 0.0678, 0.0682 | 0.2519 |
| test | semantic_graph | 0.1916 | 0.2253, 0.1333, 0.2162 | 0.2568 |
| shift | instant_mlp | 0.0752 | 0.0767, 0.0721, 0.0767 | 0.2385 |
| shift | temporal_graph | 0.0894 | 0.0792, 0.0980, 0.0909 | 0.2349 |
| shift | semantic_graph | 0.1693 | 0.2083, 0.1152, 0.1846 | 0.2403 |
| fleet200 | instant_mlp | 0.0264 | 0.0264, 0.0264, 0.0264 | 0.2514 |
| fleet200 | temporal_graph | 0.0340 | 0.0356, 0.0356, 0.0309 | 0.2455 |
| fleet200 | semantic_graph | 0.0937 | 0.1386, 0.0408, 0.1018 | 0.2520 |

The compact non-graph baseline wins plan selection in this first protocol. Graph modelling slightly improves service prediction; the semantic value head does not establish an advantage. These offline counterfactual scores are not closed-loop guarantees.

## Closed-loop qualification and correction

The first v2 learned controllers performed poorly in 600 s rollouts. Their five-second, interruption-dominated training target was mismatched to the thirty-second control commitment. The failure is retained in `closedloop.json`.

A separately recorded correction learns uninterrupted service and charges the assumed four-second interruption analytically for a thirty-second commitment. It uses new rollout seeds 131000–131002. It still does not predict every intermediate route state.

| Policy | Mean service fraction | Worst-decile mean service | Mean reconnections per rollout | Mean critical-violation robot-steps |
|---|---|---|---|---|
| stay | 0.2031 | 0.0275 | 0.0 | 8808.0 |
| legacy_hysteresis | 0.5137 | 0.3310 | 892.0 | 3753.3 |
| strongest_scan | 0.5430 | 0.4005 | 1569.7 | 3750.3 |
| observed_load_balance | 0.4663 | 0.3159 | 1730.0 | 5585.3 |
| instant_mlp | 0.2494 | 0.0754 | 141.3 | 8049.3 |
| temporal_graph | 0.2352 | 0.0461 | 86.0 | 8342.7 |
| factorized_mlp | 0.5501 | 0.4111 | 1644.0 | 3983.0 |
| factorized_graph | 0.5430 | 0.4005 | 1569.7 | 3750.3 |

The factorized graph controller matches the strongest-scan policy exactly on these three rollouts. That is not evidence of a learned advantage. The factorized MLP differs on one rollout; increased fleet service does not establish a better safety/continuity trade-off. Compare reconnections and critical violations. Each model seed is paired with one rollout seed; neither universal benefit to all cars nor physical-car performance follows.

## Equal-budget sensing and calibration

Fifteen new environments; corrected and geometry-censored maps; uniform and decision-aware scanning; 0/10/20/40 extra robot scans after 12 initial scans. The 48-feature map protocol differs from historical E7. Scan-cost sensitivities use assumed durations, not measured interruptions.

| Map / acquisition / extra scans | Mean gap to best fixed candidate |
|---|---|
| corrected_v2/uniform/0 | 0.4354 |
| corrected_v2/uniform/10 | 0.3300 |
| corrected_v2/uniform/20 | 0.2385 |
| corrected_v2/uniform/40 | 0.2147 |
| corrected_v2/decision_aware/0 | 0.4354 |
| corrected_v2/decision_aware/10 | 0.3149 |
| corrected_v2/decision_aware/20 | 0.2270 |
| corrected_v2/decision_aware/40 | 0.2338 |
| geometry_censored/uniform/0 | 0.2485 |
| geometry_censored/uniform/10 | 0.1779 |
| geometry_censored/uniform/20 | 0.1554 |
| geometry_censored/uniform/40 | 0.2106 |
| geometry_censored/decision_aware/0 | 0.2485 |
| geometry_censored/decision_aware/10 | 0.1675 |
| geometry_censored/decision_aware/20 | 0.2059 |
| geometry_censored/decision_aware/40 | 0.2197 |

## Transfer, multitask learning and site removal

- One new site, 0 adaptation examples: mean candidate regret 0.0818 across three initial model seeds.
- One new site, 10 adaptation examples: mean candidate regret 0.0883 across three initial model seeds.
- One new site, 100 adaptation examples: mean candidate regret 0.0860 across three initial model seeds.
- Repaired admission label counts: `{"test": {"hold": 129, "lower": 523, "raise": 128}, "train": {"hold": 260, "lower": 1010, "raise": 290}, "validation": {"hold": 67, "lower": 252, "raise": 71}}`.
- Admission includes HOLD and heterogeneous best-effort priorities; all thirteen current levels are counterfactual coverage, not field prevalence. Read balanced accuracy and per-label recall with accuracy.
- Actual frozen Chronos-Bolt forecast MAE: 3.059 Mbps; last-value baseline: 3.771 Mbps. These are synthetic traffic windows.
- Shared and separate small task cores are both trained with three seeds. Per-task results, rules, exact admission oracle, model sizes and negative transfer remain in `multitask.json`.
- Site adaptation: 0/3 shared candidates pass the unchanged no-site-harm release gate.
- Shared-model removal: 3/3 one-round independent-adapter recomputations match a clean retained-site retrain bit for bit. This result concerns the unconditional FedAvg research baseline, not a rejected gated release or arbitrary iterative FL.
- Three membership-attack signals and retained-site service are reported. This is not certified erasure of data, personal states or checkpoint copies.

## Actual PHY/MAC and generative dependencies

- Sionna 2.2.0: 27 coded QAM/SNR points, with raw bit/block errors. The 3GPP LDPC chain is a generic PHY study, not an IEEE 802.11 implementation.
- ns-3 3.49: 3,618 NIST payload-success curve points, integrated into the warehouse evaluator with 0/3/6 dB rate-margin sensitivity. The existing airtime/MAC-efficiency abstraction remains.
- Native packet simulator: 12 100-robot, eight-second policy comparisons from 9 distinct topology/association/seed simulations. Identical comparisons explicitly share raw results. Per-robot throughput, unreceived packets and delay histograms are retained. Static log-distance channels and default single-stream packet devices do not validate rack propagation, mobility or handover.
- Actual Qwen3.5-0.8B active-semantics accuracy: 0.583; real d3 typed reader in the same CPU session: 0.958. Raw Qwen outputs populate irrelevant fields, so strict schema accuracy is 0.000; semantic scoring compares only fields used by the chosen kind, without correcting outputs. Qwen generates fields; d3 selects a family and extracts literals, so interfaces differ.
- LWM and JEPA are research references, not checkpoints represented as evaluated here. ANSpect proprietary algorithms were not reproduced.

## Remaining qualification

Real RSSI/throughput/outage traces, switch-stage events, one-car qualification and multi-car trials require actual hardware access. There are no configured car credentials or private-network connection in this cloud. The external Claude Studio has not been republished.

The new mechanisms and experiments are implemented and executed. These results do not yet establish SOTA, a general wireless foundation model, or benefit to every physical robot.
