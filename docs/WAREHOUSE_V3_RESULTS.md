# Warehouse v3 results

These are simulated warehouse observations, actual pretrained-model inference and actual native ns-3 packet events. No physical car was contacted or qualified. Original E1–E5/E7, v2 and horizon evidence is unchanged. See `WAREHOUSE_V3.md` for the full protocol and limitations.

Actual d3 contract-family interpretation answered 4/10 author-written missions. The deterministic complete grammar remains a separately labelled baseline. Qwen complete-schema accuracy is 10.0%; invalid or truncated outputs count as failures without repairs. These interfaces differ and the language set is small.

A separately designed clause reader answered 12/12 fresh canonical composite statements. It verifies individual instructions while a complete compiler handles literals and order. The audit was designed after observing the family gate failure; it is not unrestricted language understanding or an independent physical result. Original failure receipts are retained.

An audit of the unchanged Qwen outputs accepts declared optional/default schema fields, reaching 10.0% exact contract match. The original strict score remains preserved. Defaults are static schema defaults; no labels recover missing constraints, no malformed JSON is repaired and no model inference is invented.

## Radio forecasts and hard-contract proposals

| Domain | Model | Service RMSE | Proposal coverage | False feasibility, all cases | Raw 80% interval coverage | Critical-proxy Brier |
|---|---|---|---|---|---|---|
| ordinary | numerical | 0.4109 | 60.0% | 0.0% | 100.0% | 0.1953 |
| ordinary | mlp | 0.3318 | 60.0% | 0.0% | 92.2% | 0.1584 |
| ordinary | graph_scratch | 0.3284 | 60.0% | 0.0% | 93.4% | 0.1562 |
| ordinary | graph_masked | 0.3321 | 60.1% | 0.1% | 94.8% | 0.1588 |
| ordinary | graph_jepa | 0.3267 | 60.0% | 0.0% | 95.0% | 0.1602 |
| layout | numerical | 0.3550 | 60.0% | 0.0% | 100.0% | 0.1719 |
| layout | mlp | 0.3131 | 60.0% | 0.0% | 95.5% | 0.1360 |
| layout | graph_scratch | 0.3103 | 60.0% | 0.0% | 95.6% | 0.1357 |
| layout | graph_masked | 0.3108 | 60.0% | 0.0% | 96.1% | 0.1417 |
| layout | graph_jepa | 0.3108 | 60.0% | 0.0% | 96.4% | 0.1431 |
| fleet200 | numerical | 0.3339 | 60.0% | 0.0% | 100.0% | 0.1535 |
| fleet200 | mlp | 0.2795 | 60.0% | 0.0% | 96.8% | 0.1240 |
| fleet200 | graph_scratch | 0.2766 | 60.0% | 0.0% | 96.9% | 0.1228 |
| fleet200 | graph_masked | 0.2778 | 60.0% | 0.0% | 97.4% | 0.1248 |
| fleet200 | graph_jepa | 0.2791 | 60.0% | 0.0% | 97.5% | 0.1246 |
| traffic | numerical | 0.3001 | 60.0% | 0.0% | 100.0% | 0.1679 |
| traffic | mlp | 0.2591 | 60.0% | 0.0% | 96.3% | 0.1254 |
| traffic | graph_scratch | 0.2572 | 60.0% | 0.0% | 96.4% | 0.1261 |
| traffic | graph_masked | 0.2574 | 60.0% | 0.0% | 97.0% | 0.1337 |
| traffic | graph_jepa | 0.2585 | 60.0% | 0.0% | 97.1% | 0.1370 |

Forecasting improvement over a public numerical prior does not establish a control advantage. Masked/JEPA pretraining must be compared against graph-from-scratch, including its matched training budget. Raw interval coverage is measured, not assumed to equal its nominal 80%. Floored missions can defer; coverage and feasibility must accompany regret. Candidate exhaustion does not prove global infeasibility.

Each learned row averages three independent model seeds, each evaluated on all three environments in its domain. They are not nine independent physical fleets. Dimensions/CAD are fixed; AP layouts, hidden propagation, traffic and fleet size vary. Cached-semantic, int8 and latency-aware router results are preserved separately under `frontier/` and share these holdout environments.

A separate full-priority ranker uses explicit group-aware graph pooling and trains on the complete soft-priority ordering. Its identical-architecture text/no-text ablation uses three checkpoint chains and new environments 180000–183002. Hard predicted floors and structural guards remain outside the learned scorer. The first raw case file mislabeled a feasible-set indicator as `gold_choice`; the frozen-model record audit reconstructs actual indices and verifies all quality fields unchanged. Corrected records are under `ranker_audit/cases/`, with original raw records retained under `ranker/cases/`. This audit is not another independent replication.

## Matched public-input native packet audit

The audit executed 48 native runs covering 40 distinct input/seed/RTS conditions. Identical policy inputs are repeated executions, not independent environments. The table uses three held-out scenes. A single predeclared JEPA training seed supplies the learned packet policy; it is not a three-model-seed packet comparison. All four methods have the same twelve-switch operator-policy budget.

| Policy | RTS/CTS | Service | Worst decile | Critical deadline miss fraction, generated packets | Mean maximum critical silence (s) |
|---|---|---|---|---|---|
| stay | False | 0.7150 | 0.0362 | 0.6431 | 6.937 |
| budgeted_rssi | False | 0.7135 | 0.0347 | 0.6318 | 8.000 |
| numerical | False | 0.7142 | 0.0606 | 0.5752 | 8.000 |
| radio_jepa_seed0 | False | 0.7065 | 0.0682 | 0.6213 | 8.000 |
| stay | True | 0.7913 | 0.0736 | 0.5720 | 8.000 |
| budgeted_rssi | True | 0.7661 | 0.0692 | 0.5681 | 8.000 |
| numerical | True | 0.7768 | 0.1025 | 0.5056 | 7.442 |
| radio_jepa_seed0 | True | 0.7616 | 0.1045 | 0.5550 | 8.000 |

PHY profile: NIST errors, VHT80/HT20, two streams, declared transmit powers, fixed primary channels and WMM priority six for critical traffic. Routes and public topology match; log-distance propagation still lacks rack/material calibration. An eight-second route prefix and assumed application muting do not reproduce actual authentication/DHCP or a full thirty-second control loop. Muted demand reduces service but does not manufacture packets in deadline denominators. Stored plan forecasts are numerical-belief forecasts for the selected assignments, including assignments selected by the learned model.

The earlier 20-MHz/one-stream pilot is retained as explicitly superseded partial evidence. A native-concurrency change preserves completed reports and the earlier controller sources; binary, seeds and scientific inputs remain unchanged.

## Sensing, shared learning and removal

Sensing results under `sensing/sensing.json.gz` use actual previous-interval simulated counters. Uniform and decision-aware policies share scan budgets; cost assumptions are charged in downstream service and per-flow violations. Never-scanned robots do not generate below-floor measurements. Inaudible-path error remains reported, not declared solved.

Pooled shared release gates accepted 0/3 candidates; actual FedAvg gates accepted 0/3. The removal target did incorporate the forgotten contribution. Approximate removal equals clean retained-data retraining in 0/3 seeds. Pooled training is not federation; rejected updates are not deployed. Three attack diagnostics do not certify erasure.

The new shared-removal pilot uses four sites, one 150-second episode per site/split and five counterfactual decision samples per episode. Attack samples are small and correlated; this is a contribution/removal reference, not definitive federated-unlearning evidence.

Packet-telemetry late-event test counts: {'late': 5447, 'not_late': 4153}. The event is any late received critical packet in the next one-second bin, not packet loss. Models use observed retries/queues/busy intervals with explicit denominator-validity masks. Native events remain simulations.

All bins are included, including those without a critical arrival (event false, not healthy-link evidence). The earlier outcome-conditioned pilot is retained separately. Validation-only temperature fitting is audited against the actual event on three test scenes; it reuses this holdout rather than supplying another replication.

| Packet predictor | Raw Brier | Temperature-scaled Brier | Scaled ECE | Throughput MAE (Mbps) |
|---|---|---|---|---|
| Persistence | 0.1286 | — | 0.1286 | 1.9387 |
| instant_packet_model | 0.1017 | 0.1003 | 0.0355 | 1.8228 |
| temporal_packet_model | 0.0864 | 0.0860 | 0.0180 | 1.7851 |

## Remaining acceptance boundaries

No model is declared SOTA, a vertical foundation model or a physical fleet controller from these results. The compositional CLI is proposal-only; learned controllers have not passed new closed-loop/hardware release gates. Actual switch distributions, scan interruptions, packet-service calibration and one-/small-/100-car trials require hardware access. The implementation remains Wi-Fi-specific.

All compressed JSON is recoverable byte-for-byte; original and compressed hashes are in `artifacts/warehouse_v3/manifest.json`. Checkpoints are finite and source capsules are verified. Superseded pilot evidence is preserved, not promoted to a successful completed experiment.
