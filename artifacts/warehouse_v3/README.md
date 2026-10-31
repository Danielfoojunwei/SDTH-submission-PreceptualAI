# Verified warehouse v3 evidence

These are warehouse simulations, actual pretrained inference, actual compact
training and native ns-3 packet events. No physical car was qualified or changed.
The implementation remains Wi-Fi-specific.

- `core/`: twelve independent compact checkpoints, actual d3 contexts, sealed
  calibration/routing and four-domain evaluation. Three seeds per architecture
  are crossed with three held-out environments per domain.
- `clauses/`: actual clause-by-clause d3 verification on twelve fresh canonical
  statements. The whole-family reader's original 4/10 result remains in `core/`.
  They use different text sets, so this is not a paired accuracy comparison.
- `packets/`: 48 executed native runs, 40 unique input/seed/RTS conditions,
  complete inputs, raw per-robot packet/queue/retry/deadline events and hashes.
  The learned packet policy uses one predeclared JEPA model seed. Repeated
  identical inputs are not independent environments.
- `packet_learning/`, `packet_calibration/`: unconditional next-second event
  prediction, actual raw/scaled calibration metrics and crossed scene/seed
  intervals. No-arrival bins are included; the event is lateness, not loss.
- `sensing/`: corrected equal-budget sensing using actual simulated previous
  service and a distinction between missing scans and unheard APs.
- `shared/`: initial/local/pooled/FedAvg comparisons and actual-contribution
  removal versus retraining. All release gates reject; attack scores do not
  certify deletion. This is a small four-site pilot.
- `ranker/`, `ranker_audit/`: full-priority, group-aware text/no-text heads on
  new environments. The first raw `gold_choice` field was a feasibility
  indicator; the audit reconstructs actual choices and verifies all 7,200
  quality rows unchanged. Use the corrected `ranker_audit/cases/` records.
- `frontier/`, `cost/`: cached-semantic, actual int8, routing, per-domain full
  local pipeline timing and crossed environment/model-seed intervals.
- `generative/`, `schema_audit/`: unchanged actual Qwen outputs and separately
  labelled strict/default-aware schema evaluation, without label-guided repair.
- `superseded_packets/`, `superseded_packet_learning/`: retained diagnostic
  pilots. Partial mismatched PHY runs and conditional-on-arrival models are
  not promoted to primary evidence. Controller operational retries have their
  own attempt/source receipts.
- `examples/`: actual proposal-only executions on a labelled simulated input,
  including a feasible contract, a refused floor and actual d3 interpretation.
- `native_build_verification.json`: repeated build reproduces the executed
  binary exactly; three relevant upstream suites pass.

`manifest.json` records original/compressed report hashes, source capsules and
42 finite checkpoint files, including six superseded conditional-pilot models.
This is not 42 independent successful experiments. JSON gzip files recover
their original bytes with `gzip -dc`. Startup source capsules are retained
where controllers captured them; `reproduction_source_map.json` explicitly
labels helper hashes captured after execution rather than calling them a
pre-execution seal.

Forecast-plot shading is the range across three model seeds. The packet bars
show validation-temperature-scaled Brier means; paired intervals and raw scores
are retained separately. These plots are not physical-radio evidence.

Protocols, interpretations and outstanding acceptance boundaries:
`docs/WAREHOUSE_V3.md`, `docs/WAREHOUSE_V3_RESULTS.md`,
`docs/WAREHOUSE_SCOPE_CHECKLIST.md`, `docs/PROTOCOL_BOUNDARY.md`.
