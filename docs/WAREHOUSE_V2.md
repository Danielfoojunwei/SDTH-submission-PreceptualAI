# Warehouse JEV research campaign v2

Authorised scope: structured mission interpretation and planning; sensed-input
temporal radio modelling; decision-aware sensing; per-robot service; realistic
PHY/MAC validation; independent training seeds and transfer; E3–E5 redesign.
The original E1–E5/E7 evidence is immutable and is not relabelled as v2 evidence.

## Evidence contract

All generated trajectories and labels must carry `source: simulation`. No
generated data is a hardware measurement. External simulators must execute;
missing dependencies, failed builds and unavailable devices are errors, never
replaced with plausible numbers. Every run records configuration, source hashes,
dependency versions, seeds and checkpoint lineage. Learned controllers receive
only declared observations; hidden simulation state is used for labels and
evaluation, separately from the policy input.

## Work packages and acceptance

1. A validated mission schema, text-only semantic reader, constrained candidate
   evaluator and capability router. Preserve historical E2 soft semantics in a
   separate compatibility benchmark; hard constraints belong to a new version.
   An infeasible candidate set is not a certificate of global infeasibility.
2. Train temporal graph and non-graph baselines on noisy, capped scan histories
   and simulated action interventions. Predict service, critical-link failures
   and signal; evaluate on unseen environments and fleet sizes. Run three
   independent seeds; save predictions and checkpoints.
3. Compare passive, uniform and decision-aware measurements under identical
   measurement budgets. Charge sensing interruption explicitly and report
   downstream outcomes; fit uncertainty only on validation environments.
4. Report individual-robot service and violations, worst-decile service,
   reconnections and policy timing. No claim that all 100 cars benefit follows
   from a fleet average.
5. Execute Sionna PHY experiments and ns-3 Wi-Fi packet simulations from pinned
   upstream sources. Document coding/channel assumptions and modifications.
   Neither cellular coding nor an independent WLAN experiment validates a
   whole warehouse controller without a matched integration experiment.
6. Evaluate quality, latency, compute and cross-site adaptation. Repair E3
   admission label coverage and compare traffic models against strong rules.
   E4 must compare against the unadapted checkpoint and local training. E5 must
   first demonstrate accepted contribution, then compare removal with a clean
   retrain and multiple attacks. Do not loosen release gates to manufacture an
   accepted update.

## Hardware boundary

The cloud has no configured private network or car credentials. Physical
measurements, one-car qualification and multi-car trials require real access
and remain explicitly unexecuted until it exists. Existing supervised switch
checks in `ONE_CAR_TEST.md` continue to apply. Stage timestamps must come from
actual events; absent authentication/IP/first-packet events remain missing.

## External code

Use `scripts/setup_warehouse_v2.py` to acquire pinned source and build tools.
Source downloads live outside the research repository. Adaptations and licence
notices are tracked under `third_party/`; large data and weights stay outside
Git. See the generated upstream manifest for exact versions and hashes.

## Reproduce the executed campaign

```bash
uv sync --frozen
uv run python scripts/setup_warehouse_v2.py --build
uv run python scripts/run_warehouse_v2.py --out /workspace/training/new-core --stages semantic training
uv run python scripts/run_warehouse_v2_studies.py --core-run /workspace/training/new-core --out /workspace/training/new-studies
uv run python scripts/run_warehouse_v2_factorized.py --core-run /workspace/training/new-core --out /workspace/training/new-factorized
uv run python scripts/benchmark_warehouse_strongest.py --out /workspace/training/new-factorized/strongest_baseline.json
uv run python scripts/benchmark_warehouse_generative.py --out /workspace/training/new-generative
uv run python scripts/archive_warehouse_v2.py --core /workspace/training/new-core --studies /workspace/training/new-studies --factorized /workspace/training/new-factorized --generative /workspace/training/new-generative
```

Use fresh output directories for a changed core protocol. The core controller
refuses source/configuration mismatches. Exact executed source snapshots are
retained in each workspace run, including the original core implementation
before adding the optional fixed-site environment argument. Extension retries
retain their own source attempts and completed report hashes; a failed stage
is restarted without inventing a result or discarding earlier evidence.

The factorized study was designed after the first v2 closed-loop failure and
uses new rollout seeds. It learns uninterrupted service, then charges the
assumed switch interruption analytically. Its five-second physical prediction
is not represented as a complete thirty-second temporal world model.

See `WAREHOUSE_PHY.md` for which PHY experiments actually execute and which
ANSpect-related waveform direction remains a separate design; see
`WAREHOUSE_HARDWARE_DATA.md` for the field-data boundary.
