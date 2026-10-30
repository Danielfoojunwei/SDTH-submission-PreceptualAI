# Cloud RadioJEV full-run evidence

Seven completed controller stages are archived from the CPU full run,
seed 0, with smoke mode disabled. E3 is evaluating and E4/E5 is queued;
this directory is not a claim that all training has finished.
The original Mac reports in `artifacts/radiojev/` are unchanged.

| E1 arm | Test regret ± 95% CI (n=200) | New-mission regret ± 95% CI (n=240) | Test latency |
|---|---|---|---|
| radio_lora | 0.1068 ± 0.0240 | 0.0721 ± 0.0136 | 4,310 ms |
| radio_frozen | 0.1069 ± 0.0228 | 0.0837 ± 0.0194 | 4,622 ms |
| intent | 0.0140 ± 0.0066 | 0.0251 ± 0.0074 | 468 ms |
| no_llm | 0.00042 ± 0.00018 | Not evaluated | 2.15 ms |

| E2 arm | Test normalized regret ± 95% CI (n=120) | New-mission normalized regret ± 95% CI (n=120) | Test latency |
|---|---|---|---|
| radio_q | 0.1170 ± 0.0370 | 0.0779 ± 0.0253 | 6,420 ms |
| e1_zero | 0.2905 ± 0.0594 | 0.3288 ± 0.0693 | 7,390 ms |
| intent | 0.1987 ± 0.0461 | 0.3482 ± 0.0674 | 591 ms |

These are different regret definitions for E1 and E2. Intervals are over
evaluation cases, not independent training seeds; no across-seed robustness
is established. The numeric-only arm has no language-generalisation test.
The E1 intent arm has lower observed regret than the native-radio arms.
E2 radio-Q has lower observed normalized regret than its baselines, with
substantially higher CPU latency. All evidence is synthetic.

Each JSON is an unchanged completed-stage report. `provenance.json` records
the original run manifest, stage completion times, report checksums and
verified checkpoint checksums. All 12 RadioJEV source files matched the
captured run hashes at export; checkpoints loaded with `weights_only=True`
and all tensors were finite. Checkpoint binaries and scene caches remain
in `/workspace/training/radiojev/full`, outside Git.
