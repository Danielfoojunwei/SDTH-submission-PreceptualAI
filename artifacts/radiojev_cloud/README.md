# Cloud RadioJEV full-run evidence

Eight completed controller stages are archived from the CPU full run,
seed 0, with smoke mode disabled. E4/E5 is running after a checkpoint-loader fix;
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

E3's raw report is `e3.json`. Association normalized regret changes from
0.1170 to 0.1046; DSA greedy reward is 1.000 over one 500-step evaluation
episode versus 0.263 / 0.384 for the two MLP baselines. Traffic accuracy
(trend / event / volatility) is 0.369 / 0.231 / 0.338, below the corresponding
rule baselines of 0.708 / 0.738 / 0.923. This is mixed multitask evidence.
The label audit in `e3_label_audit.json` found that all 106 admission training
and 54 test cases are labelled "lower". Admission's 100% test accuracy equals
the constant-label baseline and is not evidence of policy generalisation.
Reproduce this audit with `docs/experiments/e3_label_audit.py`; it does not
load a model, change labels or rerun training.

Each stage JSON is an unchanged completed-stage report. `provenance.json` records
the original run manifest, stage completion times, report checksums and
verified checkpoint checksums. All 12 RadioJEV source files matched the
captured run hashes at the initial E1/E2 export and during completed E3;
checkpoints loaded with `weights_only=True`
and all tensors were finite. Checkpoint binaries and scene caches remain
in `/workspace/training/radiojev/full`, outside Git.

E4/E5 initially stopped before training because the E3 checkpoint contains
`heads`, while the E2 warm-start loader expects `encoder`. The repaired
handoff restores the full E3 checkpoint directly. The failed attempt is
preserved locally, and `e45_resume_provenance.json` records the source hashes,
versioned controller and E3 checkpoint used for the resumed stage. No E1–E3
stage was rerun or changed to make this retry succeed.
