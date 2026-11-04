# Prospective JEPA enabling stack

64 confirmation cases on 8 whole sites. No training or objective selection occurred.

| Handover s | Contrast | Status | Mean loss difference | Statistical | Engineering | Experimental benefit |
|---|---|---|---:|---|---|---|
| 4 | final_minus_fifo:radiomap | tested | -0.0119823 | True | True | True |
| 4 | final_minus_fifo:frozen_fleet_jepa | tested | -0.0093521 | True | True | True |
| 4 | final_jepa_minus_final_radiomap | tested | 0.0011198 | False | False | False |

All three comparisons retain their multiplicity budget. Observed and conservative pending-command bounds must both show superiority. The final package was fixed before data; no intermediate arm is selected. Shared control admission and watchdog are unchanged.

| Handover s | Family | Controller | Command loss | Worst decile | Max silence s | Video/original | Bulk Mbps |
|---|---|---|---:|---:|---:|---:|---:|
| 4 | radiomap | fifo | 0.0489625 | 0.1476354 | 12.42 | 0.91428 | 465.56474 |
| 4 | radiomap | latest | 0.0490115 | 0.1475938 | 12.42 | 0.91445 | 465.51787 |
| 4 | radiomap | latest_single | 0.0419698 | 0.1193646 | 8.37 | 0.93119 | 472.54165 |
| 4 | radiomap | latest_single_sense | 0.0369802 | 0.1155208 | 8.37 | 0.93236 | 472.83577 |
| 4 | frozen_fleet_jepa | fifo | 0.0474521 | 0.1496771 | 8.35 | 0.91217 | 465.34894 |
| 4 | frozen_fleet_jepa | latest | 0.0491250 | 0.1466771 | 12.37 | 0.91300 | 465.22256 |
| 4 | frozen_fleet_jepa | latest_single | 0.0424302 | 0.1213333 | 8.37 | 0.93308 | 473.47808 |
| 4 | frozen_fleet_jepa | latest_single_sense | 0.0381000 | 0.1169063 | 8.37 | 0.93108 | 474.74500 |

## Nested component diagnostics

These are descriptive 95% paired-site intervals. Intervention order is fixed; this is not a factorial decomposition and no intermediate arm is selected.

| Family | Conditional component contrast | Mean loss difference | Descriptive 95% interval |
|---|---|---:|---|
| radiomap | nested:radiomap:latest_minus_fifo | 0.0000490 | [-0.0000208, 0.0001667] |
| radiomap | nested:radiomap:latest_single_minus_latest | -0.0070417 | [-0.0099260, -0.0045072] |
| radiomap | nested:radiomap:latest_single_sense_minus_latest_single | -0.0049896 | [-0.0058740, -0.0041469] |
| frozen_fleet_jepa | nested:frozen_fleet_jepa:latest_minus_fifo | 0.0016729 | [0.0002770, 0.0036292] |
| frozen_fleet_jepa | nested:frozen_fleet_jepa:latest_single_minus_latest | -0.0066948 | [-0.0099333, -0.0039802] |
| frozen_fleet_jepa | nested:frozen_fleet_jepa:latest_single_sense_minus_latest_single | -0.0043302 | [-0.0054583, -0.0033177] |

Source/causality/paid-cost/offline replay checks: True. Profile-exceeded cases: 1 (diagnostic, not an older-study waiver).

Forecast requests: 3840; accepted jobs: 3024; skipped before computation: 816; accepted jobs unconsumed before trial end: 136.
Maximum age at consumption: 17.979999999999997; maximum held age across simulated slots: 18.97 seconds.

Zero-margin engineering checks, cumulative losses and every case remain in the machine-readable report. Repetitions add no independent sites.

- Separate enabling-stack experiment; all earlier recovery failures, nulls and gates are preserved.
- Ordinary outage utility, all-handover admission and the sustained-recovery watchdog remain shared.
- Four nested arms are not a factorial experiment; conditional component effects depend on intervention order.
- The final package is fixed; descriptive component intervals cannot select or qualify an intermediate arm.
- Whole-scan suppression changes delivered information and paid sensing; equal sensing airtime is not claimed.
- Skipped requests perform no forecast work; sparse source timestamps, not accepted-job ordinals, define forecast age.
- Accepted-query RF diagnostics cover different schedules and are not matched accuracy endpoints.
- Stack effects with frozen models do not establish a JEPA training-objective or architecture advantage.
- Offline fixed-schedule replay is not proof of freshly measured deadlines.
- Fresh live-cost repetitions are sensitivity observations, not extra independent sites.
- Shared procedural CAD, assumed sensing/public routes and fixed weights limit transfer.
- Admission/scorer CPU is not separately scheduled; no complete-stack real-time guarantee.
- Empirical endpoint/tail margins and finite silence do not establish production or indefinite safety.

## Replication scope

8 exact-schedule OFFLINE replays; 8 fresh actual-cost repetitions. The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites.
