# Prospective controller and actual-cost runtime repair

48 confirmation cases on 8 whole sites. No training or objective selection occurred.

| Handover s | Contrast | Status | Mean loss difference | Statistical | Engineering | Experimental benefit |
|---|---|---|---:|---|---|---|
| 0.03 | repair_minus_legacy:radiomap | tested | -0.0006115 | True | False | False |
| 0.03 | repair_minus_legacy:frozen_fleet_jepa | tested | -0.0003073 | True | True | True |
| 0.03 | repaired_jepa_minus_repaired_radiomap | tested | -0.0000438 | False | True | False |
| 4 | repair_minus_legacy:radiomap | not_run_development_veto | not measured | False | False | False |
| 4 | repair_minus_legacy:frozen_fleet_jepa | not_run_development_veto | not measured | False | False | False |
| 4 | repaired_jepa_minus_repaired_radiomap | not_run_development_veto | not measured | False | False | False |

All six comparisons retain their multiplicity budget. Observed and conservative pending-command bounds must both show superiority. Exact-only fast arms are secondary diagnostics, never a post-test alternative candidate.

| Handover s | Family | Controller | Command loss | Worst decile | Max silence s | Video/original | Bulk Mbps |
|---|---|---|---:|---:|---:|---:|---:|
| 0.03 | radiomap | legacy | 0.0110490 | 0.0196250 | 1.42 | 0.96791 | 509.30938 |
| 0.03 | radiomap | exact_only | 0.0110490 | 0.0196250 | 1.42 | 0.96791 | 509.30684 |
| 0.03 | radiomap | repair | 0.0104375 | 0.0138333 | 1.90 | 0.96991 | 510.09114 |
| 0.03 | frozen_fleet_jepa | legacy | 0.0107010 | 0.0162500 | 1.93 | 0.96951 | 511.58266 |
| 0.03 | frozen_fleet_jepa | exact_only | 0.0107292 | 0.0165312 | 1.93 | 0.96951 | 511.58266 |
| 0.03 | frozen_fleet_jepa | repair | 0.0103938 | 0.0134375 | 1.93 | 0.97081 | 512.29162 |

Source/causality/paid-cost/offline replay checks: True. Profile-exceeded cases: 1 (diagnostic, not an older-study waiver).

Zero-margin engineering checks, cumulative losses, every case and secondary fast contrasts remain in the machine-readable report. Repetitions add no independent sites.

- New prospective actual-cost contract; older overrun vetoes and results remain unchanged.
- Controller-package effects do not establish a JEPA training-objective or architecture advantage.
- Offline fixed-schedule replay is not proof of freshly measured deadlines.
- Fresh live-cost repetitions are sensitivity observations, not extra independent sites.
- Shared procedural CAD, assumed sensing/public routes and fixed weights limit transfer.
- Admission/scorer CPU is not separately scheduled; no complete-stack real-time guarantee.
- Empirical endpoint/tail margins and finite silence do not establish production or indefinite safety.

## Replication scope

6 exact-schedule OFFLINE replays; 6 fresh actual-cost repetitions. The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites.
