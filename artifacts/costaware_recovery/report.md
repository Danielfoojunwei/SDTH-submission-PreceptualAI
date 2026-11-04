# Prospective slow cost-aware recovery

32 confirmation cases on 8 whole sites. No training or objective selection occurred.

| Handover s | Contrast | Status | Mean loss difference | Statistical | Engineering | Experimental benefit |
|---|---|---|---:|---|---|---|
| 4 | repair_minus_legacy:radiomap | tested | -0.0003125 | False | True | False |
| 4 | repair_minus_legacy:frozen_fleet_jepa | tested | -0.0005490 | False | True | False |
| 4 | repaired_jepa_minus_repaired_radiomap | tested | -0.0038583 | True | False | False |

All three comparisons retain their multiplicity budget. Observed and conservative pending-command bounds must both show superiority. Ordinary outage utility, all-handover admission and scoring remain shared; the watchdog rule was fixed before data.

| Handover s | Family | Controller | Command loss | Worst decile | Max silence s | Video/original | Bulk Mbps |
|---|---|---|---:|---:|---:|---:|---:|
| 4 | radiomap | legacy | 0.0621813 | 0.1674792 | 23.24 | 0.90912 | 451.66852 |
| 4 | radiomap | repair | 0.0618688 | 0.1623438 | 8.34 | 0.90903 | 453.13502 |
| 4 | frozen_fleet_jepa | legacy | 0.0585594 | 0.1661354 | 19.03 | 0.90364 | 456.60298 |
| 4 | frozen_fleet_jepa | repair | 0.0580104 | 0.1595729 | 10.03 | 0.90697 | 457.13109 |

Source/causality/paid-cost/offline replay checks: True. Profile-exceeded cases: 0 (diagnostic, not an older-study waiver).

Zero-margin engineering checks, cumulative losses and every case remain in the machine-readable report. Repetitions add no independent sites.

- Separate slow-only recovery experiment; earlier unfiltered-recovery failures and vetoes are preserved.
- Ordinary outage utility and all-handover admission remain shared; only the registered persistent-gap watchdog differs.
- Recovery effects with frozen models do not establish a JEPA training-objective or architecture advantage.
- Offline fixed-schedule replay is not proof of freshly measured deadlines.
- Fresh live-cost repetitions are sensitivity observations, not extra independent sites.
- Shared procedural CAD, assumed sensing/public routes and fixed weights limit transfer.
- Admission/scorer CPU is not separately scheduled; no complete-stack real-time guarantee.
- Empirical endpoint/tail margins and finite silence do not establish production or indefinite safety.

## Replication scope

4 exact-schedule OFFLINE replays; 4 fresh actual-cost repetitions. The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites.
