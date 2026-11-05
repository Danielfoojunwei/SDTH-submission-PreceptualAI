# Prospective pilot and fast-loop study

64 native confirmation cases and 16 matched forecast streams on 8 whole sites. No training or objective selection occurred.

| Contrast | Units | Status | Mean difference | Statistical | Packet engineering | Packet benefit | Matched accuracy advantage |
|---|---|---|---:|---|---|---|---|
| combined_minus_base:radiomap | command loss fraction | tested | 0.0003406 | False | False | False | False |
| combined_minus_base:frozen_fleet_jepa | command loss fraction | tested | -0.0009979 | False | False | False | False |
| combined_jepa_minus_combined_radiomap | command loss fraction | tested | -0.0020677 | True | True | True | False |
| matched_short_route_delta_rmse:jepa_minus_radiomap | dB | tested | -0.0775458 | True | None | False | True |

All four slots retain the 0.9875 upper-quantile multiplicity budget, including unrun slots. Packet superiority requires both observed and conservative pending-command bounds plus all unchanged engineering gates. Matched RF uses fully observed short route-delta RMSE on identical public history and query traces; its accuracy flag does not qualify a controller.

| Family | Controller | Command loss | Worst decile | Max silence s | Video/original | Bulk Mbps |
|---|---|---:|---:|---:|---:|---:|
| radiomap | base | 0.0476729 | 0.1337813 | 8.34 | 0.90887 | 460.34213 |
| radiomap | pilot | 0.0479667 | 0.1362917 | 8.36 | 0.90952 | 464.69170 |
| radiomap | fast | 0.0477125 | 0.1338750 | 8.34 | 0.90886 | 460.32090 |
| radiomap | pilot_fast | 0.0480135 | 0.1363958 | 8.36 | 0.90954 | 464.69366 |
| frozen_fleet_jepa | base | 0.0469437 | 0.1238438 | 8.24 | 0.91317 | 466.10843 |
| frozen_fleet_jepa | pilot | 0.0459115 | 0.1295208 | 8.34 | 0.90991 | 462.02419 |
| frozen_fleet_jepa | fast | 0.0469667 | 0.1239479 | 8.24 | 0.91309 | 466.07754 |
| frozen_fleet_jepa | pilot_fast | 0.0459458 | 0.1295938 | 8.34 | 0.90991 | 462.02820 |

## Descriptive 2x2 factorial diagnostics

Simple effects, balanced main effects and pilot-by-fast interactions use descriptive 95% paired-site intervals. No intermediate arm is selected and these intervals make no superiority claim.

| Family | Contrast | Mean loss difference | Descriptive 95% interval |
|---|---|---:|---|
| radiomap | factorial:radiomap:pilot_at_fast_off | 0.0002937 | [-0.0002406, 0.0008438] |
| radiomap | factorial:radiomap:pilot_at_fast_on | 0.0003010 | [-0.0002260, 0.0008438] |
| radiomap | factorial:radiomap:fast_at_pilot_off | 0.0000396 | [0.0000219, 0.0000583] |
| radiomap | factorial:radiomap:fast_at_pilot_on | 0.0000469 | [0.0000292, 0.0000635] |
| radiomap | factorial:radiomap:pilot_main_effect | 0.0002974 | [-0.0002344, 0.0008438] |
| radiomap | factorial:radiomap:fast_main_effect | 0.0000432 | [0.0000266, 0.0000604] |
| radiomap | factorial:radiomap:pilot_by_fast_interaction | 0.0000073 | [-0.0000021, 0.0000177] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:pilot_at_fast_off | -0.0010323 | [-0.0030010, 0.0008854] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:pilot_at_fast_on | -0.0010208 | [-0.0029896, 0.0009010] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:fast_at_pilot_off | 0.0000229 | [0.0000021, 0.0000438] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:fast_at_pilot_on | 0.0000344 | [0.0000167, 0.0000521] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:pilot_main_effect | -0.0010266 | [-0.0029943, 0.0008932] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:fast_main_effect | 0.0000286 | [0.0000099, 0.0000474] |
| frozen_fleet_jepa | factorial:frozen_fleet_jepa:pilot_by_fast_interaction | 0.0000115 | [0.0000021, 0.0000219] |

Native and matched source/causality/paid-cost/exact-replay integrity: True. Profile-exceeded cases: 0 (diagnostic).

Forecast arrivals: 3840; accepted jobs: 2752; skipped before computation: 1088; accepted jobs unconsumed: 0.
Maximum age at consumption: 2.0100000000000016; maximum held age across simulated slots: 4.340000000000003 seconds.

Zero-margin engineering checks, cumulative losses and every case remain in the machine-readable report. Native and matched repetitions add no independent sites.

- Separate enabling-stack experiment; all earlier recovery failures, nulls and gates are preserved.
- Ordinary outage utility, all-handover admission and the sustained-recovery watchdog remain shared.
- The four arms form a 2x2 factorial; secondary simple effects, balanced main effects and interactions are descriptive only.
- The final package is fixed; descriptive component intervals cannot select or qualify an intermediate arm.
- Whole-scan suppression changes delivered information and paid sensing; equal sensing airtime is not claimed.
- Skipped requests perform no forecast work; sparse source timestamps, not accepted-job ordinals, define forecast age.
- Native accepted-query RF diagnostics cover different schedules; only the separate identical-reference trace supports matched RF inference.
- Matched RF is fully observed, excludes horizon zero and long horizons, and has its own accuracy flag without packet engineering qualification.
- The frozen JEPA and RadioMap packages use different map pipelines/priors; this does not isolate the JEPA objective or prove a map-free architecture.
- Offline fixed-schedule replay is not proof of freshly measured deadlines.
- Fresh live-cost repetitions are sensitivity observations, not extra independent sites.
- Shared procedural CAD, assumed sensing/public routes and fixed weights limit transfer.
- Admission/scorer CPU is not separately scheduled; no complete-stack real-time guarantee.
- Empirical endpoint/tail margins and finite silence do not establish production or indefinite safety.

## Replication scope

8 exact-schedule OFFLINE replays; 8 fresh actual-cost repetitions. The former establish only replay under recorded availability; the latter are descriptive runtime sensitivity and add no independent sites. 16 matched offline forecasting sequences and 2 exact matched replays add no packet trials or independent sites.
