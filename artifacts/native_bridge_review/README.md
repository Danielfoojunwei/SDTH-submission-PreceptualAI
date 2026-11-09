# Independent native bridge review

Passive review passed for all 9 cases / 18 native simulations: 512 source files and 56 sealed evidence members. No model constructor, forward, fit, RT solve or native simulation was invoked. The previous 475-source / 5,161-evidence closure remains unchanged.

The fixed candidate is seed 0 of the validation-selected **regression** branch. This is not evidence of a JEPA alignment win. Each comparison covers 12 seconds and four episodes per handover cost. The 90 ms scan every 5 seconds preserves the roughly 1% deterministic control-deadline loss floor, with finite-window boundary effects.

| Handover | Map miss rate | Candidate miss rate | Change | Episodes harmed | Mean maximum silence, Map→candidate |
|---|---:|---:|---:|---:|---:|
| 0.03s | 1.0510% | 1.3542% | +0.3031 percentage points | 4/4 | 262.5→2637.5ms |
| 4s | 1.4344% | 1.3354% | -0.0990 percentage points | 2/4 | 1205.0→602.5ms |

Fast-handover outlier: seed 90473001, robot 7 (zero-based) made one AP4→AP3 move at 1.50 s. It missed 200/240 control packets, versus 3/240 for Map, and reached 9.74 s silence. Both first forecasts completed at 1.50 s. Later AP0 recovery proposals did not execute; the frozen cadence requires a successful control delivery, and this robot had none after 2.25 s. The stored AP3 forecast is optimistic relative to archived base SNR; that base excludes hidden fading/blockage. These facts identify the trace and recovery mechanism, without a counterfactual causal decomposition.

Candidate warm work exceeded the fixed 0.25 s allowance on all 11 warm jobs in every learned case. The audit verified actual overrun costs were paid. Parity forecasts/std/completion slots and entire native result bodies match exactly; the wiring case has no runtime overrun.

No deployment or packet-risk qualification follows. Native 6.5 GHz is explicit extrapolation, distinct from the RT 6.0 GHz task. Video/bulk/airtime/stop values are hash-bound native summary counters; misses, per-robot tails, gaps and cumulative trajectories were independently recomputed.

Files: [audit.json](audit.json), [read-only audit script](audit_native_bridge_independent.py), [outlier diagnosis](outlier_diagnosis.json).
