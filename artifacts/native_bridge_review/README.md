# Independent native bridge review

Passive review passed for all9cases/18native simulations:512source files and56sealed evidence members. No model constructor, forward, fit, RT solve or native simulation was invoked. The previous475-source/5161-evidence closure remains unchanged.

The fixed candidate is seed0 of the validation-selected **regression** branch. This is not evidence of a JEPA alignment win. Each comparison covers12seconds and four episodes per handover cost. The90ms scan every5seconds preserves the roughly1% deterministic control-deadline loss floor, with finite-window boundary effects.

| Handover | Map miss rate | Candidate miss rate | Change | Episodes harmed | Mean maximum silence, Map→candidate |
|---|---:|---:|---:|---:|---:|
| 0.03s | 1.0510% | 1.3542% | +0.3031 percentage points | 4/4 | 262.5→2637.5ms |
| 4s | 1.4344% | 1.3354% | -0.0990 percentage points | 2/4 | 1205.0→602.5ms |

Fast-handover outlier: seed90473001, robot7 (zero-based) made one AP4→AP3 move at1.50s. It missed200/240control packets, versus3/240for Map, and reached9.74s silence. Both first forecasts completed at1.50s. Later AP0 recovery proposals did not execute; the frozen cadence requires a successful control delivery, and this robot had none after2.25s. The stored AP3 forecast is optimistic relative to archived base SNR; that base excludes hidden fading/blockage. These facts identify the trace and recovery mechanism, without a counterfactual causal decomposition.

Candidate warm work exceeded the fixed0.25s allowance on all11warm jobs in every learned case. The audit verified actual overrun costs were paid. Parity forecasts/std/completion slots and entire native result bodies match exactly; the wiring case has no runtime overrun.

No deployment or packet-risk qualification follows. Native6.5GHz is explicit extrapolation, distinct from the RT6.0GHz task. Video/bulk/airtime/stop values are hash-bound native summary counters; misses, per-robot tails, gaps and cumulative trajectories were independently recomputed.

Files: [audit.json](audit.json), [read-only audit script](audit_native_bridge_independent.py), [outlier diagnosis](outlier_diagnosis.json).
