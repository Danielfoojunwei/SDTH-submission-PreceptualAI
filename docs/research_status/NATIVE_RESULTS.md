# Native warehouse service diagnostic

The learned field increased missed commands in **all four 0.03 s handover
episodes**. With 4 s handovers its mean miss fraction improved, but two of
four episodes still worsened. It also required more forecast computation.
These results do not qualify a controller or establish a JEPA benefit.

The candidate is **`regression-seed0`**, fixed in advance as seed 0 of the
validation-selected adaptive branch. It has no cross-view consistency loss.
Its checkpoint is bound by the [weight receipt](../../artifacts/native_bridge/weights.json);
native results did not choose its branch, seed or weights.

## What ran

Four new episode seeds, 100 simulated robots, 12 seconds and two handover
costs produce **eight paired comparisons**. Every arm releases 24,000 control
packets: 600 bytes at 20 Hz, with a 50 ms deadline. A separate strong-Map
SNR→RSSI→SNR wiring comparison brings the total to **18 native simulations**.
Its forecast hashes, completion slots and complete native result bodies match
exactly. That wiring proof is separate from learned performance.

Both arms use the same frozen native packet kernel, held-action utility,
gate/admission hooks, scan capacity, proposal expiry and completion accounting.
RadioMap uses its full 60-bin received history; the learned field selects at
most 64 balanced AP/source-band tokens from that same permitted history
capacity. Actions can change later report deliveries. Robot trajectories
remain exogenous; this is a radio/queue feedback loop, not motion validation.

Source was remotely published before fresh world generation. The
[raw summary](../../artifacts/native_bridge/summary.json),
[passing passive audit](../../artifacts/native_bridge/verification/audit.json)
and [closure](../../artifacts/native_bridge/verification/manifest.json) bind
**512 source and 56 evidence files**. All eight receipts and compressed raw
packet results remain in [the case directory](../../artifacts/native_bridge/cases).

## Every episode pair

All arrows below mean **RadioMap → learned regression**. Misses are raw fleet
counts out of 24,000. “Worst decile” averages the ten robots with the highest
individual miss fractions. Maximum gap is the largest per-robot delivery gap
in the episode. CPU columns total twelve issued forecast jobs, in seconds;
actual work and paid compute allowance are different quantities.

### Handover interruption: 0.03 seconds

| Seed | Misses | Worst-decile miss, % | Maximum gap, ms | Actual forecast CPU, s | Paid forecast compute, s |
|---|---:|---:|---:|---:|---:|
| 90473001 | 256 → 537 | 1.792 → 13.417 | 240 → **9,740** | 0.928 → 3.635 | 3.750 → 4.352 |
| 90473002 | 243 → 244 | 1.250 → 1.250 | 130 → 130 | 1.260 → 4.117 | 3.750 → 4.761 |
| 90473003 | 270 → 278 | 2.208 → 2.500 | 540 → 540 | 1.209 → 5.149 | 3.750 → 5.808 |
| 90473004 | 240 → 241 | 1.292 → 1.292 | 140 → 140 | 0.918 → 3.399 | 3.750 → 4.123 |

### Handover interruption: 4 seconds

| Seed | Misses | Worst-decile miss, % | Maximum gap, ms | Actual forecast CPU, s | Paid forecast compute, s |
|---|---:|---:|---:|---:|---:|
| 90473001 | 496 → 398 | 11.667 → 7.500 | 4,010 → 1,630 | 1.047 → 3.732 | 3.750 → 4.447 |
| 90473002 | 243 → 244 | 1.250 → 1.292 | 130 → 130 | 0.947 → 3.731 | 3.750 → 4.425 |
| 90473003 | 397 → 401 | 7.458 → 7.625 | 540 → 540 | 0.962 → 3.646 | 3.750 → 4.338 |
| 90473004 | 241 → 239 | 1.292 → 1.250 | 140 → 110 | 0.902 → 3.693 | 3.750 → 4.420 |

## Equal-episode means, kept separate by handover cost

| Endpoint | 0.03 s: Map → regression | 4 s: Map → regression |
|---|---:|---:|
| Miss fraction | 0.01051042 → 0.01354167 | 0.01434375 → 0.01335417 |
| Paired miss-fraction difference | **+0.00303125** | **−0.00098958** |
| Episodes with more misses | **4/4** | **2/4** |
| Worst-decile miss fraction | 0.01635417 → 0.04614583 | 0.05416667 → 0.04416667 |
| Episodes with worse worst-decile fraction | 2/4 | 2/4 |
| Mean episode maximum gap, ms | 262.5 → 2,637.5 | 1,205.0 → 602.5 |
| Episodes with larger maximum gap | 1/4 | 0/4 |
| Actual forecast CPU, s per 12 jobs | 1.079 → **4.075** | 0.964 → **3.700** |
| Paid forecast compute, s | 3.750 → 4.761 | 3.750 → 4.407 |
| Forecast service including transport, s | 9.750 → 10.761 | 9.750 → 10.407 |
| Planner plus admission CPU, s | 0.252 → 0.264 | 0.242 → 0.232 |
| Video delivered on time, % | 97.508 → 97.379 | 97.359 → 97.323 |
| Aggregate fleet bulk throughput, Mb/s | 544.266 → 553.259 | 538.705 → 546.364 |
| Roams | 19.25 → 10.50 | 0.50 → 0.00 |

The miss differences are absolute fractions: +0.00303125 is **+0.303125
percentage points**, and −0.00098958 is about **−0.098958 percentage points**.
The slow-handover net reduction is 95 misses across four pairs; one episode
contributes 98 fewer misses, two episodes worsen by one and four misses, and
the fourth improves by two. This is a conditional descriptive gain, not
uniform reliability improvement. Four episodes from one warehouse family
do not support a general fleet SLA or a significance claim based on treating
all packets or robots as independent sites.

Every contrast consumes all twelve forecast jobs. All eight learned cases
exceed the common 0.25 s warm allowance on eleven jobs; Map has zero recorded
forecast overruns. The queue charges actual work above its allowance. Paid
compute also includes a 1 s cold floor; twelve shared 0.5 s transport delays
add six seconds to cumulative forecast service. These totals are cloud CPU
measurements including feature/history preparation and queue handling,
not neural-network-only latency or measured SmartCar timing. No planner or
admission result was discarded as late in these cases.

## The 9.74-second gap

The [outlier receipt](../../artifacts/native_bridge/cases/dev-90473001-ho0.03/receipt.json)
binds the original packet counters, execution trace, forecasts and physical
arrays used for this read-only diagnosis.

In fast-handover seed 90473001, robot 7 misses **200/240 commands**, versus
**3/240** with Map. Its own maximum gap is 9,740 versus 90 ms; the Map fleet's
maximum is 240 ms. The candidate executes AP4→AP3 at slot 150, or 1.50 s.
Both first forecasts complete at that slot, so warm-job overrun does not
explain the initiating divergence.

The retained trace later proposes AP0 recovery, beginning with job 3 consumed
at 3.93 s, but no recovery executes. The last control delivery is at slot 225,
or 2.25 s. The frozen fast-handover cadence fires a pending move after a
successful control delivery; the subsequent lack of deliveries leaves
recovery pending. This is an observed recovery lockout in the shared control
mechanism. The external `reactive_rescue` hook passes eligibility through;
it is not an independently calibrated risk filter.

At forecast issue, the stored AP3 5 GHz prediction is −2.205 dB versus an
archived base value of −23.152 dB. The base excludes hidden fading and
blockage; it is not a measured instantaneous packet SINR. This documents a
large field-transfer error alongside the execution failure. No new model
forward or intervention was run to assign separate causal effects to radio
error, later latency or the cadence mechanism. Those effects are not
quantified by this trace inspection.

## Scope and next evidence

Scans cost 90 ms with a five-second cadence. Delivered scans add 1.5 dB
measurement noise, 5% missingness and top-eight reporting to the kernel's
faded/blocked RSSI. Initial cases set collision_lambda=0 and interference=False;
airtime contention, external occupancy, native retries, fading and blockage
remain. There is no learned packet-risk calibration.

Native 6.5 GHz is an explicit extrapolation beyond the RF study's 6.0 GHz
endpoint. The 120×80 m warehouse uses a declared public 3D rack extrusion,
zero device-0 bias, reciprocal path-gain and known TX/noise assumptions.
Native PHY success comes from the archived ns-3 NIST two-stream, 1,200-byte
HT20/VHT80 tables reused for 600-byte control; the 6 GHz table retains its
historical doubled-rate approximation. These are unqualified modeling
assumptions, not physical receiver or warehouse calibration.

Preserve this result and its outlier. Any field-transfer, latency or recovery
change needs a new source/protocol and fresh comparison; rerunning these
frozen cases after tuning would not be independent confirmation. RadioMap
remains the default, the original 24 SERVICE specifications remain unopened,
and no hardware actuation, production promotion or universal-model claim
follows from this diagnostic.
