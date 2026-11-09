# Public native-query geometry reuse

The new standalone `wirelessjev.native_bridge_ops` operator computes rack intersections once for each planned robot–horizon–AP link and broadcasts the geometry across the three bands. At the existing 100-robot, nine-horizon, 16-AP dimensions, this changes AABB work from **43,200 to 14,400 geometric links** while still returning all 43,200 carrier-specific queries.

This implementation does not alter the sealed native runner, its model, its 18 simulated native outcomes or any service result. It is an operand-construction optimization. A reduction in geometric work is not a measured runtime improvement or a service deadline claim.

## Preserved calculations and data boundary

The sixteen query features, eight physics-basis values, FSPL powers and zero offsets preserve the frozen output shape, ordering, dtype and **exact bytes**. Receiver/AP coordinates, distance and rack intersections are independent of band. Actual 2.4/5.2/6.5 GHz carriers, per-band transmit powers and per-band noise values remain distinct. The frozen FSPL and query-basis operators are used directly to preserve their arithmetic and rounding.

The new typed `PublicQueryGeometry` stores immutable copies of public AP sites and known transmit/noise arrays. The query function accepts this snapshot and a public planned-position array. It receives no `World`, measured observations, private RF, materials, model or packet simulator; it retains no cache between calls. The original native predictor continues to call its original query implementation.

Unit tests use explicitly identified analytic and fixed-public-CAD software fixtures, including all 100 robots, fractional and noncontiguous plans, changed powers/positions, and different site counts. They compare complete output bytes with the frozen reference and instrument only the new module's intersection alias to verify one 14,400-row AABB pass. They supply no accuracy or service evidence. Tests reside under `tests/native_bridge_ops/` because the old source collector freezes the top-level `test_native_bridge*.py` names.

## Finite timing protocol, before measurements

[The protocol](../../artifacts/native_bridge_ops/protocol.json) binds the sealed native manifest SHA256 `4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9`, its 512 source files and 56 evidence files. Registration binds the new operator, profiler, tests, protocol and exact test receipts. These must be committed, pushed and observed on the named remote branch before the profiler opens any saved plan arrays or measures calls. This small registration binds the old manifest; it does not duplicate the old source capsule.

The profile uses existing public planned positions for seeds 90473001–90473004, handover setting 0.03 seconds, all twelve saved jobs per seed: 48 plans in total. From each sealed world archive it reads only `sites`, `tx_dl` and `noise_db`; from each candidate archive it reads only the twelve registered `job*_plan` members. Archive byte hashing does not interpret its private members. The program constructs no world and reads no RF or observation array.

Each operator receives one untimed warmup per plan and three timed calls per plan. Pair order alternates deterministically; all three numerical thread limits are fixed at one. `perf_counter_ns` measures operand construction only. Input loading, provenance checking and byte-parity checks are outside the timing interval. Complete operand byte parity is required for each plan and every timed pair. The primary runtime ratio divides the median reference call time by the median optimized call time across the fixed 144 calls per operator. Repeated same-process timings are descriptive and correlated.

At initial source registration, no actual-plan profiling or timing had run. Source commit `0bc5b4360d8c8023af6891b53374b4a576f26077` was then pushed and observed before the registered profile executed. The run command is:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --frozen python -m wirelessjev.native_bridge_ops.profile run
```

The profiler refuses an existing attempt rather than silently repeating measurements. It checks preserved native bytes before and after and writes to the separate `artifacts/native_bridge_ops` namespace. Its report must remain explicit that no model forward, fit, new world or packet simulation took place.

Any measured speedup applies only to operand construction in the recorded CPU environment. Context processing, learned rendering, queuing, planning, native packet behavior and physical hardware latency are outside this profile. Improving those outcomes requires a separate authorized and registered evaluation; the old native evidence remains unchanged.

## Actual registered results

The finite profile completed successfully on all **48 saved public plans**, with **144 timed calls and 48 untimed warmups per operator**. Every warmup pair and timed pair returned identical operand shape, dtype and bytes. The geometric AABB work fell from 43,200 to 14,400 links per call; the number of output queries remained 43,200.

| Operand call | Median time | Observed minimum–maximum |
|---|---:|---:|
| Frozen reference | 277.992 ms | 245.183–337.103 ms |
| Geometry shared across bands | **98.194 ms** | 86.175–130.875 ms |

The registered primary ratio of median call times is **2.831×**, corresponding to 64.7% less median operand-call time. The median of the individual paired ratios is 2.824×. Reference-first and reuse-first order each occurred 72 times. These are correlated same-process measurements, not independent deployment trials or confidence intervals.

The environment used Python 3.12.14, NumPy 2.5.4 and a Linux CPU quota of four cores. OpenBLAS, OpenMP and MKL thread limits were each one. Comprehensive repository tests and adaptive prediction work were running concurrently on the managed CPU; the benchmark was not isolated. Its timers include ordinary call allocation and physical-feature arithmetic, so the result is an operand-call microbenchmark, not a pure intersection-kernel result.

Evidence: [registered report](../../artifacts/native_bridge_ops/report.json), [source publication receipt](../../artifacts/native_bridge_ops/registration/git.json), [independent review](../../artifacts/native_bridge_ops_review/review.json) and [review script](../../artifacts/native_bridge_ops_review/script.py). The report SHA256 is `38c2751d1125be80a8937d40d288fdacf93de4afad0fcdfa4fd8ab7ba4580503`.

The independent review checks all 48 public input hashes, all 144 unique timing pairs, balanced call order, exact summary arithmetic, source publication and the preserved 512-source/56-evidence native byte closure. It makes **zero extra operator calls, zero timing calls, zero model forwards and zero packet simulations**. The native evidence manifest remains SHA256 `4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9`; all 18 previously completed native simulated outcomes are unchanged.

The optimized operator is available separately. It has not been substituted into the frozen native runner, and no full-model latency, deadline, controller or service improvement is claimed.
