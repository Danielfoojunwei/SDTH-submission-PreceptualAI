# Registered native warehouse bridge

This is a finite simulated development diagnostic. It does not reopen the
original 24 SERVICE confirmation cases or qualify a controller, PHY model,
packet probability, radio uncertainty or real SmartCar deployment.

The study registers four fresh episode seeds, 90473001–90473004. Each runs
100 robots for 12 seconds with handover costs of 0.03 and 4 seconds, reported
separately. Each case executes the same frozen native packet kernel twice:
one strong numerical RadioMap and one adaptive RF field. A separate first
case sends strong RadioMap predictions through SNR→RSSI→SNR and requires
bitwise forecast equivalence, identical completion slots and identical native
service counters before interpreting learned contrasts. There are at most
18 native simulations. No native development outcome selects the model.

The learned member is fixed seed 0 of the joint/regression branch selected
only by the adaptive study's validation rule averaged over its three seeds.
All twelve adaptive fits must be complete and their checkpoint/fit receipts
must verify before the bridge loads that member. Its weight hash is bound
after fitting. A failed adaptive RF development gate does not prevent this
separately registered diagnostic, and does not grant fresh RF TEST permission.

The actual native episode and its first 60 seconds of reports come from
`fleetwm.data.episode`. `contracted.data.runtime_world` slices physical truth
and public plans to the same post-warm clock. Both arms receive the same
warm reports, five-second scan cadence, native traffic and hidden channel
draws. Each subsequent report must complete its scan and arrive on a real
successful control frame. Actions can change later telemetry, so identical
capture capacity does not imply identical subsequent observations.
During service, DeliveredScans adds 1.5 dB receiver measurement noise, a 5%
missing-scan event and top-eight reporting to the kernel's faded and blocked
RSSI. Warm reports use the separate native capped-scan generator. Neither
scan source supplies integer quantization bins.
The native radio/queue decisions close the connectivity loop; physical robot
trajectories remain exogenous. Safety-stop counters do not validate a coupled
motion controller or change the archived physical route in this probe.

Only strictly past delivered positions, measured RSSI, heard masks, actual
upper censor bounds and scan-attempt masks reach the RF predictor. Its
maximum 64 context tokens alternate AP/source-band groups, selecting newest
reports with deterministic heard/robot tie breaks. The strong Map retains
its full 60-bin history. This intentionally avoids weakening the numerical
control to the neural token limit. Source bands are 2.4 and 5.2 GHz; a
measured 6.5 GHz scan is not encoded. Native scans have continuous power
values, so the quantization feature is zero. Unknown and censored entries
are never turned into reported point measurements.

The native warehouse is a 120×80 m two-dimensional model. The bridge makes
an explicit public geometric approximation: a 6 m hall, 1 m receiver height,
4 m AP height and 33 separated rack boxes, 1.2 m deep and 3 m high. These
are declared extrusions, not hidden RT geometry or measured heights. A
vectorized segment/box operator matches the existing scalar union operator
on analytic fixtures; overlapping or touching box interiors/faces are
rejected by the fast sum operator. It evaluates the 43,200 query links for
100 robots, nine horizons, sixteen APs and three bands without a Python
query-point-by-box loop. Yard geometry is unsupported.

The field is queried at actual native carriers 2.4/5.2/**6.5 GHz**. Its 6.5
GHz output is new extrapolation, distinct from the adaptive RT study's 6.0
GHz endpoint. The bridge declares native zero device-0 bias, AP transmit
power from public geometry, 15 dBm client transmit power, equal path gain
under reciprocity, zero added antenna correction, and the native receiver
noise floor. Initial cases set collision_lambda=0 and interference=False;
native airtime contention, packet retries, hidden fading and blockage remain
active. The archived 1,200-byte/two-stream HT20/VHT80 PER tables and historical
6 GHz doubled-rate approximation remain explicit native modeling proxies.

`RSSIForecast.current_ap` is an all-zero conversion placeholder. It is never
an association observation or planning input: the completion queue consumes
only mean/spread, while native planning supplies actual current association.
The field does not receive a mission or action, and does not invent a
calibrated control-deadline probability from RSSI. RadioMap remains the
default outside this opt-in diagnostic.

Both arms use the frozen service bridge's typed surrogate utility, native
roam path, independent gate and exact admission, proposal lifetime, scan
costs, stagger and FIFO completion. Common forecast allowances are 1/0.25/1
seconds for cold/warm/refit plus a 0.5-second transport delay. Actual work
above an allowance is charged. The isolated wiring case uses a common four-
second allowance to test conversion independently of ordinary small timing
variation; any overrun or inequality is reported as proof failure. Planner
and admission computation have the same 0.1-second deadline; late results
are discarded and their measured costs remain in the evidence.

Report all four episode pairs at each handover cost, including gains and
harms, control misses, worst-decile miss fraction, p99 and maximum delivery
gap, video, bulk, airtime, roaming and paid forecast/planner costs. Four
episodes do not support a general SLA, SOTA or fleet reliability claim.
`control_gap_p99_ms` is the 99th percentile across robots' individual maximum
gaps, not the 99th percentile over every interpacket gap.
Native uncertainty and service heads remain unqualified. No hardware is
actuated and no default model is promoted.

The runner archives physical arrays separately for audit, common warm
reports, selected weights, final causal history and every selected context,
model power/spread, paid completion record, decision, execution and raw native
packet counters. A passive audit recomputes source identities, causal token
selection, RSSI→SNR hashes, event payload/time ordering, packet/gap metrics
and paired summaries without a new fit or model forward. Completed evidence
is sealed by a byte manifest.

Run `python -m wirelessjev.native_bridge.study write-spec`, then `preflight`
and `register` after the adaptive source is published. Commit and push the
resulting source capsule, and use `publish --commit SHA --remote origin
--branch BRANCH` to verify the remotely observed commit. Only then may `run`
generate the fresh native worlds and execute cases after all adaptive fits
and validation selection exist. `audit` is passive. Scientific execution
before this publication gate is rejected.

Each case writes and fsyncs an exclusive attempt record before any world
generation, model construction or native execution. An interrupted nonempty
case without a receipt is retained and cannot silently run again. Cached and
audited cases must match the exact case, handover, compute profile, source,
protocol, world and canonical archive paths. Recomputed checksums cannot
turn a different configuration into the registered case.
