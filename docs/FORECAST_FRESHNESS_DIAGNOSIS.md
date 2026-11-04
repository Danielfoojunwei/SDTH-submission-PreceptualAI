# Forecast freshness: the server is slower than its requests

**The original FleetJEPA ensemble is overloading the simulated FIFO service, so its controller receives increasingly old forecasts.** This is a concrete enabling-stack limitation, not a demonstrated explanation of every missed command or evidence that changing the JEPA objective will help. The compact context-JEPA model does not have the same sustained overload.

This is a passive, post-hoc audit of **208 already completed confirmation archives**: 128 selected-model, 48 controller-repair and 32 cost-aware-recovery cases. No world, forward pass, training, scientific timing, packet replay or candidate service result was generated. The [diagnosis](../artifacts/forecast_freshness_diagnosis/diagnosis.json), [hash receipt](../artifacts/forecast_freshness_diagnosis/receipt.json) and [standalone script](../scripts/diagnose_forecast_freshness.py) retain every case and input hash. Earlier source closures of 80, 230, 386 and 422 entries were checked before and after reading. Historical timing vetoes and packet results remain unchanged.

## What the completed ledgers show

Each trial requests one full fleet forecast per second for 60 seconds. The older ensemble's sealed warm allowance is 0.85 seconds plus the shared 0.50-second scheduling/other-work assumption. Its serialized **1.35-second** warm service exceeds the **1-second** request interval. The actual-cost repair correctly pays overruns but preserves that sealed floor, so paying work honestly does not make the queue stable.

The table shows the archived schedules, not a new benchmark. Maximum ages are descriptive extremes; rows and models are not additional independent sites.

| Forecast package | Warm charged service | Consumed forecasts per 60 requests | Largest age when consumed | Largest actual wall-age while held | Last issued job ready |
|---|---:|---:|---:|---:|---:|
| Original FleetJEPA ensemble, all three studies | 1.35 s | 43 | 17.98 s | 18.97 s | 82.93 s |
| RadioMap, controller/cost-aware studies | 0.62 s normally; one 0.64733 s paid diagnostic | 60 | 1.13 s | 2.11 s | 59.62 s |
| Compact context JEPA, selected study | 0.85 s | 60 | 2.13 s | 2.97 s | 59.85 s |
| Matched compact supervised, selected study | 0.85 s | 60 | 2.13 s | 2.97 s | 59.85 s |

Every one of the 56 archived ensemble cases leaves **17 of 60 jobs unconsumed within its trial**: 952 jobs in total. These jobs were actually computed by the synchronous simulator and assigned paid virtual completion times. This does not mean they were cancelled or that an asynchronous production worker already exists.

One concrete example is `costaware_recovery/confirmation-2134000-h4000-frozen_fleet_jepa-legacy`. The first forecast, from source decision 0, becomes usable at 1.96 seconds. The last consumed forecast comes from source decision **42**, becomes usable at **59.98 seconds**, and is 17.98 seconds old. Immediately before that arrival, the held forecast from decision 41 is 18.97 seconds old. Jobs from decisions 43 through 59 do not become usable during the episode; decision 59 is scheduled for 82.93 seconds.

The repaired studies log consumption explicitly, and this audit reproduced each logged event exactly from paid ready slots. The older selected study lacks that consumption field; its consumption events are clearly labelled as recomputed from its original ready ledger. No fresh elapsed-time claims follow from either calculation.

## Age compensation exists, but support runs out

`ForecastQueue.advance` already interpolates at `horizon + age`. Merely adding a missing age shift would therefore not be a new repair. It clips requests to the original 0–30 second forecast support.

At a 17.98-second consumption age, the old forecast has **12.02 seconds of remaining future support**. Of the controller's requested 30-second window, 17.98 seconds lie beyond the model's original prediction interval. The requested 20- and 30-second knots both become the old terminal value. Because outage utility reinterpolates the sparse advanced knots, its exactly constant terminal segment is 20–30 seconds: **10 seconds**, not 17.98 seconds. Unsupported query duration and exactly terminal-held integration duration must not be conflated.

The kernel also takes `sigma_now` from the source decision's horizon-zero standard deviation while advancing only the forecast mean. That is an interface mismatch to address when uncertainty is used; the current reactive-rescue gate overrides the ordinary confidence calculation, so this is not an isolated explanation of present losses.

Clock age alone does not establish that a route changed or a forecast became inaccurate. The archives do not retain every public plan tensor. A future service should carry query time, route/pose identity and validity support explicitly, rather than assume that retiming an old route forecast is equivalent to querying the current route.

## The clean next intervention

Use the same frozen ensemble and RadioMap, original allowances, shared 0.5-second assumption, observations and controller. Change **admission of forecast work before computing it**. With the service busy, do not append another expensive forecast to an unbounded FIFO. At the next allowed start, use the newest causally available public history and route. Log every submitted, superseded, admitted, started and completed request. Never declare already computed work free because its answer was discarded.

Two implementable choices have different semantics:

- **Busy admission at integer request times:** skip computation when the preceding forecast is still in service; reconsider with the newest request at the next integer tick. This is conservative and easy to align with existing public histories, but can leave idle fractions and reduce controller callbacks.
- **One coalesced pending request:** keep only the newest pending request and start when the active job completes. This needs a defined source-history cutoff and public route at fractional start times, plus actual work accounted at that start. Reusing future route arrays or silently shifting a request timestamp is invalid.

A shorter queue can improve freshness while reducing update count. It can therefore worsen packet control. It needs fresh paired service experiments, not only an age improvement or an unregistered rerun of old holdouts. Keep legacy and repaired controllers fixed during the first queue comparison; a second mechanism can test forecast validity/horizon handling separately.

Do not call lower queue delay a JEPA-objective advantage: the compact JEPA and supervised models already stay below the request interval and still failed their original statistical comparisons. This intervention targets delivery of the stronger, slower ensemble's information.

## Separate compute from communication only when justified

The existing FIFO serializes both forecast compute and the shared 500 ms allowance. If some of that delay is transport latency, it might overlap the next computation; if it represents CPU or dispatch work on the same server, it might not. The old studies do not identify the components. A measured pipeline with distinct compute, serialization, transport and consumption stages could preserve a 500 ms end-to-end delay without treating all of it as server occupancy. That is a separate prospective assumption requiring evidence, not permission to retrospectively reduce old charges.

## Mechanical checks needed before new sites

- Sparse request times must remain distinct from job ordinals and array storage indices. The old `_consume` assumes all three are equal and computes age from the ordinal; blindly reusing it after skipping requests is wrong.
- A skipped or superseded request must not call the predictor, copy a history snapshot or mutate fleet/map memory. Admitted jobs must see strictly increasing source histories, including genuine reports between admitted times.
- Completion uses actual measured work and the same sealed allowance. No forecast or controller consumes it before verified completion; the pending buffer stays bounded.
- Every consumed forecast records source decision, source cutoff, start, finish, consumed slot and age. Test `d=0,2,4` admissions explicitly rather than only contiguous decisions.
- Age advancement, admission lookup and storage all use the correct source identity. Initial fallback, long refits, overruns, zero duration, final unconsumed jobs and invalid/nonfinite timings fail predictably.
- Distinguish logged age at new arrival from actual wall-age throughout its holding interval. Report issued/completed/consumed/superseded counts, effective callback cadence and time outside model support alongside packet endpoints.
- Preserve exact offline recorded-schedule replay as a reproducibility check. Fresh measured-cost replications remain separate and add no independent sites.

The causal performance question remains open. No queue redesign, hardware continuity guarantee, default activation, production qualification or new model benefit is claimed by this diagnosis.
