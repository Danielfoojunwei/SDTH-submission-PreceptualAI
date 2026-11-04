# JEPA enabling stack: freshness, radio resources and sensing

The [protocol](../artifacts/freshness_stack/protocol.json) tests changes around fixed original FleetJEPA and strong RadioMap. Both receive the same stack improvements. No model is retrained, selected or assumed to win.

The [passive freshness diagnosis](FORECAST_FRESHNESS_DIAGNOSIS.md) reproduces 208 old confirmation archives. The ensemble issues 60 forecasts but consumes 43 before the episode ends. Consumption age reaches 17.98 s; held age 18.97 s. Compact JEPA/supervised do not have sustained overload, so this does not overturn their earlier null results.

Four nested stages are fixed before data: original paid FIFO; latest-only admission; latest-only with simultaneous STR control duplication disabled; and that package with conditional whole scans. Busy requests are rejected before inference. All accepted work retains actual measured costs and the same serialized 0.5 s allowance. Source timestamp, job ordinal and array index remain distinct. Sequential cross-link retries remain enabled.

A full 90 ms scan can cover a 50 ms control opportunity. The final stage postpones an ordinary due scan only with fresh association-matching pilots and an actually consumed forecast predicting a strong fixed band at its six native query points over the next five seconds. The −67 dBm threshold is fixed from existing −75 dBm poor-link threshold plus 8 dB recovery gain. This heuristic is not a safety guarantee. Unknown/stale evidence allows the scan. Postponement creates no measurement, leaves the scan due, and never blocks the existing 200 ms silence watchdog. Every performed scan pays its full 90 ms outage.

The [sensing diagnosis](STACK_SENSING_DIAGNOSIS.md) distinguishes scan timing from combined scan/handover outage attribution. The native simulator already supplies lagged pilots across three bands of the associated AP; this is an inherited assumption, not real SmartCar capability evidence. The new gate checks the actual capture association and refresh clock, and reads no additional pilots.

The study reserves 16 development cases and conditionally 64 confirmation cases on two/eight new sites, respectively, plus eight offline exact replays and eight fresh-cost repetitions. Every case has 100 simulated robots for 60 s at assumed four-second handovers. Final-package versus baseline within each family and final JEPA versus final RadioMap are the only three primary tests. Intermediate stages are descriptive, not alternative selected winners or a full factorial design. Source publication precedes collection.

All old source/evidence, timing vetoes and negative outcomes remain intact. Source/cost/causal checks, pending-command sensitivity and worst-decile/silence/video/bulk margins still gate benefit. This is simulation research; no default, physical, production, JEPA-objective, SOTA or indefinite reliability claim follows.
