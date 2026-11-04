# Sustained recovery for costly handovers

This separately registered experiment tests a repair for the slow-handover failure found in the [local recovery study](JEPAREPAIR_DESIGN.md). The earlier aggressive package roughly doubled command loss on its two four-second development sites. Its severe-regression gate rejected that cost before confirmation; those four-second confirmation cases were not run; the same world IDs were evaluated only at the fast cost. The negative result and its source stay unchanged.

## Preserve ordinary admission; require stronger emergency evidence

A short RF fluctuation is poor evidence for paying a four-second interruption. Both new variants retain the original outage-utility candidate scores, confidence behavior, ordinary cached-neighbor behavior and admission for **both planned and reactive moves**. The repaired variant changes only local observation recovery and the sustained-failure watchdog.

An already-due periodic scan may start once control silence reaches the existing 200 ms stop threshold, even when no successful control frame arrives. It pays the native 90 ms scan outage. A scan becomes available locally only after completion; its central report still requires successful control transport.

The handover watchdog requires **420 primary-radio-available silent slots**:

`existing_stop_threshold + ceil(configured_handover_outage / slot_duration)`.

At four-second handovers and 10 ms slots, this is at least 4.2 seconds of available-radio time without delivered control. Successful control delivery, an association change or known primary handover unavailability resets this evidence. Scans pause it. The watchdog cannot count its own preceding four-second outage as evidence for another move.

When deciding, the current measured pilot must also be below −75 dBm, and a genuinely heard, completed local neighbor must be at least 8 dB better and no older than the configured scan period. The timestamp belongs to capture, not completion or later delivery. Unknown and stale alternatives remain unavailable. This does **not** prove that RSSI was continuously poor throughout the silent interval; it is an explicitly cost-derived recovery heuristic.

The watchdog runs independently of cloud forecast arrival. An emergency move pays the configured four-second interruption, clears stale pending targets and resets its old-link evidence. Both forecasters retain fixed weights. This is a controller repair, not a newly trained JEPA objective or an optimal stopping theorem.

## Independent comparison

The [protocol](../artifacts/costaware_recovery/protocol.json) fixes only four-second handovers, strong RadioMap and the original three-member FleetJEPA ensemble, each with legacy and repaired control. Two new development sites yield eight evaluations. Eight additional fresh confirmation sites yield thirty-two evaluations if the fixed severe-regression check permits them. Every case has 100 simulated robots for sixty seconds.

The prior development results are already observed and motivate this new rule. New source, criteria and execution order are published before opening these new sites. No old rejected reservation is reused. There is no model fitting, controller selection or timing recalibration after observing the new outcomes.

Three primary comparisons share the declared multiplicity budget: repair versus legacy for each forecaster, and repaired JEPA versus repaired RadioMap. Whole sites are the independent units. Conservative pending-command bounds, worst-decile loss, maximum silence, original video and bulk delivery remain required; zero-margin diagnostics stay visible. Development stops confirmation if either repaired mean loses more than one absolute percentage point or a mechanical/paid-cost proof fails.

Every fresh arm uses the same actual-cost queue and original sealed profiles. Forecast work is charged as the maximum of its allowance, predictor time and measured issue-path time, plus the common 500 ms scheduling/other-work delay. A paid profile exceedance is reported separately from an undercharged or premature release. The older studies' different vetoes remain unchanged. Admission/scoring CPU work is not separately scheduled, so full-stack real-time qualification is not claimed.

Four explicit offline replays retain their original charged schedules while recomputing forecasts and packet outcomes. Four separate fresh measured-cost repetitions assess runtime sensitivity, adding no independent sites. Offline equality is not a freshly measured deadline guarantee.

Capture masks, measured values and timestamps, payload hashes, local completions, watchdog evidence, paid outages and forecast consumption are archived for verification. The simulator still uses shared procedural CAD, conditional warm history, assumed localization/scan capabilities and fixed packet/PHY mechanisms. No physical command, default activation, production guarantee, arbitrary-warehouse transfer, EMA necessity or SOTA claim follows.

## Why a cost-derived threshold is not a safety guarantee

Waiting 4.2 seconds establishes past control failure. It does not establish
that the failed link would remain unusable for another four seconds, or that
the candidate would deliver commands after switching. Waiting and then paying
the four-second interruption can itself create approximately 8.2 seconds of
silence before successful delivery, with slot/packet alignment adding delay.
Excluding the controller's own outage from its evidence prevents one feedback
loop; it does not remove this continuity cost.

The study therefore retains its separate maximum-silence check. A development
GO means only that the fixed severe-regression/mechanical gate permits the
confirmation experiment. It is not an improvement, tail-safety or deployment
approval. Future recovery designs need an explicit prospective packet-loss
and silence budget, as well as evidence that the alternative is reachable.
With a sole control radio, a four-second break-before-make handover cannot
guarantee a tighter uninterrupted-delivery target. Another usable link or an
appropriate robot stop/fallback mode is a system requirement for such a target;
changing the JEPA loss alone cannot supply it.
