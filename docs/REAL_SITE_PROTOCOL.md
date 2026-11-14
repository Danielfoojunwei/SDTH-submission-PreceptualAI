# Real-site analysis: fixed before the data exists

Site visit: 17 Nov 2026, one SmartCar, operator-supplied APs. This fixes how
the logs will be analysed, so results are not shaped after the fact.

## Data
Laps from `onecar lap`: 10 Hz link state, 20 Hz gateway ping, scan-cache snapshots, waypoint marks. Handover timings come from `switch --timing` and `roam`.

**Split:** survey laps and odd-numbered route laps (`r01`, `r03`, …) are for fitting. Even-numbered route laps and all `block*` laps are held out. `load*` laps are reported separately.

## Questions and endpoints
1. **Handover cost.** Median and worst ping outage for today's switch versus fast roam; whether the IP is kept. The twin's assumed 4 s and 30 ms are replaced by these.
2. **Signal prediction on held-out laps.** RMSE of predicted versus measured RSSI at held-out positions, for:
   - the leave-one-lap-out nearest-neighbour map (`hil.trace_twin`);
   - Codex's portable numerical map;
   - Codex's three frozen RadioJEPA checkpoints in shadow.

   No model is retrained on held-out laps.
3. **Outage prediction on held-out laps.** An outage is a ping-reply gap over 0.2 s. For horizons 0.1, 0.5 and 1 s: Brier score of the persistence baseline (`hil.analyse`) and of any learned model trained on the fitting laps only. A learned model "wins" only if its Brier score is lower on the held-out laps at all three horizons.
4. **Safe plan.** How many gaps exceed 0.2, 0.5, 1 and 2 s: how often a robot carrying a plan of that length would have stopped.
5. **Twin calibration.** Fading spread, blockage depth and duration, and path-loss slope estimated from fitting laps; twin-versus-real error reported on held-out laps.
6. **Ten-robot trace twin.** Fixed rule, single radio and same-card second path on laps `r01`–`r10`, labelled "1 real car, replayed".

7. **Blockage rule on real signal** (added 15 Nov, before the visit). The simulator's detector rule (signal more than 10 dB below the car's own running level for 2 consecutive 100 ms readings; `docs/BLOCKAGE_DETECTOR.md`) is replayed over the held-out and `block*` laps with its settings unchanged (`hil.analyse`). Endpoints:
   - outages caught: the rule fired between 1 s before the outage and its end;
   - median lead time before the last reply;
   - false alarms per minute of driving.

   The rule is "useful on real signal" only if it catches at least half of the `block*` outages with at most 1 false alarm per minute. One car reports only the link it is using, so this is the single-link form of the rule; the "every band at once" check is not testable on Tuesday.

## What this cannot show
Fleet interference between real robots, control by a learned model, or anything about private 5G.
