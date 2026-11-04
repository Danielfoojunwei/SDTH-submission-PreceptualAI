# Repairing JEPA execution and local recovery

This is a new, prospective controller/runtime experiment with fixed model weights. The prior compact-JEPA comparison remains negative. Its source, timing veto, archives and original unused reservations are preserved. This work tests the controller package around the strongest previously tested JEPA ensemble and RadioMap; it does not establish a new JEPA objective.

## Diagnosis before another fit

The [architecture audit](JEPAREPAIR_DIAGNOSIS_ARCHITECTURE.md) finds 16% lower absolute validation SNR error but only 1.6% lower short route-increment error, which is effectively tied with supervised. Native band/MCS control cancels static forecast bias. The head is not saturating at its cap. More latent rank or a wider correction is not an evidence-based first repair.

The [service audit](../artifacts/jepa_repair_diagnosis/README.md) identifies a 20.73-second right-censored silence on a robot that stayed associated to the same AP. Every recorded JEPA plan selected that same AP; the tail does not demonstrate a rejected eligible plan. Reactive masks and local-cache freshness were not archived, so its precise cause is unproven. Factual handover outages account for a large share of slow-case misses; these are not all avoidable regret.

The [runtime diagnosis](JEPAREPAIR_RUNTIME_DIAGNOSIS.md) preserves the original isolated 285 ms/120 ms budget exceedance and its unresolved cause. The historical workers explicitly limited BLAS/OpenMP threads. An unpinned diagnostic's oversubscription does not explain that historical event.

## Concrete repair mechanics

1. **Fractional outage integration.** The old fast gate begins candidate integration at the first horizon after the outage. For 30 ms this drops 970 ms of failure exposure. The new gate interpolates at the exact outage endpoint. With constant failure probability `p`, switch-minus-stay cost is `outage × (1 − p)`.
2. **Separate planned admission from local recovery.** Rejected planned actions no longer leave the confidence mask preventing reactive recovery. The prior all-handover admission policy also vetoed reactive actions. The new package admits planned actions through the exact gate and retains a separate observable recovery rule. This changes controller semantics as well as fixing numerical integration.
3. **Break delivery dependencies locally.** A robot-local watchdog runs every 100 ms after the existing 200 ms control-silence stop threshold. It can start an already-due periodic scan without waiting for another successful control packet. It can switch from a poor current link to an actually heard, completed, recent local neighbor without waiting for a cloud forecast or the packet needed to fire a cadence-timed fast move. It clears stale pending moves. No additional arbitrary scan cadence is introduced.
4. **Keep unknown observations unknown.** Recovery candidates must come from completed heard scans no older than the configured scan period; current-link pilots remain available. An unheard `−100 dBm` placeholder cannot become a fictitious better neighbor when the current link is dead. Local scans update local recovery after completion; central reports still require successful control transport.
5. **Pay actual inference completion.** Every arm charges `max(sealed allowance, predictor elapsed, measured issue-path elapsed) + 500 ms`. Issue timing includes history snapshot, forecast validation/storage and hashing. Consumption is recorded and must occur after accounted completion. Scans and handovers retain their actual native outages.

The opt-in new kernel preserves every predecessor field when emergency recovery is disabled, adding only an execution audit. Regression cases use the real native PHY/packet path. Constructed software fixtures are not warehouse-performance evidence. The application default is unchanged.

## Fixed comparison and source boundaries

The [protocol](../artifacts/jepa_repair/protocol.json) registers two new development sites and eight new confirmation sites, 100 simulated robots, sixty seconds, 30 ms/four-second handovers and two fixed forecasters: strong RadioMap and the original three-member FleetJEPA ensemble. All arms use the same completion-safe runtime and delivered observation implementation.

Fast cases compare legacy, exact-integral-only and the complete repair. Slow cases compare legacy and the complete repair. The fast intermediate arm is a secondary ablation, not an option selected after outcomes. This yields twenty development and eighty confirmation evaluations if both costs proceed.

Development is mechanical validation and a fixed severe-regression veto: confirmation stops for a cost if either family's mean repair-minus-legacy command loss exceeds one percentage point, or integrity/paid-cost/consumption fails. No weight, controller or threshold is tuned. Confirmation permission is recomputed from raw development archives before opening new worlds.

Six primary comparisons retain one multiplicity budget: repair versus legacy for each model/cost, and repaired JEPA versus repaired RadioMap at each cost. Whole sites are the independent units. Observed and conservative pending-command contrasts, worst-decile loss, maximum silence, original video, bulk throughput and zero-margin diagnostics remain visible. Untested costs are explicit, never passes.

Execution uses a fixed seeded permutation within site blocks, two workers and explicit thread limits. The order, source, weights, criteria and dependency versions are sealed and published before any new registered site is opened. Earlier training and results have already been observed; only this new evaluation is prospective.

## Timing, replay and claim scope

The earlier study's global timing veto remains unchanged. In this separately declared actual-cost experiment, exceeding a profile is a capacity diagnostic because the additional measured cost is paid. Undercharged or prematurely consumed forecasts veto experimental benefit. This is not a retrospective waiver or physical timing qualification.

Offline trace replays recompute the first confirmation site's forecasts and native packet outcomes using its original charged schedule. They must match exactly, and are labelled offline; they do not claim that newly measured compute met the original schedule. Separate fresh measured-cost replications report runtime sensitivity and are not additional independent sites. Different live completion times can legitimately change future delivered histories.

Admission/scoring CPU work is not individually scheduled; the common 500 ms scheduling/other-work delay remains an assumption. The study uses procedural shared CAD, conditional warm history, assumed localization/scan capabilities and archived native PHY curves. Experimental benefit is distinct from production qualification. No physical car, default promotion, SOTA, EMA necessity, arbitrary-layout transfer or indefinite reliability claim follows.
