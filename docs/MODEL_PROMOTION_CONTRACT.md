# Model promotion evidence contract

`wirelessjev.production_qualification.qualify(bundle, repository_root)` is a
**read-only evidence checker**. It does not activate, download, train, schedule
or send a radio command. Existing proposal/shadow interfaces and application
defaults are unchanged. An eligibility report is not an execution capability,
scientific proof, legal opinion or safety certificate.

The current read-only inventory is
[`artifacts/ran_research/qualification_status.json`](../artifacts/ran_research/qualification_status.json).
All current candidates are blocked from production:

- Packet-action JEPA was rejected on development. Its existing summary has no
  evaluated fresh confirmation sites.
- DINOv3 features and the CPU ray-tracing fixture exist, but their visual/RF
  fusion is untrained and has no independent control qualification.
- The new continual-response experiment has no independently confirmed,
  hardware-qualified production model. Registration or completed collection
  alone cannot meet this contract.

The inventory reads and hashes actual existing receipts. It does not translate
old summary booleans into new positive qualification evidence. Missing dossiers
remain missing; their blockers include independent control, hardware timing
and actual rollback/reconciliation evidence.

## Bound evidence

A bundle declares `schema_version: 1`, `candidate_id`, a mode, `limits`, a
hardware/software profile, four file manifests (`source`, `weights`, `data`,
`licenses`) and a SHA-256 reference to a preregistered `protocol` JSON file.
Each reference is `{ "path": "relative/file", "sha256": "64 lowercase hex" }`.
References must resolve to files inside the repository. All bytes are hashed;
a receipt `{ "passed": true }` is insufficient. Receipt JSON is limited to
8 MB and individual hashed files to 1 GiB.

`subject(bundle)` hashes candidate ID, mode, hardware profile, all file
manifests, the protocol reference and the operator limits. Changing weights,
source/data, a driver, policy limits or proposal mode changes that subject.
Every fit, rights, evaluation, risk, hardware and rollback receipt must bind
the same `subject_sha256`. Receipt references are themselves hash-verified.
The protocol is included by file hash rather than recursively including its
subject; this avoids a circular hash definition.

The hardware profile names `profile_id`, `device_model`, `driver`, `firmware`,
`protocol`, `baseline_source_sha256`, `control_period_ms` and `deadline_ms`.
It also binds `baseline_endpoint`, `interface` and `role` for physical recovery.
The present SmartCar contract permits only the existing `wlan1` / `WIFI` role;
it cannot qualify the management hotspot for actuation.
This contract targets the current **20 Hz / 50 ms** warehouse task. Baseline
source must be in the bound source manifest. A different task needs an explicit
contract review, not a renamed seconds-to-milliseconds field.

`rights` contains reviewed license grants for every weight/data artifact:
artifact hash, bound license-file hash, reviewer, dated review and allowed
uses. Research permission can qualify a sandbox; `production_inference` is
required separately for production. The checker records an explicit operator
rights review and does not interpret the legal terms itself.

## Raw control and uncertainty checks

The protocol contains distinct nonoverlapping `training_sites`,
`validation_sites`, `confirmation_sites`, at least three distinct integer
`training_seeds`, and `sealed_at`. The evaluation receipt records
`confirmation_opened_at`, later than the seal. Every seed needs a separate
hash-verified fit receipt with positive optimizer steps, matching source/data
manifest hashes and a distinct bound checkpoint hash. Merely relabeling the
same checkpoint with three seeds fails.

The evaluation's `rows` cover exactly every confirmation site × training seed,
with one pair per combination. Each candidate/baseline arm supplies integer
`misses` and positive integer `sent`, `max_gap_ms`, and at least 100 raw
`latency_ms` samples. Exposures must match; booleans, NaN, negative numbers,
missing/duplicate pairs and impossible counts fail. One bundle qualifies one
specified hardware/policy profile; these rows are not interchangeable across
different handover costs, deadlines or devices.

For each site, the checker recomputes the mean seed-paired difference

\[
\Delta_s=\frac1K\sum_k10^6
\left(\frac{m^{candidate}_{s,k}}{n_{s,k}}
-\frac{m^{baseline}_{s,k}}{n_{s,k}}\right).
\]

Twenty thousand fixed-seed bootstrap draws produce a paired-site interval.
A second crossed bootstrap resamples both sites and trained seeds, retaining
pairing. Both upper endpoints must meet the operator's noninferiority margin.
Training seeds are never counted as additional independent warehouses. Raw
statistics and failed criteria remain in the returned metrics.

Those paired bootstrap intervals are **not sufficient evidence of a rare-event
budget**: eighteen commands with no misses can give a degenerate zero interval.
The checker additionally requires a preregistered independent reset-episode
sampling design and finite-sample absolute-risk bounds. The protocol's
`independent_sampling` specifies `unit: independently_reset_episode`,
`target: one_preregistered_uniform_command_per_episode`, and a hash-bound
`design_review` receipt. That review identifies its reviewer, pre-seal review
date, registered profile hash, reset/sampling implementation in the bound
source manifest, justification, and the explicit assumption
`iid_reset_episodes_within_registered_profile`. Its accepted basis is
`independent_randomized_resets`. Unknown/detected shift, inferred independence
from clock gaps, missing review, and correlated packet-trace sampling fail.
The registered policy is to invalidate qualification on detected shift.

Each paired row references an `independent_episode_ledger`: raw bytes, one per
registered episode's prespecified uniformly sampled command. Bit 0 is the
candidate miss and bit 1 the paired baseline miss. A separate hash-bound
`independent_reset_ids` file has one strictly increasing little-endian uint64
identity per outcome. Ledger lengths and miss bits must exactly reproduce
both arms' `sent` and `misses`. One long correlated trace cannot contribute
all its packets as independent trials; only the registered single sample from
each independently reset episode contributes. More commands inside an episode
do not increase this count. Reset identities alone do not establish independence.

For each site/seed cell with `k` misses among `n` justified episodes, the
checker computes the exact one-sided Clopper–Pearson upper bound. In particular,
zero failures gives `U = 1 - alpha^(1/n)`, **not zero**. Half the default 0.05
error budget is split across all control site/seed cells; the other half is
split across registered calibration bins. Each control cell needs at least
100,000 actual sampled commands **and** an upper bound no greater than 10 ppm.
At 95% for one cell, even zero failures need 299,572 independent episodes to
reach 10 ppm; multiplicity is stricter. No packet-level effective sample size
is inferred. Reused data across model comparisons are never pooled as extra
independent episodes. Binary outcome/identity evidence is capped at 32 MB/file.

Default engineering limits are six confirmation sites, three trained seeds,
**0 ppm additional mean loss**, at most **10 ppm loss in every site/seed**,
maximum **200 ms silence**, and **50 ms p99 latency**. Gap and p99 latency
must also be no worse than the paired baseline in each site/seed. These are
explicit release budgets, not externally certified safety requirements.
Any policy adjustment must be bound into new preregistered evidence.

`risk` supplies empirical held-out calibration bins with site, predicted
deadline-miss probability, actual miss count and trial count. Calibration
sites must be disjoint from training and confirmation. At least 1,000 trials,
both outcomes, two distinct probability bins and empirical weighted absolute
calibration error at most 0.02 are required. All-zero labels cannot establish
useful risk discrimination. Calibration is reported as **empirical only**;
an asserted distribution-free/exchangeability certificate is rejected. The
checker cannot establish distributional exchangeability or guarantee transfer.

ECE is only a diagnostic. The protocol also registers every risk `bin_id`,
whether its predictions may be `admitted`, and its risk `ceiling`. All those
bins must appear exactly once. Each bin has matching raw binary outcome and
reset-identity ledgers, so a stated trial count cannot inflate exposure. Its
one-sided episode upper bound must be covered by the reported conservative
probability and stay below the preregistered ceiling. Every admitted bin must
also meet the operator's **10 ppm** limit. Thus predictions near `1e-8` with
1% observed misses fail even when their global ECE is below 0.02. At least one
admitted bin is required; rejecting every action cannot qualify a useful model.

These finite-sample statements are **conditional on the reviewed sampling
assumptions**. Hashes establish byte identity; they do not prove that measurements
are honest, reset episodes are genuinely distinct or IID, or that a reset-episode
distribution transfers to continuous warehouse operation. Arbitrary dependence
inside an episode is allowed because only its prespecified single command is
sampled. Between-episode independence and the registered stationary scope remain
explicit assumptions requiring review. This checker does not infer them from
timestamps, assign a generic exchangeability certificate, or qualify operation
under an unsupported shift.

## Physical execution prerequisites

The `hardware` receipt must identify an actual physical-device run and include
at least 1,000 raw samples. Every sample records observation, inference,
transport, actuator and total **end-to-end milliseconds**, plus an explicit
timeout flag. Total latency cannot be less than the sum of its stages, no
sample may time out or exceed the 50 ms deadline, and p99 must meet its cap.
Sample IDs must be unique and monotonic start timestamps strictly increasing;
copying one successful measurement a thousand times cannot inflate its count.
A neural-head microbenchmark or simulated timing profile cannot replace this.

The `rollback` receipt must identify the same physical device and contain an
ordered `outcome_unknown → status_read → baseline_verified` event trace.
Reconciliation must verify the bound baseline controller through hardware
status within the default 200 ms recovery budget. A boolean assertion that
rollback passed is insufficient. The final observed baseline endpoint,
interface and role must match the bound profile. The checker only reports
eligibility for that profile/device evidence. Production also requires review of the
underlying physical measurement and operator authorization; hashes prove
byte identity, not that a measurement was honestly performed.

`proposal_only` artifacts **always** have `execution_eligible: false`, even
if their other evidence passes. They may separately have `sandbox_eligible`
or `latency_shadow_eligible`; the latter also requires physical timing evidence.
`active_candidate` can receive an eligible dossier result only when all checks
pass. This module has no bridge that turns that result into radio actuation.

## Verification

```bash
PYTHONPATH=src uv run --frozen python -m pytest tests/test_production_qualification.py -q
```

Positive tests use clearly named synthetic fixtures in temporary directories,
with plausible native units and one million independently indexed episode
outcomes per test cohort; the default 10 ppm budget is not relaxed to make them
pass. They are never written into the actual candidate
inventory or represented as hardware evidence. Repository tests independently
check that all current candidates remain blocked.
