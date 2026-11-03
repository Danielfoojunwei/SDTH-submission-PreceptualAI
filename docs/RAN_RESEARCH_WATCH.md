# Bounded RAN / JEPA research watch

The watch collects **unreviewed bibliographic leads**, with identifiers and
source receipts. It does not establish SOTA, verify experimental claims, or
qualify a model for WirelessJEV. Search results never trigger execution,
repository changes, dependency installation, model/PDF downloads, controller
promotion, GitHub comments or external messages.

[`research_watch.py`](../src/wirelessjev/research_watch.py) queries only the
official HTTPS discovery endpoints:

- `https://export.arxiv.org/api/query`
- `https://api.openalex.org/works`

Five fixed topic searches cover wireless JEPA, wireless world models,
predictive representations for network control, RAN control/resource allocation,
and wireless deadline risk/uncertainty. These searches are finite and imperfect;
their relevance and coverage require human review. Metadata is not a literature
review. A successful empty response cannot establish that no relevant work exists.

Run from the repository root:

```bash
PYTHONPATH=src uv run --frozen python scripts/update_ran_research.py \
  --output-dir artifacts/ran_research/watch \
  --sources arxiv openalex --since 2026-01-01 \
  --max-results 5 --timeout 5 --max-attempts 1 --max-run-seconds 90
```

The default search uses both sources and five results per source/query. Hard
bounds are 50 results, 2 MB per provider response, 30 seconds per socket/response
budget, three attempts and a 600-second request-start budget. Retries apply only
to transient transport errors and HTTP 408/429/5xx statuses. Retry waits are
capped at five seconds; arXiv requests are paced at least three seconds apart.
Only two redirects are permitted, and each must stay on the exact official
HTTPS endpoint. Read loops check elapsed time as well as byte limits to stop
trickling responses. The overall budget stops new requests/retries; an already
blocking socket operation may finish after that deadline. Scheduled execution
should retain a separate job timeout.

Publication/update timestamps after the retrieval clock are explicitly excluded.
The inclusive `--since` window is sent to both providers and checked again on
returned metadata. Malformed entries, invalid identifiers, provider result-cap
violations, malformed feeds, authentication failures, timeouts and unavailable
sources produce explicit partial/failed request records. They are never silently
converted into a successful empty search. An unqueried provider also marks
coverage incomplete. arXiv/DOI versions are normalized and records are merged
only through exact arXiv, DOI or OpenAlex identifiers, including arXiv DOIs.
Title similarity never merges papers.

The output directory contains:

- `latest.json`: current snapshot, with per-query outcomes, exact request URLs,
  response hashes, excluded entries, lead records, changes and coverage flags.
- `snapshots/`: immutable dated snapshots and retained previous snapshots. Missing
  papers remain in history as `retained_previous_not_seen`; they are not deleted.
- `responses/<sha256>.bin`: actual bounded provider response bytes, including HTTP
  failure bodies. Every retained response has a hash, byte count and relative path
  in its attempt receipt. Bodies are metadata/API responses; they are not paper
  or model downloads. Transport failures have an error receipt and no invented body.
- `.publish.lock`: local nonblocking writer lock. A competing publisher or changed
  `latest.json` causes publication to fail without replacing the current snapshot.

Writes use a same-directory temporary file, file `fsync` and atomic replacement.
Previous snapshots are validated before publication. A malformed previous file,
missing/tampered response body, disk failure or stale writer preserves the old
`latest.json`. Snapshots have a 20 MB size cap; retention pruning or archival
requires an explicit maintenance decision. The watch does not silently discard
old records to fit that limit.

New records always start as `lead`. An existing explicit review is retained only
when relevant metadata is unchanged and the prior record includes reviewer,
review date and evidence URL. Changed metadata returns the record to `lead` and
retains the earlier review as history. No discovered item receives automatic
scientific review or production/SOTA status.

The CLI publishes a valid partial snapshot and exits **2** when coverage is
incomplete. A scheduler should preserve/upload those artifacts and report the
degraded outcome. `--allow-incomplete` changes only the exit code to zero; it
retains all incomplete-coverage warnings. It is not a scientific approval flag.
OpenAlex authentication/limits and arXiv availability can change independently
of this repository; the watch never bypasses provider access controls.

Network-free contract tests use explicit provider fixtures, simulated time and
controlled filesystem failures:

```bash
PYTHONPATH=src uv run --frozen python -m pytest tests/test_research_watch.py tests/test_research_workflow.py tests/test_research_restore.py -q
```

The retained live snapshot in `artifacts/ran_research/watch/` records what the
actual provider requests returned, separately from these unit fixtures. Read
its `coverage` and `request_results` before treating its leads as current.
All 44 metadata/history tests also pass in an isolated environment containing
only pytest; they do not require the model-training dependency stack.

## Daily workflow and activation boundary

[`.github/workflows/ran-research-watch.yml`](../.github/workflows/ran-research-watch.yml)
configures daily discovery at **08:00 Asia/Singapore (00:00 UTC)**. GitHub
scheduled workflows run on the default branch: the daily schedule becomes
active only after the workflow is merged there and Actions is enabled.
Push/PR changes and manual dispatch also request bounded verification. A local
successful discovery does not prove that a future scheduled run occurred.

The workflow uses pinned official action commits, read-only repository
permissions, a ten-minute job limit and metadata-only tests. A stable cache key
preserves history across implementation changes. Every restore checks the
latest trusted artifact metadata, even on a cache hit: a valid old cache must
not roll back a newer completed publication whose cache save failed. On cache
loss, the helper recovers a verified artifact from this repository's same
workflow on the default branch. Prior history without a recoverable artifact fails the
job; it cannot silently reset to the committed seed. Initial bootstrap is
allowed only when no prior default-branch run exists. Incompatible stores need
an explicit migration.

Verified evidence is uploaded with a requested **90-day retention**, subject
to repository policy, before default-branch history is cached. Invalid-store
diagnostics use a separate artifact name and cannot become recovery snapshots.
Artifact retention and caches are finite: expiry or deletion can still require
operator-managed archival recovery. Provider failures fail the job after
preserving available evidence. The 20 MB snapshot cap requires deliberate
archival maintenance; no evidence is silently discarded.
`scripts/verify_research_watch.py` checks every historical snapshot, predecessor
link, observation provenance and retained provider-response body, including
old failed-query responses.

Recovery uses the Ubuntu runner's Linux atomic directory exchange (`renameat2`)
and authenticated `gh` API calls. Unsupported exchange, API uncertainty,
divergent lineage, expired newest artifacts or exceeded storage/listing limits
fail without publishing a replacement. Temporary ZIP/API fixtures test these
paths; they are not evidence of a live GitHub disaster-recovery exercise.

Nothing in the workflow opens a PR, comments on an issue, downloads a model,
executes discovered code or changes the radio controller.

## Observed GitHub execution blocker

The initial push and PR checks did not execute any step. GitHub's annotation
states: **“The job was not started because your account is locked due to a
billing issue.”** [Retained CI receipts](../artifacts/ran_research/ci/status.json)
link both runs and their actual annotation bodies. Billing resolution is an
additional activation requirement; local tests cannot bypass the account lock.

A local read-only invocation of the restore helper against actual GitHub API
metadata succeeded with `initial_genesis_committed_seed`, verified ten response
bodies and retained 47 leads. It confirmed no prior default-branch runs existed.
Recovery from an actual prior workflow artifact remains unexercised live; its
failure/recovery cases are covered by explicit temporary ZIP/API fixtures.
