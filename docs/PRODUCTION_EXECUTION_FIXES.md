# Process-local execution and recovery fixes

The active execution API now enforces ledger ownership, atomic single-use issue
and idempotent reservation settlement. This fixes concrete software failure
paths; it does not qualify a real radio controller or supply crash recovery.
No hardware command or network session was used to implement or test this change.

The later opt-in [durable wrapper](DURABLE_EXECUTION.md) adds local SQLite
obligation recovery around this API. The process-local limits below describe
the core ledger; callers need the wrapper for its tested restart behavior.

## Active version and historical evidence

[`execution.py`](../src/wirelessjev/execution.py) declares
`EXECUTION_SEMANTICS_VERSION = 2`. New results from
[`closedloop.py`](../src/wirelessjev/closedloop.py) carry
`execution_semantics_version: 2`, including the CLI report envelope. They use
the corrected reservation semantics and must not be presented as replays of
historical version-1 results.

Before editing, the current-hash checks for the contracted training/evaluation,
closed-loop confirmation, service-world, packet-value and frontier selection
capsules were inspected. Those manifests do not bind the two active root-level
execution modules. Older complete-source/provenance records do contain them;
their archived bytes, manifests and outcomes were left unchanged. The archived
root-level copies in `artifacts/contracted/complete_source` still match:

| Module | Historical SHA-256 |
|---|---|
| `src/wirelessjev/execution.py` | `e21ec87f3c544db76868536e15145d0d37c0f8e327d7d901ab32971b59e0c0f3` |
| `src/wirelessjev/closedloop.py` | `9ad3093a2c2670a0033f5deb16c898f91e24b195be09a19a025727d3b9aa6e3b` |

Use an executed historical capsule to reproduce its old behavior. This change
does not rewrite sealed source or update old metrics to the new semantics.

## Admission and single-use issue

`Ledger` accepts finite nonnegative domain budgets and airtime, and a positive
integer in-flight limit. Admission checks run under one ledger lock. A duplicate
transaction identity cannot replace its owner, a car cannot have two unresolved
operations, and concurrent admission cannot exceed the domain/in-flight limit.
Unknown domains are not admitted. A new in-memory ledger refuses populated
reservation/transaction dictionaries because they are not a verified recovery
journal.

Admission snapshots the car, origin/target BSSID, client interface/role and
credential-free connect request. The subsequent operation must be the exact
registered `Transaction` object with that unchanged identity and request.
Changing mutable request annotations such as `_airtime` cannot change reservation
ownership. The transport receives a separate copy of the admitted wire request.

`execute()` atomically claims an admitted PREPARED operation before releasing
the lock for transport I/O. A concurrent duplicate or any later call for an
already issued, UNKNOWN or terminal operation sends nothing. A private issued
flag prevents a caller from enabling another send by resetting the public
transaction state. An unregistered/impostor or altered pre-issue transaction is
rejected before transport use.

## Reservation ownership and outcomes

The ledger owns reservation amounts separately from mutable requests. Terminal
COMMITTED, ABORTED and pre-issue CANCELLED outcomes settle the reservation once.
`release()` is idempotent and cannot free PREPARED, ISSUED or UNKNOWN work.
`cancel()` is limited to a never-issued PREPARED operation.

The old closed-loop caller released a clean failure after `execute()` had
already released it. That could subtract another live operation's reservation.
The caller no longer settles terminal operations: execution and reconciliation
own settlement. UNKNOWN-to-COMMITTED reconciliation now releases its reservation
too. These changes preserve the identity of every unresolved reservation.

Successful connect replies require literal `ok: true`, `connected: true` and
the expected SSID/BSSID. Supplied interface, role, car or transaction identity
must agree with the admitted operation. Existing SmartCar connect replies do
not always include interface/role, so those fields are checked when supplied
on the initial reply and required on reconciliation status.

An explicit `ok: false` with `activation_attempted: false` establishes a clean
nonapplication. `PreSendRefused` is a separate transport assertion that nothing
was sent. A generic exception, malformed reply, truthy string in `ok`, wrong
connection identity or ambiguous outcome becomes UNKNOWN and retains its
reservation. An arbitrary `PermissionError` alone is not proof of nonapplication.

## Fresh status reconciliation

Only a ledger-owned UNKNOWN transaction can claim a reconciliation query, and
only one such query can be outstanding for it. The reconciler sends **status**,
never another connect. A resolving status needs:

- literal `ok: true` and `connected: true`;
- the expected client interface and `WIFI` role;
- a syntactically valid MAC BSSID matching the admitted old or new association;
- matching car/transaction identity when those fields are supplied;
- matching target SSID when supplied for the new association;
- a finite, forward-moving query/completion clock within `max_status_age_s`.

Symbolic AP identities are accepted only by the existing `DryRunTransport`,
preserving digital-twin compatibility without authorizing physical radios.
Missing BSSID is never treated as proof of the previous state, including when
the original association was `None`. Other APs, disconnected replies, malformed
status, wrong interface/role and failed reads retain UNKNOWN.

Legacy hardware status has no required measurement timestamp. Its freshness
therefore relies on the transport fulfilling a new synchronous status query;
the generic API cannot detect an adapter silently returning cached data. If
`observed_at_monotonic` is supplied, it must use the reconciler's clock and lie
within that query. Old, future or invalid timestamps cannot settle the operation.
The maximum query age rejects late output; it cannot interrupt a transport that
blocks. An actual deployment still needs bounded transport I/O and clock support.

## Verified scope and remaining production gap

[`test_execution_recovery.py`](../tests/test_execution_recovery.py) exercises
invalid budgets, duplicate identities, concurrent admission/issue, immutable
request identity, cancellation, repeated release, ambiguous sends, malformed
replies, stale/invalid status and both terminal reconciliation outcomes.
The new recovery tests plus existing contract, one-car and handover tests pass:
**90 tests**, run with one BLAS/OpenMP thread. These are offline
fault/concurrency tests, rather than measurements of model or RF performance.

The ledger remains **process-local and memory-only**. There is no durable
write-ahead issue journal, restart recovery, multi-process coordination or
bounded history-retention policy. A process crash can lose an ambiguous
operation and its reservation. Do not infer exactly-once physical execution or
production crash safety from the within-process single-send property. A future
physical bridge must persist issue/reconciliation obligations before sending,
recover them before new mutations, bind capabilities/device identities and
validate real status/timeout behavior. The generic SSH transport remains a stub;
the separately supervised one-car path is not activated by this change.
