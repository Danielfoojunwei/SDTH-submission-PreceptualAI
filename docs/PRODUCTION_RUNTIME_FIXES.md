# Runtime fault recovery and bounded shadow work

These changes harden opt-in software proposal services. They do not activate a
learned packet critic, alter RadioMap admission, change radio settings, or
establish hardware qualification. Existing research source capsules, evidence,
checkpoints and latency profiles remain unchanged.

The separate [spawned proposal worker](PROCESS_ISOLATION.md) now implements
supervised terminate/kill/reap for proposal computation. The thread services
described here retain their original behavior and are not migrated by it.

## Forecast proposals

`AsyncForecastService` owns one real `ThreadPoolExecutor` worker and a bounded
pending queue. Its positive integer `max_pending` defaults to two. Queue
backpressure, invalid submissions and calls after close return `False`.
Input snapshot construction reserves that same capacity before allocating,
so concurrent submitters do not bypass the queue's memory bound.
`poll()` returns `None` whenever an acceptable completed forecast is unavailable;
the caller must continue its existing baseline control.

The service validates shapes and finite real numeric fields before copying a
history. Scan/heard masks must be boolean and consistent. Routes must have
shape `N × 9 × 2`; histories must have matching timestamp, robot and link axes.
Each immutable submitted snapshot has explicit limits: **600 timestamps** and
**64 MiB** by default. The same limits protect direct
`OptimizedWorldModel.forecast()` calls. This is a finite-prefix interface, not
an unlimited rolling-stream implementation. A caller reaching the limit must
retain baseline control and manage an explicit, quiescent reseeding boundary.
Silently clipping a prefix would corrupt the existing fleet-memory time index,
so this implementation rejects it instead.

Submission clocks and polling clocks must each be finite and nondecreasing.
Polling before a forecast's issue time returns `None`. Predictions expire
**five seconds after issue** by default; configuration must remain within the
30-second forecast support. A completed but already expired job cannot become
a fresh result merely because its inference finished late. These checks use
the caller's logical clock; they do not authenticate upstream observation
timestamps or synchronize device clocks.

Worker results must contain correctly shaped finite floating arrays,
nonnegative uncertainty and bounded scalar metadata. The wrapper copies the
validated result before publishing it, so mutation of a worker-owned output
cannot change a cached forecast. Invalid output and worker exceptions clear
cached advice and disable that service instance. Subsequent control uses the
baseline until a fresh, known model/service is initialized. This conservative
failure latch avoids reusing model memory that may have been partially changed
before an exception. Worker-originated base exceptions are contained inside the
worker; the control thread's own interrupt is not swallowed.

Diagnostic events use a bounded ring (64 records by default). They retain
status and exception type rather than arbitrary exception messages or tracebacks.

## Packet critics remain shadow-only

`ShadowPacketAdmission` evaluates and copies the authoritative baseline mask
before doing any critic work. Its result is that exact mask after critic
abstention, rejection, invalid advice, exception, executor submission failure,
queue saturation, or close. Baseline contract errors remain visible; the wrapper
does not invent an admission decision if the baseline itself fails.

A critic receives a separate `RoamDecision` snapshot with copied arrays. It
cannot mutate the original controller context through that supplied argument.
Snapshot allocation is capped at **1 MiB** by default. Only one critic future
can be outstanding. Calls during that work skip new critic jobs and immediately
return the baseline result. Telemetry uses a bounded ring of **128 records** by
default. An outstanding job may additionally retain its single bounded record
until completion. Critic code still runs in the same process: this is input
isolation, not a sandbox against arbitrary global side effects or allocations.

## Shutdown semantics

Both `close()` methods are idempotent and nonblocking with respect to worker
completion. They reject new work, mark queued wrappers to skip inference, and
call executor shutdown without joining its thread. They deliberately avoid
`Future.cancel()` on the control thread: cancellation invokes arbitrary future
callbacks synchronously and could itself hang control. Queued wrappers drain
on the one existing worker, which also runs completion callbacks. `False`
means that tracked work or its worker thread is still alive. `True` means both
have finished. Calls after close stay on the baseline and never start another
worker.

Python cannot kill a running thread safely. A worker that hangs can retain its
bounded in-flight/queued snapshots and model state, and Python may still wait
for that thread during interpreter shutdown. Hard termination, total process-memory limits and strict
CPU isolation require a supervised process boundary. Creating replacement
thread pools indefinitely would defeat the resource bound and is not a recovery
strategy implemented here. No hardware deadline follows from these software
shutdown semantics.

## Verification

`tests/test_runtime_recovery.py` uses real thread-pool workers and synchronization
events. It exercises worker exceptions, invalid/nonfinite results, stale and
backward clocks, finite-history and byte limits, caller/output mutation,
submission to a shut-down executor, saturated queues, bounded telemetry,
worker-initiated close, blocking future callbacks and deliberately blocked workers. Each blocked
test releases its worker in cleanup; passing close tests does not imply that a
thread was killed. Existing forecast-value and packet-outcome equivalence tests
continue to check the successful paths.

The historical CPU measurements describe their archived implementation. The
new validation and copying overhead has not been qualified on a SmartCar.
