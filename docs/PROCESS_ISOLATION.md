# Opt-in process isolation for proposal workers

`wirelessjev.process_worker.ProcessProposalWorker` runs one reviewed inference
callable in a fresh Python process created with the `spawn` multiprocessing
context. It does not change the existing thread services, admission defaults,
model qualification, command execution or any radio setting.

The intended contract is **proposal computation only**. The child must not own
physical actuation handles or make external state changes. Killing a process
cannot undo an action it already sent, and a lost response cannot establish
whether such an action happened. Durable command execution needs its separate
reservation/ownership/settlement contract.

## Callable and payload contract

Supply a trusted import path such as `my_package.adapters:predict`, or use
`factory=True` with an importable factory returning a callable. The factory
runs inside the child and may initialize its model from reviewed checkpoints.
Factory keyword arguments are copied through the same bounded passive codec.
Application entry points must follow Python's spawn rules, including an
`if __name__ == '__main__':` guard. Configured modules must be importable without
starting more workers as an import side effect.

The callable receives one payload. Supported data are finite primitive values,
string-keyed dictionaries, lists/tuples and NumPy arrays of ordinary boolean,
integer and floating dtypes. Tuples return as lists. Arrays travel as bounded
base64 data with checked dtype/shape envelopes, then are copied on decoding.
Objects, dataclasses, tensors with attached execution state, object arrays,
nonfinite numbers and arbitrary Python reducers are not an IPC format.

No worker-produced pickle is deserialized in the parent. Python's own trusted
spawn bootstrap still uses its standard process-launch protocol. Factory
configuration is sent over the bounded socket after launch, rather than adding
large arguments to that bootstrap pipe. Child model logs and exception text are
not forwarded; error telemetry contains a bounded exception type.

Default wire limits are **1 MiB per input and output**, configurable from 256
bytes to 16 MiB. The actual encoded frame is checked in both directions, with
additional nesting/node/allocation limits. Encoding overhead counts against the
limit. This bounds communication and wrapper-owned pending data, not the total
memory a model allocates inside its process. A 100-robot history may exceed the
default limit; a future adapter must choose an explicit bounded schema and
preserve its absolute observation-time semantics rather than silently clipping
an existing model's prefix.

## Submission and deadlines

Construction starts the child and waits for its initialization handshake, with
a five-second startup budget by default. Initialize or restart outside the
actuation loop. OS process creation and Python bootstrap scheduling are not
hard realtime operations.

`submit(payload)` accepts at most **one outstanding request or unread result**.
It copies/serializes the bounded payload and returns `False` for backpressure,
invalid input, a busy lifecycle lock, a closed service or a disabled worker.
No unbounded job queue or replacement pool is created. Invalid input is rejected
before model dispatch and does not replace a healthy process.

`poll()` returns a completed decoded result once, or `None` for baseline fallback.
A callable may also deliberately return `None` to abstain. It performs bounded
serialization work and nonblocking local socket I/O; it never joins a child.
Both `submit()` and `poll()` refuse to wait for a concurrent lifecycle lock.
Serialization and OS scheduling still consume time, so these properties do not
establish a 50 ms robot deadline.

The request budget uses the parent's monotonic clock and includes preparation,
IPC and consumption of the response. The default is one second; per-request
and startup budgets must be finite, positive and at most sixty seconds. A reply
that is not consumed before its deadline is discarded even if the child computed
it earlier. No stale result becomes fresh simply because it arrived late.

**The application must keep calling `poll()` or `maintain()` regularly.** This
component has no autonomous watchdog thread. Those calls detect expiry/crash,
disable advice, request termination, and then escalate to kill after the shutdown
grace interval. They opportunistically reap without waiting. If supervision stops
entirely, this component does not promise to kill a hung callable on its own.

## Cleanup, restart and ownership

`close()` is idempotent. It requests termination, waits up to the configured
shutdown grace, escalates to kill, waits one further grace, and reaps/releases
the process handle. The default grace is 100 ms, configurable up to two seconds.
A return value of `False` means cleanup or an input preparation is incomplete.
The object retains ownership of that process; `restart()` refuses to create
another child until it has been reaped. Concurrent lifecycle calls are serialized
and can additionally wait for an in-progress startup budget. Run maintenance and
lifecycle operations outside hard-deadline actuation.

Crash, timeout, invalid output and worker exceptions latch the component disabled.
Only an explicit `restart()` attempts a new child. That restart has cold model
state: an adapter must reseed any temporal context before using its proposals.
It cannot recover discarded packets or an earlier process's uncommitted state.

An instance belongs to its creating parent PID. Public methods check ownership
before acquiring inherited locks or touching sockets/process handles. An object
inherited by `fork` returns fallback/refusal and cannot shut down its original
parent's socket. Create a fresh owned instance instead. This is particularly
important if another thread held the original instance's lock during the fork.

The tests verify that all owned inference child PIDs are reaped. CPython may keep
its shared multiprocessing resource-tracker helper until application exit; this
is not a replacement inference worker. A kernel-stuck process may remain
unreapable despite kill. In that case the API reports incomplete cleanup and
refuses replacement; it cannot promise stronger OS behavior.

## Scope and verification

Process separation contains many Python/native model failures and permits actual
termination that a thread pool cannot provide. It is **not a security sandbox**:
the trusted callable retains the child's OS permissions, filesystem/network
access and inherited environment. It must not start unmanaged descendants.
Process groups, credential minimization, container/cgroup memory/CPU limits and
hardware scheduling require deployment-specific isolation. Wire byte limits are
not total-RSS, GPU-memory or CPU-time limits.

`tests/test_process_worker.py` launches actual spawned children. It checks copied
input/output, factories, one-request backpressure, malformed/nonfinite/oversized
output, process crash, a SIGTERM-ignoring hung callable, terminate-to-kill
escalation, reaping, explicit restart, startup failure/hang, finite configuration,
foreign-parent ownership and nonblocking proposal calls during lifecycle work.
These are software-containment tests, not physical-car evidence, SOTA results or
production qualification. No existing service is migrated or activated here.
