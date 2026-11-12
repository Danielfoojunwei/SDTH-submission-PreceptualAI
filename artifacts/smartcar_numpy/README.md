# Torch-free SmartCar package

[Download smartcar-wirelessJEV.zip](smartcar-wirelessJEV.zip) and follow the
[NumPy deployment guide](../../docs/SMARTCAR_NUMPY_DEPLOYMENT.md). Both numerical
mapping and the explicit three-member RadioJEPA shadow now need only NumPy/SciPy
on the computer. The Pi collector remains stdlib-only. No HF download, Torch,
Sionna, GPU or training pipeline is needed to run the packaged workflow.

The archive is **118,688 bytes**, with 35 members. SHA-256:
`088d148fbae7780327e52e133bbdf8996af89650ec327f34ef1bfe7cf794165f`.
Its [receipt](smartcar-wirelessJEV.receipt.json) binds source commit
`fb152a33e5d3bac3287786f47a48bdf224dabd5b`. All three numerical exports retain
every original float32 state array and bind the original checkpoint hashes
and full-state fingerprints. Only the exactly zero-gated learned-feature MLP
calculation is omitted; the learned discrepancy model, priors, Fourier features,
three Newton steps, line search, original jitter and all local state dimensions
remain. No retraining or new RF accuracy result is involved.

## Same-weight software comparison

The [published protocol](protocol.json) fixed three members, 192 context tokens,
192 queries, AP counts 1/4/16, single-thread workers, five warmups and 25 measured
conditioning-plus-render calls. All 18 planned worker processes completed.
The original [raw results](../smartcar_numpy_review/benchmark.json) retain every sample
and output; [environment metadata](../smartcar_numpy_review/benchmark_environment.json)
records the cloud CPU and package versions. A separately [published matching CPU-only protocol](cpu_only_protocol.json)
repeats every fixed case against `torch 2.14.1+cpu`; its [complete results](../smartcar_numpy_review/benchmark_cpu_only.json)
are the primary comparison below. The table shows medians across the three
member-specific median timings, and medians of peak process RSS.

| APs | NumPy condition + render | Torch condition + render | NumPy peak RSS | Torch peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.471 ms | 9.815 ms | 56.60 MiB | 255.42 MiB |
| 4 | 3.400 ms | 10.942 ms | 56.38 MiB | 255.26 MiB |
| 16 | 6.204 ms | 12.137 ms | 56.10 MiB | 255.39 MiB |

Member-specific warm speed ratios span **1.92–3.98×** and peak-process-memory
ratios **4.30–4.65×**. Loading and input assembly are outside the warm loop.
Worker-entry-to-first-result includes heavy imports/model setup but excludes
interpreter startup and top-level standard-library imports; its member medians
are approximately 0.20–0.23 seconds for NumPy and 0.93–1.00 seconds for CPU-only Torch.
These are descriptive timings for fixed synthetic software inputs, not
confidence intervals, sensor measurements, full workflow latency, real-Pi
performance, packet service or comparisons against RadioMap.
The original comparison against the installed `torch 2.14.1+cu130` wheel on CPU
is also retained: 1.91–4.13× warm speed ratios and 9.35–9.63× peak-RSS ratios.
It is a separate reference build, rather than the primary CPU-only baseline.
NumPy workers do not import Torch in either environment.

Across the nine paired cases, maximum absolute mean/spread differences are
**3.33e-7 / 4.77e-7 dB**, within the predeclared `1e-4` absolute / `1e-6` relative
software tolerances. The 58 model tests additionally cover censoring, points,
unknowns, AP permutations, dense Gaussian reference calculations and covariance.
This preserves predictions rather than improving them.

## Verification and remaining work

The [whole suite](../smartcar_numpy_review/software_suite.json) passes **4,138
tests**, with 606 source/config/test files unchanged across execution.
The [portable check](../smartcar_numpy_review/portable_verification.json) executes
all three actual exports in a clean environment without Torch, verifies all
34 shipped inputs and exercises the map/proposal workflow. Test data are
explicit software fixtures; no SSH request, real-car scan or actuation occurs.
The earlier package and scientific reports remain unchanged.

The default remains the numerical map. Neural outputs stay in shadow, proposals
transmit nothing, and a switch still requires an interactive terminal, fresh
role/status checks, durable intent, typed confirmation and the existing trusted
SmartCarCmd. Actual hardware access, measurements, supervised return and
held-out real-site comparisons remain required. No SOTA or real-car superiority
is established by this package or performance benchmark.
