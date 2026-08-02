# Low-compute sub-nodes

A drone or small ground sensor cannot carry a Jetson, but it can carry a radio
and a link to a vault node. This document defines what such a sub-node runs, what
it must not run, and how it degrades when the link to its vault node is lost.

## The split: cheap at the edge, expensive at the vault

Kval's loop divides cleanly by computational cost.

| Stage | Cost | Runs on |
|---|---|---|
| Parse OS/link/sensor evidence | trivial | sub-node |
| Verify signed policy (Ed25519) | ~microseconds | sub-node |
| Verify + redeem a capability lease | ~microseconds + one SQLite write | sub-node |
| Spectrum band assessment (thresholds) | trivial | sub-node |
| Power/thermal governance (thresholds) | trivial | sub-node |
| Connectivity recovery FSM | trivial | sub-node |
| Deterministic profile optimizer | trivial | sub-node |
| Multimodal fusion across nodes | moderate | vault node |
| Forecasting / learned policy inference | high | vault node |
| Training, evaluation, federated aggregation | very high | DGX coordinator |

Everything in the "sub-node" rows is pure Python over small structured inputs:
no accelerator, no model weights, no TensorRT, and state measured in kilobytes.
That is the entire safety-critical path — observation, verification, bounded
reaction — and it is deliberately the cheap part.

## The thin-path modules

A sub-node build needs only:

* `validation.py`, `models.py` — types and bounds checking
* `signing.py` — Ed25519 policy/document verification
* `lease.py` — capability-lease verification and single-use nonce store
* `spectrum.py` — `SpectrumPolicy`, `SpectrumObservation`, `SpectrumAnalyzer`
* `power.py` — `PowerPolicy`, `parse_tegrastats` (or a local equivalent reading),
  `PowerGovernor`
* `recovery.py` — `LinkStateMachine`
* `optimizer.py`, `policy.py` — deterministic scoring and the Safety Shield
* `audit.py` — hash-chained ledger

`hardware.py` (the Jetson platform gate), `capabilities.py` model paths, and any
learned inference are **not** part of a sub-node build. The Jetson gate is
specific to `kval-node preflight` and the full node service; the thin path above
does not import it.

> Note: this repository is packaged today as a single Jetson-targeted
> distribution. The module split above is the supported boundary for a sub-node
> build; producing a separate reduced wheel is a packaging exercise that has not
> been done here, and is not claimed as done.

## What a sub-node may do locally

Under its last valid signed policy, and only within it:

* sense passively on approved bands and assess band status;
* enforce its power/thermal/duty limits;
* run the link recovery state machine (probe, backoff, reconnect, failover);
* select among *its own* approved connectivity profiles via the deterministic
  optimizer and the Safety Shield;
* verify and redeem an operator lease it already holds;
* append to its local audit chain.

## What a sub-node must not do

* Run or trust an unsigned model, policy, or catalogue.
* Mint, widen, or renew its own authorization.
* Transmit on, or reconfigure, any radio parameter — sensing is receive-only and
  connectivity is limited to activating an approved connection.
* Fabricate an observation to fill a gap. Missing evidence yields
  `INSUFFICIENT_EVIDENCE` / abstention, never a guess.

## Degradation when the vault link is lost

1. **Link degrades** — the recovery FSM moves `HEALTHY → DEGRADED`, increases
   probing, and holds the current profile.
2. **Link fails** — `FAILED`; reconnect with exponential backoff (bounded by the
   policy cap), then failover to the next approved profile after attempts are
   exhausted.
3. **Repeated flapping** — once the hourly flap ceiling is hit, the FSM stops
   thrashing the radio and escalates for operator attention.
4. **Prolonged isolation** — the sub-node keeps running the thin path on its last
   signed policy. When that policy **expires**, every policy-gated capability
   fails closed: spectrum sensing stops returning assessments, the power governor
   has no valid limits, and no lease can be redeemed against it. The node retains
   its current connection and keeps auditing; it does not silently continue on
   stale authority.

That last point is the important one: expiry is a safety property, not an
inconvenience. An isolated node that cannot prove what it is allowed to do stops
doing it.

## Energy notes

The controls that matter most for a battery-powered sub-node are already bounded
by signed policy: scan interval, telemetry interval, inference duty, and (on
Jetson-class hardware) the `nvpmodel` power mode from an operator allowlist.
`PowerGovernor` tightens all of them together under thermal, battery, or
sustained-watt pressure, and applies hysteresis so the node does not oscillate at
the thermal ceiling.

Absolute power figures are **not** claimed here. Energy per bit, sensing duty
cost, and battery life must be measured on the real sub-node hardware and
recorded as evidence.
