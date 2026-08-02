# The closed loop and the cross-node fabric

Two gaps are closed here. Until now the capability modules were libraries with
tests and no production caller — `service.py` imported none of them, so the
running node behaved exactly as it had before they existed. And each node sensed
alone: the vault/sub-node topology was documented but no code moved an
observation between nodes.

## The node loop

`KvalNode.run_once` is now:

```
observe telemetry
   -> check clock integrity
   -> sense spectrum        (optional, fail-closed)
   -> govern power          (optional, fail-closed)
   -> advance recovery FSM
   -> rank profiles, demoting contested bands
   -> fuse modalities with provenance
   -> Safety Shield
   -> activate or advise
   -> audit
```

Every capability loop is independently optional. When one is unconfigured it
reports `available: false` with a reason and contributes nothing, so a node
configured exactly as it was before this change behaves exactly as it did
before. The full cycle — including each loop's result — is written to
`/run/kval/latest.json` and to the audit chain under `capabilities`.

### How sensing reaches the decision

Spectrum evidence enters in exactly **one direction: it can demote a profile,
never promote one.**

A signed catalogue profile may declare `spectrum_band_ids` — the policy bands
that profile's radio depends on. The operator declares this; the node does not
guess which frequencies a connection uses. When the local sensor reports one of
those bands as `JAMMING_SUSPECTED` or `ANOMALOUS` with non-zero confidence, that
profile's utility is multiplied by `SPECTRUM_DEGRADED_MULTIPLIER` (0.6) and
carries the reason code `SPECTRUM_BAND_CONTESTED`.

The asymmetry is deliberate. A quiet band is not evidence that a link works —
only the live probe is. So sensing is allowed to warn the optimizer off a
contested medium and nothing more; a profile without complete probe evidence
stays ineligible no matter how clear its band looks. A profile that declares no
bands is unaffected, which is why a node without spectrum sensing is unchanged.

### Clock integrity

`timesync.py` measures whether the local clock is actually disciplined, from
`chronyc tracking`, `timedatectl`, and the divergence between `CLOCK_REALTIME`
and `CLOCK_MONOTONIC` over the process lifetime. It fails closed: no time daemon
means not trusted.

This matters because freshness checks are the backbone of every fail-closed gate
in the node, and they stop meaning anything on a stepped clock. The verdict is
carried as the `security` modality in the fused state, and it gates cross-node
correlation below.

## The cross-node fabric

A sub-node publishes a **signed report** of what its radio measured. A vault node
verifies each report against an operator-signed roster and fuses what survives.

```bash
# on a sub-node
kval-node fabric-report --config /etc/kval/node.json --output /srv/outbox/nodeA.json

# on the vault node, over reports collected into the inbox
kval-node fabric-view --config /etc/kval/node.json
```

Three properties are enforced:

**Every report is signed and node-bound.** A report is verified against its
author's public key from the signed roster. An unknown author is refused with a
reason and does not affect the view — a fabric that accepts anonymous
observations is a poisoning surface (threats 14 and 17 in the EW threat model).
A stranger signing a report that *claims* to be a rostered node is refused too,
because the key ID must match the roster entry.

**Correlation requires comparable clocks.** Each report carries its author's own
time-integrity verdict. A report from an undisciplined clock still counts for
presence, but is excluded from corroboration — two observations that cannot be
aligned in time cannot confirm each other.

**Corroboration is counted, not assumed.** A band is only `interference_confirmed`
when independent nodes with trusted clocks agree. One node reporting jamming is
an observation; a quorum agreeing is evidence. A band is `clear_everywhere` only
if every reporting node found it usable, so a single node seeing interference
removes it from `fleet_clear_band_ids`.

## The training corpus

`dataset.py` turns the audit ledger into a labelled corpus, which is the data
path the learning guardrails were missing.

```bash
kval-node export-corpus --config /etc/kval/node.json --output corpus.jsonl
```

Each example pairs the features available **at decision time** with an outcome
measured from a **later** cycle, so a model trained on it learns to predict
rather than to describe. Two properties are enforced:

* **No leakage** — features come only from the record being labelled; the
  outcome comes only from later records.
* **Verified provenance** — export refuses to run against a ledger whose hash
  chain does not verify. NIST AI 100-2e2025 names poisoning as a primary attack
  class, and the ledger is the obvious thing to attack.

A record with no successor inside the horizon is dropped rather than labelled
from nothing, and the export reports how many were dropped and why.

## Low-compute sub-nodes

The sub-node boundary is real and is now pinned by a test: the thin modules and
the capability entry point (`capabilities.py`) **do not import the Jetson
platform gate**. Only the full `service.py` loop does.

That means the same wheel already serves both roles — the boundary is which
commands you run, not which package you install:

| Command | Needs Jetson |
|---|---|
| `spectrum-scan`, `power-plan`, `transport-preflight`, `verify-lease`, `fabric-report`, `fabric-view`, `export-corpus`, `verify-catalog`, `verify-audit` | no |
| `once`, `run`, `benchmark`, `hil-activate`, `preflight` | yes |

A separate reduced wheel was previously listed as missing work. It is not needed:
the reduction is achieved by the import boundary, and the test enforces it so a
future change cannot quietly make a drone build refuse to run on its own
hardware.

## Not claimed

The fabric is verified end to end in software — signing, roster verification,
rejection of unrostered authors, quorum corroboration, clock-gated correlation.
It has **not** been run across physically separate nodes. Multi-node geolocation
(TDOA or RSSI multilateration) is not implemented: it needs the fabric plus real
per-node position and calibrated timing, and would be dishonest to claim from
occupancy reports alone.

No model is trained. The corpus exporter produces the data a model would need;
training, evaluation, and promotion remain outside this repository, and the
learned path still abstains until a signed manifest verifies.
