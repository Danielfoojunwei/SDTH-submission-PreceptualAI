# Restored didjital R6.1 reference

Repository: `Danielfoojunwei/didjital`.
Branch: `wirelessjev/swarmos-10-r6-1-20261022`.
Immutable source: [`a458a90d34fe559addb876a2a1c69213c6ecc968`](https://github.com/Danielfoojunwei/didjital/tree/a458a90d34fe559addb876a2a1c69213c6ecc968).
The cloud checkout is `/workspace/didjital-r6.1`; no source files there were changed.

## Executed software qualification

| Check | Result |
|---|---|
| Didjital root tests | 3 passed |
| Legacy WirelessJEV reference tests | 67 passed |
| Preceptual R6.1 verification with graph tests required | 117 passed, no failures/errors/skips |
| Root mission-planning CLI | Passed; plan written locally |
| Frozen ranker, 11-node synthetic demo | Passed; model SHA-256 unchanged |
| Native C++ decision gate (`-Wall -Wextra -Werror -O2`) | Compiled and passed |
| R6.1 synthetic campaign | 10 selected, no reservation overloads |

The machine-readable summary is `artifacts/r6_qualification/report.json`.
The full R6.1 runner output is retained under the restored checkout's
`preceptual_r6/artifacts/run-*/`. Its numerical PHY/RSMA results and graph
tests are synthetic software evidence, not a commissioned network or a
production graph-model training run.

## Keep the packages separate

This repository and the older reference both use the Python name
`wirelessjev`, but expose different models and APIs. Installing both into the
same environment would make imports ambiguous. The restored root package uses
`/workspace/.venvs/didjital-r6.1`. Run its reference tests from the documented
component directory. The existing research training environment is unchanged.

```bash
cd /workspace/didjital-r6.1
/workspace/.venvs/didjital-r6.1/bin/python -m pytest -q

cd /workspace/didjital-r6.1/wirelessjev
/workspace/.venvs/didjital-r6.1/bin/python -m unittest discover -s tests -v

cd /workspace/didjital-r6.1/preceptual_r6
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/SDTH-submission-PreceptualAI/.venv/bin/python tools/verify.py --require-graph
```

The last command uses the existing PyTorch installation while the working
directory selects the local `preceptual` package. It does not install the older
`wirelessjev` package into the research environment.

## Integration boundary

| Current research component | R6.1 reference | Work still required |
|---|---|---|
| Exact planner and candidate outcomes | Proposal and mission-contract checks | A typed read-only proposal adapter, unit/schema parity and replay tests |
| Dry-run transaction ledger | Site reservations and durable execution journal | Explicit resource-unit, ownership and UNKNOWN-outcome reconciliation mapping |
| Learned intent/readout or radio encoder | Frozen graph model/release identity | Separate schemas, dataset provenance and validated model release; weights are not interchangeable |
| Supervised SmartCar tooling | Operational edge-agent authority boundary | Exact private source audit, identity provisioning and actual one-car qualification |

The checkout and software qualification are complete. No runtime bridge was
introduced, no production release was authorized, and no hardware, SSH,
firmware, motor or radio operation was performed. The private SmartCar archive
is not present in this cloud workspace, so its source audit and legacy GUI/SSH
regression checks remain outstanding. Do not substitute synthetic reference
results for those prerequisites.
