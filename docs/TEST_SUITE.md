# Full test suite on this branch (15 Nov 2026)

## How to run
```bash
.venv/bin/python -m pytest --import-mode=importlib -q
```
- `--import-mode=importlib` is required. The test folders reuse file names (for example two different `test_native.py`), and the default import mode cannot load both; without the option 14 files fail to collect.
- Run it outside any sandbox: some tests start worker processes.
- It takes about three minutes.

## Result
**4,298 passed, 16 failed, 34 skipped** (4,348 collected).

The 16 failures also occur on Codex's unchanged tip
(`origin/codex/measured-radio-nuisance`, 2a043e54) on this Mac, with the
same tests failing. They come from this machine's setup, not from this
branch:

| Tests | Count | Cause |
|---|---|---|
| `tests/r6_bridge/test_r6_bridge_preview.py` | 11 | Expects an upstream checkout at `/workspace`, which does not exist here |
| `test_adaptive_simulation`, `test_radiojepa_rt` (Sionna) | 2 | The ray-tracing backend cannot find `libLLVM.dylib` |
| `test_research_restore` | 1 | Uses `renameat2`, a Linux-only system call |
| `test_representation_repair` | 1 | An exact numerical recomputation differs on this platform |
| `test_selected_jepa_rollout` | 1 | "Frozen source/protocol/software changed": the software versions differ from the machine the evidence was built on |

None of these was fixed; they are Codex's tests and this machine's
environment.

## What this run found and fixed
The first run had 29 failures and 12 errors. Thirteen failures and all
twelve errors were caused by this branch, for one reason: **earlier
evidence archives checksum the simulator source.**

- Several archives pin `src/wirelessjev/native/sim.py` by SHA-256, and three tests compare its result dictionary exactly.
- Two profiles checksum **every Python file in** `src/wirelessjev/native/` (also `twin/`, `contracted/`, `fleetwm/`, `frontier/`, `gate_repair/gate.py`, `pyproject.toml` and `uv.lock`). Adding a file to `native/` breaks them.

This branch had edited `sim.py` and added three files to `native/`.

**Fix:**
- `native/sim.py` is restored byte-for-byte to Codex's version, and `native/` as a whole is identical to Codex's tip.
- The extended simulator and the fleet work moved to a new package, `src/wirelessjev/native_ext/`: `sim_fleet.py` (the extended simulator), `blockers.py`, `run_fleet.py`, `coverage.py`.
- A test checks that `sim_fleet` with no feature switched on gives the same results as the pinned `sim.py` for seven policies, and that `native/` is unchanged from 2a043e54.

The campaign results committed today were produced by the code that is now
`sim_fleet.py`; only its location and imports changed.

## Rule for future work
Do not edit or add files in `src/wirelessjev/native/`, `twin/`,
`contracted/`, `fleetwm/` or `frontier/`, and do not change
`pyproject.toml` or `uv.lock`, unless the intent is to invalidate the
earlier archives. Put new simulator work in `native_ext/`.
