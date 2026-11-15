"""Extensions to the native slot simulator, kept outside `wirelessjev.native`.

Earlier evidence archives checksum every Python file in `wirelessjev/native/`
(and `sim.py` in particular), so nothing may be added or edited there. The
fleet-capability work lives here instead:

- `sim_fleet`: `native.sim` plus optional features (off by default; the
  default run matches `native.sim`, checked in tests/test_native.py).
- `blockers`: moving blockers (forklifts).
- `run_fleet`: the pre-registered campaigns.
- `coverage`: the site coverage report.
"""
