# Portable deployment software verification

The [final software suite](final/software_suite.json) passes **3,903 tests**
across 160 modules. All 591 tracked source/config/test files have identical
before/after hashes. The expected inherited-fork warning remains recorded.
The 270 new deployment fixtures are included in that total, not added again.
They are software tests, not observations of a real SmartCar.

The [final portable check](final/portable_verification.json) extracts the actual
95,786-byte ZIP, verifies every shipped input, and runs validate/fit/predict/
propose in a new environment containing NumPy/SciPy and no Torch, Sionna,
Transformers or Hugging Face. All three real pinned model checkpoints also
execute in the existing Torch environment while research/training dependencies
are blocked. The test inputs are explicitly synthetic API fixtures. No SSH
request, car scan, actuation, training or RT solve is performed. A local
read-only doctor check correctly rejects this cloud executor's missing radios;
it is not a SmartCar hardware test.

The [prior scientific guard](prior_closure_verification.json) preserves all ten
scientific closures and 27,295 named source/evidence files. Historical negative
results, original SERVICE holdouts and deferred observation-frontier work remain
unchanged. Deployment software neither revises those results nor establishes
specialist SOTA.

Independent code review checked the collector, actual ZIP, model provenance,
clock boundaries, mapping support and supervised control. Two findings were
fixed before final publication: the CLI now requires an interactive terminal
before credentials/network access, and the guide distinguishes proposal
thresholds from the separately authorized manual switch. Full tests and the
portable check were repeated after those fixes.

The initial 3,902-test [software receipt](software_suite.json),
[portable receipt](portable_verification.json) and
[first archive](../smartcar_deployment/candidates/initial/smartcar-wirelessJEV.zip)
are preserved as superseded software candidates. Their archived hashes refer
to the original bytes and source commit, not the current final package.
No scientific data were changed to obtain the final software pass.

The new control path also avoids concrete shortcomings of the frozen older
onecar diagnostic: partial-write/interrupt classification and a first-ping
timing summary that can miss a later outage. It durably journals intent before
one connect, handles ambiguous completion as UNKNOWN and permits only status
reconciliation. It makes no new measured-handover claim.

Actual installation, trusted hardware identity/access, scan permission,
position-tagged measurements, independent management routing, a supervised
switch/return and held-out real-site comparisons remain open. The next useful
research checks include a separately verified Torch-free implementation and
an explicitly registered measured-RSSI comparison; neither already has a result.
