# SmartCar deployment bundle

[Download smartcar-wirelessJEV.zip](smartcar-wirelessJEV.zip) and follow the
[deployment guide](../../docs/SMARTCAR_DEPLOYMENT.md). Extract the same archive
on the car and computer; install desktop requirements before joining the car's
hotspot. The car collector uses Python standard-library code.

Archive SHA-256:
`436bb7de7676a18ce1a60821aa3e4c1129d37facf557205f5a6dab6333c3f8f3`

The 95,786-byte archive has 28 members, including all three exact released
nested-joint checkpoints, source hashes and a stdlib integrity verifier.
It contains no passwords, trusted host-key files or private vendor archive.

The [whole software suite](../smartcar_deployment_review/final/software_suite.json)
passes 3,903 tests across 160 modules, with 591 source/config/test files
unchanged across execution. The [portable verification](../smartcar_deployment_review/final/portable_verification.json)
passes in a clean NumPy/SciPy-only map environment; all three neural members
also execute while research/training dependencies are blocked. These are
software fixtures and archived-model checks, with zero SSH requests, real-car
captures or actuation. [All ten existing scientific closures](../smartcar_deployment_review/prior_closure_verification.json)
remain unchanged.

The portable default map is a heard-only ordinary-kriging/RBF model around an
FSPL reference, with declared assumptions and bounded support. It is a separate
implementation from the frozen simulator RadioMap and the RF study's numerical
controls. The new workflow supplies no comparative hardware accuracy result.
RadioJEPA runs only as an experimental shadow; proposals return no fabricated
success probability and transmit no command. Switching is a separate typed,
journaled, operator-confirmed action using the existing trusted SmartCarCmd.

Actual management access, radio roles, scan privileges, operator-positioned
measurements, held-out site evaluation and a supervised switch/return still
need to be checked on the car. SOTA, autonomous control and physical-transfer
superiority are not established.
