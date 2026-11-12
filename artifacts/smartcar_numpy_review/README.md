# Pinned RadioJEPA numerical backend verification

The [4,138-test suite](software_suite.json) passes across 165 modules, with
606 tracked source/config/test hashes unchanged. The 235 new tests are included
in this total. They cover likelihood tails and quadrature, public geometry,
all three exact weight exports, dense posterior/covariance references, unknowns,
AP permutations, large marginal queries, real-survey admission parity and a
complete shadow command with Torch imports forbidden. These are software tests,
without new training, ray tracing, RF quality evaluation or hardware action.

The [portable proof](portable_verification.json) extracts the actual 35-member
ZIP and checks its 34 inputs. A clean NumPy/SciPy-only environment executes
validate/fit/predict/propose and all three genuine exported models; Torch is
absent and forbidden. Inputs are explicit synthetic API fixtures. The local
cloud doctor's failure to find the car radios is a software guard check,
not a physical SmartCar observation.

Two performance protocols were published before their corresponding timing
runs. The [original](../smartcar_numpy/protocol.json) uses the installed
CUDA-enabled Torch wheel on CPU; [all 18 raw rows](benchmark.json) are retained.
The [CPU-only follow-up](../smartcar_numpy/cpu_only_protocol.json) repeats the
same fixed cases against an actually installed `torch 2.14.1+cpu`, with
[all 18 raw rows](benchmark_cpu_only.json). Each worker uses one thread, five
warmups and 25 full conditioning/render calls. Same-weight numerical parity
passes in every case. There are no timing-driven implementation or tolerance
changes. Report CPU-only comparisons as the primary result.

Warm timings exclude loading and geometry/input assembly. The startup value
starts at worker entry, includes heavy imports/model setup, and excludes
interpreter launch and top-level standard-library imports. Peak process RSS
includes framework/import overhead; it is not just model-tensor size or shared
fleet memory. These are descriptive cloud CPU measurements for fixed synthetic
inputs, not confidence intervals, actual Pi timing, real-time service, physical
RF accuracy, a RadioMap compute comparison or SOTA qualification.

Independent math review found no material point/interval, mask, precision,
jitter, Schur or covariance discrepancy. Integration review checked the
admission adapter, three-member wiring, export fingerprints and bundle
allowlist. The learned discrepancy MLP remains; only evaluation of the exactly
zero-gated learned local features is omitted. Full local state dimensions are
retained. The loader rejects modified state even with rewritten manifests.

Earlier scientific sources, results and packages remain unchanged. The current
package still requires actual hardware access, trusted radio identity, scan
permission, measured positions and separately supervised switch/return before
real deployment validation. It supplies no proof of SOTA or improved real-car
predictions.
