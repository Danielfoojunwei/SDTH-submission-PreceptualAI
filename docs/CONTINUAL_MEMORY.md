# Bounded differential-response memory

[`continual_memory.py`](../src/wirelessjev/continual_memory.py) stores small,
per-device/context/radio-epoch linear response statistics. It identifies a
**requested response functional**, rather than claiming every transmitter gain
is individually known. It performs no hardware probe, backbone training,
automatic model promotion or production qualification.

## Inputs and their meaning

For transmitter `j`, a caller can construct known receiver-filter exposure in
linear units as `e_j = P_tx,j_mw × activity_j × overlap_jr`. A `DifferentialPair`
supplies signed `Δexposure_mw` between two conditions and signed
`Δreceived_power_mw` for the matching receiver. Transmit power must be converted
from dBm **before** taking differences. Neither a change in aggregate RSSI nor
a visually inferred interferer state establishes those individual exposures.

The API accepts only the declared differential pair, identity, clocks and
provenance. There is no hidden coupling-matrix `G` input. Transmitter order,
reference planes, channel/filter configuration and units must agree across
observations and queries. A change in schema or radio calibration requires a
new context/epoch key. This caller contract cannot be inferred from numeric
array shape alone.

`controlled=False` is the default. Such fits are explicitly
`evidence_kind='associative_only'`. Declaring `controlled=True` requires an
experiment-provenance string; it records a claim about how the pair was
collected, rather than proving causal control. Querying with
`controlled_only=True` excludes all undeclared observations.

## Query-specific identifiability

Within a fixed linear-response context, write normalized pair rows as
`y = Xβ`. The retained sufficient statistics are

\[
\mathrm{Gram}=X^\top X,\qquad b=X^\top y.
\]

For a query exposure change `q`, its response is determined by the data if
`q` lies in the row space of `X`, equivalently the range of `Gram`. The
pseudoinverse then gives `qᵀ Gram⁺ b`. A pseudoinverse's chosen individual
coefficient vector does not make an unidentified individual query known.

For example, one observation `[1,1]β=5` identifies `[2,2]β=10`; it leaves
`[1,0]β` unknown. Repeating synchronized observations does not supply the
missing excitation. Staggered rows `[1,0]` and `[0,1]` identify both coordinates.
With declared unit transmit power, overlap and slot duration, one joint slot
and two staggered slots both consume two transmitter-airtime units. This is an
analytic design comparison, not a measured RF/throughput benchmark.

The implementation uses a relative Gram eigenspectrum cutoff and checks the
normalized **query direction**, avoiding an absolute residual tolerance that
would falsely accept tiny SI-unit queries. The same declared exposure scale
applies to data and queries. Gram matrices square conditioning, so near-collinear
directions can conservatively remain unidentified. Regularization or a prior
does not create experimental rank. No ridge/prior precision is added here.

An unidentified, missing or expired query has `predicted_delta_mw=None`.
Signed identified responses are retained, including negative changes. With
usable evidence, an exactly zero query has the zero response of the declared
homogeneous linear model. With no usable evidence the API returns unknown,
including for a zero query. It never invents an uncertainty bound; numerical
identifiability is not a calibrated statement about real RF error.

## Time, bounds and retention

Keys are `(device_id, context_id, radio_epoch)`. Each state has its own bounded
pair window, Gram and right-hand side; epochs and contexts do not inherit each
other's evidence. A bounded LRU limits the total number of states. Pair identity
deduplication applies within the retained state/window. It is not an unlimited
exactly-once event log after eviction.

All API calls share a finite nonnegative, nondecreasing ingestion/query clock.
Equal timestamps permit a batch. A pair requires
`0 ≤ measured_at ≤ delivered_at ≤ now`, bounded delivery delay and bounded
measurement age. `measured_at` must denote the **oldest measurement in the
pair**, rather than a later upload or convenient newest capture time. The
caller must establish clock alignment and that both paired measurements belong
to the same declared context. Out-of-order delivery is allowed when these
constraints remain satisfied.

Every query needs a known, causal `query_observed_at` for its exposure inputs;
stale/unknown exposure state is not silently made current. Each retained row
expires by its own measurement time. A new joint observation cannot renew an
old independent row and falsely preserve full rank. Statistics are rebuilt
from the bounded retained window after removal, avoiding subtraction residue
that could appear to be new rank.

Configuration bounds feature dimension to at most 64, state count, pairs per
state and normalized magnitudes. Inputs must be finite signed powers. Missing
values are not converted to zero. Explicit exposure/response scales let the
caller use practical RF power units; overflow, unsafe underflow and inconsistent
coordinates are rejected. If an algebraically identified response exceeds the
finite output range, `identified` remains true but its numerical prediction is
`None` with a range-error reason; it is not a usable numeric estimate. The
memory is process-local, single-threaded and not a durable recovery
or synchronized multi-writer service.

## Evidence boundary

Eight deterministic test groups cover joint/individual identifiability,
staggering, repeated and nearly collinear excitation, controlled provenance,
individual-row expiry and query freshness, duplicate/LRU bounds, SI scaling,
numeric range and context/epoch separation. They are numerical proofs and
software fault checks, rather than fabricated training or hardware results.

A full-rank design still needs stable coupling, correct exposure measurements,
aligned differential outcomes and control of confounding changes before a
causal interpretation is warranted. Uncontrolled association alone cannot
establish the result of an intervention. `production_qualified` remains false
for every result; no memory update changes any live radio policy or pretrained
backbone.
