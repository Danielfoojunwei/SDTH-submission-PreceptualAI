# Minimal read-only R6.1 bridge

`wirelessjev.r6_bridge` connects explicit planning contracts and journal snapshots to the actual restored R6.1 reference. It is an opt-in software interoperability layer. It neither replaces the warehouse controller nor sends an action, grants a release, migrates a live journal or establishes physical service reliability.

## Actual upstream planning code

The bridge pins didjital commit `a458a90d34fe559addb876a2a1c69213c6ecc968` and the exact bytes of `preceptual_r6/preceptual/{__init__,contracts,planner,mission_plan}.py`. It invokes the actual `preceptual.mission_plan.plan_missions` function in a separate `python -I -S` interpreter, loading the verified upstream source bytes. Parent import paths and installed packages are not used to select the upstream implementation; neither `wirelessjev` package is installed over the other.

The parent and worker enforce bounded typed input, output size and subprocess time. The upstream commit/files are checked before loading and again after planning. The response carries the pinned source identity, the owner policy ID and the original upstream plan/hash. It retains `actuation_performed: false`, `transition_qualified: false`, `physical_service_guaranteed: false` and the upstream declared-prediction evidence boundary. The bridge does not call `DidjitalRuntime.plan`, whose rule-adjusted ranking outcomes are not calibrated warehouse service forecasts.

Caller inputs are explicit, immutable interchange contracts:

| Contract | Required information |
|---|---|
| `Mission` | Robot, priority, required Mbps, deadline milliseconds, interruption budget milliseconds and maximum predicted failure |
| `Resource` / `Demand` | Named resource, explicit supported unit, capacity/use or demand amount, resource version and observation clock where defined |
| `Choice` | Robot/action/mode, typed demands, utility, optional declared failure estimate, interruption, validity window, capabilities and switching flag |
| `ServiceForecast` | Mission/choice hashes; goodput floor, delay ceiling and event-specific failure upper; horizon; estimator/evidence identity and validity window |
| `OwnerPolicy` | Independently supplied estimator scope allowlist, supported capabilities, policy identity, current clock and bounded planner limits |

Supported resource units are the upstream units: `airtime_fraction`, `mbps`, `cpu_fraction`, `prb_slots` and `slots`. The adapter never derives capacity or airtime from RSSI. The policy object must be controlled independently by the application owner; its provenance ID is not proof of independent approval or field calibration. Forecast labels and owner declarations are not authenticated by this library.

The existing RF `OutcomeDistribution` does not supply the qualified service delay and event-failure heads required by this contract. `defer_unqualified_outcomes(outcomes)` therefore returns **DEFER / QUALIFIED_SERVICE_FORECAST_REQUIRED**, without inventing a `ServiceForecast`, estimator registration or backend evidence. With explicit forecasts, the actual upstream API still rejects wrong context hashes, unknown estimators/events/horizons, stale reports, unsupported capabilities, resource-unit/version mismatches and unmet mission constraints.

Usage with already supplied explicit contracts:

```python
from wirelessjev.r6_bridge.preview import preview

result = preview(
    missions, resources, choices, service_forecasts,  # bounded typed tuples
    owner=operator_policy,
    checkout="/workspace/didjital-r6.1",
)
document = result.document()  # a proposal/deferral record, never an execution call
```

## Journal view without constructor or recovery writes

The existing `DurableExecution` constructor performs recovery writes. The upstream R6 `Journal` and `ReservationLedger` constructors also initialize SQLite/schema state. None is used to open an existing journal through this adapter. No bridge API calls enrollment, preparation, issue, cancellation, recovery, transport, reconciliation or allocation release.

The journal module supports a typed view of current execution transactions, upstream R6 session journals and upstream R6 resource-reservation journals. It preserves original operation states and marks `ISSUED`, `SUBMITTED` and `UNKNOWN` as unresolved. The view never grants retry permission, infers an outcome from expiry or frees an unresolved allocation. Request/action payloads and observations are not exported as execution authority or credentials.

Current execution's generic `airtime` and budget numbers have no source-declared unit, resource version or trustworthy recovered event clock. The view leaves their unit unknown; it cannot silently reinterpret them as an R6 airtime fraction. R6 reservation resources carry explicit units, but the safe public `snapshot()` operation rows omit target demands. Parsing that public snapshot therefore preserves its reported usage while marking allocation reconstruction incomplete; it cannot prove a reservation is safe to release.

## Conservative file boundary

SQLite `mode=ro` alone is not a guarantee of a nonmutating live-file view: WAL/SHM coordination and hot-journal handling can affect sidecars. The file API consequently supports only **caller-declared quiescent, checkpointed, rollback-mode exported snapshots**, and refuses WAL-header databases or any existing `-wal`, `-shm` or `-journal` sidecar before opening SQLite. It never uses `immutable=1` to ignore potentially uncheckpointed records.

The reader copies bounded verified source bytes into its own private scratch space and opens only that copy with URI `mode=ro` and `query_only`. Source identity, bytes and sidecar membership are checked around reading. The source is never opened by SQLite. This does not create a general live-journal bridge or certify that an external caller's quiescence declaration is truthful. An active owner must supply a consistent exported snapshot or its own snapshot dictionary; this adapter does not checkpoint or change the journal mode of a live store.

Caller-owned upstream `snapshot()` dictionaries can also be normalized with explicit provenance. Such provenance is an attribution record, not authentication of hardware origin. No radio-unit conversion, missing clock, resource ownership or target allocation is invented to fill unavailable fields.

## Software evidence and remaining integration

Targeted tests use explicitly labelled temporary **software fixtures** through the actual pinned upstream API. These fixture service values are not measured predictions or scientific trials. Tests compare the bridge's full plan/schema/hash with an independent direct import of the upstream planner; exercise unit, clock, capability, estimator and source rejection; and require current unqualified RF outcomes to defer.

Journal tests cover the actual current/R6 APIs, unresolved-state retention, malformed input and strict file boundaries. Main database, WAL/SHM/hot-journal bytes and directory membership are checked around refused or accepted reads. No production journal is opened and no physical endpoint is contacted.

**All 77 targeted tests pass**: 25 planning/contract checks and 52 journal checks. The [source-bound receipt](../../artifacts/r6_bridge/software_tests.json) and [test log](../../artifacts/r6_bridge/software_tests.log) record identical before/after hashes for all seven new source/test files, the used upstream files and all seven preserved scientific source maps (452/475/487/491/504/512/5 members). The upstream checkout and frozen source bytes are unchanged. This is additional software qualification, not another RF experiment or hardware result; it does not replace the earlier full tracked-suite proof.

Remaining work is deliberate: qualified real service-forecast producers, independently operated capability/resource registries, authenticated site-clock observations, a separately reviewed transition/ownership bridge and actual hardware qualification. This read-only slice does not close those gaps, provide shared control authority or promote either model into production.
