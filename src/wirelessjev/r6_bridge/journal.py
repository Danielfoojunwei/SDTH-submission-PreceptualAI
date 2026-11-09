"""Bounded journal inspection with no recovery or execution authority.

File input is deliberately narrower than a live SQLite reader: the owner must
export a quiescent, checkpointed, rollback-mode database, with no sidecars.
SQLite opens only a private byte copy, using ``mode=ro`` and ``query_only``.
Live WAL/hot-journal databases are refused before SQLite is called. A supplied
R6 snapshot dictionary is an owner assertion, not authenticated state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
import time


class SnapshotError(ValueError):
    """The input cannot establish the declared read-only snapshot boundary."""


@dataclass(frozen=True)
class ReadLimits:
    max_rows: int = 10000
    max_json_bytes: int = 131072
    max_db_bytes: int = 64 * 1024 * 1024
    timeout_seconds: float = 2.0

    def __post_init__(self):
        for key, maximum in (("max_rows", 100000), ("max_json_bytes", 1048576),
                             ("max_db_bytes", 256 * 1024 * 1024)):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise SnapshotError("Invalid read limit")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds)
                or not .01 <= self.timeout_seconds <= 10):
            raise SnapshotError("Invalid read deadline")


@dataclass(frozen=True)
class Allocation:
    resource_id: str
    unit: str
    amount: float
    version: int


@dataclass(frozen=True)
class ResourceRecord:
    resource_id: str
    unit: str
    capacity: float
    external_used: float
    version: int
    observed_ms: int
    used: float


@dataclass(frozen=True)
class RobotRecord:
    robot: str
    epoch: str
    generation: int
    current: tuple[Allocation, ...] = ()
    sequence: int | None = None
    disabled: bool | None = None


@dataclass(frozen=True)
class OperationRecord:
    operation_id: str
    robot: str
    original_state: str
    normalized_state: str
    unresolved: bool
    plan_hash: str | None = None
    source: tuple[Allocation, ...] = ()
    target: tuple[Allocation, ...] = ()
    generic_domain: str | None = None
    generic_airtime: float | None = None
    issued_ms: int | None = None
    retry_allowed: bool = field(default=False, init=False)
    generic_unit: None = field(default=None, init=False)

    @property
    def holds_target(self):
        return self.original_state in {"PREPARED", "ISSUED", "SUBMITTED", "UNKNOWN"}


@dataclass(frozen=True)
class JournalSnapshot:
    backend: str
    operations: tuple[OperationRecord, ...]
    resources: tuple[ResourceRecord, ...] = ()
    robots: tuple[RobotRecord, ...] = ()
    generic_budgets: tuple[tuple[str, float], ...] = ()
    site: str | None = None
    revision: int | None = None
    site_clock_ms: int | None = None
    source_sha256: str | None = None
    provenance: str = ""
    allocation_mapping_complete: bool = False
    usage_verified: bool = False
    declared_execution_source_sha256: str | None = None
    generic_budget_unit: None = field(default=None, init=False)
    clock_mapping_available: bool = field(default=False, init=False)
    execution_authorized: bool = field(default=False, init=False)


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_UNITS = {"airtime_fraction", "mbps", "cpu_fraction", "prb_slots", "slots"}
_ACTIVE = {"PREPARED", "ISSUED", "UNKNOWN", "SUBMITTED"}
_CURRENT_OUTCOMES = {"PREPARED": None, "ISSUED": None, "UNKNOWN": "UNKNOWN",
                     "COMMITTED": "APPLIED", "ABORTED": "NOT_APPLIED", "CANCELLED": "NOT_APPLIED"}
_R6_STATES = {"PREPARED", "SUBMITTED", "UNKNOWN", "APPLIED", "NOT_APPLIED"}
_RF_STATES = set(_CURRENT_OUTCOMES)
_SCHEMAS = {
    "metadata": (("singleton", "INTEGER"), ("config", "TEXT")),
    "transactions": (("seq", "INTEGER"), ("txn_id", "TEXT"), ("payload", "TEXT"),
                     ("sha256", "TEXT"), ("state", "TEXT"), ("outcome", "TEXT")),
    "sessions": (("robot", "TEXT"), ("epoch", "TEXT"), ("generation", "INTEGER"),
                 ("seq", "INTEGER"), ("disabled", "INTEGER")),
    "operations": (("op_id", "TEXT"), ("robot", "TEXT"), ("plan_hash", "TEXT"),
                   ("body", "TEXT"), ("state", "TEXT"), ("result", "TEXT")),
    "rf_meta": (("k", "TEXT"), ("v", "TEXT")),
    "rf_resources": (("id", "TEXT"), ("body", "TEXT")),
    "rf_robots": (("robot", "TEXT"), ("epoch", "TEXT"), ("generation", "INTEGER"), ("current", "TEXT")),
    "rf_ops": (("id", "TEXT"), ("robot", "TEXT"), ("state", "TEXT"), ("body", "TEXT"),
               ("issued_ms", "INTEGER"), ("evidence", "TEXT")),
}
_PRIMARY = {"metadata": "singleton", "transactions": "seq", "sessions": "robot",
            "operations": "op_id", "rf_meta": "k", "rf_resources": "id", "rf_robots": "robot", "rf_ops": "id"}


def _ident(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise SnapshotError("Invalid journal identifier")
    return value


def _number(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise SnapshotError("Invalid journal amount")
    return float(value)


def _integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise SnapshotError("Invalid journal integer")
    return value


def _hash(value):
    if type(value) is not str or not _HASH.fullmatch(value):
        raise SnapshotError("Invalid declared SHA256")
    return value


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise SnapshotError("Unsupported journal document schema")
    return value


def _list(value, limits):
    if type(value) not in (list, tuple) or len(value) > limits.max_rows:
        raise SnapshotError("Journal row budget exceeded")
    return value


def _json(text, limits, *, maximum=None):
    if type(text) is not str or len(text.encode("utf-8")) > min(maximum or limits.max_json_bytes, limits.max_json_bytes):
        raise SnapshotError("Journal JSON budget exceeded")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SnapshotError("Duplicate journal JSON key")
            result[key] = value
        return result
    try:
        return json.loads(text, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(SnapshotError("Nonfinite journal JSON")))
    except (ValueError, RecursionError, UnicodeError) as error:
        if isinstance(error, SnapshotError):
            raise
        raise SnapshotError("Invalid journal JSON") from None


def _operation(op_id, robot, state, *, states, **kwargs):
    _ident(op_id); _ident(robot)
    if type(state) is not str or state not in states:
        raise SnapshotError("Unknown journal operation state")
    normalized = ("prepared" if state == "PREPARED" else "unresolved" if state in _ACTIVE
                  else "applied" if state in {"APPLIED", "COMMITTED"} else "not_applied")
    return OperationRecord(op_id, robot, state, normalized, state in {"ISSUED", "SUBMITTED", "UNKNOWN"}, **kwargs)


def _unique(records, key):
    if len({getattr(r, key) for r in records}) != len(records):
        raise SnapshotError("Duplicate journal identity")


def _open_unique(operations):
    robots = [r.robot for r in operations if r.holds_target]
    if len(set(robots)) != len(robots):
        raise SnapshotError("Multiple unresolved operations for a robot")


def _allocation(value, resources):
    _shape(value, {"resource_id", "unit", "amount", "version"})
    key = _ident(value["resource_id"])
    resource = resources.get(key)
    version = _integer(value["version"], 1)
    if resource is None or value["unit"] != resource.unit or version > resource.version:
        raise SnapshotError("Allocation lacks authoritative resource unit/version")
    return Allocation(key, resource.unit, _number(value["amount"]), version)


def _allocations(values, resources, limits):
    if type(values) not in (tuple, list) or len(values) > min(64, limits.max_rows):
        raise SnapshotError("Invalid allocation vector")
    result = tuple(_allocation(v, resources) for v in values)
    _unique(result, "resource_id")
    return result


def _resources(values, limits):
    if type(values) is not dict or len(values) > limits.max_rows:
        raise SnapshotError("Invalid resource snapshot")
    result = {}
    for key, value in values.items():
        _shape(value, {"resource_id", "unit", "capacity", "external_used", "version", "observed_ms"})
        if (_ident(key) != _ident(value["resource_id"]) or type(value["unit"]) is not str
                or value["unit"] not in _UNITS):
            raise SnapshotError("Invalid authoritative resource identity/unit")
        capacity, external = _number(value["capacity"]), _number(value["external_used"])
        if external > capacity:
            raise SnapshotError("Invalid external resource occupancy")
        result[key] = ResourceRecord(key, value["unit"], capacity, external,
                                     _integer(value["version"], 1), _integer(value["observed_ms"]), external)
    return result


def _robots(values, resources, limits):
    robots = []
    for value in _list(values, limits):
        _shape(value, {"robot", "epoch", "generation", "current"})
        current = _json(value["current"], limits) if type(value["current"]) is str else value["current"]
        robots.append(RobotRecord(_ident(value["robot"]), _ident(value["epoch"]),
                                  _integer(value["generation"], 1), _allocations(current, resources, limits)))
    _unique(robots, "robot")
    return tuple(robots)


def _usage(resources, robots, operations=()):
    usage = {key: item.external_used for key, item in resources.items()}
    for robot in robots:
        for demand in robot.current:
            usage[demand.resource_id] += demand.amount
    for operation in operations:
        if operation.holds_target:
            for demand in operation.target:
                usage[demand.resource_id] += demand.amount
    if not all(math.isfinite(v) for v in usage.values()):
        raise SnapshotError("Nonfinite aggregate resource occupancy")
    return usage


def _with_usage(resources, usage):
    return tuple(ResourceRecord(r.resource_id, r.unit, r.capacity, r.external_used,
                                r.version, r.observed_ms, usage[r.resource_id])
                 for _, r in sorted(resources.items()))


def parse_snapshot(kind, document, *, provenance, limits=ReadLimits()):
    """Normalize an owner-supplied R6 snapshot; no DB or upstream API calls.

    R6's safe reservation snapshot omits operation target vectors. Its supplied
    usage is retained, but cannot be independently reconstructed by this API.
    Neither this dictionary nor a provenance label grants actuation authority.
    """
    if not isinstance(limits, ReadLimits):
        raise SnapshotError("Typed read limits required")
    if type(provenance) is not str or not 1 <= len(provenance.encode()) <= 256:
        raise SnapshotError("Explicit bounded owner provenance required")
    if kind == "r6_journal":
        _shape(document, {"robots", "observations", "operations"})
        _list(document["observations"], limits)  # Intentionally discard observation payloads.
        robots = []
        for value in _list(document["robots"], limits):
            _shape(value, {"robot", "epoch", "generation", "seq", "disabled"})
            if type(value["disabled"]) is not int or value["disabled"] not in (0, 1):
                raise SnapshotError("Invalid session disabled flag")
            robots.append(RobotRecord(_ident(value["robot"]), _ident(value["epoch"]),
                                      _integer(value["generation"], 1), sequence=_integer(value["seq"], -1),
                                      disabled=bool(value["disabled"])))
        operations = []
        for value in _list(document["operations"], limits):
            _shape(value, {"op_id", "robot", "plan_hash", "state"})
            operations.append(_operation(value["op_id"], value["robot"], value["state"],
                                         states=_R6_STATES, plan_hash=_hash(value["plan_hash"])))
        _unique(robots, "robot"); _unique(operations, "operation_id"); _open_unique(operations)
        if any(o.robot not in {r.robot for r in robots} for o in operations):
            raise SnapshotError("Operation has no journal robot")
        return JournalSnapshot(kind, tuple(operations), robots=tuple(robots), provenance=provenance)
    if kind != "r6_reservations":
        raise SnapshotError("Unsupported owner snapshot backend")
    _shape(document, {"site", "revision", "usage", "overloaded", "resources", "robots", "operations",
                      "radio_actuation_performed", "capacity_source"})
    if document["radio_actuation_performed"] is not False or document["capacity_source"] != "OWNER_DECLARED_NOT_INFERRED_FROM_RSSI":
        raise SnapshotError("Unsupported reservation snapshot scope")
    resources = _resources(document["resources"], limits)
    robots = _robots(document["robots"], resources, limits)
    operations = []
    for value in _list(document["operations"], limits):
        _shape(value, {"id", "robot", "state", "issued_ms"})
        issued = None if value["issued_ms"] is None else _integer(value["issued_ms"])
        operation = _operation(value["id"], value["robot"], value["state"], states=_RF_STATES, issued_ms=issued)
        if ((operation.original_state in {"PREPARED", "CANCELLED"}) != (issued is None)):
            raise SnapshotError("Reservation issue-state mismatch")
        operations.append(operation)
    _unique(operations, "operation_id"); _open_unique(operations)
    robot_ids = {r.robot for r in robots}
    if any(o.robot not in robot_ids for o in operations):
        raise SnapshotError("Operation has no reservation robot")
    usage = document["usage"]
    if type(usage) is not dict or set(usage) != set(resources):
        raise SnapshotError("Resource usage identity mismatch")
    usage = {k: _number(v) for k, v in usage.items()}
    floor = _usage(resources, robots)
    if any(usage[k] + 1e-10 < floor[k] for k in usage):
        raise SnapshotError("Snapshot erases confirmed current occupancy")
    overloaded = sorted(k for k in usage if usage[k] > resources[k].capacity + 1e-10)
    if type(document["overloaded"]) is not list or document["overloaded"] != overloaded:
        raise SnapshotError("Overload declaration disagrees with resource usage")
    return JournalSnapshot(kind, tuple(operations), _with_usage(resources, usage), robots,
                           site=_ident(document["site"]), revision=_integer(document["revision"]), provenance=provenance)


def _schema(conn, table):
    row = conn.execute("SELECT type,sql FROM sqlite_master WHERE name=?", (table,)).fetchone()
    if row is None or row[0] != "table" or not row[1] or "VIRTUAL" in row[1].upper():
        raise SnapshotError("Unsupported journal table")
    info = conn.execute("PRAGMA table_xinfo(" + table + ")").fetchall()
    columns = tuple((r[1], r[2].upper()) for r in info)
    if (columns != _SCHEMAS[table] or any(r[6] != 0 for r in info)
            or [(r[1], r[5]) for r in info if r[5]] != [(_PRIMARY[table], 1)]):
        raise SnapshotError("Unsupported journal SQLite schema")


def _rows(conn, table, columns, limits, *, order="rowid"):
    rows = conn.execute("SELECT " + columns + " FROM " + table + " ORDER BY " + order + " LIMIT ?",
                        (limits.max_rows + 1,)).fetchall()
    if len(rows) > limits.max_rows:
        raise SnapshotError("Journal row budget exceeded")
    return rows


def _current(conn, limits):
    for table in ("metadata", "transactions"):
        _schema(conn, table)
    rows = _rows(conn, "metadata", "singleton,config", limits)
    if len(rows) != 1 or rows[0]["singleton"] != 1:
        raise SnapshotError("Invalid current journal metadata")
    config = _json(rows[0]["config"], limits)
    _shape(config, {"schema", "execution_semantics", "execution_source_sha256", "budgets", "max_in_flight", "max_records"})
    if type(config["schema"]) is not int or config["schema"] != 1 or type(config["execution_semantics"]) is not int or config["execution_semantics"] != 2:
        raise SnapshotError("Unsupported current execution semantics")
    source = _hash(config["execution_source_sha256"])
    _integer(config["max_in_flight"], 1); maximum = _integer(config["max_records"], 1)
    budgets = config["budgets"]
    if type(budgets) is not dict or len(budgets) > min(128, limits.max_rows):
        raise SnapshotError("Invalid generic budget vector")
    budgets = {_ident(k): _number(v) for k, v in budgets.items()}
    operations = []
    for row in _rows(conn, "transactions", "seq,txn_id,payload,sha256,state,outcome", limits, order="seq"):
        _integer(row["seq"], 1)
        text = row["payload"]
        if type(text) is not str or hashlib.sha256(text.encode()).hexdigest() != _hash(row["sha256"]):
            raise SnapshotError("Current payload checksum mismatch")
        payload = _json(text, limits, maximum=4096)
        _shape(payload, {"txn_id", "car_id", "from_bssid", "to_bssid", "request", "expected_interface", "expected_role", "domain", "airtime"})
        _shape(payload["request"], {"action", "ssid", "bssid", "security"})
        if (payload["txn_id"] != row["txn_id"] or type(row["state"]) is not str
                or row["state"] not in _CURRENT_OUTCOMES or row["outcome"] != _CURRENT_OUTCOMES[row["state"]]):
            raise SnapshotError("Current journal identity/state/outcome mismatch")
        domain = _ident(payload["domain"])
        if domain not in budgets:
            raise SnapshotError("Current journal domain has no generic budget")
        operations.append(_operation(row["txn_id"], payload["car_id"], row["state"], states=_RF_STATES,
                                     generic_domain=domain, generic_airtime=_number(payload["airtime"])))
    if len(operations) > maximum:
        raise SnapshotError("Current journal exceeds persisted record capacity")
    _unique(operations, "operation_id"); _open_unique(operations)
    return JournalSnapshot("current", tuple(operations), generic_budgets=tuple(sorted(budgets.items())),
                           declared_execution_source_sha256=source)


def _journal(conn, limits):
    for table in ("sessions", "operations"):
        _schema(conn, table)
    document = {"robots": [dict(r) for r in _rows(conn, "sessions", "robot,epoch,generation,seq,disabled", limits)],
                "observations": [],
                "operations": [dict(r) for r in _rows(conn, "operations", "op_id,robot,plan_hash,state", limits)]}
    return parse_snapshot("r6_journal", document, provenance="quiescent_sqlite_copy", limits=limits)


def _reservations(conn, limits):
    for table in ("rf_meta", "rf_resources", "rf_robots", "rf_ops"):
        _schema(conn, table)
    meta = dict((r["k"], r["v"]) for r in _rows(conn, "rf_meta", "k,v", limits))
    if set(meta) != {"site", "revision", "clock"}:
        raise SnapshotError("Unsupported reservation metadata")
    def counter(text):
        if type(text) is not str or not re.fullmatch(r"0|[1-9][0-9]{0,18}", text):
            raise SnapshotError("Invalid reservation metadata integer")
        return _integer(int(text))
    site, revision, clock = _ident(meta["site"]), counter(meta["revision"]), counter(meta["clock"])
    resources = _resources({r["id"]: _json(r["body"], limits)
                            for r in _rows(conn, "rf_resources", "id,body", limits)}, limits)
    robots = _robots([dict(r) for r in _rows(conn, "rf_robots", "robot,epoch,generation,current", limits)], resources, limits)
    robot_map = {r.robot: r for r in robots}
    operations = []
    for row in _rows(conn, "rf_ops", "id,robot,state,body,issued_ms", limits):
        body = _json(row["body"], limits)
        _shape(body, {"site", "op_id", "robot", "epoch", "generation", "source", "target", "action", "plan_hash", "revision", "expires_ms"})
        if (body["op_id"] != row["id"] or body["robot"] != row["robot"] or body["site"] != site
                or row["robot"] not in robot_map or type(body["action"]) is not dict):
            raise SnapshotError("Reservation operation binding mismatch")
        _ident(body["epoch"]); _integer(body["generation"], 1); _integer(body["expires_ms"])
        if _integer(body["revision"]) > revision:
            raise SnapshotError("Reservation operation from future revision")
        source = _allocations(body["source"], resources, limits)
        target = _allocations(body["target"], resources, limits)
        issued = None if row["issued_ms"] is None else _integer(row["issued_ms"])
        operation = _operation(row["id"], row["robot"], row["state"], states=_RF_STATES,
                               plan_hash=_hash(body["plan_hash"]), source=source, target=target, issued_ms=issued)
        if ((operation.original_state in {"PREPARED", "CANCELLED"}) != (issued is None)):
            raise SnapshotError("Reservation issue-state mismatch")
        robot = robot_map[operation.robot]
        if operation.holds_target and (source != robot.current or body["epoch"] != robot.epoch
                                       or body["generation"] != robot.generation):
            raise SnapshotError("Open operation differs from current allocation/session")
        operations.append(operation)
    _unique(operations, "operation_id"); _open_unique(operations)
    return JournalSnapshot("r6_reservations", tuple(operations), _with_usage(resources, _usage(resources, robots, operations)), robots,
                           site=site, revision=revision, site_clock_ms=clock,
                           allocation_mapping_complete=True, usage_verified=True)


def _source_bytes(path, limits, deadline):
    if not isinstance(path, (str, os.PathLike)) or len(os.fsencode(path)) > 4096:
        raise SnapshotError("Bounded local snapshot path required")
    path = Path(path).absolute()
    try:
        if path.resolve(strict=True) != path or path.is_symlink():
            raise SnapshotError("Symlinked snapshots are unsupported")
        def sidecars():
            if any(os.path.lexists(str(path) + suffix) for suffix in ("-wal", "-shm", "-journal")):
                raise SnapshotError("Live WAL/SHM or rollback journal refused; export a quiescent snapshot")
        sidecars()
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 100 <= before.st_size <= limits.max_db_bytes:
                raise SnapshotError("Regular bounded unaliased SQLite snapshot required")
            chunks = []
            remaining = before.st_size
            while remaining:
                if time.monotonic() >= deadline:
                    raise SnapshotError("Snapshot read deadline exceeded")
                chunk = os.read(fd, min(remaining, 1048576))
                if not chunk:
                    raise SnapshotError("Snapshot changed while reading")
                chunks.append(chunk); remaining -= len(chunk)
            data = b"".join(chunks)
            if data[:16] != b"SQLite format 3\x00" or data[18:20] != b"\x01\x01":
                raise SnapshotError("Only checkpointed rollback-mode SQLite snapshots are supported")
            identity = lambda s: (s.st_dev, s.st_ino, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(os.fstat(fd)) or identity(before) != identity(path.stat()):
                raise SnapshotError("Snapshot identity changed while reading")
            sidecars()
            os.lseek(fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            for _ in range((before.st_size + 1048575) // 1048576):
                if time.monotonic() >= deadline:
                    raise SnapshotError("Snapshot read deadline exceeded")
                digest.update(os.read(fd, 1048576))
            if (hashlib.sha256(data).hexdigest() != digest.hexdigest()
                    or identity(before) != identity(os.fstat(fd)) or identity(before) != identity(path.stat())):
                raise SnapshotError("Snapshot bytes changed while reading")
            sidecars()
            return data
        finally:
            os.close(fd)
    except (OSError, ValueError) as error:
        if isinstance(error, SnapshotError):
            raise
        raise SnapshotError("Cannot inspect local snapshot") from None


def read_snapshot(path, *, kind="auto", quiescent=False, limits=ReadLimits()):
    """Read an owner-exported offline database; never open a live DB in SQLite.

    ``quiescent=True`` declares an external owner guarantee, not a discovered
    lock or a freshness/clock assertion. Stable bytes are checked twice. This
    function cannot turn an actively written database into a coherent backup.
    """
    if quiescent is not True or not isinstance(limits, ReadLimits):
        raise SnapshotError("Explicit quiescent snapshot declaration and typed limits required")
    if kind not in {"auto", "current", "r6_journal", "r6_reservations"}:
        raise SnapshotError("Unsupported journal backend")
    started = time.monotonic()
    deadline = started + limits.timeout_seconds
    data = _source_bytes(path, limits, deadline)
    if time.monotonic() >= deadline:
        raise SnapshotError("Snapshot read deadline exceeded")
    try:
        with tempfile.TemporaryDirectory(prefix="r6-journal-read-") as scratch:
            copied = Path(scratch) / "snapshot.db"
            copied.write_bytes(data)
            conn = sqlite3.connect(copied.as_uri() + "?mode=ro", uri=True, timeout=.05, isolation_level=None)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only=ON")
                conn.execute("PRAGMA trusted_schema=OFF")
                conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, limits.max_json_bytes)
                conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 4096)
                steps = [0]
                def progress():
                    steps[0] += 1000
                    return int(time.monotonic() >= deadline or steps[0] > 10000000)
                conn.set_progress_handler(progress, 1000)
                # All SQL below is fixed. Deny writes/attachment even in the private copy.
                allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_TRANSACTION}
                def authorize(action, arg1, arg2, db, trigger):
                    if action == sqlite3.SQLITE_PRAGMA and arg1 == "table_xinfo" and arg2 in _SCHEMAS:
                        return sqlite3.SQLITE_OK
                    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
                conn.set_authorizer(authorize)
                conn.execute("BEGIN")
                tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1001")}
                if len(tables) > 1000:
                    raise SnapshotError("Schema table budget exceeded")
                candidates = {name for name, required in (("current", {"metadata", "transactions"}),
                              ("r6_journal", {"sessions", "operations"}),
                              ("r6_reservations", {"rf_meta", "rf_resources", "rf_robots", "rf_ops"})) if required <= tables}
                if kind == "auto":
                    if len(candidates) != 1:
                        raise SnapshotError("Unsupported or ambiguous journal backend")
                    kind = candidates.pop()
                elif kind not in candidates:
                    raise SnapshotError("Declared journal backend differs from SQLite schema")
                snapshot = {"current": _current, "r6_journal": _journal, "r6_reservations": _reservations}[kind](conn, limits)
                conn.execute("ROLLBACK")
                if time.monotonic() >= deadline:
                    raise SnapshotError("Snapshot read deadline exceeded")
            finally:
                conn.close()
    except (sqlite3.Error, OSError, OverflowError, RecursionError) as error:
        raise SnapshotError("Cannot validate read-only SQLite snapshot") from None
    # Build a fresh frozen DTO; no raw SQLite row, payload, request or action escapes.
    fields = {key: getattr(snapshot, key) for key, definition in snapshot.__dataclass_fields__.items() if definition.init}
    fields.update(source_sha256=hashlib.sha256(data).hexdigest(), provenance="quiescent_sqlite_copy")
    return JournalSnapshot(**fields)
