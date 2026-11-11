"""Software-fixture SQLite tests: no radio, world, transport or live DB calls."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

from wirelessjev.execution import Transaction, build_connect
from wirelessjev.execution_journal import DurableExecution
from wirelessjev.r6_bridge import journal as reader
from wirelessjev.r6_bridge.journal import ReadLimits, SnapshotError, parse_snapshot, read_snapshot


# These are genuine pinned upstream implementations, used only to CREATE new
# temporary SOFTWARE_FIXTURE databases. No existing user DB is opened by them.
UPSTREAM_COMMIT = "a458a90d34fe559addb876a2a1c69213c6ecc968"
FIXTURE_SOURCES = {
    "journal.py": "0b7fb6d18e47f590d0d714c3d485e627a3b3d0824d8ed947fd4daa56bfd691be",
    "reservations.py": "0596050c5dfca64ab5f0a8443a0b33af51e06f4bf691453cc58ebb595314cfff",
    "contracts.py": "ac9111e4408960ae268a412ad3531511cb224d4ce9caeb9e9f6c78a7e2e28514",
}
CURRENT_FIXTURE_SOURCES = {
    "execution.py": "2d05582bb540b11d1a03be505f8a361196e6dde4bfb20c55f7bab6589f38b0cc",
    "execution_journal.py": "14c62f75ff32adba98c819d515275e2bd57e2e3f2ed272633e33751aa5ef73ea",
}


@pytest.fixture(scope="module")
def upstream():
    root = Path(os.environ.get("DIDJITAL_R6_ROOT", "/workspace/didjital-r6.1"))
    package = root / "preceptual_r6" / "preceptual"
    if not package.is_dir():
        pytest.skip("Pinned real didjital R6.1 checkout is unavailable; no substitute fixture implementation")
    for name, expected in FIXTURE_SOURCES.items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected
    name = "_r6_bridge_actual_fixture"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return tuple(importlib.import_module(name + "." + part) for part in ("journal", "reservations", "contracts"))


def _files(directory):
    return {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in directory.iterdir() if p.is_file()}


def _export(db, path):
    """The FIXTURE OWNER exports a consistent, checkpointed DELETE-mode DB.

    The bridge never performs this write/backup or upgrades a live DB itself.
    """
    with sqlite3.connect(path) as copy:
        db.backup(copy)
        assert copy.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    assert path.read_bytes()[18:20] == b"\x01\x01"
    return path


def _current(tmp_path, state="ISSUED"):
    live = tmp_path / "SOFTWARE_FIXTURE-current-owner.db"
    snapshot = tmp_path / "SOFTWARE_FIXTURE-current-offline.db"
    with DurableExecution(live, {"warehouse": 1.}) as owner:
        txn = Transaction("operation-1", "car-001", "02:00:00:00:00:01", "02:00:00:00:00:02",
                          build_connect("PRIVATE_SSID_NOT_FOR_DTO", "02:00:00:00:00:02"))
        assert owner.prepare(txn, "warehouse", .3)
        # Actual persisted transition implementation, on a new software fixture.
        # No transport is invoked; ISSUED represents the durable crash boundary.
        if state not in {"PREPARED", "CANCELLED"}:
            owner._persist_state(txn.txn_id, "ISSUED", None)
        if state in {"UNKNOWN", "COMMITTED", "ABORTED"}:
            owner._persist_state(txn.txn_id, state, {"UNKNOWN": "UNKNOWN", "COMMITTED": "APPLIED", "ABORTED": "NOT_APPLIED"}[state])
        if state == "CANCELLED":
            assert owner.cancel(txn.txn_id)
        _export(owner._conn, snapshot)
    return snapshot


def _r6_journal(tmp_path, upstream, state="SUBMITTED"):
    module, _, _ = upstream
    live = tmp_path / "SOFTWARE_FIXTURE-r6-journal-owner.db"
    snapshot = tmp_path / "SOFTWARE_FIXTURE-r6-journal-offline.db"
    owner = module.Journal(live)
    try:
        owner.enroll("car-001", "boot-1", expected_generation=0)
        owner.prepare({"op_id": "operation-1", "robot": "car-001", "plan_hash": "a" * 64,
                       "epoch": "boot-1", "generation": 1, "expires_ms": 1000,
                       "action": {"credential": "NEVER_EXPOSE_JOURNAL_ACTION"}})
        if state != "PREPARED":
            owner.claim("operation-1", now_ms=100)
        if state in {"UNKNOWN", "APPLIED", "NOT_APPLIED"}:
            owner.finish("operation-1", state, {"private_result": "NEVER_EXPOSE_JOURNAL_RESULT"})
        document = owner.snapshot()
        _export(owner.db, snapshot)
    finally:
        owner.close()
    return snapshot, document


def _r6_reservations(tmp_path, upstream, state="UNKNOWN", newer_resources=False):
    _, module, contracts = upstream
    live = tmp_path / "SOFTWARE_FIXTURE-r6-ledger-owner.db"
    snapshot = tmp_path / "SOFTWARE_FIXTURE-r6-ledger-offline.db"
    owner = module.ReservationLedger(live, site="site-1")
    try:
        owner.replace_resources([contracts.Resource(key, "airtime_fraction", 1., .05, 1, 100)
                                 for key in ("radio:a", "radio:b")], expected_revision=0, now_ms=100)
        owner.enroll("car-001", "boot-1", 1, [contracts.Demand("radio:a", "airtime_fraction", .6, 1)], now_ms=100)
        owner.prepare("operation-1", "car-001", epoch="boot-1", generation=1,
                      target=[contracts.Demand("radio:b", "airtime_fraction", .6, 1)],
                      action={"credential": "NEVER_EXPOSE_RESERVATION_ACTION"}, plan_hash="b" * 64,
                      expected_revision=1, expires_ms=1000, now_ms=110)
        if state != "PREPARED":
            owner.issue("operation-1", now_ms=120)
        if state == "UNKNOWN":
            owner.mark_unknown("operation-1", now_ms=130)
        if newer_resources:
            owner.replace_resources([contracts.Resource(key, "airtime_fraction", .5, .05, 2, 140)
                                     for key in ("radio:a", "radio:b")], expected_revision=1, now_ms=140)
        document = owner.snapshot()
        _export(owner.db, snapshot)
    finally:
        owner.close()
    return snapshot, document


def _edit(path, sql, args=()):
    """Explicit corruption of a caller-owned software fixture before reading."""
    with sqlite3.connect(path) as owner:
        owner.execute(sql, args)


def _edit_body(path, table, change):
    column = "payload" if table == "transactions" else "body"
    with sqlite3.connect(path) as owner:
        original = owner.execute("SELECT " + column + " FROM " + table).fetchone()[0]
        body = json.loads(original)
        change(body)
        body = json.dumps(body, sort_keys=True, separators=(",", ":"))
        if table == "transactions":
            owner.execute("UPDATE transactions SET payload=?,sha256=?", (body, hashlib.sha256(body.encode()).hexdigest()))
        else:
            owner.execute("UPDATE " + table + " SET " + column + "=?", (body,))


def test_current_fixture_source_pins():
    root = Path(reader.__file__).parents[1]
    for name, expected in CURRENT_FIXTURE_SOURCES.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("state,normalized,unresolved", [
    ("PREPARED", "prepared", False), ("ISSUED", "unresolved", True), ("UNKNOWN", "unresolved", True),
    ("COMMITTED", "applied", False), ("ABORTED", "not_applied", False), ("CANCELLED", "not_applied", False)])
def test_current_original_states_units_and_bytes(tmp_path, state, normalized, unresolved):
    path = _current(tmp_path, state)
    before = _files(tmp_path)
    result = read_snapshot(path, quiescent=True)
    assert _files(tmp_path) == before
    operation, = result.operations
    assert (operation.original_state, operation.normalized_state, operation.unresolved) == (state, normalized, unresolved)
    assert operation.holds_target == (state in {"PREPARED", "ISSUED", "UNKNOWN"})
    assert operation.retry_allowed is False
    assert operation.generic_domain == "warehouse" and operation.generic_airtime == .3
    assert operation.generic_unit is None and result.generic_budget_unit is None
    assert result.generic_budgets == (("warehouse", 1.),)
    assert result.resources == () and result.allocation_mapping_complete is False
    assert result.clock_mapping_available is False and result.execution_authorized is False
    assert result.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "PRIVATE_SSID" not in repr(result) and "02:00:" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        operation.original_state = "COMMITTED"


@pytest.mark.parametrize("state", ["PREPARED", "SUBMITTED", "UNKNOWN", "APPLIED", "NOT_APPLIED"])
def test_r6_journal_actual_states_file_and_owner_dict(tmp_path, upstream, state):
    path, document = _r6_journal(tmp_path, upstream, state)
    before = _files(tmp_path)
    file = read_snapshot(path, kind="r6_journal", quiescent=True)
    supplied = parse_snapshot("r6_journal", document, provenance="software-fixture:actual-upstream-api")
    assert file.operations == supplied.operations
    assert file.operations[0].original_state == state
    assert file.operations[0].unresolved == (state in {"SUBMITTED", "UNKNOWN"})
    assert file.operations[0].retry_allowed is False
    assert file.allocation_mapping_complete is False and file.resources == ()
    assert supplied.source_sha256 is None
    assert "NEVER_EXPOSE" not in repr(file) + repr(supplied)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("state", ["PREPARED", "ISSUED", "UNKNOWN"])
def test_reservation_units_keep_both_allocations_and_unknown(tmp_path, upstream, state):
    path, document = _r6_reservations(tmp_path, upstream, state)
    before = _files(tmp_path)
    result = read_snapshot(path, quiescent=True)
    operation, = result.operations
    assert operation.original_state == state and operation.retry_allowed is False
    assert operation.holds_target is True
    assert operation.source[0].resource_id == "radio:a" and operation.target[0].resource_id == "radio:b"
    assert operation.target[0].unit == "airtime_fraction"
    assert {r.resource_id: r.used for r in result.resources} == pytest.approx({"radio:a": .65, "radio:b": .65})
    assert result.usage_verified is True and result.allocation_mapping_complete is True
    assert result.site == "site-1" and result.revision == 1
    assert result.site_clock_ms == {"PREPARED": 110, "ISSUED": 120, "UNKNOWN": 130}[state]
    assert result.clock_mapping_available is False
    supplied = parse_snapshot("r6_reservations", document, provenance="software-fixture:actual-upstream-api")
    assert supplied.operations[0].original_state == state
    assert supplied.operations[0].target == ()  # Upstream safe snapshot omits this vector.
    assert supplied.resources == result.resources
    assert supplied.usage_verified is False and supplied.allocation_mapping_complete is False
    assert "NEVER_EXPOSE" not in repr(result) + repr(supplied)
    assert _files(tmp_path) == before


def test_old_allocation_versions_and_overload_are_not_freed(tmp_path, upstream):
    path, _ = _r6_reservations(tmp_path, upstream, newer_resources=True)
    result = read_snapshot(path, quiescent=True)
    assert result.revision == 2
    assert all(r.version == 2 and r.used > r.capacity for r in result.resources)
    assert result.operations[0].target[0].version == 1
    assert result.robots[0].current[0].version == 1


def test_reader_uses_only_private_uri_ro_query_only_and_whitelisted_journal_sql(tmp_path, upstream, monkeypatch):
    path, _ = _r6_journal(tmp_path, upstream)
    calls, statements = [], []
    actual = sqlite3.connect
    def checked(database, *args, **kwargs):
        calls.append((database, kwargs.copy()))
        assert str(database).startswith("file:") and str(database).endswith("?mode=ro")
        assert kwargs.get("uri") is True
        assert str(path) not in str(database)
        conn = actual(database, *args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn
    monkeypatch.setattr(reader.sqlite3, "connect", checked)
    result = read_snapshot(path, quiescent=True)
    assert result.backend == "r6_journal" and len(calls) == 1
    assert "PRAGMA query_only=ON" in statements and "BEGIN" in statements and "ROLLBACK" in statements
    assert any(s.startswith("SELECT op_id,robot,plan_hash,state FROM operations ") for s in statements)
    assert not any(s.startswith("SELECT body") or s.startswith("SELECT *") for s in statements)


def test_live_uncheckpointed_wal_refuses_before_sqlite_and_preserves_shm(tmp_path, upstream, monkeypatch):
    module, _, _ = upstream
    path = tmp_path / "SOFTWARE_FIXTURE-live-WAL.db"
    owner = module.Journal(path)
    try:
        owner.enroll("car-001", "boot-1", expected_generation=0)
        assert Path(str(path) + "-wal").stat().st_size > 0
        assert Path(str(path) + "-shm").is_file()
        before = _files(tmp_path)
        def forbidden(*args, **kwargs):
            pytest.fail("SQLite must not open live/WAL input, even read-only")
        monkeypatch.setattr(reader.sqlite3, "connect", forbidden)
        with pytest.raises(SnapshotError, match="WAL/SHM"):
            read_snapshot(path, quiescent=True)
        assert _files(tmp_path) == before  # main, WAL, SHM bytes and membership.
    finally:
        owner.close()


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_any_sidecar_refuses_without_creating_files(tmp_path, suffix, monkeypatch):
    path = _current(tmp_path)
    Path(str(path) + suffix).write_bytes(b"SOFTWARE_FIXTURE-hot-or-stale-sidecar")
    before = _files(tmp_path)
    monkeypatch.setattr(reader.sqlite3, "connect", lambda *a, **k: pytest.fail("No SQLite call allowed"))
    with pytest.raises(SnapshotError, match="WAL/SHM"):
        read_snapshot(path, quiescent=True)
    assert _files(tmp_path) == before


def test_checkpointed_wal_header_is_still_refused(tmp_path, upstream):
    module, _, _ = upstream
    path = tmp_path / "SOFTWARE_FIXTURE-closed-WAL.db"
    owner = module.Journal(path)
    owner.enroll("car-001", "boot-1", expected_generation=0)
    owner.close()
    assert path.read_bytes()[18:20] == b"\x02\x02"
    before = _files(tmp_path)
    with pytest.raises(SnapshotError, match="rollback-mode"):
        read_snapshot(path, quiescent=True)
    assert _files(tmp_path) == before


def test_explicit_offline_declaration_backend_and_no_sql_paths(tmp_path):
    path = _current(tmp_path)
    for quiescent in (False, None, 1, "yes"):
        with pytest.raises(SnapshotError, match="quiescent"):
            read_snapshot(path, quiescent=quiescent)
    with pytest.raises(SnapshotError, match="differs"):
        read_snapshot(path, kind="r6_journal", quiescent=True)
    with pytest.raises(SnapshotError, match="backend"):
        read_snapshot(path, kind="transactions; DROP TABLE transactions", quiescent=True)
    with pytest.raises(SnapshotError):
        read_snapshot("file:" + str(path) + "?mode=rw", quiescent=True)


@pytest.mark.parametrize("change", [
    lambda p: p.update(txn_id="different"),
    lambda p: p.update(domain="unbound"),
    lambda p: p.update(airtime=True),
    lambda p: p.update(extra="unsupported"),
    lambda p: p["request"].update(password="SECRET_MUST_NEVER_ESCAPE"),
])
def test_current_hashed_but_corrupt_schema_is_rejected(tmp_path, change):
    path = _current(tmp_path)
    _edit_body(path, "transactions", change)
    before = _files(tmp_path)
    with pytest.raises(SnapshotError):
        read_snapshot(path, quiescent=True)
    assert _files(tmp_path) == before


@pytest.mark.parametrize("sql,args", [
    ("UPDATE transactions SET sha256=?", ("0" * 64,)),
    ("UPDATE transactions SET state='UNRECOGNIZED'", ()),
    ("UPDATE transactions SET state='UNKNOWN',outcome=NULL", ()),
    ("ALTER TABLE transactions ADD COLUMN unsupported TEXT", ()),
])
def test_current_bad_hash_state_or_sql_schema_is_rejected(tmp_path, sql, args):
    path = _current(tmp_path)
    _edit(path, sql, args)
    with pytest.raises(SnapshotError):
        read_snapshot(path, quiescent=True)


@pytest.mark.parametrize("change", [
    lambda b: b.update(epoch="other-boot"),
    lambda b: b.update(generation=2),
    lambda b: b.update(site="other-site"),
    lambda b: b.update(op_id="other-op"),
    lambda b: b["target"][0].update(unit="mbps"),
    lambda b: b["target"][0].update(version=2),
    lambda b: b["target"][0].update(amount=True),
    lambda b: b.update(source=[]),
])
def test_reservation_bindings_units_or_current_occupancy_corruption_refuses(tmp_path, upstream, change):
    path, _ = _r6_reservations(tmp_path, upstream)
    _edit_body(path, "rf_ops", change)
    with pytest.raises(SnapshotError):
        read_snapshot(path, quiescent=True)


def test_unknown_resource_unit_is_not_guessed(tmp_path, upstream):
    path, _ = _r6_reservations(tmp_path, upstream)
    _edit_body(path, "rf_resources", lambda b: b.update(unit="percent"))
    with pytest.raises(SnapshotError, match="unit"):
        read_snapshot(path, quiescent=True)


def test_unknown_r6_states_never_become_terminal(tmp_path, upstream):
    path, document = _r6_journal(tmp_path, upstream)
    _edit(path, "UPDATE operations SET state='RETRY_ME'")
    with pytest.raises(SnapshotError, match="state"):
        read_snapshot(path, quiescent=True)
    document["operations"][0]["state"] = "RETRY_ME"
    with pytest.raises(SnapshotError, match="state"):
        parse_snapshot("r6_journal", document, provenance="software-fixture")


def test_owner_dict_is_detached_and_does_not_release_unknown(tmp_path, upstream):
    _, document = _r6_reservations(tmp_path, upstream)
    result = parse_snapshot("r6_reservations", document, provenance="software-fixture")
    document["usage"]["radio:b"] = 0
    document["resources"]["radio:b"]["unit"] = "mbps"
    document["operations"][0]["state"] = "COMMITTED"
    assert result.operations[0].original_state == "UNKNOWN"
    assert result.resources[1].unit == "airtime_fraction" and result.resources[1].used == pytest.approx(.65)
    assert result.operations[0].retry_allowed is False


@pytest.mark.parametrize("change", [
    lambda d: d.update(radio_actuation_performed=True),
    lambda d: d.update(capacity_source="INFERRED_FROM_RSSI"),
    lambda d: d["usage"].update({"radio:a": 0}),
    lambda d: d["operations"][0].update(issued_ms=None),
    lambda d: d.update(overloaded=["radio:b"]),
    lambda d: d.update(extra="unrecognized"),
])
def test_owner_dict_scope_usage_issue_and_schema_refuse(tmp_path, upstream, change):
    _, document = _r6_reservations(tmp_path, upstream)
    change(document)
    with pytest.raises(SnapshotError):
        parse_snapshot("r6_reservations", document, provenance="software-fixture")


def test_limits_bad_paths_and_no_dynamic_sql(tmp_path):
    path = _current(tmp_path)
    with pytest.raises(SnapshotError, match="bounded"):
        read_snapshot(path, quiescent=True, limits=ReadLimits(max_db_bytes=100))
    with pytest.raises(SnapshotError):
        read_snapshot(path, quiescent=True, limits=ReadLimits(max_json_bytes=32))
    for kwargs in ({"max_rows": True}, {"timeout_seconds": float("nan")}, {"max_db_bytes": 2**40}):
        with pytest.raises(SnapshotError):
            ReadLimits(**kwargs)
    alias = tmp_path / "SOFTWARE_FIXTURE-alias.db"
    alias.symlink_to(path)
    with pytest.raises(SnapshotError, match="Symlink"):
        read_snapshot(alias, quiescent=True)
    alias.unlink()
    os.link(path, alias)
    with pytest.raises(SnapshotError, match="unaliased"):
        read_snapshot(path, quiescent=True)


def test_row_budget_and_ambiguous_schema(tmp_path, upstream):
    path, _ = _r6_reservations(tmp_path, upstream)
    with pytest.raises(SnapshotError, match="row budget"):
        read_snapshot(path, quiescent=True, limits=ReadLimits(max_rows=1))
    # A caller must explicitly choose if its offline DB carries both schemas.
    with sqlite3.connect(path) as owner:
        owner.execute("CREATE TABLE sessions(robot TEXT PRIMARY KEY,epoch TEXT NOT NULL,generation INTEGER NOT NULL,seq INTEGER NOT NULL,disabled INTEGER NOT NULL)")
        owner.execute("CREATE TABLE operations(op_id TEXT PRIMARY KEY,robot TEXT NOT NULL,plan_hash TEXT NOT NULL,body TEXT NOT NULL,state TEXT NOT NULL,result TEXT)")
    with pytest.raises(SnapshotError, match="ambiguous"):
        read_snapshot(path, quiescent=True)
    assert read_snapshot(path, kind="r6_reservations", quiescent=True).backend == "r6_reservations"


def test_dto_has_no_secret_containers_or_execution_switch(tmp_path, upstream):
    path, _ = _r6_reservations(tmp_path, upstream)
    result = asdict(read_snapshot(path, quiescent=True))
    text = json.dumps(result)
    assert all('"' + key + '":' not in text for key in ("credential", "action", "request", "evidence"))
    assert result["execution_authorized"] is False
