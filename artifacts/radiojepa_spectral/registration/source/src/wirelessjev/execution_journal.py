"""Opt-in durable execution obligations; no hardware activation or exactly-once claim."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from . import execution as core
try:
    import fcntl
except ImportError:
    fcntl = None


class JournalError(RuntimeError):
    pass


class JournalOwnershipError(JournalError):
    pass


_STATES = {'PREPARED': None, 'ISSUED': None, 'UNKNOWN': 'UNKNOWN',
           'COMMITTED': 'APPLIED', 'ABORTED': 'NOT_APPLIED', 'CANCELLED': 'NOT_APPLIED'}
_NEXT = {'PREPARED': {'ISSUED', 'CANCELLED'}, 'ISSUED': {'UNKNOWN', 'COMMITTED', 'ABORTED'},
         'UNKNOWN': {'UNKNOWN', 'COMMITTED', 'ABORTED'}}
_PAYLOAD = {'txn_id', 'car_id', 'from_bssid', 'to_bssid', 'request',
            'expected_interface', 'expected_role', 'domain', 'airtime'}
_REQUEST = {'action', 'ssid', 'bssid', 'security'}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class DurableExecution:
    """Exclusive local POSIX owner, fixed capacity, serialized transport calls.

    Terminal IDs are retained. Reopen cancels PREPARED work and restores any
    issue intent as UNKNOWN. No monotonic timestamp survives as fresh evidence.
    The wrapper belongs outside the fast packet path and changes no defaults.
    """
    def __init__(self, path, budgets, *, max_in_flight=3, max_records=4096):
        self._lock = threading.RLock(); self._owner = os.getpid()
        self._failed = False; self._closed = False; self._conn = None; self._lock_fd = None
        if fcntl is None: raise JournalOwnershipError('POSIX ownership locking is required')
        if type(max_records) is not int or max_records < 1: raise ValueError('max_records must be positive integer')
        self._ledger = core.Ledger(budgets, max_in_flight=max_in_flight)
        if len(self._ledger.budgets) > 128: raise ValueError('At most 128 domains are supported')
        self._max_records = max_records; self.path = Path(path).resolve()
        config = _json({'schema': 1, 'execution_semantics': core.EXECUTION_SEMANTICS_VERSION,
            'execution_source_sha256': hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
            'budgets': self._ledger.budgets, 'max_in_flight': max_in_flight, 'max_records': max_records})
        try:
            if not self.path.parent.is_dir() or (self.path.exists() and not self.path.is_file()):
                raise JournalError('Journal requires a local file in an existing directory')
            if self.path.exists() and self.path.stat().st_nlink != 1:
                raise JournalOwnershipError('Hard-linked journal aliases are unsupported')
            self._lock_fd = os.open(str(self.path)+'.lock', os.O_CREAT | os.O_RDWR, 0o600)
            try: fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: raise JournalOwnershipError('Journal already has an owner') from None
            existed = self.path.exists()
            self._conn = sqlite3.connect(self.path, timeout=.2, isolation_level=None, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            if self._conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok': raise JournalError('Corrupt journal')
            if self._conn.execute('PRAGMA journal_mode=WAL').fetchone()[0].lower() != 'wal':
                raise JournalError('WAL durability is unavailable')
            self._conn.execute('PRAGMA synchronous=FULL')
            self._conn.execute('PRAGMA wal_autocheckpoint=128')
            self._conn.execute('PRAGMA journal_size_limit=4194304')
            if existed:
                row = self._conn.execute('SELECT config FROM metadata WHERE singleton=1').fetchone()
                if row is None or row[0] != config: raise JournalError('Journal configuration/schema differs')
            else:
                def initialize():
                    self._conn.execute('CREATE TABLE metadata(singleton INTEGER PRIMARY KEY CHECK(singleton=1),config TEXT NOT NULL)')
                    self._conn.execute('INSERT INTO metadata VALUES(1,?)', (config,))
                    self._conn.execute('CREATE TABLE transactions(seq INTEGER PRIMARY KEY,txn_id TEXT UNIQUE NOT NULL,'
                        'payload TEXT NOT NULL,sha256 TEXT NOT NULL,state TEXT NOT NULL,outcome TEXT)')
                self._write(initialize)
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(directory)
                finally: os.close(directory)
            self._restore()
        except BaseException as error:
            self._failed = True; self.close()
            if isinstance(error, (JournalError, KeyboardInterrupt, SystemExit)): raise
            raise JournalError('Cannot open/restore durable journal: '+str(error)) from error

    def _owner_check(self):
        if os.getpid() != self._owner: raise JournalOwnershipError('Inherited journal cannot be used after fork')

    def _ensure(self):
        self._owner_check()
        if self._closed or self._failed: raise JournalError('Journal closed/failed; further issue is forbidden')

    def _write(self, operation):
        try:
            self._conn.execute('BEGIN IMMEDIATE'); operation(); self._conn.commit()
        except BaseException as error:
            self._failed = True
            try: self._conn.rollback()
            except Exception: pass
            if isinstance(error, (KeyboardInterrupt, SystemExit)): raise
            raise JournalError('Durable journal commit failed') from error

    def _persist_state(self, txn_id, state, outcome):
        def update():
            row = self._conn.execute('SELECT state FROM transactions WHERE txn_id=?', (txn_id,)).fetchone()
            if (state not in _STATES or _STATES[state] != outcome or row is None
                    or (state != row[0] and state not in _NEXT.get(row[0], set()))):
                raise JournalError('Invalid durable transition')
            self._conn.execute('UPDATE transactions SET state=?,outcome=? WHERE txn_id=?', (state, outcome, txn_id))
        self._write(update)

    def _store(self, txn_id, state, outcome):
        try: self._persist_state(txn_id, state, outcome)
        except BaseException as error:
            self._failed = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)): raise
            raise JournalError('State persistence failed; issue remains ambiguous') from error

    def _rows(self):
        try:
            count = self._conn.execute('SELECT count(*) FROM transactions').fetchone()[0]
            if count > self.max_records: raise JournalError('Journal exceeds configured capacity')
            return self._conn.execute('SELECT * FROM transactions ORDER BY seq').fetchall()
        except Exception as error:
            self._failed = True
            raise JournalError('Cannot read journal obligations') from error

    @staticmethod
    def _decode(row):
        text = row['payload']
        if not isinstance(text, str) or len(text.encode()) > 4096 or _digest(text) != row['sha256']:
            raise JournalError('Immutable admission payload is corrupt')
        payload = json.loads(text)
        if (not isinstance(payload, dict) or set(payload) != _PAYLOAD or payload['txn_id'] != row['txn_id']
                or not isinstance(payload['request'], dict) or set(payload['request']) != _REQUEST
                or row['state'] not in _STATES or row['outcome'] != _STATES[row['state']]):
            raise JournalError('Invalid journal record/schema/outcome')
        return payload

    @staticmethod
    def _payload(entry):
        return {'txn_id': entry.txn_id, 'car_id': entry.car_id, 'from_bssid': entry.from_bssid,
                'to_bssid': entry.to_bssid, 'request': entry.wire, 'expected_interface': entry.interface,
                'expected_role': entry.role, 'domain': entry.domain, 'airtime': entry.airtime}

    def _restore(self):
        rows = self._rows()
        for row in rows: self._decode(row)
        def recover():
            self._conn.execute("UPDATE transactions SET state='CANCELLED',outcome='NOT_APPLIED' WHERE state='PREPARED'")
            self._conn.execute("UPDATE transactions SET state='UNKNOWN',outcome='UNKNOWN' WHERE state='ISSUED'")
        self._write(recover); rows = self._rows()
        rows.sort(key=lambda r: r['state'] == 'UNKNOWN')  # settled history before active reservations
        for row in rows:
            p = self._decode(row)
            txn = core.Transaction(p['txn_id'], p['car_id'], p['from_bssid'], p['to_bssid'], p['request'],
                                   expected_interface=p['expected_interface'], expected_role=p['expected_role'])
            if not self._ledger.prepare(txn, p['domain'], p['airtime']):
                raise JournalError('Recovered ownership/reservations exceed configuration')
            if row['state'] == 'CANCELLED': self._ledger.cancel(txn)
            else:
                entry, _ = self._ledger._claim_issue(txn)
                self._ledger._finish_issue(entry, row['state'], row['outcome'])
            txn.history.clear()  # reconstruction is not a new issue or a recovered original event clock

    def _lookup(self, txn_id):
        self._ensure()
        try:
            row = self._conn.execute('SELECT * FROM transactions WHERE txn_id=?', (txn_id,)).fetchone()
            if row is None: raise KeyError(txn_id)
            payload = self._decode(row)
            if txn_id not in self._ledger.transactions: raise JournalError('Unrestored journal obligation')
            txn = self._ledger.transactions[txn_id]
            entry = self._ledger._entry(txn)
            if entry.state != row['state'] or _json(payload) != _json(self._payload(entry)):
                raise JournalError('Live and durable identity/state differ')
            return txn, row['state']
        except KeyError: raise
        except Exception as error:
            self._failed = True
            raise JournalError('Cannot validate journal obligation') from error

    def prepare(self, txn, domain, airtime):
        self._owner_check()
        with self._lock:
            self._ensure()
            if not isinstance(txn, core.Transaction): raise ValueError('Use execution.Transaction')
            if any(row['txn_id'] == txn.txn_id for row in self._rows()): return False
            if len(self._ledger.transactions) >= self.max_records: return False
            if not isinstance(txn.request, dict) or set(txn.request) != _REQUEST:
                raise ValueError('Only fixed credential-free connect fields may be journaled')
            if not self._ledger.prepare(txn, domain, airtime): return False
            entry = self._ledger._entry(txn)
            payload = _json(self._payload(entry))
            if len(payload.encode()) > 4096:
                self._ledger.cancel(txn); raise ValueError('Bounded admission payload exceeded')
            try:
                self._write(lambda: self._conn.execute('INSERT INTO transactions(txn_id,payload,sha256,state,outcome) VALUES(?,?,?,?,?)',
                                                       (txn.txn_id, payload, _digest(payload), 'PREPARED', None)))
            except BaseException:
                self._ledger.cancel(txn); raise
            return True

    def execute(self, txn_id, transport):
        self._owner_check()
        with self._lock:
            txn, state = self._lookup(txn_id)
            if state != 'PREPARED': return txn
            entry = self._ledger._entry(txn); self._ledger._unchanged(entry)
            self._store(txn_id, 'ISSUED', None)  # FULL synchronous commit BEFORE transport
            try: core.execute(txn, self._ledger, transport)
            except BaseException:
                with self._ledger._lock:
                    entry.issued = True; self._ledger._mark(entry, 'UNKNOWN', 'UNKNOWN')
                self._store(txn_id, 'UNKNOWN', 'UNKNOWN')
                raise
            self._store(txn_id, entry.state, _STATES[entry.state])
            return txn

    def reconcile(self, txn_id, transport, **kwargs):
        self._owner_check()
        with self._lock:
            txn, state = self._lookup(txn_id)
            if state != 'UNKNOWN': return txn
            entry = self._ledger._entry(txn)
            try: core.reconcile(txn, self._ledger, transport, **kwargs)
            finally: self._store(txn_id, entry.state, _STATES[entry.state])
            return txn

    def cancel(self, txn_id):
        self._owner_check()
        with self._lock:
            txn, state = self._lookup(txn_id)
            if state != 'PREPARED': return False
            self._store(txn_id, 'CANCELLED', 'NOT_APPLIED')
            if not self._ledger.cancel(txn):
                self._failed = True; raise JournalError('Durable cancellation and live ownership differ')
            return True

    def get(self, txn_id):
        self._owner_check()
        with self._lock: return self._lookup(txn_id)[0]

    def in_flight(self):
        self._owner_check()
        with self._lock:
            self._ensure(); return tuple(self._ledger.in_flight())

    @property
    def reserved(self):
        self._owner_check()
        with self._lock:
            self._ensure(); return self._ledger.reserved.copy()

    @property
    def max_records(self):
        return self._max_records

    @property
    def failed(self):
        self._owner_check()
        return self._failed

    def close(self):
        self._owner_check()
        with self._lock:
            if self._closed: return
            self._closed = True
            try:
                if self._conn is not None: self._conn.close()
            finally:
                if self._lock_fd is not None:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN); os.close(self._lock_fd); self._lock_fd = None

    def __enter__(self): self._ensure(); return self
    def __exit__(self, *exception): self.close()
