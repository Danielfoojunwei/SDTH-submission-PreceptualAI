"""Execution: turn an approved fleet plan into per-car SmartCarCmd requests.

The wire format is exactly what SmartCar.zip's PI/SmartCarCmd.py reads on
stdin, one JSON object per call:

    {"action": "status"}
    {"action": "connect", "ssid": ..., "bssid": ..., "security": ...}

Credentials are never in the plan or the request built here. SmartCarCmd's
connect does need the Wi-Fi password in the request (the PC app sends it from
WiFi.json); the real transport (carlink.py) adds it only at send time, so
model inputs, receipts and logs carry no secrets. SmartCarCmd itself re-checks interface roles (never the HOTPOT
management radio, never 192.168.1.11), serialises radio access with a lock,
and verifies SSID/BSSID after nmcli activation; this layer adds the fleet
transaction semantics around it:

    PREPARED -> ISSUED -> COMMITTED | ABORTED | UNKNOWN
    CANCELLED (only before ISSUED)

- A failure before the request is sent is NOT_APPLIED (safe to retry later).
- AP_NOT_VISIBLE / INTERFACE_UNSAFE responses carry activation_attempted=False
  and are NOT_APPLIED.
- Any other failure after sending is UNKNOWN: never blindly resent; it is
  resolved only by a fresh status read (reconcile).
- Destination airtime stays reserved until the outcome is known.

Transports: DryRunTransport (default, simulates the car in the twin) and
SSHTransport (stub that refuses to run until explicitly enabled per car).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
import copy
import json
import math
import random
import re
import threading
import time
from typing import Protocol

EXECUTION_SEMANTICS_VERSION = 2
_MAC = re.compile(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\Z")
_SIM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_TERMINAL = frozenset(("COMMITTED", "ABORTED", "CANCELLED"))


class PreSendRefused(PermissionError):
    """Transport certifies that no mutation was sent or attempted."""


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not _SIM_ID.fullmatch(value):
        raise ValueError(f"Invalid {name}")
    return value


def _same_bssid(a, b):
    if isinstance(a, str) and isinstance(b, str) and _MAC.fullmatch(a) and _MAC.fullmatch(b):
        return a.lower() == b.lower()
    return a == b


def _bssid(value, transport):
    if not isinstance(value, str):
        return None
    if _MAC.fullmatch(value):
        octets = bytes.fromhex(value.replace(":", ""))
        if octets[0] & 1 or not any(octets):
            return None                 # multicast/broadcast/zero cannot identify an AP
        return value.lower()
    # Symbolic AP IDs belong only to the existing digital-twin transport.
    if isinstance(transport, DryRunTransport) and _SIM_ID.fullmatch(value):
        return value
    return None


class Transport(Protocol):
    def send(self, car_id: str, request: dict) -> dict: ...


@dataclass
class DryRunTransport:
    """Simulates SmartCarCmd responses. failure_rate gives clean failures
    (AP_NOT_VISIBLE, activation not attempted); unknown_rate drops the reply
    after the car may or may not have switched."""
    state: dict[str, str]                       # car -> bssid actually in use
    failure_rate: float = 0.0
    unknown_rate: float = 0.0
    seed: int = 0
    sent: list[tuple[str, dict]] = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def send(self, car_id: str, request: dict) -> dict:
        self.sent.append((car_id, request))
        if request["action"] == "status":
            return {"ok": True, "result": {"connected": True,
                                           "bssid": self.state.get(car_id),
                                           "interface": "wlan1", "role": "WIFI"}}
        r = self._rng.random()
        if r < self.failure_rate:
            return {"ok": False, "code": "AP_NOT_VISIBLE",
                    "activation_attempted": False, "error": "target AP not visible"}
        if r < self.failure_rate + self.unknown_rate:
            if self._rng.random() < 0.5:           # it may have switched
                self.state[car_id] = request["bssid"]
            raise TimeoutError("SSH channel closed; outcome unknown")
        self.state[car_id] = request["bssid"]
        return {"ok": True, "result": {"connected": True, "bssid": request["bssid"],
                                       "ssid": request["ssid"]}}


class SSHTransport:
    """Placeholder for the real per-car SSH path to SmartCarCmd.py. Refuses
    to send until a car is explicitly enabled; enabling is an operator action,
    not something the planner can do."""

    def __init__(self, enabled_cars: set[str] | None = None):
        self.enabled = set(enabled_cars or ())

    def send(self, car_id: str, request: dict) -> dict:
        if car_id not in self.enabled:
            raise PreSendRefused(f"{car_id}: real transport not enabled")
        raise NotImplementedError("wire to SmartCar ssh_manager on the site PC")


@dataclass
class Transaction:
    txn_id: str
    car_id: str
    from_bssid: str | None
    to_bssid: str
    request: dict
    state: str = "PREPARED"
    outcome: str | None = None
    history: list[tuple[float, str]] = field(default_factory=list)
    expected_interface: str = "wlan1"
    expected_role: str = "WIFI"

    def mark(self, state: str) -> None:
        self.state = state
        self.history.append((time.time(), state))


@dataclass
class _Admission:
    transaction: Transaction
    txn_id: str
    car_id: str
    from_bssid: str | None
    to_bssid: str
    wire: dict
    interface: str
    role: str
    domain: str
    airtime: float
    state: str = "PREPARED"
    issued: bool = False
    reservation_active: bool = True
    reconciling: bool = False


@dataclass
class Ledger:
    """Thread-safe, process-local reservation ownership and at-most-once issue.

    Public transaction/request objects are checked against their admitted
    snapshot. Private ownership, not request annotations, controls release.
    This is NOT durable crash recovery; outstanding state is memory-only.
    """
    budgets: dict[str, float]
    reserved: dict[str, float] = field(default_factory=dict)
    max_in_flight: int = 3
    transactions: dict[str, Transaction] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _entries: dict[str, _Admission] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if type(self.max_in_flight) is not int or self.max_in_flight < 1:
            raise ValueError("max_in_flight must be a positive integer")
        if not isinstance(self.budgets, Mapping):
            raise ValueError("Provide domain budgets")
        self.budgets = {_identifier(d, "domain"): _finite_nonnegative(v, "airtime budget")
                        for d, v in self.budgets.items()}
        self._budgets = self.budgets.copy()
        if self.reserved or self.transactions:
            raise ValueError("An unjournaled ledger cannot restore unowned reservations/transactions")
        self.reserved = {}
        self.transactions = {}

    def _entry(self, txn):
        if not isinstance(txn, Transaction):
            raise ValueError("Use a registered Transaction")
        entry = self._entries.get(txn.txn_id)
        if entry is None or entry.transaction is not txn or self.transactions.get(txn.txn_id) is not txn:
            raise ValueError("Transaction is not owned by this ledger")
        return entry

    def _total(self, domain):
        return math.fsum(e.airtime for e in self._entries.values() if e.domain == domain and e.reservation_active)

    def _release(self, entry):
        if not entry.reservation_active:
            return False
        entry.reservation_active = False
        self.reserved[entry.domain] = self._total(entry.domain)
        return True

    def _mark(self, entry, state, outcome):
        entry.state = state
        entry.transaction.mark(state)
        entry.transaction.outcome = outcome
        if state in _TERMINAL:
            self._release(entry)

    def in_flight(self) -> list[Transaction]:
        with self._lock:
            return [e.transaction for e in self._entries.values()
                    if e.state in ("PREPARED", "ISSUED", "UNKNOWN")]

    def prepare(self, txn: Transaction, domain: str, airtime: float) -> bool:
        airtime = _finite_nonnegative(airtime, "airtime")
        _identifier(domain, "domain")
        with self._lock:
            if not isinstance(txn, Transaction):
                raise ValueError("Use a Transaction")
            _identifier(txn.txn_id, "transaction identity"); _identifier(txn.car_id, "car identity")
            _identifier(txn.to_bssid, "target BSSID")
            if txn.from_bssid is not None: _identifier(txn.from_bssid, "origin BSSID")
            if txn.txn_id in self._entries:
                return False
            if txn.state != "PREPARED" or txn.outcome is not None:
                raise ValueError("Only a new PREPARED transaction can be admitted")
            if not isinstance(txn.request, dict): raise ValueError("Invalid connect request")
            wire = copy.deepcopy({k: v for k, v in txn.request.items() if not str(k).startswith("_")})
            if (set(wire) != {"action", "ssid", "bssid", "security"} or wire["action"] != "connect"
                    or not _same_bssid(wire["bssid"], txn.to_bssid)
                    or not isinstance(wire["ssid"], str) or not 1 <= len(wire["ssid"].encode()) <= 32
                    or wire["security"] not in ("wpa-psk", "sae", "open")):
                raise ValueError("Connect request differs from the declared target or format")
            _identifier(txn.expected_interface, "interface")
            if txn.expected_role != "WIFI": raise ValueError("Only the WIFI client role can be admitted")
            live = [e for e in self._entries.values() if e.state in ("PREPARED", "ISSUED", "UNKNOWN")]
            if (domain not in self._budgets or len(live) >= self.max_in_flight
                    or any(e.car_id == txn.car_id for e in live)
                    or math.fsum((self._total(domain), airtime)) > self._budgets[domain]):
                return False
            entry = _Admission(txn, txn.txn_id, txn.car_id, txn.from_bssid, txn.to_bssid,
                               wire, txn.expected_interface, txn.expected_role, domain, airtime)
            self._entries[txn.txn_id] = entry
            self.transactions[txn.txn_id] = txn
            self.reserved[domain] = self._total(domain)
            txn.request["_domain"], txn.request["_airtime"] = domain, airtime
            txn.mark("PREPARED")
            return True

    def release(self, txn: Transaction) -> bool:
        """Idempotent terminal settlement; never free an unresolved operation."""
        with self._lock:
            entry = self._entry(txn)
            return self._release(entry) if entry.state in _TERMINAL else False

    def cancel(self, txn: Transaction) -> bool:
        with self._lock:
            entry = self._entry(txn)
            if entry.issued or entry.state != "PREPARED": return False
            self._mark(entry, "CANCELLED", "NOT_APPLIED")
            return True

    def _unchanged(self, entry):
        txn = entry.transaction
        if not isinstance(txn.request, dict):
            raise ValueError("Transaction request differs from its admitted format")
        wire = {k: v for k, v in txn.request.items() if not str(k).startswith("_")}
        if ((txn.txn_id, txn.car_id, txn.from_bssid, txn.to_bssid, txn.expected_interface, txn.expected_role)
                != (entry.txn_id, entry.car_id, entry.from_bssid, entry.to_bssid, entry.interface, entry.role)
                or wire != entry.wire):
            raise ValueError("Transaction differs from its admitted identity/request")

    def _claim_issue(self, txn):
        with self._lock:
            entry = self._entry(txn)
            if entry.issued or entry.state != "PREPARED": return None
            self._unchanged(entry)
            if txn.state != "PREPARED" or not entry.reservation_active:
                raise ValueError("Transaction is not an admitted PREPARED operation")
            entry.issued = True
            self._mark(entry, "ISSUED", None)
            return entry, copy.deepcopy(entry.wire)

    def _finish_issue(self, entry, state, outcome):
        with self._lock:
            if entry.state == "ISSUED": self._mark(entry, state, outcome)

    def _claim_reconcile(self, txn):
        with self._lock:
            entry = self._entry(txn)
            if entry.state != "UNKNOWN" or entry.reconciling: return None
            self._unchanged(entry)
            entry.reconciling = True
            return entry

    def _finish_reconcile(self, entry, state=None, outcome=None):
        with self._lock:
            if entry.state == "UNKNOWN" and state is not None: self._mark(entry, state, outcome)
            entry.reconciling = False


def build_connect(ssid: str, bssid: str, security: str = "wpa-psk") -> dict:
    """SmartCarCmd connect request without credentials."""
    return {"action": "connect", "ssid": ssid, "bssid": bssid, "security": security}


def execute(txn: Transaction, ledger: Ledger, transport: Transport) -> Transaction:
    """Claim a ledger-owned operation once; retries of issued work never send.

    Network I/O occurs outside the ledger lock. Only a certified pre-send
    refusal or an explicit nonattempted response establishes NOT_APPLIED.
    """
    claim = ledger._claim_issue(txn)
    if claim is None: return txn
    entry, wire = claim
    if _bssid(entry.to_bssid, transport) is None:
        ledger._finish_issue(entry, "ABORTED", "NOT_APPLIED")
        return txn
    try:
        response = transport.send(entry.car_id, wire)
        if _connection_bssid(response, entry, transport, status=False) == _bssid(entry.to_bssid, transport):
            state, outcome = "COMMITTED", "APPLIED"
        elif (isinstance(response, Mapping) and response.get("ok") is False
              and response.get("activation_attempted") is False and _reply_identity(response, entry)):
            state, outcome = "ABORTED", "NOT_APPLIED"
        else:
            state, outcome = "UNKNOWN", "UNKNOWN"
    except PreSendRefused:
        state, outcome = "ABORTED", "NOT_APPLIED"
    except Exception:
        state, outcome = "UNKNOWN", "UNKNOWN"
    except BaseException:
        # Preserve reconciliation obligations even when an operator interrupts
        # a transport after issue. Do not swallow the shutdown/interrupt itself.
        ledger._finish_issue(entry, "UNKNOWN", "UNKNOWN")
        raise
    ledger._finish_issue(entry, state, outcome)
    return txn


def _reply_identity(response, entry):
    for part in (response, response.get("result")):
        if isinstance(part, Mapping):
            if "car_id" in part and part["car_id"] != entry.car_id: return False
            if "txn_id" in part and part["txn_id"] != entry.txn_id: return False
    return True


def _connection_bssid(response, entry, transport, *, status):
    if not isinstance(response, Mapping) or response.get("ok") is not True or not _reply_identity(response, entry):
        return None
    result = response.get("result")
    if not isinstance(result, Mapping) or result.get("connected") is not True:
        return None
    for key, expected in (("interface", entry.interface), ("role", entry.role)):
        if (status or key in result) and result.get(key) != expected: return None
    bssid = _bssid(result.get("bssid"), transport)
    if bssid is None: return None
    if _same_bssid(bssid, entry.to_bssid):
        if (not status or "ssid" in result) and result.get("ssid") != entry.wire["ssid"]: return None
    return bssid


def reconcile(txn: Transaction, ledger: Ledger, transport: Transport, *,
              max_status_age_s: float = 5., clock=time.monotonic) -> Transaction:
    """Resolve UNKNOWN from one fresh status query, never another connect.

    An untimestamped legacy status relies on the transport's fresh-query
    contract. If observed_at_monotonic is supplied, it must share the caller's
    clock and lie within this query. Slow/backwards/invalid reads stay UNKNOWN.
    The timeout rejects late output; it cannot interrupt a blocking transport.
    """
    max_age = _finite_nonnegative(max_status_age_s, "maximum status age")
    if max_age == 0: raise ValueError("maximum status age must be positive")
    entry = ledger._claim_reconcile(txn)
    if entry is None: return txn
    state = outcome = None
    try:
        started = _finite_nonnegative(clock(), "status query clock")
        response = transport.send(entry.car_id, {"action": "status"})
        finished = _finite_nonnegative(clock(), "status completion clock")
        bssid = _connection_bssid(response, entry, transport, status=True)
        fresh = started <= finished and finished - started <= max_age
        if isinstance(response, Mapping) and isinstance(response.get("result"), Mapping):
            result = response["result"]
            if "observed_at_monotonic" in result:
                measured = _finite_nonnegative(result["observed_at_monotonic"], "status observation clock")
                fresh = fresh and started <= measured <= finished and finished - measured <= max_age
        if fresh and bssid is not None:
            if _same_bssid(bssid, entry.to_bssid): state, outcome = "COMMITTED", "APPLIED"
            elif entry.from_bssid is not None and _same_bssid(bssid, entry.from_bssid):
                state, outcome = "ABORTED", "NOT_APPLIED"
    except Exception:
        pass                         # malformed, failed or stale status proves nothing
    finally:
        ledger._finish_reconcile(entry, state, outcome)
    return txn


def redact(obj) -> str:
    """Log helper: refuses to serialise objects with secret-bearing fields
    (the field names SmartCarCmd redacts: password, psk, ...)."""
    secret_keys = ("password", "psk", "passwd", "secret", "token")

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(word in str(key).lower() for word in secret_keys):
                    raise ValueError("SECRET_IN_LOG")
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)

    walk(obj)
    return json.dumps(obj, default=str)
