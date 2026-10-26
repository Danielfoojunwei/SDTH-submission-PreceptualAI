"""Execution: turn an approved fleet plan into per-car SmartCarCmd requests.

The wire format is exactly what SmartCar.zip's PI/SmartCarCmd.py reads on
stdin, one JSON object per call:

    {"action": "status"}
    {"action": "connect", "ssid": ..., "bssid": ..., "security": ...}

Credentials are never in the plan or the request built here: the on-car agent
fills the password from its local WiFi.json, so model inputs and logs carry no
secrets. SmartCarCmd itself re-checks interface roles (never the HOTPOT
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
import json
import random
import time
from typing import Protocol


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
            raise PermissionError(f"{car_id}: real transport not enabled")
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

    def mark(self, state: str) -> None:
        self.state = state
        self.history.append((time.time(), state))


@dataclass
class Ledger:
    """Single-site reservation ledger keyed by contention domain."""
    budgets: dict[str, float]
    reserved: dict[str, float] = field(default_factory=dict)
    max_in_flight: int = 3
    transactions: dict[str, Transaction] = field(default_factory=dict)

    def in_flight(self) -> list[Transaction]:
        return [t for t in self.transactions.values()
                if t.state in ("PREPARED", "ISSUED", "UNKNOWN")]

    def prepare(self, txn: Transaction, domain: str, airtime: float) -> bool:
        if len(self.in_flight()) >= self.max_in_flight:
            return False
        if any(t.car_id == txn.car_id for t in self.in_flight()):
            return False
        if self.reserved.get(domain, 0.0) + airtime > self.budgets.get(domain, 0.0) + 1e-9:
            return False
        self.reserved[domain] = self.reserved.get(domain, 0.0) + airtime
        txn.request["_domain"], txn.request["_airtime"] = domain, airtime
        self.transactions[txn.txn_id] = txn
        txn.mark("PREPARED")
        return True

    def release(self, txn: Transaction) -> None:
        d, a = txn.request.get("_domain"), txn.request.get("_airtime", 0.0)
        if d is not None:
            self.reserved[d] = max(0.0, self.reserved.get(d, 0.0) - a)


def build_connect(ssid: str, bssid: str, security: str = "wpa-psk") -> dict:
    """SmartCarCmd connect request without credentials."""
    return {"action": "connect", "ssid": ssid, "bssid": bssid, "security": security}


def execute(txn: Transaction, ledger: Ledger, transport: Transport) -> Transaction:
    wire = {k: v for k, v in txn.request.items() if not k.startswith("_")}
    txn.mark("ISSUED")
    try:
        response = transport.send(txn.car_id, wire)
    except PermissionError:
        txn.mark("CANCELLED")
        txn.outcome = "NOT_APPLIED"
        ledger.release(txn)
        return txn
    except Exception:
        txn.mark("UNKNOWN")
        txn.outcome = "UNKNOWN"            # reservation stays held
        return txn
    if response.get("ok"):
        txn.mark("COMMITTED")
        txn.outcome = "APPLIED"
    elif response.get("activation_attempted") is False:
        txn.mark("ABORTED")
        txn.outcome = "NOT_APPLIED"
        ledger.release(txn)
    else:
        txn.mark("UNKNOWN")
        txn.outcome = "UNKNOWN"
    return txn


def reconcile(txn: Transaction, ledger: Ledger, transport: Transport) -> Transaction:
    """Resolve UNKNOWN only from a fresh status read; never resend connect."""
    if txn.state != "UNKNOWN":
        return txn
    try:
        status = transport.send(txn.car_id, {"action": "status"})
    except Exception:
        return txn
    bssid = status.get("result", {}).get("bssid")
    if bssid == txn.to_bssid:
        txn.mark("COMMITTED")
        txn.outcome = "APPLIED"
    elif bssid == txn.from_bssid:
        txn.mark("ABORTED")
        txn.outcome = "NOT_APPLIED"
        ledger.release(txn)
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
