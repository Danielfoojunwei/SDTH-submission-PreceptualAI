"""One supervised SmartCar connect, with conservative transport outcomes.

This module does not provide SSH, install software, select an AP, or retry a
mutation. Its injected journal must durably flush an event before returning.
Unit tests of this boundary do not establish actual hardware or route safety.
"""
from __future__ import annotations

import re
from typing import Callable

_BSSID = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")
_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_READ_TIMEOUT = 15.0
_CONNECT_TIMEOUT = 150.0
_RECONCILE_READS = 3


def _bssid(value):
    if not isinstance(value, str):
        return None
    value = value.lower()
    if not _BSSID.fullmatch(value):
        return None
    if value == "00:00:00:00:00:00" or int(value[:2], 16) & 1:
        return None
    return value


def _profile(value):
    if not isinstance(value, dict) or value.get("enabled") is not True:
        return None
    ssid, security, password = (value.get(key) for key in ("ssid", "security", "password"))
    if not isinstance(ssid, str):
        return None
    try:
        if not 1 <= len(ssid.encode("utf-8")) <= 32:
            return None
    except UnicodeError:
        return None
    if any(ord(char) < 32 or ord(char) == 127 for char in ssid):
        return None
    if security not in ("wpa-psk", "sae", "open") or not isinstance(password, str):
        return None
    if security != "open" and not password:
        return None
    if security == "open" and password:
        return None
    # Public receipt metadata must not itself contain the credential.
    if password and password in ssid:
        return None
    return {"ssid": ssid, "security": security, "password": password}


def _status(transport):
    reply = transport("smartcar", {"action": "status"}, _READ_TIMEOUT)
    if not isinstance(reply, dict) or reply.get("ok") is not True:
        return None
    value = reply.get("result")
    if not isinstance(value, dict):
        return None
    if value.get("interface") != "wlan1" or value.get("role") != "WIFI":
        return None
    if value.get("connected") is not True or _bssid(value.get("bssid")) is None:
        return None
    if not isinstance(value.get("ssid"), str):
        return None
    return {"ssid": value["ssid"], "bssid": _bssid(value["bssid"])}


def _doctor_safe(transport):
    reply = transport("agent", {"action": "doctor"}, _READ_TIMEOUT)
    if not isinstance(reply, dict) or reply.get("ok") is not True or reply.get("action") != "doctor":
        return False
    doctor = reply.get("doctor")
    role = doctor.get("role_check") if isinstance(doctor, dict) else None
    return (isinstance(role, dict) and role.get("safe") is True
            and role.get("management_guard") is True
            and role.get("interface") == "wlan1" and role.get("role") == "WIFI"
            and role.get("management_interface") == "wlan0")


def supervised_switch(transport: Callable, journal: Callable, *, profiledict: dict,
                      target_bssid: str, confirm: Callable = input) -> dict:
    """Request one same-SSID BSSID switch after a durable, typed intent.

    ``transport(key, payload, timeout)`` returns the raw remote JSON envelope.
    ``journal(event, **fields)`` must fsync before returning. Failed intent
    journaling prevents the connect. No exception text or remote error body is
    included in receipts. ``UNKNOWN`` permits only status reconciliation; a
    return to the original BSSID is a separate operator-confirmed invocation.

    The doctor attests the radio roles, not the independence of a VPN/SSH route.
    The caller must establish a management path that survives the radio change.
    """
    profile, target = _profile(profiledict), _bssid(target_bssid)
    receipt = {"outcome": "NOT_SENT", "connect_requests": 0,
               "mutation_attempted": False, "journal_complete": True,
               "return_requires_separate_confirmation": True}

    def emit(event, **fields):
        try:
            journal(event, **fields)
            return True
        except (Exception, KeyboardInterrupt):
            receipt["journal_complete"] = False
            return False

    def finish(outcome, reason, **fields):
        receipt.update(outcome=outcome, reason=reason, **fields)
        if receipt["mutation_attempted"] and not receipt["journal_complete"]:
            receipt.update(outcome="UNKNOWN", reason="JOURNAL_NOT_DURABLE",
                           instruction="Read status; do not resend the connect.")
        logged = emit("switch_result", **receipt)
        if receipt["mutation_attempted"] and (not logged or not receipt["journal_complete"]):
            receipt.update(outcome="UNKNOWN", reason="JOURNAL_NOT_DURABLE",
                           instruction="Read status; do not resend the connect.")
        return dict(receipt)

    if profile is None or target is None:
        return finish("NOT_SENT", "INVALID_PROFILE_OR_BSSID")
    if profile["password"] and profile["password"] in target:
        return finish("NOT_SENT", "CREDENTIAL_IN_PUBLIC_METADATA")

    try:
        if not _doctor_safe(transport):
            return finish("NOT_SENT", "RADIO_ROLES_UNSAFE_OR_UNKNOWN")
        original = _status(transport)
    except (Exception, KeyboardInterrupt):
        return finish("NOT_SENT", "PRECONDITION_READ_FAILED")
    if original is None or original["ssid"] != profile["ssid"]:
        return finish("NOT_SENT", "CONNECTED_ORIGINAL_LINK_REQUIRED")
    if original["bssid"] == target:
        return finish("NOT_SENT", "ALREADY_ON_TARGET")
    if profile["password"] and profile["password"] in original["bssid"]:
        return finish("NOT_SENT", "CREDENTIAL_IN_PUBLIC_METADATA")
    receipt.update(from_bssid=original["bssid"], from_ssid=original["ssid"],
                   target_bssid=target, ssid=profile["ssid"])
    if not emit("switch_precondition", **receipt):
        return finish("NOT_SENT", "JOURNAL_NOT_DURABLE")
    try:
        typed = confirm(f"Switch wlan1 from {original['bssid']} to {target} on "
                        f"{profile['ssid']!r}. Re-type the target BSSID: ")
    except (Exception, KeyboardInterrupt):
        return finish("NOT_SENT", "CONFIRMATION_CANCELLED")
    if not isinstance(typed, str) or typed.strip().lower() != target:
        return finish("NOT_SENT", "CONFIRMATION_MISMATCH")

    # Human confirmation may take arbitrarily long. Do not use its stale status.
    try:
        if not _doctor_safe(transport):
            return finish("NOT_SENT", "RADIO_ROLES_CHANGED_OR_UNKNOWN")
        current = _status(transport)
    except (Exception, KeyboardInterrupt):
        return finish("NOT_SENT", "PRECONDITION_RECHECK_FAILED")
    if current != original:
        return finish("NOT_SENT", "ORIGINAL_LINK_CHANGED")

    def execute_once():
        receipt.update(connect_requests=1, mutation_attempted=True)
        initial_outcome, reason, interrupted = "UNKNOWN", "UNVERIFIED_REMOTE_REPLY", False
        try:
            reply = transport("smartcar", {"action": "connect", "ssid": profile["ssid"],
                              "security": profile["security"], "password": profile["password"],
                              "bssid": target}, _CONNECT_TIMEOUT)
            if isinstance(reply, dict) and reply.get("ok") is True and isinstance(reply.get("result"), dict):
                initial_outcome, reason = "ACKNOWLEDGED", "FRESH_STATUS_REQUIRED"
            elif (isinstance(reply, dict) and reply.get("ok") is False
                  and reply.get("activation_attempted") is False
                  and isinstance(reply.get("code"), str) and _CODE.fullmatch(reply["code"])):
                # Never echo arbitrary remote strings, even machine error codes.
                public_code = reply["code"] if reply["code"] in {
                    "AP_NOT_VISIBLE", "INTERFACE_UNSAFE"} else None
                if profile["password"] and public_code and profile["password"] in public_code:
                    public_code = None
                return finish("NOT_APPLIED", "REMOTE_REFUSED_BEFORE_ACTIVATION",
                              refusal_code=public_code)
        except KeyboardInterrupt:
            interrupted, reason = True, "INTERRUPTED_AFTER_INTENT"
        except Exception:
            reason = "TRANSPORT_OUTCOME_UNKNOWN"
        receipt["initial_outcome"] = initial_outcome
        if initial_outcome == "UNKNOWN" and not emit(
                "switch_unknown", **{**receipt, "outcome": "UNKNOWN", "reason": reason}):
            return finish("UNKNOWN", "JOURNAL_NOT_DURABLE",
                          instruction="Read status; do not resend the connect.")
        if interrupted:
            return finish("UNKNOWN", reason, instruction="Read status; do not resend the connect.")

        # This is the only recovery operation: three read-only status calls max.
        for attempt in range(_RECONCILE_READS):
            try:
                observed = _status(transport)
            except KeyboardInterrupt:
                raise
            except Exception:
                observed = None
            matched = observed == {"ssid": profile["ssid"], "bssid": target}
            if not emit("switch_status_verification", read=attempt + 1, target_observed=matched,
                        status_valid=observed is not None):
                return finish("UNKNOWN", "JOURNAL_NOT_DURABLE",
                              instruction="Read status; do not resend the connect.")
            if matched:
                return finish("APPLIED", "TARGET_OBSERVED_IN_FRESH_STATUS",
                              reconciled_by_status=initial_outcome == "UNKNOWN")
        return finish("UNKNOWN", "TARGET_NOT_VERIFIED",
                      instruction="Read status; do not resend the connect.")

    # Cover the complete post-intent region, including local parsing/reduction
    # and callbacks, rather than just the blocking transport call.
    try:
        if not emit("mutation_intent", action="connect", interface="wlan1",
                    connect_requests_planned=1, **receipt):
            return finish("NOT_SENT", "JOURNAL_NOT_DURABLE")
        return execute_once()
    except (Exception, KeyboardInterrupt):
        receipt["initial_outcome"] = "UNKNOWN"
        emit("switch_unknown", **{**receipt, "outcome": "UNKNOWN",
                                  "reason": "INTERRUPTED_OR_FAILED_AFTER_INTENT"})
        return finish("UNKNOWN", "INTERRUPTED_OR_FAILED_AFTER_INTENT",
                      instruction="Read status; do not resend the connect.")
