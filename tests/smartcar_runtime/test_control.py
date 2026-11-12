"""Synthetic control-boundary checks, with no car, SSH or network access."""
from copy import deepcopy
import json

import pytest

from wirelessjev.smartcar_runtime.control import supervised_switch

OLD, NEW = "02:00:00:00:00:01", "02:00:00:00:00:02"
PASSWORD = "unit-only-private-passphrase"
PROFILE = {"ssid": "unit-fleet", "security": "wpa-psk", "password": PASSWORD, "enabled": True}


def status(bssid=OLD, **changes):
    result = {"connected": True, "interface": "wlan1", "role": "WIFI",
              "ssid": PROFILE["ssid"], "bssid": bssid}
    result.update(changes)
    return {"ok": True, "result": result}


def doctor(**changes):
    role = {"safe": True, "management_guard": True, "interface": "wlan1", "role": "WIFI",
            "management_interface": "wlan0", "reasons": []}
    role.update(changes)
    return {"ok": True, "action": "doctor", "doctor": {"role_check": role}}


class Boundary:
    def __init__(self, reply=None, error=None, after=None):
        self.reply = {"ok": True, "result": {}} if reply is None else reply
        self.error, self.after = error, after
        self.calls, self.events, self.trace = [], [], []
        self.doctor_replies, self.status_replies = [], []

    def journal(self, event, **fields):
        self.trace.append(("journal", event))
        self.events.append({"event": event, **deepcopy(fields)})

    def transport(self, key, payload, timeout):
        self.calls.append((key, deepcopy(payload), timeout))
        action = payload["action"]
        self.trace.append(("transport", action))
        if key == "agent":
            assert payload == {"action": "doctor"}
            return self.doctor_replies.pop(0) if self.doctor_replies else doctor()
        assert key == "smartcar"
        if action == "status":
            if self.status_replies:
                value = self.status_replies.pop(0)
            else:
                value = self.after if self.connects else status()
            if isinstance(value, BaseException):
                raise value
            return deepcopy(value)
        assert action == "connect"
        if self.error:
            raise self.error
        return deepcopy(self.reply)

    @property
    def connects(self):
        return [payload for _, payload, _ in self.calls if payload["action"] == "connect"]

    def run(self, **kwargs):
        return supervised_switch(self.transport, self.journal, profiledict=PROFILE,
                                 target_bssid=NEW, confirm=lambda _: NEW, **kwargs)


def test_intent_precedes_single_connect_and_fresh_status_required():
    boundary = Boundary(after=status(NEW))
    receipt = boundary.run()
    assert receipt["outcome"] == "APPLIED" and len(boundary.connects) == 1
    assert receipt["from_bssid"] == OLD and receipt["from_ssid"] == PROFILE["ssid"]
    assert receipt["return_requires_separate_confirmation"] is True
    assert boundary.trace.index(("journal", "mutation_intent")) < boundary.trace.index(("transport", "connect"))
    assert [payload["action"] for _, payload, _ in boundary.calls] == [
        "doctor", "status", "doctor", "status", "connect", "status"]
    assert PASSWORD not in json.dumps([receipt, boundary.events])
    assert boundary.connects[0]["password"] == PASSWORD


@pytest.mark.parametrize("change", [
    {"safe": False}, {"safe": 1}, {"management_guard": False}, {"management_guard": 1},
    {"interface": "wlan0"}, {"role": "HOTPOT"}, {"management_interface": "wlan1"}])
def test_unsafe_or_ambiguous_roles_block_before_confirmation(change):
    boundary = Boundary()
    boundary.doctor_replies = [doctor(**change)]
    def unexpected(_):
        pytest.fail("must not request confirmation for unsafe roles")
    receipt = supervised_switch(boundary.transport, boundary.journal, profiledict=PROFILE,
                                target_bssid=NEW, confirm=unexpected)
    assert receipt["outcome"] == "NOT_SENT" and not boundary.connects


@pytest.mark.parametrize("reply", [{}, {"ok": False}, {"ok": True, "doctor": {}},
    {"ok": True, "action": "doctor", "doctor": {"role_check": None}}])
def test_missing_doctor_attestation_blocks(reply):
    boundary = Boundary()
    boundary.doctor_replies = [reply]
    assert boundary.run()["outcome"] == "NOT_SENT" and not boundary.connects


@pytest.mark.parametrize("change", [{"connected": False}, {"connected": 1}, {"role": "HOTPOT"},
    {"interface": "wlan0"}, {"ssid": "other"}, {"bssid": None}])
def test_unknown_disconnected_or_different_ssid_original_blocks(change):
    boundary = Boundary()
    boundary.status_replies = [status(**change)]
    assert boundary.run()["outcome"] == "NOT_SENT" and not boundary.connects


def test_mismatched_confirmation_cannot_send():
    boundary = Boundary()
    receipt = supervised_switch(boundary.transport, boundary.journal, profiledict=PROFILE,
                                target_bssid=NEW, confirm=lambda _: OLD)
    assert receipt["outcome"] == "NOT_SENT" and not boundary.connects
    assert not any(event["event"] == "mutation_intent" for event in boundary.events)


def test_status_and_roles_are_rechecked_after_human_delay():
    boundary = Boundary()
    boundary.status_replies = [status(), status("02:00:00:00:00:03")]
    receipt = boundary.run()
    assert receipt["reason"] == "ORIGINAL_LINK_CHANGED" and not boundary.connects
    boundary = Boundary()
    boundary.doctor_replies = [doctor(), doctor(safe=False)]
    assert boundary.run()["outcome"] == "NOT_SENT" and not boundary.connects


def test_failing_durable_intent_prevents_the_connect():
    boundary = Boundary()
    def failing_journal(event, **fields):
        if event == "mutation_intent":
            raise OSError("synthetic fsync failure")
        boundary.journal(event, **fields)
    receipt = supervised_switch(boundary.transport, failing_journal, profiledict=PROFILE,
                                target_bssid=NEW, confirm=lambda _: NEW)
    assert receipt["outcome"] == "NOT_SENT" and receipt["journal_complete"] is False
    assert not boundary.connects


@pytest.mark.parametrize("error", [OSError(PASSWORD), ValueError(PASSWORD), ConnectionError(PASSWORD)])
def test_partial_write_or_malformed_reply_is_unknown_without_retry(error):
    boundary = Boundary(error=error)
    receipt = boundary.run()
    assert receipt["outcome"] == "UNKNOWN" and receipt["initial_outcome"] == "UNKNOWN"
    assert len(boundary.connects) == 1
    assert sum(payload["action"] == "status" for _, payload, _ in boundary.calls) == 5
    assert any(event["event"] == "switch_unknown" for event in boundary.events)
    assert PASSWORD not in json.dumps([receipt, boundary.events])


def test_status_can_reconcile_a_lost_ack_without_a_second_connect():
    boundary = Boundary(error=OSError("synthetic interrupted write"), after=status(NEW))
    receipt = boundary.run()
    assert receipt["outcome"] == "APPLIED" and receipt["reconciled_by_status"] is True
    assert len(boundary.connects) == 1
    assert any(event["event"] == "switch_unknown" for event in boundary.events)


def test_ctrl_c_after_intent_is_unknown_and_stops_without_further_reads():
    boundary = Boundary(error=KeyboardInterrupt())
    receipt = boundary.run()
    assert receipt["outcome"] == "UNKNOWN" and receipt["reason"] == "INTERRUPTED_AFTER_INTENT"
    assert len(boundary.connects) == 1 and boundary.calls[-1][1]["action"] == "connect"
    assert any(event["event"] == "switch_unknown" for event in boundary.events)
    assert "do not resend" in receipt["instruction"]


@pytest.mark.parametrize("code", ["AP_NOT_VISIBLE", "INTERFACE_UNSAFE"])
def test_explicit_pre_activation_refusal_is_not_applied(code):
    boundary = Boundary(reply={"ok": False, "activation_attempted": False, "code": code,
                               "error": PASSWORD})
    receipt = boundary.run()
    assert receipt["outcome"] == "NOT_APPLIED" and receipt["refusal_code"] == code
    assert len(boundary.connects) == 1 and boundary.calls[-1][1]["action"] == "connect"
    assert PASSWORD not in json.dumps([receipt, boundary.events])


@pytest.mark.parametrize("reply", [[], {}, {"ok": "true"}, {"ok": True, "result": None},
    {"ok": False, "code": "FAILED"}, {"ok": False, "activation_attempted": True, "code": "FAILED"},
    {"ok": False, "activation_attempted": 0, "code": "FAILED"}])
def test_ambiguous_remote_result_requires_status_and_cannot_become_not_applied(reply):
    boundary = Boundary(reply=reply)
    receipt = boundary.run()
    assert receipt["outcome"] == "UNKNOWN" and len(boundary.connects) == 1


def test_success_reply_without_verified_target_remains_unknown():
    boundary = Boundary(after=status())
    receipt = boundary.run()
    assert receipt["outcome"] == "UNKNOWN" and receipt["initial_outcome"] == "ACKNOWLEDGED"
    assert len(boundary.connects) == 1


@pytest.mark.parametrize("change", [{"role": "HOTPOT"}, {"interface": "wlan0"},
                                    {"ssid": "different"}, {"connected": False}])
def test_verification_rejects_wrong_interface_role_ssid_or_disconnected(change):
    boundary = Boundary(after=status(NEW, **change))
    assert boundary.run()["outcome"] == "UNKNOWN" and len(boundary.connects) == 1


@pytest.mark.parametrize("profile", [None, {}, {k: v for k, v in PROFILE.items() if k != "enabled"},
    {**PROFILE, "enabled": False}, {**PROFILE, "enabled": 1},
    {**PROFILE, "security": "unknown"}, {**PROFILE, "password": None},
    {**PROFILE, "ssid": "line\nbreak"}, {**PROFILE, "ssid": "\ud800"}, {**PROFILE, "ssid": PASSWORD}])
def test_invalid_disabled_or_unsafe_public_profile_makes_no_remote_call(profile):
    boundary = Boundary()
    receipt = supervised_switch(boundary.transport, boundary.journal, profiledict=profile,
                                target_bssid=NEW, confirm=lambda _: NEW)
    assert receipt["outcome"] == "NOT_SENT" and not boundary.calls
    assert PASSWORD not in json.dumps([receipt, boundary.events])


@pytest.mark.parametrize("target", ["any", OLD, "ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00", NEW + "; reboot"])
def test_target_must_be_a_different_single_unicast_bssid(target):
    boundary = Boundary()
    receipt = supervised_switch(boundary.transport, boundary.journal, profiledict=PROFILE,
                                target_bssid=target, confirm=lambda _: target)
    assert receipt["outcome"] == "NOT_SENT" and not boundary.connects


def test_return_to_original_is_a_separate_confirmed_request():
    outward = Boundary(after=status(NEW))
    receipt = outward.run()
    assert receipt["outcome"] == "APPLIED" and len(outward.connects) == 1
    returning = Boundary(after=status(OLD))
    returning.status_replies = [status(NEW), status(NEW)]
    result = supervised_switch(returning.transport, returning.journal, profiledict=PROFILE,
                               target_bssid=receipt["from_bssid"], confirm=lambda _: OLD)
    assert result["outcome"] == "APPLIED" and len(returning.connects) == 1
    assert returning.connects[0]["bssid"] == OLD


@pytest.mark.parametrize("event", ["switch_status_verification", "switch_result"])
@pytest.mark.parametrize("error", [OSError("synthetic fsync failure"), KeyboardInterrupt()])
def test_post_send_journal_failure_cannot_report_success(event, error):
    boundary = Boundary(after=status(NEW))
    def interrupted_journal(name, **fields):
        if name == event:
            raise error
        boundary.journal(name, **fields)
    receipt = supervised_switch(boundary.transport, interrupted_journal, profiledict=PROFILE,
                                target_bssid=NEW, confirm=lambda _: NEW)
    assert receipt["outcome"] == "UNKNOWN" and receipt["journal_complete"] is False
    assert len(boundary.connects) == 1


def test_interrupt_outside_transport_after_intent_is_still_unknown(monkeypatch):
    from wirelessjev.smartcar_runtime import control
    boundary = Boundary(after=status(NEW))
    read_status = control._status
    class InterruptComparison:
        def __eq__(self, other):
            raise KeyboardInterrupt()
    def status_then_interrupt(transport):
        value = read_status(transport)
        return InterruptComparison() if boundary.connects else value
    monkeypatch.setattr(control, "_status", status_then_interrupt)
    receipt = boundary.run()
    assert receipt["outcome"] == "UNKNOWN"
    assert receipt["reason"] == "INTERRUPTED_OR_FAILED_AFTER_INTENT"
    assert any(event["event"] == "switch_unknown" for event in boundary.events)
    assert len(boundary.connects) == 1


def test_remote_machine_code_cannot_echo_a_credential():
    boundary = Boundary(reply={"ok": False, "activation_attempted": False,
                               "code": "UNIT_PRIVATE_PASSWORD"})
    profile = {**PROFILE, "password": "UNIT_PRIVATE_PASSWORD"}
    receipt = supervised_switch(boundary.transport, boundary.journal, profiledict=profile,
                                target_bssid=NEW, confirm=lambda _: NEW)
    assert receipt["outcome"] == "NOT_APPLIED" and receipt["refusal_code"] is None
    assert profile["password"] not in json.dumps([receipt, boundary.events])


def test_unknown_journal_event_explicitly_records_unknown_outcome():
    boundary = Boundary(error=OSError("synthetic send error"))
    boundary.run()
    events = [event for event in boundary.events if event["event"] == "switch_unknown"]
    assert len(events) == 1 and events[0]["outcome"] == "UNKNOWN"
