"""Offline fault and concurrency checks; no network or hardware performance."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import pytest

from wirelessjev.execution import (
    DryRunTransport, Ledger, Transaction, build_connect, execute, reconcile,
)


OLD = '02:00:00:00:00:01'
NEW = '02:00:00:00:00:02'
OTHER = '02:00:00:00:00:03'


def transaction(name='t1', car='car-01', previous=OLD):
    return Transaction(name, car, previous, NEW, build_connect('fleet', NEW))


def connected(bssid=NEW, **changes):
    result = dict(connected=True, bssid=bssid, ssid='fleet',
                  interface='wlan1', role='WIFI')
    result.update(changes)
    return {'ok': True, 'result': result}


class ScriptedTransport:
    """An in-memory reply script for exercising the transport contract only."""
    def __init__(self, connect_reply=None, status_reply=None):
        self.connect_reply = connect_reply
        self.status_reply = status_reply
        self.sent = []
        self.lock = threading.Lock()

    def send(self, car, request):
        with self.lock:
            self.sent.append((car, dict(request)))
        reply = self.status_reply if request['action'] == 'status' else self.connect_reply
        if isinstance(reply, Exception):
            raise reply
        return reply


def uncertain(ledger=None, *, previous=OLD):
    ledger = ledger or Ledger({'d': 1.})
    txn = transaction(previous=previous)
    transport = ScriptedTransport(TimeoutError('ambiguous send'), connected())
    assert ledger.prepare(txn, 'd', .3)
    execute(txn, ledger, transport)
    assert txn.state == 'UNKNOWN'
    return ledger, txn, transport


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), True])
def test_nonphysical_budgets_and_airtime_cannot_create_reservations(value):
    with pytest.raises((TypeError, ValueError)):
        Ledger({'d': value})
    ledger = Ledger({'d': 1.})
    with pytest.raises((TypeError, ValueError)):
        ledger.prepare(transaction(), 'd', value)
    assert not ledger.in_flight() and ledger.reserved.get('d', 0.) == 0.


@pytest.mark.parametrize('value', [0, -1, True, 1.5, float('nan'), float('inf')])
def test_inflight_limit_requires_a_finite_positive_integer(value):
    with pytest.raises((TypeError, ValueError)):
        Ledger({'d': 1.}, max_in_flight=value)


def test_duplicate_identity_car_and_unknown_domain_leave_admission_unchanged():
    ledger = Ledger({'d': 1.}, max_in_flight=3)
    original = transaction()
    assert ledger.prepare(original, 'd', .3)
    assert not ledger.prepare(transaction('t1', 'car-02'), 'd', .4)
    assert not ledger.prepare(transaction('t2', 'car-01'), 'd', .4)
    assert not ledger.prepare(transaction('t3', 'car-03'), 'missing-domain', 0.)
    assert ledger.transactions['t1'] is original
    assert ledger.in_flight() == [original]
    assert ledger.reserved['d'] == pytest.approx(.3)


def test_concurrent_budget_admission_is_atomic():
    ledger = Ledger({'d': 1.}, max_in_flight=3)
    barrier = threading.Barrier(8)
    def prepare(index):
        barrier.wait()
        return ledger.prepare(transaction(f't{index}', f'car-{index}'), 'd', .3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(prepare, range(8)))
    assert sum(admitted) == 3
    assert len(ledger.in_flight()) == 3
    assert ledger.reserved['d'] == pytest.approx(.9)


def test_concurrent_duplicate_transaction_ids_cannot_overwrite_the_owner():
    ledger = Ledger({'d': 1.}, max_in_flight=8)
    barrier = threading.Barrier(8)
    candidates = [transaction('shared', f'car-{i}') for i in range(8)]
    def prepare(txn):
        barrier.wait()
        return ledger.prepare(txn, 'd', .1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(prepare, candidates))
    assert sum(admitted) == 1
    assert ledger.transactions['shared'] is candidates[admitted.index(True)]
    assert ledger.reserved['d'] == pytest.approx(.1)


def test_clean_failure_and_commit_release_once_without_consuming_other_reservations():
    ledger = Ledger({'d': 1.})
    aborted = transaction('failed', 'car-01')
    committed = transaction('succeeded', 'car-02')
    assert ledger.prepare(aborted, 'd', .3) and ledger.prepare(committed, 'd', .4)
    failure = ScriptedTransport({'ok': False, 'activation_attempted': False, 'code': 'AP_NOT_VISIBLE'})
    execute(aborted, ledger, failure)
    assert aborted.state == 'ABORTED'
    assert ledger.reserved['d'] == pytest.approx(.4)
    assert ledger.release(aborted) is False
    assert ledger.reserved['d'] == pytest.approx(.4)
    success = ScriptedTransport(connected())
    execute(committed, ledger, success)
    assert committed.state == 'COMMITTED' and committed.outcome == 'APPLIED'
    assert ledger.reserved['d'] == pytest.approx(0.)
    assert ledger.release(committed) is False
    execute(committed, ledger, success)
    assert len(success.sent) == 1


def test_prepared_and_unknown_reservations_cannot_be_released_early():
    ledger = Ledger({'d': 1.})
    txn = transaction()
    assert ledger.prepare(txn, 'd', .3)
    assert ledger.release(txn) is False
    transport = ScriptedTransport(TimeoutError('unknown'))
    execute(txn, ledger, transport)
    assert ledger.release(txn) is False
    assert ledger.reserved['d'] == pytest.approx(.3)


def test_preissue_cancellation_and_private_reservation_ownership_are_idempotent():
    ledger = Ledger({'d': 1.})
    cancelled = transaction('cancelled', 'car-01')
    other = transaction('other', 'car-02')
    assert ledger.prepare(cancelled, 'd', .3) and ledger.prepare(other, 'd', .4)
    # Mutable request bookkeeping must not be the authority for freeing airtime.
    cancelled.request['_airtime'] = 999.
    cancelled.request['_domain'] = 'another-domain'
    assert ledger.cancel(cancelled) is True
    assert cancelled.state == 'CANCELLED' and cancelled.outcome == 'NOT_APPLIED'
    assert ledger.reserved['d'] == pytest.approx(.4)
    assert ledger.cancel(cancelled) is False and ledger.release(cancelled) is False
    transport = ScriptedTransport(connected())
    assert execute(cancelled, ledger, transport) is cancelled and transport.sent == []


def test_only_exact_registered_prepared_transaction_can_issue_a_mutation():
    ledger = Ledger({'d': 1.})
    transport = ScriptedTransport(connected())
    txn = transaction()
    with pytest.raises(ValueError):
        execute(txn, ledger, transport)
    assert ledger.prepare(txn, 'd', .3)
    with pytest.raises(ValueError):
        execute(replace(txn), ledger, transport)
    assert transport.sent == []


@pytest.mark.parametrize('field,value', [('car_id', 'wrong-car'), ('to_bssid', OTHER),
                                         ('request', build_connect('other', NEW))])
def test_mutating_admitted_request_identity_does_not_change_the_wire_command(field, value):
    ledger = Ledger({'d': 1.})
    txn = transaction()
    assert ledger.prepare(txn, 'd', .3)
    setattr(txn, field, value)
    transport = ScriptedTransport(connected())
    with pytest.raises(ValueError):
        execute(txn, ledger, transport)
    assert transport.sent == [] and ledger.reserved['d'] == pytest.approx(.3)


def test_unknown_and_rewound_public_state_cannot_repeat_an_ambiguous_send():
    ledger, txn, transport = uncertain()
    execute(txn, ledger, transport)
    assert len(transport.sent) == 1
    txn.mark('PREPARED')
    try:
        execute(txn, ledger, transport)
    except ValueError:
        pass
    assert len(transport.sent) == 1
    assert ledger.reserved['d'] == pytest.approx(.3)


def test_two_concurrent_execution_calls_issue_exactly_one_request():
    entered, release = threading.Event(), threading.Event()
    class BlockingTransport(ScriptedTransport):
        def send(self, car, request):
            with self.lock:
                self.sent.append((car, dict(request)))
            entered.set()
            assert release.wait(2), 'test did not release the offline transport'
            return connected()
    ledger = Ledger({'d': 1.})
    txn = transaction()
    assert ledger.prepare(txn, 'd', .3)
    transport = BlockingTransport()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute, txn, ledger, transport)
        try:
            assert entered.wait(1)
            second = pool.submit(execute, txn, ledger, transport)
            assert second.result(timeout=1) is txn
            assert len(transport.sent) == 1
        finally:
            release.set()
        assert first.result(timeout=1) is txn
    assert txn.state == 'COMMITTED' and ledger.reserved['d'] == pytest.approx(0.)


@pytest.mark.parametrize('reply', [None, [], 'bad response', {'ok': 'true', 'result': {}},
    {'ok': True, 'result': None}, connected(connected=False), connected(OTHER),
    connected(ssid='wrong-ssid'), connected(interface='wlan0'), connected(role='HOTPOT'),
    connected(car_id='wrong-car')])
def test_malformed_or_mismatched_mutation_reply_becomes_unknown_not_issued(reply):
    ledger = Ledger({'d': 1.})
    txn = transaction()
    assert ledger.prepare(txn, 'd', .3)
    transport = ScriptedTransport(reply)
    assert execute(txn, ledger, transport) is txn
    assert txn.state == 'UNKNOWN' and txn.outcome == 'UNKNOWN'
    assert ledger.reserved['d'] == pytest.approx(.3)


@pytest.mark.parametrize('reply', [None, [], {'ok': False}, {'ok': False, 'result': connected()['result']},
    {'ok': True, 'result': {}}, connected(connected=False), connected(interface='wlan0'),
    connected(role='HOTPOT'), connected(bssid=None), connected('ap1-5g'),
    connected(OTHER), connected(car_id='wrong-car'), connected(ssid='wrong-ssid')])
def test_reconciliation_requires_valid_connected_identity_status(reply):
    ledger, txn, transport = uncertain()
    transport.status_reply = reply
    assert reconcile(txn, ledger, transport) is txn
    assert txn.state == 'UNKNOWN' and ledger.reserved['d'] == pytest.approx(.3)
    assert [r['action'] for _, r in transport.sent] == ['connect', 'status']


def test_missing_status_bssid_cannot_resolve_an_initially_disconnected_transaction():
    ledger, txn, transport = uncertain(previous=None)
    transport.status_reply = {'ok': False}
    reconcile(txn, ledger, transport)
    assert txn.state == 'UNKNOWN' and ledger.reserved['d'] == pytest.approx(.3)


@pytest.mark.parametrize('bssid,state,outcome', [(NEW, 'COMMITTED', 'APPLIED'),
                                               (OLD, 'ABORTED', 'NOT_APPLIED')])
def test_fresh_reconciliation_settles_and_releases_either_terminal_outcome_once(bssid, state, outcome):
    ledger, txn, transport = uncertain()
    transport.status_reply = connected(bssid)
    reconcile(txn, ledger, transport)
    assert txn.state == state and txn.outcome == outcome
    assert ledger.reserved['d'] == pytest.approx(0.)
    calls = len(transport.sent)
    reconcile(txn, ledger, transport)
    execute(txn, ledger, transport)
    assert len(transport.sent) == calls
    assert ledger.release(txn) is False


def test_status_freshness_uses_the_query_and_optional_observation_clock():
    ledger, txn, transport = uncertain()
    now = time.monotonic()
    transport.status_reply = connected(observed_at_monotonic=now - 20.)
    reconcile(txn, ledger, transport, clock=lambda: now, max_status_age_s=5.)
    assert txn.state == 'UNKNOWN'
    transport.status_reply = connected(observed_at_monotonic=now + 1.)
    reconcile(txn, ledger, transport, clock=lambda: now, max_status_age_s=5.)
    assert txn.state == 'UNKNOWN'
    transport.status_reply = connected(observed_at_monotonic=float('nan'))
    reconcile(txn, ledger, transport, clock=lambda: now, max_status_age_s=5.)
    assert txn.state == 'UNKNOWN'
    transport.status_reply = connected(observed_at_monotonic=now)
    reconcile(txn, ledger, transport, clock=lambda: now, max_status_age_s=5.)
    assert txn.state == 'COMMITTED'


def test_status_arriving_after_its_query_budget_does_not_release_unknown():
    ledger, txn, _ = uncertain()
    current = [time.monotonic()]
    class DelayedStatus(ScriptedTransport):
        def send(self, car, request):
            current[0] += 6.
            return connected()
    reconcile(txn, ledger, DelayedStatus(), clock=lambda: current[0], max_status_age_s=5.)
    assert txn.state == 'UNKNOWN' and ledger.reserved['d'] == pytest.approx(.3)


def test_dry_run_symbolic_bssids_still_reconcile_without_hardware_authorization():
    ledger = Ledger({'d': 1.})
    txn = Transaction('symbolic', 'car-01', 'ap1-5g', 'ap2-5g', build_connect('fleet', 'ap2-5g'))
    transport = DryRunTransport({'car-01': 'ap1-5g'}, unknown_rate=1., seed=1)
    assert ledger.prepare(txn, 'd', .3)
    execute(txn, ledger, transport)
    reconcile(txn, ledger, transport)
    assert txn.state in ('COMMITTED', 'ABORTED')
    assert ledger.reserved['d'] == pytest.approx(0.)
    assert sum(req['action'] == 'connect' for _, req in transport.sent) == 1


def test_zero_budget_does_not_gain_capacity_from_a_tolerance():
    ledger=Ledger({'d':0.})
    assert not ledger.prepare(transaction(),'d',1e-12)
    assert ledger.prepare(transaction(),'d',0.)


@pytest.mark.parametrize('invalid',['00:00:00:00:00:00','ff:ff:ff:ff:ff:ff','01:00:00:00:00:01'])
def test_non_ap_mac_status_does_not_resolve_unknown(invalid):
    ledger,txn,transport=uncertain()
    transport.status_reply=connected(invalid)
    reconcile(txn,ledger,transport)
    assert txn.state=='UNKNOWN' and ledger.reserved['d']==pytest.approx(.3)


def test_operator_interrupt_after_issue_preserves_unknown_and_propagates_shutdown():
    class InterruptedTransport(ScriptedTransport):
        def send(self,car,request):
            self.sent.append((car,dict(request)))
            raise KeyboardInterrupt('offline interrupt injection')
    ledger=Ledger({'d':1.});txn=transaction();transport=InterruptedTransport()
    assert ledger.prepare(txn,'d',.3)
    with pytest.raises(KeyboardInterrupt):execute(txn,ledger,transport)
    assert txn.state=='UNKNOWN' and ledger.reserved['d']==pytest.approx(.3)
    assert execute(txn,ledger,transport) is txn and len(transport.sent)==1
