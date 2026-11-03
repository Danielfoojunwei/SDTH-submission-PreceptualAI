"""Real SQLite/restart/process-crash checks; transports never access hardware."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest
from wirelessjev.execution import Transaction, build_connect
from wirelessjev.execution_journal import DurableExecution, JournalError, JournalOwnershipError

OLD='02:00:00:00:00:01'; NEW='02:00:00:00:00:02'
ROOT=Path(__file__).resolve().parents[1]


def transaction(identity='t',car='car'):
    return Transaction(identity,car,OLD,NEW,build_connect('fleet',NEW))


class OfflineTransport:
    def __init__(self,unknown=False): self.requests=[]; self.unknown=unknown
    def send(self,car,request):
        self.requests.append((car,request.copy()))
        if request['action']=='connect' and self.unknown: raise TimeoutError('offline ambiguous acknowledgement')
        return {'ok':True,'result':{'connected':True,'bssid':NEW,'ssid':'fleet','interface':'wlan1','role':'WIFI'}}


def child(code,*arguments):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    return subprocess.run([sys.executable,'-c',code,*map(str,arguments)],cwd=ROOT,env=env,
                          capture_output=True,text=True,timeout=15)


def test_full_durability_duplicate_issue_and_terminal_ids_survive_reopen(tmp_path):
    path=tmp_path/'journal.sqlite'; transport=OfflineTransport()
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal._conn.execute('PRAGMA journal_mode').fetchone()[0]=='wal'
        assert journal._conn.execute('PRAGMA synchronous').fetchone()[0]==2
        assert journal.prepare(transaction(),'d',.3)
        assert journal.execute('t',transport).state=='COMMITTED'
        assert journal.reserved['d']==0.
        journal.execute('t',transport)
        assert len(transport.requests)==1
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal.get('t').state=='COMMITTED' and journal.in_flight()==()
        assert not journal.prepare(transaction(),'d',.3)
        journal.execute('t',transport)
        assert len(transport.requests)==1


def test_reopened_prepared_work_is_cancelled_not_resumed_without_authorization(tmp_path):
    path=tmp_path/'journal.sqlite'
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal.prepare(transaction(),'d',.3)
    with DurableExecution(path,{'d':1.}) as journal:
        transport=OfflineTransport()
        assert journal.get('t').state=='CANCELLED' and journal.reserved['d']==0.
        journal.execute('t',transport)
        assert transport.requests==[] and not journal.prepare(transaction(),'d',.3)
        assert journal.prepare(transaction('fresh-authorization'),'d',.3)


CRASH = r'''
import os,sqlite3,sys
from wirelessjev.execution import Transaction,build_connect
from wirelessjev.execution_journal import DurableExecution
path,marker,stage=sys.argv[1:]
j=DurableExecution(path,{'d':1.})
assert j.prepare(Transaction('t','car','02:00:00:00:00:01','02:00:00:00:00:02',
                            build_connect('fleet','02:00:00:00:00:02')),'d',.3)
class CrashTransport:
 def send(self,car,request):
  with sqlite3.connect(path) as check:
   assert check.execute("SELECT state FROM transactions WHERE txn_id='t'").fetchone()[0]=='ISSUED'
  if stage in ('after_send','after_reply'):
   with open(marker,'wb') as output:
    output.write(b'one offline mutation analogue\n');output.flush();os.fsync(output.fileno())
  if stage=='after_reply':
   return {'ok':True,'result':{'connected':True,'bssid':'02:00:00:00:00:02','ssid':'fleet'}}
  os._exit(77)
original=j._persist_state
def persist(identity,state,outcome):
 if stage=='after_reply' and state=='COMMITTED':os._exit(77)
 return original(identity,state,outcome)
j._persist_state=persist
j.execute('t',CrashTransport())
'''


@pytest.mark.parametrize('stage',['before_send','after_send','after_reply'])
def test_real_process_crash_restores_unknown_reservation_without_resend(tmp_path,stage):
    path=tmp_path/'journal.sqlite'; marker=tmp_path/'sent'
    process=child(CRASH,path,marker,stage)
    assert process.returncode==77,process.stderr
    assert marker.exists()==(stage!='before_send')
    transport=OfflineTransport()
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal.get('t').state=='UNKNOWN' and journal.reserved['d']==pytest.approx(.3)
        assert journal.get('t').history==[]  # no fabricated issue timestamps during reconstruction
        assert len(journal.in_flight())==1
        journal.execute('t',transport)
        assert transport.requests==[]
        assert not journal.prepare(transaction('blocked','car'),'d',.3)
        assert journal.reconcile('t',transport).state=='COMMITTED'
        assert [request['action'] for _,request in transport.requests]==['status']
        assert journal.reserved['d']==0.
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal.get('t').state=='COMMITTED' and journal.in_flight()==()


@pytest.mark.parametrize('during_reconcile',[False,True])
def test_terminal_persistence_failure_poisoned_owner_recovers_ambiguity(tmp_path,during_reconcile):
    path=tmp_path/'journal.sqlite'; journal=DurableExecution(path,{'d':1.})
    assert journal.prepare(transaction(),'d',.3)
    if during_reconcile:journal.execute('t',OfflineTransport(unknown=True))
    original=journal._persist_state
    def unavailable(identity,state,outcome):
        if state=='COMMITTED':raise OSError('injected durable commit failure')
        return original(identity,state,outcome)
    journal._persist_state=unavailable
    try:
        with pytest.raises(JournalError):
            (journal.reconcile if during_reconcile else journal.execute)('t',OfflineTransport())
        assert journal.failed
        with pytest.raises(JournalError):journal.prepare(transaction('new','car-new'),'d',.1)
    finally:journal.close()
    transport=OfflineTransport()
    with DurableExecution(path,{'d':1.}) as reopened:
        assert reopened.get('t').state=='UNKNOWN' and reopened.reserved['d']==pytest.approx(.3)
        reopened.execute('t',transport);assert transport.requests==[]
        reopened.reconcile('t',transport)
    with DurableExecution(path,{'d':1.}) as reopened:
        assert reopened.get('t').state=='COMMITTED' and reopened.reserved['d']==0.


def test_failed_issue_intent_commit_sends_nothing_and_old_prepared_is_cancelled(tmp_path):
    path=tmp_path/'journal.sqlite'; transport=OfflineTransport()
    journal=DurableExecution(path,{'d':1.});assert journal.prepare(transaction(),'d',.3)
    journal._conn.execute('PRAGMA query_only=ON')
    try:
        with pytest.raises(JournalError):journal.execute('t',transport)
        assert journal.failed and transport.requests==[]
        with pytest.raises(JournalError):journal.execute('t',transport)
    finally:journal.close()
    with DurableExecution(path,{'d':1.}) as reopened:
        assert reopened.get('t').state=='CANCELLED'
        reopened.execute('t',transport);assert transport.requests==[]


def test_fixed_capacity_retains_ids_instead_of_forgetting_duplicate_history(tmp_path):
    path=tmp_path/'journal.sqlite'
    with DurableExecution(path,{'d':1.},max_records=1) as journal:
        assert journal.prepare(transaction(),'d',.3) and journal.cancel('t')
        assert not journal.prepare(transaction(),'d',.3)
        assert not journal.prepare(transaction('second'),'d',.3)
    with DurableExecution(path,{'d':1.},max_records=1) as journal:
        assert not journal.prepare(transaction('second'),'d',.3)


@pytest.mark.parametrize('mode',['garbage','empty','directory','payload'])
def test_corrupt_or_unavailable_database_fails_closed(tmp_path,mode):
    path=tmp_path/'journal.sqlite'
    if mode=='garbage':path.write_bytes(b'not a SQLite database')
    elif mode=='empty':path.touch()
    elif mode=='directory':path.mkdir()
    else:
        with DurableExecution(path,{'d':1.}) as journal:assert journal.prepare(transaction(),'d',.3)
        with sqlite3.connect(path) as connection:connection.execute("UPDATE transactions SET payload='{}'")
    with pytest.raises(JournalError):DurableExecution(path,{'d':1.})
    if mode!='directory':
        with pytest.raises(JournalError):DurableExecution(path,{'d':1.})


def test_exclusive_ownership_same_process_and_real_other_process(tmp_path):
    path=tmp_path/'journal.sqlite'
    with DurableExecution(path,{'d':1.}) as journal:
        with pytest.raises(JournalOwnershipError):DurableExecution(path,{'d':1.})
        code='''
import sys
from wirelessjev.execution_journal import DurableExecution,JournalOwnershipError
try:j=DurableExecution(sys.argv[1],{'d':1.})
except JournalOwnershipError:sys.exit(0)
else:j.close();sys.exit(9)
'''
        process=child(code,path);assert process.returncode==0,process.stderr
        assert journal.prepare(transaction(),'d',.3)
    with DurableExecution(path,{'d':1.}) as reopened:assert reopened.get('t').state=='CANCELLED'


def test_immutable_configuration_and_credential_free_request_schema(tmp_path):
    path=tmp_path/'journal.sqlite'
    with DurableExecution(path,{'d':1.}) as journal:
        for field in ('password','psk','arbitrary_field','_hidden_secret'):
            txn=transaction(field);txn.request[field]='not-a-real-credential'
            with pytest.raises(ValueError):journal.prepare(txn,'d',.3)
        assert journal.in_flight()==()
    for config in ({'budgets':{'d':.5}},{'budgets':{'d':1.},'max_in_flight':4},
                   {'budgets':{'d':1.},'max_records':9}):
        with pytest.raises(JournalError):DurableExecution(path,**config)
    with DurableExecution(path,{'d':1.}) as journal:assert journal.in_flight()==()


def test_same_owner_concurrent_duplicate_execution_sends_once(tmp_path):
    path=tmp_path/'journal.sqlite';transport=OfflineTransport();barrier=threading.Barrier(2)
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal.prepare(transaction(),'d',.3)
        def issue():barrier.wait();return journal.execute('t',transport).state
        with ThreadPoolExecutor(max_workers=2) as workers:
            results=list(workers.map(lambda _:issue(),range(2)))
        assert results==['COMMITTED','COMMITTED'] and len(transport.requests)==1


def test_recovery_reconstructs_multiple_unresolved_reservations_and_inflight_limit(tmp_path):
    path=tmp_path/'journal.sqlite'
    with DurableExecution(path,{'d':1.},max_in_flight=2) as journal:
        for identity,car in [('one','car-1'),('two','car-2')]:
            assert journal.prepare(transaction(identity,car),'d',.4)
            journal.execute(identity,OfflineTransport(unknown=True))
    with DurableExecution(path,{'d':1.},max_in_flight=2) as journal:
        assert len(journal.in_flight())==2 and journal.reserved['d']==pytest.approx(.8)
        assert not journal.prepare(transaction('three','car-3'),'d',.1)
        journal.reconcile('one',OfflineTransport())
        assert journal.reserved['d']==pytest.approx(.4)
        assert journal.prepare(transaction('three','car-3'),'d',.1)


def test_inherited_owner_rejects_before_an_inherited_locked_mutex(tmp_path):
    code=r'''
import os,signal,sys,threading
from wirelessjev.execution import Transaction,build_connect
from wirelessjev.execution_journal import DurableExecution,JournalOwnershipError
j=DurableExecution(sys.argv[1],{'d':1.})
assert j.prepare(Transaction('t','car','02:00:00:00:00:01','02:00:00:00:00:02',
                            build_connect('fleet','02:00:00:00:00:02')),'d',.3)
entered=threading.Event();release=threading.Event()
class Blocking:
 def send(self,car,request):
  entered.set();assert release.wait(8)
  return {'ok':True,'result':{'connected':True,'bssid':'02:00:00:00:00:02','ssid':'fleet'}}
t=threading.Thread(target=lambda:j.execute('t',Blocking()));t.start();assert entered.wait(5)
pid=os.fork()
if pid==0:
 signal.alarm(3)
 calls=[lambda:j.get('t'),lambda:j.reserved,lambda:j.in_flight(),lambda:j.execute('t',Blocking()),j.close]
 for call in calls:
  try:call()
  except JournalOwnershipError:pass
  else:os._exit(8)
 os._exit(0)
_,status=os.waitpid(pid,0)
release.set();t.join(5);assert not t.is_alive()
assert os.waitstatus_to_exitcode(status)==0
assert j.get('t').state=='COMMITTED';j.close()
'''
    process=child(code,tmp_path/'journal.sqlite')
    assert process.returncode==0,process.stderr


@pytest.mark.parametrize('during_reconcile',[False,True])
def test_persistence_uses_private_outcome_not_mutable_public_transaction(tmp_path,monkeypatch,during_reconcile):
    import wirelessjev.execution_journal as module
    path=tmp_path/'journal.sqlite'
    with DurableExecution(path,{'d':1.}) as journal:
        assert journal.prepare(transaction(),'d',.3)
        if during_reconcile:journal.execute('t',OfflineTransport(unknown=True))
        operation='reconcile' if during_reconcile else 'execute'; original=getattr(module.core,operation)
        def changed(*args,**kwargs):
            result=original(*args,**kwargs)
            result.state='ABORTED';result.outcome='NOT_APPLIED'  # injected external-reference mutation
            return result
        monkeypatch.setattr(module.core,operation,changed)
        getattr(journal,operation)('t',OfflineTransport())
        assert journal._conn.execute("SELECT state,outcome FROM transactions WHERE txn_id='t'").fetchone()[:]==('COMMITTED','APPLIED')
    monkeypatch.undo()
    with DurableExecution(path,{'d':1.}) as journal:assert journal.get('t').state=='COMMITTED'


def test_hard_link_alias_cannot_evade_exclusive_journal_ownership(tmp_path):
    path=tmp_path/'journal.sqlite';alias=tmp_path/'alias.sqlite'
    with DurableExecution(path,{'d':1.}):
        os.link(path,alias)
        try:
            with pytest.raises(JournalOwnershipError):DurableExecution(alias,{'d':1.})
        finally:alias.unlink()
