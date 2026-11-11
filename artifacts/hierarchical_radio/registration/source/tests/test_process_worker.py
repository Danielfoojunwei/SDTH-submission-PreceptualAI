"""Actual spawned processes; software fault fixtures, not robot evidence."""
import multiprocessing as mp
import os,signal,time
from pathlib import Path
from threading import Thread,Event
import numpy as np
import pytest
import wirelessjev.process_worker as process_worker
from wirelessjev.process_worker import ProcessProposalWorker


def echo(payload):return payload

def gated_reply(payload):
    Path(payload['entered']).write_text('entered')
    while not Path(payload['release']).exists():time.sleep(.003)
    return payload['value']

def behavior(payload):
    mode=payload.get('mode','echo')
    if mode=='hang':
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        if payload.get('ready_path'):Path(payload['ready_path']).write_text('entered')
        while True:time.sleep(.01)
    if mode=='crash':os._exit(23)
    if mode=='object':return object()
    if mode=='nan':return np.array([np.nan],dtype='f4')
    if mode=='large':return np.zeros(10000,dtype='f4')
    if mode=='raise':raise RuntimeError('private worker exception text')
    if mode=='sleep':time.sleep(payload['seconds'])
    return payload.get('value','ok')

class Counter:
    def __init__(self,scale):self.calls=0;self.scale=scale
    def __call__(self,payload):
        self.calls+=1;return {'calls':self.calls,'value':np.asarray(payload)*self.scale}

def counter_factory(scale=2.):return Counter(scale)
def failing_factory():raise ValueError('private factory detail')
def hanging_factory(config=None):
    while True:time.sleep(.01)

@pytest.fixture
def workers():
    owned=[]
    def create(target='echo',**kwargs):
        worker=ProcessProposalWorker(__name__+':'+target,**kwargs);owned.append(worker);return worker
    yield create
    for worker in owned:assert worker.close(),worker.status
    pids={w.status['last_pid'] for w in owned}
    assert not any(p.pid in pids for p in mp.active_children())

def reply(worker,seconds=3):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=worker.poll()
        if worker.disabled:return None
        if worker.events and worker.events[-1]['event']=='completed':return value
        time.sleep(.003)
    raise AssertionError(worker.status)

def reaped(worker,seconds=2):
    deadline=time.monotonic()+seconds
    while worker.pid is not None and time.monotonic()<deadline:worker.maintain();time.sleep(.003)
    assert worker.pid is None,worker.status

@pytest.fixture
def gated_response(workers,tmp_path):
    """Keep a real spawned response out of the parent until a fault is installed."""
    def create():
        worker=workers('gated_reply',timeout_s=1.,shutdown_grace_s=.03)
        entered=tmp_path/'entered';release=tmp_path/'release'
        assert worker.submit({'entered':str(entered),'release':str(release),'value':'proposal'})
        deadline=time.monotonic()+.5
        while not entered.exists() and time.monotonic()<deadline:time.sleep(.003)
        assert entered.exists(),worker.status
        return worker,release
    return create

@pytest.fixture
def delay_parent_decode(monkeypatch):
    def install(worker):
        unpack=process_worker._unpack;decoded=[]
        def delayed(value,budget,depth=0):
            result=unpack(value,budget,depth)
            if depth==0:
                decoded.append(True)
                time.sleep(max(0,worker._request[1]-time.monotonic())+.02)
            return result
        monkeypatch.setattr(process_worker,'_unpack',delayed)
        return decoded
    return install

def cache_response(worker):
    deadline=time.monotonic()+.5
    while not worker._has_result and not worker.disabled and time.monotonic()<deadline:
        worker.maintain();time.sleep(.003)
    assert worker._has_result and not worker.disabled,worker.status

def test_parent_decode_cannot_make_an_expired_reply_available(gated_response,delay_parent_decode):
    worker,release=gated_response();decoded=delay_parent_decode(worker)
    release.write_text('release')
    assert reply(worker) is None and decoded
    assert worker.disabled and worker.status['reason']=='deadline_exceeded'
    assert not worker._has_result and not worker.submit('stale advice')
    reaped(worker)

def test_cached_reply_expires_before_late_consumption(gated_response):
    worker,release=gated_response();release.write_text('release');cache_response(worker)
    assert not worker.submit('unread result still applies backpressure')
    time.sleep(max(0,worker._request[1]-time.monotonic())+.02)
    assert worker.poll() is None and worker.disabled
    assert worker.status['reason']=='deadline_exceeded' and not worker._has_result
    reaped(worker)

def test_poll_rechecks_deadline_after_supervision_before_exposure(gated_response,monkeypatch):
    worker,release=gated_response();release.write_text('release');cache_response(worker)
    maintain=worker.maintain
    def delayed_supervision():
        status=maintain()
        time.sleep(max(0,worker._request[1]-time.monotonic())+.02)
        return status
    monkeypatch.setattr(worker,'maintain',delayed_supervision)
    assert worker.poll() is None and worker.disabled
    assert worker.status['reason']=='deadline_exceeded' and not worker._has_result
    monkeypatch.setattr(worker,'maintain',maintain)
    reaped(worker)

def test_real_spawn_copies_inputs_outputs_and_preserves_one_worker(workers):
    worker=workers();assert worker.ready
    pid=worker.pid;values=np.arange(12,dtype='f4')[::2];expected=values.copy()
    assert worker.submit({'array':values,'flags':[True,None,3]});values[:]=999
    result=reply(worker);np.testing.assert_array_equal(result['array'],expected)
    assert result['flags']==[True,None,3];result['array'][:]=-8
    assert worker.submit({'array':expected})
    np.testing.assert_array_equal(reply(worker)['array'],expected)
    assert worker.pid==pid and worker.poll() is None

def test_factory_state_and_backpressure_are_explicit(workers):
    worker=workers('counter_factory',factory=True,factory_kwargs={'scale':3.})
    assert worker.ready and worker.submit(np.array([2.],dtype='f4'))
    assert not worker.submit(np.array([9.],dtype='f4'))
    result=reply(worker);assert result['calls']==1;np.testing.assert_array_equal(result['value'],[6.])
    assert worker.submit(np.array([4.],dtype='f4'));assert reply(worker)['calls']==2

def test_timeout_kills_and_reaps_before_explicit_restart(workers,tmp_path):
    worker=workers('behavior',timeout_s=.3,shutdown_grace_s=.03)
    marker=tmp_path/'entered';old=worker.pid
    assert worker.submit({'mode':'hang','ready_path':str(marker)})
    deadline=time.monotonic()+.25
    while not marker.exists() and time.monotonic()<deadline:worker.poll();time.sleep(.003)
    assert marker.exists()
    assert reply(worker) is None and worker.disabled and not worker.submit({'value':'later'})
    reaped(worker)
    assert worker.status['reason']=='deadline_exceeded' and worker.status['last_exitcode']==-signal.SIGKILL
    assert not any(p.pid==old for p in mp.active_children())
    assert worker.restart() and worker.pid!=old
    assert worker.submit({'value':'restarted'}) and reply(worker)=='restarted'

def test_crash_never_restarts_automatically(workers):
    worker=workers('behavior');pid=worker.pid;worker.submit({'mode':'crash'})
    assert reply(worker) is None and worker.disabled
    reaped(worker);assert worker.status['last_exitcode']==23
    for _ in range(5):assert not worker.submit({'value':'no automatic restart'})
    assert worker.status['last_pid']==pid

@pytest.mark.parametrize('mode',['object','nan','large','raise'])
def test_invalid_output_disables_child(workers,mode):
    worker=workers('behavior',max_output_bytes=512);worker.submit({'mode':mode})
    assert reply(worker) is None and worker.disabled
    assert worker.status['reason'] in ('worker_error','malformed_output','worker_exit')
    assert 'private worker exception text' not in str(worker.status);reaped(worker)

def test_invalid_inputs_do_not_dispatch_or_replace_worker(workers):
    worker=workers('counter_factory',factory=True,max_input_bytes=512);pid=worker.pid
    for value in (object(),np.array([np.inf]),'x'*1000,{'bad':set([1])}):assert not worker.submit(value)
    assert not worker.submit('x',timeout_s=np.nan)
    assert worker.ready and not worker.disabled and worker.pid==pid
    assert worker.submit(np.array([1.],dtype='f4'));assert reply(worker)['calls']==1

def test_close_is_bounded_for_stuck_callable_and_restart_is_explicit(workers,tmp_path):
    worker=workers('behavior',timeout_s=10.,shutdown_grace_s=.03)
    marker=tmp_path/'entered';assert worker.submit({'mode':'hang','ready_path':str(marker)})
    deadline=time.monotonic()+1
    while not marker.exists() and time.monotonic()<deadline:time.sleep(.003)
    assert marker.exists();start=time.monotonic();assert worker.close()
    assert time.monotonic()-start<.5 and worker.close() and worker.pid is None
    assert worker.poll() is None and not worker.submit('closed')
    assert worker.restart() and worker.submit({'value':'ok'}) and reply(worker)=='ok'

def test_startup_factory_fault_and_hang_are_reaped(workers):
    failed=workers('failing_factory',factory=True)
    assert failed.disabled and not failed.ready and failed.pid is None
    assert failed.status['error_type']=='ValueError'
    start=time.monotonic();hung=workers('hanging_factory',factory=True,factory_kwargs={'config':'x'*50000},startup_timeout_s=.5,shutdown_grace_s=.03)
    assert hung.disabled and hung.pid is None and time.monotonic()-start<2
    assert hung.status['reason']=='startup_timeout'

@pytest.mark.parametrize('option,value',[('timeout_s',np.nan),('timeout_s',0),('timeout_s',True),
    ('startup_timeout_s',np.inf),('shutdown_grace_s',0),('max_input_bytes',True),('max_output_bytes',1.5)])
def test_invalid_configuration_never_starts_child(option,value):
    before={p.pid for p in mp.active_children()}
    with pytest.raises(ValueError):ProcessProposalWorker(__name__+':echo',**{option:value})
    assert {p.pid for p in mp.active_children()}==before


def foreign_probe(worker,connection):
    connection.send({'submit':worker.submit('foreign'),'poll':worker.poll(),'close':worker.close(),
                     'restart':worker.restart(),'pid':worker.pid,'alive':worker.alive,'status':worker.status})
    connection.close()


@pytest.mark.skipif('fork' not in mp.get_all_start_methods(),reason='Fork ownership regression is POSIX-specific')
def test_inherited_handle_fails_before_lock_or_socket_access(workers):
    worker=workers();entered=Event();release=Event()
    def lock_owner():
        with worker._lock:entered.set();release.wait()
    blocker=Thread(target=lock_owner);blocker.start();assert entered.wait(1)
    receiving,sending=mp.get_context('fork').Pipe(duplex=False)
    child=mp.get_context('fork').Process(target=foreign_probe,args=(worker,sending));child.start();sending.close()
    try:
        assert receiving.poll(2),'foreign handle waited on an inherited lock'
        result=receiving.recv()
        assert result['submit'] is False and result['poll'] is None and result['close'] is False
        assert result['restart'] is False and result['pid'] is None and not result['alive']
        assert result['status']['reason']=='foreign_owner'
    finally:
        release.set();blocker.join(2);receiving.close();child.join(2)
        if child.is_alive():child.kill();child.join(2)
        child.close()
    assert worker.submit('parent unaffected') and reply(worker)=='parent unaffected'


def test_proposal_calls_do_not_wait_for_a_lifecycle_lock(workers):
    worker=workers();entered=Event();release=Event()
    def maintenance():
        with worker._lock:entered.set();release.wait()
    holder=Thread(target=maintenance);holder.start();assert entered.wait(1)
    try:
        start=time.monotonic()
        assert worker.poll() is None and not worker.submit('busy supervisor')
        assert time.monotonic()-start<.1
    finally:release.set();holder.join(2)
