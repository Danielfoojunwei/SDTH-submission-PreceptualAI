"""Opt-in spawned proposal worker: bounded passive IPC, deadlines and reaping.

Configured imports are trusted code. Process fault containment is not a
security sandbox or qualification of any physical control deadline.
"""
from __future__ import annotations
import base64,importlib,json,math,multiprocessing as mp,os,re,select,socket,struct,time
from collections import deque
from numbers import Real
from threading import RLock,Event
import numpy as np

_DTYPES={np.dtype(x).str for x in ('bool','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float16','float32','float64')}
_MAX_BYTES=16*1024*1024

class PayloadError(ValueError):pass

class _Budget:
    def __init__(self,limit):self.left=limit;self.nodes=0
    def charge(self,size,depth):
        self.left-=size;self.nodes+=1
        if self.left<0 or self.nodes>100000 or depth>16:raise PayloadError('Payload exceeds structural/byte limit')

def _pack(value,budget,depth=0):
    budget.charge(64,depth)
    if isinstance(value,np.generic):value=value.item()
    if value is None or type(value) is bool:return value
    if type(value) is int:
        if not -(2**63)<=value<2**64:raise PayloadError('Integer exceeds supported range')
        return value
    if type(value) is float:
        if not math.isfinite(value):raise PayloadError('Nonfinite numeric payload')
        return value
    if type(value) is str:
        if len(value)>budget.left:raise PayloadError('String exceeds payload limit')
        budget.charge(len(value.encode('utf-8'))*6,depth);return value
    if type(value) is np.ndarray:
        if value.dtype.str not in _DTYPES or value.ndim>8:raise PayloadError('Unsupported array type')
        budget.charge(4*((value.nbytes+2)//3)+16*value.ndim,depth)
        if not np.isfinite(value).all():raise PayloadError('Nonfinite array payload')
        return {'t':'array','dtype':value.dtype.str,'shape':list(value.shape),
                'data':base64.b64encode(value.tobytes(order='C')).decode('ascii')}
    if type(value) in (list,tuple):return {'t':'list','v':[_pack(x,budget,depth+1) for x in value]}
    if type(value) is dict:
        out={}
        for key,item in value.items():
            if type(key) is not str:raise PayloadError('Mapping keys must be strings')
            _pack(key,budget,depth+1);out[key]=_pack(item,budget,depth+1)
        return {'t':'dict','v':out}
    raise PayloadError('Unsupported payload value')

def _unpack(value,budget,depth=0):
    budget.charge(1,depth)
    if value is None or type(value) in (bool,str):return value
    if type(value) is int:
        if not -(2**63)<=value<2**64:raise PayloadError('Invalid integer')
        return value
    if type(value) is float:
        if not math.isfinite(value):raise PayloadError('Invalid float')
        return value
    if type(value) is not dict:raise PayloadError('Invalid payload envelope')
    if value.get('t')=='list' and set(value)=={'t','v'} and type(value['v']) is list:
        return [_unpack(x,budget,depth+1) for x in value['v']]
    if value.get('t')=='dict' and set(value)=={'t','v'} and type(value['v']) is dict:
        return {k:_unpack(v,budget,depth+1) for k,v in value['v'].items()}
    if value.get('t')=='array' and set(value)=={'t','dtype','shape','data'}:
        dtype=value['dtype'];shape=value['shape'];data=value['data']
        if (type(dtype) is not str or dtype not in _DTYPES or type(shape) is not list or len(shape)>8
                or any(type(x) is not int or x<0 or x>_MAX_BYTES for x in shape) or type(data) is not str):
            raise PayloadError('Invalid array envelope')
        size=math.prod(shape)*np.dtype(dtype).itemsize
        if size>budget.left or len(data)>4*((size+2)//3):raise PayloadError('Array allocation exceeds limit')
        raw=base64.b64decode(data,validate=True)
        if len(raw)!=size:raise PayloadError('Array byte size differs')
        budget.charge(size,depth);result=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
        if not np.isfinite(result).all():raise PayloadError('Nonfinite decoded array')
        return result
    raise PayloadError('Unknown payload envelope')

def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise PayloadError('Duplicate protocol field')
        result[key]=value
    return result

def _encode(message,limit):
    raw=json.dumps(message,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>limit:raise PayloadError('Serialized payload exceeds limit')
    return raw

def _decode(raw,limit):
    if len(raw)>limit:raise PayloadError('Received payload exceeds limit')
    return json.loads(raw,object_pairs_hook=_unique)

def _send(sock,data,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('IPC deadline exceeded')
    sock.settimeout(remaining);sock.sendall(struct.pack('!I',len(data))+data)

def _receive(sock,limit,deadline):
    def read(size):
        chunks=[]
        while size:
            remaining=None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining<=0:raise TimeoutError('IPC deadline exceeded')
            sock.settimeout(remaining);part=sock.recv(min(size,65536))
            if not part:raise EOFError('Worker connection closed')
            chunks.append(part);size-=len(part)
        return b''.join(chunks)
    size=struct.unpack('!I',read(4))[0]
    if size>limit:raise PayloadError('Reply exceeds byte limit')
    return read(size)

def _resolve(path):
    module,name=path.split(':',1);value=importlib.import_module(module)
    for part in name.split('.'):value=getattr(value,part)
    if not callable(value):raise TypeError('Configured target is not callable')
    return value

def _child(sock,path,factory,input_limit,output_limit):
    null=os.open(os.devnull,os.O_WRONLY)
    try:os.dup2(null,1);os.dup2(null,2)
    finally:os.close(null)
    identity=None
    try:
        configuration=_receive(sock,input_limit,time.monotonic()+60)
        target=_resolve(path)
        if factory:target=target(**_unpack(_decode(configuration,input_limit),_Budget(input_limit)))
        if not callable(target):raise TypeError('Factory did not return a callable')
        _send(sock,_encode({'kind':'ready'},output_limit),time.monotonic()+5)
        while True:
            message=_decode(_receive(sock,input_limit,None),input_limit)
            if type(message) is not dict or set(message)!={'id','payload'}:raise PayloadError('Invalid request')
            identity=message['id']
            if type(identity) is not int or identity<1:raise PayloadError('Invalid request identifier')
            value=target(_unpack(message['payload'],_Budget(input_limit)))
            result={'kind':'result','id':identity,'payload':_pack(value,_Budget(output_limit-128))}
            _send(sock,_encode(result,output_limit),time.monotonic()+5)
    except BaseException as error:
        try:_send(sock,_encode({'kind':'error','id':identity,'error_type':type(error).__name__[:80]},output_limit),time.monotonic()+.1)
        except BaseException:pass
    finally:sock.close()

def _seconds(value,name,maximum=60.):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value) or not 0<float(value)<=maximum:
        raise ValueError(f'{name} must be finite, positive and at most {maximum:g} seconds')
    return float(value)

def _limit(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not 256<=value<=_MAX_BYTES:
        raise ValueError(f'{name} must be an integer from 256 to {_MAX_BYTES}')
    return int(value)

class ProcessProposalWorker:
    """One spawned callable, one outstanding proposal, no automatic restart.

    submit/poll perform only bounded serialization and nonblocking socket I/O.
    Call poll/maintain regularly to supervise deadlines. close/restart may join
    for at most two shutdown grace intervals, plus ordinary OS scheduling.
    """
    def __init__(self,target,*,factory=False,factory_kwargs=None,timeout_s=1.,startup_timeout_s=5.,
                 shutdown_grace_s=.1,max_input_bytes=1024*1024,max_output_bytes=1024*1024):
        identifier=r'[A-Za-z_][A-Za-z0-9_]*'
        if type(target) is not str or len(target)>512 or not re.fullmatch(identifier+r'(?:\.'+identifier+r')*:'+identifier+r'(?:\.'+identifier+r')*',target):
            raise ValueError('Use a trusted importable module:callable path')
        if type(factory) is not bool or (factory_kwargs is not None and type(factory_kwargs) is not dict):
            raise ValueError('Factory configuration must be an explicit dictionary')
        self.timeout_s=_seconds(timeout_s,'timeout_s');self.startup_timeout_s=_seconds(startup_timeout_s,'startup_timeout_s')
        self.shutdown_grace_s=_seconds(shutdown_grace_s,'shutdown_grace_s',2.)
        self.max_input_bytes=_limit(max_input_bytes,'max_input_bytes');self.max_output_bytes=_limit(max_output_bytes,'max_output_bytes')
        self.target=target;self.factory=factory
        self._configuration=_encode(_pack(factory_kwargs or {},_Budget(self.max_input_bytes)),self.max_input_bytes)
        self._owner_pid=os.getpid();self._context=mp.get_context('spawn');self._lock=RLock();self.events=deque(maxlen=32)
        self._process=None;self._socket=None;self._last_pid=None;self._last_exitcode=None;self._generation=0;self._preparing=Event()
        self.closed=True;self.disabled=True;self.ready=False;self.reason='not_started';self.error_type=None
        self._clear();self.restart()

    def _clear(self):
        self._busy=False;self._request=None;self._outgoing=b'';self._sent=0;self._incoming=bytearray()
        self._has_result=False;self._result=None;self._terminate_at=None;self._killed=False;self._starting=False

    @property
    def pid(self):
        if os.getpid()!=self._owner_pid:return None
        with self._lock:return None if self._process is None else self._process.pid

    @property
    def alive(self):
        if os.getpid()!=self._owner_pid:return False
        with self._lock:return self._process is not None and self._process.is_alive()

    @property
    def status(self):
        if os.getpid()!=self._owner_pid:return {'ready':False,'disabled':True,'closed':True,'reason':'foreign_owner','pid':None}
        with self._lock:return {'ready':self.ready,'disabled':self.disabled,'closed':self.closed,
            'reason':self.reason,'error_type':self.error_type,'pending':int(self._busy),
            'pid':self.pid,'last_pid':self._last_pid,'last_exitcode':self._last_exitcode,'generation':self._generation}

    def _event(self,name):self.events.append({'event':name,'reason':self.reason,'pid':self._last_pid})

    def _disable(self,reason,error_type=None):
        self.disabled=True;self.ready=False;self._starting=False;self.reason=reason;self.error_type=error_type
        self._busy=False;self._request=None;self._outgoing=b'';self._incoming.clear();self._result=None;self._has_result=False
        if self._socket is not None:
            try:self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self._socket.close();self._socket=None
        if self._process is not None and self._terminate_at is None:
            self._terminate_at=time.monotonic()
            try:
                if self._process.is_alive():self._process.terminate()
            except OSError:self.reason='termination_failed'
        self._event('disabled')

    def _reap(self,blocking=False):
        process=self._process
        if process is None:return True
        if process.pid is None:
            process.close();self._process=None;return True
        if blocking and process.is_alive():process.join(self.shutdown_grace_s)
        if (process.is_alive() and self._terminate_at is not None
                and (blocking or time.monotonic()-self._terminate_at>=self.shutdown_grace_s) and not self._killed):
            try:process.kill();self._killed=True
            except OSError:self.reason='kill_failed'
        if blocking and process.is_alive():process.join(self.shutdown_grace_s)
        if process.is_alive():return False
        process.join(0);self._last_exitcode=process.exitcode;process.close();self._process=None;self._event('reaped');return True

    def _pump(self):
        if self._socket is None:return
        try:
            while self._sent<len(self._outgoing):
                try:sent=self._socket.send(memoryview(self._outgoing)[self._sent:])
                except BlockingIOError:break
                if sent==0:raise EOFError('Worker socket closed')
                self._sent+=sent
            if self._sent==len(self._outgoing):self._outgoing=b'';self._sent=0
            while True:
                try:part=self._socket.recv(min(65536,self.max_output_bytes+4-len(self._incoming)))
                except BlockingIOError:return
                if not part:raise EOFError('Worker exited')
                self._incoming.extend(part)
                if len(self._incoming)<4:continue
                size=struct.unpack('!I',self._incoming[:4])[0]
                if size>self.max_output_bytes:raise PayloadError('Reply exceeds wire byte limit')
                if len(self._incoming)<size+4:continue
                if len(self._incoming)!=size+4:raise PayloadError('Unexpected extra protocol frame')
                message=_decode(bytes(self._incoming[4:]),self.max_output_bytes);self._incoming.clear()
                if type(message) is not dict:raise PayloadError('Invalid response')
                if message.get('kind')=='error':
                    error=message.get('error_type')
                    if type(error) is not str or len(error)>80:raise PayloadError('Invalid worker error')
                    self._disable('worker_error',error);return
                if self._starting and message=={'kind':'ready'}:
                    self._starting=False;self.disabled=False;self.ready=True;self.reason=None;self._event('ready');return
                if (self._starting or set(message)!={'kind','id','payload'} or message['kind']!='result'
                        or self._request is None or type(message['id']) is not int or message['id']!=self._request[0]):
                    raise PayloadError('Response does not match outstanding request')
                deadline=self._request[1]
                if time.monotonic()>=deadline:raise TimeoutError('Reply expired')
                result=_unpack(message['payload'],_Budget(self.max_output_bytes))
                if time.monotonic()>=deadline:raise TimeoutError('Reply expired during decoding')
                # Keep the request deadline while a decoded result remains unread.
                self._result=result;self._has_result=True
                return
        except TimeoutError:self._disable('deadline_exceeded')
        except (OSError,EOFError):self._disable('worker_exit')
        except Exception as error:self._disable('malformed_output',type(error).__name__[:80])

    def maintain(self):
        """Nonblocking deadline supervision and opportunistic reaping."""
        if os.getpid()!=self._owner_pid:return self.status
        with self._lock:
            if not self.disabled:
                if self._request is not None and time.monotonic()>self._request[1]:self._disable('deadline_exceeded')
                else:
                    self._pump()
                    if not self.disabled and self._process is not None and not self._process.is_alive():self._disable('worker_exit')
            if self.disabled:self._reap()
            return self.status

    def submit(self,payload,*,timeout_s=None):
        if os.getpid()!=self._owner_pid:return False
        try:timeout=self.timeout_s if timeout_s is None else _seconds(timeout_s,'timeout_s')
        except ValueError:return False
        if not self._lock.acquire(False):return False
        try:
            self.maintain()
            if self.closed or self.disabled or not self.ready or self._busy or self._preparing.is_set():return False
            self._preparing.set();generation=self._generation
            identity=getattr(self,'_next_id',0)+1;self._next_id=identity;deadline=time.monotonic()+timeout
        finally:self._lock.release()
        try:raw=_encode({'id':identity,'payload':_pack(payload,_Budget(self.max_input_bytes-128))},self.max_input_bytes)
        except Exception as error:
            self._preparing.clear()
            if self._lock.acquire(False):
                try:
                    if generation==self._generation:self.reason='invalid_input';self.error_type=type(error).__name__[:80]
                finally:self._lock.release()
            return False
        except BaseException:
            self._preparing.clear();raise
        if not self._lock.acquire(False):
            self._preparing.clear();return False
        try:
            self._preparing.clear()
            if self.closed or self.disabled or generation!=self._generation or time.monotonic()>deadline:return False
            self.reason=None;self.error_type=None
            self._busy=True;self._request=(identity,deadline);self._outgoing=struct.pack('!I',len(raw))+raw;self._sent=0
            self._event('submitted');self._pump();return not self.disabled
        finally:self._lock.release()

    def poll(self):
        if os.getpid()!=self._owner_pid or not self._lock.acquire(False):return None
        try:
            self.maintain()
            if self.closed or self.disabled or not self._has_result:return None
            result=self._result;deadline=self._request[1]
            self._result=None;self._has_result=False;self._busy=False;self._request=None
            if time.monotonic()>=deadline:
                self._disable('deadline_exceeded');return None
            self._event('completed');return result
        finally:self._lock.release()

    def close(self):
        """Bounded cleanup; False forbids replacing a child that was not reaped."""
        if os.getpid()!=self._owner_pid:return False
        with self._lock:
            self.closed=True
            if not self.disabled:self._disable('closed')
            elif self._process is not None and self._terminate_at is None:self._disable(self.reason or 'closed')
            return self._reap(blocking=True) and not self._preparing.is_set()

    def restart(self):
        """Explicit opt-in restart, only after the previous child was reaped."""
        if os.getpid()!=self._owner_pid:return False
        with self._lock:
            if not self.close():return False
            self._generation+=1;self._clear();self._next_id=0;self.closed=False;self.disabled=True
            self.reason='starting';self.error_type=None;self._starting=True
            parent,child=socket.socketpair();parent.setblocking(False);self._socket=parent
            process=self._context.Process(target=_child,args=(child,self.target,self.factory,
                self.max_input_bytes,self.max_output_bytes),daemon=True)
            self._process=process;deadline=time.monotonic()+self.startup_timeout_s
            try:
                process.start();self._last_pid=process.pid
                self._outgoing=struct.pack('!I',len(self._configuration))+self._configuration;self._sent=0
            except Exception as error:self._disable('startup_failure',type(error).__name__[:80])
            finally:child.close()
            while self._starting and self._socket is not None:
                self._pump()
                if not self._starting:break
                if time.monotonic()>=deadline:self._disable('startup_timeout');break
                if not process.is_alive():self._disable('startup_failure');break
                select.select([self._socket],[],[],min(.01,max(0,deadline-time.monotonic())))
            if not self.ready:self._reap(blocking=True)
            return self.ready

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
