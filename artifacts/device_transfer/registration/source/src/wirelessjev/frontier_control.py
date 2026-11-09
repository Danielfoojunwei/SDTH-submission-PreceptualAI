"""Opt-in single-model runtime and a nonblocking forecast worker.

No radio/SSH actions are performed here. This serves causal forecast proposals;
physical control and the application's RadioMap default remain separate.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from threading import RLock,current_thread
import json,time
import numpy as np
import torch
from .frontier.study import ROOT,OUT,P,sources,digest,load_student,tensors
from .frontier.fast import FleetMemory
from .frontier.timing import ForecastQueue
from .fleetwm.data import Geometry,Observations


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _history_snapshot(history, plan, *, max_steps, max_bytes, sites=None, copy_inputs=True):
    """Bound the wrapper's input allocation before scanning/copying arrays."""
    if not isinstance(history, Observations):
        raise TypeError('Use an explicit observable history')
    arrays=(history.positions,history.value,history.heard,history.bound,history.scanned)
    if any(type(a) is not np.ndarray for a in arrays) or type(plan) is not np.ndarray:
        raise ValueError('History and route fields must be NumPy arrays')
    value=history.value
    if value.ndim != 3:
        raise ValueError('Expected T x N x links observations')
    t,n,links=value.shape
    if not 60 <= t <= max_steps or n < 1 or links < 3 or links % 3:
        raise ValueError('History exceeds its explicit runtime window or has invalid dimensions')
    if sites is not None and links != sites*3:
        raise ValueError('History does not match the public geometry')
    if (history.positions.shape != (t,n,2) or history.heard.shape != (t,n,links)
            or history.bound.shape != (t,n,links) or history.scanned.shape != (t,n)
            or plan.shape != (n,9,2)):
        raise ValueError('Inconsistent history or route shape')
    if sum(a.nbytes for a in (*arrays,plan)) > max_bytes:
        raise ValueError('Observable snapshot exceeds its byte limit')
    if history.heard.dtype != np.bool_ or history.scanned.dtype != np.bool_:
        raise ValueError('Scan and heard masks must be boolean')
    for a in (history.positions,value,history.bound,plan):
        if (not np.issubdtype(a.dtype,np.number) or np.issubdtype(a.dtype,np.complexfloating)
                or not np.isfinite(a).all()):
            raise ValueError('Observable numeric fields must be finite real numbers')
    if np.any(history.heard & ~history.scanned[...,None]):
        raise ValueError('A heard entry requires an actual scan')
    if copy_inputs:
        arrays=tuple(np.array(a,copy=True) for a in arrays)
        plan=np.array(plan,copy=True)
        for a in (*arrays,plan):a.flags.writeable=False
        return _history_snapshot(Observations(*arrays),plan,max_steps=max_steps,
                                 max_bytes=max_bytes,sites=sites,copy_inputs=False)
    return Observations(*arrays),plan,(n,9,links//3,3),t


class _ForecastFailure(Exception):
    def __init__(self,error_type):
        self.error_type=str(error_type)[:80]
        super().__init__(self.error_type)


def _validated_forecast(result, shape, cutoff, byte_limit):
    if type(result) is not tuple or len(result) != 3:
        raise ValueError('Forecast worker must return mean, uncertainty and metadata')
    mean,std,metadata=result
    if (type(mean) is not np.ndarray or type(std) is not np.ndarray
            or mean.shape != shape or std.shape != shape
            or not np.issubdtype(mean.dtype,np.floating) or not np.issubdtype(std.dtype,np.floating)
            or mean.nbytes+std.nbytes > byte_limit or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std < 0).any()):
        raise ValueError('Invalid or nonfinite forecast arrays')
    if type(metadata) is not dict or len(metadata)>16:
        raise ValueError('Forecast metadata must be a small scalar dictionary')
    clean={}
    for key,value in metadata.items():
        if type(key) is not str or len(key)>64:
            raise ValueError('Invalid forecast metadata key')
        if isinstance(value,np.generic):value=value.item()
        if type(value) is str:
            if len(value)>256:raise ValueError('Forecast metadata string exceeds limit')
        elif type(value) in (int,float):
            if not np.isfinite(value):raise ValueError('Nonfinite forecast metadata')
        elif type(value) not in (bool,type(None)):
            raise ValueError('Forecast metadata values must be bounded scalars')
        clean[key]=value
    supplied=clean.get('history_cutoff_exclusive',cutoff)
    if type(supplied) is not int or supplied != cutoff:
        raise ValueError('Forecast metadata has the wrong observable cutoff')
    clean['history_cutoff_exclusive']=cutoff
    with np.errstate(over='ignore',invalid='ignore'):
        mean=mean.astype('f4',copy=True);std=std.astype('f4',copy=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Forecast cannot be represented finitely')
    mean.flags.writeable=False;std.flags.writeable=False
    return mean,std,clean


class OptimizedWorldModel:
    def __init__(self,geometry: Geometry,*,family='development_best',max_history_steps=600,max_snapshot_bytes=64*1024*1024):
        if not isinstance(geometry,Geometry):raise TypeError('Use public Geometry')
        self.max_history_steps=_positive_integer(max_history_steps,'max_history_steps')
        self.max_snapshot_bytes=_positive_integer(max_snapshot_bytes,'max_snapshot_bytes')
        if self.max_history_steps<60:raise ValueError('Need at least sixty timestamps')
        selected=json.loads((OUT/'selection.json').read_text())
        if not selected['sealed'] or selected['source_sha256']!=sources():raise RuntimeError('The research source no longer matches selection')
        if family=='development_best':family=selected['selected_family']
        if family not in P['families']:raise ValueError('Choose a trained frontier family')
        for name,h in selected['checkpoint_sha256'].items():
            if digest(ROOT/name)!=h:raise RuntimeError('Frozen checkpoint identity differs')
        self.geometry=geometry;self.family=family;self.model=load_student(family);self.memory=FleetMemory(geometry)
        self.previous=None;self.previous_t=None;self.innovation=np.zeros(len(geometry.sites)*3,dtype='f4')

    @torch.no_grad()
    def forecast(self,history: Observations,planned_positions):
        history,planned_positions,_,_=_history_snapshot(history,planned_positions,
            max_steps=self.max_history_steps,max_bytes=self.max_snapshot_bytes,
            sites=len(self.geometry.sites),copy_inputs=False)
        a=time.perf_counter();b,c=self.memory.inputs(history,planned_positions);p=self.model(tensors(b,c));mean=p['mean'].numpy();sd=p['std'].numpy();t=len(history.value);g=self.geometry
        if self.family=='temporal':
            if self.previous_t==t-1 and self.previous is not None:
                heard=history.heard[-1]&(history.value[-1]>-75)
                observed=history.value[-1]+np.tile(15-g.tx-g.noise,len(g.sites))+1.174
                error=np.where(heard,observed-self.previous,0);count=heard.sum(0)
                update=error.sum(0)/np.maximum(count,1)
                self.innovation=np.clip(self.innovation+np.where(count>=3,.2*np.clip(update,-6,6),0),-10,10)
            elif self.previous_t is not None:self.innovation[:]=0
            mean=mean+self.innovation[None,None,:];self.previous=mean[:,0].copy()
        self.previous_t=t;shape=(len(planned_positions),9,len(g.sites),3)
        result=(mean.reshape(shape).astype('f4'),sd.reshape(shape).astype('f4'),
                {'family':self.family,'history_cutoff_exclusive':t,'total_seconds':time.perf_counter()-a,'research_only':True})
        return _validated_forecast(result,shape,t,self.max_snapshot_bytes)


class AsyncForecastService:
    """Bounded proposals; errors/expiry return None and retain baseline control.

    Worker failure disables this instance until explicitly replaced. close()
    never joins a running thread; False means a worker is still alive. Python
    threads cannot be killed here, including at interpreter shutdown.
    """
    def __init__(self,model,max_pending=2,*,max_history_steps=600,max_snapshot_bytes=64*1024*1024,max_forecast_age_s=5.,max_events=64):
        self.max_pending=_positive_integer(max_pending,'max_pending')
        self.max_history_steps=_positive_integer(max_history_steps,'max_history_steps')
        self.max_snapshot_bytes=_positive_integer(max_snapshot_bytes,'max_snapshot_bytes')
        max_events=_positive_integer(max_events,'max_events')
        if self.max_history_steps<60:raise ValueError('Need at least sixty timestamps')
        if (isinstance(max_forecast_age_s,(bool,np.bool_)) or not isinstance(max_forecast_age_s,Real)
                or not np.isfinite(max_forecast_age_s) or not 0<max_forecast_age_s<=30):
            raise ValueError('Forecast expiry must be finite and within thirty seconds')
        self.max_forecast_age_s=float(max_forecast_age_s)
        self.model=model;self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='radio-forecast')
        self.pending=deque();self.latest=None;self.closed=False;self.failed=False
        self.events=deque(maxlen=max_events);self._lock=RLock();self._closing_futures=()
        self._last_issued=None;self._last_poll=None;self._history_end=0;self._fleet_shape=None
        self._reservations=0;self._worker_thread=None

    def _event(self,status,error=None):
        record={'status':status}
        if error is not None:record['error_type']=getattr(error,'error_type',type(error).__name__)[:80]
        self.events.append(record)

    @staticmethod
    def _clock(value):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value):
            raise ValueError('Use a finite monotonic clock value')
        return float(value)

    def _predict(self,snapshot,route,shape,cutoff):
        with self._lock:
            self._worker_thread=current_thread()
            if self.closed or self.failed:raise _ForecastFailure('DisabledService')
        try:
            return _validated_forecast(self.model.forecast(snapshot,route),shape,cutoff,self.max_snapshot_bytes)
        except BaseException as error:
            # Base exceptions raised inside this worker are worker faults; the
            # control thread's own KeyboardInterrupt is not swallowed.
            with self._lock:self.failed=True;self.latest=None
            raise _ForecastFailure(type(error).__name__) from None

    def submit(self,history,plan,issued_at):
        with self._lock:
            self._collect()
            if self.closed or self.failed or len(self.pending)+self._reservations>=self.max_pending:return False
            # Reserve before copying so concurrent callers cannot each allocate
            # a full snapshot outside the pending-work bound.
            self._reservations+=1
        try:
            issued=self._clock(issued_at)
            snapshot,route,shape,cutoff=_history_snapshot(history,plan,
                max_steps=self.max_history_steps,max_bytes=self.max_snapshot_bytes)
        except Exception as error:
            with self._lock:
                self._reservations-=1;self.latest=None;self._event('invalid_input_fallback',error)
            return False
        except BaseException:
            with self._lock:self._reservations-=1
            raise
        with self._lock:
            self._reservations-=1
            self._collect()
            if self.closed or self.failed or len(self.pending)>=self.max_pending:return False
            if ((self._last_issued is not None and issued<self._last_issued) or cutoff<self._history_end
                    or (self._fleet_shape is not None and self._fleet_shape!=shape)):
                self.latest=None;self._event('nonmonotonic_input_fallback');return False
            try:future=self.pool.submit(self._predict,snapshot,route,shape,cutoff)
            except Exception as error:
                self.latest=None;self.failed=True;self._event('submission_failure_fallback',error);return False
            self.pending.append((issued,future));self._last_issued=issued;self._history_end=cutoff;self._fleet_shape=shape
            return True

    def _collect(self):
        while self.pending and self.pending[0][1].done():
            issued,future=self.pending.popleft()
            try:result=future.result()
            except Exception as error:
                self.latest=None;self.failed=True;self._event('worker_failure_fallback',error)
                continue
            if not self.closed and not self.failed:self.latest=(issued,result)

    def poll(self,now):
        try:now=self._clock(now)
        except (TypeError,ValueError,OverflowError) as error:
            with self._lock:self.latest=None;self._event('invalid_clock_fallback',error)
            return None
        with self._lock:
            if self._last_poll is not None and now<self._last_poll:
                self.latest=None;self._event('backward_clock_fallback');return None
            self._last_poll=now;self._collect()
            if self.closed or self.failed or self.latest is None:return None
            issued,(mean,std,metadata)=self.latest
            if now<issued:return None
            age=now-issued
            if age>self.max_forecast_age_s:
                self.latest=None;self._event('expired_forecast_fallback');return None
            q=ForecastQueue([0.]);q.age=age
            return q.advance(mean),q.advance(std),metadata|{'forecast_age_seconds':age,'issued_at':issued}

    def close(self):
        with self._lock:
            first=not self.closed;self.closed=True;self.latest=None
            if first:
                self._closing_futures=tuple(f for _,f in self.pending)
                self.pending.clear()
        # Future.cancel() invokes arbitrary done callbacks synchronously. Keep
        # cancellation callbacks off the control thread; queued wrappers skip
        # their model call after close and drain on the one existing worker.
        if first:self.pool.shutdown(wait=False,cancel_futures=False)
        with self._lock:
            self._closing_futures=tuple(f for f in self._closing_futures if not f.done())
            alive=self._worker_thread is not None and self._worker_thread.is_alive()
            return not self._closing_futures and self._reservations==0 and not alive
