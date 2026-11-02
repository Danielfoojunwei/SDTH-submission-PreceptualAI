"""Opt-in single-model runtime and a nonblocking forecast worker.

No radio/SSH actions are performed here. This serves causal forecast proposals;
physical control and the application's RadioMap default remain separate.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json,time
import numpy as np
import torch
from .frontier.study import ROOT,OUT,P,sources,digest,load_student,tensors
from .frontier.fast import FleetMemory
from .frontier.timing import ForecastQueue
from .fleetwm.data import Geometry,Observations


class OptimizedWorldModel:
    def __init__(self,geometry: Geometry,*,family='development_best'):
        if not isinstance(geometry,Geometry):raise TypeError('Use public Geometry')
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
        if not isinstance(history,Observations):raise TypeError('Use an observable scan prefix')
        if len(history.value)<60:raise ValueError('The tested profile requires sixty past timestamps')
        if planned_positions.shape!=(history.positions.shape[1],9,2):raise ValueError('Expected N x 9 x 2 planned positions')
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
        return mean.reshape(shape).astype('f4'),sd.reshape(shape).astype('f4'),{'family':self.family,'history_cutoff_exclusive':t,'total_seconds':time.perf_counter()-a,'research_only':True}


class AsyncForecastService:
    """A single real worker, bounded pending work, and nonblocking polling.

    A caller must keep sending control with its existing safe fallback while
    poll returns None. Logical issued_at/now values use the caller's clock.
    This does not qualify a hardware control-loop deadline.
    """
    def __init__(self,model,max_pending=2):
        if max_pending<1:raise ValueError('Need a positive queue bound')
        self.model=model;self.max_pending=max_pending;self.pool=ThreadPoolExecutor(max_workers=1)
        self.pending=deque();self.latest=None;self.closed=False

    def submit(self,history,plan,issued_at):
        if self.closed:raise RuntimeError('Forecast service is closed')
        self._collect()
        if len(self.pending)>=self.max_pending:return False
        # Capture the submitted observable prefix and route, not mutable views.
        snapshot=history.before(len(history.value));route=np.array(plan,copy=True)
        future=self.pool.submit(self.model.forecast,snapshot,route)
        self.pending.append((float(issued_at),future));return True

    def _collect(self):
        while self.pending and self.pending[0][1].done():
            issued,future=self.pending.popleft();self.latest=(issued,future.result())

    def poll(self,now):
        self._collect()
        if self.latest is None:return None
        issued,(mean,std,metadata)=self.latest
        if float(now)<issued:return None
        age=float(now)-issued;q=ForecastQueue([0.]);q.age=age
        return q.advance(mean),q.advance(std),metadata|{'forecast_age_seconds':age,'issued_at':issued}

    def close(self):
        self.closed=True;self.pool.shutdown(wait=True)
