"""Opt-in causal world-model forecasts and the frozen research controller."""
from dataclasses import replace
import json,time
import numpy as np
from .fleetwm.data import Geometry,Observations,inputs,fit_map
from .fleetwm.study import ROOT,OUT,P,load,predict,sources,digest
from .contracted.data import READINGS
from .native.intent_policy import intent_policy
from .gate_repair.gate import reactive_rescue


class _UnusedReference:
    def __init__(self,sites):self.sites=sites
    def predict(self,x,y):
        shape=x.shape+(len(self.sites),3)
        return np.zeros(shape,dtype='f4'),np.ones(shape,dtype='f4')


class FleetWorldModel:
    """No World/truth argument; history must end before the decision time.

    This is a research profile, not a real-car deployment or safety release.
    The standalone branch performs no per-BSS RadioMap fit.
    """
    def __init__(self,geometry: Geometry,*,family='development_best'):
        if not isinstance(geometry,Geometry):raise TypeError('Use known Geometry')
        selected=json.loads((OUT/'selection.json').read_text())
        if not selected['sealed'] or selected['source_sha256']!=sources():raise RuntimeError('The research profile no longer matches the executed source')
        if family=='development_best':family=selected['development_best_model']
        if family not in P['families']:raise ValueError('Unsupported world-model family')
        for name,h in selected['checkpoint_sha256'].items():
            if digest(ROOT/name)!=h:raise RuntimeError('A frozen checkpoint changed')
        self.geometry=geometry;self.family=family;self.models=load(family)
        self.gate=selected['choices'][family]['gate'];self.rmap=None;self.refit_at=None;self.last_time=None

    def forecast(self,history: Observations,planned_positions):
        if not isinstance(history,Observations):raise TypeError('Use observable device history')
        t=len(history.value)
        if t<60:raise ValueError('The validated profile needs sixty past scans')
        if self.last_time is not None and t<self.last_time:raise ValueError('History cannot move backwards')
        plan=np.asarray(planned_positions)
        if plan.shape!=(history.positions.shape[1],9,2):raise ValueError('Expected N x 9 x 2 planned positions')
        started=time.perf_counter();refit=0.
        if self.family=='fleet_jepa':
            self.rmap=_UnusedReference(self.geometry.sites)
        elif self.rmap is None or t-self.refit_at>=P['map_refresh_seconds']:
            a=time.perf_counter();self.rmap=fit_map(self.geometry,history);refit=time.perf_counter()-a;self.refit_at=t
        b=inputs(self.geometry,history,plan,self.rmap);mean,std=predict(self.models,b)
        self.last_time=t;shape=(len(plan),9,len(self.geometry.sites),3)
        return mean.reshape(shape),std.reshape(shape),{'family':self.family,'history_cutoff_exclusive':t,
            'total_seconds':time.perf_counter()-started,'map_refit_seconds':refit,'research_only':True}

    def controller(self,mission='bulk_uploads'):
        if mission!=P['primary_mission']:raise ValueError('The reusable profile is scoped to bulk_uploads')
        reading=READINGS[mission]
        policy=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),
            name=self.family+'_validated',online_map=False,gate=True,belief_scale=1 if self.gate=='rescue' else self.gate)
        return policy,reactive_rescue if self.gate=='rescue' else None
