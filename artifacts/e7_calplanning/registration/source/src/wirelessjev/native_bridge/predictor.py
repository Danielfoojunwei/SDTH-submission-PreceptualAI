"""Public-only adapter of an adaptive field to the native warehouse domain.

The 2D warehouse is explicitly extruded, not passed off as RT ground truth.
No class here receives a World, material labels, hidden channel or future RF.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
import torch

from ..adaptive_radio.service import (RSSIForecast,RadioAssumptions,RSSIOutcomeAdapter,
                                     BridgedPredictor,NATIVE_CARRIERS_HZ)
from ..native.world import HORIZONS
from ..spectral_scene.operators import query_physics_basis,fspl_received_power_dbm
from ..twin.warehouse import RACK_ROWS,RACK_SEGMENTS,WIDTH,HEIGHT

RX_HEIGHT_M=1.
AP_HEIGHT_M=4.
RACK_HEIGHT_M=3.
HALL_HEIGHT_M=6.
MAX_CONTEXT=64
SOURCE_ID='adaptive_scene_VAL_branch_seed0_native_extrapolation'


def array_digest(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str,a.shape)).encode()+a.tobytes()).hexdigest()


def rack_boxes():
    """Public 33 disjoint rack boxes; depth/height are declared approximations."""
    return np.asarray([((x0,y-.6,0.),(x1,y+.6,RACK_HEIGHT_M))
                       for y in RACK_ROWS for x0,x1 in RACK_SEGMENTS],dtype=float)


def vectorized_box_features(rx_xyz,ap_xyz,boxes):
    """Vectorized segment/AABB intersections for DISJOINT public boxes.

    Summed interval lengths equal the union only for disjoint interiors. Such
    boxes are checked explicitly. There is no Python query-point×box loop.
    """
    rx=np.asarray(rx_xyz,dtype=float);ap=np.asarray(ap_xyz,dtype=float);box=np.asarray(boxes,dtype=float)
    if (rx.ndim!=2 or rx.shape[1]!=3 or ap.shape!=rx.shape or box.ndim!=3 or box.shape[1:]!=(2,3)
            or not all(np.isfinite(x).all() for x in (rx,ap,box)) or np.any(box[:,1]<box[:,0])):
        raise ValueError('Finite endpoint arrays and ordered public XYZ boxes required')
    if not len(box):return np.zeros(len(rx)),np.zeros(len(rx))
    overlap=np.minimum(box[:,None,1],box[None,:,1])-np.maximum(box[:,None,0],box[None,:,0])
    if np.any(np.all(overlap>=-1e-12,axis=-1)&~np.eye(len(box),dtype=bool)):
        raise ValueError('Fast interval sums require separated boxes, including no touching faces')
    delta=(rx-ap)[:,None,:];parallel=np.abs(delta)<1e-14
    low=box[None,:,0,:]-ap[:,None,:];high=box[None,:,1,:]-ap[:,None,:]
    outside=np.any(parallel&((low>0)|(high<0)),axis=-1)
    divisor=np.where(parallel,1.,delta)
    first=low/divisor;second=high/divisor
    entry=np.max(np.where(parallel,-np.inf,np.minimum(first,second)),axis=-1)
    exit=np.min(np.where(parallel,np.inf,np.maximum(first,second)),axis=-1)
    entry=np.maximum(entry,0.);exit=np.minimum(exit,1.)
    valid=(exit>entry+1e-12)&~outside
    lengths=np.where(valid,exit-entry,0.)
    return valid.sum(1).astype(float),np.clip(lengths.sum(1),0.,1.)


def public_features(rx,ap,frequency,noise,tx,*,boxes=None):
    rx=np.asarray(rx,dtype=float);ap=np.asarray(ap,dtype=float)
    if rx.ndim!=2 or rx.shape[1]!=3 or ap.shape!=rx.shape:raise ValueError('Per-link public XYZ required')
    n=len(rx);frequency=np.broadcast_to(np.asarray(frequency,dtype=float),(n,))
    noise=np.broadcast_to(np.asarray(noise,dtype=float),(n,));tx=np.broadcast_to(np.asarray(tx,dtype=float),(n,))
    distance=np.linalg.norm(rx-ap,axis=-1)
    if not all(np.isfinite(x).all() for x in (rx,ap,frequency,noise,tx)) or np.any(distance<=0) or np.any(frequency<=0):
        raise ValueError('Finite nondegenerate public physical links required')
    crossings,fraction=vectorized_box_features(rx,ap,rack_boxes() if boxes is None else boxes)
    # Device0 reference is assumed zero for this simulator; no relative paired
    # calibration samples, invented quantization bins or device IDs are added.
    g=np.column_stack((rx/np.array((WIDTH,HEIGHT,HALL_HEIGHT_M)),
        ap/np.array((WIDTH,HEIGHT,HALL_HEIGHT_M)),np.log10(distance)/2,
        np.log(frequency/2.4e9),-np.ones(n),crossings/8,fraction,
        np.zeros(n),np.zeros(n),np.zeros(n),noise/100,tx/30))
    return g.astype('f4')


def select_context(past,geometry,*,maximum=MAX_CONTEXT):
    """Round-robin AP×2 source-band groups, latest real delivered scans first.

    At equal time prefer heard to censored reports, then robot index. Censor
    bounds remain bounds. A missing scan is never converted to an observation.
    Bands 2.4/5.2 only; measured 6.5 GHz reports are deliberately not encoded.
    """
    positions=np.asarray(past.positions);value=np.asarray(past.value);heard=np.asarray(past.heard)
    bound=np.asarray(past.bound);scanned=np.asarray(past.scanned)
    if (value.ndim!=3 or value.shape[-1]!=len(geometry.sites)*3 or heard.shape!=value.shape
            or bound.shape!=value.shape or positions.shape!=value.shape[:2]+(2,)
            or scanned.shape!=value.shape[:2] or heard.dtype!=bool or scanned.dtype!=bool
            or type(maximum) is not int or not 1<=maximum<=MAX_CONTEXT
            or not np.isfinite(positions).all() or np.any(heard&~scanned[...,None])):
        raise ValueError('Actual delivered observation history axes/masks required')
    times,robots=np.nonzero(scanned)
    if not len(times):raise ValueError('No delivered scan evidence; do not invent a scheduled measurement')
    groups=[]
    for site in range(len(geometry.sites)):
        for band in (0,1):
            link=site*3+band
            order=np.lexsort((robots,~heard[times,robots,link],-times))
            groups.append([(int(times[i]),int(robots[i]),site,band) for i in order[:maximum]])
    selected=[]
    for ordinal in range(maximum):
        for group in groups:
            if ordinal<len(group):selected.append(group[ordinal])
            if len(selected)>=maximum:break
        if len(selected)>=maximum:break
    index=np.asarray(selected,dtype=np.int64);t,r,s,b=index.T;link=s*3+b
    rx=np.column_stack((positions[t,r],np.full(len(t),RX_HEIGHT_M)))
    ap=np.column_stack((np.asarray(geometry.sites)[s],np.full(len(t),AP_HEIGHT_M)))
    carriers=np.asarray(NATIVE_CARRIERS_HZ)[b]
    q=public_features(rx,ap,carriers,np.asarray(geometry.noise)[b],np.asarray(geometry.tx)[b])
    fspl=fspl_received_power_dbm(rx,ap,carriers,np.asarray(geometry.tx)[b])
    h=heard[t,r,link];y=value[t,r,link];upper=bound[t,r,link]
    if not np.isfinite(y[h]).all() or not np.isfinite(upper[~h]).all():
        raise ValueError('Known real reports/bounds must be finite')
    context=np.column_stack((np.where(h,(y-fspl)/30.,0.),np.where(~h,(upper-fspl)/30.,0.),h,~h,q)).astype('f4')
    return {'context':context,'known':np.ones(len(context),bool),'scheduled':np.ones(len(context),bool),
            'selected_rows':index,'heard':h,'source_history_length':len(value)}


def query_operands(geometry,plan):
    plan=np.asarray(plan,dtype=float)
    sites=np.asarray(geometry.sites,dtype=float)
    if plan.ndim!=3 or plan.shape[1:]!=(len(HORIZONS),2) or not np.isfinite(plan).all():
        raise ValueError('Only a finite public native robot×nine-horizon×XY plan is accepted')
    n,h=plan.shape[:2];s=len(sites);shape=(n,h,s,3)
    rx=np.broadcast_to(np.concatenate((plan,np.full((n,h,1),RX_HEIGHT_M)),axis=-1)[:,:,None,None,:],shape+(3,)).reshape(-1,3)
    ap=np.broadcast_to(np.column_stack((sites,np.full(s,AP_HEIGHT_M)))[None,None,:,None,:],shape+(3,)).reshape(-1,3)
    carrier=np.broadcast_to(np.asarray(NATIVE_CARRIERS_HZ),shape).reshape(-1)
    noise=np.broadcast_to(np.asarray(geometry.noise),shape).reshape(-1)
    tx=np.broadcast_to(np.asarray(geometry.tx),shape).reshape(-1)
    q=public_features(rx,ap,carrier,noise,tx)
    return {'query':q,'basis':query_physics_basis(q),'fspl':fspl_received_power_dbm(rx,ap,carrier,tx).astype('f4'),
            'offset':np.zeros(len(q),dtype='f4'),'shape':shape}


def radio_assumptions(geometry):
    return RadioAssumptions(np.asarray(NATIVE_CARRIERS_HZ),np.asarray(geometry.tx),np.full(3,15.),
        np.asarray(geometry.noise),np.zeros(3),np.zeros(3),np.zeros(3),
        'native simulator zero device0 bias assumed; not identified physical hardware calibration',
        'initial registered DEV probe: collision_lambda=0 and interference=False; native contention/fading/blockage remain',True)


class AdaptiveNativePredictor:
    """Frozen selected RF model; encode <=64 past reports, render actual 6.5 GHz.

    Geometry/scale/array/domain transfer is unqualified. The all-zero current
    association in RSSIForecast is an unused conversion placeholder; the queue
    consumes only RF mean/std, and native planning supplies real current AP.
    """
    def __init__(self,geometry,model,checkpoint_sha256,*,source_id=SOURCE_ID):
        if getattr(geometry,'yard',False):raise ValueError('Warehouse rack extrusion cannot describe a yard geometry')
        if model.config['cell'] not in ('joint','regression'):raise ValueError('Only the published VAL-selected RF branch is eligible')
        self.geometry=geometry;self.model=model.eval();self.checkpoint_sha256=checkpoint_sha256
        self.source_id=source_id;self.records=[];self.last_history=None

    @torch.no_grad()
    def step(self,past,plan):
        started=time.perf_counter();context=select_context(past,self.geometry);q=query_operands(self.geometry,plan)
        self.last_history={name:np.array(getattr(past,name),copy=True)
                           for name in ('positions','value','heard','bound','scanned')}
        def tensor(x):return torch.from_numpy(np.array(x,copy=True))[None]
        belief=self.model.encode(tensor(context['context']),tensor(context['known']),tensor(context['scheduled']))
        mean,std=self.model.render(belief,tensor(q['query']),tensor(q['basis']),tensor(q['fspl']),tensor(q['offset']))
        m=mean[0].cpu().numpy().reshape(q['shape']);s=std[0].cpu().numpy().reshape(q['shape'])
        forecast=RSSIForecast(HORIZONS,np.zeros(len(plan),dtype=np.int64),m,s,NATIVE_CARRIERS_HZ,
                              self.source_id,self.checkpoint_sha256)
        self.records.append({'history_length':len(past.value),'context':context['context'].copy(),
            'known':context['known'].copy(),'scheduled':context['scheduled'].copy(),
            'selected_rows':context['selected_rows'].copy(),'plan':np.array(plan,copy=True),
            'mean_dbm':forecast.mean_dbm.copy(),'std_db':forecast.std_db.copy(),
            'belief_mean':belief['mean'][0].cpu().numpy().copy(),
            'belief_covariance':belief['covariance'][0].cpu().numpy().copy()})
        elapsed=time.perf_counter()-started
        return forecast,'cold' if len(self.records)==1 else 'warm',elapsed


class MapRoundTripPredictor:
    """A separate wiring control: actual strong Map SNR→RSSI→same bridge."""
    def __init__(self,geometry,source_receipt_sha256):
        from ..closed_loop.runtime import SinglePredictor
        self.geometry=geometry;self.inner=SinglePredictor(geometry,'radiomap')
        self.source_receipt_sha256=source_receipt_sha256;self.records=[];self.last_history=None

    def step(self,past,plan):
        started=time.perf_counter();m,s,kind,elapsed=self.inner.step(past,plan)
        self.last_history={name:np.array(getattr(past,name),copy=True)
                           for name in ('positions','value','heard','bound','scanned')}
        rssi=m.astype('f8')+self.geometry.tx-15.+self.geometry.noise
        forecast=RSSIForecast(HORIZONS,np.zeros(len(plan),dtype=np.int64),rssi,s,NATIVE_CARRIERS_HZ,
                              'radiomap_roundtrip',self.source_receipt_sha256)
        self.records.append({'history_length':len(past.value),'source_snr':m.copy(),'source_std':s.copy(),
                             'plan':np.array(plan,copy=True),'mean_dbm':rssi.copy()})
        return forecast,kind,max(elapsed,time.perf_counter()-started)


def bridged(predictor):
    source=predictor.source_id if isinstance(predictor,AdaptiveNativePredictor) else 'radiomap_roundtrip'
    adapter=RSSIOutcomeAdapter(radio_assumptions(predictor.geometry),source,site_id='native_public_warehouse',
                               device_id='native_zero_gauge_simulation')
    return BridgedPredictor(predictor,adapter)
