"""Public receiver and route features; hidden packet labels are separate."""
from collections import deque
import numpy as np
from ..native.phy import LinkCurves
from ..frontier.model import failure_table
from ..closed_loop.control import integral_from


class PublicHistory:
    def __init__(self,g,cost):
        self.geometry,self.cost=g,cost;self.rows=deque(maxlen=6)
        self.previous=None;self.last_switch=None

    def update(self,obs):
        if self.rows and obs['slot']<=self.rows[-1][0]:raise ValueError('History must advance')
        g=self.geometry;site=np.asarray(obs['current']);n=len(site);ar=np.arange(n)
        if self.last_switch is None:self.last_switch=np.full(n,-3000)
        local=np.asarray(obs['local_rssi']);valid=local>=-90;snr=local-g.tx+15-g.noise
        cur=np.where(valid[ar,site],snr[ar,site],0)/30
        best=np.where(valid,snr,-40).max(1)/30;coverage=valid.mean(1)
        pos=obs['positions'];route=(obs['plan'][:,[1,5,6]]-pos[:,None]).reshape(n,6)/20
        load=np.bincount(site,minlength=len(g.sites))
        sent=np.zeros(n) if self.previous is None else np.asarray(obs['sent'])-self.previous['sent']
        missed=np.zeros(n) if self.previous is None else np.asarray(obs['misses'])-self.previous['misses']
        frame=np.concatenate([np.asarray(obs['pilot_snr'])/30,cur,best,coverage,pos/[120,80],route,
            g.sites[site]/[120,80],load[site,None]/100,(missed/np.maximum(sent,1))[:,None],
            (np.maximum(sent-missed,0)/20)[:,None],(np.asarray(obs['gap'])/100)[:,None],
            (np.asarray(obs['outage_remaining'])>0)[:,None],(np.asarray(obs['scan_remaining'])>0)[:,None],
            np.full((n,1),self.cost/4),np.minimum((obs['slot']-self.last_switch)/3000,1)[:,None],
            np.asarray(obs['roams'])[:,None]/10,np.full((n,1),load.mean()/100)],1).astype('f4')
        assert frame.shape==(n,32) and np.isfinite(frame).all()
        self.rows.append((int(obs['slot']),frame))
        self.previous={'sent':np.array(obs['sent'],copy=True),'misses':np.array(obs['misses'],copy=True)}
        return frame

    def context(self):
        if not self.rows:raise ValueError('No public history')
        rows=[r for _,r in self.rows];return np.stack([rows[0]]*(6-len(rows))+rows,1).astype('f4')

    def executed(self,slot,ids):self.last_switch[np.asarray(ids,dtype=int)]=slot


def action_queries(g,obs,cost):
    fc=np.asarray(obs['forecast']);n,_,s,_=fc.shape;ar=np.arange(n);site=np.asarray(obs['current'])
    local=np.asarray(obs['local_rssi']);known=local>=-90;cache=np.where(known,local-g.tx+15-g.noise,0)
    candidate=fc[:,:7].max(-1).transpose(0,2,1)/30
    current=np.broadcast_to(fc[ar,:7,site].max(-1)[:,None,:]/30,(n,s,7))
    load=np.bincount(site,minlength=s)
    distance=np.linalg.norm(obs['positions'][:,None]-g.sites[None],axis=-1)/100
    sigma=np.asarray(obs.get('forecast_std',np.full((n,s,3),5.))).max(-1)/10
    q=np.concatenate([candidate,current,cache/30,known.astype('f4'),
        np.broadcast_to(load[None,:,None]/100,(n,s,1)),distance[...,None],
        np.full((n,s,1),cost/4),sigma[...,None]],-1).astype('f4')
    assert q.shape==(n,s,24) and np.isfinite(q).all()
    return q


class NumericPacketValue:
    def __init__(self,cost):
        self.cost=cost;self.table=failure_table();self.grid=np.linspace(-40,60,501)

    def __call__(self,obs):
        fc=np.asarray(obs['forecast']);site=np.asarray(obs['current']);ar=np.arange(len(site))
        p=np.stack([np.interp(fc[...,b],self.grid,self.table[b]) for b in range(3)],-1).min(-1)
        stay=integral_from(p,0,10)[ar,site];move=self.cost+integral_from(p,self.cost,10)
        delta=20*(move-stay[:,None]);delta[ar,site]=0;return delta.astype('f4')


def immediate_mask(g,obs):
    curves=LinkCurves();th=np.asarray([curves.threshold_db(b,.99)[0] for b in ('2g','5g','6g')])
    local=np.asarray(obs['local_rssi']);known=local>=-90
    margin=np.where(known,local-g.tx+15-g.noise-th,-np.inf).max(-1)
    current=(np.asarray(obs['pilot_snr'])-th).max(-1)
    return (margin>=3)&(margin>=current[:,None]-3)
