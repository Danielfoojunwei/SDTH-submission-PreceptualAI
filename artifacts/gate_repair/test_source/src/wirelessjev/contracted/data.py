"""Causal capped device scans, site splits, and a separate open-yard twin."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from ..native.world import World,build_world,HORIZONS,POS_DT,NOISE_DBM,BANDS,CLIENT_TX_DBM
from ..native.foundation import random_site
from ..native.radiomap import RadioMap
from ..twin.warehouse import make_warehouse

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=json.loads((ROOT/'artifacts/contracted/protocol.json').read_text())
READINGS=json.loads((ROOT/'artifacts/native/intent_flip.json').read_text())['readings']
KEYS=('ho','U','V','fair','focus','viol','prot','W','B')
HISTORY=6


def capped_scans(rssi,rng,missing=.05):
    """Unknown values stay unknown; a top-k bound is not an RSSI measurement."""
    measured=rssi+10*np.log10(rng.gamma(2,.5,size=rssi.shape))-(rng.random(rssi.shape[:-1])<.05)[...,None]*15
    flat=measured.reshape(*rssi.shape[:2],-1);order=np.argsort(-flat,axis=-1)[...,:8]
    heard=np.zeros_like(flat,dtype=bool);np.put_along_axis(heard,order,True,axis=-1);heard&=flat>=-90
    scanned=rng.random(flat.shape[:2])>=missing
    bound=np.where(heard.sum(-1)==8,np.take_along_axis(flat,order[...,-1:],-1)[...,0],-90)
    heard&=scanned[...,None]
    return {'value':np.where(heard,flat,0).astype('f4'),'heard':heard,'bound':np.broadcast_to(bound[...,None],flat.shape).astype('f4').copy(),'scanned':scanned}


def yard_world(seed,duration,n):
    """Declared outdoor rectangle routes and perimeter APs; not warehouse racks."""
    rng=np.random.default_rng(seed);sites=np.array([(x,y) for x in [5,50,95] for y in [5,95]],float)
    # Robots circulate around independently offset rectangular logistics routes.
    offset=rng.uniform(0,320,n);speed=rng.uniform(.6,1.2,n)
    def position(t):
        phase=(np.asarray(t)[...,None]*speed+offset)%320
        x=np.where(phase<80,10+phase,np.where(phase<160,90,np.where(phase<240,250-phase,10)))
        y=np.where(phase<80,10,np.where(phase<160,phase-70,np.where(phase<240,90,330-phase)))
        return np.stack([x,y],-1)
    T=int(round(duration/POS_DT));pos=position(np.arange(T)*POS_DT)
    plan=position(np.arange(int(duration))[:,None]+np.asarray(HORIZONS)[None,:]).transpose(0,2,1,3)
    noise=np.array([NOISE_DBM[b] for b in BANDS]);tx=np.array([21.,23.,23.]);phase=rng.uniform(0,6.28,(len(sites),3));slope=rng.uniform(24,30)
    def loss(p):
        d=np.maximum(1,np.linalg.norm(p[...,None,:]-sites,axis=-1))
        shadow=3*np.sin(p[...,None,None,0]/14+phase)*np.cos(p[...,None,None,1]/18)
        # A loading building creates an explicit geometric obstruction in the yard.
        block=((p[...,0]>35)&(p[...,0]<65)&(p[...,1]>35)&(p[...,1]<65))[...,None,None]*8
        return np.array([40.,46.7,48.64])+slope*np.log10(d)[...,None]+shadow+block
    pl=loss(pos);fc=CLIENT_TX_DBM-loss(plan)-noise
    domain=np.arange(len(sites)*3).reshape(len(sites),3)
    return World(seed,n,sites,(CLIENT_TX_DBM-pl-noise).astype('f4'),(tx-pl).astype('f4'),fc.astype('f4'),domain,np.full(len(sites)*3,.8),np.where(np.arange(n)%10==0,8.,0.),np.full(n,.5),duration,4.,False,np.full((int(duration),n,len(sites),3),4.,dtype='f4'),noise,None,np.tile([1,36,7],(len(sites),1)),None,plan,pos,tx)


def make_stream(seed,*,n=8,decisions=32,condition='main'):
    duration=60+decisions+31
    if condition=='yard':full=yard_world(seed,duration,n)
    else:full=build_world(seed,duration_s=duration,n_robots=n,sigma_map=0,site_params=random_site(seed),noise_2g_extra_db=10 if condition=='stress' else 0)
    scan=capped_scans(full.rssi_dl[::5],np.random.default_rng(seed+900000))
    return full,scan


def scan_features(world,scan):
    T,N,L=scan['value'].shape;S=len(world.sites)
    shift=np.tile(CLIENT_TX_DBM-world.tx_dl-world.noise_db,S)
    value=(scan['value']+shift)*scan['heard'];bound=scan['bound']+shift
    xy=world.pos[::5,:N];delta=xy[:,:,None,:]-np.repeat(world.sites,3,axis=0)[None,None]
    feat=np.stack([value/30,bound/30,scan['heard'],np.broadcast_to(scan['scanned'][...,None],(T,N,L)),delta[...,0]/120,delta[...,1]/80,np.log10(np.maximum(1,np.linalg.norm(delta,axis=-1)))/3,np.broadcast_to(np.tile(np.arange(3)/2,S),(T,N,L))],-1).astype('f4')
    feat[...,:2]*=scan['scanned'][...,None,None]
    return feat,value.astype('f4'),bound.astype('f4')


def examples(world,scan,decisions,*,targets=True):
    """Only ctx/route/prior reach the predictor. Truth is a separate label."""
    feat,value,bound=scan_features(world,scan);N=world.n;S=len(world.sites);L=S*3
    times=np.arange(60,60+decisions);past=times[:,None]-np.arange(HISTORY-1,-1,-1)[None,:]
    ctx=feat[past].transpose(0,2,3,1,4).reshape(-1,L,HISTORY*8)
    plan=world.plan[times];delta=plan[...,None,:]-np.repeat(world.sites,3,axis=0)[None,None,None]
    dist=np.log10(np.maximum(1,np.linalg.norm(delta,axis=-1)))
    prior=CLIENT_TX_DBM-np.tile([40,46.7,48.64],S)-33*dist-np.tile(world.noise_db,S)
    route=np.stack([delta[...,0]/120,delta[...,1]/80,dist/3,np.broadcast_to(np.asarray(HORIZONS)[None,None,:,None]/30,dist.shape),prior/30],-1)
    future=times[:,None]+np.asarray(HORIZONS,dtype=int)[None,:]
    heard=scan['heard'][future].transpose(0,2,1,3)
    result={'ctx':ctx,'route':route.reshape(-1,len(HORIZONS),L,5).astype('f4'),'prior':prior.reshape(-1,len(HORIZONS),L).astype('f4'),
            'truth':world.snr[future*5].transpose(0,2,1,3,4).reshape(-1,len(HORIZONS),L).astype('f4'),
            'heard':heard.reshape(-1,len(HORIZONS),L),'site':np.full(len(ctx),world.seed,dtype='i4')}
    observed=np.where(scan['heard'][times],value[times],-99).reshape(len(times),N,S,3).max(-1)
    current=observed.argmax(-1);loads=np.array([np.bincount(c,minlength=S) for c in current])
    result['current']=current.reshape(-1).astype('i8');result['load']=np.repeat(loads[:,None,:],N,axis=1).reshape(-1,S).astype('f4')
    if targets:
        target_times=future[:,:,None]-np.arange(HISTORY-1,-1,-1)[None,None,:]
        result['future_ctx']=feat[target_times].transpose(0,3,1,4,2,5).reshape(-1,len(HORIZONS),L,HISTORY*8)
        result['target']=value[future].transpose(0,2,1,3).reshape(-1,len(HORIZONS),L)
        result['bound']=bound[future].transpose(0,2,1,3).reshape(-1,len(HORIZONS),L)
        result['valid']=np.broadcast_to(scan['scanned'][future].transpose(0,2,1)[...,None],heard.shape).reshape(-1,len(HORIZONS),L)
    return result


class YardRadioMap(RadioMap):
    def _features(self,s,b,x,y):
        d=np.maximum(1,np.hypot(x-self.sites[s,0],y-self.sites[s,1]))
        arg=x[:,None]*self.w[s,b,:,0]+y[:,None]*self.w[s,b,:,1]+self.ph[s,b]
        return np.column_stack([np.ones_like(x),np.log10(d),np.zeros_like(x),np.sqrt(2/48)*np.cos(arg)])


def bayesian(world,scan,condition='main'):
    obs={};cens={};S=len(world.sites);pos=world.pos[::5]
    for s in range(S):
        for b in range(3):
            j=s*3+b;h=scan['heard'][:60,:,j];v=scan['scanned'][:60]&~h
            robots=np.broadcast_to(np.arange(world.n)[None,:],h.shape)
            obs[s,b]=(pos[:60,...,0][h],pos[:60,...,1][h],scan['value'][:60,:,j][h],robots[h])
            # Same thinning and compensation as the existing Bayesian fitter.
            v&=(np.arange(60)[:,None]%5==0)
            cens[s,b]=(pos[:60,...,0][v],pos[:60,...,1][v],scan['bound'][:60,:,j][v],robots[v])
    aps=[a for a in make_warehouse(world.seed,n_robots=1).aps if a.band=='5g']
    cls=YardRadioMap if condition=='yard' else RadioMap
    return cls(world.sites,aps,world.seed).fit(obs,cens,np.arange(max(1,world.n//5)))


def runtime_world(full,decisions,forecast,sigma):
    start=int(60/POS_DT);stop=start+int(decisions/POS_DT)
    return replace(full,snr=full.snr[start:stop].copy(),rssi_dl=full.rssi_dl[start:stop].copy(),pos=full.pos[start:stop].copy(),plan=full.plan[60:60+decisions].copy(),forecast=np.asarray(forecast,dtype='f4'),sigma_belief=np.asarray(sigma,dtype='f4'),duration_s=float(decisions),rmap=None)
