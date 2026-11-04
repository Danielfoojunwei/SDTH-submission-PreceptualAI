"""Faster reference covariance and causal rolling fleet sufficient statistics."""
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
from ..fleetwm.data import geometry,scan_context,FleetInputs,Observations
from ..native.world import CLIENT_TX_DBM,HORIZONS
from ..native.radiomap import DISC_SCALE_M


def fit_reference(g,past,strong=False):
    from ..native.radiomap import RadioMap
    from ..fleet_strong_reference import prior
    pos=past.positions[-60:];value=past.value[-60:];heard=past.heard[-60:]
    scanned=past.scanned[-60:];bound=past.bound[-60:]
    robots=np.broadcast_to(np.arange(pos.shape[1]),pos.shape[:2]);ordinal=np.cumsum(scanned,axis=0)
    obs={};cens={}
    for s in range(len(g.sites)):
        for b in range(3):
            j=s*3+b;h=heard[:,:,j];c=scanned&~h&((ordinal-1)%5==0)
            obs[s,b]=(pos[...,0][h],pos[...,1][h],value[:,:,j][h],robots[h])
            cens[s,b]=(pos[...,0][c],pos[...,1][c],bound[:,:,j][c],robots[c])
    return RadioMap(g.sites,g.aps,0,prior() if strong else None).fit(obs,cens,np.arange(max(1,pos.shape[1]//5)))


class FastMap:
    """Same posterior/features as RadioMap; a BLAS quadratic form, no refit."""
    def __init__(self,rmap):self.rmap=rmap
    def predict(self,x,y):
        r=self.rmap;fx=np.asarray(x).ravel();fy=np.asarray(y).ravel()
        mean=np.zeros((len(fx),len(r.sites),3),dtype='f4');std=np.zeros_like(mean)
        for (s,b),post in r.post.items():
            w,Ainv=post[:2];phi=r._features(s,b,fx,fy)
            mean[:,s,b]=phi@w
            var=np.maximum(np.sum((phi@Ainv)*phi,axis=1),1e-6)+r.disc_var
            if getattr(r,'disc_far',0)>0:
                seen=r.seen.get((s,b))
                d=cKDTree(seen).query(np.column_stack([fx,fy]))[0] if len(seen) else np.full(len(fx),np.inf)
                var+=r.disc_far*(1-np.exp(-d/DISC_SCALE_M))
            std[:,s,b]=np.sqrt(var)
        return mean.reshape(x.shape+(len(r.sites),3)),std.reshape(x.shape+(len(r.sites),3))


class FleetMemory:
    """O(new reports) updates; stores sixty genuine observed scan summaries."""
    def __init__(self,g):
        self.g=g;self.end=0;self.rows=deque();self.total=None;self.n=None

    def update(self,past):
        if not isinstance(past,Observations):raise TypeError('Use observable past scans')
        T,N,L=past.value.shape
        if T<self.end:raise ValueError('History cannot move backwards')
        if self.n is not None and self.n!=N:raise ValueError('Fleet membership changed; reset memory')
        self.n=N
        if self.end==0:self.end=max(0,T-60)
        for t in range(self.end,T):
            pos=past.positions[t];_,ld,racks,phy=geometry(self.g,pos)
            h=past.heard[t];sc=past.scanned[t,:,None];c=sc&~h
            shift=np.tile(CLIENT_TX_DBM-self.g.tx-self.g.noise,len(self.g.sites))
            residual=past.value[t]+shift-phy.reshape(N,L)
            bound=past.bound[t]+shift-phy.reshape(N,L)
            X=np.stack([np.ones_like(ld),ld-1.5,racks/3],-1).repeat(3,axis=1)
            raw=np.where(h,residual,0).astype('f8');clipped=np.where(h,np.clip(residual,-50,50),0).astype('f8')
            xy=np.broadcast_to(pos[:,None,:],(N,L,2))
            row={'count':h.sum(0).astype('f8'),'raw':raw.sum(0),'sum':clipped.sum(0),'square':(clipped**2).sum(0),
                'gram':np.einsum('nli,nlj,nl->lij',X,X,h,optimize=True),
                'rhs':np.einsum('nli,nl->li',X,clipped,optimize=True),
                'xy':(xy*h[...,None]).sum(0),'xy2':(xy**2*h[...,None]).sum(0),
                'c_count':c.sum(0).astype('f8'),'c_sum':np.where(c,bound,0).sum(0).astype('f8')}
            if self.total is None:self.total={k:np.zeros_like(v) for k,v in row.items()}
            if len(self.rows)==60:
                expired=self.rows.popleft()[1]
                for k in self.total:self.total[k]-=expired[k]
            for k in self.total:self.total[k]+=row[k]
            self.rows.append((t,row));self.end=t+1
        return self

    def features(self):
        a=self.total;c=a['count'];safe=np.maximum(c,1);L=len(c)
        coefs=np.linalg.solve(a['gram']+np.diag([3.,10.,10.])[None],a['rhs'][...,None])[...,0]
        coefs[c==0]=0
        memory=np.zeros((L,13),dtype='f4');memory[:,0]=c/(len(self.rows)*self.n)
        memory[:,1]=a['sum']/safe/20
        memory[:,2]=np.sqrt(np.maximum(a['square']/safe-(a['sum']/safe)**2,0))/10
        recent=list(self.rows)[-10:];rc=sum(r['count'] for _,r in recent);rr=sum(r['raw'] for _,r in recent)
        memory[:,3]=np.where(rc>0,(rr/np.maximum(rc,1)-a['sum']/safe)/10,0)
        memory[:,4:7]=coefs/10
        memory[:,7:9]=np.sqrt(np.maximum(a['xy2']/safe[:,None]-(a['xy']/safe[:,None])**2,0))/[60,40]
        age=np.full(L,60.)
        for t,r in self.rows:age=np.where(r['count']>0,(self.end-1)-t,age)
        memory[:,9]=np.where(c>0,age/60,1)
        memory[:,10]=a['c_sum']/np.maximum(a['c_count'],1)/20
        bands=np.arange(L)%3
        band_count=np.bincount(bands,weights=c,minlength=3);band_raw=np.bincount(bands,weights=a['raw'],minlength=3)
        memory[:,11]=(band_raw/np.maximum(band_count,1))[bands]/20
        memory[:,12]=self.g.noise[bands]/100
        return memory,coefs,c.copy()

    def inputs(self,past,plan,rmap=None):
        self.update(past);memory,coefs,count=self.features();g=self.g
        N,H=plan.shape[:2];L=len(g.sites)*3
        delta,ld,racks,physics=geometry(g,plan);qld=ld.repeat(3,-1);rc=racks.repeat(3,-1);d=delta.repeat(3,-2)
        h=np.broadcast_to(np.asarray(HORIZONS)[None,:,None]/30,(N,H,L));band=np.broadcast_to(np.tile(np.arange(3)/2,len(g.sites)),(N,H,L))
        px=np.broadcast_to(plan[...,0,None],(N,H,L));py=np.broadcast_to(plan[...,1,None],(N,H,L))
        q=np.stack([d[...,0]/120,d[...,1]/80,qld/3,rc/5,h,band,np.sin(px/15),np.cos(px/15),np.sin(py/15),np.cos(py/15),qld-qld[:,:1]],-1).astype('f4')
        X=np.stack([np.ones_like(qld),qld-1.5,rc/3],-1)
        fleet=physics.reshape(N,H,L)+np.einsum('nhli,li->nhl',X,coefs)
        if rmap is None:mm=np.zeros((N,H,L),dtype='f4');sd=np.ones_like(mm)
        else:
            m,sd=FastMap(rmap).predict(plan[...,0],plan[...,1]);mm=(m-g.tx+CLIENT_TX_DBM-g.noise).reshape(N,H,L);sd=sd.reshape(N,H,L)
        b=FleetInputs(scan_context(g,past),q,np.broadcast_to(memory,(N,L,13)).copy(),physics.reshape(N,H,L),mm,sd,fleet.astype('f4'))
        return b,np.broadcast_to(np.log1p(count)[None,:,None]/10,(N,L,1)).copy().astype('f4')
