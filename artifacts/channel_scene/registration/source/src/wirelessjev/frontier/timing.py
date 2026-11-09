"""FIFO forecast availability; fast radio control does not wait for inference."""
import numpy as np
from ..native.world import HORIZONS


class ForecastQueue:
    def __init__(self,service_seconds):
        service=np.asarray(service_seconds,dtype=float)
        if service.ndim!=1 or not np.isfinite(service).all() or (service<0).any():raise ValueError('Invalid service durations')
        ready=[];previous=0.
        for d,s in enumerate(service):
            previous=max(float(d),previous)+float(s);ready.append(int(np.ceil(previous*100-1e-10)))
        self.ready=np.asarray(ready,dtype=int);self.last=-1;self.age=0.

    def take(self,k):
        ix=int(np.searchsorted(self.ready,k,side='right'))-1
        if ix<=self.last:return None
        self.last=ix;self.age=k/100-ix
        return ix

    def advance(self,forecast):
        x=np.asarray(HORIZONS);y=np.asarray(forecast);target=np.clip(x+self.age,x[0],x[-1])
        right=np.searchsorted(x,target,side='right').clip(1,len(x)-1);left=right-1
        w=((target-x[left])/(x[right]-x[left]))[None,:,None,None]
        return (y[:,left]*(1-w)+y[:,right]*w).astype(y.dtype)


def reliability_scores(fc):
    """Saturate excess margin; a policy score, never relabelled as SNR."""
    # Two smooth scales reward usable margin without endlessly chasing high SNR.
    p=1/(1+np.exp(-np.clip((fc-5)/4,-50,50)))
    best=p[:,:6].max(-1)
    return 15*np.log(np.maximum(best.min(1),1e-6)) + 5*best.mean(1)
