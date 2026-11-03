"""A gain-envelope certificate and a counterexample to rank-only guarantees."""
import numpy as np


def lower_bound(predicted_gain,true_gain,u,*,z=2.,c=2.,B=100.,eligible=None):
    """Empirical counterpart of the bounded-gain theorem in CONTRACTED_THEORY.md.

    Gain is the declared one-decision AP score, not cumulative packet reward.
    """
    if z<c or c<0 or B<=0:raise ValueError('Require z >= c >= 0 and B > 0')
    predicted_gain,true_gain,u=np.broadcast_arrays(predicted_gain,true_gain,u)
    if np.any(u<0) or np.any(np.abs(true_gain)>B):raise ValueError('Envelope/gain bounds violated')
    accept=predicted_gain>=z*u
    if eligible is not None:accept&=np.asarray(eligible,dtype=bool)
    margin=np.clip(predicted_gain-c*u,0,B)
    failed=np.abs(true_gain-predicted_gain)>c*u
    observed=float(np.mean(accept*true_gain));bound=float(np.mean(accept*margin)-2*B*np.mean(accept&failed))
    return {'mean_gain':observed,'empirical_lower_bound':bound,'accepted_fraction':float(accept.mean()),'accepted_envelope_failure_mass':float(np.mean(accept&failed)),'bound_holds':bool(observed+1e-8>=bound)}


def rank_counterexample():
    predicted=np.array([2.,3.]);truth=np.array([1.,-4.]);u=np.array([1.,2.])
    return [{'scale':scale,'error_rank_correlation':1.,'accepted':(predicted>=2*scale*u).tolist(),'mean_gain':float(np.mean(truth*(predicted>=2*scale*u)))} for scale in [.5,1.,2.]]
