"""Typed, unit-normalised public radio events; no simulator truth argument."""
from collections import deque
import numpy as np

LINK_FEATURES=('cached_snr_over30','cache_heard','current_pilot_age_over30s',
    'current_pilot_snr_over30','current_pilot_valid','completed_map_snr_over30',
    'completed_map_std_over10','frequency_ghz_over6','log_distance_over3',
    'controlled_fleet_ap_load_over100','configured_mcs_over11','forecast_age_over30s')
GLOBAL_FEATURES=('control_gap_over10s','pending_age_over_deadline','handover_remaining_over4s',
    'scan_remaining_over90ms','recent_miss_fraction','recent_delivery_fraction',
    'log_video_queue_bits_over20','log_bulk_queue_bits_over20')


def public_features(g,public):
    site=np.asarray(public['current']);n=len(site);s=len(g.sites);ar=np.arange(n)
    x=np.zeros((n,s,3,12),dtype='f4');valid=np.ones_like(x,dtype=bool)
    local=public.get('local_rssi')
    heard=np.zeros((n,s,3),dtype=bool) if local is None else np.asarray(public['local_heard'],dtype=bool)
    if local is not None:x[...,0]=np.where(heard,np.asarray(local)-g.tx+15-g.noise,0)/30
    valid[...,0]=heard;x[...,1]=heard
    # Neighbor ages are not supplied by the scanner and remain unknown.
    valid[...,2:5]=False
    associated=np.asarray(public['pilot_site'])==site
    ids=np.flatnonzero(associated)
    x[ids,site[ids],:,2]=np.asarray(public['pilot_age_slots'])[ids,None]*.01/30
    valid[ids,site[ids],:,2]=True
    # Below-sensitivity simulator values must not become exact RF observations.
    pilot=(np.asarray(public['pilot_valid']) &
           (np.asarray(public['pilot_snr'])+g.noise+g.tx-15 >= -90.))
    x[ar,site,:,3]=np.where(pilot,public['pilot_snr'],0)/30
    valid[ar,site,:,3]=pilot;x[ar,site,:,4]=pilot
    valid[...,4]=True  # A known absence of a valid pilot is itself observable.
    forecast=public.get('forecast')
    if forecast is None:valid[...,5:7]=False;valid[...,11]=False
    else:
        x[...,5]=np.asarray(forecast)[:,0]/30
        if 'forecast_std' in public:x[...,6]=np.asarray(public['forecast_std'])/10
        else:valid[...,6]=False
        x[...,11]=public['forecast_age_seconds']/30
    x[...,7]=np.array([2.4,5.2,6.])/6
    d=np.linalg.norm(np.asarray(public['positions'])[:,None]-g.sites[None],axis=-1)
    x[...,8]=np.log10(np.maximum(d,1))[...,None]/3
    x[...,9]=np.asarray(public['load'])[None,:,None]/100
    valid[...,10]=False;x[ar,site,:,10]=np.asarray(public['mcs'])/11;valid[ar,site,:,10]=True
    history=np.asarray(public['outcome_history'])[-10:]
    delivered=history[...,0].sum(0);missed=history[...,1].sum(0);outcomes=np.maximum(delivered+missed,1)
    glob=np.stack([np.asarray(public['gap'])*.01/10,np.maximum(public['pending_packet_age'],0)/5,
        np.asarray(public['outage_remaining'])*.01/4,np.asarray(public['scan_remaining'])*.01/.09,
        missed/outcomes,delivered/outcomes,np.log1p(public['video_queue_bits'])/20,
        np.log1p(public['bulk_queue_bits'])/20],-1).astype('f4')
    assert x.shape==(n,s,3,12) and glob.shape==(n,8)
    assert np.isfinite(x[valid]).all() and np.isfinite(glob).all()
    return x,glob,valid


class RadioHistory:
    def __init__(self,geometry,length=20):
        self.geometry=geometry;self.length=length;self.rows=deque(maxlen=length)
        self.observations=deque(maxlen=60);self.last_slot=None;self.event_count=0

    def observe(self,public):
        slot=int(public['slot'])
        if self.last_slot is not None and slot<=self.last_slot:raise ValueError('Event time must advance')
        links,glob,mask=public_features(self.geometry,public)
        dt=0. if self.last_slot is None else (slot-self.last_slot)*.01
        self.rows.append((slot,links,glob,mask,dt));self.last_slot=slot;self.event_count+=1
        # Compact recorder rows are for decoder provenance, not a second state.
        self.observations.append({'slot':slot,'frame':glob.copy()})

    def window(self):
        if not self.rows:raise ValueError('No causal observations')
        rows=list(self.rows)
        return {'links':np.stack([r[1] for r in rows],1),
                'globals':np.stack([r[2] for r in rows],1),
                'observed':np.stack([r[3] for r in rows],1),
                'elapsed':np.broadcast_to(np.array([r[4] for r in rows],dtype='f4'),(len(rows[0][1]),len(rows))).copy()}
