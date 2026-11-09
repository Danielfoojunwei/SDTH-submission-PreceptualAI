"""Managed interference and delayed measurements for a new offline study.

Public plans/receipts contain no propagation labels. The private runtime alone
uses native warehouse propagation. Slot-averaged power addition, scheduled
channel occupancy, dedicated monitoring and reliable reserved report transport
are explicit simulator assumptions, not measured hardware capabilities.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _number(value, name, low=0., high=math.inf):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Invalid '+name)
    value=float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid '+name)
    return value


@dataclass(frozen=True)
class PublicProbeSpec:
    positions_m: object
    power_mw: object
    channel: int
    activity_plan: object
    probe_starts: tuple = ()
    design: str = 'none'
    pulse_airtime_fraction: float = .02
    capture_airtime_fraction: float = 0.
    report_bytes: int = 128
    report_rate_mbps: float = 6.5
    report_delay_slots: int = 2
    noise_floor_fraction: float = .02
    missing_probability: float = .05
    shared_receiver_blackout: bool = False

    def __post_init__(self):
        xy=np.asarray(self.positions_m,dtype=float);p=np.asarray(self.power_mw,dtype=float)
        plan=np.asarray(self.activity_plan,dtype=float)
        if xy.shape!=(2,2) or not np.isfinite(xy).all():raise ValueError('Two finite public XY positions required')
        if p.shape!=(2,) or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Two positive configured mW powers required')
        if plan.ndim!=2 or plan.shape[1]!=2 or not len(plan) or not np.isfinite(plan).all() or (plan<0).any() or (plan>1).any():
            raise ValueError('Activity plan must be finite Tx2 duty fractions in [0,1]')
        if self.design not in ('none','synchronized','staggered'):raise ValueError('Unknown probe design')
        starts=tuple(_integer(v,'probe start') for v in self.probe_starts)
        if tuple(sorted(set(starts)))!=starts:raise ValueError('Probe starts must be ordered and unique')
        for i,start in enumerate(starts):
            if start+2>=len(plan) or start%10>7 or (i and start<starts[i-1]+3):
                raise ValueError('Each complete probe cycle must lie within one native 100ms block')
            if np.any(plan[start:start+3]!=0):raise ValueError('Probe windows must be zero in the ordinary activity plan for every arm')
        _integer(self.channel,'5GHz channel',1)
        _integer(self.report_bytes,'report payload',1);_integer(self.report_delay_slots,'report delay',1)
        _number(self.pulse_airtime_fraction,'pulse fraction',0.,1.)
        _number(self.capture_airtime_fraction,'capture reservation',0.,1.)
        if self.pulse_airtime_fraction==0 and self.design!='none':raise ValueError('A probe must have positive exposure')
        _number(self.report_rate_mbps,'report transport rate',1e-9)
        _number(self.noise_floor_fraction,'measurement noise',0.,1e3)
        _number(self.missing_probability,'measurement missing probability',0.,1.)
        if type(self.shared_receiver_blackout) is not bool:raise ValueError('Blackout must be boolean')
        for name in ('channel','report_bytes','report_delay_slots'):
            object.__setattr__(self,name,int(getattr(self,name)))
        for name in ('pulse_airtime_fraction','capture_airtime_fraction','report_rate_mbps','noise_floor_fraction','missing_probability'):
            object.__setattr__(self,name,float(getattr(self,name)))
        # Tuples prevent callers from editing a future plan after construction.
        object.__setattr__(self,'positions_m',tuple(tuple(float(v) for v in row) for row in xy))
        object.__setattr__(self,'power_mw',tuple(float(v) for v in p))
        object.__setattr__(self,'activity_plan',tuple(tuple(float(v) for v in row) for row in plan))
        object.__setattr__(self,'probe_starts',starts)

    def phase(self,slot):
        slot=_integer(slot,'native slot')
        if self.design=='none':return None
        # A binary search avoids a per-slot linear pass over a long run plan.
        index=int(np.searchsorted(self.probe_starts,slot,side='right'))-1
        if index<0:return None
        phase=slot-self.probe_starts[index]
        return (index,phase) if phase<3 else None

    def activity(self,slot):
        """Actual slot-average duty; the declared finite plan extends by zeros."""
        slot=_integer(slot,'native slot');phase=self.phase(slot)
        if phase is not None:
            pattern=(((0.,0.),(1.,1.),(0.,0.)) if self.design=='synchronized'
                     else ((0.,0.),(1.,0.),(0.,1.)))
            return np.asarray(pattern[phase[1]])*self.pulse_airtime_fraction
        return np.asarray(self.activity_plan[slot],dtype=float).copy() if slot<len(self.activity_plan) else np.zeros(2)

    def digest(self):
        from dataclasses import asdict
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def geometry_prior(spec,receiver_positions_m,receiver_channels,*,pl_coef=33.,height_m=3.):
    """Public CAD/log-distance/rack prior; never the seeded shadowing field."""
    from ..twin.radio import AP
    from ..twin.warehouse import rack_obstruction
    xy=np.asarray(receiver_positions_m,dtype=float);channels=np.asarray(receiver_channels)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all() or channels.shape!=(len(xy),):
        raise ValueError('Public AP geometry/channel dimensions differ')
    _number(pl_coef,'public path-loss coefficient',0.);_number(height_m,'public height',0.)
    tx=np.asarray(spec.positions_m);d=np.sqrt(((xy[:,None]-tx[None])**2).sum(-1)+height_m**2)
    rack=np.stack([rack_obstruction(AP('public-'+str(i),*position,spec.channel,'5g'),tx[:,0],tx[:,1]) for i,position in enumerate(xy)])
    gain=10**(-(46.7+pl_coef*np.log10(np.maximum(d,1.))+rack)/10)
    return np.where((channels==spec.channel)[:,None],gain,0.)


def managed_overlap(estimated_gain_S2,power2,activity2,public_domainS,public_channelsS,channel):
    """Pure public-model CS/RF split for aligned nested duty windows.

    Kernel supplies private G; a predictor supplies its estimated/prior G.
    A source sensed by any AP defers its whole native domain. Hidden RF is
    conditioned on the remaining time outside that sensed prefix window.
    """
    from ..twin.radio import CS_THRESHOLD_DBM
    g=np.asarray(estimated_gain_S2,dtype=float);power=np.asarray(power2,dtype=float);a=np.asarray(activity2,dtype=float)
    domains=np.asarray(public_domainS);channels=np.asarray(public_channelsS)
    if (g.ndim!=2 or g.shape[1]!=2 or domains.shape!=(len(g),) or channels.shape!=(len(g),)
            or domains.dtype.kind not in 'iu' or (domains<0).any() or not len(g)
            or power.shape!=(2,) or a.shape!=(2,) or not np.isfinite(g).all() or (g<0).any()
            or not np.isfinite(power).all() or (power<=0).any() or not np.isfinite(a).all() or (a<0).any() or (a>1).any()):
        raise ValueError('Invalid managed overlap gains, powers, duties or public domains')
    received=np.where((channels==channel)[:,None],g*power[None],0.)
    threshold=10**(CS_THRESHOLD_DBM/10);busy=np.zeros(int(domains.max())+1);rf=np.zeros(len(g))
    for domain in np.unique(domains):
        ids=np.flatnonzero(domains==domain);sensed=(received[ids]>=threshold).any(0)
        deferred=float(a[sensed].max()) if sensed.any() else 0.;busy[domain]=deferred
        concurrent=np.maximum(a-deferred,0.)/max(1.-deferred,1e-30)
        concurrent=np.where(sensed,0.,concurrent)
        rf[ids]=(received[ids]*concurrent[None]).sum(1)
    return rf,busy


@dataclass(frozen=True)
class ProbeReport:
    report_id: str
    channel: int
    measured_slots: tuple
    completed_slot: int
    delivered_slot: int
    exposure_mw: tuple
    received_mw: tuple
    observed: tuple
    noise_std_mw: tuple
    receiver_sites: tuple
    payload_bytes: int
    transmitter_ids: tuple = ('managed-0','managed-1')

    def __post_init__(self):
        k=len(self.receiver_sites)
        if (len(set(self.receiver_sites))!=k or len(self.measured_slots)!=3
                or np.shape(self.exposure_mw)!=(3,2) or np.shape(self.received_mw)!=(3,k)
                or np.shape(self.observed)!=(3,k) or np.shape(self.noise_std_mw)!=(k,)):
            raise ValueError('Probe report columns must match receiver_sites order')


class ProbeMailbox:
    """Only delivered receipts are shared with the predictor, never a runtime."""
    def __init__(self):self._clock=-1;self._rows=[];self._last_drain=-1

    def _advance(self,slot):
        slot=_integer(slot,'mailbox slot')
        if slot<self._clock:raise ValueError('Mailbox clock moved backwards')
        self._clock=slot

    def _publish(self,report):
        if not isinstance(report,ProbeReport) or report.delivered_slot!=self._clock:
            raise ValueError('Only an actually delivered current-slot report may be published')
        self._rows.append(report)

    def drain(self,now):
        now=_integer(now,'mailbox query slot')
        if now>self._clock or now<self._last_drain:raise ValueError('Unavailable/future or reversed mailbox clock')
        ready=[r for r in self._rows if r.delivered_slot<=now]
        self._rows=[r for r in self._rows if r.delivered_slot>now];self._last_drain=now
        return ready


@dataclass(frozen=True)
class ManagedSlot:
    extra_interference_mw: object
    reserved_airtime: object
    blackout: object


class ProbeRuntime:
    """Private stationary coupling, channel resources and delayed report jobs."""
    @classmethod
    def from_world(cls,world,spec,mailbox,*,measurement_seed=0,site_params=None):
        from ..twin.warehouse import make_warehouse
        from ..native.world import _path_losses
        sp=site_params or {};scene=make_warehouse(world.seed,n_robots=1,rack_scale=sp.get('rack_scale',1.))
        scene.pl_coef=sp.get('pl_coef',scene.pl_coef);scene.shadow_db=sp.get('shadow_db',scene.shadow_db)
        aps5=[a for a in scene.aps if a.band=='5g'];aps2=[a for a in scene.aps if a.band=='2g']
        if not np.array_equal(world.sites,np.array([[a.x,a.y] for a in aps5])):raise ValueError('Managed RF and native AP geometry differ')
        tx=np.asarray(spec.positions_m)
        pl=_path_losses(scene,aps5,aps2,tx[:,0],tx[:,1])[:,:,1].T
        gain=10**(-pl.astype(float)/10)
        return cls(world,spec,mailbox,gain,measurement_seed=measurement_seed)

    def __init__(self,world,spec,mailbox,coupling,*,measurement_seed=0):
        if not isinstance(spec,PublicProbeSpec) or not isinstance(mailbox,ProbeMailbox):raise TypeError('Use a public spec and separate mailbox')
        self.spec=spec;self.mailbox=mailbox;self._seed=_integer(measurement_seed,'measurement RNG seed')
        self._g=np.array(coupling,dtype=float,copy=True);self._s=len(world.sites)
        if self._g.shape!=(self._s,2) or not np.isfinite(self._g).all() or (self._g<0).any():raise ValueError('Invalid private coupling labels')
        self._selected=np.flatnonzero(np.asarray(world.channel)[:,1]==spec.channel)
        self._channels=np.asarray(world.channel)[:,1].copy()
        self._g[np.asarray(world.channel)[:,1]!=spec.channel]=0.
        self._domains=np.asarray(world.domain)[:,1].copy();self._avail=np.asarray(world.avail,dtype=float).copy()
        self._noise=10**(np.asarray(world.noise_db,dtype=float)/10)
        self._sigma=np.full(self._s,self._noise[1]*spec.noise_floor_fraction)
        self._world_seed=world.seed;self._clock=-1;self._captured=-1;self._cycles={};self._transport=[];self._delivery=[]
        self._reservation=np.zeros(len(self._avail));self._records=[];self._delivered=0;self._bytes=0
        self._requested=0.;self._allocated=0.;self._capture_count=0;self._report_slots=[]
        self._bulk_generated=0.;self._bulk_overflow=0.
        self._control_allocator_slots=0

    def interference(self,slot):
        rf,_=managed_overlap(self._g,self.spec.power_mw,self.spec.activity(slot),self._domains,self._channels,self.spec.channel)
        out=np.zeros((self._s,3));out[:,1]=rf;return out

    def _base_reservation(self,slot):
        reserve=np.zeros_like(self._avail)
        _,busy=managed_overlap(self._g,self.spec.power_mw,self.spec.activity(slot),self._domains,self._channels,self.spec.channel)
        reserve[:len(busy)]=busy
        if self.spec.phase(slot) is not None:reserve[np.unique(self._domains[self._selected])]+=self.spec.capture_airtime_fraction
        return reserve

    def advance(self,slot):
        slot=_integer(slot,'native slot')
        if slot==self._clock:return
        if slot!=self._clock+1:raise ValueError('Advance the private probe runtime once per native slot')
        self._clock=slot;self.mailbox._advance(slot)
        base=self._base_reservation(slot);self._requested+=float(base.sum())
        self._reservation=np.minimum(base,self._avail);left=self._avail-self._reservation
        fraction=(self.spec.report_bytes*8/(self.spec.report_rate_mbps*1e6)+100e-6)/.01
        for job in self._transport:
            if job['complete']>slot:continue
            for site in sorted(tuple(job['remaining'])):
                domain=self._domains[site]
                if fraction>left[domain]+1e-12:continue
                left[domain]-=fraction;self._reservation[domain]+=fraction
                job['remaining'].remove(site);job['last_service']=slot
                self._bytes+=self.spec.report_bytes;self._report_slots.append({'slot':slot,'site':site,'bytes':self.spec.report_bytes,'airtime_fraction':fraction})
            if not job['remaining']:
                job['deliver']=job['last_service']+1+self.spec.report_delay_slots;self._delivery.append(job)
        self._transport=[j for j in self._transport if j['remaining']]
        self._allocated+=float(self._reservation.sum())
        for job in list(self._delivery):
            if job['deliver']>slot:continue
            report=ProbeReport(job['id'],self.spec.channel,job['slots'],job['complete'],slot,
                job['exposure'],tuple(tuple(row[i] for i in self._selected) for row in job['received']),
                tuple(tuple(row[i] for i in self._selected) for row in job['observed']),
                tuple(float(self._sigma[i]) for i in self._selected),tuple(int(i) for i in self._selected),
                self.spec.report_bytes*len(self._selected))
            self.mailbox._publish(report);self._records.append(report);self._delivered+=1;self._delivery.remove(job)

    def capture(self,slot,ambient_mw):
        if slot!=self._clock or slot<=self._captured:raise ValueError('Capture must occur once after current-slot advance')
        self._captured=slot;phase=self.spec.phase(slot)
        if phase is None:return
        ambient=np.asarray(ambient_mw,dtype=float)
        if ambient.shape!=(self._s,3) or not np.isfinite(ambient).all() or (ambient<0).any():raise ValueError('Native ambient N+I dimensions/units invalid')
        cycle,part=phase;start=self.spec.probe_starts[cycle]
        row=self._cycles.setdefault(cycle,{'received':[[None]*self._s for _ in range(3)],'observed':[[False]*self._s for _ in range(3)]})
        # Captures intentionally observe the TX windows, including sensed TXs;
        # packet reception instead uses conditional RF outside CS deferral.
        truth=ambient[:,1]+self._g@(np.asarray(self.spec.power_mw)*self.spec.activity(slot))
        for site in self._selected:
            rng=np.random.default_rng(np.random.SeedSequence([self._seed,cycle,part,int(site),2_019_337]))
            missing=rng.random()<self.spec.missing_probability
            measured=float(truth[site]+rng.normal()*self._sigma[site])
            if not missing and math.isfinite(measured) and measured>0:
                row['received'][part][site]=measured;row['observed'][part][site]=True
        self._capture_count+=1
        if part==2:
            job={'id':f'probe-{cycle}-{start}','slots':tuple(range(start,start+3)),'complete':slot+1,
                 'exposure':tuple(tuple(np.asarray(self.spec.power_mw)*self.spec.activity(t)) for t in range(start,start+3)),
                 'received':tuple(tuple(r) for r in row['received']),'observed':tuple(tuple(r) for r in row['observed']),
                 'remaining':set(int(i) for i in self._selected),'last_service':slot+1}
            if self._selected.size:
                self._transport.append(job)
                self._requested+=len(self._selected)*(self.spec.report_bytes*8/(self.spec.report_rate_mbps*1e6)+100e-6)/.01
            del self._cycles[cycle]

    def step(self,slot,ambient_mw):
        self.advance(slot);self.capture(slot,ambient_mw);return self.effect(slot)

    def effect(self,slot):
        if slot!=self._clock:raise ValueError('Resources are available only for current native slot')
        blackout=np.zeros((self._s,3),dtype=bool)
        if self.spec.shared_receiver_blackout and self.spec.phase(slot) is not None:blackout[self._selected,1]=True
        return ManagedSlot(self.interference(slot),self._reservation.copy(),blackout)

    def summary(self):
        return {'spec_sha256':self.spec.digest(),'measurement_seed':self._seed,'world_seed':self._world_seed,
            'simulation_only':True,'stationary_native_coupling':True,'coupling_used_as_policy_input':False,
            'captured_slots':self._capture_count,'cycles_delivered':self._delivered,'report_bytes_transmitted':self._bytes,
            'report_transport_slots':list(self._report_slots),'requested_domain_slot_airtime':self._requested,
            'allocated_domain_slot_airtime':self._allocated,'pending_transport_cycles':len(self._transport),
            'pending_delivery_cycles':len(self._delivery),'known_measurement_sigma_mw':self._sigma.tolist(),
            'control_allocator':'per_retry_band_domain_every_managed_slot',
            'control_allocator_slots':self._control_allocator_slots,
            'assumptions':['Aligned nested duty windows; sensed managed TXs defer native domains and add no concurrent RF.',
                'Nonsensed TX power is conditioned on the remaining time outside the sensed union.',
                'Raw captures observe all managed source power; they differ from concurrent packet RF.',
                'Reports use explicit fixed-rate reliable transport after actual per-domain reservation.',
                'Dedicated capture monitor unless shared_receiver_blackout is enabled; not hardware-qualified.',
                'Capture background is actual native fleet N+I; rank does not establish stationarity or uncertainty.']}
