"""Explicit physical units and deterministic service constraints.

These helpers express established relationships, not an architectural novelty
or a learned-performance result. An unavailable interval [a,b) contains integer
attempt slots a,...,b-1. A released command r has deadline window [r,r+D).
Only a window fully covered by known unavailability is a guaranteed miss;
unknown fading, interference and collisions may add further misses.
"""
from dataclasses import dataclass
import numpy as np


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f'{name} must be an integer number of slots')
    value=int(value)
    if positive and value<=0:raise ValueError(f'{name} must be positive')
    return value


def merge_unavailable(intervals):
    """Merge overlapping AND adjacent half-open integer intervals exactly."""
    rows=[]
    for pair in intervals:
        if len(pair)!=2:raise ValueError('Each unavailable interval needs two endpoints')
        start,end=(_integer(pair[0],'start'),_integer(pair[1],'end'))
        if end<start:raise ValueError('Unavailable interval ends before it starts')
        if end>start:rows.append((start,end))
    rows.sort();merged=[]
    for start,end in rows:
        if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    return tuple(merged)


def _forced_ranges(intervals,phase,period_slots,deadline_slots,start_slot,stop_slot):
    period=_integer(period_slots,'period_slots',positive=True)
    deadline=_integer(deadline_slots,'deadline_slots',positive=True)
    phase=_integer(phase,'phase')%period
    start=None if start_slot is None else _integer(start_slot,'start_slot')
    stop=None if stop_slot is None else _integer(stop_slot,'stop_slot')
    if start is not None and stop is not None and stop<start:
        raise ValueError('Observation window ends before it starts')
    merged=merge_unavailable(intervals);ranges=[]
    for a,b in merged:
        if start is not None:a=max(a,start)
        if stop is not None:b=min(b,stop)
        first=phase+((a-phase+period-1)//period)*period
        last=b-deadline
        if first<=last:ranges.append((first,last,period))
    return ranges,merged,phase,period,deadline,start,stop


def _pending_forced(merged,pending_release_slots,phase,period,deadline,start,stop):
    releases=set()
    pending=list(pending_release_slots)
    if pending and start is None:
        raise ValueError('Known pending packets require an explicit start_slot')
    for value in pending:
        release=_integer(value,'pending release')
        if release>=start or release+deadline<=start:
            raise ValueError('A supplied pending command must precede start and remain unexpired')
        if (release-phase)%period:
            raise ValueError('Pending release does not match the configured traffic phase')
        end=release+deadline
        if stop is not None and end>stop:continue
        if any(a<=start and end<=b for a,b in merged):releases.add(release)
    return releases


def forced_deadline_release_slots(intervals,*,phase=0,period_slots=5,deadline_slots=5,
                                  start_slot=None,stop_slot=None,pending_release_slots=()):
    """Release slots of commands guaranteed to expire without service.

    With start/stop, count only new releases inside that window whose deadlines
    also expire within it. Explicit pending releases can precede start: because
    they are already known undelivered, only their REMAINING window must be
    unavailable. Other pre-window packets are not silently counted as pending.
    Duplicate interval coverage and duplicate pending releases count once.
    """
    ranges,merged,phase,period,deadline,start,stop=_forced_ranges(
        intervals,phase,period_slots,deadline_slots,start_slot,stop_slot)
    pending=_pending_forced(merged,pending_release_slots,phase,period,deadline,start,stop)
    releases=[np.arange(a,b+1,p,dtype=np.int64) for a,b,p in ranges]
    if pending:releases.append(np.asarray(sorted(pending),dtype=np.int64))
    return np.sort(np.concatenate(releases)) if releases else np.empty(0,dtype=np.int64)


def forced_deadline_misses(intervals,*,phase=0,period_slots=5,deadline_slots=5,
                           start_slot=None,stop_slot=None,pending_release_slots=()):
    """Exact guaranteed count, without allocating one element per packet.

    This is a LOWER bound on absolute misses, not total predicted loss and not
    a bound on the difference between two policies. It assumes each command's
    attempt window is its configured relative deadline. The default matches
    the native kernel's five-slot period and five-slot deadline.
    """
    ranges,merged,phase,period,deadline,start,stop=_forced_ranges(
        intervals,phase,period_slots,deadline_slots,start_slot,stop_slot)
    pending=_pending_forced(merged,pending_release_slots,phase,period,deadline,start,stop)
    return sum((b-a)//p+1 for a,b,p in ranges)+len(pending)


def _real_array(value,name,*,allow_negative_infinity=False):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'iuf':raise TypeError(f'{name} must be real numeric values')
    array=raw.astype(np.float64)
    valid=np.isfinite(array)|(np.isneginf(array) if allow_negative_infinity else False)
    if not valid.all():raise ValueError(f'{name} contains nonfinite values')
    return array


def _scalar_or_array(value):
    return value.item() if np.ndim(value)==0 else value


def dbm_to_mw(dbm):
    """Total power dBm -> mW. -inf dBm denotes exactly zero power."""
    x=_real_array(dbm,'power dBm',allow_negative_infinity=True)
    with np.errstate(over='ignore'):out=np.power(10.,x/10)
    if not np.isfinite(out).all():raise ValueError('Power conversion exceeds finite mW range')
    return _scalar_or_array(out)


def mw_to_dbm(mw):
    """Nonnegative total mW -> dBm; zero maps to -inf dBm."""
    x=_real_array(mw,'power mW')
    if (x<0).any():raise ValueError('Physical power cannot be negative')
    with np.errstate(divide='ignore'):out=10*np.log10(x)
    return _scalar_or_array(out)


def db_to_power_ratio(db):
    """A POWER ratio uses 10log10, not the amplitude-ratio convention."""
    return dbm_to_mw(db)


def power_ratio_to_db(ratio):
    return mw_to_dbm(ratio)


def sinr_linear(signal_mw,noise_mw,interference_mw=0.):
    """S/(N+I), with all THREE inputs in the same linear mW units."""
    signal,noise,interference=np.broadcast_arrays(
        _real_array(signal_mw,'signal mW'),_real_array(noise_mw,'noise mW'),
        _real_array(interference_mw,'interference mW'))
    if (signal<0).any() or (noise<0).any() or (interference<0).any():
        raise ValueError('Signal, noise and interference must be nonnegative mW')
    denominator=noise+interference
    if not np.isfinite(denominator).all() or (denominator<=0).any():
        raise ValueError('Total noise plus interference must be finite and positive')
    with np.errstate(over='ignore'):out=signal/denominator
    if not np.isfinite(out).all():raise ValueError('SINR exceeds finite linear range')
    return _scalar_or_array(out)


def sinr_db(signal_mw,noise_mw,interference_mw=0.):
    return power_ratio_to_db(sinr_linear(signal_mw,noise_mw,interference_mw))


def thermal_noise_dbm(bandwidth_hz,noise_figure_db=0.,temperature_k=290.):
    """Conventional -174 dBm/Hz at 290 K + 10log10(B_Hz) + NF.

    Temperature is scaled as 10log10(T/290). -174 is the conventional rounded
    room-temperature density, not an exact Boltzmann-constant calculation.
    External occupancy/interference is additional; it is not hidden inside NF.
    """
    bandwidth,nf,temp=np.broadcast_arrays(_real_array(bandwidth_hz,'bandwidth Hz'),
        _real_array(noise_figure_db,'noise figure dB'),_real_array(temperature_k,'temperature K'))
    if (bandwidth<=0).any() or (temp<=0).any() or (nf<0).any():
        raise ValueError('Require positive Hz/K and nonnegative conventional noise figure')
    return _scalar_or_array(-174.+10*np.log10(bandwidth)+nf+10*np.log10(temp/290.))


@dataclass(frozen=True)
class AbsoluteServicePrediction:
    expected_misses: object
    maximum_gap_slots: object
    hazard: object = None


@dataclass(frozen=True)
class ServiceAdvantage:
    delta_misses: object
    delta_maximum_gap_slots: object
    delta_hazard: object = None


def project_absolute_service(predicted_misses,predicted_gap_slots,*,forced_miss_floor,
                             released_plus_pending,gap_floor_slots,gap_upper_slots,
                             predicted_hazard=None,certain_failure=None,
                             prediction_kind='absolute'):
    """Project ABSOLUTE inference heads onto supplied deterministic bounds.

    Count capacity includes all released and still-pending commands that could
    contribute a miss. Gap bounds are supplied explicitly: an initial silence
    can predate the prediction window, so window length alone is not an upper
    bound. Inputs use counts and integer-time-slot units; expectations may be
    continuous. Predicted hazards are event probabilities, not logits.

    ``certain_failure`` may identify only an event already physically certain,
    such as a command whose WHOLE remaining deadline window is unavailable.
    A single down radio slot does not imply a whole command must miss.

    This is an elementwise feasibility projection, not a calibrated joint risk
    certificate. It is a NumPy inference/evaluation helper, not a differentiable
    training operation. Never pass paired signed advantages: their negative
    values represent improvement and must remain negative.
    """
    if prediction_kind!='absolute':
        raise ValueError('Absolute physical floors must not be applied to paired advantages')
    misses,floor,capacity=np.broadcast_arrays(_real_array(predicted_misses,'predicted misses'),
        _real_array(forced_miss_floor,'forced miss floor'),
        _real_array(released_plus_pending,'released plus pending count'))
    gap,gap_floor,gap_upper=np.broadcast_arrays(_real_array(predicted_gap_slots,'predicted gap slots'),
        _real_array(gap_floor_slots,'gap floor slots'),_real_array(gap_upper_slots,'gap upper slots'))
    if ((floor<0).any() or (capacity<0).any() or (floor>capacity).any()
        or (floor!=np.floor(floor)).any() or (capacity!=np.floor(capacity)).any()):
        raise ValueError('Miss floor/capacity must be consistent nonnegative integer counts')
    if ((gap_floor<0).any() or (gap_upper<0).any() or (gap_floor>gap_upper).any()
        or (gap_floor!=np.floor(gap_floor)).any() or (gap_upper!=np.floor(gap_upper)).any()):
        raise ValueError('Gap bounds must be consistent nonnegative integer slots')
    hazard=None
    if predicted_hazard is not None:
        hazard=np.clip(_real_array(predicted_hazard,'predicted hazard'),0.,1.)
        if certain_failure is not None:
            certain=np.asarray(certain_failure)
            if certain.dtype.kind!='b':raise TypeError('certain_failure must be a boolean event mask')
            hazard,certain=np.broadcast_arrays(hazard,certain)
            hazard=np.where(certain,1.,hazard)
        hazard=_scalar_or_array(hazard)
    elif certain_failure is not None:
        raise ValueError('A certain-failure mask requires a hazard prediction with defined event axes')
    return AbsoluteServicePrediction(_scalar_or_array(np.clip(misses,floor,capacity)),
                                     _scalar_or_array(np.clip(gap,gap_floor,gap_upper)),hazard)


def paired_service_advantage(intervention,baseline):
    """Signed intervention-minus-baseline differences; never clip below zero."""
    if not isinstance(intervention,AbsoluteServicePrediction) or not isinstance(baseline,AbsoluteServicePrediction):
        raise TypeError('Compute paired advantage from two absolute service predictions')
    hazard=None
    if intervention.hazard is not None and baseline.hazard is not None:
        hazard=_scalar_or_array(np.asarray(intervention.hazard)-np.asarray(baseline.hazard))
    return ServiceAdvantage(_scalar_or_array(np.asarray(intervention.expected_misses)-np.asarray(baseline.expected_misses)),
        _scalar_or_array(np.asarray(intervention.maximum_gap_slots)-np.asarray(baseline.maximum_gap_slots)),hazard)
