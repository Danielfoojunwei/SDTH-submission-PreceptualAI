"""Dynamic-axis public report reader and matched spatial hierarchy examples.

Only public device reports and CAD vertex geometry are read. Private rendered
channels, XML materials, condition IDs and solver seeds never enter the model
tensor dictionaries. Encoder events use residual dB; query targets retain raw
reported RSSI intervals. No event-time/packet claim accompanies static RT data.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..spectral_scene.operators import fspl_received_power_dbm, link_geometry_features, query_physics_basis
from ..spectral_scene.data import _read_cad_box
from . import simulation as sim


def immutable(value,dtype=None):
    a=np.array(value,dtype=dtype,copy=True)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def array_digest(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(sim.encoded({'dtype':a.dtype.str,'shape':list(a.shape)})+a.tobytes()).hexdigest()


def _source_name(path):
    path=Path(path)
    return str(path.relative_to(sim.ROOT)) if path.is_relative_to(sim.ROOT) else str(path)


@dataclass(frozen=True)
class PublicScene:
    stage:str
    index:int
    state:int
    layout_id:str
    case_id:str
    ap_xyz:np.ndarray
    rx_xyz:np.ndarray
    dimensions:np.ndarray
    cad_boxes:np.ndarray
    carriers:np.ndarray
    tx_power_dbm:float
    noise_floor_dbm:np.ndarray
    rssi_dbm:np.ndarray
    heard:np.ndarray
    censored:np.ndarray
    censor_upper_dbm:np.ndarray
    quantization_interval_dbm:np.ndarray
    detection_threshold_dbm:np.ndarray
    source_files_sha256:dict

    def __post_init__(self):
        if (self.stage not in sim.COUNTS or type(self.index) is not int or self.index<0
                or type(self.state) is not int or self.state not in sim.STATES
                or not self.layout_id or not self.case_id or not math.isfinite(self.tx_power_dbm)):
            raise ValueError('Explicit public scene audit metadata and known TX required')
        ap=np.asarray(self.ap_xyz);rx=np.asarray(self.rx_xyz);dims=np.asarray(self.dimensions)
        if (ap.ndim!=2 or ap.shape[1:]!=(3,) or len(ap) not in (4,8) or rx.shape!=(48,3)
                or dims.shape!=(3,) or not all(np.isfinite(a).all() for a in (ap,rx,dims)) or np.any(dims<=0)):
            raise ValueError('Registered dynamic AP axis and48 finite public receiver XYZ poses required')
        shape=(48,len(ap),3,2,2)
        for name in ('heard','censored'):
            a=np.asarray(getattr(self,name))
            if a.shape!=shape or a.dtype!=bool:raise ValueError('Observed Boolean report axes required')
        if np.any(self.heard&self.censored):raise ValueError('Heard and censored events must be disjoint')
        if np.any(np.asarray(self.heard).sum(axis=1)>sim.TOP_K):raise ValueError('Reported heard list exceeds its per-band top-K cap')
        for name,mask in (('rssi_dbm',self.heard),('censor_upper_dbm',self.censored)):
            a=np.asarray(getattr(self,name))
            if a.shape!=shape or not np.isfinite(a[mask]).all():raise ValueError('Known report values/bounds must be finite')
        intervals=np.asarray(self.quantization_interval_dbm);floor=np.asarray(self.detection_threshold_dbm)
        if (intervals.shape!=shape+(2,) or floor.shape!=shape or not np.isfinite(floor).all()
                or not np.isfinite(intervals[self.heard]).all()
                or np.any(intervals[...,0][self.heard]>=intervals[...,1][self.heard])
                or np.any(self.rssi_dbm[self.heard]<intervals[...,0][self.heard])
                or np.any(self.rssi_dbm[self.heard]>intervals[...,1][self.heard])
                or np.any(intervals[...,1][self.heard]<floor[self.heard])):
            raise ValueError('Actual quantization intervals and public detector floors required')
        if (np.asarray(self.carriers).shape!=(3,) or not np.array_equal(self.carriers,sim.BANDS_HZ)
                or np.asarray(self.noise_floor_dbm).shape!=(2,) or not np.isfinite(self.noise_floor_dbm).all()):
            raise ValueError('Explicit carrier and receiver noise metadata required')
        boxes=np.asarray(self.cad_boxes)
        if boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or not np.isfinite(boxes).all() or np.any(boxes[:,1]<boxes[:,0]):
            raise ValueError('Public CAD AABB vertex bounds required')
        for f in fields(self):
            value=getattr(self,f.name)
            if isinstance(value,np.ndarray):object.__setattr__(self,f.name,immutable(value))
        object.__setattr__(self,'source_files_sha256',MappingProxyType(dict(self.source_files_sha256)))


def load_scene(folder,*,expected_spec=None,source_sha256=None,registry_sha256=None):
    """Read reported NPZ and CAD vertices; never open XML/private channel files."""
    folder=Path(folder)
    if folder.is_symlink() or not folder.is_dir():raise ValueError('Actual immutable scene folder required')
    receipt_path=folder/'receipt.json';receipt=json.loads(receipt_path.read_text())
    case=receipt.get('case',{})
    spec=sim.case_spec(case.get('stage'),case.get('index'),case.get('state'))
    if expected_spec is not None and spec!=expected_spec:raise RuntimeError('Scene differs from admitted exact case')
    if (receipt.get('case_id')!=spec.case_id or receipt.get('layout_id')!=spec.layout_id
            or receipt.get('public_geometry_sha256')!=spec.geometry_sha256
            or receipt.get('public_geometry')!=json.loads(sim.encoded(sim.public_geometry(spec.geometry)))
            or receipt.get('known_tx_power_dbm')!=spec.tx_power_dbm
            or receipt.get('real_device_measurements') is not False
            or receipt.get('unresolved_paths_are_censored') is not False):
        raise RuntimeError('Public scene identity or observation scope changed')
    for key,expected in (('source_receipt_sha256',source_sha256),('registry_sha256',registry_sha256)):
        if expected is not None and receipt.get(key)!=expected:raise RuntimeError('Public scene source binding changed')
    sources={_source_name(receipt_path):sim.digest(receipt_path)}
    public=folder/'public_views.npz'
    if public.is_symlink() or sim.digest(public)!=receipt['files_sha256'].get('public_views.npz'):
        raise RuntimeError('Reported public observations changed')
    sources[_source_name(public)]=sim.digest(public)
    boxes=[]
    mesh_names=sorted(name for name in receipt['files_sha256'] if name.startswith('scene/meshes/') and name.endswith('.ply'))
    if len(mesh_names)!=8:raise RuntimeError('Exact public scene mesh membership required')
    for name in mesh_names:
        relative=Path(name);path=folder/relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or sim.digest(path)!=receipt['files_sha256'][name]:
            raise RuntimeError('Public CAD vertex source changed')
        boxes.append(_read_cad_box(path));sources[_source_name(path)]=sim.digest(path)
    names=('rssi_dbm','heard','censored','censor_upper_dbm','quantization_interval_dbm',
           'reported_detection_threshold_dbm','noise_floor_dbm','ap_positions_m','rx_positions_m','bbox_dims_m','carrier_hz')
    with np.load(public,allow_pickle=False) as archive:values={name:archive[name].copy() for name in names}
    if (not np.allclose(values['ap_positions_m'],spec.geometry.ap_positions_m,rtol=0,atol=2e-5)
            or not np.allclose(values['rx_positions_m'],spec.geometry.rx_positions_m,rtol=0,atol=2e-5)
            or not np.array_equal(values['bbox_dims_m'],(spec.geometry.width_m,spec.geometry.depth_m,spec.geometry.ceiling_m))):
        raise RuntimeError('Public numeric positions/dimensions changed')
    return PublicScene(spec.stage,spec.index,spec.state,spec.layout_id,spec.case_id,
        values['ap_positions_m'],values['rx_positions_m'],values['bbox_dims_m'],np.asarray(boxes),values['carrier_hz'],
        spec.tx_power_dbm,values['noise_floor_dbm'],values['rssi_dbm'],values['heard'],values['censored'],
        values['censor_upper_dbm'],values['quantization_interval_dbm'],values['reported_detection_threshold_dbm'],sources)


def fit_calibration(scenes):
    """TRAIN-only relative median offsets from co-located same-direction reports."""
    scenes=tuple(scenes)
    if not scenes or any(s.stage!='training' for s in scenes) or len({s.case_id for s in scenes})!=len(scenes):
        raise ValueError('Distinct TRAIN scenes only for receiver calibration')
    pairs=[];widths={0:[],1:[]};sources={}
    for s in scenes:
        sources.update(s.source_files_sha256)
        for pose in sim.CONTEXT_POSES:
            for ap in range(len(s.ap_xyz)):
                for band in (0,1):
                    for dev in (0,1):
                        if s.heard[pose,ap,band,0,dev]:
                            interval=s.quantization_interval_dbm[pose,ap,band,0,dev]
                            widths[dev].append(float(interval[1]-interval[0]))
                    if s.heard[pose,ap,band,0,0] and s.heard[pose,ap,band,0,1]:
                        pairs.append({'case_id':s.case_id,'stage':s.stage,'pose':pose,'ap':ap,'band':band,
                            'direction':0,'reference_dbm':float(s.rssi_dbm[pose,ap,band,0,0]),
                            'device_dbm':float(s.rssi_dbm[pose,ap,band,0,1]),
                            'reference_interval':[float(x) for x in s.quantization_interval_dbm[pose,ap,band,0,0]],
                            'device_interval':[float(x) for x in s.quantization_interval_dbm[pose,ap,band,0,1]]})
    offset=float(np.median([p['device_dbm']-p['reference_dbm'] for p in pairs])) if pairs else 0.
    return {'schema':1,'reference_device':0,'absolute_hardware_offset_identified':False,
        'reference_gauge':'device0 reported RSSI; zero reference offset assumed, not physically identified',
        'estimates':{'0':{'offset_db':0.,'count':len(widths[0]),
                           'quantization_db':float(np.median(widths[0])) if widths[0] else 0.,'unknown':False},
                     '1':{'offset_db':offset,'count':len(pairs),
                           'quantization_db':float(np.median(widths[1])) if widths[1] else 0.,'unknown':not bool(pairs)}},
        'paired_reports':pairs,'source_files_sha256':dict(sorted(sources.items())),
        'query_or_held_band_or_true_offset_used':False,
        'paired_same_direction':True,'scope':'offline full TRAIN context reports; shared across schedules'}


def context_schedule(number_aps,regime):
    if type(number_aps) is not int or number_aps not in (4,8) or regime not in ('full','sparse'):
        raise ValueError('Registered AP count and full/sparse budget required')
    rows=[]
    for ap in range(number_aps):
        poses=sim.CONTEXT_POSES if number_aps==4 else tuple(p for i,p in enumerate(sim.CONTEXT_POSES) if (i+ap)%2==0)
        if regime=='sparse':poses=poses[::2]
        rows.extend((pose,ap,band) for pose in poses for band in (0,1))
    return tuple(sorted(rows))


def _estimate(calibration,device):
    try:estimate=calibration['estimates'][str(device)]
    except (KeyError,TypeError):raise ValueError('A TRAIN-bound frontend calibration is required') from None
    if (type(estimate['count']) is not int or estimate['count']<0
            or not math.isfinite(estimate['offset_db']) or not math.isfinite(estimate['quantization_db'])
            or estimate['quantization_db']<0 or type(estimate['unknown']) is not bool):
        raise ValueError('Malformed context-estimated receiver calibration')
    return estimate


def _rows(scene,triples,direction,device,calibration):
    estimate=_estimate(calibration,device)
    index=np.asarray(triples,dtype='i8');pose,ap,band=index.T
    rx=scene.rx_xyz[pose];tx=scene.ap_xyz[ap];frequency=scene.carriers[band]
    q=link_geometry_features(rx,tx,frequency,direction,scene.dimensions,scene.cad_boxes,
        calibration_pair_count=estimate['count'],calibration_unknown=estimate['unknown'],
        quantization_width_db=estimate['quantization_db'],noise_floor_dbm=float(scene.noise_floor_dbm[device]),
        tx_power_dbm=scene.tx_power_dbm)
    fspl=fspl_received_power_dbm(rx,tx,frequency,scene.tx_power_dbm).astype('f4')
    offset=np.full(len(index),estimate['offset_db'],dtype='f4');reference=fspl+offset
    h=scene.heard[pose,ap,band,direction,device];c=scene.censored[pose,ap,band,direction,device]
    center=scene.rssi_dbm[pose,ap,band,direction,device]
    lo=scene.quantization_interval_dbm[pose,ap,band,direction,device,0]
    hi=scene.quantization_interval_dbm[pose,ap,band,direction,device,1]
    bound=scene.censor_upper_dbm[pose,ap,band,direction,device]
    detector=scene.detection_threshold_dbm[pose,ap,band,direction,device]
    context=np.column_stack((np.where(h,(center-reference)/30.,0.),np.where(c,(bound-reference)/30.,0.),h,c,q)).astype('f4')
    lower=np.where(h,np.maximum(lo,detector)-reference,-np.inf).astype('f4')
    upper=np.where(h,hi-reference,np.where(c,bound-reference,np.inf)).astype('f4')
    view={'context':context,'known':h|c,'scheduled':np.ones(len(index),dtype=bool),'ap':ap.astype('i8'),
        'rx_xyz':rx.astype('f4'),'lower':lower,'upper':upper,'floor':(detector-reference).astype('f4'),
        'fspl':fspl,'offset':offset}
    target={'center_dbm':np.where(h,center,0.).astype('f4'),'lower_dbm':np.where(h,lo,0.).astype('f4'),
        'upper_dbm':np.where(h,hi,0.).astype('f4'),'heard':h,'censored':c,
        'censor_upper_dbm':np.where(c,bound,0.).astype('f4'),'detection_threshold_dbm':detector.astype('f4')}
    query={'q':q,'ap':ap.astype('i8'),'rx_xyz':rx.astype('f4'),'basis':query_physics_basis(q),
           'fspl':fspl,'offset':offset}
    ids=tuple(f'{scene.case_id}:pose{p}:ap{s}:band{b}:dir{direction}:dev{device}' for p,s,b in triples)
    return ({k:immutable(v) for k,v in view.items()}, {k:immutable(v) for k,v in query.items()},
            {k:immutable(v) for k,v in target.items()}, ids)


def make_example(scene,calibration,*,regime='full',held_band=False,context_scene=None):
    if not isinstance(scene,PublicScene) or type(held_band) is not bool:
        raise TypeError('Explicit reported public scene and held-band flag required')
    if held_band and scene.stage in ('training','validation'):raise ValueError('Held6GHz cannot enter TRAIN or VAL')
    context_scene=scene if context_scene is None else context_scene
    if (not isinstance(context_scene,PublicScene) or scene.layout_id!=context_scene.layout_id
            or not np.array_equal(scene.ap_xyz,context_scene.ap_xyz) or not np.array_equal(scene.rx_xyz,context_scene.rx_xyz)
            or scene.tx_power_dbm!=context_scene.tx_power_dbm):
        raise ValueError('Stale-state diagnostic must retain exact public geometry/TX')
    rows=context_schedule(len(scene.ap_xyz),regime)
    a,_,_,aid=_rows(context_scene,rows,0,0,calibration)
    b,_,_,bid=_rows(context_scene,rows,1,1,calibration)
    target_rows=tuple((pose,ap,band) for pose in sim.QUERY_POSES for ap in range(len(scene.ap_xyz))
                      for band in ((2,) if held_band else (0,1)))
    _,query,target,qid=_rows(scene,target_rows,0,0,calibration)
    # Only geometry/calibration fields from query are returned as model input;
    # query scan cutoffs and reported values remain target-side quantities.
    binding={'stage':scene.stage,'index':scene.index,'state':scene.state,'condition':scene.state,
        'layout_id':scene.layout_id,'case_id':scene.case_id,'context_case_id':context_scene.case_id,
        'regime':regime,'held_band':held_band,'stale_requested':context_scene.case_id!=scene.case_id,
        'view_ids':{'a':{'direction':0,'device':0},'b':{'direction':1,'device':1}},
        'context_measurement_ids_a':list(aid),'context_measurement_ids_b':list(bid),'query_measurement_ids':list(qid),
        'context_poses':sorted({x[0] for x in rows}),'query_poses':list(sim.QUERY_POSES),
        'source_files_sha256':dict(scene.source_files_sha256)|dict(context_scene.source_files_sha256),
        'context_event_units':'residual dB after public FSPL and TRAIN-estimated relative offset',
        'target_units':'raw device0 reported RSSI; quantized intervals and actual censor bounds',
        'scheduled_attempts_per_view':len(rows),'aps':len(scene.ap_xyz),
        'known_a':int(a['known'].sum()),'heard_a':int((a['context'][:,2]>.5).sum()),
        'censored_a':int((a['context'][:,3]>.5).sum()),'unknown_a':int((~a['known']).sum()),
        'query_heard':int(target['heard'].sum()),'query_censored':int(target['censored'].sum()),
        'query_unknown':int((~(target['heard']|target['censored'])).sum()),
        'temporal_or_network_delivery_claimed':False,
        'query_features_use_actual_target_topK_cutoff':False}
    for view,ids in ((a,aid),(b,bid)):
        physical={x.rsplit(':dir',1)[0] for x in ids};target_physical={x.rsplit(':dir',1)[0] for x in qid}
        if physical&target_physical:raise RuntimeError('A physical context link overlaps a query target')
    result={'a':a,'b':b,'query':query,'target':target,'ap_xyz':immutable(scene.ap_xyz,'f4'),'binding':binding}
    binding['model_input_sha256']={name:{k:array_digest(v) for k,v in result[name].items()}
                                    for name in ('a','b','query')}
    binding['model_input_sha256']['ap_xyz']=array_digest(result['ap_xyz'])
    binding['target_sha256']={k:array_digest(v) for k,v in target.items()}
    return result
