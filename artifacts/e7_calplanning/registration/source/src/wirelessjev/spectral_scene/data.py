"""Reported-only static scene views for the spectral scene experiment.

The reader opens public device reports and public CAD vertices. It never opens
physical_channels.npz, CIR/CFR files or material labels. Metadata identities
exist only for audit; model_inputs() is a numeric, explicit whitelist.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .operators import (CONTEXT_FEATURE_NAMES,GEOMETRY_FEATURE_NAMES,PHYSICS_BASIS_NAMES,
    fspl_received_power_dbm,link_geometry_features,query_physics_basis)

CONTEXT_POSES=(0,2,4,6,8,10)
QUERY_POSES=(1,3,5,7,9,11)
TRAIN_BANDS=(0,1)
TRAIN_DEVICES=(0,1,2)
STAGES=('training','validation','development','field_test')


def _common():
    return importlib.import_module('.common',__package__)


def _freeze(value,dtype=None):
    array=np.array(value,dtype=dtype,copy=True)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def _digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


@dataclass(frozen=True,order=True)
class ViewKey:
    band:int
    direction:int
    device:int

    def __post_init__(self):
        if any(type(x) is not int for x in (self.band,self.direction,self.device)) or not 0<=self.band<=2 or self.direction not in (0,1) or not 0<=self.device<=3:
            raise ValueError('A physical view needs registered band/direction/device indices')


VIEW_A=(ViewKey(0,0,0),ViewKey(1,0,0))
VIEW_B=(ViewKey(0,1,1),ViewKey(1,1,1))


@dataclass(frozen=True)
class PublicScene:
    stage:str
    index:int
    scene_id:str  # audit only, never encoded
    role:str
    ap_positions_m:np.ndarray
    rx_positions_m:np.ndarray
    bbox_dims_m:np.ndarray
    cad_boxes_m:np.ndarray
    band_indices:tuple[int,...]
    carrier_hz:np.ndarray
    device_indices:tuple[int,...]
    device_ids:tuple[str,...]
    rssi_dbm:np.ndarray
    heard:np.ndarray
    censored:np.ndarray
    censor_upper_dbm:np.ndarray
    quantization_interval_dbm:np.ndarray
    noise_floor_dbm:np.ndarray
    bandwidth_hz:float
    tx_power_dbm:float
    source_files_sha256:Mapping[str,str]

    def __post_init__(self):
        if self.stage not in STAGES or type(self.index) is not int or self.index<0 or not isinstance(self.scene_id,str) or not self.scene_id:
            raise ValueError('Scene audit identity must be explicit')
        bands,devices=tuple(self.band_indices),tuple(self.device_indices)
        _,expected_bands,expected_devices=_role_axes(self.stage,self.role)
        if bands!=expected_bands or devices!=expected_devices:raise ValueError('Public axes differ from the explicitly authorized role')
        if len(set(bands))!=len(bands) or len(set(devices))!=len(devices) or not bands or not devices:
            raise ValueError('View axes must be distinct and nonempty')
        if not set(bands)<={0,1,2} or not set(devices)<={0,1,2,3}:raise ValueError('Unknown physical view axes')
        if self.stage in ('training','validation') and (not set(bands)<=set(TRAIN_BANDS) or not set(devices)<=set(TRAIN_DEVICES)):
            raise ValueError('Held band/device cannot enter training or validation')
        shape=(12,2,len(bands),2,len(devices))
        for name,dtype in (('heard',bool),('censored',bool)):
            raw=np.asarray(getattr(self,name))
            if raw.shape!=shape or raw.dtype!=np.bool_:raise ValueError('Reported masks must be Boolean with registered axes')
            object.__setattr__(self,name,_freeze(raw,dtype))
        if np.any(self.heard&self.censored):raise ValueError('A point reading cannot also be censored')
        for name in ('rssi_dbm','censor_upper_dbm'):
            raw=np.asarray(getattr(self,name),dtype=float)
            mask=self.heard if name=='rssi_dbm' else self.censored
            if raw.shape!=shape or not np.isfinite(raw[mask]).all():raise ValueError('Reported values/bounds must be finite where known')
            object.__setattr__(self,name,_freeze(np.where(mask,raw,0.),np.float32))
        interval=np.asarray(self.quantization_interval_dbm,dtype=float)
        if (interval.shape!=shape+(2,) or not np.isfinite(interval[self.heard]).all()
                or np.any(interval[...,0][self.heard]>=interval[...,1][self.heard])
                or np.any(self.rssi_dbm[self.heard]<interval[...,0][self.heard])
                or np.any(self.rssi_dbm[self.heard]>interval[...,1][self.heard])):
            raise ValueError('A reported quantization interval must contain its actual point report')
        object.__setattr__(self,'quantization_interval_dbm',_freeze(np.where(self.heard[...,None],interval,0.),np.float32))
        geometry=(('ap_positions_m',(2,3)),('rx_positions_m',(12,3)),('bbox_dims_m',(3,)),
                  ('carrier_hz',(len(bands),)),('noise_floor_dbm',(len(devices),)))
        for name,expected in geometry:
            array=np.asarray(getattr(self,name),dtype=float)
            if array.shape!=expected or not np.isfinite(array).all():raise ValueError('Malformed public geometry/receiver metadata: '+name)
            object.__setattr__(self,name,_freeze(array))
        if np.any(self.bbox_dims_m<=0) or np.any(self.carrier_hz<=0):raise ValueError('Physical dimensions/carriers must be positive')
        boxes=np.asarray(self.cad_boxes_m,dtype=float)
        if boxes.size==0:boxes=np.empty((0,2,3))
        if boxes.ndim!=3 or boxes.shape[1:]!=(2,3) or not np.isfinite(boxes).all() or np.any(boxes[:,1]<boxes[:,0]):
            raise ValueError('Public CAD boxes require ordered XYZ bounds')
        object.__setattr__(self,'cad_boxes_m',_freeze(boxes))
        if len(self.device_ids)!=len(devices) or any(not isinstance(x,str) or not x for x in self.device_ids):raise ValueError('Device metadata must match the selected axes')
        if not math.isfinite(self.bandwidth_hz) or self.bandwidth_hz<=0 or not math.isfinite(self.tx_power_dbm):raise ValueError('Known bandwidth/transmit power required')
        object.__setattr__(self,'band_indices',bands);object.__setattr__(self,'device_indices',devices)
        object.__setattr__(self,'device_ids',tuple(self.device_ids))
        object.__setattr__(self,'source_files_sha256',MappingProxyType(dict(self.source_files_sha256)))

    def axis(self,key:ViewKey):
        if not isinstance(key,ViewKey) or key.band not in self.band_indices or key.device not in self.device_indices:
            raise ValueError('Requested view is not available in this authorized scene role')
        return self.band_indices.index(key.band),key.direction,self.device_indices.index(key.device)


def _role_axes(stage,role):
    if stage not in STAGES:raise ValueError('Unknown stage')
    role=('fit' if stage in ('training','validation') else 'development' if stage=='development' else 'field_test') if role is None else role
    if stage in ('training','validation'):
        if role!='fit':raise ValueError('Training/validation cannot open held-band/device roles')
        return role,TRAIN_BANDS,TRAIN_DEVICES
    if stage=='development':
        if role=='development':return role,TRAIN_BANDS,TRAIN_DEVICES
        if role=='held_band':return role,(0,1,2),TRAIN_DEVICES
        if role=='onboarding':return role,TRAIN_BANDS,(0,1,2,3)
        raise ValueError('Explicit development, held_band or onboarding role required')
    if role!='field_test':raise ValueError('Field objects require the field_test role')
    return role,(0,1,2),(0,1,2,3)


def _read_cad_box(path):
    # Public vertex geometry only. No scene XML/material labels are parsed.
    lines=Path(path).read_text().splitlines()
    if not lines or lines[0]!='ply' or 'format ascii 1.0' not in lines:raise ValueError('Expected the registered ASCII box CAD mesh')
    count=next((int(line.split()[-1]) for line in lines if line.startswith('element vertex ')),None)
    if count is None or 'end_header' not in lines:raise ValueError('Malformed public CAD vertex header')
    start=lines.index('end_header')+1
    vertices=np.array([[float(x) for x in line.split()[:3]] for line in lines[start:start+count]])
    if vertices.shape!=(count,3) or not np.isfinite(vertices).all():raise ValueError('Malformed public CAD vertices')
    return np.stack((vertices.min(0),vertices.max(0)))


def _stage_manifest(c,stage):
    path=c.OUT/'rt_data'/f'{stage}.json'
    if not path.is_file():raise RuntimeError('The immutable RT collector stage receipt is missing')
    manifest=c.read_json(path)
    if (manifest.get('schema')!=1 or manifest.get('stage')!=stage or manifest.get('completed') is not True
            or manifest.get('registry_sha256')!=c.registry_sha256()
            or manifest.get('source_receipt_sha256')!=_digest(c.OUT/'registration/source.json')
            or manifest.get('real_device_measurements') is not False
            or manifest.get('original_service_confirmation_consumed') is not False
            or len(manifest.get('scene_receipts',{}))!=c.protocol()['scene_counts'][stage]):
        raise RuntimeError('RT collector stage source/registry/scope or membership changed')
    return path,manifest


def load_scene(stage,index,*,role=None):
    """Read only a published, authorized new-stage public report cache."""
    role,bands,devices=_role_axes(stage,role)
    c=_common();c.check_sources();c.authorize_stage(stage)
    folder=Path(c.rt_scene_path(stage,index))
    receipt_path=folder/'receipt.json';public_path=folder/'public_views.npz'
    if not receipt_path.is_file() or not public_path.is_file():raise RuntimeError('Collect the authorized RT public cache before reading it')
    receipt=json.loads(receipt_path.read_text())
    stage_path,stage_receipt=_stage_manifest(c,stage)
    expected=c.field_spec(stage,index)
    registered=stage_receipt['scene_receipts'].get(receipt['scene_id'])
    if (receipt['scene_id']!=expected.scene_id or receipt.get('spec_sha256')!=expected.sha256
            or registered!={'path':str(receipt_path.relative_to(c.ROOT)),'sha256':_digest(receipt_path)}):
        raise RuntimeError('Scene receipt differs from the immutable registered collector stage')
    if _digest(public_path)!=receipt['files_sha256']['public_views.npz']:raise RuntimeError('Public reported data bytes changed')
    sources={str(path.relative_to(c.ROOT)):_digest(path) for path in (receipt_path,public_path,stage_path)}
    boxes=[]
    for name,expected in sorted(receipt['files_sha256'].items()):
        relative=Path(name)
        if name.startswith('scene/meshes/') and relative.suffix=='.ply':
            if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe CAD path')
            path=folder/relative
            if path.is_symlink() or _digest(path)!=expected:raise RuntimeError('Public CAD bytes changed')
            boxes.append(_read_cad_box(path));sources[str(path.relative_to(c.ROOT))]=expected
    names=('rssi_dbm','heard','censored','censor_upper_dbm','quantization_interval_dbm')
    with np.load(public_path,allow_pickle=False) as saved:
        # Never request physical_link_solved or any private physics archive.
        selected={name:np.take(np.take(saved[name],bands,axis=2),devices,axis=4) for name in names}
        metadata={name:saved[name].copy() for name in ('ap_positions_m','rx_positions_m')}
        metadata.update(carrier_hz=saved['carrier_hz'][list(bands)].copy(),
            device_ids=tuple(str(x) for x in saved['device_ids'][list(devices)]),
            noise_floor_dbm=saved['noise_floor_dbm'][list(devices)].copy())
    spec=receipt['spec']
    # This whitelist excludes family, seed, obstacle material and receiver offsets.
    dimensions=np.array([spec['width_m'],spec['depth_m'],spec['ceiling_m']],dtype=float)
    scene=PublicScene(stage,index,receipt['scene_id'],role,**metadata,bbox_dims_m=dimensions,
        cad_boxes_m=np.asarray(boxes),band_indices=bands,device_indices=devices,
        bandwidth_hz=float(receipt['bandwidth_hz']),tx_power_dbm=float(receipt['reference_tx_dbm']),
        source_files_sha256=sources,**selected)
    record={'schema':1,'stage':stage,'index':index,'role':role,'scene_id':scene.scene_id,
        'source_files_sha256':sources,'band_indices':list(bands),'device_indices':list(devices),
        'context_poses':list(CONTEXT_POSES),'query_poses':list(QUERY_POSES),
        'context_feature_names':list(CONTEXT_FEATURE_NAMES),'query_feature_names':list(GEOMETRY_FEATURE_NAMES),
        'physics_basis_names':list(PHYSICS_BASIS_NAMES),'private_physics_opened':False,
        'material_seed_or_scene_id_encoded':False,'source_is_simulated':True}
    c.write_once(c.OUT/'data'/f'scene-{stage}-{index:02d}-{role}.json',record)
    return scene


@dataclass(frozen=True)
class DeviceEstimate:
    device:int
    offset_db:float
    pair_count:int
    offset_interval_db:tuple[float,float] | None
    quantization_db:float | None
    known:bool

    def __post_init__(self):
        if type(self.device) is not int or not 0<=self.device<=3 or type(self.pair_count) is not int or self.pair_count<0 or not math.isfinite(self.offset_db) or type(self.known) is not bool:
            raise ValueError('Invalid observed-pair calibration estimate')
        if self.offset_interval_db is not None:
            low,high=self.offset_interval_db
            if not math.isfinite(low) or not math.isfinite(high) or not low<=self.offset_db<=high:raise ValueError('Offset estimate must respect the identified reported-interval intersection')
            object.__setattr__(self,'offset_interval_db',(float(low),float(high)))
        if self.quantization_db is not None and (not math.isfinite(self.quantization_db) or self.quantization_db<=0):raise ValueError('Quantization must be inferred as a positive reported interval width')
        if self.device!=0 and self.known and (self.pair_count==0 or self.offset_interval_db is None):raise ValueError('An unknown device cannot be declared calibrated without actual pairs')


@dataclass(frozen=True)
class FrontendCalibration:
    estimates:Mapping[int,DeviceEstimate]
    paired_reports:tuple[Mapping,...]
    source_files_sha256:Mapping[str,str]
    reference_device:int=0
    absolute_offset_identified:bool=False

    def __post_init__(self):
        estimates=dict(self.estimates)
        if self.reference_device!=0 or self.absolute_offset_identified is not False or 0 not in estimates:
            raise ValueError('Only the explicit device0 reference gauge is supported; absolute bias is not identified')
        if any(key!=value.device for key,value in estimates.items()):raise ValueError('Calibration device identities differ')
        if estimates[0].offset_db!=0.:raise ValueError('Reference offset is a chosen zero gauge, not a registry label')
        object.__setattr__(self,'estimates',MappingProxyType(estimates))
        reports=[]
        for row in self.paired_reports:
            record=dict(row)
            for name in ('reference_interval_dbm','device_interval_dbm'):record[name]=tuple(record[name])
            reports.append(MappingProxyType(record))
        object.__setattr__(self,'paired_reports',tuple(reports))
        object.__setattr__(self,'source_files_sha256',MappingProxyType(dict(self.source_files_sha256)))

    def estimate(self,device):
        return self.estimates.get(device,DeviceEstimate(device,0.,0,None,None,False))


def _paired_reports(scene,device,poses=CONTEXT_POSES):
    if any(p not in CONTEXT_POSES for p in poses):raise ValueError('Onboarding/offset calibration cannot read query-position RF')
    if device==0:return []
    reports=[]
    for pose in poses:
        for ap in range(2):
            for band in TRAIN_BANDS:
                for direction in range(2):
                    b,d,dev=scene.axis(ViewKey(band,direction,device));_,_,reference=scene.axis(ViewKey(band,direction,0))
                    if scene.heard[pose,ap,b,d,dev] and scene.heard[pose,ap,b,d,reference]:
                        actual=scene.quantization_interval_dbm[pose,ap,b,d,dev]
                        anchor=scene.quantization_interval_dbm[pose,ap,b,d,reference]
                        reports.append({'scene_id':scene.scene_id,'stage':scene.stage,'pose':pose,'ap':ap,
                            'band':band,'direction':direction,'device':device,
                            'reference_reported_dbm':float(scene.rssi_dbm[pose,ap,b,d,reference]),
                            'device_reported_dbm':float(scene.rssi_dbm[pose,ap,b,d,dev]),
                            'reference_interval_dbm':[float(x) for x in anchor],
                            'device_interval_dbm':[float(x) for x in actual]})
    return reports


def _estimate(device,reports):
    if not reports:return DeviceEstimate(device,0.,0,None,None,False)
    differences=np.array([r['device_reported_dbm']-r['reference_reported_dbm'] for r in reports])
    lower=max(r['device_interval_dbm'][0]-r['reference_interval_dbm'][1] for r in reports)
    upper=min(r['device_interval_dbm'][1]-r['reference_interval_dbm'][0] for r in reports)
    consistent=lower<=upper+1e-9
    median=float(np.median(differences))
    interval=(float(lower),float(max(lower,upper))) if consistent else None
    offset=float(np.clip(median,*interval)) if interval is not None else median
    quant=float(np.median([r['device_interval_dbm'][1]-r['device_interval_dbm'][0] for r in reports]))
    return DeviceEstimate(device,offset,len(reports),interval,quant,consistent)


def fit_frontends(scenes):
    """Global TRAIN-only relative offsets from co-observed reported device0 pairs.

    No per-scene calibration side channel and no registry/true offset lookup.
    """
    scenes=tuple(scenes)
    if not scenes or any(scene.stage!='training' for scene in scenes):raise ValueError('Frontend fitting is restricted to TRAIN scenes')
    if len({scene.scene_id for scene in scenes})!=len(scenes):raise ValueError('Duplicate calibration scene')
    pairs={device:[] for device in (1,2)};sources={};reference_widths=[];reference_count=0
    for scene in scenes:
        sources.update(scene.source_files_sha256)
        ref=scene.device_indices.index(0)
        for pose in CONTEXT_POSES:
            for band in TRAIN_BANDS:
                b=scene.band_indices.index(band)
                mask=scene.heard[pose,:,b,:,ref]
                intervals=scene.quantization_interval_dbm[pose,:,b,:,ref]
                reference_widths.extend((intervals[...,1]-intervals[...,0])[mask].tolist());reference_count+=int(mask.sum())
        for device in (1,2):pairs[device].extend(_paired_reports(scene,device))
    reference=DeviceEstimate(0,0.,reference_count,(0.,0.),float(np.median(reference_widths)) if reference_widths else None,True)
    estimates={0:reference,**{device:_estimate(device,rows) for device,rows in pairs.items()}}
    return FrontendCalibration(estimates,tuple(row for rows in pairs.values() for row in rows),sources)


def onboard_device(scene,base_calibration,*,device=3,n_pairs=0):
    """Use exactly N DEV context pairs; query positions and band2 stay excluded."""
    if scene.stage not in ('development','field_test') or scene.role not in ('onboarding','field_test') or device!=3:
        raise ValueError('Held-device onboarding requires an authorized DEV/FIELD context role')
    if type(n_pairs) is not int or n_pairs not in (0,1,4,16):raise ValueError('Use a registered onboarding budget 0/1/4/16')
    available=_paired_reports(scene,device)
    if len(available)<n_pairs:raise ValueError('Insufficient genuine context pairs for the requested onboarding budget')
    selected=available[:n_pairs]
    estimates=dict(base_calibration.estimates);estimates[device]=_estimate(device,selected)
    return FrontendCalibration(estimates,base_calibration.paired_reports+tuple(selected),
        dict(base_calibration.source_files_sha256)|dict(scene.source_files_sha256))


def calibration_record(calibration):
    from dataclasses import asdict
    return {'schema':1,'completed':True,'reference_device':0,'absolute_offset_identified':False,
        'offset_gauge':'device0 reported coordinate; not an independently identified hardware bias',
        'estimates':{str(k):asdict(v)|{'offset_interval_db':None if v.offset_interval_db is None else list(v.offset_interval_db)}
                     for k,v in sorted(calibration.estimates.items())},
        'paired_reports':[dict(row)|{name:list(row[name]) for name in ('reference_interval_dbm','device_interval_dbm')}
                          for row in calibration.paired_reports],
        'source_files_sha256':dict(calibration.source_files_sha256),
        'query_reports_or_band2_used':False,'registry_true_offsets_used':False}


def save_calibration(calibration,path=None):
    c=_common();c.check_sources()
    return c.write_once(c.OUT/'data/frontend-training.json' if path is None else Path(path),calibration_record(calibration))


def load_calibration():
    """Load the fixed TRAIN-only calibration, guarding its original reported inputs."""
    c=_common();c.check_sources()
    path=c.OUT/'data/frontend-training.json'
    if not path.is_file():raise RuntimeError('Prepare the published TRAIN calibration first')
    record=c.read_json(path)
    prepared_path=c.OUT/'data/prepared-training.json'
    if prepared_path.exists():
        prepared=c.read_json(prepared_path)
        if prepared.get('calibration_receipt')!=str(path.relative_to(c.ROOT)) or prepared.get('calibration_sha256')!=_digest(path):
            raise RuntimeError('Fixed TRAIN calibration differs from the prepared training identity')
        for stage,name in prepared['stage_receipt_paths'].items():
            if _digest(c.safe_path(name))!=prepared['stage_receipts_sha256'][stage]:raise RuntimeError('Prepared TRAIN/VAL stage receipt changed')
    for name,expected in record['source_files_sha256'].items():
        source=c.safe_path(name)
        if _digest(source)!=expected:raise RuntimeError('Calibration reported-source bytes changed')
    estimates={int(key):DeviceEstimate(**(value|{'offset_interval_db':None if value['offset_interval_db'] is None else tuple(value['offset_interval_db'])}))
               for key,value in record['estimates'].items()}
    cal=FrontendCalibration(estimates,tuple(record['paired_reports']),record['source_files_sha256'])
    if calibration_record(cal)!=record:raise RuntimeError('Frontend calibration schema or scope changed')
    for device in (1,2):
        reports=[r for r in cal.paired_reports if r['device']==device]
        if any(r['stage']!='training' or r['pose'] not in CONTEXT_POSES or r['band'] not in TRAIN_BANDS for r in reports):
            raise RuntimeError('Calibration used non-TRAIN/query/held-band reports')
        if _estimate(device,reports)!=cal.estimates[device]:raise RuntimeError('Calibration estimates differ from their actual reported pairs')
    return cal


def prepare_training():
    """Bind a deterministic TRAIN context-only calibration before any model fit."""
    c=_common();c.check_sources()
    path=c.OUT/'data/frontend-training.json'
    if path.exists():cal=load_calibration()
    else:
        scenes=[load_scene('training',i,role='fit') for i in range(c.protocol()['scene_counts']['training'])]
        cal=fit_frontends(scenes);save_calibration(cal,path)
    receipt={'schema':1,'completed':True,'training_scenes':c.protocol()['scene_counts']['training'],
        'calibration_receipt':str(path.relative_to(c.ROOT)),'calibration_sha256':_digest(path),
        'source_files_sha256':dict(cal.source_files_sha256),
        'reference_gauge':'device0 report coordinate, not independently identified absolute RF bias',
        'context_positions':list(CONTEXT_POSES),'bands_used':[0,1],
        'true_offsets_materials_or_private_channels_used':False}
    stage_manifests={stage:_stage_manifest(c,stage)[0] for stage in ('training','validation')}
    receipt['stage_receipts_sha256']={stage:_digest(path) for stage,path in stage_manifests.items()}
    receipt['stage_receipt_paths']={stage:str(path.relative_to(c.ROOT)) for stage,path in stage_manifests.items()}
    c.write_once(c.OUT/'data/prepared-training.json',receipt)
    return receipt


@dataclass(frozen=True)
class ReportedTargets:
    center_dbm:np.ndarray
    lower_dbm:np.ndarray
    upper_dbm:np.ndarray
    heard:np.ndarray
    censored:np.ndarray
    censor_upper_dbm:np.ndarray

    def __post_init__(self):
        heard=np.asarray(self.heard);censored=np.asarray(self.censored)
        if heard.ndim!=1 or heard.dtype!=np.bool_ or censored.shape!=heard.shape or censored.dtype!=np.bool_ or np.any(heard&censored):
            raise ValueError('Reported target masks must be disjoint Boolean vectors')
        for name,mask in (('center_dbm',heard),('lower_dbm',heard),('upper_dbm',heard),('censor_upper_dbm',censored)):
            value=np.asarray(getattr(self,name),dtype=float)
            if value.shape!=heard.shape or not np.isfinite(value[mask]).all():raise ValueError('Target quantities must be finite where observed')
            object.__setattr__(self,name,_freeze(np.where(mask,value,0.),np.float32))
        if np.any(self.lower_dbm[heard]>=self.upper_dbm[heard]):raise ValueError('Reported point intervals must have positive width')
        if np.any(self.center_dbm[heard]<self.lower_dbm[heard]) or np.any(self.center_dbm[heard]>self.upper_dbm[heard]):
            raise ValueError('Reported target point must lie inside its actual quantization interval')
        object.__setattr__(self,'heard',_freeze(heard));object.__setattr__(self,'censored',_freeze(censored))

    def arrays(self):
        return {name:getattr(self,name) for name in self.__dataclass_fields__}

    @property
    def center(self):return self.center_dbm
    @property
    def lower(self):return self.lower_dbm
    @property
    def upper(self):return self.upper_dbm
    @property
    def censor_upper(self):return self.censor_upper_dbm


@dataclass(frozen=True)
class LinkRows:
    scene_id:str
    stage:str
    rx_xyz:np.ndarray
    ap_xyz:np.ndarray
    carrier_hz:np.ndarray
    direction:np.ndarray
    device:np.ndarray
    band_index:np.ndarray
    pose_index:np.ndarray
    ap_index:np.ndarray
    reported_rssi_dbm:np.ndarray
    lower_dbm:np.ndarray
    upper_dbm:np.ndarray
    heard:np.ndarray
    censored:np.ndarray
    censor_upper_dbm:np.ndarray
    fspl_dbm:np.ndarray
    offset_db:np.ndarray
    geometry_features:np.ndarray
    measurement_ids:tuple[str,...]

    def __post_init__(self):
        n=len(self.measurement_ids)
        if not n or len(set(self.measurement_ids))!=n:raise ValueError('RF row identities must be distinct')
        for name in ('rx_xyz','ap_xyz','geometry_features'):
            value=np.asarray(getattr(self,name));expected=(n,16) if name=='geometry_features' else (n,3)
            if value.shape!=expected or not np.isfinite(value).all():raise ValueError('Malformed public link geometry')
            object.__setattr__(self,name,_freeze(value))
        for name in ('carrier_hz','direction','device','band_index','pose_index','ap_index','reported_rssi_dbm',
                     'lower_dbm','upper_dbm','heard','censored','censor_upper_dbm','fspl_dbm','offset_db'):
            value=np.asarray(getattr(self,name))
            if value.shape!=(n,) or not np.isfinite(value).all():raise ValueError('Malformed physical link row: '+name)
            if name in ('heard','censored') and value.dtype!=np.bool_:raise ValueError('Link masks must be Boolean')
            object.__setattr__(self,name,_freeze(value))
        if np.any(self.heard&self.censored) or not np.isin(self.direction,(0,1)).all():raise ValueError('Invalid view/mask semantics')
        object.__setattr__(self,'measurement_ids',tuple(self.measurement_ids))

    @property
    def reported_rssi(self):return self.reported_rssi_dbm
    @property
    def estimated_offset_db(self):return self.offset_db
    @property
    def lower(self):return self.lower_dbm
    @property
    def upper(self):return self.upper_dbm
    @property
    def censor_upper(self):return self.censor_upper_dbm
    @property
    def calibrated_rssi_dbm(self):return np.where(self.heard,self.reported_rssi_dbm-self.offset_db,0.)
    @property
    def calibrated_lower_dbm(self):return np.where(self.heard,self.lower_dbm-self.offset_db,0.)
    @property
    def calibrated_upper_dbm(self):return np.where(self.heard,self.upper_dbm-self.offset_db,0.)
    @property
    def calibrated_censor_upper_dbm(self):return np.where(self.censored,self.censor_upper_dbm-self.offset_db,0.)
    @property
    def targets(self):return ReportedTargets(self.reported_rssi_dbm,self.lower_dbm,self.upper_dbm,self.heard,self.censored,self.censor_upper_dbm)

    def context_features(self):
        point=np.where(self.heard,(self.reported_rssi_dbm-self.offset_db-self.fspl_dbm)/30.,0.)
        bound=np.where(self.censored,(self.censor_upper_dbm-self.offset_db-self.fspl_dbm)/30.,0.)
        return _freeze(np.column_stack((point,bound,self.heard,self.censored,self.geometry_features)),np.float32)


def view_keys(scene,regime='in_band'):
    if regime=='in_band':bands,devices=TRAIN_BANDS,TRAIN_DEVICES
    elif regime=='held_band':bands,devices=(2,),TRAIN_DEVICES
    elif regime=='held_device':bands,devices=TRAIN_BANDS,(3,)
    else:raise ValueError('Unknown observed/held view regime')
    keys=tuple(ViewKey(b,d,v) for b in bands for d in (0,1) for v in devices)
    for key in keys:scene.axis(key)
    return keys


def _scene_view(value):
    value=(value,) if isinstance(value,ViewKey) else tuple(value)
    if not value or any(not isinstance(k,ViewKey) for k in value) or len(set(value))!=len(value):raise ValueError('A scene view needs distinct physical view keys')
    return value


def link_rows(scene,keys,calibration,*,poses):
    keys=_scene_view(keys);poses=tuple(poses)
    if len(set(poses))!=len(poses) or any(type(p) is not int or not 0<=p<12 for p in poses):raise ValueError('Use distinct registered public positions')
    fields={name:[] for name in ('rx_xyz','ap_xyz','carrier_hz','direction','device','band_index','pose_index','ap_index',
        'reported_rssi_dbm','lower_dbm','upper_dbm','heard','censored','censor_upper_dbm','offset_db',
        'calibration_count','calibration_unknown','quantization_width','noise_floor')}
    ids=[]
    for key in keys:
        b,d,v=scene.axis(key);estimate=calibration.estimate(key.device)
        for pose in poses:
            for ap in range(2):
                values={'rx_xyz':scene.rx_positions_m[pose],'ap_xyz':scene.ap_positions_m[ap],
                    'carrier_hz':scene.carrier_hz[b],'direction':d,'device':key.device,'band_index':key.band,
                    'pose_index':pose,'ap_index':ap,'reported_rssi_dbm':scene.rssi_dbm[pose,ap,b,d,v],
                    'lower_dbm':scene.quantization_interval_dbm[pose,ap,b,d,v,0],
                    'upper_dbm':scene.quantization_interval_dbm[pose,ap,b,d,v,1],
                    'heard':scene.heard[pose,ap,b,d,v],'censored':scene.censored[pose,ap,b,d,v],
                    'censor_upper_dbm':scene.censor_upper_dbm[pose,ap,b,d,v],'offset_db':estimate.offset_db,
                    'calibration_count':estimate.pair_count,'calibration_unknown':not estimate.known,
                    'quantization_width':estimate.quantization_db or 0.,'noise_floor':scene.noise_floor_dbm[v]}
                for name,value in values.items():fields[name].append(value)
                ids.append(f'{scene.scene_id}:pose{pose}:ap{ap}:band{key.band}:direction{d}:device{key.device}')
    arrays={name:np.asarray(value) for name,value in fields.items()}
    geometry=link_geometry_features(arrays['rx_xyz'],arrays['ap_xyz'],arrays['carrier_hz'],arrays['direction'],
        scene.bbox_dims_m,scene.cad_boxes_m,calibration_pair_count=arrays.pop('calibration_count'),
        calibration_unknown=arrays.pop('calibration_unknown'),quantization_width_db=arrays.pop('quantization_width'),
        noise_floor_dbm=arrays.pop('noise_floor'),tx_power_dbm=scene.tx_power_dbm)
    arrays['fspl_dbm']=fspl_received_power_dbm(arrays['rx_xyz'],arrays['ap_xyz'],arrays['carrier_hz'],scene.tx_power_dbm)
    return LinkRows(scene.scene_id,scene.stage,**{name:_freeze(value) for name,value in arrays.items()},
                    geometry_features=_freeze(geometry),measurement_ids=tuple(ids))


def training_rows(scene,calibration):
    if scene.stage!='training':raise ValueError('Global numerical prior may learn from TRAIN rows only')
    return link_rows(scene,view_keys(scene),calibration,poses=tuple(range(12)))


@dataclass(frozen=True)
class SceneExample:
    context_a:np.ndarray
    context_b:np.ndarray
    query:np.ndarray
    query_basis:np.ndarray
    mask_a:np.ndarray
    mask_b:np.ndarray
    query_fspl_dbm:np.ndarray
    query_offset_db:np.ndarray
    targets:ReportedTargets
    context_rows_a:LinkRows
    context_rows_b:LinkRows
    query_rows:LinkRows

    def __post_init__(self):
        a,b=np.asarray(self.context_a),np.asarray(self.context_b);q=np.asarray(self.query)
        if a.ndim!=2 or b.shape!=a.shape or a.shape[1]!=20 or q.ndim!=2 or q.shape[1]!=16:
            raise ValueError('A scene example needs paired context20 and query16 arrays')
        if not all(np.isfinite(x).all() for x in (a,b,q,self.query_basis,self.query_fspl_dbm,self.query_offset_db)):
            raise ValueError('Public model arrays must be finite')
        for name,shape in (('mask_a',(len(a),)),('mask_b',(len(b),)),('query_basis',(len(q),8)),
                           ('query_fspl_dbm',(len(q),)),('query_offset_db',(len(q),))):
            value=np.asarray(getattr(self,name))
            if value.shape!=shape or name.startswith('mask') and value.dtype!=np.bool_:raise ValueError('Malformed scene model input '+name)
            object.__setattr__(self,name,_freeze(value))
        if len(self.targets.heard)!=len(q):raise ValueError('Query and reported target counts differ')
        for name,value in (('context_a',a),('context_b',b),('query',q)):object.__setattr__(self,name,_freeze(value,np.float32))

    def model_inputs(self):
        return {name:getattr(self,name) for name in ('context_a','context_b','query','query_basis',
            'mask_a','mask_b','query_fspl_dbm','query_offset_db')}

    def target_arrays(self):return self.targets.arrays()


def make_example(scene,context_a=VIEW_A,context_b=VIEW_B,calibration=None,*,query_keys=None,query_poses=QUERY_POSES):
    if calibration is None:raise ValueError('Supply TRAIN-fitted or explicitly onboarded frontend calibration')
    a,b=_scene_view(context_a),_scene_view(context_b)
    if set(a)&set(b):raise ValueError('Good-view encoders cannot receive the same recorded features')
    for keys in (a,b):
        if (len(keys)!=2 or {k.band for k in keys}!=set(TRAIN_BANDS)
                or len({(k.direction,k.device) for k in keys})!=1):
            raise ValueError('Each identifiable context view must contain both observed bands at one receiver/direction, never band2')
        if scene.stage in ('training','validation') and any(k.device not in TRAIN_DEVICES for k in keys):raise ValueError('Held device cannot enter TRAIN/VAL context')
    queries=view_keys(scene) if query_keys is None else _scene_view(query_keys)
    query_poses=tuple(query_poses)
    if query_poses!=QUERY_POSES:
        if (query_poses!=CONTEXT_POSES or scene.stage not in ('development','field_test')
                or any(key.band!=2 for key in queries)):
            raise ValueError('Context-position targets are reserved for held-band DEV/FIELD transport only')
    if scene.stage in ('training','validation') and any(k.band not in TRAIN_BANDS or k.device not in TRAIN_DEVICES for k in queries):
        raise ValueError('Held-band/device labels cannot enter fitting or validation')
    left=link_rows(scene,a,calibration,poses=CONTEXT_POSES);right=link_rows(scene,b,calibration,poses=CONTEXT_POSES)
    query=link_rows(scene,queries,calibration,poses=query_poses)
    if set(left.measurement_ids)&set(right.measurement_ids) or set(left.measurement_ids+right.measurement_ids)&set(query.measurement_ids):
        raise ValueError('Input RF and prediction target RF must be disjoint')
    return SceneExample(left.context_features(),right.context_features(),query.geometry_features,
        _freeze(query_physics_basis(query.geometry_features)),_freeze(left.heard|left.censored),
        _freeze(right.heard|right.censored),_freeze(query.fspl_dbm,np.float32),
        _freeze(query.offset_db,np.float32),query.targets,left,right,query)


def examples(stage,*,role=None,onboarding_pairs=None,query_keys=None,query_poses=QUERY_POSES):
    """Standard fixed-calibration examples; model targets remain a separate object."""
    if stage not in STAGES:raise ValueError('Unknown example stage')
    c=_common();c.check_sources();calibration=load_calibration()
    result=[]
    for index in range(c.protocol()['scene_counts'][stage]):
        scene=load_scene(stage,index,role=role)
        current=calibration if onboarding_pairs is None else onboard_device(scene,calibration,n_pairs=onboarding_pairs)
        result.append(make_example(scene,calibration=current,query_keys=query_keys,query_poses=query_poses))
    return result
