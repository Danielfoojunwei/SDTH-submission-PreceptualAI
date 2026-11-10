"""Reported-only adaptive RF cases and explicit report-attempt schedules.

Pure sealed spectral builders are reused. Their registered loaders/common are
never called. No hidden material, condition index, true gain or private CFR is
included in model_inputs(). The simulator supplies actual censor/unknown masks.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass,fields,replace
import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from ..spectral_scene import data as sd
from . import simulation as sim

CONTEXT_POSES=sd.CONTEXT_POSES
QUERY_POSES=sd.QUERY_POSES
TRAIN_BANDS=sd.TRAIN_BANDS
TRAIN_DEVICES=sd.TRAIN_DEVICES
VIEW_A=sd.VIEW_A
VIEW_B=sd.VIEW_B
ViewKey=sd.ViewKey


def _common():return importlib.import_module('.common',__package__)


def _freeze(value):
    array=np.array(value,copy=True)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


def array_digest(value):
    array=np.ascontiguousarray(value)
    header=json.dumps({'dtype':array.dtype.str,'shape':list(array.shape)},sort_keys=True).encode()
    return hashlib.sha256(header+array.tobytes()).hexdigest()


@dataclass(frozen=True)
class AdaptiveScene(sd.PublicScene):
    source_stage:str='training'
    layout_id:str=''
    case_id:str=''
    condition:int=0
    geometry_sha256:str=''
    reported_detection_threshold_dbm:np.ndarray|None=None

    def __post_init__(self):
        super().__post_init__()
        if self.source_stage not in sim.COUNTS or not self.layout_id or self.case_id!=self.scene_id or self.condition not in sim.CONDITIONS:
            raise ValueError('Adaptive scene audit identity is malformed')
        threshold=np.asarray(self.reported_detection_threshold_dbm)
        if threshold.shape!=self.rssi_dbm.shape or not np.isfinite(threshold).all():
            raise ValueError('Simulated event thresholds must cover every reported link/view')
        if (not np.allclose(threshold[self.censored],self.censor_upper_dbm[self.censored],rtol=0,atol=1e-5)
                or np.any(self.quantization_interval_dbm[...,1][self.heard]+1e-5<threshold[self.heard])):
            raise ValueError('Detection thresholds disagree with actual heard/censored reports')
        object.__setattr__(self,'reported_detection_threshold_dbm',_freeze(threshold))


@dataclass(frozen=True)
class AdaptiveTargets(sd.ReportedTargets):
    detection_threshold_dbm:np.ndarray|None=None

    def __post_init__(self):
        super().__post_init__()
        threshold=np.asarray(self.detection_threshold_dbm)
        if threshold.shape!=self.heard.shape or not np.isfinite(threshold).all():
            raise ValueError('Detection event thresholds are required on heard and censored targets')
        if (not np.allclose(threshold[self.censored],self.censor_upper_dbm[self.censored],rtol=0,atol=1e-5)
                or np.any(self.upper_dbm[self.heard]+1e-5<threshold[self.heard])):
            raise ValueError('A target event threshold conflicts with its actual observation')
        object.__setattr__(self,'detection_threshold_dbm',_freeze(threshold))


@dataclass(frozen=True)
class AdaptiveLinkRows(sd.LinkRows):
    layout_id:str=''
    case_id:str=''
    condition:int=0
    regime:str='full'
    detection_threshold_dbm:np.ndarray|None=None

    def __post_init__(self):
        super().__post_init__()
        if not self.layout_id or not self.case_id or self.condition not in sim.CONDITIONS or self.regime not in ('full','sparse'):
            raise ValueError('Adaptive RF row audit identity is malformed')
        threshold=np.asarray(self.detection_threshold_dbm)
        if threshold.shape!=self.heard.shape or not np.isfinite(threshold).all():raise ValueError('Missing actual per-report event threshold')
        object.__setattr__(self,'detection_threshold_dbm',_freeze(threshold))

    @property
    def targets(self):
        return AdaptiveTargets(self.reported_rssi_dbm,self.lower_dbm,self.upper_dbm,self.heard,
                               self.censored,self.censor_upper_dbm,self.detection_threshold_dbm)


@dataclass(frozen=True)
class AdaptiveExample(sd.SceneExample):
    stage:str='training'
    layout_id:str=''
    case_id:str=''
    condition:int=0
    regime:str='full'
    context_case_id:str=''
    scheduled_mask_a:np.ndarray|None=None
    scheduled_mask_b:np.ndarray|None=None
    stale_requested:bool=False

    def __post_init__(self):
        super().__post_init__()
        if self.stage not in sim.COUNTS or not self.layout_id or not self.case_id or self.condition not in sim.CONDITIONS:
            raise ValueError('Adaptive example audit identity is malformed')
        if self.regime not in ('full','sparse') or type(self.stale_requested) is not bool or not self.context_case_id:
            raise ValueError('An explicit context regime and case identity are required')
        for name,known in (('scheduled_mask_a',self.mask_a),('scheduled_mask_b',self.mask_b)):
            scheduled=np.asarray(getattr(self,name))
            if scheduled.shape!=known.shape or scheduled.dtype!=np.bool_ or np.any(known&~scheduled):
                raise ValueError('Sensed-known reports must be a subset of scheduled report attempts')
            object.__setattr__(self,name,_freeze(scheduled))

    def model_inputs(self):
        result=super().model_inputs()
        result.update(scheduled_mask_a=self.scheduled_mask_a,scheduled_mask_b=self.scheduled_mask_b)
        return result


def _stage_receipt(c,stage):
    path=c.OUT/'rt_data'/f'{stage}.json'
    if not path.is_file():raise RuntimeError('Collect the authorized adaptive RT stage first')
    receipt=c.read_json(path)
    if (receipt.get('schema')!=1 or receipt.get('stage')!=stage or receipt.get('completed') is not True
            or receipt.get('source_receipt_sha256')!=c.digest(c.OUT/'registration/source.json')
            or receipt.get('registry_sha256')!=c.registry_sha256()
            or len(receipt.get('case_receipts',{}))!=3*sim.COUNTS[stage]):
        raise RuntimeError('Adaptive RT stage source/registry/membership changed')
    return path,receipt


def load_scene(stage,index,condition,*,role=None):
    """Read reported data/public CAD, never the private reference20 channels."""
    c=_common();c.check_sources();c.authorize_stage(stage)
    spec=sim.case_spec(stage,index,condition);folder=Path(c.case_path(stage,index,condition))
    manifest_path,manifest=_stage_receipt(c,stage)
    receipt_path=folder/'receipt.json';public_path=folder/'public_views.npz'
    if not receipt_path.is_file() or not public_path.is_file():raise RuntimeError('Missing collected adaptive case')
    binding={'path':str(receipt_path.relative_to(c.ROOT)),'sha256':c.digest(receipt_path)}
    if manifest['case_receipts'].get(spec.case_id)!=binding:raise RuntimeError('Case differs from its immutable collector stage')
    receipt=c.read_json(receipt_path)
    if (receipt.get('case_id')!=spec.case_id or receipt.get('layout_id')!=spec.layout_id
            or receipt.get('public_geometry_sha256')!=spec.geometry_sha256
            or receipt.get('known_tx_power_dbm')!=spec.tx_power_dbm
            or receipt.get('source_receipt_sha256')!=c.digest(c.OUT/'registration/source.json')
            or receipt.get('registry_sha256')!=c.registry_sha256()
            or receipt.get('real_device_measurements') is not False
            or receipt.get('unresolved_paths_are_censored') is not False):
        raise RuntimeError('Adaptive case public specification/source changed')
    files=receipt['files_sha256']
    if c.digest(public_path)!=files['public_views.npz']:raise RuntimeError('Reported device values changed')
    sources={str(p.relative_to(c.ROOT)):c.digest(p) for p in (manifest_path,receipt_path,public_path)}
    boxes=[]
    for name,sha in sorted(receipt['mesh_sha256'].items()):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.ply':raise RuntimeError('Unsafe public CAD path')
        path=folder/relative
        if path.is_symlink() or c.digest(path)!=sha:raise RuntimeError('Public CAD geometry changed')
        boxes.append(sd._read_cad_box(path));sources[str(path.relative_to(c.ROOT))]=sha
    internal_stage='field_test' if stage=='test' else stage
    internal_role,bands,devices=sd._role_axes(internal_stage,role)
    names=('rssi_dbm','heard','censored','censor_upper_dbm','quantization_interval_dbm','reported_detection_threshold_dbm')
    with np.load(public_path,allow_pickle=False) as saved:
        selected={name:np.take(np.take(saved[name],bands,axis=2),devices,axis=4) for name in names}
        metadata={name:saved[name].copy() for name in ('ap_positions_m','rx_positions_m')}
        metadata.update(carrier_hz=saved['carrier_hz'][list(bands)].copy(),
            noise_floor_dbm=saved['noise_floor_dbm'][list(devices)].copy(),
            device_ids=tuple(str(x) for x in saved['device_ids'][list(devices)]))
    if (not np.allclose(metadata['ap_positions_m'],spec.geometry.ap_positions_m,rtol=0,atol=2e-5)
            or not np.allclose(metadata['rx_positions_m'],spec.geometry.rx_positions_m,rtol=0,atol=2e-5)):
        raise RuntimeError('Reported public positions differ from the registered layout')
    public=receipt['public_geometry']
    # Only physical coordinates/dimensions/box vertices enter the old pure builder.
    scene=AdaptiveScene(internal_stage,index,spec.case_id,internal_role,**metadata,
        bbox_dims_m=np.array([public['width_m'],public['depth_m'],public['ceiling_m']]),
        cad_boxes_m=np.asarray(boxes),band_indices=bands,device_indices=devices,
        bandwidth_hz=float(receipt['bandwidth_hz']),tx_power_dbm=float(receipt['known_tx_power_dbm']),
        source_files_sha256=sources,source_stage=stage,layout_id=spec.layout_id,case_id=spec.case_id,
        condition=condition,geometry_sha256=spec.geometry_sha256,**selected)
    c.write_once(c.OUT/'data/scenes'/f'{spec.case_id}-{internal_role}.json',{
        'schema':1,'stage':stage,'layout_id':spec.layout_id,'case_id':spec.case_id,'condition':condition,
        'source_files_sha256':sources,'public_geometry_sha256':spec.geometry_sha256,
        'bands_exposed':list(bands),'devices_exposed':list(devices),
        'private_gain_or_CFR_files_read':False,'hidden_material_or_condition_encoded':False,
        'static_simulation_only':True})
    return scene


def _tag_rows(rows,scene,regime):
    values={f.name:getattr(rows,f.name) for f in fields(sd.LinkRows)}
    values['stage']=scene.source_stage
    thresholds=np.array([scene.reported_detection_threshold_dbm[
        int(p),int(a),scene.band_indices.index(int(b)),int(d),scene.device_indices.index(int(v))]
        for p,a,b,d,v in zip(rows.pose_index,rows.ap_index,rows.band_index,rows.direction,rows.device)])
    return AdaptiveLinkRows(**values,layout_id=scene.layout_id,case_id=scene.case_id,
                            condition=scene.condition,regime=regime,detection_threshold_dbm=thresholds)


def scheduled_reports(rows,regime):
    if regime not in ('full','sparse'):raise ValueError('Unknown report-attempt schedule')
    return np.ones(len(rows.pose_index),bool) if regime=='full' else np.isin(rows.pose_index,sim.SPARSE_CONTEXT_POSES)


def _apply_schedule(rows,scene,regime):
    scheduled=scheduled_reports(rows,regime)
    point=rows.heard&scheduled;censor=rows.censored&scheduled
    # Withheld reports are unknown. They do not acquire a sensitivity bound.
    updated=replace(rows,heard=_freeze(point),censored=_freeze(censor),
        reported_rssi_dbm=_freeze(np.where(point,rows.reported_rssi_dbm,0.)),
        lower_dbm=_freeze(np.where(point,rows.lower_dbm,0.)),upper_dbm=_freeze(np.where(point,rows.upper_dbm,0.)),
        censor_upper_dbm=_freeze(np.where(censor,rows.censor_upper_dbm,0.)))
    return _tag_rows(updated,scene,regime),_freeze(scheduled)


def make_example(scene,calibration,*,regime='full',held_band=False,context_scene=None,stale=False):
    context_scene=scene if context_scene is None else context_scene
    if (scene.layout_id!=context_scene.layout_id or scene.geometry_sha256!=context_scene.geometry_sha256
            or scene.tx_power_dbm!=context_scene.tx_power_dbm
            or not np.array_equal(scene.rx_positions_m,context_scene.rx_positions_m)
            or not np.array_equal(scene.ap_positions_m,context_scene.ap_positions_m)
            or not np.array_equal(scene.cad_boxes_m,context_scene.cad_boxes_m)):
        raise ValueError('Stale-condition comparison requires identical public geometry and known power')
    queries=sd.view_keys(scene,'held_band') if held_band else sd.view_keys(scene)
    query_poses=CONTEXT_POSES if held_band else QUERY_POSES
    current=sd.make_example(scene,calibration=calibration,query_keys=queries,query_poses=query_poses)
    # No query values from the source/stale condition are required or read here.
    a=sd.link_rows(context_scene,VIEW_A,calibration,poses=CONTEXT_POSES)
    b=sd.link_rows(context_scene,VIEW_B,calibration,poses=CONTEXT_POSES)
    a,scheduled_a=_apply_schedule(a,context_scene,regime)
    b,scheduled_b=_apply_schedule(b,context_scene,regime)
    q=_tag_rows(current.query_rows,scene,regime)
    return AdaptiveExample(a.context_features(),b.context_features(),current.query,current.query_basis,
        _freeze(a.heard|a.censored),_freeze(b.heard|b.censored),current.query_fspl_dbm,
        current.query_offset_db,q.targets,a,b,q,stage=scene.source_stage,layout_id=scene.layout_id,
        case_id=scene.case_id,condition=scene.condition,regime=regime,context_case_id=context_scene.case_id,
        scheduled_mask_a=scheduled_a,scheduled_mask_b=scheduled_b,stale_requested=bool(stale))


def training_rows(scene,calibration):
    if scene.source_stage!='training':raise ValueError('The global prior may fit TRAIN reports only')
    return _tag_rows(sd.training_rows(scene,calibration),scene,'full')


def calibration_record(calibration):
    value=sd.calibration_record(calibration)
    value.update(scope='offline TRAIN paired context reports; identical calibration for all report schedules',
                 receiver_random_noise_added=False,quantization_is_not_a_sensor_gaussian_sigma=True)
    return value


def load_calibration():
    c=_common();c.check_sources();path=c.OUT/'data/frontend-training.json'
    if not path.is_file():raise RuntimeError('Prepare the adaptive TRAIN calibration first')
    record=c.read_json(path)
    prepared=c.OUT/'data/prepared-training.json'
    if prepared.exists() and c.read_json(prepared)['calibration_sha256']!=c.digest(path):
        raise RuntimeError('Adaptive calibration differs from the prepared training identity')
    for name,sha in record['source_files_sha256'].items():
        if c.digest(c.safe_path(name))!=sha:raise RuntimeError('Calibration reported inputs changed')
    estimates={int(k):sd.DeviceEstimate(**(v|{'offset_interval_db':None if v['offset_interval_db'] is None else tuple(v['offset_interval_db'])}))
               for k,v in record['estimates'].items()}
    result=sd.FrontendCalibration(estimates,tuple(record['paired_reports']),record['source_files_sha256'])
    if calibration_record(result)!=record:raise RuntimeError('Adaptive calibration scope/schema changed')
    for device in (1,2):
        pairs=[p for p in result.paired_reports if p['device']==device]
        if any(p['stage']!='training' or p['pose'] not in CONTEXT_POSES or p['band'] not in TRAIN_BANDS for p in pairs):
            raise RuntimeError('Adaptive calibration used forbidden query/test/band reports')
        if sd._estimate(device,pairs)!=result.estimates[device]:raise RuntimeError('Adaptive offset estimates differ from their paired reports')
    return result


def prepare_training():
    c=_common();c.check_sources();path=c.OUT/'data/frontend-training.json'
    if path.exists():calibration=load_calibration()
    else:
        scenes=[load_scene('training',i,k) for i in range(sim.COUNTS['training']) for k in sim.CONDITIONS]
        calibration=sd.fit_frontends(scenes)
        c.write_once(path,calibration_record(calibration))
    stage_paths={stage:_stage_receipt(c,stage)[0] for stage in ('training','validation')}
    value={'schema':1,'completed':True,'training_layouts':sim.COUNTS['training'],
        'training_cases':3*sim.COUNTS['training'],'calibration_receipt':str(path.relative_to(c.ROOT)),
        'calibration_sha256':c.digest(path),'source_files_sha256':dict(calibration.source_files_sha256),
        'stage_receipts_sha256':{stage:c.digest(p) for stage,p in stage_paths.items()},
        'stage_receipt_paths':{stage:str(p.relative_to(c.ROOT)) for stage,p in stage_paths.items()},
        'known_tx_is_shared_by_all_conditions_of_layout':True,'hidden_condition_or_true_gain_used_as_input':False,
        'observation_schedules':{'full':list(CONTEXT_POSES),'sparse':list(sim.SPARSE_CONTEXT_POSES)},
        'calibration_schedule_scope':'offline full TRAIN contexts; shared by all online schedules'}
    c.write_once(c.OUT/'data/prepared-training.json',value)
    return value


def example_binding(example):
    return {'layout_id':example.layout_id,'case_id':example.case_id,'condition':example.condition,
        'stage':example.stage,'regime':example.regime,'context_case_id':example.context_case_id,
        'stale_requested':example.stale_requested,
        'model_input_sha256':{k:array_digest(v) for k,v in example.model_inputs().items()},
        'target_sha256':{k:array_digest(v) for k,v in example.target_arrays().items()},
        'scheduled_attempts_a':int(example.scheduled_mask_a.sum()),'scheduled_attempts_b':int(example.scheduled_mask_b.sum()),
        'sensed_known_a':int(example.mask_a.sum()),'sensed_known_b':int(example.mask_b.sum()),
        'point_a':int(example.context_rows_a.heard.sum()),'censored_a':int(example.context_rows_a.censored.sum()),
        'unknown_or_withheld_a':int((~example.mask_a).sum()),
        'input_RF_identity_sha256':hashlib.sha256('\n'.join(example.context_rows_a.measurement_ids).encode()).hexdigest(),
        'target_RF_identity_sha256':hashlib.sha256('\n'.join(example.query_rows.measurement_ids).encode()).hexdigest(),
        'network_delivery_or_real_capture_timing_claimed':False}


def examples(stage,*,regime='full',held_band=False,stale=False):
    if stage not in sim.COUNTS or regime not in ('full','sparse') or type(held_band) is not bool or type(stale) is not bool:
        raise ValueError('Unknown adaptive stage/regime/counterfactual')
    if held_band and stage in ('training','validation'):raise ValueError('Held-band reports cannot enter TRAIN/VAL')
    c=_common();c.check_sources();calibration=load_calibration();result=[]
    role='held_band' if held_band and stage=='development' else None
    for index in range(sim.COUNTS[stage]):
        scenes=[load_scene(stage,index,k,role=role) for k in sim.CONDITIONS]
        for scene in scenes:
            context=scenes[0] if stale else scene
            result.append(make_example(scene,calibration,regime=regime,held_band=held_band,
                                       context_scene=context,stale=stale))
    stage_path,_=_stage_receipt(c,stage)
    name=f'{stage}-{regime}-'+('heldband' if held_band else 'main')+('-stale' if stale else '-current')
    receipt={'schema':1,'stage':stage,'regime':regime,'held_band':held_band,'stale':stale,
        'case_count':len(result),'independent_layouts':sim.COUNTS[stage],
        'calibration_sha256':c.digest(c.OUT/'data/frontend-training.json'),
        'stage_receipt_sha256':c.digest(stage_path),'examples':[example_binding(e) for e in result],
        'conditions_or_schedules_add_independent_layouts':False,'static_simulation_only':True}
    c.write_once(c.OUT/'data'/f'{name}.json',receipt)
    return result
