"""Actual RT fields with three hidden materials per identical public CAD.

The conditions are static counterfactual scene states, not a simulated temporal
transition or real hardware. Scheduled report availability is implemented by
the separate reported-only data loader, not by inventing MAC delivery events.
"""
from __future__ import annotations

from dataclasses import dataclass,replace
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from ..radiojepa import rt_data as rt

ROOT=Path(__file__).resolve().parents[3]
COUNTS={'training':24,'validation':6,'development':4,'test':24}
CONDITIONS=(0,1,2)
MATERIALS=('metal','wood','brick')
TX_POWER_DBM=(-20.,-10.,0.,10.)
SPARSE_CONTEXT_POSES=(0,10)
PUBLIC_GEOMETRY_FIELDS=('family','width_m','depth_m','ceiling_m','obstacle_height_m',
                        'ap_positions_m','rx_positions_m')


def _common():return importlib.import_module('.common',__package__)


def _encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def public_geometry(spec):
    raw=spec.to_dict() if isinstance(spec,rt.GeometrySpec) else dict(spec)
    return {name:raw[name] for name in PUBLIC_GEOMETRY_FIELDS}


def geometry_fingerprint(spec):return hashlib.sha256(_encoded(public_geometry(spec))).hexdigest()


def geometry_spec(stage,index):
    if stage not in COUNTS or type(index) is not int or not 0<=index<COUNTS[stage]:
        raise ValueError('Unknown registered adaptive layout')
    split=list(COUNTS).index(stage)
    # Moderate warehouse dimensions, with new coordinates rather than new seeds
    # over a reused CAD. All material conditions below retain these exact values.
    width=15.137+.947*index+.079*split
    depth=20.283+1.417*((index*7)%24)+.131*split
    ceiling=4.267+.31*(index%5)+.017*split
    height=2.4+.19*(index%4)
    aps=((-0.36*width,-0.35*depth,ceiling-.57),
         (.35*width,.36*depth,ceiling-.63))
    rx=tuple((x*width,y*depth,1.05+.02*(index%2))
             for y in (-.34,-.12,.12,.34) for x in (-.34,0.,.34))
    return rt.GeometrySpec(f'adaptive-{stage}-{index:02d}',
        ('rack_aisles','partitioned_room')[(index+split)%2],width,depth,ceiling,height,
        'metal',aps,rx,981000+100*split+index)


@dataclass(frozen=True)
class CaseSpec:
    stage:str
    index:int
    condition:int
    geometry:rt.GeometrySpec
    tx_power_dbm:float

    def __post_init__(self):
        if self.stage not in COUNTS or type(self.index) is not int or not 0<=self.index<COUNTS[self.stage] or type(self.condition) is not int or self.condition not in CONDITIONS:
            raise ValueError('Unknown layout/condition case')
        if not isinstance(self.geometry,rt.GeometrySpec) or not math.isfinite(self.tx_power_dbm):
            raise ValueError('A physical CAD and known transmit power are required')

    @property
    def layout_id(self):return self.geometry.scene_id
    @property
    def case_id(self):return self.layout_id+f'-rf{self.condition}'
    @property
    def hidden_material(self):return MATERIALS[(self.index+self.condition)%len(MATERIALS)]
    @property
    def rt_spec(self):
        # Same sampling seed for all three conditions: candidate-path sampling
        # is paired rather than gratuitously resampled with the hidden material.
        return replace(self.geometry,scene_id=self.case_id,obstacle_material=self.hidden_material)
    @property
    def geometry_sha256(self):return geometry_fingerprint(self.geometry)

    def to_dict(self):
        return {'stage':self.stage,'layout_index':self.index,'condition':self.condition,
            'layout_id':self.layout_id,'case_id':self.case_id,'public_geometry_sha256':self.geometry_sha256,
            'known_tx_power_dbm':self.tx_power_dbm,'rt_spec':self.rt_spec.to_dict(),
            'hidden_material_is_model_input':False}


def case_spec(stage,index,condition):
    geometry=geometry_spec(stage,index)
    return CaseSpec(stage,index,condition,geometry,TX_POWER_DBM[index%len(TX_POWER_DBM)])


def geometry_registry():
    prior=[];prior_files={}
    for relative in ('artifacts/radiojepa/rt_registry.json','artifacts/radiojepa_spectral/field_registry.json'):
        path=ROOT/relative;value=json.loads(path.read_text());prior_files[relative]=digest(path)
        prior.extend(geometry_fingerprint(spec) for specs in value['stages'].values() for spec in specs)
    if len(prior)!=120:raise RuntimeError('Expected the original36 plus sealed spectral84 public CADs')
    stages={};fresh=[]
    for stage,count in COUNTS.items():
        stages[stage]=[]
        for index in range(count):
            spec=geometry_spec(stage,index);fingerprint=geometry_fingerprint(spec);fresh.append(fingerprint)
            stages[stage].append({'layout_id':spec.scene_id,'public_geometry':public_geometry(spec),
                'public_geometry_sha256':fingerprint,'known_tx_power_dbm':TX_POWER_DBM[index%4],
                'cases':[case_spec(stage,index,c).to_dict() for c in CONDITIONS]})
    if len(set(fresh))!=sum(COUNTS.values()) or set(fresh)&set(prior):
        raise RuntimeError('Adaptive layouts are not geometrically disjoint from previous or other splits')
    return {'schema':1,'scope':'static actual Sionna RT counterfactual fields; not real hardware or a temporal simulator',
        'counts':dict(COUNTS),'conditions_per_layout':3,'case_count':3*sum(COUNTS.values()),
        'all_conditions_share_exact_public_CAD_TX_and_solver_seed':True,
        'all_new_public_geometries_disjoint_from_previous120':True,'prior_registry_sha256':prior_files,
        'known_tx_power_dbm_cycle_per_layout':list(TX_POWER_DBM),'hidden_materials':list(MATERIALS),
        'sparse_context_pose_attempts':list(SPARSE_CONTEXT_POSES),
        'network_transport_or_physical_delivery_clocks_simulated':False,'stages':stages}


def reports_from_rendered_gain(gain,solved,tx_power_dbm,*,profiles):
    """Physical linear power scaling, followed by genuine receiver thresholds.

    This function belongs to simulation, never the learned data boundary.
    Unresolved paths remain unknown and are not assigned invented weak powers.
    """
    gain=np.asarray(gain,dtype=float);solved=np.asarray(solved)
    if gain.shape!=solved.shape or solved.dtype!=bool or not math.isfinite(tx_power_dbm):
        raise ValueError('Invalid rendered gain/availability/power')
    if np.any(gain[solved]<=0) or not np.isfinite(gain[solved]).all():raise ValueError('Solved channel gains must be positive and finite')
    signal=np.zeros(gain.shape,dtype=float)
    signal[solved]=float(tx_power_dbm)+10*np.log10(gain[solved])
    reports=rt.device_observations(signal,solved,profiles=profiles,bandwidth_hz=rt.BANDWIDTH_HZ)
    thermal=10*math.log10(1.380649e-23*290.*rt.BANDWIDTH_HZ*1000.)
    threshold=np.array([thermal+p.noise_figure_db+p.detection_snr_db+p.rssi_offset_db
                        for p in profiles],dtype='f4')
    reports['reported_detection_threshold_dbm']=np.broadcast_to(threshold,gain.shape+(len(profiles),)).copy()
    return reports


def _trace_case(spec,folder,*,samples_per_src=32768,max_depth=3,max_num_paths_per_src=4096):
    """Internal known-fixture/scientific seam; public caller guards publication."""
    folder=Path(folder)
    raw=folder/'reference20'
    profiles=tuple(p for p in rt.DEVICE_PROFILES if not p.held_out) if spec.stage in ('training','validation') else rt.DEVICE_PROFILES
    raw_receipt=rt._trace_scene(spec.rt_spec,raw,profiles=profiles,samples_per_src=samples_per_src,
        max_depth=max_depth,max_num_paths_per_src=max_num_paths_per_src)
    # The generator may read its own actual physical channel. No learned loader
    # opens this archive; the newly emitted observation is the only model data.
    with np.load(raw/'physical_channels.npz',allow_pickle=False) as physical:
        reports=reports_from_rendered_gain(physical['channel_power_gain'],physical['solved'],
                                          spec.tx_power_dbm,profiles=profiles)
    reports.update(ap_positions_m=np.asarray(spec.geometry.ap_positions_m,dtype='f4'),
        rx_positions_m=np.asarray(spec.geometry.rx_positions_m,dtype='f4'),
        carrier_hz=np.asarray(rt.BANDS_HZ,dtype='f8'),
        training_device=np.array([not p.held_out for p in profiles]),
        device_ids=np.array([p.device_id for p in profiles]))
    public=folder/'public_views.npz';public.parent.mkdir(parents=True,exist_ok=True)
    with public.open('xb') as stream:np.savez_compressed(stream,**reports)
    files={str(path.relative_to(folder)):digest(path) for path in folder.rglob('*') if path.is_file()}
    mesh={name:sha for name,sha in files.items() if '/scene/meshes/' in name}
    receipt={'schema':1,'case':spec.to_dict(),'case_id':spec.case_id,'layout_id':spec.layout_id,
        'stage':spec.stage,'layout_index':spec.index,'condition':spec.condition,
        'public_geometry_sha256':spec.geometry_sha256,'public_geometry':public_geometry(spec.geometry),
        'known_tx_power_dbm':spec.tx_power_dbm,'bands_hz':list(rt.BANDS_HZ),'bandwidth_hz':rt.BANDWIDTH_HZ,
        'receiver_profiles':[{'device_id':p.device_id,'noise_figure_db':p.noise_figure_db,
                             'quantization_db':p.quantization_db,'held_out':p.held_out} for p in profiles],
        'mesh_sha256':mesh,'files_sha256':files,'reference20_receipt_sha256':digest(raw/'receipt.json'),
        'source_reference_tx_dbm':rt.REFERENCE_TX_DBM,'power_scaling':'linear channel gain times known transmit power',
        'real_device_measurements':False,'static_counterfactual_condition':True,
        'temporal_blockage_transition_simulated':False,'network_delivery_simulated':False,
        'detection_threshold_scope':'simulated reported-gauge event metadata for evaluation/targets, not a student feature',
        'unresolved_paths_are_censored':False,'finite_path_search_converged':raw_receipt['finite_path_search_converged']}
    return receipt


def trace_case(spec,folder):
    c=_common();c.check_sources();authorization=c.authorize_stage(spec.stage)
    if spec!=case_spec(spec.stage,spec.index,spec.condition):raise ValueError('Case differs from the registered physical specification')
    folder=Path(folder)
    if folder.resolve()!=Path(c.case_path(spec.stage,spec.index,spec.condition)).resolve():raise ValueError('Case output path differs from the registered namespace')
    receipt_path=folder/'receipt.json'
    if receipt_path.exists():return validate_case_receipt(spec,folder)
    if folder.exists() and any(folder.rglob('*')):raise RuntimeError('Partial RT case exists; refuse an unrecorded retime/overwrite')
    receipt=_trace_case(spec,folder)
    receipt.update(source_receipt_sha256=c.digest(c.OUT/'registration/source.json'),
        registry_sha256=c.registry_sha256(),authorization=authorization)
    c.write_once(receipt_path,receipt)
    return validate_case_receipt(spec,folder)


def validate_case_receipt(spec,folder):
    """Passive byte/spec validation; no model, renderer or regeneration."""
    c=_common();folder=Path(folder);receipt=c.read_json(folder/'receipt.json')
    if (receipt.get('schema')!=1 or receipt.get('case')!=json.loads(_encoded(spec.to_dict()))
            or receipt.get('case_id')!=spec.case_id or receipt.get('layout_id')!=spec.layout_id
            or receipt.get('public_geometry_sha256')!=spec.geometry_sha256
            or receipt.get('known_tx_power_dbm')!=spec.tx_power_dbm
            or receipt.get('source_receipt_sha256')!=c.digest(c.OUT/'registration/source.json')
            or receipt.get('registry_sha256')!=c.registry_sha256()
            or receipt.get('real_device_measurements') is not False
            or receipt.get('unresolved_paths_are_censored') is not False):
        raise RuntimeError('RT case source/specification/scope changed')
    expected=set(receipt['files_sha256'])
    actual={str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file() and path.name!='receipt.json'}
    # The nested frozen reference20/receipt.json is a bound physical source.
    actual.add('reference20/receipt.json')
    if actual!=expected:raise RuntimeError('RT case membership changed')
    for name,expected_sha in receipt['files_sha256'].items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or (folder/path).is_symlink() or digest(folder/path)!=expected_sha:
            raise RuntimeError('RT case raw bytes changed')
    # Independent, passive reconstruction of device reports from the rendered
    # channel. This is a simulator/auditor operation, not a learned data input.
    raw=folder/'reference20';raw_receipt=c.read_json(raw/'receipt.json')
    profiles=tuple(rt.DeviceProfile(**value) for value in raw_receipt['receiver_profiles'])
    with np.load(raw/'physical_channels.npz',allow_pickle=False) as physical:
        expected_reports=reports_from_rendered_gain(physical['channel_power_gain'],physical['solved'],
                                                   spec.tx_power_dbm,profiles=profiles)
    with np.load(folder/'public_views.npz',allow_pickle=False) as public:
        for name,expected_array in expected_reports.items():
            if not np.array_equal(public[name],expected_array):raise RuntimeError('Reported values/events differ from the actual rendered gain')
    return receipt


def trace_fixture(geometry,folder,*,condition=0,tx_power_dbm=-10.,**kwargs):
    """Explicit known-ID software fixture; registered cases cannot bypass guards."""
    if not isinstance(geometry,rt.GeometrySpec) or geometry.scene_id.startswith('adaptive-'):
        raise RuntimeError('Registered layouts require source publication and stage authorization')
    spec=CaseSpec('training',0,condition,geometry,float(tx_power_dbm))
    return _trace_case(spec,folder,**kwargs)
