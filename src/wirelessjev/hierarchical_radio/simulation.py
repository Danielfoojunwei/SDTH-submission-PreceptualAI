"""Fresh multi-AP ray-traced fields with localized material interventions.

Public geometry, radio-source power and private material assignment use
independent deterministic streams. Material states share exact CAD, sampling
seed and observation schedules. Nothing here claims chronological dynamics.
Scientific tracing requires an external published-source authorization callback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import time
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from ..radiojepa import rt_data as rt

ROOT = Path(__file__).resolve().parents[3]
COUNTS = {'training':24, 'validation':6, 'development':4, 'test':24, 'ap_transfer':4}
STATES = (0, 1, 2)
CONDITIONS = STATES
BANDS_HZ = (2.4e9, 5.2e9, 6.0e9)
BANDWIDTH_HZ = 20e6
PROFILES = rt.DEVICE_PROFILES[:2]
TOP_K = 2
MEASUREMENT_SIGMA_DB = .75
X_FRACTIONS = (-.42, -.30, -.10, .10, .30, .42)
Y_FRACTIONS = (-.42, -.30, -.18, -.06, .06, .18, .30, .42)
CONTEXT_POSES = tuple(i for i in range(48) if ((i//6)+(i%6))%2 == 0)
QUERY_POSES = tuple(i for i in range(48) if i not in CONTEXT_POSES)
SPARSE_CONTEXT_POSES = CONTEXT_POSES[::2]
PUBLIC_GEOMETRY_FIELDS = ('family','width_m','depth_m','ceiling_m','obstacle_height_m',
                          'ap_positions_m','rx_positions_m')


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(data);f.flush()
        import os
        os.fsync(f.fileno())


def _stream(stage,index,stream):
    return np.random.default_rng(np.random.SeedSequence([10112026,list(COUNTS).index(stage),index,stream]))


def geometry_spec(stage,index):
    if stage not in COUNTS or type(index) is not int or not 0<=index<COUNTS[stage]:
        raise ValueError('Unknown registered hierarchical layout')
    rng=_stream(stage,index,11)
    width=float(rng.uniform(22.,43.));depth=float(rng.uniform(27.,55.))
    ceiling=float(rng.uniform(4.6,6.4));height=float(rng.uniform(2.3,min(3.8,ceiling-.7)))
    base=((- .36,-.36),(.36,-.36),(-.36,.36),(.36,.36))
    if stage=='ap_transfer':base=tuple((x,y) for y in (-.36,-.12,.12,.36) for x in (-.36,.36))
    aps=tuple((float((x+rng.uniform(-.012,.012))*width),
               float((y+rng.uniform(-.012,.012))*depth),float(ceiling-rng.uniform(.5,.8))) for x,y in base)
    rx=tuple((x*width,y*depth,float(1.04+rng.uniform(0.,.04))) for y in Y_FRACTIONS for x in X_FRACTIONS)
    return rt.GeometrySpec(f'hierarchical-{stage}-{index:02d}',
        ('rack_aisles','partitioned_room')[int(rng.integers(2))],width,depth,ceiling,height,
        'wood',aps,rx,1112000+1000*list(COUNTS).index(stage)+index)


def public_geometry(geometry):
    raw=geometry.to_dict() if isinstance(geometry,rt.GeometrySpec) else dict(geometry)
    return {key:raw[key] for key in PUBLIC_GEOMETRY_FIELDS}


def geometry_fingerprint(geometry):
    return hashlib.sha256(encoded(public_geometry(geometry))).hexdigest()


@dataclass(frozen=True)
class CaseSpec:
    stage: str
    index: int
    state: int
    geometry: rt.GeometrySpec
    materials: tuple[str,str]
    tx_power_dbm: float

    def __post_init__(self):
        if (self.stage not in COUNTS or type(self.index) is not int or not 0<=self.index<COUNTS[self.stage]
                or type(self.state) is not int or self.state not in STATES
                or not isinstance(self.geometry,rt.GeometrySpec)
                or type(self.materials) is not tuple or len(self.materials)!=2
                or any(x not in ('wood','brick','concrete','metal') for x in self.materials)
                or type(self.tx_power_dbm) not in (int,float) or not math.isfinite(self.tx_power_dbm)):
            raise ValueError('Explicit registered layout/state/material/power identity required')

    @property
    def condition(self):return self.state
    @property
    def layout_id(self):return self.geometry.scene_id
    @property
    def case_id(self):return self.layout_id+f'-state{self.state}'
    @property
    def geometry_sha256(self):return geometry_fingerprint(self.geometry)

    def to_dict(self):
        return {'stage':self.stage,'index':self.index,'state':self.state,'layout_id':self.layout_id,
                'case_id':self.case_id,'public_geometry_sha256':self.geometry_sha256,
                'geometry':self.geometry.to_dict(),'private_obstacle_materials':list(self.materials),
                'per_mesh_materials_override_legacy_geometry_material_field':True,
                'known_tx_power_dbm':float(self.tx_power_dbm),
                'material_state_identity_or_solver_seed_are_model_inputs':False}


def case_spec(stage,index,state):
    geometry=geometry_spec(stage,index)
    if type(state) is not int or state not in STATES:raise ValueError('Unknown localized material state')
    base=tuple(_stream(stage,index,23).choice(('wood','brick','concrete'),size=2,replace=False).tolist())
    materials=base if state==0 else ('metal',base[1]) if state==1 else (base[0],'metal')
    power=float(_stream(stage,index,37).choice((-20.,-10.,0.,10.)))
    return CaseSpec(stage,index,state,geometry,materials,power)


def obstacle_boxes(geometry):
    """Same public solids across states; endpoints lie outside obstacle interiors."""
    w,d=geometry.width_m/2,geometry.depth_m/2
    if geometry.family=='rack_aisles':
        return tuple((f'rack{i}',(x-.2,-.25*geometry.depth_m,0.),
                      (x+.2,.25*geometry.depth_m,geometry.obstacle_height_m))
                     for i,x in enumerate((-.18*geometry.width_m,.18*geometry.width_m)))
    gap=.18*geometry.width_m
    return (('partition_left',(-w,-.08,0.),(-gap,.08,geometry.obstacle_height_m)),
            ('partition_right',(gap,-.08,0.),(w,.08,geometry.obstacle_height_m)))


def geometry_registry(root=ROOT):
    root=Path(root);prior=[];prior_files={}
    for relative in ('artifacts/radiojepa/rt_registry.json','artifacts/radiojepa_spectral/field_registry.json'):
        path=root/relative;value=json.loads(path.read_text());prior_files[relative]=digest(path)
        prior.extend(geometry_fingerprint(spec) for specs in value['stages'].values() for spec in specs)
    path=root/'artifacts/adaptive_radio/geometry_registry.json'
    value=json.loads(path.read_text());prior_files[str(path.relative_to(root))]=digest(path)
    for layouts in value['stages'].values():
        for layout in layouts:prior.append(geometry_fingerprint(layout['public_geometry']))
    stages={};fresh=[]
    for stage,count in COUNTS.items():
        stages[stage]=[]
        for index in range(count):
            geometry=geometry_spec(stage,index);fingerprint=geometry_fingerprint(geometry);fresh.append(fingerprint)
            stages[stage].append({'layout_id':geometry.scene_id,'public_geometry':public_geometry(geometry),
                'public_geometry_sha256':fingerprint,'APs':len(geometry.ap_positions_m),
                'cases':[case_spec(stage,index,state).to_dict() for state in STATES]})
    if len(prior)!=178 or len(set(fresh))!=sum(COUNTS.values()) or set(fresh)&set(prior):
        raise RuntimeError('New layouts must be geometrically disjoint from all178 prior public layouts and each other')
    return {'schema':1,'counts':COUNTS,'states_per_layout':3,'case_count':3*sum(COUNTS.values()),
        'context_poses':list(CONTEXT_POSES),'query_poses':list(QUERY_POSES),
        'source_bands':[0,1],'held_band':2,'bands_hz':list(BANDS_HZ),
        'top_k_per_band':TOP_K,'receiver_measurement_sigma_db':MEASUREMENT_SIGMA_DB,
        'context_tokens_full':192,'context_tokens_sparse':96,
        'observation_schedule_independent_of_private_material_state':True,
        'localized_states':'state1 changes obstacle0 only; state2 changes obstacle1 only',
        'geometry_material_and_power_use_independent_rng_streams':True,
        'material_states_share_public_CAD_TX_solver_seed_and_receiver_noise_draws':True,
        'all_new_geometries_disjoint_from_prior178':True,'prior_registry_sha256':prior_files,
        'temporal_dynamics_real_hardware_or_network_delivery_claimed':False,'stages':stages}


def write_scene(spec,folder):
    """Fresh material-aware CAD emitter; does not modify a frozen generator."""
    g=spec.geometry;folder=Path(folder);w,d,h=g.width_m/2,g.depth_m/2,g.ceiling_m
    boxes=[('floor','concrete',(-w,-d,-.15),(w,d,0.)),
        ('ceiling','concrete',(-w,-d,h),(w,d,h+.15)),
        ('wall_left','brick',(-w-.15,-d,0.),(-w,d,h)),
        ('wall_right','brick',(w,-d,0.),(w+.15,d,h)),
        ('wall_front','brick',(-w,-d-.15,0.),(w,-d,h)),
        ('wall_back','brick',(-w,d,0.),(w,d+.15,h))]
    boxes.extend((name,f'obstacle{i}',lo,hi) for i,(name,lo,hi) in enumerate(obstacle_boxes(g)))
    root=ET.Element('scene',version='2.1.0')
    material_specs=(('concrete','concrete'),('brick','brick'),('obstacle0',spec.materials[0]),('obstacle1',spec.materials[1]))
    for name,material in material_specs:
        node=ET.SubElement(root,'bsdf',type='itu-radio-material',id=name)
        ET.SubElement(node,'string',name='type',value=material)
        ET.SubElement(node,'float',name='thickness',value='.15')
    files=[]
    for name,material,lo,hi in boxes:
        mesh=folder/'meshes'/(name+'.ply');_write(mesh,rt._ply_box(lo,hi));files.append(mesh)
        node=ET.SubElement(root,'shape',type='ply',id=name)
        ET.SubElement(node,'string',name='filename',value='meshes/'+mesh.name)
        ET.SubElement(node,'boolean',name='face_normals',value='true')
        ET.SubElement(node,'ref',name='bsdf',id=material)
    xml=folder/'scene.xml';_write(xml,ET.tostring(root,encoding='utf-8',xml_declaration=True));files.append(xml)
    return xml,files


def reports_from_gain(gain,solved,tx_power_dbm,*,noise_seed,profiles=PROFILES):
    """Noisy measured scans, per-band top-K, and genuine interval censor events."""
    gain=np.asarray(gain,dtype=float);solved=np.asarray(solved)
    if (gain.ndim!=4 or gain.shape[2:]!=(3,2) or gain.shape!=solved.shape or solved.dtype!=bool
            or not np.isfinite(gain[solved]).all() or np.any(gain[solved]<=0)
            or type(noise_seed) is not int or noise_seed<0 or not math.isfinite(tx_power_dbm)):
        raise ValueError('Finite positive solved multi-AP channel gains and explicit noise seed required')
    signal=np.zeros(gain.shape,dtype=float);signal[solved]=tx_power_dbm+10*np.log10(gain[solved])
    shape=gain.shape+(len(profiles),)
    value=np.zeros(shape,dtype='f4');heard=np.zeros(shape,dtype=bool);censored=np.zeros(shape,dtype=bool)
    bounds=np.zeros(shape,dtype='f4');interval=np.zeros(shape+(2,),dtype='f4');floors=np.zeros(shape,dtype='f4')
    thermal=10*math.log10(1.380649e-23*290.*BANDWIDTH_HZ*1000.)
    for device,profile in enumerate(profiles):
        rng=np.random.default_rng(np.random.SeedSequence([noise_seed,device,619]))
        measured=signal+profile.rssi_offset_db+rng.normal(0.,MEASUREMENT_SIGMA_DB,gain.shape)
        floor=thermal+profile.noise_figure_db+profile.detection_snr_db+profile.rssi_offset_db
        quant=np.rint(measured/profile.quantization_db)*profile.quantization_db
        lo,hi=quant-profile.quantization_db/2,quant+profile.quantization_db/2
        floors[...,device]=floor
        for pose in range(gain.shape[0]):
            for band in range(3):
                for direction in range(2):
                    known=solved[pose,:,band,direction]
                    candidates=np.flatnonzero(known & (measured[pose,:,band,direction]>=floor))
                    order=candidates[np.lexsort((candidates,-measured[pose,candidates,band,direction]))]
                    selected=order[:TOP_K]
                    heard[pose,selected,band,direction,device]=True
                    censor=known.copy();censor[selected]=False
                    censored[pose,:,band,direction,device]=censor
                    bound=max(floor,float(hi[pose,selected[-1],band,direction])) if len(selected)==TOP_K else floor
                    bounds[pose,censor,band,direction,device]=bound
        h=heard[...,device]
        value[...,device]=np.where(h,quant,0.)
        interval[...,device,0]=np.where(h,lo,0.);interval[...,device,1]=np.where(h,hi,0.)
    return {'rssi_dbm':value,'heard':heard,'censored':censored,'censor_upper_dbm':bounds,
        'quantization_interval_dbm':interval,'reported_detection_threshold_dbm':floors,
        'noise_floor_dbm':np.asarray([thermal+p.noise_figure_db for p in profiles],dtype='f4')}


def _trace_case(spec,folder,*,samples_per_src=32768,max_depth=3,max_num_paths_per_src=4096):
    """Only trace_case's published-source guard may enter this scientific seam."""
    folder=Path(folder);xml,source=write_scene(spec,folder/'scene')
    import drjit as dr
    dr.set_thread_count(4)
    mi,load_scene,PlanarArray,Transmitter,Receiver,PathSolver=rt._rt()
    if dr.thread_count()!=4:raise RuntimeError('Declared four-thread CPU RT budget was not applied')
    g=spec.geometry;nr,na=len(g.rx_positions_m),len(g.ap_positions_m)
    cfr=np.empty((nr,na,3,2,64),dtype='complex64');solved=np.empty(cfr.shape[:-1],dtype=bool)
    solves=[];raw=[]
    for band,frequency in enumerate(BANDS_HZ):
        for direction in (0,1):
            began=time.perf_counter();scene=load_scene(str(xml));scene.frequency=frequency
            materials={name:{'relative_permittivity':float(np.asarray(m.relative_permittivity).reshape(-1)[0]),
                'conductivity_s_m':float(np.asarray(m.conductivity).reshape(-1)[0]),
                'thickness_m':float(np.asarray(m.thickness).reshape(-1)[0])} for name,m in scene.radio_materials.items()}
            scene.tx_array=PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
            scene.rx_array=PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,horizontal_spacing=.5,pattern='iso',polarization='V')
            tx,rx=(g.ap_positions_m,g.rx_positions_m) if direction==0 else (g.rx_positions_m,g.ap_positions_m)
            for i,p in enumerate(tx):scene.add(Transmitter(f'tx{i}',position=list(p)))
            for i,p in enumerate(rx):scene.add(Receiver(f'rx{i}',position=list(p)))
            paths=PathSolver()(scene,max_depth=max_depth,samples_per_src=samples_per_src,
                max_num_paths_per_src=max_num_paths_per_src,los=True,specular_reflection=True,
                diffuse_reflection=False,refraction=False,seed=g.seed+band)
            a,tau=paths.cir(normalize_delays=False,out_type='numpy')
            offsets=(np.arange(64,dtype='f4')-31.5)*(BANDWIDTH_HZ/64)
            channel=paths.cfr(frequencies=mi.Float(offsets),normalize_delays=False,normalize=False,out_type='numpy')
            coefficients,delays=a[:,0,:,0,:,0],tau;response=channel[:,0,:,0,0,:]
            if direction:coefficients,delays,response=coefficients.transpose(1,0,2),delays.transpose(1,0,2),response.transpose(1,0,2)
            if response.shape!=(nr,na,64) or not all(np.isfinite(x).all() for x in (coefficients,delays,response)):
                raise RuntimeError('Actual Sionna physical response has invalid axes or values')
            valid=(delays>=0)&(np.abs(coefficients)>0)
            distance=np.linalg.norm(np.asarray(g.rx_positions_m)[:,None]-np.asarray(g.ap_positions_m)[None],axis=-1)
            if np.any(valid & (delays+2e-11<distance[...,None]/rt.C)):raise RuntimeError('RT propagation causality failed')
            cfr[:,:,band,direction]=response;solved[:,:,band,direction]=valid.any(-1)
            path=folder/f'band-{band}-direction-{direction}-cir.npz'
            import io
            stream=io.BytesIO();np.savez_compressed(stream,coefficients=coefficients,delay_s=delays,cfr=response,
                frequency_offsets_hz=offsets,valid_path=valid);_write(path,stream.getvalue());raw.append(path)
            solves.append({'band_index':band,'carrier_hz':frequency,'direction':direction,
                'frequency_updated_materials':materials,'max_found_paths':int(valid.sum(-1).max(initial=0)),
                'solved_links':int(valid.any(-1).sum()),'total_links':nr*na,
                'elapsed_seconds':time.perf_counter()-began})
    gain=np.mean(np.abs(cfr.astype('complex128'))**2,axis=-1);valid=solved&(gain>0)
    reports=reports_from_gain(gain,valid,spec.tx_power_dbm,noise_seed=g.seed)
    reports.update(ap_positions_m=np.asarray(g.ap_positions_m,dtype='f4'),rx_positions_m=np.asarray(g.rx_positions_m,dtype='f4'),
        carrier_hz=np.asarray(BANDS_HZ,dtype='f8'),bbox_dims_m=np.asarray((g.width_m,g.depth_m,g.ceiling_m),dtype='f8'),
        device_ids=np.asarray([p.device_id for p in PROFILES]))
    import io
    for filename,values in (('public_views.npz',reports),('physical_channels.npz',{'cfr':cfr,'channel_power_gain':gain,'solved':valid})):
        stream=io.BytesIO();np.savez_compressed(stream,**values);path=folder/filename;_write(path,stream.getvalue());raw.append(path)
    return {'schema':1,'case':spec.to_dict(),'case_id':spec.case_id,'layout_id':spec.layout_id,
        'public_geometry_sha256':spec.geometry_sha256,'public_geometry':public_geometry(g),
        'known_tx_power_dbm':spec.tx_power_dbm,'receiver_profiles':[asdict(p) for p in PROFILES],
        'solver':{'samples_per_src':samples_per_src,'max_depth':max_depth,'max_num_paths_per_src':max_num_paths_per_src,'seed':g.seed},
        'runtime':{'backend':mi.variant(),'drjit_threads':int(dr.thread_count()),'requested_drjit_threads':4},
        'physical_axis_order':['rx_pose','ap','carrier','direction'],
        'public_rssi_axis_order':['rx_pose','ap','carrier','direction','device'],
        'bands_hz':list(BANDS_HZ),'bandwidth_hz':BANDWIDTH_HZ,
        'top_k_per_band':TOP_K,'receiver_measurement_sigma_db':MEASUREMENT_SIGMA_DB,
        'receiver_noise_draws_shared_across_material_states':True,
        'power_reference':'RT voltage channel gain; known per-layout TX applied as linear power scaling',
        'normalization_or_passivity_clipping_applied':False,
        'solves':solves,'versions':{n:importlib.metadata.version(n) for n in ('sionna-rt','mitsuba','drjit','numpy')},
        'scope':'actual Sionna RT static localized material fields and simulated noisy receiver reports',
        'real_device_measurements':False,'network_or_delivery_clocks_claimed':False,
        'finite_path_search_converged':False,'unresolved_paths_are_censored':False,
        'files_sha256':{str(p.relative_to(folder)):digest(p) for p in source+raw}}


def expected_case_files(spec):
    meshes=('floor','ceiling','wall_left','wall_right','wall_front','wall_back')+tuple(x[0] for x in obstacle_boxes(spec.geometry))
    return {'scene/scene.xml','public_views.npz','physical_channels.npz'}|{
        'scene/meshes/'+name+'.ply' for name in meshes}|{
        f'band-{band}-direction-{direction}-cir.npz' for band in range(3) for direction in range(2)}


def validate_case_receipt(spec,folder,*,source_sha256=None,registry_sha256=None,reconstruct_reports=False):
    """Passive byte/identity check; no RF regeneration or private target reads."""
    folder=Path(folder);receipt=json.loads((folder/'receipt.json').read_text())
    if (receipt.get('case')!=json.loads(encoded(spec.to_dict())) or receipt.get('case_id')!=spec.case_id
            or receipt.get('public_geometry_sha256')!=spec.geometry_sha256
            or receipt.get('unresolved_paths_are_censored') is not False):raise RuntimeError('RT case identity changed')
    if source_sha256 is not None and receipt.get('source_receipt_sha256')!=source_sha256:raise RuntimeError('RT source identity changed')
    if registry_sha256 is not None and receipt.get('registry_sha256')!=registry_sha256:raise RuntimeError('RT registry identity changed')
    expected_files=expected_case_files(spec)
    if set(receipt['files_sha256'])!=expected_files:raise RuntimeError('Canonical RT raw-file membership changed')
    expected=expected_files|{'receipt.json','attempt.json'}
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual!=expected:raise RuntimeError('Exact RT case membership changed')
    for name,h in receipt['files_sha256'].items():
        relative=Path(name);path=folder/relative
        if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or digest(path)!=h:
            raise RuntimeError('RT case evidence bytes changed')
    attempt=json.loads((folder/'attempt.json').read_text())
    if (attempt.get('case_id')!=spec.case_id or type(attempt.get('created_unix_ns')) is not int
            or attempt['created_unix_ns']<=0 or attempt.get('authorized') is not True
            or attempt.get('source_receipt_sha256')!=receipt.get('source_receipt_sha256')
            or attempt.get('registry_sha256')!=receipt.get('registry_sha256')):
        raise RuntimeError('Durable RT attempt identity changed')
    if (receipt.get('solver')!={'samples_per_src':32768,'max_depth':3,'max_num_paths_per_src':4096,'seed':spec.geometry.seed}
            or receipt.get('bands_hz')!=list(BANDS_HZ) or receipt.get('bandwidth_hz')!=BANDWIDTH_HZ
            or receipt.get('top_k_per_band')!=TOP_K or receipt.get('receiver_measurement_sigma_db')!=MEASUREMENT_SIGMA_DB
            or receipt.get('runtime',{}).get('drjit_threads')!=4 or receipt.get('runtime',{}).get('requested_drjit_threads')!=4
            or receipt.get('runtime',{}).get('backend')!='llvm_ad_mono_polarized'
            or receipt.get('receiver_profiles')!=[asdict(p) for p in PROFILES]):
        raise RuntimeError('Registered RT solver/receiver/thread configuration changed')
    if reconstruct_reports:
        with np.load(folder/'physical_channels.npz',allow_pickle=False) as raw:
            reports=reports_from_gain(raw['channel_power_gain'],raw['solved'],spec.tx_power_dbm,noise_seed=spec.geometry.seed)
        with np.load(folder/'public_views.npz',allow_pickle=False) as public:
            if any(not np.array_equal(public[name],value) for name,value in reports.items()):
                raise RuntimeError('Reported scans differ from their actual physical solve and receiver process')
    return receipt


def trace_case(spec,folder,*,authorize):
    """Once-only scientific collector, with guard before any scene/solver work."""
    if not isinstance(spec,CaseSpec) or spec!=case_spec(spec.stage,spec.index,spec.state):raise ValueError('Exact registered RT case required')
    if not callable(authorize):raise TypeError('External source/publication authorization required')
    permission=authorize(spec)
    if (not isinstance(permission,Mapping) or permission.get('authorized') is not True
            or permission.get('case_id')!=spec.case_id
            or not all(isinstance(permission.get(k),str) and re.fullmatch('[0-9a-f]{64}',permission[k])
                       for k in ('source_receipt_sha256','registry_sha256'))
            or not isinstance(permission.get('case_path'),str) or not permission['case_path']):
        raise RuntimeError('RT publication authorization refused')
    folder=Path(folder)
    expected=Path(permission['case_path'])
    if not expected.is_absolute() or folder.resolve()!=expected.resolve() or folder.is_symlink():
        raise RuntimeError('RT output differs from the authorized canonical case path')
    if (folder/'receipt.json').exists():return validate_case_receipt(spec,folder,
        source_sha256=permission['source_receipt_sha256'],registry_sha256=permission['registry_sha256'])
    if folder.exists():raise RuntimeError('Partial attempted RT case exists; no replacement/retiming')
    _write(folder/'attempt.json',encoded({'case_id':spec.case_id,'created_unix_ns':time.time_ns(),**dict(permission)}))
    receipt=_trace_case(spec,folder)
    permission_after=authorize(spec)
    if not isinstance(permission_after,Mapping) or dict(permission_after)!=dict(permission):
        raise RuntimeError('RT authorization changed during trace')
    receipt.update(source_receipt_sha256=permission['source_receipt_sha256'],registry_sha256=permission['registry_sha256'])
    _write(folder/'receipt.json',encoded(receipt))
    return validate_case_receipt(spec,folder,source_sha256=permission['source_receipt_sha256'],registry_sha256=permission['registry_sha256'],reconstruct_reports=True)
