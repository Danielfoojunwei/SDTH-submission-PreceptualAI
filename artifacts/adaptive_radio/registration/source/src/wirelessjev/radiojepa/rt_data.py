"""Actual CPU ray-traced static scene views, with a publication authorization seam.

These are simulated electromagnetic channels and simulated receiver profiles.
They are never labeled as hardware measurements. Public scene specifications
can be written before a study starts; tracing registered scenes requires an
explicit source/publication permission callback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping
import xml.etree.ElementTree as ET

import numpy as np

BANDS_HZ = (2.4e9, 5.2e9, 6.0e9)
BANDWIDTH_HZ = 20e6
REFERENCE_TX_DBM = 20.
C = 299792458.
STAGES = ('training', 'validation', 'development', 'confirmation')


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def fingerprint(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise RuntimeError('Refuse to overwrite RT evidence: '+str(path))
    return path


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    rssi_offset_db: float
    quantization_db: float
    noise_figure_db: float
    detection_snr_db: float = 3.
    held_out: bool = False

    def __post_init__(self):
        if (not self.device_id or not all(math.isfinite(x) for x in
                (self.rssi_offset_db, self.quantization_db, self.noise_figure_db, self.detection_snr_db))
                or self.quantization_db <= 0 or self.noise_figure_db < 0
                or type(self.held_out) is not bool):
            raise ValueError('Invalid explicitly simulated receiver profile')


DEVICE_PROFILES = (
    DeviceProfile('sim-rssi-a', 0., 1., 4.),
    DeviceProfile('sim-rssi-b', -3., .5, 7.),
    DeviceProfile('sim-rssi-c', 4., 1., 9.),
    DeviceProfile('sim-rssi-heldout-d', 6., 2., 11., held_out=True),
)


@dataclass(frozen=True)
class GeometrySpec:
    scene_id: str
    family: str
    width_m: float
    depth_m: float
    ceiling_m: float
    obstacle_height_m: float
    obstacle_material: str
    ap_positions_m: tuple[tuple[float, float, float], ...]
    rx_positions_m: tuple[tuple[float, float, float], ...]
    seed: int

    def __post_init__(self):
        if (not self.scene_id or '/' in self.scene_id or '..' in self.scene_id
                or self.family not in ('rack_aisles', 'partitioned_room')
                or self.obstacle_material not in ('metal', 'wood', 'brick', 'concrete')
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError('Invalid scene identity/material/family/seed')
        if (not all(math.isfinite(v) and v > 0 for v in
                    (self.width_m, self.depth_m, self.ceiling_m, self.obstacle_height_m))
                or self.obstacle_height_m >= self.ceiling_m):
            raise ValueError('Invalid physical scene dimensions')
        for points in (self.ap_positions_m, self.rx_positions_m):
            a = np.asarray(points, dtype=float)
            if (a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all()
                    or (np.abs(a[:, 0]) >= self.width_m/2).any()
                    or (np.abs(a[:, 1]) >= self.depth_m/2).any()
                    or (a[:, 2] <= 0).any() or (a[:, 2] >= self.ceiling_m).any()):
                raise ValueError('Antenna positions must lie strictly inside the room')
        distance = np.linalg.norm(np.asarray(self.rx_positions_m)[:, None]
                                  -np.asarray(self.ap_positions_m)[None], axis=-1)
        if (distance < .5).any():
            raise ValueError('This far-field generator requires AP/client separation>=0.5m')
        # Frozen dataclasses must not retain mutable lists after a JSON round trip.
        object.__setattr__(self,'ap_positions_m',tuple(tuple(float(x) for x in p) for p in self.ap_positions_m))
        object.__setattr__(self,'rx_positions_m',tuple(tuple(float(x) for x in p) for p in self.rx_positions_m))

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return fingerprint(self.to_dict())


def _spec(stage, i):
    if stage not in STAGES:
        raise ValueError('Unknown registered stage')
    # A public specification, independent of traced RF and downstream outcomes.
    offset = {'training': 0, 'validation': 101, 'development': 201, 'confirmation': 401}[stage]
    k = i+offset
    if stage == 'confirmation':
        width, depth = 7.+(i%6)*1.9, 9.+(i//6)*3.1
        ceiling = 3.6+.25*(i%4)
    else:
        width, depth = 9.+(k%4)*1.2, 11.+((k//2)%4)*1.3
        ceiling = 4.+.2*(k%3)
        # A new scene ID/seed alone is not a held-out physical layout.
        if stage == 'validation':
            width,depth,ceiling = width+.37,depth+.61,ceiling+.11
        elif stage == 'development':
            width,depth,ceiling = width+.73,depth+.29,ceiling+.17
    height = min(ceiling-.6, 2.1+.2*(k%4))
    aps = ((-.34*width, -.34*depth, ceiling-.55),
           (.34*width, .34*depth, ceiling-.55))
    rx = tuple((x*width, y*depth, 1.1) for y in (-.34, -.12, .12, .34)
               for x in (-.34, 0., .34))
    material = ('metal', 'wood')[k%2]
    if stage == 'confirmation' and i%5 == 0:
        material = 'brick'
    return GeometrySpec(f'radiojepa-{stage}-{i:02d}',
        ('rack_aisles', 'partitioned_room')[k%2], width, depth, ceiling,
        height, material, aps, rx, 731000+k)


def geometry_registry():
    """Build only public specs: no Mitsuba/Sionna import or RF generation."""
    counts = dict(training=8, validation=2, development=2, confirmation=24)
    return {'schema': 1,
        'scope': 'Static simulated geometric/material configurations, not captured physical warehouses',
        'bands_hz': list(BANDS_HZ), 'bandwidth_hz': BANDWIDTH_HZ,
        'reference_tx_dbm': REFERENCE_TX_DBM,
        'device_profiles': [asdict(p) for p in DEVICE_PROFILES],
        'training_device_ids': [p.device_id for p in DEVICE_PROFILES if not p.held_out],
        'held_out_device_ids': [p.device_id for p in DEVICE_PROFILES if p.held_out],
        'stages': {stage: [s.to_dict() | {'spec_sha256': s.sha256}
                   for s in (_spec(stage, i) for i in range(count))]
                   for stage, count in counts.items()},
        'reserved_confirmation_contains_specs_only': True}


def write_registry(path):
    value = geometry_registry()
    _write_once(path, _encoded(value))
    return {'path': str(path), 'registry_sha256': fingerprint(value)}


def _ply_box(low, high):
    x0, y0, z0 = low
    x1, y1, z1 = high
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError('Degenerate mesh box')
    vertices = ((x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1))
    faces = ((0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7))
    header = 'ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n'
    return (header+''.join(' '.join(map(str,v))+'\n' for v in vertices)
            +''.join('3 '+' '.join(map(str,f))+'\n' for f in faces)).encode()


def write_scene(spec: GeometrySpec, folder):
    """Write deterministic CAD/XML; this function performs no ray tracing."""
    folder = Path(folder)
    w, d, h = spec.width_m/2, spec.depth_m/2, spec.ceiling_m
    boxes = [
        ('floor','concrete',(-w,-d,-.15),(w,d,0.)),
        ('ceiling','concrete',(-w,-d,h),(w,d,h+.15)),
        ('wall_left','brick',(-w-.15,-d,0.),(-w,d,h)),
        ('wall_right','brick',(w,-d,0.),(w+.15,d,h)),
        ('wall_front','brick',(-w,-d-.15,0.),(w,-d,h)),
        ('wall_back','brick',(-w,d,0.),(w,d+.15,h)),
    ]
    if spec.family == 'rack_aisles':
        for j, x in enumerate((-.18*spec.width_m, .18*spec.width_m)):
            boxes.append((f'rack{j}', 'obstacle', (x-.2,-.25*spec.depth_m,0.),
                          (x+.2,.25*spec.depth_m,spec.obstacle_height_m)))
    else:
        gap = .18*spec.width_m
        boxes += [('partition_left','obstacle',(-w,-.08,0.),(-gap,.08,spec.obstacle_height_m)),
                  ('partition_right','obstacle',(gap,-.08,0.),(w,.08,spec.obstacle_height_m))]
    root = ET.Element('scene', version='2.1.0')
    for name, material in (('concrete','concrete'),('brick','brick'),('obstacle',spec.obstacle_material)):
        bsdf = ET.SubElement(root,'bsdf',type='itu-radio-material',id=name)
        ET.SubElement(bsdf,'string',name='type',value=material)
        ET.SubElement(bsdf,'float',name='thickness',value='.15')
    files = []
    for name, material, low, high in boxes:
        mesh = folder/'meshes'/(name+'.ply')
        _write_once(mesh,_ply_box(low,high)); files.append(mesh)
        shape = ET.SubElement(root,'shape',type='ply',id=name)
        ET.SubElement(shape,'string',name='filename',value='meshes/'+mesh.name)
        ET.SubElement(shape,'boolean',name='face_normals',value='true')
        ET.SubElement(shape,'ref',name='bsdf',id=material)
    xml = folder/'scene.xml'
    _write_once(xml,ET.tostring(root,encoding='utf-8',xml_declaration=True)); files.append(xml)
    return xml, files


def device_observations(signal_dbm, solved, *, profiles=DEVICE_PROFILES, bandwidth_hz=BANDWIDTH_HZ):
    """Simulated deterministic receiver response, preserving censor/unknown masks.

    Noise figure changes noise and sensitivity, not the incident signal power.
    No-path numerical failures remain unresolved, never invented weak labels.
    """
    signal, solved = np.asarray(signal_dbm, dtype=float), np.asarray(solved)
    if (signal.shape != solved.shape or solved.dtype != bool or not np.isfinite(signal[solved]).all()
            or not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0):
        raise ValueError('Invalid physical signal/mask/bandwidth')
    values, heard, censored, upper, intervals, noise = [], [], [], [], [], []
    thermal = 10*math.log10(1.380649e-23*290.*bandwidth_hz*1000.)
    for profile in profiles:
        floor = thermal+profile.noise_figure_db
        threshold = floor+profile.detection_snr_db
        detected = solved & (signal >= threshold)
        # Quantize to nearest bin, with the corresponding interval retained.
        clean = np.where(solved, signal, 0.)+profile.rssi_offset_db
        quantized = np.floor(clean/profile.quantization_db+.5)*profile.quantization_db
        values.append(np.where(detected,quantized,0.))
        heard.append(detected)
        censored.append(solved & ~detected)
        upper.append(np.where(solved & ~detected,threshold+profile.rssi_offset_db,0.))
        intervals.append(np.stack((np.where(detected,quantized-profile.quantization_db/2,0.),
                                   np.where(detected,quantized+profile.quantization_db/2,0.)),-1))
        noise.append(floor)
    return {'rssi_dbm':np.stack(values,-1).astype('f4'),
        'heard':np.stack(heard,-1), 'censored':np.stack(censored,-1),
        'censor_upper_dbm':np.stack(upper,-1).astype('f4'),
        'quantization_interval_dbm':np.stack(intervals,-2).astype('f4'),
        'noise_floor_dbm':np.asarray(noise,dtype='f4'),
        'physical_link_solved':solved}


def _rt():
    import mitsuba as mi
    mi.set_variant('llvm_ad_mono_polarized')
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
    return mi, load_scene, PlanarArray, Transmitter, Receiver, PathSolver


def _trace_scene(spec, folder, *, bands_hz=BANDS_HZ, samples_per_src=32768,
                 max_depth=3, max_num_paths_per_src=4096, profiles=DEVICE_PROFILES):
    """Internal tracing seam. Scientific callers must use collect_stage."""
    if (not bands_hz or any(not math.isfinite(f) or f <= 0 for f in bands_hz)
            or type(samples_per_src) is not int or samples_per_src < 1
            or type(max_depth) is not int or max_depth < 0
            or type(max_num_paths_per_src) is not int or max_num_paths_per_src < 1):
        raise ValueError('Invalid solver configuration')
    folder = Path(folder)
    xml, source_files = write_scene(spec,folder/'scene')
    mi, load_scene, PlanarArray, Transmitter, Receiver, PathSolver = _rt()
    nr, na = len(spec.rx_positions_m), len(spec.ap_positions_m)
    cfr = np.empty((nr,na,len(bands_hz),2,64),dtype='complex64')
    solved = np.empty(cfr.shape[:-1],dtype=bool)
    summaries, raw_files = [], []
    for band_index, frequency in enumerate(bands_hz):
        for direction in range(2):
            began = time.perf_counter()
            scene = load_scene(str(xml)); scene.frequency = frequency
            materials = {name:{'relative_permittivity':float(np.asarray(material.relative_permittivity).reshape(-1)[0]),
                                'conductivity_s_m':float(np.asarray(material.conductivity).reshape(-1)[0]),
                                'thickness_m':float(np.asarray(material.thickness).reshape(-1)[0])}
                         for name,material in scene.radio_materials.items()}
            scene.tx_array = PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,
                horizontal_spacing=.5,pattern='iso',polarization='V')
            scene.rx_array = PlanarArray(num_rows=1,num_cols=1,vertical_spacing=.5,
                horizontal_spacing=.5,pattern='iso',polarization='V')
            tx_positions, rx_positions = ((spec.ap_positions_m,spec.rx_positions_m)
                                         if direction == 0 else (spec.rx_positions_m,spec.ap_positions_m))
            for i, p in enumerate(tx_positions): scene.add(Transmitter(f'tx{i}',position=list(p)))
            for i, p in enumerate(rx_positions): scene.add(Receiver(f'rx{i}',position=list(p)))
            paths = PathSolver()(scene,max_depth=max_depth,samples_per_src=samples_per_src,
                max_num_paths_per_src=max_num_paths_per_src,los=True,specular_reflection=True,
                diffuse_reflection=False,refraction=False,seed=spec.seed+band_index)
            a,tau = paths.cir(normalize_delays=False,out_type='numpy')
            offsets = (np.arange(64,dtype='f4')-31.5)*(BANDWIDTH_HZ/64)
            channel = paths.cfr(frequencies=mi.Float(offsets),normalize_delays=False,
                                normalize=False,out_type='numpy')
            coefficients, delays = a[:,0,:,0,:,0], tau
            response = channel[:,0,:,0,0,:]
            if direction:
                coefficients,delays,response = (coefficients.transpose(1,0,2),
                                                delays.transpose(1,0,2),response.transpose(1,0,2))
            if (response.shape != (nr,na,64) or not np.isfinite(response).all()
                    or not np.isfinite(coefficients).all() or not np.isfinite(delays).all()):
                raise RuntimeError('Sionna returned malformed/nonfinite physical paths')
            valid = (delays >= 0) & (np.abs(coefficients) > 0)
            distance = np.linalg.norm(np.asarray(spec.rx_positions_m)[:,None]
                                      -np.asarray(spec.ap_positions_m)[None],axis=-1)
            if np.any(valid & (delays+2e-11 < distance[...,None]/C)):
                raise RuntimeError('RT path violates absolute propagation causality')
            cfr[:,:,band_index,direction] = response
            solved[:,:,band_index,direction] = valid.any(-1)
            raw_path = folder/f'band-{band_index}-direction-{direction}-cir.npz'
            if raw_path.exists(): raise RuntimeError('Refuse to retime/rewrite an existing RT solve')
            with raw_path.open('xb') as stream:
                np.savez_compressed(stream,coefficients=coefficients,delay_s=delays,cfr=response,
                    frequency_offsets_hz=offsets,valid_path=valid)
            raw_files.append(raw_path)
            summaries.append({'band_index':band_index,'carrier_hz':frequency,'direction':direction,
                'frequency_updated_materials':materials,
                'max_found_paths':int(valid.sum(-1).max(initial=0)),
                'solved_links':int(valid.any(-1).sum()),'total_links':nr*na,
                'elapsed_seconds':time.perf_counter()-began})
    gain = np.mean(np.abs(cfr.astype('complex128'))**2,axis=-1)
    valid_signal = solved & (gain > 0)
    signal = np.zeros(gain.shape,dtype='f8')
    signal[valid_signal] = REFERENCE_TX_DBM+10*np.log10(gain[valid_signal])
    public = device_observations(signal,valid_signal,profiles=profiles)
    public.update(ap_positions_m=np.asarray(spec.ap_positions_m,dtype='f4'),
        rx_positions_m=np.asarray(spec.rx_positions_m,dtype='f4'),
        carrier_hz=np.asarray(bands_hz,dtype='f8'),
        training_device=np.array([not p.held_out for p in profiles]),
        device_ids=np.array([p.device_id for p in profiles]))
    public_path,physics_path = folder/'public_views.npz',folder/'physical_channels.npz'
    for path,arrays in ((public_path,public),(physics_path,{'cfr':cfr,'signal_dbm':signal,
            'channel_power_gain':gain,'solved':valid_signal})):
        if path.exists(): raise RuntimeError('Refuse to rewrite traced scene')
        with path.open('xb') as stream: np.savez_compressed(stream,**arrays)
    files = source_files+raw_files+[public_path,physics_path]
    receipt = {'schema':1,'scene_id':spec.scene_id,'spec':spec.to_dict(),'spec_sha256':spec.sha256,
        'scope':'actual Sionna ray tracing; explicitly simulated static scene and receiver profiles',
        'real_device_measurements':False,'static_scene_time_s':0.,
        'capture_delivery_clock_claimed':False,'backend':mi.variant(),
        'versions':{n:importlib.metadata.version(n) for n in ('sionna-rt','mitsuba','drjit','numpy')},
        'bands_hz':list(bands_hz),'bandwidth_hz':BANDWIDTH_HZ,'reference_tx_dbm':REFERENCE_TX_DBM,
        'receiver_profiles':[asdict(p) for p in profiles],
        'receiver_profile_scope':'RSSI offset/quantization/NF/sensitivity only; no CSI front-end impairments',
        'links_are_isolated_channel_probes':True,
        'interference_or_mac_simulated':False,
        'directions':['AP_to_client','client_to_AP'],
        'physical_axis_order':['rx_pose','ap','carrier','direction'],
        'public_rssi_axis_order':['rx_pose','ap','carrier','direction','device'],
        'rssi_definition':'20dBm +10log10(mean(abs(CFR)**2)), offset and quantization; thermal floor affects detection only',
        'power_reference':'dimensionless physical channel voltage ratios from isotropic V-polarized matched simulation arrays',
        'absolute_delays_identified':True,'normalization_or_passivity_clipping_applied':False,
        'finite_path_search_converged':False,
        'unresolved_links_are_censored':False,
        'solver':{'max_depth':max_depth,'samples_per_src':samples_per_src,
            'max_num_paths_per_src':max_num_paths_per_src,'seed_base':spec.seed,
            'los':True,'specular_reflection':True,'refraction':False,'diffuse_reflection':False},
        'solves':summaries,
        'files_sha256':{str(p.relative_to(folder)):_digest(p) for p in files}}
    _write_once(folder/'receipt.json',_encoded(receipt))
    return receipt


def trace_fixture(spec, folder, **kwargs):
    """Known-ID software integration only; reserved scientific IDs are rejected."""
    names = {s['scene_id'] for group in geometry_registry()['stages'].values() for s in group}
    if spec.scene_id in names:
        raise RuntimeError('Registered scene requires published stage authorization')
    return _trace_scene(spec,folder,**kwargs)


def _permission(stage, authorize, registry):
    proof = authorize(stage)
    if (not isinstance(proof,Mapping) or proof.get('authorized') is not True
            or proof.get('stage') != stage or proof.get('registry_sha256') != fingerprint(registry)):
        raise RuntimeError('Source publication or registered geometry authorization failed')
    if stage == 'confirmation' and proof.get('confirmation_permitted') is not True:
        raise RuntimeError('Reserved confirmation geometry may not be traced before qualification')
    return dict(proof)


def validate_scene_receipt(spec, folder, stage):
    """Passive resume checks: never reconstruct missing CAD or RF artifacts."""
    folder = Path(folder)
    receipt = json.loads((folder/'receipt.json').read_text())
    names = ['floor','ceiling','wall_left','wall_right','wall_front','wall_back']
    names += ['rack0','rack1'] if spec.family == 'rack_aisles' else ['partition_left','partition_right']
    expected = {'scene/scene.xml','public_views.npz','physical_channels.npz'}
    expected.update('scene/meshes/'+name+'.ply' for name in names)
    expected.update(f'band-{band}-direction-{direction}-cir.npz' for band in range(3) for direction in range(2))
    profiles = tuple(p for p in DEVICE_PROFILES if not p.held_out) if stage in ('training','validation') else DEVICE_PROFILES
    if (receipt['spec_sha256'] != spec.sha256 or fingerprint(receipt['spec']) != spec.sha256
            or receipt.get('real_device_measurements') is not False
            or receipt['bands_hz'] != list(BANDS_HZ) or receipt['bandwidth_hz'] != BANDWIDTH_HZ
            or receipt['reference_tx_dbm'] != REFERENCE_TX_DBM
            or receipt['receiver_profiles'] != [asdict(p) for p in profiles]
            or set(receipt['files_sha256']) != expected):
        raise RuntimeError('Existing RT identity/profile/raw membership changed')
    visible = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()
               and not p.name.endswith(('.lock','.part')) and not p.name.startswith('.')}
    if visible != expected | {'receipt.json'}:
        raise RuntimeError('Missing or extra RT scene files')
    for name,expected_sha in receipt['files_sha256'].items():
        path = folder/name
        if (not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink()
                or _digest(path) != expected_sha):
            raise RuntimeError('Existing RT raw data changed')
    with np.load(folder/'public_views.npz',allow_pickle=False) as arrays:
        shape = (len(spec.rx_positions_m),len(spec.ap_positions_m),3,2,len(profiles))
        if arrays['rssi_dbm'].shape != shape or arrays['device_ids'].tolist() != [p.device_id for p in profiles]:
            raise RuntimeError('Existing RT public device axes changed')
        if stage in ('training','validation') and not arrays['training_device'].all():
            raise RuntimeError('Held-out receiver leaked into a fitting stage')
    return receipt


def collect_stage(stage, output, *, authorize: Callable[[str], Mapping] | None = None):
    """Trace a finite registered stage only after source/publication permission."""
    if stage not in STAGES or authorize is None:
        raise RuntimeError('A registered stage and explicit authorization callback are required')
    registry = geometry_registry()
    proof = _permission(stage,authorize,registry)
    output = Path(output)
    receipts=[]
    for value in registry['stages'][stage]:
        _permission(stage,authorize,registry)
        spec = GeometrySpec(**{k:v for k,v in value.items() if k != 'spec_sha256'})
        folder = output/stage/spec.scene_id
        if (folder/'receipt.json').exists():
            receipt = validate_scene_receipt(spec,folder,stage)
        else:
            if folder.exists() and any(folder.rglob('*')):
                raise RuntimeError('Partial RT evidence exists; refuse an untracked rerun')
            profiles = tuple(p for p in DEVICE_PROFILES if not p.held_out) if stage in ('training','validation') else DEVICE_PROFILES
            receipt = _trace_scene(spec,folder,profiles=profiles)
        receipts.append({'scene_id':spec.scene_id,'receipt_sha256':_digest(folder/'receipt.json')})
    _permission(stage,authorize,registry)
    result={'schema':1,'stage':stage,'completed':True,'registry_sha256':fingerprint(registry),
            'authorization':dict(proof),'scenes':receipts,'real_device_measurements':False}
    _write_once(output/(stage+'.json'),_encoded(result))
    return result
