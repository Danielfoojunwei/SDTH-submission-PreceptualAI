"""Explicit admission of real or simulated complex-channel OBSERVATIONS.

Previous RT studies treated CFR as private simulator truth. Reading it here
requires a NEW protocol declaring those exact bytes to be observations. This
does not retroactively change any earlier baseline or evidence contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .calibration import PhaseCoherenceProof, immutable, sto_cpo_correct


MEASURED_ROOT = Path('/workspace/upstream/scene-radio/csi/dichasus')
_RECORDINGS = {'dichasus-cf06': 'cf0x', 'dichasus-ad00': 'adxx', 'dichasus-dc01': 'dcxx'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class ObservationAdmission:
    protocol_id: str
    split: str
    files_sha256: Mapping[str, str]
    purpose: str = 'inspection'
    admit_ideal_rt_phase: bool = False
    public_mobile_pose: bool = False
    data_role: str = 'complex_cfr_observations'

    def check(self, path, logical_name):
        if (not isinstance(self.protocol_id, str) or not self.protocol_id
                or self.purpose not in ('inspection', 'training', 'evaluation')
                or self.split not in ('training', 'validation', 'development', 'test', 'field_test')
                or self.data_role != 'complex_cfr_observations'
                or type(self.admit_ideal_rt_phase) is not bool
                or type(self.public_mobile_pose) is not bool):
            raise ValueError('A new explicit complex-observation protocol admission is required')
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or redirected admitted observation file')
        expected = self.files_sha256.get(logical_name)
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Observation file is not declared by the new protocol: ' + logical_name)
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Admitted observation bytes changed: ' + logical_name)
        return actual


@dataclass(frozen=True)
class ChannelGauge:
    delay_reference: str
    power_reference: str
    sto_cpo_applied: bool
    ideal_simulation_admitted: bool
    qualifications: tuple[str, ...]


@dataclass(frozen=True)
class ComplexObservation:
    name: str
    cfr: np.ndarray
    frequency_offsets_hz: np.ndarray
    carrier_hz: np.ndarray
    observed: np.ndarray
    infrastructure_xyz_m: np.ndarray
    group_ids: np.ndarray
    capture_time_s: np.ndarray | None
    link_roles: tuple[str, ...]
    gauge: ChannelGauge
    source_sha256: Mapping[str, str]
    coherence_proof: PhaseCoherenceProof | None
    public_mobile_xyz_m: np.ndarray | None = None
    _private_path: Path = field(default=Path(), repr=False, compare=False)
    _private_sha_key: str = field(default='', repr=False, compare=False)
    _pose_indices: np.ndarray | None = field(default=None, repr=False, compare=False)
    _private_pose_field: str = field(default='', repr=False, compare=False)

    def encoder_features(self):
        """No material, ray delay, mobile pose, solver target or context ID."""
        return {'cfr': self.cfr, 'frequency_offsets_hz': self.frequency_offsets_hz,
                'carrier_hz': self.carrier_hz, 'observed': self.observed,
                'infrastructure_xyz_m': self.infrastructure_xyz_m, 'group_ids': self.group_ids}


def dichasus_sources(name, root=MEASURED_ROOT):
    """Inventory paths for protocol publication; does not interpret CSI values."""
    if name not in _RECORDINGS:
        raise ValueError('Recording is outside the acquired complex-CSI inventory')
    root = Path(root)
    prefix = 'measured:' + name + ':'
    family = _RECORDINGS[name]
    return {prefix + 'npz': root / (name + '-first1024.npz'),
            prefix + 'receipt': root / (name + '-first1024.json'),
            prefix + 'binding': root / (name + '-binding.json'),
            prefix + 'calibration': root.parent / 'research/dichasus' /
                f'content__datasets__data__dichasus-{family}__reftx-offsets-{name}.json'}


def load_dichasus(name, admission, root=MEASURED_ROOT):
    if not isinstance(admission, ObservationAdmission):
        raise TypeError('Supply the new protocol observation admission')
    if admission.purpose == 'inspection' and (name != 'dichasus-cf06' or admission.split != 'training'):
        raise ValueError('Pre-registration inspection is limited to the observed training recording')
    paths = dichasus_sources(name, root)
    identities = {key: admission.check(path, key) for key, path in paths.items()}
    base = 'measured:' + name + ':'
    receipt = json.loads(paths[base + 'receipt'].read_text())
    binding = json.loads(paths[base + 'binding'].read_text())
    offsets = json.loads(paths[base + 'calibration'].read_text())
    if (receipt.get('source_recording') != name or binding.get('recording') != name
            or receipt.get('parsed_npz_sha256') != identities[base + 'npz']
            or receipt.get('all_length_and_payload_crc32c_pass') is not True):
        raise ValueError('Actual acquired source identity or CRC receipt differs')
    with np.load(paths[base + 'npz'], allow_pickle=False) as saved:
        raw = saved['csi']
        times = np.asarray(saved['time'], dtype='f8').reshape(-1)
        if raw.dtype != np.float32 or raw.ndim != 4 or raw.shape[2:] != (1024, 2):
            raise ValueError('Actual source complex CSI axes/dtype changed')
        h = raw[..., 0] + 1j * raw[..., 1]
    if (len(h) != receipt.get('acquired_records') or len(times) != len(h)
            or not np.isfinite(h).all() or not np.isfinite(times).all()):
        raise ValueError('Invalid acquired complex samples/timestamps')
    h = np.fft.fftshift(h, axes=-1)
    h = sto_cpo_correct(h, offsets['sto'], offsets['cpo'], fft_size=1024)
    groups = np.asarray(binding['antenna_group_ids'])
    xyz = np.asarray(binding['array_centers_by_antenna_m'], dtype='f8')
    if (groups.shape != (h.shape[1],) or groups.dtype.kind not in 'iu'
            or (groups < 0).any() or xyz.shape != (h.shape[1], 3) or not np.isfinite(xyz).all()):
        raise ValueError('Invalid public array identities/CAD centers')
    carrier, bandwidth = float(binding['frequency_hz']), float(binding['bandwidth_hz'])
    if (not np.isfinite(carrier) or not np.isfinite(bandwidth) or bandwidth <= 0
            or carrier <= bandwidth / 2 or carrier != receipt['frequency_hz']
            or bandwidth != receipt['bandwidth_hz']):
        raise ValueError('Actual carrier/bandwidth metadata differs')
    frequencies = (np.arange(1024, dtype='f8') - 512) * bandwidth / 1024
    mobile = None
    if admission.public_mobile_pose:
        with np.load(paths[base + 'npz'], allow_pickle=False) as saved:
            mobile = _positions(saved['pos-tachy'], len(h))
    gauge = ChannelGauge('unknown_sampling_origin', 'receiver_referenced_voltage', True, False,
        ('STO/CPO estimates are applied; no bounded global phase-coherence proof is available.',
         'CFO is not Doppler ground truth; raw coefficient magnitude is not absolute received mW.',
         'Public geometry uses array centers; absent array roll/element orientation is not invented.'))
    return ComplexObservation(name, immutable(h), immutable(frequencies),
        immutable(np.full(len(h), carrier)), immutable(np.ones(h.shape[:2], dtype=bool)),
        immutable(xyz), immutable(groups.astype('i8')), immutable(times),
        tuple('mobile_tx_to_array_rx' for _ in range(len(h))), gauge, identities, None,
        mobile, paths[base + 'npz'], base + 'npz', None, 'pos-tachy')


def sionna_sources(folder):
    folder = Path(folder)
    if not re.fullmatch('[A-Za-z0-9_-]+', folder.name):
        raise ValueError('Invalid scene identity')
    prefix = 'rt:' + folder.name + ':'
    return {prefix + 'cfr': folder / 'physical_channels.npz',
            prefix + 'public': folder / 'public_views.npz', prefix + 'receipt': folder / 'receipt.json'}


def load_sionna(folder, admission):
    """Read only aggregate CFR and public metadata, never per-ray truth files."""
    if not isinstance(admission, ObservationAdmission):
        raise TypeError('Supply the new complex-observation protocol admission')
    folder = Path(folder)
    if admission.purpose == 'inspection' and (admission.split != 'training' or 'training' not in folder.parts):
        raise ValueError('Pre-registration RT inspection is limited to training data')
    paths = sionna_sources(folder)
    identities = {key: admission.check(path, key) for key, path in paths.items()}
    base = 'rt:' + folder.name + ':'
    receipt = json.loads(paths[base + 'receipt'].read_text())
    if (receipt.get('scene_id') != folder.name or receipt.get('real_device_measurements') is not False
            or receipt.get('physical_axis_order') != ['rx_pose', 'ap', 'carrier', 'direction']
            or receipt.get('directions') != ['AP_to_client', 'client_to_AP']):
        raise ValueError('RT source identity/simulation scope changed')
    for logical, filename in (('cfr', 'physical_channels.npz'), ('public', 'public_views.npz')):
        if receipt['files_sha256'].get(filename) != identities[base + logical]:
            raise ValueError('RT observation bytes differ from existing immutable receipt')
    with np.load(paths[base + 'cfr'], allow_pickle=False) as saved:
        raw = saved['cfr']  # Other keys include old private targets: do not read them.
    with np.load(paths[base + 'public'], allow_pickle=False) as saved:
        xyz = np.asarray(saved['ap_positions_m'], dtype='f8')
        bands = np.asarray(saved['carrier_hz'], dtype='f8')
        mask = saved['physical_link_solved']
        mobile = np.asarray(saved['rx_positions_m'], dtype='f8') if admission.public_mobile_pose else None
    if (raw.dtype.kind != 'c' or raw.ndim != 5 or raw.shape[3] != 2
            or raw.shape[-1] != 64 or mask.dtype.kind != 'b' or mask.shape != raw.shape[:-1]
            or xyz.shape != (raw.shape[1], 3) or not np.isfinite(xyz).all()
            or bands.shape != (raw.shape[2],) or not np.isfinite(bands).all() or (bands <= 0).any()):
        raise ValueError('RT observable CFR/public axes differ')
    if not np.isfinite(raw[mask]).all():
        raise ValueError('Nonfinite observed RT channel')
    nr, links, nb, nd, nf = raw.shape
    h = np.where(mask[..., None], raw, 0.).transpose(0, 2, 3, 1, 4).reshape(-1, links, nf)
    observed = mask.transpose(0, 2, 3, 1).reshape(-1, links)
    carrier = np.broadcast_to(bands[None, :, None], (nr, nb, nd)).reshape(-1)
    bandwidth = float(receipt['bandwidth_hz'])
    if not np.isfinite(bandwidth) or bandwidth <= 0 or not np.array_equal(bands, receipt['bands_hz']):
        raise ValueError('RT public frequency metadata differs')
    frequencies = (np.arange(nf, dtype='f8') - (nf - 1) / 2) * bandwidth / nf
    pose_indices = np.repeat(np.arange(nr), nb * nd)
    if mobile is not None:
        mobile = immutable(_positions(mobile, nr)[pose_indices])
    absolute = (receipt.get('absolute_delays_identified') is True
        and receipt.get('normalization_or_passivity_clipping_applied') is False)
    proof = None
    if admission.admit_ideal_rt_phase:
        if not absolute or receipt.get('receiver_profile_scope') != 'RSSI offset/quantization/NF/sensitivity only; no CSI front-end impairments':
            raise ValueError('Source does not justify the admitted ideal phase reference')
        proof = PhaseCoherenceProof(admission.protocol_id + ':ideal-renderer-reference', 'global',
            'ideal_simulation', identities, tuple(range(links)), 0., 0.)
    gauge = ChannelGauge('ideal_simulator_absolute' if absolute else 'unknown_sampling_origin',
        'dimensionless_physical_channel_voltage', False, admission.admit_ideal_rt_phase,
        ('Old private CFR is admitted as an observation ONLY by this new protocol.',
         'Static isolated probes have no real acquisition timestamps, MAC, motion or CSI front-end impairments.',
         'Finite path search is not a converged full electromagnetic ground truth.',
         'AP CAD centers are public; mobile poses remain private unless explicitly admitted.'))
    roles = tuple(['AP_to_client', 'client_to_AP'][i % 2] for i in range(len(h)))
    return ComplexObservation(folder.name, immutable(h), immutable(frequencies), immutable(carrier),
        immutable(observed), immutable(xyz), immutable(np.arange(links, dtype='i8')), None,
        roles, gauge, identities, proof, mobile, paths[base + 'public'], base + 'public',
        immutable(pose_indices), 'rx_positions_m')


def _positions(value, count):
    a = np.asarray(value, dtype='f8')
    if a.shape != (count, 3) or not np.isfinite(a).all():
        raise ValueError('Malformed explicitly accessed mobile-pose labels')
    return immutable(a)


def private_labels(observation, *, purpose):
    """Explicit split/frozen-probe labels. Materials and ray paths are not exposed."""
    if purpose not in ('partition', 'frozen_probe', 'evaluation'):
        raise ValueError('Private pose access requires an explicit evaluation/probe purpose')
    if sha256(observation._private_path) != observation.source_sha256.get(observation._private_sha_key):
        raise ValueError('Source changed before private-label access')
    with np.load(observation._private_path, allow_pickle=False) as saved:
        value = saved[observation._private_pose_field]
    if observation._pose_indices is None:
        return {'mobile_position_m': _positions(value, len(observation.cfr))}
    source = _positions(value, int(observation._pose_indices.max()) + 1)
    return {'mobile_position_m': immutable(source[observation._pose_indices])}


def require_coherent_view(observation, proof=None, *, budget=None):
    proof = observation.coherence_proof if proof is None else proof
    if not isinstance(proof, PhaseCoherenceProof):
        raise ValueError('Phase-coherent views are unavailable without explicit bound evidence')
    return proof.require(observation.source_sha256, observation.group_ids,
                         ideal_simulation_admitted=observation.gauge.ideal_simulation_admitted,
                         budget=budget)
