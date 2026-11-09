"""Analytic and negative loader fixtures; no scientific results or acquisition."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.radiojepa import data as d


def fixture_dataset(tmp_path, *, name='dichasus-cf06', csi=None, pose=None,
                    times=None, groups=None, centers=None):
    n, antennas = 120, 4
    if csi is None:
        csi = np.zeros((n, antennas, 1024, 2), 'f4')
        csi[..., 0] = 1.
    n, antennas = csi.shape[:2]
    if pose is None:
        pose = np.stack((np.zeros(n), np.linspace(-5, 5, n), np.ones(n)), -1)
    if times is None:
        times = np.arange(n, dtype='f8') / 10
    if groups is None:
        groups = np.arange(antennas)
    if centers is None:
        centers = np.stack((np.arange(antennas) * 10., np.zeros(antennas), np.ones(antennas)), -1)
    path = tmp_path / (name + '-first1024.npz')
    np.savez_compressed(path, csi=csi, time=times[:, None], **{'pos-tachy': pose})
    receipt = {'source_recording': name, 'all_length_and_payload_crc32c_pass': True,
        'parsed_npz_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'acquired_records': n, 'frequency_hz': 1.272e9, 'bandwidth_hz': 50.05586592e6}
    binding = {'recording': name, 'array_centers_by_antenna_m': np.asarray(centers).tolist(),
        'antenna_group_ids': np.asarray(groups).tolist(), 'frequency_hz': receipt['frequency_hz'],
        'bandwidth_hz': receipt['bandwidth_hz']}
    (tmp_path / (name + '-first1024.json')).write_text(json.dumps(receipt))
    (tmp_path / (name + '-binding.json')).write_text(json.dumps(binding))
    return d.load_recording(name, tmp_path)


def test_fft_order_linear_power_mean_then_db_and_known_frequency_bins(tmp_path):
    shifted = np.ones((2, 4, 1024, 2), 'f4')
    shifted[..., 1] = 0
    shifted[:, :, 8:16, 0] = 3
    raw = np.fft.ifftshift(shifted, axes=2)
    r = fixture_dataset(tmp_path, csi=raw)
    assert r.log_power_db.shape == (2, 4, 64)
    np.testing.assert_allclose(r.log_power_db[..., 0], 10*np.log10(5), atol=1.e-6)
    np.testing.assert_array_equal(r.log_power_db[..., 1:], 0)
    expected = r.carrier_hz + r.bandwidth_hz * (16*np.arange(64)+7.5-512)/1024
    np.testing.assert_allclose(r.bin_frequency_hz, expected, rtol=0, atol=1.e-6)
    assert 'not dBm' in r.power_unit
    with pytest.raises(ValueError):
        r.log_power_db[0, 0, 0] = 2


def test_absolute_referenced_power_is_preserved_not_normalized_away(tmp_path):
    first = tmp_path / 'one'; first.mkdir()
    second = tmp_path / 'two'; second.mkdir()
    base = np.zeros((4,4,1024,2), 'f4');base[...,0] = 1
    a = fixture_dataset(first,csi=base)
    b = fixture_dataset(second,csi=10*base)
    np.testing.assert_allclose(b.log_power_db-a.log_power_db, 20, atol=1.e-6)


def test_pose_labels_are_not_loaded_or_returned_as_encoder_features(tmp_path):
    pose = np.full((120,3), np.nan)
    recording = fixture_dataset(tmp_path, pose=pose)
    assert np.isfinite(recording.log_power_db).all()
    assert not {'pos-tachy', 'pose', 'source_pose'} & set(recording.public_features())
    with pytest.raises(ValueError, match='Private measured poses'):
        d.private_pose_labels(recording)


def test_source_bytes_changed_are_rejected_before_features_or_labels(tmp_path):
    recording = fixture_dataset(tmp_path)
    with recording._npz_path.open('ab') as stream:
        stream.write(b'altered-source')
    with pytest.raises(ValueError,match='bytes differ'):
        d.load_recording(recording.name,tmp_path)
    with pytest.raises(ValueError,match='Source NPZ changed'):
        d.private_pose_labels(recording)


def test_zero_power_has_explicit_floor_mask_and_count(tmp_path):
    recording = fixture_dataset(tmp_path,csi=np.zeros((3,4,1024,2),'f4'))
    np.testing.assert_array_equal(recording.log_power_db,-120)
    assert not recording.positive_power_mask.any()
    assert recording.clipped_power_bin_count == 3*4*64


def test_static_views_are_disjoint_whole_physical_arrays(tmp_path):
    r = fixture_dataset(tmp_path)
    a,b = d.view_indices(r)
    np.testing.assert_array_equal(a,[0,2]);np.testing.assert_array_equal(b,[1,3])
    assert not np.intersect1d(a,b).size
    assert not set(r.antenna_group_ids[a]) & set(r.antenna_group_ids[b])


def test_single_array_does_not_become_fabricated_second_device(tmp_path):
    r = fixture_dataset(tmp_path,name='dichasus-0152',groups=np.zeros(4,dtype=int),
        centers=np.tile([0.,0.,1.2],(4,1)))
    with pytest.raises(ValueError,match='one physical array'):
        d.view_indices(r)
    with pytest.raises(ValueError,match='whole held-out'):
        d.spatial_split(r)


def test_spatial_split_is_disjoint_complete_with_guard_and_no_pose_feature(tmp_path):
    r = fixture_dataset(tmp_path)
    split = d.spatial_split(r)
    all_indices = np.concatenate((split.train,split.validation,split.test,split.excluded))
    np.testing.assert_array_equal(np.sort(all_indices),np.arange(r.samples))
    assert split.report['minimum_train_test_xy_distance_m'] >= .25
    assert min(len(split.train),len(split.validation),len(split.test)) > 0
    assert split.report['private_pose_used_for_partition_only'] is True
    assert not hasattr(r,'positions')
    with pytest.raises(ValueError):
        split.train[0] = 3


def test_temporal_pairs_sort_real_clocks_and_cannot_cross_reserved_cohorts(tmp_path):
    times = np.arange(120,dtype=float)/10
    times[1::2] = times[1::2][::-1]
    r = fixture_dataset(tmp_path,times=times)
    np.testing.assert_array_equal(r.time_s,times)  # Loader must not silently reorder.
    split = d.spatial_split(r)
    pairs = d.temporal_partners(r,split.train)
    assert np.isin(pairs,split.train).all()
    assert (r.time_s[pairs[:,1]]-r.time_s[pairs[:,0]] >= 2).all()
    with pytest.raises(ValueError,match='spatial training split'):
        d.temporal_partners(r,np.arange(r.samples))
    with pytest.raises(ValueError,match='at least a2s'):
        d.temporal_partners(r,split.train,1.)


@pytest.mark.parametrize('bad',['../outside','/absolute','not-a-recording'])
def test_recording_names_cannot_redirect_dataset_reads(tmp_path,bad):
    with pytest.raises(ValueError,match='identifier'):
        d.load_recording(bad,tmp_path)


def test_conflicting_public_geometry_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='conflicting array centers'):
        fixture_dataset(tmp_path,groups=np.zeros(4,dtype=int))


def test_dataset_identity_and_private_labels_survive_root_relocation(tmp_path):
    import shutil
    first = tmp_path/'first'; first.mkdir()
    a = fixture_dataset(first)
    second = tmp_path/'second';shutil.copytree(first,second)
    b = d.load_recording(a.name,second)
    assert a.source_sha256 == b.source_sha256
    np.testing.assert_array_equal(a.log_power_db,b.log_power_db)
    np.testing.assert_array_equal(d.private_pose_labels(a),d.private_pose_labels(b))
