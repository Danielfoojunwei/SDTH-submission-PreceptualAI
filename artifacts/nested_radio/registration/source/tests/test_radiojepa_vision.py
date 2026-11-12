"""Tiny source-proof unit fixtures, not publisher data or neural feature runs."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from wirelessjev.radiojepa import vision as v


@pytest.fixture
def alignment_fixture(tmp_path,monkeypatch):
    # A tiny artificial publisher fixture exercises verification logic. It is
    # never represented as real DICHASUS data and never reaches a backbone.
    size, count = 256, 4
    source_md5=hashlib.md5(b'x'*size).hexdigest()
    video_bytes=b'unit-video-identity-not-a-real-movie'
    video_md5=hashlib.md5(video_bytes).hexdigest()
    monkeypatch.setattr(v,'SOURCE_BYTES',size)
    monkeypatch.setattr(v,'SOURCE_RECORDS',count)
    monkeypatch.setattr(v,'SOURCE_MD5',source_md5)
    monkeypatch.setattr(v,'VIDEO_MD5',video_md5)
    root=tmp_path
    research=root/'research/dichasus';research.mkdir(parents=True)
    publisher=research/'darus-4062.json'
    publisher.write_text(json.dumps({'data':{'latestVersion':{'files':[
        {'dataFile':{'filename':'dichasus-ad00.tfrecords','filesize':size,
                     'checksum':{'type':'MD5','value':source_md5}}},
        {'dataFile':{'filename':'dichasus-ad00.mkv','filesize':len(video_bytes),
                     'checksum':{'type':'MD5','value':video_md5}}}]}}}))
    contract=research/'dataset-details.html'
    contract.write_text('frame zero = earliest CSI datapoint; accuracy1-2 video frames')
    directory=root/'dichasus/video';directory.mkdir(parents=True)
    video=directory/'dichasus-ad00.mkv';video.write_bytes(video_bytes)
    metadata=directory/'time-metadata.npz'
    # Prefix first row is12, but true earliest full-source timestamp is10.
    np.savez_compressed(metadata,time_s=np.array([12.,10.,13.,11.]),
        record_length_bytes=np.full(count,64,dtype='i8'),pose_m=np.zeros((count,3)))
    images=directory/'aligned-samples';images.mkdir()
    rows=[]
    for i,(source_row,source_time,pts) in enumerate(((3,11.,1.),(0,12.,2.))):
        path=images/f'sample-{i+1:03d}.png'
        Image.new('RGB',(5,3),(i*30,20,90)).save(path)
        rows.append({'source_csi_row':source_row,'source_csi_time_s':source_time,
            'video_target_offset_s':pts,'selected_frame_number':int(pts*50),
            'actual_video_pts_s':pts,'extraction_time_mismatch_s':0.,
            'file':str(path.relative_to(root)),'sha256':v._digest(path)})
    log=images/'ffmpeg.log'
    log.write_text('[Parsed_showinfo_0 @ fixture] n:0 pts:50 pts_time:1.0\n'
                   '[Parsed_showinfo_0 @ fixture] n:1 pts:100 pts_time:2.0\n')
    proof={'schema':1,'status':'verified_full_recording_stream',
        'full_recording_checksum_verified':True,'all_record_length_and_payload_crc32c_pass':True,
        'received_bytes':size,'complete_records':count,'publisher_md5':source_md5,
        'received_md5':source_md5,'received_sha256':hashlib.sha256(b'x'*size).hexdigest(),
        'official_dataset_doi':v.SOURCE_DOI,'publisher_metadata_sha256':v._digest(publisher),
        'official_sync_contract_sha256':v._digest(contract),
        'full_recording_time_pose_metadata':str(metadata.relative_to(root)),
        'metadata_sha256':v._digest(metadata),'global_earliest_csi_time_s':10.,
        'global_latest_csi_time_s':13.,'video_frame_zero_csi_time_s':10.,
        'earliest_source_row':1,'latest_source_row':2,'video_sha256':v._digest(video),
        'publisher_video_md5':video_md5,'video_duration_s':3.5,
        'paired_frames':{'publisher_sync_uncertainty_frames':[1,2],
            'publisher_sync_uncertainty_seconds':[.02,.04],
            'not_guaranteed_subframe_simultaneity':True,
            'ffmpeg_stderr_sha256':v._digest(log),'actual_frame_samples':rows}}
    path=root/'video_alignment.json';path.write_text(json.dumps(proof))
    return path,proof


def test_missing_full_alignment_proof_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='prefix origin'):
        v.load_verified_alignment(tmp_path/'video_alignment.json')


def test_global_origin_and_actual_extracted_pts_pairing(alignment_fixture):
    path,_=alignment_fixture
    alignment=v.load_verified_alignment(path)
    assert alignment.global_earliest_csi_time_s == 10.
    pairs=v.pair_csi_rows(alignment,[0,3],csi_times_s=[12.,11.])
    assert [p.actual_video_pts_s for p in pairs] == [2.,1.]
    assert all(p.publisher_uncertainty_seconds == (.02,.04) for p in pairs)
    assert all(p.conservative_alignment_error_seconds == .04 for p in pairs)
    assert not hasattr(alignment,'pose_m')


@pytest.mark.parametrize('field,value',[
    ('full_recording_checksum_verified',False),
    ('all_record_length_and_payload_crc32c_pass',False),
    ('received_md5','0'*32),('received_bytes',255),('complete_records',3),
    ('global_earliest_csi_time_s',12.),('video_frame_zero_csi_time_s',12.),
])
def test_prefix_or_incomplete_source_cannot_be_claimed_aligned(alignment_fixture,field,value):
    path,proof=alignment_fixture
    proof[field]=value;path.write_text(json.dumps(proof))
    with pytest.raises(ValueError):v.load_verified_alignment(path)


def test_actual_pts_and_uncertainty_cannot_be_replaced(alignment_fixture):
    path,proof=alignment_fixture
    proof['paired_frames']['actual_frame_samples'][0]['actual_video_pts_s']=1.3
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='PTS'):v.load_verified_alignment(path)
    proof['paired_frames']['actual_frame_samples'][0]['actual_video_pts_s']=1.
    proof['paired_frames']['publisher_sync_uncertainty_seconds']=[0.,0.]
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='uncertainty'):v.load_verified_alignment(path)


def test_no_unextracted_frames_or_wrong_csi_time_assignment(alignment_fixture):
    alignment=v.load_verified_alignment(alignment_fixture[0])
    with pytest.raises(ValueError,match='no verified extracted'):
        v.pair_csi_rows(alignment,[1])
    with pytest.raises(ValueError,match='Caller timestamps'):
        v.pair_csi_rows(alignment,[0],csi_times_s=[0.])
    with pytest.raises(ValueError,match='integral'):
        v.pair_csi_rows(alignment,[0.])


def test_images_are_read_only_and_bound_to_proven_source(alignment_fixture):
    alignment=v.load_verified_alignment(alignment_fixture[0])
    pair=v.pair_csi_rows(alignment)[0];before=pair.image_path.read_bytes()
    image=v.open_frame(pair)
    assert image.mode == 'RGB' and image.size == (5,3)
    assert pair.image_path.read_bytes() == before
    pair.image_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frame bytes'):v.open_frame(pair)


def test_feature_execution_requires_source_publication_before_any_images(tmp_path,monkeypatch):
    from wirelessjev.radiojepa import common
    def denied():raise RuntimeError('fixture source not published')
    monkeypatch.setattr(common,'check_sources',denied)
    with pytest.raises(RuntimeError,match='not published'):
        v.extract_frozen_features(tmp_path/'missing-proof.json',tmp_path/'features')
    assert not (tmp_path/'features').exists()


def test_vision_output_cannot_mutate_main_trial_closure(tmp_path,monkeypatch):
    from wirelessjev.radiojepa import common
    monkeypatch.setattr(common,'check_sources',lambda:None)
    monkeypatch.setattr(common,'OUT',tmp_path/'main-trial')
    with pytest.raises(ValueError,match='outside the main RF'):
        v.extract_frozen_features(tmp_path/'missing-proof.json',common.OUT/'vision')
