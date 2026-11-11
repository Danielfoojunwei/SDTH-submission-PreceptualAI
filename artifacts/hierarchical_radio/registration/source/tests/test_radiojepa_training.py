"""Small synthetic software integration, never evidence of model performance."""
import numpy as np
import pytest
from wirelessjev.radiojepa import common as c,training as t


def test_two_epoch_fixture_keeps_position_labels_after_frozen_checkpoint(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(c,'OUT',tmp_path/'artifacts/radiojepa')
    monkeypatch.setattr(c,'WORK',tmp_path/'work')
    monkeypatch.setattr(c,'check_sources',lambda:None)
    p={'training_recording':'fixture','heldout_recordings':[],'cells':['joint_none'],'training_seeds':[0],
       'epochs':2,'batch_size':4,'learning_rate':.001,'weight_decay':.0002,'bins':64,'width':8,'paths':4,
       'alignment_weight':.1,'prior_weight':.01,'interventions':['z_zero','tokens_zero','power_zero','all_zero','all_shuffle']}
    monkeypatch.setattr(c,'protocol',lambda:p)
    ident={'source_receipt_sha256':'fixture-source','protocol_sha256':'fixture-protocol','data_receipt_sha256':'fixture-data'}
    monkeypatch.setattr(t,'identity',lambda:ident)
    rng=np.random.default_rng(8)
    public={'power_db':rng.normal(size=(12,4,64)).astype('f4'), 'xyz':rng.normal(size=(4,3)),
            'carrier_hz':np.array(1.272e9),'bin_frequency_hz':1.272e9+np.linspace(-24e6,24e6,64)}
    split={'train':np.arange(4),'validation':np.arange(4,8),'test':np.arange(8,12),
           'temporal_pairs':np.array([[0,1],[1,2],[2,3],[3,0]])}
    monkeypatch.setattr(t,'prepared',lambda name:public)
    monkeypatch.setattr(t,'split_arrays',lambda:split)
    c.write_once(c.OUT/'data/receipt.json',{'recordings':{'fixture':{'view_a':[0,1],'view_b':[2,3]}},
                                         'files':{'fixture.public':{'path':'fixture','sha256':'fixture'}}})
    label_reads=[]
    def labels(name):
        assert (c.OUT/'checkpoints/joint_none-seed0.pt').is_file()
        label_reads.append(name)
        return np.column_stack((np.arange(12),np.arange(12)*.5,np.ones(12)))
    monkeypatch.setattr(t,'labels',labels)
    result=t.fit(('joint_none',0))
    assert result['completed'] is True and len(result['history'])==2
    assert label_reads==['fixture']
    assert result['private_pose_used_for_encoder_training'] is False
    assert result['domains']['fixture']['forward_timing']['end_monotonic_ns']>=result['domains']['fixture']['forward_timing']['begin_monotonic_ns']
    # Completed fit integrity cannot silently route a different seed/run.
    path=c.OUT/'fits/joint_none-seed0.json'
    damaged=c.read_json(path);damaged['completed']=False
    path.write_bytes(c.encoded(damaged))
    with pytest.raises(RuntimeError,match='identity'):
        t.fit(('joint_none',0))


def test_source_paths_reject_symlink_and_escape(tmp_path):
    outside=tmp_path/'outside';outside.write_text('x')
    root=tmp_path/'root';root.mkdir();(root/'link').symlink_to(outside)
    with pytest.raises(RuntimeError):c.safe_path('../outside',root)
    with pytest.raises(RuntimeError):c.safe_path('link',root)
