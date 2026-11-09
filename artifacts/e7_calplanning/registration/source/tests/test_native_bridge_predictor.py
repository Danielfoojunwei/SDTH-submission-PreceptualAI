"""Public RF feature/unit proofs; no new warehouse episode or fit is created."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wirelessjev.native_bridge import predictor as p
from wirelessjev.spectral_scene.operators import segment_box_features,link_geometry_features
from wirelessjev.adaptive_radio.model import AdaptiveScene
from wirelessjev.fleetwm.data import Observations
from wirelessjev.twin.warehouse import AP_SITES


def geometry():
    return SimpleNamespace(sites=np.asarray(AP_SITES),tx=np.array((21.,23.,23.)),
                           noise=np.array((-94.,-88.,-85.)),yard=False)


def observations():
    rng=np.random.default_rng(19);t,n,s=60,4,len(AP_SITES)
    positions=rng.uniform((1.,1.),(119.,79.),(t,n,2)).astype('f4')
    scanned=np.ones((t,n),bool);heard=rng.random((t,n,s*3))<.4
    value=np.where(heard,rng.uniform(-88.,-35.,heard.shape),0.).astype('f4')
    bound=np.full_like(value,-80.)
    return Observations(positions,value,heard,bound,scanned)


def test_vectorized_actual33_racks_match_slow_union_operator():
    rng=np.random.default_rng(177)
    rx=rng.uniform((0,0,0),(120,80,6),(128,3));ap=rng.uniform((0,0,0),(120,80,6),(128,3))
    actual=p.vectorized_box_features(rx,ap,p.rack_boxes())
    expected=segment_box_features(rx,ap,p.rack_boxes())
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_allclose(actual[1],expected[1],rtol=0,atol=2e-15)
    assert p.rack_boxes().shape==(33,2,3)


def test_fast_box_operator_rejects_shared_faces_instead_of_doublecounting():
    boxes=np.array((((0,0,0),(1,1,1)),((1,0,0),(2,1,1))))
    with pytest.raises(ValueError,match='touching'):
        p.vectorized_box_features(np.array(((1.,2.,.5),)),np.array(((1.,0.,.5),)),boxes)


def test_public_features_equal_shared_sixteen_column_operator():
    rng=np.random.default_rng(11);rx=rng.uniform((0,0,1),(120,80,1),(32,3))
    ap=np.tile((15.,12.,4.),(32,1));carrier=np.resize(p.NATIVE_CARRIERS_HZ,32)
    actual=p.public_features(rx,ap,carrier,-88.,23.)
    expected=link_geometry_features(rx,ap,carrier,0,(120.,80.,6.),p.rack_boxes(),
        calibration_pair_count=0,calibration_unknown=False,quantization_width_db=0.,noise_floor_dbm=-88.,tx_power_dbm=23.)
    np.testing.assert_array_equal(actual,expected)
    assert np.all(actual[:,13]==0)  # continuous scans, no invented quantization


def test_full_native100_query_geometry_is_vectorized_and_actual_carriers():
    plan=np.broadcast_to(np.array((3.,4.)),(100,9,2)).copy()
    q=p.query_operands(geometry(),plan)
    assert q['shape']==(100,9,16,3) and q['query'].shape==(43200,16)
    assert q['basis'].shape==(43200,8) and np.isfinite(q['query']).all()
    np.testing.assert_allclose(q['query'][:,7].reshape(q['shape'])[0,0,0],np.log(np.array(p.NATIVE_CARRIERS_HZ)/2.4e9))
    assert q['query'][2,7]!=pytest.approx(np.log(6./2.4))


def test_context_balanced_latest_actual_and_no_six_point_five_input():
    past=observations();g=geometry();context=p.select_context(past,g)
    index=context['selected_rows'];t,r,s,b=index.T
    assert len(index)==64 and set(b)=={0,1} and set(s)==set(range(16))
    assert (t==59).all() and np.bincount(s*2+b).tolist()==[2]*32
    np.testing.assert_array_equal(context['context'][:,2].astype(bool),past.heard[t,r,s*3+b])
    np.testing.assert_array_equal(context['context'][:,3].astype(bool),~past.heard[t,r,s*3+b])
    corrupted=past.value.copy();corrupted[:,:,2::3]=100000
    changed=Observations(past.positions,corrupted,past.heard,past.bound,past.scanned)
    np.testing.assert_array_equal(context['context'],p.select_context(changed,g)['context'])
    np.testing.assert_array_equal(context['context'],p.select_context(past,g)['context'])


def test_unknown_and_nonheard_center_never_become_readings():
    past=observations();g=geometry()
    value=past.value.copy();value[~past.heard]=1e9
    corrupt=Observations(past.positions,value,past.heard,past.bound,past.scanned)
    np.testing.assert_array_equal(p.select_context(corrupt,g)['context'],p.select_context(past,g)['context'])
    missing=Observations(past.positions,np.zeros_like(value),np.zeros_like(past.heard),past.bound,np.zeros_like(past.scanned))
    with pytest.raises(ValueError,match='No delivered scan'):p.select_context(missing,g)


def test_untrained_actual_model_software_boundary_and_causal_input_archive():
    torch.manual_seed(0);torch.set_num_threads(1)
    model=AdaptiveScene('joint',width=16);g=geometry();raw=p.AdaptiveNativePredictor(g,model,'a'*64)
    bridge=p.bridged(raw);past=observations();plan=np.full((2,9,2),5.)
    mean,std,kind,elapsed=bridge.step(past,plan)
    assert mean.shape==std.shape==(2,9,16,3) and kind=='cold' and elapsed>=0
    assert np.isfinite(mean).all() and (std>0).all()
    assert raw.records[0]['history_length']==60 and raw.last_history['value'].shape==past.value.shape
    np.testing.assert_array_equal(raw.records[0]['selected_rows'],p.select_context(past,g)['selected_rows'])
    assert bridge.adapter.packet_calibration is None and bridge.adapter.rssi_bound is None
    # Software execution is untrained and supplies no accuracy evidence.
    with pytest.raises(ValueError,match='yard'):
        p.AdaptiveNativePredictor(SimpleNamespace(**(g.__dict__|{'yard':True})),model,'a'*64)


def test_native_map_unit_roundtrip_recovers_float32_bits():
    rng=np.random.default_rng(32);g=geometry();mean=rng.uniform(-60,60,(5,9,16,3)).astype('f4')
    rssi=mean.astype('f8')+g.tx-15.+g.noise
    restored=p.radio_assumptions(g).convert(rssi).astype('f4')
    np.testing.assert_array_equal(restored,mean)
