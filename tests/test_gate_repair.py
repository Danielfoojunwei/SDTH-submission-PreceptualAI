"""Decision-level uncertainty, trace isolation and default-kernel compatibility."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest
import torch
from wirelessjev.native.sim import simulate,policies
from wirelessjev.native.world import build_world
from wirelessjev.native.gate import RoamDecision,checked_mask
from wirelessjev.contracted.data import make_stream,examples,READINGS,KEYS
from wirelessjev.contracted.runtime import DeviceInputs,forecast
from wirelessjev.contracted.train import load_models
from wirelessjev.gate_repair.bank import forecasts,ForecastBank
from wirelessjev.gate_repair.gate import GainCalibrator,GainGate,Recorder,reactive_rescue,fit_model


def test_hook_does_not_change_default_kernel_and_trace_has_no_side_effects():
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('wirelessjev.native._prior_gate_test',root/'artifacts/contracted/evaluation_source/src/wirelessjev/native/sim.py')
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    w=build_world(250020,duration_s=4,n_robots=8);p={p.name:p for p in policies()}['native'];seen=[]
    for gate in [False,True]:
        pol=replace(p,gate=gate);baseline=old.simulate(w,pol)
        assert simulate(w,pol)==baseline
        assert simulate(w,pol,decision_trace=lambda *r:seen.append(r))==baseline
    assert seen and all(not r[0].current.flags.writeable for r in seen)
    assert simulate(w,replace(p,gate=True),roam_gate=lambda ctx:np.ones(w.n,dtype=bool))==simulate(w,replace(p,gate=False))


def test_gate_boundary_rejects_invalid_masks_and_cannot_expose_truth():
    with pytest.raises(ValueError):checked_mask(np.array([np.nan]),1)
    with pytest.raises(ValueError):checked_mask(np.ones((2,1),dtype=bool),2)
    assert set(RoamDecision.__dataclass_fields__)=={'decision','current','candidate','gain','eligible','scores'}
    c=RoamDecision.capture(0,np.array([0]),np.array([1]),np.array([5.]),np.array([True]),np.array([[1.,6.]]))
    with pytest.raises(ValueError):c.current[0]=1
    np.testing.assert_array_equal(reactive_rescue(c),[True])


def test_bank_preserves_original_frozen_ensemble_forecasts():
    torch.set_num_threads(1);w,s=make_stream(250021,n=2,decisions=2);d=examples(w,s,2,targets=False);inputs=DeviceInputs(d['ctx'],d['route'],d['prior']);models=load_models('jepa');weights=[READINGS['bulk_uploads']['weights'][k] for k in KEYS]
    m,sd=forecast(models,inputs,weights);b=forecasts(models,inputs,weights,2,2,1.)
    np.testing.assert_array_equal(m.reshape(b.mean.shape),b.mean);np.testing.assert_array_equal(sd.reshape(b.std.shape),b.std)


def test_pair_spread_cancels_shared_ap_bias():
    mean=np.ones((1,1,9,2,3))*10;std=np.ones_like(mean)*8
    member=np.array([[[[10.,20.]]],[[[30.,40.]]],[[[-5.,5.]]]])
    b=ForecastBank(mean,std,member,np.ones((1,1,2)),np.ones((1,1,2))*10,np.zeros((1,1,2)))
    c=RoamDecision.capture(0,[0],[1],[10.],[True],[[10.,20.]])
    u=b.uncertainty(c);assert u['ensemble_pair_u'][0]==0 and u['ensemble_independent_u'][0]>0
    assert b.features(c)[0,1]==0


def test_gain_gate_defers_ineligible_actions_and_rejects_overpredicted_gain():
    class Bank:
        def features(self,c):return np.zeros((3,12))
    model=GainCalibrator(np.zeros(12),np.ones(12),np.r_[3.,np.zeros(12)],np.zeros(13),{'0.9':2.})
    c=RoamDecision.capture(0,[0,0,0],[1,1,1],[4.,7.,100.],[True,True,False],np.zeros((3,2)))
    np.testing.assert_array_equal(GainGate(Bank(),model,.9)(c),[False,True,False])


def test_calibration_uses_one_sided_error_and_roundtrips():
    rng=np.random.default_rng(3);x=rng.normal(size=(300,12));y=2*x[:,0]+rng.normal(size=300);cx=rng.normal(size=(100,12));cy=2*cx[:,0]+rng.normal(size=100)
    m=fit_model(x,y,cx,cy);n=GainCalibrator.from_dict(m.to_dict())
    for a,b in zip(m.estimate(cx),n.estimate(cx)):np.testing.assert_allclose(a,b)
    assert m.quantiles['0.5']<=m.quantiles['0.75']<=m.quantiles['0.9']
