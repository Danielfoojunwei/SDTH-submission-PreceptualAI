"""Analytic/software fixtures; no hardware or learned-model result claims."""
from types import SimpleNamespace
import numpy as np
import pytest

from wirelessjev.closed_loop.control import OutageUtility
from wirelessjev.native.world import HORIZONS
from wirelessjev.worldmodel.interface import CandidateAction, OutcomeDistribution, SceneState
from wirelessjev.worldmodel.radiomap_adapter import RadioMapAdapter


def state(n=5, sites=4):
    rng = np.random.default_rng(717)
    return SceneState(np.array(HORIZONS), np.arange(n) % sites,
                      rng.normal(11, 15, (n, 9, sites, 3)).astype('f4'), 'fixture-prediction')


@pytest.mark.parametrize('horizon,cost', [(5., .03), (30., 4.), (8., 1.73)])
def test_adapter_exactly_reproduces_existing_outage_utility(horizon, cost):
    s = state(100, 16)
    actions = tuple(CandidateAction(f'ap-{i}', i, handover_seconds=cost) for i in range(16))
    result = RadioMapAdapter().query(s, actions, horizon, 'mid')
    old = OutageUtility(None, None, cost, horizon)
    expected = old.scores(s.snr_db, s.current_ap)
    np.testing.assert_array_equal(-result.outage_integral_seconds, expected)
    assert result.metadata['calibration'] == 'uncalibrated'
    assert result.outage_upper is result.silence_upper_seconds is None


def test_same_ap_has_no_association_outage_but_roam_has_explicit_blackout():
    s = SceneState(np.array(HORIZONS), np.array([0]), np.full((1,9,2,3), 40.), 'fixture')
    r = RadioMapAdapter().query(s, (CandidateAction('stay',0),CandidateAction('roam',1)),30,'mid')
    np.testing.assert_array_equal(r.switch_seconds,[[0.,4.]])
    assert np.all(r.outage_probability[0,1,:4] == 1.)
    assert np.all(r.rate_mbps[0,1,:4] == 0.)
    assert r.outage_integral_seconds[0,1] >= 4.
    assert r.outage_integral_seconds[0,0] < 1.


def test_explicit_mcs_zero_matches_native_band_risk_and_invalid_mcs_rejects():
    s = state(2,2)
    adapter = RadioMapAdapter()
    a = adapter.query(s,(CandidateAction('fixed',0,band='5g'),),30,'fast')
    b = adapter.query(s,(CandidateAction('mcs0',0,band='5g',mcs=0),),30,'fast')
    np.testing.assert_array_equal(a.outage_probability,b.outage_probability)
    with pytest.raises(ValueError,match='MCS'):
        adapter.query(s,(CandidateAction('bad',0,band='5g',mcs=100),),30,'fast')


def test_model_receipt_reference_cannot_self_qualify_legacy_adapter():
    s = state(1,1)
    altered = SceneState(s.times_s,s.current_ap,s.snr_db,s.source_id,
                        np.zeros_like(s.snr_db), {'outage':'a'*64})
    adapter = RadioMapAdapter()
    action = (CandidateAction('stay',0),)
    a,b = adapter.query(s,action,30,'mid'),adapter.query(altered,action,30,'mid')
    np.testing.assert_array_equal(a.outage_probability,b.outage_probability)
    assert b.metadata['calibration'] == 'uncalibrated'
    assert b.outage_upper is None


def test_boundary_arrays_are_owned_and_cannot_be_made_writeable():
    raw = np.zeros((1,9,1,3))
    s = SceneState(np.array(HORIZONS),np.array([0]),raw,'fixture')
    raw[:] = 100
    assert not s.snr_db.any()
    with pytest.raises(ValueError):
        s.snr_db.setflags(write=True)
    with pytest.raises(TypeError):
        s.qualified_head['outage'] = 'a'*64


@pytest.mark.parametrize('change', ['float_ap','bad_times','nan_snr','invalid_probability','receipt'])
def test_invalid_scene_cannot_enter_planner(change):
    s = state(1,1)
    values = dict(times_s=s.times_s,current_ap=s.current_ap,snr_db=s.snr_db,source_id=s.source_id)
    if change == 'float_ap': values['current_ap'] = np.array([0.])
    if change == 'bad_times': values['times_s'] = np.array(HORIZONS)[::-1]
    if change == 'nan_snr': values['snr_db'] = np.full_like(s.snr_db,np.nan)
    if change == 'invalid_probability': values['learned_outage_probability'] = np.full_like(s.snr_db,1.1)
    if change == 'receipt': values['qualified_head'] = {'outage':'not-a-receipt'}
    with pytest.raises(ValueError): SceneState(**values)


def test_unimplemented_actions_and_private_world_objects_are_rejected():
    with pytest.raises(ValueError,match='MLO'):
        CandidateAction('mlo',0,mechanism='mlo_link')
    with pytest.raises(ValueError,match='band'):
        CandidateAction('mcs',0,mcs=1)
    with pytest.raises(TypeError):
        CandidateAction('power',0,power_dbm=15)
    with pytest.raises(TypeError,match='predicted SceneState'):
        RadioMapAdapter().query(SimpleNamespace(snr=np.zeros((1,1,1,3))),
                                (CandidateAction('a',0),),30,'mid')
