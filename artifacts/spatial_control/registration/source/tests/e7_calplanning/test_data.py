"""Analytic software fixtures only: no registered environment is generated."""
from types import SimpleNamespace

import numpy as np
import pytest

from wirelessjev.e7_calplanning import common as c,data
from wirelessjev.envlearn import Survey,belief_evaluator
from wirelessjev.fastplan import FastPlanner
from wirelessjev.missions import WAREHOUSE_MISSIONS
from wirelessjev.oracle import _TABLES


def fixture():
    aps=[SimpleNamespace(x=4.,y=3.,band='5g',tx_power_dbm=20.,channel=36),
         SimpleNamespace(x=20.,y=9.,band='2g',tx_power_dbm=17.,channel=1)]
    public=data.public_fit_inputs(aps,3.)
    x=np.tile(np.linspace(1.,25.,28),2);y=np.tile(np.linspace(4.,11.,28),2)
    ap=np.repeat([0,1],28)
    rssi=np.asarray([aps[j].tx_power_dbm-40.-28.*np.log10(np.sqrt((aps[j].x-xx)**2+(aps[j].y-yy)**2+9.))
                     for j,xx,yy in zip(ap,x,y)])
    sv=Survey(x.tolist(),y.tolist(),ap.tolist(),rssi.tolist(),[24.,24.],[12.,12.],[0,1],[-75.,-78.],
              [(1.,4.,0,10.),(20.,8.,1,8.)],{('5g',36):[.1,.2],('2g',1):[.2,.3]},censor_weight=3.)
    shared={'n':np.asarray(2),'m':np.asarray(2),'nd':np.asarray(2),'nkeys':np.asarray(2),
        'dom_of_ap':np.arange(2),'noise_mw':10**(np.array([-94.,-96.])/10),'is5':np.array([True,False]),
        'bc_key':np.arange(2),'ext':np.array([False,False]),'crit':np.array([1.,1.]),'be':np.array([2.,2.]),
        'demand':np.array([3.,3.]),'cf_dem':np.ones((2,1)),'cf_floor':np.full((2,1),.5),
        'current':np.array([0,1]),'t5':_TABLES['5g'][0],'r5':_TABLES['5g'][1],
        't2':_TABLES['2g'][0],'r2':_TABLES['2g'][1],'interferer':np.zeros((2,2),bool),'leak':np.zeros((2,2)),
        'car_ids':np.array(['toy0','toy1']),'ap_ids':np.array(['a','b']),'visible':np.ones((2,2),bool),
        'client_tx_dbm':np.asarray(15.),'band_names':np.array(['5g','2g']),'channels':np.array([36,1])}
    positions=np.array([[[3.,4.],[19.,8.]]])
    return aps,public,sv,shared,positions


def test_public_fit_inputs_copy_only_whitelisted_attributes():
    class AP:
        x=1.;y=2.;band='5g';tx_power_dbm=20.
        def __getattr__(self,name):raise AssertionError('private attribute read: '+name)
    public=data.public_fit_inputs([AP()],3.)
    assert set(public.__dict__)=={'aps','ap_height'}
    assert set(public.aps[0].__dict__)=={'x','y','band','tx_power_dbm'}
    with pytest.raises(ValueError):data.public_fit_inputs([AP()],float('nan'))
    with pytest.raises(ValueError):data.fit_map('invented',public,Survey(),0)


@pytest.mark.parametrize('kind',c.KINDS)
def test_actual_fixed_map_fits_and_mean_state_replays(kind):
    _,public,sv,_,positions=fixture()
    model=data.fit_map(kind,public,sv,seed=912)
    x,y=positions[0].T
    expected,spread=model.predict(x,y)
    np.testing.assert_allclose(data.replay_mean(data.mean_state(model,kind),public,x,y),expected,rtol=1e-12,atol=1e-12)
    assert np.isfinite(spread).all()
    assert not {'Ainv','covariance','calibrated_spread'} & set(data.mean_state(model,kind))


def test_tail_subset_and_archive_preserve_all_report_channels():
    n=541
    sv=Survey(list(range(n)),list(range(n)),[0]*n,list(range(n)),list(range(61)),list(range(61)),[0]*61,list(range(61)),
              [(i,0,0,i) for i in range(31)],{('5g',36):list(range(31)),('2g',1):list(range(17))},censor_weight=3.)
    for b in c.BUDGETS:
        sub=data.tail_subset(sv,b);restored=data.survey_from_arrays(data.survey_arrays(sub))
        assert restored==sub
        assert sub.rssi==sv.rssi[-b*12*(n//180):]
        assert sub.cbound==sv.cbound[-int(61*b/15):]
        assert sub.link==sv.link[-int(31*b/15):]
        assert sub.busy[('2g',1)]==sv.busy[('2g',1)][-max(1,int(17*b/15)):]
    # Deliberately preserve the legacy zero-count slicing edge, not a new rule.
    small=Survey([1.],[2.],[0],[-70.],link=[(1.,2.,0,3.)])
    assert len(data.tail_subset(small,1).rssi)==1
    assert len(data.tail_subset(small,1).link)==1


def test_observed_map_belief_matches_frozen_e7_helper_without_private_map_inputs():
    aps,public,sv,shared,positions=fixture()
    truth=data.evaluator(shared,[(np.full((2,2),-65.),np.full((2,2),-60.))],external_noise=np.array([9.,8.]),availability=np.array([.2,.3]))
    scene=SimpleNamespace(aps=aps,cars=[SimpleNamespace(x=x,y=y,tx_power_dbm=15.) for x,y in positions[0]],mobility=None)
    old=belief_evaluator(truth,scene,SimpleNamespace(ap_height=3.,seed=912),sv,'learned',kappa=0.)
    model=data.fit_map('corrected_v2',public,sv,912)
    new=data.make_belief(shared,positions,public,sv,lambda x,y:model.predict(x,y)[0])
    for key in ('ext_noise_mw','avail','leak'):
        np.testing.assert_allclose(getattr(new,key),getattr(old,key),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(new.snaps[0].ul,old.snaps[0].ul,rtol=1e-12,atol=1e-12)
    actions=np.array([[0,0],[0,1],[1,0],[1,1]])
    for mission in WAREHOUSE_MISSIONS.values():
        np.testing.assert_allclose(new.utility(actions,mission.weights),old.utility(actions,mission.weights),rtol=1e-12,atol=1e-12)
    # No Scene object or true external powers remain on the constructed view.
    assert not hasattr(new,'scene')
    assert not np.array_equal(new.ext_noise_mw,truth.ext_noise_mw)


def test_same_frozen_analytical_planner_matches_toy_exhaustive_objective():
    _,public,sv,shared,positions=fixture()
    means=np.array([[-55.,-86.],[-86.,-55.]])
    belief=data.make_belief(shared,positions,public,sv,lambda x,y:means[:len(x)])
    mission=WAREHOUSE_MISSIONS['teleop_assist'];actions=np.array([[0,0],[0,1],[1,0],[1,1]])
    plan,value,stats=FastPlanner(None,evaluator=belief,device=None).search(mission.weights,starts=actions,max_iter=40,screen_iters=1,multi=True)
    assert np.isclose(value,np.max(belief.utility(actions,mission.weights)))
    assert 0<=stats['iterations']<=40
    assert np.isclose(value,belief.utility(plan,mission.weights)[0])
