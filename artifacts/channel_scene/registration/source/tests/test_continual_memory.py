"""Numerical identifiability/retention proofs, not a wireless benchmark."""
import numpy as np
import pytest
from wirelessjev.continual_memory import ContinualResponseMemory, DifferentialPair

KEY=('robot-1','warehouse-A/channel-36',0)


def pair(identity,x,y,t=1.,controlled=True,provenance='declared staggered-power protocol'):
    return DifferentialPair(identity,x,y,t,t,controlled,provenance)


def query(memory,x,now=1.,key=KEY,**kwargs):
    return memory.query(key,x,now=now,query_observed_at=now,**kwargs)


def test_rank_one_evidence_identifies_joint_query_but_not_individual_couplings():
    memory=ContinualResponseMemory(2)
    assert memory.ingest(KEY,pair('one',[1.,1.],5.),now=1.)
    joint=query(memory,[2.,2.]); individual=query(memory,[1.,0.])
    assert joint.identified and joint.rank==1 and joint.predicted_delta_mw==pytest.approx(10.)
    assert not individual.identified and individual.predicted_delta_mw is None
    assert joint.uncertainty_bound is None and not joint.production_qualified


def test_staggered_equal_exposure_budget_identifies_individual_queries():
    memory=ContinualResponseMemory(2)
    first,second=[1.,0.],[0.,1.]
    assert sum(first)==sum(second)==1.  # caller-declared equal transmit exposure budget
    # With unit transmit power/overlap and unit-duration slots, both protocols
    # use two transmitter-airtime units in total: one joint slot or two staggered.
    assert sum([1.,1.])==sum(first)+sum(second)==2.
    memory.ingest(KEY,pair('tx-A',first,2.),now=1.)
    memory.ingest(KEY,pair('tx-B',second,3.),now=1.)
    assert query(memory,[1.,0.]).predicted_delta_mw==pytest.approx(2.)
    assert query(memory,[0.,1.]).predicted_delta_mw==pytest.approx(3.)
    joint=query(memory,[2.,2.])
    assert joint.rank==2 and joint.predicted_delta_mw==pytest.approx(10.)


def test_repeated_synchronized_data_does_not_create_identifying_excitation():
    memory=ContinualResponseMemory(2)
    for index in range(20):
        memory.ingest(KEY,pair(f'sync-{index}',[1.,1.],5.),now=1.)
    result=query(memory,[1.,0.])
    assert result.observations==20 and result.rank==1 and not result.identified
    assert query(memory,[-2.,-2.]).predicted_delta_mw==pytest.approx(-10.)
    near=ContinualResponseMemory(2)
    near.ingest(KEY,pair('near-a',[1.,1.],5.),now=1.)
    near.ingest(KEY,pair('near-b',[1.,1.+1e-8],5.+3e-8),now=1.)
    assert not query(near,[1.,0.]).identified  # Gram conditioning cannot invent reliable rank


def test_uncontrolled_pairs_remain_associative_and_control_requires_provenance():
    memory=ContinualResponseMemory(2)
    observational=DifferentialPair('obs',[1.,1.],5.,1.,1.)
    assert memory.ingest(KEY,observational,now=1.)
    result=query(memory,[2.,2.])
    assert result.evidence_kind=='associative_only' and result.predicted_delta_mw==pytest.approx(10.)
    assert not query(memory,[2.,2.],controlled_only=True).identified
    with pytest.raises(ValueError,match='provenance'):
        memory.ingest(KEY,pair('undeclared',[1.,0.],2.,provenance=''),now=1.)
    memory.ingest(KEY,pair('controlled',[1.,0.],2.),now=1.)
    result=query(memory,[1.,0.],controlled_only=True)
    assert result.evidence_kind=='controlled_declared' and result.predicted_delta_mw==pytest.approx(2.)
    assert result.provenance==('declared staggered-power protocol',) and not result.production_qualified


def test_each_old_row_expires_even_when_new_joint_evidence_refreshes_the_state():
    memory=ContinualResponseMemory(2,max_age_seconds=5.,max_delivery_seconds=2.)
    memory.ingest(KEY,pair('old-independent',[1.,0.],2.,t=1.),now=1.)
    memory.ingest(KEY,pair('new-joint',[1.,1.],5.,t=5.),now=5.)
    assert query(memory,[1.,0.],now=5.).identified
    assert not query(memory,[1.,0.],now=7.).identified
    assert query(memory,[2.,2.],now=7.).predicted_delta_mw==pytest.approx(10.)
    stale=memory.query(KEY,[2.,2.],now=7.,query_observed_at=1.)
    assert not stale.identified and stale.reason=='stale_query'
    assert not memory.query(KEY,[2.,2.],now=7.).identified
    assert not query(memory,[2.,2.],now=11.).identified and memory.size=={'states':0,'pairs':0}
    with pytest.raises(ValueError,match='backwards'):query(memory,[1.,0.],now=10.)
    with pytest.raises(ValueError,match='Noncausal'):
        memory.ingest(KEY,DifferentialPair('future',[1.,0.],2.,12.,13.),now=11.)
    assert not memory.ingest(KEY,DifferentialPair('late',[1.,0.],2.,8.,11.),now=11.)


def test_pair_dedup_lru_state_and_pair_limits_remain_bounded():
    memory=ContinualResponseMemory(2,max_states=2,max_pairs_per_state=2)
    for index in range(3):memory.ingest(KEY,pair(f'p{index}',[1.,0.],2.),now=1.)
    assert memory.size=={'states':1,'pairs':2}
    assert not memory.ingest(KEY,pair('p2',[1.,0.],2.),now=1.)
    other=('robot-2',KEY[1],0); third=('robot-3',KEY[1],0)
    memory.ingest(other,pair('p',[1.,0.],3.),now=1.)
    query(memory,[1.,0.])  # touch first state, making the other LRU
    memory.ingest(third,pair('p',[1.,0.],4.),now=1.)
    assert not query(memory,[1.,0.],key=other).identified
    assert memory.size=={'states':2,'pairs':3}


def test_si_scale_checks_do_not_turn_tiny_unidentified_queries_into_zero():
    for scale in (1.,1e-12):
        memory=ContinualResponseMemory(2,exposure_scale_mw=scale,response_scale_mw=1e-9)
        memory.ingest(KEY,pair('tiny',[1e-12,1e-12],5e-9),now=1.)
        assert query(memory,[2e-12,2e-12]).predicted_delta_mw==pytest.approx(10e-9)
        assert not query(memory,[1e-12,0.]).identified
    for values in ([np.nan,0.],[np.inf,0.],[1e20,0.],[1e-200,0.]):
        with pytest.raises(ValueError):memory.ingest(KEY,pair('bad',values,2.),now=1.)
    extreme=ContinualResponseMemory(2,response_scale_mw=1e308)
    extreme.ingest(KEY,pair('finite-response',[1.,0.],1e308),now=1.)
    result=query(extreme,[2.,0.])
    assert result.identified and result.predicted_delta_mw is None and result.reason=='numerical_overflow'
    with pytest.raises(ValueError,match='underflow'):
        extreme.ingest(KEY,pair('lost-response',[1.,0.],1e-200),now=1.)
    lost_exposure=ContinualResponseMemory(2,exposure_scale_mw=1e308)
    with pytest.raises(ValueError,match='underflow'):
        lost_exposure.ingest(KEY,pair('lost-exposure',[1e-200,0.],1.),now=1.)


def test_context_epoch_device_separation_missing_data_and_configuration_validation():
    memory=ContinualResponseMemory(2)
    keys=[KEY,(KEY[0],KEY[1],1),(KEY[0],'warehouse-B',0),('robot-2',KEY[1],0)]
    for index,key in enumerate(keys):memory.ingest(key,pair('same-local-id',[1.,0.],index+1.),now=1.)
    for index,key in enumerate(keys):assert query(memory,[1.,0.],key=key).predicted_delta_mw==pytest.approx(index+1.)
    assert not memory.ingest(KEY,pair('missing',None,2.),now=1.)
    assert not memory.ingest(KEY,pair('missing-y',[1.,0.],None),now=1.)
    assert not query(memory,None).identified
    for kwargs in (dict(max_states=0),dict(max_pairs_per_state=True),dict(exposure_scale_mw=0.),
                   dict(max_age_seconds=np.inf),dict(rcond=1.),dict(max_scaled_value=1e308)):
        with pytest.raises(ValueError):ContinualResponseMemory(2,**kwargs)
