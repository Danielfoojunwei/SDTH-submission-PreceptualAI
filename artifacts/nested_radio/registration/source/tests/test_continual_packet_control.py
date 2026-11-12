"""Numerical public-report fixtures; no scientific training or hardware data."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from wirelessjev.continual_packet.control import ResponsePredictor
from wirelessjev.continual_packet.probes import ProbeMailbox, ProbeReport, PublicProbeSpec
from wirelessjev.fleetwm.data import Geometry
from wirelessjev.twin.radio import AP


def setup(estimator='memory', design='staggered', *, domains=(0, 1), channels=(36, 52), powers=(1., 1.)):
    sites = np.array([[15., 12.], [45., 12.]])
    aps = (AP('first', 15., 12., channels[0]), AP('other', 45., 12., channels[1]))
    geometry = Geometry(sites, np.array([20., 20., 20.]), np.array([-90., -88., -85.]), aps)
    spec = PublicProbeSpec([[20., 20.], [30., 20.]], powers, 36,
                           np.zeros((100, 2)), (0,), design=design, pulse_airtime_fraction=1.)
    box = ProbeMailbox()
    predictor = ResponsePredictor(geometry, spec, box, estimator, receiver_domains=np.asarray(domains))
    exposure = ((0., 0.), (powers[0], 0.), (0., powers[1])) if design == 'staggered' else ((0., 0.), tuple(powers), (0., 0.))
    signal = [[1e-7 + 2e-6 * a + 3e-6 * b] for a, b in exposure]
    report = ProbeReport('fixture-0', 36, (0, 1, 2), 3, 5, exposure,
                         tuple(tuple(row) for row in signal), ((True,),) * 3,
                         (1e-8,), (0,), 128)
    return predictor, box, report


def deliver(box, report):
    box._advance(report.delivered_slot)
    box._publish(report)


def test_public_staggered_reports_identify_both_effects_without_hidden_kernel():
    p, box, report = setup()
    deliver(box, report)
    assert p.consume(5) == (1, 2)
    for q, expected in [([1., 0.], 2e-6), ([0., 1.], 3e-6), ([1., 1.], 5e-6)]:
        result = p.memories[0].query(('ap-0', 'managed-5g', 0), q,
                                    now=.05, query_observed_at=.05, controlled_only=True)
        assert result.identified and result.predicted_delta_mw == pytest.approx(expected)
    assert not hasattr(p, 'world') and not hasattr(p, 'runtime')


def test_synchronized_reports_identify_joint_effect_but_not_individual_source():
    p, box, report = setup(design='synchronized')
    deliver(box, report)
    p.consume(5)
    key = ('ap-0', 'managed-5g', 0)
    joint = p.memories[0].query(key, [1., 1.], now=.05, query_observed_at=.05)
    single = p.memories[0].query(key, [1., 0.], now=.05, query_observed_at=.05)
    assert joint.identified and joint.predicted_delta_mw == pytest.approx(5e-6)
    assert not single.identified and single.predicted_delta_mw is None


def test_missing_baseline_never_becomes_a_zero_measurement():
    p, box, report = setup()
    report = replace(report, observed=((False,), (True,), (True,)),
                     received_mw=((None,), report.received_mw[1], report.received_mw[2]))
    deliver(box, report)
    assert p.consume(5) == (1, 0)
    assert p.memories[0].size['pairs'] == 0


def test_bayesian_update_uses_correlated_measured_differences_not_the_prior_as_truth():
    p, box, report = setup('bayesian')
    deliver(box, report)
    p.consume(5)
    posterior = p._bayesian(0, np.array([.5e-6, .5e-6]), np.array([4., 4.]), .05)
    np.testing.assert_allclose(posterior, [2e-6, 3e-6], atol=1e-8, rtol=0)


def test_unavailable_or_wrong_schema_reports_fail_before_assimilation():
    p, box, report = setup()
    with pytest.raises(ValueError):
        p.consume(5)  # A future mailbox cannot be queried before native advance.
    with pytest.raises(ValueError):
        deliver(box, replace(report, transmitter_ids=('other', 'schema')))
        p.consume(5)


def test_old_evidence_expires_and_ignored_receipts_do_not_fit_a_memory():
    p, box, report = setup()
    deliver(box, report)
    p.consume(5)
    result = p.memories[0].query(('ap-0', 'managed-5g', 0), [1., 0.],
                                now=31., query_observed_at=31.)
    assert not result.identified and result.predicted_delta_mw is None
    ignored, box, report = setup('ignore')
    deliver(box, report)
    assert ignored.consume(5) == (1, 0)
    assert all(memory.size['pairs'] == 0 for memory in ignored.memories)


def fixture_step(monkeypatch, predictor, mailbox, gain, activity):
    """A numeric public map fixture isolates prediction math from simulation."""
    from wirelessjev.continual_packet import control
    plan=np.zeros((1,9,2));shape=(1,9,len(predictor.geometry.sites),3)
    mean=np.full(shape,30.,dtype='f4');sd=np.full(shape,2.,dtype='f4')
    class FixedBase:
        rmap=object()
        def step(self,past,route):return mean.copy(),sd.copy(),'warm',0.
    class PublicMap:
        def __init__(self,rmap):pass
        def predict(self,x,y):
            rssi=np.zeros((2,len(predictor.geometry.sites),3),dtype=float)
            rssi[:,:,1]=predictor.geometry.tx[1]+10*np.log10(np.asarray(gain).T)
            return rssi,np.full_like(rssi,4.)
    predictor.base=FixedBase()
    monkeypatch.setattr(control,'FastMap',PublicMap)
    activities=np.zeros((3201,2));activities[3:]=activity
    predictor.spec=replace(predictor.spec,activity_plan=activities)
    mailbox._advance(100)
    result=predictor.step(SimpleNamespace(value=np.zeros((61,1,6))),plan)
    return result,mean


def assert_rf_prediction(output,baseline,geometry,expected_rf):
    penalty=10*np.log10(1+np.asarray(expected_rf)/10**(geometry.noise[1]/10))
    np.testing.assert_allclose(output[:,:, :,1],baseline[:,:,:,1]-penalty[None,None,:],atol=2e-6,rtol=0)
    np.testing.assert_array_equal(output[:,:,:,[0,2]],baseline[:,:,:,[0,2]])


def test_step_sensed_sources_only_defer_airtime_and_do_not_become_rf_penalties(monkeypatch):
    p,box,_=setup('prior')
    result,mean=fixture_step(monkeypatch,p,box,[[1e-6,2e-6],[1e-12,2e-12]],[.1,.2])
    np.testing.assert_array_equal(result[0],mean)
    assert result[2]=='warm' and result[3]>=0


def test_step_hidden_sources_add_only_their_public_predicted_overlap_power(monkeypatch):
    p,box,_=setup('prior')
    result,mean=fixture_step(monkeypatch,p,box,[[1e-12,2e-12],[1e-6,1e-6]],[.1,.2])
    assert_rf_prediction(result[0],mean,p.geometry,[.1e-12+.4e-12,0.])


@pytest.mark.parametrize('hidden_duty,expected',[(.05,0.),(.2,1e-12*(.2-.1)/.9)])
def test_step_mixed_sources_condition_rf_on_time_outside_sensed_window(monkeypatch,hidden_duty,expected):
    p,box,_=setup('prior')
    result,mean=fixture_step(monkeypatch,p,box,[[1e-6,1e-12],[1e-6,1e-6]],[.1,hidden_duty])
    assert_rf_prediction(result[0],mean,p.geometry,[expected,0.])


@pytest.mark.parametrize('domains,second_rf',[
    ((0,0),2e-12*(.2-.1)/.9),
    ((0,1),1e-12*.1+2e-12*.2),
])
def test_step_uses_public_domain_wide_sensing_from_estimated_gains(monkeypatch,domains,second_rf):
    p,box,_=setup('prior',domains=domains,channels=(36,36))
    result,mean=fixture_step(monkeypatch,p,box,[[1e-6,1e-12],[1e-12,2e-12]],[.1,.2])
    assert_rf_prediction(result[0],mean,p.geometry,[1e-12*(.2-.1)/.9,second_rf])


def test_step_memory_synchronized_joint_sum_cannot_replace_unknown_individual_gains(monkeypatch):
    p,box,report=setup(design='synchronized');deliver(box,report)
    queries=[];query=p.memories[0].query
    def track(key,exposure,**kwargs):
        queries.append(np.array(exposure,copy=True));return query(key,exposure,**kwargs)
    monkeypatch.setattr(p.memories[0],'query',track)
    result,mean=fixture_step(monkeypatch,p,box,[[1e-12,2e-12],[1e-6,1e-6]],[.2,.2])
    np.testing.assert_array_equal(queries,[[1.,0.],[0.,1.]])
    assert_rf_prediction(result[0],mean,p.geometry,[.6e-12,0.])
    assert p.records[-1]['identified_queries']==0 and p.records[-1]['fallback_queries']==2


def test_step_memory_uses_identified_basis_and_prior_for_the_unidentified_source(monkeypatch):
    p,box,report=setup()
    report=replace(report,observed=((True,),(True,),(False,)),
                   received_mw=(report.received_mw[0],report.received_mw[1],(None,)))
    deliver(box,report)
    result,mean=fixture_step(monkeypatch,p,box,[[1e-12,2e-12],[1e-6,1e-6]],[.1,.2])
    assert_rf_prediction(result[0],mean,p.geometry,[2e-12*(.2-.1)/.9,0.])
    assert p.records[-1]['identified_queries']==1 and p.records[-1]['fallback_queries']==1


def test_step_memory_normalizes_nonunit_power_basis_into_individual_gain(monkeypatch):
    p,box,report=setup(powers=(2.,4.))
    received=tuple((1e-7+1e-12*a+2e-12*b,) for a,b in report.exposure_mw)
    deliver(box,replace(report,received_mw=received,noise_std_mw=(1e-18,)))
    result,mean=fixture_step(monkeypatch,p,box,[[1e-11,1e-11],[1e-6,1e-6]],[.1,.2])
    assert_rf_prediction(result[0],mean,p.geometry,[1e-12*2*.1+2e-12*4*.2,0.])
    assert p.records[-1]['identified_queries']==2 and p.records[-1]['fallback_queries']==0


def test_step_bayesian_posterior_changes_estimated_carrier_sense_without_true_mask(monkeypatch):
    p,box,report=setup('bayesian');deliver(box,report)
    result,mean=fixture_step(monkeypatch,p,box,[[5e-9,5e-9],[1e-6,1e-6]],[.1,.2])
    np.testing.assert_array_equal(result[0],mean)  # measured posterior now senses both sources
    prior,other_box,_=setup('prior')
    prior_result,prior_mean=fixture_step(monkeypatch,prior,other_box,[[5e-9,5e-9],[1e-6,1e-6]],[.1,.2])
    assert_rf_prediction(prior_result[0],prior_mean,prior.geometry,[1.5e-9,0.])


@pytest.mark.parametrize('domains',[None,(0,),(-1,0),(False,True),(0.,1.)])
def test_missing_or_invalid_public_receiver_domains_are_not_silently_invented(domains):
    with pytest.raises(ValueError):setup(domains=domains)
