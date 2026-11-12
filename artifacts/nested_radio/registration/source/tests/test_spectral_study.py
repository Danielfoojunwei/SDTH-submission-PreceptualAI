"""Known-ID software fixtures only; no registered RF scene is traced or fitted."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from wirelessjev.spectral_scene import common as c,study,verification


def targets(values):
    values=np.asarray(values,dtype=float)
    return {'center_dbm':values,'lower_dbm':values-.5,'upper_dbm':values+.5,
            'heard':np.ones(len(values),bool),'censored':np.zeros(len(values),bool),
            'censor_upper_dbm':np.full(len(values),-90.)}


def evaluation(rmse=(1.,1.1,1.2),reference=2.,zero=1.5,p95=1.,reference_p95=2.):
    def member(name,seed,score,p95score):
        return {'member':name,'cell':'joint_none','seed':seed,'intervention':None,
            'layouts':[{'scene_id':f'fixture-{i}','metrics':{'rmse_db':score+i*.01,
                'p95_abs_db':p95score+i*.01,'finite':True,'heard_count':3,'censored_count':1,'missing_count':2}}
                for i in range(4)]}
    members={'radiomap_operator':member('radiomap_operator',None,reference,reference_p95)}
    for seed,score in enumerate(rmse):
        for suffix,value in (('',score),('-zero',zero),('-shuffle',zero+.1)):
            key=f'joint_none-seed{seed}'+suffix;members[key]=member(key,seed,value,p95)
    return {'members':{'in_band':members}}


def test_interval_metrics_do_not_invent_censored_point_labels():
    t=targets([-40.,-50.,-9999.]);t['heard'][-1]=False;t['censored'][-1]=True
    first=study.observation_metrics(np.array([-41.,-48.,-95.]),np.full(3,4.),t)
    t['center_dbm'][-1]=99999.
    second=study.observation_metrics(np.array([-41.,-48.,-95.]),np.full(3,4.),t)
    assert first==second
    assert first['rmse_db']==pytest.approx(np.sqrt(2.5))
    assert first['heard_count']==2 and first['censored_count']==1


def test_censor_likelihood_penalizes_prediction_above_bound():
    t=targets([-40.,-999.]);t['heard'][1]=False;t['censored'][1]=True
    below=study.observation_metrics(np.array([-40.,-100.]),np.full(2,4.),t)
    above=study.observation_metrics(np.array([-40.,-60.]),np.full(2,4.),t)
    assert above['observation_nll']>below['observation_nll']


def test_interval_tail_is_finite_and_missing_reading_is_excluded():
    t=targets([100.,999.]);t['heard'][1]=False
    score=study.observation_metrics(np.array([-100.,-1.]),np.full(2,4.),t)
    assert np.isfinite(score['observation_nll']) and score['missing_count']==1


@pytest.mark.parametrize('mutation',[lambda t:t.update(heard=np.zeros(2,bool)),
    lambda t:t.update(censored=np.ones(2,bool)),lambda t:t.update(upper_dbm=t['lower_dbm'])])
def test_invalid_endpoint_arrays_rejected(mutation):
    t=targets([-40.,-50.]);mutation(t)
    with pytest.raises(ValueError):study.observation_metrics(np.zeros(2),np.full(2,4.),t)


def test_nonfinite_prediction_rejected():
    with pytest.raises(FloatingPointError):study.observation_metrics([np.nan],[4.],targets([-40.]))


def test_prior_selection_averages_independent_scores_and_tie_is_registered_order():
    fits={(cell,seed):{'validation_mean_layout_rmse_db':1.+seed*.1} for cell in study.INITIAL for seed in study.SEEDS}
    result=study.select_prior(fits)
    assert result['selected_prior']=='none'
    fits[('joint_sigreg',0)]['validation_mean_layout_rmse_db']=.1
    assert study.select_prior(fits)['selected_prior']=='sigreg'


def test_gate_all_requirements_and_each_seed():
    assert study.development_gate(evaluation(),'joint_none')['field_test_permitted'] is True
    result=study.development_gate(evaluation(rmse=(.2,.2,2.5)),'joint_none')
    assert result['checks']['mean_rf_rmse_better'] is True
    assert result['checks']['every_seed_rmse_no_worse'] is False
    assert result['field_test_permitted'] is False


def test_gate_zero_latent_and_p95_are_separate_requirements():
    result=study.development_gate(evaluation(zero=1.1,p95=2.1),'joint_none')
    assert result['checks']['zero_latent_hurts_at_least_one_percent'] is False
    assert result['checks']['mean_p95_better'] is False
    assert result['field_test_permitted'] is False
    assert result['original_service_confirmation_permitted'] is False


def test_gate_finite_bit_and_strict_improvement():
    value=evaluation(rmse=(2.,2.,2.),reference=2.)
    value['members']['in_band']['joint_none-seed0']['layouts'][0]['metrics']['finite']=False
    result=study.development_gate(value,'joint_none')
    assert not result['checks']['mean_rf_rmse_better'] and not result['checks']['all_predictions_finite']


def test_seed_metrics_are_averaged_never_predictions():
    value=evaluation(rmse=(1.,3.,5.))
    np.testing.assert_allclose(study.layout_values(value,'joint_none','rmse_db'),[3.,3.01,3.02,3.03])


def test_paired_bootstrap_uses_central_bonferroni_interval():
    a=np.array([0.,1.,4.,3.]);b=np.array([1.,2.,3.,4.])
    result=study.paired_bootstrap(a,b,samples=1000,seed=7)
    rng=np.random.default_rng(7);boot=(a-b)[rng.integers(4,size=(1000,4))].mean(1)
    np.testing.assert_array_equal(result['adjusted_interval'],np.quantile(boot,[.00625,.99375]))
    assert result['adjusted_confidence']==.9875
    assert 'conditional on three' in result['interval_scope']


def test_bootstrap_is_paired_layout_not_individual_rows():
    result=study.paired_bootstrap(np.arange(24.),np.arange(24.)+1.)
    assert result['layouts']==24 and result['adjusted_interval']==[-1.,-1.]
    assert result['candidate_better_adjusted'] is True
    with pytest.raises(ValueError):study.paired_bootstrap([1.],[2.])


def test_public_registry_is_geometry_disjoint_and_specs_only(monkeypatch):
    from wirelessjev.radiojepa import rt_data
    monkeypatch.setattr(rt_data,'_trace_scene',lambda *a,**k:pytest.fail('Registry traced a field'))
    registry=c.field_registry()
    assert registry['counts']=={'training':48,'validation':8,'development':4,'field_test':24}
    all_specs=[row for rows in registry['stages'].values() for row in rows]
    assert len({(x['width_m'],x['depth_m'],x['ceiling_m']) for x in all_specs})==84
    assert registry['original_service_confirmation_geometries_consumed']==0


def test_reserved_field_path_requires_gate_even_without_trace(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path)
    def denied(stage):raise RuntimeError('gate veto')
    monkeypatch.setattr(c,'authorize_stage',denied)
    with pytest.raises(RuntimeError,match='gate veto'):c.rt_scene_path('field_test',0)
    assert not list(tmp_path.rglob('*'))


def test_immutable_write_and_path_guard(tmp_path):
    path=tmp_path/'record.json';c.write_once(path,{'a':1});c.write_once(path,{'a':1})
    with pytest.raises(RuntimeError):c.write_once(path,{'a':2})
    with pytest.raises(RuntimeError):c.safe_path('../escaped',tmp_path)
    (tmp_path/'redirect').symlink_to('/tmp')
    with pytest.raises(RuntimeError):c.safe_path('redirect/x',tmp_path)


def test_passive_guard_rejects_neural_constructor_and_missing_cache(tmp_path):
    import torch
    with verification.passive_guard():
        with pytest.raises(RuntimeError,match='cannot fit'):torch.nn.Linear(2,2)
        with pytest.raises(RuntimeError,match='existing identical'):c.write_once(tmp_path/'missing.json',{})
    assert not (tmp_path/'missing.json').exists()


def test_phase_guard_checks_previous_even_when_body_fails(monkeypatch):
    calls=[]
    monkeypatch.setattr(c,'check_sources',lambda:calls.append('source'))
    monkeypatch.setattr(c,'verify_previous',lambda:calls.append('previous') or {'ok':True})
    with pytest.raises(ValueError):
        with c.phase_guard('fixture'):raise ValueError('expected')
    assert calls==['source','previous','source','previous']


def test_authorize_field_requires_recomputed_gate_not_literal_qualification(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);(tmp_path/'registration').mkdir()
    c.write_once(tmp_path/'registration/source.json',{})
    monkeypatch.setattr(c,'check_sources',lambda:None);monkeypatch.setattr(c,'registry_sha256',lambda:'registry')
    monkeypatch.setattr(study,'verified_development',lambda:{'field_test_permitted':False})
    c.write_once(tmp_path/'qualification.json',{'field_test_permitted':True})
    with pytest.raises(RuntimeError):c.authorize_stage('field_test')


def test_gate_writer_replays_actual_reported_arrays_before_qualification(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(study,'verify_evaluation',lambda stage:evaluation())
    monkeypatch.setattr(study,'verified_selection',lambda:{'selected_cell':'joint_none'})
    def rejected(stage):raise RuntimeError('public target differs from actual raw report')
    monkeypatch.setattr(verification,'verify_public_arrays',rejected)
    with pytest.raises(RuntimeError,match='actual raw report'):study.write_development()
    assert not (tmp_path/'qualification.json').exists()
    assert not (tmp_path/'development.json').exists()


def test_exact_member_membership_rejected_before_scoring(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(study,'identity',lambda:{'fixture':True})
    monkeypatch.setattr(study,'verify_reference',lambda value:tmp_path/'selection.json')
    monkeypatch.setattr(study,'verified_selection',lambda:{'selected_cell':'joint_none'})
    value={'stage':'development','identity':{'fixture':True},'selection':{},'members':{
        k:{} for k in ('in_band','held_band','onboarding_0','onboarding_1','onboarding_4','onboarding_16')}}
    monkeypatch.setattr(c,'read_json',lambda path:value)
    with pytest.raises(RuntimeError,match='member membership'):study.verify_evaluation('development')


def test_onboarding_unavailability_retains_counts_and_never_replaces_support(monkeypatch):
    from types import SimpleNamespace
    from wirelessjev.spectral_scene import data
    monkeypatch.setattr(c,'COUNTS',{'development':2})
    monkeypatch.setattr(data,'load_calibration',lambda:object())
    scenes=[SimpleNamespace(scene_id='fixture-a',index=0),SimpleNamespace(scene_id='fixture-b',index=1)]
    monkeypatch.setattr(data,'load_scene',lambda stage,index,role=None:scenes[index])
    monkeypatch.setattr(data,'_paired_reports',lambda scene,device:[{}]*(4 if scene.index==0 else 20))
    monkeypatch.setattr(data,'make_example',lambda scene,**kw:SimpleNamespace(targets=SimpleNamespace(heard=np.array([scene.index==0]))))
    monkeypatch.setattr(data,'onboard_device',lambda *a,**k:pytest.fail('Unavailable assay invented onboarding support'))
    rows=study._examples('development','onboarding_16')
    assert [r['scene_id'] for r in rows]==['fixture-a','fixture-b']
    assert rows[0]['reason']=='insufficient_genuine_context_pairs' and rows[0]['available_context_pairs']==4
    assert rows[1]['reason']=='no_heard_held_device_query' and rows[1]['available_context_pairs']==20
    assert all(r['requested_context_pairs']==16 and r['available'] is False for r in rows)


def test_sparse_onboarding_reports_denominator_and_cannot_drop_primary():
    value=evaluation()
    for n in (0,1,4,16):
        group=deepcopy(value['members']['in_band'])
        unavailable=study.onboarding_availability('fixture-0',0,n,0,0)
        for member in group.values():member['layouts'][0]={'scene_id':'fixture-0','available':False,'availability':unavailable}
        value['members'][f'onboarding_{n}']=group
    curve=study.onboarding_curve(value,'joint_none')
    assert all(row['registered_layouts']==4 and row['available_layouts']==3 for row in curve)
    assert curve[3]['unavailable_layouts'][0]['available_context_pairs']==0
    value['members']['in_band']['joint_none-seed0']['layouts'][0]['available']=False
    with pytest.raises(RuntimeError,match='cannot drop'):study.development_gate(value,'joint_none')


def test_verified_gate_checks_complete_raw_development_first(monkeypatch):
    calls=[]
    def rejected(stage):calls.append(stage);raise RuntimeError('raw CIR bytes changed')
    monkeypatch.setattr(study,'validate_stage',rejected)
    monkeypatch.setattr(study,'verify_evaluation',lambda stage:pytest.fail('Gate read predictions before checking raw stage'))
    with pytest.raises(RuntimeError,match='raw CIR'):study.verified_development()
    assert calls==['development']


def test_passive_replay_requires_exact_unavailable_assay_membership(monkeypatch):
    status=study.onboarding_availability('fixture-a',0,16,4,2)
    row={'scene_id':'fixture-a','available':False,'availability':deepcopy(status)}
    value={'same_link_frequency':{'fixture':True},'members':{'onboarding_16':{'joint_none-seed0':{'seed':0,'layouts':[row]}}}}
    monkeypatch.setattr(study,'verify_evaluation',lambda stage:value)
    monkeypatch.setattr(study,'verified_selection',lambda:{'selected_cell':'joint_none'})
    monkeypatch.setattr(study,'frequency_diagnostic',lambda *a,**k:{'fixture':True})
    monkeypatch.setattr(study,'_examples',lambda stage,regime:[status])
    assert verification.verify_public_arrays('development')['prediction_records_replayed']==0
    row['availability']['available_context_pairs']=99
    with pytest.raises(RuntimeError,match='actual support'):verification.verify_public_arrays('development')
