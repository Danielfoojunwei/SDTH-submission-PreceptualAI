"""Known-ID software fixtures: no adaptive RF scene or learned fit is executed."""
from copy import deepcopy
import json

import pytest

from wirelessjev.adaptive_radio import common as c


@pytest.fixture
def cache_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    checks=[];monkeypatch.setattr(c,'check_sources',lambda:checks.append('source'))
    c.write_once(c.OUT/'registration/source.json',{'fixture':True})
    data=tmp_path/'inputs';data.mkdir();(data/'raw.json').write_text('{"fixture":1}')
    gate={'test_permitted':True,'original_service_confirmation_permitted':False,'production_qualified':False,'score':{'gain':1.}}
    def original():checks.append('full');return deepcopy(gate)
    def dependencies():return {str(p.relative_to(tmp_path)):c.digest(p) for p in sorted(data.rglob('*')) if p.is_file()}
    cache=c.EvidenceGateCache(original,dependencies)
    return cache,gate,data,checks


def test_intrinsic_gate_cache_full_once_deepcopy_and_final_replay(cache_fixture):
    cache,gate,data,checks=cache_fixture
    assert cache()['test_permitted'] is True
    result=cache();result['score']['gain']=999
    assert cache()['score']['gain']==1. and checks.count('full')==1
    cache.finalize();assert checks.count('full')==2
    files_before={str(p):p.read_bytes() for p in c.OUT.rglob('*') if p.is_file()}
    cache();assert files_before=={str(p):p.read_bytes() for p in c.OUT.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError):cache.finalize()


@pytest.mark.parametrize('mutation',['gate','dependencies','original','enumerator','bootstrap','raw','new_file','deleted_file'])
def test_cached_permission_cannot_outlive_source_or_dependency_change(cache_fixture,mutation):
    cache,gate,data,checks=cache_fixture;cache.bootstrap()
    if mutation=='gate':cache._gate['score']['gain']=10
    elif mutation=='dependencies':cache._bound.clear()
    elif mutation=='original':cache.original=lambda:deepcopy(cache._gate)
    elif mutation=='enumerator':cache.dependencies=lambda:deepcopy(cache._bound)
    elif mutation=='bootstrap':cache.path.write_text('{}')
    elif mutation=='raw':(data/'raw.json').write_text('changed')
    elif mutation=='new_file':(data/'extra.json').write_text('{}')
    else:(data/'raw.json').unlink()
    with pytest.raises((RuntimeError,KeyError)):cache()


@pytest.mark.parametrize('key,value',[('test_permitted',False),('test_permitted',1),
    ('original_service_confirmation_permitted',True),('original_service_confirmation_permitted',0),
    ('production_qualified',True),('production_qualified',None)])
def test_original_permission_flags_cannot_waive_service_or_production(cache_fixture,key,value):
    cache,gate,data,checks=cache_fixture;gate[key]=value
    with pytest.raises(RuntimeError):cache.bootstrap()
    assert not cache.path.exists()


def test_changed_original_result_rejected_by_final_full_replay(cache_fixture):
    cache,gate,data,checks=cache_fixture;cache.bootstrap();gate['score']['gain']=5.
    with pytest.raises(RuntimeError):cache.finalize()
    assert not (c.OUT/'gates/development-final.json').exists()


def test_gate_receipt_is_stable_across_restart_but_full_replay_is_required(cache_fixture):
    cache,gate,data,checks=cache_fixture;cache.bootstrap();before=cache.path.read_bytes()
    other=c.EvidenceGateCache(cache.original,cache.dependencies);other.bootstrap()
    assert other.path.read_bytes()==before and checks.count('full')==2


def test_hit_log_tamper_rejected(cache_fixture):
    cache,gate,data,checks=cache_fixture;cache.bootstrap();cache()
    path=next((c.OUT/'gates').glob('*.jsonl'));json.loads(path.read_text())
    path.write_text('{}\n')
    with pytest.raises(RuntimeError,match='log changed'):cache()


def test_phase_guard_checks_prior_after_body_error(monkeypatch):
    calls=[];monkeypatch.setattr(c,'check_sources',lambda:calls.append('source'))
    monkeypatch.setattr(c,'verify_previous',lambda:calls.append('previous') or {'ok':True})
    with pytest.raises(ValueError):
        with c.phase_guard('fixture'):raise ValueError('fixture error')
    assert calls==['source','previous','source','previous']


def _targets(center,heard=None):
    import numpy as np
    center=np.asarray(center,dtype=float);heard=np.ones(len(center),bool) if heard is None else np.asarray(heard,bool)
    return {'center_dbm':center,'lower_dbm':center-.5,'upper_dbm':center+.5,'heard':heard,
        'censored':~heard,'censor_upper_dbm':np.full(len(center),-80.),'detection_threshold_dbm':np.full(len(center),-80.)}


def _examples(targets):
    import numpy as np
    from types import SimpleNamespace
    result=[]
    for i,t in enumerate(targets):
        n=len(t['center_dbm']);rows=SimpleNamespace(pose_index=np.arange(n)*2+1,ap_index=np.zeros(n,int),
            band_index=np.zeros(n,int),direction=np.zeros(n,int),device=np.zeros(n,int))
        result.append(SimpleNamespace(layout_id='known-fixture',condition=i,case_id=f'known-fixture-c{i}',
            query_rows=rows,target_arrays=lambda t=t:t))
    return result


def test_condition_changes_cancel_constant_scalar_bias_and_keep_geometry_unit():
    import numpy as np
    from wirelessjev.adaptive_radio import study
    targets=[_targets([-50.,-55.]),_targets([-45.,-50.]),_targets([-52.,-57.])]
    vals=_examples(targets);pred=[(t['center_dbm']+1,np.ones(2)) for t in targets]
    result=study.condition_metrics(pred,vals)
    assert len(result)==1 and result[0]['delta_rmse_db']==0. and result[0]['rmse_db']==1.
    assert result[0]['delta_both_heard_count']==4 and len(result[0]['case_ids'])==3
    stale=[(targets[0]['center_dbm']+1,np.ones(2)) for _ in targets]
    assert study.condition_metrics(stale,vals)[0]['delta_rmse_db']==pytest.approx(np.sqrt(14.5))


def test_one_censored_state_retains_allcases_but_delta_is_explicitly_unavailable():
    import numpy as np
    from wirelessjev.adaptive_radio import study
    targets=[_targets([-50.,-55.]),_targets([-85.,-50.],[False,True]),_targets([-85.,-90.],[False,False])]
    result=study.condition_metrics([(np.full(2,-55.),np.full(2,4.)) for _ in targets],_examples(targets))[0]
    assert result['delta_available'] is False and result['delta_rmse_db'] is None
    assert len(result['case_ids'])==3 and result['censored_count']==3
    assert result['pair_detection'][1]['both_heard_count']==0
    assert result['detector_brier'] is not None


def test_condition_groups_cannot_drop_or_mismatch_physical_query_cases():
    import numpy as np
    from wirelessjev.adaptive_radio import study
    targets=[_targets([-50.]) for _ in range(3)];vals=_examples(targets);p=[(np.array([-50.]),np.ones(1)) for _ in range(3)]
    with pytest.raises(ValueError,match='All three'):study.condition_metrics(p[:2],vals[:2])
    vals[2].query_rows.pose_index=np.array([9])
    with pytest.raises(ValueError,match='different physical'):study.condition_metrics(p,vals)


def test_heard_bin_likelihood_intersects_threshold_and_brier_ignores_zero_censor_placeholder():
    import numpy as np
    from scipy.special import ndtr
    from wirelessjev.adaptive_radio import study
    t=_targets([-80.]);t['censor_upper_dbm'][0]=0.
    result=study.observation_metrics(np.array([-80.]),np.ones(1),t)
    assert result['observation_nll']==pytest.approx(-np.log(ndtr(.5)-ndtr(0.)))
    assert result['detector_brier']==.25


def test_censored_point_center_never_enters_delta_or_point_error():
    import numpy as np
    from wirelessjev.adaptive_radio import study
    t=_targets([-50.,-9000.],[True,False]);a=study.observation_metrics(np.array([-51.,-85.]),np.full(2,4.),t)
    t['center_dbm'][1]=99999.;b=study.observation_metrics(np.array([-51.,-85.]),np.full(2,4.),t)
    assert a==b and a['rmse_db']==1.


def test_branch_selection_is_VALonly_joint_regression_with_canonical_tie():
    from wirelessjev.adaptive_radio import study
    fits={(cell,s):{'validation_mean_layout_delta_rmse_db':1.} for cell in ('joint','regression') for s in (0,1,2)}
    fits[('rf_erased',0)]={'validation_mean_layout_delta_rmse_db':0.}
    assert study.select_branch(fits)['selected_cell']=='joint'
    fits[('regression',2)]['validation_mean_layout_delta_rmse_db']=.1
    assert study.select_branch(fits)['selected_cell']=='regression'


def test_bootstrap_uses_whole_layout_pairs_and_registered_central_interval():
    import numpy as np
    from wirelessjev.adaptive_radio import study
    a=np.array([1.,2.,3.,4.]);b=np.array([2.,3.,4.,5.]);score=study.paired_bootstrap(a,b)
    assert score['layouts']==4 and score['adjusted_interval']==[-1.,-1.]
    assert 'three RF conditions' not in score['scope'] and 'whole geometry' in score['unit']


def test_only_exact_registered_test_data_names_can_escape_gate_closure(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from wirelessjev.adaptive_radio import study
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'COUNTS',{'test':1})
    monkeypatch.setattr(c,'case_spec',lambda stage,index,condition:SimpleNamespace(case_id=f'fixture-c{condition}',layout_id='fixture'))
    meta={'stage':'test','regime':'sparse','held_band':False,'stale':False}
    assert study._is_registered_test_data(tmp_path/'data/test-sparse-main-current.json',meta)
    assert not study._is_registered_test_data(tmp_path/'data/DEV-counterfeit.json',meta)
    assert not study._is_registered_test_data(tmp_path/'data/test-sparse-main-current.json',meta|{'stale':True})
    case={'stage':'test','case_id':'fixture-c2','layout_id':'fixture','condition':2}
    assert study._is_registered_test_data(tmp_path/'data/scenes/fixture-c2-field_test.json',case)
    assert not study._is_registered_test_data(tmp_path/'data/scenes/fixture-c2-development.json',case)


def _gate_evaluation(seed_delta=(1.,1.,1.),censored=1,usable=4):
    from wirelessjev.adaptive_radio import study
    def member(cell,seed,delta):
        return {'cell':cell,'seed':seed,'layouts':[{'layout_id':f'known-fixture-{i}',
            'delta_available':i<usable,'delta_rmse_db':delta if i<usable else None,
            'rmse_db':1.,'p95_abs_db':2.,'censored_count':censored,'finite':True} for i in range(4)]}
    members={'radiomap_operator':member('radiomap_operator',None,2.)}
    for seed in (0,1,2):
        members[f'joint-seed{seed}']=member('joint',seed,seed_delta[seed])
        members[f'joint-seed{seed}-stale']=member('joint',seed,2.)
        members[f'rf_erased-seed{seed}']=member('rf_erased',seed,2.)
    return {'members':{'sparse':members,'full':deepcopy(members)}}


def test_practical_gate_requires_real_censor_support_and_minimum_geometries():
    from wirelessjev.adaptive_radio import study
    assert study.development_gate(_gate_evaluation(),'joint')['test_permitted'] is True
    for kwargs in ({'censored':0},{'usable':2}):
        assert study.development_gate(_gate_evaluation(**kwargs),'joint')['test_permitted'] is False


def test_gate_cannot_hide_bad_seed_behind_favorable_mean():
    from wirelessjev.adaptive_radio import study
    value=study.development_gate(_gate_evaluation(seed_delta=(.2,.2,2.1)),'joint')
    assert value['checks']['five_percent_better_than_radiomap'] is True
    assert value['checks']['every_seed_delta_no_worse_than_radiomap'] is False
    assert value['test_permitted'] is False


def test_gate_stale_context_and_two_percent_tail_margin_are_independent():
    from wirelessjev.adaptive_radio import study
    evaluation=_gate_evaluation()
    for seed in (0,1,2):
        for row in evaluation['members']['sparse'][f'joint-seed{seed}-stale']['layouts']:row['delta_rmse_db']=1.01
        for row in evaluation['members']['sparse'][f'joint-seed{seed}']['layouts']:row['p95_abs_db']=2.05
    value=study.development_gate(evaluation,'joint')
    assert not value['checks']['stale_context_hurts_at_least_five_percent']
    assert not value['checks']['p95_within_two_percent']
    assert value['original_service_confirmation_permitted'] is False and value['production_qualified'] is False


def test_finalized_cache_restart_does_not_append_after_seal_boundary(cache_fixture):
    cache,gate,data,checks=cache_fixture;cache.bootstrap();cache.finalize()
    snapshot={str(p):p.read_bytes() for p in c.OUT.rglob('*') if p.is_file()}
    restarted=c.EvidenceGateCache(cache.original,cache.dependencies);restarted.bootstrap();restarted()
    assert restarted._logging_frozen is True
    assert snapshot=={str(p):p.read_bytes() for p in c.OUT.rglob('*') if p.is_file()}
