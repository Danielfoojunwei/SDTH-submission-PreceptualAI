"""Analytic metric/probe fixtures, never public benchmark result numbers."""
import numpy as np
import pytest

from wirelessjev.radiojepa.evaluation import (
    CELLS, RidgeModel, aggregate_fit_reports, mean_template, metrics, power_level_transfer, representation_metrics, ridge_fit,
)


def test_power_metrics_and_row_tail_use_referenced_db_not_dbm():
    truth = np.zeros((20,2,3))
    prediction = np.zeros_like(truth)
    prediction[-2] = 3.
    prediction[-1] = 4.
    result = metrics(prediction,truth)
    assert result['rmse_db'] == pytest.approx(np.sqrt(25/20))
    assert result['mae_db'] == pytest.approx(7/20)
    assert result['worst_decile_rmse_db'] == 3.5
    assert result['samples'] == 20 and result['target_elements'] == 120
    assert 'not dBm' in result['power_unit']


@pytest.mark.parametrize('bad',[np.array([np.nan]),np.array([np.inf]),np.array([1+1j]),np.array([]),np.array(['x'])])
def test_metrics_reject_invalid_values_instead_of_filling_or_silently_casting(bad):
    with pytest.raises(ValueError): metrics(bad,bad)


def test_metrics_reject_shape_mismatch_and_numerical_overflow():
    with pytest.raises(ValueError): metrics(np.zeros((2,3)),np.zeros((2,1)))
    with pytest.raises(ValueError): metrics(np.array([1e308]),np.array([0.]))


def test_constant_representations_have_zero_rank_and_undefined_alignment():
    result = representation_metrics(np.full((30,48),.1),np.ones((30,16)))
    assert result['effective_rank_a'] == result['effective_rank_b'] == 0
    assert result['mean_coordinate_std_a'] == result['mean_coordinate_std_b'] == 0
    assert result['mutual_knn_alignment'] is None and not result['alignment_defined']
    assert result['alignment_undefined_reason'] == 'constant_coordinates'
    assert result['random_alignment_expected'] == pytest.approx(5/29)


def test_single_row_and_zero_norm_vectors_are_not_perfectly_aligned():
    assert representation_metrics([[1.,2.]],[[2.,1.]])['mutual_knn_alignment'] is None
    a = np.array([[0.,0.],[1.,2.],[3.,4.]])
    assert representation_metrics(a,a)['alignment_undefined_reason'] == 'undefined_cosine_for_zero_norm_row'


def test_rotations_and_positive_global_scaling_preserve_cosine_neighbors():
    rng = np.random.default_rng(82)
    a = rng.normal(size=(90,12))
    q,_ = np.linalg.qr(rng.normal(size=(12,12)))
    value = representation_metrics(a,3*a@q)
    assert value['mutual_knn_alignment'] == 1.
    assert value['effective_rank_a'] == pytest.approx(value['effective_rank_b'])
    assert value['mean_coordinate_std_b'] > value['mean_coordinate_std_a']


def test_independent_graphs_match_the_declared_random_neighbor_baseline():
    rng = np.random.default_rng(202)
    value = representation_metrics(rng.normal(size=(600,16)),rng.normal(size=(600,9)),k=7)
    assert value['random_alignment_expected'] == pytest.approx(7/599)
    assert abs(value['mutual_knn_alignment']-7/599) < .008


def test_cosine_boundary_ties_are_neutral_not_resolved_by_matching_row_index():
    a = np.array([[1.,0.],[1.,0.],[-1.,0.],[-1.,0.]])
    value = representation_metrics(a,a,k=2)
    # One unique neighbor + two half-weight tied neighbors: (1+.25+.25)/2.
    assert value['mutual_knn_alignment'] == .75


def test_rank_cannot_exceed_centered_sample_or_coordinate_dimension():
    rng = np.random.default_rng(1)
    value = representation_metrics(rng.normal(size=(5,48)),rng.normal(size=(5,48)))
    assert 0 < value['effective_rank_a'] <= 4+1e-12
    assert value['k'] == 4
    assert value['alignment_minus_random'] == 0.


@pytest.mark.parametrize('features,samples,solver',[(3,40,'primal'),(40,8,'dual')])
def test_ridge_matches_analytic_solution_in_both_solver_regimes(features,samples,solver):
    rng = np.random.default_rng(74)
    x = rng.normal(size=(samples,features))
    y = rng.normal(size=(samples,2))
    vx = rng.normal(size=(5,features))
    vy = rng.normal(size=(5,2))
    fitted = ridge_fit(x,y,vx,vy,alphas=(2.,))
    scale = np.maximum(x.std(0),1e-8)
    tx = (x-x.mean(0))/scale
    expected = np.linalg.solve(tx.T@tx+2*np.eye(features),tx.T@(y-y.mean(0)))
    np.testing.assert_allclose(fitted.coef,expected,atol=1e-11,rtol=1e-10)
    assert fitted.metadata['solver'] == solver


def test_standardization_and_intercept_only_see_training_rows():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100,4))
    y = x[:,:2]*2+3
    vx = rng.normal(size=(20,4))+1000
    vy = vx[:,:2]*2+3
    fitted = ridge_fit(x,y,vx,vy)
    np.testing.assert_array_equal(fitted.x_mean,x.mean(0))
    np.testing.assert_array_equal(fitted.y_mean,y.mean(0))
    assert fitted.metadata['test_accessed'] is False
    best = min(fitted.metadata['validation_grid'],key=lambda v:v['validation_mse'])
    assert fitted.alpha == best['alpha']


@pytest.mark.parametrize('scalar_target',[True,False])
def test_frozen_probe_parameters_roundtrip_without_refitting(scalar_target):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30,5))
    y = rng.normal(size=30 if scalar_target else (30,2,3))
    fitted = ridge_fit(x[:20],y[:20],x[20:],y[20:])
    loaded = RidgeModel.from_arrays(fitted.as_arrays())
    np.testing.assert_array_equal(loaded.predict(x),fitted.predict(x))
    assert loaded.predict(x).shape == y.shape
    with pytest.raises(ValueError): loaded.coef.setflags(write=True)


def test_constant_feature_columns_and_invalid_penalties_are_explicit():
    x = np.ones((10,4))
    model = ridge_fit(x,np.arange(10.),x[:2],np.array([1.,2.]))
    np.testing.assert_array_equal(model.predict(x[:2]),np.full(2,4.5))
    for alphas in [(0.,),(1.,1.),(True,),()]:
        with pytest.raises(ValueError): ridge_fit(x,np.arange(10.),x[:2],np.array([1.,2.]),alphas)


def test_template_and_power_level_transfer_do_not_read_query_targets():
    x = np.array([[[1.,2.]],[[3.,4.]]])
    y = x+10
    query = np.array([[[11.,12.]],[[21.,22.]]])
    baseline = power_level_transfer(x,y,query)
    np.testing.assert_array_equal(baseline,mean_template(y,2)+np.array([9.,19.])[:,None,None])


def test_descriptive_aggregate_requires_all_members_and_preserves_undefined_alignment():
    import copy
    domain={'metrics':{'rmse_db':2.,'mae_db':1.,'worst_decile_rmse_db':3.,'samples':5},
            'representation_z':{'effective_rank_a':0.,'effective_rank_b':0.,'mean_coordinate_std_a':0.,
                'mean_coordinate_std_b':0.,'mutual_knn_alignment':None},
            'representation_projected':{'effective_rank_a':0.,'effective_rank_b':0.,'mean_coordinate_std_a':0.,
                'mean_coordinate_std_b':0.,'mutual_knn_alignment':None},
            'interventions':{'z_zero':{'rmse_db':2.}},'ab_metrics':{'rmse_db':2.}}
    fit=[{'completed':True,'cell':cell,'seed':seed,'parameters':20,'domains':{'fixture':copy.deepcopy(domain)}}
         for cell in CELLS for seed in (0,1,2)]
    baseline={'domains':{'fixture':{'metrics':{'constant':{'rmse_db':3.}}}}}
    with pytest.raises(ValueError,match='eighteen'):aggregate_fit_reports(fit[:-1],baseline)
    result=aggregate_fit_reports(fit,baseline)
    alignment=result['cells']['joint_none']['domains']['fixture']['representation_z']['mutual_knn_alignment']
    assert alignment['mean'] is None and alignment['defined_runs']==0
    assert result['completed_fits']==18 and result['prediction_ensembling'] is False
    assert result['robot_service_qualified'] is result['global_sota_claimed'] is False


def test_passive_ridge_audit_rejects_tampered_coefficients_and_validation_score(monkeypatch):
    import copy
    from wirelessjev.radiojepa import verification as v
    monkeypatch.setattr(v.c,'protocol',lambda:{'baselines':{'ridge_alphas':[.1,1.,10.,100.]}})
    rng=np.random.default_rng(57)
    x=rng.normal(size=(30,5));y=rng.normal(size=(30,2));vx=rng.normal(size=(8,5));vy=rng.normal(size=(8,2))
    fitted=ridge_fit(x,y,vx,vy)
    with v.passive_guard():
        loaded=v.verify_ridge(fitted.as_arrays(),fitted.metadata,x,y,vx,vy)
    np.testing.assert_array_equal(loaded.predict(vx),fitted.predict(vx))
    bad=fitted.as_arrays();bad['coef'][0,0]+=1
    with pytest.raises(RuntimeError,match='training-only objective'):
        v.verify_ridge(bad,fitted.metadata,x,y,vx,vy)
    meta=copy.deepcopy(fitted.metadata)
    winner=next(row for row in meta['validation_grid'] if row['alpha']==fitted.alpha)
    winner['validation_mse']+=.01
    with pytest.raises(RuntimeError):v.verify_ridge(fitted.as_arrays(),meta,x,y,vx,vy)


def test_passive_guard_blocks_new_models_optimizers_and_fits_but_restores_methods():
    import torch
    from wirelessjev.radiojepa import verification as v
    parameter=torch.nn.Parameter(torch.ones(1))
    original=np.linalg.solve
    with v.passive_guard():
        with pytest.raises(RuntimeError,match='Passive'):torch.nn.Linear(2,2)
        with pytest.raises(RuntimeError,match='Passive'):torch.optim.Adam([parameter])
        with pytest.raises(RuntimeError,match='Passive'):np.linalg.solve(np.eye(2),np.ones(2))
    assert np.linalg.solve is original
    np.testing.assert_array_equal(np.linalg.solve(np.eye(2),np.ones(2)),np.ones(2))


def test_group_power_transform_matches_linear_average_and_has_no_pose_feature():
    from wirelessjev.radiojepa.verification import measured_transform
    csi=np.zeros((2,4,1024,2),dtype='f4')
    csi[...,0]=np.array([1.,3.,2.,4.])[None,:,None]
    raw={'csi':csi,'time':np.array([1.,2.]),'pos-tachy':np.zeros((2,3))}
    binding={'antenna_group_ids':[0,0,1,1],
             'array_centers_by_antenna_m':[[0,0,0],[0,0,0],[1,0,0],[1,0,0]],
             'frequency_hz':3e9,'bandwidth_hz':50e6}
    public,count=measured_transform(raw,binding)
    assert count==4 and public['power_db'].shape==(2,2,64)
    np.testing.assert_allclose(public['power_db'][:,0],10*np.log10(5),atol=1e-6)
    np.testing.assert_allclose(public['power_db'][:,1],10.,atol=1e-6)
    raw['pos-tachy'][:]=1e5
    changed,_=measured_transform(raw,binding)
    assert 'positions' not in public
    np.testing.assert_array_equal(public['power_db'],changed['power_db'])


def test_recursive_audit_comparison_never_accepts_boolean_as_a_metric():
    from wirelessjev.radiojepa.verification import equivalent
    with pytest.raises(RuntimeError):equivalent(True,1.)
    with pytest.raises(RuntimeError):equivalent(1,True)
    with pytest.raises(RuntimeError):equivalent({'missing':0.},{'expected':0.})


def _prediction_fixture(tmp_path,monkeypatch):
    from wirelessjev.radiojepa import verification as v
    monkeypatch.setattr(v.c,'ROOT',tmp_path);monkeypatch.setattr(v.c,'OUT',tmp_path/'artifacts/radiojepa')
    kinds=['z_zero','tokens_zero','power_zero','all_zero','all_shuffle']
    monkeypatch.setattr(v.c,'protocol',lambda:{'training_recording':'fixture','width':4,'interventions':kinds})
    rng=np.random.default_rng(225)
    public={'power_db':rng.normal(size=(7,4,64)).astype('f4')}
    rows=np.array([3,4,5,6]);va=np.array([0,2]);vb=np.array([1,3])
    ta,tb=public['power_db'][rows][:,va],public['power_db'][rows][:,vb]
    saved={'source_rows':rows,'view_a':va,'view_b':vb,'pred_ab':tb+1.,'pred_ba':ta-.5,
           'z_a':rng.normal(size=(4,4)),'z_b':rng.normal(size=(4,4)),
           'projected_a':rng.normal(size=(4,4)),'projected_b':rng.normal(size=(4,4)),
           'shuffle_rows':np.random.default_rng(2168000).permutation(4)}
    saved.update({'intervention_ab_'+kind:saved['pred_ab']+i for i,kind in enumerate(kinds)})
    path=v.c.OUT/'predictions/fixture.npz';path.parent.mkdir(parents=True)
    np.savez_compressed(path,**saved)
    ref={'path':str(path.relative_to(tmp_path)),'sha256':v.c.digest(path)}
    public_ref={'path':'fixture-public','sha256':'fixture'}
    v.c.write_once(v.c.OUT/'data/receipt.json',{'files':{'fixture.public':public_ref}})
    v.c.write_once(str(path)+'.before-targets.json',{'predictions':ref,'prepared_public':public_ref,
        'source_rows':rows.tolist(),'prediction_complete_before_metric_computation':True})
    record={'prediction_path':ref['path'],'prediction_sha256':ref['sha256'],
        'metrics':metrics(np.concatenate((saved['pred_ab'],saved['pred_ba']),1),np.concatenate((tb,ta),1)),
        'ab_metrics':metrics(saved['pred_ab'],tb),'representation_z':representation_metrics(saved['z_a'],saved['z_b']),
        'representation_projected':representation_metrics(saved['projected_a'],saved['projected_b']),
        'interventions':{kind:metrics(saved['intervention_ab_'+kind],tb) for kind in kinds},
        'intervention_scope':'all_zero/all_shuffle alter both pooledz andreceiver tokens; measuredpower is held fixed; power_zero tests it separately',
        'forward_timing':{'begin_monotonic_ns':1,'end_monotonic_ns':1000001,'seconds':.001,
                         'scope':'both views, preparedpower input, encoder+projector+decoder; excludes CSI preprocessing/loading/transport; sharedCPU; not realtime'}}
    return v,record,public,{'view_a':va.tolist(),'view_b':vb.tolist()},{'test':rows},path


def test_passive_prediction_audit_recomputes_values_and_rejects_metric_tamper(tmp_path,monkeypatch):
    import copy
    v,record,public,info,split,_=_prediction_fixture(tmp_path,monkeypatch)
    with v.passive_guard():v.verify_prediction(record,'fixture',public,info,split)
    broken=copy.deepcopy(record);broken['interventions']['z_zero']['rmse_db']+=1.
    with pytest.raises(RuntimeError,match='intervention'):
        v.verify_prediction(broken,'fixture',public,info,split)


def test_passive_prediction_audit_rejects_changed_archive_bytes(tmp_path,monkeypatch):
    v,record,public,info,split,path=_prediction_fixture(tmp_path,monkeypatch)
    with path.open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(RuntimeError,match='bytes changed'):
        v.verify_prediction(record,'fixture',public,info,split)


def test_byte_closure_checks_document_and_extra_members_without_repeating_raw_audit(tmp_path,monkeypatch):
    from wirelessjev.radiojepa import verification as v
    monkeypatch.setattr(v.c,'ROOT',tmp_path);monkeypatch.setattr(v.c,'OUT',tmp_path/'artifacts/radiojepa')
    monkeypatch.setattr(v.c,'check_sources',lambda:None);monkeypatch.setattr(v.c,'verify_previous',lambda:{'fixture':True})
    source=tmp_path/'source.py';source.write_text('fixture source')
    monkeypatch.setattr(v.c,'source_hashes',lambda:{'source.py':v.c.digest(source)})
    v.c.write_once(v.c.OUT/'registration/source.json',{'fixture':True})
    v.c.write_once(v.c.OUT/'registration/git.json',{'fixture':True})
    v.c.write_once(v.c.OUT/'report.json',{'fixture':True})
    audit={'passes':True,'source_sha256':v.c.source_hashes(),'evidence_sha256':v._visible(),
           'report_sha256':v.c.digest(v.c.OUT/'report.json')}
    v.c.write_once(v.c.OUT/'verification/raw_audit.json',audit)
    doc=tmp_path/'results.md';doc.write_text('Fixture interpretation')
    v.seal((doc,))
    monkeypatch.setattr(v,'verify_campaign',lambda **kwargs:pytest.fail('raw audit repeated'))
    assert v.verify_closure()['raw_audit_repeated'] is False
    doc.write_text('Changed interpretation')
    with pytest.raises(RuntimeError,match='document'):v.verify_closure()
    doc.write_text('Fixture interpretation')
    (v.c.OUT/'unregistered-evidence').write_text('extra')
    with pytest.raises(RuntimeError,match='closure'):v.verify_closure()
