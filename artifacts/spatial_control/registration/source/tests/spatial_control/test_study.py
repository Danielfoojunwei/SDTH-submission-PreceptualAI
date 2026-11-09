"""Finite protocol and numerical reduction checks, without scientific execution."""
import copy
import hashlib
import json
import numpy as np
import pytest

from wirelessjev.spatial_control import study


def test_exact_finite_budget_contains_every_prespecified_arm_once():
    p=study.spec()
    assert len(p['seeds'])==len(set(p['seeds']))==4
    assert p['robots']==100 and p['duration_seconds']==20
    assert len(p['cases'])==8 and len({c['id'] for c in p['cases']})==8
    assert p['fresh_comparisons']==len(p['cases'])*len(study.ARMS)==48
    assert p['maximum_native_simulations']==p['fresh_comparisons']+p['disabled_parity_simulations']==50
    for case in p['cases']:
        assert len(case['order'])==len(set(case['order']))==6
        assert set(case['order'])==set(study.ARMS)
        for arm in case['order']: assert study.admit_case(case,arm)==case
    assert p['weights']['new_fits']==0
    assert p['weights']['encoder_context_selection_unchanged'] is True
    assert p['recovery']['realized_observations_equal'] is False
    assert not p['original_SERVICE_opened'] and not p['default_changed'] and not p['production_qualified']


@pytest.mark.parametrize('change',[
    {'seed':123}, {'id':'unregistered'}, {'handover_seconds':.05},
    {'order':[]}, {'extra':'ignored_fields_are_not_permitted'},
])
def test_case_mutations_are_not_silently_admitted(change):
    case=study.spec()['cases'][0]|change
    with pytest.raises(ValueError): study.admit_case(case,'radiomap')


@pytest.mark.parametrize('arm',['unknown','regression_seed1','radiomap__original','REGRESSION'])
def test_fresh_arm_membership_is_exact(arm):
    with pytest.raises(ValueError): study.admit_case(study.spec()['cases'][0],arm)


def test_disabled_parity_is_a_separate_two_simulation_identity():
    parity={'id':'disabled-parity','seed':study.SEEDS[0],'handover_seconds':.03}
    assert study.admit_case(parity,'radiomap__original','original')==parity
    assert study.admit_case(parity,'radiomap__disabled','disabled')==parity
    for arm,kind in [('radiomap__disabled','original'),('radiomap','disabled'),('radiomap__original','unknown')]:
        with pytest.raises(ValueError): study.admit_case(parity,arm,kind)
    with pytest.raises(ValueError): study.admit_case(parity,'radiomap')
    with pytest.raises(ValueError): study.admit_case(study.spec()['cases'][0],'radiomap__original','original')


def test_no_seed_or_threshold_is_selected_from_native_results():
    p=study.spec()
    assert p['weights']['selected_cell']=='regression' and p['weights']['seed']==0
    assert p['analysis']['no_post_result_architecture_threshold_or_seed_changes']
    assert not p['analysis']['confirmatory_p_values']
    assert p['central']['all_handover_costs_require_delivery']
    assert p['central']['delivery_strictly_after_creation']
    assert p['recovery']['own_completed_scans_and_current_pilots_only']
    assert p['recovery']['warm_scan_exact_times_unknown_so_not_used_as_recovery_payloads']


def counter_fixture(*,spatial=True):
    """Explicit software numbers, never written to actual experiment directories."""
    misses=np.arange(100)%4; sent=np.full(100,400);gap=np.arange(100)%31
    native={'packet_detail':{'misses':misses.tolist(),'sent':sent.tolist(),'max_gap':gap.tolist()},
        'control_miss':float(misses.sum()/sent.sum()),
        'control_miss_worst_decile':float(np.sort(misses/sent)[-10:].mean()),
        'control_gap_p99_ms':float(np.percentile(gap,99)*10),
        'video_on_time':.9,'bulk_mbps':100.,'roams':2.,'safety_stop_fraction':.01}
    if spatial:
        native['spatial_execution']={'pending_control_packets':3,
            'local_events':[{'event':'executed'},{'event':'queued'}],
            'central_events':[{'event':'executed'},{'event':'cancelled'}],
            'local_decisions':[],'central_decisions':[],'scan_events':[]}
    return {'native':native,'forecast_jobs':[
        {'charged_compute_seconds':1.,'actual_compute_seconds':.2},
        {'charged_compute_seconds':.25,'actual_compute_seconds':.15}],
        'forecast_consumption':[{'decision':0}],
        'telemetry':{'scans_started':2},'support':[{'permitted':True},{'permitted':False}]}


def test_metrics_recompute_integer_packet_counts_and_keep_risk_units():
    result=counter_fixture();m=study.metrics(result)
    assert m['control_misses']==150 and m['control_sent']==40000
    assert m['control_miss']==.00375
    assert m['control_miss_worst_decile']==pytest.approx(.0075)
    assert m['maximum_silence_ms']==300.
    assert m['end_pending_commands']==3
    assert m['observed_plus_pending_miss_fraction']==153/40000
    assert m['paid_forecast_compute_seconds']==1.25
    assert m['actual_forecast_compute_seconds']==.35
    assert m['local_handovers']==m['central_handovers']==1
    assert m['support_vetoes']==1


@pytest.mark.parametrize('field',['control_miss','control_miss_worst_decile','control_gap_p99_ms'])
def test_changed_aggregate_cannot_override_raw_packet_counters(field):
    r=counter_fixture();r['native'][field]+=.01
    with pytest.raises(RuntimeError): study.metrics(r)


def test_disabled_predecessor_missing_pending_status_is_not_invented():
    m=study.metrics(counter_fixture(spatial=False))
    assert m['end_pending_commands'] is None
    assert m['observed_plus_pending_miss_fraction'] is None
    assert m['local_handovers']==m['central_handovers']==0


def test_array_identity_includes_dtype_shape_and_exact_bytes():
    a=np.arange(6,dtype='f4').reshape(2,3)
    assert study.array_hash(a)==study.array_hash(a.copy())
    assert study.array_hash(a)!=study.array_hash(a.astype('f8'))
    assert study.array_hash(a)!=study.array_hash(a.reshape(3,2))
    b=a.copy();b[0,0]=1
    assert study.array_hash(a)!=study.array_hash(b)


def test_canonical_json_accepts_native_numpy_scalars_and_rejects_nan():
    raw=study.canonical({'int':np.int64(2),'array':np.array([1.,2.])})
    assert json.loads(raw)=={'int':2,'array':[1.,2.]}
    with pytest.raises(ValueError): study.canonical({'fake':float('nan')})


def test_once_only_evidence_write_refuses_replacing_a_partial_attempt(tmp_path):
    path=tmp_path/'attempt.json'
    study.write(path,{'started':True})
    before=path.read_bytes()
    with pytest.raises(FileExistsError): study.write(path,{'replacement':True})
    assert path.read_bytes()==before


def test_prior_manifest_membership_is_fixed_and_fresh_namespace_is_separate():
    assert set(study.PRIOR)=={'radiojepa','radiojepa_spectral','device_transfer',
        'channel_scene','adaptive_radio','native_bridge','e7_calplanning'}
    assert study.OUT.name=='spatial_control'
    assert study.OUT.parent.name=='artifacts'
