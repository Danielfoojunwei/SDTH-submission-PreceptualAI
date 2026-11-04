"""Pure arithmetic fixtures; no worlds, predictions, training or study data."""
from copy import deepcopy
import json
from pathlib import Path
import random
import pytest
from wirelessjev.costaware_recovery.report import aggregate, arms, development_gate, markdown, replication_summary

MECHANICAL = dict(source_and_archive_integrity=True, causality_and_kernel_checks=True, paid_cost_and_consumption_checks=True)
VALIDATION = dict(MECHANICAL, exact_offline_replay=True)

@pytest.fixture
def p():
    value = json.loads((Path(__file__).resolve().parents[1] / 'artifacts/costaware_recovery/protocol.json').read_text())
    value.update(robots=10, duration_seconds=2, development_sites=[11, 12], confirmation_sites=[21, 22, 23],
                 cumulative_metrics_seconds=[1, 2], bootstrap_replicates=500, worst_decile_margin=1/40)
    return value

def one(p, stage, site, cost, family, variant, miss):
    n, total = p['robots'], 20*p['duration_seconds']
    m, sent, ages = [miss]*n, [total]*n, [-1]*n
    detail = {'misses':m, 'sent':sent, 'max_gap':[2]*n, 'pending_control_packet_age_slots':ages,
        'pending_control_packets':0, 'cumulative':[
            {'slot_end':100,'misses':[miss//2]*n,'sent':[20]*n,'max_gap':[1]*n},
            {'slot_end':200,'misses':m.copy(),'sent':sent.copy(),'max_gap':[2]*n}]}
    account = {'video_requested_bits':1000.,'video_generated_bits':1000.,'video_delivered_bits':900.,
        'video_pending_bits':100.,'video_expired_bits':0.,'video_flow_conservation_error_bits':0.,
        'bulk_generated_bits':100.,'bulk_delivered_bits':90.,'final_bulk_backlog_bits':10.,
        'bulk_overflow_drop_bits':0.,'bulk_flow_conservation_error_bits':0.,
        'pending_control_packet_age_slots':ages.copy(),'pending_control_packets':0}
    return {'stage':stage,'site':site,'handover_seconds':cost,'family':family,'controller_variant':variant,
        'training_seed':None,'runtime_overrun':False,'predictor_runtime_overrun':False,'causality_pass':True,
        'paid_cost_and_consumption_pass':True,'replay':False,'live_replication':False,'accounting_mode':'actual_cost',
        'forecast_metrics':{},'jobs':[{'charged_seconds':.6,'charged_compute_seconds':.1,
            'actual_compute_seconds':.04,'runtime_overrun':False} for _ in range(p['duration_seconds'])],
        'forecast_consumption':[{'slot':60,'forecast_decision':0,'ready_slot':60,'age_seconds':.6}],
        'result':{'control_misses':miss*n,'control_sent':total*n,'control_miss':miss/total,
                  'packet_detail':detail,'packet_accounting':account}}

def matrix(p, stage='confirmation', costs=None):
    if costs is None: costs=p['handover_seconds']
    values={('radiomap','legacy'):8,('radiomap','repair'):4,
        ('frozen_fleet_jepa','legacy'):7,('frozen_fleet_jepa','repair'):2}
    return [one(p,stage,site,cost,family,variant,values[family,variant])
        for site in p[stage+'_sites'] for cost in costs for family,variant in arms(cost)]

def set_counts(row, counts):
    detail=row['result']['packet_detail'];detail['misses']=list(counts)
    detail['cumulative'][0]['misses']=[c//2 for c in counts];detail['cumulative'][-1]['misses']=list(counts)
    row['result']['control_misses']=sum(counts);row['result']['control_miss']=sum(counts)/row['result']['control_sent']

def test_slow_matrix_site_units_and_three_primary_comparisons(p):
    r=aggregate(matrix(p),p,VALIDATION)
    assert r['case_count']==12 and r['independent_sites']==3
    assert len(arms(4.))==4
    assert len(r['primary'])==len(r['engineering_gates'])==3
    assert 'secondary_fast_contrasts' not in r and len(r['cumulative_excess'])==6
    assert len(r['means'])==4 and r['primary_multiplicity_budget']==3
    assert all(len(x['paired_site_differences'])==3 and x['upper_quantile']==1-.05/3 for x in r['primary'])
    assert r['all_three_experimental_benefits']
    assert not r['production_qualified'] and not r['hardware_qualified'] and not r['activated']
    assert not r['ema_necessity_claimed'] and r['new_fits']==0
    assert r['timing']['consumptions']==12 and r['timing']['maximum_forecast_age_seconds']==.6


def test_registered_eight_site_count_is_thirty_two(p):
    p['confirmation_sites']=list(range(21,29));r=aggregate(matrix(p),p,VALIDATION)
    assert r['case_count']==32 and r['independent_sites']==8
    assert development_gate(matrix(p,'development'),p,MECHANICAL)['case_count']==8

@pytest.mark.parametrize('kind',['missing','duplicate','foreign_site','extra_variant','fast_cost','wrong_stage',
                                  'invented_seed','offline','live','mode','numeric_flag'])
def test_exact_matrix_and_accounting_namespaces(p,kind):
    rows=matrix(p)
    if kind=='missing':rows.pop()
    if kind=='duplicate':rows.append(deepcopy(rows[0]))
    if kind=='foreign_site':rows[0]['site']=999
    if kind=='extra_variant':rows[0]['controller_variant']='alternative'
    if kind=='fast_cost':rows[0]['handover_seconds']=.03
    if kind=='wrong_stage':rows[0]['stage']='development'
    if kind=='invented_seed':rows[0]['training_seed']=0
    if kind=='offline':rows[0]['replay']=True
    if kind=='live':rows[0]['live_replication']=True
    if kind=='mode':rows[0]['accounting_mode']='sealed_budget'
    if kind=='numeric_flag':rows[0]['replay']=0
    with pytest.raises(ValueError):aggregate(rows,p,VALIDATION)

@pytest.mark.parametrize('validation',[None,{},MECHANICAL,{**VALIDATION,'exact_offline_replay':False},
    {**VALIDATION,'paid_cost_and_consumption_checks':False},{**VALIDATION,'source_and_archive_integrity':1},
    {**VALIDATION,'causality_and_kernel_checks':False}])
def test_mechanical_and_exact_offline_proofs_fail_closed(p,validation):
    r=aggregate(matrix(p),p,validation)
    assert all(x['statistical_pass'] for x in r['primary'])
    assert not any(x['experimental_benefit'] for x in r['primary'])

@pytest.mark.parametrize('field',['causality_pass','paid_cost_and_consumption_pass'])
def test_one_failed_case_proof_vetoes_all_benefits(p,field):
    rows=matrix(p);rows[0][field]=False;r=aggregate(rows,p,VALIDATION)
    assert all(x['statistical_pass'] for x in r['primary'])
    assert not any(x['experimental_benefit'] for x in r['primary'])

def test_paid_profile_exceedance_is_only_diagnostic(p):
    rows=matrix(p);rows[0].update(runtime_overrun=True,predictor_runtime_overrun=True)
    for j in rows[0]['jobs']:j.update(runtime_overrun=True,actual_compute_seconds=.25,charged_compute_seconds=.25,charged_seconds=.75)
    r=aggregate(rows,p,VALIDATION)
    assert r['timing']['profile_exceeded_cases']==1
    assert r['timing']['profile_exceeded_jobs']==p['duration_seconds']
    assert r['timing']['profile_exceedance_is_diagnostic_not_a_veto']
    assert r['proof']['paid_cost_and_consumption_pass'] and r['all_three_experimental_benefits']
    assert sum(x['profile_exceeded'] for x in r['every_case'])==1
    assert all('runtime_overrun' not in x for x in r['every_case'])

def test_pending_endpoint_blocks_one_command_gain(p):
    rows=matrix(p)
    for r in rows:
        if r['family']=='radiomap' and r['controller_variant']=='repair':
            set_counts(r,[7]*p['robots'])
            for x in (r['result']['packet_detail'],r['result']['packet_accounting']):
                x['pending_control_packet_age_slots']=[0]*p['robots'];x['pending_control_packets']=p['robots']
    out=aggregate(rows,p,VALIDATION)
    c=next(x for x in out['primary'] if x['handover_seconds']==4. and x['contrast_id']=='repair_minus_legacy:radiomap')
    assert c['observed_passes'] and not c['endpoint_passes'] and not c['experimental_benefit']

@pytest.mark.parametrize('change',['silence','worst','video','bulk'])
def test_mean_gain_cannot_hide_tail_or_traffic_regression(p,change):
    rows=matrix(p)
    for r in rows:
        if r['family']!='frozen_fleet_jepa' or r['controller_variant']!='repair':continue
        d,a=r['result']['packet_detail'],r['result']['packet_accounting']
        if change=='silence':d['max_gap']=[20]*p['robots'];d['cumulative'][-1]['max_gap']=[20]*p['robots']
        if change=='worst':counts=d['misses'].copy();counts[0]=30;set_counts(r,counts)
        if change=='video':a.update(video_delivered_bits=800.,video_pending_bits=200.)
        if change=='bulk':a.update(bulk_delivered_bits=80.,final_bulk_backlog_bits=20.)
    out=aggregate(rows,p,VALIDATION)
    c=next(x for x in out['primary'] if x['handover_seconds']==4. and x['contrast_id']=='repair_minus_legacy:frozen_fleet_jepa')
    assert c['statistical_pass'] and not c['engineering_pass'] and not c['experimental_benefit']

def test_no_confirmation_is_never_success(p):
    r=aggregate([],p,VALIDATION,passed_costs=[])
    assert r['case_count']==r['independent_sites']==0 and len(r['primary'])==3
    assert all(x['status']=='not_run_development_veto' and x['mean_difference'] is None and x['multiplicity_upper'] is None and not x['experimental_benefit'] for x in r['primary'])
    assert not r['all_three_experimental_benefits']
    assert r['registered_sites']==p['confirmation_sites'] and 'not measured' in markdown(r)
    with pytest.raises(ValueError):aggregate(matrix(p),p,VALIDATION,passed_costs=[])

def test_development_exact_one_percentage_point_threshold(p):
    rows=matrix(p,'development')
    for r in rows:
        if r['handover_seconds']==4. and r['controller_variant'] in ('legacy','repair'):
            counts=[1]*p['robots']
            if r['controller_variant']=='repair':counts[:4]=[2]*4
            set_counts(r,counts)
    gate=development_gate(rows,p,MECHANICAL)
    assert gate['case_count']==8 and gate['passed_costs']==p['handover_seconds']
    assert all(x['repair_minus_legacy']==.01 and not x['severe_regression'] for x in gate['per_cost'][0]['family_checks'])
    target=next(r for r in rows if r['handover_seconds']==4. and r['controller_variant']=='repair')
    counts=target['result']['packet_detail']['misses'].copy();counts[0]+=1;set_counts(target,counts)
    failed=development_gate(rows,p,MECHANICAL)
    assert failed['passed_costs']==[] and failed['per_cost'][0]['status']=='development_severe_regression_veto'
    assert not failed['model_or_controller_selection_performed'] and not failed['statistical_superiority_claimed']

def test_development_is_diagnostic_not_selection(p):
    rows=matrix(p,'development')
    assert development_gate(rows,p,MECHANICAL)['passed_costs']==p['handover_seconds']
    rows[0]['runtime_overrun']=True
    assert development_gate(rows,p,MECHANICAL)['passed_costs']==p['handover_seconds']
    rows[0]['paid_cost_and_consumption_pass']=False
    assert development_gate(rows,p,MECHANICAL)['passed_costs']==[]
    assert development_gate(matrix(p,'development'),p)['passed_costs']==[]

def test_randomized_execution_order_does_not_change_seals(p):
    rows=matrix(p,'development');shuffled=deepcopy(rows);random.Random(5).shuffle(shuffled)
    assert development_gate(rows,p,MECHANICAL)==development_gate(shuffled,p,MECHANICAL)
    rows=matrix(p);shuffled=deepcopy(rows);random.Random(6).shuffle(shuffled)
    assert aggregate(rows,p,VALIDATION)==aggregate(shuffled,p,VALIDATION)

@pytest.mark.parametrize('mode',['fresh_live_cost','recorded_schedule_offline'])
def test_replications_add_no_independent_sites(p,mode):
    main=matrix(p);rows=[deepcopy(r) for r in main if r['site']==p['confirmation_sites'][0]]
    for r in rows:r.update(replay=mode=='recorded_schedule_offline',live_replication=mode=='fresh_live_cost',accounting_mode='recorded_schedule_offline' if mode=='recorded_schedule_offline' else 'actual_cost')
    if mode=='fresh_live_cost':set_counts(rows[0],[9]*p['robots'])
    summary=replication_summary(rows,p,mode,original_rows=main)
    assert summary['case_count']==4 and summary['independent_sites_added']==0
    assert not summary['statistical_superiority_claimed'] and not summary['fresh_measured_deadline_claim']
    assert any(not r['packet_result_exactly_matches_original'] for r in summary['differences_from_main']) is (mode=='fresh_live_cost')
    with pytest.raises(ValueError):aggregate(rows,p,VALIDATION)

def test_raw_validation_and_nonpromotion_wording(p):
    rows=matrix(p);rows[0]['result']['control_misses']+=1
    with pytest.raises(ValueError):aggregate(rows,p,VALIDATION)
    text=markdown(aggregate(matrix(p),p,VALIDATION))
    assert 'exact_only' not in text and 'frozen_fleet_jepa' in text
    assert 'All three' in text and 'earlier unfiltered-recovery failures' in text
    assert 'Experimental benefit' in text and 'training-objective' in text
    assert 'qualified' not in text

@pytest.mark.parametrize('change',['old_costs','old_quantile','old_variants'])
def test_previous_registration_cannot_leak_into_new_study(p,change):
    if change=='old_costs':p['handover_seconds']=[.03,4.]
    if change=='old_quantile':p['primary_upper_quantile']=1-.05/6
    if change=='old_variants':p['variants_by_cost']['4.0'].append('exact_only')
    with pytest.raises(ValueError):aggregate([],p,VALIDATION,passed_costs=[])
    with pytest.raises(ValueError):arms(.03)


def test_previous_report_globals_are_not_monkeypatched(p):
    from wirelessjev.jepa_repair import report as previous
    assert previous.COSTS==(.03,4.)
    result=aggregate(matrix(p),p,VALIDATION)
    assert result['all_three_experimental_benefits']
    assert previous.COSTS==(.03,4.)
    assert len(previous.arms(.03))==6


def test_whole_site_pairing_and_conservative_pending_units(p):
    rows=matrix(p)
    for row in rows:
        if row['family']=='frozen_fleet_jepa' and row['controller_variant']=='repair':
            set_counts(row,[{21:1,22:2,23:3}[row['site']]]*p['robots'])
    result=aggregate(rows,p,VALIDATION)
    contrast=next(row for row in result['primary'] if row['contrast_id']=='repair_minus_legacy:frozen_fleet_jepa')
    assert contrast['paired_site_differences']==pytest.approx([-6/40,-5/40,-4/40])
    assert result['independent_sites']==3
    assert 'whole site' in result['bootstrap_unit']
    assert len(contrast['paired_site_differences'])!=len(rows)
