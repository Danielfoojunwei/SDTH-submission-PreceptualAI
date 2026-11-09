"""Software export fixtures; no synthetic figures enter the real Studio export."""
from copy import deepcopy
import hashlib
import json

import pytest

from wirelessjev.research_export import bundle as b


def put(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value))
    return b.file_sha(path)


def test_seal_requires_current_source_and_evidence_bytes_and_bound_report(tmp_path):
    source=put(tmp_path,'src/source.py','source')
    report=put(tmp_path,'artifacts/report.json',{'actual_metric':7})
    put(tmp_path,'artifacts/seal.json',{'source_sha256':{'src/source.py':source},
        'evidence_sha256':{'artifacts/report.json':report}})
    e=b.Evidence(tmp_path);seal,proof=e.seal('artifacts/seal.json')
    assert proof['source_files']==1 and proof['evidence_files']==1
    assert e.report('artifacts/report.json',seal)=={'actual_metric':7}
    put(tmp_path,'artifacts/unsealed.json',{'metric':999})
    with pytest.raises(ValueError,match='not bound'):e.report('artifacts/unsealed.json',seal)
    put(tmp_path,'src/source.py','changed')
    with pytest.raises(ValueError,match='changed'):b.Evidence(tmp_path).seal('artifacts/seal.json')


def test_mutation_after_verification_is_not_hidden_by_digest_cache(tmp_path):
    put(tmp_path,'evidence.json',{'status':'before'})
    e=b.Evidence(tmp_path);e.load('evidence.json')
    put(tmp_path,'evidence.json',{'status':'after'})
    with pytest.raises(ValueError,match='changed'):e.load('evidence.json')
    with pytest.raises(ValueError,match='changed'):e.finish()


def test_pending_adaptive_never_reads_or_exports_unsealed_test_metrics(tmp_path):
    put(tmp_path,'artifacts/adaptive_radio/report.json',{'test_rmse':999999})
    e=b.Evidence(tmp_path);result=b.adaptive(e)
    assert result['status']=='pending_seal' and 'metrics' not in result
    assert not e.sources  # Neither report nor selection is inspected without a final seal.


def test_pending_e7_planning_never_exports_partial_utility_results(tmp_path):
    put(tmp_path,'artifacts/e7_calplanning/report.json',{'summary':{'15':{'mean_environment_utility_delta':999999}}})
    e=b.Evidence(tmp_path);result=b.e7_calplanning(e);b.view_fields('e7_calplanning',result)
    assert result['status']=='pending_seal' and 'metrics' not in result and not e.sources
    assert 'not chronological minutes' in result['units']['nominal_budget']


def test_e7_partial_source_map_cannot_activate_planning_metrics(tmp_path):
    source=put(tmp_path,'src/source.py','one source is not the complete study')
    report=put(tmp_path,'artifacts/e7_calplanning/report.json',{'partial':'not publishable'})
    put(tmp_path,'artifacts/e7_calplanning/verification/manifest.json',{
        'source_sha256':{'src/source.py':source},'evidence_sha256':{'artifacts/e7_calplanning/report.json':report}})
    with pytest.raises(ValueError,match='525-source'):b.e7_calplanning(b.Evidence(tmp_path))


def test_e7_complete_source_closure_preserves_all_budget_harms_and_oracle_scope(tmp_path):
    protocol={'budget_semantics':'nominal nested tail fractions, not chronological minutes',
        'shared_ORACLE_metadata':['oracle infrastructure','truth-derived legacy start']}
    sources={'artifacts/e7_calplanning/protocol.json':put(tmp_path,'artifacts/e7_calplanning/protocol.json',protocol)}
    for index in range(524):
        name=f'src/unit_source_{index}.py';sources[name]=put(tmp_path,name,f'unit source{index}')
    summary={str(budget):{'environment_units':15,'mission_pairs':75,'mean_environment_utility_delta':-.125,
        'environment_harms':9,'mission_pair_harms':40,'per_environment_delta':[-.125]*15} for budget in (1,5,15)}
    report={'scope':'ORACLE-CONDITIONED DESCRIPTIVE unit fixture','production_qualified':False,'default_changed':False,
        'original_SERVICE_holdouts_opened':False,'environment_units':15,'paired_rows':225,'summary':summary,
        'map_fits':90,'belief_searches':450,'reference_searches':75}
    name='artifacts/e7_calplanning/report.json';digest=put(tmp_path,name,report)
    put(tmp_path,'artifacts/e7_calplanning/verification/manifest.json',{'scope':report['scope'],
        'source_sha256':sources,'evidence_sha256':{name:digest}})
    result=b.e7_calplanning(b.Evidence(tmp_path));b.view_fields('e7_calplanning',result)
    assert result['status']=='sealed' and result['proof']['source_files']==525
    assert result['metrics']['summary']==summary and not result['production_qualified']
    assert result['scope']==report['scope'] and 'tail' in result['metrics']['budget_semantics']


def test_completed_adaptive_report_must_be_in_verified_seal(tmp_path):
    source=put(tmp_path,'src/source.py','registered')
    report=put(tmp_path,'artifacts/adaptive_radio/report.json',{'selected_cell':'regression','test_error':3.})
    put(tmp_path,'artifacts/adaptive_radio/verification/manifest.json',
        {'source_sha256':{'src/source.py':source},'evidence_sha256':{'artifacts/adaptive_radio/report.json':report}})
    result=b.adaptive(b.Evidence(tmp_path))
    assert result['status']=='sealed' and result['metrics']['selected_cell']=='regression'


def test_whole_suite_count_comes_from_bound_log_and_keeps_scope(tmp_path):
    log='some progress\n2547 passed, 1 warning in 256.86s (0:04:16)\n'
    digest=put(tmp_path,'artifacts/research_review/whole_suite.log',log)
    source=put(tmp_path,'src/source.py','fixed')
    proof={'exit_code':0,'unchanged':True,'log_sha256':digest,'duration_seconds':259.6,
        'source_before_sha256':{'src/source.py':source},'source_after_sha256':{'src/source.py':source},
        'test_files':['tests/test_one.py'],'scope':'Tracked prior modules only'}
    put(tmp_path,'artifacts/research_review/whole_suite.json',proof)
    result=b.suite(b.Evidence(tmp_path))
    assert result['passed']==2547 and result['pytest_seconds']==256.86
    assert result['whole_process_seconds']==259.6 and not result['includes_this_new_exporter']
    put(tmp_path,'artifacts/research_review/whole_suite.log',log+'1 passed, 0 warnings in 1.0s\n')
    with pytest.raises(ValueError,match='changed'):b.suite(b.Evidence(tmp_path))


def test_evidence_paths_cannot_escape_or_follow_symlinks(tmp_path):
    e=b.Evidence(tmp_path)
    with pytest.raises(ValueError,match='Unsafe'):e.path('../outside.json')
    put(tmp_path,'real.json',{})
    (tmp_path/'alias.json').symlink_to(tmp_path/'real.json')
    with pytest.raises(ValueError,match='Redirected'):e.path('alias.json')


def test_augmented_bundle_preserves_every_legacy_value_without_calling_old_generator(tmp_path,monkeypatch):
    legacy={'generated_at':'old actual time','layout':{'actual':[1,2]},'evidence_class':'legacy simulated'}
    put(tmp_path,'studio/data/bundle.json',legacy)
    names=('measured_power','spectral','phones','csi','native','operator_profile','context','vision')
    for name in names:
        monkeypatch.setattr(b,name,lambda e:b.track('fixture software','fixture','verified',scope='unit test only',units={}))
    monkeypatch.setattr(b,'suite',lambda e:{'passed':0,'test_modules':0,'warnings':0})
    monkeypatch.setattr(b,'software_capabilities',lambda e:{})
    monkeypatch.setattr(b,'original_runs',lambda e:{'status':'software fixture'})
    monkeypatch.setattr(b,'view_fields',lambda key,value:value.update(headline='Software fixture',findings=['unit only']))
    result,manifest=b.build_bundle(tmp_path,captured_at='fixed test time')
    assert {k:v for k,v in result.items() if k!='research_evidence'}==legacy
    assert result['research_evidence']['tracks']['adaptive_rf']['status']=='pending_seal'
    assert result['research_evidence']['tracks']['e7_calplanning']['status']=='pending_seal'
    assert not result['research_evidence']['remote_publication_performed']
    assert json.loads((tmp_path/'studio/data/bundle.json').read_text())==legacy
    assert manifest['scientific_actions']=={'fits':0,'model_forwards':0,'simulations':0}


def test_standalone_html_embeds_data_safely_and_uses_no_remote_assets():
    payload={'research_evidence':{'untrusted':'</script><script>alert(1)</script> & <img src=x>'}}
    html=b.review_html(payload)
    embedded=html.split('<script id="evidence" type="application/json">',1)[1].split('</script>',1)[0]
    assert '</script>' not in embedded and json.loads(embedded)==payload
    assert 'src="https://' not in html and 'href="https://' not in html
    assert 'textContent' in html and 'innerHTML' not in html


def test_missing_required_external_cache_is_fatal(tmp_path):
    e=b.Evidence(tmp_path,cache_root=tmp_path/'cache')
    with pytest.raises(FileNotFoundError):e.digest('predictions/missing.npz',expected='a'*64,external=True)


def test_concise_headlines_use_actual_metrics_and_keep_negative_findings_visible():
    power={'metrics':{'factory_comparison':{'best_reported_neural_rmse_db':7.25,'constant_rmse_db':2.5}}}
    b.view_fields('measured_power',power)
    assert '7.250 vs 2.500 dB' in power['headline'] and 'loses' in power['findings'][0]
    csi={'metrics':{'summary':{'dichasus-dc01':{'joint':{'seed_mean_packet_nmse':1.75},'linear':{'mean_packet_nmse':.1234}}}}}
    b.view_fields('complex_csi',csi)
    assert '1.750 vs 0.1234 NMSE' in csi['headline']
    assert len(power['findings'])==len(csi['findings'])==2
    ops={'metrics':{'ratio_of_median_call_times':2.2,'reference_median_ms':22.,'reuse_median_ms':10.}}
    b.view_fields('geometry_ops',ops)
    assert ops['headline']=='2.200× faster operand construction'


def test_e7_headline_retains_higher_better_utility_and_harms_without_promotion():
    row={'mean_environment_utility_delta':-.125,'environment_harms':9,'mission_pair_harms':40}
    value={'status':'sealed','metrics':{'summary':{'15':row}}}
    b.view_fields('e7_calplanning',value)
    assert '−' not in value['headline'] and '-0.1250' in value['headline']
    assert '9/15' in value['findings'][0] and '40/75' in value['findings'][0]
    assert 'not packet-service qualification' in value['findings'][1]
