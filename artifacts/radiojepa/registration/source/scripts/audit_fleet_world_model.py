"""Independent read-only evidence checks for fleet world models."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/fleet_world_model'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    manifest=json.loads((OUT/'manifest.json').read_text());p=json.loads((OUT/'protocol.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    for name,h in manifest['files_sha256'].items():assert digest(OUT/name)==h,name
    for name,h in manifest['checkpoint_sha256'].items():assert digest(ROOT/name)==h,name
    for name,h in manifest['previous_archive_manifests'].items():assert digest(ROOT/name)==h,name
    for stage in ['training','selection','test','analysis']:
        receipt=json.loads((OUT/(stage+'_source.json')).read_text())
        for name,h in receipt['source_sha256'].items():assert digest(OUT/(stage+'_source')/name)==h,name
    training=json.loads((OUT/'training.json').read_text());assert training['completed'] and training['fits']==9 and not training['test_accessed']
    fits=[json.loads(f.read_text()) for f in (OUT/'fits').glob('*.json')]
    assert {(r['family'],r['seed']) for r in fits}=={(f,s) for f in p['families'] for s in p['training_seeds']}
    for r in fits:
        assert r['completed'] and len(r['history'])==p['epochs'] and not r['test_accessed']
        assert r['source_sha256']==json.loads((OUT/'training_source.json').read_text())['source_sha256']
        assert digest(OUT/'checkpoints'/f'{r["family"]}-{r["seed"]}.pt')==r['checkpoint_sha256']
    assert selection['sealed'] and not selection['test_accessed']
    assert training['time_unix']<selection['sealed_at_unix']<json.loads((OUT/'test_source.json').read_text())['time_unix']
    groups=[set(p['training_sites']),set(p['validation_sites']),
        set(sum([p['selection_'+c+'_sites'] for c in ['main','stress','yard','small','drift']],[])),
        set(sum([p['test_'+c+'_sites'] for c in ['main','stress','yard','small','drift']],[]))]
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not a&b
    best=min(p['families'],key=lambda f:selection['choices'][f]['score']);assert best==selection['development_best_model']==summary['development_best_model']
    for f,choice in selection['choices'].items():
        best_candidate=min(choice['candidates'],key=lambda r:r['score']);assert best_candidate['gate']==choice['gate']
    cases=[json.loads(f.read_text()) for f in (OUT/'cases').glob('*.json')]
    tests=[c for c in cases if c['stage']=='test'];dev=[c for c in cases if c['stage']=='selection']
    assert len(tests)==32 and len(dev)==12 and sum(len(c['rows']) for c in dev)==372
    assert {c['scene'] for c in tests}==groups[-1] and {c['scene'] for c in dev}==groups[-2]
    assert sum(len(c['rows']) for c in tests)==256
    seal=json.loads((OUT/'comparison_seal.json').read_text());refchoice=json.loads((OUT/'strong_reference/selection.json').read_text())
    assert seal['core_selection_sha256']==digest(OUT/'selection.json') and seal['reference_selection_sha256']==digest(OUT/'strong_reference/selection.json')
    assert seal['sealed_at_unix']<json.loads((OUT/'test_source.json').read_text())['time_unix']
    assert seal['sealed_at_unix']<json.loads((OUT/'strong_reference/test_source.json').read_text())['time_unix']
    assert seal['strongest_development_radiomap']==min(seal['reference_scores'],key=seal['reference_scores'].get)
    refcases=[json.loads(f.read_text()) for f in (OUT/'strong_reference/cases').glob('*.json')]
    assert sum(len(c['rows']) for c in refcases if c['stage']=='selection')==60
    assert sum(len(c['rows']) for c in refcases if c['stage']=='test')==40
    for stage in ['selection','test']:
        rec=json.loads((OUT/'strong_reference'/(stage+'_source.json')).read_text())
        for name,h in rec['source_sha256'].items():assert digest(OUT/'strong_reference'/(stage+'_source')/name)==h
    for c in refcases:
        assert c['source_sha256']==json.loads((OUT/'strong_reference'/(c['stage']+'_source.json')).read_text())['source_sha256']
        if c['stage']=='test':assert all(r['gate']==refchoice['gate'] for r in c['rows'])
    for c in cases:
        receipt=json.loads((OUT/(c['stage']+'_source.json')).read_text())
        assert c['source_sha256']==receipt['source_sha256'] and c['checkpoint_sha256']==selection['checkpoint_sha256']
        assert not c['runtime_inputs_have_truth'] and c['latest_observation_strictly_before_decision']
        for r in c['rows']:
            z=r['result'];assert z['control_sent']==c['robots']*c['duration_s']*20
            assert np.isclose(z['control_miss'],z['control_misses']/z['control_sent'],rtol=0,atol=1e-15)
            assert sum(z['miss_causes'].values())==z['control_misses']
            if c['stage']=='test' and r['family']!='fixed':assert r['gate']==selection['choices'][r['family']]['gate']
        if c['stage']=='selection':
            for family in selection['choices']:
                hashes={r['forecast_sha256'] for r in c['rows'] if r['family']==family};assert len(hashes)==1
    for condition,g in summary['conditions'].items():
        cs=[c for c in tests if c['condition']==condition]
        for c in cs:
            r=next(x for x in refcases if x['stage']=='test' and x['scene']==c['scene'])
            c['rows']+=r['rows']
        for family,a in g['arms'].items():
            rows=[next(r['result'] for r in c['rows'] if r['family']==family and r['mission']==p['primary_mission']) for c in cs]
            assert a['misses']==sum(r['control_misses'] for r in rows)
            assert a['commands']==sum(r['control_sent'] for r in rows)
            assert np.isclose(a['ppm'],a['misses']/a['commands']*1e6)
    return {'passed':True,'independent_fits':9,'selection_rollouts':432,'test_rollouts':296,'fresh_scenes':32,'prior_archives_preserved':True,'development_selection_frozen':True,'simulation_only':True}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
