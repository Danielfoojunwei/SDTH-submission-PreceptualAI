"""Independent, read-only verification of the gate-repair evidence."""
from pathlib import Path
import hashlib,json
import numpy as np
from wirelessjev.gate_repair.gate import diagnostics

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/gate_repair'


def audit():
    manifest=json.loads((OUT/'manifest.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'summary.json').read_text());selection=json.loads((OUT/'selection.json').read_text())
    for name,expected in manifest['files_sha256'].items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==expected,name
    assert hashlib.sha256((ROOT/'artifacts/contracted/manifest.json').read_bytes()).hexdigest()==manifest['original_archive_manifest_sha256']
    for name,expected in manifest['checkpoint_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    for stage in ['fit','selection','test','analysis']:
        capsule=json.loads((OUT/(stage+'_source.json')).read_text())
        for name,expected in capsule['source_sha256'].items():assert hashlib.sha256((OUT/(stage+'_source')/name).read_bytes()).hexdigest()==expected
    stages=[set(protocol['fit_sites']+protocol['calibration_sites']),set(sum([protocol['selection_'+c+'_sites'] for c in ['main','stress','yard','small']],[])),set(sum([protocol['test_'+c+'_sites'] for c in ['main','stress','yard','small']],[]))]
    assert not(stages[0]&stages[1] or stages[0]&stages[2] or stages[1]&stages[2])
    assert selection['sealed_at_unix']<json.loads((OUT/'test_source.json').read_text())['time_unix'] and not selection['test_accessed']
    assert selection['gain_model_sha256']==hashlib.sha256((OUT/'gain_model.json').read_bytes()).hexdigest()
    cases=[json.loads(p.read_text()) for p in (OUT/'cases').glob('*.json')];assert len(cases)==40 and sum(len(c['rows']) for c in cases)==270
    assert len([c for c in cases if c['stage']=='test'])==24
    for stage,ids in zip(['fit','select','test'],stages):assert {c['scene'] for c in cases if c['stage']==stage}==ids
    for c in cases:
        stage='selection' if c['stage']=='select' else c['stage'];receipt=json.loads((OUT/(stage+'_source.json')).read_text())
        assert c['source_sha256']==receipt['source_sha256'] and c['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert len(c['rows'])=={'fit':2,'select':9,'test':7}[c['stage']]
        for r in c['rows']:
            z=r['result'];assert z['control_sent']==c['robots']*c['duration_s']*20 and z['control_miss']==z['control_misses']/z['control_sent']
            if r['trace']:
                with np.load(OUT/r['trace']) as t:d=diagnostics({k:t[k] for k in t.files})
                for key,value in d.items():
                    if value is None:assert r['gate_diagnostics'][key] is None
                    else:assert np.isclose(value,r['gate_diagnostics'][key],rtol=1e-10,atol=1e-10),key
        if c['stage']=='test':
            neural=[r for r in c['rows'] if r['kind'] not in ['fixed','bayesian']];assert len({r['forecast_sha256'] for r in neural})==1
            assert not c['truth_used_by_runtime_gate']
    for condition,group in summary['fresh_results'].items():
        cs=[c for c in cases if c['stage']=='test' and c['condition']==condition];assert len(cs)==6
        for kind,facts in group['arms'].items():
            rs=[next(r for r in c['rows'] if r['kind']==kind)['result'] for c in cs];miss=sum(r['control_misses'] for r in rs);sent=sum(r['control_sent'] for r in rs)
            assert facts['misses']==miss and facts['sent']==sent and np.isclose(facts['ppm'],miss/sent*1e6)
    historical=json.loads((OUT/'historical_diagnosis.json').read_text());assert historical['no_tuning_on_historical_replays']
    assert historical['selection_sha256_before_and_after']==hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest()
    for p in (OUT/'historical_diagnosis').glob('*.json'):
        r=json.loads(p.read_text());assert r['original_counters_match']
        old=json.loads((ROOT/f'artifacts/contracted/cases/main-{r["scene"]}.json').read_text());reference=next(x for x in old['rows'] if x['arm']=='jepa_gated_base' and x['mission']==protocol['mission'])
        for key in r['counter_fields_compared']:assert r['result'][key]==reference[key]
    assert len(list((OUT/'historical_diagnosis').glob('*.json')))==6
    return {'passed':True,'native_study_runs':270,'fresh_test_runs':168,'fresh_scenes':24,'historical_replays_matched':6,'frozen_members':5,'forecast_identity_verified':True,'original_archive_preserved':True,'simulation_only':True}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
