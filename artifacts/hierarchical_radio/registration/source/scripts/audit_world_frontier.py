"""Independent read-only audit of the source-sealed optimisation frontier."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/world_frontier'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    manifest=json.loads((OUT/'manifest.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text());training=json.loads((OUT/'training.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    for name,h in manifest['files_sha256'].items():assert digest(OUT/name)==h,name
    for name,h in manifest['checkpoint_sha256'].items():assert digest(ROOT/name)==h,name
    for name,h in manifest['previous_manifests'].items():assert digest(ROOT/name)==h,name
    for stage in ['training','selection','test']:
        receipt=json.loads((OUT/(stage+'_source.json')).read_text())
        for name,h in receipt['source_sha256'].items():assert digest(OUT/(stage+'_source')/name)==h,name
    latency=json.loads((OUT/'latency.json').read_text());base=json.loads((OUT/'latency_base.json').read_text())
    assert latency['tensor_packing_accounted'] and latency['base_profile_sha256']==digest(OUT/'latency_base.json')
    assert latency['calibration_source_sha256']==digest(OUT/'timing_calibration_source/calibrate_frontier_timing.py')
    for cadence,methods in latency['p95_seconds'].items():
        for family,costs in methods.items():
            for kind,value in costs.items():
                overhead=latency['tensor_packing_p95_seconds'][cadence] if family in protocol['families'] else 0
                assert np.isclose(value,base['p95_seconds'][cadence][family][kind]+overhead)
    assert training['fits']==9 and not training['test_accessed'] and selected['sealed'] and not selected['test_accessed']
    fits=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')]
    assert len(fits)==9
    for family in protocol['families']:
        fs=[f for f in fits if f['family']==family];assert {f['seed'] for f in fs}==set(protocol['training_seeds'])
        assert training['single_members'][family]==min(fs,key=lambda r:(r['best_validation_score'],r['seed']))['seed']
    for fit in fits:assert fit['completed'] and len(fit['history'])==protocol['epochs'] and not fit['test_accessed']
    assert selected['selected_family']==min(protocol['families'],key=lambda f:selected['choices'][f]['score'])
    assert latency['calibration_time_unix']<selected['sealed_at_unix']<json.loads((OUT/'test_source.json').read_text())['time_unix']
    assert selected['latency_sha256']==digest(OUT/'latency.json')
    splits=[set(protocol['training_sites']),set(protocol['validation_sites']),set(sum([protocol['selection_'+c+'_sites'] for c in ['main','drift','sparse']],[])),set(sum([protocol['test_'+c+'_sites'] for c in ['main','drift','sparse','slow']],[]))]
    for i,a in enumerate(splits):
        for b in splits[i+1:]:assert not a&b
    cases=[json.loads(p.read_text()) for p in (OUT/'cases').glob('*.json')];tests=[c for c in cases if c['stage']=='test'];dev=[c for c in cases if c['stage']=='selection']
    assert len(dev)==8 and len(tests)==24
    assert sum(len(c['rows']) for c in dev)==96 and sum(len(c['rows']) for c in tests)==160
    for c in cases:
        assert c['no_truth_in_forecast_inputs'] and c['latency_sha256']==selected['latency_sha256']
        assert c['source_sha256']==json.loads((OUT/(c['stage']+'_source.json')).read_text())['source_sha256']
        assert c['checkpoint_sha256']==selected['checkpoint_sha256']
        for r in c['rows']:
            z=r['result'];assert z['control_sent']==100*c['duration_s']*20
            assert z['control_miss']==z['control_misses']/z['control_sent'] and sum(z['miss_causes'].values())==z['control_misses']
            if c['stage']=='test':assert r['planner']==selected['choices'][r['family']]['planner']
    for condition,group in summary['conditions'].items():
        cs=[c for c in tests if c['condition']==condition]
        for f,a in group['arms'].items():
            rs=[next(r['result'] for r in c['rows'] if r['family']==f and r['timing_cost']) for c in cs]
            assert a['misses']==sum(r['control_misses'] for r in rs) and a['commands']==sum(r['control_sent'] for r in rs)
    return {'passed':True,'independent_fits':9,'selection_runs':96,'test_runs':160,'fresh_sites':24,'charged_inference_and_scan_costs':True,'old_archives_preserved':True,'simulation_only':True}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
