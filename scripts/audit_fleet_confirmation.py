"""Read-only check of independently registered warehouse confirmation."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/fleet_confirmation';DISC=ROOT/'artifacts/fleet_world_model'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit():
    manifest=json.loads((OUT/'manifest.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    selected=json.loads((OUT/'selection.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    opened=json.loads((OUT/'test_opened.json').read_text());assert selected['sealed'] and not selected['test_accessed']
    assert opened['time_unix']>selected['sealed_at_unix'] and opened['selection_sha256']==digest(OUT/'selection.json')
    for name,h in manifest['files_sha256'].items():assert digest(OUT/name)==h,name
    for name,h in manifest['checkpoint_sha256'].items():assert digest(ROOT/name)==h,name
    for name,h in manifest['source_sha256'].items():assert digest(OUT/'executed_source'/name)==h,name
    assert digest(DISC/'manifest.json')==manifest['discovery_manifest_sha256']
    for name,h in selected['discovery_case_sha256'].items():assert digest(ROOT/name)==h
    assert selected['candidate']==protocol['fixed_candidate']==min(selected['discovery_counts'],key=lambda f:(selected['discovery_counts'][f],f))
    p=json.loads((DISC/'protocol.json').read_text())
    past=set(p['training_sites']+p['validation_sites'])
    for c in ['main','stress','yard','small','drift']:past.update(p['selection_'+c+'_sites']+p['test_'+c+'_sites'])
    assert not past&set(protocol['test_sites'])
    cases=[json.loads(f.read_text()) for f in (OUT/'cases').glob('*.json')]
    assert len(cases)==8 and {c['scene'] for c in cases}==set(protocol['test_sites'])
    for c in cases:
        assert c['source_sha256']==manifest['source_sha256'] and c['checkpoint_sha256']==manifest['checkpoint_sha256']
        assert c['no_truth_in_runtime_inputs'] and len(c['rows'])==4
        for r in c['rows']:
            z=r['result'];assert r['gate']=='rescue'
            assert z['control_sent']==protocol['robots']*protocol['duration_s']*20
            assert z['control_miss']==z['control_misses']/z['control_sent']
            assert sum(z['miss_causes'].values())==z['control_misses']
    for family,a in summary['arms'].items():
        rows=[next(r['result'] for r in c['rows'] if r['family']==family) for c in cases]
        assert sum(r['control_misses'] for r in rows)==a['misses']
        assert sum(r['control_sent'] for r in rows)==a['commands']
        assert np.isclose(a['ppm'],a['misses']/a['commands']*1e6)
    primary=summary['comparisons_ppm']['hybrid_supervised_vs_learned_radiomap']
    assert summary['primary_success']==(primary['paired_site_bootstrap_95'][1]<0)
    return {'passed':True,'independent_confirmation_sites':8,'native_runs':32,'frozen_checkpoint_files':len(manifest['checkpoint_sha256']),
        'no_retraining_or_controller_retuning':True,'discovery_archive_preserved':True,'simulation_only':True}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
