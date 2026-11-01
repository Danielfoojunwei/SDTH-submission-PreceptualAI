"""Read-only independent audit of the portable world-model campaign archive."""
from pathlib import Path
import hashlib,json
import torch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/contracted'


def audit():
    manifest=json.loads((OUT/'manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
    for relative,digest in manifest['files_sha256'].items():assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==digest,relative
    for stage in ['training','evaluation']:
        capsule=json.loads((OUT/(stage+'_source.json')).read_text())
        for relative,digest in capsule['source_sha256'].items():assert hashlib.sha256((OUT/(stage+'_source')/relative).read_bytes()).hexdigest()==digest,(stage,relative)
    training=set(protocol['training_sites']);development=set(protocol['development_sites']);testing=set().union(*(set(protocol[k]) for k in ['primary_sites','small_sites','stress_sites','yard_sites']))
    assert not (training&development or training&testing or development&testing)
    fits=list((OUT/'fits').glob('*.json'));assert len(fits)==20
    assert {(r['family'],r['seed']) for r in map(lambda p:json.loads(p.read_text()),fits)}=={(f,s) for f in protocol['families'] for s in range(5)}
    for file in fits:
        fit=json.loads(file.read_text());assert fit['completed'] and fit['epochs']==12 and not fit['test_accessed']
        path=OUT/'checkpoints'/f'{fit["family"]}-{fit["seed"]}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==fit['checkpoint_sha256']
        state=torch.load(path,weights_only=True,map_location='cpu')['state'];assert all(torch.isfinite(t).all() for t in state.values())
    cases=[json.loads(p.read_text()) for p in (OUT/'cases').glob('*.json')];assert len(cases)==24 and {c['scene'] for c in cases}==testing
    assert sum(len(c['rows']) for c in cases)==366
    primary=[c for c in cases if c['condition']=='main']
    for case in cases:
        assert case['completed'] and len(case['rows'])==(34 if case['condition']=='main' else 9)
        for row in case['rows']:
            assert row['control_sent']==case['robots']*case['duration_s']*20
            assert row['control_miss']==row['control_misses']/row['control_sent']
    for arm,facts in summary['primary'].items():
        rows=[next(r for r in c['rows'] if r['arm']==arm and r['mission']=='bulk_uploads') for c in primary]
        misses=sum(r['control_misses'] for r in rows);sent=sum(r['control_sent'] for r in rows)
        assert facts['misses']==misses and facts['commands']==sent and abs(facts['ppm']-misses/sent*1e6)<1e-10
    for c in primary:
        metrics={r['arm']:r for r in c['forecast_metrics'] if r['mission']=='bulk_uploads'};base=metrics['jepa_gated_base']
        for kind in ['half','double','noise','shuffle']:
            changed=metrics['jepa_gated_'+kind];assert base['mean_forecast_sha256']==changed['mean_forecast_sha256']
            assert base['rmse_near_db']==changed['rmse_near_db']
        for kind in ['half','double']:assert abs(base['ap_score_error_sigma_spearman']-metrics['jepa_gated_'+kind]['ap_score_error_sigma_spearman'])<1e-10
        assert metrics['jepa_gated_half']['coverage_90']<=base['coverage_90']<=metrics['jepa_gated_double']['coverage_90']
    return {'passed':True,'finite_fits':20,'fresh_scenes':24,'native_rollouts':366,'source_and_output_hashes_verified':True,'uncertainty_intervention_forecast_identity_verified':True,'packet_counts_recomputed':True,'simulation_only':True}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
