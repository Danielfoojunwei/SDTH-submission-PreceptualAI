"""Report and independently audit the fixed compact-model confirmation."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/frontier_confirmation';DISC=ROOT/'artifacts/world_frontier'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(values):
    d=np.asarray(values);rng=np.random.default_rng(310000);b=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
    return {'mean_difference_ppm':float(d.mean()),'paired_site_bootstrap_95_ppm':np.quantile(b,[.025,.975]).tolist(),'wins':int((d<0).sum()),'ties':int((d==0).sum()),'sites':len(d)}


def report():
    from wirelessjev.frontier_confirmation import source_hashes
    selected=json.loads((OUT/'selection.json').read_text());assert selected['source_sha256']==source_hashes()
    p=json.loads((OUT/'protocol.json').read_text());cases=[json.loads(f.read_text()) for f in sorted((OUT/'cases').glob('*.json'))];assert len(cases)==8
    groups={}
    for f in p['arms']:
        rows=[next(r for r in c['rows'] if r['family']==f) for c in cases];miss=sum(r['result']['control_misses'] for r in rows);sent=sum(r['result']['control_sent'] for r in rows)
        groups[f]={'misses':miss,'commands':sent,'ppm':miss/sent*1e6,'scene_ppm':[r['result']['control_miss']*1e6 for r in rows],
            'forecast_metrics':{k:float(np.mean([r['forecast_metrics'][k] for r in rows])) for k in rows[0]['forecast_metrics']},
            'traffic':{k:float(np.mean([r['result'][k] for r in rows])) for k in ['bulk_mbps','video_on_time','safety_stop_fraction']}}
    contrasts={f:paired(np.asarray(groups['distilled']['scene_ppm'])-groups[f]['scene_ppm']) for f in p['arms'] if f!='distilled'}
    primary=contrasts['radiomap'];latency=json.loads((DISC/'latency.json').read_text())['p95_seconds']['1']
    summary={'completed':True,'built_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'candidate':'distilled','fresh_sites':8,'native_runs':32,'arms':groups,
        'candidate_vs_reference':contrasts,'primary_success':primary['paired_site_bootstrap_95_ppm'][1]<0,'latency_seconds':latency,
        'warm_speed_ratio_vs_hybrid':latency['confirmed_hybrid']['warm']/latency['distilled']['warm'],'charged_compute_and_sensing':True,'simulation_only':True}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    names={'radiomap':'Optimised RadioMap','previous_jepa':'Previous JEPA','confirmed_hybrid':'Three-member hybrid','distilled':'Single distilled model'}
    a=groups['distilled'];b=groups['radiomap'];ci=primary['paired_site_bootstrap_95_ppm']
    lines=['# Compact-model independent confirmation','',
        f'With sensing and compute charged, the fixed distilled model produces **{a["ppm"]:.2f}** misses per million versus **{b["ppm"]:.2f}** for optimised RadioMap. The paired difference is **{primary["mean_difference_ppm"]:.2f} ppm**, 95% site-bootstrap interval **[{ci[0]:.2f}, {ci[1]:.2f}]**.',
        'The primary superiority criterion '+('passes.' if summary['primary_success'] else 'fails.'),'',
        'The candidate was fixed after the first eight primary warehouses and before these eight new sites. All weights, the planner and the measured service ledger were frozen. The original frontier selection and all its negative results remain unchanged.','',
        '| Method | Misses / million | Forecast RMSE (dB) | Route-change RMSE (dB) | Warm compute charge (ms) |','|---|---:|---:|---:|---:|']
    for f in p['arms']:
        m=groups[f]['forecast_metrics'];lines.append(f'| {names[f]} | {groups[f]["ppm"]:.2f} | {m["rmse_db"]:.2f} | {m["delta_rmse_db"]:.2f} | {1000*latency[f]["warm"]:.2f} |')
    lines+=['','Eight fresh warehouses, 100 robots, 120 seconds each; 1,920,000 commands per arm. Scans occur at the declared 1 Hz cadence with 30 ms outage; 30 ms handovers and a single-worker measured-service FIFO are charged.','',
        f'The single model has a {summary["warm_speed_ratio_vs_hybrid"]:.2f}× smaller warm CPU budget than the frozen three-member hybrid. This ratio is from the fixed microbenchmark, not a physical robot measurement. Secondary paired comparisons are in `summary.json`; a RadioMap win is not automatically a win over the older neural references.','',
        '## Remaining limits','',
        'This remains the same synthetic canonical warehouse family, with idealised scanner phases and attached-link measurements, precomputed motion and mean-field supervision. Timing charges are measured CPU service profiles; scan/handover durations are assumptions. Arbitrary CAD, real radio hardware, full native counterfactual packet training and end-to-end fleet timing remain open.','',
        'Reproduce after the frontier study with `uv run --frozen python -m wirelessjev.frontier_confirmation --workers 4`, then `uv run --frozen python scripts/report_frontier_confirmation.py`. Audit with the latter script and `--audit`.']
    (ROOT/'docs/FRONTIER_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(4)
    colors=['#3e79ad','#d3aa73','#9972b0','#16a6ae']
    axes[0].bar(x,[groups[f]['ppm'] for f in p['arms']],color=colors)
    for c in cases:axes[0].scatter(x,[next(r['result']['control_miss']*1e6 for r in c['rows'] if r['family']==f) for f in p['arms']],s=13,color='#173243',alpha=.6)
    axes[0].set_xticks(x,[names[f] for f in p['arms']],rotation=18,ha='right');axes[0].set_ylabel('Control misses / million');axes[0].set_title('Eight independent warehouses · costs charged')
    axes[1].bar(x,[1000*latency[f]['warm'] for f in p['arms']],color=colors);axes[1].set_xticks(x,[names[f] for f in p['arms']],rotation=18,ha='right');axes[1].set_ylabel('Warm compute charge (ms / 100 robots)');axes[1].set_title('Fixed one-thread CPU service profile')
    fig.tight_layout();fig.savefig(figdir/'confirmation.png',dpi=180);fig.savefig(figdir/'confirmation.pdf');plt.close(fig)
    source=Path(__file__).resolve();target=OUT/'analysis_source'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    manifest={'completed':True,'files_sha256':{str(f.relative_to(OUT)):digest(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='manifest.json'},
        'checkpoint_sha256':selected['checkpoint_sha256'],'frontier_manifest_sha256':digest(DISC/'manifest.json'),'preregistration_commit':'53d5535','simulation_only':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary,indent=2))


def audit():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text());s=json.loads((OUT/'selection.json').read_text());opened=json.loads((OUT/'test_opened.json').read_text())
    assert s['sealed'] and not s['test_accessed'] and s['sealed_at_unix']<opened['time_unix'] and opened['selection_sha256']==digest(OUT/'selection.json')
    for name,h in m['files_sha256'].items():assert digest(OUT/name)==h,name
    for name,h in m['checkpoint_sha256'].items():assert digest(ROOT/name)==h,name
    for name,h in s['source_sha256'].items():assert digest(OUT/'executed_source'/name)==h,name
    for name,h in s['discovery_sha256'].items():assert digest(ROOT/name)==h,name
    assert digest(DISC/'manifest.json')==m['frontier_manifest_sha256'] and digest(DISC/'latency.json')==s['latency_sha256']
    assert s['candidate']==min(s['discovery_counts'],key=lambda f:(s['discovery_counts'][f],f))=='distilled'
    old=json.loads((DISC/'protocol.json').read_text());used=set(old['training_sites']+old['validation_sites'])
    for k,v in old.items():
        if k.startswith(('selection_','test_')) and k.endswith('_sites'):used.update(v)
    assert not used&set(p['test_sites'])
    cases=[json.loads(f.read_text()) for f in (OUT/'cases').glob('*.json')];assert len(cases)==8 and {c['scene'] for c in cases}==set(p['test_sites'])
    for c in cases:
        assert c['source_sha256']==s['source_sha256'] and c['checkpoint_sha256']==s['checkpoint_sha256'] and c['no_truth_in_runtime_inputs']
        assert len(c['rows'])==4 and c['latency_sha256']==s['latency_sha256']
        for r in c['rows']:
            z=r['result'];assert z['control_sent']==100*120*20 and z['control_miss']==z['control_misses']/z['control_sent'] and sum(z['miss_causes'].values())==z['control_misses']
    summary=json.loads((OUT/'summary.json').read_text())
    for family,a in summary['arms'].items():
        rs=[next(r['result'] for r in c['rows'] if r['family']==family) for c in cases]
        assert a['misses']==sum(r['control_misses'] for r in rs) and a['commands']==sum(r['control_sent'] for r in rs)
        assert np.isclose(a['ppm'],a['misses']/a['commands']*1e6)
    assert summary['primary_success']==(summary['candidate_vs_reference']['radiomap']['paired_site_bootstrap_95_ppm'][1]<0)
    return {'passed':True,'independent_sites':8,'native_runs':32,'fixed_models_and_timing':True,'original_frontier_preserved':True,'simulation_only':True}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:print(json.dumps(audit(),indent=2))
    else:report()
