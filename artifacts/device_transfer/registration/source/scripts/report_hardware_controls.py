"""Report and audit the slow-hardware policy and admission experiments."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LABEL={'radiomap':'Optimised RadioMap','previous_jepa':'Previous JEPA','confirmed_hybrid':'Three-member hybrid','distilled':'Distilled model'}


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(v):
    d=np.asarray(v);rng=np.random.default_rng(316000);b=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
    return {'mean_difference_ppm':float(d.mean()),'paired_site_bootstrap_95_ppm':np.quantile(b,[.025,.975]).tolist(),'wins':int((d<0).sum()),'sites':len(d)}


def report(stage):
    out=ROOT/'artifacts'/stage;p=json.loads((out/'protocol.json').read_text());complete=json.loads((out/'completion.json').read_text());assert complete['completed']
    files=sorted((out/'cases').glob('test-*.json' if stage=='hardware_budget' else '*.json'));cases=[json.loads(f.read_text()) for f in files]
    variants=['default','selected'] if stage=='hardware_budget' else ['planned_only','all_handovers']
    selected=json.loads((out/'selection.json').read_text()) if stage=='hardware_budget' else None
    methods=p['methods'];arms={}
    for f in methods:
        arms[f]={}
        for variant in variants:
            rs=[]
            for c in cases:
                if selected:
                    cadence=1 if variant=='default' else selected['choices'][f]['scan_period'];gate='rescue' if variant=='default' else selected['choices'][f]['gate']
                    r=next(r for r in c['rows'] if r['family']==f and r['scan_period']==cadence and r['gate']==gate)
                else:r=next(r for r in c['rows'] if r['family']==f and r['variant']==variant)
                rs.append(r['result'])
            miss=sum(r['control_misses'] for r in rs);sent=sum(r['control_sent'] for r in rs)
            arms[f][variant]={'misses':miss,'commands':sent,'ppm':miss/sent*1e6,'percent':100*miss/sent,'scene_ppm':[r['control_miss']*1e6 for r in rs],
                'bulk_mbps':float(np.mean([r['bulk_mbps'] for r in rs])),'video_on_time':float(np.mean([r['video_on_time'] for r in rs])),
                'mean_roams':float(np.mean([r['roams'] for r in rs])),'safety_stop_fraction':float(np.mean([r['safety_stop_fraction'] for r in rs]))}
    deltas={f:paired(np.asarray(arms[f][variants[1]]['scene_ppm'])-arms[f][variants[0]]['scene_ppm']) for f in methods}
    primary=paired(np.asarray(arms['distilled'][variants[1]]['scene_ppm'])-arms['radiomap'][variants[1]]['scene_ppm']) if selected else deltas['distilled']
    summary={'completed':True,'study':stage,'fresh_sites':len(cases),'native_runs':complete.get('native_runs',complete.get('development_runs',0)+complete.get('test_runs',0)),
        'variants':variants,'arms':arms,'within_method_comparisons':deltas,'primary_comparison':primary,'primary_success':primary['paired_site_bootstrap_95_ppm'][1]<0,'simulation_only':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    title='Slow-hardware sensing and handover budget' if selected else 'Admission applies to both planned and reactive handovers'
    lines=['# '+title,'',
        '100 simulated robots per site, 60 seconds per run, four-second handovers, 90 ms scan outages and an additional 500 ms compute delay. These are assumed hardware timings, not device measurements.','',
        ('Four development sites selected cadence and gate independently for every reference. Four additional sites test the frozen configurations.' if selected else 'Six fresh sites compare the same fixed physical-outage rule on planned proposals alone versus both planned and reactive proposals. No model or threshold was fitted.'),'',
        '| Forecaster | '+variants[0]+' loss (%) | '+variants[1]+' loss (%) | Selected/all bulk Mbps | Selected/all handovers per run |','|---|---:|---:|---:|---:|']
    for f in methods:
        a,b=[arms[f][v] for v in variants];lines.append(f'| {LABEL[f]} | {a["percent"]:.3f} | {b["percent"]:.3f} | {b["bulk_mbps"]:.2f} | {b["mean_roams"]:.1f} |')
    ci=primary['paired_site_bootstrap_95_ppm']
    lines+=['',f'Primary paired difference: {primary["mean_difference_ppm"]:.2f} ppm, nominal 95% site-bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}]. '+('The interval excludes zero.' if summary['primary_success'] else 'The interval spans zero; superiority is not established.'),'',
        '## Mechanism and trade-offs','',
        'The physical-outage budget estimates failure area over thirty seconds with the archived ns-3/Nakagami surrogate. It admits a planned switch only when the estimated loss from staying exceeds four seconds of certain outage plus candidate loss afterward. This is an approximate admission heuristic, not a risk certificate.','',
        ('The selected experiment keeps the original reactive fallback. Consequently, a rejected planned switch can still be followed by a reactive handover. The separate all-handover experiment tests that interface distinction.' if selected else 'The Boolean confidence gate chooses whether the planner or reactive rule proposes a move. The new admission hook separately filters proposals from both paths. It can also withhold a necessary emergency switch when the forecast is wrong, so live risk monitoring and hardware qualification remain required.'),'',
        'Sensing cadence changes both data availability and scan outage. The forecasters see masked real scan streams at the selected cadence, not dense scans at a sparse-scan cost. The FIFO uses the already frozen cadence-specific processing profile.','',
        '## Limits','',
        'Synthetic canonical warehouse, known CAD/localisation/noise, idealised scanner phases and attached-link measurements, fixed robot trajectories and mean-field training. Scan-report transport availability is not coupled to each policy\'s radio outages. The expected-failure surrogate omits full MAC/queue interactions; actual packet outcomes are reported separately. High remaining loss is a failed deployment condition, not a successful physical-car trial.','',
        'Compare percentages only within this cohort. The earlier 30 ms and slow-hardware cohorts use different seeds. A lower miss rate can cost throughput; bulk, video, handover counts and stop time are all archived in `summary.json`.']
    name='HARDWARE_BUDGET_RESULTS.md' if selected else 'HANDOVER_ADMISSION_RESULTS.md';(ROOT/'docs'/name).write_text('\n'.join(lines)+'\n')
    figdir=out/'figures';figdir.mkdir(exist_ok=True);fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(len(methods))
    for i,v in enumerate(variants):ax.bar(x+(i-.5)*.36,[arms[f][v]['percent'] for f in methods],width=.36,label=v,color=['#a6b9ce','#12a28e'][i])
    ax.set_xticks(x,[LABEL[f] for f in methods],rotation=10,ha='right');ax.set_ylabel('Missed control commands (%)');ax.set_title(title);ax.legend();fig.tight_layout()
    fig.savefig(figdir/'controls.png',dpi=180);fig.savefig(figdir/'controls.pdf');plt.close(fig)
    source=Path(__file__).resolve();target=out/'analysis_source'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target)
    checkpoints=cases[0]['checkpoint_sha256'];previous={str(q.relative_to(ROOT)):digest(q) for q in [ROOT/'artifacts/world_frontier/manifest.json',ROOT/'artifacts/frontier_confirmation/manifest.json']}
    manifest={'completed':True,'files_sha256':{str(q.relative_to(out)):digest(q) for q in sorted(out.rglob('*')) if q.is_file() and q.name!='manifest.json'},
        'checkpoint_sha256':checkpoints,'previous_manifests':previous,'simulation_only':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(summary,indent=2))


def audit(stage):
    out=ROOT/'artifacts'/stage;p=json.loads((out/'protocol.json').read_text());m=json.loads((out/'manifest.json').read_text())
    for name,h in m['files_sha256'].items():assert digest(out/name)==h,name
    for name,h in m['checkpoint_sha256'].items():assert digest(ROOT/name)==h,name
    for name,h in m['previous_manifests'].items():assert digest(ROOT/name)==h,name
    receipts=['development','test'] if stage=='hardware_budget' else ['test']
    source_receipts={}
    for kind in receipts:
        receipt=json.loads((out/(kind+'_source.json')).read_text());root=out/(kind+'_source') if stage=='hardware_budget' else out/'executed_source'
        source_receipts[kind]=receipt
        for name,h in receipt['source_sha256'].items():assert digest(root/name)==h,name
    if stage=='hardware_budget':
        selected=json.loads((out/'selection.json').read_text());assert selected['sealed'] and not selected['test_accessed']
        assert selected['sealed_at_unix']<json.loads((out/'test_source.json').read_text())['time_unix']
        for f,c in selected['choices'].items():
            expected=min(c['candidates'],key=lambda r:(r['misses'],-r['mean_bulk_mbps']));assert expected['gate']==c['gate'] and expected['scan_period']==c['scan_period']
    cases=[json.loads(q.read_text()) for q in (out/'cases').glob('*.json')];fresh=[c for c in cases if c.get('stage','test')=='test']
    assert {c['scene'] for c in fresh}==set(p['test_sites'])
    if stage=='hardware_budget':assert not set(p['development_sites'])&set(p['test_sites'])
    for c in cases:
        assert c.get('no_truth_in_runtime_inputs',c.get('no_truth_in_admission_inputs',False))
        receipt=source_receipts[c.get('stage','test')]
        assert c['source_sha256']==receipt['source_sha256'] and c['checkpoint_sha256']==m['checkpoint_sha256']
        assert c['latency_sha256']==receipt['latency_sha256']
        for r in c['rows']:
            z=r['result'];assert z['control_sent']==100*60*20 and z['control_miss']==z['control_misses']/z['control_sent'] and sum(z['miss_causes'].values())==z['control_misses']
    complete=json.loads((out/'completion.json').read_text());count=complete.get('native_runs',complete.get('development_runs',0)+complete.get('test_runs',0))
    assert sum(len(c['rows']) for c in cases)==count
    return {'passed':True,'study':stage,'fresh_sites':len(fresh),'native_runs':count,'no_model_retuning':True,'simulation_only':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',choices=['hardware_budget','handover_admission']);p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit:print(json.dumps(audit(a.study),indent=2))
    else:report(a.study)
