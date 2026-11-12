"""Report every frontier result; never choose a new model from fresh outcomes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wirelessjev.frontier.study import ROOT,OUT,P,METHODS,sources,digest,checkpoint_hashes
from wirelessjev.contracted.train import write

LABEL={'radiomap':'Optimised RadioMap','previous_jepa':'Previous JEPA','confirmed_hybrid':'Confirmed hybrid','distilled':'Distilled student','packet':'Route/reliability student','temporal':'Temporal + innovation'}


def interval(delta):
    d=np.asarray(delta);rng=np.random.default_rng(308000);b=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
    return {'mean_difference_ppm':float(d.mean()),'paired_site_bootstrap_95_ppm':np.quantile(b,[.025,.975]).tolist(),'wins':int((d<0).sum()),'ties':int((d==0).sum()),'sites':len(d)}


def row(c,f,cost=True):return next(r for r in c['rows'] if r['family']==f and r['timing_cost']==cost)


def main():
    completion=json.loads((OUT/'completion.json').read_text());assert completion['completed']
    selected=json.loads((OUT/'selection.json').read_text());assert selected['source_sha256']==sources() and selected['checkpoint_sha256']==checkpoint_hashes()
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('test-*.json'))];assert len(cases)==24
    groups={}
    for condition in ['main','drift','sparse','slow']:
        cs=[c for c in cases if c['condition']==condition];arms={}
        for f in METHODS:
            rs=[row(c,f)['result'] for c in cs];miss=sum(r['control_misses'] for r in rs);sent=sum(r['control_sent'] for r in rs)
            arms[f]={'misses':miss,'commands':sent,'ppm':miss/sent*1e6,'scene_ppm':[r['control_miss']*1e6 for r in rs],
                'forecast_metrics':{k:float(np.mean([c['forecast_metrics'][f][k] for c in cs])) for k in cs[0]['forecast_metrics'][f]},
                'traffic':{k:float(np.mean([r[k] for r in rs])) for k in ['bulk_mbps','video_on_time','video_delivered_vs_requested','safety_stop_fraction','roams']}}
        comparisons={}
        for f in P['families']:
            for ref in ['radiomap','previous_jepa','confirmed_hybrid']:
                d=np.asarray(arms[f]['scene_ppm'])-arms[ref]['scene_ppm'];comparisons[f+'_vs_'+ref]=interval(d)
        groups[condition]={'sites':[c['scene'] for c in cs],'arms':arms,'comparisons':comparisons}
    family=selected['selected_family'];comparison=groups['main']['comparisons'][family+'_vs_radiomap']
    ablations={}
    main_cases=[c for c in cases if c['condition']=='main']
    for f in ['radiomap',family]:
        delta=[(row(c,f)['result']['control_miss']-row(c,f,False)['result']['control_miss'])*1e6 for c in main_cases]
        ablations[f]=interval(delta)|{'timing_free_ppm':float(np.mean([row(c,f,False)['result']['control_miss']*1e6 for c in main_cases]))}
    latency=json.loads((OUT/'latency.json').read_text())
    summary={'completed':True,'built_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'fits':9,'selection_runs':96,'test_runs':160,'fresh_sites':24,
        'selected_family':family,'selected_planners':{f:v['planner'] for f,v in selected['choices'].items()},'conditions':groups,'timing_ablations':ablations,
        'latency_p95_seconds':latency['p95_seconds'],'primary_control_superiority':comparison['paired_site_bootstrap_95_ppm'][1]<0,
        'simulation_only':True,'physical_timing_qualified':False}
    write(OUT/'summary.json',summary)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));x=np.arange(len(METHODS));colors=['#3e79ad','#d3aa73','#9972b0','#57a694','#16a6ae','#cb785b']
    axes[0].bar(x,[groups['main']['arms'][f]['ppm'] for f in METHODS],color=colors)
    for c in main_cases:axes[0].scatter(x,[row(c,f)['result']['control_miss']*1e6 for f in METHODS],s=12,color='#173243',alpha=.6)
    axes[0].set_xticks(x,[LABEL[f] for f in METHODS],rotation=18,ha='right');axes[0].set_ylabel('Control misses / million');axes[0].set_title('Measured compute queue + charged scans')
    for f,col in zip(METHODS,colors):axes[1].scatter(1000*latency['p95_seconds']['1'][f]['warm'],groups['main']['arms'][f]['ppm'],s=75,color=col,label=LABEL[f])
    axes[1].set_xlabel('p95 warm compute charge (ms / 100 robots)');axes[1].set_ylabel('Control misses / million');axes[1].legend(fontsize=8);axes[1].set_title('Compute / control frontier')
    fig.tight_layout();fig.savefig(figdir/'frontier.png',dpi=180);fig.savefig(figdir/'frontier.pdf');plt.close(fig)
    ci=comparison['paired_site_bootstrap_95_ppm'];a=groups['main']['arms']
    lines=['# Warehouse world-model optimisation frontier','',
        f'Development selected **{LABEL[family]}**. On eight fresh primary warehouses, with compute and scan costs charged, it reaches **{a[family]["ppm"]:.2f}** control misses per million versus **{a["radiomap"]["ppm"]:.2f}** for optimised RadioMap. Paired difference: {comparison["mean_difference_ppm"]:.2f} ppm, 95% site-bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}].',
        'The prespecified primary control-superiority interval '+('excludes zero.' if summary['primary_control_superiority'] else 'includes zero; the superiority claim fails.'),'',
        '## Implemented optimisation axes','',
        '- Rolling fleet sufficient statistics, eliminating repeated full-minute reconstruction.',
        '- BLAS covariance projection for RadioMap itself, so the reference benefits from the numerical optimisation too.',
        '- Three compact single-member families: teacher distillation, route/reliability loss, and GRU history plus causal innovation correction. Each has three independent fits; validation fixes one checkpoint per family.',
        '- Absolute report count, AP-local cross-band pooling, variable fleet density and sparse-scanning training.',
        '- Route-change and ns-3 MCS0 failure-surrogate losses. The surrogate is not end-to-end native packet supervision.',
        '- Independently selected native versus saturated-margin planners for every reference and candidate.',
        '- An asynchronous FIFO of measured processing costs, forecast-age interpolation, and actual scan outages in the packet kernel.','',
        '## Fresh results','',
        '| Condition | '+' | '.join(LABEL[f] for f in METHODS)+' |','|---|'+'---:|'*len(METHODS)]
    for c,g in groups.items():lines.append('| '+c+' | '+' | '.join(f'{g["arms"][f]["ppm"]:.2f}' for f in METHODS)+' |')
    lines+=['','Missed control commands per million; eight primary sites, six each drift/sparse and four slow-hardware stress sites. All have 100 robots.','',
        '![Optimisation frontier](../artifacts/world_frontier/figures/frontier.png)','',
        '## Forecast and timing quality','',
        '| Model | Forecast RMSE | Route-change RMSE | AP-choice regret | Warm charge (ms) | Cold charge (ms) |','|---|---:|---:|---:|---:|---:|']
    for f in METHODS:
        m=a[f]['forecast_metrics'];t=latency['p95_seconds']['1'][f]
        lines.append(f'| {LABEL[f]} | {m["rmse_db"]:.2f} | {m["delta_rmse_db"]:.2f} | {m["ap_regret"]:.3f} | {1000*t["warm"]:.2f} | {1000*t["cold"]:.2f} |')
    lines+=['','Timing is measured in a quiet single-thread CPU profile on validation inputs, including tensor packing. Sparse scans use their own profile. Cold has one probe and refit has two; the latter includes a seventeen-timestamp catch-up probe, so its processing budget is deliberately larger than an ordinary cached update. This is a microbenchmark envelope, not a production latency guarantee or a monotonic bound on packet losses.','',
        '## Timing-free ablation','',
        'This replay removes forecast delay and restores the former 30-second unscheduled background-scan policy. It changes both timing assumptions together, not inference alone.','',
        '| Model | Charged misses / million | Former timing assumptions | Difference |','|---|---:|---:|---:|']
    for f,v in ablations.items():lines.append(f'| {LABEL[f]} | {a[f]["ppm"]:.2f} | {v["timing_free_ppm"]:.2f} | {v["mean_difference_ppm"]:.2f} |')
    lines+=['','## Evidence, limits and remaining options','',
        'Nine real fits, 96 native development runs and 160 test runs on 24 fresh sites. Every reference is recomputed with the same strictly past inputs, charged timing and scan cadence. Earlier evidence is preserved; these are new scenarios and cannot be compared directly with the old 167/133-ppm cohort.','',
        'The kernel models 30 ms scans and 30 ms handovers normally. Slow stress uses 90 ms scans, four-second handovers and an added 500 ms compute delay. These durations are assumptions, not hardware measurements. Scanner report phases are idealised; attached-link measurements remain idealised; robot trajectories and warehouse CAD are fixed; forecast targets include planned-versus-realised route mismatch. No physical fleet or arbitrary-layout qualification follows.','',
        'This is a finite search, not exhaustion of all world-model options. Remaining work includes native counterfactual packet-loss training, varied CAD layouts, calibrated joint uncertainty, live observation/scan coupling, measured hardware timing and resource-aware fleet scheduling. Negative results and failed promotion checks remain evidence.','',
        'Reproduce with `bash scripts/run_world_frontier.sh`. Audit with `uv run --frozen python scripts/audit_world_frontier.py`.']
    (ROOT/'docs/WORLD_FRONTIER_RESULTS.md').write_text('\n'.join(lines)+'\n')
    analysis={}
    for p in [ROOT/'scripts/report_world_frontier.py',ROOT/'scripts/audit_world_frontier.py',ROOT/'scripts/calibrate_frontier_timing.py',ROOT/'scripts/run_world_frontier.sh',ROOT/'scripts/report_frontier_confirmation.py',ROOT/'scripts/report_hardware_controls.py',ROOT/'src/wirelessjev/frontier_control.py']:
        name=str(p.relative_to(ROOT));dest=OUT/'analysis_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);analysis[name]=digest(p)
    write(OUT/'analysis_source.json',{'source_sha256':analysis})
    files={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    previous=[ROOT/'artifacts/contracted/manifest.json',ROOT/'artifacts/gate_repair/manifest.json',ROOT/'artifacts/fleet_world_model/manifest.json',ROOT/'artifacts/fleet_confirmation/manifest.json']
    write(OUT/'manifest.json',{'completed':True,'files_sha256':files,'checkpoint_sha256':checkpoint_hashes(),'previous_manifests':{str(p.relative_to(ROOT)):digest(p) for p in previous},'preregistration_commit':'51efe24','simulation_only':True})
    print(json.dumps({'selected_family':family,'primary_comparison':comparison,'primary_control_superiority':summary['primary_control_superiority']},indent=2))


if __name__=='__main__':main()
