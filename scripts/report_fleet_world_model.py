"""Report matched fleet-model tests without selecting on test outcomes."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from wirelessjev.fleetwm.study import ROOT,OUT,P,sources,digest
from wirelessjev.contracted.train import write

FAMILIES=['fixed','static_radiomap','rolling_radiomap','learned_radiomap','previous_jepa',*P['families']]
LABELS={'fixed':'Fixed split','static_radiomap':'Static RadioMap','rolling_radiomap':'Rolling RadioMap',
        'learned_radiomap':'Learned-prior RadioMap','previous_jepa':'Previous JEPA','hybrid_jepa':'Hybrid JEPA','hybrid_supervised':'Hybrid supervised','fleet_jepa':'Fleet JEPA'}


def paired(delta):
    d=np.asarray(delta,dtype=float);rng=np.random.default_rng(289000)
    draws=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
    return {'mean_difference_ppm':float(d.mean()),'paired_scene_bootstrap_95_ppm':np.quantile(draws,[.025,.975]).tolist(),
            'wins':int((d<0).sum()),'ties':int((d==0).sum()),'scenes':len(d)}


def row(c,f,mission=None):
    return next(r for r in c['rows'] if r['family']==f and r['mission']==(mission or P['primary_mission']))


def summarize(cases,selection,mission=None,reference='rolling_radiomap'):
    families=FAMILIES if mission is None else ['rolling_radiomap','learned_radiomap',*P['families']]
    arms={}
    for f in families:
        rs=[row(c,f,mission)['result'] for c in cases];miss=sum(r['control_misses'] for r in rs);sent=sum(r['control_sent'] for r in rs)
        arms[f]={'misses':miss,'commands':sent,'ppm':miss/sent*1e6,'scene_ppm':[r['control_miss']*1e6 for r in rs],
                 'roams':sum(r['roams'] for r in rs)}
        arms[f]['traffic_and_safety']={k:float(np.mean([r[k] for r in rs])) for k in
            ['bulk_mbps','video_on_time','video_delivered_vs_requested','safety_stop_fraction','safety_stops','control_miss_worst_decile','control_gap_p99_ms']}
        if mission is None and f!='fixed':
            arms[f]['forecast_metrics']={k:float(np.mean([c['forecast_metrics'][f][k] for c in cases if c['forecast_metrics'][f][k] is not None])) for k in cases[0]['forecast_metrics'][f]}
            # The executed kernel splits at mean SNR=0, not the RSSI hearing
            # threshold. Preserve raw case files but use precise report labels.
            m=arms[f]['forecast_metrics']
            m['positive_snr_rmse_db']=m.pop('audible_rmse_db')
            m['nonpositive_snr_rmse_db']=m.pop('unheard_rmse_db')
            arms[f]['mean_predict_seconds_per_fleet_decision']=float(np.mean([c['predict_seconds'][f]/c['duration_s'] for c in cases]))
    comparisons={}
    for f in families:
        if f=='rolling_radiomap':continue
        delta=np.asarray(arms[f]['scene_ppm'])-arms['rolling_radiomap']['scene_ppm']
        comparisons[f+'_vs_rolling_radiomap']=paired(delta)|{'relative_reduction':1-arms[f]['ppm']/arms['rolling_radiomap']['ppm'] if arms['rolling_radiomap']['ppm'] else None}
    if reference in arms:
        for f in families:
            if f==reference:continue
            delta=np.asarray(arms[f]['scene_ppm'])-arms[reference]['scene_ppm']
            comparisons[f+'_vs_'+reference]=paired(delta)|{'relative_reduction':1-arms[f]['ppm']/arms[reference]['ppm'] if arms[reference]['ppm'] else None}
    for ref in ['static_radiomap','rolling_radiomap','learned_radiomap']:
        if ref not in arms:continue
        for f in P['families']:
            delta=np.asarray(arms[f]['scene_ppm'])-arms[ref]['scene_ppm']
            comparisons[f+'_vs_'+ref]=paired(delta)|{'relative_reduction':1-arms[f]['ppm']/arms[ref]['ppm'] if arms[ref]['ppm'] else None}
            if mission is None:
                errors=[c['forecast_metrics'][f]['rmse_near_db']-c['forecast_metrics'][ref]['rmse_near_db'] for c in cases]
                evidence=paired(errors)
                comparisons[f+'_forecast_vs_'+ref]={'mean_rmse_difference_db':evidence['mean_difference_ppm'],
                    'paired_scene_bootstrap_95_db':evidence['paired_scene_bootstrap_95_ppm'],'wins':evidence['wins'],'scenes':len(cases)}
    if 'hybrid_supervised' in arms:
        comparisons['hybrid_jepa_vs_hybrid_supervised']=paired(np.asarray(arms['hybrid_jepa']['scene_ppm'])-arms['hybrid_supervised']['scene_ppm'])
    return {'scenes':[c['scene'] for c in cases],'arms':arms,'comparisons':comparisons}


def main():
    completion=json.loads((OUT/'completion.json').read_text());assert completion['completed']
    selection=json.loads((OUT/'selection.json').read_text());assert selection['source_sha256']==sources()
    seal=json.loads((OUT/'comparison_seal.json').read_text());reference=seal['strongest_development_radiomap']
    assert seal['core_selection_sha256']==digest(OUT/'selection.json')
    assert seal['reference_selection_sha256']==digest(OUT/'strong_reference/selection.json')
    assert json.loads((OUT/'strong_reference/completion.json').read_text())['completed']
    for name,h in selection['checkpoint_sha256'].items():assert digest(ROOT/name)==h
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('test-*.json'))];assert len(cases)==32
    for c in cases:
        ref=json.loads((OUT/'strong_reference/cases'/f'test-{c["condition"]}-{c["scene"]}.json').read_text())
        c['rows']+=ref['rows'];c['forecast_metrics']['learned_radiomap']=ref['forecast_metrics'];c['predict_seconds']['learned_radiomap']=ref['predict_seconds']
    groups={c:summarize([x for x in cases if x['condition']==c],selection,reference=reference) for c in ['main','stress','yard','small','drift']}
    heldout=summarize([x for x in cases if x['condition']=='main'],selection,P['heldout_mission'])
    best=selection['development_best_model'];main=groups['main'];comparison=main['comparisons'][best+'_vs_'+reference]
    claim=comparison['paired_scene_bootstrap_95_ppm'][1]<0
    beats_all=all(main['comparisons'][best+'_vs_'+ref]['paired_scene_bootstrap_95_ppm'][1]<0 for ref in ['static_radiomap','rolling_radiomap','learned_radiomap'])
    limitations=['Simulation only; no physical smartcar qualification',
        'Warehouse seeds share the canonical rack/AP layout and randomise propagation, routes and traffic; arbitrary warehouse-layout transfer is not tested',
        'Stress, yard and drift examples appear in training; their fresh-site tests do not establish transfer to entirely unseen environment families',
        'All native controllers retain the same idealised reactive measurement model and 30 ms handover assumption',
        'Known AP positions, perfect robot localisation, CAD rack geometry and configured noise floors are assumed',
        'Targets use realised future mean SNR while inputs contain planned positions; route mismatch contributes to forecast error, so this is not pure fixed-position map reconstruction',
        'All policies share precomputed robot trajectories; safety-stop time is measured without resimulating robot motion',
        'Forecasts are generated causally but supplied to the simulator offline; measured inference/refit time is not charged to packet deadlines',
        'Training uses synthetic mean-field truth for all links; real device-only training has not been established',
        'The hybrid depends on RadioMap fitting; its neural-head time is not an end-to-end compute advantage',
        'The standalone fleet model uses reports-only geometric summaries; its results must be assessed separately',
        'Three training seeds are ensembled; site intervals do not quantify retraining variance',
        'Forecast inputs pool scans from fixed simulated trajectories rather than a resource-constrained hardware scan scheduler',
        'Drift is an explicitly constructed intervention; the changing attenuation schedule is hidden from forecasters',
        'No safety certificate or universal Earth/environment/protocol generalisation is established']
    fits=[json.loads(p.read_text()) for p in sorted((OUT/'fits').glob('*.json'))]
    summary={'completed':True,'built_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'fits':len(fits),**completion,
        'development_best_model':best,'strongest_development_radiomap':reference,
        'gate_choices':{f:r['gate'] for f,r in selection['choices'].items()}|{'learned_radiomap':json.loads((OUT/'strong_reference/selection.json').read_text())['gate']},
        'core_test_rollouts':completion['test_rollouts'],'test_rollouts':completion['test_rollouts']+40,
        'total_selection_rollouts':completion['selection_rollouts']+60,'total_native_rollouts':completion['test_rollouts']+completion['selection_rollouts']+100,
        'primary_superiority_interval_excludes_zero':claim,'conditions':groups,'heldout_mission':heldout,
        'primary_beats_all_three_radiomap_references':beats_all,
        'mean_rolling_refit_seconds_per_case':float(np.mean([c['rolling_refits_seconds'] for c in cases])),
        'parameters_per_member':fits[0]['parameters'],'inference_parameters_per_member':fits[0]['parameters']-4800,'limitations':limitations}
    summary['runtime_benchmark']=json.loads((OUT/'runtime_benchmark.json').read_text())
    write(OUT/'summary.json',summary)
    figure_dir=OUT/'figures';figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#83949d','#9dabb6','#245471','#3e79ad','#d3aa73','#12a28e','#9a72b0','#cf6f62']
    fig,ax=plt.subplots(figsize=(11,4.7));x=np.arange(len(FAMILIES))
    ax.bar(x,[main['arms'][f]['ppm'] for f in FAMILIES],color=colors)
    for c in [c for c in cases if c['condition']=='main']:
        ax.plot(x,[row(c,f)['result']['control_miss']*1e6 for f in FAMILIES],color='#273a4a',alpha=.15,lw=.6)
        ax.scatter(x,[row(c,f)['result']['control_miss']*1e6 for f in FAMILIES],s=12,color='#273a4a',alpha=.55)
    ax.set_xticks(x,[LABELS[f] for f in FAMILIES],rotation=12,ha='right');ax.set_ylabel('Missed control commands / million')
    ax.set_title('Eight fresh 100-robot warehouses · separately development-tuned controllers · simulation')
    fig.tight_layout();fig.savefig(figure_dir/'primary.png',dpi=180);fig.savefig(figure_dir/'primary.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for f,col in zip(FAMILIES[1:],colors[1:]):
        m=main['arms'][f]['forecast_metrics'];axes[0].scatter(m['rmse_near_db'],main['arms'][f]['ppm'],s=75,color=col,label=LABELS[f])
    axes[0].set_xlabel('Forecast RMSE over first six horizons (dB)');axes[0].set_ylabel('Control misses / million');axes[0].legend(fontsize=8)
    conditions=list(groups);ratios=np.array([[groups[c]['arms'][f]['ppm']/max(groups[c]['arms']['rolling_radiomap']['ppm'],1e-9) for c in conditions] for f in FAMILIES])
    im=axes[1].imshow(np.log2(np.maximum(ratios,.001)),vmin=-2,vmax=2,cmap='RdBu_r',aspect='auto')
    axes[1].set_yticks(range(len(FAMILIES)),[LABELS[f] for f in FAMILIES]);axes[1].set_xticks(range(5),conditions,rotation=20)
    for i in range(len(FAMILIES)):
        for j in range(5):axes[1].text(j,i,f'{ratios[i,j]:.2f}×',ha='center',va='center',fontsize=8)
    axes[1].set_title('Miss-rate ratio to tuned rolling RadioMap');fig.colorbar(im,ax=axes[1],label='log₂ ratio')
    fig.tight_layout();fig.savefig(figure_dir/'tradeoffs.png',dpi=180);fig.savefig(figure_dir/'tradeoffs.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,11),ylim=(0,4));ax.axis('off')
    boxes=[(.1,1.2,1.8,1.4,'Past device scans\nFleet positions\nKnown AP / CAD'),(2.5,2.1,2.1,1.1,'Causal fleet memory\nGeometry summaries'),(2.5,.5,2.1,1.1,'Rolling RadioMap\nHybrid branch only'),(5.3,1.2,2.3,1.4,'JEPA context encoder\nRoute-conditioned predictor\nMean-field uncertainty'),(8.3,1.2,2.3,1.4,'Typed native controller\nDevelopment-tuned gate\nRecovery + link adaptation')]
    for x0,y0,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.08',facecolor='#e7f4f1' if x0>5 else '#e9eef4',edgecolor='#31566a'))
        ax.text(x0+w/2,y0+h/2,label,ha='center',va='center',fontsize=9)
    for a,b in [((1.95,2.1),(2.4,2.6)),((1.95,1.7),(2.4,1.05)),((4.7,2.6),(5.2,2.2)),((4.7,1.05),(5.2,1.6)),((7.7,1.9),(8.2,1.9))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=14,color='#31566a'))
    ax.text(5.5,3.65,'Mean-field labels and future EMA targets supervise training only',ha='center',fontsize=11)
    ax.text(5.5,.05,'Three seeds / family · shared causal observations · no simulator truth in forecast inputs',ha='center',fontsize=10)
    fig.tight_layout();fig.savefig(figure_dir/'architecture.png',dpi=180);fig.savefig(figure_dir/'architecture.pdf');plt.close(fig)
    ci=comparison['paired_scene_bootstrap_95_ppm']
    lines=['# Fleet world-model results','',
        f'The model chosen on development scenes was **{LABELS[best]}**. On eight fresh primary scenes it produces **{main["arms"][best]["ppm"]:.2f}** misses per million, versus **{main["arms"][reference]["ppm"]:.2f}** for **{LABELS[reference]}**, the RadioMap reference chosen on development scenes.',
        f'The paired difference is {comparison["mean_difference_ppm"]:.2f} ppm, with a 95% scene-bootstrap interval [{ci[0]:.2f}, {ci[1]:.2f}] and {comparison["wins"]}/8 wins. The interval '+('supports a primary-condition reduction.' if claim else 'does not establish a primary-condition reduction.'),'',
        '![Architecture](../artifacts/fleet_world_model/figures/architecture.png)','',
        '## What was built','',
        '- A 28,346-parameter per-member model pooling six causal robot scans, a sixty-second fleet memory, known rack geometry and a planned route.',
        '- A hybrid that corrects rolling RadioMap forecasts, plus a standalone fleet JEPA with no dependence on per-BSS RadioMap predictions.',
        '- A supervised hybrid ablation with the same architecture and inputs but no EMA/JEPA objective.',
        '- True mean-field training targets, horizon weighting and AP-score regret, separating forecast-mean uncertainty from noisy scan spread.',
        '- Matched censored rolling RadioMap, updated every twenty seconds from the trailing minute, with its own controller tuning.',
        '- A stronger RadioMap using the existing forty-site learned foundation prior, with its own development tuning and a separate pre-test comparison seal.',
        '- Nine real model fits on 32 sites, six distinct validation sites and twelve closed-loop selection sites. The choice was sealed before 32 fresh test sites.','',
        '## Primary results','',
        'Eight warehouses, 100 robots each, 120 seconds per warehouse. RadioMap and world-model arms share the same native hardware and 30 ms handover assumptions. Fixed split retains the original default-MLO policy and its 60 ms handover setting.','',
        '| Method | Misses / million | Forecast RMSE (dB) | AP-choice regret (dB) |','|---|---:|---:|---:|']
    for f in FAMILIES:
        a=main['arms'][f];m=a.get('forecast_metrics')
        lines.append(f'| {LABELS[f]} | {a["ppm"]:.2f} | '+(f'{m["rmse_near_db"]:.2f} | {m["ap_choice_regret_db"]:.3f}' if m else '— | —')+' |')
    lines+=['','![Primary](../artifacts/fleet_world_model/figures/primary.png)','',
        'Forecast RMSE and AP-choice regret describe mean-field prediction, not actual packet-loss regret. Lower forecast error can still produce worse control decisions. The split error metrics use positive versus nonpositive mean SNR; they are not observed-versus-unheard scan labels (the raw case kernel uses older audible/unheard key names for that split).','',
        '## All fresh conditions','',
        '| Condition | '+ ' | '.join(LABELS[f] for f in FAMILIES)+' |','|---|'+'---:|'*len(FAMILIES)]
    for c,group in groups.items():lines.append('| '+c+' | '+' | '.join(f'{group["arms"][f]["ppm"]:.2f}' for f in FAMILIES)+' |')
    lines+=['','The condition table reports misses per million. Primary has eight sites; other conditions have six. Small has ten robots. Generalisation runs last sixty seconds.','',
        'Primary traffic and stop-time metrics (scene means):','',
        '| Method | Bulk Mbps | Video on time | Video delivered / requested | Safety-stop fraction |','|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        q=main['arms'][f]['traffic_and_safety']
        lines.append(f'| {LABELS[f]} | {q["bulk_mbps"]:.2f} | {q["video_on_time"]:.4f} | {q["video_delivered_vs_requested"]:.4f} | {q["safety_stop_fraction"]:.6f} |')
    lines+=['','![Tradeoffs](../artifacts/fleet_world_model/figures/tradeoffs.png)','',
        '## JEPA ablation and held-out mission','',
        'The supervised hybrid is the test of whether the EMA/JEPA objective adds value to the same information and architecture. Full paired comparisons and held-out teleoperation-mission results are in `summary.json`; the development-best designation is never changed using test results.','',
        '## Compute and reproducibility','',
        'Study prediction timings include input construction and the neural head but exclude map refits. The separate runtime benchmark below times the actual API, including any initial or refresh fit. Each call predicts nine horizons for 100 robots. It uses one CPU thread on this host and four warm calls; these are microbenchmarks, not hardware latency guarantees. Checkpoint loading is recorded separately in `runtime_benchmark.json`. Each member has 23,546 inference parameters; the reported 28,346 total also includes the frozen EMA teacher.','',
        '| Method | Initial call (ms) | Warm call (ms) | Refresh call (ms) |','|---|---:|---:|---:|']
    for f,b in summary['runtime_benchmark']['results'].items():
        lines.append(f'| {LABELS[f]} | {1000*b["initial_forecast_seconds"]:.2f} | {1000*b["warm_forecast_mean_seconds"]:.2f} | {1000*b["refresh_forecast_seconds"]:.2f} |')
    lines+=['',
        'Run `bash scripts/run_fleet_research.sh` to reproduce the core study, stronger-reference amendment, independent forecast replay, reports, audit and tests sequentially. Dataset receipts hash the deterministic training caches; the caches are rebuilt locally. Checkpoints, histories, all native counters and source capsules are archived under `artifacts/fleet_world_model`. Audit with `uv run --frozen python scripts/audit_fleet_world_model.py`.','',
        '## Limits','']+['- '+s+'.' for s in limitations]
    (ROOT/'docs/FLEET_WORLD_MODEL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    analysis={}
    for path in [ROOT/'scripts/report_fleet_world_model.py',ROOT/'scripts/audit_fleet_world_model.py',ROOT/'scripts/verify_fleet_forecasts.py',ROOT/'scripts/benchmark_fleet_runtime.py',ROOT/'scripts/analyze_fleet_validation.py',ROOT/'scripts/run_fleet_research.sh',ROOT/'src/wirelessjev/fleet_control.py']:
        name=str(path.relative_to(ROOT));target=OUT/'analysis_source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);analysis[name]=digest(path)
    write(OUT/'analysis_source.json',{'source_sha256':analysis})
    files={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(OUT/'manifest.json',{'completed':True,'files_sha256':files,'checkpoint_sha256':selection['checkpoint_sha256'],
        'previous_archive_manifests':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'artifacts/contracted/manifest.json',ROOT/'artifacts/gate_repair/manifest.json']},
        'preregistration_commit':'0385ac2','simulation_only':True})
    print(json.dumps({'development_best_model':best,'primary':main,'interval_supports_reduction':claim},indent=2))


if __name__=='__main__':main()
