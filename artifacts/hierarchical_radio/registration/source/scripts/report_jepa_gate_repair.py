"""Report the frozen gate-repair tests without retuning on their outcomes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wirelessjev.gate_repair.study import OUT,P,sources,checkpoints
from wirelessjev.contracted.data import ROOT
from wirelessjev.contracted.train import write


def interval(values):
    v=np.asarray(values,dtype=float);rng=np.random.default_rng(272000);b=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return {'mean_ppm':float(v.mean()),'paired_scene_bootstrap_95_ppm':np.quantile(b,[.025,.975]).tolist(),'wins':int((v<0).sum()),'ties':int((v==0).sum()),'scenes':len(v)}


def aggregate_trace(files):
    arrays=[]
    for path in files:
        with np.load(path) as z:arrays.append({k:z[k] for k in z.files})
    from wirelessjev.gate_repair.gate import diagnostics
    return diagnostics({k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]})


def main():
    completion=json.loads((OUT/'completion.json').read_text());assert completion['completed']
    selection=json.loads((OUT/'selection.json').read_text());assert selection['source_sha256']==sources() and selection['checkpoint_sha256']==checkpoints()
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('test-*.json'))];assert len(cases)==24
    kinds=['fixed','bayesian','ungated','legacy','rescue','scale','gain'];groups={}
    def row(c,k):return next(r for r in c['rows'] if r['kind']==k)
    for condition in ['main','stress','yard','small']:
        cs=[c for c in cases if c['condition']==condition];assert len(cs)==6;arms={}
        for kind in kinds:
            rs=[row(c,kind) for c in cs];miss=sum(r['result']['control_misses'] for r in rs);sent=sum(r['result']['control_sent'] for r in rs)
            arms[kind]={'misses':miss,'sent':sent,'ppm':miss/sent*1e6}
            if kind not in ['fixed','bayesian']:arms[kind]['decision_diagnostics']=aggregate_trace([OUT/r['trace'] for r in rs])
        comparisons={}
        for kind,reference in [('scale','legacy'),('gain','legacy'),('scale','ungated'),('gain','ungated'),('scale','rescue'),('gain','rescue'),('scale','bayesian')]:
            delta=[(row(c,kind)['result']['control_miss']-row(c,reference)['result']['control_miss'])*1e6 for c in cs]
            comparisons[kind+'_vs_'+reference]=interval(delta)|{'relative_reduction':1-arms[kind]['ppm']/arms[reference]['ppm'] if arms[reference]['ppm'] else None}
        groups[condition]={'arms':arms,'paired_comparisons':comparisons}
    historical_files=sorted((OUT/'historical_diagnosis').glob('*.json'));assert len(historical_files)==6
    historical=aggregate_trace([p.with_suffix('.npz') for p in historical_files]);historical['original_counters_reproduced']=all(json.loads(p.read_text())['original_counters_match'] for p in historical_files)
    recommended=min(['scale','gain'],key=lambda k:(selection[k]['score'],0 if k=='scale' else 1))
    summary={'completed':True,'built_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'study_native_runs':270,'diagnostic_native_replays':6,'fresh_test_scenes':24,'fresh_test_runs':168,'frozen_jepa_members':5,
        'selection':{k:{j:selection[k][j] for j in ['parameter','score','rejection_fraction']} for k in ['scale','gain']},'development_best':recommended,
        'historical_diagnosis':historical,'fresh_results':groups,'selection_sealed_before_tests':selection['sealed_at_unix']<json.loads((OUT/'test_source.json').read_text())['time_unix'],
        'limitations':['Simulation only; the native reactive fallback retains its idealised measurement model','Native handovers assume 30 ms; stock-car handover timing and field qualification remain pending','Bulk-upload mission only; no on-car inference or switching was qualified','The Bayesian reference was not retuned in this JEPA-focused repair','The absolute uncertainty still undercovers some links; the selected multiplier is a control penalty, not a corrected 90% signal interval','Gain residual quantiles are empirical and do not give a physical safety certificate','The gain model is not consistently better than the simpler reactive-rescue ablation','Neither JEPA model weights nor its mean forecasts changed between gate arms','The test stress condition still favors fixed split over the repaired JEPA','Old test scenes were replayed only after the new choice was sealed, solely for diagnosis and exact-counter compatibility']}
    write(OUT/'summary.json',summary)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Fixed split','Bayesian gate','JEPA ungated','Old JEPA gate','Reactive rescue','Tuned scale','Gain bound'];x=np.arange(7)
    fig,ax=plt.subplots(figsize=(11,4.5));main_group=groups['main'];ax.bar(x,[main_group['arms'][k]['ppm'] for k in kinds],color=['#8a999c','#447fa5','#dba77b','#c96457','#afbad0','#24a99a','#9263b6'])
    for c in [c for c in cases if c['condition']=='main']:ax.scatter(x,[row(c,k)['result']['control_miss']*1e6 for k in kinds],s=16,c='#173243',alpha=.55)
    ax.set_xticks(x,names,rotation=12,ha='right');ax.set_ylabel('Control misses / million commands');ax.set_title('Six fresh warehouse scenes · identical frozen JEPA forecasts · simulation');fig.tight_layout();fig.savefig(figdir/'fresh_primary.png',dpi=180);fig.savefig(figdir/'fresh_primary.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['Eligible','Rejected','Positive gain\nrejected'],[historical['opportunities'],historical['rejected'],historical['beneficial_rejected_without_rescue']],color=['#8a999c','#c96457','#dba77b']);axes[0].set_ylabel('One-second robot decision opportunities');axes[0].set_title('Exact replay of the original six scenes')
    paths=[p.with_suffix('.npz') for p in historical_files];us=[];es=[]
    for p in paths:
        with np.load(p) as z:us.extend(z['legacy_u']);es.extend(np.abs(z['error']))
    axes[1].boxplot([us,es],tick_labels=['Old pair uncertainty','Actual gain error'],showfliers=False);axes[1].set_ylabel('dB');axes[1].set_title('Absolute-link spread versus relative error');fig.tight_layout();fig.savefig(figdir/'diagnosis.png',dpi=180);fig.savefig(figdir/'diagnosis.pdf');plt.close(fig)
    def ci(c):return '['+', '.join(f'{v:.2f}' for v in c['paired_scene_bootstrap_95_ppm'])+']'
    m=groups['main'];a=m['arms'];r=m['paired_comparisons'];scale=selection['scale']['parameter'];q=selection['gain']['parameter']
    lines=['# JEPA gate repair: diagnosis and fresh-scene results','',
        f'The development-selected **{scale}× gate penalty** reduces primary misses from **{a["legacy"]["ppm"]:.2f} to {a["scale"]["ppm"]:.2f} per million** on six fresh warehouse scenes: **{100*r["scale_vs_legacy"]["relative_reduction"]:.1f}% fewer**. The new gain-bound gate reaches {a["gain"]["ppm"]:.2f}. The five JEPA models and their forecasts are unchanged.','',
        '## Why the original gate was poor','',
        f'Forensic replays reproduce all six original packet/roam counters exactly. Of **{historical["opportunities"]:,} eligible decision opportunities**, the old gate rejected **{historical["rejected"]:,}** ({100*historical["rejection_fraction"]:.1f}%). **{historical["beneficial_rejected_without_rescue"]:,} rejected opportunities had positive true AP-score gain and no reactive rescue**. These are repeated robot/second opportunities, not independent packets or distinct handovers.','',
        f'The old pair uncertainty median was **{historical["median_legacy_uncertainty_db"]:.2f} dB**, versus **{historical["median_abs_gain_error_db"]:.2f} dB** median actual gain error. Its max-band rule gave {historical["median_active_band_uncertainty_db"]:.2f} dB even when restricted to the band with highest predicted current SNR: the larger problem was the uncertainty target and scale, not merely which band was selected.','',
        'The planner compares a route-averaged *difference between two AP scores*. The JEPA head was trained on noisy future scan readings, whereas the gate needed uncertainty about the mean AP-score advantage. Its spread was calibrated over all links, including unheard links, then maximised over bands and combined across APs as if independent. That made the gate excessively reluctant to approve useful early moves. The reactive rule could not rescue them while its separate weak-link trigger remained false.','',
        'Ensemble member predictions also share offsets. Their paired gain spread is computed directly in the repair, preserving covariance instead of adding independent marginal variances. A small Ridge bias/scale model then calibrates one-sided gain-overprediction error. No twin truth enters the runtime gate; labels are added to recorded traces only after a rollout.','',
        '![Original diagnosis](../artifacts/gate_repair/figures/diagnosis.png)','',
        '## What changed','',
        f'- A simple gate penalty was selected from {P["scale_candidates"]} using new closed-loop development scenes. The choice was **{scale}**, not a threshold chosen from the fresh test results.',
        f'- A decision-aligned gain bound was fit on four new sites, calibrated on two others and selected at empirical quantile **{q}**. It can still reject moves; it is not a renamed ungated controller.',
        '- When no planned move is eligible, the gain gate explicitly defers to the reactive rule. The separate reactive-rescue ablation measures this routing benefit without an uncertainty filter.',
        '- The optional simulator gate/trace interface receives forecasts and current decision state only. Tests reproduce the archived kernel exactly when the hook is unused and verify that tracing changes no result.',
        '- Historical artifacts remain unchanged. The old report writer now refuses a changed execution kernel before rewriting any old output.','',
        '## Six fresh primary scenes','',
        '100 robots, 120 seconds per scene: 1,440,000 commands per arm. These are different scenes from the original 151.39-ppm study; before/after comparisons below use the same new scenes.','',
        '| Controller | Missed commands | Misses / million |','|---|---:|---:|']
    for kind,label in zip(kinds,names):lines.append(f'| {label} | {a[kind]["misses"]} | {a[kind]["ppm"]:.2f} |')
    lines+=['',f'Tuned scale minus old gate: **{r["scale_vs_legacy"]["mean_ppm"]:.2f} ppm**, paired-scene 95% bootstrap interval {ci(r["scale_vs_legacy"])}; wins {r["scale_vs_legacy"]["wins"]}/6. Gain bound minus old gate: **{r["gain_vs_legacy"]["mean_ppm"]:.2f} ppm**, interval {ci(r["gain_vs_legacy"])}; wins {r["gain_vs_legacy"]["wins"]}/6.','',
        f'Tuned scale minus reactive rescue: {r["scale_vs_rescue"]["mean_ppm"]:.2f} ppm, interval {ci(r["scale_vs_rescue"])}. Gain bound minus reactive rescue: {r["gain_vs_rescue"]["mean_ppm"]:.2f} ppm, interval {ci(r["gain_vs_rescue"])}. Much of the improvement therefore comes from permitting useful planned moves and retaining reactive recovery; the added gain model is not the main-condition winner.','',
        '![Fresh primary results](../artifacts/gate_repair/figures/fresh_primary.png)','',
        '## Generalisation','',
        '| Condition | Fixed split | Bayesian gate | Ungated JEPA | Old gate | Rescue | Tuned scale | Gain bound |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in ['main','stress','yard','small']:lines.append('| '+condition+' | '+' | '.join(f'{groups[condition]["arms"][k]["ppm"]:.2f}' for k in kinds)+' |')
    lines+=['', 'Values are misses per million. Each condition has six fresh scenes. Generalisation runs last 60 seconds; `small` has 10 robots, others 100. The repaired methods improve the old gate but still lose to fixed split under stress. This is not universal controller superiority.','',
        '## Use the validated research profile','',
        'The simplest development-best profile is available explicitly; the RadioMap default is unchanged:', '', '```python','from wirelessjev.jepa_control import controller','from wirelessjev.native.sim import simulate','policy, roam_gate, profile = controller()  # development-best, currently scale','result = simulate(world_with_jepa_forecasts, policy, roam_gate=roam_gate)','```','',
        'The helper checks the frozen checkpoint identities and limits its validated mission to `bulk_uploads`. For the gain-bound variant, pass `controller(bank, method="gain")`. The scale is an operational gate penalty, not a claim that the original 90% SNR prediction interval has been repaired. Neither profile is a hardware/safety qualification.','',
        '## Evidence and limits','',f'The study contains **270 native runs**: 12 fit-trace runs, 90 development-selection runs and 168 fresh tests on 24 sites. Six additional historical diagnostic replays match the old counters and are not counted as fresh tests. The choice was sealed before test access. Development-best was determined from frozen development scores ({recommended}), not test ranking.','']
    lines.extend('- '+v+'.' for v in summary['limitations'])
    lines+=['', 'All reports, raw eligible-decision traces, gain coefficients, selection receipts, source capsules and checksums are in [artifacts/gate_repair](../artifacts/gate_repair/). The original [contracted study](CONTRACTED_RESULTS.md) remains intact. Audit with `uv run --frozen python scripts/audit_jepa_gate_repair.py`.']
    (ROOT/'docs/JEPA_GATE_REPAIR_RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Freeze analysis/runtime additions separately from the untouched test kernel.
    paths=[ROOT/'scripts/report_jepa_gate_repair.py',ROOT/'scripts/diagnose_original_jepa_gate.py',ROOT/'scripts/audit_jepa_gate_repair.py',ROOT/'src/wirelessjev/jepa_control.py']
    capsule={}
    for p in paths:
        relative=str(p.relative_to(ROOT));dest=OUT/'analysis_source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);capsule[relative]=hashlib.sha256(p.read_bytes()).hexdigest()
    write(OUT/'analysis_source.json',{'source_sha256':capsule})
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(OUT/'manifest.json',{'completed':True,'files_sha256':files,'original_archive_manifest_sha256':hashlib.sha256((ROOT/'artifacts/contracted/manifest.json').read_bytes()).hexdigest(),'checkpoint_sha256':checkpoints(),'study_native_runs':270,'diagnostic_replays':6,'test_native_runs':168,'preregistration_commit':'412b649','simulation_only':True})
    print(json.dumps({'fresh_primary':groups['main'],'historical_diagnosis':historical,'development_best':recommended},indent=2))


if __name__=='__main__':main()
