"""Separate preregistered confirmation; never rewrites discovery evidence."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,shutil,time
import numpy as np
import torch
from .fleetwm.study import ROOT,OUT as DISC,load,predict,sources as core_sources,forecast_metrics,digest
from .fleetwm.data import episode,inputs,fit_map,map_prediction
from .fleet_strong_reference import fit_strong_map
from .contracted.data import READINGS,KEYS,examples,runtime_world
from .contracted.train import write,load_models
from .contracted.runtime import DeviceInputs,forecast
from .native.world import HORIZONS
from .native.intent_policy import intent_policy
from .native.sim import simulate
from .gate_repair.gate import reactive_rescue

OUT=ROOT/'artifacts/fleet_confirmation';P=json.loads((OUT/'protocol.json').read_text())


def source_hashes():
    paths=[ROOT/'src/wirelessjev/fleet_confirmation.py',ROOT/'src/wirelessjev/fleet_strong_reference.py',
        OUT/'protocol.json',ROOT/'artifacts/native/foundation.json',ROOT/'artifacts/contracted/calibration.json']
    return core_sources()|{str(p.relative_to(ROOT)):digest(p) for p in paths}


def seal():
    dest=OUT/'selection.json';hashes=source_hashes()
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['source_sha256']!=hashes:raise RuntimeError('Confirmation source changed')
        return old
    counts={f:0 for f in ['hybrid_jepa','hybrid_supervised','fleet_jepa']};discovery={}
    for seed in P['discovery_sites']:
        path=DISC/'cases'/f'test-main-{seed}.json';c=json.loads(path.read_text());discovery[str(path.relative_to(ROOT))]=digest(path)
        for f in counts:counts[f]+=next(r['result']['control_misses'] for r in c['rows'] if r['family']==f and r['mission']==P['mission'])
    candidate=min(counts,key=lambda f:(counts[f],f));assert candidate==P['fixed_candidate'] and counts[candidate]==P['candidate_discovery_misses']
    selected=json.loads((DISC/'selection.json').read_text());ref=json.loads((DISC/'strong_reference/selection.json').read_text())
    for f in ['hybrid_jepa','hybrid_supervised','previous_jepa']:assert selected['choices'][f]['gate']==P['gate']
    assert ref['gate']==P['gate']
    checkpoints=selected['checkpoint_sha256']|{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'artifacts/contracted/checkpoints').glob('jepa-*.pt')}
    for name,h in checkpoints.items():assert digest(ROOT/name)==h
    for name in hashes:
        path=OUT/'executed_source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    choice={'sealed':True,'sealed_at_unix':time.time(),'candidate':candidate,'discovery_counts':counts,'discovery_case_sha256':discovery,
        'source_sha256':hashes,'checkpoint_sha256':checkpoints,'test_accessed':False,'original_comparison_seal_sha256':digest(DISC/'comparison_seal.json')}
    write(dest,choice);return choice


def case(seed):
    torch.set_num_threads(1);dest=OUT/'cases'/f'{seed}.json';choice=json.loads((OUT/'selection.json').read_text())
    if dest.exists():
        old=json.loads(dest.read_text())
        if old['source_sha256']!=source_hashes():raise RuntimeError('Confirmation case changed')
        return old
    began=time.time();n=P['robots'];duration=P['duration_s'];full,scan,g,obs=episode(seed,n,duration,'main')
    models={f:load(f) for f in ['hybrid_jepa','hybrid_supervised']}
    predictions={f:[[],[]] for f in ['learned_radiomap','hybrid_jepa','hybrid_supervised']}
    for d in range(duration):
        t=60+d
        if d%20==0:
            generic=fit_map(g,obs.before(t));strong=fit_strong_map(g,obs.before(t))
        b=inputs(g,obs.before(t),full.plan[t],generic)
        m,s=map_prediction(g,strong,full.plan[t]);predictions['learned_radiomap'][0].append(m);predictions['learned_radiomap'][1].append(s)
        for f,ms in models.items():
            m,s=predict(ms,b);predictions[f][0].append(m.reshape(n,9,-1,3));predictions[f][1].append(s.reshape(n,9,-1,3))
    old=examples(full,scan,duration,targets=False);weights=[READINGS[P['mission']]['weights'][k] for k in KEYS]
    m,s=forecast(load_models('jepa'),DeviceInputs(old['ctx'],old['route'],old['prior']),weights)
    scale=json.loads((ROOT/'artifacts/contracted/calibration.json').read_text())['jepa']['scale']
    predictions['previous_jepa']=[m.reshape(duration,n,9,-1,3),s.reshape(duration,n,9,-1,3)*scale]
    future=np.arange(60,60+duration)[:,None]+np.asarray(HORIZONS,dtype=int);truth=full.snr[future*5].transpose(0,2,1,3,4)
    rows=[];reading=READINGS[P['mission']]
    for f in P['arms']:
        m,s=[np.asarray(x,dtype='f4') for x in predictions[f]]
        policy=replace(intent_policy(reading['weights'],reading['focus'],dual_hardware='none'),name='confirmation_'+f,online_map=False,gate=True)
        result=simulate(runtime_world(full,duration,m,s[:,:,0]),policy,roam_gate=reactive_rescue)
        rows.append({'family':f,'gate':P['gate'],'result':result,'forecast_metrics':forecast_metrics(m,s,truth,g),'forecast_sha256':hashlib.sha256(m.tobytes()).hexdigest()})
        print(json.dumps({'confirmation_site':seed,'family':f,'misses':result['control_misses']}),flush=True)
    result={'completed':True,'scene':seed,'robots':n,'duration_s':duration,'rows':rows,'source_sha256':source_hashes(),
        'checkpoint_sha256':choice['checkpoint_sha256'],'no_truth_in_runtime_inputs':True,'elapsed_s':time.time()-began}
    write(dest,result);return result


def interval(values):
    v=np.asarray(values);rng=np.random.default_rng(290000);b=v[rng.integers(len(v),size=(20000,len(v)))].mean(1)
    return {'mean_difference':float(v.mean()),'paired_site_bootstrap_95':np.quantile(b,[.025,.975]).tolist(),'wins':int((v<0).sum()),'ties':int((v==0).sum()),'sites':len(v)}


def report():
    selected=json.loads((OUT/'selection.json').read_text());assert selected['source_sha256']==source_hashes()
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))];assert len(cases)==8
    arms={}
    for f in P['arms']:
        rows=[next(r for r in c['rows'] if r['family']==f) for c in cases];miss=sum(r['result']['control_misses'] for r in rows);sent=sum(r['result']['control_sent'] for r in rows)
        arms[f]={'misses':miss,'commands':sent,'ppm':miss/sent*1e6,'scene_ppm':[r['result']['control_miss']*1e6 for r in rows],
            'forecast_metrics':{k:float(np.mean([r['forecast_metrics'][k] for r in rows])) for k in rows[0]['forecast_metrics']},
            'traffic':{k:float(np.mean([r['result'][k] for r in rows])) for k in ['bulk_mbps','video_on_time','video_delivered_vs_requested','safety_stop_fraction']}}
        m=arms[f]['forecast_metrics'];m['positive_snr_rmse_db']=m.pop('audible_rmse_db');m['nonpositive_snr_rmse_db']=m.pop('unheard_rmse_db')
    pairs={}
    for a,b in [('hybrid_supervised','learned_radiomap'),('hybrid_jepa','learned_radiomap'),('hybrid_supervised','previous_jepa'),('hybrid_jepa','hybrid_supervised')]:
        pairs[a+'_vs_'+b]=interval(np.asarray(arms[a]['scene_ppm'])-arms[b]['scene_ppm'])
    main=pairs['hybrid_supervised_vs_learned_radiomap'];passed=main['paired_site_bootstrap_95'][1]<0
    summary={'completed':True,'built_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'discovery_was_used_for_candidate_selection':True,
        'independent_confirmation_sites':8,'native_runs':32,'fixed_candidate':selected['candidate'],'arms':arms,'comparisons_ppm':pairs,
        'primary_success':passed,'simulation_only':True,'original_study_preserved':True}
    write(OUT/'summary.json',summary)
    names={'learned_radiomap':'Learned-prior RadioMap','hybrid_supervised':'Supervised hybrid','hybrid_jepa':'Hybrid JEPA','previous_jepa':'Previous JEPA'}
    ci=main['paired_site_bootstrap_95'];a=arms['hybrid_supervised'];b=arms['learned_radiomap']
    lines=['# Independent warehouse confirmation','',
        f'On eight additional warehouses, the fixed supervised hybrid produces **{a["ppm"]:.2f}** missed commands per million versus **{b["ppm"]:.2f}** for learned-prior RadioMap. The paired difference is **{main["mean_difference"]:.2f} ppm**, 95% site-bootstrap interval **[{ci[0]:.2f}, {ci[1]:.2f}]**; wins {main["wins"]}/8.',
        'The preregistered primary reduction criterion '+('passes.' if passed else 'does not pass.'),'',
        'This candidate was chosen from the first eight warehouses, then frozen before these eight new sites. These results are separate from the original study and do not replace its development-selected standalone model. No model, gate or controller was retuned.','',
        '| Method | Misses | Misses / million | Forecast RMSE (dB) | AP-choice regret (dB) |','|---|---:|---:|---:|---:|']
    for f in P['arms']:
        r=arms[f];m=r['forecast_metrics'];lines.append(f'| {names[f]} | {r["misses"]} | {r["ppm"]:.2f} | {m["rmse_near_db"]:.2f} | {m["ap_choice_regret_db"]:.3f} |')
    lines+=['','![Independent confirmation](../artifacts/fleet_confirmation/figures/confirmation.png)','',
        '100 robots per site, 120 seconds per run, 1,920,000 commands per arm. All four arms use the same 30 ms native handover and reactive-rescue controller.','',
        '## Interpretation','',
        'Mean-field forecast accuracy, control outcomes and the value of the JEPA objective are distinct questions. The supervised hybrid is the primary candidate; the JEPA hybrid and previous model are secondary comparisons. All paired comparisons, traffic metrics and interval coverage are in `summary.json`. A win for the supervised hybrid does not establish that JEPA pretraining is necessary.','',
        '## Reproduction and limits','',
        'Run `uv run --frozen python -m wirelessjev.fleet_confirmation --workers 4`. Source capsules, frozen selection, checkpoint hashes, discovery-case hashes and all confirmation counters are archived in `artifacts/fleet_confirmation`. Audit with `uv run --frozen python scripts/audit_fleet_confirmation.py`.','',
        'Simulation only, canonical warehouse layout with varied propagation/routes/traffic, idealised reactive observations, known localisation/CAD and a 30 ms handover assumption. The models use synthetic mean-field supervision and the hybrid still requires RadioMap fitting. Forecast error includes planned-versus-realised route mismatch; policies share precomputed trajectories. Physical scanning, on-car inference and stock-car switching remain unqualified.']
    (ROOT/'docs/FLEET_CONFIRMATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(4)
    axes[0].bar(x,[arms[f]['ppm'] for f in P['arms']],color=['#3e79ad','#926eb0','#12a28e','#d3aa73'])
    for c in cases:
        axes[0].scatter(x,[next(r['result']['control_miss']*1e6 for r in c['rows'] if r['family']==f) for f in P['arms']],s=14,color='#173243',alpha=.6)
    axes[0].set_xticks(x,[names[f] for f in P['arms']],rotation=15,ha='right');axes[0].set_ylabel('Missed commands / million');axes[0].set_title('Eight independent confirmation warehouses')
    axes[1].bar(x,[arms[f]['forecast_metrics']['rmse_near_db'] for f in P['arms']],color=['#3e79ad','#926eb0','#12a28e','#d3aa73'])
    axes[1].set_xticks(x,[names[f] for f in P['arms']],rotation=15,ha='right');axes[1].set_ylabel('Mean-field forecast RMSE (dB)');axes[1].set_title('Forecast accuracy is a separate outcome')
    fig.tight_layout();fig.savefig(figdir/'confirmation.png',dpi=180);fig.savefig(figdir/'confirmation.pdf');plt.close(fig)
    analysis=ROOT/'scripts/audit_fleet_confirmation.py';dest=OUT/'analysis_source/scripts/audit_fleet_confirmation.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(analysis,dest)
    files={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(OUT/'manifest.json',{'completed':True,'files_sha256':files,'source_sha256':selected['source_sha256'],'checkpoint_sha256':selected['checkpoint_sha256'],
        'discovery_manifest_sha256':digest(DISC/'manifest.json'),'preregistration_commit':'57321e4','simulation_only':True})
    print(json.dumps(summary,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    selected=seal();assert selected['source_sha256']==source_hashes()
    for name,h in selected['checkpoint_sha256'].items():assert digest(ROOT/name)==h
    if not (OUT/'test_opened.json').exists():write(OUT/'test_opened.json',{'time_unix':time.time(),'selection_sha256':digest(OUT/'selection.json')})
    with ProcessPoolExecutor(args.workers) as pool:list(pool.map(case,P['test_sites']))
    report();print('INDEPENDENT_CONFIRMATION_COMPLETE',flush=True)


if __name__=='__main__':main()
