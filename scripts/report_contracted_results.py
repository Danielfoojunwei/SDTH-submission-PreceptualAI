"""Summarise completed experiments without changing models or selecting policies."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wirelessjev.contracted.train import OUT,RUN,write,load_models
from wirelessjev.contracted.data import ROOT,PROTOCOL,KEYS,READINGS,examples,make_stream
from wirelessjev.contracted.theory import rank_counterexample


def interval(values):
    values=np.asarray(values);rng=np.random.default_rng(330000);draw=values[rng.integers(len(values),size=(10000,len(values)))].mean(-1)
    return {'mean':float(values.mean()),'paired_scene_bootstrap_95':np.quantile(draw,[.025,.975]).tolist(),'n_independent_scenes':len(values)}


def member_interval(matrix):
    m=np.asarray(matrix);rng=np.random.default_rng(330001);rows=rng.integers(m.shape[0],size=(10000,m.shape[0]));cols=rng.integers(m.shape[1],size=(10000,m.shape[1]));draw=m[rows[:,:,None],cols[:,None,:]].mean((1,2))
    return {'mean':float(m.mean()),'crossed_scene_training_seed_bootstrap_95':np.quantile(draw,[.025,.975]).tolist(),'scenes':m.shape[0],'training_seeds':m.shape[1]}


def markdown(s):
    p=s['primary'];p1=s['paper1'];p2=s['paper2'];delta=p2['regret_vs_jepa'];seed=p2['five_seed_paired_comparison']
    names={'fixed':'Fixed split','bayesian_ungated_base':'Bayesian, ungated','bayesian_gated_base':'Bayesian, gated','jepa_ungated_base':'JEPA, ungated','jepa_gated_base':'JEPA, gated','regret_gated_base':'Regret JEPA, gated','no_route_gated_base':'No-route regret, gated','scratch_ungated_base':'Scratch, ungated','scratch_gated_base':'Scratch, gated'}
    def ci(values):return '['+', '.join(f'{v:.3f}' for v in values)+']'
    lines=['# Contracted world-model study: completed results','',
        'All **20 model fits, 24 fresh scenes and 366 native rollouts** completed. This is simulation evidence. Completing the implementation does not establish the proposed papers’ novelty or hypotheses.','',
        'The contract was not necessary for JEPA to beat the fixed rule; uncertainty scale changed control despite identical rankings; regret training did not establish a robust control advantage. RadioMap remains the default.','',
        'The model, protocol, derivation and literature review are in [CONTRACTED_MODEL.md](CONTRACTED_MODEL.md), [CONTRACTED_WORLD_MODELS.md](CONTRACTED_WORLD_MODELS.md), [CONTRACTED_THEORY.md](CONTRACTED_THEORY.md) and [CONTRACTED_PRIOR_ART.md](CONTRACTED_PRIOR_ART.md). The default native system remains RadioMap + WirelessJEV.','',
        '## What was built and executed','',
        '- A causal six-scan context, explicit heard/censored/missing masks, future-route conditioning, EMA target encoder and per-link Gaussian head at the nine native horizons.',
        '- Five independently trained seeds each for JEPA, regret-weighted JEPA, no-route regret and scratch: 24 randomised training twins, six development twins, twelve epochs per fit. Checkpoints are selected using development censored likelihood only.',
        '- A five-member uncertainty ensemble with development-only residual scaling; typed inference inputs exclude future observations and truth labels. An adapter supplies the existing forecast and uncertainty fields without changing the planner.',
        '- Matched gate/no-gate runs, four uncertainty-only interventions, three-way objective comparison, five-seed control robustness, held-out mission readings, route ablation, 10-robot/stress/yard generalisation and a verified literature pass.','',
        '## Primary endpoint: six paired warehouse scenes','',
        '100 robots × 120 seconds × 20 commands/second = **240,000 commands per scene and 1,440,000 per arm**. Packets share scene conditions and are not independent replicates. All rows use the bulk-upload mission reading.','',
        '| Arm | Missed / sent | Misses per million | Near-horizon all-link RMSE (dB) | 90% interval coverage |',
        '|---|---:|---:|---:|---:|']
    for arm,label in names.items():
        r=p[arm];rmse=f'{r["rmse_near_db"]:.2f}' if 'rmse_near_db' in r else '—';coverage=f'{100*r["coverage_90"]:.1f}%' if 'coverage_90' in r else '—'
        lines.append(f'| {label} | {r["misses"]} / {r["commands"]:,} | {r["ppm"]:.2f} | {rmse} | {coverage} |')
    lines+=['','![Primary six-scene comparison](../artifacts/contracted/figures/primary.png)','',
        'The Bayesian reference pools a 60-second fleet survey. Each JEPA inference sees six recent scans from one robot, plus known geometry/route and cached intent weights. This is a strong-system reference with unequal new-site information budgets; the JEPA versus regret and gate versus no-gate contrasts are matched.','',
        '## Paper 1: test the contract, do not assume necessity','',
        f'Gated JEPA beats the fixed rule in **{p1["contracted_vs_fixed"]["candidate_wins"]}/6** scenes. Ungated JEPA also beats it in **{p1["uncontracted_vs_fixed"]["candidate_wins"]}/6**. The gated-minus-ungated difference is {p1["contracted_vs_uncontracted"]["mean"]:.2f} misses/million; its six-scene paired bootstrap interval is {ci(p1["contracted_vs_uncontracted"]["paired_scene_bootstrap_95"])}. Lower is better.',
        '', 'The proposed necessity claim is '+('**rejected**: ungated JEPA already wins on at least five scenes.' if p1['ungated_already_wins_5_of_6'] else '**not established** by an advantage against one fixed rule.'),'',
        'The uncertainty interventions below hold mean forecasts bit-identical. Half and double preserve rankings exactly, so changed control outcomes cannot be attributed to changed forecast RMSE or uncertainty ordering.','',
        '| Uncertainty | Misses / million | 90% coverage | AP error–spread Spearman | Changed miss-count scenes |',
        '|---|---:|---:|---:|---:|']
    for kind,r in s['uncertainty_interventions'].items():lines.append(f'| {kind} | {r["control_ppm"]:.2f} | {100*r["coverage_90"]:.1f}% | {r["ap_score_error_sigma_spearman"]:.3f} | {r["changed_miss_count_scenes"]}/6 |')
    lines+=['','![Uncertainty-only interventions](../artifacts/contracted/figures/uncertainty.png)','',
        '**Rank correlation alone is insufficient.** The executable two-decision counterexample proves this even at perfect ranking. The derived gain-envelope bound requires coverage assumptions; its empirical audit is not an ex-ante safety guarantee. Multiplying uncertainty by 0.5 is a diagnostic intervention, not a policy selected for deployment from these test results.','',
        '## Paper 2: decision-regret training','',
        f'The ensemble regret-minus-JEPA difference is **{delta["mean"]:.3f} misses/million**, with paired-scene 95% bootstrap interval {ci(delta["paired_scene_bootstrap_95"])}. It improves on {delta["candidate_wins"]}/6 scenes. The crossed five-training-seed/six-scene comparison is **{seed["mean"]:.3f}**, interval {ci(seed["crossed_scene_training_seed_bootstrap_95"])}.',
        '', 'These are separate uncertainty summaries: one compares the deployed five-member ensembles across scenes; the other pairs individual trained members and resamples both scene and training-seed axes. A tiny mean difference or an interval spanning zero does not establish a robust control advantage.','',
        'The loss uses finite native AP-score regret, directly involving roam and load intent knobs. It is not an exact cross-layer deadline-loss objective. Other intent dimensions condition the predictor/controller, but complete named-robot packet utility is not the training oracle.','',
        '![Forecast error and control](../artifacts/contracted/figures/forecast_control.png)','',
        '## Generalisation, held-out missions and routes','',
        '| Condition | Fixed ppm | Bayesian gate ppm | JEPA ungated ppm | JEPA gated ppm | Regret gated ppm |',
        '|---|---:|---:|---:|---:|---:|']
    for condition,r in s['generalisation'].items():lines.append('| '+condition+' | '+' | '.join(f'{r["arms"][a]["ppm"]:.2f}' for a in ['fixed','bayesian_gated_base','jepa_ungated_base','jepa_gated_base','regret_gated_base'])+' |')
    lines+=['', 'Each generalisation panel has six fresh scenes at 60 seconds: 10 robots for `small`, 100 for `stress` and `yard`. The yard has six AP sites and perimeter routes with separate analytic propagation; it is another synthetic environment, not an outdoor measurement.','',
        '| Held-out cached mission | Bayesian gate ppm | JEPA gate ppm | Regret gate ppm |',
        '|---|---:|---:|---:|']
    for mission,r in s['heldout_missions'].items():lines.append('| '+mission+' | '+' | '.join(f'{r["arms"][a]["ppm"]:.2f}' for a in ['bayesian_gated_base','jepa_gated_base','regret_gated_base'])+' |')
    lines+=['', f'The route-conditioned regret family minus its no-route ablation is {s["route_ablation"]["mean"]:.2f} misses/million, interval {ci(s["route_ablation"]["paired_scene_bootstrap_95"])}. The no-route ablation is a separately trained family, not a test-time mask.', '',
        'The two held-out readings were excluded from world-model training. Four cached d3 readings do not demonstrate arbitrary new mission-language transfer.','',
        '## Literature and claim readiness','',
        'Eighteen open full texts were retrieved and their relevant methods inspected. Uncertainty-based learned/conventional switching already appears in IEEE TVT (open preprint 2401.13206); Telecom World Models includes intent, uncertainty, guardrails and fallback; coupled robot/wireless JEPA models appear in 2609.04851. Decision-focused losses, value-equivalent models and selective prediction also have established prior art.', '',
        'Therefore the broad architecture/contract novelty claim is not supported. The direct IEEE index page returned HTTP 418, so an exhaustive authenticated index search is not represented as complete. The reviewed open sources are enough to demonstrate substantial overlap.','',
        '## Boundaries and audit','']
    lines.extend('- '+x+'.' for x in s['limitations'])
    lines+=['', 'Raw reports, checkpoints, calibration, source capsules, figures and source/output hashes are in [artifacts/contracted](../artifacts/contracted/). [summary.json](../artifacts/contracted/summary.json) contains every aggregate, interval and seed matrix. Audit without retraining:', '', '```sh','uv run --frozen python scripts/audit_contracted_results.py','```','',
        'The archive audit checks all 20 finite, distinct fits, checkpoint selection, all 366 raw counter reports, disjoint scene splits, source/output hashes and forecast invariance during the uncertainty interventions. No simulation was relabelled as a hardware result.', '',
        'The raw native `sigma_map` field records the base generator’s synthetic map-error injection setting, not learned forecast accuracy. Use `forecast_metrics` and the table above for actual learned error and coverage.']
    (ROOT/'docs/CONTRACTED_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    if not (OUT/'evaluation.json').exists():raise RuntimeError('Evaluation has not completed')
    completion=json.loads((OUT/'evaluation.json').read_text());assert completion['completed']
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('*.json'))]
    assert len(cases)==24 and sum(len(c['rows']) for c in cases)==366
    for c in cases:assert len(c['rows'])==(34 if c['condition']=='main' else 9)
    def row(c,arm,mission='bulk_uploads'):return next(r for r in c['rows'] if r['arm']==arm and r['mission']==mission)
    def metric(c,arm,mission='bulk_uploads'):return next(r for r in c['forecast_metrics'] if r['arm']==arm and r['mission']==mission)
    primary=[c for c in cases if c['condition']=='main']
    arms=['fixed','bayesian_ungated_base','bayesian_gated_base','jepa_ungated_base','jepa_gated_base','regret_gated_base','no_route_gated_base','scratch_ungated_base','scratch_gated_base']
    def arm_report(group,arm,mission='bulk_uploads'):
        rs=[row(c,arm,mission) for c in group];r=interval([x['control_miss']*1e6 for x in rs]);r.update({'misses':sum(x['control_misses'] for x in rs),'commands':sum(x['control_sent'] for x in rs),'ppm':sum(x['control_misses'] for x in rs)/sum(x['control_sent'] for x in rs)*1e6})
        if arm!='fixed':
            family=rs[0]['family'];ms=[metric(c,family+'_gated_base',mission) for c in group]
            for name in ['rmse_near_db','coverage_90','finite_ap_choice_regret','reported_link_rmse_near_db','unreported_link_rmse_near_db']:r[name]=float(np.mean([m[name] for m in ms]))
        return r
    main={a:arm_report(primary,a) for a in arms}
    def compare(a,b,group=primary,mission='bulk_uploads'):
        dif=[(row(c,a,mission)['control_miss']-row(c,b,mission)['control_miss'])*1e6 for c in group]
        return interval(dif)|{'candidate_wins':int(sum(d<0 for d in dif)),'ties':int(sum(d==0 for d in dif)),'comparison':a+' minus '+b,'unit':'misses per million'}
    contracted=compare('jepa_gated_base','fixed');uncontracted=compare('jepa_ungated_base','fixed');objective=compare('regret_gated_base','jepa_gated_base')
    seed_matrix=[[(row(c,f'regret_gated_base_seed{s}')['control_miss']-row(c,f'jepa_gated_base_seed{s}')['control_miss'])*1e6 for s in range(5)] for c in primary]
    uncertainty={}
    for kind in ['base','half','double','noise','shuffle']:
        arm='jepa_gated_'+kind;ms=[metric(c,arm) for c in primary];rs=[row(c,arm) for c in primary]
        assert all(metric(c,arm)['mean_forecast_sha256']==metric(c,'jepa_gated_base')['mean_forecast_sha256'] for c in primary)
        uncertainty[kind]={'control_ppm':float(np.mean([r['control_miss']*1e6 for r in rs])),'roams':int(sum(r['roams'] for r in rs)),'coverage_90':float(np.mean([m['coverage_90'] for m in ms])),
            'ap_score_error_sigma_spearman':float(np.mean([m['ap_score_error_sigma_spearman'] for m in ms])),'link_error_sigma_spearman':float(np.mean([m['link_error_sigma_spearman'] for m in ms])),
            'changed_miss_count_scenes':int(sum(row(c,arm)['control_misses']!=row(c,'jepa_gated_base')['control_misses'] for c in primary)),
            'all_empirical_theory_bounds_hold':all(m['theory_audit']['bound_holds'] for m in ms),'mean_empirical_theory_lower_bound':float(np.mean([m['theory_audit']['empirical_lower_bound'] for m in ms]))}
    general={condition:{'arms':{a:arm_report([c for c in cases if c['condition']==condition],a) for a in arms},'gate_vs_fixed':compare('jepa_gated_base','fixed',[c for c in cases if c['condition']==condition]),'gate_vs_ungated':compare('jepa_gated_base','jepa_ungated_base',[c for c in cases if c['condition']==condition])} for condition in ['small','stress','yard']}
    shifted={mission:{'arms':{a:arm_report(primary,a,mission) for a in ['bayesian_gated_base','jepa_gated_base','regret_gated_base']},'regret_vs_jepa':compare('regret_gated_base','jepa_gated_base',mission=mission)} for mission in PROTOCOL['heldout_missions']}
    summary={'completed':True,'built_at':datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),'fits':20,'test_scenes':24,'native_rollouts':366,'primary':main,
        'paper1':{'contracted_vs_fixed':contracted,'uncontracted_vs_fixed':uncontracted,'contracted_vs_uncontracted':compare('jepa_gated_base','jepa_ungated_base'),'proposed_5_of_6_gate_claim_passes':contracted['candidate_wins']>=5,'ungated_already_wins_5_of_6':uncontracted['candidate_wins']>=5},
        'uncertainty_interventions':uncertainty,'forecasts_identical_under_uncertainty_corruption':True,'rank_only_guarantee':'Refuted mathematically; see counterexample and calibrated-envelope theorem','rank_counterexample':rank_counterexample(),
        'paper2':{'regret_vs_jepa':objective,'five_seed_paired_comparison':member_interval(seed_matrix),'seed_scene_difference_ppm':seed_matrix,'regret_vs_bayesian':compare('regret_gated_base','bayesian_gated_base'),'worse_forecast_better_control_tradeoff':bool(main['regret_gated_base']['rmse_near_db']>main['jepa_gated_base']['rmse_near_db'] and objective['mean']<0)},
        'route_ablation':compare('regret_gated_base','no_route_gated_base'),'generalisation':general,'heldout_missions':shifted,
        'limitations':['Simulation only; no physical car/RAN testbed','24 randomised training sites and six development sites; not broad vertical pretraining','One five-member ensemble per family; the primary JEPA/regret comparison also crosses five individual training seeds','Bayesian baseline pools 60 seconds of fleet scans; each JEPA context is six recent scans from one robot','Native reactive fallback retains its idealised uncapped measurement model','Finite AP-score regret is not exact packet-loss regret','Four cached d3 mission readings; held-out weights do not establish general semantic transfer','Marginal residual calibration does not certify conditional gain coverage','One-step theorem is not a cumulative packet-delivery or safety certificate','Selective fallback and intent-aware world-model planning already exist in prior art']}
    write(OUT/'summary.json',summary)
    markdown(summary)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Fixed split','Bayes no gate','Bayes gate','JEPA no gate','JEPA gate','Regret gate','No-route regret','Scratch no gate','Scratch gate'];aa=arms
    fig,ax=plt.subplots(figsize=(12,4.8));x=np.arange(len(aa));vals=[main[a]['ppm'] for a in aa];ax.bar(x,vals,color=['#8a999c','#b0c4d4','#447fa5','#e6a26e','#24a99a','#9263b6','#afbad0','#d5d29c','#aaa15c'])
    for c in primary:ax.scatter(x,[row(c,a)['control_miss']*1e6 for a in aa],c='#183142',s=12,alpha=.5)
    ax.set_xticks(x,names,rotation=15,ha='right');ax.set_ylabel('Control misses per million commands');ax.set_title('Primary six paired scenes · 100 robots · 120 seconds · simulation');fig.tight_layout();fig.savefig(figdir/'primary.pdf');fig.savefig(figdir/'primary.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));kinds=list(uncertainty)
    axes[0].bar(kinds,[uncertainty[k]['control_ppm'] for k in kinds],color='#447fa5');axes[0].set_ylabel('Control misses / million');axes[0].set_title('Identical forecasts, changed uncertainty')
    axes[1].plot(kinds,[uncertainty[k]['coverage_90'] for k in kinds],'o-',label='90% interval coverage');axes[1].plot(kinds,[uncertainty[k]['ap_score_error_sigma_spearman'] for k in kinds],'s-',label='AP error / sigma Spearman');axes[1].legend(fontsize=8);axes[1].axhline(.9,c='gray',ls=':',lw=1);fig.tight_layout();fig.savefig(figdir/'uncertainty.pdf');fig.savefig(figdir/'uncertainty.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for a,label,offset in [('bayesian_gated_base','Bayesian',(7,-14)),('jepa_gated_base','JEPA',(7,8)),('regret_gated_base','Regret JEPA',(-80,-18)),('no_route_gated_base','No-route regret',(7,7)),('scratch_gated_base','Scratch',(7,7))]:
        ax.scatter(main[a]['rmse_near_db'],main[a]['ppm'],s=60);ax.annotate(label,(main[a]['rmse_near_db'],main[a]['ppm']),xytext=offset,textcoords='offset points')
    ax.margins(.18);ax.set_xlabel('Near-horizon all-link SNR RMSE (dB)');ax.set_ylabel('Control misses / million');ax.set_title('Prediction error and control are separate endpoints');fig.tight_layout();fig.savefig(figdir/'forecast_control.pdf');fig.savefig(figdir/'forecast_control.png',dpi=180);plt.close(fig)
    # Keep unchanged import dependencies portable in addition to the executed capsules.
    deps=['src/wirelessjev/warehouse_v3/jepa.py','third_party/vjepa2/SOURCE.json','third_party/vjepa2/LICENSE','pyproject.toml','uv.lock']
    dependency_hashes={}
    for name in deps:
        p=ROOT/name
        if not p.exists():continue
        dest=OUT/'dependencies'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);dependency_hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['training.data.json','development.data.json']:
        if (RUN/name).exists():shutil.copyfile(RUN/name,OUT/name)
    import importlib.metadata
    write(OUT/'dependency_source.json',{'dependency_sha256':dependency_hashes,'versions':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','matplotlib','httpx','pymupdf']},'note':'EMA helper is unchanged from execution. The lock also declares literature/report dependencies added after training; training code and exact versions have separate receipts.'})
    for report in (OUT/'fits').glob('*.json'):
        r=json.loads(report.read_text());ckpt=OUT/'checkpoints'/f'{r["family"]}-{r["seed"]}.pt';assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==r['checkpoint_sha256']
        state=torch.load(ckpt,map_location='cpu',weights_only=True)['state'];assert all(torch.isfinite(v).all() for v in state.values())
    analysis_paths=['scripts/report_contracted_results.py','scripts/audit_contracted_results.py','scripts/audit_contracted_prior_art.py','docs/CONTRACTED_THEORY.md']
    analysis_hashes={}
    for name in analysis_paths:
        dest=OUT/'analysis_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);analysis_hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write(OUT/'analysis_source.json',{'source_sha256':analysis_hashes})
    # Include the entire Python import tree, not just the critical-file receipts.
    # The fit/evaluation capsules remain immutable; verify their executed core
    # still matches before supplementing it with unchanged legacy dependencies.
    for stage in ['training','evaluation']:
        receipt=json.loads((OUT/(stage+'_source.json')).read_text())
        for relative,expected in receipt['source_sha256'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,('Executed source changed',stage,relative)
    complete_paths=list((ROOT/'src/wirelessjev').rglob('*.py'))+[ROOT/'pyproject.toml',ROOT/'uv.lock',OUT/'protocol.json',OUT/'evaluation_amendment.json',ROOT/'artifacts/native/intent_flip.json',ROOT/'artifacts/warehouse_v2/studies/wifi_link_curves.json']
    complete_hashes={}
    for path in complete_paths:
        relative=str(path.relative_to(ROOT));dest=OUT/'complete_source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);complete_hashes[relative]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write(OUT/'complete_source.json',{'source_sha256':complete_hashes,'legacy_dependency_commit':'998fe3736a28d30cf04931e9ef182bda8cd4caf8','executed_core_matches_immutable_capsules':True})
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['manifest.json']}
    write(OUT/'manifest.json',{'completed':True,'files_sha256':files,'fit_count':20,'native_rollouts':366,'preregistration_commit':'2c2cc59','pre_test_amendment_commit':'a2e4162','simulation_only':True})
    print(json.dumps({'paper1':summary['paper1'],'paper2':summary['paper2'],'primary':{a:round(main[a]['ppm'],2) for a in main}},indent=2))


if __name__=='__main__':main()
