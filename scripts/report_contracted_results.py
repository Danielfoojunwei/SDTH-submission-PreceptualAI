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
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Fixed split','Bayesian gate','JEPA no gate','JEPA gate','Regret JEPA','No-route regret','Scratch gate'];aa=['fixed','bayesian_gated_base','jepa_ungated_base','jepa_gated_base','regret_gated_base','no_route_gated_base','scratch_gated_base']
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(aa));vals=[main[a]['ppm'] for a in aa];ax.bar(x,vals,color=['#8a999c','#447fa5','#e6a26e','#24a99a','#9263b6','#afbad0','#aaa15c'])
    for c in primary:ax.scatter(x,[row(c,a)['control_miss']*1e6 for a in aa],c='#183142',s=12,alpha=.5)
    ax.set_xticks(x,names,rotation=15,ha='right');ax.set_ylabel('Control misses per million commands');ax.set_title('Primary six paired scenes · 100 robots · 120 seconds · simulation');fig.tight_layout();fig.savefig(figdir/'primary.pdf');fig.savefig(figdir/'primary.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4));kinds=list(uncertainty)
    axes[0].bar(kinds,[uncertainty[k]['control_ppm'] for k in kinds],color='#447fa5');axes[0].set_ylabel('Control misses / million');axes[0].set_title('Identical forecasts, changed uncertainty')
    axes[1].plot(kinds,[uncertainty[k]['coverage_90'] for k in kinds],'o-',label='90% interval coverage');axes[1].plot(kinds,[uncertainty[k]['ap_score_error_sigma_spearman'] for k in kinds],'s-',label='AP error / sigma Spearman');axes[1].legend(fontsize=8);axes[1].axhline(.9,c='gray',ls=':',lw=1);fig.tight_layout();fig.savefig(figdir/'uncertainty.pdf');fig.savefig(figdir/'uncertainty.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for a,label in [('bayesian_gated_base','Bayesian'),('jepa_gated_base','JEPA'),('regret_gated_base','Regret JEPA'),('no_route_gated_base','No-route regret'),('scratch_gated_base','Scratch')]:
        ax.scatter(main[a]['rmse_near_db'],main[a]['ppm'],s=60);ax.annotate(label,(main[a]['rmse_near_db'],main[a]['ppm']),xytext=(5,5),textcoords='offset points')
    ax.set_xlabel('Near-horizon all-link SNR RMSE (dB)');ax.set_ylabel('Control misses / million');ax.set_title('Prediction error and control are separate endpoints');fig.tight_layout();fig.savefig(figdir/'forecast_control.pdf');fig.savefig(figdir/'forecast_control.png',dpi=180);plt.close(fig)
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
    files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['manifest.json']}
    write(OUT/'manifest.json',{'completed':True,'files_sha256':files,'fit_count':20,'native_rollouts':366,'preregistration_commit':'2c2cc59','pre_test_amendment_commit':'a2e4162','simulation_only':True})
    print(json.dumps({'paper1':summary['paper1'],'paper2':summary['paper2'],'primary':{a:round(main[a]['ppm'],2) for a in main}},indent=2))


if __name__=='__main__':main()
