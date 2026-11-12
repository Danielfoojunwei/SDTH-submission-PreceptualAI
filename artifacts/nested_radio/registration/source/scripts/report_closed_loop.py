"""Recompute and independently audit the delivered-telemetry evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/closed_loop'
METHOD={'radiomap':'RadioMap','distilled':'Compact world model'}
VARIANT={'ideal_reports':'Ideal model reports','delivered_reports':'Delivered model reports','delivered_utility':'Delivered + outage utility'}


def interval(comparison):
    lo,hi=comparison['paired_site_bootstrap_95_ppm']
    return f'[{lo:.2f}, {hi:.2f}] ppm'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a,b):
    d=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(322000)
    boot=d[rng.integers(len(d),size=(20000,len(d)))].mean(1)
    return {'mean_difference_ppm':float(d.mean()),'paired_site_bootstrap_95_ppm':np.quantile(boot,[.025,.975]).tolist(),
            'wins':int((d<0).sum()),'ties':int((d==0).sum()),'sites':len(d)}


def aggregate(cases,protocol):
    groups={}
    for cost in protocol['handover_seconds']:
        arms={}
        for family in protocol['models']:
            arms[family]={}
            for variant in protocol['test_variants']:
                rows=sorted([c for c in cases if c['family']==family and c['variant']==variant and c['handover_seconds']==cost],key=lambda r:r['scene'])
                assert [r['scene'] for r in rows]==protocol['test_sites']
                rs=[r['result'] for r in rows]
                miss=sum(r['control_misses'] for r in rs);sent=sum(r['control_sent'] for r in rs)
                arms[family][variant]={'misses':miss,'commands':sent,'ppm':miss/sent*1e6,'percent':miss/sent*100,
                    'scene_ppm':[r['control_miss']*1e6 for r in rs],
                    'mean_roams':float(np.mean([r['roams'] for r in rs])),
                    'bulk_mbps':float(np.mean([r['bulk_mbps'] for r in rs])),
                    'video_on_time':float(np.mean([r['video_on_time'] for r in rs])),
                    'safety_stop_fraction':float(np.mean([r['safety_stop_fraction'] for r in rs])),
                    'forecast_rmse_db':float(np.sqrt(np.mean([r['forecast']['rmse_db']**2 for r in rows]))),
                    'current_rmse_db':float(np.sqrt(np.mean([r['forecast']['current_rmse_db']**2 for r in rows]))),
                    'route_delta_rmse_db':float(np.sqrt(np.mean([r['forecast']['route_delta_rmse_db']**2 for r in rows]))),
                    'mean_reports_delivered':float(np.mean([r['telemetry']['reports_delivered'] for r in rows])),
                    'mean_report_age_seconds':float(np.mean([r['telemetry']['mean_delivery_age_seconds'] for r in rows])),
                    'miss_causes':{k:sum(r['miss_causes'][k] for r in rs) for k in rs[0]['miss_causes']}}
        within={f:paired(arms[f]['delivered_utility']['scene_ppm'],arms[f]['delivered_reports']['scene_ppm']) for f in protocol['models']}
        feedback={f:paired(arms[f]['delivered_reports']['scene_ppm'],arms[f]['ideal_reports']['scene_ppm']) for f in protocol['models']}
        model=paired(arms['distilled']['delivered_utility']['scene_ppm'],arms['radiomap']['delivered_utility']['scene_ppm'])
        groups[str(cost)]={'arms':arms,'utility_comparisons':within,'feedback_comparisons':feedback,'model_comparison':model}
    primary=groups['4.0']['utility_comparisons']['distilled']
    return {'completed':True,'fresh_sites':len(protocol['test_sites']),'development_runs':24,'test_runs':len(cases),'native_runs':24+len(cases),
            'conditions':groups,'primary_comparison':primary,'primary_success':primary['paired_site_bootstrap_95_ppm'][1]<0,
            'simulation_only':True,'no_training_or_test_tuning':True}


def figures(summary):
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    colors=['#a8b3c5','#e4a85a','#2b9b88']
    for ax,(cost,group) in zip(axes,summary['conditions'].items()):
        x=np.arange(2)
        for j,variant in enumerate(VARIANT):
            values=[group['arms'][f][variant]['percent'] for f in METHOD]
            ax.bar(x+(j-1)*.24,values,width=.22,label=VARIANT[variant],color=colors[j])
            for i,family in enumerate(METHOD):
                points=np.asarray(group['arms'][family][variant]['scene_ppm'])/10000
                ax.scatter(i+(j-1)*.24+np.linspace(-.055,.055,len(points)),points,s=8,color='#223344',alpha=.65,zorder=3)
        ax.set_xticks(x,list(METHOD.values()));ax.set_ylabel('Commands missing deadline (%)')
        ax.set_title(f'{float(cost):g} s assumed handover');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('100 simulated robots • native packet-delivery feedback • frozen models',fontsize=12)
    fig.tight_layout();fig.savefig(dest/'delivered_control.png',dpi=180);plt.close(fig)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig,ax=plt.subplots(figsize=(12,4.5));ax.set(xlim=(0,12),ylim=(0,4.5));ax.axis('off')
    boxes=[(.1,2.6,'Warehouse / channel','Movement, fading, blockage'),(3.2,2.6,'Device scans','Top 8 noisy RSSI; charged outage'),
           (6.3,2.6,'Control exchange','Report only after packet delivery'),(9.4,2.6,'Fleet memory','Strictly past delivery bins'),
           (9.4,.55,'Frozen forecaster','Compact model or RadioMap'),(6.3,.55,'Forecast FIFO','Measured CPU + 500 ms; aged outputs'),
           (3.2,.55,'Outage utility','Stay loss vs switch outage + loss'),(.1,.55,'Link controller','Native packets and local pilots')]
    for x,y,title,detail in boxes:
        ax.add_patch(FancyBboxPatch((x,y),2.5,1.05,boxstyle='round,pad=.07',facecolor='#e5f0f4',edgecolor='#27637b'))
        ax.text(x+1.25,y+.72,title,ha='center',fontweight='bold',fontsize=10)
        ax.text(x+1.25,y+.28,detail,ha='center',fontsize=7.8)
    paths=[((2.7,3.1),(3.1,3.1)),((5.8,3.1),(6.2,3.1)),((8.9,3.1),(9.3,3.1)),
           ((10.65,2.5),(10.65,1.7)),((9.3,1.1),(8.9,1.1)),((6.2,1.1),(5.8,1.1)),((3.1,1.1),(2.7,1.1)),((1.35,1.7),(1.35,2.5))]
    for a,b in paths:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#27637b',lw=1.7))
    ax.text(6,4.1,'Delivered observation → forecast → physical action → delivered observation',ha='center',fontsize=13,fontweight='bold')
    fig.tight_layout();fig.savefig(dest/'feedback_architecture.png',dpi=180);plt.close(fig)


def report():
    protocol=json.loads((OUT/'protocol.json').read_text());completion=json.loads((OUT/'completion.json').read_text())
    assert completion['completed']
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'cases').glob('test-*.json'))]
    summary=aggregate(cases,protocol)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figures(summary)
    choice=json.loads((OUT/'selection.json').read_text())['selected']
    lines=['# Delivered-telemetry warehouse confirmation','',
        'Eight untouched seeds, 100 simulated robots each, 60 seconds per run. Four stationary and four drifting mean fields share the existing CAD family. Twenty-four development runs selected one shared utility horizon; 96 fresh test runs compare two frozen forecasters, three configurations and two handover costs. No additional models were fitted. The existing bulk-upload mission fixes controller knobs; mission readers and original RadioJEV E1–E5 weights are unchanged.','',
        '![Closed feedback architecture](../artifacts/closed_loop/figures/feedback_architecture.png)','',
        '## What is now coupled','',
        'Only actual cadence-triggered scans produce runtime observations. RSSI includes the same hidden channel process as packet transmission plus receiver noise. Local neighbor knowledge uses completed scans; pilots freeze during outages. Reports reach fleet memory only after a successful control-frame delivery and become usable in a strictly later decision bin. The forecaster never receives a hidden channel array. Missing scans do not become invented below-threshold readings.','',
        'A reserved 128 bytes in each existing 600-byte control frame carries at most eight scan entries and pose metadata. Total simulated packet size is unchanged; application payload capacity is correspondingly reduced. This assumes a telemetry-capable bidirectional control exchange and does not measure separate uplink ACK loss. Initial history is a shared conditional warm start; positions and routes remain out-of-band inputs.','',
        f'One shared **{choice["horizon_seconds"]}-second** utility horizon was selected using both models on four separate development sites. Utility integrates the frozen PHY failure surrogate over the route and includes the complete physical outage. At fractional endpoints it includes the remaining partial interval. Both planned and reactive moves require admission. The surrogate omits MAC/queue uncertainty; it is not a calibrated deadline probability.','',
        '![Deadline loss by configuration](../artifacts/closed_loop/figures/delivered_control.png)','']
    for cost,group in summary['conditions'].items():
        lines += [f'## Assumed handover: {float(cost):g} seconds','',
            '| Forecaster | Forecast reports / controller | Misses per million | Mean roams / fleet | Bulk Mbps | Forecast RMSE dB |',
            '|---|---|---:|---:|---:|---:|']
        for family,label in METHOD.items():
            for variant,vlabel in VARIANT.items():
                a=group['arms'][family][variant]
                lines.append(f'| {label} | {vlabel} | {a["ppm"]:.2f} | {a["mean_roams"]:.1f} | {a["bulk_mbps"]:.2f} | {a["forecast_rmse_db"]:.2f} |')
        lines += ['']
        for family,label in METHOD.items():
            a=group['utility_comparisons'][family]
            lines.append(f'{label}, utility minus delivered-reference: **{a["mean_difference_ppm"]:.2f} ppm**, paired-site 95% interval {interval(a)}; {a["wins"]}/{a["sites"]} site wins.')
            lines.append('')
        a=group['model_comparison']
        lines += [f'Compact model minus RadioMap with delivered reports and the same utility controller: {a["mean_difference_ppm"]:.2f} ppm, interval {interval(a)}.','']
    primary=summary['primary_comparison']
    lines += ['## Confirmation and limits','',
        f'The prespecified primary utility comparison **{"passes" if summary["primary_success"] else "does not pass"}** the criterion that the upper confidence endpoint is below zero. The primary tests the controller change within the compact model; model comparisons and fast-handover results are secondary. Confidence intervals resample eight sites, not individual robot packets. They do not include additional model-training variability.','',
        'The improvement is limited to the slow-handover comparison: the compact model loses 5.849% → 5.635% of commands (3.67% fewer misses), with six site wins and two ties. RadioMap with the new utility controller is 5.603%; the model comparison spans zero. Compact forecast RMSE is 6.70 versus 8.54 dB, without corresponding demonstrated control superiority. Remaining losses are still too high for deployment.','',
        'At 30 ms handovers the same utility controller worsens every test site for both models. Compact loss rises 1.035% → 1.318%, while mean fleet roams rise 140.25 → 179.0 and bulk throughput falls. This regression rejects universal use of the selected slow-handover profile. Its secondary interval also excludes zero in the harmful direction. The original application default is unchanged; selecting a cost-dependent router after seeing these tests would need another independent confirmation.','',
        'The native cause ledger attributes 2,704 of the compact fast-profile\'s 2,715 additional missed commands to weak-link or blockage conditions, and ten to additional no-radio-up misses. That diagnoses a link-choice problem worth testing with immediate-link constraints and native packet-supervised risk, without establishing either proposed remedy.','',
        'Ideal model reports isolate the earlier independent scan stream while retaining the new observable local controller. Thus its numbers should not be substituted for historical study numbers. Delivered-report and utility runs share mechanisms but generate different observations because their actions differ; that feedback is intended.','',
        'All timing costs are assumed or CPU-microbenchmarked, not measured on a SmartCar. The new full single-family cost is charged at least the old cadence-five p95 envelope plus 500 ms; radio control continues until each FIFO result is ready. Scan-report selection/serialization and controller scoring still lack separate CPU deadlines. The Python native packet abstraction uses archived ns-3 curves, not a full ns-3 run per case. The fixed CAD, precomputed motion, MLO/pilot assumptions, shared warm start, reserved telemetry payload and modeled PHY remain transfer limits. No safety qualification, physical-car success, universal world-model win or SOTA claim follows.','',
        'Reproduce with `bash scripts/run_closed_loop.sh`. Source/weight identities, latency, the development-only selection, delivery events, forecast hashes, raw per-robot packet counters and read-only verification are archived under `artifacts/closed_loop/`. Earlier study archives are unchanged.','']
    (ROOT/'docs/CLOSED_LOOP_RESULTS.md').write_text('\n'.join(lines))
    analysis=OUT/'analysis_source';analysis.mkdir(exist_ok=True)
    shutil.copyfile(__file__,analysis/'report_closed_loop.py')
    shutil.copyfile(ROOT/'scripts/replay_closed_loop.py',analysis/'replay_closed_loop.py')
    previous={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'artifacts/world_frontier/manifest.json',ROOT/'artifacts/frontier_confirmation/manifest.json',ROOT/'artifacts/hardware_budget/manifest.json',ROOT/'artifacts/handover_admission/manifest.json']}
    files={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    manifest={'files_sha256':files,'checkpoint_sha256':json.loads((OUT/'test_source.json').read_text())['checkpoint_sha256'],
              'previous_manifests':previous,'document_sha256':digest(ROOT/'docs/CLOSED_LOOP_RESULTS.md')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'completed':True,'primary_success':summary['primary_success'],'primary':primary},indent=2))


def audit():
    p=json.loads((OUT/'protocol.json').read_text());m=json.loads((OUT/'manifest.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text());latency=json.loads((OUT/'latency.json').read_text())
    assert selection['sealed'] and not selection['test_accessed'] and not latency['test_accessed']
    assert latency['time_unix']<json.loads((OUT/'development_source.json').read_text())['time_unix']<selection['sealed_at_unix']<json.loads((OUT/'test_source.json').read_text())['time_unix']
    for name,h in m['files_sha256'].items():assert digest(OUT/name)==h,name
    for name,h in m['checkpoint_sha256'].items():assert digest(ROOT/name)==h,name
    for name,h in m['previous_manifests'].items():assert digest(ROOT/name)==h,name
    assert m['document_sha256']==digest(ROOT/'docs/CLOSED_LOOP_RESULTS.md')
    assert selection['selected']==min(selection['candidates'],key=lambda r:(r['misses'],-r['sum_bulk_mbps'],r['horizon_seconds']))
    assert not set(p['development_sites'])&set(p['test_sites'])
    receipts={stage:json.loads((OUT/(stage+'_source.json')).read_text()) for stage in ['development','test']}
    for stage,receipt in receipts.items():
        for name,h in receipt['source_sha256'].items():
            assert digest(OUT/(stage+'_source')/name)==h,name
            assert digest(ROOT/name)==h,name
        assert receipt['source_sha256']==latency['source_sha256']==selection['source_sha256']
        assert receipt['checkpoint_sha256']==m['checkpoint_sha256']
        assert receipt['latency_sha256']==digest(OUT/'latency.json')==selection['latency_sha256']
    cases=[json.loads(f.read_text()) for f in sorted((OUT/'cases').glob('*.json'))]
    dev=[c for c in cases if c['stage']=='development'];tests=[c for c in cases if c['stage']=='test']
    assert len(dev)==24 and len(tests)==96
    assert {c['scene'] for c in tests}==set(p['test_sites'])
    assert {c['scene'] for c in dev}==set(p['development_sites'])
    for c in cases:
        assert c['no_truth_in_predictor_or_admission_inputs'] and c['simulation_only']
        assert c['source_sha256']==receipts[c['stage']]['source_sha256'] and c['checkpoint_sha256']==m['checkpoint_sha256']
        assert c['latency_sha256']==digest(OUT/'latency.json')
        r=c['result'];assert r['control_sent']==100*60*20
        assert r['control_miss']==r['control_misses']/r['control_sent'] and sum(r['miss_causes'].values())==r['control_misses']
        if c['stage']=='test' and c['variant']=='delivered_utility':assert c['horizon_seconds']==selection['selected']['horizon_seconds']
        telemetry=c['telemetry'];assert telemetry['reports_delivered']==len(telemetry['events'])
        assert telemetry['reports_delivered']+telemetry['reports_pending']+telemetry['reports_replaced']+telemetry['receiver_missing']==telemetry['scans_started']
        for event in telemetry['events']:
            assert event['measurement_slot']<event['complete_slot']<=event['delivery_slot']<100*event['first_usable_decision']
            assert event['first_usable_decision']==event['delivery_slot']//100+1 and 0<=event['heard']<=8
        previous=0.
        for d,job in enumerate(c['jobs']):
            assert job['decision']==d and job['history_cutoff_exclusive']==60+d
            service=latency['p95_seconds'][c['family']][job['kind']]+p['extra_compute_seconds']
            previous=max(d,previous)+service
            assert job['charged_seconds']==service and job['ready_slot']==int(np.ceil(previous*100-1e-10))
            if d>0 and c['variant']!='ideal_reports':
                events=[e for e in telemetry['events'] if e['first_usable_decision']==d]
                assert job['scans_last_bin']==len({e['robot'] for e in events})
        assert len(c['jobs'])==60
    for option in selection['candidates']:
        rows=[c['result'] for c in dev if c['variant']=='delivered_utility' and c['horizon_seconds']==option['horizon_seconds']]
        assert option['misses']==sum(r['control_misses'] for r in rows)
        assert np.isclose(option['sum_bulk_mbps'],sum(r['bulk_mbps'] for r in rows))
    recomputed=aggregate(tests,p)
    assert recomputed==json.loads((OUT/'summary.json').read_text())
    complete=json.loads((OUT/'completion.json').read_text())
    assert complete['completed'] and complete['native_runs']==len(cases)==120
    replay=json.loads((OUT/'replay.json').read_text())
    assert replay['passed'] and len(replay['replays'])==4
    assert replay['replay_code_sha256']==digest(OUT/'analysis_source/replay_closed_loop.py')
    for r in replay['replays']:
        assert r['exact_packet_replay'] and r['exact_delivery_replay'] and r['exact_forecasts']==60
    return {'passed':True,'native_runs':120,'fresh_sites':8,'development_sites':4,'causal_delivery_and_fifo_verified':True,'simulation_only':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    if args.audit:print(json.dumps(audit(),indent=2))
    else:report()
