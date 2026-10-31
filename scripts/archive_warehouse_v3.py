"""Archive verified completed v3 evidence and retain explicitly superseded pilots."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean
import tarfile
import numpy as np
import torch
from wirelessjev.warehouse_v2.experiment import atomic_json,sha256


def read(path):return json.loads(Path(path).read_text())


def finite(value):
    if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():raise RuntimeError('Nonfinite checkpoint')
    if isinstance(value,dict):
        for child in value.values():finite(child)
    if isinstance(value,(tuple,list)):
        for child in value:finite(child)
    if isinstance(value,float) and not np.isfinite(value):raise RuntimeError('Nonfinite report')


def copy_json(source,target,*,compressed=False):
    finite(read(source));target.parent.mkdir(parents=True,exist_ok=True)
    if compressed:
        with target.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as stream:stream.write(source.read_bytes())
    else:shutil.copyfile(source,target)
    return {'sha256':sha256(target),'uncompressed_sha256':sha256(source)}


def sources(root,out):
    provenance=read(root/'provenance.json')
    for name,digest in provenance['source_sha256'].items():
        if sha256(root/'source_snapshot'/name)!=digest:raise RuntimeError('Executed source capsule mismatch')
    path=out/'executed_sources.tar.gz';out.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0) as stream:
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for source in sorted((root/'source_snapshot').rglob('*')):
                if source.is_file():
                    info=archive.gettarinfo(str(source),arcname=str(source.relative_to(root/'source_snapshot')))
                    info.mtime=info.uid=info.gid=0;info.uname=info.gname=''
                    with source.open('rb') as data:archive.addfile(info,data)
    return sha256(path)


def completion(root,report=None,*,stage=None):
    status=read(root/'status.json')
    if stage:
        entry=status.get('stages',{}).get(stage,{})
        if entry.get('state')!='completed' or sha256(root/f'{stage}.json')!=entry['sha256']:raise RuntimeError('Incomplete study stage')
    elif status.get('state')!='completed':raise RuntimeError(f'Run is not complete: {root}')
    for name,digest in status.get('reports',{}).items():
        if sha256(root/name)!=digest:raise RuntimeError('Completion report changed')
    if report and sha256(root/report)!=status['report_sha256']:raise RuntimeError('Completion receipt mismatch')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/workspace/training'))
    for name in ['core','sensing','shared','packets','packet_learning','frontier','generative','hidden','clauses','cost','ranker','ranker_audit','schema_audit','packet_calibration']:
        p.add_argument('--'+name.replace('_','-'),type=Path)
    p.add_argument('--out',type=Path,default=Path('artifacts/warehouse_v3'));args=p.parse_args()
    root,out=args.root,args.out
    runs={'core':root/'warehouse-v3','sensing':root/'warehouse-v3-sensing-corrected',
        'shared':root/'warehouse-v3-studies','packets':root/'warehouse-v3-packets-matched',
        'packet_learning':root/'warehouse-v3-packet-learning-unconditional','frontier':root/'warehouse-v3-frontier-matched',
        'generative':root/'warehouse-v3-generative','hidden':root/'warehouse-v3-hidden-matched',
        'clauses':root/'warehouse-v3-clauses','cost':root/'warehouse-v3-cost-audit',
        'ranker':root/'warehouse-v3-ranker','ranker_audit':root/'warehouse-v3-ranker-record-audit',
        'schema_audit':root/'warehouse-v3-schema-audit','packet_calibration':root/'warehouse-v3-packet-calibration'}
    runs={key:getattr(args,key) or path for key,path in runs.items()}
    completion(runs['core']);completion(runs['sensing'],stage='sensing');completion(runs['shared'],stage='adaptation')
    completion(runs['packets'],'packet_audit.json');completion(runs['packet_learning'],stage='packet_prediction')
    completion(runs['frontier'],'frontier.json');completion(runs['generative'],'generative.json');completion(runs['hidden'],'hidden_terminal.json')
    completion(runs['clauses'],'clauses.json')
    completion(runs['cost'],'intervals_and_cost.json')
    completion(runs['ranker'],'ranker.json');completion(runs['ranker_audit'],'record_audit.json');completion(runs['schema_audit'],'schema_audit.json')
    completion(runs['packet_calibration'],'packet_calibration.json')
    selection=read(runs['core']/'selection.json')
    if selection['test_accessed_before_selection']:raise RuntimeError('Test preceded model selection')
    manifest={'source':'simulation_and_actual_pretrained_inference','reports':{},'checkpoints':{},'sources':{},
        'hardware_experiments_executed':False,'sota_established':False}
    mapping={'core':['status.json','provenance.json','semantic_reader.json','selection.json','results.json'],
        'sensing':['status.json','provenance.json','sensing.json'],'shared':['status.json','provenance.json','adaptation.json','sensing_superseded.json'],
        'packets':['status.json','provenance.json','plans.json','packet_audit.json'],
        'packet_learning':['status.json','provenance.json','packet_prediction.json'],
        'frontier':['status.json','provenance.json','selection.json','frontier.json'],
        'generative':['status.json','generative.json'],'hidden':['status.json','hidden_terminal.json'],
        'clauses':['status.json','provenance.json','clauses.json'],
        'cost':['status.json','intervals_and_cost.json'],
        'ranker':['status.json','provenance.json','selection.json','ranker.json'],
        'ranker_audit':['status.json','provenance.json','record_audit.json'],
        'schema_audit':['status.json','schema_audit.json'],
        'packet_calibration':['status.json','calibration.json','packet_calibration.json']}
    for kind,names in mapping.items():
        base=runs[kind];target=out/kind
        for name in names:
            if name=='sensing_superseded.json' and not (base/name).exists():continue
            compressed=name in ['packet_audit.json','adaptation.json','sensing.json','frontier.json']
            relative=Path(kind)/(name+'.gz' if compressed else name)
            manifest['reports'][str(relative)]=copy_json(base/name,out/relative,compressed=compressed)
        if (base/'source_snapshot').exists():manifest['sources'][kind]=sources(base,target)
        for path in sorted(base.glob('train-*.json')):
            manifest['reports'][str(Path(kind)/path.name)]=copy_json(path,target/path.name)
        for path in sorted((base/'cases').glob('*.json')):
            relative=Path(kind)/'cases'/(path.name+'.gz');manifest['reports'][str(relative)]=copy_json(path,out/relative,compressed=True)
        for path in sorted((base/'native').glob('*.json')):
            relative=Path(kind)/'native'/(path.name+'.gz');manifest['reports'][str(relative)]=copy_json(path,out/relative,compressed=True)
        for path in sorted((base/'checkpoints').glob('*.pt')):
            finite(torch.load(path,map_location='cpu',weights_only=True));relative=Path(kind)/'checkpoints'/path.name
            (out/relative).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out/relative)
            manifest['checkpoints'][str(relative)]=sha256(path)
        for path in sorted((base/'inputs').glob('*.txt')):
            relative=Path(kind)/'inputs'/path.name;(out/relative).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out/relative)
            manifest['reports'][str(relative)]={'sha256':sha256(path),'uncompressed_sha256':sha256(path)}
    # The original packet attempt/source is deliberately retained, including
    # completed pilot reports. It is never treated as a matched completion.
    pilot=root/'warehouse-v3-packets'
    for name in ['status.json','provenance.json','progress.json']:
        if (pilot/name).exists():manifest['reports'][f'superseded_packets/{name}']=copy_json(pilot/name,out/'superseded_packets'/name)
    if (pilot/'source_snapshot').exists():manifest['sources']['superseded_packets']=sources(pilot,out/'superseded_packets')
    for path in sorted((pilot/'native').glob('*.json')):
        relative=Path('superseded_packets/native')/(path.name+'.gz');manifest['reports'][str(relative)]=copy_json(path,out/relative,compressed=True)
    conditional=root/'warehouse-v3-packet-learning'
    if (conditional/'packet_prediction.json').exists():
        for name in ['status.json','provenance.json','packet_prediction.json','superseded.json']:
            manifest['reports'][f'superseded_packet_learning/{name}']=copy_json(conditional/name,out/'superseded_packet_learning'/name)
        manifest['sources']['superseded_packet_learning']=sources(conditional,out/'superseded_packet_learning')
        for path in sorted((conditional/'checkpoints').glob('*.pt')):
            finite(torch.load(path,map_location='cpu',weights_only=True));relative=Path('superseded_packet_learning/checkpoints')/path.name
            (out/relative).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,out/relative);manifest['checkpoints'][str(relative)]=sha256(path)
    for kind in ['packets','clauses']:
        for attempt in sorted((runs[kind]/'attempts').glob('*')):
            relative=Path(kind)/'attempts'/attempt.name
            for path in attempt.glob('*.json'):manifest['reports'][str(relative/path.name)]=copy_json(path,out/relative/path.name)
            if (attempt/'source_snapshot').exists():manifest['sources'][str(relative)]=sources(attempt,out/relative)
    embeddings=runs['core']/'semantic_contexts.npz'
    shutil.copyfile(embeddings,out/'core/semantic_contexts.npz')
    manifest['reports']['core/semantic_contexts.npz']={'sha256':sha256(embeddings),'uncompressed_sha256':sha256(embeddings)}
    review=Path('/workspace/upstream/warehouse-v3/lwm-review.json')
    if review.exists():manifest['reports']['lwm_review.json']=copy_json(review,out/'lwm_review.json')
    core=read(runs['core']/'results.json');summary=[]
    for domain in ['ordinary','layout','fleet200','traffic']:
        for arm in ['numerical','mlp','graph_scratch','graph_masked','graph_jepa']:
            rows=[r['summary'] for r in core['results'] if r['domain']==domain and
                (r['method']=='numerical' if arm=='numerical' else r['method'] in [f'{arm}-{i}' for i in [0,1,2]])]
            summary.append({'domain':domain,'model':arm,'training_seeds':0 if arm=='numerical' else 3,
                'service_rmse':mean(r['service_rmse'] for r in rows),'proposal_coverage':mean(r['proposal_coverage'] for r in rows),
                'false_feasible_fraction':mean(r['false_feasible_fraction'] for r in rows),
                'raw_quantile_80_coverage':mean(r['raw_quantile_80_coverage'] for r in rows),
                'critical_risk_brier':mean(r['risk']['brier'] for r in rows)})
    packet=read(runs['packets']/'packet_audit.json');packet_summary=[]
    for rts in [False,True]:
        for policy in ['stay','budgeted_rssi','numerical','radio_jepa_seed0']:
            rows=[r for r in packet['records'] if r['scene']>=146003 and r['rts_cts']==rts and r['policy']==policy]
            packet_summary.append({'policy':policy,'rts_cts':rts,'test_scenes':len(rows),
                'mean_service':mean(r['metrics']['mean_service'] for r in rows),
                'worst_decile_service':mean(r['metrics']['worst_decile_service'] for r in rows),
                'critical_deadline_miss_fraction_generated':mean(r['metrics']['critical_deadline_miss_fraction_generated'] for r in rows),
                'maximum_critical_silence_s_mean':mean(r['metrics']['maximum_critical_silence_s'] for r in rows)})
    shared=read(runs['shared']/'adaptation.json');semantics=read(runs['core']/'semantic_reader.json');generative=read(runs['generative']/'generative.json')
    packet_model=read(runs['packet_learning']/'packet_prediction.json')
    clause_report=read(runs['clauses']/'clauses.json')
    schema=read(runs['schema_audit']/'schema_audit.json')
    ranker=read(runs['ranker']/'ranker.json')
    calibrated=read(runs['packet_calibration']/'packet_calibration.json')
    packet_predictors=[]
    for temporal,label in [(False,'instant_packet_model'),(True,'temporal_packet_model')]:
        selected=[r for r in calibrated['records'] if r['model'].endswith('-'+str(temporal))]
        trained=[r for r in packet_model['records'] if r['temporal']==temporal]
        packet_predictors.append({'model':label,'seeds':3,'calibrated_late_event_brier':mean(r['calibrated']['brier'] for r in selected),
            'raw_late_event_brier':mean(r['raw']['brier'] for r in selected),
            'calibrated_late_event_ece':mean(r['calibrated']['ece'] for r in selected),
            'throughput_mae_mbps':mean(r['next_second_throughput_mae_mbps'] for r in trained)})
    unique=len({(p['scene'],p['topology_sha256'],rts) for p in packet['plans'] for rts in [False,True]})
    snapshot={'source':'warehouse_v3_simulation_and_native_packet_events','core_models':summary,'native_packet_audit':packet_summary,
        'd3_answered':sum(not r['abstained'] for r in semantics['readings']),'d3_total':len(semantics['readings']),
        'clause_reader_answered':clause_report['answered'],'clause_reader_total':clause_report['total'],
        'qwen_complete_contract_accuracy':generative['exact_schema_accuracy'],
        'qwen_with_declared_defaults_accuracy':schema['accuracy_with_declared_defaults'],
        'full_priority_rankers':ranker['records'],
        'model_seeds':[0,1,2],'native_executions':len(packet['records']),'unique_native_conditions':unique,
        'pooled_gates_accepted':sum(r['shared_pooled_gate_accepted'] for r in shared['records']),
        'federation_gates_accepted':sum(r['federation_gate_accepted'] for r in shared['records']),
        'approximate_removal_equals_retrain':sum(r['approximate_equals_clean_weights'] for r in shared['records']),
        'packet_late_event_test_counts':packet_model['test_label_counts'],
        'packet_predictors':packet_predictors,'packet_persistence':packet_model['persistence'],
        'packet_event_discrimination_testable':all(v>0 for v in packet_model['test_label_counts'].values()),
        'protocol':'Wi-Fi','hardware_qualified':False,'sota_established':False}
    atomic_json(out/'snapshot.json',snapshot);manifest['snapshot_sha256']=sha256(out/'snapshot.json')
    atomic_json(out/'manifest.json',manifest)
    lines=['# Warehouse v3 results','',
        'These are simulated warehouse observations, actual pretrained-model inference and actual native ns-3 packet events. No physical car was contacted or qualified. Original E1–E5/E7, v2 and horizon evidence is unchanged. See `WAREHOUSE_V3.md` for the full protocol and limitations.','',
        f'Actual d3 contract-family interpretation answered {snapshot["d3_answered"]}/{snapshot["d3_total"]} author-written missions. The deterministic complete grammar remains a separately labelled baseline. Qwen complete-schema accuracy is {generative["exact_schema_accuracy"]:.1%}; invalid or truncated outputs count as failures without repairs. These interfaces differ and the language set is small.','',
        f'A separately designed clause reader answered {snapshot["clause_reader_answered"]}/{snapshot["clause_reader_total"]} fresh canonical composite statements. It verifies individual instructions while a complete compiler handles literals and order. The audit was designed after observing the family gate failure; it is not unrestricted language understanding or an independent physical result. Original failure receipts are retained.','',
        f'An audit of the unchanged Qwen outputs accepts declared optional/default schema fields, reaching {schema["accuracy_with_declared_defaults"]:.1%} exact contract match. The original strict score remains preserved. Defaults are static schema defaults; no labels recover missing constraints, no malformed JSON is repaired and no model inference is invented.','',
        '## Radio forecasts and hard-contract proposals','',
        '| Domain | Model | Service RMSE | Proposal coverage | False feasibility, all cases | Raw 80% interval coverage | Critical-proxy Brier |',
        '|---|---|---|---|---|---|---|']
    for row in summary:lines.append(f'| {row["domain"]} | {row["model"]} | {row["service_rmse"]:.4f} | {row["proposal_coverage"]:.1%} | {row["false_feasible_fraction"]:.1%} | {row["raw_quantile_80_coverage"]:.1%} | {row["critical_risk_brier"]:.4f} |')
    lines += ['', 'Forecasting improvement over a public numerical prior does not establish a control advantage. Masked/JEPA pretraining must be compared against graph-from-scratch, including its matched training budget. Raw interval coverage is measured, not assumed to equal its nominal 80%. Floored missions can defer; coverage and feasibility must accompany regret. Candidate exhaustion does not prove global infeasibility.', '',
        'Each learned row averages three independent model seeds, each evaluated on all three environments in its domain. They are not nine independent physical fleets. Dimensions/CAD are fixed; AP layouts, hidden propagation, traffic and fleet size vary. Cached-semantic, int8 and latency-aware router results are preserved separately under `frontier/` and share these holdout environments.', '',
        'A separate full-priority ranker uses explicit group-aware graph pooling and trains on the complete soft-priority ordering. Its identical-architecture text/no-text ablation uses three checkpoint chains and new environments 180000–183002. Hard predicted floors and structural guards remain outside the learned scorer. The first raw case file mislabeled a feasible-set indicator as `gold_choice`; the frozen-model record audit reconstructs actual indices and verifies all quality fields unchanged. Corrected records are under `ranker_audit/cases/`, with original raw records retained under `ranker/cases/`. This audit is not another independent replication.', '',
        '## Matched public-input native packet audit','',
        f'The audit executed {len(packet["records"])} native runs covering {unique} distinct input/seed/RTS conditions. Identical policy inputs are repeated executions, not independent environments. The table uses three held-out scenes. A single predeclared JEPA training seed supplies the learned packet policy; it is not a three-model-seed packet comparison. All four methods have the same twelve-switch operator-policy budget.', '',
        '| Policy | RTS/CTS | Service | Worst decile | Critical deadline miss fraction, generated packets | Mean maximum critical silence (s) |',
        '|---|---|---|---|---|---|']
    for row in packet_summary:lines.append(f'| {row["policy"]} | {row["rts_cts"]} | {row["mean_service"]:.4f} | {row["worst_decile_service"]:.4f} | {row["critical_deadline_miss_fraction_generated"]:.4f} | {row["maximum_critical_silence_s_mean"]:.3f} |')
    lines += ['', 'PHY profile: NIST errors, VHT80/HT20, two streams, declared transmit powers, fixed primary channels and WMM priority six for critical traffic. Routes and public topology match; log-distance propagation still lacks rack/material calibration. An eight-second route prefix and assumed application muting do not reproduce actual authentication/DHCP or a full thirty-second control loop. Muted demand reduces service but does not manufacture packets in deadline denominators. Stored plan forecasts are numerical-belief forecasts for the selected assignments, including assignments selected by the learned model.', '',
        'The earlier 20-MHz/one-stream pilot is retained as explicitly superseded partial evidence. A native-concurrency change preserves completed reports and the earlier controller sources; binary, seeds and scientific inputs remain unchanged.', '',
        '## Sensing, shared learning and removal','',
        'Sensing results under `sensing/sensing.json.gz` use actual previous-interval simulated counters. Uniform and decision-aware policies share scan budgets; cost assumptions are charged in downstream service and per-flow violations. Never-scanned robots do not generate below-floor measurements. Inaudible-path error remains reported, not declared solved.', '',
        f'Pooled shared release gates accepted {snapshot["pooled_gates_accepted"]}/3 candidates; actual FedAvg gates accepted {snapshot["federation_gates_accepted"]}/3. The removal target did incorporate the forgotten contribution. Approximate removal equals clean retained-data retraining in {snapshot["approximate_removal_equals_retrain"]}/3 seeds. Pooled training is not federation; rejected updates are not deployed. Three attack diagnostics do not certify erasure.', '',
        'The new shared-removal pilot uses four sites, one 150-second episode per site/split and five counterfactual decision samples per episode. Attack samples are small and correlated; this is a contribution/removal reference, not definitive federated-unlearning evidence.', '',
        f'Packet-telemetry late-event test counts: {packet_model["test_label_counts"]}. The event is any late received critical packet in the next one-second bin, not packet loss. Models use observed retries/queues/busy intervals with explicit denominator-validity masks. Native events remain simulations.', '',
        'All bins are included, including those without a critical arrival (event false, not healthy-link evidence). The earlier outcome-conditioned pilot is retained separately. Validation-only temperature fitting is audited against the actual event on three test scenes; it reuses this holdout rather than supplying another replication.', '',
        '| Packet predictor | Raw Brier | Temperature-scaled Brier | Scaled ECE | Throughput MAE (Mbps) |',
        '|---|---|---|---|---|',
        f'| Persistence | {packet_model["persistence"]["late_arrival_risk"]["brier"]:.4f} | — | {packet_model["persistence"]["late_arrival_risk"]["ece"]:.4f} | {packet_model["persistence"]["throughput_mae_mbps"]:.4f} |']
    for row in packet_predictors:lines.append(f'| {row["model"]} | {row["raw_late_event_brier"]:.4f} | {row["calibrated_late_event_brier"]:.4f} | {row["calibrated_late_event_ece"]:.4f} | {row["throughput_mae_mbps"]:.4f} |')
    lines += ['',
        '## Remaining acceptance boundaries','',
        'No model is declared SOTA, a vertical foundation model or a physical fleet controller from these results. The compositional CLI is proposal-only; learned controllers have not passed new closed-loop/hardware release gates. Actual switch distributions, scan interruptions, packet-service calibration and one-/small-/100-car trials require hardware access. The implementation remains Wi-Fi-specific.', '',
        'All compressed JSON is recoverable byte-for-byte; original and compressed hashes are in `artifacts/warehouse_v3/manifest.json`. Checkpoints are finite and source capsules are verified. Superseded pilot evidence is preserved, not promoted to a successful completed experiment.']
    Path('docs/WAREHOUSE_V3_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('Archived verified v3 evidence:',len(manifest['checkpoints']),'finite checkpoints;',len(packet['records']),'native packet runs')


if __name__=='__main__':main()
