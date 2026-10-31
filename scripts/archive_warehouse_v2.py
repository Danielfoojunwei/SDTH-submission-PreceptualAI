"""Validate completed real runs, archive their reports, and write the findings."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
from statistics import mean
import tarfile

from wirelessjev.warehouse_v2.experiment import atomic_json, sha256


def load(path):
    return json.loads(Path(path).read_text())


def average(values):
    return f'{mean(values):.4f}'


def source_capsule(source, target):
    """Portable exact executed source, with deterministic archive metadata."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as output, gzip.GzipFile(fileobj=output, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file():
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(source)))
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        archive.addfile(info, stream)
    return sha256(target)


def validate_sources(root):
    receipt = load(root/'provenance.json')
    for relative, expected in receipt['source_sha256'].items():
        if sha256(root/relative) != expected:
            raise RuntimeError(f'Executed source snapshot changed: {root/relative}')


def summary(core, studies, factorized, generative):
    text = ['# Warehouse JEV v2 findings', '',
            'Evidence: simulation, real pretrained-model inference, native ns-3 packet simulation and Sionna coded-PHY execution. No physical car test is claimed.', '',
            'Original E1–E5/E7 reports are preserved. These are new protocols, not interchangeable reruns of the original experiments. Three independent compact-model seeds do not turn the historical large-model runs into multi-seed evidence.', '',
            '## Structured missions', '']
    compatibility = load(studies/'compatibility.json')
    for reader, splits in compatibility['summary'].items():
        for split, report in splits.items():
            text.append(f'- {reader}, {split}: {report["answered"]}/{report["n"]} answered; normalized candidate regret {report["norm_regret_answered"]["mean"]:.4f} on answered cases.')
    text += ['', 'The exact simulator outcomes and common candidate set are available here. This establishes a stronger semantic-interface baseline; it does not establish prediction from real radio measurements. Historical E2 does not establish a learned-scorer advantage over this structured evaluator.', '',
             '## Sensed-input models', '',
             'Training uses 24 environments, validation 6, ordinary test 12, shifted-layout test 6, and shifted-layout 200-robot test 6. Inputs are noisy capped scan histories, public route/demand/AP metadata and reported current service. Future truth is confined to intervention labels.', '',
             '| Test split | Model | Mean normalized regret across three seeds | Per-seed regret | Mean service RMSE |',
             '|---|---|---|---|---|']
    core_rows = []
    for split in ['test', 'shift', 'fleet200']:
        for arm in ['instant_mlp', 'temporal_graph', 'semantic_graph']:
            reports = [load(core/f'eval-{arm}-{seed}-{split}.json') for seed in [0, 1, 2]]
            values = [r['regret']['mean'] for r in reports]
            row = {'split': split, 'arm': arm, 'regret_seeds': values,
                   'mean_regret': mean(values), 'service_rmse': mean(r['service_rmse'] for r in reports)}
            core_rows.append(row)
            text.append(f'| {split} | {arm} | {average(values)} | '+', '.join(f'{v:.4f}' for v in values)+f' | {row["service_rmse"]:.4f} |')
    text += ['', 'The compact non-graph baseline wins plan selection in this first protocol. Graph modelling slightly improves service prediction; the semantic value head does not establish an advantage. These offline counterfactual scores are not closed-loop guarantees.', '',
             '## Closed-loop qualification and correction', '',
             'The first v2 learned controllers performed poorly in 600 s rollouts. Their five-second, interruption-dominated training target was mismatched to the thirty-second control commitment. The failure is retained in `closedloop.json`.', '',
             'A separately recorded correction learns uninterrupted service and charges the assumed four-second interruption analytically for a thirty-second commitment. It uses new rollout seeds 131000–131002. It still does not predict every intermediate route state.', '',
             '| Policy | Mean service fraction | Worst-decile mean service | Mean reconnections per rollout | Mean critical-violation robot-steps |',
             '|---|---|---|---|---|']
    prospective = load(factorized/'factorized.json')['records']
    prospective += load(factorized/'strongest_baseline.json')['records']
    policies = ['stay', 'legacy_hysteresis', 'strongest_scan', 'observed_load_balance', 'instant_mlp', 'temporal_graph', 'factorized_mlp', 'factorized_graph']
    fleet_rows = []
    for policy in policies:
        rows = [r for r in prospective if r['policy'] == policy]
        metrics = [r['metrics'] for r in rows]
        item = {'policy': policy, 'mean_service': mean(r['mean_service'] for r in metrics),
                'worst_decile_mean_service': mean(r['worst_decile_mean_service'] for r in metrics),
                'reconnections': mean(r['reconnections'] for r in metrics),
                'critical_violation_robot_steps': mean(r['critical_violation_robot_steps'] for r in metrics)}
        fleet_rows.append(item)
        text.append(f'| {policy} | {item["mean_service"]:.4f} | {item["worst_decile_mean_service"]:.4f} | {item["reconnections"]:.1f} | {item["critical_violation_robot_steps"]:.1f} |')
    text += ['', 'The factorized graph controller matches the strongest-scan policy exactly on these three rollouts. That is not evidence of a learned advantage. The factorized MLP differs on one rollout; increased fleet service does not establish a better safety/continuity trade-off. Compare reconnections and critical violations. Each model seed is paired with one rollout seed; neither universal benefit to all cars nor physical-car performance follows.', '',
             '## Equal-budget sensing and calibration', '',
             'Fifteen new environments; corrected and geometry-censored maps; uniform and decision-aware scanning; 0/10/20/40 extra robot scans after 12 initial scans. The 48-feature map protocol differs from historical E7. Scan-cost sensitivities use assumed durations, not measured interruptions.', '',
             '| Map / acquisition / extra scans | Mean gap to best fixed candidate |', '|---|---|']
    sensing = load(studies/'sensing.json')
    for key, value in sensing['summary'].items():
        text.append(f'| {key} | {value["mean"]:.4f} |')
    text += ['', '## Transfer, multitask learning and site removal', '']
    transfer = load(studies/'transfer.json')
    for count in [0, 10, 100]:
        values = [r['metrics']['mean_regret'] for r in transfer['records'] if r['adaptation_examples'] == count]
        text.append(f'- One new site, {count} adaptation examples: mean candidate regret {average(values)} across three initial model seeds.')
    multitask = load(studies/'multitask.json')
    text += [f'- Repaired admission label counts: `{json.dumps(multitask["admission_label_counts"], sort_keys=True)}`.',
             '- Admission includes HOLD and heterogeneous best-effort priorities; all thirteen current levels are counterfactual coverage, not field prevalence. Read balanced accuracy and per-label recall with accuracy.',
             f'- Actual frozen Chronos-Bolt forecast MAE: {multitask["baselines"]["chronos"]["forecast_mae_mbps"]:.3f} Mbps; last-value baseline: {multitask["baselines"]["forecast_last_value_mae_mbps"]:.3f} Mbps. These are synthetic traffic windows.',
             '- Shared and separate small task cores are both trained with three seeds. Per-task results, rules, exact admission oracle, model sizes and negative transfer remain in `multitask.json`.']
    federation = load(studies/'federation.json')
    accepted = sum(r['shared_gate_accepted'] for r in federation['records'])
    exact = sum(r['removal']['equals_clean_retrain_weights'] for r in federation['records'])
    text += [f'- Site adaptation: {accepted}/3 shared candidates pass the unchanged no-site-harm release gate.',
             f'- Shared-model removal: {exact}/3 one-round independent-adapter recomputations match a clean retained-site retrain bit for bit. This result concerns the unconditional FedAvg research baseline, not a rejected gated release or arbitrary iterative FL.',
             '- Three membership-attack signals and retained-site service are reported. This is not certified erasure of data, personal states or checkpoint copies.', '',
             '## Actual PHY/MAC and generative dependencies', '']
    phy = load(studies/'phy.json')
    ns3 = load(studies/'ns3.json')
    gen = load(generative/'generative.json')
    gen_analysis = load(generative/'generative_analysis.json')['comparisons']
    text += [f'- Sionna {phy["sionna_version"]}: {len(phy["points"])} coded QAM/SNR points, with raw bit/block errors. The 3GPP LDPC chain is a generic PHY study, not an IEEE 802.11 implementation.',
             '- ns-3 3.49: 3,618 NIST payload-success curve points, integrated into the warehouse evaluator with 0/3/6 dB rate-margin sensitivity. The existing airtime/MAC-efficiency abstraction remains.',
             f'- Native packet simulator: {len(ns3["snapshots"])} 100-robot, eight-second policy comparisons from {ns3.get("unique_native_simulations", len(ns3["snapshots"]))} distinct topology/association/seed simulations. Identical comparisons explicitly share raw results. Per-robot throughput, unreceived packets and delay histograms are retained. Static log-distance channels and default single-stream packet devices do not validate rack propagation, mobility or handover.',
             f'- Actual Qwen3.5-0.8B active-semantics accuracy: {gen_analysis["qwen"]["active_semantics_accuracy"]:.3f}; real d3 typed reader in the same CPU session: {gen_analysis["d3"]["active_semantics_accuracy"]:.3f}. Raw Qwen outputs populate irrelevant fields, so strict schema accuracy is {gen_analysis["qwen"]["strict_schema_accuracy"]:.3f}; semantic scoring compares only fields used by the chosen kind, without correcting outputs. Qwen generates fields; d3 selects a family and extracts literals, so interfaces differ.',
             '- LWM and JEPA are research references, not checkpoints represented as evaluated here. ANSpect proprietary algorithms were not reproduced.', '',
             '## Remaining qualification', '',
             'Real RSSI/throughput/outage traces, switch-stage events, one-car qualification and multi-car trials require actual hardware access. There are no configured car credentials or private-network connection in this cloud. The external Claude Studio has not been republished.', '',
             'The new mechanisms and experiments are implemented and executed. These results do not yet establish SOTA, a general wireless foundation model, or benefit to every physical robot.']
    snapshot = {'source': 'simulation_and_real_model_simulator_execution', 'training_seeds': [0, 1, 2],
                'core_models': core_rows, 'prospective_closedloop': fleet_rows,
                'hardware_experiments_executed': False, 'sota_established': False}
    return '\n'.join(text)+'\n', snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, default=Path('/workspace/training/warehouse-v2-core'))
    p.add_argument('--studies', type=Path, default=Path('/workspace/training/warehouse-v2-studies'))
    p.add_argument('--factorized', type=Path, default=Path('/workspace/training/warehouse-v2-factorized'))
    p.add_argument('--generative', type=Path, default=Path('/workspace/training/warehouse-v2-generative'))
    p.add_argument('--archive', type=Path, default=Path('artifacts/warehouse_v2'))
    args = p.parse_args()
    for root in [args.core, args.studies, args.factorized]:
        if load(root/'status.json')['state'] != 'completed':
            raise RuntimeError(f'Run is not complete: {root}')
    if not (args.generative/'generative.json').exists():
        raise RuntimeError('Actual generative baseline has not completed')
    for seed in [0, 1, 2]:
        for arm in ['instant_mlp', 'temporal_graph', 'semantic_graph']:
            metadata = load(args.core/f'train-{arm}-{seed}.json')
            actual_hash = sha256(args.core/'checkpoints'/f'{arm}-{seed}.pt')
            if actual_hash != metadata['checkpoint_sha256']:
                raise RuntimeError('Core checkpoint changed since training')
            for split in ['test', 'shift', 'fleet200']:
                evaluation = load(args.core/f'eval-{arm}-{seed}-{split}.json')
                if evaluation['checkpoint_sha256'] != actual_hash or sha256(evaluation['predictions']) != evaluation['predictions_sha256']:
                    raise RuntimeError('Evaluation/checkpoint/predictions provenance mismatch')
        for graph in [False, True]:
            metadata = load(args.factorized/f'factorized-train-{seed}-{graph}.json')
            if metadata['checkpoint_sha256'] != sha256(args.factorized/'checkpoints'/f'factorized-{seed}-{graph}.pt'):
                raise RuntimeError('Factorized checkpoint changed since training')
    if sha256(args.factorized/'factorized.json') != load(args.factorized/'status.json')['report_sha256']:
        raise RuntimeError('Factorized completion receipt mismatch')
    core_provenance = load(args.core/'provenance.json')
    for relative, expected in core_provenance['source_sha256'].items():
        if sha256(args.core/'source_snapshot'/relative) != expected:
            raise RuntimeError('Original core source capsule changed')
    for relative, expected in load(args.factorized/'provenance.json')['source_sha256'].items():
        if sha256(args.factorized/'source_snapshot'/relative) != expected:
            raise RuntimeError('Factorized source snapshot changed')
    import torch
    weights = {}
    for root in [args.core, args.studies, args.factorized]:
        for checkpoint in (root/'checkpoints').glob('*.pt'):
            content = torch.load(checkpoint, map_location='cpu', weights_only=True)
            def finite(value):
                if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                    raise RuntimeError(f'Nonfinite checkpoint: {checkpoint}')
                if isinstance(value, dict):
                    for child in value.values(): finite(child)
            finite(content)
            weights[str(checkpoint)] = sha256(checkpoint)
    args.archive.mkdir(parents=True, exist_ok=True)
    capsules = {}
    capsule_roots = {'core': args.core/'source_snapshot', 'factorized': args.factorized/'source_snapshot'}
    capsule_roots.update({f'studies-{p.name}': p for p in (args.studies/'source_snapshots').iterdir() if p.is_dir()})
    for name, source in capsule_roots.items():
        if name.startswith('studies-'):
            validate_sources(source)
        target = args.archive/'executed_sources'/f'{name}.tar.gz'
        capsules[str(target)] = source_capsule(source, target)
    reports = {}
    for group, root in [('core', args.core), ('studies', args.studies), ('factorized', args.factorized), ('generative', args.generative)]:
        for source in sorted(root.glob('*.json')):
            if source.name in ['progress.json', 'federation-progress.json', 'multitask-progress.json', 'factorized-progress.json']:
                continue
            load(source)  # Parse actual file; no synthetic replacement on failure.
            dest = args.archive/group/source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            reports[str(dest)] = {'sha256': sha256(dest), 'bytes': dest.stat().st_size}
    # Validate the stage receipts, not merely their human-readable state.
    for name, receipt in load(args.studies/'status.json')['stages'].items():
        if receipt['state'] != 'completed' or receipt['report_sha256'] != sha256(args.studies/f'{name}.json'):
            raise RuntimeError(f'Extension receipt/report mismatch: {name}')
    findings, snapshot = summary(args.core, args.studies, args.factorized, args.generative)
    Path('docs/WAREHOUSE_V2_RESULTS.md').write_text(findings)
    atomic_json(args.archive/'snapshot.json', snapshot)
    atomic_json(args.archive/'provenance.json', {'reports': reports, 'checkpoint_sha256': weights, 'executed_source_capsules': capsules,
                'runs': {'core': str(args.core), 'studies': str(args.studies), 'factorized': str(args.factorized), 'generative': str(args.generative)},
                'evidence': 'simulation_and_actual_dependency_execution', 'hardware_experiments_executed': False})
    (args.archive/'README.md').write_text('# Warehouse v2 evidence\n\nSee `docs/WAREHOUSE_V2_RESULTS.md` for interpretation. '
        'All reports are copied unchanged from completed runs. Provenance records raw report and retained checkpoint hashes. '
        'Full numeric caches, source snapshots and checkpoints remain in the recorded workspace run directories. '
        'No physical car or ANSpect implementation is claimed.\n')
    print(f'Archived {len(reports)} completed reports; verified {len(weights)} finite checkpoints')


if __name__ == '__main__':
    main()
