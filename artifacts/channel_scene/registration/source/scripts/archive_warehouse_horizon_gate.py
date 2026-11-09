"""Archive the completed selector and verify every numerical fallback outcome."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
from statistics import mean
import tarfile

from wirelessjev.warehouse_v2.experiment import atomic_json, sha256


def summarize(report, policy):
    records = report['records']
    baseline = {r['scene']: r for r in records if r['method'] == 'numerical_planner'}
    verified = 0
    for row in records:
        if row['constraint_violations']:
            raise RuntimeError('A prospective audit proposal violated its contract')
        if row['method'] == 'guarded_residual' and not policy['models'][row['model_key']]['whole_schedule_accepted']:
            old = baseline[row['scene']]
            for field in ['metrics', 'phase_metrics', 'phase_utility', 'mean_mission_utility']:
                if old[field] != row[field]:
                    raise RuntimeError('Rejected residual did not reproduce numerical outcomes')
            verified += 1
    summary = []
    for method in ['budgeted_strongest', 'numerical_planner', 'guarded_residual']:
        rows = [r for r in records if r['method'] == method]
        summary.append({'method': method, 'runs': len(rows),
            'mean_service': mean(r['metrics']['mean_service'] for r in rows),
            'worst_decile_service': mean(r['metrics']['worst_decile_mean_service'] for r in rows),
            'critical_violation_robot_steps': mean(r['metrics']['critical_violation_robot_steps'] for r in rows),
            'mean_mission_utility': mean(r['mean_mission_utility'] for r in rows)})
    return {'source': 'prospective_warehouse_simulation', 'audit_environments': report['audit_environments'],
            'validation_environment_count': len(report['validation_baseline']),
            'validation_rollouts': len(report['validation_baseline'])+sum(map(len, report['validation_candidates'].values())),
            'audit_rollouts': len(records), 'accepted_learned_models': sum(v['whole_schedule_accepted'] for v in policy['models'].values()),
            'candidate_models': len(policy['models']), 'verified_identical_fallbacks': verified,
            'designed_after_original_holdout': True, 'summary': summary,
            'guarded_rows_are_independent_fleets': False, 'hardware_qualified': False,
            'physical_safety_guarantee': False, 'sota_established': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('/workspace/training/warehouse-horizon-selector'))
    parser.add_argument('--parent', type=Path, default=Path('artifacts/warehouse_horizon'))
    args = parser.parse_args()
    root, out = args.run, args.parent/'selector'
    status = json.loads((root/'status.json').read_text())
    if status['state'] != 'completed' or sha256(root/'gate_audit.json') != status['report_sha256']:
        raise RuntimeError('Incomplete or modified selector audit')
    if sha256(root/'gate_policy.json') != status['gate_policy_sha256']:
        raise RuntimeError('Selector changed after audit')
    policy = json.loads((root/'gate_policy.json').read_text())
    if policy['parent_selection_sha256'] != sha256(args.parent/'selection.json'):
        raise RuntimeError('Archive parent differs from audited parent')
    provenance = json.loads((root/'provenance.json').read_text())
    for relative, digest in provenance['source_sha256'].items():
        if sha256(root/'source_snapshot'/relative) != digest:
            raise RuntimeError('Selector executed-source capsule mismatch')
    report = json.loads((root/'gate_audit.json').read_text())
    if report['gate_policy_sha256'] != status['gate_policy_sha256']:
        raise RuntimeError('Raw audit references a different selector')
    snapshot = summarize(report, policy)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ['status.json', 'gate_policy.json', 'provenance.json']:
        shutil.copyfile(root/name, out/name); files[name] = sha256(out/name)
    with (out/'gate_audit.json.gz').open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
        zipped.write((root/'gate_audit.json').read_bytes())
    capsule = out/'executed_sources.tar.gz'
    with capsule.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for path in sorted((root/'source_snapshot').rglob('*')):
                if path.is_file():
                    info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root/'source_snapshot')))
                    info.mtime = info.uid = info.gid = 0; info.uname = info.gname = ''
                    with path.open('rb') as data: archive.addfile(info, data)
    atomic_json(out/'snapshot.json', snapshot)
    for name in ['gate_audit.json.gz', 'executed_sources.tar.gz', 'snapshot.json']:
        files[name] = sha256(out/name)
    atomic_json(out/'manifest.json', {'files': files, 'audit_uncompressed_sha256': status['report_sha256'],
                'parent_selection_sha256': policy['parent_selection_sha256'], 'source': 'simulation'})
    lines = ['# Prospective closed-loop selector audit', '',
        'This selector was designed after the initial horizon holdout exposed lower-tail regressions. It uses six validation environments to accept or reject each already-trained model, seals its decision, then evaluates three new environments (143000–143002). No training or calibration uses those audit environments.', '',
        'Both mission phases must improve mean validation utility without increasing critical violations or lowering worst-decile service in any validation environment. The whole schedule is accepted together; independently mixing phases would change the state at the mission transition.', '',
        f'**{snapshot["accepted_learned_models"]}/{snapshot["candidate_models"]} learned models accepted.** All rejected models use the numerical belief and its original calibration. The audit checks numerical fallback outcomes for exact equality; these repetitions are not independent learned successes.', '',
        '| Audit controller | Runs | Mean service | Worst-decile service | Critical-violation robot-steps |',
        '|---|---|---|---|---|']
    for row in snapshot['summary']:
        lines.append(f'| {row["method"]} | {row["runs"]} | {row["mean_service"]:.4f} | {row["worst_decile_service"]:.4f} | {row["critical_violation_robot_steps"]:.1f} |')
    lines += ['', f'The study executes {snapshot["validation_rollouts"]} validation rollouts and {snapshot["audit_rollouts"]} prospective audit rollouts. {snapshot["verified_identical_fallbacks"]} rejected-model audit outcomes exactly match their numerical reference. The fresh comparison has three independent simulated environments, not eighteen independent fleets.', '',
        'The same operator-policy budget of twelve switches per decision applies. This does not establish superiority to unconstrained roaming, hardware reliability or a safe learned controller. Three audit environments are a small evaluation sample. All observations, telemetry and switch outcomes are simulated.', '',
        'The proposal CLI defaults to numerical planning and uses this guard when a residual is requested. An explicit simulation-only research option exposes the residuals for continued experiments. Unsupported clauses are rejected; input provenance remains self-reported. No car command is sent.', '',
        'The compressed raw audit is byte-for-byte recoverable (`gzip -dc gate_audit.json.gz`) with its original completion SHA-256 in `manifest.json`. Executed sources, sealed policy, validation differences and provenance are archived alongside it.']
    Path('docs/WAREHOUSE_HORIZON_SELECTOR.md').write_text('\n'.join(lines)+'\n')
    print(f'Archived selector: {snapshot["audit_rollouts"]} fresh audit rollouts; {snapshot["accepted_learned_models"]}/{snapshot["candidate_models"]} accepted learned models')


if __name__ == '__main__':
    main()
