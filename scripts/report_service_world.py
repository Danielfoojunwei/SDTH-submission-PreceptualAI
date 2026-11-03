"""Audit completed service-world receipts and report descriptive native results.

This reporter never imports a model, opens a simulator or trains anything.
Independent counterfactual branches remain individual diagnostic examples;
their hindsight minima cannot be added into a joint fleet-policy result.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool), f'{name}: expected an integer')
    return value


def cost_key(value):
    return str(float(value))


def count_metrics(misses, sent):
    require(0 <= misses <= sent, 'Invalid raw miss/sent counts')
    return {'misses': misses, 'sent': sent,
            'miss_fraction': misses / sent if sent else None,
            'misses_per_million_sent': misses / sent * 1e6 if sent else None}


def packet_counts(result, robots, *, detailed):
    """Use integer counters, never rounded per-robot fractions."""
    misses = integer(result['control_misses'], 'control_misses')
    sent = integer(result['control_sent'], 'control_sent')
    values = count_metrics(misses, sent)
    require(math.isclose(result['control_miss'], misses / max(sent, 1),
                         rel_tol=1e-12, abs_tol=1e-15), 'Scalar miss fraction disagrees with raw counters')
    if detailed:
        packet = result['packet_detail']
        require(len(packet['misses']) == len(packet['sent']) == robots,
                'Detailed packet counter length differs from robot count')
        for m, s in zip(packet['misses'], packet['sent']):
            count_metrics(integer(m, 'robot misses'), integer(s, 'robot sent'))
        require(sum(packet['misses']) == misses and sum(packet['sent']) == sent,
                'Detailed integer packet counters disagree with scalar totals')
        cumulative = packet['cumulative']
        require(cumulative, 'Missing factual cumulative counters')
        require(cumulative[-1]['misses'] == misses and cumulative[-1]['sent'] == sent,
                'Final cumulative counters disagree with factual totals')
    return values


def effect_counts(records):
    differences = [r['delta_misses'] for r in records]
    return {'actions': len(differences),
            'beneficial': sum(d < 0 for d in differences),
            'harmful': sum(d > 0 for d in differences),
            'zero': sum(d == 0 for d in differences),
            'minimum_delta_misses': min(differences) if differences else None,
            'maximum_delta_misses': max(differences) if differences else None}


def audit(root):
    out = root / 'artifacts/service_world'
    completion_path = out / 'completion.json'
    require(completion_path.is_file(), 'No completion.json: wait for the native diagnostic to finish')
    completion = read(completion_path)
    require(completion.get('completed') is True, 'Native diagnostic is not completed')
    require(completion.get('trained_models') == 0 and completion.get('learned_policy_evaluations') == 0,
            'This reporter supports the untrained native diagnostic, not a learned-policy evaluation')
    require(completion.get('original_holdouts_unused') is True, 'Holdout-access declaration missing')
    require(completion.get('sota_established') is False and completion.get('novelty_established') is False,
            'Unexpected SOTA/novelty qualification in completion receipt')

    protocol_path = out / 'protocol.json'
    snapshot_path = out / 'executed_source.json'
    protocol, snapshot = read(protocol_path), read(snapshot_path)
    hashes = snapshot['source_sha256']
    require(isinstance(hashes, dict) and hashes, 'Empty executed-source manifest')
    require(completion['source_sha256'] == hashes, 'Completion/executed-source hash manifest mismatch')
    require(hashes.get('artifacts/service_world/protocol.json') == digest(protocol_path),
            'Protocol differs from executed-source manifest')
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path')
        archived, current = out / 'executed_source' / relative, root / relative
        require(archived.is_file() and digest(archived) == expected,
                f'Archived executed source missing or changed: {name}')
        require(current.is_file() and digest(current) == expected,
                f'Current source differs from executed diagnostic: {name}')

    expected_cases = {(int(scene), float(cost)) for scene in protocol['diagnostic_sites']
                      for cost in protocol['handover_seconds']}
    paths = sorted((out / 'cases').glob('*.json'))
    cases = [read(path) for path in paths]
    actual_cases = [(c['scene'], float(c['cost'])) for c in cases]
    require(len(actual_cases) == len(set(actual_cases)) and set(actual_cases) == expected_cases,
            'Case files are incomplete, duplicated or outside the diagnostic protocol')
    require(completion['cases'] == len(cases), 'Completion case count mismatch')
    case_rows, best_rows, all_records, links = [], [], [], []
    for path, case in zip(paths, cases):
        scene, cost, robots = case['scene'], float(case['cost']), case['robots']
        key = f'{scene}-{cost:g}'
        require(path.name == key + '.json', 'Unexpected native case filename')
        require(case['completed'] is True and case['source_sha256'] == hashes,
                f'{key}: incomplete receipt or different executed source')
        require(case.get('simulation_only') is True and case.get('no_learned_policy_evaluated') is True
                and case.get('no_heldout_access') is True, f'{key}: unsupported evaluation scope')
        require(robots == protocol['robots'], f'{key}: robot count differs from protocol')
        data_path = out / 'inputs' / (key + '.npz')
        require(data_path.is_file() and digest(data_path) == case['inputs_sha256'],
                f'{key}: native input-window hash mismatch')
        prototype = case['prototype']
        require(prototype.get('untrained') is True and prototype.get('accuracy_or_control_result') is False
                and prototype.get('zero_residual_exact') is True
                and prototype.get('unknown_hazard_is_none') is True,
                f'{key}: prototype receipt exceeds untrained identity-probe scope')
        factual = packet_counts(case['factual'], robots, detailed=True)
        legacy = packet_counts(case['legacy'], robots, detailed=False)
        require(factual['sent'] == legacy['sent'], f'{key}: classical arms have different sent counts')
        require(case['factual']['duration_s'] == case['legacy']['duration_s'] == protocol['duration_seconds'],
                f'{key}: unexpected simulated duration')
        packet = case['factual']['packet_detail']
        require(packet.get('native_proposal_ttl_slots') == round(protocol['proposal_expiry_seconds'] * 100),
                f'{key}: factual proposal lifetime differs from protocol')
        require(packet.get('observation_period_slots') == round(protocol['observation_period_seconds'] * 100),
                f'{key}: observation cadence differs from protocol')
        records = case['records']
        groups = defaultdict(list)
        action_ids = set()
        for index, record in enumerate(records):
            for field in ('slot', 'robot', 'current', 'target', 'hold_slots', 'baseline_misses',
                          'intervention_misses', 'delta_misses', 'baseline_window_max_gap_slots',
                          'intervention_window_max_gap_slots', 'delta_window_max_gap_slots'):
                integer(record[field], key + ': ' + field)
            require(0 <= record['robot'] < robots and record['slot'] >= 0,
                    f'{key}: invalid focal robot or checkpoint')
            require(record['hold_slots'] == round(protocol['intervention_hold_seconds'] * 100)
                    and record['canonical_recovery'] is True, f'{key}: intervention contract differs')
            require(record['baseline_misses'] >= 0 and record['intervention_misses'] >= 0,
                    f'{key}: negative absolute miss count')
            require(record['delta_misses'] == record['intervention_misses'] - record['baseline_misses'],
                    f'{key}: miss difference is inconsistent')
            require(record['delta_window_max_gap_slots'] == record['intervention_window_max_gap_slots']
                    - record['baseline_window_max_gap_slots'], f'{key}: gap difference is inconsistent')
            require(record['kind'] in ('stay', 'move'), f'{key}: unknown action kind')
            require((record['kind'] == 'stay') == (record['target'] == record['current']),
                    f'{key}: action kind/target mismatch')
            identity = (record['slot'], record['robot'], record['kind'], record['target'])
            require(identity not in action_ids, f'{key}: duplicate focal action')
            action_ids.add(identity)
            row = {'scene': scene, 'cost_seconds': cost, 'record_index': index, **record}
            groups[(record['slot'], record['robot'])].append(row)
            all_records.append(row)
        require(len(groups) == case['input_windows'], f'{key}: input-window/group count mismatch')
        slots = {slot for slot, _ in groups}
        require(len(slots) == case['factual_replays'] == len(protocol['checkpoint_forecast_issues_seconds']),
                f'{key}: factual replay/checkpoint count mismatch')
        require(case['native_runs'] == 2 + len(records) + case['factual_replays'],
                f'{key}: native runs do not equal factual + legacy + suffixes + interventions')
        for (slot, robot), actions in sorted(groups.items()):
            require(sum(a['kind'] == 'stay' for a in actions) == 1,
                    f'{key}: each checkpoint/robot needs one stay intervention')
            require(len({a['baseline_misses'] for a in actions}) == 1,
                    f'{key}: actions in one group have different factual baselines')
            negative = sorted((a for a in actions if a['delta_misses'] < 0),
                              key=lambda a: (a['delta_misses'], a['kind'], a['target']))
            best = negative[0] if negative else None
            best_rows.append({'scene': scene, 'cost_seconds': cost, 'slot': slot, 'robot': robot,
                              'actions_recorded': len(actions), 'best_negative_action': best,
                              'tied_best_actions': [a for a in negative if a['delta_misses'] == best['delta_misses']]
                              if best else [], 'selection': 'NONCAUSAL hindsight from independent forks',
                              'joint_fleet_policy_result': False})
        case_rows.append({'scene': scene, 'cost_seconds': cost, 'robots': robots,
                          'duration_seconds': protocol['duration_seconds'],
                          'factual_expiring_proposals': factual, 'legacy_unbounded_pending': legacy,
                          'classical_factual_minus_legacy_misses': factual['misses'] - legacy['misses'],
                          'interventions': effect_counts(records), 'native_runs': case['native_runs'],
                          'exact_factual_suffix_replays': case['factual_replays'],
                          'input_windows': case['input_windows'], 'prototype': prototype})
        links.append({'path': str(path.relative_to(root)), 'sha256': digest(path),
                      'inputs_path': str(data_path.relative_to(root)), 'inputs_sha256': digest(data_path),
                      'source_manifest_matches_completion': True})
    for field in ('native_runs', 'factual_replays'):
        require(sum(c[field] for c in cases) == completion[field], f'Completion {field} total mismatch')
    by_cost = {}
    for cost in protocol['handover_seconds']:
        selected = [c for c in case_rows if c['cost_seconds'] == float(cost)]
        arms = {}
        for arm in ('factual_expiring_proposals', 'legacy_unbounded_pending'):
            arms[arm] = count_metrics(sum(c[arm]['misses'] for c in selected),
                                      sum(c[arm]['sent'] for c in selected))
        records = [r for r in all_records if r['cost_seconds'] == float(cost)]
        grouped = [r for r in best_rows if r['cost_seconds'] == float(cost)]
        by_cost[cost_key(cost)] = {'handover_seconds': float(cost), 'cases': len(selected), **arms,
                                 'classical_pending_expiry_ablation': {
                                     'factual_ttl_seconds': protocol['proposal_expiry_seconds'],
                                     'legacy_ttl_seconds': None,
                                     'delta_misses_factual_minus_legacy': arms['factual_expiring_proposals']['misses']
                                     - arms['legacy_unbounded_pending']['misses'],
                                     'delta_misses_per_million_sent':
                                     arms['factual_expiring_proposals']['misses_per_million_sent']
                                     - arms['legacy_unbounded_pending']['misses_per_million_sent'],
                                     'learned_model_contribution': False},
                                 'interventions': effect_counts(records),
                                 'interventions_by_kind': {kind: effect_counts([r for r in records if r['kind'] == kind])
                                                           for kind in ('stay', 'move')},
                                 'checkpoint_robot_groups': len(grouped),
                                 'groups_with_a_negative_recorded_action': sum(r['best_negative_action'] is not None
                                                                              for r in grouped)}
    return {'completed': True, 'status': 'native_diagnostic_complete_untrained_prototype',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'simulation_only': True, 'diagnostic_only': True, 'robots_per_case': protocol['robots'],
            'duration_seconds_per_case': protocol['duration_seconds'],
            'diagnostic_sites': protocol['diagnostic_sites'], 'trained_models': 0,
            'learned_policy_evaluations': 0, 'learned_accuracy_or_control_score': None,
            'learned_policy_qualified': False, 'original_holdouts_unused': True,
            'sota_established': False, 'novelty_established': False,
            'counts': {'cases': len(cases), 'native_runs': completion['native_runs'],
                       'exact_factual_suffix_replays': completion['factual_replays'],
                       'input_windows': sum(c['input_windows'] for c in cases),
                       'recorded_interventions': len(all_records)},
            'replay_check_scope': 'Study asserts equality of per-robot cumulative miss counters against the factual endpoint; reporter checks receipts/source and does not rerun the simulator.',
            'by_cost': by_cost, 'cases': case_rows,
            'best_negative_action_per_checkpoint_robot': best_rows,
            'hindsight_warning': 'NONCAUSAL. Each action starts an independent fork of one factual checkpoint. These choices and their miss differences cannot be combined or summed into the result of a joint sequential fleet policy.',
            'prototype': {'untrained': True, 'accuracy_or_control_result': False,
                          'inference_parameter_counts': sorted({c['prototype']['inference_parameters'] for c in cases}),
                          'probe': 'Real-input forward/finite-latent, exact zero-residual reference identity, and unknown hazard remaining None; not trained prediction or control performance.'},
            'linkage': {'completion_path': str(completion_path.relative_to(root)),
                        'completion_sha256': digest(completion_path),
                        'executed_source_manifest_path': str(snapshot_path.relative_to(root)),
                        'executed_source_manifest_sha256': digest(snapshot_path),
                        'protocol_path': str(protocol_path.relative_to(root)),
                        'protocol_sha256': digest(protocol_path), 'source_sha256': hashes,
                        'archived_and_current_source_files_verified': len(hashes), 'case_receipts': links,
                        'reporter_path': 'scripts/report_service_world.py',
                        'reporter_sha256': digest(root / 'scripts/report_service_world.py')}}


def number(value):
    return '—' if value is None else f'{value:,.2f}'


def markdown(summary, root):
    counts = summary['counts']
    lines = ['# Service-world native diagnostic results', '',
             '**This is a completed simulation diagnostic with an untrained prototype.** '
             'It is not a learned-model result, a physical 100-car trial, SOTA validation or a novelty certification.', '',
             f'Each case contains {summary["robots_per_case"]} simulated robots over '
             f'{summary["duration_seconds_per_case"]} seconds. The four diagnostic sites are '
             + ', '.join(str(s) for s in summary['diagnostic_sites']) + '. '
             'Original held-out evaluation sites were not accessed. These are descriptive diagnostic counts; '
             'packets and sampled robots are not independent training replications.', '',
             '## Actual factual and legacy packet counts', '',
             'Both arms use the numerical RadioMap controller. The factual arm expires pending proposals '
             'after one second; the legacy arm has no added proposal lifetime (`None`). '
             'Their difference isolates a classical pending-command-expiry change. It is not a JEPA gain. '
             'Rates below are calculated from total integer misses/sent, not averages of rounded robot rates.', '',
             '| Handover | Cases | Factual 1 s TTL misses / sent | Legacy `None` misses / sent | Factual − legacy misses | Factual / legacy ppm |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary['by_cost'].values():
        factual, legacy = row['factual_expiring_proposals'], row['legacy_unbounded_pending']
        delta = row['classical_pending_expiry_ablation']['delta_misses_factual_minus_legacy']
        lines.append(f'| {row["handover_seconds"]:g} s | {row["cases"]} | '
                     f'{factual["misses"]:,} / {factual["sent"]:,} | {legacy["misses"]:,} / {legacy["sent"]:,} | '
                     f'{delta:+,} | {number(factual["misses_per_million_sent"])} / '
                     f'{number(legacy["misses_per_million_sent"])} |')
    lines += ['', '## Independent one-second interventions', '',
              'Each recorded stay/move intervention is compared with the actual unforced factual continuation '
              'from the same checkpoint. The intervention lasts one second, then canonical recovery resumes; '
              'miss differences are measured over ten seconds. Negative means fewer focal-robot misses. '
              'Focal robots are selected from public pre-intervention measurements, rather than future outcomes.', '',
              '| Handover | Recorded actions | Beneficial | Harmful | Zero | Checkpoint/robot groups with a negative action |',
              '|---|---:|---:|---:|---:|---:|']
    for row in summary['by_cost'].values():
        effect = row['interventions']
        lines.append(f'| {row["handover_seconds"]:g} s | {effect["actions"]} | {effect["beneficial"]} | '
                     f'{effect["harmful"]} | {effect["zero"]} | '
                     f'{row["groups_with_a_negative_recorded_action"]} / {row["checkpoint_robot_groups"]} |')
    lines += ['', '**Best-action selection is NONCAUSAL hindsight.** '
              'The [full summary](../artifacts/service_world/summary.json) records the best negative action '
              'and tied best actions for every checkpoint/robot group, including groups with none. '
              'Each action starts an independent fork of a factual checkpoint. Combining these choices or '
              'adding their gains does not produce a joint fleet-policy result. No trained model selected these actions.', '',
              '## Executed work and source linkage', '',
              f'The audit contains **{counts["cases"]} complete cases, {counts["native_runs"]:,} native simulator runs, '
              f'{counts["exact_factual_suffix_replays"]} factual suffix replay checks, '
              f'{counts["input_windows"]} recorded input windows and '
              f'{counts["recorded_interventions"]} intervention branches**. '
              'The replay assertion checks per-robot cumulative miss counters against the matching full factual endpoint; '
              'it does not establish equality of every simulator state field.', '',
              'The reporter verifies every case/input checksum and agreement between '
              '[completion](../artifacts/service_world/completion.json), '
              '[executed-source manifest](../artifacts/service_world/executed_source.json), '
              '[protocol](../artifacts/service_world/protocol.json), archived source copies and current source. '
              f'{summary["linkage"]["archived_and_current_source_files_verified"]} source files were verified. '
              'The JSON summary preserves the individual hashes and reporter hash. This reporting step does not '
              'rerun the simulator or train a model.', '',
              '## What the prototype establishes', '',
              'There are **zero trained models, zero learned-policy evaluations and no learned accuracy/control score**. '
              'The inactive prototype has '
              + ', '.join(f'{n:,}' for n in summary['prototype']['inference_parameter_counts'])
              + ' inference parameters in the recorded probes. Real causal input windows exercise its forward path; '
              'a zero residual preserves the supplied physical reference exactly, and an unknown hazard remains `None`. '
              'This checks an implementation identity. The physical reference is a known-unavailability lower bound, '
              'rather than an estimated or calibrated expected service value. It is not evidence of improvement over RadioMap.', '',
              '## Scientific claim boundary', '',
              'The [implemented model](SERVICE_WORLD_MODEL.md), '
              '[primary-source prior-art review](SERVICE_WORLD_PRIOR_ART.md) and '
              '[official baseline access audit](SERVICE_WORLD_BASELINES.md) describe the current evidence. '
              'Action-conditioned wireless JEPA, mechanistic residuals, robot/wireless coupling, matched '
              'counterfactuals and stochastic graph dynamics overlap directly with existing literature. '
              'No broad first-of-kind claim is supported. Official upstream component forwards are not matched '
              'warehouse experiments; unavailable code/data and unmatched tasks prevent a global SOTA conclusion.']
    principles = [p for p in sorted((root / 'docs').glob('*.md'))
                  if ('FIRST_PRINCIPLE' in p.stem.upper() or 'FIRSTPRINCIPLE' in p.stem.upper())
                  and ('WIRELESS' in p.stem.upper() or 'SERVICE' in p.stem.upper())]
    physics_link = principles[0].name if principles else '../src/wirelessjev/service_world/physics.py'
    lines += ['', f'The [first-principles service bound]({physics_link}) explains why packets whose slack '
              'is shorter than a link-exclusive unavailable interval cannot be rescued by a better predictor alone. '
              'Separate links, redundancy or a different intervention change that premise and require their own '
              'validated model and measurements. Training independent seeds, matched non-JEPA/classical comparators, '
              'sealed evaluation and real warehouse validation remain necessary before claiming a learned advance.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Repository root (default: this checkout)')
    args = parser.parse_args()
    root = args.root.resolve()
    summary = audit(root)
    text = markdown(summary, root)
    output = root / 'artifacts/service_world/summary.json'
    document = root / 'docs/SERVICE_WORLD_RESULTS.md'
    for path, content in ((output, json.dumps(summary, indent=2, allow_nan=False) + '\n'),
                          (document, text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(content)
        temporary.replace(path)
    print(json.dumps({'completed': True, 'status': summary['status'], **summary['counts'],
                      'trained_models': 0, 'learned_policy_evaluations': 0,
                      'sota_established': False, 'novelty_established': False}, indent=2))


if __name__ == '__main__':
    main()
