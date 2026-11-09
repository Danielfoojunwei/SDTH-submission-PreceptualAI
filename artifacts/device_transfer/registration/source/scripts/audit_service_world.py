"""Read-only integrity audit of the native service-world diagnostic archive.

This checks receipts, public-input archives and executed source. It neither
trains a model nor reruns a simulator, and it cannot establish novelty, SOTA,
causal-policy performance or physical deployment readiness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/service_world'


class AuditError(ValueError):
    """The archived evidence does not satisfy its registered contract."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer(value, name, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f'{name} must be an integer >= {minimum}')
    return value


def safe_relative(name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and name != '.',
            f'Unsafe source path: {name}')
    return relative


def original_holdout_sites(root):
    """Inspect registrations, never open their held-out cases or inputs."""
    sites = set()
    for path in (root / 'artifacts').glob('*/protocol.json'):
        if path.parent.name == 'service_world':
            continue
        for name, values in read(path).items():
            if ('test' in name or 'holdout' in name) and name.endswith('_sites'):
                if isinstance(values, list):
                    sites.update(v for v in values if isinstance(v, int))
    return sites


def check_sources(root, out, manifest):
    hashes = manifest['source_sha256']
    require(isinstance(hashes, dict) and hashes, 'Missing executed-source identities')
    required = {str(p.relative_to(root)) for p in (root / 'src/wirelessjev/service_world').glob('*.py')}
    required.add('artifacts/service_world/protocol.json')
    require(len(required) >= 7 and required <= set(hashes), 'Executed-source manifest omits service-world source')
    copied = out / 'executed_source'
    require({str(p.relative_to(copied)) for p in copied.rglob('*') if p.is_file()} == set(hashes),
            'Executed-source copy file set differs from manifest')
    for name, expected in hashes.items():
        relative = safe_relative(name)
        require(isinstance(expected, str) and len(expected) == 64, f'Invalid source digest: {name}')
        require(digest(root / relative) == expected, f'Current source digest mismatch: {name}')
        require(digest(copied / relative) == expected, f'Copied source digest mismatch: {name}')
    require(manifest.get('learned_policy_qualified') is False, 'Snapshot must not qualify learned control')
    return hashes


def packet_counters(result, robots, duration, name, *, require_detail=False):
    missed = integer(result['control_misses'], f'{name} total misses')
    total = integer(result['control_sent'], f'{name} total commands')
    require(total == robots * duration * 20 and missed <= total,
            f'{name}: incomplete 20-Hz native traffic or invalid loss count')
    detail = result.get('packet_detail')
    require(not require_detail or detail is not None, f'{name}: missing detailed factual counters')
    if detail is not None:
        misses = np.asarray(detail['misses'])
        sent = np.asarray(detail['sent'])
        require(misses.shape == sent.shape == (robots,), f'{name}: wrong packet counter axes')
        require(misses.dtype.kind in 'iu' and sent.dtype.kind in 'iu', f'{name}: packet counts need integer units')
        require((misses >= 0).all() and (sent >= misses).all(), f'{name}: invalid absolute packet counts')
        require(int(misses.sum()) == missed, f'{name}: per-robot miss count mismatch')
        require(int(sent.sum()) == total, f'{name}: per-robot sent count mismatch')
    require(sum(result['miss_causes'].values()) == result['control_misses'], f'{name}: miss causes do not reconcile')
    require(np.isclose(result['control_miss'], result['control_misses'] / max(result['control_sent'], 1)),
            f'{name}: miss fraction does not reconcile')


def forecast_forks(case, protocol):
    timing = case['forecast_timing']
    require(len(timing) == protocol['duration_seconds'], 'Incomplete forecast timing ledger')
    previous = 0.
    for decision, row in enumerate(timing):
        require(row['decision'] == decision and row['history_cutoff_exclusive'] == 60 + decision,
                'Forecast issue order or causal cutoff differs')
        service = row['charged_seconds']
        require(isinstance(service, (int, float)) and np.isfinite(service) and service >= 0,
                'Invalid charged forecast duration')
        previous = max(float(decision), previous) + service
        require(row['ready_slot'] == int(np.ceil(previous * 100 - 1e-10)),
                'Forecast completion is inconsistent with FIFO compute charges')
    forks = {timing[d]['ready_slot'] for d in protocol['checkpoint_forecast_issues_seconds']}
    require(len(forks) == len(protocol['checkpoint_forecast_issues_seconds']), 'Checkpoint forecasts collapsed to one event')
    horizon = int(protocol['label_horizon_seconds'] * 100)
    require(all(slot + horizon <= protocol['duration_seconds'] * 100 for slot in forks),
            'A label suffix exceeds the factual scenario')
    return forks


def public_input_keys(path, case, protocol):
    require(digest(path) == case['inputs_sha256'], f'Public-input digest mismatch: {path.name}')
    with np.load(path, allow_pickle=False) as data:
        base_fields = {'slot', 'robot', 'links', 'globals', 'observed', 'elapsed'}
        selection_fields = {'selection_observation_slot', 'decision_age_seconds'}
        require(set(data.files) in (base_fields, base_fields | selection_fields),
                f'Unexpected public-input fields: {path.name}')
        slot, robot = data['slot'], data['robot']
        count = integer(case['input_windows'], 'input_windows', 1)
        require(slot.shape == robot.shape == (count,), 'Public-input index count mismatch')
        require(slot.dtype.kind in 'iu' and robot.dtype.kind in 'iu', 'Public-input indices need integer units')
        require(((robot >= 0) & (robot < protocol['robots'])).all(), 'Public-input robot outside fleet')
        keys = list(zip(slot.tolist(), robot.tolist()))
        require(len(set(keys)) == count, 'Duplicate public-input window identity')
        links, glob, observed, elapsed = (data[k] for k in ('links', 'globals', 'observed', 'elapsed'))
        require(links.ndim == 5 and links.shape[0] == count and links.shape[-2:] == (3, 12),
                'Public input must preserve window/AP/band/feature axes')
        require(links.shape[1] > 0 and links.shape[2] > 0, 'Empty public radio history')
        require(observed.dtype.kind == 'b' and observed.shape == links.shape, 'Missing radio observation masks')
        require(glob.shape == (count, links.shape[1], 8) and elapsed.shape == (count, links.shape[1]),
                'Global features or elapsed-time axes differ')
        require(np.isfinite(links[observed]).all() and np.isfinite(glob).all()
                and np.isfinite(elapsed).all() and (elapsed >= 0).all(), 'Nonfinite or negative public-input values')
        selection = {}
        if selection_fields <= set(data.files):
            observed_slot, age = data['selection_observation_slot'], data['decision_age_seconds']
            require(observed_slot.shape == age.shape == (count,), 'Selection timestamp axes differ')
            require(observed_slot.dtype.kind in 'iu' and age.dtype.kind == 'f',
                    'Selection timestamps need integer slots and floating-point seconds')
            require((observed_slot <= slot).all() and np.isfinite(age).all()
                    and ((age >= 0) & (age < .1)).all()
                    and np.allclose(age, (slot - observed_slot) * .01, rtol=1e-7, atol=1e-9),
                    'Selection evidence must be causal and younger than 100 ms')
            selection = dict(zip(keys, observed_slot.tolist()))
        return set(keys), links.shape[2], selection


def check_case(case, key, root, out, protocol, hashes):
    require(case.get('completed') is True, f'Incomplete case: {key}')
    require(case['source_sha256'] == hashes, f'Case source identity differs: {key}')
    require(case.get('simulation_only') is True and case.get('no_learned_policy_evaluated') is True
            and case.get('no_heldout_access') is True, f'Case scope overclaim: {key}')
    robots = integer(case['robots'], 'robots', 1)
    require(robots == protocol['robots'], f'Case fleet differs: {key}')
    input_keys, aps, selection = public_input_keys(out / 'inputs' / (key + '.npz'), case, protocol)
    forks = forecast_forks(case, protocol)
    require({slot for slot, _ in input_keys} == forks, f'Public windows do not cover checkpoint set: {key}')
    for slot in forks:
        require(0 < sum(s == slot for s, _ in input_keys) <= protocol['focal_robots_per_checkpoint'],
                f'Incorrect focal sampling count: {key}/{slot}')
    records = case['records']
    seen = set()
    for row in records:
        identity = (integer(row['slot'], 'record slot'), integer(row['robot'], 'record robot'))
        require(identity in input_keys, f'Record lacks its causal public window: {key}/{identity}')
        if 'selection_observation_slot' in row:
            require(identity in selection and row['selection_observation_slot'] == selection[identity],
                    f'Record selection timestamp differs from causal public input: {key}/{identity}')
        require(row['hold_slots'] == 100 and row['canonical_recovery'] is True,
                f'Intervention disables canonical recovery: {key}/{identity}')
        current = integer(row['current'], 'current AP')
        target = integer(row['target'], 'target AP')
        require(current < aps and target < aps, 'Record AP outside public input axes')
        require(row['kind'] in ('stay', 'move') and ((row['kind'] == 'stay') == (current == target)),
                'Typed action disagrees with target AP')
        action_key = (*identity, row['kind'])
        require(action_key not in seen, f'Duplicate action: {key}/{action_key}')
        seen.add(action_key)
        baseline = integer(row['baseline_misses'], 'baseline misses')
        intervention = integer(row['intervention_misses'], 'intervention misses')
        require(integer(row['delta_misses'], 'delta misses', -10**12) == intervention - baseline,
                f'Raw packet delta does not equal intervention minus baseline: {key}/{identity}')
        baseline_gap = integer(row['baseline_window_max_gap_slots'], 'baseline maximum gap slots')
        intervention_gap = integer(row['intervention_window_max_gap_slots'], 'intervention maximum gap slots')
        require(integer(row['delta_window_max_gap_slots'], 'delta gap slots', -10**12) == intervention_gap - baseline_gap,
                f'Raw gap delta does not equal intervention minus baseline: {key}/{identity}')
    require(all((*identity, 'stay') in seen for identity in input_keys), 'Public window lacks its explicit stay intervention')
    replays = integer(case['factual_replays'], 'factual_replays')
    require(replays == len(forks), f'Factual suffix replay count differs: {key}')
    runs = integer(case['native_runs'], 'native_runs')
    require(runs == 2 + replays + len(records), f'Native run count does not reconcile: {key}')
    prototype = case['prototype']
    require(prototype.get('untrained') is True and prototype.get('accuracy_or_control_result') is False,
            'Untrained prototype is presented as learned performance')
    require(prototype.get('zero_residual_exact') is True and prototype.get('unknown_hazard_is_none') is True,
            'Prototype loses its known-physics/unknown-hazard contract')
    integer(prototype['inference_parameters'], 'prototype inference parameters', 1)
    require(0 < prototype['clients'] <= min(8, robots), 'Invalid prototype probe fleet')
    require(prototype['physical_reference'].startswith('Known-unavailability lower bounds only;'),
            'Physical lower bound must not be relabeled expected service')
    for name in ('factual', 'legacy'):
        packet_counters(case[name], robots, protocol['duration_seconds'], f'{key}/{name}',
                        require_detail=name == 'factual')
    require(case['factual']['packet_detail']['native_proposal_ttl_slots'] == 100,
            'Canonical comparator must expire stale proposals after one second')
    # The legacy ablation stores aggregate counters, not a second checkpoint
    # archive. Its configured None TTL is in the verified executed study source.
    if 'packet_detail' in case['legacy']:
        require(case['legacy']['packet_detail']['native_proposal_ttl_slots'] is None,
                'Legacy pending-command reference must retain unbounded proposal semantics')
    return {'native_runs': runs, 'factual_replays': replays, 'records': len(records),
            'input_windows': len(input_keys)}


def check_optional_summary(summary, totals, *, cases=None, root=None, hashes=None):
    """Check scope/count fields if emitted; never interpret headroom as policy skill."""
    expected = totals | {'trained_models': 0, 'learned_policy_evaluations': 0,
                         'sota_established': False, 'novelty_established': False,
                         'original_holdouts_unused': True, 'simulation_only': True,
                         'learned_policy_qualified': False}
    for name, value in expected.items():
        if name in summary and not (name == 'cases' and isinstance(summary[name], list)):
            require(summary[name] == value, f'Summary field does not reconcile: {name}')
    forbidden_true = {'sota_established', 'novelty_established', 'learned_policy_qualified',
                      'learned_control_qualified', 'superiority_established', 'physical_deployment_qualified',
                      'joint_fleet_policy_result', 'learned_model_contribution'}
    def walk(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in forbidden_true:
                    require(item is False, f'Summary overclaims {name}')
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(summary)
    if cases is None:
        return
    expected_counts = {'cases': totals['cases'], 'native_runs': totals['native_runs'],
                       'exact_factual_suffix_replays': totals['factual_replays'],
                       'input_windows': totals['input_windows'], 'recorded_interventions': totals['records']}
    require(summary.get('counts') == expected_counts, 'Summary count ledger differs from receipts')
    require(summary.get('learned_accuracy_or_control_score') is None,
            'An untrained prototype cannot have a learned accuracy/control score')
    prototype = summary['prototype']
    require(prototype.get('untrained') is True and prototype.get('accuracy_or_control_result') is False,
            'Summary presents the untrained probe as learned performance')
    require(prototype['inference_parameter_counts'] ==
            sorted({case['prototype']['inference_parameters'] for case in cases}), 'Summary prototype size differs')

    def metrics(value, rows, arm):
        misses = sum(c[arm]['control_misses'] for c in rows)
        sent = sum(c[arm]['control_sent'] for c in rows)
        require(value['misses'] == misses and value['sent'] == sent, 'Summary absolute packet counts differ')
        require(np.isclose(value['miss_fraction'], misses / sent, rtol=1e-12, atol=1e-15)
                and np.isclose(value['misses_per_million_sent'], misses / sent * 1e6, rtol=1e-12, atol=1e-12),
                'Summary packet rates differ from integer counters')

    def effects(value, records):
        deltas = [row['delta_misses'] for row in records]
        expected = {'actions': len(deltas), 'beneficial': sum(d < 0 for d in deltas),
                    'harmful': sum(d > 0 for d in deltas), 'zero': sum(d == 0 for d in deltas),
                    'minimum_delta_misses': min(deltas) if deltas else None,
                    'maximum_delta_misses': max(deltas) if deltas else None}
        require(value == expected, 'Summary intervention signs/range differ from raw deltas')

    case_rows = summary['cases']
    lookup = {(c['scene'], float(c['cost'])): c for c in cases}
    require(len(case_rows) == len(cases) and
            {(r['scene'], float(r['cost_seconds'])) for r in case_rows} == set(lookup),
            'Summary case identity set differs')
    for row in case_rows:
        case = lookup[(row['scene'], float(row['cost_seconds']))]
        metrics(row['factual_expiring_proposals'], [case], 'factual')
        metrics(row['legacy_unbounded_pending'], [case], 'legacy')
        effects(row['interventions'], case['records'])
        require(row['classical_factual_minus_legacy_misses'] ==
                case['factual']['control_misses'] - case['legacy']['control_misses'], 'Summary classical delta differs')
    require(set(summary['by_cost']) == {str(float(c['cost'])) for c in cases}, 'Summary cost set differs')
    for cost, row in summary['by_cost'].items():
        selected = [c for c in cases if float(c['cost']) == float(cost)]
        records = [r for c in selected for r in c['records']]
        require(row['cases'] == len(selected), 'Summary cost case count differs')
        metrics(row['factual_expiring_proposals'], selected, 'factual')
        metrics(row['legacy_unbounded_pending'], selected, 'legacy')
        effects(row['interventions'], records)
        for kind in ('stay', 'move'):
            effects(row['interventions_by_kind'][kind], [r for r in records if r['kind'] == kind])
        ablation = row['classical_pending_expiry_ablation']
        delta = sum(c['factual']['control_misses'] - c['legacy']['control_misses'] for c in selected)
        sent = sum(c['factual']['control_sent'] for c in selected)
        require(ablation['delta_misses_factual_minus_legacy'] == delta
                and np.isclose(ablation['delta_misses_per_million_sent'], delta / sent * 1e6),
                'Summary classical pending-expiry effect differs')
        groups = {(c['scene'], r['slot'], r['robot']) for c in selected for r in c['records']}
        negatives = {(c['scene'], r['slot'], r['robot']) for c in selected
                     for r in c['records'] if r['delta_misses'] < 0}
        require(row['checkpoint_robot_groups'] == len(groups)
                and row['groups_with_a_negative_recorded_action'] == len(negatives),
                'Summary hindsight group counts differ')
    linkage = summary['linkage']
    require(linkage['source_sha256'] == hashes, 'Summary executed-source identity differs')
    for name in ('completion', 'executed_source_manifest', 'protocol'):
        require(digest(root / safe_relative(linkage[name + '_path'])) == linkage[name + '_sha256'],
                f'Summary {name} linkage differs')
    require(digest(root / safe_relative(linkage['reporter_path'])) == linkage['reporter_sha256'],
            'Summary reporter source has changed')
    links = linkage['case_receipts']
    require(len(links) == len(cases) and
            {Path(link['path']).stem for link in links} ==
            {f"{c['scene']}-{c['cost']:g}" for c in cases}, 'Summary case linkage set differs')
    for link in links:
        require(digest(root / safe_relative(link['path'])) == link['sha256']
                and digest(root / safe_relative(link['inputs_path'])) == link['inputs_sha256'],
                'Summary case/input linkage digest differs')


def audit(root=ROOT, out=None):
    root = Path(root)
    out = root / 'artifacts/service_world' if out is None else Path(out)
    protocol, manifest, completion = (read(out / name) for name in
                                      ('protocol.json', 'executed_source.json', 'completion.json'))
    require(completion.get('completed') is True, 'Native diagnostic is not complete')
    hashes = check_sources(root, out, manifest)
    require(completion['source_sha256'] == hashes, 'Completion source identity differs')
    require(completion['trained_models'] == 0 and completion['learned_policy_evaluations'] == 0,
            'This archive must not imply model training or learned control evaluation')
    require(completion.get('sota_established') is False and completion.get('novelty_established') is False,
            'Diagnostic evidence cannot certify SOTA or novelty')
    require(completion.get('original_holdouts_unused') is True and protocol.get('original_holdouts_unused') is True,
            'Original holdouts must remain unused')
    require(protocol.get('simulation_only') is True and protocol['intervention_hold_seconds'] == 1
            and protocol['canonical_proposal_expiry_seconds'] == 1, 'Protocol scope or canonical timing differs')
    expected = {(seed, float(cost)) for seed in protocol['diagnostic_sites'] for cost in protocol['handover_seconds']}
    require(len(expected) == len(protocol['diagnostic_sites']) * len(protocol['handover_seconds']),
            'Protocol duplicates a diagnostic site/cost')
    require(not set(protocol['diagnostic_sites']) & original_holdout_sites(root),
            'Diagnostic sites overlap an original registered holdout')
    cases = [read(path) for path in sorted((out / 'cases').glob('*.json'))]
    require(len(cases) == completion['cases'] == len(expected), 'Completed case count differs from protocol')
    identities = {(case['scene'], float(case['cost'])) for case in cases}
    require(identities == expected and len(identities) == len(cases), 'Case site/cost set differs from protocol')
    totals = {'cases': len(cases), 'native_runs': 0, 'factual_replays': 0, 'records': 0, 'input_windows': 0}
    for case in cases:
        key = f"{case['scene']}-{case['cost']:g}"
        require(read(out / 'cases' / (key + '.json')) == case, 'Case filename does not match site/cost identity')
        values = check_case(case, key, root, out, protocol, hashes)
        for name, count in values.items():
            totals[name] += count
    require({p.stem for p in (out / 'inputs').glob('*.npz')} ==
            {f"{c['scene']}-{c['cost']:g}" for c in cases}, 'Public-input archive set differs from case receipts')
    for name in ('native_runs', 'factual_replays'):
        require(completion[name] == totals[name], f'Completion total does not reconcile: {name}')
    summary_path = out / 'summary.json'
    if summary_path.exists():
        check_optional_summary(read(summary_path), totals, cases=cases, root=root, hashes=hashes)
    return {'passed': True, **totals, 'source_files': len(hashes), 'trained_models': 0,
            'learned_policy_evaluations': 0, 'original_holdouts_unused': True,
            'sota_established': False, 'novelty_established': False,
            'simulation_only': True, 'summary_checked': summary_path.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = audit(args.root)
    except (AuditError, KeyError, TypeError, OSError, ValueError) as error:
        print(json.dumps({'passed': False, 'error': str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
