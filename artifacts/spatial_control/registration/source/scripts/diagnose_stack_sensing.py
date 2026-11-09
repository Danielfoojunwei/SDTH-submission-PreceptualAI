"""Passive diagnosis of archived scan interruptions and STR trade-offs.

Only the standard library is used. No scientific package is imported and no
world, model, forward pass, timing measurement or counterfactual is executed.
--write creates derived outputs; the default verifies their exact contents.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/stack_sensing_diagnosis'
BASE_COMMIT = '1441f68089173279536afc11fc17a3926115cd5c'
SOURCE_COUNTS = {'semantic_jepa': 80, 'selected_jepa': 230,
                 'jepa_repair': 386, 'costaware_recovery': 422}
CAUSES = ('no_radio_up', 'all_paths_blocked', 'weak_link', 'contention', 'other')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(inputs, name, expected=None):
    path = ROOT / name
    actual = digest(path)
    require(expected is None or actual == expected, 'Changed input: ' + name)
    require(name not in inputs or inputs[name] == actual, 'Inconsistent input: ' + name)
    inputs[name] = actual
    return path


def pinned(inputs, name):
    raw = subprocess.check_output(['git', 'show', BASE_COMMIT + ':' + name], cwd=ROOT)
    return bind(inputs, name, hashlib.sha256(raw).hexdigest())


def protected_inputs(inputs):
    counts = {}
    for study, count in SOURCE_COUNTS.items():
        directory = 'artifacts/' + study + '/registration/'
        registered = read(pinned(inputs, directory + 'source.json'))['source_sha256']
        require(len(registered) == count, 'Protected source count changed: ' + study)
        for name, expected in registered.items():
            bind(inputs, name, expected)
            bind(inputs, directory + 'source/' + name, expected)
        counts[study] = {'registered_source_files': count,
                         'current_and_capsule_bytes_verified': True}
    return counts


def arithmetic_proof(inputs):
    name = 'src/wirelessjev/costaware_recovery/sim.py'
    text = bind(inputs, name).read_text()
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                   for node in ast.parse(text).body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                   and isinstance(node.value, ast.Constant)}
    require((assignments['SLOT'], assignments['CONTROL_PERIOD'], assignments['CONTROL_DEADLINE'])
            == (.01, 5, 5), 'Native control clock changed')
    for fragment in ('scanning = scan_until > k', 'up = (outage_until <= k) & ~scanning',
                     'new_ctrl = ((k + scan_phase) % CONTROL_PERIOD) == 0',
                     'late = ctrl_age >= CONTROL_DEADLINE',
                     'go_scan = (delivered | watchdog_scan) & due',
                     'k + 1 + int(round((SCAN_S if scan_duration_s is None else scan_duration_s) / SLOT))'):
        require(fragment in text, 'Scan/control source semantics changed: ' + fragment)
    slots = round(.09 / assignments['SLOT'])
    phases = [{'first_next_generation_offset_slots': offset,
               'deadline_window_slot_offsets': list(range(offset, offset + 5)),
               'entire_window_inside_scan_outage': offset + 4 <= slots}
              for offset in range(1, 6)]
    require(slots == 9 and all(row['entire_window_inside_scan_outage'] for row in phases),
            'Five-phase arithmetic failed')
    return {'slot_seconds': .01, 'control_period_slots': 5, 'control_deadline_slots': 5,
            'scan_work_seconds': .09, 'unavailable_offsets_after_delivered_control': list(range(1, 10)),
            'first_usable_slot_offset': 10, 'phase_arithmetic': phases,
            'conclusion': 'Every uninterrupted nine-slot ordinary scan contains a complete five-slot control opportunity, unless the episode boundary truncates it; no second usable control path is assumed.',
            'not_a_counterfactual_count': True,
            'limits': ['Overlapping handover outages are not additional independent misses.',
                      'Finite episode endpoints and pending commands require separate accounting.',
                      'The no_radio_up attribution combines scanning and handovers.']}


def archived_groups(inputs):
    groups = []
    for study, cost, first in (('jepa_repair', .03, 2124000),
                               ('costaware_recovery', 4., 2134000)):
        folder = 'artifacts/' + study + '/'
        manifest = read(pinned(inputs, folder + 'verification/manifest.json'))
        evidence = manifest['evidence_sha256']
        report = read(bind(inputs, folder + 'report.json', evidence[folder + 'report.json']))
        protocol = read(bind(inputs, folder + 'protocol.json', evidence[folder + 'protocol.json']))
        require(protocol['scan_outage_seconds'] == .09 and protocol['scan_period_seconds'] == 5
                and protocol['robots'] == 100 and protocol['duration_seconds'] == 60,
                'Archived scan assumptions differ')
        for family in ('radiomap', 'frozen_fleet_jepa'):
            rows = []
            for site in range(first, first + 8):
                name = folder + f'cases/confirmation-{site}-h{round(cost * 1000):04d}-{family}-repair'
                receipt = read(bind(inputs, name + '.receipt.json', evidence[name + '.receipt.json']))
                archive = bind(inputs, name + '.json.gz', evidence[name + '.json.gz'])
                require(digest(archive) == receipt['archive_sha256'], 'Archive/receipt differs')
                row = json.loads(gzip.decompress(archive.read_bytes()))
                require(receipt['case'] == row['case'] == ['confirmation', site, cost, family, 'repair']
                        and value_digest(row['result']) == receipt['result_sha256']
                        and row['completed'] is True and row['replay'] is False
                        and row['live_replication'] is False, 'Case/result membership differs')
                result = row['result']
                require(result['control_sent'] == 120000
                        and sum(result['miss_causes'].values()) == result['control_misses'],
                        'Native counters or exclusive cause attribution differ')
                rows.append(row)
            missed = sum(row['result']['control_misses'] for row in rows)
            sent = sum(row['result']['control_sent'] for row in rows)
            mean = next(item for item in report['means'] if item['family'] == family
                        and item['controller_variant'] == 'repair' and item['handover_seconds'] == cost)
            require(math.isclose(missed / sent, mean['command_loss'], abs_tol=1e-12),
                    'Report differs from native packet counts')
            jobs = [job for row in rows for job in row['jobs']]
            groups.append({'study': study, 'handover_seconds': cost, 'family': family,
                'controller_variant': 'repair', 'independent_sites': list(range(first, first + 8)),
                'cases': 8, 'control_sent': sent, 'control_misses': missed,
                'command_loss': missed / sent,
                'exclusive_first_condition_causes': {cause: sum(row['result']['miss_causes'][cause]
                                                               for row in rows) for cause in CAUSES},
                'mean_dup_fraction_of_fleet_slots': math.fsum(row['result']['dup_fraction'] for row in rows) / 8,
                'scans_started': sum(row['telemetry']['scans_started'] for row in rows),
                'scans_reported': sum(row['telemetry']['reports_delivered'] for row in rows),
                'jobs_with_no_latest_bin_heard_entries': sum(job['heard_entries'] == 0 for job in jobs),
                'forecast_jobs': len(jobs),
                'nominal_periodic_scan_count': 100 * (60 // 5) * 8,
                'nominal_scan_count_is_not_an_isolated_miss_count': True,
                'contention_cause_count_is_zero': all(row['result']['miss_causes']['contention'] == 0 for row in rows),
                'collision_lambda': sorted({row['result']['collision_lambda'] for row in rows}),
                'interference': sorted({row['result']['interference'] for row in rows})})
    return groups


def old_str_ablation(inputs):
    campaign = read(pinned(inputs, 'artifacts/native/campaign.json'))
    pinned(inputs, 'docs/NATIVE_STACK_RESULTS.md')
    selected = {arm: sorted([row for row in campaign['rows'] if row['arm'] == arm],
                           key=lambda row: row['scene'])
                for arm in ('native', 'native-no_duplication')}
    require(all([row['scene'] for row in rows] == list(range(500, 506))
                for rows in selected.values()), 'Historical ablation scenes changed')
    means = {arm: math.fsum(row['control_miss'] for row in rows) / 6
             for arm, rows in selected.items()}
    return {'earlier_separate_cohort': True, 'scenes': list(range(500, 506)),
            'duration_seconds': campaign['duration_s'], 'mean_command_loss': means,
            'observed_relative_reduction_without_duplication': 1 - means['native-no_duplication'] / means['native'],
            'mechanism': 'The current STR surrogate subtracts 3 dB and halves each link rate; cross-link retries remain a separate mechanism.',
            'current_study_effect_not_established': True,
            'not_paired_with_current_cohorts': True}


def diagnose():
    inputs = {}
    closures = protected_inputs(inputs)
    proof = arithmetic_proof(inputs)
    groups = archived_groups(inputs)
    str_ablation = old_str_ablation(inputs)
    value = {'schema': 1, 'base_commit': BASE_COMMIT,
        'post_hoc_passive_diagnosis': True, 'scientific_execution_performed': False,
        'new_worlds': 0, 'new_forwards': 0, 'new_training': 0, 'new_timing': 0,
        'protected_source_closures': closures, 'archive_cases_verified': 32,
        'source_arithmetic': proof, 'archived_groups': groups,
        'earlier_str_ablation': str_ablation,
        'chosen_next_design': {
            'nested_variants': ['fifo', 'latest', 'latest_single', 'latest_single_sense'],
            'fixed_forecasters': ['radiomap', 'frozen_fleet_jepa'], 'handover_seconds': [4.0],
            'whole_scan_preserved': True, 'scan_outage_seconds': .09,
            'skip_only_ordinary_due_scan_when_all': ['control silence is below 200 ms',
                'current best available observed pilot RSSI is at least -67 dBm, its captured AP matches the association, and refresh age is below the existing 100 ms cadence',
                'consumed forecast age is at most five seconds and the next five seconds remain within its thirty-second support',
                'one fixed band stays at or above -67 dBm across the next zero-to-five seconds, using current pilot plus age-aligned forecast increments'],
            'unknown_or_stale_forecast': 'Perform the ordinary due scan.',
            'emergency_scan_after_200ms_silence': 'Never suppressed by the risk gate.',
            'suppressed_scan': 'Retain due status and reassess ordinary starts after five seconds; the emergency watchdog may override that interval. Only a genuine full scan start clears due/reassessment.',
            'not_selected': 'Chunked/resumable scanner behavior and extra control radios are not assumed.',
            'benefit_not_yet_established': True,
            'risk': 'Skipping observations can create overconfidence or delay discovering better APs. Fresh independent native evaluations must retain all packet, worst-decile, silence, video and bulk gates.'},
        'limits': ['Cause attribution is descriptive and assigns each missed packet to its first highest-priority encountered condition.',
                   'no_radio_up combines scans and handovers; it is not isolated avoidable regret.',
                   'Nominal scan arithmetic is not a measured counterfactual benefit.',
                   'Fast and slow cohorts have different sites and cannot be subtracted as a paired cost effect.',
                   'The old STR ablation is a different study and does not establish a new-controller benefit.',
                   'Zero recorded contention does not establish that real warehouses have no contention.',
                   'No new model/objective, hardware, production, arbitrary-layout or SOTA claim follows.']}
    # This second pass detects changes to every old source, capsule and raw
    # input that was read, including files shared across nested closures.
    for name, expected in inputs.items():
        require(digest(ROOT / name) == expected, 'Input changed during diagnosis: ' + name)
    return value, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write derived diagnosis and hash receipt')
    args = parser.parse_args()
    value, inputs = diagnose()
    raw = encoded(value)
    receipt = {'schema': 1, 'base_commit': BASE_COMMIT, 'passive_only': True,
               'source_and_capsules_verified_before_and_after': True,
               'no_worlds_forwards_fits_or_timing': True,
               'script_sha256': digest(Path(__file__)), 'inputs_sha256': dict(sorted(inputs.items())),
               'outputs_sha256': {'artifacts/stack_sensing_diagnosis/diagnosis.json': hashlib.sha256(raw).hexdigest(),
                                  'docs/STACK_SENSING_DIAGNOSIS.md': digest(ROOT / 'docs/STACK_SENSING_DIAGNOSIS.md')}}
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'diagnosis.json').write_bytes(raw)
        (OUT / 'receipt.json').write_bytes(encoded(receipt))
    else:
        require((OUT / 'diagnosis.json').read_bytes() == raw, 'Derived diagnosis differs; explicit --write required')
        require(read(OUT / 'receipt.json') == receipt, 'Derived input/script receipt differs')
    print(json.dumps({'cases': value['archive_cases_verified'], 'protected_closures': SOURCE_COUNTS,
                      'old_inputs_verified_before_and_after': len(inputs), 'derived_outputs_match': True,
                      'scientific_execution_performed': False}, indent=2))


if __name__ == '__main__':
    main()
