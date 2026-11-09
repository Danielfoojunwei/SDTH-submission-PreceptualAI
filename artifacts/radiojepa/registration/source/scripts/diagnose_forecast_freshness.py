"""Passively inspect existing FIFO ledgers; never import scientific runners.

This is a derived, post-hoc diagnosis. It creates no worlds, forwards, timings,
counterfactual packet outcomes, or new model selection.
"""
from __future__ import annotations

import ast
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/forecast_freshness_diagnosis'
STUDIES = ('selected_jepa', 'jepa_repair', 'costaware_recovery')
SOURCES = ('src/wirelessjev/jepa_repair/runtime.py',
    'src/wirelessjev/closed_loop/runtime.py', 'src/wirelessjev/frontier/timing.py',
    'src/wirelessjev/closed_loop/control.py', 'src/wirelessjev/costaware_recovery/sim.py',
    'src/wirelessjev/costaware_recovery/rollout.py',
    'src/wirelessjev/semantic_jepa/runtime.py', 'src/wirelessjev/native/world.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def summary(values):
    return {'count': len(values), 'mean': sum(values) / len(values) if values else None,
            'median': quantile(values, .5), 'p95': quantile(values, .95),
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def protected_sources():
    """Check earlier scientific source bytes without executing their modules."""
    closures = {}
    for study, count in (('semantic_jepa', 80), ('selected_jepa', 230),
                         ('jepa_repair', 386), ('costaware_recovery', 422)):
        path = ROOT / 'artifacts' / study / 'registration/source.json'
        value = read(path)
        sources = value['source_sha256']
        assert len(sources) == count, study
        for name, expected in sources.items():
            assert digest(ROOT / name) == expected, (study, name)
        closures[study] = {'source_entries': len(sources),
                           'registration_sha256': digest(path)}
    return closures


def case_summary(path, horizons):
    row = json.loads(gzip.decompress(path.read_bytes()))
    receipt_path = path.with_name(path.name.removesuffix('.json.gz') + '.receipt.json')
    receipt = read(receipt_path)
    assert digest(path) == receipt['archive_sha256'], path
    jobs = row['jobs']
    duration = int(row['result']['duration_s'])
    assert len(jobs) == duration == 60
    previous = 0.
    ready = []
    for decision, job in enumerate(jobs):
        assert job['decision'] == decision and job['history_cutoff_exclusive'] == 60 + decision
        previous = max(float(decision), previous) + job['charged_seconds']
        assert job['ready_slot'] == math.ceil(previous * 100 - 1e-10)
        ready.append(job['ready_slot'])
    consumed, held_ages = [], []
    last = -1
    for slot in range(duration * 100):
        index = bisect_right(ready, slot) - 1
        if index < 0:
            continue
        held_ages.append(slot / 100 - jobs[index]['decision'])
        if index > last:
            consumed.append({'slot': slot, 'forecast_decision': jobs[index]['decision'],
                             'ready_slot': ready[index], 'age_seconds': slot / 100 - jobs[index]['decision']})
            last = index
    logged = row.get('forecast_consumption')
    if logged is not None:
        assert logged == consumed, path
    ages = [entry['age_seconds'] for entry in consumed]
    clipped_count = [sum(h + age > horizons[-1] for h in horizons) for age in ages]
    # The advanced forecast is regridded at HORIZONS. Do not describe all
    # unsupported query time as an exactly constant terminal segment.
    constant_segments = [horizons[-1] - next((h for h in horizons if h + age >= horizons[-1]), horizons[-1])
                         for age in ages]
    unused = [job for job in jobs if job['ready_slot'] >= duration * 100]
    return {'study': path.parts[-3], 'case': row['case'], 'family': row['family'],
        'handover_seconds': row['handover_seconds'], 'controller_variant': row.get('controller_variant'),
        'training_seed': row.get('training_seed'), 'site': row.get('site', row['case'][1]),
        'archive': str(path.relative_to(ROOT)), 'archive_sha256': digest(path),
        'consumption_source': 'logged_and_recomputed_exact' if logged is not None else 'recomputed_from_original_paid_ledger',
        'jobs_issued': len(jobs), 'forecasts_consumed': len(consumed),
        'jobs_not_consumed_in_episode': len(unused),
        'actual_predictor_seconds_for_unconsumed_jobs': sum(j['actual_prediction_seconds'] for j in unused),
        'first_forecast_consumption': consumed[0], 'last_forecast_consumption': consumed[-1],
        'last_issued_job_ready_seconds': ready[-1] / 100,
        'age_at_consumption_seconds': summary(ages),
        'actual_wall_age_of_held_forecast_seconds': summary(held_ages),
        'requested_30s_window_outside_original_support_seconds': summary([min(horizons[-1], age) for age in ages]),
        'clipped_query_knot_count': summary(clipped_count),
        'constant_terminal_segment_on_advanced_grid_seconds': summary(constant_segments),
        'warm_charged_service_seconds': summary([j['charged_seconds'] for j in jobs if j['kind'] == 'warm']),
        'logged_command_loss_unchanged': row['result']['control_miss']}


def main():
    before = protected_sources()
    horizons = None
    for node in ast.parse((ROOT / 'src/wirelessjev/native/world.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HORIZONS' for t in node.targets):
            horizons = ast.literal_eval(node.value)
    assert tuple(horizons) == (0., 1., 2., 3., 4., 5., 10., 20., 30.)
    archives = [path for study in STUDIES for path in sorted((ROOT / 'artifacts' / study / 'cases').glob('confirmation-*.json.gz'))]
    assert len(archives) == 128 + 48 + 32
    inputs = {str(path.relative_to(ROOT)): digest(path) for path in archives}
    for path in archives:
        receipt = path.with_name(path.name.removesuffix('.json.gz') + '.receipt.json')
        inputs[str(receipt.relative_to(ROOT))] = digest(receipt)
    for name in SOURCES:
        inputs[name] = digest(ROOT / name)
    rows = [case_summary(path, horizons) for path in archives]
    groups = defaultdict(list)
    for row in rows:
        groups[row['study'], row['family'], row['handover_seconds']].append(row)
    group_rows = []
    for (study, family, cost), cases in sorted(groups.items()):
        group_rows.append({'study': study, 'family': family, 'handover_seconds': cost,
            'archive_cases': len(cases), 'distinct_sites': len({r['site'] for r in cases}),
            'total_jobs': sum(r['jobs_issued'] for r in cases),
            'total_consumptions': sum(r['forecasts_consumed'] for r in cases),
            'unconsumed_jobs': sum(r['jobs_not_consumed_in_episode'] for r in cases),
            'maximum_consumption_age_seconds': max(r['age_at_consumption_seconds']['maximum'] for r in cases),
            'maximum_held_actual_wall_age_seconds': max(r['actual_wall_age_of_held_forecast_seconds']['maximum'] for r in cases),
            'maximum_last_issued_ready_seconds': max(r['last_issued_job_ready_seconds'] for r in cases),
            'maximum_warm_charged_service_seconds': max(r['warm_charged_service_seconds']['maximum'] for r in cases)})
    assert protected_sources() == before
    assert all(digest(ROOT / name) == expected for name, expected in inputs.items())
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'schema': 1, 'derived_post_hoc': True, 'new_worlds_forwards_fits_retiming_or_packet_replays': False,
        'new_loss_or_superiority_claim': False, 'horizons_seconds': list(horizons),
        'confirmation_archives': len(rows), 'protected_source_closures': before,
        'input_sha256': inputs, 'groups': group_rows, 'every_archive': rows,
        'limits': ['Descriptive schedule arithmetic on already observed archives, not a counterfactual service result.',
            'Per-slot held age is time since source decision; the original controller does not re-run planning each slot.',
            'Unsupported 30s query duration differs from constant terminal interpolation segment after sparse regridding.',
            'No source, historical criterion, timing veto or packet result was changed.']}
    target = OUT / 'diagnosis.json'
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    receipt = {'schema': 1, 'derived_only': True, 'script_sha256': digest(Path(__file__)),
        'inputs_sha256': inputs, 'outputs_sha256': {str(target.relative_to(ROOT)): digest(target)},
        'protected_sources_checked_before_and_after': before,
        'scientific_execution': False}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'confirmation_archives': len(rows), 'groups': group_rows}, indent=2))


if __name__ == '__main__':
    main()
