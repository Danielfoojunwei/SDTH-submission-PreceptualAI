"""Passive post-hoc comparison of the published JEPA/RadioMap packages.

Reads only previously completed archives. No scientific package is imported;
no world, model, forward pass, fit, or timing measurement is executed.
Descriptive intervals cannot select or qualify a new candidate.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/jepa_advantage_diagnosis'
DOC = ROOT / 'docs/JEPA_ADVANTAGES_DIAGNOSIS.md'
BASE = '9a8577d2f5334bbf1f4ea805811b40c35a11212a'
FAMILIES = ('radiomap', 'frozen_fleet_jepa')
VARIANTS = ('fifo', 'latest', 'latest_single', 'latest_single_sense')
CLOSURES = {'semantic_jepa': 80, 'selected_jepa': 230, 'jepa_repair': 386,
            'costaware_recovery': 422, 'freshness_stack': 446}
BOOTSTRAP_SEED, DRAWS = 20261105, 20000
# Direction describes the reported quantity, never a new success criterion.
METRICS = {
    'command_loss': ('fraction', 'lower'),
    'worst_decile_loss': ('fraction', 'lower'),
    'worst_decile_pending_as_misses': ('fraction', 'lower'),
    'max_silence_seconds': ('s', 'lower'),
    'p99_robot_max_silence_seconds': ('s', 'lower'),
    'video_vs_original': ('fraction', 'higher'),
    'bulk_mbps': ('fleet Mbps', 'higher'),
    'snr_rmse_db': ('dB, native accepted query sets', 'lower'),
    'route_delta_rmse_db': ('dB, native accepted query sets', 'lower'),
    'ap_choice_regret_db': ('dB, native accepted query sets', 'lower'),
    'measured_forecast_elapsed_seconds': ('summed elapsed wall s per trial', 'lower'),
    'measured_elapsed_per_accepted_job_seconds': ('elapsed wall s/job; different queries', 'lower'),
    'accepted_jobs': ('jobs/trial', None),
    'consumed_forecasts': ('callbacks/trial', None),
    'maximum_held_source_age_seconds': ('s, source clock not measurement clock', 'lower'),
    'whole_scan_starts': ('starts/trial', None),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read(path):
    return json.loads(Path(path).read_text())


def bind(inputs, name, expected=None):
    actual = digest(ROOT / name)
    require(expected is None or actual == expected, 'Changed protected input: ' + name)
    require(name not in inputs or inputs[name] == actual, 'Inconsistent input: ' + name)
    inputs[name] = actual
    return ROOT / name


def pinned(inputs, name):
    raw = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
    return bind(inputs, name, hashlib.sha256(raw).hexdigest())


def protect(inputs):
    closures = {}
    for study, count in CLOSURES.items():
        directory = 'artifacts/' + study + '/registration/'
        names = read(pinned(inputs, directory + 'source.json'))['source_sha256']
        require(len(names) == count, 'Protected closure count differs: ' + study)
        for name, expected in names.items():
            bind(inputs, name, expected)
            bind(inputs, directory + 'source/' + name, expected)
        closures[study] = count
    name = 'artifacts/freshness_stack/verification/manifest.json'
    manifest = read(pinned(inputs, name))
    require(len(manifest['sources_sha256']) == 376 and len(manifest['evidence_sha256']) == 665,
            'Published final manifest count differs')
    for group in ('sources_sha256', 'evidence_sha256'):
        for name, expected in manifest[group].items():
            bind(inputs, name, expected)
    # Current narrative documents can evolve. Their old hashes apply to BASE,
    # not to the parent agent's next result documentation.
    for name, expected in manifest['documents_sha256'].items():
        raw = subprocess.check_output(['git', 'show', BASE + ':' + name], cwd=ROOT)
        require(hashlib.sha256(raw).hexdigest() == expected, 'Historical document differs: ' + name)
    return {'scientific_closures': closures, 'named_source_entries': 376,
            'named_evidence_entries': 665, 'historical_document_entries': 5,
            'current_narrative_documents_not_required_to_match_old_commit': True}


def verified_rows(inputs):
    report = read(pinned(inputs, 'artifacts/freshness_stack/report.json'))
    protocol = read(pinned(inputs, 'artifacts/freshness_stack/protocol.json'))
    require(protocol['confirmation_sites'] == list(range(2144000, 2144008)), 'Unexpected old sites')
    expected = {(site, family, variant) for site in protocol['confirmation_sites']
                for family in FAMILIES for variant in VARIANTS}
    summaries = report['every_case']
    require(len(summaries) == len(expected) == 64, 'Incomplete published main matrix')
    rows = []
    for row in summaries:
        key = row['site'], row['family'], row['controller_variant']
        require(key in expected, 'Duplicate or unexpected main identity')
        expected.remove(key)
        name = ('artifacts/freshness_stack/cases/confirmation-' + str(key[0])
                + '-h4000-' + key[1] + '-' + key[2])
        receipt = read(bind(inputs, name + '.receipt.json'))
        archive = bind(inputs, name + '.json.gz', receipt['archive_sha256'])
        raw = json.loads(gzip.decompress(archive.read_bytes()))
        require(raw['case'] == ['confirmation', key[0], 4., key[1], key[2]]
                and raw['completed'] is True and raw['replay'] is False
                and raw['live_replication'] is False, 'Archive identity differs')
        for metric in ('control_misses', 'control_sent', 'bulk_mbps'):
            require(math.isclose(raw['result'][metric], row[metric], rel_tol=1e-12, abs_tol=1e-12),
                    'Reported native metric differs: ' + metric)
        require(row['forecast_metrics'] == raw['forecast_metrics'], 'Reported RF diagnostics differ')
        t = row['timing']; consumed = raw['forecast_consumption']; jobs = raw['jobs']
        require(len(jobs) == t['forecast_jobs'] and len(consumed) == t['consumptions']
                and len(raw['arrivals']) == t['forecast_arrivals'] == 60,
                'Recorded work membership differs')
        require([d['decision'] for d in raw['decision_trace']]
                == [d['forecast_decision'] for d in consumed], 'Control callbacks differ from arrivals')
        elapsed = sum(job['actual_compute_seconds'] for job in jobs)
        require(math.isclose(elapsed, t['actual_compute_seconds_total'], rel_tol=1e-12),
                'Measured elapsed-work sum differs')
        held = max((consumed[i+1]['slot'] - 1 if i+1 < len(consumed) else 5999) / 100
                   - event['forecast_decision'] for i, event in enumerate(consumed))
        require(math.isclose(held, t['maximum_held_forecast_age_seconds'], abs_tol=1e-12),
                'Held source age differs')
        metrics = {name: row[name] for name in ('command_loss', 'worst_decile_loss',
            'worst_decile_pending_as_misses', 'max_silence_seconds',
            'p99_robot_max_silence_seconds', 'video_vs_original', 'bulk_mbps')}
        metrics.update(snr_rmse_db=row['forecast_metrics']['rmse_db'],
            route_delta_rmse_db=row['forecast_metrics']['route_delta_rmse_db'],
            ap_choice_regret_db=row['forecast_metrics']['ap_choice_regret_db'],
            measured_forecast_elapsed_seconds=elapsed,
            measured_elapsed_per_accepted_job_seconds=elapsed / len(jobs),
            accepted_jobs=len(jobs), consumed_forecasts=len(consumed),
            maximum_held_source_age_seconds=held,
            whole_scan_starts=row['sensing']['scan_starts'])
        require(set(metrics) == set(METRICS) and all(math.isfinite(v) for v in metrics.values()),
                'Invalid metric fields')
        rows.append({'site': key[0], 'family': key[1], 'variant': key[2],
            'condition': 'drift' if key[0] % 2 else 'stationary',
            'metrics': metrics, 'archive': name + '.json.gz',
            'archive_sha256': receipt['archive_sha256']})
    require(not expected, 'Main cases missing')
    return report, rows


def comparisons(rows):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for variant in VARIANTS:
        for condition in ('all', 'stationary', 'drift'):
            subset = [row for row in rows if row['variant'] == variant
                      and (condition == 'all' or row['condition'] == condition)]
            sites = sorted({row['site'] for row in subset})
            require(len(sites) == (8 if condition == 'all' else 4), 'Condition membership differs')
            paired = {(row['site'], row['family']): row for row in subset}
            draws = rng.integers(0, len(sites), size=(DRAWS, len(sites)))
            metrics = {}
            for name, (units, preferred) in METRICS.items():
                a = np.asarray([paired[site, 'frozen_fleet_jepa']['metrics'][name] for site in sites])
                b = np.asarray([paired[site, 'radiomap']['metrics'][name] for site in sites])
                difference = a - b
                lo, hi = np.quantile(difference[draws].mean(1), [.025, .975])
                metrics[name] = {'units': units, 'preferred_direction': preferred,
                    'jepa_mean': float(a.mean()), 'radiomap_mean': float(b.mean()),
                    'mean_jepa_minus_radiomap': float(difference.mean()),
                    'descriptive_95_interval': [float(lo), float(hi)],
                    'relative_jepa_minus_radiomap': float(a.mean() / b.mean() - 1) if b.mean() else None,
                    'lower_at_sites': int((difference < 0).sum()),
                    'equal_at_sites': int((difference == 0).sum()),
                    'higher_at_sites': int((difference > 0).sum()),
                    'paired_site_differences': difference.tolist()}
            results.append({'variant': variant, 'condition': condition, 'sites': sites,
                'independent_units': len(sites), 'post_hoc_descriptive_only': True,
                'multiple_comparison_adjustment': None, 'qualification_or_selection_performed': False,
                'metrics': metrics,
                'maximum_over_all_robots_and_sites': {
                    family: max(paired[site, family]['metrics']['max_silence_seconds'] for site in sites)
                    for family in FAMILIES}})
    return results


def diagnose():
    inputs = {}; preserved = protect(inputs); report, rows = verified_rows(inputs)
    sources = ['src/wirelessjev/fleet_control.py', 'src/wirelessjev/semantic_jepa/runtime.py',
               'src/wirelessjev/fleetwm/data.py', 'src/wirelessjev/frontier/fast.py',
               'src/wirelessjev/freshness_stack/sim.py', 'src/wirelessjev/freshness_stack/rollout.py']
    texts = {name: pinned(inputs, name).read_text() for name in sources}
    require("family='hybrid_jepa'" in texts[sources[1]] and "if self.family=='fleet_jepa':" in texts[sources[0]]
            and 'self.rmap=fit_map(self.geometry,history)' in texts[sources[0]], 'Hybrid map dependence changed')
    require("fit_reference(self.geometry, past, strong=True)" in texts[sources[1]], 'Strong numerical prior differs')
    require("'drift' if site % 2 else 'main'" in texts[sources[-1]], 'Condition assignment differs')
    require('else arrived is not None' in texts[sources[-2]]
            and 'ctrl_link = np.argmax(pmin, axis=1)' in texts[sources[-2]], 'Native callback interface differs')
    prior = read(pinned(inputs, 'artifacts/fleet_confirmation/summary.json'))
    selected = read(pinned(inputs, 'artifacts/selected_jepa/report.json'))
    response = read(pinned(inputs, 'artifacts/continual_radio/summary.json'))
    value = {'schema': 1, 'base_commit': BASE, 'post_hoc_passive_diagnosis': True,
        'new_worlds': 0, 'new_forwards': 0, 'new_training': 0, 'new_timing': 0,
        'published_confirmation_cases_verified': len(rows), 'preserved': preserved,
        'comparison_sign': 'JEPA minus RadioMap; units and direction stated per metric',
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'replicates': DRAWS,
            'unit': 'paired whole procedural site', 'interval': 'descriptive two-sided 95%',
            'all_metrics_stages_and_conditions_retained': True, 'new_qualification': False},
        'comparisons': comparisons(rows), 'every_case': rows,
        'original_registered_primary_tests_unchanged': report['primary'],
        'architecture_facts': {'current_jepa_reference_is_hybrid_with_operational_radiomap_fits': True,
            'radiomap_reference_uses_strong_prior': True,
            'jepa_map_and_strong_reference_map_are_not_identical_pipelines': True,
            'rf_metrics_are_different_accepted_query_sets_and_histories': True,
            'rf_diagnostic_intervals_are_not_matched_input_accuracy_superiority': True,
            'forecast_arrival_drives_native_planning_and_control_link_callbacks': True,
            'current_native_phy_mcs_measurement_refresh_is_already_faster': True,
            'neural_objective_necessity_proven': False, 'map_free_or_intrinsically_cheaper_proven': False},
        'earlier_context_not_pooled': {
            'fleet_confirmation': {'primary_candidate': prior['fixed_candidate'], 'primary_success': prior['primary_success'],
                'arm_ppm': {name: row['ppm'] for name, row in prior['arms'].items()},
                'jepa_was_secondary': True, 'idealized_observations_and_30ms_handovers': True},
            'compact_context_all_references_pass': selected['both_costs_all_references_pass'],
            'compact_context_objective_superiority_established': False,
            'continual_response_jepa_specific_gate': response['jepa_specific_gate_passed']},
        'limits': ['All analyses are post-hoc and descriptive; no new winner or qualification is selected.',
            'Native RF query schedules, histories and map priors differ; RF differences are not matched accuracy tests.',
            'Four-site condition groups are especially small and share the same procedural CAD.',
            'Means of site maxima differ from the maximum over all robots/sites.',
            'Measured work is elapsed wall time, not processor utilization; per-job workloads differ.',
            'Bulk/video/tail intervals are not new registered superiority claims.',
            'Learning memory, uncertainty and numerical assimilation are not unique to JEPA.',
            'Lower RF error or nonconstant latents do not imply deadline improvement.',
            'No new model fit, timing calibration, packet replay, counterfactual or physical observation occurred.',
            'Earlier cohorts and their assumptions are not pooled or subtracted as paired cost effects.',
            'No production, hardware, map-free, objective-necessity, SOTA or arbitrary-layout claim follows.']}
    for name, expected in inputs.items():
        require(digest(ROOT / name) == expected, 'Input changed during passive audit: ' + name)
    return value, inputs


def markdown(value):
    final = next(row for row in value['comparisons'] if row['variant'] == 'latest_single_sense' and row['condition'] == 'all')
    m = final['metrics']
    lines = ['# Where JEPA has evidence, and where it still loses', '',
        '**JEPA has promising native RF diagnostics, but the current final package does not beat RadioMap on command delivery.** This is a post-hoc audit of all 64 published confirmation cases, with no new worlds, model calls, fitting, timing or selection.', '',
        'The [machine-readable diagnosis](../artifacts/jepa_advantage_diagnosis/diagnosis.json), [input/output receipt](../artifacts/jepa_advantage_diagnosis/receipt.json) and [audit script](../scripts/audit_jepa_advantages.py) preserve every stage, site and condition. All five earlier source closures and the published 376-source/665-evidence entries were checked before and after.', '',
        '## Final-stack comparison on all eight sites', '',
        'Differences are JEPA minus RadioMap. Intervals below are **unadjusted descriptive paired whole-site 95% intervals**, not new superiority tests or a way to select a winner. These 20,000 descriptive resamples use seed 20261105 and do not replace the original registered intervals or decisions. Different accepted query times, histories and map pipelines prevent interpreting RF rows as matched-input accuracy tests.', '',
        '| Quantity | JEPA | RadioMap | Difference | Descriptive 95% interval | Site directions |',
        '|---|---:|---:|---:|---|---|']
    specs = [('Command loss (%)', 'command_loss', 100), ('Worst-decile loss (%)', 'worst_decile_loss', 100),
        ('Original video (%)', 'video_vs_original', 100), ('Bulk (fleet Mbps)', 'bulk_mbps', 1),
        ('Native SNR RMSE (dB)', 'snr_rmse_db', 1), ('Native route-delta RMSE (dB)', 'route_delta_rmse_db', 1),
        ('Native AP regret (dB)', 'ap_choice_regret_db', 1),
        ('Measured forecast elapsed work (s/trial)', 'measured_forecast_elapsed_seconds', 1)]
    for label, key, scale in specs:
        x = m[key]; lo, hi = x['descriptive_95_interval']
        lines.append(f"| {label} | {x['jepa_mean']*scale:.6f} | {x['radiomap_mean']*scale:.6f} | {x['mean_jepa_minus_radiomap']*scale:+.6f} | [{lo*scale:+.6f}, {hi*scale:+.6f}] | {x['lower_at_sites']} lower / {x['equal_at_sites']} tied / {x['higher_at_sites']} higher |")
    lines += ['', 'The strongest consistent RF pattern is route-delta error: JEPA is lower on all eight native site summaries. SNR RMSE and AP regret are lower on six. These are leads for a fair matched-query study, not proof that the JEPA objective is better. Its RF error is lower while its final command-loss mean is higher; prediction quality is not the service endpoint.', '',
        'The bulk point mean is slightly higher for JEPA, but only three sites favor it and one large gain drives the average. The interval retains that uncertainty. Original-video and worst-decile point means favor RadioMap; their descriptive intervals retain uncertainty. The maximum over every robot and all eight sites is **8.37 seconds for both**; the mean of site maxima is a different quantity and does not create a new continuity claim.', '',
        '## Stationary and drifting conditions', '',
        'Even site IDs use the stationary case; odd IDs introduce hidden mean-field drift. This grouping follows the frozen source but is a **post-hoc four-site diagnostic**, not an independently qualified subgroup. All subgroup values and intervals remain in the JSON.', '',
        '| Condition | Sites | JEPA loss (%) | Map loss (%) | JEPA native SNR RMSE | Map native SNR RMSE | JEPA route-delta RMSE | Map route-delta RMSE |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in ('stationary', 'drift'):
        group = next(x for x in value['comparisons'] if x['variant'] == 'latest_single_sense' and x['condition'] == c)
        k = group['metrics']
        lines.append(f"| {c} | 4 | {k['command_loss']['jepa_mean']*100:.6f} | {k['command_loss']['radiomap_mean']*100:.6f} | {k['snr_rmse_db']['jepa_mean']:.6f} | {k['snr_rmse_db']['radiomap_mean']:.6f} | {k['route_delta_rmse_db']['jepa_mean']:.6f} | {k['route_delta_rmse_db']['radiomap_mean']:.6f} |")
    lines += ['', 'Both groups have lower native RF point-error means for JEPA but higher command-loss means. The drift group therefore does not rescue a claim of superior dynamic control. Query/history differences remain in both groups.', '',
        '## What the architecture offers, and what it does not establish', '',
        'The current learned reference conditions on delivered fleet histories, geometry and planned route queries, and shares neural weights across robots/AP-band queries. Those components can represent nonlinear context and route-conditioned effects. Their practical value still needs task-level tests against numerical and matched-supervised controls.', '',
        'This particular JEPA is a **hybrid** that continues fitting a RadioMap. Its map pipeline differs from the strong-prior numerical reference. It is neither map-free nor demonstrated intrinsically cheaper. The final JEPA package spends about three times the summed measured forecast elapsed work despite making fewer calls. RadioMap already has online assimilation and posterior uncertainty, so those properties are not exclusive JEPA advantages.', '',
        'Earlier evidence is mixed rather than erased: a separate fast/idealized-observation hybrid study has a positive supervised primary and a favorable secondary JEPA point result; it does not prove JEPA-objective necessity. The compact context study fails its registered superiority comparisons, and the RT-backed response-identification JEPA loses to numerical Bayesian identification. Those cohorts cannot be pooled with this one.', '',
        '## Concrete leads for new gains', '',
        '1. **Decouple fast radio control from slow model arrivals.** The current controller updates native planning/control-link choices on forecast consumption. Latest admission makes the ensemble fresher but reduces its callbacks from 43 to 29 per trial; latest-only service worsens. Reusing a cached forecast with current pilots for a separately timed local control loop could retain prediction value without more neural forwards. The causal effect requires a new fixed comparison; existing faster MCS updates must not be presented as newly invented.',
        '2. **Improve genuinely delivered observation density.** Associated-link pilot reports may provide positive measurements without full scans. Partial reports must not mark every unmeasured AP as censored, must preserve capture position/time/AP/band identity, and must pay reporting/availability costs. Numerical assimilation must receive the same evidence. A model must handle or be trained for the resulting mask distribution.',
        '3. **Measure decision-relevant prediction on common inputs.** Route-delta and AP-relative contrasts are more directly used by control than a static absolute offset. Match history, map prior, query times and labels, then compare JEPA with a matched supervised model. Lower native error in this audit is not that test.', '',
        'These are hypotheses for prospective work, not newly validated improvements. There is no physical-car, production, SOTA, objective-necessity, arbitrary-layout or indefinite-safety claim. No previous result, model or qualification rule changed.']
    return ('\n'.join(lines) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write derived diagnosis, document and receipt')
    args = parser.parse_args()
    value, inputs = diagnose(); raw = encoded(value); doc = markdown(value)
    receipt = {'schema': 1, 'base_commit': BASE, 'passive_only': True,
        'source_and_capsules_verified_before_and_after': True,
        'no_worlds_forwards_fits_or_timing': True, 'script_sha256': digest(Path(__file__)),
        'inputs_sha256': dict(sorted(inputs.items())), 'outputs_sha256': {
            'artifacts/jepa_advantage_diagnosis/diagnosis.json': hashlib.sha256(raw).hexdigest(),
            'docs/JEPA_ADVANTAGES_DIAGNOSIS.md': hashlib.sha256(doc).hexdigest()}}
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'diagnosis.json').write_bytes(raw); DOC.write_bytes(doc)
        (OUT / 'receipt.json').write_bytes(encoded(receipt))
    else:
        require((OUT / 'diagnosis.json').read_bytes() == raw and DOC.read_bytes() == doc,
                'Derived diagnosis/document changed')
        require(read(OUT / 'receipt.json') == receipt, 'Derived input/output receipt changed')
    print(json.dumps({'cases': len(value['every_case']), 'paired_sites': 8,
        'all_stages_and_post_hoc_conditions_retained': True, 'bound_inputs': len(inputs),
        'preserved_source_closures': value['preserved']['scientific_closures'],
        'new_scientific_execution': False, 'mode': 'write' if args.write else 'verify'}))


if __name__ == '__main__':
    main()
