"""Independent passive arithmetic audit; never imports the research implementation.

Reads archived reports/predictions/checkpoint tensors only. No RT, model
construction, neural forward, fitting, or scientific evaluation is available.
All writes stay in this sibling review directory.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/adaptive_radio'
REVIEW = Path(__file__).resolve().parent
TARGETS = ('center_dbm', 'lower_dbm', 'upper_dbm', 'heard', 'censored',
           'censor_upper_dbm', 'detection_threshold_dbm')
INPUTS = {'context_a', 'context_b', 'mask_a', 'mask_b', 'scheduled_mask_a',
          'scheduled_mask_b', 'query', 'query_basis', 'query_fspl_dbm', 'query_offset_db'}
CELLS = ('joint', 'regression', 'rf_erased', 'pure_geometry')
SEEDS = (0, 1, 2)
HASHED = {}


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    path = Path(path)
    if path not in HASHED:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                h.update(block)
        HASHED[path] = h.hexdigest()
    return HASHED[path]


def read(path):
    return json.loads(Path(path).read_text())


def ref(value):
    relative = Path(value['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe reference')
    path = ROOT / relative
    require(sha(path) == value['sha256'], f'Reference bytes differ: {relative}')
    return path


def close(actual, expected, label):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f'{label}: dictionary keys differ')
        for key, value in expected.items():
            close(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f'{label}: sequence length differs')
        for index, value in enumerate(expected):
            close(actual[index], value, f'{label}[{index}]')
    elif isinstance(expected, (float, np.floating)):
        require(math.isclose(actual, expected, rel_tol=2e-10, abs_tol=2e-10),
                f'{label}: {actual!r} != {expected!r}')
    else:
        require(actual == expected, f'{label}: {actual!r} != {expected!r}')


def scores(a):
    """Independently score archived Gaussian forecasts against actual reports."""
    mean = a['mean'].astype('f8'); std = a['std'].astype('f8')
    heard = a['heard']; censored = a['censored']; known = heard | censored
    require(heard.dtype == bool and censored.dtype == bool and not (heard & censored).any(),
            'Invalid detection masks')
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(),
            'Invalid forecast')
    error = mean[heard] - a['center_dbm'][heard]
    lo = (np.maximum(a['lower_dbm'][heard], a['detection_threshold_dbm'][heard]) - mean[heard]) / std[heard]
    hi = (a['upper_dbm'][heard] - mean[heard]) / std[heard]
    require((hi > lo).all(), 'Empty detected quantization event')
    # Compute each bin using its better-conditioned CDF or survival-function tail.
    log_mass = np.empty_like(lo)
    for i, (left, right) in enumerate(zip(lo, hi)):
        upper, lower = (log_ndtr(-left), log_ndtr(-right)) if left > 0 else (log_ndtr(right), log_ndtr(left))
        log_mass[i] = upper + np.log(-np.expm1(min(lower - upper, -np.finfo(float).eps)))
    log_censor = log_ndtr((a['censor_upper_dbm'][censored] - mean[censored]) / std[censored])
    nll = -np.r_[log_mass, log_censor]
    p = ndtr((mean[known] - a['detection_threshold_dbm'][known]) / std[known])
    require(np.isfinite(nll).all(), 'Nonfinite observation likelihood')
    return {'rmse_db': float(np.sqrt(np.mean(error ** 2))) if error.size else None,
            'p95_abs_db': float(np.quantile(abs(error), .95)) if error.size else None,
            'observation_nll': float(np.mean(nll)) if nll.size else None,
            'detector_brier': float(np.mean((p - heard[known]) ** 2)) if p.size else None,
            'heard_count': int(heard.sum()), 'censored_count': int(censored.sum()),
            'missing_count': int((~known).sum()), 'finite': True}


def pairs(cases):
    require(set(cases) == {0, 1, 2}, 'Three material cases required')
    anchor = cases[0]; errors = []; detection = []; usable = True
    key_names = ('pose_index', 'ap_index', 'band_index', 'direction', 'device')
    for condition in (1, 2):
        changed = cases[condition]
        require(all(np.array_equal(anchor[k], changed[k]) for k in key_names), 'Physical query correspondence differs')
        ah, bh = anchor['heard'], changed['heard']
        ak, bk = ah | anchor['censored'], bh | changed['censored']
        both = ah & bh; count = int(both.sum()); usable &= count > 0
        errors.extend(((changed['mean'].astype('f8') - anchor['mean'].astype('f8')) -
                       (changed['center_dbm'] - anchor['center_dbm']))[both].astype('f8'))
        detection.append({'pair': [0, condition], 'both_heard_count': count,
            'anchor_heard_only_count': int((ah & ~bh & bk).sum()),
            'changed_heard_only_count': int((bh & ~ah & ak).sum()),
            'both_censored_count': int((ak & bk & ~ah & ~bh).sum()),
            'missing_pair_count': int((~(ak & bk)).sum())})
    pooled = {name: np.concatenate([cases[k][name] for k in (0, 1, 2)])
              for name in (*TARGETS, 'mean', 'std')}
    return {'delta_available': bool(usable),
            'delta_rmse_db': float(np.sqrt(np.mean(np.square(errors)))) if usable else None,
            'delta_unavailable_reason': None if usable else 'at_least_one_registered_pair_has_no_both_heard_query',
            'delta_both_heard_count': len(errors), 'pair_detection': detection, **scores(pooled)}


def raw_stage(stage, registry):
    manifest_path = OUT / 'rt_data' / f'{stage}.json'
    manifest = read(manifest_path)
    require(manifest['completed'] and manifest['stage'] == stage, 'Incomplete RT stage')
    require(manifest['source_receipt_sha256'] == sha(OUT / 'registration/source.json'), 'Stage source mismatch')
    require(manifest['registry_sha256'] == sha(OUT / 'geometry_registry.json'), 'Stage registry mismatch')
    require(set(manifest['case_receipts']) == {case['case_id'] for layout in registry['stages'][stage]
                                             for case in layout['cases']}, 'Stage receipt cohort differs')
    cases = {}; total_heard = total_censored = total_missing = 0
    for layout in registry['stages'][stage]:
        previous = None; materials = []
        for registered in layout['cases']:
            case = registered['case_id']; receipt_path = ref(manifest['case_receipts'][case]); folder = receipt_path.parent
            receipt = read(receipt_path)
            close(receipt['case'], registered, f'{case}.spec')
            require(receipt['public_geometry'] == layout['public_geometry'], 'Public CAD differs across conditions')
            require(receipt['known_tx_power_dbm'] == layout['known_tx_power_dbm'], 'Known TX differs across conditions')
            for name, digest in receipt['files_sha256'].items():
                require(sha(folder / name) == digest, f'Raw file changed: {case}/{name}')
            raw = read(folder / 'reference20/receipt.json')
            common = (receipt['public_geometry_sha256'], receipt['mesh_sha256'], raw['spec']['seed'],
                      receipt['known_tx_power_dbm'])
            if previous is not None:
                require(common == previous, f'{case}: CAD/mesh/solver-seed/TX changed with material')
            previous = common; materials.append(raw['spec']['obstacle_material'])
            with np.load(folder / 'reference20/physical_channels.npz', allow_pickle=False) as archive:
                gain = archive['channel_power_gain']; solved = archive['solved']; cfr = archive['cfr']
                require(np.allclose(np.mean(abs(cfr.astype('complex128')) ** 2, axis=-1), gain, rtol=1e-12, atol=0),
                        'Stored physical power differs from linear CFR mean')
            with np.load(folder / 'public_views.npz', allow_pickle=False) as archive:
                public = {k: archive[k].copy() for k in archive.files}
            physical = np.zeros_like(gain)
            physical[solved] = layout['known_tx_power_dbm'] + 10 * np.log10(gain[solved])
            thermal = 10 * math.log10(1.380649e-23 * 290 * receipt['bandwidth_hz'] * 1000)
            for device, profile in enumerate(raw['receiver_profiles']):
                threshold = thermal + profile['noise_figure_db'] + profile['detection_snr_db']
                heard = solved & (physical >= threshold); censored = solved & ~heard
                require(np.array_equal(public['heard'][..., device], heard), 'Invented/missing heard events')
                require(np.array_equal(public['censored'][..., device], censored), 'Unresolved path mislabeled censored')
                reported_threshold = np.float32(threshold + profile['rssi_offset_db'])
                require(np.all(public['reported_detection_threshold_dbm'][..., device] == reported_threshold),
                        'Detection thresholds must also exist on heard rows')
                quant = profile['quantization_db']; clean = np.where(solved, physical, 0) + profile['rssi_offset_db']
                center = np.floor(clean / quant + .5) * quant
                require(np.array_equal(public['rssi_dbm'][..., device], np.where(heard, center, 0).astype('f4')),
                        'Reported point does not follow rendered gain/quantizer')
                interval = np.stack((np.where(heard, center - quant / 2, 0),
                                     np.where(heard, center + quant / 2, 0)), axis=-1).astype('f4')
                require(np.array_equal(public['quantization_interval_dbm'][..., device, :], interval), 'Quantization bins changed')
                require(np.array_equal(public['censor_upper_dbm'][..., device],
                                       np.where(censored, threshold + profile['rssi_offset_db'], 0).astype('f4')),
                        'Censor bound differs from physical sensitivity')
            total_heard += int(public['heard'].sum()); total_censored += int(public['censored'].sum())
            total_missing += int((~(public['heard'] | public['censored'])).sum())
            cases[case] = public
        require(set(materials) == {'metal', 'wood', 'brick'}, 'Material counterfactuals missing')
    require(len(cases) == 3 * len(registry['stages'][stage]), 'RT cohort differs')
    return cases, {'layouts': len(registry['stages'][stage]), 'cases': len(cases),
                   'heard': total_heard, 'censored': total_censored, 'missing': total_missing,
                   'same_CAD_mesh_TX_solver_seed_verified': True, 'material_counterfactuals_verified': True}


def frontend(training_cases):
    calibration = read(OUT / 'data/frontend-training.json')
    require(calibration['reference_device'] == 0 and not calibration['absolute_offset_identified']
            and not calibration['registry_true_offsets_used'] and not calibration['query_reports_or_band2_used'],
            'Frontend gauge/source declaration differs')
    for name, digest in calibration['source_files_sha256'].items():
        require(name.startswith('artifacts/adaptive_radio/rt_data/training') and 'physical_channels' not in name,
                'Calibration source outside public TRAIN reports/CAD')
        require(sha(ROOT / name) == digest, 'Calibration source bytes differ')
    seen = set()
    for row in calibration['paired_reports']:
        require(row['stage'] == 'training' and row['scene_id'] in training_cases and row['pose'] in (0, 2, 4, 6, 8, 10)
                and row['band'] in (0, 1) and row['device'] in (1, 2), 'Calibration queried a forbidden view')
        key = (row['scene_id'], row['pose'], row['ap'], row['band'], row['direction'], row['device'])
        require(key not in seen, 'Duplicated calibration pair'); seen.add(key)
        public = training_cases[row['scene_id']]; ix = (row['pose'], row['ap'], row['band'], row['direction'])
        require(public['heard'][ix + (0,)] and public['heard'][ix + (row['device'],)], 'Calibration pair was not co-observed')
        for prefix, device in (('reference', 0), ('device', row['device'])):
            require(row[f'{prefix}_reported_dbm'] == public['rssi_dbm'][ix + (device,)], 'Calibration point differs from actual report')
            close(row[f'{prefix}_interval_dbm'], public['quantization_interval_dbm'][ix + (device,)].tolist(), 'Calibration interval')
    for device in (1, 2):
        rows = [r for r in calibration['paired_reports'] if r['device'] == device]
        estimate = calibration['estimates'][str(device)]
        require(estimate['pair_count'] == len(rows), 'Calibration pair denominator differs')
        if rows:
            lower = max(r['device_interval_dbm'][0] - r['reference_interval_dbm'][1] for r in rows)
            upper = min(r['device_interval_dbm'][1] - r['reference_interval_dbm'][0] for r in rows)
            median = float(np.median([r['device_reported_dbm'] - r['reference_reported_dbm'] for r in rows]))
            known = lower <= upper + 1e-9
            expected = float(np.clip(median, lower, max(lower, upper))) if known else median
            close(estimate['offset_db'], expected, 'TRAIN-only relative offset')
            require(estimate['known'] == known, 'Calibration identifiability differs')
    return calibration


def prediction_stage(stage, public_cases):
    evaluation = read(OUT / 'results' / f'{stage}.json'); out = {}
    selected = read(OUT / 'selection.json')['selected_cell']
    calibration = read(OUT / 'data/frontend-training.json')
    for regime, members in evaluation['members'].items():
        expected = ({f'{selected}-seed{s}' for s in SEEDS} | {'radiomap_operator', 'bayesian_operator'}) if regime == 'sparse-held_band' else (
            {f'{cell}-seed{s}' for cell in CELLS for s in SEEDS} | {'radiomap_operator', 'bayesian_operator'} |
            {f'frozen_old-seed{s}' for s in SEEDS} | {f'{selected}-seed{s}-stale' for s in SEEDS})
        require(set(members) == expected, 'Evaluation member/seed cohort changed')
        out[regime] = {}
        targets_by_case = {}; query_by_case = {}
        for member_name, member in members.items():
            grouped = {}; metadata = {}; all_available = True
            for reference in member['cases']:
                receipt = read(ref(reference)); m = receipt['metadata']
                require(m['member'] == member_name and m['stage'] == stage and m['source_view'] == 'a', 'Prediction identity mismatch')
                path = ROOT / receipt['npz_path']; require(sha(path) == receipt['npz_sha256'], 'Prediction NPZ changed')
                with np.load(path, allow_pickle=False) as saved:
                    a = {k: saved[k].copy() for k in saved.files}
                if m['cell'] in CELLS:
                    beta, covariance = a['beta_mean'], a['beta_covariance']
                    require(beta.shape == (8,) and covariance.shape == (8, 8)
                            and np.isfinite(beta).all() and np.isfinite(covariance).all(), 'Invalid physical posterior archive')
                    require(np.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-6)
                            and np.linalg.eigvalsh((covariance + covariance.T) / 2).min() >= -1e-6,
                            'Physical posterior covariance is not symmetric positive semidefinite')
                require(INPUTS <= set(a) and not (INPUTS & set(TARGETS)), 'Target appears in model-input schema')
                source = public_cases[m['case_id']]
                ix = tuple(a[k] for k in ('pose_index', 'ap_index', 'band_index', 'direction', 'device'))
                expected_targets = {'center_dbm': source['rssi_dbm'][ix], 'lower_dbm': source['quantization_interval_dbm'][ix + (0,)],
                    'upper_dbm': source['quantization_interval_dbm'][ix + (1,)], 'heard': source['heard'][ix],
                    'censored': source['censored'][ix], 'censor_upper_dbm': source['censor_upper_dbm'][ix],
                    'detection_threshold_dbm': source['reported_detection_threshold_dbm'][ix]}
                for k in TARGETS:
                    require(np.array_equal(a[k], expected_targets[k]), f'{member_name}/{m["case_id"]}: target not actual report: {k}')
                poses = {0, 2, 4, 6, 8, 10} if m['held_band'] else {1, 3, 5, 7, 9, 11}
                require(set(a['pose_index']) == poses and set(a['band_index']) == ({2} if m['held_band'] else {0, 1}), 'Held band/query pose boundary changed')
                require(set(a['device']) == {0, 1, 2}, 'Held device leaked into main endpoint')
                for view in ('a', 'b'):
                    context, known, scheduled = a[f'context_{view}'], a[f'mask_{view}'], a[f'scheduled_mask_{view}']
                    require(context.shape == (24, 20), 'Context shape differs')
                    require(int(scheduled.sum()) == (8 if m['regime'] == 'sparse' else 24), 'Attempt schedule changed')
                    require(np.all(~known | scheduled), 'Known outside attempts')
                    require(np.array_equal(known, (context[:, 2] > 0) | (context[:, 3] > 0)), 'Known != heard or censored')
                    require(not context[~scheduled, :4].any(), 'Withheld rows became RF/status evidence')
                    require(np.all(np.isclose(context[:, 11], 0., atol=1e-6) |
                                   np.isclose(context[:, 11], math.log(5.2 / 2.4), atol=1e-6)), 'Context contains held band')
                    context_source = public_cases[m['context_case_id']]
                    device, direction = (0, 0) if view == 'a' else (1, 1)
                    estimate = calibration['estimates'][str(device)]
                    positions = [(p, ap, b, direction, device) for b in (0, 1)
                                 for p in (0, 2, 4, 6, 8, 10) for ap in (0, 1)]
                    ci = tuple(np.array([r[j] for r in positions]) for j in range(5))
                    planned = np.ones(24, bool) if m['regime'] == 'full' else np.isin(ci[0], (0, 10))
                    ch, cc = context_source['heard'][ci] & planned, context_source['censored'][ci] & planned
                    require(np.array_equal(scheduled, planned) and np.array_equal(known, ch | cc), 'Context availability differs from actual schedule/reports')
                    distance = np.linalg.norm(context_source['rx_positions_m'][ci[0]].astype('f8') -
                                              context_source['ap_positions_m'][ci[1]].astype('f8'), axis=-1)
                    tx = context[:, 19].astype('f8') * 30
                    fspl = tx - 20 * np.log10(4 * math.pi * distance * context_source['carrier_hz'][ci[2]] / 299792458.)
                    raw_context = np.column_stack((np.where(ch, (context_source['rssi_dbm'][ci] - estimate['offset_db'] - fspl) / 30, 0),
                        np.where(cc, (context_source['censor_upper_dbm'][ci] - estimate['offset_db'] - fspl) / 30, 0), ch, cc))
                    require(np.allclose(context[:, :4], raw_context, rtol=1e-6, atol=2e-7), 'Context RF values are not actual source reports')
                if m['stale']:
                    require(m['context_case_id'] == m['case_id'].rsplit('-rf', 1)[0] + '-rf0', 'Stale intervention source changed')
                else:
                    require(m['context_case_id'] == m['case_id'], 'Current context is stale')
                t = tuple(a[k].tobytes() for k in TARGETS)
                require(m['case_id'] not in targets_by_case or targets_by_case[m['case_id']] == t, 'Method-specific target cohort')
                targets_by_case[m['case_id']] = t
                query = tuple(a[k].tobytes() for k in ('query', 'query_basis', 'query_fspl_dbm', 'query_offset_db'))
                require(m['case_id'] not in query_by_case or query_by_case[m['case_id']] == query, 'Method-specific public query')
                query_by_case[m['case_id']] = query
                if m['available']:
                    close(receipt['metrics'], scores(a), 'case metrics')
                else:
                    require(m['cell'] == 'frozen_old' and receipt['metrics'] is None, 'Primary method abstained')
                    all_available = False
                grouped.setdefault(m['layout_id'], {})[m['condition']] = a
                metadata[m['layout_id']] = m
            require(len(grouped) == (4 if stage == 'development' else 24), 'Whole-geometry evaluation cohort differs')
            rows = []
            for saved in member['layouts']:
                if all_available:
                    if member['cell'] == 'pure_geometry' or member['stale']:
                        conditions = grouped[saved['layout_id']]
                        require(all(np.array_equal(conditions[0][key], conditions[k][key])
                                    for k in (1, 2) for key in ('mean', 'std')),
                                'Identical public geometry and fixed RF/schedule forecast changed across hidden conditions')
                    actual = pairs(grouped[saved['layout_id']])
                    close({k: saved[k] for k in actual}, actual, f'{regime}/{member_name}/{saved["layout_id"]}')
                    rows.append(actual | {'layout_id': saved['layout_id']})
                else:
                    # Supplementary abstention is explicitly retained; primary endpoints never use it.
                    rows.append(saved)
            out[regime][member_name] = rows
    return evaluation, out


def metric(rows, cell, name, *, stale=False):
    if cell in ('radiomap_operator', 'bayesian_operator'):
        return np.array([np.nan if r[name] is None else r[name] for r in rows[cell]], float)
    members = [[np.nan if r[name] is None else r[name] for r in rows[f'{cell}-seed{s}' + ('-stale' if stale else '')]] for s in SEEDS]
    a = np.asarray(members, float)
    return np.where(np.isfinite(a).all(axis=0), a.mean(axis=0), np.nan)


def gate(rows, selected):
    sparse = rows['sparse']
    arrays = [metric(sparse, c, 'delta_rmse_db', stale=stale) for c, stale in
              ((selected, False), ('radiomap_operator', False), ('rf_erased', False), (selected, True))]
    cohort = np.isfinite(arrays).all(axis=0); n = int(cohort.sum())
    current, radio_map, erased, stale = [float(a[cohort].mean()) if n else None for a in arrays]
    def means(name):
        a, b = metric(sparse, selected, name), metric(sparse, 'radiomap_operator', name)
        valid = np.isfinite(a) & np.isfinite(b)
        return (float(a[valid].mean()), float(b[valid].mean()), int(valid.sum())) if valid.any() else (None, None, 0)
    scalar, scalar_map, scalar_n = means('rmse_db'); tail, tail_map, tail_n = means('p95_abs_db')
    seed_scores = [float(np.mean([sparse[f'{selected}-seed{s}'][i]['delta_rmse_db'] for i in np.flatnonzero(cohort)])) if n else None for s in SEEDS]
    censored = sum(r['censored_count'] for r in sparse[f'{selected}-seed0'])
    finite = all(r['finite'] for regime in ('full', 'sparse') for member, data in rows[regime].items()
                 if not member.startswith('frozen_old') for r in data)
    checks = {'minimum_three_usable_layouts': n >= 3,
        'five_percent_better_than_rf_erased': n > 0 and current <= .95 * erased and current < erased,
        'five_percent_better_than_radiomap': n > 0 and current <= .95 * radio_map and current < radio_map,
        'every_seed_delta_no_worse_than_radiomap': n > 0 and all(x <= radio_map for x in seed_scores),
        'scalar_rmse_within_two_percent': scalar_n >= 3 and scalar <= 1.02 * scalar_map,
        'p95_within_two_percent': tail_n >= 3 and tail <= 1.02 * tail_map,
        'stale_context_hurts_at_least_five_percent': n > 0 and stale >= 1.05 * current and stale > current,
        'actual_censored_events_present': censored > 0, 'all_primary_predictions_and_scores_finite': finite}
    return {'test_permitted': all(checks.values()), 'checks': checks, 'usable_layouts': n,
        'selected_delta_rmse_db': current, 'radiomap_delta_rmse_db': radio_map,
        'rf_erased_delta_rmse_db': erased, 'stale_delta_rmse_db': stale,
        'selected_seed_delta_rmse_db': seed_scores, 'scalar_rmse_db': scalar,
        'radiomap_scalar_rmse_db': scalar_map, 'scalar_available_layouts': scalar_n,
        'p95_abs_db': tail, 'radiomap_p95_abs_db': tail_map, 'p95_available_layouts': tail_n,
        'censored_query_events': censored}


def bootstrap(a, b):
    diff = np.asarray(a, 'f8') - np.asarray(b, 'f8')
    draw = np.random.default_rng(21749).integers(len(diff), size=(20000, len(diff)))
    boot = diff[draw].mean(axis=1); interval = np.quantile(boot, [.00625, .99375])
    return {'layouts': len(diff), 'candidate_mean': float(np.mean(a)), 'reference_mean': float(np.mean(b)),
        'mean_paired_difference': float(diff.mean()), 'adjusted_interval': interval.tolist(), 'adjusted_confidence': .9875,
        'descriptive_95_interval': np.quantile(boot, [.025, .975]).tolist(), 'better_adjusted': bool(interval[1] < 0),
        'bootstrap_samples': 20000, 'bootstrap_seed': 21749,
        'unit': 'whole geometry, three RF conditions nested', 'scope': 'conditional on three trained members; seed spread descriptive'}


def audit():
    require((OUT / 'report.json').exists(), 'Campaign not complete: report.json is not available')
    require((OUT / 'verification/manifest.json').exists(), 'Wait for the campaign passive closure before independent replay')
    before_paths = {path for path in OUT.rglob('*') if path.is_file()}
    before = {path: sha(path) for path in before_paths}
    source = read(OUT / 'registration/source.json'); protocol = read(OUT / 'protocol.json')
    registry = read(OUT / 'geometry_registry.json'); publication = read(OUT / 'registration/git.json')
    require(len(source['source_sha256']) == 504, 'Unexpected frozen source cohort')
    for name, digest in source['source_sha256'].items():
        require(sha(ROOT / name) == digest, f'Frozen source changed: {name}')
    for name, digest in source['preflight_sha256'].items():
        require(sha(ROOT / name) == digest, 'Published preflight bytes differ')
    preflight = read(OUT / 'preflight/tests.json')
    require(preflight['exit_code'] == 0 and preflight['sources_before_sha256'] ==
            preflight['sources_after_sha256'] == source['source_sha256'], 'Passing preflight was not bound to frozen source')
    require(publication['commit'] == publication['remote_observed_commit'] == '9bd34ab688d8b74dc2e05df8a6c55fed87257f20', 'Publication differs')
    require(publication['source_receipt_sha256'] == sha(OUT / 'registration/source.json'), 'Publication source receipt differs')
    previous = ROOT / 'artifacts/radiojepa_spectral/verification/manifest.json'
    require(sha(previous) == source['previous']['manifest_sha256'], 'Previous manifest changed')
    old = read(previous)
    for group in ('source_sha256', 'evidence_sha256', 'documents_sha256'):
        for name, digest in old[group].items():
            require(sha(ROOT / name) == digest, f'Prior sealed bytes changed: {name}')
    expected_runs = {f'{cell}-seed{s}' for cell in CELLS for s in SEEDS}
    require({p.stem for p in (OUT / 'fits').glob('*.json')} == expected_runs, 'Missing or extra fits')
    fit_summary = {}; scores_by_cell = {}
    training_identity = {'source_receipt_sha256': sha(OUT / 'registration/source.json'),
        'protocol_sha256': sha(OUT / 'protocol.json'),
        'data_receipt_sha256': sha(OUT / 'data/prepared-training.json'),
        'rt_stage_receipts_sha256': {stage: sha(OUT / 'rt_data' / f'{stage}.json')
                                    for stage in ('training', 'validation')}}
    # Loading inert checkpoint tensors is permitted; no model constructor or forward exists here.
    import torch
    for run in sorted(expected_runs):
        receipt = read(OUT / 'fits' / f'{run}.json'); history = receipt['history']
        require(receipt['run_id'] == run == f'{receipt["cell"]}-seed{receipt["seed"]}', 'Fit cell/seed identity differs')
        close(receipt['identity'], training_identity, 'Fit data/source identity')
        close(receipt['settings'], {'cell': receipt['cell'], 'width': 64, 'em_steps': 3}, 'Fit architecture budget')
        require(receipt['training_bands'] == [0, 1] and receipt['batch_distinct_layouts'] == 24,
                'Fit band/layout admission changed')
        require(len(history) == 60 and all(len(h['updates']) == 8 for h in history), 'Training budget differs')
        require(all(h['epoch'] == i and all(np.isfinite(list(u.values())).all() for u in h['updates'])
                    for i, h in enumerate(history)), 'Training update/history is nonfinite or misordered')
        require(receipt['updates'] == 480 and receipt['epochs'] == 60 and receipt['completed'], 'Incomplete fit')
        v = np.array([h['validation_mean_layout_delta_rmse_db'] for h in history])
        require(np.isfinite(v).all() and receipt['selected_epoch'] == int(v.argmin()), 'First-minimum VAL checkpoint differs')
        require(receipt['validation_mean_layout_delta_rmse_db'] == float(v.min()), 'Selected VAL score differs')
        checkpoint = OUT / 'checkpoints' / f'{run}.pt'
        require(sha(checkpoint) == receipt['checkpoint_sha256'], 'Checkpoint bytes differ')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(all(torch.is_tensor(t) and torch.isfinite(t).all() for t in ck['state'].values()), 'Nonfinite learned tensors')
        for key in ('cell', 'seed', 'selected_epoch', 'settings', 'identity'):
            close(ck[key], receipt[key], f'{run}.checkpoint.{key}')
        scores_by_cell.setdefault(receipt['cell'], {})[receipt['seed']] = float(v.min())
        fit_summary[run] = {'selected_epoch': int(v.argmin()), 'validation_delta_rmse_db': float(v.min()), 'updates': 480}
    selected = min(('joint', 'regression'), key=lambda c: np.mean([scores_by_cell[c][s] for s in SEEDS]))
    selection = read(OUT / 'selection.json')
    require(selection['selected_cell'] == selected, 'VAL-only branch choice differs')
    for cell in ('joint', 'regression'):
        close(selection['candidates'][cell]['seed_delta_rmse_db'], [scores_by_cell[cell][s] for s in SEEDS], 'VAL seed scores')
    all_fingerprints = [layout['public_geometry_sha256'] for layouts in registry['stages'].values() for layout in layouts]
    require(len(all_fingerprints) == len(set(all_fingerprints)) == 58, 'Public CAD split duplication')
    public_fields = ('family', 'width_m', 'depth_m', 'ceiling_m', 'obstacle_height_m', 'ap_positions_m', 'rx_positions_m')
    def fingerprint(spec):
        value = {key: spec[key] for key in public_fields}
        return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()
    prior_fingerprints = set()
    for name in registry['prior_registry_sha256']:
        require(sha(ROOT / name) == registry['prior_registry_sha256'][name], 'Prior geometry registry changed')
        prior_registry = read(ROOT / name)
        prior_fingerprints.update(fingerprint(spec) for stage in prior_registry['stages'].values() for spec in stage)
    require(not (set(all_fingerprints) & prior_fingerprints) and len(prior_fingerprints) == 120,
            'Fresh CAD duplicates prior public geometry')
    require(all(fingerprint(layout['public_geometry']) == layout['public_geometry_sha256']
                for layouts in registry['stages'].values() for layout in layouts), 'Public CAD fingerprint differs')
    raw_summary = {}
    for stage in ('training', 'validation', 'development'):
        cases, raw_summary[stage] = raw_stage(stage, registry)
        if stage == 'training':
            calibration = frontend(cases)
        if stage == 'development':
            _, development_rows = prediction_stage(stage, cases)
    development = read(OUT / 'development.json'); replay = gate(development_rows, selected)
    close({key: development[key] for key in replay}, replay, 'Development gate')
    report = read(OUT / 'report.json'); contrast_summary = []
    if replay['test_permitted']:
        require((OUT / 'gates/development-final.json').exists(), 'Final full gate replay is absent')
        cases, raw_summary['test'] = raw_stage('test', registry)
        _, test_rows = prediction_stage('test', cases)
        sufficient = sum(r['delta_available'] for r in test_rows['sparse'][f'{selected}-seed0']) >= 12
        for spec, contrast in zip(protocol['primary_tests'], report['contrasts']):
            cell = selected if spec['candidate'] == 'selected' else spec['candidate']
            a = metric(test_rows['sparse'], cell, spec['metric'])
            b = metric(test_rows['sparse'], spec['reference'], spec['metric'])
            common = np.isfinite(a) & np.isfinite(b)
            statistics = bootstrap(a[common], b[common]) if common.sum() >= 2 else None
            close(contrast['statistics'], statistics, f'Contrast {spec["slot"]}')
            supported = bool(sufficient and statistics is not None and statistics['better_adjusted'])
            require(contrast['supported'] == supported, 'Claim support differs')
            contrast_summary.append({'slot': spec['slot'], 'candidate': cell, 'reference': spec['reference'],
                                     'metric': spec['metric'], 'statistics': statistics, 'supported': supported})
        require(report['RF_adaptation_supported'] == all(c['supported'] for c in contrast_summary[:2]), 'Adaptation claim differs')
        require(report['matched_alignment_benefit_supported'] == contrast_summary[2]['supported'], 'Alignment claim differs')
        require(report['detector_brier_benefit_supported'] == contrast_summary[3]['supported'], 'Detector claim differs')
    else:
        require(report['status'] == 'development_veto' and not (OUT / 'rt_data/test').exists(), 'Vetoed TEST opened')
        require(report['test_opened'] is False, 'Veto report opened TEST')
        require(not any(report[k] for k in ('RF_adaptation_supported', 'matched_alignment_benefit_supported',
                                           'detector_brier_benefit_supported')), 'Veto report asserts scientific benefit')
    require(report['production_qualified'] is False and report['original_service_confirmation_permitted'] is False,
            'Simulation results improperly qualified production/SERVICE')
    require({path for path in OUT.rglob('*') if path.is_file()} == before_paths, 'Evidence membership changed during review')
    HASHED.clear()
    require(all(sha(path) == digest for path, digest in before.items()), 'Evidence bytes changed during review')
    summary = {'schema': 1, 'status': 'pass', 'independent_passive_replay': True,
        'research_implementation_imported': False, 'RT_model_forward_fit_or_evaluation_run': False,
        'source_files_verified': len(source['source_sha256']),
        'prior_sealed_evidence_verified': len(old['evidence_sha256']), 'publication': publication,
        'campaign_files_preserved': len(before), 'campaign_membership_and_bytes_preserved': True,
        'review_script_sha256': sha(Path(__file__)), 'source_receipt_sha256': sha(OUT / 'registration/source.json'),
        'report_sha256': sha(OUT / 'report.json'), 'selected_cell': selected,
        'fits': fit_summary, 'raw_stages': raw_summary, 'development_gate': replay,
        'frontend': {'TRAIN_only_coobserved_pairs_verified': len(calibration['paired_reports']),
                     'estimated_offsets_db': {k: v['offset_db'] for k, v in calibration['estimates'].items()},
                     'absolute_offset_identified': False},
        'test_opened': replay['test_permitted'], 'contrasts': contrast_summary,
        'RF_adaptation_supported': report['RF_adaptation_supported'],
        'matched_alignment_benefit_supported': report['matched_alignment_benefit_supported'],
        'detector_brier_benefit_supported': report['detector_brier_benefit_supported'],
        'production_qualified': False, 'original_service_confirmation_permitted': False,
        'limits': ['Static isolated simulated channels; no packet/MAC/hardware qualification.',
                   'Receiver reports are deterministic quantization/sensitivity transforms; no random measurement noise is added.',
                   'Finite RT path search is not a convergence proof; unresolved paths are unknown and excluded.',
                   'Condition-delta point error is conditional on both-heard reports, with explicit censored/missing coverage.',
                   'Geometry bootstrap describes this finite prespecified simulation benchmark; it does not certify arbitrary deployments.',
                   'Checkpoint VAL scores are replayed from archived history, not rerun model inference.',
                   'Bootstrap conditions on three trained members; geometry is the independent unit.']}
    path = REVIEW / 'result.json'
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(encoded)
    print(json.dumps({'status': 'pass', 'result': str(path), 'selected_cell': selected,
                      'test_opened': replay['test_permitted'], 'raw_stages': raw_summary}, sort_keys=True))


if __name__ == '__main__':
    audit()
