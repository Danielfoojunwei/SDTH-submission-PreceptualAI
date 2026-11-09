"""Independent read-only TUJI1 result review, outside the frozen study.

Uses the licensed source CSVs and saved arrays directly. It creates no fit,
new experiment prediction stream, support sample or modified study evidence.
Scalar inference spot checks and all error/calibration arithmetic are separate
implementations from the registered benchmark. The canonical audit additionally
recomputes every stored classical prediction using its frozen implementation.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'artifacts/device_transfer'
OUT = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b, name, atol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=0,atol=atol):
        raise RuntimeError('Independent review failed: '+name)


def arrays(reference):
    path = ROOT/reference['path']
    if sha(path) != reference['sha256']:
        raise RuntimeError('Changed saved array archive')
    with np.load(path, allow_pickle=False) as saved:
        return {name:saved[name].copy() for name in saved.files}


def scalar_knn(query, query_heard, reference, reference_heard, xyz, neighbors, minimum_shared):
    common = reference_heard & query_heard
    count = common.sum(1)
    difference = np.where(common, reference-query, 0.)
    squared = (difference*difference).sum(1)/np.maximum(count,1)
    distance = np.sqrt(squared)
    distance[count < minimum_shared] = np.inf
    nearest = np.argsort(distance, kind='stable')[:neighbors]
    valid = np.isfinite(distance[nearest])
    if not valid.any():
        return xyz.mean(0)
    chosen = nearest[valid]
    weights = 1./(distance[chosen]+1.)
    return (xyz[chosen]*weights[:,None]).sum(0)/weights.sum()


def raw_metrics(predicted, truth, supported):
    xy_error = np.sqrt(np.square(predicted[:,:2]-truth[:,:2]).sum(1))
    spatial_groups = defaultdict(list)
    for point,error in zip(truth,xy_error):
        spatial_groups[tuple(point)].append(float(error))
    return {'mean_2d_error_m':float(np.mean(xy_error)),
        'median_2d_error_m':float(np.median(xy_error)),
        'p95_2d_error_m':float(np.percentile(xy_error,95)),
        'location_balanced_mean_2d_error_m':float(np.mean([np.mean(x) for x in spatial_groups.values()])),
        'mean_3d_error_m':float(np.sqrt(np.square(predicted-truth).sum(1)).mean()),
        'shared_ap_fallback_fraction':float(np.count_nonzero(~supported)/len(supported))}


def run():
    protocol = read(STUDY/'protocol.json')
    report = read(STUDY/'report.json')
    manifest_path = STUDY/'verification/manifest.json'
    manifest = read(manifest_path)
    before = {name:sha(ROOT/name) for name in {**manifest['source_sha256'],**manifest['evidence_sha256']}}
    if before != {**manifest['source_sha256'],**manifest['evidence_sha256']}:
        raise RuntimeError('Frozen source/evidence bytes differ before review')
    manifest_sha = sha(manifest_path)
    directory = os.environ.get('DEVICE_TRANSFER_DATA_ROOT','/workspace/upstream/scene-radio/rssi/downloads')
    source = Path(directory)/'7641701-DATASET.zip'
    if sha(source) != protocol['archive_sha256']:
        raise RuntimeError('Actual source archive differs')
    with zipfile.ZipFile(source) as archive:
        def matrix(name):
            with archive.open('DATASET/'+name) as stream:
                return np.loadtxt(stream, delimiter=',')
        train_rssi = matrix('RSS_training.csv')
        train_labels = matrix('Coordinates_training.csv')
        test_rssi = matrix('RSS_testing.csv')
        test_labels = matrix('Coordinates_testing.csv')
    train_heard, test_heard = train_rssi != 100, test_rssi != 100
    train_xyz, test_xyz = train_labels[:,:3], test_labels[:,:3]
    train_device, test_device = train_labels[:,5].astype(int), test_labels[:,5].astype(int)
    if set(map(tuple,train_xyz)) & set(map(tuple,test_xyz)):
        raise RuntimeError('TRAIN/TEST reference positions overlap')
    if train_rssi.shape != (6752,310) or test_rssi.shape != (2147,310):
        raise RuntimeError('Actual source corpus shape differs')
    inventory = read(STUDY/'inventory.json')
    prelabel = read(STUDY/'before-labels.json')
    by_device = {row['device']:row for row in prelabel['predictions']}
    cases = {(row['device'],row['model'],row['support_size'],row['support_seed']):row for row in report['cases']}
    wanted = {(d,m,n,s) for d in protocol['target_devices'] for m in protocol['models']
              for n in protocol['support_sizes'] for s in protocol['support_seeds']}
    if len(cases) != 300 or set(cases) != wanted or len(report['cases']) != 300:
        raise RuntimeError('Not the exact registered300-case matrix')
    independently_computed, fit_checks = [], []
    scalar_checks = calibration_checks = 0
    for device in protocol['target_devices']:
        state_receipt = read(STUDY/'fits'/f'device-{device}.json')
        state = arrays(state_receipt['state'])
        saved = arrays(by_device[device]['predictions'])
        source_rows = np.flatnonzero(train_device != device)
        target_rows = np.flatnonzero(test_device == device)
        if not np.array_equal(state['source_rows'],source_rows) or not np.array_equal(saved['test_rows'],target_rows):
            raise RuntimeError('Source/test phone membership differs')
        close(state['source_rssi_dbm'],np.where(train_heard[source_rows],train_rssi[source_rows],0.),'real source fingerprint values',atol=0)
        close(state['source_heard'],train_heard[source_rows],'real source masks',atol=0)
        close(state['source_xyz_m'],train_xyz[source_rows],'real source reference coordinates',atol=0)
        colocated = defaultdict(list)
        for row in source_rows:
            colocated[tuple(train_xyz[row])].append(int(row))
        for si,seed in enumerate(protocol['support_seeds']):
            order = inventory['support_orders'][str(device)][str(seed)]
            if not np.array_equal(state['support_order'][si],order):
                raise RuntimeError('Published support order changed')
            for ni,n in enumerate(protocol['support_sizes']):
                chosen = order[:n]
                if (len({tuple(train_xyz[row]) for row in chosen}) != n
                        or any(train_device[row] != device for row in chosen)):
                    raise RuntimeError('Support is duplicated or crosses devices')
                residuals = defaultdict(list)
                for row in chosen:
                    count = 0
                    for ap in np.flatnonzero(train_heard[row]):
                        values = [train_rssi[ref,ap] for ref in colocated[tuple(train_xyz[row])] if train_heard[ref,ap]]
                        if values:
                            residuals[int(ap)].append(float(train_rssi[row,ap]-np.median(values)))
                            count += 1
                    if count < 3:
                        raise RuntimeError('Support lacks three observed co-location APs')
                flat = [value for values in residuals.values() for value in values]
                global_bias = float(np.median(flat)) if flat else 0.
                per_ap = np.full(310,global_bias)
                counts = np.zeros(310,int)
                for ap,values in residuals.items():
                    counts[ap] = len(values)
                    per_ap[ap] = (len(values)*np.median(values)+protocol['ap_shrinkage_pseudocount']*global_bias)/(len(values)+protocol['ap_shrinkage_pseudocount'])
                close(state['global_offsets_db'][si,ni],global_bias,'independent global calibration')
                close(state['ap_offsets_db'][si,ni],per_ap,'independent AP shrinkage')
                close(state['ap_pair_counts'][si,ni],counts,'independent paired counts',atol=0)
                calibration_checks += 1
        # Independent scalar/broadcast distance implementation, with three fixed
        # actual query rows per device and every distinct saved prediction case.
        for key,prediction in saved.items():
            if not key.endswith('xyz'):
                continue
            if key == 'ridge_xyz':
                heard = train_heard[source_rows]
                counts = heard.sum(0)
                mean = np.divide(np.where(heard,train_rssi[source_rows],0.).sum(0),np.maximum(counts,1))
                scale = np.maximum(np.sqrt(np.where(heard,train_rssi[source_rows]-mean,0.).var(0)),1.)
                # The ridge variance divides observed squared residuals by the
                # observed count, rather than all rows; recompute explicitly.
                scale = np.maximum(np.sqrt(np.square(np.where(heard,train_rssi[source_rows]-mean,0.)).sum(0)/np.maximum(counts,1)),1.)
                close(state['ridge_mean'],mean,'independent ridge train-only mean')
                close(state['ridge_scale'],scale,'independent ridge train-only std')
                features = np.concatenate((np.where(heard,(train_rssi[source_rows]-mean)/scale,0.),heard),1)
                feature_mean = features.mean(0)
                centered = features-feature_mean
                lhs = (centered.T@centered+protocol['ridge_alpha']*np.eye(features.shape[1]))@state['ridge_coefficient']
                rhs = centered.T@(train_xyz[source_rows]-train_xyz[source_rows].mean(0))
                close(lhs,rhs,'independent ridge normal equations',atol=1e-8*max(1.,float(np.abs(rhs).max())))
                continue
            reference_rows = source_rows
            offset = np.zeros(310)
            if key != 'base_xyz':
                name,seed_token,n_token,_ = key.split('_')
                seed, n = int(seed_token[1:]), int(n_token[1:])
                si,ni = protocol['support_seeds'].index(seed), protocol['support_sizes'].index(n)
                if name == 'support':
                    reference_rows = np.r_[source_rows,state['support_order'][si,:n]]
                if name == 'global': offset[:] = state['global_offsets_db'][si,ni]
                if name == 'ap': offset = state['ap_offsets_db'][si,ni]
            for local_row in sorted({0,len(target_rows)//2,len(target_rows)-1}):
                row = target_rows[local_row]
                independent = scalar_knn(test_rssi[row]-offset,test_heard[row],
                    train_rssi[reference_rows],train_heard[reference_rows],train_xyz[reference_rows],
                    protocol['neighbors'],protocol['minimum_shared_observed_aps'])
                close(independent,prediction[local_row],'scalar nearest-neighbor inference spot',atol=1e-7)
                scalar_checks += 1
        fit_checks.append({'device':device,'source_rows':len(source_rows),'test_rows':len(target_rows),
                           'target_fitting_rows_excluded_from_source':True})
        for model in protocol['models']:
            for n in protocol['support_sizes']:
                for seed in protocol['support_seeds']:
                    row = cases[device,model,n,seed]
                    if model == 'source_ridge':
                        pred_key,support_key = 'ridge_xyz',None
                    elif model == 'source_knn' or n == 0:
                        pred_key,support_key = 'base_xyz','base_supported'
                    else:
                        prefix = {'source_knn_with_target_support':'support','global_offset_knn':'global','ap_shrinkage_knn':'ap'}[model]
                        pred_key,support_key = f'{prefix}_s{seed}_n{n}_xyz',f'{prefix}_s{seed}_n{n}_supported'
                    supported = np.ones(len(target_rows),bool) if support_key is None else saved[support_key]
                    metrics = raw_metrics(saved[pred_key],test_xyz[target_rows],supported)
                    for field,value in metrics.items():
                        close(row['metrics'][field],value,'actual XYZ error '+field)
                    independently_computed.append({'device':device,'model':model,'n':n,'seed':seed,'metrics':metrics})
    summaries = []
    for n in protocol['support_sizes']:
        for model in protocol['models']:
            rows = [row for row in independently_computed if row['n']==n and row['model']==model]
            expected = next(row for row in report['summaries'] if row['support_size']==n and row['model']==model)
            for metric in protocol['metrics']:
                close(expected['balanced_phone_'+metric],np.mean([row['metrics'][metric] for row in rows]),'balanced phone aggregation '+metric)
            seed_means = [np.mean([row['metrics']['mean_2d_error_m'] for row in rows if row['seed']==seed]) for seed in protocol['support_seeds']]
            close(expected['support_draw_std_balanced_mean_m'],np.std(seed_means,ddof=1),'support-draw variation')
            summaries.append(expected)
    from wirelessjev.device_transfer import benchmark
    canonical_audit = benchmark.audit(source,write=False)
    if canonical_audit['passes'] is not True:
        raise RuntimeError('Canonical passive audit failed')
    closure = benchmark.verify_closure()
    if before != {name:sha(ROOT/name) for name in before} or sha(manifest_path) != manifest_sha:
        raise RuntimeError('Frozen study changed during independent review')
    result = {'schema':1,'passes':True,'reviewer_scope':'independent real-data arithmetic plus canonical full passive audit',
        'source_archive_sha256':sha(source),'study_manifest_sha256':manifest_sha,
        'review_script_sha256':sha(__file__),'reviewed_source_files':len(manifest['source_sha256']),
        'reviewed_evidence_files':len(manifest['evidence_sha256']),
        'independent_actual_xyz_case_checks':len(independently_computed),
        'independent_metric_summary_checks':len(summaries),'independent_calibration_checks':calibration_checks,
        'independent_scalar_inference_spots':scalar_checks,'fold_membership':fit_checks,
        'independent_summaries':summaries,'canonical_audit_pass':True,'closure':closure,
        'frozen_study_unchanged':True,'new_fits':0,'new_synthetic_observations':0,
        'new_experiment_prediction_streams':0,'classical_scalar_inference_spots_recomputed':True,
        'radiojepa_encoder_transfer_claimed':False,'physical_sites':1,
        'note':'Support draws are not independent neural training runs; device/day/body confounding remains.',
        'completed_utc':datetime.now(timezone.utc).isoformat()}
    destination = OUT/'review.json'
    if destination.exists():
        raise RuntimeError('Refuse to overwrite an existing independent review')
    destination.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('passes','independent_actual_xyz_case_checks',
        'independent_metric_summary_checks','independent_calibration_checks','independent_scalar_inference_spots',
        'frozen_study_unchanged')}))


if __name__ == '__main__':
    run()
