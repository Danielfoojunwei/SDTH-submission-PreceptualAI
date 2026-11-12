#!/usr/bin/env python3
"""Render completed archived evidence, without model or simulator imports.

This presentation is deliberately separate from the scientific collector.
It refuses incomplete reports and reads only public CAD vertices, reported
query targets, archived predictions and completed aggregate reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STUDY = ROOT / 'artifacts/nested_radio'
INPUTS: dict[str, str] = {}
SEALED = None
CELLS = ('global8_reg','global8_joint','hier_reg','nested_reg','nested_joint')
NUMERICAL = ('radiomap-5m','radiomap-10m','radiomap-20m','hierarchy-fixed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Symlink inputs are not presentation artifacts')
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Regular contained repository artifact required')
    name = str(path.relative_to(ROOT)); digest = sha(path)
    if SEALED is not None and path.is_relative_to(STUDY) and path != STUDY/'verification/manifest.json':
        if SEALED.get(name) != digest:
            raise RuntimeError('Presentation input differs from completed scientific seal: '+name)
    INPUTS[name] = digest
    return path


def read(path):
    return json.loads(record(path).read_bytes())


def referenced(value):
    if set(value) != {'path', 'sha256'}:
        raise ValueError('Exact repository artifact reference required')
    p = Path(value['path'])
    if p.is_absolute() or '..' in p.parts or p.as_posix() != value['path']:
        raise ValueError('Canonical repository artifact reference required')
    p = record(ROOT / p)
    if sha(p) != value['sha256']:
        raise ValueError('Referenced artifact bytes changed: ' + value['path'])
    return p


def arrays(path):
    with np.load(record(path), allow_pickle=False) as a:
        return {key: a[key].copy() for key in a.files}


def cad_box(path):
    """Read actual public ASCII PLY vertex bounds, never XML materials."""
    lines = record(path).read_text().splitlines()
    if lines[:2] != ['ply', 'format ascii 1.0']:
        raise ValueError('Expected archived ASCII public CAD')
    count = int(next(line.split()[-1] for line in lines if line.startswith('element vertex ')))
    start = lines.index('end_header') + 1
    vertices = np.array([[float(v) for v in line.split()[:3]] for line in lines[start:start+count]])
    if vertices.shape != (count, 3) or not np.isfinite(vertices).all():
        raise ValueError('Malformed public CAD vertex positions')
    return [vertices.min(0).tolist(), vertices.max(0).tolist()]


def build():
    global SEALED
    # Guard before any case read or output write; no partial/scaffolding data.
    required = [STUDY/'report.json', STUDY/'verification/manifest.json']
    if not all(path.is_file() for path in required):
        raise RuntimeError('Completed report.json AND verification/manifest.json must precede presentation')
    scientific_seal = read(required[1])
    SEALED = {**scientific_seal['source_sha256'], **scientific_seal['evidence_sha256']}
    field = read(required[0])
    if (field.get('complete') is not True or field.get('primary_layouts') != 24
            or field.get('members_each') != 3 or field.get('fits') != 15
            or field.get('fresh_RT_cases') != 84 or field.get('reused_TRAIN_VAL_RT_cases') != 90
            or field.get('native_simulations') != 0 or field.get('production_qualified') is not False):
        raise RuntimeError('Fixed completed RF-only experiment membership required')
    selection = read(STUDY/'selection.json')
    if (sha(STUDY/'selection.json') != field['selection_sha256']
            or selection['selected_cell'] != field['selected_cell']
            or selection['selected_numerical'] != field['best_numerical']):
        raise RuntimeError('Immutable validation-only selection changed')
    selected = field['selected_cell']+'-seed0'
    best_numeric = field['best_numerical']
    fits = []
    for cell in CELLS:
        for seed in range(3):
            member = f'{cell}-seed{seed}'
            row = read(STUDY/'fits'/f'{member}.json')
            checks = [r for r in row['history'] if r['epoch'] in (40,80,120,160)]
            if (row.get('complete') is not True or row.get('committed_updates') != 1920
                    or row['cell'] != cell or row['seed'] != seed or len(checks) != 4
                    or row['selected_epoch'] != min(checks,key=lambda r:r['validation_rmse'])['epoch']):
                raise RuntimeError('Complete fixed-budget fit and first RMSE minimum required')
            fits.append({'member':member,'cell':cell,'seed':seed,'selected_epoch':row['selected_epoch'],
                'validation_rmse':row['validation_rmse'],'validation_nll':row['validation_nll'],
                'allocated_parameters':row['learned_parameters'],'committed_updates':row['committed_updates'],
                'checks':[{'epoch':r['epoch'],'rmse':r['validation_rmse'],'nll':r['validation_nll']} for r in checks]})
    for cell in CELLS:
        mean=float(np.mean([r['validation_rmse'] for r in fits if r['cell']==cell]))
        if abs(mean-selection['mean_VAL_RMSE'][cell])>1e-12:
            raise RuntimeError('Selection does not average all three fixed fit minima')
    eligible=('global8_reg','global8_joint','nested_reg','nested_joint')
    if selection['selected_cell'] != min(eligible,key=lambda cell:selection['mean_VAL_RMSE'][cell]):
        raise RuntimeError('Selected model does not follow predeclared validation/tie rule')
    numeric = read(referenced(selection['numerical_selection']))
    if (numeric['selected_name'] != best_numeric or set(numeric['scores'])!=set(NUMERICAL)
            or choose_first(NUMERICAL,numeric['scores'])!=best_numeric):
        raise RuntimeError('Best numerical reference does not follow the same VAL RMSE rule')
    representation=read(referenced(field['validation_representation']))
    expected_members=[f'{cell}-seed{seed}' for cell in CELLS for seed in range(3)]
    if [r['member'] for r in representation['rows']]!=expected_members or representation['used_for_selection'] is not False:
        raise RuntimeError('All fifteen predeclared validation diagnostics required')
    if field['selected_cell']=='global8_reg':
        alias=field['primary_contrasts'][1]
        if field['summary']['sparse']['selected']!=field['summary']['sparse']['global8_reg'] or alias['mean_delta']!=0 or alias['central_98_75_interval']!=[0,0]:
            raise RuntimeError('Selected baseline must remain an exact alias/null, not a residual gain')
    registry = read(STUDY / 'registry.json')
    geometries = registry['stages']['test']
    if len(geometries) != 24:
        raise RuntimeError('All 24 TEST geometries required')
    manifest = read(STUDY / 'predictions/test-sparse.json')
    group = read(referenced(manifest['example_group']))
    values = arrays(referenced(group['arrays']))
    bindings = group['bindings']
    if (len(bindings) != 72 or len(manifest['case_receipts']) != 72
            or group['stage'] != 'test' or group['regime'] != 'sparse'
            or group['held_band'] or group['stale']):
        raise RuntimeError('Complete sparse current-state source-band TEST presentation required')
    cases = []
    scene_data = []
    for layout_index, item in enumerate(geometries):
        public = item['public_geometry']
        if item['APs'] != 4:
            raise RuntimeError('Four-AP main experiment required')
        source_binding = bindings[layout_index * 3]
        meshes = []
        for name, expected in source_binding['source_files_sha256'].items():
            p = Path(name)
            if p.suffix == '.ply' and p.stem.startswith(('rack', 'partition')):
                p = record(ROOT / p)
                if sha(p) != expected:
                    raise RuntimeError('Public CAD identity changed')
                meshes.append({'name': p.stem, 'bounds': cad_box(p)})
        if len(meshes) != 2:
            raise RuntimeError('Both real public obstacle meshes required')
        meshes.sort(key=lambda m: m['name'])
        scene_data.append({'id': item['layout_id'], 'geometry': public, 'obstacles': meshes})
        for state in range(3):
            i = layout_index * 3 + state
            binding = bindings[i]
            case_id = f"{item['layout_id']}-state{state}"
            if binding['case_id'] != case_id or binding['layout_id'] != item['layout_id']:
                raise RuntimeError('Exact layout/state order required')
            receipt = read(referenced(manifest['case_receipts'][case_id]))
            if receipt['case_id'] != case_id or receipt['example_index'] != i:
                raise RuntimeError('Prediction example binding changed')
            predicted = arrays(referenced(receipt['predictions']))
            indices = [receipt['methods'].index(m) for m in (selected, best_numeric)]
            if (receipt['methods'] != [f'{cell}-seed{s}' for cell in CELLS for s in range(3)]+list(NUMERICAL)
                    or predicted['mean'].shape != (19,192) or predicted['std'].shape != (19,192)):
                raise RuntimeError('Archived main query shape changed')
            ids = binding['query_measurement_ids']
            tokens = [re.fullmatch(r'.+:pose(\d+):ap(\d+):band(\d+):dir0:dev0', token) for token in ids]
            if not all(tokens):
                raise RuntimeError('Reported query IDs required')
            pose, ap, band = np.array([[int(v) for v in token.groups()] for token in tokens]).T
            if (not np.array_equal(ap,values['query__ap'][i]) or len(pose)!=192
                    or set(band)!={0,1} or np.any(pose<0) or np.any(pose>=48)
                    or not np.allclose(values['query__rx_xyz'][i],np.asarray(public['rx_positions_m'])[pose],rtol=0,atol=2e-5)):
                raise RuntimeError('Physical query identities or receiver coordinates changed')
            target = {key[len('target__'):]: value[i] for key, value in values.items() if key.startswith('target__')}
            if (not np.isfinite(predicted['mean'][indices]).all() or not np.isfinite(predicted['std'][indices]).all()
                    or np.any(predicted['std'][indices]<=0)):
                raise RuntimeError('Finite archived forecasts required')
            context = np.unique(values['a__rx_xyz'][i], axis=0)
            cases.append({'id': case_id, 'layout': layout_index, 'state': state,
                'pose': pose.tolist(), 'ap': ap.tolist(), 'band': band.tolist(),
                'positions': values['query__rx_xyz'][i].tolist(), 'context_positions': context.tolist(),
                'heard': target['heard'].tolist(), 'censored': target['censored'].tolist(),
                'reported_center': target['center_dbm'].tolist(), 'reported_lower': target['lower_dbm'].tolist(),
                'reported_upper': target['upper_dbm'].tolist(), 'censor_upper': target['censor_upper_dbm'].tolist(),
                'hier_mean': predicted['mean'][indices[0]].tolist(), 'hier_std': predicted['std'][indices[0]].tolist(),
                'map_mean': predicted['mean'][indices[1]].tolist(), 'map_std': predicted['std'][indices[1]].tolist()})
    selected_fit=next(row for row in fits if row['member']==selected)
    return {'schema':1,'selected_member':selected,'best_numerical_name':best_numeric,
        'learned_parameters':selected_fit['allocated_parameters'],'field':field,'selection':selection,
        'numerical':numeric,'fits':fits,'representation':representation,'control_order':list(CELLS)+list(NUMERICAL),
        'scenes':scene_data,'cases':cases,
        'sealed_source_files':len(scientific_seal['source_sha256']),'sealed_evidence_files':len(scientific_seal['evidence_sha256']),
        'scope':'Presentation of completed archived RF evidence; no forward, fit, ray solve or simulation',
        'single_case_scope':'Sparse current-state source bands, validation-selected model seed0 versus validation-selected numerical reference; descriptive, not the primary estimator',
        'native_scope':'RF-only study: zero new warehouse simulations or hardware actions; no service qualification'}


def choose_first(order,scores):
    return min(order,key=lambda key:scores[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Replace only presentation outputs, retaining the same report/input identities')
    args = parser.parse_args()
    data = build()
    template = record(HERE / 'template.html').read_text()
    text = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    html = template.replace('__DATA_JSON__', text)
    if '__DATA_JSON__' in html:
        raise RuntimeError('Unresolved presentation template')
    outputs = {'index.html': html.encode(), 'data.json': (text+'\n').encode()}
    if not args.overwrite and any((HERE / name).exists() for name in (*outputs, 'provenance.json')):
        raise FileExistsError('Presentation already exists; explicit --overwrite required')
    if args.overwrite and (HERE / 'provenance.json').exists():
        old = json.loads((HERE / 'provenance.json').read_bytes())
        report_names = ['artifacts/nested_radio/report.json','artifacts/nested_radio/verification/manifest.json']
        if any(old['inputs_sha256'].get(name) != INPUTS.get(name) for name in report_names):
            raise RuntimeError('Cannot replace a presentation with different scientific report identities')
    for name, content in outputs.items():
        (HERE / name).write_bytes(content)
    provenance = {'schema': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator_sha256': sha(__file__), 'inputs_sha256': dict(sorted(INPUTS.items())),
        'output_sha256': {name: sha(HERE / name) for name in outputs},
        'primary_geometries': 24, 'members_in_aggregate': 3, 'illustrated_member': data['selected_member'],
        'illustrated_cases': 72, 'fresh_RT_cases':84,'reused_TRAIN_VAL_RT_cases':90,'native_simulations':0,
        'scientific_calls': 0, 'external_assets': 0,
        'scope': data['scope'], 'production_qualified': False}
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'complete': True, 'illustrated_cases': 72, 'files': list(outputs), 'inputs': len(INPUTS)}))


if __name__ == '__main__':
    main()
