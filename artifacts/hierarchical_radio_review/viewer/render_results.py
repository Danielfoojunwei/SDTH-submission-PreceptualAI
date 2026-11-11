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
STUDY = ROOT / 'artifacts/hierarchical_radio'
INPUTS: dict[str, str] = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Symlink inputs are not presentation artifacts')
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Regular contained repository artifact required')
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
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
    # Check existence before touching any case data or writing final outputs.
    required = [STUDY / 'field_report.json', STUDY / 'native/report.json']
    if not all(path.is_file() for path in required):
        raise RuntimeError('Complete field_report.json and native/report.json must precede presentation')
    field, native = map(read, required)
    if (field.get('complete') is not True or native.get('complete') is not True
            or field.get('primary_layouts') != 24 or field.get('members') != 3
            or field.get('actual_neural_fits') != 15 or field.get('actual_RT_cases') != 186
            or native.get('fresh_simulations') != 32 or native.get('parity_simulations') != 2):
        raise RuntimeError('Fixed completed experiment membership required')
    if referenced(native['binding']['field_report']) != required[0]:
        raise RuntimeError('Native study does not bind this field report')
    selection = read(STUDY / 'selection.json')
    if (sha(STUDY / 'selection.json') != field['selection_sha256']
            or selection['selected_hierarchy'] != field['selected_hierarchy']):
        raise RuntimeError('Immutable VAL selection changed')
    selected = field['selected_hierarchy'] + '-seed0'
    fit = read(STUDY / 'fits' / (selected + '.json'))
    if fit.get('complete') is not True or fit.get('committed_updates') != 480:
        raise RuntimeError('Completed selected member identity required')
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
            indices = [receipt['methods'].index(m) for m in (selected, 'radiomap_operator')]
            if predicted['mean'].shape != predicted['std'].shape or predicted['mean'].shape[1] != 192:
                raise RuntimeError('Archived main query shape changed')
            ids = binding['query_measurement_ids']
            tokens = [re.fullmatch(r'.+:pose(\d+):ap(\d+):band(\d+):dir0:dev0', token) for token in ids]
            if not all(tokens):
                raise RuntimeError('Reported query IDs required')
            pose, ap, band = np.array([[int(v) for v in token.groups()] for token in tokens]).T
            if not np.array_equal(ap, values['query__ap'][i]) or len(pose) != 192:
                raise RuntimeError('Physical query identities changed')
            target = {key[len('target__'):]: value[i] for key, value in values.items() if key.startswith('target__')}
            if not np.isfinite(predicted['mean'][indices]).all() or not np.isfinite(predicted['std'][indices]).all():
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
    return {'schema': 1, 'selected_member': selected, 'learned_parameters': fit['learned_parameters'], 'field': field, 'native': native,
        'scenes': scene_data, 'cases': cases,
        'scope': 'Presentation of completed archived evidence; no forward, fit, ray solve or simulation',
        'single_case_scope': 'Sparse current-state source bands, VAL-selected hierarchy seed0; descriptive, not the primary estimator',
        'native_scope': 'Four procedural episodes per handover, 100 simulated robots, 20 seconds; matched rules/budget, realized histories may differ'}


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
        report_names = ['artifacts/hierarchical_radio/field_report.json', 'artifacts/hierarchical_radio/native/report.json']
        if any(old['inputs_sha256'].get(name) != INPUTS.get(name) for name in report_names):
            raise RuntimeError('Cannot replace a presentation with different scientific report identities')
    for name, content in outputs.items():
        (HERE / name).write_bytes(content)
    provenance = {'schema': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'generator_sha256': sha(__file__), 'inputs_sha256': dict(sorted(INPUTS.items())),
        'output_sha256': {name: sha(HERE / name) for name in outputs},
        'primary_geometries': 24, 'members_in_aggregate': 3, 'illustrated_member': data['selected_member'],
        'illustrated_cases': 72, 'native_fresh_simulations': 32, 'native_parity_simulations': 2,
        'scientific_calls': 0, 'external_assets': 0,
        'scope': data['scope'], 'production_qualified': False}
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'complete': True, 'illustrated_cases': 72, 'files': list(outputs), 'inputs': len(INPUTS)}))


if __name__ == '__main__':
    main()
