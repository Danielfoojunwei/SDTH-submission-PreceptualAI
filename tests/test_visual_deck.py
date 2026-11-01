"""Check that the visual deck is concise, real 3D, and evidence-grounded."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/slides/concise';OUT=BASE/'generated'


def test_ten_slide_exports_have_real_models_editable_copy_and_current_sources():
    source=json.loads((BASE/'deck.json').read_text())
    spec=importlib.util.spec_from_file_location('visual_deck',ROOT/'scripts/build_visual_deck.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    assert len(builder.validate(source,OUT))==10
    for s in source['slides']:
        assert not re.search(r'development history|\blineage\b|previous versions?',s['title']+' '+s['body'],re.I)
        assert len(s['body'].split())<=18
        assert all((ROOT/p).exists() for p in s['sources'])
    manifest=json.loads((OUT/'manifest.json').read_text())
    for relative,digest in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    for relative,digest in manifest['outputs_sha256'].items():assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==digest,relative
    assert manifest['models_are_schematic'] and not manifest['external_claude_deck_republished']
    for name in ['warehouse','architecture','diversity']:
        payload=(OUT/'models'/f'{name}.glb').read_bytes()
        assert payload[:4]==b'glTF' and struct.unpack_from('<II',payload,4)==(2,len(payload))
        length=struct.unpack_from('<I',payload,12)[0];model=json.loads(payload[20:20+length])
        assert len(model['meshes'])>=15 and len(model['nodes'])>=30
        assert (OUT/'models'/f'{name}.blend').read_bytes().startswith(b'BLENDER')


def test_cinematic_scene_data_uses_archived_counts_and_an_actual_map_grid():
    data=json.loads((BASE/'assets/scene_data.json').read_text())
    rows=json.loads((ROOT/'artifacts/native/wifi_vs_5g.json').read_text())['rows']
    for condition,arms in data['results'].items():
        for arm,facts in arms.items():
            selected=[r for r in rows if r['condition']==condition and r['arm']==arm]
            assert len(selected)==6 and facts['commands']==sum(r['control_sent'] for r in selected)==3_600_000
            assert facts['misses']==sum(r['control_misses'] for r in selected)
            assert abs(facts['ppm']-facts['misses']/facts['commands']*1e6)<1e-12
    main=data['results']['main'];assert 17<data['ratio']==main['nr_30ms']['ppm']/main['vertical_client']['ppm']<18
    stress=data['results']['stress'];assert stress['vertical_client']['ppm']>stress['nr_30ms']['ppm']
    grid=data['map']['mean_dbm'];assert len(grid)==33 and all(len(row)==49 for row in grid)
    assert max(v for row in grid for v in row)-min(v for row in grid for v in row)>10
    assert data['conceptual_model'] and not data['field_measurement']
