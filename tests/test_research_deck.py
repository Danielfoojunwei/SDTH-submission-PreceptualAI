"""The presentation must match the current system and its archived evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/slides/generated'


def module():
    spec=importlib.util.spec_from_file_location('research_deck_builder',ROOT/'scripts/build_research_deck.py')
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def test_deck_is_current_system_with_a_visual_and_sources_for_every_slide():
    source=json.loads((ROOT/'docs/slides/deck.json').read_text());slides=source['slides']
    assert len(slides)==32 and len({s['id'] for s in slides})==32
    for s in slides:
        assert s['sources'] and s['notes'] and s['insight'] and s['evidence']
        visible=' '.join(s[k] for k in ['title','subtitle','insight','notes'])
        assert not re.search(r'development history|\blineage\b|previous versions?|\bE[127]\b|\bv[234]\b',visible,re.I),s['id']
        assert all((ROOT/p).exists() for p in s['sources'])
    assert {'architecture','design_rationale','allocation','traffic_control','operator_view','execution_safety'} <= {s['id'] for s in slides}


def test_generated_formats_have_unique_visuals_editable_notes_and_current_hashes():
    source=json.loads((ROOT/'docs/slides/deck.json').read_text());builder=module()
    checks=builder.validate(source,OUT)
    assert len(checks)==32 and all(r['substantive_visual'] and r['pdf_title_verified'] for r in checks)
    manifest=json.loads((OUT/'manifest.json').read_text())
    for relative,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    for relative,digest in manifest['outputs_sha256'].items():
        assert hashlib.sha256((OUT/relative).read_bytes()).hexdigest()==digest,relative


def test_headline_chart_uses_actual_paired_campaign_counts():
    rows=json.loads((ROOT/'artifacts/native/wifi_vs_5g.json').read_text())['rows']
    def values(arm):return [r for r in rows if r['condition']=='main' and r['arm']==arm]
    dual=values('vertical_client');cell=values('nr_30ms')
    dual_ppm=sum(r['control_misses'] for r in dual)/sum(r['control_sent'] for r in dual)*1e6
    cell_ppm=sum(r['control_misses'] for r in cell)/sum(r['control_sent'] for r in cell)*1e6
    facts=json.loads((OUT/'figure_facts.json').read_text())['main_results']
    assert abs(facts['dual_ppm']-dual_ppm)<1e-9
    assert abs(facts['cellular_30ms_ppm']-cell_ppm)<1e-9
    assert facts['commands_per_arm_condition']==sum(r['control_sent'] for r in dual)==3_600_000
    assert facts['independent_scene_realisations']==6
    assert facts['packet_iid_intervals_used'] is False
