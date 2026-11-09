"""Score unchanged real outputs for both strict schema and active semantics."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
from statistics import mean

from wirelessjev.warehouse_v2.experiment import atomic_json, sha256
from wirelessjev.warehouse_v2.missions import MissionSpec, parse_mission


def active_semantics(spec):
    fields = {'video_floor': ['floor'], 'move_budget': ['max_moves'],
              'zone': ['axis', 'boundary_m', 'side'], 'worst': [], 'safety': []}
    value = asdict(spec)
    return {key: value[key] for key in ['kind', *fields[spec.kind]]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    raw = args.run/'generative.json'
    report = json.loads(raw.read_text())
    comparisons = {}
    for label, key in [('qwen', 'readings'), ('d3', 'd3_readings_same_session')]:
        rows = []
        for reading in report[key]:
            expected = parse_mission(reading['text'], semantics='historical_e2')
            actual = MissionSpec(**reading['spec']) if 'spec' in reading else None
            rows.append({'text': reading['text'], 'answered': actual is not None,
                         'strict_match': actual == expected,
                         'active_semantics_match': actual is not None and active_semantics(actual) == active_semantics(expected)})
        comparisons[label] = {'n': len(rows), 'answered': sum(r['answered'] for r in rows),
                              'strict_schema_accuracy': mean(r['strict_match'] for r in rows),
                              'active_semantics_accuracy': mean(r['active_semantics_match'] for r in rows),
                              'latency_ms_mean': mean(r['latency_ms'] for r in report[key]), 'cases': rows}
    atomic_json(args.run/'generative_analysis.json', {'source_report_sha256': sha256(raw),
                'analysis_source_sha256': sha256(Path(__file__)), 'comparisons': comparisons,
                'method': 'Compare kind and only the fields used by that kind; no output rewriting or label-guided correction',
                'limits': ['Strict schema accuracy also penalises irrelevant populated fields',
                           'The typed reader extracts literal values deterministically, while Qwen generates them',
                           'One prompt and a small author-written set are not a general capability ranking']})
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
