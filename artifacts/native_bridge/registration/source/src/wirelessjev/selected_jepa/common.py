"""Independent confirmation identity; the rejected source study is preserved."""
import json
from pathlib import Path
import shutil
import time
from fractions import Fraction
from ..semantic_jepa import common as previous
from ..semantic_jepa.common import digest, encoded, locked, value_digest, write_once, write_text_once

ROOT = previous.ROOT
OUT = ROOT / 'artifacts/selected_jepa'
PROTOCOL = OUT / 'protocol.json'


def protocol():
    return json.loads(PROTOCOL.read_text())


def selection():
    return json.loads((OUT / 'selection.json').read_text())


def source_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ROOT / 'tests').glob('test_selected_jepa*.py'))
    paths += [PROTOCOL, OUT / 'selection.json']
    old = previous.OUT
    paths += [old / name for name in ('registration/source.json', 'protocol.json', 'development.json',
                                      'training.json', 'latency.json')]
    paths += sorted((old / 'fits').glob('*.json'))
    paths += sorted((old / 'checkpoints').glob('*.pt'))
    paths += sorted((old / 'cases').glob('development-*.receipt.json'))
    return previous.source_hashes() | {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def verify_selection():
    """Verify observed development; never construct a fresh test world."""
    from ..semantic_jepa import study
    previous.check_sources()
    dev = json.loads((previous.OUT / 'development.json').read_text())
    rows = [study.load_case(key) for key in study.case_keys('development')]
    if any(row is None for row in rows):
        raise RuntimeError('Prior development archive incomplete')
    study.validate_development_seal(dev, rows)
    if dev['passed_costs']:
        raise RuntimeError('Preserve the recorded source-study semantic veto')
    record = selection()
    for name, expected in record['source_receipt_sha256'].items():
        if digest(previous.OUT / name) != expected:
            raise RuntimeError('Prior selection evidence changed: ' + name)
    scores = []
    for family in ('context_jepa', 'semantic_jepa'):
        ratios = []
        for cost in protocol()['handover_seconds']:
            candidate = [r for r in rows if r['family'] == family and r['handover_seconds'] == cost]
            reference = [r for r in rows if r['family'] == 'radiomap' and r['handover_seconds'] == cost]
            rate = Fraction(sum(r['result']['control_misses'] for r in candidate),
                            sum(r['result']['control_sent'] for r in candidate))
            ref = Fraction(sum(r['result']['control_misses'] for r in reference),
                           sum(r['result']['control_sent'] for r in reference))
            ratios.append(rate / ref)
        scores.append((sum(ratios) / len(ratios), family))
    chosen = min(scores)[1]
    if chosen != record['selected_family'] or chosen != protocol()['candidate_family']:
        raise RuntimeError('Selected family differs from frozen development rule')
    training = study.training_receipt()
    latency = json.loads((previous.OUT / 'latency.json').read_text())
    for field, family in [('candidate_checkpoint_sha256', chosen), ('supervised_checkpoint_sha256', 'supervised')]:
        expected = {name: sha for name, sha in training['checkpoint_sha256'].items()
                    if Path(name).name.startswith(family + '-')}
        if record[field] != expected or len(expected) != 3:
            raise RuntimeError('Selected checkpoint set differs from frozen three-member family')
    names = ['radiomap', 'frozen_fleet_jepa'] + [f'{family}-s{seed}'
            for family in (chosen, 'supervised') for seed in protocol()['training_seeds']]
    if record['reused_timing_profiles'] != {name: latency['profiles'][name] for name in names}:
        raise RuntimeError('Selected timing profiles differ from pre-existing calibration')
    for score, family in scores:
        saved = next(row for row in record['candidates'] if row['family'] == family)
        if Fraction(saved['exact_score_numerator'], saved['exact_score_denominator']) != score:
            raise RuntimeError('Selection score differs from raw prior counters')
    return chosen


def verify_fresh_sites():
    sites = protocol()['confirmation_sites']
    if len(sites) != 8 or len(set(sites)) != 8 or any(type(s) is not int for s in sites):
        raise RuntimeError('Need eight distinct registered sites')
    def numbers(value):
        if isinstance(value, dict):
            return set().union(*(numbers(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(numbers(v) for v in value)) if value else set()
        return {value} if type(value) is int else set()
    for path in (ROOT / 'artifacts').glob('*/protocol.json'):
        if path != PROTOCOL and set(sites) & numbers(json.loads(path.read_text())):
            raise RuntimeError('New confirmation overlaps prior registration: ' + str(path))


def register():
    verify_fresh_sites()
    verify_selection()
    path = OUT / 'registration/source.json'
    with locked(OUT / 'registration/.lock'):
        if path.exists():
            return check_sources()
        hashes = source_hashes()
        for name, expected in hashes.items():
            target = OUT / 'registration/source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) != expected:
                    raise RuntimeError('Partial source capsule differs')
            else:
                shutil.copyfile(ROOT / name, target)
        value = {'schema': 1, 'time_unix': time.time(), 'source_sha256': hashes,
                 'software_versions': previous.versions(), 'confirmation_data_accessed': False,
                 'source_training_and_development_already_observed': True,
                 'selected_family': selection()['selected_family'], 'new_fits': 0}
        write_once(path, value)
        return value


def check_sources():
    previous.check_sources()
    saved = json.loads((OUT / 'registration/source.json').read_text())
    if saved['source_sha256'] != source_hashes() or saved['software_versions'] != previous.versions():
        raise RuntimeError('Selected-JEPA frozen source/input/software changed')
    for name, expected in saved['source_sha256'].items():
        if digest(OUT / 'registration/source' / name) != expected:
            raise RuntimeError('Selected-JEPA source capsule changed: ' + name)
    if selection()['selected_family'] != protocol()['candidate_family']:
        raise RuntimeError('Selected model/protocol mismatch')
    return saved
