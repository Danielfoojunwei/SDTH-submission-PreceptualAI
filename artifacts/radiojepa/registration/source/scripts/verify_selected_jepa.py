"""Verify the completed selected-JEPA confirmation without new rollouts.

Default checks an existing manifest. --write-manifest explicitly creates or
refreshes derived metadata. Both modes reproduce frozen report arithmetic and
read existing checkpoints/receipts/archives. Neither constructs worlds, trains
models, selects new weights or retimes inference. Both scientific Git histories
must be available in the local clone.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/selected_jepa'
SELECTED_COMMIT = 'f5c69d55cf110578f7c48de96899a5879bb9f106'
PARENT_COMMIT = '9ff5fa56e7327bccde15a10a1a3952a82c699e02'
PRIOR_PR18_COMMIT = '18c294988c242106f4f01e2a85ce1ef55f099e2b'
DOCUMENTS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
    'docs/SELECTED_JEPA_DESIGN.md', 'docs/SELECTED_JEPA_RESULTS.md',
    'docs/SEMANTIC_JEPA_DESIGN.md', 'docs/SEMANTIC_JEPA_RESULTS.md')
sys.path.insert(0, str(ROOT / 'src'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def hash_value(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def committed_bytes(commit, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe registered relative path')
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def git_tree(commit):
    return subprocess.check_output(['git', 'rev-parse', commit + '^{tree}'], cwd=ROOT, text=True).strip()


def visible_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return set()
    return {item.name for item in folder.iterdir()
            if not item.name.startswith('.') and not item.name.endswith(('.lock', '.part'))}


def check_membership(folder, names):
    expected = {name + suffix for name in names for suffix in ('.json.gz', '.receipt.json')}
    require(visible_files(folder) == expected, 'Archive membership differs: ' + str(folder))


def check_registration(c):
    frozen = c.check_sources()
    require(len(frozen['source_sha256']) == 230 and frozen['confirmation_data_accessed'] is False
            and frozen['source_training_and_development_already_observed'] is True
            and frozen['new_fits'] == 0, 'Selected registration scope/source count differs')
    meta = read_json(OUT / 'registration/git.json')
    tree = git_tree(SELECTED_COMMIT)
    require(meta['scientific_source_commit'] == SELECTED_COMMIT
            and meta['scientific_source_tree'] == tree and meta['before_new_confirmation'] is True,
            'Selected pre-confirmation Git record differs from the pinned commit')
    require((OUT / 'registration/source.json').read_bytes() == committed_bytes(SELECTED_COMMIT,
            'artifacts/selected_jepa/registration/source.json'),
            'Registration receipt differs from the pre-confirmation Git commit')
    for name, expected in frozen['source_sha256'].items():
        value = committed_bytes(SELECTED_COMMIT, name)
        require(hashlib.sha256(value).hexdigest() == expected
                and (OUT / 'registration/source' / name).read_bytes() == value,
                'Scientific source/capsule differs from committed bytes: ' + name)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', SELECTED_COMMIT, 'artifacts'],
                                    cwd=ROOT, text=True).splitlines()
    require(not any(name.startswith(('artifacts/selected_jepa/cases/',
                                    'artifacts/selected_jepa/replays/')) for name in names),
            'Pre-confirmation Git commit already contains new scientific cases')
    current_sites = set(c.protocol()['confirmation_sites'])
    def numbers(value):
        if isinstance(value, dict):
            return set().union(*(numbers(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(numbers(v) for v in value)) if value else set()
        return {value} if type(value) is int else set()
    for name in names:
        if re.fullmatch(r'artifacts/[^/]+/protocol\.json', name) and name != 'artifacts/selected_jepa/protocol.json':
            require(not current_sites & numbers(json.loads(committed_bytes(SELECTED_COMMIT, name))),
                    'Selected confirmation overlaps a protocol present at registration: ' + name)
    return frozen, tree


def check_selection(c):
    """Derive the choice from all 112 verified, already-observed old cases."""
    from wirelessjev.semantic_jepa import common as previous, study as parent
    previous_frozen = previous.check_sources()
    require((previous.OUT / 'registration/source.json').read_bytes() == committed_bytes(
            PARENT_COMMIT, 'artifacts/semantic_jepa/registration/source.json'),
            'Parent pre-training registration receipt changed')
    selection = c.selection()
    flags = {'sealed': True, 'source_scientific_commit': PARENT_COMMIT,
        'source_training_already_complete': True, 'source_development_already_observed': True,
        'source_semantic_confirmation_veto_preserved': True, 'new_confirmation_accessed': False,
        'no_new_training_or_timing_calibration': True}
    require(all(selection.get(key) == value for key, value in flags.items()),
            'Selection lost its retrospective-development/prospective-test boundary')
    require(set(selection['source_receipt_sha256']) ==
            {'development.json', 'training.json', 'latency.json', 'protocol.json'},
            'Selection source receipt membership differs')
    for name, expected in selection['source_receipt_sha256'].items():
        require(digest(previous.OUT / name) == expected, 'Selection input changed: ' + name)
    keys = parent.case_keys('development')
    require(len(keys) == 112, 'Expected all 112 prior development cases')
    check_membership(previous.OUT / 'cases', [parent.case_name(key) for key in keys])
    check_membership(previous.OUT / 'replays', [])
    rows = [parent.load_case(key) for key in keys]
    require(all(row is not None for row in rows), 'Prior selection input archive missing')
    dev = read_json(previous.OUT / 'development.json')
    require(parent.validate_development_seal(dev, rows) == [],
            'Original semantic study must retain its recorded rejection')
    old_confirmation = read_json(previous.OUT / 'confirmation.json')
    require(old_confirmation['completed'] is True and old_confirmation['passed_costs'] == []
            and old_confirmation['confirmation_cases'] == 0 and old_confirmation['replays'] == []
            and old_confirmation['confirmation_not_run_if_development_rejected'] is True,
            'Original semantic confirmation holdouts are no longer unused')
    require(parent.report() == read_json(previous.OUT / 'report.json'),
            'Original rejected study report differs from preserved archives')
    require({row['family'] for row in selection['candidates']} == {'context_jepa', 'semantic_jepa'}
            and len(selection['candidates']) == 2, 'Selection suppressed an eligible JEPA family')
    scores = []
    for family in ('context_jepa', 'semantic_jepa'):
        ratios, rates = [], {}
        for cost in c.protocol()['handover_seconds']:
            candidate = [row for row in rows if row['family'] == family and row['handover_seconds'] == cost]
            reference = [row for row in rows if row['family'] == 'radiomap' and row['handover_seconds'] == cost]
            rate = Fraction(sum(row['result']['control_misses'] for row in candidate),
                            sum(row['result']['control_sent'] for row in candidate))
            ref = Fraction(sum(row['result']['control_misses'] for row in reference),
                           sum(row['result']['control_sent'] for row in reference))
            ratios.append(rate / ref); rates[str(float(cost))] = float(rate)
        score = sum(ratios) / len(ratios)
        recorded = next(row for row in selection['candidates'] if row['family'] == family)
        require(Fraction(recorded['exact_score_numerator'], recorded['exact_score_denominator']) == score,
                'Development selection differs from exact raw-counter ratios')
        require(math.isclose(recorded['mean_loss_ratio_to_radiomap'], float(score), rel_tol=0, abs_tol=1e-15)
                and recorded['loss_by_cost'] == rates, 'Descriptive selection score differs')
        scores.append((score, family))
    chosen = min(scores)[1]
    require(chosen == selection['selected_family'] == c.protocol()['candidate_family']
            == c.verify_selection(), 'Selected family differs from the frozen exact ranking')
    training = parent.training_receipt()
    for name, expected in training['fit_receipt_sha256'].items():
        path = previous.OUT / 'fits' / (name + '.json')
        require(digest(path) == expected, 'Source fit receipt changed: ' + name)
        fit = read_json(path)
        require(fit['completed'] is True and fit['data_receipt_sha256'] == training['data_receipt_sha256'],
                'Source fit or data identity differs')
    require(len(training['fit_receipt_sha256']) == 12, 'Source fit count differs')
    for split, expected in training['data_receipt_sha256'].items():
        require(digest(previous.OUT / 'data' / (split + '.json')) == expected,
                'Source training data receipt changed')
    latency = read_json(previous.OUT / 'latency.json')
    require(latency['sealed'] is True and latency['outcomes_used'] is False
            and latency['source_sha256'] == previous_frozen['source_sha256']
            and latency['checkpoint_sha256'] == training['checkpoint_sha256'],
            'Reused timing receipt is not the original sealed profile')
    for field, family in [('candidate_checkpoint_sha256', chosen), ('supervised_checkpoint_sha256', 'supervised')]:
        expected = {name: value for name, value in training['checkpoint_sha256'].items()
                    if Path(name).name.startswith(family + '-')}
        require(len(expected) == 3 and selection[field] == expected, 'Selected fixed members differ')
    return selection, training, latency, len(rows)


def check_jobs(jobs, duration, profile, extra):
    require(len(jobs) == duration, 'Incomplete forecast ledger')
    previous, overrun_count = 0., 0
    for decision, row in enumerate(jobs):
        require(row['decision'] == decision and row['history_cutoff_exclusive'] == 60 + decision,
                'Forecast consumes a noncausal prefix')
        kind = 'cold' if decision == 0 else 'refit' if decision % 20 == 0 else 'warm'
        require(row['kind'] == kind, 'Map-refit regime differs')
        charge = profile[kind] + extra
        require(math.isclose(row['charged_seconds'], charge, rel_tol=1e-13, abs_tol=1e-13),
                'Job charge differs from original timing profile')
        previous = max(float(decision), previous) + charge
        require(row['ready_slot'] == int(math.ceil(previous * 100 - 1e-10)), 'FIFO completion differs')
        actual = row['actual_prediction_seconds']
        require(type(actual) in (int, float) and math.isfinite(actual) and actual >= 0,
                'Malformed measured prediction time')
        overrun = actual > profile[kind]
        require(row['runtime_overrun'] is overrun and is_hash(row['forecast_sha256']),
                'Job runtime flag or forecast hash differs')
        overrun_count += int(overrun)
    return overrun_count


def check_cases(c, rollout, keys, identity, latency, *, replay=False):
    rows, overrun_cases = [], 0
    p = c.protocol()
    for key in keys:
        row = rollout.load_case(key, replay=replay)
        require(row is not None, 'Missing selected case: ' + rollout.case_name(key))
        require(all(row.get(name) == value for name, value in identity.items()),
                'Archive source/selection/checkpoint/input identity differs')
        require(row['causality_pass'] is True and row['hardware_qualified'] is False
                and row['simulation_only'] is True, 'Case observation/qualification boundary differs')
        events = row['telemetry']['events']
        require(row['telemetry']['reports_delivered'] == len(events), 'Delivered report count differs')
        for event in events:
            require(type(event['robot']) is int and 0 <= event['robot'] < p['robots']
                    and 0 <= event['measurement_slot'] < event['complete_slot'] <= event['delivery_slot']
                    < p['duration_seconds'] * 100
                    and event['first_usable_decision'] == event['delivery_slot'] // 100 + 1
                    and is_hash(event['payload_sha256']), 'RF report violates capture/delivery causality')
        profile = latency['profiles'][rollout.arm_name(row['family'], row['training_seed'])]
        overrun = bool(check_jobs(row['jobs'], p['duration_seconds'], profile, p['extra_compute_seconds']))
        require(row['runtime_overrun'] is overrun, 'Case runtime-veto summary differs')
        require(is_hash(row['forecast_mean_sha256']) and is_hash(row['forecast_std_sha256'])
                and row['action_trace_sha256'] == hash_value(row['result']['packet_detail']['action_trace']),
                'Forecast/action trace digest differs')
        rows.append(row); overrun_cases += int(overrun)
    return rows, overrun_cases


def preserve_prior():
    path = ROOT / 'artifacts/continual_packet/verification/manifest.json'
    require(path.read_bytes() == committed_bytes(PRIOR_PR18_COMMIT, str(path.relative_to(ROOT))),
            'Prior PR18 verification manifest changed')
    manifest = read_json(path)
    counts = {}
    for group in ('sources_sha256', 'evidence_sha256'):
        for name, expected in manifest[group].items():
            require(digest(ROOT / name) == expected, 'Prior PR18 source/evidence changed: ' + name)
        counts[group] = len(manifest[group])
    return digest(path), counts


def verify(*, write_manifest=False):
    require((OUT / 'report.json').is_file() and (OUT / 'report.md').is_file(),
            'Complete all selected confirmation/replay/report artifacts first')
    from wirelessjev.selected_jepa import common as c, study, rollout
    frozen, tree = check_registration(c)
    selection, training, latency, old_dev_count = check_selection(c)
    keys, replay_keys = rollout.case_keys(), study.replay_keys()
    require(len(keys) == 128 and len(replay_keys) == 8, 'Expected 128 main cases and eight replays')
    check_membership(OUT / 'cases', [rollout.case_name(key) for key in keys])
    check_membership(OUT / 'replays', [rollout.case_name(key) for key in replay_keys])
    identity = {'source_sha256': frozen['source_sha256'],
        'registration_sha256': digest(OUT / 'registration/source.json'),
        'selection_sha256': digest(OUT / 'selection.json'),
        'checkpoint_sha256': training['checkpoint_sha256'],
        'parent_training_sha256': digest(ROOT / 'artifacts/semantic_jepa/training.json'),
        'latency_sha256': digest(ROOT / 'artifacts/semantic_jepa/latency.json')}
    rows, main_overruns = check_cases(c, rollout, keys, identity, latency)
    replays, replay_overruns = check_cases(c, rollout, replay_keys, identity, latency, replay=True)
    comparisons = study.compare_replays(rows, replays)
    require(len(comparisons) == 8 and all(row['exact'] for row in comparisons),
            'Required exact result/forecast/telemetry/action replay failed')
    archived = read_json(OUT / 'report.json')
    require(study.report() == archived, 'Frozen report differs from actual archives')
    require(archived['candidate_family'] == selection['selected_family']
            and archived['case_count'] == 128 and archived['independent_sites'] == 8
            and archived['primary_multiplicity_budget'] == 6 and len(archived['primary']) == 6
            and archived['any_runtime_overrun'] is bool(main_overruns + replay_overruns),
            'Report identity/statistical units/runtime veto differs')
    require(archived['hardware_qualified'] is False and archived['production_qualified'] is False
            and archived['activated'] is False and archived['ema_necessity_claimed'] is False,
            'Unsupported deployment or isolated-EMA claim')
    if main_overruns or replay_overruns:
        # Statistical and engineering diagnostics stay intact. The registered
        # global runtime veto independently blocks EVERY qualified claim.
        require(all(row['qualified'] is False and row['passes'] is False
                    for row in archived['primary']),
                'An actual runtime overrun did not veto every reference qualification')
        claim_flags = ('numerical_benefit', 'original_jepa_improvement',
                       'objective_advantage', 'all_references_pass', 'passes')
        require(all(row[name] is False for row in archived['per_cost'] for name in claim_flags)
                and archived['headline_numerical_benefit_at_4s'] is False
                and archived['both_costs_all_references_pass'] is False,
                'An actual runtime overrun left a positive service claim')
    log_path = OUT / 'logs/preflight-tests.txt'
    require(re.search(r'^895 passed, 1 warning in 144\.27s', log_path.read_text(), re.M) is not None,
            'Archived 895-test execution log differs')
    prior_sha, prior_counts = preserve_prior()
    sources = [path for folder in ('src', 'scripts', 'tests') for path in (ROOT / folder).rglob('*.py')
               if '__pycache__' not in path.parts] + [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    documents = [ROOT / name for name in DOCUMENTS]
    proof = OUT / 'verification'; path = proof / 'manifest.json'
    # Only this experiment's outputs enter this evidence map. In particular,
    # the parent semantic verification manifest is not included, avoiding a
    # cross-experiment manifest cycle during final publication.
    evidence = [item for item in OUT.rglob('*') if item.is_file() and item != path
                and not any(part.startswith('.') for part in item.relative_to(OUT).parts)
                and not item.name.endswith(('.lock', '.part'))]
    hashes = lambda paths: {str(item.relative_to(ROOT)): digest(item) for item in sorted(paths)}
    value = {'schema': 1, 'scientific_source_commit': SELECTED_COMMIT, 'scientific_source_tree': tree,
        'source_registration_sha256': digest(OUT / 'registration/source.json'),
        'registered_source_files': len(frozen['source_sha256']),
        'registration_precedes_only_new_confirmation_not_source_training': True,
        'candidate_family': selection['selected_family'], 'selection_sha256': digest(OUT / 'selection.json'),
        'selection_rederived_from_prior_development_cases': old_dev_count,
        'original_semantic_study_veto_and_unused_holdouts_preserved': True,
        'source_study_independent_fits': 12, 'new_fits': 0,
        'main_cases': len(rows), 'exact_replays': len(replays), 'independent_sites': 8,
        'full_tests': {'passed': 895, 'warnings': 1, 'seconds': 144.27,
                       'log': str(log_path.relative_to(ROOT)), 'sha256': digest(log_path)},
        'per_cost_claims': archived['per_cost'],
        'headline_numerical_benefit_at_4s': archived['headline_numerical_benefit_at_4s'],
        'all_registered_source_archive_input_checks_passed': True,
        'frozen_report_recomputed_from_actual_archives': True,
        'report_sha256': digest(OUT / 'report.json'),
        'runtime_overrun_cases': {'main': main_overruns, 'replays': replay_overruns},
        'prior_PR18_manifest_sha256': prior_sha, 'prior_PR18_preserved_entry_counts': prior_counts,
        'previous_global_document_hashes_apply_to_their_original_commits': True,
        'hardware_qualified': False, 'production_qualified': False, 'activated': False,
        'ema_necessity_claimed': False,
        'new_worlds_training_or_timing_calibration_executed_by_verifier': False,
        'sources_sha256': hashes(sources), 'documents_sha256': hashes(documents),
        'evidence_sha256': hashes(evidence)}
    if write_manifest:
        proof.mkdir(exist_ok=True)
        value['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
        fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=proof)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write(json.dumps(value, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    else:
        stored = read_json(path); stored.pop('verified_at_utc')
        require(stored == value, 'Final source/document/evidence manifest differs')
    print(json.dumps({key: value[key] for key in ('candidate_family', 'main_cases', 'exact_replays',
        'headline_numerical_benefit_at_4s', 'runtime_overrun_cases', 'production_qualified')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true',
                        help='Explicitly create/refresh final derived verification metadata')
    args = parser.parse_args()
    verify(write_manifest=args.write_manifest)
