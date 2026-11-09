"""Passive final source/document/evidence manifest for pilot/fast-loop.

Run after measured scientific execution stops. --write creates the immutable
verification/manifest.json; plain mode recomputes and checks its exact bytes.
No World/model, training, forward, or timing execution occurs in either mode.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/pilot_fastloop'
MANIFEST = OUT / 'verification/manifest.json'
RAW_AUDIT = OUT / 'verification/raw_audit.json'
sys.path.insert(0, str(ROOT / 'src'))

# Actual published pre-data metadata; the verifier is outside its source capsule.
SCIENTIFIC_SOURCE_COMMIT = 'f8821588456f4ca8873211d8157987f40e6a3928'
SCIENTIFIC_SOURCE_TREE = '182443bc5c6ed9706629cdb8c714e71408ffe829'
EXPECTED_SOURCE_COUNT = 467
EXPECTED_TESTS = (1861, 1, 206.13)
EXPECTED_TEST_LOG_SHA256 = '0414f44ce47c1172aa7c9a5bf5440e3119450d3f39f47feba05190dc46f6b534'
EXPECTED_TESTED_SOURCE_SHA256 = '8fd0abd757dd43cea32f498c70da8106c206cda8ed79ff74ce2eed3320ad8fa1'
EXPECTED_TESTED_SOURCE_COUNT = 2806
OPERATIONAL_SOURCE_COMMIT = 'cc019928864ffa8ebea686b2e02becc302e54c7d'
OPERATIONAL_SOURCE_TREE = '867d5fca4503d9c3efd8c7a3c9be0cf7dba0d5b3'
OPERATIONAL_WRAPPER_SHA256 = 'b7f9ea6bfa460b7f28164bd4af23202eafd730322bb75c318ebe3dcc70550200'
OPERATIONAL_FIXTURE_SHA256 = '1aabf4a8fdca1a6e25d2b01a710dcb93317d3e8a2cc91a76b43c5e0be49c09d9'
OPERATIONAL_TEST_LOG_SHA256 = '7ecfea79df9bfe7c3c0d2bca7aeca6eccee8bc9564a6164fcf8fc45bdcc59c5c'
OPERATIONAL_PREFLIGHT_SHA256 = '1cf933b403737e752d3e8d0a9c3a233fbf4d8559060d0d609a2f93d3eac343eb'
OPERATIONAL_TESTS = {'passed': 50, 'warnings': 0, 'seconds': .52}
OPERATIONAL_METADATA_COMMIT = '8a01db8679c2d276151168c37fea07070b0556b7'
OPERATIONAL_RESUME_SHA256 = '10f15f62705e35c10e25c519026cc50cfa01ba30a24787e09789c8732e1c88e5'
PREVIOUS_PUBLISHED_COMMIT = '9a8577d2f5334bbf1f4ea805811b40c35a11212a'
DOCUMENTS = ('README.md', 'docs/HANDOFF.md', 'docs/PAPER_PROPOSAL.md',
             'docs/JEPA_ADVANTAGES_DIAGNOSIS.md', 'docs/PILOT_FASTLOOP_DESIGN.md',
             'docs/PILOT_FASTLOOP_RESULTS.md')
AUDIT_PROOFS = ('independent_pilot_payload_provenance_verified',
                'independent_fast_local_arithmetic_verified',
                'independent_four_primary_bootstrap_verified', 'report_recomputed',
                'exact_native_and_matched_replays_verified',
                'no_worlds_models_training_or_timing_executed')
FIGURE_STEMS = ('pilot-fastloop-packet-factorial', 'pilot-fastloop-primary-and-rf',
                'pilot-fastloop-resources')
FIGURE_INPUTS = ('protocol.json', 'registration/source.json', 'qualification.json',
                 'completion.json', 'development.json', 'execution_order.json',
                 'report.json', 'verification/raw_audit.json')


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _file(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts),
            'Unsafe verification path: ' + str(name))
    full = ROOT / path
    require(full.resolve().is_relative_to(ROOT.resolve()) and full.is_file(),
            'Missing or escaped verification file: ' + str(name))
    return full


def committed(commit, name):
    _file(name)
    return subprocess.check_output(['git', 'show', commit + ':' + name], cwd=ROOT)


def _tree(commit):
    return subprocess.check_output(['git', 'rev-parse', commit + '^{tree}'], cwd=ROOT, text=True).strip()


def _git_names(commit, *folders):
    return subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, *folders],
                                   cwd=ROOT, text=True).splitlines()


def _hashes(paths):
    return {str(path.relative_to(ROOT)): digest(_file(str(path.relative_to(ROOT))))
            for path in sorted(set(paths))}


def _check_map(entries, message):
    require(isinstance(entries, dict) and entries, 'Missing verification digest map')
    for name, expected in entries.items():
        require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected)
                and digest(_file(name)) == expected, message + name)


def _value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _bound_development_files():
    """Independent fixed membership for the wrapper's permission dependencies."""
    base = [OUT / name for name in ('registration/source.json', 'registration/git.json',
             'protocol.json', 'execution_order.json', 'development.json', 'qualification.json')]
    base.extend(path for path in (OUT / 'registration/source').rglob('*') if path.is_file())
    paths = list(base)
    p = read(OUT / 'protocol.json')
    for site in p['development_sites']:
        for family in p['families']:
            for variant in p['variants_by_cost']['4.0']:
                name = f'development-{site}-h4000-{family}-{variant}'
                paths.extend(OUT / 'cases' / (name + suffix) for suffix in ('.json.gz', '.receipt.json'))
            name = f'development-{site}-{family}'
            paths.extend(OUT / 'matched/cases' / (name + suffix) for suffix in ('.json.gz', '.receipt.json'))
            paths.append(OUT / 'matched/predictions' / (name + '.npz'))
        paths.extend(OUT / 'matched/inputs' / (f'development-{site}' + suffix)
                     for suffix in ('.public.npz', '.labels.npz', '.metadata.json'))
    expected = {str(path.relative_to(ROOT)) for path in paths}
    actual = {str(path.relative_to(ROOT)) for path in base}
    for name in ('cases', 'matched/cases', 'matched/predictions', 'matched/inputs'):
        actual.update(str(path.relative_to(ROOT)) for path in (OUT / name).glob('development*')
                      if path.suffix != '.lock')
    require(actual == expected, 'Cached development dependency membership differs from the fixed complete matrix')
    require(all(path.is_file() and not path.is_symlink() for path in paths)
            and all(not (OUT / name).is_symlink() for name in
                    ('registration/source', 'cases', 'matched/cases', 'matched/predictions', 'matched/inputs')),
            'Cached development dependency was redirected')
    return _hashes(paths)


def _cached_gate_records(resume, launch, frozen):
    """Authenticate original bootstraps and guarded permission reuse."""
    paths = sorted((OUT / 'logs').glob('cached-development-*.bootstrap.json'))
    require(len(paths) == 2, 'Require one original full bootstrap for each resumed phase')
    bound = _bound_development_files()
    gate = read(OUT / 'development.json')['gate']
    gate_sha, bound_sha = _value_sha256(gate), _value_sha256(bound)
    phase_records, receipts, expected_journals = {}, [], set()
    for path in paths:
        value = read(path)
        phase, cache_id = value.get('phase'), value.get('cache_id')
        match = re.fullmatch(r'(confirmation|report)-([0-9]+)-([0-9]+)', str(cache_id))
        parent_pid = value.get('cache_parent_pid')
        require(match is not None and match[1] == phase and phase not in phase_records
                and type(parent_pid) is int and parent_pid > 0 and int(match[3]) == parent_pid
                and path.name == f'cached-development-{cache_id}.bootstrap.json',
                'Malformed or duplicated operational bootstrap identity')
        require(value.get('schema') == 1 and value.get('original_full_verified_development') is True
                and value.get('source_guard_checked') is True and value.get('unchanged') is True
                and value.get('registered_scientific_source_changed') is False
                and value.get('new_worlds_models_or_training_called_by_wrapper') is False
                and value.get('wrapper_path') == resume['wrapper_path']
                and value.get('wrapper_sha256') == resume['wrapper_sha256']
                and value.get('original_common_path') == 'src/wirelessjev/pilot_fastloop/common.py'
                and value.get('original_common_sha256') == frozen['source_sha256']['src/wirelessjev/pilot_fastloop/common.py']
                and value.get('before_sha256') == value.get('after_sha256') == bound
                and value.get('file_count') == len(bound) and value.get('bound_files_sha256') == bound_sha
                and value.get('gate') == gate and value.get('gate_sha256') == gate_sha
                and _finite(value.get('recorded_unix'))
                and value['recorded_unix'] >= launch['recorded_before_first_resumed_phase_unix']
                and value['recorded_unix'] * 1e9 >= int(match[2]),
                'Operational full verification, dependencies, source or gate proof differs')
        journal = path.with_name(path.name.removesuffix('.bootstrap.json') + '.cache-hits.jsonl')
        require(journal.is_file(), 'Missing cached permission hit journal')
        expected_journals.add(journal)
        hits = _cache_hits(journal, path, value, resume, bound_sha, len(bound), gate_sha)
        phase_records[phase] = {'cache_id': cache_id, 'cache_parent_pid': parent_pid,
            'original_full_bootstrap_sha256': digest(path), 'cache_hits_sha256': digest(journal),
            'cache_hits': len(hits), 'processes': sorted({hit['pid'] for hit in hits}),
            'recorded_unix': value['recorded_unix']}
        receipts.extend((path, journal))
    require(set(phase_records) == {'confirmation', 'report'}
            and phase_records['confirmation']['recorded_unix'] <= phase_records['report']['recorded_unix']
            and set((OUT / 'logs').glob('cached-development-*.cache-hits.jsonl')) == expected_journals,
            'Resumed phase order or cache journal membership differs')
    return phase_records, receipts, gate_sha


def _cache_hits(journal, bootstrap_path, bootstrap, resume, bound_sha, file_count, gate_sha):
    raw = journal.read_bytes()
    require(bool(raw) and raw.endswith(b'\n'), 'Empty or partial cached permission hit journal')
    hits, previous = [], {}
    parent_pid = bootstrap['cache_parent_pid']
    bootstrap_sha = digest(bootstrap_path)
    for line in raw.splitlines():
        hit = json.loads(line)
        pid, ppid, index = hit.get('pid'), hit.get('parent_pid'), hit.get('process_hit_index')
        require(type(pid) is int and pid > 0 and type(ppid) is int and ppid > 0
                and (pid == parent_pid or ppid == parent_pid)
                and type(index) is int and index > 0,
                'Cached permission process is outside its parent-verified fork tree')
        require(hit.get('schema') == 1 and hit.get('phase') == bootstrap['phase']
                and hit.get('cache_id') == bootstrap['cache_id']
                and hit.get('cache_parent_pid') == parent_pid
                and hit.get('bootstrap_path') == str(bootstrap_path.relative_to(ROOT))
                and hit.get('bootstrap_sha256') == bootstrap_sha
                and hit.get('wrapper_sha256') == resume['wrapper_sha256']
                and hit.get('source_guard_checked') is True
                and hit.get('bytes_and_membership_checked') is True
                and hit.get('original_full_verification_reexecuted') is False
                and hit.get('bound_files_sha256') == bound_sha and hit.get('file_count') == file_count
                and hit.get('gate_sha256') == gate_sha and _finite(hit.get('recorded_unix'))
                and hit['recorded_unix'] >= bootstrap['recorded_unix'],
                'Cached permission omitted source/byte/membership/gate guards')
        old_index, old_time = previous.get(pid, (0, bootstrap['recorded_unix']))
        require(index == old_index + 1 and hit['recorded_unix'] >= old_time,
                'Cached permission process ledger is incomplete or out of order')
        previous[pid] = index, hit['recorded_unix']
        hits.append(hit)
    return hits


def _operational_tests():
    """The wrapper has separate published fixtures; original tests stay scoped."""
    wrapper = ROOT / 'scripts/run_pilot_fastloop_cached.py'
    fixture = OUT / 'logs/cached-wrapper-fixture.py'
    log = OUT / 'logs/cached-wrapper-tests.txt'
    path = OUT / 'logs/cached-wrapper-preflight.json'
    value = read(path)
    expected_map = {'scripts/run_pilot_fastloop_cached.py': OPERATIONAL_WRAPPER_SHA256,
                    'fixture_source': OPERATIONAL_FIXTURE_SHA256}
    require(digest(wrapper) == OPERATIONAL_WRAPPER_SHA256
            and digest(fixture) == OPERATIONAL_FIXTURE_SHA256
            and digest(log) == OPERATIONAL_TEST_LOG_SHA256
            and digest(path) == OPERATIONAL_PREFLIGHT_SHA256
            and value.get('schema') == 1
            and value.get('wrapper_path') == str(wrapper.relative_to(ROOT))
            and value.get('fixture_path') == str(fixture.relative_to(ROOT))
            and value.get('test_log_path') == str(log.relative_to(ROOT))
            and value.get('wrapper_sha256') == OPERATIONAL_WRAPPER_SHA256
            and value.get('fixture_sha256') == OPERATIONAL_FIXTURE_SHA256
            and value.get('test_log_sha256') == OPERATIONAL_TEST_LOG_SHA256
            and value.get('tests') == OPERATIONAL_TESTS
            and value.get('before_sha256') == value.get('after_sha256') == expected_map
            and value.get('unchanged') is True
            and value.get('new_worlds_models_or_training') is False
            and value.get('registered_scientific_source_changed') is False,
            'Published supplemental wrapper fixture/source/test receipt differs')
    matches = re.findall(r'(?m)^([0-9]+) passed(?:, ([0-9]+) warnings?)? in ([0-9.]+)s', log.read_text())
    require(len(matches) == 1 and {'passed': int(matches[0][0]), 'warnings': int(matches[0][1] or 0),
            'seconds': float(matches[0][2])} == OPERATIONAL_TESTS,
            'Supplemental wrapper test output differs from actual fixture summary')
    return [wrapper, fixture, log, path], value


def operational_check(frozen, audit):
    """Check separately published post-development caching; never import it."""
    resume_path, pause_path = (OUT / 'logs' / name for name in ('operational-resume.json', 'operational-pause.json'))
    preservation_path, supervisor_path, launch_path = (OUT / 'logs' / name for name in
                    ('operational-preservation.json', 'resume-supervisor.py', 'resume-launch.json'))
    resume, pause, preservation, launch = (read(path) for path in
                    (resume_path, pause_path, preservation_path, launch_path))
    tests_paths, tests = _operational_tests()
    require(resume.get('schema') == 1 and resume.get('scientific_source_commit') == SCIENTIFIC_SOURCE_COMMIT
            and resume.get('registered_source_count') == EXPECTED_SOURCE_COUNT
            and resume.get('operational_source_commit') == OPERATIONAL_SOURCE_COMMIT
            and resume.get('operational_source_tree') == _tree(OPERATIONAL_SOURCE_COMMIT) == OPERATIONAL_SOURCE_TREE
            and resume.get('wrapper_path') == 'scripts/run_pilot_fastloop_cached.py'
            and resume.get('wrapper_sha256') == OPERATIONAL_WRAPPER_SHA256
            and resume.get('source_published_before_resume') is True
            and resume.get('completed_confirmation_receipts_at_resume') == {}
            and resume.get('scientific_code_models_controller_criteria_or_seeds_changed') is False
            and resume.get('cache_only_verified_development') is True
            and resume.get('full_original_raw_bootstrap_once_per_phase') is True
            and resume.get('source_and_all_bound_development_bytes_checked_each_reuse') is True
            and resume.get('amendment_after_development') is True
            and resume.get('supplemental_tests') == OPERATIONAL_TESTS
            and digest(resume_path) == OPERATIONAL_RESUME_SHA256
            and resume_path.read_bytes() == committed(OPERATIONAL_METADATA_COMMIT, str(resume_path.relative_to(ROOT))),
            'Operational publication/cache scope or fixed scientific identity differs')
    bindings = {'wrapper_test_source': tests_paths[1], 'wrapper_test_log': tests_paths[2],
                'wrapper_test_receipt': tests_paths[3], 'resume_supervisor': supervisor_path}
    for prefix, path in bindings.items():
        require(resume.get(prefix + '_path') == str(path.relative_to(ROOT))
                and resume.get(prefix + '_sha256') == digest(path),
                'Operational supplemental source/log/receipt binding differs: ' + prefix)
    _operational_launch(resume, pause, preservation, launch, pause_path, preservation_path, supervisor_path, resume_path)
    published = tests_paths + [pause_path, preservation_path, supervisor_path]
    for path in published:
        require(path.read_bytes() == committed(OPERATIONAL_SOURCE_COMMIT, str(path.relative_to(ROOT))),
                'Operational wrapper/test/supervisor differs from published pre-resume Git bytes')
    forbidden = tuple('artifacts/pilot_fastloop/' + folder + '/confirmation'
                     for folder in ('cases', 'replays', 'live_replications', 'matched/cases',
                                    'matched/replays', 'matched/predictions', 'matched/replay_predictions', 'matched/inputs'))
    for commit in (OPERATIONAL_SOURCE_COMMIT, OPERATIONAL_METADATA_COMMIT):
        require(not any(name.startswith(forbidden) for name in _git_names(commit, 'artifacts/pilot_fastloop')),
                'Published operational pre-resume commit contains confirmation outputs')
    for parent, child in ((SCIENTIFIC_SOURCE_COMMIT, OPERATIONAL_SOURCE_COMMIT),
                          (OPERATIONAL_SOURCE_COMMIT, OPERATIONAL_METADATA_COMMIT)):
        require(subprocess.run(['git', 'merge-base', '--is-ancestor', parent, child], cwd=ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
                'Operational publication ancestry differs from the original scientific source')
    phase_records, cache_paths, gate_sha = _cached_gate_records(resume, launch, frozen)
    core_paths = tests_paths + [pause_path, resume_path] + cache_paths
    operational = audit.get('operational_provenance', {})
    require(operational.get('operational_files_sha256') == _hashes(core_paths)
            and operational.get('operational_resume_sha256') == digest(resume_path)
            and operational.get('development_gate_sha256') == gate_sha,
            'Independent uncached raw audit operational provenance differs')
    return {'scientific_source_unchanged': True, 'registered_source_files': EXPECTED_SOURCE_COUNT,
            'operational_source_commit': OPERATIONAL_SOURCE_COMMIT, 'operational_source_tree': OPERATIONAL_SOURCE_TREE,
            'operational_metadata_commit': OPERATIONAL_METADATA_COMMIT,
            'operational_files_sha256': _hashes(core_paths + [preservation_path, supervisor_path, launch_path]),
            'supplemental_wrapper_tests': tests['tests'], 'development_gate_sha256': gate_sha,
            'phase_cache_records': phase_records, 'original_full_raw_bootstrap_once_per_phase': True,
            'source_and_bound_file_bytes_and_membership_verified_on_every_logged_hit': True,
            'final_raw_audit_used_original_uncached_verification': True, 'amendment_after_development': True,
            'scope': 'Execution verification cache only; original model/controller/criteria/seeds/order and actual forecast cost remain unchanged.'}


def _operational_launch(resume, pause, preservation, launch, pause_path, preservation_path, supervisor_path, resume_path):
    require(resume.get('original_launch_sha256') == digest(OUT / 'logs/launch.json')
            and resume.get('pause_sha256') == digest(pause_path)
            and resume.get('preservation_sha256') == digest(preservation_path)
            and pause.get('source_commit') == SCIENTIFIC_SOURCE_COMMIT
            and pause.get('completed_confirmation_receipts') == {} and pause.get('selection_or_tuning') is False
            and preservation.get('scientific_source_commit') == SCIENTIFIC_SOURCE_COMMIT
            and preservation.get('registered_source_count') == EXPECTED_SOURCE_COUNT
            and preservation.get('tested_source_count') == EXPECTED_TESTED_SOURCE_COUNT
            and preservation.get('tested_source_receipt_sha256') == EXPECTED_TESTED_SOURCE_SHA256
            and preservation.get('all_tested_source_inputs_unchanged') is True
            and preservation.get('original_source_and_capsule_checks_pass') is True
            and preservation.get('no_scientific_code_or_test_change') is True
            and preservation.get('new_worlds_models_training_or_forwards_run_by_this_check') is False
            and preservation.get('completed_confirmation_receipts') == 0,
            'Operational pause/preservation/original-launch evidence changed')
    scope = {'module': 'wirelessjev.pilot_fastloop.study', 'phase_order': ['confirmation', 'report'],
             'workers': 2, 'thread_limits': {name: '1' for name in
                          ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
    require(all(resume.get(name) == value and launch.get(name) == value for name, value in scope.items())
            and launch.get('scientific_source_commit') == SCIENTIFIC_SOURCE_COMMIT
            and launch.get('operational_source_commit') == OPERATIONAL_SOURCE_COMMIT
            and launch.get('operational_resume_sha256') == digest(resume_path)
            and launch.get('wrapper_path') == resume['wrapper_path']
            and launch.get('wrapper_sha256') == OPERATIONAL_WRAPPER_SHA256
            and launch.get('resume_supervisor_sha256') == digest(supervisor_path),
            'Actual resumed wrapper/target/order/worker/thread launch differs')
    times = (pause.get('recorded_unix'), preservation.get('recorded_unix'),
             resume.get('recorded_before_resumed_confirmation_unix'),
             launch.get('recorded_before_first_resumed_phase_unix'))
    require(all(_finite(value) for value in times) and list(times) == sorted(times),
            'Operational pause/publication/resume chronology differs')


def preserve_prior():
    """Historical document hashes apply at their original published commits."""
    import verify_freshness_stack as prior
    result = prior.preserve_prior()
    name = 'artifacts/freshness_stack/verification/manifest.json'
    raw = committed(PREVIOUS_PUBLISHED_COMMIT, name)
    require(_file(name).read_bytes() == raw, 'Published PR21 manifest changed')
    manifest = json.loads(raw)
    require(len(manifest['sources_sha256']) == 376 and len(manifest['evidence_sha256']) == 665,
            'Published PR21 global source/evidence membership differs')
    counts = {}
    for field in ('sources_sha256', 'evidence_sha256'):
        _check_map(manifest[field], 'Previously published source/evidence changed: ')
        counts[field] = len(manifest[field])
    for path, expected in manifest['documents_sha256'].items():
        require(hashlib.sha256(committed(PREVIOUS_PUBLISHED_COMMIT, path)).hexdigest() == expected,
                'Previously published historical document changed: ' + path)
    result['freshness_stack'] = {'manifest_sha256': digest(_file(name)), 'preserved_entries': counts}
    return result


def registration(c):
    import verify_freshness_stack as prior
    require(SCIENTIFIC_SOURCE_COMMIT is not None and EXPECTED_SOURCE_COUNT is not None,
            'Pin actual published pilot/fast-loop commit and source count before final verification')
    parent, _, closures = prior.registration(c.parent)
    require(len(parent['source_sha256']) == 446, 'Original freshness closure differs')
    frozen = c.check_sources()
    meta = read(OUT / 'registration/git.json')
    commit = meta['scientific_source_commit']
    require(commit == SCIENTIFIC_SOURCE_COMMIT and re.fullmatch('[0-9a-f]{40}', commit)
            and meta['scientific_source_tree'] == _tree(commit) == SCIENTIFIC_SOURCE_TREE
            and meta['before_new_evaluations'] is True
            and meta['source_published_before_new_evaluations'] is True,
            'Pilot/fast-loop pre-data published Git metadata differs')
    require(len(frozen['source_sha256']) == EXPECTED_SOURCE_COUNT
            and frozen['new_evaluation_data_accessed'] is False
            and type(frozen['new_fits']) is int and frozen['new_fits'] == 0
            and frozen['parent_results_already_observed'] is True
            and frozen['source_training_and_results_already_observed'] is True,
            'Pilot/fast-loop registered source count/scope differs')
    name = 'artifacts/pilot_fastloop/registration/source.json'
    require(_file(name).read_bytes() == committed(commit, name),
            'Pilot/fast-loop registration differs from published Git')
    capsule = OUT / 'registration/source'
    actual = {str(path.relative_to(capsule)) for path in capsule.rglob('*') if path.is_file()}
    require(actual == set(frozen['source_sha256']), 'Registered capsule membership differs')
    for name, expected in frozen['source_sha256'].items():
        raw = committed(commit, name)
        require(digest(_file(name)) == expected == hashlib.sha256(raw).hexdigest()
                and (capsule / name).read_bytes() == raw,
                'Pilot/fast-loop current/capsule/published Git differs: ' + name)
    require(frozen['previous_source_and_evidence_preserved'] == c.verify_previous_evidence()
            and frozen['passive_diagnosis_source_inputs_and_outputs_verified'] == c.verify_diagnosis_inputs()
            and frozen['observed_input_sha256'] == {str(path.relative_to(ROOT)): digest(path) for path in c.prior_inputs()},
            'Registered observed prior/diagnosis bindings changed')
    names = _git_names(commit, 'artifacts')
    forbidden = tuple('artifacts/pilot_fastloop/' + folder + '/'
                      for folder in ('cases', 'replays', 'live_replications', 'matched'))
    require(not any(name.startswith(forbidden) for name in names),
            'Published pre-data commit contains new evaluations/caches')
    c.verify_fresh_sites()
    p = c.protocol()
    require(p['development_sites'] == [2153000, 2153001]
            and p['confirmation_sites'] == list(range(2154000, 2154008))
            and p['families'] == ['radiomap', 'frozen_fleet_jepa']
            and p['handover_seconds'] == [4]
            and p['variants_by_cost'] == {'4.0': ['base', 'pilot', 'fast', 'pilot_fast']}
            and p['robots'] == 100 and p['duration_seconds'] == 60 and p['new_fits'] == 0
            and p['primary_multiplicity_budget'] == 4 and p['primary_upper_quantile'] == .9875
            and p['bootstrap_replicates'] == 20000,
            'Pilot/fast-loop prospective identity or four-test budget differs')
    new = set(p['development_sites'] + p['confirmation_sites'])
    for name in names:
        if re.fullmatch(r'artifacts/[^/]+/protocol\.json', name) and name != 'artifacts/pilot_fastloop/protocol.json':
            require(not new & c._integers(json.loads(committed(commit, name))),
                    'Pilot/fast-loop sites overlap an earlier protocol')
    require(read(OUT / 'execution_order.json') == c.execution_order(), 'Native/matched execution order changed')
    return frozen, meta, dict(closures, freshness_stack=446)


def tests_check():
    require(all(value is not None for value in (EXPECTED_TESTS, EXPECTED_TEST_LOG_SHA256,
                EXPECTED_TESTED_SOURCE_SHA256, EXPECTED_TESTED_SOURCE_COUNT)),
            'Pin actual full preflight metadata before final verification')
    log = OUT / 'logs/preflight-tests.txt'
    matches = re.findall(r'(?m)^([0-9]+) passed(?:, ([0-9]+) warnings?)? in ([0-9.]+)s', log.read_text())
    require(len(matches) == 1, 'No unique full preflight summary')
    passed, warnings, seconds = matches[0]
    actual = int(passed), int(warnings or 0), float(seconds)
    require(actual == EXPECTED_TESTS and digest(log) == EXPECTED_TEST_LOG_SHA256,
            'Actual full preflight log differs')
    path = OUT / 'logs/tested-source-sha256.json'
    receipt = read(path)
    before, after = receipt['before_sha256'], receipt['after_sha256']
    require(digest(path) == EXPECTED_TESTED_SOURCE_SHA256
            and receipt['source_count'] == len(before) == len(after) == EXPECTED_TESTED_SOURCE_COUNT
            and before == after and receipt['unchanged'] is True
            and receipt['new_evaluation_data_accessed'] is False
            and receipt['tests'] == {'passed': actual[0], 'warnings': actual[1], 'seconds': actual[2]}
            and receipt['test_log_sha256'] == EXPECTED_TEST_LOG_SHA256,
            'Full before/after tested-source receipt differs')
    _check_map(before, 'Source/input changed since actual full preflight: ')
    launch = read(OUT / 'logs/launch.json')
    require(launch['scientific_source_commit'] == SCIENTIFIC_SOURCE_COMMIT
            and launch['workers'] == 2 and launch['module'] == 'wirelessjev.pilot_fastloop.study'
            and launch['phase_order'] == ['development', 'confirmation', 'report']
            and launch['thread_limits'] == {name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
            'Actual source/launch/worker/thread contract differs')
    return {'passed': actual[0], 'warnings': actual[1], 'seconds': actual[2],
            'test_log_sha256': digest(log), 'tested_source_receipt_sha256': digest(path),
            'tested_source_entries': len(before), 'launch_sha256': digest(OUT / 'logs/launch.json')}


def audit_check(meta, frozen):
    """Recompute the independent raw audit; never trust a saved success flag."""
    require(RAW_AUDIT.is_file(), 'Write the independent raw audit before the final manifest')
    auditor = importlib.import_module('audit_pilot_fastloop')
    before = RAW_AUDIT.read_bytes()
    expected = auditor.audit(write=False)
    require(RAW_AUDIT.read_bytes() == before, 'Passive raw audit rewrote its receipt')
    audit = read(RAW_AUDIT)
    require(expected == audit and audit['schema'] == 1
            and audit['scientific_source_commit'] == meta['scientific_source_commit']
            and audit['scientific_source_tree'] == meta['scientific_source_tree']
            and audit['registered_source_files'] == len(frozen['source_sha256'])
            and audit['report_sha256'] == digest(OUT / 'report.json')
            and audit['script_sha256'] == digest(ROOT / 'scripts/audit_pilot_fastloop.py')
            and all(audit.get(name) is True for name in AUDIT_PROOFS),
            'Independent raw audit identity/proof differs')
    return audit


def figures_check(report, audit):
    """Require the complete derived figure set and authenticate its provenance."""
    folder = OUT / 'figures'
    receipt_path = folder / 'receipt.json'
    require(receipt_path.is_file(), 'Generate all audited figures before the final manifest')
    receipt = read(receipt_path)
    expected_outputs = {str((folder / (stem + '.' + extension)).relative_to(ROOT))
                        for stem in FIGURE_STEMS for extension in ('svg', 'pdf', 'png')}
    expected_inputs = {str((OUT / name).relative_to(ROOT)) for name in FIGURE_INPUTS}
    require(receipt.get('schema') == 1 and receipt.get('derived_only') is True
            and receipt.get('scientific_execution') is False
            and receipt.get('new_fitting_or_selection') is False
            and receipt.get('secondary_components_descriptive_only') is True
            and receipt.get('no_physical_or_production_qualification') is True
            and receipt.get('independent_unit') == 'whole site'
            and receipt.get('script_sha256') == digest(ROOT / 'scripts/plot_pilot_fastloop.py')
            and receipt.get('report_sha256') == digest(OUT / 'report.json'),
            'Figure receipt source/report identity or derived-only scope differs')
    require(isinstance(receipt.get('inputs_sha256'), dict)
            and set(receipt['inputs_sha256']) == expected_inputs
            and isinstance(receipt.get('outputs_sha256'), dict)
            and set(receipt['outputs_sha256']) == expected_outputs,
            'Figure receipt must bind all eight inputs and exactly nine SVG/PDF/PNG outputs')
    _check_map(receipt['inputs_sha256'], 'Figure input changed: ')
    _check_map(receipt['outputs_sha256'], 'Figure output changed or missing: ')
    visible = {str(path.relative_to(ROOT)) for path in folder.rglob('*') if path.is_file()
               and not any(part.startswith('.') for part in path.relative_to(folder).parts)
               and not path.name.endswith(('.lock', '.part'))}
    require(visible == expected_outputs | {str(receipt_path.relative_to(ROOT))},
            'Figure output directory has missing or unreceipted files')
    require(receipt.get('primary_comparisons') == report['primary']
            and receipt.get('all_four_primary_supported') is report['all_four_primary_supported']
            and receipt.get('matched_forecast_advantage') is report['matched_forecast_advantage']
            and type(receipt.get('independent_sites')) is int
            and receipt.get('independent_sites') == report['independent_sites']
            and receipt.get('limits') == report['limits']
            and receipt.get('audit_flags') == {name: audit[name] for name in AUDIT_PROOFS}
            and all(receipt['audit_flags'].get(name) is True for name in AUDIT_PROOFS),
            'Figure receipt statistical values, limits or raw-audit proofs differ')
    return {'receipt_sha256': digest(receipt_path), 'plot_script_sha256': receipt['script_sha256'],
            'inputs_sha256': receipt['inputs_sha256'], 'outputs_sha256': receipt['outputs_sha256'],
            'derived_only': True, 'scientific_execution': False}


def _visible_evidence():
    """Bind all new data/results/caches/outputs; omit mutable lock/temp files."""
    paths = []
    for root in (OUT, ROOT / 'artifacts/jepa_advantage_diagnosis'):
        require(root.is_dir(), 'Missing new evidence directory: ' + str(root))
        for path in root.rglob('*'):
            if not path.is_file() or path == MANIFEST:
                continue
            parts = path.relative_to(root).parts
            if any(part.startswith('.') for part in parts) or path.name.endswith(('.lock', '.part')):
                continue
            paths.append(path)
    return paths


def _encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.read_bytes() == data, 'Immutable final manifest differs; refusing overwrite')
    finally:
        os.unlink(temporary)


def verify(*, write=False):
    require(type(write) is bool, 'Write mode must be an explicit boolean')
    require((OUT / 'report.json').is_file() and (OUT / 'report.md').is_file(),
            'Complete the registered study/report before final verification')
    from wirelessjev.pilot_fastloop import common as c
    frozen, meta, closures = registration(c)
    previous = preserve_prior()
    tests = tests_check()
    audit = audit_check(meta, frozen)
    operational = operational_check(frozen, audit)
    report = read(OUT / 'report.json')
    require(report['primary_multiplicity_budget'] == 4 and len(report['primary']) == 4
            and report['primary_upper_quantile'] == .9875 and report['new_fits'] == 0
            and all(report[name] is False for name in ('hardware_qualified', 'production_qualified', 'activated')),
            'Final report promoted scope or changed its four-primary budget')
    figures = figures_check(report, audit)
    sources = [path for folder in ('src', 'scripts', 'tests') for path in (ROOT / folder).rglob('*.py')
               if '__pycache__' not in path.parts]
    sources += [ROOT / 'pyproject.toml', ROOT / 'uv.lock']
    documents = [ROOT / name for name in DOCUMENTS]
    value = {'schema': 1, 'scientific_source_commit': meta['scientific_source_commit'],
             'scientific_source_tree': meta['scientific_source_tree'],
             'registration_precedes_only_new_development_and_confirmation': True,
             'registered_source_files': len(frozen['source_sha256']), 'original_source_closures': closures,
             'source_registration_sha256': digest(OUT / 'registration/source.json'),
             'prior_scientific_source_and_evidence': previous,
             'previous_global_document_hashes_apply_to_their_original_commits': True,
             'full_preflight': tests, 'raw_audit_sha256': digest(RAW_AUDIT),
             'operational_resume': operational,
             'raw_audit_script_sha256': digest(ROOT / 'scripts/audit_pilot_fastloop.py'),
             'raw_audit_proofs': {name: audit[name] for name in AUDIT_PROOFS},
             'raw_audit_limits': audit.get('limits', []), 'report_sha256': digest(OUT / 'report.json'),
             'derived_figures': figures,
             'development_sha256': digest(OUT / 'development.json'),
             'qualification_sha256': digest(OUT / 'qualification.json'),
             'completion_sha256': digest(OUT / 'completion.json'),
             'execution_order_sha256': digest(OUT / 'execution_order.json'),
             'primary_multiplicity_budget': 4, 'primary_upper_quantile': .9875,
             'primary': report['primary'], 'passed_development_costs': report['passed_development_costs'],
             'case_count': report['case_count'], 'matched_case_count': report['matched_case_count'],
             'independent_confirmation_sites': report['independent_sites'],
             'matched_independent_sites_added': 0, 'new_fits': 0,
             'all_three_packet_benefits': report['all_three_packet_benefits'],
             'matched_forecast_advantage': report['matched_forecast_advantage'],
             'all_four_primary_supported': report['all_four_primary_supported'],
             'sources_sha256': _hashes(sources), 'documents_sha256': _hashes(documents),
             'evidence_sha256': _hashes(_visible_evidence()),
             'hardware_qualified': False, 'production_qualified': False, 'activated': False,
             'new_worlds_models_forwards_training_or_timing_executed_by_verifier': False,
             'verifier_sha256': digest(Path(__file__))}
    if MANIFEST.exists():
        existing = read(MANIFEST)
        require(isinstance(existing.get('verified_at_utc'), str), 'Missing final verification timestamp')
        value['verified_at_utc'] = existing['verified_at_utc']
    elif write:
        value['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
    else:
        raise RuntimeError('Final manifest missing; use --write after final documents are complete')
    data = _encoded(value)
    if write:
        _write_once(MANIFEST, data)
    else:
        require(MANIFEST.read_bytes() == data, 'Final source/document/evidence manifest bytes differ')
    print(json.dumps({'registered_source_files': value['registered_source_files'],
                      'global_sources': len(value['sources_sha256']),
                      'global_documents': len(value['documents_sha256']),
                      'new_evidence_files': len(value['evidence_sha256']),
                      'case_count': value['case_count'], 'matched_case_count': value['matched_case_count'],
                      'all_four_primary_supported': value['all_four_primary_supported'],
                      'production_qualified': False}, indent=2))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Create the immutable manifest after final docs')
    args = parser.parse_args()
    verify(write=args.write)
