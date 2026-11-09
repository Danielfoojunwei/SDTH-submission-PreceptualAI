"""Read-only profiling-ledger audit; never import or invoke either operator."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np

ROOT = Path('/workspace/SDTH-submission-PreceptualAI')
OUT = ROOT / 'artifacts/native_bridge_ops'
NATIVE = ROOT / 'artifacts/native_bridge'
DEST = ROOT / 'artifacts/native_bridge_ops_review/independent.json'
NATIVE_SHA = '4d9f95316d9e029365ed64d4b04e43128a49239c4115ebb8a6bb49f11c12aeb9'
HASHES = {}


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    path = Path(path)
    if path not in HASHES:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                value.update(block)
        HASHES[path] = value.hexdigest()
    return HASHES[path]


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def safe(name):
    value = Path(name)
    require(not value.is_absolute() and '..' not in value.parts, 'Unsafe relative reference')
    path = ROOT / value
    require(path.resolve().is_relative_to(ROOT.resolve()) and path.is_file(), 'Missing or escaped input')
    for item in (path, *path.parents):
        if item == ROOT:
            break
        require(not item.is_symlink(), 'Redirected source or evidence')
    return path


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def check_summary(actual, expected):
    require(set(actual) == set(expected), 'Summary membership differs')
    for name, value in expected.items():
        require(np.allclose(actual[name], value, rtol=1e-12, atol=1e-12), 'Timing summary differs: ' + name)


def main():
    p = read(OUT / 'protocol.json')
    registration = read(OUT / 'registration/source.json')
    publication = read(OUT / 'registration/git.json')
    report = read(OUT / 'report.json')
    attempt = read(OUT / 'attempt.json')
    before_report = sha(OUT / 'report.json')
    source = registration['source_sha256']
    require(len(source) == 5, 'Expected exact five-file operations source')
    for name, expected in source.items():
        require(sha(safe(name)) == expected, 'Published operations source changed')
    for name, expected in registration['preflight_sha256'].items():
        require(sha(safe(name)) == expected, 'Operations preflight changed')
    preflight = read(OUT / 'preflight/tests.json')
    require(preflight['exit_code'] == 0
            and preflight['source_sha256_before'] == preflight['source_sha256_after'] == source,
            'Passing preflight is not bound to actual sources')
    require(sha(OUT / 'protocol.json') == registration['protocol_sha256'] == report['protocol_sha256'],
            'Protocol identity differs')
    require(publication['commit'] == publication['remote_observed_commit']
            == '0bc5b4360d8c8023af6891b53374b4a576f26077', 'Wrong operations publication')
    require(publication['source_receipt_sha256'] == report['source_receipt_sha256']
            == attempt['source_receipt_sha256'] == sha(OUT / 'registration/source.json'),
            'Operations source receipt identity differs')
    require(report['publication_sha256'] == attempt['publication_sha256']
            == sha(OUT / 'registration/git.json'), 'Operations publication receipt differs')
    members = dict(source) | registration['preflight_sha256']
    members['artifacts/native_bridge_ops/registration/source.json'] = sha(OUT / 'registration/source.json')
    process = subprocess.Popen(['git', 'archive', '--format=tar', publication['commit'], *members],
                               cwd=ROOT, stdout=subprocess.PIPE)
    found = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if member.isdir():
                    continue
                require(member.isfile() and member.name in members and member.name not in found,
                        'Unexpected published operations member')
                require(hashlib.sha256(archive.extractfile(member).read()).hexdigest() == members[member.name],
                        'Git-published operations source bytes differ')
                found.add(member.name)
        require(process.wait() == 0 and found == set(members), 'Incomplete operations publication')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    receipt_commit = subprocess.check_output(['git', 'rev-parse', '7798184^{commit}'], cwd=ROOT, text=True).strip()
    receipt_bytes = subprocess.check_output(['git', 'show', receipt_commit + ':artifacts/native_bridge_ops/registration/git.json'], cwd=ROOT)
    require(hashlib.sha256(receipt_bytes).hexdigest() == sha(OUT / 'registration/git.json'),
            'Published observation receipt differs from its recorded commit')
    committed_seconds = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', receipt_commit], cwd=ROOT, text=True))
    require(type(attempt['started_unix_ns']) is int and attempt['started_unix_ns'] >= committed_seconds * 10**9,
            'Profiling attempt predates its published observation receipt')

    frozen = NATIVE / 'verification/manifest.json'
    require(sha(frozen) == NATIVE_SHA, 'Frozen native manifest changed')
    native = read(frozen)
    require(len(native['source_sha256']) == 512 and len(native['evidence_sha256']) == 56,
            'Native source/evidence membership differs')
    for field in ('source_sha256', 'evidence_sha256'):
        for name, expected in native[field].items():
            require(sha(safe(name)) == expected, 'Frozen native source/evidence bytes differ')
    visible = {str(path.relative_to(ROOT)) for path in NATIVE.rglob('*')
               if path.is_file() and path != frozen and not path.is_relative_to(NATIVE / 'registration/source')}
    require(visible == set(native['evidence_sha256']), 'Frozen native evidence membership changed')
    for field in ('frozen_native_manifest_sha256_before', 'frozen_native_manifest_sha256_after'):
        require(report[field] == NATIVE_SHA, 'Profiling did not retain frozen-native identity')
    require(registration['frozen_native_manifest_sha256'] == NATIVE_SHA
            and registration['actual_plan_arrays_opened_before_registration'] is False
            and registration['timing_calls_before_registration'] == 0,
            'Operations preregistration scope differs')
    require(report['passes'] is True and report['exact_operand_byte_parity_every_plan_and_timed_pair'] is True,
            'Recorded runtime byte-parity execution gate failed')
    for key in ('new_worlds', 'new_native_cases', 'model_forwards', 'fits'):
        require(report[key] == 0, 'Profiling reports forbidden scientific execution')
    for key in ('native_runner_changed', 'SLA_claim', 'service_improvement_claim'):
        require(report[key] is False, 'Operations claim exceeds registered scope')
    require(attempt['finite_public_plan_inputs'] == 48 and attempt['timed_calls_per_operator'] == 144
            and attempt['new_native_cases'] == attempt['model_forwards'] == 0,
            'Finite attempt budget differs')
    require(p['timed_paired_rounds_per_plan'] == 3 and p['warmup_calls_per_operator_per_plan'] == 1,
            'Finite timing design differs')
    require(report['geometry_work'] == p['geometry_work']
            == {'full_queries_per_call': 43200, 'rack_boxes': 33,
                'reference_AABB_links_per_call': 43200, 'reuse_AABB_links_per_call': 14400},
            'Geometry work accounting differs')
    require(report['limitations'] == p['limitations']
            and report['environment']['thread_limits'] == p['thread_limits'], 'Scope or numerical thread configuration differs')

    expected_inputs = []
    expected_pairs = []
    for seed_index, (seed, entry) in enumerate(zip((90473001, 90473002, 90473003, 90473004), p['inputs'])):
        require(entry['case_id'] == f'dev-{seed}-ho0.03', 'Selected a different profiling episode')
        require(entry['plan_keys'] == [f'job{i}_plan' for i in range(12)], 'Plan membership differs')
        for name, expected_path in (
            ('candidate_archive', f'artifacts/native_bridge/cases/dev-{seed}-ho0.03/candidate_inputs_forecasts.npz'),
            ('public_geometry_archive', f'artifacts/native_bridge/worlds/{seed}/world.npz')):
            ref = entry[name]
            require(ref['path'] == expected_path
                    and ref['sha256'] == native['evidence_sha256'][expected_path]
                    and sha(safe(expected_path)) == ref['sha256'], 'Canonical public input archive differs')
        # Decode only the three declared public arrays, never any RF array.
        with np.load(safe(entry['public_geometry_archive']['path']), allow_pickle=False) as saved:
            sites = np.asarray(saved['sites'], dtype=float)
            tx = np.asarray(saved['tx_dl'], dtype=float)
            noise = np.asarray(saved['noise_db'], dtype=float)
        require(sites.shape == (16, 2) and tx.shape == noise.shape == (3,)
                and all(np.isfinite(a).all() for a in (sites, tx, noise)), 'Malformed public geometry')
        with np.load(safe(entry['candidate_archive']['path']), allow_pickle=False) as saved:
            for plan_index, key in enumerate(entry['plan_keys']):
                plan = saved[key].copy()
                require(plan.shape == (100, 9, 2) and np.isfinite(plan).all(), 'Malformed public planned positions')
                expected_inputs.append({'case_id': entry['case_id'], 'plan_key': key,
                    'plan_sha256': array_sha(plan), 'sites_sha256': array_sha(sites),
                    'tx_sha256': array_sha(tx), 'noise_sha256': array_sha(noise), 'plan_shape': [100, 9, 2]})
                ordinal = seed_index * 12 + plan_index
                for trial in range(3):
                    expected_pairs.append((entry['case_id'], key, trial,
                        ['reference', 'reuse'] if (ordinal + trial) % 2 == 0 else ['reuse', 'reference']))
    require(len(p['inputs']) == 4 and report['public_inputs'] == expected_inputs and len(expected_inputs) == 48,
            'Actual public plan/geometry hashes or membership differ')
    rows = report['timed_pairs']
    require(len(rows) == len(expected_pairs) == 144, 'Timed pair membership differs')
    first_method = {'reference': 0, 'reuse': 0}
    for row, expected in zip(rows, expected_pairs):
        require(set(row) == {'case_id', 'plan_key', 'round', 'call_order', 'reference_ns', 'reuse_ns'},
                'Unexpected timing ledger fields')
        require((row['case_id'], row['plan_key'], row['round'], row['call_order']) == expected,
                'Timed plan/round/order differs')
        require(type(row['reference_ns']) is int and type(row['reuse_ns']) is int
                and min(row['reference_ns'], row['reuse_ns']) > 0, 'Nonpositive or noninteger timing')
        first_method[row['call_order'][0]] += 1
    require(first_method == {'reference': 72, 'reuse': 72}, 'Paired order is unbalanced')
    a = np.asarray([row['reference_ns'] for row in rows], dtype=float)
    b = np.asarray([row['reuse_ns'] for row in rows], dtype=float)
    computed = {'reference_median_ms': float(np.median(a) / 1e6),
                'reuse_median_ms': float(np.median(b) / 1e6),
                'ratio_of_median_call_times': float(np.median(a) / np.median(b)),
                'median_paired_speedup': float(np.median(a / b)),
                'reference_min_max_ms': [float(a.min() / 1e6), float(a.max() / 1e6)],
                'reuse_min_max_ms': [float(b.min() / 1e6), float(b.max() / 1e6)],
                'timed_calls_per_operator': 144, 'warmup_calls_per_operator': 48}
    check_summary(report['summary'], computed)
    owner = read(ROOT / 'artifacts/native_bridge_ops_review/review.json')
    require(owner['summary'] == report['summary'], 'Owner review and independent arithmetic disagree')
    for path, expected in HASHES.items():
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Input changed during passive review')
    result = {'passes': True, 'scope': 'independent published-source, public-input and raw-timing ledger reconciliation',
        'registered_source_files': 5, 'preserved_native_sources': 512, 'preserved_native_evidence': 56,
        'source_publication': publication['commit'], 'publication_receipt_commit': receipt_commit,
        'public_plans_rehashed': 48, 'timed_pairs_reconciled': 144, 'first_method_counts': first_method,
        'summary': computed, 'frozen_native_manifest_sha256': NATIVE_SHA,
        'report_sha256_before': before_report, 'report_sha256_after': sha(OUT / 'report.json'),
        'extra_operator_calls': 0, 'extra_timing_calls': 0, 'models_fits_forwards_or_packet_simulations': 0,
        'byte_parity_evidence': 'Exact runtime byte assertions are bound to published source and passing original execution; output operands were not archived or independently regenerated.',
        'independent_numeric_operand_replay': False,
        'qualifications': ['Descriptive operator-only speed gain on these48 existing public plans in one managed CPU environment.',
            'CPU was shared with other adaptive/full-suite work as recorded in the owner review; this was not an isolated-machine benchmark.',
            'Repeated calls and plans are correlated; no iid confidence or deployment claim.',
            'No full predictor-head, queue, radio, control-miss, SLA or service speed improvement was measured.'],
        'co_running_cpu_context': owner['co_running_cpu_context'],
        'audit_script_sha256': sha(Path(__file__)),
        'input_files_sha256': {str(path.relative_to(ROOT)): digest for path, digest in HASHES.items() if path.is_relative_to(ROOT)}}
    require(not DEST.exists(), 'Independent review already exists; preserve original bytes')
    DEST.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'passes': True, 'summary': computed, 'record': str(DEST), 'sha256': sha(DEST)}, indent=2))


if __name__ == '__main__':
    main()
