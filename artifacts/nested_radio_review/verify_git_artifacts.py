"""Check exact archived study bytes in published Git HEAD; no scientific calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BRANCH = 'codex/nested-radio-shrinkage'
FOLDER = 'artifacts/nested_radio'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def verify():
    assert git('branch', '--show-current').decode().strip() == BRANCH
    assert git('rev-parse', '--show-object-format').decode().strip() == 'sha1'
    head = git('rev-parse', 'HEAD').decode().strip()
    remote_rows = git('ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH).decode().splitlines()
    assert len(remote_rows) == 1
    remote, remote_ref = remote_rows[0].split()
    assert remote_ref == 'refs/heads/' + BRANCH and remote == head

    # Check committed objects, rather than trusting potentially staged index bytes.
    entries = {}
    for record in git('ls-tree', '-r', '-z', 'HEAD', '--', FOLDER).split(b'\0'):
        if not record:
            continue
        info, name = record.split(b'\t', 1)
        mode, kind, blob = info.decode().split()
        assert mode in ('100644', '100755') and kind == 'blob', name
        entries[name.decode()] = blob

    folder = ROOT / FOLDER
    assert folder.resolve().is_relative_to(ROOT) and not folder.is_symlink()
    files = [p for p in folder.rglob('*') if p.is_file()]
    assert len(files) == len(entries) == 3349
    assert {str(p.relative_to(ROOT)) for p in files} == set(entries)
    sizes = 0
    for path in files:
        name = str(path.relative_to(ROOT))
        assert not path.is_symlink() and path.resolve().is_relative_to(folder), name
        size = path.stat().st_size
        digest = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        assert entries[name] == digest.hexdigest(), name
        sizes += size

    return {
        'schema': 1, 'passes': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'science_commit': head, 'remote_observed_commit': remote, 'branch': BRANCH,
        'study_files_in_Git_matching_exact_bytes': len(files), 'study_bytes': sizes,
        'committed_HEAD_tree_and_local_membership_equal': True,
        'scientific_calls': 0,
        'scope': 'Study archives including source capsule; final documentation commit can follow without changing these bytes.'}


if __name__ == '__main__':
    proof = verify()
    with (ROOT / 'artifacts/nested_radio_review/git_artifact_verification.json').open('x') as handle:
        json.dump(proof, handle, sort_keys=True, indent=2)
        handle.write('\n')
    print(json.dumps({
        'passes': True,
        'published_study_files': proof['study_files_in_Git_matching_exact_bytes'],
        'bytes': proof['study_bytes'], 'science_commit': proof['science_commit']}))
