"""Execute the prospective horizon correction with explicit source provenance."""
from pathlib import Path
import argparse
import json
import shutil
from datetime import datetime, timezone
import torch

from wirelessjev.warehouse_v2.data import build_dataset
from wirelessjev.warehouse_v2.experiment import atomic_json, sha256
from wirelessjev.warehouse_v2.factorized import study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out/'core_run.json', {'path': str(args.core_run.resolve())})
    repo = Path(__file__).resolve().parents[1]
    files = [*repo.glob('src/wirelessjev/**/*.py'), Path(__file__)]
    hashes = {}
    for file in files:
        relative = file.relative_to(repo)
        dest = args.out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) != sha256(file):
            raise RuntimeError('Use a fresh output directory for changed factorized source')
        shutil.copyfile(file, dest)
        hashes[str(relative)] = sha256(file)
    atomic_json(args.out/'provenance.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'source_sha256': hashes, 'source': 'simulation', 'core_provenance_sha256': sha256(args.core_run/'provenance.json')})
    atomic_json(args.out/'status.json', {'state': 'running'})
    try:
        data, _ = build_dataset(args.core_run/'cache'/'observations')
        atomic_json(args.out/'factorized.json', study(args.out, data, [0, 1, 2]))
    except BaseException as error:
        atomic_json(args.out/'status.json', {'state': 'failed', 'error': str(error)})
        raise
    atomic_json(args.out/'status.json', {'state': 'completed', 'report_sha256': sha256(args.out/'factorized.json')})


if __name__ == '__main__':
    main()
