"""Acquire real, pinned upstream simulators without adding nested Git repos."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

UPSTREAMS = {
    "sionna": ("NVlabs/sionna", "2f0a3820269b53fd022780a6f2f5f720f7ceb9b4", "Apache-2.0"),
    "ns3": ("nsnam/ns-3-dev-git", "418210e8914e9f6bcb5cddbcc94a2cecb7853b68", "GPL-2.0-only; per-file notices apply"),
}
ARCHIVE_SHA256 = {
    'sionna': 'ffed5d0e909830fe52f8c770b568d39bde0ae1b191eba4935daa3604888b44ce',
    'ns3': '37ef24b3ed834eae4e4b17c1ef9fa4c67ea1dbbe70032d3dde2b6e53828cbcc0',
}


def acquire(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, commit, licence) in UPSTREAMS.items():
        dest = root / name
        receipt = root / f"{name}.json"
        if receipt.exists() and dest.is_dir():
            item = json.loads(receipt.read_text())
            if item["commit"] != commit or item['archive_sha256'] != ARCHIVE_SHA256[name]:
                raise RuntimeError(f"Refusing to reuse unpinned {name}")
        else:
            if dest.exists():
                raise RuntimeError(f"Unverified upstream directory: {dest}")
            url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
            print(f"Downloading {repo}@{commit}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                blob = response.read()
            if hashlib.sha256(blob).hexdigest() != ARCHIVE_SHA256[name]:
                raise RuntimeError(f'Pinned archive digest mismatch: {name}')
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
                members = archive.getmembers()
                prefix = members[0].name.split('/')[0]
                archive.extractall(root, filter="data")
            (root / prefix).rename(dest)
            item = {"repository": f"https://github.com/{repo}", "commit": commit,
                    "archive_sha256": hashlib.sha256(blob).hexdigest(),
                    "licence": licence, "path": str(dest)}
            receipt.write_text(json.dumps(item, indent=2) + "\n")
        manifest[name] = item
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/workspace/upstream/warehouse-v2'))
    parser.add_argument('--manifest', type=Path, default=Path('/workspace/.cloud-setup/warehouse-v2-upstreams.json'))
    parser.add_argument('--build', action='store_true', help='Configure and compile both adapted native ns-3 programs')
    args = parser.parse_args()
    manifest = acquire(args.root)
    if args.build:
        repo = Path(__file__).resolve().parents[1]
        ns3 = args.root/'ns3'
        env = dict(os.environ, PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH', ''))
        for name in ['warehouse-wifi', 'warehouse-link-curves']:
            shutil.copyfile(repo/'third_party/ns3'/f'{name}.cc', ns3/'scratch'/f'{name}.cc')
        subprocess.run([sys.executable, 'ns3', 'configure', '--enable-modules=wifi,internet,mobility,applications,flow-monitor',
                        '--disable-examples', '--disable-tests', '--build-profile=optimized', '--out=build-warehouse'],
                       cwd=ns3, env=env, check=True)
        manifest['adaptations'] = {}
        for name in ['warehouse-wifi', 'warehouse-link-curves']:
            subprocess.run([sys.executable, 'ns3', 'build', name, '-j', '2'], cwd=ns3, env=env, check=True)
            binary = ns3/'build-warehouse/scratch'/f'ns3.49-{name}-optimized'
            manifest['adaptations'][name] = {
                'source_sha256': hashlib.sha256((repo/'third_party/ns3'/f'{name}.cc').read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'binary': str(binary)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
