#!/usr/bin/env python3
"""Verify pristine extracted ZIP bytes with Python stdlib before installation."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_bytes())
    expected = manifest["files_sha256"]
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe bundle manifest path")
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("Missing or unsafe bundle member")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("A bundle file differs from the manifest: " + name)
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != set(expected) | {"MANIFEST.json"}:
        raise ValueError("Unexpected files in the pristine bundle; verify before creating runtime data")
    return {"passes": True, "files_checked": len(expected), "source_commit": manifest["source_commit"],
            "hardware_validation": False, "integrity_is_not_external_attestation": True}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
