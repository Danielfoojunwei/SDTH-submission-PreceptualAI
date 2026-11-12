#!/usr/bin/env python3
"""Build an allowlisted portable deployment ZIP, excluding all user credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (
    "wirelessjev/__init__.py",
    "wirelessjev/smartcar_runtime/__init__.py",
    "wirelessjev/smartcar_runtime/contracts.py",
    "wirelessjev/smartcar_runtime/workflow.py",
    "wirelessjev/smartcar_runtime/control.py",
    "wirelessjev/smartcar_runtime/field.py",
    "wirelessjev/nested_radio/__init__.py",
    "wirelessjev/nested_radio/model.py",
    "wirelessjev/hierarchical_radio/__init__.py",
    "wirelessjev/hierarchical_radio/model.py",
    "wirelessjev/spectral_scene/__init__.py",
    "wirelessjev/spectral_scene/operators.py",
    "wirelessjev/radiojepa/__init__.py",
    "wirelessjev/radiojepa/likelihoods.py",
    "wirelessjev/radiojepa/operators.py",
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def build(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("A published bundle cannot be overwritten")
    members = {"src/" + name: (ROOT / "src" / name).read_bytes() for name in COMPONENTS}
    portable = ROOT / "deployment/smartcar"
    for name in ("README.md", "PI/wj_car_agent.py", "PC/run.py", "verify_bundle.py"):
        members[name] = (portable / name).read_bytes()
    members["docs/SMARTCAR_DEPLOYMENT.md"] = (ROOT / "docs/SMARTCAR_DEPLOYMENT.md").read_bytes()
    members["PC/requirements-map.txt"] = b"numpy==2.5.4\nscipy==1.18.1\n"
    members["PC/requirements-radiojepa.txt"] = b"-r requirements-map.txt\ntorch==2.14.1\n"
    members["site.template.json"] = encoded({"schema": 1, "site_id": "replace-with-your-site",
        "dimensions_m": [None, None, None], "aps": [], "cad_boxes_m": [],
        "receiver": {"noise_floor_dbm": None, "offset_db": 0., "quantization_db": 0.,
                     "reference_gauge_assumed": False}})
    selection_path = ROOT / "artifacts/nested_radio/selection.json"
    selection = json.loads(selection_path.read_bytes())
    weights = []
    for seed in (0, 1, 2):
        name = f"nested_joint-seed{seed}"
        reference = selection["checkpoints"][name]
        data = (ROOT / reference["path"]).read_bytes()
        if digest(data) != reference["sha256"]:
            raise RuntimeError("A frozen selected checkpoint changed")
        path = name + ".pt"
        members["models/" + path] = data
        weights.append({"name": name, "path": path, "sha256": digest(data), "seed": seed})
    members["models/manifest.json"] = encoded({"schema": 1, "cell": "nested_joint",
        "experimental_shadow_only": True, "hardware_transfer_qualified": False,
        "selection_source_path": str(selection_path.relative_to(ROOT)),
        "selection_source_sha256": digest(selection_path.read_bytes()), "members": weights,
        "note": "Actual published weights. Do not select members using local survey outcomes."})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"schema": 1, "source_commit": revision,
        "files_sha256": {name: digest(data) for name, data in sorted(members.items())},
        "scope": "Car collection and desktop numerical mapping; RadioJEPA shadow only",
        "credentials_included": False, "vendor_SmartCar_scripts_included": False,
        "actual_hardware_validation_completed": False, "SOTA_established": False,
        "provenance": "Hash-bound integrity, not a cryptographic signature"}
    members["MANIFEST.json"] = encoded(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            entry = zipfile.ZipInfo("smartcar-wirelessJEV/" + name, (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100644 << 16)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    proof = {"schema": 1, "archive": str(destination), "archive_sha256": digest(destination.read_bytes()),
        "archive_bytes": destination.stat().st_size, "bundle_members": len(members),
        "source_commit": revision, "source_files_sha256": manifest["files_sha256"],
        "hardware_calls": 0, "scientific_training_or_RT_calls": 0,
        "credentials_or_private_vendor_archive_included": False}
    with destination.with_suffix(".receipt.json").open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {key: proof[key] for key in ("archive", "archive_sha256", "archive_bytes", "bundle_members", "source_commit")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    print(json.dumps(build(parser.parse_args().out), indent=2))
