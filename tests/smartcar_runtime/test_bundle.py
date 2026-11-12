"""Build and inspect actual allowlisted archives; no credentials or car calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def builder():
    spec = importlib.util.spec_from_file_location("software_bundle_builder", ROOT / "scripts/build_smartcar_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_is_deterministic_contains_exact_published_weights_and_no_private_archive(tmp_path):
    module = builder()
    first, second = tmp_path/"a.zip", tmp_path/"b.zip"
    module.build(first)
    module.build(second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert all(name.startswith("smartcar-wirelessJEV/") for name in names)
        assert not any(Path(name).name in ("WiFi.json", "known_hosts", "SmartCarCmd.py", "SmartCatScan.py") for name in names)
        selection = json.loads((ROOT/"artifacts/nested_radio/selection.json").read_text())
        for seed in range(3):
            member = f"nested_joint-seed{seed}"
            data = archive.read(f"smartcar-wirelessJEV/models/{member}.pt")
            assert hashlib.sha256(data).hexdigest() == selection["checkpoints"][member]["sha256"]


def test_extracted_bundle_passes_stdlib_integrity_and_runs_its_own_entrypoint(tmp_path):
    path = tmp_path / "bundle.zip"
    builder().build(path)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp_path / "extracted")
    root = tmp_path / "extracted/smartcar-wirelessJEV"
    proof = subprocess.run([sys.executable, str(root/"verify_bundle.py")], capture_output=True, text=True, check=True)
    assert json.loads(proof.stdout)["passes"] is True
    help_result = subprocess.run([sys.executable, str(root/"PC/run.py"), "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0 and "fit-map" in help_result.stdout
    # Real source corruption must fail the extracted integrity check.
    with (root/"PI/wj_car_agent.py").open("a") as stream:
        stream.write("\n# software corruption test\n")
    corrupted = subprocess.run([sys.executable, str(root/"verify_bundle.py")], capture_output=True)
    assert corrupted.returncode != 0


def test_builder_refuses_to_replace_an_existing_archive(tmp_path):
    path = tmp_path / "bundle.zip"
    path.write_bytes(b"existing-user-file")
    with pytest.raises(FileExistsError):
        builder().build(path)
    assert path.read_bytes() == b"existing-user-file"
