"""Exact released-state conversion and hostile inert-archive intake checks."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from wirelessjev.smartcar_numpy import export as ex


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "artifacts/nested_radio/checkpoints"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("exact-radiojepa-exports")
    manifest = ex.export_release(CHECKPOINTS, folder)
    return folder, manifest


def _member(exported, seed=0):
    folder, manifest = exported
    row = manifest["members"][seed]
    return folder / row["path"], row


def _read(exported, seed=0):
    path, row = _member(exported, seed)
    return ex.load_export(path, row["sha256"], row["source_checkpoint_sha256"])


def _repack(path, replacements=None, *, omit=None, duplicate=None, extra=None,
            compression=zipfile.ZIP_STORED, comment=b""):
    replacements = replacements or {}
    result = io.BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(result, "w", compression=compression) as target:
        for name in source.namelist():
            if name != omit:
                target.writestr(name, replacements.get(name, source.read(name)))
        if duplicate:
            with pytest.warns(UserWarning, match="Duplicate name"):
                target.writestr(duplicate, source.read(duplicate))
        if extra:
            target.writestr(extra, b"unexpected")
        target.comment = comment
    return result.getvalue()


def _load_bad(tmp_path, payload, source_sha=None):
    path = tmp_path / "bad.npz"
    path.write_bytes(payload)
    return ex.load_export(path, hashlib.sha256(payload).hexdigest(),
                          source_sha or ex.RELEASED_CHECKPOINTS["nested_joint-seed0"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exact_full_state_roundtrip_including_scalar_and_dormant_weights(exported, seed):
    import torch

    arrays, metadata = _read(exported, seed)
    checkpoint = torch.load(CHECKPOINTS / f"nested_joint-seed{seed}.pt", map_location="cpu", weights_only=True)
    assert set(arrays) == set(checkpoint["state"])
    for name, tensor in checkpoint["state"].items():
        original = tensor.detach().cpu().numpy()
        assert arrays[name].dtype == original.dtype == np.float32
        assert arrays[name].shape == original.shape
        assert arrays[name].tobytes() == original.tobytes()
        assert not arrays[name].flags.writeable
    assert arrays["raw_discrepancy"].shape == ()
    assert metadata["seed"] == seed
    assert metadata["settings"] == checkpoint["settings"]
    assert metadata["source_checkpoint_sha256"] == ex.RELEASED_CHECKPOINTS[f"nested_joint-seed{seed}"]
    assert metadata["state_sha256"] == ex.RELEASED_STATES[seed]
    assert np.array_equal(arrays["local_gates"][8:], np.zeros(8, np.float32))
    assert np.count_nonzero(arrays["local_gates"][:8]) == (5, 5, 6)[seed]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_export_bytes_deterministic_and_existing_output_never_overwritten(exported, tmp_path, seed):
    source = CHECKPOINTS / f"nested_joint-seed{seed}.pt"
    second = tmp_path / "different-name.npz"
    row = ex.export_checkpoint(source, second)
    first, original = _member(exported, seed)
    assert second.read_bytes() == first.read_bytes()
    assert row["sha256"] == original["sha256"]
    before = second.read_bytes()
    with pytest.raises(FileExistsError):
        ex.export_checkpoint(source, second)
    assert second.read_bytes() == before


def test_manifest_binds_exactly_three_members_and_refuses_nonempty_destination(exported):
    folder, manifest = exported
    assert manifest["seeds"] == [0, 1, 2]
    assert manifest["runtime_torch_required"] is False
    assert manifest["scientific_fit_or_prediction_performed"] is False
    assert {r["name"] for r in manifest["members"]} == set(ex.RELEASED_CHECKPOINTS)
    assert json.loads((folder / "manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="new or empty"):
        ex.export_release(CHECKPOINTS, folder)


def test_loader_in_clean_process_rejects_any_torch_or_huggingface_import(exported):
    path, member = _member(exported)
    program = """
import importlib.abc,sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','huggingface_hub','safetensors'}:
            raise AssertionError('Forbidden runtime import: '+fullname)
sys.meta_path.insert(0,Forbidden())
from wirelessjev.smartcar_numpy.export import load_export
arrays,meta=load_export(sys.argv[1],sys.argv[2],sys.argv[3])
assert len(arrays)==18 and meta['seed']==0 and meta['export_sha256']==sys.argv[2]
assert not any(name.split('.')[0] in {'torch','transformers','huggingface_hub'} for name in sys.modules)
print('inert runtime PASS')
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run([sys.executable, "-c", program, str(path), member["sha256"],
                             member["source_checkpoint_sha256"]], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "inert runtime PASS"


def test_export_rejects_unknown_source_before_importing_torch(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("Untrusted checkpoint reached deserialization")
        return original(name, *args, **kwargs)

    source = tmp_path / "untrusted.pt"
    source.write_bytes(b"untrusted checkpoint")
    monkeypatch.setattr(builtins, "__import__", checked)
    with pytest.raises(ValueError, match="exact released"):
        ex.export_checkpoint(source, tmp_path / "out.npz")
    assert not (tmp_path / "out.npz").exists()


def test_loader_requires_matching_export_and_released_source_hash(exported):
    path, member = _member(exported)
    with pytest.raises(ValueError, match="Export SHA"):
        ex.load_export(path, "0" * 64, member["source_checkpoint_sha256"])
    with pytest.raises(ValueError, match="released source"):
        ex.load_export(path, member["sha256"], "0" * 64)
    with pytest.raises(ValueError, match="exact released source"):
        ex.load_export(path, member["sha256"], ex.RELEASED_CHECKPOINTS["nested_joint-seed1"])


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "compressed", "comment"])
def test_zip_membership_and_compression_are_checked_before_numpy_load(exported, tmp_path, monkeypatch, change):
    path, _ = _member(exported)
    kwargs = {"missing": {"omit": "local_gates.npy"}, "extra": {"extra": "../escape.npy"},
              "duplicate": {"duplicate": "local_gates.npy"},
              "compressed": {"compression": zipfile.ZIP_DEFLATED}, "comment": {"comment": b"unexpected"}}[change]
    payload = _repack(path, **kwargs)
    monkeypatch.setattr(ex.np, "load", lambda *a, **k: pytest.fail("ZIP preflight must precede np.load"))
    with pytest.raises(ValueError):
        _load_bad(tmp_path, payload)


@pytest.mark.parametrize("kind", ["object", "float64", "huge_shape", "trailing_bytes"])
def test_npy_header_and_payload_bounds_precede_array_loading(exported, tmp_path, monkeypatch, kind):
    path, _ = _member(exported)
    stream = io.BytesIO()
    if kind == "object":
        np.save(stream, np.array(["untrusted"] * 8, dtype=object), allow_pickle=True)
    elif kind == "float64":
        np.save(stream, np.zeros(8, dtype=np.float64), allow_pickle=False)
    elif kind == "huge_shape":
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": False, "shape": (2**60,)})
    else:
        np.save(stream, np.zeros(8, dtype=np.float32), allow_pickle=False)
        stream.write(b"unaccounted trailing bytes")
    payload = _repack(path, {"global_prior_mean.npy": stream.getvalue()})
    monkeypatch.setattr(ex.np, "load", lambda *a, **k: pytest.fail("NPY preflight must precede np.load"))
    with pytest.raises(ValueError):
        _load_bad(tmp_path, payload)


@pytest.mark.parametrize("key", ["global_prior_mean", "kernel.0.weight", "rff_phase", "local_gates"])
def test_rewritten_archive_and_self_declared_hashes_cannot_change_released_weights(exported, tmp_path, key):
    arrays, metadata = _read(exported)
    arrays = {name: value.copy() for name, value in arrays.items()}
    value = arrays[key]
    flat = value.reshape(-1)
    flat[0] = np.nextafter(flat[0], np.float32(np.inf))
    metadata.pop("export_sha256")
    # The attacker updates all per-array checksums and the outer NPZ checksum.
    # The pinned full-state identity still rejects the modified source state.
    metadata["arrays"] = ex._arrays_metadata(arrays)
    payload = ex._archive(arrays, metadata)
    with pytest.raises(ValueError, match="exact released source state"):
        _load_bad(tmp_path, payload)


@pytest.mark.parametrize("field,value", [("seed", True), ("schema", True), ("cell", "nested_reg"),
                                         ("source_receipt_sha256", "0" * 64), ("state_sha256", "0" * 64),
                                         ("settings", {"cell": "nested_joint", "width": 32.0,
                                                        "inference_steps": 3})])
def test_metadata_identity_cannot_relabel_valid_arrays(exported, tmp_path, field, value):
    arrays, metadata = _read(exported)
    metadata.pop("export_sha256")
    metadata[field] = value
    with pytest.raises(ValueError, match="exact released source"):
        _load_bad(tmp_path, ex._archive(arrays, metadata))


@pytest.mark.parametrize("encoded", [b'{"schema":1,"schema":1}', b'{"schema":NaN}', b'\xff'])
def test_metadata_rejects_duplicate_nonfinite_or_non_utf8_json(exported, tmp_path, encoded):
    path, _ = _member(exported)
    raw = ex._npy(np.frombuffer(encoded, dtype=np.uint8))
    payload = _repack(path, {"__metadata__.npy": raw})
    with pytest.raises(ValueError):
        _load_bad(tmp_path, payload)


def test_nonregular_and_oversize_inputs_refused(exported, tmp_path):
    path, member = _member(exported)
    link = tmp_path / "linked.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        ex.load_export(link, member["sha256"], member["source_checkpoint_sha256"])
    huge = tmp_path / "oversize.npz"
    with huge.open("wb") as stream:
        stream.truncate(ex.MAX_EXPORT_BYTES + 1)
    with pytest.raises(ValueError, match="bounded"):
        ex.load_export(huge, "0" * 64, member["source_checkpoint_sha256"])
