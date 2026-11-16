"""Inert, deterministic exports of exactly three released RadioJEPA checkpoints.

Only ``export_checkpoint`` imports Torch, after authenticating the source bytes.
The runtime reader checks ZIP/NPY bounds before using ``allow_pickle=False``.
Neither conversion nor loading fits, evaluates or selects a scientific model.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import zipfile

import numpy as np


FORMAT = "wirelessjev.smartcar_numpy.weights.v1"
SETTINGS = {"cell": "nested_joint", "width": 32, "inference_steps": 3}
SOURCE_RECEIPT_SHA256 = "a9a1fa1670ab5c6801cb828d1e059d1ff25e09b15720252e8a73174bfadbd06b"
PROTOCOL_SHA256 = "cdf2e5af7875f640324a0c636305135eb167e22c572dc01c78b95ecfbb9ac696"
DATA_RECEIPT_SHA256 = "5026f21b97eea6662b602325b56f3a7ec922907e8b1cad5a25ec7a32a23a6598"
RELEASED_CHECKPOINTS = {
    "nested_joint-seed0": "57c710af2db4f4bd7abfd031a03a76359c27100568d3ebe277ab3c738a69c3bf",
    "nested_joint-seed1": "2de1c28d4ea41298cac58ca64cabe6d86a8c76688646a3dd4f479ab6686a9b52",
    "nested_joint-seed2": "16571ca0adb781349e1988fae74a648caf06c76c13c1cf5f605b0e5a4fee974c",
}
# Fingerprints cover every exact key, dtype, shape and C-order byte sequence.
# Derived from the authenticated source PTs, including unused kernel buffers.
RELEASED_STATES = {
    0: "6b501b9c3b2eb492d4f7b5e5db54f7da59d4179de2e4979ad671673ba9f60580",
    1: "62eeabbc9afc49f00f750305a1378560bbc5d2d77c20b5b4398cb9852be3b8b4",
    2: "be6f70314bdd3fc9f5987e4c31e1977c3441e3ab182010df630c4b746f3622e5",
}
_ACTIVE_FOURIER = {0: (0, 2, 4, 5, 6), 1: (0, 2, 5, 6, 7), 2: (0, 2, 3, 4, 5, 6)}
MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024
MAX_EXPORT_BYTES = 256 * 1024
MAX_MEMBER_BYTES = 32 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024
MAX_METADATA_BYTES = 16 * 1024
_METADATA_KEY = "__metadata__"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("A lowercase SHA-256 is required")
    return value


def _read_regular(path, limit):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("A regular non-symlink file is required")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("A bounded nonempty regular file is required")
        payload = stream.read(limit + 1)
    if len(payload) != info.st_size or len(payload) > limit:
        raise ValueError("File size changed or exceeded its bound")
    return payload


def _source(path):
    payload = _read_regular(path, MAX_CHECKPOINT_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    for seed in range(3):
        if digest == RELEASED_CHECKPOINTS[f"nested_joint-seed{seed}"]:
            return payload, digest, seed
    raise ValueError("Only the three exact released nested_joint checkpoints may be exported")


def state_sha256(arrays):
    """Canonical state identity; does not transpose, scale or absorb gates."""
    result = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        header = _canonical({"name": name, "dtype": array.dtype.str, "shape": list(array.shape)})
        result.update(len(header).to_bytes(8, "big"))
        result.update(header)
        result.update(array.tobytes(order="C"))
    return result.hexdigest()


def _arrays_metadata(arrays):
    return {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                   "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
            for name, value in sorted(arrays.items())}


def _validate_state(arrays, seed):
    from .model import validate_arrays

    values = validate_arrays(arrays)
    if any(value.dtype.str != "<f4" for value in values.values()):
        raise ValueError("Preserve every source array as little-endian float32")
    gates = values["local_gates"]
    if (not np.array_equal(gates[8:], np.zeros(8, dtype=np.float32))
            or tuple(np.flatnonzero(gates[:8] > 0)) != _ACTIVE_FOURIER[seed]):
        raise ValueError("Released seed-specific fixed/learned feature gates changed")
    if state_sha256(values) != RELEASED_STATES[seed]:
        raise ValueError("Exported arrays differ from the exact released source state")
    return values


def _npy(array):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _archive(arrays, metadata):
    members = {name: value for name, value in arrays.items()}
    members[_METADATA_KEY] = np.frombuffer(_canonical(metadata), dtype=np.uint8)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy(members[name]))
    return stream.getvalue()


def _write_exclusive(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def export_checkpoint(checkpoint_path, output_path):
    """Convert authenticated PT bytes once; Torch is a build-time dependency."""
    payload, source_digest, seed = _source(checkpoint_path)
    import torch

    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if (not isinstance(checkpoint, dict) or checkpoint.get("cell") != "nested_joint"
            or type(checkpoint.get("seed")) is not int or checkpoint["seed"] != seed
            or checkpoint.get("settings") != SETTINGS
            or checkpoint.get("identity") != {"source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                                               "protocol_sha256": PROTOCOL_SHA256,
                                               "data_receipt_sha256": DATA_RECEIPT_SHA256}
            or not isinstance(checkpoint.get("state"), dict)):
        raise ValueError("Source checkpoint identity/settings are inconsistent")
    arrays = {}
    for name, tensor in checkpoint["state"].items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError("Every released state entry must be a float32 tensor")
        arrays[name] = tensor.detach().cpu().numpy().copy(order="C")
    arrays = _validate_state(arrays, seed)
    metadata = {"schema": 1, "format": FORMAT, "name": f"nested_joint-seed{seed}",
                "cell": "nested_joint", "seed": seed, "settings": dict(SETTINGS),
                "source_checkpoint_sha256": source_digest,
                "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                "protocol_sha256": PROTOCOL_SHA256, "data_receipt_sha256": DATA_RECEIPT_SHA256,
                "state_sha256": RELEASED_STATES[seed], "arrays": _arrays_metadata(arrays)}
    exported = _archive(arrays, metadata)
    digest = hashlib.sha256(exported).hexdigest()
    _write_exclusive(output_path, exported)
    return {"name": metadata["name"], "path": Path(output_path).name, "seed": seed,
            "cell": "nested_joint", "settings": dict(SETTINGS), "sha256": digest,
            "export_sha256": digest, "source_checkpoint_sha256": source_digest,
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256, "state_sha256": RELEASED_STATES[seed]}


def export_release(checkpoint_directory, output_directory):
    """Write all three pinned exports and a manifest, refusing existing content."""
    checkpoint_directory, output_directory = Path(checkpoint_directory), Path(output_directory)
    # Authenticate every input before writing the first output.
    for seed in range(3):
        _, _, actual = _source(checkpoint_directory / f"nested_joint-seed{seed}.pt")
        if actual != seed:
            raise ValueError("Checkpoint filenames do not match their pinned seeds")
    if output_directory.is_symlink() or output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("Exports require a new or empty output directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    members = [export_checkpoint(checkpoint_directory / f"nested_joint-seed{seed}.pt",
                                 output_directory / f"nested_joint-seed{seed}.npz")
               for seed in range(3)]
    manifest = {"schema": 1, "format": FORMAT, "cell": "nested_joint", "settings": dict(SETTINGS),
                "seeds": [0, 1, 2], "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
                "members": members, "runtime_torch_required": False,
                "scientific_fit_or_prediction_performed": False}
    _write_exclusive(output_directory / "manifest.json", _canonical(manifest) + b"\n")
    return manifest


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate metadata keys are forbidden")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON constants are forbidden")


def _checked_members(payload):
    from .model import STATE_SHAPES

    expected = {name + ".npy" for name in STATE_SHAPES} | {_METADATA_KEY + ".npy"}
    members = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if (archive.comment or len(infos) != len(expected)
                    or {info.filename for info in infos} != expected
                    or len({info.filename for info in infos}) != len(infos)
                    or sum(info.file_size for info in infos) > MAX_PAYLOAD_BYTES):
                raise ValueError("Export ZIP membership or size is invalid")
            for info in infos:
                if (info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1
                        or not 0 < info.file_size <= MAX_MEMBER_BYTES
                        or info.compress_size != info.file_size):
                    raise ValueError("Only bounded uncompressed inert NPY entries are permitted")
                value = archive.read(info)
                stream = io.BytesIO(value)
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("Only exported NPY version 1.0 entries are permitted")
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=1024)
                name = info.filename[:-4]
                if name == _METADATA_KEY:
                    valid = (dtype == np.dtype("u1") and len(shape) == 1
                             and 0 < shape[0] <= MAX_METADATA_BYTES)
                else:
                    valid = dtype.str == "<f4" and shape == tuple(STATE_SHAPES[name])
                if (not valid or fortran or dtype.hasobject
                        or stream.tell() + math.prod(shape) * dtype.itemsize != len(value)):
                    raise ValueError("Export NPY shape, dtype or payload length is invalid")
                members[name] = value
    except (zipfile.BadZipFile, EOFError, KeyError, OverflowError) as exc:
        raise ValueError("Malformed inert export archive") from exc
    return members


def _validate_metadata(metadata, source_sha256):
    keys = {"schema", "format", "name", "cell", "seed", "settings", "source_checkpoint_sha256",
            "source_receipt_sha256", "protocol_sha256", "data_receipt_sha256", "state_sha256", "arrays"}
    if not isinstance(metadata, dict) or set(metadata) != keys:
        raise ValueError("Exact export metadata fields are required")
    seed = metadata["seed"]
    if (type(seed) is not int or seed not in RELEASED_STATES or type(metadata["schema"]) is not int
            or metadata["schema"] != 1 or metadata["format"] != FORMAT
            or metadata["name"] != f"nested_joint-seed{seed}" or metadata["cell"] != "nested_joint"
            or _canonical(metadata["settings"]) != _canonical(SETTINGS)
            or metadata["source_checkpoint_sha256"] != source_sha256
            or source_sha256 != RELEASED_CHECKPOINTS[f"nested_joint-seed{seed}"]
            or metadata["source_receipt_sha256"] != SOURCE_RECEIPT_SHA256
            or metadata["protocol_sha256"] != PROTOCOL_SHA256
            or metadata["data_receipt_sha256"] != DATA_RECEIPT_SHA256
            or metadata["state_sha256"] != RELEASED_STATES[seed]):
        raise ValueError("Export metadata does not identify an exact released source")
    return seed


def load_export(path, expected_sha256, expected_source_sha256):
    """Return read-only exact state arrays and identity without importing Torch."""
    expected_sha256, expected_source_sha256 = _sha(expected_sha256), _sha(expected_source_sha256)
    if expected_source_sha256 not in RELEASED_CHECKPOINTS.values():
        raise ValueError("An exact released source checkpoint identity is required")
    payload = _read_regular(path, MAX_EXPORT_BYTES)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Export SHA-256 differs from the supplied pinned identity")
    members = _checked_members(payload)
    # Header validation above precedes every array allocation/payload load.
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        encoded = archive[_METADATA_KEY].tobytes()
        try:
            metadata = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique,
                                  parse_constant=_invalid_constant)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Export metadata is not finite UTF-8 JSON") from exc
        seed = _validate_metadata(metadata, expected_source_sha256)
        arrays = {name: archive[name].copy(order="C") for name in members if name != _METADATA_KEY}
    arrays = _validate_state(arrays, seed)
    if _canonical(metadata["arrays"]) != _canonical(_arrays_metadata(arrays)):
        raise ValueError("Array hashes/shapes/dtypes disagree with export metadata")
    return arrays, {**metadata, "export_sha256": digest}
