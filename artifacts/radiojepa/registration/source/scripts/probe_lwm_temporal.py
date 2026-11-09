"""Verify and execute one reviewed LWM-Temporal checkpoint on a shape fixture.

This is an offline executability probe, not a CSI accuracy or control benchmark.
The downloaded implementation remains outside this application. No dependency
installation, remote-code loader, training, or hardware access is performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "wi-lab/lwm-temporal"
REVISION = "df32f4beb6c7fe11d83031edbe5d6312798b81c3"
REVIEWED = {
    "LICENSE": "db8d07cbbc61a50354d22437da225c2803af9b7eae56a4e0baf89d2f37c63df2",
    "README.md": "6ac59d21ecbbb504cc077e70f0348878d9a4a67b513ab41f4d2f2049703dc4fe",
    "LWMTemporal/models/lwm.py": "c8ef6e6ccbc5124edf29d79d190d7c065fd98e668af251ff09979ec1d0d39f72",
    "checkpoints/config.json": "556fa34bbb1e5649385a97e576ffba81c1c6d4fa21e1635664f72a42d208c392",
    "checkpoints/pytorch_model.bin": "90be5447ee15f13547f1d0819de4d80fc1b49c28737378824513cb293335e0be",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def download(source: Path, output: Path) -> None:
    """Fetch pinned public files only; never import or execute downloaded code."""
    metadata_url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=45) as response:
        raw = response.read()
    metadata = json.loads(raw)
    if metadata["sha"] != REVISION:
        raise ValueError("Model metadata revision changed")
    (output / "upstream_metadata.json").write_bytes(raw)
    receipts = []
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name.endswith(".ipynb"):
            continue
        target = source / name
        if not target.resolve().is_relative_to(source.resolve()):
            raise ValueError("Upstream filename escapes source directory")
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        sha256 = digest(data)
        if len(data) != item["size"]:
            raise ValueError(f"Download size mismatch: {name}")
        if item.get("lfs"):
            if sha256 != item["lfs"]["sha256"]:
                raise ValueError(f"LFS SHA-256 mismatch: {name}")
        else:
            header = f"blob {len(data)}\0".encode()
            if hashlib.sha1(header + data).hexdigest() != item["blobId"]:
                raise ValueError(f"Git object mismatch: {name}")
        if name in REVIEWED and sha256 != REVIEWED[name]:
            raise ValueError(f"Reviewed file changed: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        receipts.append({"path": name, "bytes": len(data), "sha256": sha256,
                         "url": url, "git_blob_id": item["blobId"],
                         "lfs_sha256_verified": bool(item.get("lfs")),
                         "git_blob_verified": not bool(item.get("lfs"))})
    write_json(output / "download_manifest.json", {
        "repository": REPOSITORY, "revision": REVISION,
        "source_directory": str(source), "metadata_url": metadata_url,
        "metadata_sha256": digest(raw),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "omitted_files": ["notebooks/tutorial.ipynb"], "files": receipts,
        "upstream_code_executed_during_download": False,
    })


def probe(source: Path, output: Path) -> dict:
    # Refuse changed implementation, license, configuration, or checkpoint bytes
    # before executing the one audited source module or deserializing tensors.
    for name, expected in REVIEWED.items():
        if digest((source / name).read_bytes()) != expected:
            raise ValueError(f"Reviewed SHA-256 mismatch: {name}")
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location(
        "reviewed_lwm_temporal", source / "LWMTemporal/models/lwm.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load reviewed source module")
    upstream = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = upstream  # dataclass resolves its defining module
    spec.loader.exec_module(upstream)

    checkpoint = source / "checkpoints/pytorch_model.bin"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty tensor state_dict")
    if not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Checkpoint contains non-tensor or non-finite entries")
    config = upstream.LWMConfig.from_dict(json.loads((source / "checkpoints/config.json").read_text()))
    # This is the upstream loader; the environment enforces weights-only loading
    # even though upstream omits that keyword. Then enforce stricter coverage
    # than its default strict=False to rule out randomly initialized parameters.
    import os
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    started = time.perf_counter()
    model = upstream.LWMBackbone.from_pretrained(checkpoint, config=config, map_location="cpu")
    remapped = upstream.LWMBackbone._remap_state_dict(state)
    coverage = model.load_state_dict(remapped, strict=True)
    if coverage.missing_keys or coverage.unexpected_keys:
        raise ValueError("Incomplete pretrained weight coverage")
    loaded_state = model.state_dict()
    if not all(torch.equal(loaded_state[k], v) for k, v in remapped.items()):
        raise ValueError("Loaded model differs from pretrained checkpoint")
    model.requires_grad_(False).eval()
    load_seconds = time.perf_counter() - started

    # Deterministic complex angle-delay-shaped unit fixture. No measured CSI,
    # DeepMIMO scenario, RSSI conversion, or claimed physical calibration.
    batch, frames, height, width = 1, 11, 32, 8
    index = torch.arange(frames * height * width, dtype=torch.float32)
    amplitude = 0.7 + 0.3 * torch.cos(index * 0.013)
    sequence = torch.complex(amplitude * torch.cos(index * 0.031),
                             amplitude * torch.sin(index * 0.031))
    sequence = sequence.reshape(batch, frames, height, width)
    tokens, mask = model.tokenizer(sequence, model.config.patch_size)
    mask[:, torch.arange(tokens.size(1)) % 5 < 3] = True
    with torch.inference_mode():
        started = time.perf_counter()
        first = model(sequence, mask=mask)["reconstruction"]
        first_seconds = time.perf_counter() - started
        started = time.perf_counter()
        repeated = model(sequence, mask=mask)["reconstruction"]
        repeat_seconds = time.perf_counter() - started
    if first.shape != tokens.shape or not torch.isfinite(first).all():
        raise ValueError("Pretrained forward produced invalid reconstruction")
    max_difference = float((first - repeated).abs().max())
    if max_difference != 0.0:
        raise ValueError("Repeated CPU forward was not deterministic")
    np.savez_compressed(output / "fixture_forward.npz", input=sequence.numpy(),
                        mask=mask.numpy(), reconstruction=first.numpy())
    result = {
        "schema": "wirelessjev.lwm_temporal_probe.v1", "status": "passed",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY, "revision": REVISION, "license": "MIT",
        "source_directory": str(source), "reviewed_sha256": REVIEWED,
        "executed_upstream_files": ["LWMTemporal/models/lwm.py"],
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "device": "cpu", "torch_threads": 2,
        "dependency_installation": "none; existing torch/numpy; no upstream package installation",
        "weights_only_deserialization": True, "strict_pretrained_load": True,
        "missing_keys": [], "unexpected_keys": [], "checkpoint_tensors": len(state),
        "pretrained_parameters": sum(p.numel() for p in model.parameters()),
        "all_loaded_tensors_equal_checkpoint": True,
        "randomly_initialized_parameters_remaining": False,
        "configuration": model.config.to_dict(),
        "input": {"kind": "deterministic analytical complex shape fixture",
                  "measured_or_simulated_csi": False, "shape": list(sequence.shape),
                  "dtype": str(sequence.dtype), "masked_tokens": int(mask.sum()),
                  "total_tokens": tokens.size(1)},
        "output": {"shape": list(first.shape), "dtype": str(first.dtype),
                   "finite": True, "mean": float(first.mean()),
                   "std": float(first.std()), "l2_norm": float(first.norm()),
                   "sha256": digest(first.numpy().tobytes()),
                   "repeat_max_absolute_difference": max_difference},
        "timing_seconds": {"model_load_and_strict_check": load_seconds,
                           "first_forward": first_seconds, "repeat_forward": repeat_seconds},
        "application_radio_observations_used": False,
        "application_task_accuracy_tested": False,
        "control_or_service_world_integration": False,
        "quality_conclusion": "Executability only; fixture outputs do not measure wireless prediction or control quality",
        "limits": ["No paired complex CSI dataset is supplied by the current application observation contract",
                   "Default checkpoint attention includes future-frame offsets; this reconstruction probe is not a causal forecast",
                   "No downstream channel-prediction checkpoint, fine-tuning, baseline comparison, or hardware measurement was used"],
    }
    write_json(output / "probe_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/workspace/upstream/continual-radio/lwm-temporal"))
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/ran_research/lwm_temporal")
    parser.add_argument("--download-only", action="store_true",
                        help="Download pinned source/weights without executing upstream code")
    args = parser.parse_args()
    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.download_only:
        download(args.source_dir, args.output_dir)
        print("Pinned source/weights verified; no upstream code executed")
    else:
        result = probe(args.source_dir, args.output_dir)
        print(json.dumps({k: result[k] for k in ("status", "pretrained_parameters", "output", "timing_seconds")}))


if __name__ == "__main__":
    main()
