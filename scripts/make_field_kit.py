"""Build the field kit: one zip with the whole field system, ready to install.

    python3 scripts/make_field_kit.py                -> artifacts/field_kit/wirelessjev-field-kit-<date>.zip
    python3 scripts/make_field_kit.py --code-only    -> the small kit without the model and wheels (for updates)

Contents:
- the code (src/), the starter script, the pinned requirements and the guides;
- WirelessJEV's d3-lite model (vllm-sr/d3-lite at the pinned revision, Apache-2.0, LICENSE and
  NOTICE included) as a ready Hugging Face cache, so the WirelessJEV proposal runs offline;
- the Python packages as wheels for Apple-Silicon Macs with Python 3.12 (macOS 14+), so the
  install there needs no internet; other laptops download the same pinned versions once.

Nothing secret goes in: car credentials (WiFi.json, known hosts), lap data and local settings are
refused by name. Every model weight file is checked against its SHA-256 before it is packed."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
INCLUDE = ["src", "field_platform.sh", "requirements-field.txt", "requirements-wirelessjev.txt", "FIELD_KIT.md",
           "docs/PLATFORM.md", "docs/TUESDAY_RUNBOOK.md", "docs/REAL_SITE_PROTOCOL.md",
           # data the field code reads at run time (found by tracing every field command's file opens):
           "artifacts/warehouse_v2/studies/wifi_link_curves.json"]          # ns-3 link curves for the 10-robot replay
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
REFUSE = ("wifi.json", "known_hosts", ".env", "platform_state.json", "platform_config.json", "platform_audit.jsonl",
          "id_rsa", "id_ed25519", ".pem")
WHEELS = ROOT / "artifacts" / "field_kit" / "wheels-macos-arm64-py312"
TOP = "wirelessjev-field-kit"


def code_files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_DIRS)
                    and f.suffix != ".pyc"]
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"missing: {item}")
    return out


def files() -> list[Path]:                                         # kept for the tests: the code part
    return code_files()


def model_snapshot() -> tuple[str, str, Path]:
    from wirelessjev.policy import MODEL_REPO, MODEL_REVISION
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    snap = hub / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots" / MODEL_REVISION
    if not (snap / "model-00001-of-00001.safetensors").exists():
        raise SystemExit(f"d3-lite {MODEL_REVISION} is not in {hub}. With internet: python -c \"from huggingface_hub import "
                         f"snapshot_download; snapshot_download('{MODEL_REPO}', revision='{MODEL_REVISION}')\"")
    return MODEL_REPO, MODEL_REVISION, snap


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-only", action="store_true", help="leave out the model and the wheels")
    args = ap.parse_args(argv)
    fs = code_files()
    bad = [f for f in fs if any(r in f.name.lower() for r in REFUSE) or "-lap-" in f.name]
    if bad:
        print("refusing to build: these look like credentials or lap data:", *bad, sep="\n  ", file=sys.stderr)
        return 1
    entries = [(f, f"{TOP}/{f.relative_to(ROOT)}") for f in fs]
    manifest = []
    if not args.code_only:
        repo, rev, snap = model_snapshot()
        base = f"{TOP}/hf-cache/hub/models--{repo.replace('/', '--')}"
        for f in sorted(snap.rglob("*")):
            real = f.resolve()
            if not real.is_file():
                continue
            rel = f.relative_to(snap)
            if rel.suffix == ".safetensors":
                # Hugging Face names a weight blob by its SHA-256: the snapshot's own link target (later links may
                # point on into a deduplicated store with other names).
                blob = Path(os.readlink(f)).name if f.is_symlink() else real.name
                digest = sha256(real)
                if len(blob) == 64 and digest != blob:
                    raise SystemExit(f"model file {rel} does not match its SHA-256; download it again")
                manifest.append(f"{digest}  hf-cache/hub/models--{repo.replace('/', '--')}/snapshots/{rev}/{rel}")
            entries.append((real, f"{base}/snapshots/{rev}/{rel}"))
        # The cache's own metadata, so the model resolves with no internet: the saved file listing
        # (trees/<revision>.json), the "no such file" markers and the refs. Without trees/,
        # snapshot_download asks the Hub for the listing even in offline mode.
        model_root = snap.parents[1]
        for sub in ("trees", ".no_exist", "refs"):
            for f in sorted((model_root / sub).rglob("*")) if (model_root / sub).is_dir() else []:
                if f.is_file():
                    entries.append((f.resolve(), f"{base}/{f.relative_to(model_root)}"))
        if not (model_root / "trees" / f"{rev}.json").is_file():
            raise SystemExit(f"{model_root}/trees/{rev}.json is missing: the bundled model would need the internet to load")
        if not WHEELS.is_dir() or not any(WHEELS.glob("*.whl")):
            raise SystemExit(f"no wheels in {WHEELS}: download them first (see docs/PLATFORM.md, field kit)")
        entries += [(w, f"{TOP}/wheels/macos-arm64-py312/{w.name}") for w in sorted(WHEELS.glob("*.whl"))]
    out_dir = ROOT / "artifacts" / "field_kit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"wirelessjev-field-kit-{date.today():%Y%m%d}{'-code-only' if args.code_only else ''}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for src, name in entries:
            info = zipfile.ZipInfo.from_file(src, name)
            if src.name == "field_platform.sh":
                info.external_attr = 0o100755 << 16             # stays executable after unzipping
            packed = src.suffix in (".safetensors", ".whl", ".png", ".mp4")     # already compressed or incompressible
            info.compress_type = zipfile.ZIP_STORED if packed else zipfile.ZIP_DEFLATED
            with src.open("rb") as fin, z.open(info, "w", force_zip64=True) as fout:
                for chunk in iter(lambda: fin.read(1 << 22), b""):
                    fout.write(chunk)
        if not args.code_only:
            z.writestr(f"{TOP}/MODEL_SHA256.txt", "\n".join(manifest) + "\n")
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB, {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
