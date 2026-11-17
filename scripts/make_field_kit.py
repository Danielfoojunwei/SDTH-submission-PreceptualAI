"""Build the field kit: a small zip with everything needed to run the Field Platform.

    python3 scripts/make_field_kit.py            -> artifacts/field_kit/wirelessjev-field-kit-<date>.zip

The whole repository is about 600 MB of files and a 4.8 GB history; the kit is the code
(src/), the starter script, the pinned field requirements and the guides, a few megabytes.
Nothing secret goes in: car credentials (WiFi.json, known hosts), lap data and local
settings are refused by name, and the kit is checked for them before it is written."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "field_platform.sh", "requirements-field.txt", "FIELD_KIT.md", "docs/PLATFORM.md",
           "docs/TUESDAY_RUNBOOK.md", "docs/REAL_SITE_PROTOCOL.md"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
REFUSE = ("wifi.json", "known_hosts", ".env", "platform_state.json", "platform_config.json", "platform_audit.jsonl",
          "id_rsa", "id_ed25519", ".pem")


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file() and not (set(f.relative_to(ROOT).parts) & SKIP_DIRS)
                    and f.suffix not in (".pyc",)]
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"missing: {item}")
    return out


def main() -> int:
    fs = files()
    bad = [f for f in fs if any(r in f.name.lower() for r in REFUSE) or "-lap-" in f.name]
    if bad:
        print("refusing to build: these look like credentials or lap data:", *bad, sep="\n  ", file=sys.stderr)
        return 1
    out_dir = ROOT / "artifacts" / "field_kit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"wirelessjev-field-kit-{date.today():%Y%m%d}.zip"
    top = "wirelessjev-field-kit"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in fs:
            info = zipfile.ZipInfo.from_file(f, f"{top}/{f.relative_to(ROOT)}")
            if f.name == "field_platform.sh":
                info.external_attr = 0o100755 << 16             # stays executable after unzipping
            with f.open("rb") as src:
                z.writestr(info, src.read(), zipfile.ZIP_DEFLATED)
    print(f"{out}  {out.stat().st_size / 1e6:.1f} MB, {len(fs)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
