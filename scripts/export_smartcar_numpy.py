"""Build inert NumPy exports of the exact three released RadioJEPA members."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path,
                        default=Path("artifacts/nested_radio/checkpoints"))
    parser.add_argument("--out", type=Path, required=True,
                        help="A new or empty directory; exports are never overwritten")
    args = parser.parse_args(argv)
    from wirelessjev.smartcar_numpy.export import export_release

    manifest = export_release(args.checkpoints, args.out)
    print(json.dumps({"manifest": str(args.out / "manifest.json"),
                      "members": [{"name": row["name"], "sha256": row["sha256"],
                                   "source_checkpoint_sha256": row["source_checkpoint_sha256"]}
                                  for row in manifest["members"]],
                      "scientific_fit_or_prediction_performed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
