#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kval_node.signing import (  # noqa: E402
    generate_operator_keypair,
    load_private_key,
    public_key_id,
    sign_catalog,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sign an operator-approved Kval profile catalogue")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-key", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite signed catalogue: {args.output}")
    if args.generate_key:
        generate_operator_keypair(args.private_key, args.public_key)
    private_key = load_private_key(args.private_key)
    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    envelope = sign_catalog(payload, private_key)
    args.output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "algorithm": "Ed25519",
                "key_id": public_key_id(private_key.public_key()),
                "profiles": len(payload["profiles"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

