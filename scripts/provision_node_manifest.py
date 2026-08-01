from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from kval_node.signing import canonical_json, load_private_key, public_key_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign one Kval node identity manifest offline")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing signed node manifest")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    private_key = load_private_key(args.private_key)
    signature = private_key.sign(canonical_json(payload))
    envelope = {
        "algorithm": "Ed25519",
        "key_id": public_key_id(private_key.public_key()),
        "payload": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    args.output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
