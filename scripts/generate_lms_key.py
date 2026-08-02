#!/usr/bin/env python3
"""Generate a CNSA 2.0-approved LMS signing key for the policy trust root.

Run this on the **operator's** machine, never on a node. The node receives only
the public key; the seed and the signature-index state stay with whoever signs.

The tree height is chosen once and cannot be changed: it fixes how many
documents this key can ever sign. Height 10 gives 1,024 signatures, which is
years of catalogue and policy revisions. Running out means generating and
distributing a new public key to every node, so choose with headroom.

NIST SP 800-208 requires key generation and state custody inside a FIPS 140
Level 3 hardware module. This script writes to files, which is materially
weaker and is stated in the output rather than glossed over.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kval_node.pqc import (  # noqa: E402
    DEFAULT_LMOTS_TYPE,
    LMS_SHA256_M32_H5,
    LMS_SHA256_M32_H10,
    LMS_SHA256_M32_H15,
    generate_lms_key,
)

HEIGHTS = {5: LMS_SHA256_M32_H5, 10: LMS_SHA256_M32_H10, 15: LMS_SHA256_M32_H15}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", required=True, help="seed destination (0600)")
    parser.add_argument("--public", required=True, help="public key destination")
    parser.add_argument("--height", type=int, default=10, choices=sorted(HEIGHTS))
    args = parser.parse_args()

    private_path = Path(args.private)
    public_path = Path(args.public)
    if private_path.exists() or public_path.exists():
        print("refusing to overwrite an existing key", file=sys.stderr)
        return 1

    key = generate_lms_key(lms_type=HEIGHTS[args.height], ots_type=DEFAULT_LMOTS_TYPE)
    public = key.public_key()

    # Create the file 0600 *before* any secret bytes are written; writing first
    # and chmod-ing after leaves a window where the seed is world-readable.
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(base64.b64encode(key.seed + key.identifier).decode("ascii") + "\n")
    public_path.write_text(base64.b64encode(public.to_bytes()).decode("ascii") + "\n", encoding="utf-8")

    print(f"key_id          : {public.key_id()}")
    print(f"algorithm       : LMS_SHA256_M32 (CNSA 2.0 approved for software/firmware signing)")
    print(f"signatures       : {public.signatures_available()} — irrevocable, chosen at generation")
    print(f"private (0600)  : {private_path}")
    print(f"public          : {public_path}")
    print()
    print("NIST SP 800-208 requires key generation and signature-index custody in a")
    print("FIPS 140 Level 3 hardware module. This key lives in a file: an attacker who")
    print("can roll the index back can still induce one-time key reuse. Use an HSM for")
    print("anything accredited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
