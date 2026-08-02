#!/usr/bin/env python3
"""Sign Kval capability policies and capability leases offline.

The operator private key never touches a node. This tool runs on an
operator/admin machine, validates the document against its parser, signs it, and
writes an envelope the node can verify with the operator public key alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kval_node.capabilities import (  # noqa: E402
    POWER_DOCUMENT_TYPE,
    SPECTRUM_DOCUMENT_TYPE,
    TRANSPORT_DOCUMENT_TYPE,
)
from kval_node.lease import sign_lease  # noqa: E402
from kval_node.power import PowerPolicy  # noqa: E402
from kval_node.signing import (  # noqa: E402
    canonical_json,
    generate_operator_keypair,
    load_private_key,
    public_key_id,
    sign_document,
)
from kval_node.spectrum import SpectrumPolicy  # noqa: E402
from kval_node.transport_security import TransportPolicy  # noqa: E402

DOCUMENT_TYPES = {
    "spectrum": (SPECTRUM_DOCUMENT_TYPE, SpectrumPolicy.from_dict),
    "power": (POWER_DOCUMENT_TYPE, PowerPolicy.from_dict),
    "transport": (TRANSPORT_DOCUMENT_TYPE, TransportPolicy.from_dict),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sign a Kval capability policy or lease")
    parser.add_argument("kind", choices=[*DOCUMENT_TYPES, "lease"])
    parser.add_argument("--input", type=Path, required=True, help="policy or lease body JSON")
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-key", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    if args.generate_key:
        generate_operator_keypair(args.private_key, args.public_key)
    private_key = load_private_key(args.private_key)
    payload = json.loads(args.input.read_text(encoding="utf-8"))

    if args.kind == "lease":
        envelope = sign_lease(payload, private_key)
        summary = {"kind": "lease", "action": payload.get("action")}
    else:
        document_type, validator = DOCUMENT_TYPES[args.kind]
        # Validate before signing so a malformed policy never gets a signature.
        validator(payload)
        envelope = sign_document(document_type, payload, private_key)
        # Emit the hash a lease's policy_hash field must carry to bind to this
        # exact policy; without it an operator has to compute it by hand.
        policy_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
        summary = {"kind": args.kind, "document_type": document_type, "policy_hash": policy_hash}

    args.output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.update({"output": str(args.output), "key_id": public_key_id(private_key.public_key())})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
