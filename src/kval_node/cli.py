from __future__ import annotations

import argparse
import json

from .audit import AuditLedger
from .config import NodeConfig
from .hardware import verify_jetson_platform
from .service import KvalNode
from .signing import load_verified_catalog


DEFAULT_CONFIG = "/etc/kval/node.json"


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _with_node(config_path: str, action: str, **kwargs: object) -> object:
    node = KvalNode(NodeConfig.load(config_path))
    try:
        if action == "once":
            return node.run_once()
        if action == "benchmark":
            return node.benchmark_optimizer(int(kwargs["iterations"]))
        if action == "activate":
            return node.activate_for_hil(str(kwargs["profile_id"]))
        if action == "run":
            node.run_forever()
            return {"status": "stopped"}
        raise ValueError(f"unknown node action: {action}")
    finally:
        node.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kval-node")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight")

    verify_catalog_parser = subparsers.add_parser("verify-catalog")
    verify_catalog_parser.add_argument("--config", default=DEFAULT_CONFIG)

    for name in ("once", "run"):
        item = subparsers.add_parser(name)
        item.add_argument("--config", default=DEFAULT_CONFIG)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--config", default=DEFAULT_CONFIG)
    benchmark.add_argument("--iterations", type=int, default=100)

    activation = subparsers.add_parser("hil-activate")
    activation.add_argument("--config", default=DEFAULT_CONFIG)
    activation.add_argument("--profile-id", required=True)

    verify_audit = subparsers.add_parser("verify-audit")
    verify_audit.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        _print(verify_jetson_platform().to_dict())
        return
    if args.command == "verify-catalog":
        config = NodeConfig.load(args.config)
        catalog = load_verified_catalog(config.catalog_path, config.catalog_public_key_path)
        _print(
            {
                "valid": True,
                "version": catalog.version,
                "profiles": [profile.to_public_dict() for profile in catalog.profiles],
                "expires_at": catalog.expires_at.isoformat(),
                "payload_hash": catalog.payload_hash,
                "key_id": catalog.key_id,
            }
        )
        return
    if args.command == "verify-audit":
        config = NodeConfig.load(args.config)
        ledger = AuditLedger(config.audit_database_path)
        try:
            _print(ledger.verify().__dict__)
        finally:
            ledger.close()
        return
    if args.command == "benchmark":
        _print(_with_node(args.config, "benchmark", iterations=args.iterations))
        return
    if args.command == "hil-activate":
        _print(_with_node(args.config, "activate", profile_id=args.profile_id))
        return
    if args.command in {"once", "run"}:
        result = _with_node(args.config, args.command)
        if result is not None:
            _print(result)
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()
