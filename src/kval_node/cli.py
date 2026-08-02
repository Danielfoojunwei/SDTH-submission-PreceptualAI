from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AuditLedger
from .capabilities import CapabilityServices
from .config import NodeConfig
from .dataset import FEATURE_NAMES, CorpusExporter, write_jsonl
from .hardware import verify_jetson_platform
from .lease import LeaseAction
from .service import KvalNode
from .signing import load_verified_catalog


DEFAULT_CONFIG = "/etc/kval/node.json"


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _print_capability(result: dict[str, object]) -> None:
    """Print a capability result and exit non-zero unless it genuinely passed.

    A capability that is unavailable, unverified, or invalid must not look like
    success to a shell, a systemd unit, or the HIL script.
    """
    _print(result)
    if result.get("available") is not True:
        raise SystemExit(3)
    for key in ("verified", "valid"):
        if key in result and result[key] is not True:
            raise SystemExit(4)


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

    spectrum = subparsers.add_parser("spectrum-scan")
    spectrum.add_argument("--config", default=DEFAULT_CONFIG)

    power = subparsers.add_parser("power-plan")
    power.add_argument("--config", default=DEFAULT_CONFIG)
    power.add_argument("--tegrastats-line", default=None)
    power.add_argument("--current-nvpmodel-id", type=int, default=None)

    transport = subparsers.add_parser("transport-preflight")
    transport.add_argument("--config", default=DEFAULT_CONFIG)

    analyze = subparsers.add_parser("analyze-outcomes")
    analyze.add_argument("--config", default=DEFAULT_CONFIG)
    analyze.add_argument("--window", type=int, default=8)

    verify_model = subparsers.add_parser("verify-model")
    verify_model.add_argument("--config", default=DEFAULT_CONFIG)

    export = subparsers.add_parser("export-corpus")
    export.add_argument("--config", default=DEFAULT_CONFIG)
    export.add_argument("--output", required=True, help="JSONL destination")
    export.add_argument("--horizon-seconds", type=float, default=60.0)

    fabric_report = subparsers.add_parser("fabric-report")
    fabric_report.add_argument("--config", default=DEFAULT_CONFIG)
    fabric_report.add_argument("--output", default=None, help="write the signed report here")

    fabric_view = subparsers.add_parser("fabric-view")
    fabric_view.add_argument("--config", default=DEFAULT_CONFIG)

    locate = subparsers.add_parser("locate-emitter")
    locate.add_argument("--config", default=DEFAULT_CONFIG)
    locate.add_argument("--band-id", required=True)

    fed_update = subparsers.add_parser("federated-update")
    fed_update.add_argument("--config", default=DEFAULT_CONFIG)
    fed_update.add_argument("--output", default=None)

    fed_agg = subparsers.add_parser("federated-aggregate")
    fed_agg.add_argument("--config", default=DEFAULT_CONFIG)
    fed_agg.add_argument("--round-index", type=int, default=0)

    topology = subparsers.add_parser("fabric-topology")
    topology.add_argument("--config", default=DEFAULT_CONFIG)

    memory = subparsers.add_parser("audit-memory")
    memory.add_argument("--config", default=DEFAULT_CONFIG)

    forecast = subparsers.add_parser("forecast-links")
    forecast.add_argument("--config", default=DEFAULT_CONFIG)

    track = subparsers.add_parser("track-emitter")
    track.add_argument("--config", default=DEFAULT_CONFIG)
    track.add_argument("--band-id", required=True)
    track.add_argument("--horizon-seconds", type=float, default=600.0)

    anomaly = subparsers.add_parser("detect-fleet-anomaly")
    anomaly.add_argument("--config", default=DEFAULT_CONFIG)

    excise = subparsers.add_parser("excise-node")
    excise.add_argument("--config", default=DEFAULT_CONFIG)
    excise.add_argument("--node-id", required=True)

    crypto = subparsers.add_parser("crypto-posture")
    crypto.add_argument("--config", default=DEFAULT_CONFIG)

    lease = subparsers.add_parser("verify-lease")
    lease.add_argument("--config", default=DEFAULT_CONFIG)
    lease.add_argument("--lease", required=True)
    lease.add_argument("--action", choices=[item.value for item in LeaseAction], default=None)
    lease.add_argument("--redeem", action="store_true")
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
    if args.command == "spectrum-scan":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).sense_spectrum())
        return
    if args.command == "power-plan":
        services = CapabilityServices(NodeConfig.load(args.config))
        _print_capability(
            services.plan_power(
                tegrastats_line=args.tegrastats_line,
                current_nvpmodel_id=args.current_nvpmodel_id,
            )
        )
        return
    if args.command == "transport-preflight":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).transport_preflight())
        return
    if args.command == "analyze-outcomes":
        services = CapabilityServices(NodeConfig.load(args.config))
        _print_capability(services.analyze_outcomes(window=args.window))
        return
    if args.command == "verify-model":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).verify_model())
        return
    if args.command == "export-corpus":
        config = NodeConfig.load(args.config)
        exporter = CorpusExporter(config.audit_database_path, horizon_seconds=args.horizon_seconds)
        examples, stats = exporter.export()
        write_jsonl(examples, Path(args.output))
        _print({"output": args.output, "feature_names": list(FEATURE_NAMES), **stats.to_dict()})
        return
    if args.command == "fabric-report":
        result = CapabilityServices(NodeConfig.load(args.config)).publish_report()
        if args.output and result.get("available") is True:
            Path(args.output).write_text(
                json.dumps(result["report"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        _print_capability(result)
        return
    if args.command == "fabric-view":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).aggregate_fleet())
        return
    if args.command == "locate-emitter":
        services = CapabilityServices(NodeConfig.load(args.config))
        _print_capability(services.locate_emitter(args.band_id))
        return
    if args.command == "federated-update":
        result = CapabilityServices(NodeConfig.load(args.config)).federated_update()
        if args.output and result.get("available") is True:
            Path(args.output).write_text(
                json.dumps(result["update"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        _print_capability(result)
        return
    if args.command == "federated-aggregate":
        _print_capability(
            CapabilityServices(NodeConfig.load(args.config)).federated_aggregate(
                round_index=args.round_index
            )
        )
        return
    if args.command == "fabric-topology":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).fabric_topology())
        return
    if args.command == "audit-memory":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).audit_memory())
        return
    if args.command == "forecast-links":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).forecast_links())
        return
    if args.command == "track-emitter":
        services = CapabilityServices(NodeConfig.load(args.config))
        _print_capability(
            services.track_emitter(args.band_id, horizon_seconds=args.horizon_seconds)
        )
        return
    if args.command == "detect-fleet-anomaly":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).detect_fleet_anomaly())
        return
    if args.command == "excise-node":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).excise_node(args.node_id))
        return
    if args.command == "crypto-posture":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).crypto_posture())
        return
    if args.command == "verify-lease":
        services = CapabilityServices(NodeConfig.load(args.config))
        envelope = json.loads(Path(args.lease).read_text(encoding="utf-8"))
        expected_action = LeaseAction(args.action) if args.action else None
        _print_capability(
            services.verify_lease_envelope(
                envelope,
                expected_action=expected_action,
                redeem=args.redeem,
            )
        )
        return
    if args.command in {"once", "run"}:
        result = _with_node(args.config, args.command)
        if result is not None:
            _print(result)
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()
