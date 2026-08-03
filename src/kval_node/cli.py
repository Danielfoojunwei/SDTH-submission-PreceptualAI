from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import AuditLedger
from .capabilities import CapabilityServices
from .config import NodeConfig
from .telemetry import TelemetryCollector
from .dataset import FEATURE_NAMES, CorpusExporter, DatasetError, write_jsonl
from .hardware import verify_jetson_platform
from .lease import LeaseAction
from .service import KvalNode
from .signing import load_verified_catalog


DEFAULT_CONFIG = "/etc/kval/node.json"


def _print(value: object) -> None:
    # allow_nan=False: Python emits bare ``Infinity`` and ``NaN``, which RFC
    # 8259 does not permit. Python and jq tolerate them; a Go or JavaScript
    # consumer rejects the whole document. Refusals carry an infinite
    # uncertainty radius, so this fired on every declined prediction.
    print(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False))


def _print_capability(result: dict[str, object]) -> None:
    """Print a capability result and exit non-zero unless it genuinely passed.

    A capability that is unavailable, unverified, or invalid must not look like
    success to a shell, a systemd unit, or the HIL script.
    """
    _print(result)
    if result.get("available") is not True:
        raise SystemExit(3)
    # Every verdict field a capability can fail on. ``audit-memory`` reports
    # under ``trustworthy`` and exited 0 on a ledger it had just proven
    # poisoned, because only ``verified`` and ``valid`` were checked here.
    for key in ("verified", "valid", "trustworthy", "granted"):
        if key in result and result[key] is not True:
            raise SystemExit(4)


def _with_node(config_path: str, action: str, **kwargs: object) -> object:
    node = KvalNode(NodeConfig.load(config_path))
    try:
        if action == "once":
            return node.run_once()
        if action == "benchmark":
            return node.benchmark_optimizer(int(kwargs["iterations"]))
        if action == "c2-status":
            return node.c2_status()
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






    heterogeneity = subparsers.add_parser("fleet-heterogeneity")
    heterogeneity.add_argument("--config", default=DEFAULT_CONFIG)

    fed_update = subparsers.add_parser("federated-update")
    fed_update.add_argument("--config", default=DEFAULT_CONFIG)
    fed_update.add_argument("--output", default=None)

    fed_agg = subparsers.add_parser("federated-aggregate")
    fed_agg.add_argument("--config", default=DEFAULT_CONFIG)
    fed_agg.add_argument("--round-index", type=int, default=0)


    memory = subparsers.add_parser("audit-memory")
    memory.add_argument("--config", default=DEFAULT_CONFIG)

    forecast = subparsers.add_parser("forecast-links")
    forecast.add_argument("--config", default=DEFAULT_CONFIG)


    anomaly.add_argument("--config", default=DEFAULT_CONFIG)

    excise = subparsers.add_parser("excise-node")
    excise.add_argument("--config", default=DEFAULT_CONFIG)
    excise.add_argument("--node-id", required=True)

    subparsers.add_parser("plan-slices")
    subparsers.add_parser("assess-link-anomaly")
    emcon = subparsers.add_parser("emcon-posture")
    emcon.add_argument("--config", default=DEFAULT_CONFIG)


    c2 = subparsers.add_parser("c2-status")
    c2.add_argument("--config", default=DEFAULT_CONFIG)

    renew = subparsers.add_parser("renew-grant")
    renew.add_argument("--config", default=DEFAULT_CONFIG)
    renew.add_argument("--grant", required=True)
    renew.add_argument("--attestation", action="append", default=[], required=True)
    renew.add_argument("--extend-seconds", type=float, default=3600.0)

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
        # The one command whose job is reporting platform state must not answer
        # with a traceback when the platform is wrong.
        try:
            report = verify_jetson_platform().to_dict()
        except Exception as exc:  # noqa: BLE001 - report the refusal, do not raise it
            _print({"supported": False, "reason": f"{type(exc).__name__}: {exc}"})
            raise SystemExit(3) from None
        _print(report)
        if report.get("supported") is not True:
            raise SystemExit(3)
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
        # Was _print, so a refused activation exited 0 and run_hil_validation.sh
        # read it as a pass.
        result = _with_node(args.config, "activate", profile_id=args.profile_id)
        _print(result)
        if isinstance(result, dict) and result.get("activation_error"):
            raise SystemExit(4)
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
        try:
            examples, stats = exporter.export()
        except DatasetError as exc:
            # Every other corpus consumer refuses structurally; this one raised
            # a traceback at a tampered ledger, which is the moment structure
            # matters most.
            _print_capability({"available": False, "reason": str(exc)})
            return
        write_jsonl(examples, Path(args.output))
        _print({"output": args.output, "feature_names": list(FEATURE_NAMES), **stats.to_dict()})
        return
    if args.command == "fleet-heterogeneity":
        services = CapabilityServices(NodeConfig.load(args.config))
        _print_capability(services.assess_fleet_heterogeneity())
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
    if args.command == "audit-memory":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).audit_memory())
        return
    if args.command == "forecast-links":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).forecast_links())
        return
    if args.command == "excise-node":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).excise_node(args.node_id))
        return
    if args.command == "plan-slices":
        config = NodeConfig.load(args.config)
        services = CapabilityServices(config)
        snapshot = TelemetryCollector().collect(
            load_verified_catalog(config.catalog_path, config.catalog_public_key_path)
        )
        _print_capability(services.plan_slices(snapshot))
        return
    if args.command == "assess-link-anomaly":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).assess_link_anomaly())
        return
    if args.command == "emcon-posture":
        _print_capability(CapabilityServices(NodeConfig.load(args.config)).emcon_posture())
        return
    if args.command == "c2-status":
        _print_capability(_with_node(args.config, "c2-status"))
        return
    if args.command == "renew-grant":
        services = CapabilityServices(NodeConfig.load(args.config))
        _print_capability(
            services.renew_grant_from_files(
                Path(args.grant),
                [Path(item) for item in args.attestation],
                extend_seconds=args.extend_seconds,
            )
        )
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
