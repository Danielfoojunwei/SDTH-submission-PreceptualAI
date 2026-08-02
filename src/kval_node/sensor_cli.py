"""Daemon entry point for the receive-only spectrum sensor adapter.

Serves the ``kval-spectrum-agent/v1`` contract over a root-owned Unix socket,
answering scans with real measurements taken by the host's own radio. It refuses
to start when no measurement backend is present, so a node can never be served
observations the hardware did not produce.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import socketserver
from pathlib import Path

from .capabilities import SPECTRUM_DOCUMENT_TYPE
from .sensor_adapter import ADAPTER_SCHEMA, SpectrumSensorAdapter, build_backends
from .signing import load_verified_document
from .spectrum import SpectrumPolicy

MAX_REQUEST_BYTES = 65_536


class _Handler(socketserver.StreamRequestHandler):
    timeout = 10.0

    def handle(self) -> None:
        adapter: SpectrumSensorAdapter = self.server.adapter  # type: ignore[attr-defined]
        policy: SpectrumPolicy = self.server.policy  # type: ignore[attr-defined]
        try:
            raw = self.rfile.readline(MAX_REQUEST_BYTES)
            if not raw:
                return
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            response = adapter.handle(request, policy)
        except Exception as exc:  # never leak a traceback to the client
            response = {"schema": ADAPTER_SCHEMA, "success": False, "error": str(exc)}
        self.wfile.write(
            json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )


class _Server(socketserver.ThreadingUnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kval-spectrum-sensord",
        description="Kval receive-only spectrum sensor daemon",
    )
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="signed spectrum policy")
    parser.add_argument("--public-key", type=Path, required=True, help="operator public key")
    parser.add_argument("--interface", default=None, help="Wi-Fi interface to survey, e.g. wlan0")
    parser.add_argument("--adapter-id", default=None, help="defaults to kval-sensor@<hostname>")
    parser.add_argument(
        "--allow-active-scan",
        action="store_true",
        help=(
            "enable the iw scan backend, which TRANSMITS 802.11 probe requests. "
            "Off by default: observations it contributes to are marked non-passive "
            "and the node will refuse them"
        ),
    )
    parser.add_argument(
        "--passive-scan-only",
        action="store_true",
        help="accepted and redundant: passive operation is now the default",
    )
    parser.add_argument("--no-sdr", action="store_true", help="do not use an attached RTL-SDR")
    parser.add_argument("--no-modem", action="store_true", help="do not use ModemManager serving-cell data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if os.geteuid() != 0:
        raise SystemExit("kval-spectrum-sensord must run as root so the socket is root-owned")

    payload, policy_hash = load_verified_document(
        args.policy, args.public_key, expected_type=SPECTRUM_DOCUMENT_TYPE
    )
    policy = SpectrumPolicy.from_dict(payload)

    backends = build_backends(
        interface=args.interface,
        # Emitting has to be asked for. It was the default, and the observation
        # was stamped passive either way, so a deployment that never chose to
        # transmit transmitted anyway and could not tell from the evidence.
        # ``--passive-scan-only`` no longer changes anything; it is kept so
        # existing units keep working and keep meaning what they say.
        enable_scan=args.allow_active_scan and not args.passive_scan_only,
        enable_sdr=not args.no_sdr,
        enable_modem=not args.no_modem,
    )
    if not backends:
        raise SystemExit(
            "no real measurement backend is available (need iw/nmcli on a wireless "
            "interface, an attached RTL-SDR, or a ModemManager modem); refusing to "
            "serve observations"
        )

    adapter = SpectrumSensorAdapter(
        backends,
        adapter_id=args.adapter_id or f"kval-sensor@{socket.gethostname()}",
    )

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        args.socket.unlink()
    # Create the socket restrictive from the outset; the node client refuses a
    # socket that is group- or world-writable.
    previous_umask = os.umask(0o077)
    try:
        server = _Server(str(args.socket), _Handler)
    finally:
        os.umask(previous_umask)
    server.adapter = adapter  # type: ignore[attr-defined]
    server.policy = policy  # type: ignore[attr-defined]

    def shutdown(_signum: int, _frame: object) -> None:
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(
        json.dumps(
            {
                "status": "listening",
                "socket": str(args.socket),
                "policy_id": policy.policy_id,
                "policy_hash": policy_hash,
                "bands": sorted(policy.approved_band_ids()),
                "backends": [backend.name for backend in backends],
                "adapter_id": adapter.adapter_id,
                "conformance_hash": adapter.conformance_hash,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.socket.exists():
            args.socket.unlink()


if __name__ == "__main__":
    main()
