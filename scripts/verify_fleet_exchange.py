#!/usr/bin/env python3
"""Prove, on the deployed nodes, that peer reports are trusted for a reason.

Two nodes exchanging signed reports is easy to show and proves very little: a
node that accepted everything would look identical. What distinguishes a fabric
from a shared folder is what it *refuses*, so this runs the refusals against the
real roster on the real machine and records the verdicts.

Nothing here fabricates evidence. The genuine report is whatever the peer's own
sensor produced — including nothing, if that node has no spectrum hardware, which
is a true statement about the deployment and not a gap to be papered over. What
is constructed is the *attacks*: a flipped field, a stranger's signature, a
replayed timestamp. Those are adversarial inputs to a security test, which is the
one thing that has to be manufactured to be tested at all.

Run it on the receiving node, against a report that actually arrived over the
network:

    python3 scripts/verify_fleet_exchange.py \\
        --config /etc/kval/node.json \\
        --report /var/lib/kval/fabric-inbox/from-node-02.json \\
        --evidence /var/lib/kval/evidence/exchange-$(date -u +%Y%m%dT%H%M%SZ).json

Exits non-zero if a hostile variant is accepted or the genuine report refused,
so it can gate a deployment rather than merely describe one.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from kval_node.capabilities import CapabilityServices  # noqa: E402
from kval_node.config import NodeConfig  # noqa: E402
from kval_node.fabric import FabricReport, PeerRoster, sign_report, verify_report  # noqa: E402
from kval_node.models import utc_now  # noqa: E402

#: How far past the correlation window the replay case moves the clock. Any
#: value beyond the window proves the same point; this one is far enough that a
#: slow run cannot land back inside it and turn a pass into a flake.
REPLAY_OFFSET_SECONDS = 3600.0


def _hostile_variants(
    genuine: dict[str, Any], roster: PeerRoster
) -> list[tuple[str, dict[str, Any], dict[str, Any], str]]:
    """Each case: (label, envelope, verify kwargs, what accepting it would mean)."""
    stranger = Ed25519PrivateKey.generate()
    author = str(genuine["body"]["node_id"])

    tampered = copy.deepcopy(genuine)
    tampered["body"]["policy_hash"] = "0" * 64

    # The one that matters most: clock_trusted is what decides whether a report
    # may corroborate. A node that could set it would be able to promote its own
    # unsynchronised word into evidence two nodes agreed on.
    clock_claim = copy.deepcopy(genuine)
    clock_claim["body"]["clock_trusted"] = not genuine["body"]["clock_trusted"]

    def resigned(node_id: str) -> dict[str, Any]:
        body = copy.deepcopy(genuine["body"])
        body["node_id"] = node_id
        report = FabricReport.from_dict(body)
        return sign_report(report, stranger)

    unknown_id = "kval-node-not-on-this-roster"
    while unknown_id in roster.by_id():  # pragma: no cover - defensive
        unknown_id += "-x"

    return [
        (
            "tampered_payload", tampered, {},
            "a peer's readings could be rewritten in flight",
        ),
        (
            "clock_trust_flipped", clock_claim, {},
            "an undisciplined node could claim the right to corroborate",
        ),
        (
            "replayed_report", genuine,
            {"now": utc_now() + timedelta(seconds=REPLAY_OFFSET_SECONDS)},
            "yesterday's quiet band could be replayed over today's jamming",
        ),
        (
            "author_not_on_roster", resigned(unknown_id), {},
            "any machine that can write to the inbox could join the fleet",
        ),
        (
            "roster_peer_impersonated", resigned(author), {},
            "a peer's identity could be worn by whoever holds any key",
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify peer-report enforcement against this node's real roster",
    )
    parser.add_argument("--config", required=True, type=Path, help="this node's config")
    parser.add_argument(
        "--report", required=True, type=Path,
        help="a signed report that actually arrived from a peer",
    )
    parser.add_argument("--evidence", type=Path, default=None, help="write the evidence bundle here")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = NodeConfig.load(args.config)
    services = CapabilityServices(config)

    # The node's own loader, so this cannot drift from what the node believes.
    roster, roster_hash = services.load_peer_roster()
    genuine = json.loads(args.report.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    try:
        accepted = verify_report(genuine, roster)
        results.append({
            "case": "genuine_peer_report",
            "expected": "accept",
            "outcome": "accepted",
            "author": accepted.node_id,
            "clock_trusted": accepted.clock_trusted,
            "bands_reported": len(accepted.bands),
        })
    except Exception as exc:
        results.append({
            "case": "genuine_peer_report", "expected": "accept",
            "outcome": "refused", "detail": str(exc),
        })
        failures.append(f"the genuine report from the peer was refused: {exc}")
        # Everything below tests refusal. If the accept case is broken the
        # refusals prove nothing, so say so rather than reporting five passes.
        print(json.dumps({"failed": failures, "results": results}, indent=2), file=sys.stderr)
        return 1

    for label, envelope, kwargs, consequence in _hostile_variants(genuine, roster):
        try:
            verify_report(envelope, roster, **kwargs)
        except Exception as exc:
            results.append({
                "case": label, "expected": "refuse", "outcome": "refused", "detail": str(exc),
            })
        else:
            results.append({
                "case": label, "expected": "refuse", "outcome": "ACCEPTED",
                "consequence": consequence,
            })
            failures.append(f"{label} was accepted: {consequence}")

    bundle = {
        "schema": "kval-fleet-exchange-evidence/v1",
        "recorded_at": utc_now().isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "system": f"{platform.system()} {platform.release()}",
        },
        "node_id": config.node_id,
        "roster_id": roster.roster_id,
        "roster_hash": roster_hash,
        "roster_peers": sorted(roster.by_id()),
        "report_source": str(args.report),
        "results": results,
        "passed": not failures,
    }
    if failures:
        bundle["failures"] = failures

    if args.evidence:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(bundle, indent=2, sort_keys=True))
    if failures:
        print(f"\n{len(failures)} enforcement failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
