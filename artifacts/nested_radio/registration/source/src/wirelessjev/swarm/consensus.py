"""Serverless swarm merge with validation before release.

Follows the swarm-learning DSA protocol of Wang, Shen, Li & Lam (WCNC 2025)
and Li, Wang, Lam, Shen & Wang (IEEE IoT-J 2025):
1. admission: a node may join only if it authenticates and runs the agreed
   algorithm/aggregation versions;
2. sharing starts once enough nodes are ready;
3. leader election by performance score P (the first to finish raises P);
4. the leader merges updates with FedAvg (weighted by sample count here);
5. the merged model is validated against a target before it is released;
   a failed validation is rejected and the previous release stays live;
6. every step is recorded in an auditable log.

The papers use a permissioned blockchain for (6). Ten cars on one site do not
need consensus over adversarial ledgers, so we use an append-only,
hash-chained, HMAC-signed log instead and say so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import time
from typing import Callable

import torch

ALGORITHM_VERSION = "wirelessjev.readout-kl.v1"
AGGREGATION_VERSION = "fedavg-weighted.v1"


@dataclass
class SwarmNode:
    node_id: str
    key: bytes
    algorithm: str = ALGORITHM_VERSION
    aggregation: str = AGGREGATION_VERSION
    score: float = 0.0              # performance score P
    reputation: float = 1.0


@dataclass
class MergeResult:
    released: bool
    leader: str | None
    participants: list[str]
    validation: dict
    reason: str
    readout: torch.Tensor | None = None


@dataclass
class SwarmConsensus:
    site_key: bytes
    min_ready: int = 3
    min_reputation: float = 0.5
    nodes: dict[str, SwarmNode] = field(default_factory=dict)
    log: list[dict] = field(default_factory=list)
    released_readout: torch.Tensor | None = None

    # --- audit log ---------------------------------------------------------
    def _append(self, event: dict) -> None:
        prev = self.log[-1]["hash"] if self.log else "0" * 64
        body = json.dumps({"prev": prev, "t": round(time.time(), 3), **event},
                          sort_keys=True, default=str)
        digest = hashlib.sha256(body.encode()).hexdigest()
        sig = hmac.new(self.site_key, digest.encode(), hashlib.sha256).hexdigest()
        self.log.append({"body": body, "hash": digest, "sig": sig})

    def verify_log(self) -> bool:
        prev = "0" * 64
        for entry in self.log:
            body = json.loads(entry["body"])
            if body["prev"] != prev:
                return False
            if hashlib.sha256(entry["body"].encode()).hexdigest() != entry["hash"]:
                return False
            expect = hmac.new(self.site_key, entry["hash"].encode(),
                              hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expect, entry["sig"]):
                return False
            prev = entry["hash"]
        return True

    # --- membership ----------------------------------------------------------
    def admit(self, node: SwarmNode, challenge: bytes, response: str) -> bool:
        expected = hmac.new(node.key, challenge, hashlib.sha256).hexdigest()
        ok = (hmac.compare_digest(expected, response)
              and node.algorithm == ALGORITHM_VERSION
              and node.aggregation == AGGREGATION_VERSION)
        if ok:
            self.nodes[node.node_id] = node
        self._append({"event": "admit", "node": node.node_id, "ok": ok})
        return ok

    # --- one merge round ----------------------------------------------------
    def merge_round(self, updates: dict[str, tuple[torch.Tensor, int]],
                    finish_order: list[str],
                    validate: Callable[[torch.Tensor], dict]) -> MergeResult:
        """updates: node_id -> (readout, n_samples). validate(readout) must
        return {"passed": bool, ...metrics}."""
        eligible = [n for n in updates
                    if n in self.nodes and self.nodes[n].reputation >= self.min_reputation]
        excluded = sorted(set(updates) - set(eligible))
        if len(eligible) < self.min_ready:
            self._append({"event": "round_skipped", "ready": len(eligible),
                          "excluded": excluded})
            return MergeResult(False, None, eligible, {}, "NOT_ENOUGH_READY_NODES")

        # Leader: first eligible finisher raises its score; highest score leads.
        for node_id in finish_order:
            if node_id in eligible:
                self.nodes[node_id].score += 1.0
                break
        leader = max(eligible, key=lambda n: (self.nodes[n].score, n))

        total = sum(updates[n][1] for n in eligible)
        merged = sum(updates[n][0].float() * (updates[n][1] / total) for n in eligible)
        validation = validate(merged)
        released = bool(validation.get("passed"))
        if released:
            self.released_readout = merged
        self._append({"event": "merge", "leader": leader, "participants": eligible,
                      "excluded": excluded, "released": released,
                      "validation": validation})
        return MergeResult(released, leader, eligible, validation,
                           "RELEASED" if released else "VALIDATION_FAILED",
                           merged if released else None)
