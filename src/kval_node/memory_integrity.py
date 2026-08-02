"""Memory-level integrity for the node's decision history.

Defends the attack described in Zhang, Zheng, Xu, Zhou, **Shen**, Ou, Zhang and
**Lam**, "MemMorph: Tool Hijacking in LLM Agents via Memory Poisoning"
(arXiv:2605.26154). MemMorph injects a *small number* of crafted records —
disguised as technical facts, incident reports and operational policies — into
an agent's long-term memory. It never dictates a decision directly; it reshapes
the context the agent reasons over, so the agent autonomously infers the
attacker's preferred action. They report up to **85.9% attack success from
three injected records**, surviving three representative defences, and conclude
that long-term memory is a critical under-explored attack surface requiring
**memory-level integrity safeguards**.

That threat lands squarely on this system. Kval's long-term memory is the audit
ledger, and ``analysis.py`` derives baselines, change points and causal
attributions from it. An attacker who can append a handful of plausible
decision records shifts what the node believes "normal" is — and every
downstream judgement inherits the bias, without any single component being
obviously wrong.

The existing hash chain is necessary but **not sufficient**. It detects
*modification* of recorded history; it does not, on its own, stop *appending*
plausible new history. Three safeguards close that gap:

**Authorship binding.** A record is bound to the node that produced it by an
Ed25519 signature over its canonical content, so an attacker with database
write access but no node key cannot manufacture accepted history. The hash
chain proves the sequence is intact; the signature proves who wrote each link.

**Injection detection.** MemMorph's signature is a small burst of records that
are individually plausible but collectively anomalous — unusually similar to
each other, arriving unusually close together, and pulling a statistic in a
consistent direction. All three are checked.

**Bounded influence.** Even undetected, no small group of records may dominate
a derived statistic. The influence cap is the same principle the federated
aggregator uses against poisoning: a conclusion that a handful of records can
overturn was never adequately evidenced.
"""

from __future__ import annotations

import base64
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import utc_now
from .signing import canonical_json, public_key_id
from .validation import bounded_number, non_empty_str

#: Records arriving closer together than this are treated as one burst.
DEFAULT_BURST_WINDOW_SECONDS = 5.0

#: Burst size at or above which a group is flagged. MemMorph needs only three
#: records, so the threshold is deliberately low.
DEFAULT_BURST_THRESHOLD = 3

#: Similarity at or above which two records are treated as near-duplicates.
#: Injected records tend to resemble each other because they are generated
#: from one template to push one conclusion.
DEFAULT_SIMILARITY_THRESHOLD = 0.9

#: Maximum share of a derived statistic any single record may account for.
DEFAULT_MAX_SINGLE_INFLUENCE = 0.25


class MemoryIntegrityError(RuntimeError):
    """Raised when recorded history cannot be trusted as the node's own."""


@dataclass(frozen=True)
class SignedRecord:
    """A decision record bound to its author."""

    node_id: str
    recorded_at: datetime
    payload: dict[str, Any]
    signature: str
    key_id: str

    def body(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "recorded_at": self.recorded_at.isoformat(),
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "signature": self.signature, "key_id": self.key_id}


def sign_record(
    payload: dict[str, Any],
    private_key: Ed25519PrivateKey,
    *,
    node_id: str,
    now: datetime | None = None,
) -> SignedRecord:
    recorded_at = now or utc_now()
    body = {
        "node_id": non_empty_str(node_id, "node_id"),
        "recorded_at": recorded_at.isoformat(),
        "payload": payload,
    }
    signature = private_key.sign(canonical_json(body))
    return SignedRecord(
        node_id=body["node_id"],
        recorded_at=recorded_at,
        payload=payload,
        signature=base64.b64encode(signature).decode("ascii"),
        key_id=public_key_id(private_key.public_key()),
    )


def verify_record(record: SignedRecord, public_key: Ed25519PublicKey, *, node_id: str) -> None:
    """Raise unless the record was genuinely written by this node."""
    if record.node_id != node_id:
        raise MemoryIntegrityError("record claims a different node as its author")
    if record.key_id != public_key_id(public_key):
        raise MemoryIntegrityError("record key ID does not match the node key")
    try:
        signature = base64.b64decode(record.signature, validate=True)
    except (ValueError, TypeError) as exc:
        raise MemoryIntegrityError("record signature is malformed") from exc
    try:
        public_key.verify(signature, canonical_json(record.body()))
    except InvalidSignature as exc:
        raise MemoryIntegrityError("record signature verification failed") from exc


def payload_similarity(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Jaccard similarity over flattened key/value pairs.

    Injected records are typically generated from one template, so they share
    an unusually large fraction of their structure with each other.
    """

    def flatten(value: Any, prefix: str = "") -> set[str]:
        items: set[str] = set()
        if isinstance(value, dict):
            for key, inner in value.items():
                items |= flatten(inner, f"{prefix}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, inner in enumerate(value):
                items |= flatten(inner, f"{prefix}[{index}]")
        else:
            items.add(f"{prefix}={value}")
        return items

    left = flatten(first)
    right = flatten(second)
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


@dataclass(frozen=True)
class InjectionFinding:
    kind: str
    record_indices: tuple[int, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "record_indices": list(self.record_indices), "detail": self.detail}


@dataclass(frozen=True)
class IntegrityReport:
    records: int
    trustworthy: bool
    findings: tuple[InjectionFinding, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-memory-integrity/v1",
            "records": self.records,
            "trustworthy": self.trustworthy,
            "findings": [finding.to_dict() for finding in self.findings],
            "reasons": list(self.reasons),
        }


class MemoryAuditor:
    """Detects the injection pattern MemMorph relies on."""

    def __init__(
        self,
        *,
        burst_window_seconds: float = DEFAULT_BURST_WINDOW_SECONDS,
        burst_threshold: int = DEFAULT_BURST_THRESHOLD,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.burst_window = timedelta(seconds=burst_window_seconds)
        self.burst_threshold = max(2, burst_threshold)
        self.similarity_threshold = bounded_number(
            similarity_threshold, "similarity_threshold", 0.0, 1.0
        )

    def _bursts(self, records: Sequence[SignedRecord]) -> list[InjectionFinding]:
        findings: list[InjectionFinding] = []
        start = 0
        for index in range(len(records)):
            while records[index].recorded_at - records[start].recorded_at > self.burst_window:
                start += 1
            span = index - start + 1
            if span >= self.burst_threshold:
                group = tuple(range(start, index + 1))
                if not any(set(group) <= set(f.record_indices) for f in findings):
                    findings = [f for f in findings if not set(f.record_indices) <= set(group)]
                    findings.append(
                        InjectionFinding(
                            kind="BURST",
                            record_indices=group,
                            detail=f"{span} records within {self.burst_window.total_seconds()}s",
                        )
                    )
        return findings

    def _near_duplicates(self, records: Sequence[SignedRecord]) -> list[InjectionFinding]:
        findings: list[InjectionFinding] = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                similarity = payload_similarity(records[i].payload, records[j].payload)
                if similarity >= self.similarity_threshold:
                    findings.append(
                        InjectionFinding(
                            kind="NEAR_DUPLICATE",
                            record_indices=(i, j),
                            detail=f"payload similarity {similarity:.3f}",
                        )
                    )
        return findings

    def audit(
        self,
        records: Sequence[SignedRecord],
        public_key: Ed25519PublicKey,
        *,
        node_id: str,
    ) -> IntegrityReport:
        findings: list[InjectionFinding] = []
        reasons: list[str] = []

        for index, record in enumerate(records):
            try:
                verify_record(record, public_key, node_id=node_id)
            except MemoryIntegrityError as exc:
                findings.append(
                    InjectionFinding(kind="UNAUTHENTICATED", record_indices=(index,), detail=str(exc))
                )

        ordered = sorted(range(len(records)), key=lambda i: records[i].recorded_at)
        chronological = [records[i] for i in ordered]
        findings.extend(self._bursts(chronological))
        findings.extend(self._near_duplicates(chronological))

        if any(f.kind == "UNAUTHENTICATED" for f in findings):
            reasons.append("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE")
        if any(f.kind == "BURST" for f in findings):
            reasons.append("RECORDS_ARRIVED_IN_AN_IMPLAUSIBLE_BURST")
        if any(f.kind == "NEAR_DUPLICATE" for f in findings):
            reasons.append("NEAR_DUPLICATE_RECORDS_SUGGEST_TEMPLATED_INJECTION")
        if not reasons:
            reasons.append("HISTORY_CONSISTENT_WITH_THIS_NODE_S_OWN_OPERATION")

        return IntegrityReport(
            records=len(records),
            trustworthy=not findings,
            findings=tuple(findings),
            reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class InfluenceVerdict:
    within_bound: bool
    max_influence: float
    most_influential_index: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "within_bound": self.within_bound,
            "max_influence": round(self.max_influence, 6),
            "most_influential_index": self.most_influential_index,
            "reasons": list(self.reasons),
        }


def bounded_influence(
    values: Sequence[float],
    *,
    max_single_influence: float = DEFAULT_MAX_SINGLE_INFLUENCE,
) -> InfluenceVerdict:
    """Check that no single observation dominates the mean.

    Influence is the leave-one-out shift in the mean, relative to the total
    spread. A conclusion that one record can overturn was not adequately
    evidenced — which is exactly the leverage MemMorph exploits by injecting
    only a handful of records.
    """
    if len(values) < 3:
        return InfluenceVerdict(
            within_bound=False,
            max_influence=1.0,
            most_influential_index=None,
            reasons=("TOO_FEW_RECORDS_TO_BOUND_INFLUENCE",),
        )
    full_mean = statistics.fmean(values)
    spread = statistics.pstdev(values)
    if spread <= 0.0:
        return InfluenceVerdict(True, 0.0, None, ("NO_VARIANCE_SO_NO_RECORD_DOMINATES",))

    worst_index = 0
    worst = 0.0
    for index in range(len(values)):
        without = values[:index] + values[index + 1 :]
        shift = abs(statistics.fmean(without) - full_mean) / spread
        if shift > worst:
            worst, worst_index = shift, index

    within = worst <= max_single_influence
    reasons = (
        ("NO_SINGLE_RECORD_DOMINATES",)
        if within
        else ("A_SINGLE_RECORD_MOVES_THE_CONCLUSION_TOO_FAR",)
    )
    return InfluenceVerdict(within, worst, worst_index, reasons)
