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

**Injection detection.** MemMorph's records are individually plausible but
collectively anomalous: generated from one template, so unusually similar to
each other, and *unlike the history they were inserted into*.

That last condition is what makes the test usable, and getting it wrong makes
the detector worse than useless. A node's ledger is *supposed* to repeat — a
link that stays healthy produces the same decision every cycle, and several
records land within the same second of one loop iteration. A detector that
flagged repetition, or flagged records arriving close together, would fire
constantly on a correctly-operating node and be switched off within a day. So
a group is flagged only when its members closely resemble **each other** and
are clearly distinct from **everything else** — the signature of a template
inserted into a history, rather than of a system doing the same thing twice.

The honest limit: this catches an outsider injecting from a template. It does
not catch a compromised node that holds the signing key and writes poisoned
records resembling its own normal output. Nothing at this layer could. The
backstop for that case is bounded influence, below, and reputation weighting
across the fleet.

**Bounded influence.** Even undetected, no small group of records may dominate
a derived statistic. The influence cap is the same principle the federated
aggregator uses against poisoning: a conclusion that a handful of records can
overturn was never adequately evidenced.
"""

from __future__ import annotations

import base64
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import utc_now
from .signing import canonical_json, public_key_id
from .validation import bounded_number, non_empty_str

#: Group size at or above which mutually near-duplicate records are treated as
#: a templated injection. MemMorph needs only three records, so the threshold
#: is deliberately low — which is affordable precisely because the group must
#: also be distinct from the surrounding history.
DEFAULT_GROUP_THRESHOLD = 3

#: Similarity at or above which two records are treated as near-duplicates.
#: Injected records resemble each other because one template generated them.
DEFAULT_SIMILARITY_THRESHOLD = 0.9

#: How much *less* similar a group must be to the rest of history than to
#: itself before it counts as foreign. Without this margin the test fires on a
#: node repeating a decision, which is what a healthy node does all day.
DEFAULT_NOVELTY_MARGIN = 0.35

#: How many times a typical record's influence one record may carry before it
#: is treated as dominating the conclusion. Measured against the *median*
#: record's influence rather than as an absolute share, because an absolute
#: share is not a meaningful bar: with ten records every record legitimately
#: carries about a tenth, so a fixed cap condemns healthy short histories and
#: exonerates long poisoned ones. Clean histories measure 1.5–3 across every
#: length and spread tested; a three-record injection measures 8–20.
DEFAULT_MAX_DISPROPORTION = 5.0


class MemoryIntegrityError(RuntimeError):
    """Raised when recorded history cannot be trusted as the node's own."""


@dataclass(frozen=True)
class SignedRecord:
    """A decision record bound to its author.

    ``record_type`` is inside the signed body, not merely alongside it. A
    signature that covered only the payload would let an attacker relabel a
    signed observation as, say, an activation, producing history the node never
    wrote out of parts it did.
    """

    node_id: str
    recorded_at: datetime
    record_type: str
    payload: dict[str, Any]
    signature: str
    key_id: str

    def body(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "recorded_at": self.recorded_at.isoformat(),
            "record_type": self.record_type,
            "payload": self.payload,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "signature": self.signature, "key_id": self.key_id}


def sign_record(
    payload: dict[str, Any],
    private_key: Ed25519PrivateKey,
    *,
    node_id: str,
    record_type: str = "decision",
    now: datetime | None = None,
) -> SignedRecord:
    recorded_at = now or utc_now()
    body = {
        "node_id": non_empty_str(node_id, "node_id"),
        "recorded_at": recorded_at.isoformat(),
        "record_type": non_empty_str(record_type, "record_type"),
        "payload": payload,
    }
    signature = private_key.sign(canonical_json(body))
    return SignedRecord(
        node_id=body["node_id"],
        recorded_at=recorded_at,
        record_type=body["record_type"],
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
        group_threshold: int = DEFAULT_GROUP_THRESHOLD,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        novelty_margin: float = DEFAULT_NOVELTY_MARGIN,
    ) -> None:
        self.group_threshold = max(2, group_threshold)
        self.similarity_threshold = bounded_number(
            similarity_threshold, "similarity_threshold", 0.0, 1.0
        )
        self.novelty_margin = bounded_number(novelty_margin, "novelty_margin", 0.0, 1.0)

    def _templated_groups(self, records: Sequence[SignedRecord]) -> list[InjectionFinding]:
        """Groups of mutually near-duplicate records unlike the rest of history.

        Both halves are required. Mutual similarity alone is satisfied by a
        node repeating a stable decision; distinctness alone is satisfied by
        any unusual event. Together they describe a template inserted into a
        history that did not contain it.
        """
        count = len(records)
        if count < self.group_threshold:
            return []

        similarity = [[0.0] * count for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                value = payload_similarity(records[i].payload, records[j].payload)
                similarity[i][j] = similarity[j][i] = value

        # Group members by mutual near-duplication. Transitive grouping is
        # adequate here: a template produces one tight cluster, not a chain.
        assigned: dict[int, int] = {}
        groups: list[list[int]] = []
        for i in range(count):
            if i in assigned:
                continue
            members = [i]
            assigned[i] = len(groups)
            for j in range(i + 1, count):
                if j not in assigned and similarity[i][j] >= self.similarity_threshold:
                    members.append(j)
                    assigned[j] = len(groups)
            groups.append(members)

        findings: list[InjectionFinding] = []
        for members in groups:
            if len(members) < self.group_threshold:
                continue
            outsiders = [index for index in range(count) if index not in members]
            if not outsiders:
                # Every record resembles every other: a node doing one thing
                # consistently, which is not evidence of anything.
                continue
            internal = statistics.fmean(
                similarity[a][b]
                for position, a in enumerate(members)
                for b in members[position + 1 :]
            )
            external = statistics.fmean(
                similarity[a][b] for a in members for b in outsiders
            )
            if internal - external < self.novelty_margin:
                continue
            span = (
                records[max(members)].recorded_at - records[min(members)].recorded_at
            ).total_seconds()
            findings.append(
                InjectionFinding(
                    kind="TEMPLATED_GROUP",
                    record_indices=tuple(members),
                    detail=(
                        f"{len(members)} records with internal similarity {internal:.3f} "
                        f"against {external:.3f} to the rest of history, spanning {span:.1f}s"
                    ),
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

        ordered = sorted(range(len(records)), key=lambda i: records[i].recorded_at)
        chronological = [records[i] for i in ordered]

        for index, record in enumerate(chronological):
            try:
                verify_record(record, public_key, node_id=node_id)
            except MemoryIntegrityError as exc:
                findings.append(
                    InjectionFinding(kind="UNAUTHENTICATED", record_indices=(index,), detail=str(exc))
                )

        findings.extend(self._templated_groups(chronological))

        if any(f.kind == "UNAUTHENTICATED" for f in findings):
            reasons.append("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE")
        if any(f.kind == "TEMPLATED_GROUP" for f in findings):
            reasons.append("A_GROUP_OF_RECORDS_LOOKS_TEMPLATED_AND_FOREIGN_TO_THIS_HISTORY")
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
    disproportion: float
    most_influential_index: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "within_bound": self.within_bound,
            "max_influence": round(self.max_influence, 6)
            if math.isfinite(self.max_influence)
            else None,
            "disproportion": round(self.disproportion, 6)
            if math.isfinite(self.disproportion)
            else None,
            "most_influential_index": self.most_influential_index,
            "reasons": list(self.reasons),
        }


def _leave_one_out_influences(values: Sequence[float]) -> list[float]:
    """Each record's leave-one-out shift, in units of the *remaining* spread.

    Measuring against the remaining spread rather than the total spread is the
    whole mechanism, not a detail. Normalising by the spread of *all* values
    lets the record under test inflate the denominator judging it, and the
    ratio then converges to ``1/sqrt(n-1)`` however extreme the record is: past
    seventeen records a single injected value of any magnitude at all — 10^12 —
    sat inside the old bound. That is the same feedback error the NLOS residual
    scale had, and it made this defence inert exactly where a real ledger would
    put it. The spread of the records that remain is something the tested
    record cannot touch.
    """
    full_mean = statistics.fmean(values)
    influences: list[float] = []
    for index in range(len(values)):
        without = list(values[:index]) + list(values[index + 1 :])
        remaining_spread = statistics.pstdev(without)
        shift = abs(statistics.fmean(without) - full_mean)
        if remaining_spread <= 0.0:
            # Every other record agrees exactly. Any disagreement from this one
            # is then total, and any agreement is no influence at all.
            influences.append(0.0 if shift <= 0.0 else math.inf)
        else:
            influences.append(shift / remaining_spread)
    return influences


def bounded_influence(
    values: Sequence[float],
    *,
    max_disproportion: float = DEFAULT_MAX_DISPROPORTION,
) -> InfluenceVerdict:
    """Check that no single record carries a disproportionate share of the mean.

    A conclusion one record can overturn was not adequately evidenced — exactly
    the leverage MemMorph buys by injecting a handful of records.

    The verdict rests on how the most influential record compares to the
    **typical** record, not on an absolute share. An absolute cap has no
    defensible value: with ten records every record legitimately carries about
    a tenth, so any cap tight enough to catch poisoning in a long history
    condemns every short one. The ratio is stable across history length and
    across how much a node's readings naturally vary.

    This bounds *single-record* influence and nothing more. A group of injected
    records covers for each other under a leave-one-out test — remove one and
    the rest still hold the mean — so groups are the templated-group detector's
    job, not this one's. The two are meant to be used together.
    """
    if len(values) < 3:
        return InfluenceVerdict(
            within_bound=False,
            max_influence=math.inf,
            disproportion=math.inf,
            most_influential_index=None,
            reasons=("TOO_FEW_RECORDS_TO_BOUND_INFLUENCE",),
        )

    influences = _leave_one_out_influences(values)
    worst = max(influences)
    worst_index = influences.index(worst)
    typical = statistics.median(influences)

    if worst <= 0.0:
        return InfluenceVerdict(
            True, 0.0, 0.0, None, ("NO_VARIANCE_SO_NO_RECORD_DOMINATES",)
        )
    if typical <= 0.0:
        # The typical record moves the mean not at all while one moves it: that
        # record is carrying the conclusion by itself.
        return InfluenceVerdict(
            False,
            worst,
            math.inf,
            worst_index,
            ("ONE_RECORD_CARRIES_THE_CONCLUSION_AND_THE_REST_CARRY_NONE",),
        )

    disproportion = worst / typical
    within = disproportion <= max_disproportion
    reasons = (
        ("NO_SINGLE_RECORD_DOMINATES",)
        if within
        else ("A_SINGLE_RECORD_MOVES_THE_CONCLUSION_TOO_FAR",)
    )
    return InfluenceVerdict(within, worst, disproportion, worst_index, reasons)
