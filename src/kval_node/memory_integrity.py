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

**Injection detection.** This detector catches *templated* injection: records
individually plausible but collectively anomalous because they came from one
template, so unusually similar to each other and *unlike the history they were
inserted into*.

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

The larger limit, and the one this docstring previously got wrong by claiming
the opposite: **it does not catch MemMorph as that paper describes it.**
MemMorph's three records are "disguised as technical facts, incident reports,
and operational policies" — three *deliberately different* kinds, whose force
comes from appearing to corroborate one conclusion from independent
directions. Records like that are not similar to each other, so
``_templated_groups`` never groups them and nothing here fires. What is built
is keyed on the signature of a weaker, single-style variant.

``_convergent_groups`` is the second arm, keyed on the opposite signature:
records *dissimilar* to each other that deviate from the history's median in
the same direction across several numeric fields at once, written off the
node's cadence. Deviation is measured in median-absolute-deviation units, not
standard deviations, because the injected records sit inside the sample whose
spread is being measured and would inflate the very threshold meant to catch
them.

Both arms share the cadence gate, and for the same reason: a node whose
environment genuinely degrades produces many records deviating the same way,
and only the rate separates that from an insertion. The evasion is therefore
also shared and equally stated — an attacker who holds the signing key and
paces the injection to the node's own cadence defeats both.

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
#:
#: This was 0.9, which made the detector inert. Jaccard runs over ``key=value``
#: pairs, so a single differing value is punished twice — it leaves the
#: intersection *and* adds two terms to the union. A record needs nineteen
#: fields before one varying field still scores 0.9, and a decision record here
#: has six. A templated injection whose records differ only in a sequence
#: number scored 0.714 and never grouped at all, so the detector could not see
#: even a single-style templated injection. Measured, not reasoned — see the
#: timing gate below for why lowering this is safe. Note this fixed the
#: threshold, not the premise: multi-style injection still never groups.
DEFAULT_SIMILARITY_THRESHOLD = 0.65

#: How much faster than the node's own recording cadence a group must have been
#: written before it counts as foreign.
#:
#: Similarity alone cannot carry this detector, and that is not a tuning
#: problem. A node flapping between links under jamming — this system's normal
#: condition — produces exactly what an injection produces: small groups of
#: mutually similar records unlike the rest of history. Measured over ten
#: profiles at three cycles each, every one of the ten regimes matched the
#: injection on group size, share of history, internal similarity (0.714) and
#: external similarity (0.333) alike. No threshold separates them, so a
#: sensitive detector would fire on every node in contested spectrum.
#:
#: What does separate them is *rate*. A link change happens at the cadence the
#: node decides at; injected records are written into the store in one
#: operation. The cadence is taken from the node's own surrounding history
#: rather than configured, so the test calibrates itself to whatever loop
#: interval a deployment runs.
#:
#: Applied in both directions — see ``_written_off_the_node_s_cadence``.
#:
#: The evasion is real and stated: an attacker who holds the signing key, knows
#: the node's cadence, and is willing to space the injection to match it within
#: a factor of four defeats this test. That attacker is a long way past the
#: signature check that runs first, and has to pace an attack whose whole appeal
#: is that three records are enough.
DEFAULT_CADENCE_FACTOR = 4.0

#: How much *less* similar a group must be to the rest of history than to
#: itself before it counts as foreign. Without this margin the test fires on a
#: node repeating a decision, which is what a healthy node does all day.
DEFAULT_NOVELTY_MARGIN = 0.35

#: Fewest numeric fields on which a set of *dissimilar* records must all deviate
#: the same way before their agreement counts as convergence rather than
#: coincidence. With one field, any two records that happen to be above the
#: median "agree"; the whole signal is that independent-looking evidence points
#: one way across several quantities at once.
DEFAULT_CONVERGENCE_FIELDS = 2

#: How far from the median a field must sit, in units of the history's own
#: median absolute deviation, before that record counts as pushing on it.
#: Robust rather than a standard-deviation multiple, because the injected
#: records are in the sample whose spread is being measured, and would inflate
#: the very threshold meant to catch them.
DEFAULT_CONVERGENCE_DEVIATION_MADS = 3.0

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


def _flatten_payload(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a payload to dotted keys, keeping values typed.

    ``payload_similarity`` flattens to ``key=value`` *strings* because Jaccard
    needs set members. The convergence arm needs the numbers themselves, and
    both arms must read the same surface or a field could be invisible to one
    and decisive to the other.
    """
    items: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, inner in payload.items():
            items.update(_flatten_payload(inner, f"{prefix}.{key}"))
    elif isinstance(payload, (list, tuple)):
        for index, inner in enumerate(payload):
            items.update(_flatten_payload(inner, f"{prefix}[{index}]"))
    else:
        items[prefix] = payload
    return items


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
    """Detects *templated* injection. Not MemMorph's multi-style form — see above."""

    def __init__(
        self,
        *,
        group_threshold: int = DEFAULT_GROUP_THRESHOLD,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        convergence_fields: int = DEFAULT_CONVERGENCE_FIELDS,
        convergence_deviation_mads: float = DEFAULT_CONVERGENCE_DEVIATION_MADS,
        novelty_margin: float = DEFAULT_NOVELTY_MARGIN,
        cadence_factor: float = DEFAULT_CADENCE_FACTOR,
    ) -> None:
        self.group_threshold = max(2, group_threshold)
        self.convergence_fields = max(1, convergence_fields)
        self.convergence_deviation_mads = bounded_number(
            convergence_deviation_mads, "convergence_deviation_mads", 0.0, 100.0
        )
        self.similarity_threshold = bounded_number(
            similarity_threshold, "similarity_threshold", 0.0, 1.0
        )
        self.novelty_margin = bounded_number(novelty_margin, "novelty_margin", 0.0, 1.0)
        self.cadence_factor = bounded_number(cadence_factor, "cadence_factor", 1.0, 1000.0)

    def _written_off_the_node_s_cadence(
        self, records: Sequence[SignedRecord], members: Sequence[int], outsiders: Sequence[int]
    ) -> tuple[bool, float, float]:
        """Was this group written at a rate the node's own loop does not produce?

        Deliberately two-sided. A burst written in one operation is the obvious
        case, but a template drip-fed hours apart in a history recorded every
        few minutes did not come from the loop either — and treating only the
        fast side as suspicious would hand the attacker an evasion for the cost
        of waiting. What a link change has, and neither of those has, is the
        node's own cadence.

        The cadence comes from the records *outside* the group, so an injected
        burst cannot drag down the very baseline that judges it. Returns the
        verdict along with both rates so the finding can state them.
        """
        def median_gap(indices: Sequence[int]) -> float:
            times = sorted(records[i].recorded_at for i in indices)
            gaps = [
                (later - earlier).total_seconds()
                for earlier, later in zip(times, times[1:])
            ]
            return statistics.median(gaps) if gaps else math.nan

        cadence = median_gap(outsiders)
        internal = median_gap(members)
        if not math.isfinite(cadence) or not math.isfinite(internal) or cadence <= 0.0:
            # Too little surrounding history to calibrate against. Refusing to
            # judge is the honest answer; the novelty test alone is what fires
            # on every link change, which is why it cannot govern by itself.
            return False, internal, cadence
        too_fast = internal * self.cadence_factor < cadence
        too_slow = internal > cadence * self.cadence_factor
        return too_fast or too_slow, internal, cadence

    @staticmethod
    def _numeric_fields(records: Sequence[SignedRecord]) -> dict[str, list[float | None]]:
        """Numeric payload fields present in enough records to have a median.

        Flattened one level, matching ``payload_similarity``'s view of a record,
        so the two arms judge the same surface.
        """
        columns: dict[str, list[float | None]] = {}
        names: set[str] = set()
        for record in records:
            for key, value in _flatten_payload(record.payload).items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if math.isfinite(float(value)):
                    names.add(key)
        for name in sorted(names):
            column: list[float | None] = []
            for record in records:
                value = _flatten_payload(record.payload).get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    column.append(None)
                elif math.isfinite(float(value)):
                    column.append(float(value))
                else:
                    column.append(None)
            if sum(1 for item in column if item is not None) >= 3:
                columns[name] = column
        return columns

    def _convergent_groups(self, records: Sequence[SignedRecord]) -> list[InjectionFinding]:
        """Records that do *not* resemble each other yet push the same conclusion.

        This is the arm the templated test cannot cover, and the reason it is
        needed is that MemMorph's own description is the opposite of a template:
        three records "disguised as technical facts, incident reports, and
        operational policies", whose persuasive force comes precisely from
        looking like independent evidence converging on one answer. Mutual
        similarity is not merely absent from that attack — it is designed out of
        it, because three copies of the same claim are less convincing than
        three different kinds of claim agreeing.

        So the signature inverts. A group qualifies when its members are
        *dissimilar* to each other, deviate from the history's median in the
        *same direction* on several numeric fields at once, and were written off
        the node's own recording cadence. All three are required, and the third
        carries the same weight it does in the templated arm: a node whose
        environment genuinely degrades produces many records deviating the same
        way, and only the rate distinguishes that from an insertion.

        Deviation is measured in median-absolute-deviation units rather than
        standard deviations because the injected records sit inside the sample
        whose spread is being measured. A standard deviation would be inflated
        by the very records the test is looking for.
        """
        count = len(records)
        if count < self.group_threshold + 2:
            # Needs a surrounding history to have a median and a cadence at all.
            return []

        columns = self._numeric_fields(records)
        if not columns:
            return []

        # Direction of each record's departure from the history's own median,
        # per field, counted only when the departure is large in robust terms.
        directions: list[dict[str, int]] = [{} for _ in range(count)]
        for name, column in columns.items():
            present = [value for value in column if value is not None]
            if len(present) < 3:
                continue
            median = statistics.median(present)
            spread = statistics.median([abs(value - median) for value in present])
            if spread <= 0.0:
                continue
            floor = spread * self.convergence_deviation_mads
            for index, value in enumerate(column):
                if value is None:
                    continue
                delta = value - median
                if abs(delta) >= floor:
                    directions[index][name] = 1 if delta > 0 else -1

        candidates = [i for i in range(count) if len(directions[i]) >= self.convergence_fields]
        if len(candidates) < self.group_threshold:
            return []

        similarity: dict[tuple[int, int], float] = {}
        for a_pos, i in enumerate(candidates):
            for j in candidates[a_pos + 1 :]:
                similarity[(i, j)] = payload_similarity(records[i].payload, records[j].payload)

        findings: list[InjectionFinding] = []
        used: set[int] = set()
        for a_pos, seed in enumerate(candidates):
            if seed in used:
                continue
            members = [seed]
            for other in candidates[a_pos + 1 :]:
                if other in used:
                    continue
                pair = similarity[(seed, other)]
                if pair >= self.similarity_threshold:
                    # Similar records are the *other* arm's business. Letting
                    # them in here would double-report a template and would let
                    # a healthy node's repeated decisions form a group.
                    continue
                agreed = [
                    name
                    for name, sign in directions[seed].items()
                    if directions[other].get(name) == sign
                ]
                if len(agreed) >= self.convergence_fields:
                    members.append(other)
            if len(members) < self.group_threshold:
                continue

            shared = set(directions[members[0]])
            for index in members[1:]:
                shared &= {
                    name for name, sign in directions[index].items()
                    if directions[members[0]].get(name) == sign
                }
            if len(shared) < self.convergence_fields:
                continue

            outsiders = [index for index in range(count) if index not in members]
            if not outsiders:
                continue
            off_cadence, group_gap, node_gap = self._written_off_the_node_s_cadence(
                records, members, outsiders
            )
            if not off_cadence:
                continue

            used.update(members)
            findings.append(
                InjectionFinding(
                    kind="CONVERGENT_GROUP",
                    record_indices=tuple(members),
                    detail=(
                        f"{len(members)} mutually dissimilar records agree on "
                        f"{sorted(shared)} against this history's median, written at "
                        f"{group_gap:.3f}s spacing against a node cadence of {node_gap:.3f}s"
                    ),
                )
            )
        return findings

    def _templated_groups(self, records: Sequence[SignedRecord]) -> list[InjectionFinding]:
        """Groups of near-duplicate records, unlike the rest of history, written
        faster than the node records.

        All three are required, and the third is what makes the other two
        usable. Mutual similarity alone is satisfied by a node repeating a
        stable decision. Distinctness alone is satisfied by any unusual event —
        including an ordinary link change, which is a small group of similar
        records unlike everything around it and is therefore *indistinguishable
        from an injection* by similarity alone. Only the rate separates them: a
        link change arrives at the cadence the node decides at, an injection is
        written in one operation.
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
            off_cadence, group_gap, cadence = self._written_off_the_node_s_cadence(
                records, members, outsiders
            )
            if not off_cadence:
                # Similar, and unlike the rest of history, but arriving at the
                # rate the node itself records at. That is a link change, which
                # is what this node does for a living.
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
                        f"against {external:.3f} to the rest of history, spanning {span:.3f}s "
                        f"at {group_gap:.3f}s between records against a node cadence of "
                        f"{cadence:.1f}s"
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
        findings.extend(self._convergent_groups(chronological))

        if any(f.kind == "UNAUTHENTICATED" for f in findings):
            reasons.append("HISTORY_CONTAINS_RECORDS_THIS_NODE_DID_NOT_WRITE")
        if any(f.kind == "TEMPLATED_GROUP" for f in findings):
            reasons.append("A_GROUP_OF_RECORDS_LOOKS_TEMPLATED_AND_FOREIGN_TO_THIS_HISTORY")
        if any(f.kind == "CONVERGENT_GROUP" for f in findings):
            reasons.append("DISSIMILAR_RECORDS_PUSH_THE_SAME_CONCLUSION_OFF_THIS_NODE_S_CADENCE")
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
