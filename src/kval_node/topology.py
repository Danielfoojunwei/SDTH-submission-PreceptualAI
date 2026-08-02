"""Graph-structured propagation of spectrum evidence across the fabric.

Adapted from Zhang, **Lam**, **Shen**, Wang and **Feng Li**'s graph-model line of
DSA work — "Dynamic spectrum access for Internet-of-Things with graph neural
network and deep Q-network", Ad Hoc Networks 163:103596, 2024,
[10.1016/j.adhoc.2024.103596](https://doi.org/10.1016/j.adhoc.2024.103596) —
whose contribution is that the spectrum environment is **relational**: what a
node should believe depends on its neighbours, and a graph carries that
structure where a flat feature vector cannot.

**What is taken, and what is not.** Their message passing is *learned* — a GNN
whose weights are trained, feeding a DQN that chooses a transmit action. Neither
half ships here. No model ships at all, and the action half is transmit-side.
What transfers is the graph itself and deterministic message passing over it,
which is what actually delivers the capability this fleet needs:

**A node whose own sensing is down still gets a view.** That is the fabric's
whole premise — eyes on the ground when a node's own eyes are shut. A node with
a failed radio, a saturated front end, or a band it physically cannot tune is
not blind if a neighbour can see; it inherits the neighbour's belief, weakened
by distance, and knows that is what happened.

Three properties keep inherited belief honest:

**Evidence weakens with every hop, and is never manufactured.** Propagation uses
Shafer's discounting: a fraction of a neighbour's committed mass is moved to Θ
(ignorance) per hop, never redistributed to the opposite hypothesis. A node two
hops from the only observer ends up *less* certain than the observer, never more
and never differently. Ignorance is the sink, so an unreachable node ends at
total ignorance rather than a default of "clear".

**Belief does not circulate.** Naive iterative fusion double-counts: A tells B,
B tells C, C tells A, and the fleet talks itself into certainty from one
observation. Each node's inherited belief is computed from the shortest path to
each *original observer*, so every observation contributes exactly once no
matter how densely the graph is connected.

**The graph's own fragility is reported.** A fabric whose connectivity depends
on one node is one capture away from partition. Articulation points and
components are computed from the same graph, so an operator sees which node's
loss splits the fleet before it happens rather than afterwards.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .evidence import BeliefMass, fuse
from .geolocate import EARTH_RADIUS_M, NodePosition
from .validation import bounded_number, non_empty_str

#: Fraction of a neighbour's committed belief retained per hop. The remainder
#: moves to ignorance. 0.6 means second-hand evidence carries a little over half
#: the weight of first-hand, and third-hand about a third.
DEFAULT_HOP_RETENTION = 0.6

#: Hops beyond which inherited evidence is not worth carrying. At the default
#: retention this is already below 8% of the original commitment.
DEFAULT_MAX_HOPS = 4


class TopologyError(ValueError):
    """Raised when the declared fabric graph cannot be used."""


@dataclass(frozen=True)
class FleetGraph:
    """Undirected adjacency over node IDs.

    Links are *declared*, not inferred from traffic: a node claiming to be
    adjacent to another is a spoofing surface, so the graph comes from the
    signed roster the same way node positions do.
    """

    nodes: tuple[str, ...]
    links: tuple[frozenset[str], ...]

    @classmethod
    def from_pairs(cls, pairs: Iterable[Sequence[str]]) -> "FleetGraph":
        nodes: list[str] = []
        links: list[frozenset[str]] = []
        for pair in pairs:
            if len(pair) != 2:
                raise TopologyError("each link must name exactly two nodes")
            left = non_empty_str(pair[0], "node_id")
            right = non_empty_str(pair[1], "node_id")
            if left == right:
                raise TopologyError(f"node {left!r} cannot link to itself")
            for node in (left, right):
                if node not in nodes:
                    nodes.append(node)
            link = frozenset({left, right})
            if link not in links:
                links.append(link)
        return cls(tuple(sorted(nodes)), tuple(links))

    @classmethod
    def from_positions(
        cls, positions: Sequence[NodePosition], *, link_range_m: float
    ) -> "FleetGraph":
        """Link every pair within radio range of each other.

        A range-based graph is an *upper bound* on connectivity: it assumes
        line of sight and says nothing about terrain, obstruction or
        interference. It is a planning aid, and a link that does not carry
        traffic in practice must be removed from the roster rather than
        inferred away here.
        """
        limit = bounded_number(link_range_m, "link_range_m", 1.0, 1_000_000.0)
        pairs: list[tuple[str, str]] = []
        for index, first in enumerate(positions):
            for second in positions[index + 1 :]:
                if _metres_between(first, second) <= limit:
                    pairs.append((first.node_id, second.node_id))
        graph = cls.from_pairs(pairs)
        # Isolated nodes have no links, so from_pairs never sees them. They are
        # still part of the fleet and must appear, unreachable, in the result.
        declared = {position.node_id for position in positions}
        missing = sorted(declared - set(graph.nodes))
        return cls(tuple(sorted(set(graph.nodes) | set(missing))), graph.links)

    def neighbours(self, node_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(next(iter(link - {node_id})) for link in self.links if node_id in link)
        )

    def hop_distances(self, source: str) -> dict[str, int]:
        """Breadth-first hop count from ``source`` to every reachable node."""
        if source not in self.nodes:
            raise TopologyError(f"node {source!r} is not in the fleet graph")
        distances = {source: 0}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbour in self.neighbours(current):
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1
                    queue.append(neighbour)
        return distances

    def components(self) -> tuple[tuple[str, ...], ...]:
        seen: set[str] = set()
        found: list[tuple[str, ...]] = []
        for node in self.nodes:
            if node in seen:
                continue
            reachable = self.hop_distances(node)
            seen |= reachable.keys()
            found.append(tuple(sorted(reachable)))
        return tuple(found)

    def articulation_points(self) -> tuple[str, ...]:
        """Nodes whose loss increases the number of components.

        Computed by removal rather than by Tarjan's algorithm: fleets here are
        tens of nodes, and the direct definition is the one an operator can
        check against the result.
        """
        if len(self.nodes) <= 2:
            return ()
        baseline = len(self.components())
        critical: list[str] = []
        for node in self.nodes:
            reduced = FleetGraph(
                nodes=tuple(n for n in self.nodes if n != node),
                links=tuple(link for link in self.links if node not in link),
            )
            if not reduced.nodes:
                continue
            if len(reduced.components()) > baseline:
                critical.append(node)
        return tuple(critical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "links": [sorted(link) for link in sorted(self.links, key=sorted)],
            "components": [list(component) for component in self.components()],
            "articulation_points": list(self.articulation_points()),
        }


def _metres_between(first: NodePosition, second: NodePosition) -> float:
    cos_lat = math.cos(math.radians((first.latitude_deg + second.latitude_deg) / 2.0))
    east = math.radians(second.longitude_deg - first.longitude_deg) * EARTH_RADIUS_M * cos_lat
    north = math.radians(second.latitude_deg - first.latitude_deg) * EARTH_RADIUS_M
    return math.hypot(east, north)


def discount(mass: BeliefMass, retention: float) -> BeliefMass:
    """Shafer discounting: move ``1 - retention`` of committed mass to Θ.

    The direction matters. Weakening evidence must *increase ignorance*, never
    transfer belief to the opposite hypothesis — "my neighbour thinks this band
    is busy, but they are far away" is not evidence the band is clear.
    """
    kept = bounded_number(retention, "retention", 0.0, 1.0)
    return BeliefMass(
        clear=mass.clear * kept,
        busy=mass.busy * kept,
        unknown=1.0 - (mass.clear + mass.busy) * kept,
    )


#: Conflict K above which neighbours are treated as materially disagreeing.
#: Lower than the fusion module's own 0.85 ceiling, and deliberately so:
#: discounting caps how much conflict inherited evidence can ever express, so a
#: threshold set for first-hand sources would almost never fire here.
DEFAULT_NEIGHBOUR_CONFLICT = 0.4


@dataclass(frozen=True)
class PropagatedBelief:
    node_id: str
    belief: BeliefMass
    conflict: float
    hops_to_nearest_observer: int | None
    observers_used: tuple[str, ...]
    first_hand: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "belief": self.belief.to_dict(),
            "conflict": round(self.conflict, 6),
            "hops_to_nearest_observer": self.hops_to_nearest_observer,
            "observers_used": list(self.observers_used),
            "first_hand": self.first_hand,
            "reasons": list(self.reasons),
        }


class EvidencePropagator:
    """Gives every node a view of a band, including nodes that cannot sense it.

    Each node's belief is the Dempster combination of every observation in its
    component, discounted by the shortest hop distance to that observation's
    author. Shortest-path attribution is what stops belief circulating: a
    triangle of nodes cannot amplify one observation into three.
    """

    def __init__(
        self,
        graph: FleetGraph,
        *,
        hop_retention: float = DEFAULT_HOP_RETENTION,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_neighbour_conflict: float = DEFAULT_NEIGHBOUR_CONFLICT,
    ) -> None:
        self.graph = graph
        self.hop_retention = bounded_number(hop_retention, "hop_retention", 0.0, 1.0)
        self.max_hops = max(0, int(max_hops))
        self.max_neighbour_conflict = bounded_number(
            max_neighbour_conflict, "max_neighbour_conflict", 0.0, 1.0
        )

    def propagate(self, observations: dict[str, BeliefMass]) -> tuple[PropagatedBelief, ...]:
        unknown_nodes = sorted(set(observations) - set(self.graph.nodes))
        if unknown_nodes:
            raise TopologyError(
                f"observation from node(s) not in the fleet graph: {', '.join(unknown_nodes)}"
            )
        # Distances from each observer outward, computed once.
        reach = {
            observer: self.graph.hop_distances(observer)
            for observer, mass in observations.items()
            if mass.unknown < 1.0
        }

        results: list[PropagatedBelief] = []
        for node in self.graph.nodes:
            contributions: list[BeliefMass] = []
            used: list[str] = []
            nearest: int | None = None
            for observer, distances in sorted(reach.items()):
                hops = distances.get(node)
                if hops is None or hops > self.max_hops:
                    continue
                nearest = hops if nearest is None else min(nearest, hops)
                contributions.append(
                    discount(observations[observer], self.hop_retention**hops)
                )
                used.append(observer)

            reasons: list[str] = []
            conflict = 0.0
            if not contributions:
                belief = BeliefMass(0.0, 0.0, 1.0)
                reasons.append(
                    "NO_OBSERVER_WITHIN_REACH_SO_TOTAL_IGNORANCE"
                    if not reach
                    else "UNREACHABLE_FROM_ANY_OBSERVER_SO_TOTAL_IGNORANCE"
                )
            else:
                fused = fuse(contributions)
                belief = BeliefMass(fused.clear, fused.busy, fused.unknown)
                conflict = fused.conflict
                if not fused.trustworthy:
                    reasons.append("NEIGHBOUR_EVIDENCE_IRRECONCILABLY_CONFLICTED")
                elif conflict > self.max_neighbour_conflict:
                    reasons.append("NEIGHBOURS_MATERIALLY_DISAGREE_ABOUT_THIS_BAND")

            first_hand = node in reach
            if not first_hand and contributions:
                reasons.append(f"INHERITED_FROM_NEIGHBOURS_AT_{nearest}_HOPS")
            elif first_hand:
                reasons.append("OBSERVED_DIRECTLY")

            results.append(
                PropagatedBelief(
                    node_id=node,
                    belief=belief,
                    conflict=conflict,
                    hops_to_nearest_observer=nearest,
                    observers_used=tuple(used),
                    first_hand=first_hand,
                    reasons=tuple(reasons),
                )
            )
        return tuple(results)


@dataclass(frozen=True)
class FragilityReport:
    """Where the fabric breaks if one node is lost."""

    nodes: int
    components: tuple[tuple[str, ...], ...]
    articulation_points: tuple[str, ...]
    isolated: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "kval-fabric-fragility/v1",
            "nodes": self.nodes,
            "components": [list(component) for component in self.components],
            "articulation_points": list(self.articulation_points),
            "isolated": list(self.isolated),
            "reasons": list(self.reasons),
        }


def assess_fragility(graph: FleetGraph) -> FragilityReport:
    components = graph.components()
    critical = graph.articulation_points()
    isolated = tuple(node for node in graph.nodes if not graph.neighbours(node))
    reasons: list[str] = []
    if len(components) > 1:
        reasons.append("FABRIC_IS_ALREADY_PARTITIONED")
    if critical:
        reasons.append("LOSS_OF_A_SINGLE_NODE_WOULD_PARTITION_THE_FABRIC")
    if isolated:
        reasons.append("NODES_WITH_NO_LINKS_CANNOT_CONTRIBUTE_OR_INHERIT")
    if not reasons:
        reasons.append("FABRIC_TOLERATES_THE_LOSS_OF_ANY_SINGLE_NODE")
    return FragilityReport(
        nodes=len(graph.nodes),
        components=components,
        articulation_points=critical,
        isolated=isolated,
        reasons=tuple(reasons),
    )
