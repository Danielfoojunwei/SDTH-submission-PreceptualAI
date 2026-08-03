# Two-node validation on physically separate hardware

This closes the gap `DEFENCE_STANDARDS.md` ranks last and calls the largest
evidential one: *"No cross-node result on physically separate hardware."*
Everything cross-node — corroborated sensing, emitter localisation, cohort
clustering, federated statistics, quorum renewal — has only ever run in one
process on one machine.

Nothing in the code needed to change for this. The fabric has no transport by
design: a node writes a signed report, and the report reaches its peers by
whatever means the deployment already trusts. What was missing was the ability
to *provision* two nodes at all, since the roster naming both and carrying their
public keys had to be hand-authored with PEM blocks pasted into JSON.

## What this proves, and what it does not

It proves that two nodes on separate machines exchange signed spectrum reports
over a real network, and that each verifies the other against an operator-signed
roster — accepting a genuine report and refusing a tampered, replayed,
off-roster or wrong-key one.

It does **not** prove that one node's measurement changes the other node's
decision, because no code path does that. The autonomous loop runs on
`FlightServices`, which has no access to the fabric at all: `aggregate_fleet` is
reachable only from the CLI, the `degraded_band_ids` that reach the optimizer
come solely from this node's own `sense_spectrum`, and `FleetView`'s
`fleet_clear_band_ids` has no production consumer. A peer's measurement reaches
an operator's terminal and the JSIR interference report. It does not reach link
selection. Wiring it there is a design decision — a peer report is weaker
evidence than a live probe and would have to enter demote-only, below measured
sensing — not a missing call.

It does **not** prove RF performance. Occupancy comes from `iw dev <iface>
survey dump` — the kernel's channel survey on the onboard Wi-Fi radio, which is
a genuine measured noise floor and busy ratio, but it is a Wi-Fi chipset's view
of the 2.4/5 GHz ISM bands, not a calibrated receiver's view of an arbitrary
band. Anything requiring a calibrated measurement still needs an SDR and, for
the emissions claim, a chamber.

Write the claim to match the run. *Two processes*, *two containers*, and *two
machines* are three different statements, and only the last one closes this gap.

## Hardware

Two nodes that pass `kval-node preflight` — Jetson Orin Nano on Jetson Linux
39.2, per `JETSON_VALIDATION.md` — on the same network, each with a Wi-Fi
interface. A third machine acts as the operator. The operator machine must not
be one of the nodes: the whole point of an operator-signed roster is that a node
cannot mint its own peers.

## 1. Provision the fleet (operator machine)

```bash
python3 scripts/provision_fleet.py \
    --node kval-node-01 --node kval-node-02 \
    --position kval-node-01=1.2966,103.7764 \
    --position kval-node-02=1.3040,103.8050 \
    --operator-private-key operator.private.pem \
    --operator-public-key  operator.public.pem \
    --generate-operator-key \
    --output-dir fleet/
```

Positions are optional and operator-**surveyed**, not self-reported: they live
in the signed roster precisely because a node that could claim to be anywhere
could drag every emitter estimate with it. A node without a position still
corroborates spectrum; it just cannot contribute to localisation.

Then copy `fleet/peer-roster.signed.json` and `operator.public.pem` to **both**
nodes, copy each `fleet/<node-id>/node.private.pem` to **its own** node, and
delete the private keys from the operator machine. A node key that exists in two
places is a node that can be impersonated by whoever holds the copy.

## 2. Point each node at its identity

In each node's config:

```json
{
  "fabric_roster_path": "/etc/kval/peer-roster.signed.json",
  "fabric_node_key_path": "/etc/kval/node.private.pem",
  "fabric_inbox_path": "/var/lib/kval/fabric-inbox"
}
```

## 3. Exchange a report

On node 02:

```bash
kval-node fabric-report --config /etc/kval/node.json --output /tmp/node-02.json
```

Move it to node 01's inbox over whatever transport the deployment trusts — the
WireGuard tunnel `transport-preflight` already verifies is the intended one:

```bash
scp /tmp/node-02.json kval-node-01:/var/lib/kval/fabric-inbox/
```

On node 01:

```bash
kval-node fabric-view --config /etc/kval/node.json
```

The fleet view should now name `kval-node-02` as a contributor. If node 02 has
no spectrum hardware configured, `fabric-report` reports unavailable rather than
emitting an empty report dressed as a measurement — that refusal is correct
behaviour, and the exchange is not demonstrable until sensing is configured.

## 4. Prove the refusals, which is the part that matters

Two machines exchanging files proves nothing on its own: a node that accepted
everything would look identical. Run the adversarial harness on the **receiving**
node, against a report that actually arrived:

```bash
python3 scripts/verify_fleet_exchange.py \
    --config /etc/kval/node.json \
    --report /var/lib/kval/fabric-inbox/node-02.json \
    --evidence /var/lib/kval/evidence/exchange-$(date -u +%Y%m%dT%H%M%SZ).json
```

It derives hostile variants from that genuine report and runs each through the
node's own roster:

| Case | Accepting it would mean |
|---|---|
| `tampered_payload` | a peer's readings could be rewritten in flight |
| `clock_trust_flipped` | an undisciplined node could claim the right to corroborate |
| `replayed_report` | yesterday's quiet band could be replayed over today's jamming |
| `author_not_on_roster` | any machine that can write to the inbox could join the fleet |
| `roster_peer_impersonated` | a peer's identity could be worn by whoever holds any key |

Exit status is 0 only if the genuine report is accepted **and** every hostile
variant is refused, so this gates a deployment rather than describing one. The
evidence bundle records the hostname, roster ID and hash, and each verdict.

Nothing here fabricates evidence. The genuine report carries whatever the peer's
sensor actually produced. What is constructed is the attacks, which is the one
thing a security test has to manufacture to test at all.

## 5. Show the fused view, and where it stops

With both nodes sensing, put a load on one band near node 02 only and run
`kval-node fabric-view` on node 01. The band appears contested in the fused
view, corroborated by a node that node 01 never measured for itself.

Then look at node 01's `latest.json` and see that its **ranking has not moved**.
That is not a misconfiguration to debug — it is where the fabric currently ends.
The fused view is an operator artefact; the loop ranks on local sensing only.
Demonstrate the stopping point deliberately, because a reader who assumes
otherwise will believe the fleet is doing something it is not.

One property that *is* wired, and is counter-intuitive enough to show:

- **A blind fleet reports `unknown`, not `clear`.** Stop sensing on both nodes
  and the fused belief goes to total ignorance. Vote-counting would have
  reported quiet, which is the defect Dempster-Shafer fusion was brought in to
  fix.

And one that governs any future wiring:

- **Sensing demotes, never promotes.** A band reported clear does not raise a
  profile's score; only this node's own live probe can. So if peer evidence is
  ever admitted to the decision, a fleet still cannot be talked onto a link by
  peers alone.

## What remains open after this runs

Closing this gap does not close the others in `DEFENCE_STANDARDS.md` §7, and the
ranking there does not change: transport is still classically secure only, there
is still no hardware root of trust for the signing state, and emissions are
still unmeasured against MIL-STD-461 RE102 / STANAG 2319. A two-node result
makes the fabric real. It says nothing about whether the node can be found by
its own leakage, which remains the sharpest threat to the passive claim.
