# Kval

**Kval keeps an uncrewed vehicle's command link alive as its radios degrade —
and refuses to act on evidence it doesn't have.**

A Jetson-class box on the platform. It measures which radio links actually work,
picks one inside operator-signed policy, and says so in a tamper-evident record.
It never transmits to do any of that.

---

## The problem

An uncrewed vehicle — air, land or sea — carries several links: Wi-Fi, 4G/5G,
satellite, Ethernet. They degrade at different times, and some degrade because
someone wants them to.

When the command link goes, the vehicle either flies on blind or turns for home.
Both are mission failures, and an adversary only has to cause one of them.

## Three things that make this different

### 1. The command link is not tradeable

CNPC is a separately regulated class — ITU-R sets aside 5030–5091 MHz and
960–977 MHz, with its own MOPS, radio standard and mandated loss behaviour.
Payload video has none of that.

Rank them on one scale and a camera downlink can outscore the link that steers
the vehicle. Here, **ranking never crosses roles**, and a link cannot become
safety-critical by omission — it has to be declared one in the signed catalogue.

### 2. It knows ordered silence from jamming

Under EMCON a platform is *told* to go dark. To every lost-link detector, that
is indistinguishable from being jammed — so a vehicle that runs its contingency
because it was **obeying an order** has been defeated by its own doctrine, and an
adversary who jams during a known EMCON window gets a return-to-base for free.

Kval resolves it directionally: EMCON constrains what the *platform* emits and
says nothing about what the *operator* emits. A dark return path under commanded
silence is compliance. A dark command path is a lost link regardless.

```
jamming, no order          →  degraded → lost → CONTINGENCY
commanded EMCON SILENT     →  nominal at t+0, +31s, +600s
jamming during EMCON       →  still reaches CONTINGENCY
order expired, still dark  →  suppression lifts
```

No civil standard covers this, because civil aviation has no reason to fall
silent on purpose.

### 3. Nothing it cannot act on is in it

The system used to carry a passive spectrum-sensing stack, a cross-node fabric
built on band reports, and a weather-discrimination tier built on those
observations — together larger than the decision they served, and none of it
reaching the flying node's choice of link. It has been removed rather than
carried.

What is left is what the loop runs on: the link probes it measures itself.
Degradation is detected the way the vehicle experiences it — latency, loss,
jitter, throughput, signal — not inferred from an RF picture the decision never
consulted.

---

## The discipline underneath

One rule runs through everything: **the system refuses rather than guesses.**

- Anything that is not a live probe may **demote** a link, never **promote**
  one. Nothing that merely looks promising is evidence that a link works. So a
  node cannot be *talked into* a link.
- A capability with missing evidence returns **unavailable**, never a plausible
  answer. "Could not measure" is never reported as "it is fine."
- Nothing names a source or an intent. A degraded link cannot tell you who, or
  why.
- Every answer carries the reasons it reached that answer and the sample count
  it rests on.

A system that always answers is a system whose answers mean nothing.

---

## What it runs

The autonomous loop, on the platform, unattended:

```
observe → clock integrity → govern power → forecast degradation
  → recovery → rank profiles → fuse evidence → safety shield → act → audit
```

Everything else — the operator tools and the research tier — loads when a person
asks for it, on a machine where a person is watching. That boundary is
deliberate and enforced by the build.

---

## Depth, on request

Behind the three claims sit implementations of published mechanisms, applied to
a receive-only node. They are not the proposition; they are the answer to *how
do you know?*

| Question | What answers it |
|---|---|
| How do you know a link is failing? | Live probes bound to the interface — latency, loss, jitter, throughput — scored against a service-class objective, with confidence zero if the interface is absent |
| Why did it not switch? | Dwell, switch-rate and hysteresis gates in the Safety Shield, each with a named refusal code |
| How does a fleet learn without pooling data? | Federated statistics, clustered by environment, perturbed locally before release |
| What happens when the fleet is cut off? | k-of-n peers may extend an existing grant's life, never widen its scope and never undo a revocation — then it fails closed |
| Will it still verify in 2030? | CNSA 2.0: LMS signatures per RFC 8554, pinned against the RFC's own test vector |

Full mapping with DOIs in [`DSA_RESEARCH_MAPPING.md`](DSA_RESEARCH_MAPPING.md).
Standards analysis and the honest gap ledger in
[`DEFENCE_STANDARDS.md`](DEFENCE_STANDARDS.md).

---

## What it does not do

No transmit. No jamming or electronic attack. No packet inspection. No vehicle
control — STANAG 4586 LOI 1: it selects and assures links, it does not fly
anything. No intent attribution. No source naming.

**No RF sensing.** It does not observe spectrum, locate emitters, or tell
weather from jamming. It knows a link is bad because it measured that link.

Zero runtime dependencies, so a fielded box has nothing to resolve.

## What is not yet proven

No cross-node result on two physically separate machines. Peer identity and the
federated tier are verified in one process; nothing has been run on two hosts.

Emissions are unmeasured against MIL-STD-461 RE102. Kval no longer senses, so
the exposure is narrower than it was — but a box that leaks is still findable,
and settling that needs a chamber.

No model is trained. No FIPS-validated module, RMF package, or independent
security assessment exists. LMS key custody is a file, not a FIPS 140 Level 3
module — the algorithm requirement of CNSA 2.0 met without the implementation
profile of SP 800-208.
