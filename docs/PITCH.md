# Kval

**Kval keeps an uncrewed vehicle's command link alive in contested spectrum —
and refuses to act on evidence it doesn't have.**

A Jetson-class box on the platform. It measures which radio links actually work,
picks one inside operator-signed policy, and says so in a tamper-evident record.
It never transmits to do any of that.

---

## The problem

An uncrewed vehicle — air, land or sea — carries several links: Wi-Fi, 4G/5G,
satellite, Ethernet. In contested spectrum they degrade at different times, and
some degrade because someone wants them to.

When the command link goes, the vehicle either flies on blind or turns for home.
Both are mission failures, and an adversary only has to cause one of them.

## Three things that make this different

### 1. It never transmits to sense

Every measurement comes from listening. The build fails if a transmit path
appears in the source, and an observation obtained by emitting is refused rather
than used — passivity is derived from which sensor backends actually ran, not
asserted by a flag.

A sensor that emits to sense is a sensor that can be found.

### 2. The command link is not tradeable

CNPC is a separately regulated class — ITU-R sets aside 5030–5091 MHz and
960–977 MHz, with its own MOPS, radio standard and mandated loss behaviour.
Payload video has none of that.

Rank them on one scale and a camera downlink can outscore the link that steers
the vehicle. Here, **ranking never crosses roles**, and a link cannot become
safety-critical by omission — it has to be declared one in the signed catalogue.

### 3. It knows ordered silence from jamming

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

---

## The discipline underneath

One rule runs through everything: **the system refuses rather than guesses.**

- Sensing can **demote** a link, never **promote** one. A quiet band is not
  evidence a link works — only a live probe is. So a fleet cannot be *talked
  into* a link.
- **"Couldn't measure" is never reported as "it's clear."** Vote-counting can't
  tell those apart, which made a fleet of blind nodes look like a fleet
  reporting quiet. Fixed with evidence theory: blind nodes now return `unknown`.
- **A position with an uncertainty radius is a place, not an actor.** Nothing
  here names a source or an intent — a noise floor cannot tell you who, or why.
- Every answer carries the reasons it reached that answer and the sample count
  it rests on.

A system that always answers is a system whose answers mean nothing.

---

## What it runs

The autonomous loop, on the platform, unattended:

```
observe → clock integrity → sense spectrum → govern power → forecast
  → recovery → rank profiles → fuse evidence → safety shield → act → audit
```

**25 modules load on a flying node.** Everything else — the operator tools and
the research tier — loads when a person asks for it, on a machine where a person
is watching. That boundary is deliberate and enforced by the build.

---

## Depth, on request

Behind the three claims sit implementations of published mechanisms, applied to
a receive-only fleet. They are not the proposition; they are the answer to *how
do you know?*

| Question | What answers it |
|---|---|
| How do you know the band is contested? | Passive occupancy + noise floor; peers corroborate over signed reports |
| How do you trust a peer? | Operator-signed roster; undisciplined clocks count for presence, never for corroboration |
| How do you know it's jamming and not weather? | Signal drops with a steady noise floor is attenuation; a rising floor is energy arriving. Weather mechanisms separate by frequency law |
| Where is the emitter? | Multi-node localisation with NLOS rejection — and a refusal when no anchor removal reconciles the rest |
| How does the fleet learn without pooling data? | Federated statistics, clustered by RF environment, perturbed locally before release |
| Will it still verify in 2030? | CNSA 2.0: LMS signatures per RFC 8554, pinned against the RFC's own test vector |

Full mapping with DOIs in [`DSA_RESEARCH_MAPPING.md`](DSA_RESEARCH_MAPPING.md).
Standards analysis and the honest gap ledger in
[`DEFENCE_STANDARDS.md`](DEFENCE_STANDARDS.md).

---

## What it does not do

No transmit. No jamming or electronic attack. No packet inspection. No vehicle
control — STANAG 4586 LOI 1: it selects and assures links, it does not fly
anything. No intent attribution. No source naming.

Zero runtime dependencies, so a fielded box has nothing to resolve.

## What is not yet proven

Nothing cross-node has run on two physically separate machines. The provisioning
and the adversarial verification exist and are tested; they have not been
executed on two hosts, and the claim will be written to match whatever is
actually run.

Emissions are unmeasured against MIL-STD-461 RE102. A passive node findable by
its own leakage fails at its primary claim, and settling that needs a chamber.

The weather discrimination is derived from published physics and exercised
against constructed signatures. That tests the logic, not the calibration — no
measured campaign data has been through it.

No model is trained. No FIPS-validated module, RMF package, or independent
security assessment exists.
