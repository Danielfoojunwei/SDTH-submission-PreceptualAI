# Defence standards: what this system meets, and what it does not

Written for the Singapore Defence Tech Hackathon, 25–27 September 2026
(European Defense Tech Hub with NUS Enterprise and TUM Venture Labs), whose
stated tracks include **Resilient Communications**, **Electronic Warfare**,
**Autonomous Command & Control**, and **AI & Agentic Systems** —
[luma.com/sdth-2026](https://luma.com/sdth-2026).

This document exists because a defence-adjacent system that has not been
measured against defence standards is a prototype with good intentions. Each
section below states the standard, what Kval actually does, and — where it
falls short — says so with the specific requirement it misses. Nothing here
claims a certification. No accreditation body has assessed this code.

---

## 1. Cryptography — CNSA 2.0

**The standard.** NSA's Commercial National Security Algorithm Suite 2.0 sets
the algorithms mandatory for National Security Systems. It approves:

| Function | Algorithm | Reference |
|---|---|---|
| Key establishment | ML-KEM-1024 | FIPS 203 |
| Digital signatures | ML-DSA-87 | FIPS 204 |
| Software & firmware signing | LMS or XMSS, single-tree | NIST SP 800-208 |
| Symmetric encryption | AES-256 | FIPS 197 |
| Hashing | SHA-384 / SHA-512 | FIPS 180-4 |

CNSA 2.0 **retires all classical asymmetric cryptography** — RSA, finite-field
Diffie-Hellman, ECDH and ECDSA. Ed25519 is not on the approved list either.

**The dates matter more than the list.** NSA's transition table puts software
and firmware signing first, because a verification algorithm baked into fielded
hardware is locked in for that hardware's operational life:

| Category | Support and prefer | Exclusive use |
|---|---|---|
| **Software & firmware signing** | **2025** | **2030** |
| Networking equipment (VPN, routers) | 2026 | 2030 |
| Operating systems | 2027 | 2033 |
| Constrained / niche devices | 2030 | 2033 |

And the procurement gate: **from 1 January 2027, every new National Security
System acquisition must support CNSA 2.0 or carry a documented exception.**
SDTH is September 2026. A system pitched to a defence buyer then is three
months from that gate.

**Where Kval was.** Every trust decision rested on Ed25519 — the catalogue,
spectrum and power policies, capability leases, peer reports, federated
contributions, audit-record authorship, quorum attestations. A signed-policy
trust root *is* firmware signing in everything but name: it decides what the
node is permitted to do.

**What is now built.** `pqc.py` implements LMS per RFC 8554 and SP 800-208,
using SHA-256 and nothing else — no runtime dependency, consistent with
`dependencies = []`. NSA names LMS as the near-term recommendation for firmware
precisely because its security rests on a hash function alone.

Correctness is not asserted from round-tripping. The implementation verifies
**RFC 8554 Appendix F Test Case 1** — the RFC's own published public key,
message and signature bytes — and that vector is pinned in CI. A signer and
verifier that agree with each other prove nothing; agreeing with the published
bytes is the property that matters.

`signing.py` gained algorithm agility: documents carry their algorithm, the
verifier dispatches on it, and the trust anchors come from the caller so an
attacker cannot downgrade a document by relabelling it. `require_cnsa_2_0` in
node config refuses non-approved algorithms outright.

`kval-node crypto-posture` reports each trust anchor, the algorithm it uses,
whether that algorithm is approved, and the days remaining to the acquisition
gate.

### What is *not* met, precisely

**SP 800-208 §5 requires key generation and state management inside a FIPS 140
Level 3 hardware module that cannot export private key material.** Kval keeps
the signature index in a file. That is materially weaker: an attacker with
filesystem access who rolls the index back can induce one-time key reuse, and
reuse of an LM-OTS leaf lets them recover that leaf's private key and forge
freely (RFC 8554 §9.1).

The file store does what a file store can — reserve-before-sign with `fsync` on
both file and directory, refusal on any index that moves backwards, hard stop
on exhaustion — and the signing interface is shaped so swapping in an HSM is a
contained change. **Using this satisfies the algorithm requirement of CNSA 2.0
and not the implementation profile of SP 800-208.** An operator seeking
accreditation needs the hardware.

**ML-KEM and ML-DSA are not implemented.** Kval establishes no keys of its own
(transport security is delegated to WireGuard, which is verified rather than
implemented), so ML-KEM has no place here yet — but a WireGuard tunnel is
itself classical, and `transport_security.py` verifies a tunnel it cannot make
quantum-resistant. That is a real gap in the transport layer, not the signing
layer. ML-DSA would need a vetted implementation this repository cannot supply
without a dependency.

---

## 2. Uncrewed systems: the command link is not one link among many

**This is the finding that reframed the system.** Kval ranked every link on one
utility scale — latency, loss, jitter, throughput, cost, energy — and took the
best. For a fixed sensor that is right. For an autonomous uncrewed platform,
air, ground, surface or subsurface, it is wrong in a way that is unsafe *and*
non-compliant, and the reason is regulatory before it is technical.

**Control and Non-Payload Communications is a protected, separately regulated
class.** The C2 link carries what flies, drives or steers the platform. It has:

| | |
|---|---|
| Dedicated spectrum | **5030–5091 MHz** terrestrial LOS (WRC-12), **960–977 MHz** L-band, FSS for BLOS |
| Sized by ITU-R | 34 MHz terrestrial, 56 MHz satellite |
| Its own MOPS | RTCA **DO-362** (terrestrial), **DO-377** (satellite) |
| Its own radio standard | FAA **TSO-C213** |
| Its own rule part | FCC **Part 88** (August 2024) |
| Mandated loss behaviour | RTCA **DO-400**, ICAO contingency procedures |

Payload video has none of that.

### What is now built

`c2link.py` makes three properties structural:

**Ranking never crosses roles.** `Profile.link_role` is declared in the signed
catalogue and the optimizer confines ranking to one role. Comparing a command
link against a payload link on one utility scale lets a high-value sensor
downlink outscore the link that steers the vehicle — and utility weights are
exactly what gets retuned for throughput by someone who does not know that.
The default role is `payload`, so **a link can never become safety-critical by
omission**, and an undeclared link can never be selected as C2.

**Loss is an event with a pre-planned response.** DO-400 requires a declared
lost-link route, orbit points, altitudes and a flight-termination point.
`ContingencyAction` is a closed vocabulary the *operator* fills in before the
mission; the node selects nothing. Two timers separate confirmation from
commitment, so one bad sample cannot commit the platform — a contingency
triggered on a single observation is one an adversary fires with a single
pulse. A declared contingency is not silently unwound by the link returning:
the platform is executing a plan, and cancelling it automatically would leave
it somewhere neither the plan nor the operator expects.

**The budget is a floor, not a preference.** A C2 link outside its declared
latency and loss budget is degraded even when it is the best available.
"Best of a bad set" is how a vehicle ends up flying on a link that cannot
command it.

### The gap the civil standards do not cover

Civil aviation has no reason to fall silent on purpose, so nothing in DO-400 or
ICAO addresses this: **under emission control, a platform is *ordered* to stop
transmitting, and to every lost-link detector ever built that is
indistinguishable from jamming.**

A vehicle that runs its lost-link contingency because it was obeying an EMCON
order has been defeated by its own doctrine while the adversary did nothing.
Worse, an adversary who jams during a known EMCON window gets a return-to-base
or a self-termination for free.

`emcon.py` and `c2link.py` resolve it directionally, and the asymmetry is the
whole design:

- **EMCON constrains what the *platform* emits. It says nothing about what the
  *operator* emits.** An operator can transmit commands to a silent vehicle all
  day. So a dark **return** path under commanded silence is compliance; a dark
  **command** path is a lost link regardless of EMCON state. Jamming during an
  EMCON window still reaches the contingency, and this is tested.
- **Silence is bounded.** An order carries an expiry, and an expired order
  stops suppressing — otherwise an adversary who jams the *lifting* order gets
  indefinite suppression of the contingency.
- **The order is signed.** An unauthenticated EMCON order is a denial-of-service
  primitive: anyone who can assert "you are silent" suppresses the platform's
  lost-link response for as long as they like.
- **Every suppression is recorded**, so an operator reading the audit sees
  which silences were doctrine and which were an adversary.

EMCON levels are also encoded rather than labelled: each state names the link
roles it suppresses, from housekeeping only, through payload, to total silence
— and receive-only sensing continues at every level, because receiving emits
nothing a silence order governs.

### Interoperability, claimed honestly

**STANAG 4586** defines five Levels of Interoperability, from indirect receipt
of data (1) through payload control (3) and vehicle control (4) to full control
including launch and recovery (5). **STANAG 4817** extends the framework to
multi-domain UxV control across air, sea and subsurface.

Kval declares **LOI 1** and says why: it selects and assures links, it does not
control a vehicle or a payload, and LOI 4/5 require a control station this is
not. Its domain-agnosticism across land, sea, air and subsurface is real — link
selection does not care what the platform is — but that is a strength, not
compliance with STANAG 4817.

**Remaining gap.** No STANAG 4586 message-format bridge exists. A control
station speaking 4586 cannot ingest Kval's output today without a translation
layer, and building one from the abstract without the (controlled-distribution)
message specification would be guesswork.

---

## 3. Electromagnetic spectrum operations

**The standard.** NATO recognises the electromagnetic environment as an
operational domain (MC 64/11), with the discipline named Electromagnetic
Spectrum Operations (EMSO). The US joint manual is **CJCSM 3320.01D, *Joint
Electromagnetic Spectrum Operations*, 24 January 2025**.

**Where Kval sits.** Passive, receive-only sensing is squarely inside EMSO's
spectrum-monitoring and situational-awareness function. The architecture is
EMCON-compatible by construction: there is no transmit path, and CI fails the
build if one appears.

**Gaps, honestly.**

- **EMCON is not modelled as a state.** EMCON is a matter of degree, from
  unrestricted emission to total silence, and a node ought to know which state
  it is under and behave differently. Kval is *always* receive-only, which is
  conservative and correct, but it cannot express "the platform is at EMCON
  Alpha, so suppress even the backhaul heartbeat". The scan rate limiter is the
  nearest thing and it governs duty, not doctrine.
- **Receive-only is not emission-free.** A Jetson plus an SDR radiates:
  local-oscillator leakage from the receiver front end, and digital switching
  noise from the compute. **MIL-STD-461 (RE102)** and **NATO STANAG 2319**
  (electromagnetic compatibility) govern exactly this, and nothing in this
  repository measures it. A "passive" node that is detectable by its own
  emissions is a liability in the mission this system is for. This needs a
  chamber, not a code change.
- **No spectrum supportability record.** Fielded equipment normally carries
  spectrum certification (in US practice, DD Form 1494) and host-nation
  coordination. `config/band-plans.json` cites civil allocations — FCC, ETSI,
  ITU — and has no military allocation or host-nation dimension.
- **Interference reporting — now closed.** `interference.py` emits
  JSIR-compatible fields per CJCSM 3320.01D: observer, time, band, centre
  frequency, bandwidth, victim system role, and characterised operational
  effect. Two things it deliberately refuses: it does not classify **intent**,
  because nothing in a noise floor says whether a microwave oven, a faulty
  amplifier or an adversary put it there; and it names no **source**, because
  a position with an uncertainty radius is a place, not an actor. A corroborated
  C2 victim is escalated above a payload one, since that is the report a
  spectrum manager needs first.
- **EMCON — now closed.** See §2. Modelled as ordered states with encoded
  consequences, bounded in time, signed, and consumed by the lost-link logic.

---

## 4. Open architecture — MOSA, SOSA, CMOSS

**The standard.** The US DoD mandates a **Modular Open Systems Approach**
(MOSA) for major acquisitions. The aligned technical standards are **SOSA**
(Sensor Open Systems Architecture, covering communications, EW, radar and
SIGINT sensors) and **CMOSS** (C5ISR/EW Modular Open Suite of Standards, built
on OpenVPX).

**Where Kval sits.** The software architecture is modular and openly specified
in the MOSA spirit: signed documents with declared schemas, a closed action
vocabulary, capability boundaries that are structural rather than promised.

**Gap.** SOSA and CMOSS are substantially *hardware* standards — OpenVPX card
form factors, backplane profiles, chassis management. Kval targets a Jetson
Orin Nano, which is not an OpenVPX card. Claiming SOSA alignment would be
false. The honest position: the software is modular and portable, and a SOSA
integration would be a port to a compliant card, not a rewrite.

---

## 5. Autonomy and meaningful human control

SDTH's Track C asks: *"Can we keep meaningful human judgment in the loop when
autonomous warfare has already outrun human reaction time?"*

This is the question the capability-lease architecture was built for, and it is
worth stating precisely what it does and does not answer.

**What it does.** Authority is not a runtime decision. A `LeaseAction` is one of
nine values in a closed enum, every one operator-signed, node-bound,
single-use, expiring, with declared bounds, a rollback and a post-condition.
The node cannot invent an action, cannot widen its own authority, and no
learned component may promote itself — `verify-model` returns
`ADVISORY_UNTIL_PROMOTED` even for a manifest that verifies, because provenance
is not fitness. Every refusal carries a machine-readable reason, so an operator
reads *why* rather than inferring it.

**What it does not answer.** Human control here is *pre-authorised*, not
*supervisory*: a human decides in advance what the node may do, not in the
moment. That is the right trade when the link is contested — supervision needs
a link, and the whole premise is that the link fails — but it is not the same
thing as a human in the loop, and it should not be sold as one. The quorum
renewal in `threshold_ca.py` deliberately cannot grant new authority for
exactly this reason: a partition may keep an approved capability alive and may
never approve a new one.

---

## 6. Assurance and accreditation

Not claimed, and worth listing so the absence is deliberate rather than
overlooked:

- **No FIPS 140 validated module.** `cryptography` is used for Ed25519 and the
  LMS implementation is this repository's own. Neither is validated.
- **No Risk Management Framework package**, no ATO, no STIG mapping. The
  systemd hardening is real (namespace isolation, read-only paths, capability
  bounding) but has not been mapped to a control catalogue such as NIST
  SP 800-53.
- **No safety-critical software process.** DO-178C, DEF STAN 00-055/00-056 and
  IEC 61508 all require artefacts — requirements traceability, structural
  coverage analysis, tool qualification — that do not exist here.
- **No independent security assessment.** The adversarial reviews in this
  repository's history were self-conducted.

---

## 7. Where the honest gaps are, ranked

Ordered by how much they would matter to a defence buyer, not by how hard they
are:

1. **Transport is classically secure only.** WireGuard uses X25519. Kval
   *verifies* the tunnel rather than implementing it, so this cannot be fixed
   inside this repository — but a harvest-now-decrypt-later adversary against a
   fleet whose traffic reveals sensor siting is a real threat, and CNSA 2.0
   puts networking equipment at exclusive use by 2030.
2. **No hardware root of trust for the signing state.** SP 800-208's Level 3
   requirement, above. The algorithm is right; the custody is not.
3. **Unmeasured emissions.** MIL-STD-461 / STANAG 2319. A passive system that
   can be found by its own leakage fails at its primary claim.
4. **No STANAG 4586 message bridge.** A 4586 control station cannot ingest
   Kval's output without a translation layer, and the message specification is
   controlled-distribution. This is the clearest remaining interoperability
   gap for a NATO-facing deployment.
5. **C2 timing budgets are operator-declared, not measured against DO-362.**
   The module enforces whatever floor the operator signs. It does not verify
   that floor is airworthy for the platform, which is a certification activity.
6. **No cross-node result on physically separate hardware.** Stated throughout
   this repository and repeated here because it remains the largest evidential
   gap: everything cross-node is verified in software.

---

## Sources

- NSA, *Announcing the Commercial National Security Algorithm Suite 2.0*
  ([media.defense.gov](https://media.defense.gov/2025/May/30/2003728741/-1/-1/0/CSA_CNSA_2.0_ALGORITHMS.PDF)),
  and the CNSA 2.0 FAQ
  ([media.defense.gov](https://media.defense.gov/2022/Sep/07/2003071836/-1/-1/0/CSI_CNSA_2.0_FAQ_.PDF))
- NIST SP 800-208, *Recommendation for Stateful Hash-Based Signature Schemes*
  ([csrc.nist.gov](https://csrc.nist.gov/pubs/sp/800/208/final))
- McGrew, Curcio, Fluhrer, *Leighton-Micali Hash-Based Signatures*, RFC 8554
  ([rfc-editor.org](https://www.rfc-editor.org/rfc/rfc8554.txt))
- CJCSM 3320.01D, *Joint Electromagnetic Spectrum Operations*, 24 January 2025
  ([jcs.mil](https://www.jcs.mil/Portals/36/Documents/Library/Manuals/CJCSM%203320.01D.pdf))
- NATO STANAG 2319, electromagnetic compatibility
  ([betalight-tactical.com](https://betalight-tactical.com/knowledge-base/regulations/stanag-2319))
- EMSOpedia, *Electromagnetic Spectrum Operations*
  ([emsopedia.org](https://www.emsopedia.org/entries/electro-magnetic-spectrum-operation-emso/))
- US DoD OUSD(R&E), *Modular Open Systems Approach*
  ([cto.mil](https://www.cto.mil/sea/mosa/)); Curtiss-Wright, *What is MOSA,
  SOSA and CMOSS in Defense?*
  ([defense-solutions.curtisswright.com](https://defense-solutions.curtisswright.com/resources/white-papers/what-is-mosa-sosa-cmoss))
- US Army, *Adapting to Multi-Domain Battlefield: Developing Emissions Control
  SOP* ([army.mil](https://www.army.mil/article/284546/adapting_to_multi_domain_battlefield_developing_emissions_control_sop))
- STANAG 4586, *Standard Interfaces of UAV Control System (UCS) for NATO UAV
  Interoperability*
  ([sto.nato.int](https://www.sto.nato.int/document/stanag-4586-standard-interfaces-of-ucs-for-nato-uav-interoperability/),
  [publications.sto.nato.int](https://publications.sto.nato.int/publications/STO%20Educational%20Notes/STO-EN-SCI-271/EN-SCI-271-03.pdf))
- STANAG 4817, multi-domain UxV control (air, sea, subsurface), via the
  Fraunhofer INTERACT project report
  ([iosb.fraunhofer.de](https://www.iosb.fraunhofer.de/content/dam/iosb/iosbtest/documents/projekte/interact/Interoperability%20of%20Armed%20Forces%20Unmanned%20Systems_korr.pdf))
- NASA, *Control and Non-Payload Communications (CNPC) link*
  ([ntrs.nasa.gov](https://ntrs.nasa.gov/api/citations/20170010707/downloads/20170010707.pdf));
  FAA TSO-C213a draft
  ([faa.gov](https://www.faa.gov/sites/faa.gov/files/2022-08/TSO_C213a_Draft_Public_Comment.pdf));
  FCC Part 88 UAS spectrum
  ([uavionix.com](https://uavionix.com/protected-c2-for-uas-part-1/))
- FAA Notice JO 7110.724, *UAS Lost Link*
  ([faa.gov](https://www.faa.gov/documentLibrary/media/Notice/N_JO_7110.724_5-2-9_UAS_Lost_Link.pdf));
  RTCA DO-400 guidance on C2 link unavailability
  ([unmannedairspace.info](https://www.unmannedairspace.info/latest-news-and-information/rtca-publishes-new-guidance-for-uas-in-case-command-and-control-link-becomes-unavailable/))
- Singapore Defence Tech Hackathon 2026 ([luma.com](https://luma.com/sdth-2026))
