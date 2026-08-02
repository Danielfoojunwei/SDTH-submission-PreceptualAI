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

## 2. Electromagnetic spectrum operations

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
- **Interference reports are not in a recognised format.** Kval detects and
  corroborates interference across nodes, which is genuinely the hard part, but
  emits its own JSON. Joint Spectrum Interference Resolution (JSIR) reporting
  under CJCSM 3320.01D has a defined format, and a report a spectrum manager
  cannot ingest is a report that does not reach anyone.

---

## 3. Open architecture — MOSA, SOSA, CMOSS

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

## 4. Autonomy and meaningful human control

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

## 5. Assurance and accreditation

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

## 6. Where the honest gaps are, ranked

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
4. **No EMCON state model.** Doctrinally, emission control is a spectrum of
   states, and the node cannot represent where it sits on it.
5. **Interference reporting is not interoperable.** JSIR format under
   CJCSM 3320.01D.
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
- Singapore Defence Tech Hackathon 2026 ([luma.com](https://luma.com/sdth-2026))
