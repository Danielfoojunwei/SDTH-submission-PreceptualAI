# Secure transport and the electronic-warfare threat model (defensive)

## Encryption is not invisibility

The request was to encrypt everything sent over the air "such that nothing can be
tapped". That goal cannot be met by encryption alone, and it would be dishonest
to claim otherwise. Encryption protects the **content** of a transmission. It
does not hide:

* that a transmission happened at all — RF energy is detectable by anyone with a
  receiver;
* **when** it happened, how long it lasted, how big it was, and in which
  direction it flowed (traffic analysis);
* **where** the transmitter is — direction finding works on any emission,
  encrypted or not;
* the endpoints, if the outer headers are visible.

Encryption also does nothing against **jamming** (an attacker does not need to
read your traffic to deny it) or against a **compromised endpoint** (where the
plaintext already exists). What Kval therefore enforces is a *verified encrypted
path with no plaintext leak around it*, plus honest documentation of the residual
observability.

## What is enforced

`transport_security.py` verifies, against an operator-signed transport policy:

| Control | Enforcement |
|---|---|
| Tunnel identity | WireGuard peers matched **exactly** by public key; unexpected peers fail |
| Tunnel scope | Each peer's `AllowedIPs` must equal the policy set exactly |
| Tunnel liveness | Peer must show a recent handshake (never-handshaked fails) |
| No leak around the tunnel | **Both** IPv4 and IPv6 default routes must leave exclusively via the tunnel interface |
| Egress default-deny | nftables `output` chain must default to `drop` |
| Ruleset integrity | Normalized nftables ruleset hash must equal the pinned policy hash (handles and counters stripped so reloads don't churn the hash) |
| Wi-Fi link layer | WPA3-SAE or WPA3-Enterprise-192 only; management-frame protection (PMF) required; optional BSSID allowlist |
| Enterprise identity | WPA3-Enterprise-192 requires a pinned CA hash and EAP server domain-suffix match |
| Evidence hygiene | `assert_no_secret_material` refuses to emit any evidence containing private-key or PSK material |

Every check **fails closed**: a missing binary, an unreadable interface, a
malformed ruleset, or an unverifiable policy yields `verified: false` with a
reason — never a pass by default.

```bash
kval-node transport-preflight --config /etc/kval/node.json
```

### Why these specific controls

WireGuard's own design associates a peer's identity with the addresses it may
use: "The fundamental principle of a secure VPN is an association between peers
and the IP addresses each is allowed to use as source IPs. In WireGuard, peers
are identified strictly by their public key, a 32-byte Curve25519 point."
(Donenfeld, *WireGuard: Next Generation Kernel Network Tunnel*, §2 "Cryptokey
Routing"). The protocol "avoids key-compromise impersonation, avoids replay
attacks, provides perfect forward secrecy…" (§5), with replay handled by a
sliding window of received message counters (§6). Kval's contribution is to
verify that this association is configured *as the operator signed it*, and that
no traffic can bypass it.

Kval's tunnel checks sit **below** application security and do not replace it.
Application traffic should still use TLS 1.3 / mTLS — now specified by
**RFC 9846**, *The Transport Layer Security (TLS) Protocol Version 1.3* (July
2026), which obsoletes RFC 8446 — with deployment hardening per **RFC 9325**
(BCP 195), and SSH per RFC 4251–4254. Cellular link security remains governed by
**3GPP TS 33.501**, *Security architecture and procedures for 5G System*, and is
the operator network's responsibility, not Kval's.

## Defensive EW and adversary threat register

Kval is a defensive system. The table below lists the threats it is designed to
survive or detect, and what actually mitigates each. **No offensive capability is
implemented**: no jamming, no spoofing, no covert signalling, no directed RF
energy, no arbitrary transmit control. Nothing in the lease vocabulary can
authorise one.

| # | Threat | Mitigation in this repository | Residual risk |
|---|---|---|---|
| 1 | Rogue access point / evil twin | WPA3-only + PMF required + BSSID allowlist + EAP CA/domain pinning | Operator must provision a correct allowlist |
| 2 | Passive wiretap of payload | WireGuard tunnel verified end to end; default-drop egress | Metadata still observable (see below) |
| 3 | Man-in-the-middle on the tunnel | Exact peer public-key match; handshake required | Endpoint compromise not covered |
| 4 | Route bypass / plaintext leak | Effective IPv4 **and** IPv6 default-route verification; nftables default-drop output | Requires policy to pin the ruleset hash |
| 5 | Replay of tunnel traffic | WireGuard sliding-window counters (protocol-level) | — |
| 6 | Replay of an operator authorization | Single-use lease nonces in a durable store; expiring leases | Operator must not reuse nonces |
| 7 | Narrowband / selective jamming | Passive sensing flags `JAMMING_SUSPECTED`; recovery FSM fails over to another approved band/profile | Cannot prevent denial, only detect and route around |
| 8 | Broadband jamming | Multi-profile failover incl. NTN; degraded-service modes | Physics: sufficient power denies the band |
| 9 | Traffic analysis (size/timing/direction) | Documented as residual; telemetry batching and duty limits reduce pattern richness | **Not eliminated** — see below |
| 10 | Direction finding of our own emitters | Duty-cycle and scan-interval limits reduce emission time; sensing itself is receive-only and emits nothing | **Not eliminated** — any transmission can be located |
| 11 | GNSS / time spoofing | Signed policies carry their own validity windows; observation freshness is enforced against local monotonic and wall-clock checks | A sustained clock attack can still stall decisions (fails closed) |
| 12 | Endpoint capture / physical access | Root-owned sockets; private keys never on the node for catalogue/policy signing; audit hash chain | Physical compromise of a node is not survivable by software alone |
| 13 | Firmware / kernel compromise | Out of scope for Kval; requires platform secure boot and attestation | Explicitly not claimed |
| 14 | Poisoned telemetry | Provenance hash, source identity, quality and missingness per modality; fail-closed fusion | A signed-but-lying adapter is trusted by design |
| 15 | Model poisoning | Signed model manifest (engine, feature schema, calibration, lineage); drift/OOD abstention; quarantine action | Requires operator to run the promotion gates |
| 16 | Malicious federated update | Signed updates + quarantine lease action; robust aggregation is the coordinator's job | Aggregation robustness not implemented here |
| 17 | Sybil / impersonated nodes | Node-bound leases; exact peer keys | Fleet identity management is a coordinator concern |
| 18 | Insider / over-broad authorization | Closed `LeaseAction` set; per-action numeric bounds; expiry; rollback + post-condition | An operator can still sign a bad-but-in-bounds policy |
| 19 | Supply-chain substitution | Adapter conformance hash + implementation identity required; pinned Debian packages | Requires real conformance reports |
| 20 | Link flapping (self-inflicted DoS) | Dwell time, switch-rate limits, backoff, flap ceiling → escalate | — |
| 21 | Audit tampering | SHA-256 hash-chained SQLite ledger; verification command | Detects, does not prevent |
| 22 | Power exhaustion / thermal DoS | Signed power policy; thermal ceiling with hysteresis; battery floor; duty limits | — |

### Residual observability — stated plainly

Even with everything above verified, an adversary with a receiver can still learn
that a node is transmitting, roughly when and how much, and — with suitable
equipment — where it is. Reducing that exposure is an operational discipline
(emission control, duty-cycle limits, terrain, directional antennas, scheduled
transmission windows), not something encryption provides. Kval helps by keeping
sensing passive and by bounding scan/telemetry duty cycles under signed power
policy, but it does not claim low-probability-of-intercept or
low-probability-of-detection properties.

## What still requires physical proof

None of the transport gates are claimed as passing until they are run on the real
hardware with real keys. On the node, with provisioned WireGuard keys, real
NetworkManager UUIDs and BSSIDs, and a pinned nftables ruleset:

1. `kval-node transport-preflight` returns `verified: true`.
2. Repeat during tunnel loss, during reconnection, and against a rogue BSSID —
   each must fail closed with the expected reason code.
3. Confirm with an independent packet capture that no plaintext underlay egress
   occurs while the tunnel is down.

No pass is claimed in this repository before those artifacts exist.
