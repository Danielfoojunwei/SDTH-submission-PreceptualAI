# Spectrum-sensing fabric

The fabric turns a fleet of Kval nodes into a distributed, **passive** RF sensor
network. When cameras are down, degraded, or simply absent, the radio spectrum
itself carries usable situational intelligence: which communication bands are
clear, whether interference or jamming is present, and whether emitters are
active nearby. This is coarser than computer vision and is never presented as a
replacement for it — but it works in the dark, through smoke, and around
corners, and it costs very little compute.

## Receive-only, by construction

Kval spectrum sensing **only listens**. The node never transmits, jams,
generates a waveform, or sets frequency, gain, EIRP, or transmit power. Those are
not implemented and are not leasable actions. What Kval does:

* loads a signed `SpectrumPolicy` — an operator allowlist of bands, each with a
  regulatory region, frequency range, resolution bandwidth, occupancy baseline
  and alert threshold, and a noise-floor baseline;
* asks a **root-owned spectrum sensor adapter** (a separate, operator-approved
  radio process, over a Unix socket) for passive occupancy observations on those
  bands only;
* refuses any observation that is not marked `passive`, or that is for a band
  outside the signed policy;
* derives bounded intelligence from the observations with a transparent,
  deterministic analyzer.

The trust boundary mirrors the existing radio standards agent: Kval verifies
provenance and stays inside the allowlist; the adapter owns the radio and is
responsible for regulatory compliance. See `schemas/spectrum-agent-v1.schema.json`
and `schemas/spectrum-policy-v1.schema.json`.

## Vault nodes and low-compute sub-nodes

```
        signed policy + models                 passive occupancy
 operator ───────────────────▶ vault node ◀───────────────────── sensor adapter
 (laptop)                     (Jetson Orin)                        (root-owned)
                                  ▲   │
              lightweight scans   │   │  fused intelligence, models, power/time sync
                                  │   ▼
                      sub-node ───┴──── sub-node   (drones / small radios, no Jetson)
                    (thin sensing + connectivity to a vault node)
```

Not every node can carry a Jetson. A drone or a small ground sensor may have
only a low-power SoC and a radio, yet it can still contribute to and benefit from
the fabric:

* **Sub-nodes run the thin path.** The parsers, policy checks, safety limits,
  capability-lease verification, and the spectrum *analyzer* are pure Python over
  small structured inputs — no accelerator, no model weights, kilobytes of state.
  A sub-node can sense within its signed policy, apply power/duty limits, verify a
  lease, and act on the analyzer's band-availability output entirely locally.
* **Vault nodes carry the heavy path.** Anything that needs real compute —
  training, evaluation, multimodal model inference, federated aggregation — runs
  on the vault node (Jetson) or the DGX coordinator, never on the sub-node. The
  sub-node ships provenance-tagged observations up; the vault node fuses across
  sub-nodes and returns compact, signed guidance (which bands are clear, whether
  the area shows jamming, updated thresholds).
* **Graceful degradation.** If a sub-node loses its link to the vault node, it
  keeps running the thin path on its last signed policy and fails closed on
  anything it cannot verify locally — it does not fabricate an answer or exceed
  its bounds. The connectivity recovery state machine (`recovery.py`) governs the
  reconnect/backoff/failover behaviour.

The design goal the request names — "the capabilities and the learning in a less
compute-power package" — is met by **splitting the loop by cost**: bounded
observation, verification, and deterministic reaction are cheap and live at the
edge; learning and fusion are expensive and live at the vault/coordinator, with
signed results pushed back down.

## What intelligence the fabric yields

The analyzer (`spectrum.SpectrumAnalyzer`) is deterministic and thresholded
against the signed policy. Per band it reports one of: `CLEAR`, `OCCUPIED`,
`CONGESTED`, `ANOMALOUS`, `JAMMING_SUSPECTED`, or `INSUFFICIENT_EVIDENCE`, plus:

* **Band availability** — which approved comms-candidate bands are clear enough
  to use, so an autonomous vehicle or operator can keep a link when the primary
  channel is contested.
* **Interference / jamming suspicion** — high occupancy *together with* a
  noise-floor rise above the policy baseline is flagged as `JAMMING_SUSPECTED`.
  It is reported as *suspected*, never asserted, because occupancy sensing cannot
  by itself distinguish a jammer from heavy legitimate traffic.
* **Presence / "eyes on the ground"** — a non-zero detected-emitter count in a
  band indicates something is transmitting nearby. With a direction-finding
  adapter, an optional bearing accompanies it. This gives a rough presence and
  direction signal when cameras cannot.

Every assessment carries a confidence derived from the observation quality, and a
band with no fresh, error-free observation is `INSUFFICIENT_EVIDENCE` — never
assumed clear.

## Opportunities beyond navigation

* **Comms resilience** — pick the clearest approved band for the connectivity
  loop; avoid contested channels before a link degrades.
* **Interference and jamming *awareness*** — detect and localise (with DF)
  disruption so a fleet can route around it or fall back to an alternate band or
  NTN link. This is a defensive, situational-awareness capability.
* **Distributed sensing coverage** — many cheap sub-nodes cover more area than
  one expensive sensor; the vault node fuses their views.
* **Cross-modal corroboration** — spectrum presence can corroborate or fill gaps
  in a degraded camera feed inside the multimodal state.

## Honest limits

* Occupancy/energy sensing is **not** imaging. It gives presence, congestion, and
  interference indicators, not identification or fine localisation.
* Accuracy, sensitivity, and any bearing estimate are entirely the adapter's;
  Kval reports what a signed adapter provides and its stated quality.
* Jamming detection is a heuristic and can be fooled by unusual-but-legitimate
  traffic; it triggers investigation, not an automatic offensive response.
* No performance number is claimed here. Sensitivity, false-alarm rate, and
  bearing error must be measured with the real adapter and recorded as evidence.

## On offensive capabilities

The request asked what offensive opportunities the sensing might create. Kval's
answer is deliberate: **the fabric is defensive and receive-only.** It does not
implement jamming, spoofing, directed RF energy, covert transmission, or any
transmit path, and no lease can authorise one. The offensive threats it *defends
against* — jamming, spoofing, direction-finding of friendly emitters, traffic
analysis — are catalogued in `docs/SECURE_TRANSPORT_AND_EW_THREAT_MODEL.md`.
Building an actual electronic-attack capability is out of scope for this project.

## Running it

```bash
# On the operator machine: sign a spectrum policy.
python3 scripts/provision_policy.py spectrum \
  --input config/spectrum-policy.template.json \
  --private-key operator.private.pem --public-key operator.public.pem \
  --output spectrum-policy.signed.json

# On the node: configure spectrum_policy_path + spectrum_sensor_socket, then:
kval-node spectrum-scan --config /etc/kval/node.json
```

With no sensor adapter present the command fails closed with a clear reason and
never fabricates occupancy data.
