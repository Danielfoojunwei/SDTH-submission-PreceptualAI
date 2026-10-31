# Warehouse PHY opportunities and ANSpect boundary

ANSpect's public SCOPE project description describes carrying additional
information on strong links while serving users at a conservative MCS, using
a modified srsRAN/Open5GS testbed:
https://ccixgtestbed.org/research/projects/SCOPE.html . It does not disclose
enough of the patented mechanism to reproduce or claim the company's results.
No ANSpect source or proprietary algorithm is included in this repository.

## Layers and available actions

QAM changes amplitude and phase; PSK changes phase. Both are modulation.
Other PHY opportunities include coding/code rate, spatial streams and MIMO
combining, beamforming, channel estimation, interference cancellation, power,
and OFDM bandwidth/subcarrier allocation. Scheduling and retransmission cross
PHY/MAC boundaries. Wi-Fi MLO and 802.11k/v/r are not custom constellations.
5G HARQ must not be substituted for ordinary Wi-Fi MAC retransmissions.

The SmartCar currently exposes a client BSSID-selection interface on wlan1.
Its customer AP channels are fixed inputs. Firmware may already perform rate
adaptation and spatial-stream selection. Rate, width, power, MLO or CSI control
must be established by the real capability probe before exposing actions to a
policy. No code here transmits a custom waveform on a BE200.

## Executed v2 studies

- `warehouse_v2.phy`: actual Sionna 2.2 source -> 3GPP LDPC encoder -> QAM
  mapper -> AWGN -> demapper -> decoder. Report raw bit/block error counts.
  This is a generic coded PHY sensitivity experiment, not Wi-Fi coding.
- `third_party/ns3/warehouse-link-curves.cc`: actual ns-3 NIST Wi-Fi
  payload-success probabilities for HT 20 MHz / VHT 80 MHz, two streams,
  1,200-byte payloads. `warehouse_v2.link_curves` integrates expected rates
  into the existing planner evaluator and sweeps 0/3/6 dB margins.
- `third_party/ns3/warehouse-wifi.cc`: actual packet-level Wi-Fi contention,
  MinstrelHt adaptation, UDP traffic and per-client delay/throughput for a
  candidate association. These static, default single-stream snapshots use
  log-distance propagation; they do not reproduce the twin's rack geometry
  or qualify a moving-fleet handover controller.

## Separate programmable-waveform experiment

An independently implemented research comparison could test ordinary
adaptive modulation/coding, time/frequency division and a layered waveform
with successive interference cancellation. A learned constellation is
another experimental arm, not an assumption that shifting QAM points helps.
Constrain equal bandwidth and average/peak power; include coding, pilots,
channel-estimation error, Doppler, receiver complexity and error propagation.
Report both users' delivered bits and weak-link reliability, not aggregate
spectral efficiency alone. Select parameters on validation channels and test
on different channel draws. Do not assert legacy Wi-Fi compatibility without
testing the preamble, pilots, EVM, spectral mask and actual legacy receivers.

This programmable-waveform experiment is a design, not an executed result.
It requires a compatible receiver implementation and eventually a controlled
SDR testbed. The warehouse controller and its PHY sensitivity studies do not
establish ANSpect-equivalent functionality.
