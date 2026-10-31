# External code and modifications

`ns3/warehouse-wifi.cc` adapts the GPL-2.0-only ns-3 Wi-Fi infrastructure and
HE examples. Its file retains upstream copyright and SPDX notices. Changes:
read a fleet topology and candidate association; configure actual Wi-Fi
PHY/MAC, contention, MinstrelHt rate adaptation and UDP flows; export each
robot's throughput, unreceived packets and delay histogram quantiles. It is a
static association snapshot, not a handover or moving-fleet experiment. Its
log-distance channel does not reproduce the warehouse twin's rack geometry.
The runner must report these limits with the result. Linked ns-3 derivative
binaries retain GPL requirements. Upstream licence text is acquired alongside
the exact pinned source by `scripts/setup_warehouse_v2.py`.

`warehouse_v2/phy.py` adapts the simulation chain in NVIDIA Sionna's
`tutorials/phy/Hello_World.ipynb` (Apache-2.0): source, encoder, QAM mapper,
AWGN channel, demapper, decoder. It adds deterministic seeds, per-point raw
error counts, bounded batches and a provenance-bearing JSON report. Its
5G LDPC coding is explicitly a generic PHY sensitivity study, not IEEE
802.11 coding. It is not substituted for validated Wi-Fi link curves.

LWM and V-JEPA are research references. No LWM or V-JEPA checkpoint is
represented as downloaded, adapted or evaluated by these modules.
