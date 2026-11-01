# Bill of materials: from simulation to a measured claim

Everything in the canonical results is simulation. This list is what we need
to *measure* the same claims on real hardware. Prices are not estimated here;
each line needs a quote.

## Required for the canonical claim
| # | Item | Qty | Why we need it | Experiment it unlocks |
|---|---|---|---|---|
| 1 | Wi-Fi 7 enterprise access points: multi-link, 802.11r/k/v, 6 GHz, controller API | 4 | Overlapping cells along a test aisle, so a robot can hold two AP paths and roam between them | Cross-AP diversity, fast roam, Wi-Fi 7 default baseline |
| 2 | Additional SmartCar robots matching SmartCar.zip (Raspbot, Raspberry Pi 5, Intel BE200) | 9 | Brings the fleet to the 10-robot milestone used in the 10-robot results | 10-robot fleet trial |
| 3 | Second client radio per robot: USB Wi-Fi 6E/7 adapter with an in-kernel Linux driver | 12 (10 + 2 spares) | The second AP path, if the BE200 cannot hold two client connections (the probe checks this) and the onboard radio cannot be freed | `vertical_client` arm on real robots |
| 4 | Out-of-band management path per robot (USB Ethernet for the bench, or a management VLAN on the customer Wi-Fi) | 10 | Frees the onboard radio (`wlan0`, today the hotspot) to act as the second client radio | `vertical_onboard` arm on real robots |
| 5 | Monitor-mode capture adapters and a capture laptop | 2 adapters, 1 laptop | Independent packet capture and timing, so results are not self-reported by the robots | Handover stage timing, miss attribution |
| 6 | USB power meters | 2 | Second-radio power draw, which the simulation does not model | Battery impact of dual association |

## Needed for a like-for-like 5G comparison
| # | Item | Qty | Why |
|---|---|---|---|
| 7 | Access to a private 5G test network (rented lab or partner site) with 5G USB modems | 1 site, 2+ modems | The 5G numbers in the deck come from a generous model. A real head-to-head needs a real 5G network on the same route. |

## Optional (research, not needed for the claim)
| # | Item | Qty | Why |
|---|---|---|---|
| 8 | Programmable RF attenuators and a shield box | 1 set | Repeatable bench handovers and fades |
| 9 | Software-defined radio pair | 2 | Layered-modulation and spectrum-sensing research; both lost or did not matter in simulation |

## Software and access (no purchase)
- **Run the read-only probes on the car first:** the capability, lever and handover probes. They decide whether items 3 and 4 are needed: if one BE200 can hold two client connections on two channels, the second radio becomes a software feature.
- **Admin access to the AP controller** for 802.11r and channel plans.
