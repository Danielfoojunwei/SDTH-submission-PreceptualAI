# Keep packet critics outside unqualified switching

The packet-action JEPA profile was rejected on development. Its uncertainty
correction turned rejected proposals into ten-second stays and suppressed
canonical RadioMap recovery. The six fits and all failed development evidence
remain intact in [`PACKET_VALUE_RESULTS.md`](PACKET_VALUE_RESULTS.md).

[`ShadowPacketAdmission`](../src/wirelessjev/packet_control.py) always returns
the baseline's admission mask. Critic abstention, rejection, invalid advice or
an exception cannot replace that mask with a stay. Advice runs in one bounded
background worker; admission does not wait for it and drops shadow work while
the worker is busy. Explicit shutdown joins the worker outside actuation.
CPU contention and physical deadlines still need hardware qualification.

There is no active-override mode. The separate
`require_confirmed_packet_profile()` guard rejects a profile without completed
independent qualification, including this development-rejected candidate.
The main application default remains RadioMap. Shadow equivalence establishes
that advice does not alter the tested baseline packet decisions; it does not
establish that JEPA improves control.

An active critic would change policy continuation: uncertainty should retain
the canonical action, and reactive recovery must remain available. The existing
labels compare ten-second focal holds and therefore cannot automatically
qualify that new contract. Such work needs separately versioned state-fork
labels, development checks and a pre-holdout source seal. The eight reserved
confirmation seeds have not been accessed.

Actual missed commands cannot be recovered after their deadlines. Predicted
risk reservations are accounting, not a guarantee that factual excess losses
never accumulate. The assumed single-radio 90 ms scan/50 ms deadline already
creates approximately a one-percent raw loss floor at five-second scanning.
