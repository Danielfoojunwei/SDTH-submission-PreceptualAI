# Protocol boundary

WirelessJEV and RadioJEV currently implement Wi-Fi warehouse connectivity.
They are not validated as protocol-agnostic models or controllers.

Mission groups, service floors, ordered priorities, application demand and
proposal receipts can be reused across protocols. Current radio inputs and
actions still assume AP metadata, BSSIDs, capped Wi-Fi scans, channel contention,
Wi-Fi link curves and roaming costs. Even a graph with variable fleet size is
not automatically protocol-agnostic.

A future common link interface would need protocol adapters that declare
available observations, permitted actions, resource ownership and transition
costs. Wi-Fi would expose association and supported client controls; private
5G/LTE would need available modem/RAN interfaces and independently verified
handover/scheduling semantics; Bluetooth would need connection/relay and
payload constraints. Missing observations must be represented explicitly.
Measured throughput, loss, delay and energy offer common service outcomes;
their predictors and probabilities still need validation for each protocol.

No 5G, LTE, Bluetooth or satellite adapter is claimed implemented by this
warehouse campaign. Customer AP channels remain fixed and no physical radio
was changed.
