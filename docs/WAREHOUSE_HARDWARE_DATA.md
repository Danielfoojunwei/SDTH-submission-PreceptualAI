# Real-data qualification required for warehouse v2

The cloud has no configured SmartCar network, pinned host-key file or car
credential. Physical tests have not run. Do not rename simulator observations
as hardware data, infer packet delivery from a reported PHY bitrate, or fill
missing authentication/DHCP timestamps with estimates.

Use the existing `docs/ONE_CAR_TEST.md` preflight, capability/status collection,
scan, proposal and supervised switch. The management radio remains excluded.
The existing request/response timing measures client-observed duration only;
it cannot separate authentication, IP recovery and first successful packet.

A field dataset needs actual observation timestamps and clock identity,
device/radio identifiers, scan RSSI and its result cap, association before and
after each action, delivered byte/packet counters and measurement intervals,
offered demand, and route/position metadata. Busy-time, retries, queue lengths,
MCS and CSI are optional observed fields with explicit missingness, not
invented values. Record driver/AP versions and whether AP control is available.

For switch decomposition, obtain events from the actual on-car network stack
or instrument its existing command implementation after inspecting the real
source. Preserve event clock domains and distinguish request submission,
association, authentication, IP readiness and a packet received at the intended
endpoint. An IP may be retained, so DHCP is not automatically a stage in every
handover. A successful shell command is not first-packet evidence.

Qualification then progresses from stationary one-car measurement to a small
fleet and finally a 100-robot trial, with per-robot deadline/outage/service
metrics and an operator-defined fallback. Simulated critical-rate violations
are not a validated model of application-level safety deadlines.
