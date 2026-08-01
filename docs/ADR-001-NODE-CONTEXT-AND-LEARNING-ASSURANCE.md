# ADR-001: Signed node context before learned control

**Status:** Accepted

**Date:** 2026-08-01

## Context

Every Kval node will eventually run local inference and participate in a
distributed learning cohort. A model update is meaningless unless the system
can prove which node produced it, what that node was allowed to do, which
hardware and catalogue were active, and which observable links informed the
decision.

## Decision

Each node has an offline-signed identity manifest. The manifest binds node ID,
role, deployment, learning cohort, allowed model families, allowed service
classes, expected hardware, context schema, and validity window. Each decision
contains a runtime context digest derived from the verified manifest, live
platform report, signed catalogue, current profile, observable profiles, and
decision engine.

Learned inference remains advisory until a separately signed model manifest and
real evaluation evidence are available. The deterministic Safety Shield always
retains final authority.

## Options considered

### Unsigned configuration only

Simple, but any local configuration drift can silently change a node's claimed
role or cohort. Rejected.

### Central identity lookup on every decision

Provides central control but fails during disconnection, which contradicts the
resilient-node requirement. Rejected as the primary source.

### Signed local identity with periodic fleet reconciliation

Works offline, is auditable, and allows central revocation/renewal when links
return. Accepted.

## Consequences

- Training examples and model updates can be attributed to an authorized node
  context rather than only a mutable hostname.
- Expired or mismatched identity fails node startup.
- Node provisioning now requires a signed identity manifest in addition to the
  signed connectivity catalogue.
- TPM-backed device attestation remains a later hardware proof gate.
