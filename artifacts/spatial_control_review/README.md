# Independent passive review of spatial control

`audit_actual.py` is an independent stdlib/NumPy reducer outside the scientific
source collector. It refuses to inspect fresh scores until both the complete
official report and its sealed manifest exist. It imports no project module,
model, trainer or kernel and executes no forward, fit, RT solve or packet run.

The review checks exact source/evidence membership, retained Git publication
bytes, seven prior closures, all 48 fresh arm receipts and both disabled-parity
receipts. Numeric checks cover forecast availability and output hashes, causal
64-token contexts reconstructed from history, real local scan payload/server
report joins, per-robot packet counts, worst decile, gaps, end-pending upper
fractions, actual scan interruptions and shared cadence, central command
creation/delivery/expiry, local authority and own target/current evidence,
post-tick validation budgets, and the complete reported means/contrasts.

Own pilot capture eligibility is checked against reconstructed association and
receiver-availability histories. Guard rows retain exact measurement/arrival
provenance and newest own heard/censored reports. New server deliveries can
refresh a queued guard, while locally completed undelivered RF cannot.

A passing result establishes consistency of the archived bytes and reductions.
It does not authenticate physical measurements, prove model correctness,
rerun the simulator, qualify packet risk or establish production readiness.
The study remains a finite simulation diagnostic with negative outcomes kept.

Run only after completion:

```bash
uv run --frozen python artifacts/spatial_control_review/audit_actual.py
```

The script writes a new `result.json` exclusively; it never rewrites the
sealed study or silently replaces a previous review result.
