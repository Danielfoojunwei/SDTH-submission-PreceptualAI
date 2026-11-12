# Sealed nested-radio results viewer

This viewer reads archived reported targets, public CAD, saved predictions and
completed reports. It never imports a project model or runs a fit, ray trace,
packet simulation or diagnostic forward. The generator refuses final output
until both `artifacts/nested_radio/report.json` and its evidence seal exist.

After the registered experiment and seal complete:

```bash
python artifacts/nested_radio_review/viewer/render_results.py
python artifacts/nested_radio_review/viewer/verify_browser.py
```

The second command uses installed system Python Playwright and Chromium. It
serves unchanged local repository bytes over an ephemeral loopback HTTP server,
checks all 576 main-layout/state/AP/carrier combinations, all nine controls,
all 60 checkpoint scores and all fifteen gate diagnostics, and records input
hashes before/after. No heavy browser work should run during timed scientific
inference. Source-only Python/JavaScript syntax checks can run before execution.

Open generated `index.html` through a local HTTP server or a compatible artifact
viewer. It is standalone and loads no external assets. The linked 3D cutaways
use actual public obstacle-box vertices and query coordinates; only genuine
heard values are point targets. Censored bounds and unknown paths stay distinct.
The geometry panel illustrates selected seed zero; primary estimates retain all
three members and every geometry. Gated coefficient-prior scales are not final
predictive uncertainty or a physical-identity certificate. Global-eight raw
kernel features are inactive/untrained and are labeled accordingly.

This is a static RF-only experiment: zero new warehouse simulations, hardware
actions, temporal learning or service qualification. The prior experiment and
its viewer remain unchanged. `provenance.json` binds all direct scientific inputs,
outputs and generator source; `browser_review.json` binds the final checked HTML,
verifier and screenshots. These files are created only from actual final data.

Final browser review passed against 217 exact input hashes, with desktop and
mobile screenshots including separate mobile hero/geometry views. The fourth
contrast also displays its six-decimal agreement increment and its limited RF
interpretation. `browser_attempts/` retains the initial verifier negative-zero
formatting failure and subsequent actual reviews; no scientific data changed.
