# Shared and AP-local RF evidence viewer

`index.html` is a standalone offline presentation generated only after the
completed RF and native reports exist. It uses no external scripts, fonts,
images or data requests. Open it directly in a browser.
Managed Chromium verification serves the unchanged files over a temporary
loopback HTTP server because its policy blocks `file://` navigation.

The primary summary uses all three trained members and all 24 TEST geometries.
The rotatable scene inspector uses the VAL-selected hierarchy's fixed seed 0
and the selected numerical map on all 72 sparse TEST material-state cases.
Heard values are reported quantization centers and intervals; censored links
show upper bounds, and unresolved paths remain unknown. The inspector is
descriptive and never substitutes for the primary estimator.

The warehouse panel is a separate simulation endpoint. Its matched methods
share budgets and selection/guard rules, while their closed-loop histories may
differ. The 16-AP geometry and 6.5 GHz forecast are unqualified extrapolations.
RadioMap remains the application default.

From the repository root, after the parent study finishes:

```sh
python artifacts/hierarchical_radio_review/viewer/render_results.py
python artifacts/hierarchical_radio_review/viewer/verify_browser.py
```

`provenance.json` binds every direct input, the generator, template and outputs.
`browser_review.json` records actual browser checks and screenshots. Generation
and viewing do not fit models, run inference, ray trace or simulate packets.
The renderer needs NumPy; the browser check uses the environment's installed
Playwright and Chromium. These are presentation dependencies, not changes to
the frozen scientific environment.
