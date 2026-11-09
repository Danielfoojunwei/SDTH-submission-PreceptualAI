# Current research Studio export

A separate local view now exposes the current research evidence without
editing the original Studio generator or assets:

- [Standalone evidence view](../../artifacts/research_studio/index.html)
- [Augmented data bundle](../../artifacts/research_studio/bundle.json)
- [Input/output provenance](../../artifacts/research_studio/manifest.json)
- [Browser check](../../artifacts/research_studio/browser_review.json)
- [Rendered preview](../../artifacts/research_studio/preview.png)

The original `studio/data/bundle.json` is copied with every legacy value
preserved. A versioned `research_evidence` section adds measured array-power,
static spectral RF, real held-phone calibration, complex CSI, adaptive assay,
native service diagnostic, geometry-operator profile, passive context diagnosis
and exploratory frozen visual features. A separately verified R6 software-test
receipt adds the read-only planning/journal preview card. Evidence kinds, units, selected model,
negative results and claim limits remain separate. Each card shows at most two
core findings; complete metrics, caveats and rights remain expandable. The
factory headline retains the negative 2.650 versus 2.108 dB comparison, and the
CSI/native cards expose their poor outcomes directly. The current native candidate
is regression, not proof of a JEPA-objective gain. The 2.831× measurement is
operand-call performance only.

Completed scientific tracks require their reports to appear in the sealed
evidence map. The exporter checks all source/evidence/document hashes in those
seals. Complex-CSI export additionally requires every external prediction shard
and transport capsule to match its recorded hash; missing cache bytes are
fatal, not silently skipped. The output manifest records source hashes and
the number of repository/external files verified. No model or NPZ array is
loaded, and no scientific fit, forward or simulation runs during export.

Until `artifacts/adaptive_radio/verification/manifest.json` exists and verifies,
the adaptive card says **pending seal** and exports no unsealed TEST metrics.
Run the exporter again only after that completed closure is available to
refresh its status from actual final evidence. A DEV permission, a selection
receipt or partially written TEST report does not activate the result card.

The archived **2,547 passed tests** are parsed from the hash-bound original
whole-suite log and retain their original tracked-source scope. They exclude
this new exporter. A separate source-bound receipt records **ten exporter
tests** and identical before/after hashes for its four source/test/script
files. Do not add these overlapping software checks to scientific sample or
training-run counts. The R6 card's 77 software checks also retain their own
receipt and scope; no combined whole-suite count is claimed.

Rebuild with `PYTHONPATH=src uv run --frozen python
scripts/export_research_studio.py`. Set `CHANNEL_SCENE_CACHE_ROOT`, or pass
`--cache-root`, if the byte-identical CSI prediction cache has been relocated.
The standalone view embeds the JSON and uses no external assets. Its browser
check uses a temporary loopback HTTP server; the managed browser blocks direct
`file://` navigation. For manual review, serve `artifacts/research_studio`
locally with Python's HTTP server and open its loopback URL.

This is a reviewable local export. **The external Claude Studio URL and slide
deck have not been republished**, and no credentials or publication permission
are implied by rebuilding these files.
