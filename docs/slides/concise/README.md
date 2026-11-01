# RadioMap + WirelessJEV — ten visual slides

A concise presentation of the current warehouse system, using **real 3D scene
models** rendered in Blender: a fleet warehouse, exploded architecture, learned
radio-map surface, mission console, confidence gate, independent AP paths,
cross-layer stack, paired comparison, operating envelope and validation aisle.
It explains what the modules do and why they belong together.

- [PDF](generated/wirelessjev_10_slides.pdf)
- [Editable PowerPoint](generated/wirelessjev_10_slides.pptx)
- [Standalone browser deck](generated/wirelessjev_10_slides.html)
- [Interactive 3D explorer](generated/explore_3d.html): orbit and zoom the warehouse, architecture and path-diversity models.
- [Contact sheet](generated/contact_sheet.jpg)
- [Blender / GLB models and render assets](generated/wirelessjev_3d_assets.zip)
- [Presenter notes and sources](NOTES.md)

Every slide has its own 3D render. Titles, annotations, metrics and speaker notes
remain editable in PowerPoint. The PDF has selectable text over rendered models.
The browser deck embeds its images; the 3D explorer embeds its library and GLB
geometry. It uses WebGL and should be opened in a supported browser. Arrows move
through the slides; N shows notes, O the overview and F fullscreen.

The models are **schematic illustrations**, not photographs or measured RF rays.
The elevated radio-map mesh encodes a genuine model prediction on a spatial
grid; its height is a signal encoding. Performance numbers are calculated from
the archived six-scene campaign, not generated from these illustrations. The
second-client-radio assumption, stylised cellular comparator and stress-case
reversal remain visible. Physical fleet qualification is pending.

## Rebuild

```sh
uv sync --frozen
# Refit the illustrative map grid and extract immutable campaign counts:
uv run --frozen python scripts/prepare_visual_deck_data.py
# Requires Blender 4.3+, available in this environment. CPU rendering is supported:
uv run --frozen python scripts/build_visual_deck.py --render
# Re-export from existing 3D renders, without rendering again:
uv run --frozen python scripts/build_visual_deck.py
```

[deck.json](deck.json) specifies the story and source notes.
[render_models.py](render_models.py) creates the actual meshes, lighting, camera
composition and marker projections. The renderer uses Cycles, 64 samples and
2304 × 1296 RGBA images; no photograph or AI-generated measurement is involved.
Three scenes also ship as editable `.blend` and interoperable `.glb` models.
The manifest records sources, outputs and render metadata. Blender CPU output
may vary slightly by version; archived render hashes identify this build.

The self-contained explorer uses the official **@google/model-viewer 4.0.0**
bundle from the npm registry under [Apache-2.0](vendor/LICENSE). The pinned bundle,
upstream package metadata and license are vendored for offline use; no CDN is
needed at viewing time. Blender's uncompressed glTF export does not require Draco.

The [32-slide technical appendix](../README.md#technical-appendix) remains available
for detailed mechanisms, research figures and evidence. The original Claude-hosted
deck has not been republished by this export.
