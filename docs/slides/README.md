# Current-system presentation

The primary presentation is the **[10-slide visual deck](concise/README.md)**:
real Blender models, cinematic warehouse scenes, an exploded architecture,
spatial radio state and independent AP paths. It includes PDF, editable
PowerPoint, a standalone browser deck and an interactive 3D model explorer.

## Technical appendix

This deck explains **RadioMap + WirelessJEV as it is implemented in the canonical native-stack source**, with a separate boundary around supervised vehicle tools. It focuses on design rationale, interfaces, mechanisms and evidence. It has no development timeline or version-comparison section.

- [PowerPoint](generated/wirelessjev_system_architecture.pptx): editable titles, takeaways and speaker notes; one research figure per slide.
- [Vector PDF](generated/wirelessjev_system_architecture.pdf).
- [Standalone browser deck](generated/wirelessjev_system_architecture.html): arrows/Page Up/Page Down, N for notes, O for overview, F for fullscreen.
- [Visual overview](generated/contact_sheet.jpg).
- [SVG/PDF/PNG research figures](generated/wirelessjev_figure_assets.zip).
- [Presenter script](../PRESENTATION_SCRIPT.md).

The source is [deck.json](deck.json); [figures.py](figures.py) draws every figure. Each slide names its evidence class and source files. The original linked Claude-hosted deck has not been republished by this build.

## What the deck covers

Requirements and responsibility boundaries; the complete feedback architecture; control timescales; the warehouse model; PHY assumptions; censored radio-map learning and uncertainty; mission compilation; AP and link allocation; confidence gating; roaming; cross-AP recovery; failure attribution; ablations; supervised outcome handling; operator visibility; the cellular comparator; paired evaluation; reliability and tail metrics; operating limits; hardware options; traffic handling; capability coverage; field validation; and the bill of materials.

The Studio image is a real Chromium capture of the implemented fleet-map page. It shows a recorded warehouse replay, **not live car telemetry or a dual-radio campaign replay**. The coverage maps and trace images are simulator-rendered illustrations; performance charts use the archived campaign reports. The scan example uses actual output from the simulated capped-scan function. No stock photograph or generated RF measurement is used.

## Rebuild

```sh
uv sync --frozen
uv run --frozen python scripts/build_research_deck.py
```

The builder validates source references, text fit, page counts, unique substantive visuals and editable speaker notes. It exports 32 slides in all three formats, previews, standalone SVG/PDF/PNG figures and a manifest with source/output hashes. Figure caching is accepted only when the figure-affecting source hashes and slide-ID list match; the complete manifest still records all presentation sources:

```sh
uv run --frozen python scripts/build_research_deck.py --reuse-figures
```

Dependencies are locked in the development group: matplotlib, python-pptx, ReportLab and PyMuPDF. PowerPoint keeps slide text editable; diagrams also ship as standalone editable SVGs. The PDF retains vector diagrams and selectable text.

## Evidence boundaries

All performance comparisons are simulations. The primary comparison has six paired scenes, not millions of independent environments. Its cellular model is a scaled abstraction, not a full 3GPP stack or commercial private-5G installation. Its successful two-path arm assumes a second client radio; alternative shared-card and onboard-radio routes still need hardware verification. The stress case favors the cellular model. The native planner uses explicit numerical policies rather than a formally verified global optimizer. Richer contract proposal checks and supervised car tooling remain separate interfaces.

The deck deliberately does not promote unarchived cross-AP/ten-robot replications, protocol support that has not been probed, or a field uptime/safety guarantee.
