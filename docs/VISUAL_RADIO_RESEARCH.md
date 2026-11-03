# Visual radio representations: what DINOv3 can add and how to test it

**DINOv3 is a strong candidate visual encoder, not an established RF world
model or a demonstrated repair for our JEPA.** The useful hypothesis is that
spatially grounded visual features can add information about visible geometry
and moving blockers that a sparse radio history lacks. That hypothesis needs
matched tests against explicit geometry, RF-only inputs and supervised temporal
learning. A failure to beat RadioMap does not by itself diagnose representation
collapse.

This document is a research design. It reports primary-source reading and
official-code inspection, not new training, a completed DINOv3 integration,
benchmark superiority or a novelty claim. The [source audit](../artifacts/visual_radio/literature/audit.json)
records retrieval hashes, exact review scopes and bounded search coverage.

## What the official DINOv3 objective actually learns

The [DINOv3 technical report](https://arxiv.org/abs/2508.10104), §3.2 and §4.1–4.3,
describes several complementary objectives:

| Objective | Mechanism and relevant boundary |
|---|---|
| DINO image-level loss | A student predicts teacher prototype distributions across views/crops of the same image, using the CLS representation. This encourages useful global semantics; it is not future radio-state prediction. |
| iBOT patch loss | A masked student predicts teacher patch distributions at masked spatial locations. This provides local information beyond the global token. It does not label electromagnetic material properties. |
| KoLeo regularization | Encourages normalized representations to spread within the chosen comparison group. It helps representation diversity, without guaranteeing that retained dimensions explain wireless outcomes. |
| Gram anchoring | During refinement, matches the pairwise similarity structure of normalized patch features to a separate Gram teacher with better dense consistency. Higher-resolution teacher features can be downsampled before comparison. This preserves local relationships while allowing individual features to move. |

The published initial objective is
\(L_{\mathrm{DINO}}+L_{\mathrm{iBOT}}+0.1L_{\mathrm{DKoLeo}}\).
The refinement adds a weighted Gram term. For normalized patch matrices,

\[
L_{\mathrm{Gram}}\propto
\left\|X_SX_S^\top-X_GX_G^\top\right\|_F^2.
\]

The official [code receipt](../artifacts/visual_radio/literature/dinov3_code.json)
pins commit `6876159a11b4df116f30f667f8c9888617df0751`. Inspected
`ssl_meta_arch.py` separately computes CLS cross-entropy, masked-patch
cross-entropy, KoLeo and optional Gram losses. Teacher assignments use
Sinkhorn–Knopp. `gram_loss.py` normalizes features and compares their patch
Gram matrices by MSE; it also exposes optional negative-similarity handling.
The inspected refinement configuration uses a separate periodically refreshed
Gram teacher, rather than simply the continuously updated EMA teacher.
Configuration switches matter: inspected YAMLs leave distributed KoLeo off,
so paper-level recipe descriptions should not be mistaken for proof that
every released configuration exactly reproduces the reported training run.

The paper diagnoses a specific degradation: global classification can improve
while patch locality and segmentation deteriorate during extended training.
Patch similarities become less spatially selective and more aligned with CLS.
Gram anchoring addresses that dense-feature problem. This is more specific
than saying that every latent, every task or every JEPA has collapsed.

The official backbone returns separate `x_norm_clstoken`,
`x_norm_patchtokens` and storage tokens. A CLS-only probe tests global image
information. A dense probe retains an explicit patch grid, where a rack edge,
occluding vehicle or robot-to-AP corridor can be localized. CLS may still encode
geometry implicitly, so its inferiority is an experimental question. Dense
tokens also require a spatially aware decoder: averaging them immediately
removes their explicit coordinate structure.

DINO teacher prototype probabilities are training assignments. They are not
calibrated probabilities of a packet deadline miss or typed controller action.
A downstream service head needs its own labels and calibration evidence.
The repository has a custom DINOv3 license, inspected as `LICENSE.md`; it is
not labeled MIT here. No weights or datasets were downloaded for this review,
and no upstream source was copied into the application.

## Visual structure is useful, but appearance is not RF causal state

Camera images can reveal obstacle position, size, motion and some surface
appearance. Radio propagation also depends on transmitter/receiver geometry,
carrier frequency, antenna pattern/orientation, material properties,
penetration, multipath and interference. A path-sum illustration is

\[
H(f,t)=\sum_p \alpha_p(f,t)\,e^{-j2\pi f\tau_p(t)}.
\]

Object layout constrains possible paths; pixels do not uniquely determine their
complex coefficients. Visually similar panels can have different RF losses;
an interferer outside the camera view can change service without changing the
image. Moving one object can also change phase-sensitive interference at a
scale much smaller than a semantic object. Queue occupancy and association
interruption are not properties of visual appearance.

The current warehouse kernel independently samples hidden Nakagami fading and
Markov blockage. Those realized draws are not caused by visible blocker
trajectories. A CAD-image model can learn a geometry/propagation prior; it
cannot causally predict independent hidden draws from an unchanged image.
To test visible-blocker anticipation, object trajectories must causally feed
both a synchronized camera observation and the propagation calculation.
Rendering hidden blockage flags, true SNR, future interference or fitted target
values into colors would leak privileged information.

A rasterized CAD plan is a legitimate input when that CAD is available to the
deployed controller, but it is a **CAD representation experiment**, not proof
of camera perception. Explicit geometry already contains much of its content;
the geometry-only baseline is therefore essential. Material annotations may
be inputs only if genuinely available, with their source and uncertainty
declared. Randomly coloring shapes does not create a real visual dataset.

## Collapse, aliasing and task mismatch require different remedies

| Possible cause | Diagnostic | Implication |
|---|---|---|
| Constant or severely collapsed representations | Per-dimension variance, covariance spectrum/effective rank, pairwise distances and simple held-out geometry/RF probes, checked across meaningful input variation. | Investigate teacher/student dynamics, masking, optimization and diversity objectives. Low rank alone is not failure: useful abstractions can be compact. |
| Loss of spatial locality | Dense correspondence/segmentation or object-relative probes; patch similarity maps and CLS–patch alignment over training. | Preserve patch coordinates; consider a frozen dense encoder before retraining it. Gram anchoring targets this visual failure mode, rather than arbitrary radio error. |
| Observation aliasing | Nearly identical available histories produce different outcomes after changing hidden material, unseen interference or independent fading. | Add causally available measurements or model a conditional distribution. A larger image encoder cannot recover unobserved independent state. |
| Wrong task abstraction | Geometry probes work, but service/decision probes fail despite sufficient input information. | Train and evaluate service-relevant heads and action-conditioned dynamics; preserving every visual detail is not the objective. |
| Decoder, label or timing failure | Compare frozen-feature probes, numerical-only outputs, target units, at-risk masks, completed-observation timestamps and counterfactual continuation. | Fix the interface before attributing a bad controller result to representation collapse. |
| World-model rollout drift | Evaluate one-step versus open-loop versus observation-corrected horizons, with the same data. | Separate representation quality from accumulated transition error and action-distribution shift. |

These diagnostics should precede an architectural conclusion. A stochastic
belief can represent genuine ambiguity, but adding stochastic latents does
not itself calibrate them. Diffusion can represent multimodal futures at extra
compute cost; it cannot turn an unavailable observation into information.

## Preserve physical equivariance instead of imposing blanket invariance

Some nuisance changes should not change the service answer: for example,
relabeling APs while relabeling all inputs and action indices consistently.
Other changes should transform the answer. A joint coordinate-frame rotation
must also rotate robot/AP poses, camera calibration and antenna orientations;
an action vector or beam direction must transform with them. This is different
from declaring an arbitrary rotated photograph radio-equivalent.

Retain metric coordinates, camera intrinsics/extrinsics or a declared CAD
coordinate system, route timing, band/frequency, antenna configuration and
measurement age. Image resizing/cropping needs consistent calibration and
visibility handling. Scaling a warehouse at fixed wavelength changes distances
and loss; changing frequency changes propagation and receiver noise. Neither
is a generic invariance. Recompute physically consistent labels for those
augmentations rather than forcing the latent to ignore the change.

A practical candidate is a spatial visual/CAD branch plus a causal RF/service
history branch, with AP/robot-relative queries into the patch grid. Keep
unknown material and unobserved interference uncertain. Fuse by timestamps
and availability, rather than treating every camera frame and scan as a
simultaneous complete observation. This is a proposed adaptation, not a claim
that those mechanisms are new.

## Hierarchical planning must respect the actual clocks

| Level | Useful abstraction | Constraint retained |
|---|---|---|
| Slow scene/map reasoning | Stable geometry, uncertain materials, route-level coverage and candidate AP sequences. | Available CAD/perception only; metric scale and frequency remain explicit. |
| Intermediate planning | Bounded association options, scanning opportunities, upload-at-stop and route/service budgets. | Outage cost, proposal expiry, completion delay and a return to canonical recovery. |
| Fast native execution | Current packet age, queue state, receiver availability, MCS/retry decisions and immediate recovery. | Real packet deadline and feasible radios; a slow latent plan cannot postpone these checks. |

This is a proposed hierarchy, not three implemented controllers. In the native
study, packet slots are 10 ms, control deadlines are 50 ms, receiver/event
snapshots can be 100 ms and forecast issues occur at one second. A visual
encoder running every few hundred milliseconds can provide advance warning,
but cannot replace the fast packet loop. Charge capture, encoding, transport
and inference delays. A remote visual result becomes available only after
delivery/completion; a local encoder has its own compute and energy cost.

High-level options need explicit duration, termination and preemption rules.
Do not discard packet age or outage state through temporal pooling. A rejected
option must preserve the baseline's recovery state; a short option must release
control when its hold expires. Evaluate a flat planner at the same action and
compute budget, and isolate command-expiry changes from learned-model gains.

## Closest reviewed wireless work

| Primary source | Relevant mechanism and boundary |
|---|---|
| [Vision-Aided 6G: Blockage Prediction and Proactive Handoff](https://arxiv.org/abs/2102.09527), §IV–V | YOLO object boxes and observed beams feed recurrent future-blockage prediction and proactive handoff. Synthetic ViWi images and 28-GHz channels are synchronized. Visual object structure plus RF history for proactive wireless decisions is already direct prior art. Its one downtown scenario and sequence splits do not establish cross-warehouse Wi-Fi generalization. |
| [Environment Semantics Aided Wireless Communications](https://arxiv.org/abs/2301.05837), §III–IV-A | PSPNet semantic maps preserve object class/shape/layout; target-user location and selected semantics feed beam/blockage heads. CARLA/SUMO and Wireless InSite align visual and channel labels. Scene abstraction for radio is established; swapping in DINO does not automatically create a novel system. |
| [Coupled Control and Wireless World Models](https://arxiv.org/abs/2609.04851), §III | Visual robot dynamics and a wireless predictive latent model are coupled, with CSI/spectrogram/persistence-image alternatives and synchronized Gazebo/ROS/Sionna RT. This already links visual abstraction, stochastic state, radio prediction and robotic control. Our different inputs/task require a disclosed adaptation, not an asserted reproduction. |
| [Guess and Send / WorldGT](https://arxiv.org/abs/2610.29587), §III–IV | Spatial feature-cache dynamics, motion compensation, age/validity, ACK-synchronized receiver memory and packet counterfactuals guide semantic transmission. Spatial predictive abstraction and delivery-aware communication are established. Its video-perception metric is not our roaming/deadline leaderboard. |

As a data-method reference, [DeepSense 6G](https://arxiv.org/abs/2211.09769),
§III–V, explicitly calibrates sensor fields of view and aligns different
camera/radio/LiDAR rates with triggers and timestamps. Its real multimodal
measurements are valuable evidence that synchronization is part of the problem;
the dataset does not automatically supply warehouse deadline labels. No dataset
content was downloaded in this review.

## One narrow, falsifiable hypothesis

> With identical causal RF/service histories and controller constraints,
> spatially grounded DINOv3 patch features improve held-out-warehouse prediction
> and packet decisions over CLS, explicit geometry and RF-only features when
> visible objects causally affect propagation. An optional JEPA objective adds
> benefit beyond the identical supervised temporal encoder at a declared budget.

Failure on either clause rejects that clause; a useful visual encoder need not
imply that JEPA helps. The hypothesis does not assert uniqueness or global SOTA.
The comparison needs the following controlled ladder:

| Arm | What it isolates |
|---|---|
| Canonical RadioMap/controller | Strong existing system and preserved recovery behavior. Report any difference in its survey/information budget. |
| Geometry-only + common temporal/service head | Whether explicit distances, obstructions and poses already explain the gains attributed to an image encoder. |
| RF/service-only + same head | Whether the available radio/packet history already contains sufficient useful information. |
| Frozen DINOv3 CLS + common head | Global visual information; add a common RF branch in the matched fusion contrast. |
| Frozen dense DINOv3 + spatial head | Whether local patch structure helps beyond CLS. Keep output/head capacity and compute accounting explicit; dense tokens cost more. |
| Dense DINOv3 + geometry + RF/service fusion | Incremental multimodal value, including missingness and availability. Compare to geometry+RF fusion without DINO. |
| Supervised versus JEPA on the same fused encoder | Same architecture, inputs, labels, controller, data split and inference graph. Separately charge any extra pretraining data/updates. |
| Flat versus hierarchical planner | Whether hierarchy helps with the same model, feasible actions and charged decision budget. |

Start with frozen pretrained features and modest probes; this separates
representation transfer from another large training run. Then compare optional
adapters/fine-tuning with declared optimization and memory budgets. Include a
random/frozen-backbone control where it helps distinguish pretrained semantics
from an arbitrary nonlinear encoding. Artificial shape fixtures can check code
execution, but they are not measured visual/RF datasets or quality evidence.

## Generalization and failure tests that decide the claim

Split by **whole warehouses, collection sessions and trajectories before
forming temporal windows**, keeping overlapping frames/history out of separate
partitions. Preserve a validation set for architecture, feature selection and
thresholds; never tune those on the final test set. Use multiple independently
trained seeds and paired evaluation episodes, with uncertainty at site/episode
level rather than treating correlated packets as independent trials.

Test held-out layouts, rack dimensions, material families, AP placements and
antenna configurations; then test their compositions. Distinguish zero-shot,
few-shot calibration and fully retrained settings. Vary lighting/texture while
holding physics fixed, and vary material/interference while holding appearance
fixed. Include hidden/out-of-view blockers, crowds, independent fading,
different load/collision regimes, camera occlusion/dropout, stale frames,
pose/calibration error and modality loss. Cross-band/device tests must retain
frequency/noise/configuration inputs and compare physically matched baselines;
28-GHz visual blockage success does not establish 2.4/5/6-GHz Wi-Fi transfer.

Report representation probes, event calibration/proper scores, packet deadline
misses, outage tails, recovery, handovers and end-to-end compute/latency
together. A visually coherent feature map or lower latent loss is not a packet
improvement. A CAD-only gain is not hardware visual validation. A simulation
result for 100 cars is not a physical fleet trial, and success across sampled
warehouses cannot establish “any robot in any environment.”

The bounded search inspected four closest wireless method sources plus the
dataset reference and official DINOv3 paper/code. Two Crossref queries were
rate-limited; a DINOv3-specific wireless query does not prove absence of direct
applications. Guessed identifiers that failed or resolved to unrelated work
were explicitly excluded in the receipts. No first-of-kind, collapse-repair or
SOTA claim follows from this review.
