# Real warehouse ray tracing and synchronized visual inputs

The bounded CPU probe completed **actual Sionna RT channel computation and
Mitsuba rendering from the same full warehouse geometry**, followed by real
pretrained DINOv3 encoding of the two rendered images. Desired and cross-link
couplings are nonzero and change between the two explicit metal-blocker poses.
This establishes a small synchronized simulation fixture, not a learned RF
prediction result or a real warehouse experiment.

The [receipt](../artifacts/visual_radio/rt_probe/receipt.json) records versions,
all upstream asset hashes, materials, antenna/device positions, camera
calibration, solver settings, complex-channel/image/feature hashes and explicit
validation checks. The [probe script](../scripts/probe_visual_radio_rt.py)
contains the complete procedure. Arrays and images stay outside the repository
at `/workspace/training/visual-radio/rt/`.
The [diagnostic figure](figures/dinov3_physical_radio.png) combines the actual
renders, explicitly outlined encoder crops, joint dense-feature PCA and path
couplings. It supplies a visual record, not a learned RF prediction score.

## Actual scene and physical assumptions

The scene uses all 53 original objects from the MIT-licensed
[Gazebo–Sionna warehouse assets](https://github.com/ICONgroupCWC/Gazebo-Sionna-RT-Integration/tree/2d37c650bde5af6e5f69e899aa4e4b01243d3160/environments/aws_warehouse/sionna),
with an original 0.5 × 2 × 3 m metal-box mesh added explicitly. That identical
mesh and pose participate in optical rendering and electromagnetic path
computation. Its two centers are `(0,-3,1.5)` and `(0,0,1.5)` metres. Material
colors are ordinary renderer inputs; radio power, predicted blockage and hidden
channel labels are never painted into the images.

The runtime is Sionna/Sionna RT 2.2.0, Mitsuba 3.9.1 and Dr.Jit 1.5.0 using
the LLVM CPU backend. The carrier is 5.8 GHz, with one isotropic vertically
polarized antenna element per device. Two transmitters and two receivers yield
a 2 × 2 coupling matrix indexed `[receiver, transmitter]`; diagonal links are
designated desired and off-diagonal links potential interferers. No concurrent
traffic, transmit-power schedule, thermal noise or MAC is imposed.

The scene's ITU materials supply frequency-dependent permittivity/conductivity
for metal, chipboard, plywood, brick, concrete, glass and wood. Their numeric
values are in the receipt; these are configured material assumptions, not
measurements calibrated to a particular warehouse. The bounded solve includes
LOS and specular reflection to depth two, 2,048 samples per source and a 4,096
path cap. Refraction, diffuse reflection and diffraction are disabled.
Convergence and omitted-path error have not been established.

## Observed couplings

The actual incoherent sums of path power, `sum(abs(a_path)**2)`, were:

| Blocker pose | Desired RX0 ← TX0 | Cross RX0 ← TX1 | Cross RX1 ← TX0 | Desired RX1 ← TX1 |
|---|---:|---:|---:|---:|
| `(0,-3,1.5)` | 5.63e-8 | 1.33e-6 | 1.35e-6 | 4.29e-8 |
| `(0,0,1.5)` | 1.11e-6 | 7.26e-7 | 7.40e-7 | 1.51e-6 |

These are simulated dimensionless channel couplings, not packet delivery,
observed interference traffic, SINR, throughput or a learned model's gains.
The arrays also retain complex baseband impulse responses, absolute path
delays and coherent center-frequency powers `abs(sum(a_path))**2`, which differ
from incoherent path-power sums. No analytical or synthetic channel fallback
was used.

An initial placement across obstructed rack interiors returned zero found
paths for both poses. That attempt is preserved and explicitly excluded in
[its receipt](../artifacts/visual_radio/rt_probe/initial_geometry_attempt.json),
with original arrays/images and script archived under `initial_...` in the
external work directory. It is not useful visual/RF variation evidence.
Geometry-only downward floor-ray checks identified aisle placements; the
devices were moved there while preserving the original two blocker poses.
This is scene-fixture development, not model selection or a performance
comparison against the failed placement.

## Camera and actual DINOv3 input

The camera renders 256 × 192 pixels with a 60° horizontal field of view, four
samples per pixel and known simulated position/orientation. The receipt
includes the pinhole intrinsic matrix and both Mitsuba and OpenCV
camera-to-world transforms. Actual sensor rays at the center, right edge and
bottom edge check the convention: Mitsuba uses local X left/Y up/Z forward;
OpenCV uses X right/Y down/Z forward. This is simulated calibration, not a
calibrated physical SmartCar camera. Low-sample render noise remains visible.

The exact rendered PNGs are then passed to the
[frozen pretrained DINOv3 adapter](DINOV3_ACCESS.md), producing real
`[2,256,384]` dense patch tokens, masks and original-pixel patch boxes in
`rt_dinov3_features.npz`. DINO's crop/resize geometry and input-image hashes
are retained. The encoder receives images and provenance, **no channel
coefficients or hidden RF labels**. All returned visual features are finite.

These are independent static simulation poses. Capture time is explicitly
unknown; the adapter's observation-time field records probe ingestion only.
The pair is not a timed robot stream, a synchronized hardware capture or a
trained visual/RF dataset. The fixture makes it possible to begin such work;
two poses cannot validate an estimator, temporal hierarchy or transfer.

Reproduce using the existing environment and isolated DINO runtime:

```bash
PYTHONPATH=/workspace/upstream/dinov3-python-deps:src .venv/bin/python scripts/probe_visual_radio_rt.py --dino
```

The Dr.Jit disk-cache warning reflects an unwritable default cache directory;
the LLVM computation completed. No dependency lock or existing scientific
source was changed. Physical material calibration, more varied scenes,
measured RGB/pose/RF sequences, convergence checks, held-out prediction and
closed-loop control comparisons remain necessary before stronger claims.
