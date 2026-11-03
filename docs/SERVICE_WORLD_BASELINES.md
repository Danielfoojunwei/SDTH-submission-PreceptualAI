# Service-world model: upstream baseline access

Audit date: **3 November 2026**. The [machine-readable receipt](../artifacts/service_world/baselines/access.json)
records public source URLs, immutable code pins, licenses, actual CPU smoke
outputs, dependency deviations and remaining access gaps. This complements the
[prior-art review](SERVICE_WORLD_PRIOR_ART.md); it supplies no new performance
result and establishes neither SOTA nor architectural novelty.

## What was actually obtained

| Method/source | Verified access | Reproduction boundary |
|---|---|---|
| [RAN-JEPA](https://arxiv.org/abs/2610.17584) | Primary paper inspected. Its cited [CETIN dataset](https://doi.org/10.1038/s41597-026-07723-0) resolves to a public [Zenodo record](https://doi.org/10.5281/zenodo.17815388), CC BY 4.0. Metadata, README and the actual 8.3 MB `Dataset_01.zip` were downloaded; its catalog checksum matches and LTE/NR CSV headers were inspected. No official RAN-JEPA code release was located in the inspected paper/metadata and targeted repository searches. | Public observational PM counters are not the complete downstream action-conditioned corpus, calibrated twin or real four-cell testbed. The paper's testbed control interval is five minutes; the dataset aggregates at fifteen minutes. Neither is our warehouse packet-deadline benchmark. No RAN-JEPA reproduction is claimed. |
| [Coupled Control and Wireless World Models](https://arxiv.org/abs/2609.04851) | The paper explicitly links the three ICONgroupCWC repositories pinned below. The original W-JEPA encoder, GRU predictor and power head passed CPU forward shape checks. | The main model repository has no LICENSE/COPYING file located at this pin. Its documented W-JEPA `src/train.py` entry point is absent. No ready CSI arrays or pretrained weights were found in the checkout. ROS Noetic, Gazebo Classic and synchronized Sionna RT were not run. The released core consumes complex CSI; substituting our event/RSSI input would be an architectural adaptation. |
| [Stochastic graph dynamics / GDM](https://arxiv.org/abs/2609.28670) | Primary paper inspected; its availability statement says code/data “will be uploaded anonymously.” No official source URL was located in the reviewed paper or exact-title repository searches. | Method reference only. A new graph RSSM implementation would be an implementation inspired by this paper, not a reproduction of its experiments. |
| [AutoLab](https://arxiv.org/abs/2609.14854) | Paper links a [lab site](https://www.6g-private.com/lab) and [authoring tutorial](https://www.youtube.com/watch?v=3PegjtwTM5o&t=6s). The lab URL returned HTTP 200 with an Angular application shell. | The paper describes downloadable spreadsheets/templates and public Python validation code. A standalone versioned package was not located in the inspected paper and returned landing HTML. No remote hardware experiment was submitted or run. This is an experimental-infrastructure reference, not a reproduced algorithm score. |
| [LatentWave](https://arxiv.org/abs/2606.06373) | Primary paper inspected. Targeted GitHub searches found survey/reading-list mentions that link back to the paper, rather than an official implementation. | Wireless representation-learning reference only. No official model, training configuration or checkpoint release was located in the checked materials. No downstream service-deadline comparison was run. |
| [TD-MPC2](https://github.com/nicklashansen/tdmpc2) | Official MIT source pinned below. Original `WorldModel` encoder, transition, reward and Q heads passed a small CPU forward shape check. | The full upstream agent hardcodes `cuda:0`, and this worker has no available CUDA device. The core smoke used a reduced configuration and the worker's Torch 2.14.1, whereas upstream specifies Torch 2.7.1. It is not full-agent execution, exact-environment reproduction, training or a warehouse result. |
| [DreamerV3](https://github.com/danijar/dreamerv3) | Official MIT source pinned below. In a separate CPU environment, its original RSSM observed and imagined three steps with finite outputs. | This is an untrained reduced-size RSSM smoke, not the full agent, a benchmark task or wireless control. JAX/JAXlib 0.4.33 CPU replace upstream's default CUDA JAX extra; no source patch was made. |

“Not located” reports the bounded checks in the receipt. It does not establish
that a release cannot exist elsewhere. Public repository visibility also does
not provide a missing redistribution license. No code from the unlicensed
coupled-model repository was copied into our application.

## Immutable source pins

All checkouts live outside the application at
`/workspace/upstream/service-world-reference/`. Source files were left unchanged;
the smokes import the original model modules. Generated Python bytecode is not
an upstream source modification.

| Repository | Commit | License located |
|---|---|---|
| [ICONgroupCWC/DreamerV2-Meets-Gazebo](https://github.com/ICONgroupCWC/DreamerV2-Meets-Gazebo/tree/f2f82f88a18569558adc7b9eee35ea467033a754) | `f2f82f88a18569558adc7b9eee35ea467033a754` | None found |
| [ICONgroupCWC/Gazebo-Sionna-RT-Integration](https://github.com/ICONgroupCWC/Gazebo-Sionna-RT-Integration/tree/2d37c650bde5af6e5f69e899aa4e4b01243d3160) | `2d37c650bde5af6e5f69e899aa4e4b01243d3160` | MIT |
| [ICONgroupCWC/jetbot_world](https://github.com/ICONgroupCWC/jetbot_world/tree/d30f06b40b4f8af8c74118550a20cee7f2be65d7) | `d30f06b40b4f8af8c74118550a20cee7f2be65d7` | MIT |
| [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2/tree/332f19e4aba4bffe2dc736f45ea7f7bdc585903a) | `332f19e4aba4bffe2dc736f45ea7f7bdc585903a` | MIT |
| [danijar/dreamerv3](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c) | `e01491fad6434b2245a3b8ca201dd7faedcc458c` | MIT |

## CPU verification and its limits

These are real upstream forwards on explicit shape fixtures. They do not use
warehouse test episodes, train a model, produce benchmark scores or consume
new held-out evaluation seeds. Randomly initialized weights and artificial
input tensors verify executability only.

| Original upstream component | Actual smoke | Result |
|---|---|---|
| Coupled W-JEPA | Complex CSI `[2,384]` → latent `[2,2]`; GRU input `[2,6,4]` → prediction `[2,6,2]`; power output `[2,1]` | Passed; final power output finite |
| TD-MPC2 `WorldModel` | State `[2,16]` → latent `[2,32]`; action `[2,2]` → next latent `[2,32]`, 11-bin reward `[2,11]`, average Q `[2,1]` | Passed; final Q output finite |
| DreamerV3 RSSM | Batch 2; three observed and three imagined steps; deterministic state 32, four categorical latents with four classes | Passed on `TFRT_CPU_0`; all returned arrays finite; 27 parameter arrays |

TD-MPC2 import dependencies were installed in an isolated target directory;
Dreamer dependencies were installed in an isolated Python 3.12 CPU virtual
environment. Root project source, dependency manifests and lockfiles were not
changed. Harness source, dependency freeze, file hashes, commands and initial
harness errors are included in the JSON receipt so the result can be inspected
without mistaking it for a trained-model comparison. Full end-to-end upstream
simulator/runtime setup remains outstanding.

## How these become credible comparators

1. **Same-task evaluation** must include the canonical RadioMap/controller,
   an explicit numerical queue/service estimator and an information-matched
   non-JEPA recurrent or stochastic model. Those provide meaningful warehouse
   comparisons only when supplied with the same causal observations, permitted
   actions, controller continuation, traffic and sealed evaluation protocol.
2. **Architectural adaptations** of TD-MPC2, DreamerV3 or coupled W-JEPA must
   disclose changed observations, objectives, action spaces and training budgets.
   Keep an unmodified upstream reference and a separate adapter. A CSI encoder
   and a delivered-RSSI/event encoder see different information; equalize that
   budget before attributing differences to JEPA or model architecture.
3. **Method references** with inaccessible code/data remain limitations. A
   paper-inspired reimplementation can test a stated mechanism, but should not
   be labeled the authors' reproduced result. Obtain the actual RAN-JEPA
   downstream resources and clarify the coupled-model license before claiming
   faithful reuse or comparison.

No same-task external baseline has been trained or reproduced by this access
audit. There is no common leaderboard across the RAN sleep-control,
CSI/robot-control, graph-dynamics and warehouse deadline tasks. The present
evidence cannot establish global SOTA against methods whose official code,
full task data or comparable evaluations are unavailable to this audit.
