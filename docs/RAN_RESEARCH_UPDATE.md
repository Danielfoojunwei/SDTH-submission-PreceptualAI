# RAN research update: mechanisms to adopt and evidence required

**Review date: 3 November 2026.** This update recommends controlled experiments
and implementation patterns. It establishes neither global SOTA nor a production
SLA. Current transaction/runtime fault handling is being improved separately;
those fixes and the learned components are **not production-qualified**.

The [primary review audit](../artifacts/ran_research/primary_review/audit.json)
records 18 targeted primary-source reviews, fresh retrievals, release pins,
licenses, bounded searches and limits. No upstream model code was executed,
weights downloaded, new model trained or live radio network probed by this
literature audit. Existing sealed evidence was not edited.

Separate follow-through is documented in [CONTINUAL_RADIO_STATUS.md](CONTINUAL_RADIO_STATUS.md):
the response experiment completed with a negative JEPA result, and a licensed
[LWM-Temporal checkpoint](LWM_TEMPORAL_ACCESS.md) was subsequently downloaded
and executed on CPU. The release inventory below describes what the literature
audit itself inspected; it is not the checkpoint execution receipt.

## What our results imply for adoption

The [projection experiment](REPRESENTATION_REPAIR_RESULTS.md) improved average
normalized auxiliary future MSE from **1.623 to 0.580**, while missed-command
MAE changed from **23.322 to 23.710** and two of three seeds worsened. These
are exposed validation scenes, not independent deployment evidence. This
rejects auxiliary latent accuracy as a sufficient promotion criterion; it does
not show that all predictive representation learning fails.

The [visual/RF work](VISUAL_RADIO_STATUS.md) supplies real pretrained DINOv3
features and two synchronized Sionna warehouse poses. It does not yet supply
a trained fused estimator, continual-learning result or visual-policy service
gain. The next useful target is a **receiver-specific response to a controlled
action that can change a QoS decision**, with an explicit physical baseline.
The detailed proposed experiment is in [CONTINUAL_RADIO_DESIGN.md](CONTINUAL_RADIO_DESIGN.md).

## Fresh core-paper checks

All five PDFs below were freshly retrieved and match the hashes of the earlier
reviewed copies. Fresh bounded repository searches and inspected paper links
did not locate official code/weights for RAN-JEPA, LatentWave, JEPA-kMMVAE or
WorldGT. “Not located” describes the checked materials, not proof of absence.

| Primary source | Mechanism worth using | Boundary |
|---|---|---|
| [RAN-JEPA](https://arxiv.org/abs/2610.17584), §3–4 | Observation-only pretraining and domain adaptation, then a frozen encoder and action-conditioned learning from controlled responses. Mechanistic references plus learned corrections; matched same-history interventions. | Testbed KPIs are recorded every five minutes, with fifteen-minute prediction intervals. Our control deadlines are 50 ms. Transfer the method separation and experimental discipline, not its clock, scores or implied deployment safety. |
| [Telecom World Models](https://arxiv.org/abs/2604.06882), §V/VII | Queues, protocol state, handover interruption, counterfactual planning, uncertainty, shadow deployment and rollback. | Framework and limited proof of concept; not a ready warehouse controller. |
| [JEPA-kMMVAE](https://arxiv.org/abs/2610.13315), §3/4.2 | Frozen denoising temporal representation with separate predictor and residual future-CSI decoder; noisy history alone at inference. | Its own GRU/Transformer comparators win at lower observation SNR, while the residual variant wins at higher SNR. This supports regime-specific comparisons rather than assuming JEPA dominates. |
| [LatentWave](https://arxiv.org/abs/2606.06373), §III | Antenna-shared time/frequency patching, variable antenna subsets and structured masking. | Compatible RF/CSI inputs are needed; representation transfer is not demonstrated packet-service control. |
| [WorldGT](https://arxiv.org/abs/2610.29587), §III–IV | ACK-synchronized receiver memory, spatial age/validity, matched packet counterfactuals and defined outage continuation. | Semantic video perception, with different metrics and budget exclusions. Its scores cannot be ranked directly against warehouse roaming. |

RAN-JEPA's cited [CETIN catalog](https://zenodo.org/records/17815388) was also
freshly checked: CC BY 4.0, with `Dataset_03.zip` listed at 709,421,773 bytes.
No new dataset archive was downloaded. Those passive operator counters do not
provide the complete downstream controlled corpus, calibrated twin or testbed.

## Newly verified resources

| Resource | Verified immutable release | Practical limit |
|---|---|---|
| [LWM-Temporal](https://huggingface.co/wi-lab/lwm-temporal), [paper](https://arxiv.org/abs/2603.10024) | `df32f4beb6c7fe11d83031edbe5d6312798b81c3`; actual MIT license; model/data-generation source inventory and a listed 1,017,487-byte checkpoint. | Angle-delay-time sparse attention is relevant when compatible complex CSI exists. Reconstruction pretraining is bidirectional; forecasting must use causal attention. Checkpoint bytes were not downloaded or loaded here. |
| [LWM-CDE](https://huggingface.co/wi-lab/lwm-cde), [paper](https://arxiv.org/abs/2605.24077) | `c5a9451217fd1279db52c56c3df8bf0a90df8040`; Apache-2.0 license; source-selection/dataset-distance implementation inventory. | Frozen embeddings and label-aware sliced Wasserstein distance rank source datasets; refinement needs a measured transfer matrix. Refined paper checkpoints are explicitly absent. The paper reports failure of its refined metric across sub-6/28-GHz bands. Dataset distance is not a service-risk certificate. |
| [LWM v1.1](https://huggingface.co/wi-lab/lwm-v1.1), linked by the [LWM paper](https://arxiv.org/abs/2411.08872) | `1b68512dbe97de56841301206eacb780fcd76011`; model/preprocessing/inference source inventory and listed 9,960,738-byte checkpoint with catalog hash. | Standalone explicit license not located in the inspected release. Catalog hashes are not independently verified weight contents. The CDE package describes licensing of its bundled source separately; downloaded weight terms still need verification. |
| [WiFo](https://github.com/liuboxun/WiFo), [paper](https://arxiv.org/abs/2412.08908) | `313cc09f5576bf4c49c931cd99abb0ac5082d7e1`; official README/tree, advertised pretrained variants and datasets. | No LICENSE in the inspected complete tree. Linked weights/data were not downloaded. The README notes a large-model attention-head difference from the paper; exact reproduction must account for it. |
| [Cheap synthetic indoor-map pretraining](https://arxiv.org/abs/2610.19015), [generator](https://github.com/xoren22/cheap-synth-pretrain-neurips26) | `8b4a6fe58f4f46f89777f31e3527bb92aa46f80c`; official generator README/tree. | No code license/weights located in that tree. No roughly 6-TB corpus download. Its 2D approximation omits diffraction, scattering and 3D propagation; the paper explicitly finds that scale cannot replace missing material/measurement information. |

LWM-Temporal's per-sample RMS normalization and amplitude augmentation are
reasonable channel-shape learning choices. For service prediction, retain
absolute measured signal/noise/power and their calibration in a separate path:
invariance to amplitude can otherwise remove the distinction that changes a
deadline decision. These are compatible-input comparators, not replacements
for unavailable hardware measurements.

One plausible search result, `Snarci/Radio-DINO`, actually concerns medical
radiomics and was excluded after reading its primary README. A third-party
repository saying it follows LWM was likewise not labeled an official release.

## Ranked adoption plan

1. **Controlled response learning after passive representation learning.** Use
   RAN-JEPA's staged separation, but learn the response of specified receivers
   to recorded own-transmitter activity/power/AP interventions. Compare against
   numerical system identification and the identical supervised architecture.
   This directly tests whether the pretext becomes useful for service decisions.
2. **Delivered, event-aligned receiver state.** Use WorldGT's information
   discipline: age, identity, validity and actual delivery determine available
   state. Cross-robot teachers must align the same intervention and capture
   clocks. Delayed reports may become delayed labels, never retroactive online
   observations. Test missing events and commanded versus actual transmission.
3. **Shared physical prior with local fast adaptation.** Begin with explicit
   receiver/transmitter coupling, activity, power, spectral overlap and local
   device calibration. Test whether a learned shared prior improves small-sample
   response estimation beyond RLS/NNLS. Use FedProx only as a heterogeneous-client
   optimization comparator; it does not itself solve continual drift or forgetting.
4. **Selective physical/CSI pretraining and source selection.** Test a small
   geometry/RF or compatible CSI comparator before a large pretraining campaign.
   LWM-CDE suggests selecting compatible source data rather than indiscriminately
   pooling every robot/site. Keep geometry+RF and supervised/non-JEPA controls,
   band identity, data rights and charged compute explicit.
5. **Performance-based fallback and staged promotion.** The [wireless UQ
   fallback paper](https://arxiv.org/abs/2401.13206), §III, already qualifies a
   learned solution against task performance, invokes conventional optimization
   when untrusted and learns from refinements. Adapt this principle with our
   actual service loss and canonical recovery, rather than treating ensemble
   spread or normalized probabilities as a safety certificate.

## Production qualification and research cadence

[3GPP TS 22.104 / ETSI TS 122 104 V18.4.0](https://www.etsi.org/deliver/etsi_ts/122100_122199/122104/18.04.00_60/ts_122104v180400p.pdf),
§5.1 and Annex C.3, separates deadlines, application survival time, safe-state
behavior and recovery. Its illustrative reliability-to-availability conversion
explicitly assumes uncorrelated errors, often unrealistic. Requirement targets
are not measured availability of this system.

[O-RAN Architecture / ETSI TS 103 982 V8.0.0](https://www.etsi.org/deliver/etsi_ts/103900_103999/103982/08.00.00_60/ts_103982v080000p.pdf),
§6.2–6.3.2, separates Non-RT loops of at least one second, Near-RT loops of
10 ms–1 s and faster execution inside E2 nodes. Actuation depends on exposed
service models. Continuing baseline E2 service on RIC failure is weaker than
proving application-safe recovery. Both PDFs now have successful hash-verified
curl retrievals; initial urllib 403 responses remain in the receipts.

[Conformal Risk Control](https://arxiv.org/abs/2208.02814), §2, supplies a useful
calibration comparator under exchangeable, bounded, monotone loss assumptions.
Official MIT code is pinned at `3eff946390a8f188b1e5ab700fce21a66a215e7c`.
Its expected-risk guarantee is not a per-state or high-probability service SLA
for correlated, adaptively selected packets. More abstention need not make a
closed-loop loss monotone when fallback changes future state.

Promotion requires independent site/session evaluation, multiple trained seeds,
actual deadline/outage/recovery metrics, charged capture/transport/compute
latency, drift and missing-input monitoring, validated transaction behavior and
hardware evidence. The current runtime fixes do not waive these gates.

The bounded research watcher supplies metadata leads and provenance; a lead
is not a reviewed method or an approval to execute remote code. Each adoption
decision should retain the exact primary/version and release pin, license/data
status, mechanism, same-task comparison and failed checks. A durable schedule
must be explicitly configured; this report does not imply a background service
or promise that future publications have already been reviewed.
