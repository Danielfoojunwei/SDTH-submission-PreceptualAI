# Prior-art review for the two proposed papers

The supplied identifiers were retrieved and checked against document titles.
Eighteen open full texts were downloaded and parsed. This review inspects their
relevant methods, control mechanisms, evaluation and limitations; it is not a
claim that keyword matching proves novelty. The portable
[audit](../artifacts/contracted/prior_art/audit.json) records URLs, titles, PDF
hashes and retrieval outcomes. Full PDFs remain in the workspace literature
archive rather than being redistributed in this repository.

## Closest overlap changes the proposed claim

**Uncertainty-triggered learned control with a conventional fallback already
exists in wireless.** Lee, Kim and Min's *Self-Improving Interference Management
Based on Deep Learning With Uncertainty Quantification*, IEEE TVT (2025),
[DOI 10.1109/TVT.2025.3588226](https://doi.org/10.1109/TVT.2025.3588226), has an
open [2024 preprint](https://arxiv.org/abs/2401.13206). Sections III-B/C train
independent mean/variance networks, quantify uncertainty, assess credibility
against a performance criterion and invoke conventional optimisation when the
learned solution is insufficiently trustworthy. The improved solution also
feeds subsequent training. This is more specific overlap than the uploaded
plan acknowledged. The reviewed full text is the open preprint, not a claim
that the paywalled final typeset version was inspected.

The *Telecom World Models* paper also explicitly combines intent, predictive
world models, uncertainty propagation, constraint enforcement and rollback or
fallback. Consequently, neither a new name for the uncertainty interface nor
pairing JEPA and JEV establishes an architectural novelty claim.

## Comparison grounded in the full texts

| Work | Relevant mechanism and evidence | Consequence for this study |
|---|---|---|
| [RAN-JEPA, 2610.17584](https://arxiv.org/abs/2610.17584) | Observation-only PM pretraining, deployment adaptation and action-conditioned latent prediction; four-cell real-equipment 5G sleep/wake control. Sections 2–4 distinguish chosen actions from unknown exogenous futures. | Action-conditioned JEPA plus a separate controller is established. Our simulated warehouse results do not reproduce or surpass its real testbed evidence. |
| [Telecom World Models, 2604.06882](https://arxiv.org/abs/2604.06882) | Field, control/dynamics and foundation-model layers; intent compiles objectives and guardrails; uncertainty informs planning and rollback. Includes a network-slicing proof of concept. | Broad coupling, uncertainty and intent-driven planning overlap directly. A narrower censored-device comparison would need its own positive evidence. |
| [JEPA for AI-Native 6G, 2607.09798](https://arxiv.org/abs/2607.09798) | Shared predictive representations with task-specific uses across RAN/edge/core, and an illustrative beam-management study. | A reusable wireless latent backbone is already proposed; this small model is a warehouse benchmark, not a general wireless foundation model. |
| [LatentWave, 2606.06373](https://arxiv.org/abs/2606.06373) | Wireless JEPA representation pretraining and transfer rather than simply reconstructing every raw observation. | The JEPA objective itself is not new. Our implementation is not an official LatentWave reproduction. |
| [Field-layer JEPA, 2608.10222](https://arxiv.org/abs/2608.10222) | Predictive field/channel representation bridging channel estimation and prediction. | Channel/field world modelling is established. Device scan censoring and native packet control are the relevant benchmark differences. |
| [JEPA-kMMVAE, 2610.13315](https://arxiv.org/abs/2610.13315) | Predictive latent modelling for time-varying CSI. | A per-link future latent alone is insufficient novelty. We predict coarse scanned RSSI/SNR, not CSI. |
| [System One Models, 2610.04345](https://arxiv.org/abs/2610.04345) | Typed finite decisions, intent-conditioned tasks, calibrated confidence and explicit escalation/fallback; Section III discusses uncertainty that changes execution. Reference 20 points to the TVT work above. | The decision-native interface and confidence-triggered fallback are explicitly discussed in wireless literature. |
| [Selective Classification, 1705.08500](https://arxiv.org/abs/1705.08500) | Selective prediction, risk/coverage trade-offs and confidence rejection, including MC-dropout comparisons. | Threshold abstention and risk/coverage analysis are established concepts. |
| [Risk-Controlling Prediction Sets, 2101.02703](https://arxiv.org/abs/2101.02703) | Statistical risk control through calibrated prediction sets under stated assumptions. | Empirical marginal SNR coverage is not a conditional control-risk certificate. Our current gate is not an implementation of this guarantee. |
| [Smart Predict, then Optimize, 1710.08005](https://arxiv.org/abs/1710.08005) | Prediction trained for downstream optimisation loss; decision quality can diverge from forecast error. | Regret-aware learning is established. The current loss is an application to finite AP scoring, not a new SPO+ method or a packet-loss gradient. |
| [MuZero, 1911.08265](https://arxiv.org/abs/1911.08265) | Learned dynamics trained around reward, value and policy sufficient for planning, rather than observation reconstruction alone. | Decision-aware latent modelling predates this application. |
| [Value Equivalence, 2011.03506](https://arxiv.org/abs/2011.03506) | Models can be adequate for decisions without matching every aspect of environment dynamics. | Better decision quality with worse reconstruction would be consistent with existing theory, not a new principle. |
| [MOPO, 2005.13239](https://arxiv.org/abs/2005.13239) | Model uncertainty penalises model-based offline policy optimisation. | Uncertainty-aware reliance on a learned world model has substantial prior art outside wireless too. |
| [MOReL, 2005.05951](https://arxiv.org/abs/2005.05951) | A pessimistic learned MDP discourages unsupported model rollouts. | Restricting decisions to trusted parts of a learned model is established. |
| [KnowNo, 2307.01928](https://arxiv.org/abs/2307.01928) | Calibrated uncertainty aligns language-model planning with when a robot should ask for help. | Uncertainty-triggered escalation in agentic/robot planning is established. |
| [Homomorphic Wireless World Models, 2603.20048](https://arxiv.org/abs/2603.20048) | Structured latent dynamics for wireless CSI, discovered through the decision-focused wireless-model search. | Structured wireless latent dynamics are another established direction; this work cannot claim the broad territory. |
| [Coupled Control and Wireless World Models, 2609.04851](https://arxiv.org/abs/2609.04851) | Coupled JEPA robot/wireless models use visual and RF representations for predictive communication scheduling and discrepancy-driven adaptation, evaluated in synchronised Gazebo–ROS–Sionna simulation. | Coupling robot and wireless world models for predictive control is already explicit prior art, including a richer robot/simulator stack than this benchmark. |

## What the present implementation actually tests

The new combination is **capped and censored robot scan histories, known future
routes, a predictive EMA latent, per-link uncertainty, an unchanged native
planner and a finite AP-regret objective**, evaluated in the existing packet
simulator. This is a concrete experimental scope. Its novelty and benefit
cannot be inferred from combining component names.

The model is trained offline on pooled synthetic device observations. No
federated or on-device optimisation, real robot survey, physical RAN deployment,
raw IQ/CSI representation or general wireless foundation-model claim follows.
Bayesian RadioMap pools a 60-second fleet survey, whereas the JEPA's local input
is six recent scans from one robot; the Bayesian comparison is a strong-system
reference with a different new-site information budget. The JEPA/regret and
gate/no-gate contrasts have matched inputs.

## Search coverage and remaining limits

The companion [index-search record](../artifacts/contracted/prior_art/index_search.json)
contains the IEEE Xplore search-page request and OpenAlex queries. Crossref
search responses and rate-limit failures remain in the audit; failures are not
represented as completed searches. Open copies are used legitimately where
available. No IEEE authentication is configured, and an accessible search page
alone would not establish an exhaustive full-text search.

This review is sufficient to reject a broad claim that uncertainty contracts,
learned fallback, intent-aware planning or decision-focused objectives are new.
It does not establish that no paper has studied the exact warehouse combination.
A publication claim requires a narrower contribution and convincing results,
rather than presenting missing search coverage as evidence of novelty.
