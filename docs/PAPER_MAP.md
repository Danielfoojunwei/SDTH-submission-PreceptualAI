# Paper map: Bowen Shen, Feng Li, Kwok-Yan Lam → WirelessJEV

FT = full text read (open copies on NTU DR-NTU, arXiv, or Research Square). Abs = abstract only.
None of these papers releases code. Every learned component they use (DQN/DDQN/PPO/GNN/LSTM/LLaMA-LoRA) is replaced here by one of:
- their formulation, used as an exact oracle;
- a frozen open model (`vllm-sr/d3-lite`, `amazon/chronos-bolt-small`);
- counting statistics.

The only parameters we fit are:
- readout heads (1024 numbers for the fleet task), distilled from exact oracles;
- a 9-parameter ridge residual, used only to reproduce the traffic paper's ablation.

| Paper | Status | Mechanism used | Code |
|---|---|---|---|
| Zhang, Lam, Shen, Wang, Li — *DSA for IoT with hierarchical federated DRL*, Ad Hoc Networks 149 (2023), doi:10.1016/j.adhoc.2023.103257 | FT | Interference state {2, 1, 0, −1}; piecewise reward SNR/SINR/0/−10; two-tier personalised sharing (80/20, edge k1 = 5, global k2 = 10) | `twin/observe.py`, `oracle.py` (`piecewise`), `swarm/share.py` |
| Li, Shen, Guo, Lam, Wei, Wang — *DSA for IoT based on federated DRL*, IEEE TVT 71(7) (2022), doi:10.1109/TVT.2022.3166535 | FT | Reward SINR − μ; cluster non-interfering, similar users | `oracle.py` (`qos_margin`), `swarm/share.py` |
| Li, Yang, Lam, Shen, Wei — *DSA for IoT with joint GNN and DQN*, Ad Hoc Networks 163 (2024), doi:10.1016/j.adhoc.2024.103596 | Abs | Graph-derived spectrum state, then action selection | Deterministic contention-domain graph (`Scene.domains`) feeds digested option features; the frozen decision model replaces the DQN |
| Li, Yang, Lam, Shen, Luo — *DSA for IoT with improved federated learning*, JNCA (2025), doi:10.1016/j.jnca.2025.104189 | Abs | GNN + DQN + federated aggregation | Same as above; aggregation via the swarm protocol |
| Wang, Shen, Li, Lam — *Swarm DSA for IoT*, IEEE WCNC (2025), doi:10.1109/WCNC61545.2025.10978419 | FT | Reward +1/−0.5/−1; leader = first ready; weighted merge | `oracle.py` (`pm1`), `swarm/consensus.py` |
| Li, Wang, Lam, Shen, Wang — *Secure DSA for IoT with swarm learning*, IEEE IoT-J (2025), doi:10.1109/JIOT.2025.3583710 | FT | Admission check; leader by performance score; FedAvg; **validate before release**; auditable ledger | `swarm/consensus.py` (HMAC hash-chained log in place of a blockchain) |
| Li, Shui, Lam, Shen, Wang — *Secure DSA based on machine unlearning*, ICAIIC (2025), doi:10.1109/ICAIIC64266.2025.10920793 | FT | R = B + αU − γW + βV; location-tagged forgetting; membership-inference AUC audit | `oracle.py` (mission utility), `distill.py` (MIA AUC, exact region unlearning) |
| Li, Sun, Lam, Sun, Shen, Peng — *Dynamic spectrum optimization for IoV with DL-based mobility prediction*, Wireless Pers. Commun. 137 (2024), doi:10.1007/s11277-024-11449-w | FT (preprint) | P ∝ visits/d², top-2, dwell; collaborative max-min colouring argmax b/(D+1) | `twin/mobility.py` (no LSTM), `teachers/assign.py` (`coloring_assignment`) |
| Shen, Goh, Lam, Li — *Efficient and secure federated DSA for LEO satellite IoT*, IEEE ICCT (2024), doi:10.1109/ICCT62411.2024.10946453 | FT | max Σ SINR·idle s.t. channel exists/idle, power bounds, ≤1 channel per user; bi-level k-means | `teachers/assign.py` (`hungarian_access`), `candidates.py` (`hard_filter`), `swarm/share.py` (`bilevel_kmeans`) |
| Shen, Lam, Li (, Wang) — *Privacy-aware spectrum pricing and power control for LEO satellite IoT*, arXiv 2407.00814 / IEEE TWC (2025) | FT | Revenue = price·#{budget ≥ price}, budget ∝ SINR; ±1 price step with +1/0/−1 | `teachers/admission.py`, admission task in `bench.py` |
| Shen, Lam, Yang, Liu, Li — *Privacy-preserving framework for spectrum pricing and power control*, MSN (2024), doi:10.1109/MSN63567.2024.00079 | FT | Reputation ± p̂·l·count; threats: power over cap, access overuse, lease overrun | `swarm/reputation.py`; quarantine in `closedloop.py` |
| Shen, Lam, Zheng, Lim, Yang, Li — *Efficient and certified LEO-enabled smart management system*, ICISS (2024) | Abs | Certified aggregation | Covered by the consensus log; nothing more is claimed |
| Chen, Lam, Li — *LLMs for network traffic prediction: a trend-aware hybrid framework*, IEEE IoT-J 13(4) (2025), doi:10.1109/JIOT.2025.3636129 | FT | Labels: trend / event / volatility; per-step prompt; semantic vs same-size calendar/PCA ablation; noise / missing-data / surge robustness | `traffic.py`: d3-lite typed questions replace LLaMA-LoRA and the text parser; Chronos-Bolt replaces the CNN-LSTM |
| Gao, Chen, Jia, Gong, Lam, Wang — *LLM-as-a-judge bias in communication systems*, arXiv 2510.12462 | Summary | Explicit rubrics make judges stricter; audit for bias | Rubrics in `policy.py`; bias audit in `bench.py` |

Jev-literature anchors (2026):
- arXiv 2610.18529: Jev-class models are weak at numeric reasoning, and option-wise normalisation helps. This is why the numeric layer digests every option, and why the deployed mode is option-wise scoring.
- arXiv 2610.11692: a frozen Jev works as a reference policy, judge or replay rater inside RL.
- arXiv 2609.22753: Jev replaces LLMs for low-latency edge admission.
