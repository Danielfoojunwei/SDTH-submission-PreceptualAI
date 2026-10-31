# Prospective closed-loop selector audit

This selector was designed after the initial horizon holdout exposed lower-tail regressions. It uses six validation environments to accept or reject each already-trained model, seals its decision, then evaluates three new environments (143000–143002). No training or calibration uses those audit environments.

Both mission phases must improve mean validation utility without increasing critical violations or lowering worst-decile service in any validation environment. The whole schedule is accepted together; independently mixing phases would change the state at the mission transition.

**0/6 learned models accepted.** All rejected models use the numerical belief and its original calibration. The audit checks numerical fallback outcomes for exact equality; these repetitions are not independent learned successes.

| Audit controller | Runs | Mean service | Worst-decile service | Critical-violation robot-steps |
|---|---|---|---|---|
| budgeted_strongest | 3 | 0.2375 | 0.0684 | 8037.7 |
| numerical_planner | 3 | 0.3258 | 0.1174 | 6853.0 |
| guarded_residual | 18 | 0.3258 | 0.1174 | 6853.0 |

The study executes 42 validation rollouts and 24 prospective audit rollouts. 18 rejected-model audit outcomes exactly match their numerical reference. The fresh comparison has three independent simulated environments, not eighteen independent fleets.

The same operator-policy budget of twelve switches per decision applies. This does not establish superiority to unconstrained roaming, hardware reliability or a safe learned controller. Three audit environments are a small evaluation sample. All observations, telemetry and switch outcomes are simulated.

The proposal CLI defaults to numerical planning and uses this guard when a residual is requested. An explicit simulation-only research option exposes the residuals for continued experiments. Unsupported clauses are rejected; input provenance remains self-reported. No car command is sent.

The compressed raw audit is byte-for-byte recoverable (`gzip -dc gate_audit.json.gz`) with its original completion SHA-256 in `manifest.json`. Executed sources, sealed policy, validation differences and provenance are archived alongside it.
