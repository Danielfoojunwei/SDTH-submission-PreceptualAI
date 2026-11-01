# RadioJEV v1: cross-site radio prior, pre-registered protocol

**What is learned.** 40 training warehouses (seeds 700–739) are drawn with randomised physics:
- rack loss ×0.5–1.6;
- path-loss slope 28–38;
- shadowing 2–7 dB.

Each is fully surveyed on a 4 m truth grid, as past deployments would be. Per band, the learned prior is:
- the mean and variance of each BSS's geometry coefficients (intercept, log-distance slope, loss per rack row);
- the RMS shadowing strength.

**The baseline.** RadioJEV v0's generic prior knows only a free-space-like law: intercept −40, slope −33, −3.5 dB per rack, with broad variances.

**Test.** 6 held-out sites (seeds 600–605, their own random physics) with survey budgets of 10, 30 and 60 s. Each site is mapped twice, once with each prior; everything else is identical (censored fit, held-out-robot discrepancy). Calibration is scored against the fade-free truth, as in L4.

**Hypotheses** (each passes if it holds in at least 5 of 6 sites):
- **F1:** at a 10 s survey, the learned prior gives lower map RMSE than the generic prior.
- **F2:** at a 10 s survey, the learned prior's 90% coverage is closer to 0.90.
- **F3 (reported):** the same comparisons at 30 and 60 s.
