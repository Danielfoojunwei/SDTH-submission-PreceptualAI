# RadioMap v1 (cross-site prior): cross-site radio prior, pre-registered protocol

**What is learned.** 40 training warehouses (seeds 700–739) are drawn with randomised physics:
- rack loss ×0.5–1.6;
- path-loss slope 28–38;
- shadowing 2–7 dB.

Each is fully surveyed on a 4 m truth grid, as past deployments would be. Per band, the learned prior is:
- the mean and variance of each BSS's geometry coefficients (intercept, log-distance slope, loss per rack row);
- the RMS shadowing strength.

**The baseline.** RadioMap v0's generic prior knows only a free-space-like law: intercept −40, slope −33, −3.5 dB per rack, with broad variances.

**Test.** 6 held-out sites (seeds 600–605, their own random physics) with survey budgets of 10, 30 and 60 s. Each site is mapped twice, once with each prior; everything else is identical (censored fit, held-out-robot discrepancy). Calibration is scored against the fade-free truth, as in L4.

**Hypotheses** (each passes if it holds in at least 5 of 6 sites):
- **F1:** at a 10 s survey, the learned prior gives lower map RMSE than the generic prior.
- **F2:** at a 10 s survey, the learned prior's 90% coverage is closer to 0.90.
- **F3 (reported):** the same comparisons at 30 and 60 s.

## Results (held-out sites 600–605)
Raw data: [`artifacts/native/foundation.json`](../artifacts/native/foundation.json).

| Survey | RMSE generic | RMSE learned | 90% coverage generic | 90% coverage learned | Sites where learned has lower RMSE | Sites where learned coverage is nearer 0.90 |
|---|---|---|---|---|---|---|
| 10 s | 3.92 dB | 3.94 dB | 0.67 | 0.63 | 2/6 | 3/6 |
| 30 s | 4.07 dB | 4.09 dB | 0.62 | 0.60 | 2/6 | 2/6 |
| 60 s | 3.90 dB | 3.93 dB | 0.51 | 0.51 | 1/6 | 3/6 |

**F1 fails. F2 fails.** The learned prior is correct physics; for example, it puts the 5 GHz intercept at −28.8 dB where the generic prior assumed −40. It still changes nothing measurable.

**Why.** Even a 10 s survey with 100 robots yields about 8,000 scan reports, which swamp any prior on three geometry coefficients. The map's remaining error (about 4 dB) is local shadowing and fading bias, which a prior over the *law* cannot predict.

**Implications for the foundation claim:**
- **Low-data use only.** A cross-site prior can only matter where data is scarce: a single car, a few seconds, or a new area.
- **Local structure is the real target.** Real gains need a model that predicts *local* structure from site information the robots see before driving: CAD, rack contents, AP mounting.
- **No foundation claim yet.** The radio model is not yet a foundation model in any useful sense.
