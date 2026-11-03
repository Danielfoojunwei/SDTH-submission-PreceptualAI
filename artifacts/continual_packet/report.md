# Costed managed-probe packet study

Procedural native warehouse/ns-3-curve simulator; not RT100robot truth, JEPA benefit, hardware, novelty or SOTA

192 confirmation cases; 6 whole sites; 2 paired packet seeds; 20000 site bootstrap draws.

| Estimator | Primary (all six) | Information (both costs) | Engineering margins | Integrity | Service qualified |
|---|---|---|---|---|---|
| memory | False | False | False | True | False |
| bayesian | False | False | False | True | False |

Every primary comparison is reported, including negative and null outcomes. A negative difference favors the staggered estimator. Upper bounds use 1−0.05/12; information contrasts use 1−0.05/4. These are empirical margins, not safety certification.

| Estimator | Handover s | Reference | Loss difference | Descriptive 95% | Observed upper | Endpoint upper | Pass |
|---|---:|---|---:|---|---:|---:|---|
| memory | 0.03 | canonical | 0.000014 | [-0.000007, 0.000039] | 0.000047 | 0.000053 | False |
| memory | 0.03 | reciprocal_radiomap | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000018 | False |
| memory | 0.03 | staggered_ignored | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000018 | False |
| memory | 4.0 | canonical | 0.002239 | [-0.000116, 0.006172] | 0.007994 | 0.008173 | False |
| memory | 4.0 | reciprocal_radiomap | -0.000015 | [-0.000038, 0.000002] | 0.000005 | 0.000178 | False |
| memory | 4.0 | staggered_ignored | -0.000015 | [-0.000038, 0.000002] | 0.000005 | 0.000178 | False |
| bayesian | 0.03 | canonical | 0.000014 | [-0.000007, 0.000039] | 0.000047 | 0.000053 | False |
| bayesian | 0.03 | reciprocal_radiomap | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000018 | False |
| bayesian | 0.03 | staggered_ignored | 0.000000 | [0.000000, 0.000000] | 0.000000 | 0.000018 | False |
| bayesian | 4.0 | canonical | 0.002252 | [-0.000102, 0.006176] | 0.008014 | 0.008193 | False |
| bayesian | 4.0 | reciprocal_radiomap | -0.000003 | [-0.000008, 0.000000] | 0.000000 | 0.000183 | False |
| bayesian | 4.0 | staggered_ignored | -0.000003 | [-0.000008, 0.000000] | 0.000000 | 0.000183 | False |

Unresolved commands in positive-sign arms are misses; negative-sign arms are successes. Primary and information bounds must pass under both observed and conservative outcomes.

| Estimator | Handover s | Information difference-in-differences | Observed upper | Endpoint upper | Pass |
|---|---:|---:|---:|---:|---|
| memory | 0.03 | 0.000000 | 0.000000 | 0.000033 | False |
| memory | 4.0 | -0.000015 | 0.000003 | 0.000343 | False |
| bayesian | 0.03 | 0.000000 | 0.000000 | 0.000033 | False |
| bayesian | 4.0 | -0.000001 | 0.000000 | 0.000356 | False |

| Arm | Handover s | Command loss | Worst decile | Max silence s | Original video fraction | Bulk Mbps | Pending controls |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical | 0.03 | 0.010289 | 0.012953 | 0.64 | 0.9764 | 469.9852 | 1.50 |
| reciprocal_radiomap | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8835 | 1.50 |
| synchronized_ignored | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8835 | 1.50 |
| staggered_ignored | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8835 | 1.50 |
| synchronized_memory | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8835 | 1.50 |
| staggered_memory | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8834 | 1.50 |
| synchronized_bayesian | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8834 | 1.50 |
| staggered_bayesian | 0.03 | 0.010303 | 0.013016 | 0.64 | 0.9764 | 469.8834 | 1.50 |
| canonical | 4.0 | 0.136070 | 0.425583 | 52.31 | 0.8154 | 386.4707 | 19.25 |
| reciprocal_radiomap | 4.0 | 0.138324 | 0.427896 | 36.93 | 0.8159 | 384.5832 | 20.75 |
| synchronized_ignored | 4.0 | 0.138324 | 0.427896 | 36.93 | 0.8159 | 384.5830 | 20.75 |
| staggered_ignored | 4.0 | 0.138324 | 0.427896 | 36.93 | 0.8159 | 384.5830 | 20.75 |
| synchronized_memory | 4.0 | 0.138324 | 0.427896 | 36.93 | 0.8159 | 384.5830 | 20.75 |
| staggered_memory | 4.0 | 0.138309 | 0.427875 | 36.93 | 0.8160 | 384.5516 | 20.75 |
| synchronized_bayesian | 4.0 | 0.138322 | 0.427896 | 36.93 | 0.8159 | 384.5830 | 20.75 |
| staggered_bayesian | 4.0 | 0.138321 | 0.427896 | 36.93 | 0.8159 | 384.5830 | 20.75 |

All per-case metrics, cumulative checkpoint excess, backlog, overflow, paid resources, measurement ages, identification/fallback counts and timings are in report.json. Immutable case archives retain every per-robot second-by-second counter trace, raw paid report, scan delivery event and forecast completion ledger. Endpoint pending controls are right censored.

- Procedural simulator, shared CAD, not measured 100-robot warehouse evidence.
- AP-assisted carrier sense, aligned nested duty, mean RF before nonlinear PER and dedicated capture are assumptions.
- Bayesian posterior updates do not establish algebraic identification or calibrated service risk.
- Six sites are independent analysis units; packet seeds and robots are not extra sites.
- Pending controls are right censored. Observed maximum silence is not a safety guarantee.
- No JEPA, physical-world, novelty or SOTA claim follows from this experiment.
