# Observed JEPA service failures

This diagnostic reads all 128 completed selected-study cases and changes no scientific evidence. The eight warehouse instances are the site units; repeated members are not additional sites. These outcomes are now observed repair-design data, not fresh holdouts.

## The longest tail is sustained association failure, not a long handover

At site 2114001, robot 44 has a right-censored 20.73 s control-silence interval from 39.27 s through the trial end at 60 s. CTX all three members, SUP0/1 and RadioMap remain on AP10 throughout. SUP2 moves AP10→AP2 at 59.85 s, too late to recover. The old frozen JEPA moves AP10→AP2 at 14.11 s and exits its outage at 18.11 s.

In the common full-second window [40,60], all new members and RadioMap lose 400/400 commands on this robot; the old frozen model loses 4/400. CTX records 21 planning callbacks during the long tail, all with candidate=current AP10, gain zero and eligibility/confidence false. Its last delivered RF report was at 34.96 s. There is no planned-eligible/admission-denied evidence for this particular tail. Reactive candidates and gate return masks were not archived.

The source starts cadence scans only after a successful control delivery; continuous silence therefore prevents fresh scan starts. A scan initiated before silence may finish near its start. This is a real feedback constraint, but does not prove that allowing a scan or a different roam would causally save the observed number of losses.

## Handover cost and model timing

All 64 slow cases reconstruct 2,456 handovers, matching their native roam totals. Complete unavailable intervals imply at least 183,012 of 367,153 observed misses (49.85%). This lower bound conditions on the chosen handovers; it is not avoidable regret.

The frozen ensemble consumes 43 forecasts within each 60 s run, with maximum ready age 17.98 s. The new members consume all 60 with maximum age 2.13 s; RadioMap reaches 1.13 s. These are ages under the sealed FIFO model. Issued-frame forecast errors do not measure the error of the actually consumed, aged forecast. The old model still roams more often at fast cost (135.75 versus CTX127.00 mean roams); fewer updates alone is not an explanation.

## Proxy objectives do not imply better service

Across 48 paired CTX/RadioMap case comparisons, 14 have lower issued-frame RMSE but more missed commands. Across 48 CTX/old-frozen comparisons, 21 have lower AP-choice proxy regret but more misses. These are counterexamples to a metric implication, not independent-site significance or proof of the cause.

The largest CTX/old-model case excess is site2114005, member1: +1,002 misses. Robot31 alone contributes +282, with no CTX handover and a 16.73 s gap; the old model makes one handover and has a 4 s maximum gap on that robot. Other robots and earlier policy histories also differ, so this is not a same-state intervention.

## Repair hypotheses requiring new matched tests

- Put planned admission results in the confidence/fallback path and separately preserve reactive recovery; log planned eligibility, admission, reactive proposal and execution distinctly.
- Test bounded emergency sensing when delivery-dependent observation collection becomes stale.
- Record actual pilots, cached neighbor RSSI/ages, consumed AP scores and per-action outage predictions to distinguish wrong beliefs from wrong admission/recovery.
- Compare every controller repair across CTX, supervised, RadioMap and the old model. A shared controller gain does not establish a JEPA-objective gain.

No new world, inference, fitting, calibration or intervention was run. No collapse, SOTA, production, causal-benefit or indefinite-stability claim follows.

Reproduce with `uv run --frozen python scripts/diagnose_jepa_service.py`. `findings.json` contains every case, paired robot excess, six long-gap groups and timing/proxy limitations.
