# Adaptive field controls

Both numerical controls fit their global priors only from admitted TRAIN
reported intervals and censor events at 2.4/5.2 GHz. They adapt using the same
view-A attempted report schedule as the learned models. Private material,
condition identifiers, channel gains and held-band reports are excluded.

`radiomap_operator` calls the unchanged native RadioMap Bayesian/Tobit core
through a public-geometry matrix adapter, per AP and observed band.
`bayesian_operator` uses a pooled eight-coefficient physical field posterior.
The numerical priors use interval/censor second-moment EM. Heard bins are
intersected with the actual detector event. No invented point at a detection
floor is fitted.

The receiver generator has deterministic offset, quantization and detection,
with no added Gaussian sensor noise. The fitted Gaussian spread represents
unresolved field discrepancy. Native center-point conditioning additionally
accounts for reported-bin variance; this approximation and the native
`CENSOR_WEIGHT` are recorded in the receipts. Quantization is integrated once
in interval-based fits.

The frozen three-member `joint_none` model is a supplementary transfer
reference. It loads the literal previous checksummed weights, performs no
new fit and explicitly abstains when its source context is unsupported.
Abstentions never remove geometry from the primary comparisons.

The new `rf_erased` control keeps heard/censored status and availability but
erases RF amplitudes. `pure_geometry` uses the public attempt schedule without
RF status or values. The exact matched regression model removes only the
joint-embedding alignment objective. Numerical, learned and frozen-reference
results must be reported together, including nulls and failures.
