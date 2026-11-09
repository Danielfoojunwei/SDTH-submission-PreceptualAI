# Target observability and pilot design

The new `wirelessjev.sensing_design` package implements mathematical checks
and a sensing heuristic. It has not been evaluated as a radio controller and
does not change the sealed CSI study, any hardware or a shipped policy.

For supplied linear observations `y=A x` and linear targets `T x`,
`certify_targets(A,T)` computes a **full** right-nullspace basis `N` from SVD.
It reports rank, nullity, singular values, numerical rank thresholds,
observed-subspace condition number, target ambiguity rank and two ambiguity
sizes: `||T N||₂` and `||T N||F²`. The first is the largest target perturbation
per unit unobserved latent perturbation; the second is
`tr(T P_null(A) Tᴴ)`. Targets are identifiable from the supplied measurements
exactly when `T N=0`, at the declared numerical tolerances. An underdetermined
latent model can still identify a particular target.

An optional explicitly named positive-semidefinite `GaussianPrior` supplies
`C=L Lᴴ`. The certificate then additionally analyzes `A L` and `T L`. It keeps
the **ambient certificate** alongside the **prior-supported certificate**:
a low-rank prior can remove ambiguity only by assuming that excluded latent
directions cannot occur. A full-rank prior does not remove ambient nullspaces.
An empirical covariance's small eigenvalues are not proof that real channels
cannot vary in those directions. Covariance support and numerical eigenvalue
thresholds must therefore be disclosed.
Every positive prior eigenvalue is preserved by default. If a caller expressly
requests a relative prior-support cutoff, the certificate flags that truncation
and reports the omitted target variance. A tiny prior direction cannot silently
be treated as irrelevant when the target operator amplifies it.

The CSI mask's algebra is a concrete counterexample. A complete 64-tap DFT
dictionary sampled on 16 regularly spaced rows has rank 16 and nullity 48.
Even the union of the two views contains only the 32 even-frequency rows.
For any suitable tap index `m`, `e_m-e_(m+32)` is invisible to both views while
producing nonzero responses on every odd target row. A second view cannot
remove that ambiguity without an appropriate additional assumption. The
software fixture checks the actual 16-of-64 mask and this explicit witness.
It does not turn a truncated delay prior into physically identified paths.

`expected_posterior_covariance` and `greedy_targeted_pilots` require an explicit
proper-complex Gaussian covariance assumption and independent measurement
noise. The expected covariance is
`Cpost=C-C Aᴴ (A C Aᴴ+R)† A C`. These APIs accept operators and noise variances,
not observed values. They do not validate the consistency of an actual
singular/noiseless measurement, infer a clock model or certify calibrated
uncertainty. If real observations are later accepted, residuals outside the
innovation covariance's range must be rejected or modeled explicitly.

For an available candidate row `b` with independent noise variance `r`, the
greedy designer computes `v=C bᴴ`, `s=b C bᴴ+r` and
`gain=||T v||²/s`. It chooses an affordable unselected row by largest expected
gain divided by its positive declared cost, updates the covariance and
recomputes all marginal gains. A PSD-zero innovation supplies no information.
Noiseless duplicate rows become redundant; independent noisy repeats can
still reduce uncertainty. Existing measurements, finite budgets and maximum
choice counts are explicit. Costs have a caller-supplied common unit; calling
it “microseconds” does not measure real sounding airtime.
Affordability uses the unrounded cumulative `fsum` against one fixed tolerance
of eight machine epsilons times the budget. Reported spending is clamped only
for that bounded sum-roundoff excess, with the raw sum, adjustment and
tolerance exposed. A zero budget has zero tolerance and admits no positive
cost. Every strictly positive finite target gain is retained, including gains
small relative to unrelated target directions; overflowing variances, gains
or gain/cost ratios are refused.

The returned typed result includes choices, expected target variance drops,
spent budget, remaining target variance, prior identity and posterior
covariance. The objective is Bayesian target MSE in squared **linear** target
units, not dB, a worst-case guarantee or a service probability. Gain-per-cost
greedy selection is a heuristic, without a global budget-optimality or general
submodularity guarantee. Correlated receiver/clock noise needs a joint or
augmented nuisance model; scalar independent-noise updates do not cover it.
Improper complex distributions also require pseudo-covariance or an augmented
real representation; a single Hermitian covariance does not describe them.

Sixteen analytic software fixtures pass, including complex-conjugation checks,
target-specific nullspaces, explicit low-rank priors, the exact CSI alias,
noise-free redundancy, noisy repeats, target-amplified small positive prior
directions, immutable priors and a finite-budget
Fourier example. In that example an explicitly isotropic coefficient prior
and already-observed even rows make new odd rows informative; three such
measurements reduce expected target variance from 32 to 29. Those are analytic
linear-algebra quantities, not new measured RF results.

Physical scatterers, materials, unknown phase/time gauges and magnitude/power
observations are not generally linear in `x`. This certificate does not prove
their identifiability. A future active-sensing experiment must declare valid
observations, nuisance gauges, calibrated priors/noise, actual airtime costs,
independent evaluation and fallback behavior before reporting radio gains.
