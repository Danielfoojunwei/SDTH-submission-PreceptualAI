# What an uncertainty contract can actually guarantee

Let \(G\) be the true advantage of one candidate AP over keeping the current
AP, in the declared planner score; \(\hat G\) its prediction; and \(U\ge0\)
an uncertainty score. Let \(A=\{\hat G\ge zU\}\), with \(z\ge c\ge0\).
Additional acceptance conditions, such as a minimum roam gain, can only restrict A.
Assume \(G\in[-B,B]\). Set \(m=\min(B,\max(0,\hat G-cU))\).

On the envelope event \(E=\{|G-\hat G|\le cU\}\), an accepted decision has
\(G\ge m\). Outside E, \(G\ge-B\ge m-2B\). Therefore, pointwise,

\[
G1_A\ge m1_A-2B1_{A\cap E^c}.
\]

Taking expectations gives

\[
\mathbb E[G1_A]\ge\mathbb E[m1_A]-2B\Pr(A\cap E^c).
\]

With a **conditional**, post-selection failure bound
\(\Pr(E^c\mid X,A)\le\delta(X)\), a sharper result is
\(\mathbb E[G1_A]\ge\mathbb E[1_A((1-\delta(X))m-\delta(X)B)]\).
An advantage follows only when the positive margin exceeds the failure cost.
Neither empirical marginal link coverage nor Spearman correlation establishes
that conditional assumption. Correlated AP errors also prevent replacing an
error-difference bound with a root-sum-square bound without further assumptions.

## Why rank correlation alone is insufficient

Take predicted gains (2,3), true gains (1,−4) and uncertainties (1,2). The absolute
errors are (1,7), so the error/uncertainty ordering is perfect. At z=2, multiplying
uncertainty by 0.5 accepts both decisions and gives mean gain −1.5. Unit scale
accepts only the beneficial decision and gives +0.5; scale 2 abstains and gives 0.
All three have the same perfect ordering. Thus a condition depending only on
rank correlation and the gain distribution cannot guarantee this fixed gate's
advantage. Scale affects its selected set even if ranking is unchanged.

`contracted.theory` checks the finite-sample inequality and the counterexample.
The campaign reports the observed gain and envelope-failure mass under every
uncertainty perturbation. The empirical bound uses evaluation truth and is an
audit, **not an ex-ante certificate** or a distribution-free coverage guarantee.

The theorem compares a single AP-score decision with keeping its attachment.
The actual native fallback can roam reactively, and packet delivery has load,
handover, fading and queue effects. Consequently, this theorem does not prove
that the existing native gate improves cumulative control misses or is safe on
a physical robot. Those are separate empirical comparisons.
