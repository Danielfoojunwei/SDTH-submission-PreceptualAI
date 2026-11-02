# Independent warehouse confirmation

The original fleet study keeps its prespecified development-selected
standalone model and all of its results. Its eight primary test warehouses
are discovery data for this separate follow-up.

On those eight warehouses, the supervised hybrid had 163 missed commands,
the hybrid JEPA had 165, and the selected standalone had 192. The candidate
rule is the lowest discovery miss count among the three new families. It
therefore fixes the supervised hybrid before accessing eight new seeds.

The reference is the learned-prior RadioMap already chosen on development
data. The JEPA hybrid and the previous JEPA are secondary comparisons. All
four use the already selected reactive-rescue policy. No weights, gates,
feature definitions or controller knobs will be changed during confirmation.

Eight fresh warehouses each run 100 robots for 120 seconds. The primary
claim requires the 95% paired-site bootstrap interval for supervised hybrid
minus learned-prior RadioMap misses per million to lie entirely below zero.
Forecast accuracy is reported separately. Failure to pass that criterion
will be reported; the original study will not be relabelled as confirmation.

[Frozen protocol](../artifacts/fleet_confirmation/protocol.json).
