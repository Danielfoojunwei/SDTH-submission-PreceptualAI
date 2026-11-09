# Spectral-scene field experiment

This is an independent Phase 2a experiment following the sealed measured-power
pilot. Its source and outcomes remain unchanged. This study does not open the
original 24 robot-service confirmation geometries or activate a learned model.

The new data consists of 48 training, eight validation, four development and
24 conditionally opened RF field geometries. Every public geometry fingerprint
is disjoint from the previous 36 specifications and every other new geometry.
They are generated configurations from two CAD families, not 24 measured
warehouses or arbitrary terrestrial environments.

All scenes are traced by the unchanged Sionna RT generator only after new source,
protocol, specification and execution-order publication. Receiver offsets and
materials remain simulator audit truth. Learning and baselines use allowed
reported observations, public CAD/receiver geometry and training-only front-end
calibration. Training and validation contain no 6 GHz or held-out-device targets.

Both context views observe the same six even receiver poses, two APs and two
observed bands. View A uses direction 0/device 0; view B direction 1/device 1.
Main queries use the six odd poses. The held 6 GHz frequency assay uses the same six even physical links as the observed-band contexts, so it isolates frequency transfer from position interpolation. The model uses a shared pooled scene latent and
public query physics, with no forwarded RF tokens or mean-power decoder bypass.

Nine initial models compare no prior, SIGReg and VICReg across three independently
trained seeds. Validation RF error selects the prior. Three matched-regression
and three RF-value-erased geometry + reported-availability controls then use
that prior and the same training
budget. The operator-only control uses declared free-space/receiver operators;
it contains no learned hidden-material correction. RadioMap-operator and cheap
Bayesian-operator references receive the same permitted context observations.

Training uses 60 epochs, eight declared independently masked context repetitions
per epoch, and a batch containing all 48 training geometries. Both contexts
include both observed bands; token masking retains at least one token per band.
The fixed 4 dB likelihood spread is an optimization assumption, not calibrated
uncertainty. Hidden material/device calibration parameters
do not enter the student encoder.

The development gate requires selected-model mean RF RMSE and 95th-percentile
absolute error below RadioMap, each training seed's RMSE no worse than RadioMap,
finite outputs, and at least 1% RMSE deterioration when its scene latent is zeroed.
Rank is descriptive: four development scenes cannot establish rank above three
for centered invariant scene embeddings. Failure retains an explicit veto.

Four fixed field-test contrasts cover main RF RMSE against RadioMap and matched
regression, 95th-percentile absolute error against RadioMap, and held-band 6 GHz
RMSE against RadioMap. Main results use source view A to match the references'
information budget. Each metric is computed separately per trained member,
then averaged within a whole geometry; predictions are not ensembled.
Paired whole-geometry bootstrap intervals use a central 98.75% confidence level
for four two-sided tests, with quantiles 0.00625 and 0.99375.

Point-error endpoints use heard reported-RSSI query values. Censored observations
receive likelihood evaluation rather than invented point labels. Held-out-device
onboarding is descriptive. Source view B is a paired training view and is not
evaluated as an inference condition in this finite phase. No RF field
result establishes application deadlines, silence, production safety, or
a general necessity claim for the JEPA objective. Its exact matched-control contrast tests alignment benefit within this finite architecture and training recipe.


The encoder initializes its eight shared coefficients by an amortized
Bayesian-style inverse operator: a weighted public-geometry ridge solve on
heard residuals, plus a set-attention correction. The weights are learned
positive reliabilities; the correction is scaled by 0.25 and remains unbounded
so that unit-variance regularizers remain attainable. This is a regularized
field estimate, not identified material properties or a calibrated posterior.
The decoder maps only the shared scene code and public query geometry to RF.
SIGReg/VICReg and direct latent alignment use detached per-step gradient weights
that target 25% of the measurement-gradient RMS, bounded between 0.01 and 100.
Both branches learn; no projector, EMA teacher or stopped target is used.

The additional same-link frequency reference transports an actual heard 5.2 GHz
source report through the known free-space frequency operator. It assumes fixed
material/antenna response and calibrated reciprocity. Its comparison uses only
matched eligible heard source/query pairs and reports coverage; it never fills
unheard source values. This descriptive control is separate from the full
heard-cohort held-band primary endpoint against RadioMap.

The held-device curve uses 0, 1, 4 or 16 actual paired even-pose reported RSSI
observations to estimate the simulated fourth receiver's relative offset.
Inference still encodes known-device source view A and predicts the fourth
device at odd poses. This measures simulated RSSI offset/quantization inverse
onboarding, not held-device encoder transfer, real hardware AGC/CFO calibration
or neural fine-tuning. Nested support sets are correlated; the curve is
descriptive. No 6 GHz report enters calibration.

Results report heard, censored and unresolved/missing coverage. Point errors
therefore describe the heard cohort, not an entirely observed RF field.
Whole-layout bootstrap intervals are conditional on the three trained members;
training-seed spread is reported separately and is not a posterior over
retraining. Centered embedding ranks and singular values are descriptive.

A final passing test receipt and its stdout bind exactly the frozen source
bytes. Source, tests, protocol, public specs and execution order are registered
and remotely verified before any official RT tracing or training. Every stage
binds the raw receipt membership. The passive gate and final audit reconstruct
whitelisted public arrays and compare them to cached inputs/targets without
neural constructors, forwards, baseline fits or solver calls. Source-bound
qualification cannot substitute for recomputing this development gate. The old
452-source/864-evidence closure is checked before and after each phase.


The internal cell name `geometry_only` means RF-value-erased geometry plus
reported availability: it erases the first four RF/status token dimensions but
retains the heard-or-censored attention mask. Availability depends on RF, so
this is not a strictly CAD-only control and cannot establish RF-value necessity.


If an onboarding condition lacks its requested genuine context-pair count or
has no heard held-device query, that geometry/N is explicitly unavailable for
this descriptive assay. The result retains the actual available pair count,
heard-query count, reason and denominator. No RF values, scenes or support
budgets are substituted. The geometry remains in every mandatory primary RF
endpoint; a descriptive availability issue cannot waive the development gate.
