# Supervised toy-signal learning-contract study plan

Status: frozen before execution

Study ID: `supervised-toy-signal-v1`

Execution class: unpaid, deterministic local CPU learning mechanics only

## Question

Can one fixed, small supervised surrogate policy recover a deliberately
learnable context-to-binary-action signal on generator regimes that are held
out from fitting, beat frozen constant and random policies under a
precommitted metric, and fall back to baseline under an identically evaluated
label-shuffle control?

This is a harness and learning-contract question. It does not ask whether the
synthetic signal exists in UIFO, whether a learned policy improves the
submitted optimizer, or whether reinforcement learning should be trained.

## Frozen task, generator, and typed contracts

- The task family is synthetic and has no graph, optical layout, official
  topology, topology identifier, or topology-derived input. Its structure is
  fixed at observation dimension four, two binary actions, and horizon one.
- Each trajectory is exactly one pre-action observation followed by one
  action, evaluator-only target and reward, and `done = true`.
- The policy-visible observation is a finite `float64[4]` array with ordered
  fields `signal`, `nuisance_pair`, `nuisance_regime`, and `nuisance_cycle`.
  No identifier, split, generator regime, row position, target, reward, or
  evaluator mask is policy-visible.
- The action and evaluator-only target are scalar `int8` values in `{0, 1}`.
  The evaluator-only reward is scalar `float64`, equal to one when action and
  target agree and zero otherwise. Mean reward therefore equals accuracy.
- Every generator regime contains exactly 32 four-row blocks, hence 128
  one-step trajectories. Within a block, observations occur in the fixed
  signal-sign pattern `[-1, +1, -1, +1]`; the three nuisance coordinates are
  identical across all four rows. The true targets are `[0, 1, 0, 1]`.
- For block `b` and regime tuple `(code, signal_scale, nuisance_shift,
  nuisance_scale)`, compute all quantities directly in `float64` as
  `magnitude = signal_scale * (1 + 0.05 * (b mod 7))`,
  `nuisance_pair = nuisance_shift + nuisance_scale * (((b mod 9) - 4) / 4)`,
  `nuisance_regime = nuisance_scale * ((((5*b + code) mod 13) - 6) / 6)`, and
  `nuisance_cycle = 0.5*nuisance_shift + nuisance_scale *
  ((((7*b + code) mod 17) - 8) / 8)`. The four observations are the Cartesian
  concatenations of signed signal `[-magnitude, +magnitude, -magnitude,
  +magnitude]` with the one shared nuisance triple. No formula uses the target.
- The complete regime table is frozen as follows. Fitting uses four regimes:
  `(101, 0.80, -0.75, 0.90)`, `(103, 0.95, -0.25, 1.10)`,
  `(107, 1.05, 0.25, 0.80)`, and `(109, 1.20, 0.75, 1.20)`. Validation uses
  `(211, 0.70, -1.25, 0.70)` and `(223, 1.30, 1.25, 1.30)`. The terminal test
  uses `(307, 0.60, -1.75, 0.60)` and `(331, 1.40, 1.75, 1.40)`. Tuple fields
  are regime code, signal scale, nuisance shift, and nuisance scale.
- Regime codes and canonical sample keys `(split, code, block, row)` are
  evaluator metadata only. Canonical order is the declared regime-table
  order, ascending block `0..31`, then ascending row `0..3`. All
  regime codes and sample keys are disjoint across train, validation, and
  test. Exact observation-row hashes must also have zero cross-split overlap.
  Repeated observations inside one four-row block are intentional and cannot
  cross a split.
- Generation is closed-form and has no RNG. The only stochastic-looking
  component is the frozen random-policy baseline, which uses NumPy `PCG64`
  seed `2026082707` on canonical held-out row order. No global RNG, OS entropy,
  time-derived seed, or data-dependent seed is permitted.

The complete counts are 512 fitting trajectories, 256 validation trajectories,
and 256 test trajectories. No normalization, augmentation, calibration,
feature selection, early stopping, or hyperparameter selection is performed.

## Frozen learner, baselines, metric, and controls

- Encode fitting targets as `{-1.0, +1.0}` and prepend an intercept to the
  four policy-visible features.
- Fit one deterministic linear ridge score by solving
  `(X.T @ X + penalty) * weights = X.T @ y` in `float64`, with ridge
  coefficient `1e-6` on feature weights and zero penalty on the intercept.
  There is no learned initialization, optimizer loop, minibatch order, or
  model seed.
- Convert scores to action one only when the score is strictly positive;
  zero is the declared tie and maps to action zero. The model is fitted once
  on train only. Validation and test examples, labels, and rewards cannot
  affect fitting or any threshold.
- The constant policy always chooses action zero. It is not chosen from any
  held-out label. The random policy uses the one frozen independent PCG64
  stream and is evaluated on the exact same canonical validation and test
  rows as the learner.
- The primary metric is equal-weight macro accuracy across complete generator
  regimes. Per-regime accuracy and the minimum held-out-regime accuracy are
  also frozen guards. Accuracy is computed by canonical sample key, and a
  separately reordered evaluation must reproduce it exactly.
- The positive learner gate requires all of the following: train, validation,
  and test macro accuracy at least `0.99`; minimum validation-or-test regime
  accuracy at least `0.98`; validation gain over the constant policy at least
  `0.30`; test gain over the constant policy at least `0.30`; and test gain
  over the random policy at least `0.25`.
- The label-shuffle control keeps every observation, split, model equation,
  ridge coefficient, evaluator row, and true validation/test label fixed. It
  changes only fitting-label association using the frozen within-block
  permutation `[0, 2, 1, 3]`. Thus each block's training labels become
  `[0, 0, 1, 1]` while class counts remain unchanged. It must be nonidentity,
  preserve train-label marginals exactly, leave the train-feature hash
  unchanged, reject the complete positive gate, achieve test macro accuracy
  at most `0.55`, and trail the true-label learner by at least `0.40`.
- The attribution control fits the identical ridge learner after replacing
  the declared signal coordinate with zero in train, validation, and test.
  Its test macro accuracy must be at most `0.55`. Applying the true learner to
  signal-zeroed test observations must also achieve at most `0.55`.

## Complete frozen cases

1. `typed_task_contract`
   - Require the exact observation/action/target/reward dtypes, shapes, field
     order, value bounds, horizon-one transition, `done` value, and policy
     input field set.
2. `generator_partition`
   - Require the exact regime table and split counts, 128 trajectories per
     regime, balanced targets within every regime, deterministic replay,
     finite values, unique cross-split regime/sample identities, zero
     cross-split observation overlap, and canonical dataset commitments.
3. `leakage_guards`
   - Require fitting to receive only train observations and targets. Reject
     sentinels that expose `target`, `reward`, `regime_code`, `split`, or
     `sample_key` as policy inputs; reject train/test regime, key, and
     observation overlap sentinels made by copying the first corresponding
     train item into test; and require canonical-key scoring to stay identical
     when validation and test are each evaluated in exact reverse canonical
     order.
4. `baseline_replay`
   - Require the constant and random policies to use the declared actions,
     seed, and exact held-out row sets; require exact in-process replay and
     finite macro/per-regime metrics.
5. `supervised_recovery`
   - Require every positive learner threshold, record only aggregate metrics
     and model/action commitments, and forbid raw observations, labels,
     predictions, or weights in the result.
6. `label_shuffle_control`
   - Require the exact nonidentity permutation, unchanged train-feature and
     label-marginal commitments, the failed positive gate, test macro accuracy
     at most `0.55`, and a true-minus-shuffled test gap of at least `0.40`.
7. `signal_attribution_control`
   - Require both the refitted nuisance-only learner and the true learner on
     signal-zeroed test observations to have macro accuracy at most `0.55`.
8. `process_isolation`
   - Require two fresh credential-scrubbed, network-disabled CPU workers to
     reproduce the complete timing-free non-process projection byte-for-byte.

## Invariants

- All thresholds, formulas, regime definitions, counts, dtypes, row ordering,
  baseline seed, permutation, and ridge settings are frozen before any fixture
  result is observed. There is no seed search or validation-based choice.
- The train/validation/test split unit is the complete generator regime, not a
  row or repeated observation. Validation is a holdout guard, not a tuning
  surface.
- The learner never receives regime metadata, target/reward fields, sample
  identity, row order, or future information. A horizon-one trajectory
  prevents temporal leakage in this first rung.
- True, shuffled, constant, random, and attribution conditions use identical
  held-out examples and scoring. The random baseline has a stream independent
  of every other operation.
- Any accepted leakage sentinel, cross-split collision, nondeterministic
  projection, nonfinite metric, malformed typed field, missing case, or
  shuffle-control recovery fails the study.
- The result retains only counts, Booleans, finite aggregate metrics, versions,
  and SHA-256 commitments. It retains no examples, labels, predictions,
  actions by row, model weights, paths, credentials, private state, or
  structure identities.
- The protected submission tree, packaged defaults, submitted ZIP, and all six
  terminal study records remain unchanged.
- The controller's pinned registry digest, five approved source hashes, global
  lease, private state/events, output cap, one-hour timeout, CPU pin,
  credential scrub, network block, immutable result, and SHA sidecar remain
  mandatory.

## Stopping and decision rule

Run the complete eight-case contract exactly once. There is no formula,
threshold, regime, count, dtype, split, permutation, seed, ridge coefficient,
case, or control change after observing a result; there is no retry or top-up.

- If every case passes, record
  `synthetic_supervised_toy_signal_recovered_for_harness` and advance only to a
  new learning-mechanics question. Do not integrate this policy, start a
  contextual bandit, or modify the candidate in the same cycle.
- If any case fails, is malformed, times out, is nondeterministic, accepts a
  leakage sentinel, or encounters source drift, record or preserve the
  failure, park the learning lane, and request owner review. Do not alter the
  fixture and rerun it.

## Claim boundary

The strongest permitted positive claim is:

> On the locked local CPU runtime, one fixed linear ridge policy recovered the
> deliberately exposed synthetic binary signal across the predeclared held-out
> generator regimes, beat the frozen constant and random policies under the
> frozen accuracy rule, and returned to baseline under the identically scored
> training-label shuffle and signal-ablation controls.

This can support only the deterministic learning harness and the deliberately
learnable synthetic signal. It cannot establish useful hidden-topology
generalization, real constraint prediction, UIFO value, competition score,
candidate improvement, contextual-bandit or RL value, causal attribution to a
real diagnostic, accelerator behavior, or a reason to change the submission.
It does not authorize official-data training, private-trajectory selection,
candidate integration, paid compute, or portal action.
