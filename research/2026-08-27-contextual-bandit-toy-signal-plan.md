# Contextual-bandit toy-signal mechanics study plan

Status: frozen before execution

Study ID: `contextual-bandit-toy-signal-v1`

Execution class: unpaid, deterministic local CPU learning mechanics only

## Question

Can one fixed deterministic two-action contextual-bandit policy learn an
observation-dependent choice from chosen-action rewards on a sequential toy
family, preserve that policy without updates on held-out generator regimes,
beat frozen constant and seeded-random policies under a precommitted reward
rule, and fall back to chance under identically evaluated shuffled-context and
signal-ablation controls?

This is a bandit-harness and online-update question. It does not ask whether
the toy signal exists in UIFO, whether a learned policy improves the submitted
optimizer, or whether production reinforcement learning should be trained.

## Frozen task, generator, and typed contracts

- The family is synthetic and has no graph, optical layout, official structure,
  structure identifier, or structure-derived input. Its structure kind is
  `none`.
- Each generator regime contains 32 trajectories of exactly eight ordered
  steps. Policy state persists across all four train regimes. Validation and
  test use the final frozen train state and apply no updates.
- A policy-visible context is a finite `float64[3]` array with ordered fields
  `signal`, `nuisance_pair`, and `nuisance_cycle`. The policy selector receives
  only an immutable record containing that context.
- The evaluator-only preferred action and the selected action are scalar
  `int8` values in `{0, 1}`. Reward is an evaluator-computed scalar `float64`
  in `{0.0, 1.0}`, equal to one exactly when selected and preferred actions
  agree. Reward becomes policy-visible only after selection, through one
  update call.
- `done` is scalar Boolean: false at steps zero through six and true at step
  seven. A terminal reward is incorporated before the next trajectory begins;
  trajectory boundaries do not reset policy state.
- Evaluator logging uses the exact event order `context`, `select`, `reward`,
  `update`, `log`. Each train step must create exactly one event of each kind.
  A log record authenticates the evaluator-only key, context commitment,
  action, reward, update index before/after, exploration mode, and `done`, but
  raw logs are never retained in the study result.
- Canonical preferred actions within every trajectory are
  `[0, 1, 0, 1, 0, 1, 0, 1]`. Context signal signs follow the corresponding
  `[-1, +1, -1, +1, -1, +1, -1, +1]` pattern. The four negative rows and four
  positive rows share one magnitude and one nuisance pair per trajectory.
- For trajectory `b` and regime tuple `(code, signal_scale, nuisance_shift,
  nuisance_scale)`, compute directly in `float64`:
  `magnitude = signal_scale * (1 + 0.04 * (b mod 7))`,
  `nuisance_pair = nuisance_shift + nuisance_scale * (((b mod 11) - 5) / 5)`,
  and `nuisance_cycle = 0.4 * nuisance_shift + nuisance_scale *
  ((((5*b + code) mod 19) - 9) / 9)`. Every row concatenates its signed signal
  with that trajectory's two shared nuisance values. No formula uses the
  preferred action.
- The complete regime table is frozen as follows. Train uses
  `(401, 0.75, -0.90, 0.85)`, `(409, 0.90, -0.30, 1.15)`,
  `(419, 1.10, 0.30, 0.80)`, and `(421, 1.25, 0.90, 1.20)`.
  Validation uses `(503, 0.65, -1.40, 0.70)` and
  `(509, 1.35, 1.40, 1.30)`. Test uses `(601, 0.55, -1.90, 0.60)` and
  `(607, 1.45, 1.90, 1.40)`. Tuple fields are regime code, signal scale,
  nuisance shift, and nuisance scale.
- Canonical evaluator keys are `(split, code, trajectory, step)`, ordered by
  the declared regime table, ascending trajectory `0..31`, then ascending step
  `0..7`. Regime codes and keys are disjoint across train, validation, and
  test. Exact context-row hashes must also have zero cross-split overlap;
  repeated same-sign rows within one trajectory are intentional.
- Generation is closed-form and uses no RNG. The random baseline alone uses a
  NumPy `PCG64` stream with seed `2026082713`, consumed once per canonical
  validation row and then once per canonical test row. No global RNG, OS
  entropy, time-derived seed, or data-dependent seed is permitted.

The complete counts are 1,024 train steps, 512 validation steps, and 512 test
steps, with 128, 64, and 64 complete trajectories respectively.

## Frozen policy and online update order

- The policy is a two-context-bin, two-action empirical-reward table. Negative
  signal maps to bin zero; positive signal maps to bin one; an exactly zero
  signal maps to bin zero for the attribution control.
- State contains `int32[2,2]` chosen-action counts, `float64[2,2]` reward sums,
  one integer update counter, and at most one pending selection commitment.
  All counts and sums start at zero.
- Train trajectories alternate exploration and exploitation. Every even
  trajectory uses the fixed action schedule `[0, 0, 1, 1, 0, 0, 1, 1]`.
  Every odd trajectory selects the action with the larger empirical mean for
  the current context bin; an unobserved cell has mean zero and an exact tie
  chooses action zero. The schedule is derived from the internal update count,
  not from policy-visible split, regime, trajectory, step, target, or reward.
- At a train step, `select` reads the immutable context and pre-update state,
  validates and returns one scalar int8 action, and records one pending
  commitment. The evaluator then computes reward from the unchanged hidden
  preferred action. `update` must match the pending context and action, accept
  one finite scalar zero-or-one reward, update only the selected table cell,
  increment the counter exactly once, and clear the pending commitment. The
  evaluator then logs the completed transition. Double selection, update
  before selection, double update, mismatched context/action, malformed action,
  or malformed reward is rejected.
- Validation and test call a frozen greedy evaluator that cannot create a
  pending selection or mutate counts, sums, or the update counter. No
  validation checkpoint, early stopping, calibration, normalization,
  hyperparameter selection, or test-based choice exists.
- On the canonical train stream, each exploration trajectory samples every
  context-bin/action cell exactly twice. The analytically expected reward is
  four of eight on each exploration trajectory and eight of eight on each
  exploitation trajectory: 768 of 1,024 total, mean `0.75`, and regret `256`
  against the one-reward-per-step oracle.

## Frozen baselines, metric, thresholds, and controls

- Constant policies for both action zero and action one are evaluated; the
  comparison baseline is the better held-out constant. Each is analytically
  `0.5` because every complete trajectory is balanced.
- The seeded-random policy uses the one independent PCG64 stream and the exact
  learner held-out rows. Replay must be byte-identical.
- Primary held-out score is equal-weight macro mean reward across complete
  generator regimes. Minimum held-out-regime reward is also a guard. Scoring
  aligns action and reward records by canonical evaluator key, rejects
  duplicate/missing/noncontiguous keys, and recomputes reward from preferred
  action. Reordering only the completed evaluator log must preserve the score.
- The positive policy gate requires: train online mean reward at least `0.74`;
  train cumulative regret at most `264`; validation and test macro reward at
  least `0.99`; minimum validation-or-test regime reward at least `0.98`;
  validation gain over the better constant at least `0.30`; test gain over the
  better constant at least `0.30`; and test gain over seeded random at least
  `0.25`.
- The shuffled-context control starts a fresh identical policy and changes only
  train context assignment within every trajectory using the exact permutation
  `[0, 2, 4, 6, 1, 3, 5, 7]`. The hidden preferred-action sequence, keys,
  regime metadata, exploration schedule, reward equation, validation/test
  contexts, and validation/test preferred actions remain unchanged. Preferred
  actions are never recomputed from shuffled contexts.
- Under that permutation, each context-bin/action cell receives one preferred
  zero and one preferred one on every exploration trajectory; each shuffled
  context bin also contains two preferred zeros and two preferred ones per
  complete trajectory. The control must preserve the train-context multiset,
  all train preferred-action/key commitments, and all held-out commitments;
  change the rowwise train-context commitment; reject the complete positive
  gate; score at most `0.55` validation and test macro reward; and trail the
  true policy by at least `0.40` test reward.
- The attribution control replaces only the declared signal coordinate with
  zero in train, validation, and test. It starts another fresh identical policy
  and otherwise preserves nuisance coordinates, preferred actions, keys,
  regimes, and held-out row identities. Both the refit context-free policy and
  the true policy evaluated on signal-zeroed test contexts must score at most
  `0.55` test macro reward.

## Complete frozen cases

1. `typed_bandit_contract`
   - Require exact context/action/preferred-action/reward/done dtypes, shapes,
     values, context fields, policy-visible field set, horizon, trajectory
     boundary, and logging-event order.
2. `generator_partition`
   - Require the exact formulas, regime table, split and trajectory counts,
     balanced preferred actions, deterministic replay, finite contexts,
     canonical regime/key order, within-split key uniqueness, disjoint split
     regimes/keys/context rows, exact done pattern, and canonical dataset
     commitment.
3. `online_update_order`
   - Require exactly 1,024 selects, rewards, updates, and logs; 512 exploration
     and 512 exploitation selections; 1,024 final updates; all 128 terminal
     train rewards incorporated; exact event ordering; chosen-cell-only state
     changes; and rejection of update-before-select, double-select,
     double-update, mismatched-context, mismatched-action, and malformed-reward
     sentinels.
4. `leakage_guards`
   - Reject policy records exposing `preferred_action`, `reward`, `regime_code`,
     `split`, `sample_key`, `trajectory_id`, `step`, `done`, `next_context`, or
     `counterfactual_reward`. Reject validation and test streams presented to
     the train updater. Require zero held-out updates, unchanged train-state
     commitments across validation/test evaluation, exact keyed score after
     completed-log reversal, rejection of duplicate-key and wrong-reward logs,
     and a changed score under the swapped-action sentinel.
5. `baseline_replay`
   - Require both constant policies and the seeded-random policy to use the
     declared actions/seed and exact held-out keys; require exact random replay,
     identical evaluator rows, and finite macro/regime metrics.
6. `contextual_recovery`
   - Require every positive threshold, exact exploration/exploitation counts,
     and only aggregate metrics and state/action/log commitments in the result.
7. `shuffled_context_control`
   - Require the exact nonidentity permutation and per-trajectory independence
     table; unchanged preferred-action/key/context-multiset and held-out
     commitments; changed rowwise train-context commitment; failed positive
     gate; held-out macro reward ceilings; and true-minus-shuffled test gap.
8. `signal_attribution_control`
   - Require that only the signal coordinate changed, all evaluator metadata
     stayed fixed, and both declared attribution variants satisfy the chance
     ceiling.
9. `process_isolation`
   - Require two fresh credential-scrubbed, network-disabled CPU workers to
     reproduce the complete timing-free non-process projection byte-for-byte.

## Invariants

- All formulas, regimes, counts, dtypes, orderings, action schedule, tie rule,
  update rule, baseline seed, shuffle permutation, thresholds, cases, and
  controls are frozen before executing a policy. There is no seed search or
  validation-based choice.
- The split unit is the complete generator regime. Only train may update state;
  validation is a frozen holdout guard rather than a tuning surface, and test
  remains terminal.
- The selector never receives preferred action, reward, evaluator identity,
  row position, future context, done, RNG, or environment handles. Inferring a
  preferred action from chosen-action reward after selection is valid bandit
  feedback; access before selection is forbidden.
- Canonical, shuffled, constant, random, and attribution arms use identical
  evaluator keys and preferred-action streams. Context shuffle never changes
  hidden preferred actions. The baseline RNG is independent of every learner.
- Any accepted sentinel, cross-split collision, malformed transition,
  nonfinite value, duplicate/missing key, unexpected update, changed held-out
  state, nondeterministic projection, or negative-control recovery fails the
  study.
- The result retains only counts, Booleans, finite aggregate metrics, versions,
  and SHA-256 commitments. It retains no raw contexts, actions, rewards,
  preferred actions, trajectories, logs, table cells, policy state arrays,
  paths, credentials, private state, or structure identities.
- The protected submission tree, packaged defaults, submitted ZIP, and all
  seven terminal study records remain unchanged.
- The controller's pinned registry digest, five approved source hashes, global
  lease, private state/events, output cap, one-hour timeout, CPU pin,
  credential scrub, network block, immutable result, and SHA sidecar remain
  mandatory.

## Stopping and decision rule

Run the complete nine-case contract exactly once. There is no formula,
threshold, regime, count, dtype, ordering, action schedule, seed, permutation,
case, or control change after observing a result; there is no retry or top-up.

- If every case passes, record
  `synthetic_contextual_bandit_signal_recovered_for_harness` and advance only
  to a new unpaid learning-mechanics question. Do not integrate this policy,
  start meta-RL, or modify the candidate in the same cycle.
- If any case fails, is malformed, times out, is nondeterministic, accepts a
  sentinel, or encounters source drift, record or preserve the failure, park
  contextual-bandit research, and request owner review. Do not alter the
  fixture and rerun it.

## Claim boundary

The strongest permitted positive claim is:

> On the locked local CPU runtime, one fixed deterministic two-action
> contextual-bandit policy learned the deliberately exposed synthetic context
> signal from chosen-action reward alone, met the frozen train and held-out
> reward rules, beat the frozen constant and seeded-random policies, and
> returned to chance under the identically scored shuffled-context and
> signal-ablation controls.

This can support only the deterministic bandit harness, online update order,
split/leakage enforcement, and deliberately learnable toy signal. It cannot
establish useful hidden-structure generalization, real constraint prediction,
UIFO value, candidate improvement, production RL or meta-RL value, causal
attribution to a real diagnostic, accelerator behavior, competition score, or
a reason to change the submission. It does not authorize official-data
training, private-trajectory selection, candidate integration, paid compute,
or portal action.
