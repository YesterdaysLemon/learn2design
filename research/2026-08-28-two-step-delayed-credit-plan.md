# Frozen plan: two-step delayed-credit mechanics

Status: **frozen before implementation or result-bearing execution**  
Study ID: `two-step-delayed-credit-v1`  
Date frozen: 2026-08-28  
Execution budget: one guarded local-CPU terminal invocation

## Narrow question

Can a small deterministic tabular return learner assign a terminal reward to
both choices in a deliberately learnable two-step, action-dependent synthetic
process, then recover the optimal policy on untouched generator regimes, beat
frozen myopic, constant, and seeded-random baselines, and return to the frozen
control ceilings when either the observed transition or the delayed reward is
misassigned during training?

This is a mechanics question about a toy harness. It is not a test of the
competition task, the submitted optimizer, hidden structure, production RL,
meta-RL, accelerator performance, or score.

## Synthetic episodic family

The family is deterministic, topology-independent, and contains no graph,
network, official-data, UIFO, candidate, or private-evidence input. There are
exactly two action selections and one terminal reward in every episode.

Each regime is the tuple
`(split, code, signal_scale, nuisance_shift, nuisance_scale)`:

| split | code | signal scale | nuisance shift | nuisance scale |
|---|---:|---:|---:|---:|
| train | 701 | 0.80 | -0.80 | 0.85 |
| train | 709 | 0.95 | -0.25 | 1.10 |
| train | 719 | 1.10 | 0.25 | 0.75 |
| train | 727 | 1.25 | 0.80 | 1.20 |
| validation | 803 | 0.65 | -1.35 | 0.65 |
| validation | 811 | 1.35 | 1.35 | 1.25 |
| test | 907 | 0.55 | -1.85 | 0.55 |
| test | 919 | 1.45 | 1.85 | 1.45 |

Every regime contains exactly 32 episodes in canonical episode order
`0..31`. Therefore the complete family contains 128 train, 64 validation, and
64 test episodes and twice as many action records. The generator uses no RNG.

For local episode index `e`:

```text
sign = -1 when e is even, otherwise +1
target = 0 when sign is -1, otherwise 1
magnitude = signal_scale * (1 + 0.03 * (e mod 5))
nuisance = nuisance_shift
         + nuisance_scale * ((((7*e + code) mod 23) - 11) / 11)
state0 = [0.0, sign*magnitude, 0.0, nuisance]
```

After first action `a0`, the canonical action-dependent transition is:

```text
branch = -1.0 when a0 == 0, otherwise +1.0
state1 = [1.0, 0.0, branch, nuisance + 0.125*branch]
reward at the transition = none
done = false
```

After second action `a1`, the evaluator computes exactly one terminal scalar:

```text
terminal_reward = 1.0 iff (a0 == target and a1 == a0), else 0.0
done = true
```

The terminal reward depends on both actions and is unavailable until after the
second action. The public branch says which branch was realized; it never says
whether that branch matches the hidden target. Initial signal, nuisance, target,
keys, and both action-dependent successor coordinates are regenerated from the
closed formulas above. The implementation must commit the complete generated
family with a SHA-256 value before terminal execution.

## Typed schemas and physical information boundary

The observation at either phase is an immutable, C-contiguous, finite
`float64[4]` array with fields `(phase, signal, branch, nuisance)`. Phase is
exactly `0.0` or `1.0`; a phase-zero row has branch zero, and a phase-one row
has signal zero. An action is a scalar NumPy `int8` with value 0 or 1. Boolean,
Python integer, floating-point, wrong-shape, and out-of-range actions are
rejected before coercion.

Evaluator-only episode keys are `(split, regime_code, episode)`. Evaluator-only
action keys append phase 0 or 1. A typed completed trajectory contains exactly
two keyed observation/action records, one authenticated action-dependent
transition, one terminal `float64` reward in `{0.0, 1.0}`, and the done pattern
`[false, true]`. Duplicate, missing, noncontiguous, or cross-episode records
are invalid.

At each selection the learner receives an immutable mapping with exactly one
field, `{"observation": observation}`. It must never receive a target,
preferred action, reward, transition object, next observation, terminal state,
counterfactual reward, split, regime code, episode/action key, trajectory ID,
step, done flag, delay, RNG, generator, evaluator, environment handle, control
mode, donor identity, or reward origin.

The only learner-facing feedback is one scalar `terminal_reward` passed to the
episode updater after both selections. In particular, no zero placeholder or
callback is sent after action zero. Mutable observations and hidden-field
probes must fail closed.

The canonical evaluator event order is exactly:

```text
observe0, select0, transition, observe1, select1,
terminal_reward, update_episode, log_episode
```

The learner stores its own two observation-bin/action commitments. Exactly one
episode update is legal after the terminal scalar. Within that call it updates
the phase-one cell first and the phase-zero cell second, then clears the pending
episode. No value-table mutation may occur before the terminal reward. Missing
transition, update before either selection, consecutive phase-zero selections,
phase-one selection before a phase-zero commitment, duplicate selection,
duplicate update, wrong pending action or observation commitment, malformed
reward, stale pending episode, early reward, or nonempty pending state at a
split boundary is a hard failure.

## Learner and update rule

The learner owns `counts[4,2]` as `int32` and `return_sums[4,2]` as `float64`.
The four public state bins are, in order, start-negative, start-positive,
branch-negative, and branch-positive. No nuisance, key, or counter is a value
table key.

For every train episode, the behavior action pair is selected from the exact
eight-episode schedule below, repeated in order:

```text
episode mod 8:  0    1    2    3    4    5    6    7
(a0, a1):      00   00   01   01   10   10   11   11
```

The schedule is derived only from the learner's completed-episode count and is
identical in the true and negative-control arms. It is not passed evaluator
metadata. After the terminal reward, both visited cells receive that same
undiscounted return:

```text
counts[cell, action] += 1
return_sums[cell, action] += terminal_reward
```

The phase-one cell is updated before the phase-zero cell. The learned greedy
policy chooses the greater empirical mean; an unobserved or exact-tie choice is
action 0. Train behavior intentionally remains fixed so that action, key, and
canonical reward commitments are identical between treatment arms. Online
learning here means that the value table is updated once per completed episode,
not that the exploration schedule adapts.

Canonical train behavior has exactly 32 successes, mean return 0.25, regret 96,
128 completed episode updates, and 256 table-cell updates. These are frozen
integrity expectations. After fitting, the true learner should choose 0/0 for
negative starts and 1/1 for positive starts, producing return 1.0.

Validation and test use the frozen greedy train state. The evaluator supplies
no held-out rewards to the learner and invokes no updater. Validation cannot be
used for checkpoint selection, early stopping, normalization, calibration,
threshold selection, or policy selection. Train must complete with exploding
or absent held-out iterators, and its complete mutable-state commitment must be
identical whether held-out objects exist or not.

## Metrics and baselines

Primary performance is mean terminal return, first averaged within a complete
regime and then macro-averaged with equal regime weight. The minimum complete
held-out-regime return is also retained. Regret is `episode_count - reward_sum`.
Every arm is evaluated on the identical keyed canonical held-out episodes.

Frozen baselines are:

- constant pair `0/0`;
- constant pair `1/1`;
- a myopic learner that credits only transition-local reward, sees no reward
  after action zero, learns the phase-one terminal table, and therefore uses
  tie action 0 at phase zero;
- an independent seeded random policy using NumPy `PCG64(2026082803)`, exactly
  two draws per validation episode followed by two draws per test episode.

The random stream is independent of the generator and learner. Baseline replay
must be byte-identical in a second construction. Constants and myopic score
exactly 0.5 on every complete regime; no exact random score is used as a gate.

The positive gate requires all of:

- canonical behavior train mean exactly 0.25 and regret exactly 96;
- post-fit train, validation, and test macro return at least 0.99;
- minimum held-out-regime return at least 0.98;
- validation and test gains over the better constant at least 0.30;
- validation and test gains over myopic at least 0.30;
- validation and test gains over seeded random at least 0.30;
- exactly 128 episode updates and 256 ordered cell updates;
- zero held-out updates and an unchanged complete learner-state commitment.

## Frozen negative controls

All controls use fresh learners. They retain the canonical generator rows,
initial observations, hidden targets, episode/action keys, fixed action schedule,
episode counts, split boundaries, terminal reward formula, tie action, held-out
evaluator, and thresholds. Controls affect training only. Validation and test
always use canonical transitions and rewards without learner updates.

### Transition shuffle

For every train transition, the evaluator replaces the complete public branch
output with the opposite branch:

```text
observed_branch = -canonical_branch
state1 = [1.0, 0.0, observed_branch,
          nuisance + 0.125*observed_branch]
```

The hidden realized first action and terminal reward remain canonical. This is
equivalent to the fixed branch-label permutation `(negative, positive) ->
(positive, negative)`. Because every eight-episode behavior block is balanced,
the per-regime successor-row multiset is preserved while every rowwise
transition assignment changes. The learner receives no source or mode flag.

The control must have validation and test macro return at most 0.05, a true
minus control test gap at least 0.90, and must reject the positive gate.

### Reward-delay/misalignment control

A correctly tracked longer transport delay is not a negative control. This arm
instead tests the reward-origin bookkeeping failure that a delayed-credit
harness must detect: at each canonical terminal update, it supplies a scalar
from a different episode under a frozen hidden derangement. Within each train
regime's 32 episodes, destination episode `d` receives the canonical terminal
reward from origin index in this exact array:

```text
[3, 7, 4, 5, 6, 23, 9, 10,
 11, 12, 8, 13, 14, 17, 24, 18,
 19, 20, 21, 15, 22, 25, 26, 31,
 0, 27, 28, 29, 16, 30, 1, 2]
```

The mapping is a no-fixed-point permutation. It preserves the exact reward
multiset, transitions, actions, keys, call counts, and scalar delivery slot,
but exposes neither origin nor queue metadata. Its eight positive destinations
have local indices 0 through 7 exactly once, so every value-table cell receives
the same mean assigned return, 0.25. The tie policy is therefore 0/0.

The control must have validation and test macro return at most 0.55, a true
minus control test gap at least 0.40, and must reject the positive gate. The
same case must reject early delivery, delivery after an extra public transition,
duplicate delivery, an origin-bearing reward object, and a nonempty delayed
queue at the train/held-out boundary. The result may support only the frozen
misalignment-control behavior, not a claim that any correctly associated
transport delay is unlearnable.

### Signal attribution

Zero only the phase-zero signal coordinate in train and held-out observations.
Nuisances, transitions, hidden targets, evaluator rows, keys, action schedule,
and reward formula remain byte-identical. A fresh learner refit on the ablated
train family and the true learner evaluated on ablated test rows must each have
test macro return at most 0.55. The input-integrity check must prove that only
the signal coordinate changed. Both variants must reject the positive gate.

## Leakage, attribution, and scoring invariants

Before terminal execution the implementation and tests must establish all of:

- exact generator replay, formula conformance, finite immutable arrays, typed
  schemas, regime membership/order/counts, unique contiguous keys, balanced
  targets/actions, and disjoint regimes, keys, and exact observation rows;
- the complete generated-family SHA-256 commitment and independent replay;
- physical learner boundaries with exploding hidden-field, mutation, action,
  timing, pending-state, and held-out-access sentinels;
- exact event counts and order, no table change before terminal feedback,
  phase-one-before-phase-zero update order, and terminal update before logging;
- exact keyed scorer joins with rejection of duplicates, missing phases, wrong
  transitions, wrong rewards, wrong done values, swapped episode records,
  malformed actions, and inconsistent predecessor references;
- score invariance when completed action, transition, and reward logs are
  independently reversed and then joined by authenticated keys;
- identical canonical held-out row commitments for learner and every baseline;
- random-baseline replay and evaluator reward recomputation;
- transition-shuffle preservation of non-treatment commitments, balanced
  successor multiset, and changed rowwise transition commitment;
- reward-derangement preservation of non-treatment commitments, reward
  multiset, no fixed points, per-cell assigned-return equality, and changed
  reward-origin assignment;
- signal-ablation preservation of every coordinate except phase-zero signal;
- fresh-process equality for the complete non-process result projection.

The sanitized result may retain only bounded scalar aggregates, counts,
Booleans, version strings, case contracts, and lowercase SHA-256 commitments.
It must never retain raw observations/states, actions, rewards, returns, targets,
transitions, trajectories, logs, tables, policy state, parameter values, paths,
private evidence, or credentials.

## Complete cases and stopping rule

The registry/result case set is exactly:

1. `typed_episodic_contract`
2. `generator_partition`
3. `action_dependent_transition`
4. `delayed_update_order`
5. `leakage_guards`
6. `baseline_replay`
7. `delayed_credit_recovery`
8. `transition_shuffle_control`
9. `reward_delay_control`
10. `signal_attribution_control`
11. `process_isolation`

After the plan commit, implementation, focused tests, approved source hashes,
registry digest, and one full repository verification pass must be committed in
a second clean pre-result commit. The guarded controller is then invoked exactly
once. No terminal fixture may be rerun, topped up, reseeded, narrowed, or tuned.

All eleven cases must pass. On success, record only the sanitized aggregate
result with action:

`synthetic_two_step_delayed_credit_recovered_for_harness`

On any failed invariant or threshold, negative-control recovery, malformed or
nondeterministic result, timeout, source drift, or process-isolation mismatch,
preserve the terminal evidence, leave the controller parked, stop mutation, and
use action:

`park_delayed_credit_research`

## Claim boundary

A pass may say only that this fixed deterministic table learner and guarded
local CPU harness recovered the deliberately exposed two-step terminal-return
signal on untouched synthetic regimes, beat the frozen toy baselines, and lost
the signal under the frozen transition-assignment, reward-origin, and signal
interventions. It cannot support claims about production RL, meta-RL, longer
horizons, partial observability, official data, hidden topology, UIFO, optimizer
quality, the protected submission, candidate selection, native rewrites, GPU or
accelerator speed, leaderboard standing, or competition score.
