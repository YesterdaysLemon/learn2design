# Frozen plan: target-independent multi-step TD action prefix v2

Date: 2026-08-28  
Study ID: `multistep-td-action-prefix-v2`  
Status: rejected before terminal execution; retained as the frozen preflight
record only

> **Closed record.** Pre-result audit rejected this fixture after development
> diagnostics but before controller execution. Do not run, repair, register, or
> reuse this ID. See
> [`2026-08-28-multistep-td-action-prefix-v2-preflight-rejection.md`](2026-08-28-multistep-td-action-prefix-v2-preflight-rejection.md).

This is a fresh study, not a repair, rerun, or continuation of the rejected
`multistep-td-propagation-v1` fixture. The v1 source and worker must remain
unregistered and must never be imported by this study.

## Question and claim boundary

Ask exactly one falsifiable question:

> Can a blank deterministic tabular TD(0) learner propagate a terminal binary
> signal backward through three bootstrap boundaries in a four-step synthetic
> action-dependent process, score well on untouched held-out generator regimes,
> beat frozen constant, myopic, no-bootstrap, and seeded-random baselines, and
> lose the signal under separately frozen transition-target, reward-origin, and
> signal-ablation controls?

A pass may support only the deterministic synthetic CPU action-prefix/TD
harness and its deliberately learnable toy propagation signal. It is not
evidence about hidden structure, the official dataset, the submitted optimizer,
candidate performance, production RL, meta-RL, native rewrites, accelerators,
competition score, or a no-feedback oracle.

## Frozen topology-independent family

There are four train regimes, two validation regimes, and two test regimes.
Every regime contains 32 episodes in canonical episode order. Each episode has
four actions and exactly three nonterminal bootstrap boundaries.

| split | code | signal scale | nuisance shift | nuisance scale |
|---|---:|---:|---:|---:|
| train | 1301 | 0.70 | -0.90 | 0.80 |
| train | 1303 | 0.90 | -0.30 | 1.05 |
| train | 1307 | 1.10 | 0.30 | 0.75 |
| train | 1319 | 1.30 | 0.90 | 1.20 |
| validation | 1409 | 0.60 | -1.40 | 0.65 |
| validation | 1423 | 1.40 | 1.40 | 1.30 |
| test | 1511 | 0.50 | -1.90 | 0.55 |
| test | 1523 | 1.50 | 1.90 | 1.45 |

For episode index `e` and regime code `c`, define the canonical binary target
twice, by separately implemented generator and evaluator formulas:

```text
train:       target = 0 if e mod 16 < 8 else 1
held-out:    target = 1[((5 e + c) mod 32) >= 16]
```

The held-out formula is balanced because multiplication by five permutes the
32 residues. No regime, split, seed, or threshold may be changed after this
plan is committed.

The public observation is an immutable C-contiguous `float64[4]` with fields
`(phase, signal, action_prefix, nuisance)`. Its generator accepts regime,
episode, phase, and prefix, but no target or outcome argument. It implements:

```text
sign_bit = the split/code/episode formula above, evaluated inside the public
           generator rather than supplied as hidden target state
signal   = (-1 if sign_bit = 0 else +1)
           * signal_scale * (1 + 0.02 * ((3 e + c) mod 9))
base     = nuisance_shift
           + nuisance_scale * ((((13 e + c) mod 31) - 15) / 15)
nuisance = base + phase / 16 + action_prefix / 32
observation = [float(phase), signal, float(action_prefix), nuisance]
```

At phase zero, the prefix is zero. For phases `0, 1, 2`, action `a` produces
the public successor

```text
next_phase  = phase + 1
next_prefix = 2 * action_prefix + a
```

and the observation formula is evaluated at that pair. The successor uses no
target, correctness, reward, `alive`, donor, origin, or control mode. The
target-swap twin first freezes the already generated public observation and
successor bytes and then flips only the evaluator's hidden target; it must
preserve every public byte while changing terminal outcomes. It must not call
the public generator with a target input.

At phase three, action `a` completes `word = 2 * action_prefix + a`. The
evaluator alone materializes terminal reward `1.0` when `(target, word)` is
`(0, 0)` or `(1, 15)`, and `0.0` otherwise. Terminal `done` is true and there
is no successor. All nonterminal rewards are scalar `float64(0.0)` with
`done=false`.

### Complete family commitment

Every episode has 15 legal public prefix nodes, 14 nonterminal action rows, and
16 terminal action/outcome rows. Across all 256 episodes this is exactly 3,840
nodes, 3,584 nonterminal rows, 4,096 terminal rows, and 7,680 action rows.
Every regime is exactly target-balanced 16/16; all split/regime/episode/phase/
prefix/action keys are unique, and the train, validation, and test key sets are
pairwise disjoint. The 480 public observation-byte commitments in each regime
must also be disjoint from those of every other regime; full realized-path
commitments must be disjoint across splits. Before any learner runs, the
implementation must serialize a key-sorted projection of all 7,680 rows. Each
row commits:

- split, regime code, episode, phase, prefix, action, and hidden-target digest;
- predecessor key and immutable observation byte digest;
- successor key and observation byte digest, or explicit null at terminal;
- canonical scalar reward and a literal Python Boolean `done`;
- dtype, shape, C-contiguity, immutability, and phase/prefix legality.

A separately written replay path must regenerate every node, edge, outcome,
and Boolean without calling the production generator, successor, reward, or
projection helpers. The two projections must match exactly. Independent
mutation sentinels for predecessor, successor, terminal reward, and `done`
must each change the commitment. The exact LF-normalized commitment is to be
pinned in the implementation and registry before policy execution; computing
that hash is not a result-bearing learner run.

## Typed policy, transition, and logging contracts

The action is an immutable scalar `int8` array in `{0, 1}`. Python and NumPy
Boolean actions are always rejected. A `done` value is accepted only when
`type(value) is bool`; integers, NumPy Boolean scalars, and truthy objects are
rejected for `done`.

Before selection the learner receives exactly:

```text
{"observation": immutable float64[4]}
```

The selector cannot reach target, reward, `done`, successor, key, phase
counter, split, regime code, episode identity, RNG, evaluator, generator,
held-out source, donor, origin, control mode, or a terminal scalar. Lazy
exploding sentinels must prove that successor and reward are not materialized
until after the action validates.

After action validation, the environment constructs the actual transition and
canonical evaluator feedback. The learner update receives only its internally
pending predecessor/action plus an immutable scalar reward, a literal Boolean
`done`, and the bootstrap successor observation or null. Donor and origin
identity remain evaluator-only.

For each step the exact order is:

```text
observe -> select -> validate action -> construct actual transition
-> materialize this step's scalar feedback -> form TD target -> update once
-> log authenticated components -> observe next state or close episode
```

Every failed call must leave the complete learner state unchanged.

The evaluator stores independently keyed observation, action, transition, and
feedback records. Every component independently repeats its episode/step key,
phase, predecessor observation commitment, action, and `done`. Transition
records additionally authenticate actual successor, bootstrap successor, and
donor. Feedback records additionally authenticate canonical reward,
learner-delivered `update_reward`, and origin. Scoring rejoins by keys rather
than list position.

Valid traces must score identically when observations are reversed, actions
are rotated by one, transitions use even-then-odd order, and feedback uses
odd-then-even order. The fixed reorderings are independent. The scorer must
reject, without partial mutation, missing, duplicate, extra, malformed, or
cross-episode components; wrong phase, dtype, shape, key, predecessor,
successor, action, canonical reward, update reward, donor, origin, or `done`;
nonfinite values; and non-Boolean `done` values.

Only the declared component may differ by treatment:

- transition-target: bootstrap successor and donor only;
- reward-origin: terminal `update_reward` and origin only;
- signal ablation: the declared signal coordinate in every public observation
  only.

Actual observations, actions, canonical successors, canonical rewards, keys,
and `done` remain unchanged across the first two controls.

## Frozen learner and update order

The learner starts from an all-zero `float64[4, 2, 8, 2]` table indexed by
`(phase, sign_bin, action_prefix, action)`. `sign_bin` is zero for a nonpositive
signal and one otherwise. Unreachable prefix cells remain zero. Learning rate
and discount are both exactly one. Lower action wins every tie.

The only updates are forward TD(0):

```text
nonterminal target = 0.0 + max_a Q(next_state, a)
terminal target    = terminal update_reward
Q(current_state, chosen_action) = target
```

Exactly one chosen cell is assigned per update; at most that cell may change,
because assigning zero to an already-zero cell is permitted. Every unchosen
cell must remain byte-identical. Direct full-return, distance-based,
backward-sweep, eligibility-trace, Monte Carlo, copied-terminal, or replay
updates are forbidden.

The train behavior action is constant across the four phases and is fixed by
`episode mod 16`:

```text
[0,0,0,0, 1,1,1,1, 0,0,0,0, 1,1,1,1]
```

This schedule is fixed analytically before execution. In each train regime it
has 16 successful episodes, return `16/32 = 0.50`, and regret 16, hence total
train regret 64. Under canonical feedback, target-zero values must first appear
at phases `[3,2,1,0]` (terminal-to-root orientation) on offsets `[0,1,2,3]`;
target-one values must first appear at the same phase orientation on offsets
`[12,13,14,15]`. There are exactly 512 updates: 384 nonterminal and 128
terminal.

Dynamic dependency sentinels must prove that changing only a successor value
changes the nonterminal target by exactly the same amount, changing the
terminal scalar immediately changes only the selected terminal cell, and an
earlier cell changes only on a later visit after each bootstrap boundary.

Training accepts a train-only source, never a dictionary of splits. It must
produce identical complete state and train-trace commitments when held-out
sources are absent and when an object that explodes on attribute access,
iteration, `next`, length, indexing, truth testing, or array conversion is
present outside the train API. Validation and test are generated only after
the train state freezes; they perform no updates, create no pending state, and
leave every state field byte-identical.

## Frozen baselines and metric

All arms use identical keyed episodes and canonical terminal evaluation.
Return is the mean terminal reward per episode. Split macro return gives every
complete regime equal weight; minimum held-out-regime return is also reported.
Regret is `1 - terminal_reward` summed over complete episodes.

Baselines are:

- constant action zero at all phases;
- constant action one at all phases;
- myopic: an independently implemented terminal-only empirical-mean table fit
  from train behavior feedback; phases 0--2 always choose lower-action tie
  zero, while phase 3 greedily chooses the fitted immediate terminal mean for
  the current `(sign_bin, prefix, action)` and uses lower-action tie zero. It
  receives no evaluator formula, counterfactual outcome, or successor;
- no-bootstrap: the same train schedule and table but target equals immediate
  reward at every step (`discount=0`);
- seeded random: for train/validation/test tags `0/1/2`, construct exactly
  `Generator(PCG64(SeedSequence([2026082817, split_tag])))`; draw exactly one
  `integers(0, 2, dtype=int8)` value per action in canonical order and share no
  generator with the family or learner.

The two constants, myopic, and no-bootstrap baselines are analytically expected
to score 0.5 on balanced held-out regimes. Their exact replay and the seeded
random replay must be independently repeated; observed random metrics cannot
change any rule.

The positive gate requires all of:

- exact behavior return `0.50` in every train regime and exact total train
  regret 64;
- post-fit train, validation, and test macro return at least `0.99`;
- every validation/test regime return at least `0.98`;
- validation and test gain at least `0.30` over the best constant, myopic,
  no-bootstrap, and seeded-random comparator on the same split;
- exact 512/384/128 total/bootstrap/terminal update counts, exact analytic
  propagation offsets, no held-out updates, empty pending state at boundaries,
  and every non-control case passing.

No validation selection, early stopping, normalization fit, hyperparameter
choice, seed search, or checkpoint selection is allowed.

## Frozen negative controls

### Transition-target control

Within each train regime and each 16-episode block, bootstrap successors use
the donor permutation

```text
[8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7]
```

The mapping is a no-fixed-point bijection, flips the signal/target half, and
preserves the fixed behavior action. The actual public successor and canonical
reward remain those of the destination episode; only the learner's
nonterminal bootstrap successor and evaluator-only donor identity change.
Terminal updates are canonical. The control passes only if its validation and
test macro returns are at most `0.55`, the true-minus-control test gap is at
least `0.40`, the positive gate is rejected, and all mapping/authentication
invariants pass.

### Outcome-blind reward-origin control

Within each train regime and each 16-episode block, destination offsets use
this literal origin permutation:

```text
[12,13,4,5,14,15,8,9,0,1,6,7,2,3,10,11]
```

The mapping is constructed from fixed slots only. A poisoned reward oracle
must raise if mapping construction attempts to inspect an outcome. It is a
no-fixed-point bijection within the block.

The four destination cells are exactly
`(target, behavior_action) = (0,0), (0,1), (1,0), (1,1)`, four episodes each.
For every cell, the literal mapping assigns exactly two origin successes and
two origin failures, checked as integer counts `2/4`, not by floating-point
tolerance. The global eight-one/eight-zero multiset is unchanged.

No future reward vector or delayed scalar queue may exist. At a terminal event
only, the evaluator recomputes the frozen origin episode's terminal outcome
and gives that one scalar to the learner. At nonterminal events only the fixed
zero exists. Origin identity never crosses the learner boundary. The control
passes only if validation and test macro returns are at most `0.55`, the
true-minus-control test gap is at least `0.40`, the positive gate is rejected,
and all per-cell, timing, and authentication invariants pass.

### All-trajectory signal ablation

Set only observation index one to exact `0.0` in every legal public node for
all train, validation, and test regimes and for canonical and control
projections. The full observation digest must change whenever the canonical
signal is nonzero, while every non-signal coordinate byte slice, node/edge key,
prefix, action, target, reward, successor link, donor, origin, and `done` must
remain equal. Check the complete 7,680-row family, not only realized paths.

Run both a fresh ablated fit/evaluation and the true frozen policy on ablated
held-out observations. Each test macro return must be at most `0.55`, and the
positive gate must be rejected.

## Frozen attack matrix

The implementation must attempt and reject each of these before the terminal
case can pass, with complete state equality before and after every rejection:

1. update before selection;
2. phase-one selection before phase zero;
3. repeated, skipped, rewound, or post-terminal selection;
4. transition before action validation;
5. successor before transition or a successor for the wrong action;
6. nonterminal or terminal scalar materialized early;
7. terminal scalar late after another transition or delivered twice;
8. missing or nonzero nonterminal feedback;
9. `done=true` nonterminal or `done=false` terminal;
10. successor absent nonterminal or present terminal;
11. update with wrong observation, action, predecessor, or phase;
12. duplicate update, log before update, close before update, or update after close;
13. next episode or split while pending;
14. nonempty queue or pending state at train/held-out boundary;
15. donor identity exposed to the learner at any time, origin identity exposed
    to the learner at any time, or origin scalar materialized before terminal
    feedback; evaluator-only transition donors may exist at their declared
    nonterminal transition boundary;
16. mutable observation or pending-observation mutation;
17. held-out updater call or any held-out-source access during training;
18. malformed/reentrant call and failed-call partial mutation.

## Exact result cases

The registry and worker must expose exactly these cases and aggregate fields;
every case also has Boolean `passed`:

1. `typed_action_prefix_contract`: `actions_checked`,
   `immutable_observations_checked`, `legal_rows_checked`, `trace_sha256`.
2. `target_independent_public_successors`: `successors_checked`,
   `target_swap_twins_checked`, `target_swap_outcomes_changed`, `trace_sha256`.
3. `complete_legal_family_commitment`: `dataset_sha256`, `episodes_checked`,
   `legal_rows_checked`, `nodes_checked`, `nonterminal_rows_checked`,
   `public_rows_disjoint`, `split_keys_disjoint`, `target_balance_exact`,
   `terminal_rows_checked`.
4. `independent_family_replay`: `replay_sha256`,
   `mutation_sentinels_checked`, `mutation_sentinels_rejected`.
5. `physical_pre_action_boundary`: `forbidden_fields_checked`,
   `exploding_sentinels_checked`, `exploding_sentinels_rejected`,
   `selector_inputs_checked`, `trace_sha256`.
6. `heldout_absent_source`: `state_sha256`, `train_trace_sha256`.
7. `heldout_exploding_source`: `operations_checked`, `operations_unreached`,
   `state_sha256`, `train_trace_sha256`.
8. `td_target_dependency`: `dependency_checks`,
   `earlier_cells_unchanged`, `trace_sha256`.
9. `td_update_order_and_terminal_dependency`: `bootstrap_updates`,
   `event_orders_checked`, `propagation_offsets_exact`, `terminal_updates`,
   `total_updates`, `trace_sha256`.
10. `authenticated_component_recombination`: `components_authenticated`,
    `reorder_variants`, `scores_equal`, `trace_sha256`.
11. `malformed_and_cross_episode_rejection`: `attacks_checked`,
    `attacks_rejected`, `state_unchanged_checks`, `trace_sha256`.
12. `baseline_replay`: `best_baseline_test_macro_return`,
    `best_baseline_validation_macro_return`, `constant_returns_exact`,
    `myopic_returns_exact`, `no_bootstrap_returns_exact`,
    `random_replay_exact`, `trace_sha256`.
13. `multistep_value_recovery`: `behavior_regret`, `behavior_return`,
    `minimum_heldout_regime_return`, `postfit_test_macro_return`,
    `postfit_train_macro_return`, `postfit_validation_macro_return`,
    `test_gain_baseline`, `validation_gain_baseline`, `trace_sha256`.
14. `transition_target_control`: `donor_mapping_exact`,
    `positive_gate_rejected`, `test_macro_return`, `true_test_gap`,
    `validation_macro_return`, `trace_sha256`.
15. `outcome_blind_reward_origin_control`: `cell_balance_exact`,
    `mapping_outcome_blind`, `origin_mapping_exact`,
    `positive_gate_rejected`, `reward_multiset_unchanged`,
    `test_macro_return`, `true_test_gap`, `validation_macro_return`,
    `trace_sha256`.
16. `full_timing_attack_matrix`: `attacks_checked`, `attacks_rejected`,
    `state_unchanged_checks`, `trace_sha256`.
17. `all_trajectory_signal_ablation`: `legal_rows_checked`,
    `only_signal_changed`, `positive_gate_rejected`,
    `refit_test_macro_return`, `true_policy_test_macro_return`, `trace_sha256`.
18. `process_isolation`: `trace_sha256`.

The worker top level is exactly `action`, `cases`, `environment`, `fixture`,
`schema_version`, `status`, and `study_id`. The fixture equals the registry's
frozen identity plus `case_contract`. Environment is exactly CPU device kind,
JAX version, platform, and Python version. Results retain only aggregate
counts, metrics, and commitments: no raw observations, states, actions,
rewards, transitions, trajectories, tables, targets, policy parameters, paths,
or private data.

Two credential-scrubbed, network-disabled fresh-process projections must match
the local complete non-process projection byte-for-byte. Focused tests may mark
process isolation incomplete, but the guarded controller accepts only a
terminal Boolean.

## Stopping rule and actions

There is one terminal attempt after all of the following: focused tests pass;
an independent pre-result audit finds no substantive confound; implementation,
worker, tests, registry, source hashes, and normalized controller digest are in
a clean commit; protected submission pins still match; and the controller is
still `awaiting_study` without a stop marker or lease collision.

Success action:
`synthetic_multistep_td_action_prefix_recovered_for_harness`.

Failure action: `park_multistep_td_action_prefix_research`.

Any failed, timed-out, malformed, nondeterministic, drifted, or incomplete
terminal attempt parks the controller. No failed rule may be relaxed; no case,
seed, split, regime, threshold, permutation, or fixture may be changed or
repeated. If pre-result review finds a substantive confound, do not invoke the
controller: quarantine this ID, record a sanitized rejection, and require a
fresh versioned plan.

Implementation is local CPU only. This plan authorizes no official-data use,
candidate integration, submission edit, artifact overwrite, portal action,
leaderboard action, GPU, cloud/provider resource, Docker, SSH, paid endpoint,
spend, merge, or terminal execution before the separate clean approval commit.
