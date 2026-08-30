# Frozen plan: online SARSA latched-choice mechanics

Status: **frozen before implementation or learner execution**

Study ID: `online-sarsa-latched-choice-v1`

Date frozen: 2026-08-29

Execution budget: none in this checkpoint. A future implementation checkpoint
and a still later guarded local-CPU invocation are separate gates.

## Closed predecessors and design independence

The ten registered local studies are terminal and cannot be rerun. In
particular, `multistep-td-action-prefix-v3` closed the fixed offline
synchronous-TD propagation question. Its public cue deliberately exposed the
toy evaluator target, and its terminal pass did not validate online action
selection or learning from a changing behavior policy.

`multistep-td-propagation-v1` and `multistep-td-action-prefix-v2` remain
quarantined preflight rejections. Their source, worker, formulas, fixtures,
seeds, permutations, thresholds, development diagnostics, and observed outputs
must not be imported, executed, repaired, registered, or used to select this
study. This plan is a new family with a new state layout, episode horizon,
regime codes, behavior schedule, learner, controls, and claim boundary.

No value below is selected against V1, V2, V3, official data, a private panel,
or a development run. The family and its exact exploration counts are fixed
algebraically. The only pseudorandom component is a separately frozen random
comparator whose seed and call contract are declared below before
implementation.

If hostile review finds a substantive confound before this plan is committed,
the draft may be corrected. The commit containing the final text is the freeze
boundary. After that commit, a substantive confound quarantines this ID and
requires a new versioned plan; it is not permission to amend this contract.

## Narrow falsifiable question

Can a blank deterministic tabular SARSA(0) learner, updating online from the
actions selected by its own precommitted exploratory behavior policy, acquire a
deliberately learnable three-action choice; retain the final greedy policy on
untouched validation and test generator regimes; beat frozen constant,
feedback-only myopic, no-bootstrap, and seeded-random comparators under a
precommitted terminal-return metric; and reject the same positive gate under
transition-target-cue-swap, behavior-assignment, terminal-origin-zero, and complete
signal-ablation controls?

A pass may validate only this synthetic online-control harness, its exact
SARSA update order, and its deliberately exposed toy signal. It cannot support
a claim about production RL, sample efficiency beyond this schedule, optimal
exploration, meta-RL, hidden topology, UIFO, the competition optimizer,
candidate value, a native rewrite, accelerator speed, leaderboard standing, or
score.

## Topology-independent generator family

The family contains no topology, graph, network, official archive, UIFO input,
candidate array, private evidence, provider input, or learned generator.
`structure_kind` is exactly `none`. Each episode has three selected actions,
two nonterminal zero rewards, and one evaluator-only terminal reward.

There are exactly eight regimes:

| split | regime code | signal scale |
|---|---:|---:|
| train | 4109 | 0.61 |
| train | 4127 | 0.83 |
| train | 4153 | 1.17 |
| train | 4177 | 1.39 |
| validation | 5209 | 0.53 |
| validation | 5231 | 1.47 |
| test | 6311 | 0.44 |
| test | 6323 | 1.61 |

Each train regime has exactly 2,048 episodes arranged as 64 rounds of 32.
Training order is round, then the four regimes in table order, then block
position. Each validation and test regime has exactly 512 episodes in regime
table order. There are 8,192 train, 1,024 validation, and 1,024 test episodes.

For a train local episode with `round r in 0..63` and `position p in 0..31`:

```text
local_episode = 32*r + p
if p < 16:
    behavior_mode = greedy
    public_cue = p mod 2
    forced_code = none
else:
    j = p - 16
    behavior_mode = forced_explore
    public_cue = j mod 2
    forced_code = j // 2                 # 0..7
```

The three forced actions are the big-endian bits of `forced_code`. Thus every
train regime/round contains one forced traversal of each three-action tuple for
each cue, preceded by sixteen genuinely greedy episodes with eight episodes per
cue. Exactly 4,096 train episodes are greedy and 4,096 are forced exploration.
The forced schedule is balanced independently of the cue: every action tuple
appears exactly once for each cue in every regime/round.

For held-out local episode `e in 0..511`, `public_cue = e mod 2`. Held-out execution is
always greedy and receives no exploration directive. In every split, adjacent
episodes `(2k, 2k+1)` therefore contain opposite cues but the same public
unsigned magnitude and nuisance. Neither value can recover cue or episode
parity.

For every split and regime:

```text
evaluator_cue = public_cue                # canonical-family equality only
sign = -1.0 if public_cue == 0 else +1.0
magnitude = signal_scale * (1.0 + (((local_episode // 2) mod 32) + 1) / 256.0)
nuisance = signal_scale + (((local_episode // 2) mod 32) + 1) / 4096.0
desired_code = 2 if evaluator_cue == 0 else 5  # evaluator-local 010 or 101
action_code = 4*a0 + 2*a1 + a2           # also 2*latch_code + a2
terminal_reward = 1.0 iff action_code == desired_code else 0.0
```

`public_cue` exists only through the public signed scalar and detached family
audit. `evaluator_cue` exists only inside the detached terminal evaluator. The
canonical generator asserts their equality as a fixture fact, but the terminal
formula consumes only `evaluator_cue`; the public cue or signed observation is
not reachable from evaluator code. The evaluator-twin audit flips only
`evaluator_cue` after public bytes are frozen, proving that independence.

The regime code and split are not present in any policy-visible value. The
unsigned magnitude and nuisance distinguish regime families but are identical
for the two cues in every adjacent episode pair and are excluded from the
learner's tabular state. Independently replayed public paths remain split-
disjoint because the frozen signal scales are disjoint; the disjointness proof
uses public bytes only and never evaluator identity metadata.

The terminal rule gives exactly two forced-exploration successes per
regime/round, hence exactly 512 forced-exploration rewards over training. The
first sixteen train episodes are greedy from an all-zero table, choose action
zero at every tie, receive zero terminal rewards, and leave the table exactly
zero. Their forty-eight updates do advance the table version and visit
counters; only the twenty-eight Q-value bytes remain unchanged. Total behavior
reward after all 8,192 train episodes is bounded between
512 and 4,608; its exact value is descriptive and is not used to choose a
threshold or checkpoint.

## Public latched state and evaluator boundary

At phase `p in 0..2`, the public observation is an immutable C-contiguous
little-endian `float64[5]` array:

```text
(phase, signed_cue, latch_code, remaining_actions, nuisance)
```

The exact legal forms are:

```text
phase 0: [0, sign*magnitude, -1, 2, nuisance]
phase 1: [1, sign*magnitude, a0, 1, nuisance]
phase 2: [2, sign*magnitude, 2*a0 + a1, 0, nuisance]
```

For phases zero and one, the successor latches the selected action and changes
only phase, latch, and remaining-action fields. Conditional on already public
predecessor bytes and the selected action, successor construction has no cue,
target, correctness, reward, future action, evaluator, donor, origin, control,
or learner input. Phase two has no successor and `done=true`.

The signed cue intentionally reveals which of the two target patterns is
rewarded. A hand-programmed oracle can read its sign and emit `010` or `101`
without learning. This fixture therefore cannot establish that online learning
or delayed credit is necessary, that the public surface lacks a shortcut, or
that the target is hidden from an arbitrary policy.

The frozen learner maps observations only to:

```text
(cue_bin, phase, latch_code)
```

where `cue_bin` is zero for a nonpositive cue and one for a positive cue.
Magnitude, nuisance, regime, split, episode, round, position, behavior mode,
exploration directive, exact action code, evaluator target, donor, origin, and all
counters are excluded. Across two cue bins there are fourteen abstract states
and twenty-eight state-action cells:

```text
2 cues * (1 + 2 + 4 latched states) * 2 actions = 28 cells
```

The selector never receives that five-scalar array directly. A detached pure
projector validates it and emits exactly three policy-state bytes:

```text
PolicyState(cue_bin:uint8, phase:int8, latch_code:int8)
```

The byte order is exactly the displayed field order. Legal encodings are cue
bin `0` or `1`; phase `0`, `1`, or `2`; and latch `-1` at phase zero, `0..1`
at phase one, or `0..3` at phase two. The value is an immutable exact-length
`bytes` object, not a view, object with a `__dict__`, closure, array alias, or
handle. The state index is frozen as

```text
phase_offset = 0 if phase == 0 else (1 + latch_code if phase == 1
                                     else 3 + latch_code)
state_index = 7*cue_bin + phase_offset
cell_index = 2*state_index + action
```

Changing unsigned magnitude, nuisance, regime, split, key, or row order while
holding these three bytes fixed cannot change selection. Exact projector-input
and policy-state commitments plus the complete signal ablation below must
demonstrate that no other public coordinate reaches the learner.

## Typed identity and data contracts

Split codes are `train=uint8(0)`, `validation=uint8(1)`, and `test=uint8(2)`.
Regime ordinals are zero-based within each split in the table order above.
The exact key schemas are:

- `EpisodeKey(split:uint8, regime_code:int32, local_episode:int16)`;
- `ObservationKey(episode_key, phase:int8, latch_code:int8)`;
- `ActionKey(observation_key, action_value:int8)`, where the value is `0` or
  `1` and therefore uniquely identifies both legal rows at an observation;
- `TransitionKey(action_key, transition_ordinal:int8)`, where the ordinal also
  equals the phase;
- `FeedbackKey(transition_key, feedback_ordinal:int8)`, where the ordinal also
  equals the phase;
- `SelectionKey(episode_key, phase:int8, invocation_ordinal:int32)`, where the
  invocation ordinal is the zero-based global canonical selection ordinal;
- `AssignmentKey(selection_key, executed_action:int8,
  assigned_action:int8)`; and
- `UpdateKey(feedback_key, selection_key, assignment_key,
  table_version_before:int32)`.

The cue-swap control alone adds evaluator-side
`TransitionTargetRef(update_key, real_successor_observation_key,
real_successor_policy_state_sha256, target_policy_state_sha256,
mapping_mode:uint8)`, where mode is exact `1`. It serializes in that field
order with the key encodings above, two lowercase digest byte strings, and one
`uint8`. It binds one authenticated real successor to exactly one opposite-cue
target state. It is recorded in the control `UpdateRecord`; the learner sees
only the three target-state bytes, never the reference or real successor key.

No key contains cue, desired code, target correctness, reward, behavior mode,
or control mode. `bool` and NumPy Booleans must never compare equal to an
integer identity. All integers have the exact declared NumPy width and
little-endian representation.

Every canonical and non-assignment-control update has
`assigned_action=executed_action`, so `AssignmentKey` is always well typed.
Its separate `ActionAssignmentRef` is exact `None` in canonical mode and a
typed present record only in the assignment control.

Runtime values are exact:

- observation: immutable finite C-contiguous little-endian `float64`, shape
  `(5,)`, strides `(8,)`, and the exact legal phase/latch form;
- action: zero-dimensional `int8` scalar in `{0,1}`; Python integers,
  Booleans, floats, one-element arrays, wrong dtypes, and out-of-range values
  reject before an environment lazy value resolves;
- reward/update reward: zero-dimensional little-endian `float64`; nonterminal
  source and update reward exactly `0.0`, terminal source/update reward in
  `{0.0,1.0}` except only where a declared control substitutes the terminal
  updater scalar;
- `done`: exactly Python `False, False, True` by phase; and
- donor, origin, assignment, directive, and authorization identities: exact
  typed records or exact `None`, never integers, strings, or Booleans.

The policy's canonical train selection function receives only the exact
three-byte `PolicyState`, an exact 224-byte immutable snapshot of the current
twenty-eight-cell Q table, and an exact two-byte immutable
`ExplorationDirective(mode:uint8, forced_action:int8)`. `mode=0` means greedy
and requires the `forced_action=-1` sentinel; `mode=1` means forced exploration
and requires action `0` or `1`. Every other byte pair rejects. The directive is
an exact-length `bytes` object with no attributes, future directive, schedule
index, closure, object identity payload, environment reference, or evaluator
handle. The selector validates and decodes the Q snapshot into a newly owned,
read-only, C-contiguous little-endian `float64[28]` array; subclasses, writable
buffers, non-owned aliases, wrong shape/length/dtype, and mutation before or
after selection reject. The mutable Q array is owned only by the updater and is
never passed to selection. Evaluation uses a separate function that receives
only the three-byte state and the same kind of frozen Q snapshot and has no
directive or update capability.

Selection and evaluation are pure and stateless apart from their explicit Q
snapshot. They may not read or write a call counter, module/global/class state,
clock, process identity, environment variable, filesystem, RNG, mutable
closure, object identity, prior action/reward, or future input. The complete
policy state is exactly the Q-value bytes; visits, table version, pending
authorization, and event position are protocol state and cannot affect action
choice. Held-out evaluation is replayed under canonical order, reversed regime
order, and reversal within every adjacent cue pair. Joining by typed episode
keys must give identical action and score commitments for every public policy
state and an unchanged policy digest.

The collector, not the policy, owns a slotted opaque one-use `SelectionAuth`.
It binds the exact `SelectionKey`, policy-state digest, directive digest,
pre-action table version, selected cell index/value digest, and a fresh
authorization nonce derived from the committed schedule digest plus selection
ordinal. It has no public projection and is never passed to policy code. A
`PendingNextAction` additionally binds executed and, where applicable,
assigned action, successor identity, and the latched pre-update Q snapshot.
Duplicate, stale, future, cross-episode, wrong-phase, or reentrant use clears
the pending object and rejects.

The directive generator is a separate pure schedule function of train round,
regime ordinal, block position, and phase; it is constructed before the
environment episode and cannot access observation bytes, cue, target, reward,
Q values, action history, or evaluator. The policy receives none of those
schedule indices. Independent replay must prove that adjacent cue-paired
episodes have identical directive byte streams and identical unsigned
magnitude/nuisance bytes. This is a static conditional-balance audit, not a
separate learned-policy performance control.

## Complete legal-family commitment

Before constructing a behavior policy, collector, learner, or comparator, the
implementation must enumerate every legal environment state/action row for
all 10,240 episodes. Each episode contributes:

```text
phase 0: 1 latch * 2 actions = 2 rows
phase 1: 2 latches * 2 actions = 4 rows
phase 2: 4 latches * 2 actions = 8 rows
total: 14 rows
```

The complete legal-family projection therefore contains exactly 143,360 rows:
61,440
nonterminal rows and 81,920 terminal rows over 71,680 predecessor nodes. It
commits exact episode/observation/action/transition/feedback keys, typed
predecessor bytes and layout, action bytes, every legal typed successor and
layout or terminal sentinel, source reward, canonical update reward, `done`,
public signed-cue slot, detached hidden evaluator cue, canonical
donor/origin/assignment sentinels, and formula lineage. `desired_code` is never
stored: the detached terminal evaluator derives it only while applying the
frozen formula. At every nonterminal legal row, source reward and update reward
are both exact zero. At every terminal legal row, they are separately committed
and equal in the canonical family.

Rows enumerate in split-code order, regime-table order, ascending local
episode, phase `0,1,2`, ascending legal latch, then action `0,1`. Each scalar is
serialized in its declared little-endian NumPy width; `done` is one byte `0` or
`1`; observation and successor payloads are their exact C-order bytes preceded
by dtype code `f8`, rank byte `1`, length `5`, and stride `8`; exact `None` is
one byte `0`, and a present typed identity is one byte `1` followed by its fixed
fields. Every digest is SHA-256 over the UTF-8 domain prefix, a zero byte, the
little-endian `uint32` row count, then length-prefixed rows. Domain prefixes
are exactly:

```text
L2D-online-sarsa-v1-family
L2D-online-sarsa-v1-public-path
L2D-online-sarsa-v1-schedule
L2D-online-sarsa-v1-policy-state
L2D-online-sarsa-v1-directive
L2D-online-sarsa-v1-q-cell
L2D-online-sarsa-v1-q-table
L2D-online-sarsa-v1-visit-table
L2D-online-sarsa-v1-event-trace
L2D-online-sarsa-v1-score
```

Length prefixes are little-endian `uint32`; dictionaries are forbidden from
all digest inputs. Q-table bytes use the twenty-eight-cell index order above as
little-endian `float64`; visits use the same order as little-endian `int32`. A
Q-cell commitment is its domain prefix plus `int8(cell_index)`,
`int32(table_version)`, and exact little-endian value bytes.

A separately implemented replay, sharing no row constructor, must reproduce
every row and the same canonical SHA-256. It independently proves formula
conformance, counts, unique keys, legal successors, terminal outcomes,
cue/action balance, exact train schedule counts, and no identical public
observation or realized-path digest across train, validation, and test. The
10,240 canonical environment paths produce exactly 30,720 realized transition
rows. Disjointness digests exclude split, regime code/ordinal, episode, every
key, hidden evaluator cue, target formula, reward, donor, origin, and identity
metadata. They retain complete public observation bytes, including the public
signed cue, and every action, successor, and `done` byte.

Policy/collector/learner/comparator factory counters begin at zero. Any family,
replay, schedule, type, count, key, legality, terminal formula, target-twin, or
split-disjointness failure raises the frozen contract error while every counter
remains zero. Focused tests must corrupt every row class and prove this
fail-closed boundary.

The implementation checkpoint may compute the exact family and schedule
digests only from these formulas. Those digests must be committed in focused
tests and the later registry before any learner executes. A mismatch
quarantines this study ID.

## Evaluator-twin public invariance

For every episode, the twin audit first freezes every public observation,
legal action, successor, key/link, layout, and `done` byte. It then copies those
bytes and flips only a separate evaluator cue from zero to one or one to zero.
It must not regenerate any public row after the flip. The public signed-cue
array position and its scalar remain frozen; neither is the detached evaluator
cue.
`PublicSignedCue(float64)` and
`EvaluatorCue(uint8)` are distinct detached types with no shared object,
buffer, container, closure, or parent handle.

The twin preserves every public byte, action, successor, key/link, source
identity, schedule directive, and `done`. Only the hidden twin evaluator cue
and formula-authorized terminal twin rewards differ. Of the eight terminal
action rows per episode,
exactly the `010` and `101` rows exchange reward values. Self-comparison,
copy-only comparison without a hidden flip, public regeneration, or any wider
difference fails before policy construction.

## Deterministic behavior-policy exploration contract

The online behavior policy owns the Q table and selects every train action at
the moment declared by the event order below. In forced mode it returns exactly
the directive action without consulting evaluator state. In greedy mode it chooses
the greater Q value for the current abstract state, breaking exact ties to
action zero. No action tuple is installed wholesale: phases one and two are
selected only after their real successor observations exist.

The schedule contains no learner RNG. The exact sequence of directive modes and
forced actions is generated twice by independent schedule implementations and
must match byte-for-byte before learning. Required counts are:

- 8,192 train episodes and 24,576 action selections/updates;
- 4,096 greedy and 4,096 forced-exploration episodes;
- 12,288 greedy and 12,288 forced action selections;
- for every train regime/round/cue, each three-action forced tuple exactly
  once; and
- exactly 512 forced-exploration terminal successes.

Every selection logs only bounded commitments to the full three-byte policy
input, pre-action Q version and complete Q-table digest, selected-cell index,
value and cell digest, two-byte directive digest or evaluation sentinel,
chosen executed action, control assigned action or `None`, tie decision, and
selection ordinal. An independently implemented behavior replay shares no
state projector, selection, update, table-index, or digest helper. It joins
these commitments by typed keys, regenerates every pre-action Q snapshot from
the blank table plus authenticated prior updates, recomputes each action from
the exact snapshot and directive, and must
reproduce the complete train action stream. It must reject a stale/future Q
version, changed directive/action, re-selection after an intervening update,
unlogged policy call, duplicate selection, or action installed before its
observation exists.

Directive introspection and permutation tests replace each valid directive in
turn with every other legal two-byte directive and each malformed length/mode
combination. Only the declared action consequence may change; policy-state and
table commitments remain fixed. Adjacent cue-pair directive digests are exact
matches at every phase. No test may infer correctness from a directive object
identity because only its two committed bytes exist.

## Exact online SARSA(0) event and update order

The learner starts with all twenty-eight little-endian `float64` Q cells and
all twenty-eight little-endian `int32` visit counters exactly zero. The initial
table version is `int32(0)`. Discount is `gamma=1.0`; step size is the fixed
constant `alpha=0.25`. There is no eligibility trace, replay buffer, fitted
sweep, backward episode pass, optimizer, target network, representation
learning, checkpoint choice, early stopping, or held-out update.

For each episode, canonical order is:

```text
observe0, issue_directive0, select0, validate0, step0,
resolve_successor0, resolve_zero0, observe1, issue_directive1, select1,
validate1, update0_from_latched_action1, append0, step1,
resolve_successor1, resolve_zero1, observe2, issue_directive2, select2,
validate2, update1_from_latched_action2, append1, step2,
resolve_terminal2, update2_terminal, append2, close_episode
```

At a nonterminal boundary, the next action is selected and capability-latched
before the preceding update. The update must use the same next action and the
Q table version that existed when that next action was selected. The latched
action is then executed without re-selection even though the preceding update
has changed one Q cell. At terminal, there is no next action or bootstrap.

For an authenticated transition with assigned update actions `u_t` and
`u_{t+1}`:

```text
canonical mode: u_t = executed_action_t
                u_{t+1} = latched_executed_action_{t+1}
nonterminal_target = 0.0 + Q_pre_update[s_{t+1}, u_{t+1}]
terminal_target = terminal_update_reward
Q_new[s_t, u_t] = Q_old[s_t, u_t]
                    + 0.25 * (target - Q_old[s_t, u_t])
```

Only one current Q cell may change per update. For the zero-based global update
ordinal `k`, `table_version_before=int32(k)` and
`table_version_after=int32(k+1)`; `UpdateKey` contains the before-version.
Exactly one visit counter increments from its pre-update value to that value
plus one, even if the Q-value bytes do not change. There are three updates per
train episode and 24,576 total. Nonterminal target code has no terminal scalar,
evaluator, reward-origin, maximum-over-actions, future table version, or raw
trajectory capability. Terminal target code has no successor or bootstrap
capability.

Every update binds the exact predecessor, executed action, assigned update
action, successor or terminal sentinel, selected next executed and assigned
action authorization, source/update reward, table version before and after,
and control-specific donor/origin/assignment identity. Pending identities are
single-use and clear fail-closed on rejection.

The updater receives a freshly copied slotted `SealedUpdate`, never a legal-row
or evaluator object. Its only fields are the exact current and optional next
three-byte policy states, executed and assigned current action, optional
executed and assigned latched next action, exact bare `float64` update reward,
exact `done`, `SelectionAuth` digest, and before-version. It has no source
reward, public observation, hidden evaluator cue, target formula, donor,
origin, split/regime/key, lazy object, environment, collector, or parent
reference. Reachability, alias, mutation, `__dict__`, closure, subclass, and
pickle/reconstruction attacks must prove those fields cannot cross the updater
boundary.

## Physical timing, reentrancy, and malformed attacks

The real train source and environment install operation-counting lazy successor
and reward objects. Every object records installation, attempted access,
permitted materialization, first/last stage, and digest. Selection and update
run under separate reentrancy guards. Canonical paths must exercise every lazy
boundary non-vacuously.

One complete canonical train arm has exactly 24,576 steps/selections/updates,
16,384 installed and permitted successor materializations, 24,576 installed
and permitted reward materializations, and 8,192 closes. Each materialization
has exactly one attempt and one permit. One frozen validation-plus-test arm has
2,048 episodes, 6,144 steps/selections, 4,096 successor materializations,
6,144 reward materializations, zero directives, zero updates, and 2,048
closes. Every fresh online control has the same operation cardinalities; each
canonical-action replay control has 24,576 sealed updates but zero policy or
environment operations during replay and authenticates the separately sealed
canonical 24,576-action input digest.

Each hostile timing test uses a fresh blank learner and one named malicious
spy installed through the real source/environment callback slot. The named
lazy or reentrancy boundary records exactly one installation, one attempted
access, zero permitted materializations, and one contract error; every
unrelated counter is zero and Q/table-version/visits remain at their initial
bytes. An exception raised by argument validation before the spy records its
attempt does not satisfy the case. Read-only policy-state/Q aliases, writable
aliases, malicious array subclasses, closure captures, and mutation before and
after selection/update are separate mandatory attacks.

Production has two separate hooks. The pure stateless projector signature is
exactly `project(public_observation_bytes: bytes) -> PolicyState bytes`; it has
no source/environment/lazy handle, counter, clock, global, or closure state.
The real environment alone owns an internal
`resolve_hook(stage:uint8, successor_lazy, reward_lazy, BoundaryProbe)` called
immediately before each permitted resolution. Canonical production installs a
no-op hook. A focused hostile test substitutes a malicious resolve hook at that
same internal slot; it first records `boundary_attempted=1`, then attempts the
named forbidden materialization or nested environment/selection/update call
before the environment's production stage guard runs. Hidden-metadata and
extra-argument attempts are projector/selector schema tests, not lazy timing
tests, and cannot count toward the physical attempt totals. No lazy object is
ever passed to the projector or production selector.

At minimum, focused tests must drive the real source/environment/policy/update
path through and reject:

1. successor or reward access during any selection;
2. hidden evaluator cue, target formula, evaluator, source identity, or control
   identity offered to selection (the declared public cue bit inside the sealed
   three-byte policy state is allowed);
3. invalid action successor/reward resolution;
4. successor resolution before action validation;
5. next-phase selection before the successor observation exists;
6. nonterminal terminal scalar or missing/duplicate zero feedback;
7. update before the next action is selected and latched;
8. re-selection of a latched next action after the preceding update;
9. stale, future, duplicate, or skipped Q table version;
10. bootstrap from an action other than the latched behavior action;
11. bootstrap from a post-update rather than pre-update table snapshot;
12. terminal scalar before phase-two action or after an extra transition;
13. duplicate terminal scalar, update, episode close, or directive use;
14. terminal-origin materialization before the terminal slot;
15. behavior assignment or origin metadata crossing the policy boundary;
16. reentrant environment step during selection;
17. reentrant selection during update or update during selection;
18. nonempty pending transition/action/directive/origin at split close; and
19. held-out source construction or materialization during fit;
20. raw-public-observation, nuisance-only, directive-only, or hidden-object
    alias reaching selection; and
21. source reward, evaluator cue, origin, donor, assignment parent, or raw row
    reaching `SealedUpdate`.

Every attack must traverse the boundary it names, raise the exact contract
error, leave the learner byte-identical, clear all pending capabilities, and
record exact attempted/permitted operation counts. A substitute protocol-stage
exception that cannot reach the named physical boundary is not admissible.

The exact `online_information_boundary` attack-class set is:

```text
early_successor_select, early_reward_select, hidden_metadata_select,
invalid_action_lazy, successor_before_validation,
next_select_before_successor, nonterminal_feedback_protocol,
update_before_latch, reselection_after_update, stale_or_skipped_version,
wrong_bootstrap_action, postupdate_bootstrap_snapshot, terminal_timing,
duplicate_terminal_update_close_directive, early_origin,
assignment_or_origin_policy_leak, reentrant_environment_step,
reentrant_select_or_update, pending_at_split_close, heldout_during_fit,
policy_or_update_alias
```

Its ordered registry partition is exact and disjoint. The selection/source-side
partition is:

```text
early_successor_select, early_reward_select, hidden_metadata_select,
invalid_action_lazy, successor_before_validation,
next_select_before_successor, reselection_after_update,
assignment_or_origin_policy_leak, reentrant_environment_step,
heldout_during_fit
```

The update/trace-side partition is:

```text
nonterminal_feedback_protocol, update_before_latch,
stale_or_skipped_version, wrong_bootstrap_action,
postupdate_bootstrap_snapshot, terminal_timing,
duplicate_terminal_update_close_directive, early_origin,
reentrant_select_or_update, pending_at_split_close,
policy_or_update_alias
```

Their concatenation in that order is the twenty-one-name attack-class set
above. Each named class is attempted exactly once from fresh state. Therefore
`selection_attack_classes == selection_attacks_rejected == 10` and
`update_attack_classes == update_attacks_rejected == 11`; overlap, omission,
an extra attempt, or an accepted attack fails the case.

The exact `pending_transition_authentication` mutation-class set is:

```text
wrong_episode, wrong_phase, wrong_observation_digest, wrong_directive_digest,
wrong_q_version, wrong_q_cell_digest, wrong_executed_action,
wrong_assigned_action, stale_nonce, duplicate_nonce, cross_episode,
skipped_version, duplicate_append
```

Every class is exercised once from fresh state plus one complete canonical
path; the case reports the exact class count and exact rejection count.

The keyed trace validator independently joins observation, directive,
selection, transition, feedback, update, and episode-close
components. It authenticates exact types, layouts, keys, links, values, source,
donor, origin, assignment, Boolean fields, and Q digests under separate fixed
nonidentity permutations for every component list. Missing, duplicate,
unknown, malformed, cross-episode, cross-regime, target-twin, wrong-layout,
wrong-action, wrong-successor, wrong-reward, wrong-version, wrong-directive,
wrong-origin, wrong-assignment, and independently swapped components must all
reject deterministically.

The exact keyed malformed-class set is `missing`, `duplicate`, `unknown`,
`wrong_numpy_width`, `python_bool_for_integer`, `numpy_bool_for_integer`,
`wrong_layout`, `writable_observation`, `array_subclass`, `cross_episode`,
`cross_regime`, `target_twin`, `wrong_action`, `wrong_successor`,
`wrong_source_reward`, `wrong_update_reward`, `wrong_done`, `wrong_version`,
`wrong_directive`, `wrong_origin`, `wrong_assignment`, and
`independently_swapped_component`. Each class must reject at least one mutation
of every record type to which it applies; nonapplicable record/class pairs are
committed before the validator runs and cannot count as rejections.

The realized-trace domain is separate from the 143,360 legal-row domain. Its
canonical train component lists and field order are frozen as follows:

- `ObservationRecord`: `ObservationKey`, public observation bytes/layout,
  three-byte policy-state digest; exactly 24,576;
- `DirectiveRecord`: `SelectionKey`, two directive bytes/digest; exactly
  24,576;
- `SelectionRecord`: `SelectionKey`, `ObservationKey`, invocation ordinal,
  policy-state/directive/table/cell digests, before-version, executed action,
  tie flag; exactly 24,576;
- `TransitionRecord`: `TransitionKey`, predecessor key/digest, executed action,
  successor key/digest or terminal sentinel, source reward digest, `done`,
  donor reference; exactly 24,576;
- `FeedbackRecord`: `FeedbackKey`, transition key, source-reward digest,
  update-reward digest, origin reference; exactly 24,576;
- `UpdateRecord`: `UpdateKey`, selection and feedback keys, current/next
  real-successor and target policy-state digests, executed/assigned current and
  next actions, target/old/new value digests, before/after versions,
  before/after visit counts, assignment reference, optional exact
  `TransitionTargetRef`; exactly 24,576; and
- `EpisodeCloseRecord`: `EpisodeKey`, final transition/update keys, terminal
  source-score digest, pending-empty flag; exactly 8,192.

Every list is canonically sorted by its typed keys before its event-trace
digest. Each list is then independently reordered by a fixed reversal followed
by a one-position left rotation; reversing without rotating, using one shared
permutation, or preserving canonical order fails the component-authentication
test. No runtime object, array, raw record, or list enters the sanitized result.

## Bootstrap-attribution proof

Every nonterminal update records a bounded commitment to the exact
pre-update next-state/action Q cell used by its SARSA target. An independent
equation replay must reproduce every target, old value, new value, visit count,
and table digest in chronological order from the blank table and authenticated
action stream.

There is no value-selected witness. The proof shadows all 16,384 canonical
nonterminal updates, one at a time, from each update's authenticated frozen
pre-update snapshot. For each, a pure calculator substitutes exact zero only
for the selected next-cell scalar and recomputes only that one target and
current-cell update. It must preserve every other snapshot cell, selection,
key, reward, visit, and trace byte. The complete shadow set must contain at
least one changed target and changed current-cell value in each of the four
predeclared categories `(cue_bin 0 or 1) x (phase 0 or 1)`; unchanged shadows
are retained and counted rather than discarded. This covers both cue chains
and both bootstrap boundaries without selecting a favorable learned value.
The shadow projection cannot enter action selection, canonical learning,
scoring, comparators, or controls and reads no terminal scalar directly at a
nonterminal boundary.

The no-bootstrap comparator below and mutations of next action, next state,
table version, Q-cell digest, or terminal capability complete the attribution
case. Aggregate policy success without these exact dependencies is
insufficient.

## Train-only source and held-out freeze

The orchestration function accepts a train source plus optional validation and
test handles, but the learner API accepts only the authenticated online train
environment and behavior-policy capabilities. Every source counts factory,
iterator, row materialization, environment step, close, post-close, selection,
and update operations.

The identical fit path must run in three fresh constructions:

1. validation and test handles absent;
2. validation and test handles installed as exploding sources; and
3. validation and test handles lazy and unopened until the final train state
   seals.

All three produce byte-identical train action-stream, update-trace, and final
learner commitments. Exploding held-out operations remain zero. The lazy
handles are the real sources later used by the same orchestration path for
held-out evaluation.

An inverse check installs an exploding train source after fit and executes the
real validation/test path. Train construction, materialization, selection,
fitting, and update counts remain zero while each held-out episode is evaluated
exactly once. Final learner bytes and table version are hashed before
validation, after validation, and after test and remain identical. Validation
and test cannot affect normalization, abstraction, exploration, alpha, gamma,
tie behavior, thresholds, stopping, checkpoint selection, or any policy state.

## Frozen metrics and complete positive gate

Primary performance is terminal return averaged within each complete regime
and then macro-averaged with equal regime weight. Minimum held-out regime return
is retained. Train behavior regret is episode count minus terminal reward sum.
All policies are scored by an independently implemented keyed evaluator over
identical complete canonical episodes.

For a regime with `n` episodes and `w` terminal successes, return is the exact
rational pair `(w,n)`; a split macro is the unweighted arithmetic mean of its
regime rationals. Because every regime in a split has equal size, the
implementation may reduce the summed-success ratio but must independently
replay the per-regime values. Gain is learned return minus comparator return.
`canonical_test_gap` is exactly canonical test macro return minus the named
control's test macro return; it is never a minimum-regime or train difference.
Every threshold is compared by integer cross-multiplication:
`0.95=19/20`, `0.90=9/10`, `0.80=4/5`, `0.50=1/2`, and `0.45=9/20`.
Floating-point rounding never decides a gate.

Online acquisition is measured without an extra probe or checkpoint. The first
sixteen greedy train episodes must have exact return zero, leave all twenty-
eight Q-value bytes zero, advance the table version to 48, and produce the
exact independently replayed forty-eight visit increments. The final eight
rounds contain exactly 512 greedy episodes,
128 per train regime; their greedy behavior return is computed directly from
the canonical train stream. Only the final table after all 8,192 episodes is
used for held-out evaluation.

The complete positive gate requires:

- all family, target-twin, type, schedule, source, timing, reentrancy, pending,
  trace, behavior-replay, update-equation, bootstrap-attribution,
  control-difference, sanitizer, and process-isolation invariants passing;
- exactly 8,192 train episodes, 24,576 selections, 24,576 updates, 4,096 greedy
  episodes, 4,096 forced episodes, and 512 forced-exploration successes;
- first-sixteen greedy return exactly `0.0`, unchanged all-zero Q-value bytes,
  table version exactly `48`, and exact replayed visit-counter bytes;
- final-eight-round greedy train macro return at least `0.95` and minimum
  train-regime greedy return at least `0.90`;
- final validation and test macro return at least `0.95` and minimum validation
  and test regime return at least `0.90`;
- validation and test gain over the better constant at least `0.80`;
- validation and test gain over feedback-only myopic and no-bootstrap at least
  `0.80`;
- validation and test gain over seeded random at least `0.50`;
- exactly zero held-out updates and byte-identical frozen policy state before,
  between, and after held-out regimes; and
- no early stopping, checkpoint selection, retry, seed swap, threshold change,
  or case removal.

Every negative control starts from a fresh blank learner and uses the identical
train episode ordering, update count, hyperparameters, metric formula,
thresholds, and untouched held-out episode identities. The transition-target
control is a fresh online run and its frozen policy is scored on canonical
held-out public states and rewards. Behavior-assignment and terminal-origin
are explicit canonical-action replay attribution controls: they replay the
sealed canonical train action/transition stream without policy or environment
calls, then their frozen tables are scored on canonical held-out states and
rewards. They do not claim to be second online agents. Signal ablation is a
fresh online run on copied ablated train states and is scored on copied ablated
held-out states with the unchanged canonical evaluator reward formula; the
canonical true policy is separately scored on the same ablated held-out states.
These are the only control domains, and every result reports its domain tag.

Each control is otherwise evaluated against every applicable member of the
complete positive gate and must reject it. A replay control substitutes its
exact replay-operation contract for online-selection counts but must meet all
family, source, sealed-update, trace, scoring, freeze, and process checks.
Transition-target-cue-swap validation and test macro returns must be exact
`1/2`, with canonical-minus-control test gaps at least `9/20` whenever the
positive gate passes. Behavior-assignment and terminal-origin validation and
test returns must be exact zero, with gaps at least `19/20`. Both signal-
ablation held-out scores must be at most `1/2`, with gaps at least `9/20`.

## Frozen comparators

Comparators receive only authenticated public behavior feedback or public
evaluation observations. None receives cue formula, desired code, evaluator
formula, counterfactual reward, target twin, hidden key, schedule index, control
mode, donor, origin, or the learner's trace object.

- `constant_zero`: emits `000`; exact return is `0.0` in every regime.
- `constant_one`: emits `111`; exact return is `0.0` in every regime.
- `feedback_only_myopic`: over the sealed canonical 24,576-record behavior
  stream, a separate implementation accumulates little-endian `int32` count
  and `float64` immediate-source-reward sum for each of the same 28 public
  state/action cells. Its fitted value is sum/count when count is positive and
  exact zero otherwise; evaluation takes the larger value with action-zero tie
  rule. It consumes records in canonical key order, performs no bootstrap or
  online selection, and has no access to update rewards. Phase zero ties to
  action zero, so its exact held-out return is `0/512` per regime.
- `no_bootstrap`: a fresh online learner uses the exact canonical directive
  schedule, alpha, action selection, update ordering, and source reward, but
  every nonterminal target is exact zero. Thus phase-zero Q bytes remain zero,
  held-out phase zero selects action zero, and the deterministic continuation
  cannot equal either `010` or `101`; its exact held-out return is `0/512` per
  regime. Its complete action stream and table are independently replayed.
- `seeded_random`: a fresh
  `numpy.random.Generator(numpy.random.PCG64(271828182))` makes exactly one call
  per episode,
  `rng.integers(0, 2, size=(3,), dtype=numpy.int8)`, in canonical train,
  validation, then test order. Scalar calls, three separate calls, default
  `int64`, extra draws, or another generator are forbidden. NumPy is pinned by
  `uv.lock`; the complete stream digest is committed during the later
  pre-result implementation checkpoint. Its score is not used to tune a gate.

Every comparator action stream is generated twice from fresh state and must be
byte-identical, then independently rescored. The fitted comparators use a
separate feedback materializer and share no SARSA projection helper. Aggregate
metric agreement without exact action-stream agreement is insufficient.

## Transition-target cue-swap negative control

This is a genuine wrong-successor-target control and a fresh online run; it is
not the no-bootstrap comparator. Before a policy or reward exists, its
immutable `TransitionTargetSpec` maps every real nonterminal successor policy
state `(cue_bin, next_phase, next_latch)` to the separately committed target
state `(1-cue_bin, next_phase, next_latch)`. The latched next executed action is
unchanged. The real environment successor remains canonical and is still
constructed, selected from, executed, logged, and authenticated. Only the
three-byte successor state used for the SARSA Q lookup and its exact typed
`TransitionTargetRef` change; terminal target/reward, source reward, action,
prefix, schedule, and evaluator formula remain canonical. The mapping is an
outcome-blind no-fixed-point involution over all legal nonterminal target
states, constructed and independently replayed before any learner factory.

The control runs its own policy online under the canonical directive schedule.
Its later selected actions and realized prefixes may diverge causally from the
canonical arm, but every one is independently authenticated against the real
control environment and its own pre-action table. Independent equation replay
must use the cue-swapped target state on all 16,384 nonterminal updates and the
real terminal scalar on all 8,192 terminal updates; a zero target, real
successor target, maximum, donor action, or second cue swap rejects.

Algebraically, terminal Q values remain attached to the real cue while both
nonterminal bootstrap boundaries read the opposite cue at the same prefix.
Across a balanced held-out cue pair, the resulting deterministic frozen policy
satisfies exactly one of the two evaluator targets. The implementation must
independently replay every update and held-out action to prove exact
`256/512=1/2` return per regime; this contract does not predeclare an action
tuple and cannot be satisfied by substituting an assumed pattern. The control
must reject the full positive gate and preserve a canonical test gap of at
least `9/20`.

## Behavior-assignment negative control

This is an explicit canonical-action replay attribution control. It consumes
the already sealed and independently authenticated canonical 24,576-record
train action/transition/feedback stream; it does not call a policy, source, or
environment during replay. Observation, directive, selected and executed
environment action, successor, source reward, terminal origin, `done`, event
order, and canonical evaluator commitments are byte-identical inputs. It
changes only the actions assigned to its fresh learner's update projection:

```text
assigned_current_action = int8(0)
assigned_next_action = int8(0) at nonterminal boundaries
```

The mapping is fixed before canonical learning and independent of cue, reward,
success, Q values, action, or outcome. Exact typed `ActionAssignmentRef`
records bind the source executed and assigned actions evaluator-side. The
`SealedUpdate` receives only bare assigned `int8` actions; assignment identity
cannot cross it. `AssignmentKey` changes as declared while source,
`SelectionKey`, `FeedbackKey`, and the before-version portion of `UpdateKey`
remain fixed.

Only action-zero Q cells can become positive; action-one cells stay zero.
With the frozen action-zero tie rule, its held-out policy emits exactly `000`
and has exact `0/512` return in every regime. It must reproduce those action
and score commitments independently, reject the full positive gate, and
preserve a canonical test gap of at least `19/20`.

## Terminal-origin-zero negative control

This is a second canonical-action replay attribution control. Its immutable
`TerminalOriginSpec` is constructed before source iteration and maps every
terminal `FeedbackKey` to a detached `ZeroOriginRef` and exact bare
`float64(0.0)` update reward. Canonical public observations, directives,
selected/executed actions, successors, source evaluator cue/reward, `done`,
and event order remain byte-identical replay inputs. The source reward remains
separately authenticated and is used only for canonical behavior scoring.

Across the 8,192 terminal update records, every realized cue/state/action cell
has its exact nonnegative count, scalar sum zero, and, when nonempty, mean zero;
the cell counts sum to 8,192. The control proves those counts from the sealed
action stream before constructing its control learner and then independently
replays them. The scalar materializes exactly
at the terminal updater boundary and has zero early attempts. All Q-value bytes
remain zero through 24,576 updates while versions and visits advance exactly.
Its held-out policy emits exactly `000`, scores `0/512` in every regime,
rejects the full positive gate, and preserves a canonical test gap of at least
`19/20`.

This is deliberately a terminal-signal origin ablation. It does not test
one-to-one origin permutation, reward-multiset preservation, or wrong-origin
recombination, and no pass claim may use those stronger descriptions.

## Complete signal-attribution control

This control copies every legal train, validation, and test public observation
and changes only `signed_cue` to exact `float64(0.0)`; it never regenerates a
row. The corresponding detached `PolicyState.cue_bin` becomes zero. Phase,
latch, remaining count, unsigned magnitude, nuisance, source actions, keys,
links, directives, hidden evaluator cue/formula, source/update reward formulas,
donor/origin/assignment identities, `done`, and ordering remain fixed in the
static source projection.

A fresh online learner selects its own actions on the ablated train source, so
its realized action, prefix, reward, Q, target, and trace fields may differ
causally from canonical and are authenticated against the ablated source. It
is evaluated on ablated held-out policy states with the unchanged canonical
hidden evaluator. Because adjacent held-out cue pairs have identical unsigned
magnitude/nuisance and collapse to the same three-byte policy-state path, any
deterministic frozen policy emits the same three-action code for both cues and
can satisfy at most one; return is at most `1/2` in every regime. The canonical
true policy is separately evaluated on the same ablated held-out rows and has
the same ceiling. Directives are absent at held-out evaluation; no separate
directive-only performance score is claimed.

The static all-row ablation covers canonical, target-cue-swap, assignment, and
origin-zero source projections. Both fresh-learner and true-policy ablated
validation/test returns must be at most `1/2`, reject the full positive gate,
and preserve a canonical test gap of at least `9/20`.

## Complete control-difference whitelists

Static intervention-source comparisons and dynamic execution comparisons are
separate domains; no plan clause requires causally downstream runtime traces to
remain byte-identical.

The static legal-family/source comparison covers all 143,360 rows plus the
complete schedule and control-spec projections before any learner factory is
built. Its only permitted differences are:

- evaluator twin: detached hidden evaluator cue and formula-authorized twin
  terminal reward; every public/source byte remains fixed;
- transition-target cue swap: nonterminal target successor policy-state bytes/
  digest and exact `TransitionTargetRef` only; the complete legal environment
  family and real successor are byte-identical;
- behavior assignment: `ActionAssignmentSpec.mode` only before replay; the
  sealed canonical family/action input is byte-identical;
- terminal origin zero: terminal update-reward scalar/digest and exact
  `ZeroOriginRef`; canonical source reward is byte-identical; and
- signal ablation: public signed-cue bytes/digest, successor signed-cue bytes/
  digest, and derived three-byte policy-state digest only.

Every other static field is byte-identical. Each static whitelist has a
mandatory one-field mutation attack for every protected category, including
formula, regime, schedule, hidden evaluator, public nuisance, action,
successor, source reward, key/link, dtype/layout, Boolean, directive, and
control identity.

The exact ordered static control names are:

```text
transition_target_cue_swap, behavior_assignment, terminal_origin_zero,
signal_attribution
```

The exact ordered one-field mutation-attack names are:

```text
formula_mutation, regime_mutation, schedule_mutation,
hidden_evaluator_mutation, public_nuisance_mutation, action_mutation,
successor_mutation, source_reward_mutation, key_or_link_mutation,
dtype_or_layout_mutation, boolean_mutation, directive_mutation,
control_identity_mutation
```

Each mutation attack is run exactly once from fresh source bytes against every
static control whitelist and must be rejected. The case scalars are therefore
`static_controls == 4`, `static_mutation_classes == 13`, and
`static_mutations_rejected == 52`; omission, duplication, acceptance, or an
attack that changes more than its named field class fails the case.

Dynamic traces are independently reconstructed and authenticated within their
own arm. The declared causal graph is:

```text
intervention -> sealed update input -> Q/visit bytes -> later selection
 -> executed action/prefix/source reward -> later sealed update input
```

Transition-cue-swap and signal-ablation runs may therefore differ from canonical
in policy-state (ablation only), Q/value/target/visit commitments, selection
actions and ties, realized latch/successor/source-reward bytes, and downstream
trace digests. Assignment and origin replay runs preserve the canonical input
selection/environment/source-feedback projection but may differ in assigned
actions or update reward respectively, followed by targets, Q/visit bytes, and
evaluation action streams. Episode identities, order, directive bytes, update
count, before/after version sequence, public source formulas, hidden evaluator
formula, and control-unrelated source fields never differ. A field difference
not reachable from the arm's single static intervention, or a protected field
mutation hidden behind a changed digest, fails. Reduced digests, aggregate-only
comparisons, shared canonical/control constructors, or a declared intervention
that never changes its own source field also fail before result acceptance.

The exact ordered dynamic arm names are the four static control names above.
The exact ordered causal field-class names are:

```text
policy_state_or_target_input, assigned_action_or_update_reward,
q_target_or_visit, later_selection, realized_environment,
later_update_input, evaluation_action_stream
```

The exact arm-to-class map is:

```text
transition_target_cue_swap:
  policy_state_or_target_input, q_target_or_visit, later_selection,
  realized_environment, later_update_input, evaluation_action_stream
behavior_assignment:
  assigned_action_or_update_reward, q_target_or_visit,
  evaluation_action_stream
terminal_origin_zero:
  assigned_action_or_update_reward, q_target_or_visit,
  evaluation_action_stream
signal_attribution:
  policy_state_or_target_input, q_target_or_visit, later_selection,
  realized_environment, later_update_input, evaluation_action_stream
```

Thus `dynamic_arms == 4`, `causal_field_classes == 7`, and
`noncausal_differences == 0`. The independent comparison must encounter every
listed arm/class pair at least once and reject any difference outside that
arm's exact list; an unexercised declared class fails rather than silently
shrinking the map.

## Sanitized result and process isolation

The later controller result must have exactly:

```text
action, cases, environment, fixture, schema_version, status, study_id
```

`schema_version` and registry `result_schema_version` are exactly integer `1`;
`study_id` and worker mode are exactly `online-sarsa-latched-choice-v1`.
`status` is exactly `passed` if and only if every case passes and exactly
`failed` if and only if at least one case fails. `action` is exactly the frozen
success action if and only if status is `passed`, and exactly the frozen park
action if and only if status is `failed`; every other status/action/case
combination is malformed and fails closed. Environment has exactly `device_kind`,
`jax_version`, `platform`, and `python`; device kind and platform are exactly
`cpu`, and version strings are nonempty, bounded, and sanitized.

The later registry wiring is frozen now:

```text
worker_module = experiments.local_lab.online_sarsa_latched_choice_v1_worker
source_paths and approved-hash keys:
dependency_lock = uv.lock
fixture_source = experiments/local_lab/online_sarsa_latched_choice_v1.py
lab_protocol = docs/AUTONOMOUS_LAB.md
study_plan = research/2026-08-29-online-sarsa-latched-choice-v1-plan.md
worker_source = experiments/local_lab/online_sarsa_latched_choice_v1_worker.py
```

`worker_module` is the separate top-level registry string and is never a
`source_paths` or `approved_file_sha256` key. Those two mappings each contain
exactly the five file keys listed after the label.

No sixth source path or helper module is allowed. Only that worker module/path
may be added to the controller allowlist. The registry identity must contain
exactly these keys: `action_dtype`, `action_values`, `alpha`,
`claim_boundary`, `directive_dtype`, `directive_width`, `episode_counts`,
`event_order`, `expected_family_sha256`, `expected_random_stream_sha256`,
`expected_schedule_sha256`, `gamma`, `generator_regimes`, `horizon`,
`observation_dtype`, `observation_fields`, `observation_shape`,
`policy_state_dtype`, `policy_state_fields`, `policy_state_width`,
`random_baseline_seed`, `regime_counts`, `reward_dtype`, `reward_values`,
`structure_kind`, `thresholds`, `tie_action`, `train_rounds`, and
`updates_per_episode`. Their values are exactly the values/formulas in this
plan; the three expected digests are derived from the frozen encodings and must
be committed during the implementation checkpoint before any learner runs.

The study's top-level registry entry has exactly these keys and no others:

```text
approved_file_sha256, case_contract, case_required_fields, failure_action,
fixture_identity, plan_path, result_schema_version, source_paths,
success_action, worker_mode, worker_module
```

`plan_path`, worker mode/module, result schema, and actions are the exact values
frozen in this plan. `source_paths` is the exact five-key mapping above.
`approved_file_sha256` has the same five keys, each mapped to the lowercase
64-hex digest of committed `git show HEAD:<source_path>` bytes. `fixture_identity`
is the exact object below; `case_contract` and `case_required_fields` have
exactly the twenty-three case keys and schemas below. Missing/extra registry
keys, mismatched map keys, working-tree rather than committed hashes, or a
controller registry digest not recomputed over normalized LF bytes rejects
before a worker starts.

Exact non-derived identity values are:

```text
action_dtype = int8
action_values = [0, 1]
alpha = 0.25
claim_boundary = synthetic_cpu_online_sarsa_latched_choice_harness_only
directive_dtype = bytes
directive_width = 2
episode_counts = {train:8192, validation:1024, test:1024}
gamma = 1.0
horizon = 3
observation_dtype = float64
observation_fields = [phase, signed_cue, latch_code, remaining_actions,
                      nuisance]
observation_shape = [5]
policy_state_dtype = bytes
policy_state_fields = [cue_bin, phase, latch_code]
policy_state_width = 3
random_baseline_seed = 271828182
regime_counts = {train:4, validation:2, test:2}
reward_dtype = float64
reward_values = [0.0, 1.0]
structure_kind = none
tie_action = 0
train_rounds = 64
updates_per_episode = 3
thresholds = {
  minimum_final_window_macro_return:0.95,
  minimum_final_window_regime_return:0.90,
  minimum_validation_macro_return:0.95,
  minimum_test_macro_return:0.95,
  minimum_heldout_regime_return:0.90,
  minimum_constant_gain:0.80,
  minimum_myopic_gain:0.80,
  minimum_no_bootstrap_gain:0.80,
  minimum_random_gain:0.50,
  transition_target_exact_return:0.50,
  behavior_assignment_maximum_return:0.0,
  terminal_origin_maximum_return:0.0,
  signal_attribution_maximum_return:0.50,
  zero_control_minimum_test_gap:0.95,
  transition_control_minimum_test_gap:0.45,
  signal_control_minimum_test_gap:0.45
}
```

`generator_regimes` is the eight-row table above serialized as objects with
exact keys `split`, `code`, and `signal_scale` in table order. `event_order` is
the exact twenty-seven-string sequence in the online-order code block, without
abbreviation or inserted events. `expected_family_sha256`,
`expected_schedule_sha256`, and `expected_random_stream_sha256` are the only
derived identity values.

The result `fixture` object contains exactly those twenty-nine identity keys
plus `case_contract`, and no others. The three expected digests are lowercase
64-hex strings. `action_dtype`, `claim_boundary`, `directive_dtype`,
`observation_dtype`, `policy_state_dtype`, `reward_dtype`, and `structure_kind`
are bounded strings with the exact values above. `alpha` and `gamma` are finite
JSON numbers; horizon, directive/policy widths, seed, tie action, train rounds,
and updates per episode are JSON integers. Action/reward values, event order,
observation/policy fields, and observation shape are exact ordered JSON lists
of lengths 2, 2, 27, 5, 3, and 1 respectively. `episode_counts` and
`regime_counts` are exact three-key integer objects; `generator_regimes` is an
exact eight-object ordered list whose objects each have only `split`, `code`,
and `signal_scale`; `thresholds` is the exact sixteen-key finite-number object
above; and `case_contract` has exactly the shape specified next. The controller
must validate all container lengths, child types, key sets, and ordered values
in both directions.

The registry `case_contract` has exactly the twenty-three case names below.
Each value contains `contract_version:1` and no unspecified key. The exact
exceptions are:

- `online_information_boundary` additionally has `attack_class_names` equal to
  the ordered twenty-one-name set above, `selection_attack_class_names` equal
  to the ordered ten-name selection/source-side partition, and
  `update_attack_class_names` equal to the ordered eleven-name
  update/trace-side partition;
- `pending_transition_authentication` additionally has
  `mutation_class_names` equal to the ordered thirteen-name set above;
- `keyed_trace_authentication` additionally has `malformed_class_names` equal
  to the ordered twenty-two-name set above;
- `transition_target_cue_swap_control` additionally has
  `domain:fresh_online_canonical_heldout`;
- `behavior_assignment_control` and `terminal_origin_zero_control` each
  additionally have `domain:canonical_action_replay_canonical_heldout`;
- `signal_attribution_control` additionally has
  `domain:fresh_online_ablated_heldout`; and
- `control_difference_whitelists` additionally has `static_control_names`,
  `static_mutation_class_names`, `dynamic_arm_names`, and
  `causal_field_class_names` equal to the exact ordered sets above plus
  `causal_field_classes_by_arm` equal to the exact four-key ordered-list map
  above; and
- `process_isolation` additionally has `workers:2`.

All other case-contract values are exactly `{"contract_version":1}`.

Every case result is a flat dictionary of scalar values with exactly the
following field names; `passed` is included in every set:

```text
typed_episodic_contract:
  passed, structure_kind, horizon, observation_dtype, observation_rank,
  observation_width, policy_state_width, action_dtype, action_cardinality,
  reward_dtype, reward_cardinality, done_pattern_sha256, event_order_sha256,
  typed_keys_exact, immutable_inputs, invalid_actions_rejected
generator_partition:
  passed, train_regimes, validation_regimes, test_regimes, train_episodes,
  validation_episodes, test_episodes, train_rounds,
  episodes_per_round_regime, paired_nuisance_exact, generator_rng_calls,
  family_sha256, schedule_sha256
complete_family_replay:
  passed, legal_rows, nonterminal_rows, terminal_rows, predecessor_nodes,
  canonical_realized_rows, unique_keys, primary_sha256, replay_sha256,
  replay_exact, corruption_classes, corruptions_rejected,
  factories_before_gate
evaluator_twin_public_invariance:
  passed, episodes, public_rows_checked, evaluator_flips,
  terminal_reward_exchanges, public_bytes_preserved, detached_no_alias,
  self_compare_rejected, copy_only_rejected, twin_sha256
realized_path_disjointness:
  passed, train_paths, validation_paths, test_paths,
  train_validation_overlaps, train_test_overlaps, validation_test_overlaps,
  identity_fields_excluded, public_signal_retained, projection_sha256
exploration_schedule_commitment:
  passed, train_episodes, selections, greedy_episodes, forced_episodes,
  greedy_selections, forced_selections, forced_successes,
  cue_paired_directives, tuple_balance_exact, primary_sha256, replay_sha256,
  replay_exact
online_information_boundary:
  passed, policy_state_bytes, directive_bytes, selection_attack_classes,
  selection_attacks_rejected, update_attack_classes,
  update_attacks_rejected, lazy_attempts, lazy_permits,
  reentrancy_attempts, reentrancy_rejected, alias_attacks_rejected,
  learner_state_unchanged, boundary_sha256
online_sarsa_update_order:
  passed, alpha, gamma, selections, updates, final_table_version,
  visit_increments, exact_equation_updates, latched_actions_exact,
  preupdate_bootstraps_exact, invalid_updates_rejected, q_dtype_exact,
  visit_dtype_exact, update_trace_sha256
behavior_action_replay:
  passed, selections, replayed_actions, preaction_tables_replayed,
  action_stream_exact, stale_versions_rejected, directive_mutations_rejected,
  duplicate_selections_rejected, action_stream_sha256, replay_sha256
pending_transition_authentication:
  passed, authorization_classes, authorization_attacks_rejected,
  wrong_identity_rejected, wrong_assignment_rejected,
  duplicate_append_rejected, cross_episode_rejected,
  stale_authorization_rejected, pending_cleared_after_rejection,
  projection_sha256
keyed_trace_authentication:
  passed, component_lists, component_records, reorderings,
  malformed_classes, malformed_rejected, swapped_components_rejected,
  canonical_trace_sha256, score_projection_sha256,
  update_projection_sha256
bootstrap_attribution:
  passed, nonterminal_updates, shadow_updates, cue_boundary_categories,
  categories_with_changed_targets, offcell_bytes_preserved,
  terminal_reads_at_nonterminal, wrong_cell_mutations_rejected,
  shadow_projection_sha256
train_only_source_boundary:
  passed, fit_constructions, absent_heldout_passed,
  exploding_heldout_operations, lazy_heldout_operations_during_fit,
  train_operations_inverse, heldout_updates, train_commitments_exact,
  sealed_fit_sha256
online_acquisition:
  passed, train_episodes, behavior_reward_sum, behavior_regret,
  first_window_successes, first_window_table_version,
  first_window_q_zero, final_window_macro_return,
  minimum_final_window_regime_return, final_table_sha256
heldout_policy_freeze:
  passed, validation_macro_return, test_macro_return,
  minimum_validation_regime_return, minimum_test_regime_return,
  validation_episodes, test_episodes, heldout_updates,
  policy_state_unchanged, policy_sha256
baseline_replay:
  passed, constant_zero_validation_return, constant_zero_test_return,
  constant_one_validation_return, constant_one_test_return,
  myopic_validation_return, myopic_test_return,
  no_bootstrap_validation_return, no_bootstrap_test_return,
  random_validation_return, random_test_return, random_draws,
  action_streams_exact, replay_exact, random_stream_sha256
transition_target_cue_swap_control:
  passed, train_updates, validation_return, test_return,
  minimum_heldout_regime_return, canonical_test_gap,
  positive_gate_rejected, target_states_swapped,
  target_replay_exact, real_successors_preserved,
  source_family_preserved, intervention_sha256, runtime_trace_sha256
behavior_assignment_control:
  passed, replay_updates, policy_calls_during_replay,
  environment_steps_during_replay, validation_return, test_return,
  minimum_heldout_regime_return, canonical_test_gap,
  positive_gate_rejected, action_zero_cells_only,
  source_trace_preserved, intervention_sha256, runtime_trace_sha256
terminal_origin_zero_control:
  passed, replay_updates, terminal_origin_records, terminal_update_sum,
  zero_early_origin_attempts, validation_return, test_return,
  minimum_heldout_regime_return, canonical_test_gap,
  positive_gate_rejected, q_values_zero, source_rewards_preserved,
  intervention_sha256, runtime_trace_sha256
signal_attribution_control:
  passed, legal_rows_checked, fresh_validation_return, fresh_test_return,
  true_policy_validation_return, true_policy_test_return,
  canonical_test_gap, positive_gate_rejected, hidden_evaluator_preserved,
  intervention_sha256, runtime_trace_sha256
control_difference_whitelists:
  passed, static_rows_checked, static_controls, static_mutation_classes,
  static_mutations_rejected, dynamic_arms, causal_field_classes,
  noncausal_differences, canonical_static_sha256,
  control_static_sha256, control_runtime_sha256
sanitized_result_contract:
  passed, case_count, exact_case_fields, scalar_case_values,
  forbidden_key_samples, forbidden_keys_rejected,
  forbidden_value_samples, forbidden_values_rejected,
  frozen_containers_exact, schema_sha256
process_isolation:
  passed, workers, nonprocess_projection_exact, projection_sha256
```

Case scalar types are closed by this rule. A field ending `_sha256` is an exact
lowercase 64-hex string. `structure_kind`, `observation_dtype`, `action_dtype`,
and `reward_dtype` are exact strings frozen above. `alpha`, `gamma`, every field
ending `_return` or `_gap`, and `terminal_update_sum` are exact JSON floats
(Booleans and integers reject): alpha/gamma and returns lie in `[0.0,1.0]`,
gaps in `[-1.0,1.0]`, and terminal update sum is `0.0`. Exact JSON Booleans are
`passed`, every field ending `_exact`, `_rejected`, `_preserved`, or
`_unchanged`, plus `immutable_inputs`, `public_signal_retained`,
`detached_no_alias`, `identity_fields_excluded`,
`pending_cleared_after_rejection`, `absent_heldout_passed`,
`first_window_q_zero`, `q_values_zero`, `action_zero_cells_only`, and
`scalar_case_values`. Every remaining case field is an exact JSON integer in
`[0,2147483647]`; `bool` never satisfies an integer field. No case value is a
list or dictionary. The only allowed result containers are the top-level
object, `cases`, `environment`, `fixture`, registry-derived `case_contract`,
and the exact frozen fixture-identity lists/dictionaries. The controller must
fail closed on every other container or missing/extra case field.

Permitted case values are scalar counts, finite aggregate metrics, Booleans,
version strings, frozen contract tags, and lowercase SHA-256 commitments. The
fixture and controller sanitizers must reject any raw or row-level observation,
state, directive, token, action, reward, return, target, successor, transition,
trajectory, log, Q table, policy state, parameter, gradient, topology, path,
credential, secret, donor array, origin array, assignment array, or private
evidence, including plural, nested, aliased, misspelled, and scalar-for-
container forms. Exact aggregate keys listed above are the only exceptions to
the broad raw-word rejection.

The implementation checkpoint must add an ID-specific strict validator in both
fixture and controller; the current generic/non-strict controller path is not
sufficient and cannot authorize this study unchanged. Validation uses exact
`type(value) is bool/int/float/str/list/dict`, rejects duplicate JSON keys,
nonfinite numbers, bool/int substitution, integer/float substitution, missing
or extra keys, wrong bounds, and every non-whitelisted container. Environment
requires both `device_kind == "cpu"` and `platform == "cpu"`; version strings
must match `[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}`. Strings are at most 512 bytes,
nested depth at most eight, and the worker's single JSON object is at most
1 MiB. Focused tests must mutate every case field through missing, extra,
wrong-scalar-type, lower-bound, upper-bound, list-for-scalar, dict-for-scalar,
Boolean-for-integer, and nonfinite forms; every fixture/container key through
missing, extra, reordered-list-value, and wrong-child-type forms; every
forbidden raw/path alias through singular/plural/case/separator variants; and
both CPU fields plus every hash/version field. The controller may be allowlisted
for this ID only after all strict mutations fail closed.

Two fresh credential-scrubbed, network-disabled worker processes must match the
complete bounded non-process projection byte-for-byte. Their imports must occur
only after network denial. No private path, vector, raw trace, or unbounded
diagnostic may appear in stdout, stderr, a result, or a committed summary.

The controller continues to launch one outer dedicated worker with
`python -m experiments.local_lab.online_sarsa_latched_choice_v1_worker --mode
online-sarsa-latched-choice-v1`. That worker sets the frozen CPU/thread
environment, monkeypatches socket construction/resolution/connect/send, and
only then imports the fixture. Its normal mode calls
`run_study(include_process_isolation=True)`. The fixture sequentially launches
exactly two fresh subprocesses of the same module with mode
`online-sarsa-latched-choice-v1-trace`, the same credential-scrubbed CPU
environment, new process groups, 1,200-second timeout apiece, and no shell.
Trace mode disables network before importing the fixture, calls only
`isolated_worker_trace()`, prints one sorted compact JSON object no larger than
1 MiB, requires empty stderr and exit code zero, and cannot recursively start
process-isolation workers. The parent parses both objects with duplicate-key
rejection, validates each through the strict study-specific sanitizer, and
requires their complete bounded non-process projections byte-identical before
adding the scalar `process_isolation` case. Any timeout, extra stdout JSON,
stderr byte, import-before-denial, projection mismatch, or child process left
alive fails the terminal study.

## Complete frozen case set

The future registry and result case set is exactly:

1. `typed_episodic_contract`
2. `generator_partition`
3. `complete_family_replay`
4. `evaluator_twin_public_invariance`
5. `realized_path_disjointness`
6. `exploration_schedule_commitment`
7. `online_information_boundary`
8. `online_sarsa_update_order`
9. `behavior_action_replay`
10. `pending_transition_authentication`
11. `keyed_trace_authentication`
12. `bootstrap_attribution`
13. `train_only_source_boundary`
14. `online_acquisition`
15. `heldout_policy_freeze`
16. `baseline_replay`
17. `transition_target_cue_swap_control`
18. `behavior_assignment_control`
19. `terminal_origin_zero_control`
20. `signal_attribution_control`
21. `control_difference_whitelists`
22. `sanitized_result_contract`
23. `process_isolation`

No case may be removed, merged, renamed, weakened, or substituted after the
plan commit. The later registry must declare required fields sufficient to
authenticate every count, metric, invariant, digest, attack class, source
operation, update-order field, control gate, and process comparison stated in
this plan.

## Stopping rule and actions

This plan commit is the entire present checkpoint. It authorizes no fixture,
worker, registry, controller, learner, or result change. A later heartbeat may
implement only this frozen contract under a fresh experiment module and
dedicated network-disabled worker in `experiments/local_lab`. That
implementation must receive independent hostile family/update-order,
exploration/leakage/control, and repository/registry audits and reach a clean
pre-result commit before any controller invocation.

The implementation checkpoint may run focused development tests but may not
invoke `tools/run_local_lab.py`. It must add the exact 23-case registry contract,
committed source hashes, a dedicated worker allowlist entry, normalized pinned
registry digest, sanitizer coverage, and focused tests without changing a
frozen regime, formula, schedule, seed, hyperparameter, threshold, case,
control, stopping rule, or claim. Any substantive pre-result confound
quarantines this ID and requires a fresh plan.

Only a still later heartbeat, after a clean worktree, matching revision/source
approvals, green CI, absent stop marker, absent lease, `awaiting_study` state,
and confirmation that this ID has never run, may invoke the controller exactly
once on local CPU. No fixture or worker may run directly. No terminal study or
quarantined predecessor may be rerun.

That later invocation must also verify before and after execution the protected
submission tree `e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`, protected submitted
ZIP SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`,
and protected manifest SHA-256
`99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`.
It must not inspect any private generated result before the controller returns
its sanitized terminal projection, must write output and sidecar only beneath
the sibling `learn2design-local-lab` root, and must leave `submission/` and
`artifacts/generated/submission.zip` byte-identical. A source-approval,
registry-digest, revision, output-path, or protected-hash mismatch forbids
invocation and is never auto-recovered.

All twenty-three cases and every complete positive/control gate must pass.
Success action is:

`synthetic_online_sarsa_control_confirmed_for_harness`

Any failed invariant or threshold, comparator/control recovery, malformed or
nondeterministic projection, process mismatch, timeout, source drift, or
terminal error uses:

`park_online_control_research`

The terminal projection uses the success action with `status == "passed"` if
and only if all twenty-three `passed` fields are true. It uses the park action
with `status == "failed"` if and only if one or more are false. A contradictory
status, action, or case vector is malformed and is itself a terminal failure.

The terminal fixture has no retry, top-up, alternate seed, relaxed threshold,
case removal, or same-ID repair.

## Claim boundary

A pass may say only that this fixed deterministic local-CPU harness generated
the declared synthetic family, selected actions online under the frozen
greedy/forced behavior schedule, applied the exact SARSA(0) next-action update
order, acquired the deliberately exposed three-step toy mapping, froze the
resulting policy on untouched generator regimes, beat the frozen toy
comparators, and lost the positive gate under the exact transition,
action-assignment, terminal-origin, and signal interventions.

It cannot support a claim about online learning being necessary, absence of a
public target shortcut, optimal or general exploration, sample efficiency
outside this one schedule, general or production RL, meta-RL, partial
observability, official data, private or hidden topology, UIFO, the submitted
optimizer, candidate selection, a native rewrite, accelerator value,
leaderboard rank, competition score, or permission to change or upload the
protected submission.
