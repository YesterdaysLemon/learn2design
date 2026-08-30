# Frozen plan: online SARSA token-gated choice mechanics

Status: **frozen before implementation or learner execution; the commit
containing this file is the immutable freeze boundary**

Study ID: `online-sarsa-latched-choice-v2`

Date frozen: 2026-08-29

Execution budget: none in this checkpoint. Implementation and any guarded
local-CPU invocation are separate later gates.

## Closed predecessors and independent construction

The ten registered local studies are terminal and cannot be rerun. The latest,
`multistep-td-action-prefix-v3`, closed only its fixed offline synchronous-TD
question.

`multistep-td-propagation-v1`, `multistep-td-action-prefix-v2`, and
`online-sarsa-latched-choice-v1` are quarantined pre-result rejections. Their
fixtures, workers, generators, schedules, seeds, thresholds, case schemas,
development diagnostics, and observed outputs must not be imported, executed,
repaired, registered, reused, or used to select this study. In particular, V1
required several rejection fields to be both exact integer counts and exact
JSON Booleans; no V2 field or type rule may inherit that contradiction.

This plan independently freezes a different four-action family, six-scalar
observation, four-byte policy frame, 48-position token schedule, 36 epochs,
regime set, target codes, random seed, case names, and result schema. No value
below was selected against a predecessor metric, official data, private
evidence, or a development run. The formulas determine every non-random count
algebraically. The only pseudorandom component is the separately frozen random
comparator.

Before this file is committed, hostile review may correct the draft. The commit
containing the final text is the freeze boundary. After that commit, any
substantive confound quarantines this ID and requires a fresh versioned plan;
the contract may not be repaired in place.

## Narrow falsifiable question

Can a blank deterministic tabular SARSA(0) learner, updating online from
actions selected under its own precommitted greedy/forced exploration-token
behavior, acquire a deliberately learnable four-action terminal choice; retain
its final greedy policy on untouched validation and test regimes; beat frozen
constant, feedback-only myopic, no-bootstrap, and seeded-random comparators;
and reject the same complete positive gate under opposite-beacon bootstrap,
zero-action assignment, null-terminal-signal, and complete beacon-ablation
controls?

A pass may validate only this fixed synthetic online-control harness, its exact
next-action SARSA chronology, and its deliberately exposed toy beacon. It cannot
support a claim about online learning being necessary, optimal exploration,
sample efficiency beyond this schedule, production or general RL, meta-RL,
hidden topology, UIFO, the submitted optimizer, candidate value, a native
rewrite, accelerator value, leaderboard standing, or competition score.

## Fresh topology-independent family

The family contains no topology, graph, optical network, official archive,
UIFO input, candidate array, private evidence, provider input, or learned
generator. `structure_kind` is exactly `none`. Every episode has four selected
binary actions, three nonterminal zero rewards, and one evaluator-only terminal
reward.

There are exactly eight regimes:

| split | regime code | signal scale | nuisance offset |
|---|---:|---:|---:|
| train | 7103 | 0.57 | -1.25 |
| train | 7121 | 0.79 | -0.40 |
| train | 7159 | 1.21 | 0.40 |
| train | 7187 | 1.46 | 1.25 |
| validation | 8209 | 0.49 | -1.75 |
| validation | 8237 | 1.58 | 1.75 |
| test | 9311 | 0.41 | -2.25 |
| test | 9343 | 1.71 | 2.25 |

Every train regime has 36 epochs of 48 episodes, exactly 1,728 episodes per
regime and 6,912 train episodes total. Training order is epoch, then the four
train regimes in table order, then position `p in 0..47`. Define:

```text
code = p // 3                          # 0..15
slot = p % 3                           # 0, 1, or 2
gray(x) = x xor (x >> 1)

slot 0: forced exploration, beacon=0, forced_code=gray(code)
slot 1: greedy behavior, beacon=code mod 2, forced_code=none
slot 2: forced exploration, beacon=1, forced_code=gray(code)
```

The four forced actions are the big-endian bits of `forced_code`. Gray coding
is a fixed permutation, so every epoch/regime has one
forced traversal of every four-action tuple for each beacon plus sixteen greedy
episodes, eight per beacon. There are exactly 4,608 forced and 2,304 greedy
train episodes; 18,432 forced and 9,216 greedy train selections; and 27,648
train selections and updates total.

For paired public covariates in train:

```text
family_slot = code if slot in {0,2} else 16 + (code // 2)
pair_serial = 24*epoch + family_slot
magnitude = signal_scale * (1.0 + (pair_serial + 1) / 65536.0)
nuisance = nuisance_offset
         + (((13*epoch + 7*family_slot + regime_code) mod 53) - 26) / 64.0
```

The forced beacon-zero and beacon-one episodes for one code share magnitude
and nuisance. Greedy codes `2k` and `2k+1` likewise form an opposite-beacon
pair with identical magnitude and nuisance. Thus neither unsigned magnitude nor
nuisance recovers beacon within any declared pair.

Each validation and test regime has exactly 384 episodes, in table order. For
held-out local episode `e in 0..383`:

```text
beacon = e mod 2
pair_serial = e // 2
magnitude = signal_scale * (1.0 + (pair_serial + 1) / 32768.0)
nuisance = nuisance_offset
         + (((17*pair_serial + regime_code) mod 59) - 29) / 64.0
```

Adjacent held-out episodes are opposite-beacon pairs with byte-identical
unsigned magnitude and nuisance. Held-out execution is always greedy, receives
no exploration token, and performs no update. There are exactly 768 validation
and 768 test episodes.

The detached evaluator target is:

```text
target_code = 3  when beacon == 0      # binary 0011
target_code = 12 when beacon == 1      # binary 1100
action_code = 8*a0 + 4*a1 + 2*a2 + a3
terminal_reward = 1.0 iff action_code == target_code else 0.0
```

Every forced epoch/regime therefore contains exactly two terminal successes,
one for each beacon. The complete train schedule has exactly 288 forced
successes. Greedy successes are retained descriptively but cannot select a
checkpoint, threshold, retry, seed, or stopping time.

## Public action-prefix observation and policy frame

At phase `t in 0..3`, the public observation is an immutable finite
C-contiguous little-endian `float64[6]` array:

```text
(phase, signed_beacon, prefix_code, prefix_width, steps_left, nuisance)
```

The legal form is:

```text
phase = t
signed_beacon = (-1 if beacon==0 else +1) * magnitude
prefix_width = t
prefix_code = the integer encoded by exactly the t prior actions, or 0 at t=0
steps_left = 4 - t
```

`prefix_code` lies in `0..2^t-1`; unused high bits are zero. After a valid
action at phases zero through two, the successor is the pure transformation

```text
next_phase = phase + 1
next_prefix_code = 2*prefix_code + action
next_prefix_width = prefix_width + 1
next_steps_left = steps_left - 1
```

and otherwise preserves the signed beacon, magnitude, and nuisance bytes.
Conditional on the already public predecessor and selected action, successor
construction has no evaluator target, correctness, reward, future action,
source identity, control identity, learner, donor, or origin input. Phase three
has no successor and `done` is exact Python `True`.

The signed beacon intentionally exposes which terminal code is rewarded. A
hand-programmed oracle can solve the task without learning, so this fixture
cannot establish that online learning or delayed credit is necessary or that
the public surface lacks a shortcut.

A pure detached projector validates the public array and emits exactly four
bytes in this order:

```text
PolicyFrame(phase:uint8, beacon_bin:uint8, prefix_width:uint8,
            prefix_code:uint8)
```

The frame is an owned exact `bytes` object of length four. It has no array
alias, closure, parent, handle, `__dict__`, or hidden metadata. The tabular key
is exactly the decoded four fields. Across two beacons there are 30 states and
60 state-action cells:

```text
2 beacons * (1 + 2 + 4 + 8 prefixes) * 2 actions = 60 cells
```

Magnitude, nuisance, regime, split, episode, epoch, position, exploration
mode, target, reward, donor, origin, assignment, and counters are excluded.

## Exact typed contracts

Split codes are `train=uint8(0)`, `validation=uint8(1)`, and `test=uint8(2)`.
The exact keys are:

- `EpisodeKey(split:uint8, regime_code:int32, local_episode:int16)`;
- `ObservationKey(episode_key, phase:uint8, prefix_width:uint8,
  prefix_code:uint8)`;
- `SelectionKey(episode_key, phase:uint8, global_ordinal:int32)`;
- `ActionKey(observation_key, action:int8)`;
- `TransitionKey(action_key, phase:uint8)`;
- `FeedbackKey(transition_key, phase:uint8)`;
- `AssignmentKey(selection_key, executed_action:int8,
  assigned_action:int8)`; and
- `UpdateKey(feedback_key, assignment_key, table_version_before:int32)`.

No key contains beacon, target code, correctness, reward, exploration mode,
control mode, donor, or origin. Exact NumPy widths and little-endian bytes are
mandatory; Python or NumPy Booleans never satisfy an integer field.

The exact runtime values are:

- observation: exact `numpy.ndarray`, little-endian `float64`, shape `(6,)`,
  strides `(8,)`, C-contiguous, owned, immutable, finite, and legal;
- action: zero-dimensional immutable `numpy.ndarray`, dtype `int8`, value zero
  or one; Python integers, Booleans, floats, rank-one arrays, subclasses,
  writable arrays, and out-of-range values reject before any lazy resolves;
- reward: zero-dimensional immutable `numpy.ndarray`, little-endian
  `float64`, nonterminal value exactly `0.0`, terminal value in `{0.0,1.0}`;
- `done`: exact Python `False` for phases zero through two and exact Python
  `True` at phase three;
- source, origin, assignment, and target references: exact slotted typed
  records or exact `None`, never untyped integers or Booleans; and
- all metadata integers: exact declared NumPy types with no coercion.

The seven ordered invalid-action probes are exact Python `int(0)`, exact Python
`True`, exact Python `float(0.0)`, rank-one immutable `int8[1]`, an immutable
zero-dimensional ndarray subclass, a writable zero-dimensional `int8` ndarray,
and an immutable zero-dimensional `int8(2)` ndarray. Each rejects before any
lazy successor or reward access and contributes exactly one rejection.

The hidden `EvaluatorBeacon(uint8)` and public `SignedBeacon(float64)` are
detached types sharing no object, buffer, closure, container, or parent.

## Complete family reconstruction, twins, and split separation

Before constructing any source, policy, comparator, or learner, the primary
generator enumerates every legal state/action row for all 8,448 episodes. Each
episode has:

```text
phase 0: 1 prefix * 2 actions = 2 rows
phase 1: 2 prefixes * 2 actions = 4 rows
phase 2: 4 prefixes * 2 actions = 8 rows
phase 3: 8 prefixes * 2 actions = 16 rows
total: 30 rows
```

The projection contains exactly 253,440 rows: 118,272 nonterminal rows,
135,168 terminal rows, and 126,720 action-bearing state nodes. Here
`predecessor_node_count` means the 15 legal pre-action states per episode,
including 59,136 nonterminal states and 67,584 phase-three pre-action states;
it does not mean transition rows. Every row commits the
split/regime/episode identity, exact predecessor key and bytes, action key and
scalar bytes, successor key/bytes/layout/legality or terminal sentinel, source
reward, update reward, `done`, source identity, and canonical absent control
references. All 253,440 transition keys are unique.

A separately implemented replay, sharing no row constructor, transition
formula helper, serializer, or terminal evaluator helper with the primary
path, reconstructs every row in a different loop order. It independently
proves counts, exact formulas, unique keys, target/action balance, successor
legality, terminal outcomes, and equality of the canonical SHA-256.

The 8,448 canonical behavior/evaluation paths contain 33,792 realized rows.
Public path commitments include observation bytes/layout, actions, successors
or terminal sentinel, and `done`; they exclude every identity, split, target,
reward, and evaluator field. Train, validation, and test path sets must be
pairwise disjoint with zero overlap. Identity metadata cannot make the proof
pass.

For every episode, the evaluator-counterfactual audit freezes the source
identity and every structural public, action, key, successor, schedule, and
Boolean byte. It then copies that frozen structural projection, flips only the
detached evaluator beacon, and independently rematerializes the terminal reward
fields. It never calls the public generator after the flip. Exactly two terminal
reward records per episode may change: the `0011` and `1100` rows exchange both
source-reward and canonical update-reward bytes; every other reward and every
structural/source-identity byte is fixed. `terminal_reward_change_count` counts
changed rows, not scalar fields, so it is exactly 16,896.
Self-comparison, copy-only comparison, public regeneration, aliasing, a changed
source identity, or any broader field difference rejects before a policy
factory exists.

### Canonical commitment ABI

Every SHA-256 in this study uses one frozen domain-separated binary ABI; JSON,
`repr`, pickle, platform-native integers, locale, and dictionary insertion order
are forbidden. The encoded preimage is:

```text
ascii("online-sarsa-latched-choice-v2/") || ascii(domain) || 0x00 ||
u64_le(record_count) ||
for record in exact_order:
    u32_le(record_byte_count) || record_bytes
```

Every `record_bytes` starts with a framed ASCII record-class name and
`u16_le(field_count)`; each field then uses `u16_le(field_ordinal)`, a framed
ASCII exact type tag, `u32_le(value_byte_count)`, and the value bytes. Typed
scalars use their tag plus declared little-endian NumPy payload. A Boolean is
tag `bool` plus one byte `0x00` or `0x01`; `None` is tag `none` plus byte
`0xff`; a string or byte sequence is its distinct tag plus
`u32_le(length)||payload`; a fixed tuple is tag `tuple`, `u32_le(arity)`, then
each recursively framed typed element. An array is tag `ndarray`, framed ASCII
dtype, rank, every `u32_le` shape and stride value, C-contiguity, immutability,
payload length, and exact C-order bytes. A typed key is tag `key`, framed class
name, field count, then recursively framed fields in declaration order. Lists
and maps use the exact recursive framing specified in the result-schema section
below. No two declared types share a tag or absent encoding.

Family records are sorted by the exact typed-key bytes. Token and action-stream
records are ordered by global selection ordinal. Update records are ordered by
global update ordinal. Episode/path records are ordered by split, regime-table
ordinal, local episode, then phase; the public split-separation set digest
removes split, regime, and episode identity before lexicographic byte sorting.
Fixture/container commitments sort keys by their UTF-8 bytes and retain list
order. All digest domains, record counts, and exact field projections are
stored in the registry identity; a helper shared between primary and replay
paths is forbidden.

The exact ordered complete-family corruption probes are:

```text
duplicate_key, missing_row, extra_row, wrong_predecessor_key,
wrong_predecessor_bytes, wrong_action_key, wrong_action_scalar,
illegal_successor_key, wrong_successor_bytes, wrong_successor_layout,
wrong_nonterminal_done, wrong_terminal_done, wrong_source_reward,
wrong_update_reward, wrong_source_identity, nonabsent_control_reference,
cross_episode_successor, evaluator_twin_reward_under_canonical_beacon
```

Each of the 18 probes changes only its named field/class, enters the real
independent replay validator, and rejects before any source, policy, comparator,
or learner factory. Thus `corruption_class_count` and
`corruption_rejection_count` are both exactly 18 and `factories_before_gate` is
exactly zero.

The complete-family digest, schedule digest, and random-stream digest are
derived values, not tunable choices. A later implementation checkpoint may
compute them only from the exact frozen formulas, then commit them in focused
tests and the registry before any learner runs. Any primary/replay mismatch
quarantines V2.

## Exploration-token and behavior-policy contract

Training uses an immutable four-byte token:

```text
ExplorationToken(mode:uint8, forced_action:int8,
                 epoch_mod_64:uint8, phase:uint8)
```

`mode=0` is greedy and requires `forced_action=-1`; `mode=1` is forced and
requires the exact bit of the predeclared forced code. The token generator
receives only epoch, position, and phase. It has no observation, beacon,
target, reward, Q value, evaluator, source, or control input. Within each
declared opposite-beacon pair, tokens are byte-identical at every phase.

Two independent generators reconstruct all 27,648 tokens and their exact
order before learner construction. The complete schedule must have 2,304
greedy episodes, 4,608 forced episodes, 9,216 greedy tokens, 18,432 forced
tokens, and exactly 288 forced successes. `forced_tuple_balance_ok` means only
that the forced slot-zero and slot-two episodes traverse each of the sixteen
four-action tuples exactly once per beacon/regime/epoch; no balance claim is
made about learner-dependent greedy tuples. The 3,456 declared opposite-beacon
episode pairs have byte-identical tokens at all four phases, so
`paired_token_count` is exactly 13,824 phase-token matches.

At selection the policy receives only an owned immutable four-byte
`PolicyFrame`, an owned immutable 480-byte snapshot of the complete 60-cell
little-endian `float64` Q table, and one owned four-byte token. Forced mode
returns the token action; greedy mode returns the greater Q value with exact
action-zero tie break. No whole action tuple is installed: each later action
is selected only after its real successor exists.

Each selection produces a one-use `SelectionPermit` binding the exact
`SelectionKey`, policy-frame digest, token digest, pre-action table version,
complete Q digest, selected-cell identity/value digest, selected action, and a
deterministic nonce derived from the frozen schedule digest and selection
ordinal. The permit exposes no mutable alias. Duplicate, stale, future,
cross-episode, wrong-phase, wrong-token, wrong-action, or reentrant use clears
the pending capability and rejects.

Selection authorization alone cannot authorize environment or updater work.
The environment privately issues an opaque, non-serializable, one-use
`IssuedTransitionPermit` only after consuming the real `SelectionPermit`. It
binds a fresh run nonce, exact episode/predecessor/selection/action identity,
expected successor identity or terminal sentinel, source-reward slot, and the
before-version. After the successor exists, selection of a nonterminal next
action privately issues a `LatchedNextActionPermit` binding the exact successor,
next selection/action, pre-update Q snapshot/cell, and before-version. The
environment then issues an opaque one-use `IssuedFeedbackPermit` binding the
transition permit, exact source/update reward digests, `done`, origin identity,
and feedback nonce. These capabilities have no public constructor, serializer,
copy/deepcopy/reduce protocol, equality-by-fields, or mutable alias. Counterfeit,
clone, replay, cross-run, cross-episode, wrong-stage, or duplicate consumption
rejects and clears the complete pending chain.

Their exact private slot schemas are registry commitments (`D` is exact 32
digest bytes):

```text
SelectionPermit = (
  selection_key:K, policy_frame_digest:D, token_digest:D,
  table_version_before:int32, q_table_digest:D, q_cell_index:int16,
  q_cell_value:float64, executed_action:int8, schedule_ordinal:int32,
  run_nonce:D, consumed:bool)
IssuedTransitionPermit = (
  transition_key:K, selection_key:K, predecessor_key:K,
  executed_action:int8, expected_successor_key:K|None,
  source_reward_slot:int8, table_version_before:int32,
  selection_permit_nonce:D, transition_nonce:D, consumed:bool)
LatchedNextActionPermit = (
  predecessor_transition_key:K, successor_key:K, next_selection_key:K,
  next_action:int8, q_snapshot_digest:D, q_cell_index:int16,
  q_cell_value:float64, table_version_before:int32,
  next_action_nonce:D, consumed:bool)
IssuedFeedbackPermit = (
  feedback_key:K, transition_key:K, source_reward_digest:D,
  update_reward_digest:D, done:bool, origin_reference:OriginReference,
  transition_nonce:D, feedback_nonce:D, consumed:bool)
TargetReference = (
  transition_key:K, real_frame_digest:D, target_frame_digest:D, mode:uint8)
OriginReference = (
  feedback_key:K, source_reward_digest:D, update_reward_digest:D, mode:uint8)
AssignmentReference = (
  selection_key:K, executed_action:int8, assigned_action:int8, mode:uint8)
```

Modes are canonical `0`, opposite-target `1`, zero-assignment `2`, null-origin
`3`, and beacon-ablation `4`, valid only for the corresponding reference type.
No reference exposes a parent object or evaluator truth. The registry binds the
exact slot order, types, allowed `None` positions, mode sets, one-use state
machine, and capability-to-consumer map.

An independent behavior reconstruction shares no projector, table index,
selector, updater, or digest helper. From a blank table and authenticated prior
updates it reproduces every pre-action snapshot, token, action, tie decision,
and table version. Changed tokens/actions, unlogged calls, whole-code installs,
selection before observation, reselection after an intervening update, and
duplicate selection reject.

## Exact online SARSA(0) chronology

The learner starts with 60 little-endian `float64` Q cells and 60
little-endian `int32` visit counters exactly zero. Initial table version is
`int32(0)`. Discount is `gamma=1.0`; step size is exact `alpha=0.20`.

For each episode, the only legal order is:

```text
observe0, issue_token0, select0, validate0, step0,
resolve_successor0, resolve_zero0,
observe1, issue_token1, select1, validate1,
update0_from_latched_action1, append0, step1,
resolve_successor1, resolve_zero1,
observe2, issue_token2, select2, validate2,
update1_from_latched_action2, append1, step2,
resolve_successor2, resolve_zero2,
observe3, issue_token3, select3, validate3,
update2_from_latched_action3, append2, step3,
resolve_terminal3, update3_terminal, append3, close_episode
```

At each nonterminal boundary, the next action and its pre-update Q snapshot are
selected and latched before the preceding update. The same latched action is
then executed without reselection even though the update may change one Q
cell. For assigned update actions `u_t` and `u_(t+1)`:

```text
nonterminal_target = 0.0 + Q_pre_update[s_(t+1), u_(t+1)]
terminal_target = terminal_update_reward
Q_new[s_t,u_t] = Q_old[s_t,u_t]
                 + 0.20 * (target - Q_old[s_t,u_t])
```

Canonical mode assigns every executed action to itself. Exactly one Q cell and
one visit cell are writable per update. For global update ordinal `k`, before
version is `int32(k)` and after version `int32(k+1)`. There are exactly 27,648
updates. Nonterminal target code has no terminal scalar, evaluator, origin,
maximum-over-actions, post-update Q, or raw trajectory capability. Terminal
target code has no successor or bootstrap capability.

The exact train ordinals for epoch `e`, train-regime table ordinal `r`, position
`p`, and phase `t` are:

```text
episode_ordinal = ((e*4 + r)*48) + p
selection_ordinal = 4*episode_ordinal + t
update_ordinal = 4*episode_ordinal + t
selection_table_version(t=0,1,2,3) =
    (4*episode_ordinal, 4*episode_ordinal,
     4*episode_ordinal + 1, 4*episode_ordinal + 2)
update_version_before = update_ordinal
update_version_after = update_ordinal + 1
```

For held-out split ordinal `h` (`validation=0`, `test=1`), within-split regime
ordinal `r in {0,1}`, local episode `e in 0..383`, and phase `t`:

```text
heldout_episode_ordinal = ((h*2 + r)*384) + e
heldout_selection_ordinal = 27648 + 4*heldout_episode_ordinal + t
```

Thus held-out selection ordinals are exactly 27,648 through 33,791, validation
before test. Every held-out selection binds frozen learner version 27,648.
These formulas, not call timing or list position, define all ordinals.

The 30 states are indexed by `(beacon,phase,prefix_code)` in lexicographic
order. Cell index is `2*state_index + action`; its Q snapshot byte range is
`[8*cell_index, 8*cell_index+8)` in the exact 480-byte table, and the matching
visit range is `[4*cell_index, 4*cell_index+4)` in the 240-byte visit table.

The updater receives a fresh slotted `SealedSarsaUpdate` containing only owned
current and optional next policy-frame bytes, executed and assigned current
actions, optional executed and assigned latched next actions, one exact bare
`numpy.float64` update scalar copied only after the environment reward ndarray
is authenticated, exact `done`, the three-or-four owned capability digests, full
before-Q and visit snapshot digests, selected-cell offsets/values, and
before-version. It contains no public array, source-reward object, hidden
evaluator beacon, target formula, source, split/regime/key, token parent, donor,
origin, assignment parent, environment, lazy value, collector, ambient global,
or closure. Consumption verifies every required live private capability and the
full Q/visit state before writing the single frozen cell.

The exact slot order is:

```text
current_frame:Y[4], next_frame:Y[4]|None,
executed_current_action:int8, assigned_current_action:int8,
executed_next_action:int8|None, assigned_next_action:int8|None,
update_reward:float64, done:bool,
selection_permit_digest:D, transition_permit_digest:D,
next_action_permit_digest:D|None, feedback_permit_digest:D,
q_snapshot_digest:D, visit_snapshot_digest:D,
current_cell_offset:int16, current_cell_value:float64,
next_cell_offset:int16|None, next_cell_value:float64|None,
table_version_before:int32
```

Thus a nonterminal update binds four live capabilities and a terminal update
binds three, with the next-action slot exactly `None`; the earlier phrase
"capability digest" never means authorization by digest alone.

## Physical boundaries, capabilities, and trace authentication

The real source and environment own operation-counting lazy successors and
rewards. Canonical online training has exactly 27,648 steps, selections, and
updates; 20,736 successor materializations; 27,648 reward materializations;
and 6,912 closes. Validation plus test has 6,144 steps and selections, 4,608
successors, 6,144 rewards, zero tokens, zero updates, and 1,536 closes.

Every hostile attack begins from fresh blank state, traverses the real boundary
it names, records exactly one attempted forbidden operation when applicable,
records zero permits, raises the exact contract error, leaves Q/visits/version
byte-identical, and clears every pending token, selection, transition, reward,
origin, and close capability. An argument error before the named spy records an
attempt is not evidence.

The exact ordered policy/source attack classes are:

```text
lazy_successor_before_selector, lazy_reward_before_selector,
evaluator_handle_in_policy, mutable_policy_frame, q_snapshot_alias,
token_reuse, invalid_action_before_lazy, nested_environment_during_select,
selection_before_successor, hidden_source_identity_in_policy,
ambient_source_access_in_policy, ambient_reward_access_in_policy,
ambient_environment_access_in_policy, ambient_trace_access_in_policy,
resolver_reentrant_select, writable_observation_source,
observation_array_subclass_source, ambient_evaluator_access_in_policy,
ambient_control_mode_access_in_policy, ambient_registry_access_in_policy,
ambient_learner_global_access_in_policy
```

The exact ordered update/trace attack classes are:

```text
update_before_next_selection, changed_latched_action,
postupdate_bootstrap_snapshot, stale_future_or_skipped_version,
duplicate_selection_permit, terminal_scalar_at_nonterminal,
missing_or_duplicate_zero, early_terminal_origin,
duplicate_terminal_scalar, source_reward_in_sealed_update,
assignment_parent_in_sealed_update, nested_select_during_update,
duplicate_append_or_close, pending_state_at_split_seal,
reward_before_step, successor_before_step, late_successor_resolution,
early_terminal_reward, reentrant_step, reentrant_successor_resolution,
ambient_control_access_in_update, ambient_trace_access_in_update
```

The partitions are disjoint and contain exactly 21 and 22 classes. Every class
is attempted once through its named live spy. Reward-before-step,
successor-before-step, late-successor, and early-terminal attacks have distinct
stage sentinels; step, resolver, selector, and updater reentrancy each enter the
real named boundary once. Before and after full Q, visit, version, capability,
trace, source-operation, and environment-operation state hashes must match for
all rejected attacks. Counts are reported only in fields explicitly typed as
integers; pass/fail fields are separately typed Booleans.

One-use capability mutation classes are exactly:

```text
wrong_episode, wrong_phase, wrong_policy_digest, wrong_token_digest,
wrong_table_version, wrong_table_digest, wrong_visit_digest,
wrong_cell_identity, wrong_cell_value, wrong_executed_action,
wrong_assigned_action, wrong_predecessor_identity, wrong_successor_identity,
wrong_next_action, wrong_source_reward_digest, wrong_update_reward_digest,
wrong_done, wrong_origin, wrong_transition_nonce, wrong_feedback_nonce,
duplicate_nonce, counterfeit_capability, cloned_capability, cross_episode,
skipped_version, duplicate_append
```

There are exactly 26 capability mutation classes and 26 rejections. Every
mutation enters the consumer for the exact capability it names; construction
failure outside that consumer is not evidence. Capability and state hashes
before and after each rejection are committed in ordered outcome vectors.

The exact mutation-to-consumer map is:

```text
wrong_episode, wrong_phase, wrong_policy_digest, wrong_token_digest,
wrong_table_version, wrong_table_digest -> consume_selection_permit
wrong_visit_digest, wrong_cell_identity, wrong_cell_value,
wrong_assigned_action, wrong_successor_identity, wrong_update_reward_digest,
wrong_feedback_nonce, skipped_version -> consume_sealed_sarsa_update
wrong_executed_action, wrong_predecessor_identity,
cross_episode -> consume_issued_transition_permit
wrong_next_action -> consume_latched_next_action_permit
wrong_source_reward_digest, wrong_done, wrong_origin,
wrong_transition_nonce -> consume_issued_feedback_permit
duplicate_nonce, cloned_capability -> consume_capability_ledger
counterfeit_capability -> dispatch_exact_capability_type
duplicate_append -> consume_trace_append_permit
```

In particular, `wrong_visit_digest` mutates the exact
`SealedSarsaUpdate.visit_snapshot_digest` slot and reaches that updater consumer;
it is not inferred from a Q digest or rejected by an outer argument parser.

The sealed train trace contains separately keyed observation, token, selection,
transition, feedback, update, and close components. Exact counts are 27,648 for
each component except 6,912 closes. Every component authenticates exact keys,
links, types, layouts, bytes, actions, source/update rewards, versions, visits,
references, `done`, and commitments. Components are joined by typed keys, never
list position. For list order `Observation, Token, Selection, Transition,
Feedback, Update, Close`, rotations are exactly `[1,3,5,7,11,13,17]`. For a
list of length `N` and output index `j`, the exact source index is
`(N-1-j+rotation) mod N`. These seven independent permutations must reproduce
exact scoring and update projections, and their ordered commitments form
`reordering_outcome_sha256`.

The exact component field names, order, and scalar types are below. `K` means
the exact declared typed key, `D` exact 32 digest bytes, `Y[n]` owned exact
bytes of length `n`, and the NumPy scalar names require that exact width.
Optional fields are always present and use exact `None`; they are never omitted
or represented by a sentinel of another type.

```text
ObservationRecord = (
  observation_key:K, public_bytes:Y[48], dtype:S("<f8"),
  shape:tuple[int32](6), strides:tuple[int32](8),
  c_contiguous:bool(True), immutable:bool(True), policy_frame_digest:D)
TokenRecord = (
  selection_key:K, token_bytes:Y[4], token_digest:D)
SelectionRecord = (
  selection_key:K, observation_key:K, invocation_ordinal:int32,
  policy_frame_digest:D, token_digest:D, q_table_digest:D,
  q_cell_digest:D, table_version_before:int32, selected_cell_index:int16,
  selected_cell_value:float64, executed_action:int8, tie_flag:bool)
TransitionRecord = (
  transition_key:K, selection_key:K, predecessor_key:K,
  predecessor_bytes:Y[48], predecessor_dtype:S("<f8"),
  predecessor_shape:tuple[int32](6), predecessor_strides:tuple[int32](8),
  predecessor_digest:D, executed_action:int8, successor_key:K|None,
  successor_bytes:Y[48]|None, successor_dtype:S("<f8")|None,
  successor_shape:tuple[int32](6)|None,
  successor_strides:tuple[int32](8)|None, successor_digest:D|None,
  source_reward_bytes:Y[8], source_reward_digest:D, done:bool,
  target_reference:None, control_reference:None)
FeedbackRecord = (
  feedback_key:K, transition_key:K, source_reward_bytes:Y[8],
  source_reward_digest:D, update_reward_bytes:Y[8],
  update_reward_digest:D, origin_reference:OriginReference)
UpdateRecord = (
  update_key:K, selection_key:K, feedback_key:K, predecessor_key:K,
  successor_key:K|None, predecessor_digest:D, successor_digest:D|None,
  current_real_frame_digest:D, next_real_frame_digest:D|None,
  current_target_frame_digest:D, next_target_frame_digest:D|None,
  executed_current_action:int8, assigned_current_action:int8,
  executed_next_action:int8|None, assigned_next_action:int8|None,
  source_reward_digest:D, update_reward_digest:D, done:bool,
  target_value_digest:D, old_value_digest:D, new_value_digest:D,
  table_version_before:int32, table_version_after:int32,
  visit_before:int32, visit_after:int32,
  assignment_reference:AssignmentReference,
  target_reference:TargetReference|None)
CloseRecord = (
  episode_key:K, terminal_transition_key:K, terminal_update_key:K,
  terminal_source_reward_digest:D, done:bool(True),
  pending_empty:bool(True), close_ordinal:int32)
```

For a nonterminal transition all successor fields are non-`None`; for a
terminal transition every successor field and both next-action/frame fields are
`None`. Canonical current target-frame digest equals current real-frame digest.
Ordinary permutation of a component list must validate; the malformed class
`independently_swapped_component` means installing an otherwise valid sealed
record under a different typed key after independent cross-key substitution,
not merely changing list order.

The exact malformed trace classes are:

```text
missing_component, duplicate_component, unknown_component,
wrong_numpy_width, python_bool_for_integer, numpy_bool_for_integer,
wrong_layout, cross_episode,
cross_regime, evaluator_twin_component, wrong_predecessor,
wrong_action, wrong_successor, wrong_source_reward, wrong_update_reward,
wrong_done, wrong_version, wrong_token, wrong_origin,
wrong_assignment, independently_swapped_component, wrong_close_link
```

Every applicable record/class pair rejects; nonapplicable pairs are frozen in
a registry applicability matrix before validation and cannot count as a
rejection. Original objects are mutated after sealing to prove owned immutable
trace bytes.

The exact applicability matrix is:

```text
all seven records:
  missing_component, duplicate_component, unknown_component,
  wrong_numpy_width, python_bool_for_integer, numpy_bool_for_integer,
  cross_episode, cross_regime, independently_swapped_component
ObservationRecord, TransitionRecord:
  wrong_layout
FeedbackRecord, UpdateRecord, CloseRecord:
  evaluator_twin_component
TransitionRecord, UpdateRecord:
  wrong_predecessor, wrong_successor
SelectionRecord, TransitionRecord, UpdateRecord:
  wrong_action
TransitionRecord, FeedbackRecord, UpdateRecord, CloseRecord:
  wrong_source_reward
FeedbackRecord, UpdateRecord:
  wrong_update_reward, wrong_origin
TransitionRecord, UpdateRecord, CloseRecord:
  wrong_done
SelectionRecord, UpdateRecord:
  wrong_version
TokenRecord, SelectionRecord:
  wrong_token
UpdateRecord:
  wrong_assignment
CloseRecord:
  wrong_close_link
```

Each named class is exercised once against every record type listed for it;
the exact expected pair count is 92 and is committed in `case_contract`.
`independently_swapped_component` contributes exactly seven of those
rejections. The canonical component lists contain exactly 172,800 records:
27,648 records for each of the first six types and 6,912 closes. There are
exactly seven successful independent list reorderings. A mutation may
count only if it changes its one declared field class and traverses the real
component validator.

## Bootstrap-cell dependency

Every one of the 20,736 nonterminal updates binds the exact selected
next-state/action cell in its authenticated pre-update snapshot. An independent
chronological equation replay reproduces every target, old/new value, visit,
table version, and table digest from the blank learner and action stream.

A pure shadow replaces only the selected next-cell scalar with exact zero for
each nonterminal update and recomputes only that target and current-cell update.
All shadows, including unchanged ones, are retained. The complete set must
contain at least one changed target and current-cell value for each of the six
predeclared categories `(beacon 0 or 1) x (phase 0, 1, or 2)`. Every off-cell
snapshot byte, action, key, reward, visit, and trace commitment remains fixed.
No shadow enters canonical learning, selection, scoring, comparator fitting, or
control execution.

Wrong next action, wrong state, post-update snapshot, version, Q-cell digest,
or terminal capability mutations must reject. This authenticates use of the
latched on-policy action across all three bootstrap boundaries without claiming
that the signal requires bootstrapping.

## Train-only fitting and untouched held-out regimes

The outer orchestration owns train, validation, and test sources, but the
learner API accepts only authenticated train environment and behavior-policy
capabilities. Every source counts factory, iterator, materialization, step,
selection, update, close, and post-close operations.

Every fitted path is covered separately: canonical SARSA, feedback-only
myopic, no-bootstrap, opposite-beacon fresh online, and beacon-ablation fresh
online. Zero-assignment and null-signal are authenticated canonical-stream
replays and are covered by the same source seal plus replay-only constructors.
Each fitted path runs in three fresh processes/constructions: held-out factory
and attributes absent; held-out installed as operation-counting exploding
factory/iterator/environment/reward objects; and held-out lazy/unopened until
the train seal. Constructor calls, attribute reads, iterator calls,
materializations, steps, selections, updates, closes, and process-global cache
reads are independently counted. All three modes produce byte-identical train
action, update, Q, visit, version, policy, and comparator commitments; every
held-out counter is zero.

The harness constructs passive spy shells and zeroes their counters before the
fit-call boundary; shell allocation is not a held-out source constructor. A
`factory` counter increments only when fitted code invokes the spy to construct
or retrieve a real held-out source. All attribute/iterator/environment/reward
counters likewise begin at entry to the exact fitted path. Thus installing a
passive exploding handle is compatible with every fitted-code held-out counter
remaining exactly zero; setup counters are separately committed and are exactly
one shell installation per exploding or lazy mode/path.

The complete suite is run in frozen process orders `A` (canonical, comparators,
controls) and `B` (controls in reverse, comparators in reverse, canonical), plus
one fresh worker per fitted path. Exact per-path operation traces and fit
commitments match across orders and fresh workers. No module cache, singleton,
class attribute, closure, environment variable, or global registry may convey
a held-out source or prior fitted state.

The inverse test installs an exploding train source after fit and runs the real
validation/test path. Train operations remain zero; every held-out episode is
evaluated once; no token is issued; and Q, visits, version, and policy digest
remain byte-identical before validation, after validation, and after test.
Held-out data cannot affect abstraction, normalization, exploration, alpha,
gamma, tie behavior, threshold, stopping, checkpoint choice, baseline choice,
or any learner state.

Every held-out operation uses a split-specific source capability issued only
after the train seal. The train-only API cannot receive, import, look up, or
reflect on that capability. Ambient access probes cover source, reward,
environment, trace, control, registry, and process-global namespaces.

## Frozen metrics and positive gate

Primary performance is terminal return within each complete regime, then an
unweighted macro average over regimes. Minimum regime return is retained.
Train regret is episode count minus terminal reward sum. All policies are
rescored by an independent keyed evaluator over identical complete episodes.
Threshold comparisons use exact integer cross multiplication; floating-point
rounding never decides a gate.

The acquisition window is the final six epochs only: exactly 384 greedy
episodes, 96 per train regime and 192 per beacon. It is not used for stopping or
checkpoint selection. Only the final table after all 6,912 episodes is used for
held-out evaluation.

The registry identity freezes this exact threshold object; no synonymous key
or inferred default is allowed:

```text
minimum_final_window_macro_return = 0.95
minimum_final_window_regime_return = 0.90
minimum_validation_macro_return = 0.95
minimum_test_macro_return = 0.95
minimum_heldout_regime_return = 0.90
minimum_constant_gain = 0.70
minimum_myopic_gain = 0.70
minimum_no_bootstrap_gain = 0.70
minimum_random_gain = 0.45
opposite_beacon_maximum_return = 0.50
zero_assignment_maximum_return = 0.00
null_signal_maximum_return = 0.00
beacon_ablation_maximum_return = 0.50
opposite_beacon_minimum_test_gap = 0.45
zero_control_minimum_test_gap = 0.95
beacon_ablation_minimum_test_gap = 0.45
```

All reported `F` metrics are descriptive `float64` divisions of exact integer
counts; no `F` field decides a gate. Let `succ(P,S,R)` be the exact number of
terminal successes for policy/path `P`, split `S`, and regime `R`. Because every
regime in a scored split has the same episode count:

```text
macro(P,S) = sum_R succ(P,S,R) / total_episodes(S)
minimum_regime(P,S) = min_R succ(P,S,R) / episodes_per_regime(S)
gain(P,B,S) = (succ(P,S)-succ(B,S)) / total_episodes(S)
test_gap(P,C) = (succ(P,test)-succ(C,test)) / 768
```

For the final window the denominator is 384 and every regime denominator is 96;
validation and test each use total denominator 768 and regime denominator 384.
Every threshold is stored as a reduced integer ratio (`19/20`, `9/10`,
`7/10`, `9/20`, `1/2`, or `0/1`) in `case_contract`; validation uses
`numerator*threshold_denominator >= denominator*threshold_numerator` (or `<=`)
with exact Python integers. Result cases expose the relevant success numerators
and denominators as `I` fields as well as a domain-separated count digest.

`canonical_test_gap` for the first three controls is exactly the canonical test
success count from `frozen_heldout_evaluation` minus that control's test success
count, divided by 768. Beacon ablation has two separately gated fields:
`fresh_canonical_test_gap` and `frozen_policy_canonical_test_gap`, each using its
own ablated-path test numerator and the same canonical numerator. A single
combined gap cannot satisfy both gates.

The complete ordered positive-gate vector has exactly 32 Boolean clauses:

```text
typed_episode, generator_partition, family_replay, evaluator_twin,
path_separation, token_schedule, sealed_boundary, sarsa_chronology,
behavior_replay, capability_authentication, component_join,
bootstrap_dependency, source_separation, acquisition_counts,
acquisition_macro, acquisition_min_regime, validation_macro, test_macro,
heldout_min_regime, heldout_state_frozen, constant_validation_gain,
constant_test_gain, myopic_validation_gain, myopic_test_gain,
no_bootstrap_validation_gain, no_bootstrap_test_gain,
random_validation_gain, random_test_gain, comparator_replay,
intervention_difference, bounded_schema, worker_reproduction
```

Each executable control recomputes all 32 clauses in this order. Structural
clauses use the same frozen source/family evidence and the control's exact
declared intervention; score and state clauses use that control's own path.
Each control result reports `applicable_gate_clause_count=32`, the exact failed
clause count, a domain-separated 32-Boolean vector digest, and whether every
nonintervened structural clause matches canonical. `positive_gate_rejected`
requires at least one failed clause; omission or `N/A` is forbidden.

The complete positive gate requires:

- all family, twin, type, schedule, physical-boundary, capability, trace,
  behavior-replay, equation, bootstrap, source, difference, sanitizer, and
  process invariants passing;
- exactly 6,912 train episodes, 27,648 selections and updates, 2,304 greedy
  episodes, 4,608 forced episodes, and 288 forced successes;
- final-six-epoch greedy macro return at least `0.95` and minimum train-regime
  return at least `0.90`;
- validation and test macro return at least `0.95` and each minimum held-out
  regime return at least `0.90`;
- validation and test gain over the better constant, feedback-only myopic, and
  no-bootstrap comparators at least `0.70`;
- validation and test gain over seeded random at least `0.45`;
- exactly zero held-out updates and byte-identical frozen learner state before,
  between, and after held-out evaluation; and
- no retry, top-up, alternate seed, relaxed threshold, early stop, checkpoint
  selection, case removal, or same-ID repair.

Every control is evaluated against the complete applicable positive gate and
must reject it, not merely miss one score threshold.

## Independent comparators

Comparators receive only authenticated public behavior feedback or public
evaluation observations. None receives a target formula, counterfactual
reward, evaluator handle, hidden key, exploration schedule identity, control
mode, origin, assignment parent, or the learner trace object.

- `constant_zero` emits `0000`; exact return is zero in every regime.
- `constant_one` emits `1111`; exact return is zero in every regime.
- `feedback_only_myopic` independently fits immediate source-reward means for
  the same 60 public state/action cells from the sealed behavior stream, with
  no bootstrap and action-zero ties. Its phase-zero values remain zero and its
  exact held-out return is zero.
- `no_bootstrap` is a fresh online learner with the exact canonical schedule,
  alpha, selection, and update order but exact zero nonterminal targets. Its
  phase-zero policy ties to action zero and exact held-out return is zero.
- `seeded_random` uses one fresh
  `numpy.random.Generator(numpy.random.PCG64(161803399))`. It makes exactly one
  call per episode,
  `rng.integers(0,2,size=(4,),dtype=numpy.int8)`, in canonical train,
  validation, then test order. Scalar calls, four separate calls, default
  `int64`, extra draws, or another generator reject.

Each complete comparator action stream is generated twice from fresh state,
must be byte-identical, and is independently rescored. Fitted comparators use a
separate feedback materializer and share no SARSA projection helper. Aggregate
metric equality without action-stream equality is insufficient.

## Negative controls

### Opposite-beacon bootstrap target

This is a fresh online run. Before any source or policy exists, an immutable
control specification changes every nonterminal bootstrap target frame
`(phase,beacon,prefix_width,prefix_code)` to the otherwise identical frame with
`1-beacon`. The complete static source family, environment transition function,
token schedule, terminal evaluator, and terminal timing remain canonical. An
evaluator-side typed reference binds the real successor and substituted target;
the learner sees only target-frame bytes. Runtime actions, reached successors,
and realized rewards may differ only causally after the target-frame
intervention; the source and environment definitions themselves do not change.
`static_transition_function_unchanged` independently replays all 118,272 legal
nonterminal predecessor/action inputs through canonical and control transition
functions and requires identical successor bytes, layouts, keys, and an
identical domain-separated relation hash; it makes no claim that the two online
runs reach the same successors.

The mapping is a no-fixed-point involution over every legal nonterminal frame
and is independently replayed before a learner factory. All 20,736
nonterminal targets use the substituted frame; all 6,912 terminal updates use
the real terminal scalar. The control runs its own policy online and causal
runtime divergence is allowed only downstream of this intervention. It must
score at most `0.50` on each held-out macro, retain a canonical test gap at
least `0.45`, and reject the complete positive gate. The exact action tuple is
not assumed; every action and update is replayed.

### Zero-action assignment replay

This attribution control replays the already sealed canonical 27,648-record
train action/transition/feedback stream with zero policy and environment calls.
It preserves public observations, tokens, selected/executed actions,
successors, source rewards, terminal origins, `done`, and event order, but
assigns current and next update actions to exact `int8(0)`. Typed assignment
references remain evaluator-side; the updater receives only bare assigned
actions.

Only action-zero cells can become positive. Its frozen greedy policy emits
`0000`, has exact zero return in every regime, preserves a canonical test gap
at least `0.95`, and rejects the complete positive gate.

### Null terminal signal replay

This second canonical-action replay maps every terminal feedback key to a
detached null-origin record and bare `float64(0.0)` before source iteration.
Canonical public rows, tokens, selected/executed actions, successors, source
evaluator rewards, `done`, and ordering remain fixed. The null scalar
materializes only at the terminal updater boundary and has zero early attempts.

All Q bytes remain zero while visits and versions advance through 27,648
updates. The frozen policy emits `0000`, scores exact zero in every regime,
preserves a canonical test gap at least `0.95`, and rejects the complete
positive gate. This is a terminal-signal ablation, not an origin-permutation or
reward-multiset claim.

### Complete public-beacon ablation

This control copies every legal and realized train, validation, and test row
and changes only `signed_beacon` to exact `float64(0.0)` plus the derived
policy-frame beacon byte to zero. It never regenerates a row. Phase, prefix,
width, steps-left, unsigned magnitude, nuisance, keys, actions, links, tokens,
hidden evaluator beacon/formula, source/update rewards, references, `done`, and
ordering remain fixed in the static projection.

The canonical observation validator continues to require signed beacon
`+magnitude` or `-magnitude`. Ablation uses a distinct sealed
`AblatedObservation` validator and projector which permit exact `float64(0.0)`
only at coordinate one, require every other coordinate, layout, and key byte to
equal the canonical row, and emit beacon byte zero. This validator is reachable
only through an evaluator-side ablation capability issued after the static copy
is authenticated. Neither that capability nor the control domain is
policy-visible.

A fresh online learner acts on ablated train rows and is evaluated on ablated
held-out rows. The already frozen canonical learned greedy policy (the exact
canonical Q-table selector, not a target oracle) is separately evaluated on the
same ablated held-out rows with no evaluator or signed-beacon capability.
Within each held-out public pair, both policies must emit
one identical four-action code for opposite evaluator beacons, so each score is
at most `0.50` per regime. Both paths must retain a canonical test gap at least
`0.45` and reject the complete positive gate.

## Static and dynamic intervention differences

Static source comparisons cover all 253,440 legal rows, the full token
schedule, and every control specification before a policy factory. A static
projection describes frozen definitions, not a realized online trajectory.
The evaluator counterfactual is separately frozen by its twin case and is not
one of the four executable controls. The exact ordered 36-field static
projection and its 28-class field map are:

```text
structure_kind->structure
split,regime_code,local_episode->regime_identity
schedule_ordinal->schedule
evaluator_beacon->hidden_evaluator
terminal_formula->terminal_formula
signed_beacon->signed_beacon
magnitude->magnitude
nuisance->nuisance
phase,prefix_width,prefix_code->phase_prefix
policy_frame->policy_frame
action->action
predecessor_key->predecessor_key
predecessor_bytes->predecessor_bytes
predecessor_dtype,predecessor_shape,predecessor_strides->predecessor_layout
successor_key->successor_key
successor_bytes->successor_bytes
successor_dtype,successor_shape,successor_strides->successor_layout
source_reward->source_reward
update_reward->update_reward
done->done
source_identity->source_identity
token_bytes->token
target_frame->target_frame
target_reference->target_reference
origin_reference->origin_reference
assigned_action->assignment_value
assignment_reference->assignment_reference
control_mode->control_mode
```

The exact allowed static masks are field sets over that order:

```text
opposite_beacon_target = {target_frame,target_reference,control_mode}
zero_action_assignment = {assigned_action,assignment_reference,control_mode}
null_terminal_signal = {update_reward,origin_reference,control_mode}
beacon_ablation = {signed_beacon,policy_frame,control_mode}
```

Every unlisted field is byte-identical. After constructing each exact control,
the audit independently changes each one of all 36 fields once beyond its
declared canonical/control value and sends it through the real static validator:
36 fields times four controls is exactly 144 rejections. This includes permitted
fields, whose arbitrary second mutation is not licensed by the exact mask.
Count results use integer fields named `*_count`, never Boolean names.

Dynamic traces are independently authenticated per arm. Allowed causal field
classes are frozen as:

```text
policy_frame, assigned_action, update_reward, target_value,
q_or_visit, online_selection, heldout_selection, realized_environment
```

The exact ordered executable control names, dynamic arm names, and arm-to-class
map are:

```text
static_controls = [opposite_beacon_target, zero_action_assignment,
                   null_terminal_signal, beacon_ablation]
dynamic_arms = [opposite_beacon_target, zero_action_assignment,
                null_terminal_signal, beacon_ablation]

opposite_beacon_target = [target_value, q_or_visit, online_selection,
                          heldout_selection, realized_environment]
zero_action_assignment = [assigned_action, target_value, q_or_visit,
                          heldout_selection]
null_terminal_signal = [update_reward, target_value, q_or_visit,
                        heldout_selection]
beacon_ablation = [policy_frame, target_value, q_or_visit, online_selection,
                   heldout_selection, realized_environment]
```

The complete dynamic projection has exactly this 28-field order:

```text
train_real_policy_frame, train_target_policy_frame, train_token,
train_executed_action, train_assigned_action, train_source_reward,
train_update_reward, train_predecessor, train_successor, train_done,
train_target_value, q_value, visit_value, train_online_selection,
heldout_execution, train_environment_step, train_selection_key,
train_transition_key, train_feedback_key, train_update_key,
train_origin_reference, train_assignment_reference, train_target_reference,
table_version, event_ordinal, capability_digest, close_state,
source_definition_digest
```

The exact per-arm Boolean masks over that order, grouped only for readability in
four groups of seven bits, are:

```text
opposite_beacon_target =
  1101111 1101111 1101111 1100110
zero_action_assignment =
  0000100 0001110 1000010 1000110
null_terminal_signal =
  0000001 0001110 1000001 0000100
beacon_ablation =
  1101111 1101111 1101111 1100111
```

`heldout_execution` contains the complete held-out observation/action/successor/
reward trace; every other `train_*` field is train-only. `1` is the only
permission for that exact field to differ causally; `0` requires byte equality.
The registry stores both the 28 names and four exact 28-Boolean lists, not the
display strings or a union mask.

Every declared class must occur at least once; every noncausal difference count
must be zero. Replay controls preserve the canonical input selection,
environment, and source-feedback projection exactly. `heldout_selection` means
only the later frozen greedy validation/test action stream after the controlled
fit/replay; `online_selection` means only causally downstream selections in a
fresh online control. Neither class licenses a changed canonical train replay
action. Every
unmasked byte must match, and every masked class must have at least one causal
difference. A broad union whitelist is insufficient.

## Bounded result and strict scalar schema

Container notation is separate from scalar/digest notation: `LIST[T]` means an
exact JSON list with the frozen length and element order, and `MAP` or
`MAP[K->V]` means an exact JSON object with its frozen key set, keys serialized
in ascending UTF-8 order, and recursively typed values. Lists encode as
`ascii("LIST") || u32_le(length)` followed by individually
`u32_le(byte_length)||element_bytes`; maps encode as
`ascii("MAP") || u32_le(pair_count)` followed by framed UTF-8 key then framed
typed value in sorted-key order. Each scalar encoding begins with its scalar tag
byte (`B`, `I`, `F`, `S`, or `H`) before the canonical payload. This recursive
framing applies to all fixture and case-contract commitments. `D` remains only
the internal-record shorthand for exact 32 digest bytes and is never a JSON
schema tag.

The later terminal result has exactly the standard top-level keys and types:

```text
action:S, cases:MAP, environment:MAP, fixture:MAP,
schema_version:I, status:S, study_id:S
```

`study_id` is exact `online-sarsa-latched-choice-v2`; `schema_version` is exact
integer `1`; `status` is exact `passed` iff all cases pass and otherwise
`failed`; `action` is exactly the corresponding success/failure action below.
`environment` has exactly four string fields in this order after sorted JSON
encoding: `device_kind`, `jax_version`, `platform`, `python`; both CPU fields
are exact `cpu`, and version strings match
`[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}`. `cases` has exactly the twenty-three flat
scalar objects below. Every top-level, environment, fixture, case-contract, and
case key is encoded in ascending UTF-8 order; duplicate or out-of-order raw JSON
keys reject through an ID-specific pairs-preserving decoder before conversion
to dictionaries.

`fixture` has exactly `case_contract` plus these field/type pairs:

```text
action_dtype:S, action_values:LIST[I], alpha:F, case_field_types:MAP,
case_names:LIST[S], claim_boundary:S, episode_counts:MAP,
event_order:LIST[S], expected_family_sha256:H,
expected_random_stream_sha256:H, expected_schedule_sha256:H, gamma:F,
generator_regimes:LIST[MAP], horizon:I, observation_dtype:S,
observation_fields:LIST[S], observation_shape:LIST[I], policy_frame_dtype:S,
policy_frame_fields:LIST[S], policy_frame_width:I, random_baseline_seed:I,
reward_dtype:S, reward_values:LIST[F], schedule:MAP, structure_kind:S,
target_codes:MAP, thresholds:MAP, tie_action:I, updates_per_episode:I
```

Each list/dictionary is deep-exact to the registry value: `action_values` is
`[0,1]`; episode counts are train/validation/test `6912/768/768`; event order is
the 36-event chronology above; regimes are the eight ordered table rows with
exact split/code/scale/offset scalar types; shapes and field lists are the exact
ones above; schedule contains exact epochs, positions, forced/greedy and token
counts; target codes are beacon-zero `3` and beacon-one `12`; `thresholds` is
the exact decimal-float object above while `case_contract` carries its exact
reduced-ratio map; case names and field types are the exact
ordered table below. `case_contract` has exactly the twenty-three case keys and
the per-case nested schemas declared after the table. No extra nested key or
container type is allowed.

The only containers are those top-level objects and registry-declared frozen
fixture/case-contract containers. Every case is a flat scalar object. Exact type
checks use `type(value) is bool/int/float/str/list/dict`; Booleans never satisfy
integers, integers never satisfy floats, duplicate JSON keys and nonfinite
numbers reject, strings are at most 512 UTF-8 bytes, nesting depth is at most
eight, and the worker's single compact JSON object is at most 1,048,576 bytes.

The sanitizer lowercases ASCII field names and collapses runs of `-`, `_`, and
space to `_`, then rejects exactly this 36-name forbidden corpus wherever a key
appears:

```text
observation, observations, obs, raw_observation, frame, frames, policy_frame,
token, tokens, action, actions, reward, rewards, return, returns, target,
targets, successor, successors, transition, transitions, trajectory,
trajectories, log, logs, q_table, visit_table, policy_value, parameter,
gradient, topology, path, credential, secret, reference_array, private_evidence
```

The seven canonical top-level keys, four environment keys, exact fixture and
case-contract keys, twenty-three case names, and exact aggregate case fields are
path-specific exceptions only at their declared locations. Thus top-level
`action` is legal, while `action` at any other path rejects. There is no
unbounded "misspelling" rule. The exact 14-value hostile corpus is: one raw
observation ndarray, frame bytes, token bytes, action ndarray, reward ndarray,
Q ndarray, visit ndarray, a list, a nested dictionary, a 513-byte string, a
ninth-level dictionary, a Windows absolute path, a POSIX absolute path, and a
nonfinite float. Every key and value sample is rejected once through the real
sanitizer, so both sample and rejection counts are exactly 36 and 14.

### Explicit field-by-field type table

There is no suffix-based inference. Every field is explicitly tagged:

- `B`: exact JSON Boolean;
- `I`: exact JSON integer in `[0,2147483647]`;
- `F`: exact finite JSON float, with returns and gaps in `[-1.0,1.0]` and
  nonnegative metrics in `[0.0,2147483647.0]`;
- `S`: exact bounded enum string declared by fixture identity; and
- `H`: exact lowercase 64-hex SHA-256 string.

The case order and complete fields are:

```text
typed_four_step_episode:
  passed:B, structure_kind:S, horizon:I, observation_dtype:S,
  observation_rank:I, observation_width:I, policy_frame_width:I,
  action_dtype:S, action_value_count:I, reward_dtype:S, reward_value_count:I,
  done_pattern_sha256:H, event_order_sha256:H, typed_keys_ok:B,
  immutable_inputs_ok:B, invalid_action_rejection_count:I,
  invalid_action_outcome_sha256:H

fresh_generator_partition:
  passed:B, train_regime_count:I, validation_regime_count:I,
  test_regime_count:I, train_episode_count:I, validation_episode_count:I,
  test_episode_count:I, epoch_count:I, episodes_per_epoch_regime:I,
  paired_covariates_ok:B, generator_rng_call_count:I,
  family_sha256:H, schedule_sha256:H

full_family_reconstruction:
  passed:B, legal_row_count:I, nonterminal_row_count:I,
  terminal_row_count:I, predecessor_node_count:I,
  realized_row_count:I, unique_key_count:I, primary_sha256:H,
  replay_sha256:H, replay_match:B, corruption_class_count:I,
  corruption_rejection_count:I, factories_before_gate:I

detached_evaluator_counterfactual:
  passed:B, episode_count:I, public_row_check_count:I,
  evaluator_flip_count:I, terminal_reward_change_count:I,
  public_bytes_unchanged:B, detached_no_alias:B,
  self_comparison_rejected:B, copy_only_rejected:B, twin_sha256:H

public_path_separation:
  passed:B, train_path_count:I, validation_path_count:I,
  test_path_count:I, train_validation_overlap_count:I,
  train_test_overlap_count:I, validation_test_overlap_count:I,
  identity_fields_excluded:B, public_beacon_retained:B,
  projection_sha256:H

exploration_token_schedule:
  passed:B, train_episode_count:I, selection_count:I,
  greedy_episode_count:I, forced_episode_count:I,
  greedy_token_count:I, forced_token_count:I, forced_success_count:I,
  paired_token_count:I, forced_tuple_balance_ok:B, primary_sha256:H,
  replay_sha256:H, replay_match:B

sealed_policy_update_boundary:
  passed:B, policy_frame_byte_count:I, q_snapshot_byte_count:I,
  token_byte_count:I, policy_attack_class_count:I,
  policy_attack_rejection_count:I, update_attack_class_count:I,
  update_attack_rejection_count:I, lazy_attempt_count:I,
  lazy_permit_count:I, reentrancy_attempt_count:I,
  reentrancy_rejection_count:I, learner_state_unchanged:B,
  pre_attack_state_sha256:H, post_attack_state_sha256:H,
  attack_outcome_sha256:H, boundary_sha256:H

online_sarsa_chronology:
  passed:B, alpha:F, gamma:F, selection_count:I, update_count:I,
  final_table_version:I, visit_increment_count:I,
  exact_equation_count:I, latched_action_match_count:I,
  preupdate_bootstrap_match_count:I, invalid_update_rejection_count:I,
  q_dtype_ok:B, visit_dtype_ok:B, q_layout_sha256:H,
  ordinal_projection_sha256:H, update_trace_sha256:H

behavior_stream_reconstruction:
  passed:B, selection_count:I, replayed_action_count:I,
  replayed_preaction_table_count:I, action_stream_match:B,
  stale_version_rejected:B, token_mutation_rejected:B,
  duplicate_selection_rejected:B, action_stream_sha256:H,
  replay_sha256:H

one_use_capability_authentication:
  passed:B, mutation_class_count:I, mutation_rejection_count:I,
  wrong_identity_rejected:B, wrong_assignment_rejected:B,
  duplicate_append_rejected:B, cross_episode_rejected:B,
  stale_permit_rejected:B, pending_state_cleared:B,
  state_unchanged:B, mutation_outcome_sha256:H, projection_sha256:H

component_trace_join:
  passed:B, component_list_count:I, component_record_count:I,
  reordering_count:I, malformed_class_count:I,
  malformed_rejection_count:I, swapped_component_rejection_count:I,
  reordering_outcome_sha256:H,
  canonical_trace_sha256:H, score_projection_sha256:H,
  update_projection_sha256:H

bootstrap_cell_dependency:
  passed:B, nonterminal_update_count:I, shadow_count:I,
  beacon_phase_category_count:I, changed_category_count:I,
  offcell_bytes_unchanged:B, terminal_read_at_nonterminal_count:I,
  wrong_cell_rejection_count:I, shadow_projection_sha256:H

train_heldout_source_separation:
  passed:B, fitted_path_count:I, fit_construction_count:I,
  process_order_count:I, fresh_path_worker_count:I, absent_heldout_ok:B,
  heldout_spy_setup_count:I, heldout_factory_call_count:I,
  exploding_heldout_operation_count:I,
  lazy_heldout_fit_operation_count:I, inverse_train_operation_count:I,
  heldout_update_count:I, train_commitments_match:B,
  operation_trace_sha256:H, sealed_fit_sha256:H

online_control_acquisition:
  passed:B, train_episode_count:I, forced_success_count:I,
  behavior_reward_sum:I, behavior_regret:I,
  final_window_episode_count:I, final_window_success_count:I,
  final_window_regime_denominator:I, minimum_final_window_success_count:I,
  final_window_macro_return:F, minimum_final_window_regime_return:F,
  metric_count_sha256:H, final_table_sha256:H

frozen_heldout_evaluation:
  passed:B, validation_macro_return:F, test_macro_return:F,
  minimum_validation_regime_return:F, minimum_test_regime_return:F,
  validation_episode_count:I, test_episode_count:I,
  validation_success_count:I, test_success_count:I,
  heldout_regime_denominator:I, minimum_validation_success_count:I,
  minimum_test_success_count:I, metric_count_sha256:H,
  heldout_update_count:I, learner_state_unchanged:B,
  policy_sha256:H

independent_baseline_suite:
  passed:B, constant_zero_validation_return:F,
  constant_zero_test_return:F, constant_one_validation_return:F,
  constant_one_test_return:F, myopic_validation_return:F,
  myopic_test_return:F, no_bootstrap_validation_return:F,
  no_bootstrap_test_return:F, random_validation_return:F,
  random_test_return:F, random_draw_count:I,
  constant_zero_validation_success_count:I,
  constant_zero_test_success_count:I, constant_one_validation_success_count:I,
  constant_one_test_success_count:I, myopic_validation_success_count:I,
  myopic_test_success_count:I, no_bootstrap_validation_success_count:I,
  no_bootstrap_test_success_count:I, random_validation_success_count:I,
  random_test_success_count:I, metric_denominator:I,
  action_streams_match:B, replay_match:B, metric_count_sha256:H,
  random_stream_sha256:H

opposite_beacon_target_control:
  passed:B, train_update_count:I, validation_return:F, test_return:F,
  minimum_heldout_regime_return:F, canonical_test_gap:F,
  validation_success_count:I, test_success_count:I,
  minimum_heldout_regime_success_count:I, metric_denominator:I,
  canonical_test_gap_numerator:I, metric_count_sha256:H,
  positive_gate_rejected:B, target_state_swap_count:I,
  target_replay_match:B, static_transition_function_unchanged:B,
  applicable_gate_clause_count:I, failed_gate_clause_count:I,
  applicable_gate_vector_sha256:H, all_nonintervened_clauses_match:B,
  source_family_unchanged:B, intervention_sha256:H,
  runtime_trace_sha256:H

zero_action_assignment_control:
  passed:B, replay_update_count:I, train_replay_policy_call_count:I,
  train_replay_environment_step_count:I, heldout_policy_call_count:I,
  heldout_environment_step_count:I, validation_return:F, test_return:F,
  minimum_heldout_regime_return:F, canonical_test_gap:F,
  validation_success_count:I, test_success_count:I,
  minimum_heldout_regime_success_count:I, metric_denominator:I,
  canonical_test_gap_numerator:I, metric_count_sha256:H,
  positive_gate_rejected:B, action_zero_cells_only:B,
  applicable_gate_clause_count:I, failed_gate_clause_count:I,
  applicable_gate_vector_sha256:H, all_nonintervened_clauses_match:B,
  source_trace_unchanged:B, intervention_sha256:H,
  runtime_trace_sha256:H

null_terminal_signal_control:
  passed:B, replay_update_count:I, null_origin_record_count:I,
  terminal_update_sum:F, early_origin_attempt_count:I,
  validation_return:F, test_return:F,
  minimum_heldout_regime_return:F, canonical_test_gap:F,
  validation_success_count:I, test_success_count:I,
  minimum_heldout_regime_success_count:I, metric_denominator:I,
  canonical_test_gap_numerator:I, metric_count_sha256:H,
  positive_gate_rejected:B, q_values_zero:B,
  applicable_gate_clause_count:I, failed_gate_clause_count:I,
  applicable_gate_vector_sha256:H, all_nonintervened_clauses_match:B,
  source_rewards_unchanged:B, intervention_sha256:H,
  runtime_trace_sha256:H

beacon_ablation_control:
  passed:B, legal_row_check_count:I, fresh_validation_return:F,
  fresh_test_return:F, frozen_policy_validation_return:F,
  frozen_policy_test_return:F, fresh_canonical_test_gap:F,
  frozen_policy_canonical_test_gap:F, fresh_validation_success_count:I,
  fresh_test_success_count:I, frozen_policy_validation_success_count:I,
  frozen_policy_test_success_count:I, minimum_fresh_regime_success_count:I,
  minimum_frozen_policy_regime_success_count:I, metric_denominator:I,
  fresh_test_gap_numerator:I, frozen_policy_test_gap_numerator:I,
  metric_count_sha256:H,
  positive_gate_rejected:B, hidden_evaluator_unchanged:B,
  applicable_gate_clause_count:I, failed_gate_clause_count:I,
  applicable_gate_vector_sha256:H, all_nonintervened_clauses_match:B,
  intervention_sha256:H, runtime_trace_sha256:H

intervention_difference_contract:
  passed:B, static_row_check_count:I, static_control_count:I,
  static_field_count:I, static_mutation_class_count:I,
  static_mutation_rejection_count:I, dynamic_arm_count:I,
  dynamic_field_count:I, causal_field_class_count:I,
  noncausal_difference_count:I, per_arm_masks_exact:B,
  canonical_static_sha256:H,
  control_static_sha256:H, control_runtime_sha256:H

bounded_result_schema:
  passed:B, case_count:I, exact_case_fields:B,
  scalar_case_values:B, forbidden_key_sample_count:I,
  forbidden_key_rejection_count:I, forbidden_value_sample_count:I,
  forbidden_value_rejection_count:I, frozen_containers_ok:B,
  top_level_schema_ok:B, duplicate_json_key_rejected:B,
  out_of_order_json_key_rejected:B, corpus_sha256:H, schema_sha256:H

fresh_worker_reproduction:
  passed:B, worker_count:I, output_cap_bytes:I, stderr_byte_count:I,
  environment_key_set_exact:B, network_probe_count:I,
  network_probe_rejection_count:I, duplicate_json_key_rejected:B,
  process_probe_count:I, process_probe_rejection_count:I,
  file_probe_count:I, file_probe_rejection_count:I,
  native_loader_probe_count:I, native_loader_probe_rejection_count:I,
  permitted_child_launch_count:I, forbidden_file_open_count:I,
  nonprocess_projection_match:B, environment_sha256:H,
  isolation_outcome_sha256:H, network_outcome_sha256:H, projection_sha256:H
```

### Frozen scalar equalities and bounds

Known structural integers are exact, not merely descriptive:

```text
typed_four_step_episode:
  horizon=4, observation_rank=1, observation_width=6,
  policy_frame_width=4, action_value_count=2, reward_value_count=2,
  invalid_action_rejection_count=7
fresh_generator_partition:
  train_regime_count=4, validation_regime_count=2, test_regime_count=2,
  train_episode_count=6912, validation_episode_count=768,
  test_episode_count=768, epoch_count=36, episodes_per_epoch_regime=48,
  generator_rng_call_count=0
full_family_reconstruction:
  legal_row_count=253440, nonterminal_row_count=118272,
  terminal_row_count=135168, predecessor_node_count=126720,
  realized_row_count=33792, unique_key_count=253440,
  corruption_class_count=18, corruption_rejection_count=18,
  factories_before_gate=0
detached_evaluator_counterfactual:
  episode_count=8448, public_row_check_count=253440,
  evaluator_flip_count=8448, terminal_reward_change_count=16896
public_path_separation:
  train_path_count=6912, validation_path_count=768, test_path_count=768,
  every overlap count=0
exploration_token_schedule:
  train_episode_count=6912, selection_count=27648,
  greedy_episode_count=2304, forced_episode_count=4608,
  greedy_token_count=9216, forced_token_count=18432,
  forced_success_count=288, paired_token_count=13824
sealed_policy_update_boundary:
  policy_frame_byte_count=4, q_snapshot_byte_count=480, token_byte_count=4,
  policy_attack_class_count=21, policy_attack_rejection_count=21,
  update_attack_class_count=22, update_attack_rejection_count=22,
  lazy_attempt_count=2, lazy_permit_count=0,
  reentrancy_attempt_count=5, reentrancy_rejection_count=5
online_sarsa_chronology:
  alpha=0.20, gamma=1.0, selection_count=27648, update_count=27648,
  final_table_version=27648, visit_increment_count=27648,
  exact_equation_count=27648, latched_action_match_count=20736,
  preupdate_bootstrap_match_count=20736, invalid_update_rejection_count=22
behavior_stream_reconstruction:
  selection_count=27648, replayed_action_count=27648,
  replayed_preaction_table_count=27648
one_use_capability_authentication:
  mutation_class_count=26, mutation_rejection_count=26
component_trace_join:
  component_list_count=7, component_record_count=172800,
  reordering_count=7, malformed_class_count=22,
  malformed_rejection_count=92, swapped_component_rejection_count=7
bootstrap_cell_dependency:
  nonterminal_update_count=20736, shadow_count=20736,
  beacon_phase_category_count=6, changed_category_count=6,
  terminal_read_at_nonterminal_count=0, wrong_cell_rejection_count=6
train_heldout_source_separation:
  fitted_path_count=5, fit_construction_count=15, process_order_count=2,
  fresh_path_worker_count=5, heldout_spy_setup_count=10,
  heldout_factory_call_count=0, every forbidden source operation count=0,
  heldout_update_count=0
online_control_acquisition:
  train_episode_count=6912, forced_success_count=288,
  final_window_episode_count=384, final_window_regime_denominator=96
frozen_heldout_evaluation:
  validation_episode_count=768, test_episode_count=768,
  heldout_regime_denominator=384, heldout_update_count=0
independent_baseline_suite:
  every constant/myopic/no-bootstrap success count=0,
  random_draw_count=8448, metric_denominator=768
opposite_beacon_target_control:
  train_update_count=27648, target_state_swap_count=20736,
  metric_denominator=768, applicable_gate_clause_count=32
zero_action_assignment_control:
  replay_update_count=27648, train_replay_policy_call_count=0,
  train_replay_environment_step_count=0, heldout_policy_call_count=6144,
  heldout_environment_step_count=6144,
  validation_success_count=0, test_success_count=0,
  minimum_heldout_regime_success_count=0, metric_denominator=768,
  applicable_gate_clause_count=32
null_terminal_signal_control:
  replay_update_count=27648, null_origin_record_count=6912,
  terminal_update_sum=0.0, early_origin_attempt_count=0,
  validation_success_count=0, test_success_count=0,
  minimum_heldout_regime_success_count=0, metric_denominator=768,
  applicable_gate_clause_count=32
beacon_ablation_control:
  legal_row_check_count=253440, metric_denominator=768,
  applicable_gate_clause_count=32
intervention_difference_contract:
  static_row_check_count=1013760, static_control_count=4,
  static_field_count=36, static_mutation_class_count=28,
  static_mutation_rejection_count=144, dynamic_arm_count=4,
  dynamic_field_count=28, causal_field_class_count=8,
  noncausal_difference_count=0
bounded_result_schema:
  case_count=23, forbidden_key_sample_count=36,
  forbidden_key_rejection_count=36, forbidden_value_sample_count=14,
  forbidden_value_rejection_count=14
fresh_worker_reproduction:
  worker_count=2, output_cap_bytes=1048576, stderr_byte_count=0,
  network_probe_count=10, network_probe_rejection_count=10,
  process_probe_count=10, process_probe_rejection_count=10,
  file_probe_count=8, file_probe_rejection_count=8,
  native_loader_probe_count=3, native_loader_probe_rejection_count=3,
  permitted_child_launch_count=2, forbidden_file_open_count=0
```

All success numerators not fixed above are deterministic result-bearing values
bounded inclusively by their declared denominator. Every `failed_gate_clause_count`
is in `[1,32]` when its control passes. Gap numerators equal canonical test
successes minus the named control successes and must be nonnegative. Every
Boolean required by a passing case is exact `true`; all SHA fields use the ABI
above. These equalities, bounds, and algebraic relations are encoded as
ID-specific controller checks, not trusted from worker `passed` values.

The later registry stores this exact case order, exact field order, and exact
type tag for every field. No validator may infer a type from spelling or a
suffix. Count equalities compare only `I` fields; rejection Booleans compare
only `B` fields. `terminal_update_sum` is exact float `0.0`; alpha is exact
float `0.20`; gamma exact float `1.0`; all other field-specific constants and
bounds come from this plan and fixture identity. No case value is a list or
dictionary.

The exact registry `case_contract` has the same twenty-three ordered case names
as the table. Every entry has exactly these common nested keys and types:

```text
contract_version:I(1), ordered_fields:LIST[S], field_types:MAP[S->S],
exact_values:MAP[S->scalar], integer_bounds:MAP[S->LIST[I,I]],
float_derivations:MAP[S->S], digest_domains:MAP[S->S]
```

`ordered_fields` and `field_types` repeat the table exactly. `exact_values`
contains every frozen equality above; `integer_bounds` contains `[0,denominator]`
for each result-bearing numerator and `[1,32]` for each failed control gate
count; `float_derivations` contains the exact numerator/denominator expression
for every `F` field; `digest_domains` binds every `H` field to the canonical ABI
domain and projection. Empty dictionaries are present when a category has no
member. No suffix inference or omitted default is allowed.

These cases additionally have the exact named metadata:

- `full_family_reconstruction`: ordered 18-name corruption list;
- `sealed_policy_update_boundary`: ordered 21-name policy/source list, ordered
  22-name update/trace list, and exact spy/stage map;
- `online_sarsa_chronology`: ordinal formulas and exact Q/visit offset formulas;
- `one_use_capability_authentication`: ordered 26-name mutation list and exact
  capability consumer map;
- `component_trace_join`: ordered 22-name malformed list, exact seven-record
  schemas/absent rules, seven rotation parameters, and exact 92-pair
  applicability matrix;
- `train_heldout_source_separation`: ordered five fitted paths, three source
  modes, two process orders, and per-operation counter names;
- each executable control: its domain, the ordered 32-clause gate vector, exact
  all-applicable mask, and threshold-ratio map;
- `intervention_difference_contract`: the ordered four control names, 36-field
  static projection, 28-class static map, four dynamic arms, eight causal field
  classes, 28-field dynamic projection, and exact per-arm Boolean masks;
- `bounded_result_schema`: exact 36-key and 14-value sanitizer corpora; and
- `fresh_worker_reproduction`: workers `2`, timeout `1200`, output cap
  `1048576`, exact environment keys, ten network probes, ten unpermitted process
  probes, eight file probes, three native-loader probes, and the exact two-use
  child-launch permit.

Missing, extra, reordered, duplicated, or wrong-typed metadata rejects before
worker launch. The ID-specific controller deep-validates this complete contract
and all result relations; generic case-field membership is insufficient.

The frozen fixture identity contains exactly: action dtype/values; alpha;
claim boundary; episode counts; event order; expected family/schedule/random
digests; gamma; ordered generator regimes; horizon; observation dtype/fields/
shape; policy-frame dtype/fields/width; random seed; reward dtype/values;
schedule epochs/positions/counts; structure kind; target codes; thresholds;
tie action; updates per episode; ordered case names; ordered field-type map;
and `case_contract`. The registry later binds exactly five committed sources:
`uv.lock`, the V2 fixture, `docs/AUTONOMOUS_LAB.md`, this frozen plan, and the
dedicated V2 worker. No sixth helper source is permitted. `studies.json` and
`tools/run_local_lab.py` are controller infrastructure rather than worker
sources: the implementation checkpoint must nevertheless commit their exact V2
registry entry, dedicated worker allowlist, ID-specific raw-JSON/type/relation
validator, per-study output cap, stderr rule, approved-source hashes, and the
normalized pinned registry digest in the same clean pre-result revision. Their
git blobs and the registry digest are checked before invocation.

The implementation checkpoint creates a separate infrastructure manifest in
its dated pre-result record containing exact lowercase SHA-256 and Git blob OID
for `experiments/local_lab/studies.json` and `tools/run_local_lab.py`, plus the
normalized registry digest and the clean implementation commit's parent plan
OID. These are not counted among the five worker-source approvals. The allowed
infrastructure diff is closed now:

- `studies.json`: add only the single V2 entry, its exact five source hashes,
  fixture identity, 23-case field/type/contract maps, worker identity/actions,
  and the one normalized top-level registry digest change;
- `tools/run_local_lab.py`: add only the V2 worker-module/path pair, V2 scalar
  type/relation constants, V2 pairs-preserving decoder and output/stderr cap,
  V2 process/file/network validation, and the normalized pinned digest; and
- no existing study entry, generic decision rule, protected hash, timeout,
  lease/state/event protocol, or other worker allowlist entry may change.

Focused tests compare both files against that symbol/key-level allowed-diff
contract and their manifest OIDs. A later controller invocation is authorized
only at the exact clean PR head revision whose parent is the plan commit and
whose two infrastructure blobs and normalized digest equal the manifest.

## Fresh-process and network boundary

A later dedicated worker receives exactly these environment keys and no others:

```text
COMSPEC, LD_LIBRARY_PATH, PATH, PATHEXT, SYSTEMROOT, TEMP, TMP, TMPDIR,
VIRTUAL_ENV, WINDIR, CUDA_VISIBLE_DEVICES, JAX_PLATFORMS,
LEARN2DESIGN_LOCAL_LAB_NETWORK, PYTHONHASHSEED,
XLA_PYTHON_CLIENT_PREALLOCATE
```

The first ten values are copied from the controller or exact empty strings when
absent. The last five are exactly `""`, `"cpu"`, `"disabled"`, `"0"`, and
`"false"`. Credential-like and all unlisted variables are absent. Before
fixture import, both parent and children install fail-closed probes for exactly:

```text
socket_constructor, getaddrinfo, gethostbyname, create_connection, connect,
connect_ex, send, sendall, sendto, recvfrom
```

Each probe is attempted once and must reject. The worker emits one UTF-8,
lexicographically sorted, separators-compact JSON object no larger than exactly
1,048,576 bytes and zero stderr bytes. The ID-specific controller uses an
`object_pairs_hook` which rejects duplicate and out-of-order keys at every
depth, rejects nonfinite numeric tokens before object construction, and applies
the 1 MiB limit to stdout alone plus the zero-stderr rule; the generic 5 MiB
retention cap remains only an outer emergency ceiling.

Network denial is not the only isolation boundary. After required interpreter
and dependency imports, a fail-closed broker permits reads only from the exact
five approved source files, the running Python executable/standard-library and
locked environment roots, and the task-specific empty scratch directory. It
permits no writes except broker-owned scratch files; the V2 worker itself writes
none. Path resolution is realpath-based and rejects symlink/junction escapes.
The exact ordered file probes are:

```text
private_control_path, private_generated_result_path, official_dataset_path,
protected_submission_path, git_config_path, unapproved_repo_path,
outside_scratch_write, symlink_or_junction_escape
```

They are synthetic path attempts resolved before open and never inspect the
target. The exact unpermitted process probes are:

```text
subprocess_popen, subprocess_run, subprocess_call, subprocess_check_call,
subprocess_check_output, os_system, os_popen, os_spawnv, os_startfile,
multiprocessing_process
```

The parent fixture alone owns a non-serializable two-use `ChildProcessPermit`
which authorizes only the exact current Python executable, dedicated V2 worker
module, nonrecursive mode, scrubbed environment, no shell, and new process
group. No child receives it. The exact native-loader probes are `ctypes_cdll`,
`ctypes_pydll`, and `ctypes_windll`. Every one of the 8 file, 10 unpermitted
process, and 3 native-loader probes rejects with no child/file/network effect;
the two permitted child launches are separately authenticated.

The fixture sequentially launches exactly two fresh child workers in
nonrecursive trace mode, new process groups, no shell, and a 1,200-second
timeout each. The parent applies the same pairs-preserving decoder and rejects
extra stdout, any stderr byte, timeout, surviving child/process tree, wrong
environment/CPU fields, malformed projection, or byte mismatch. Both complete
bounded non-process projections must be byte-identical before the scalar
`fresh_worker_reproduction` case is added.

This plan checkpoint creates no worker, fixture, registry entry, source
approval, controller allowlist, result, or sidecar.

The later implementation is developed on a focused branch based exactly on the
immutable plan commit and uses a separate clean pre-result commit. It pushes to
a draft PR stacked on the V1-rejection PR, never amends the plan commit, never
executes the learner or controller in that checkpoint, and cannot merge itself
or any stacked lab PR. A result-bearing heartbeat is a later gate after green
CI and fresh owner/live-gate authorization.

## Stopping rule and actions

This plan commit is the entire checkpoint. It is created on
`codex/lab-online-sarsa-latched-choice-v2-plan` based exactly on the clean PR
#33 head, pushed without amendment, and opened as a draft successor PR whose
base is `codex/lab-online-sarsa-latched-choice-v1-plan`. Only this plan and the
minimal truthful `docs/CURRENT_HANDOFF.md` transition belong in that commit.
The older README links to V1 are historical status prose; because this gate
authorizes only the plan plus handoff, `docs/CURRENT_HANDOFF.md` must explicitly
mark V1 rejected and is the authoritative live gate until a later documentation
checkpoint may update those summaries.

A later heartbeat may implement only
this contract in a fresh self-contained fixture and dedicated network-disabled
worker beneath `experiments/local_lab`. It must add focused tests, the exact
23-case registry/type contract, exactly five committed-source approvals, only
the V2 worker allowlist entry, and the normalized pinned registry digest; pass
independent hostile family/chronology, leakage/control/schema, and repository
audits; and obtain a separate clean pre-result commit. It must not invoke
`tools/run_local_lab.py` in that checkpoint.

Only a still later heartbeat, after green CI, exact source/revision/protected
hash checks, clean worktree, `awaiting_study`, absent stop marker, absent lease,
and confirmation that V2 has never run, may invoke the guarded controller at
most once on local CPU. Direct fixture or worker execution for result-bearing
metrics is forbidden. Every terminal or quarantined ID remains excluded.

Success requires all twenty-three cases and every positive/control gate and
uses action:

`synthetic_online_sarsa_token_control_confirmed_for_harness`

Any failed threshold or invariant, control recovery, malformed or
nondeterministic result, process mismatch, timeout, source drift, or terminal
error uses:

`park_online_control_v2_research`

A substantive confound found before controller execution produces no terminal
result: quarantine V2, keep the controller `awaiting_study`, and require a
fresh versioned plan. There is no retry, top-up, alternate seed, relaxed rule,
case removal, or same-ID repair.

The protected submission tree must remain
`e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`; the protected ZIP and manifest
must remain, respectively,
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`
and `99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`.

## Claim boundary

A future pass may say only that this fixed local-CPU synthetic harness
generated the declared family, selected actions online under the frozen
greedy/forced token schedule, applied exact SARSA(0) next-action updates,
acquired the deliberately exposed four-action beacon mapping, froze the policy
on untouched generator regimes, beat the frozen toy comparators, and lost the
positive gate under the exact bootstrap-target, assignment, terminal-signal,
and beacon interventions.

It cannot support a claim about learning being necessary, absence of a public
shortcut, optimal exploration, sample efficiency outside this schedule,
general or production RL, meta-RL, partial observability, official data,
private or hidden topology, UIFO, the submitted optimizer, candidate selection,
a native rewrite, accelerator value, leaderboard rank, competition score, or
permission to change or upload the protected submission.
