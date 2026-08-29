# Frozen plan: four-step synchronous TD action-prefix mechanics

Status: **frozen before implementation or any learner execution**
Study ID: `multistep-td-action-prefix-v3`
Date frozen: 2026-08-28
Execution budget: at most one guarded local-CPU terminal invocation, only after
a separate clean pre-result implementation commit

## Closed predecessors and design independence

This is a fresh study, not a repair, continuation, or rerun of either rejected
predecessor:

- `multistep-td-propagation-v1` exposed prior target agreement in public state;
- `multistep-td-action-prefix-v2` had false-positive target-swap, baseline,
  held-out, authentication, timing, control-difference, and dependency checks.

Neither rejected source or worker may be imported, registered, executed, or
used as a fixture. No threshold, regime, seed, mapping, schema, attack, or case
below was selected against a v1 or v2 development metric. The generator has no
RNG; its 16-block full-factorial symmetry determines the exact behavior and
control expectations analytically. The sole random-baseline seed is the fixed
decimal prefix `314159265`, chosen before implementation and used only for
baseline replay.

The implementation must treat this committed plan as immutable. A substantive
confound discovered before terminal execution quarantines this ID and requires
another fresh plan; it is not permission to alter this contract in place.

## Narrow falsifiable question

Can a blank deterministic tabular fitted-TD learner, using only authenticated
train behavior feedback, propagate a terminal signal backward through all
three nonterminal bootstrap boundaries of a four-action process with
target-independent successor dynamics and public action-prefix state; achieve
the deliberately learnable optimal policy on untouched generator regimes; beat
frozen constant, feedback-only myopic, no-bootstrap, and seeded-random
comparators; and reject the same positive gate under transition-target,
reward-origin, and signal-ablation controls?

A pass would validate only this synthetic synchronous-TD harness and toy
propagation signal. It is not evidence about production RL, meta-RL, the
competition optimizer, hidden topology, UIFO, candidate performance, native
rewrites, accelerators, leaderboard standing, or score.

## Topology-independent generator contract

The family contains no topology, graph, network, official archive, UIFO input,
candidate array, private evidence, or provider input. Its structure kind is
exactly `none`. Every episode has four actions, three nonterminal zero rewards,
and one evaluator-only terminal reward.

There are exactly eight regimes:

| split | code | signal scale | nuisance shift | nuisance scale |
|---|---:|---:|---:|---:|
| train | 1009 | 0.72 | -1.10 | 0.82 |
| train | 1013 | 0.91 | -0.35 | 1.07 |
| train | 1019 | 1.13 | 0.35 | 0.74 |
| train | 1021 | 1.34 | 1.10 | 1.19 |
| validation | 2003 | 0.63 | -1.60 | 0.67 |
| validation | 2011 | 1.43 | 1.60 | 1.31 |
| test | 3001 | 0.54 | -2.10 | 0.58 |
| test | 3011 | 1.52 | 2.10 | 1.42 |

Each regime has exactly 512 episodes: 16 blocks of 32. For local episode
`e in 0..511`:

```text
block = e // 32                         # 0..15
cell = e % 32                           # 0..31
action_code = cell // 2                 # 0..15
target_slot = cell % 2                  # 0 or 1
sign = -1.0 if target_slot == 0 else +1.0
target = target_slot                    # evaluator only
magnitude = signal_scale * (1.0 + block / 64.0)
nuisance = nuisance_shift
         + nuisance_scale
         * ((((11*block + 5*action_code + code) % 37) - 18) / 18.0)
```

The behavior action tuple is the four-bit big-endian expansion of
`action_code`:

```text
a0 = (action_code >> 3) & 1
a1 = (action_code >> 2) & 1
a2 = (action_code >> 1) & 1
a3 = action_code & 1
```

Thus every block contains every four-action tuple once for each target slot.
The behavior policy is derived only from the completed train-episode index; it
does not receive a target, signal formula, split, key, evaluator, or reward.

The terminal evaluator returns exactly:

```text
terminal_reward = 1.0 iff a0 == a1 == a2 == a3 == target else 0.0
```

There are exactly 2,048 train, 1,024 validation, and 1,024 test episodes.
Canonical train behavior therefore has 128 successes, mean return `0.0625`,
regret `1920`, 8,192 transition records, 6,144 nonterminal zero rewards, and
2,048 terminal rewards. No generated scalar is selected from observed output.

## Public action-prefix state

At phase `p in 0..3`, the public observation is an immutable C-contiguous
`float64[6]` array with fields:

```text
(phase, signed_signal, prefix_a0, prefix_a1, prefix_a2, nuisance)
```

Unused prefix slots are exactly `-1.0`; committed actions are exactly `0.0` or
`1.0`. The legal forms are:

```text
phase 0: [0, sign*magnitude, -1, -1, -1, nuisance]
phase 1: [1, sign*magnitude, a0, -1, -1, nuisance]
phase 2: [2, sign*magnitude, a0, a1, -1, nuisance]
phase 3: [3, sign*magnitude, a0, a1, a2, nuisance]
```

After a valid action at phases zero through two, the next public state appends
that action and changes only the phase/prefix fields above. This successor is a
pure function of already public bytes and the chosen action. It never depends
on target, correctness, reward, future action, donor, origin, or control mode.
After the phase-three action there is no successor observation and `done` is
true. Terminal success remains evaluator-only.

The initial signed signal is deliberately predictive in this toy task and is
carried forward byte-for-byte. “Target-independent successor” means that,
conditional on the already public predecessor and selected action, successor
construction has no target or correctness input. It does not mean the intended
toy signal is uninformative. The sign directly encodes the hidden evaluator
target in this deliberately learnable family, so this fixture cannot establish
that delayed credit is necessary, that public state is target-independent, or
that a target-correlated shortcut is absent.

The learner maps observations to the abstract state key
`(sign_bin, phase, prefix_bits)`, where `sign_bin` is zero for a nonpositive
signal and one for a positive signal. Magnitude and nuisance are deliberately
ignored by this frozen abstraction. Across both signs there are exactly 30
abstract states and 60 state-action cells:

```text
2 signs * (1 + 2 + 4 + 8 prefix states) * 2 actions = 60 cells
```

No episode key, regime, split, block, cell, exact magnitude, nuisance, target,
counter, donor, or origin may enter the value-table key.

## Complete legal-family commitment and fail-closed replay

Before constructing a behavior collector, comparator, or TD learner, the
implementation must enumerate every legal state/action row for every episode,
not only the realized behavior path. Each of the 4,096 episodes contributes:

```text
phase 0: 1 prefix * 2 actions = 2 rows
phase 1: 2 prefixes * 2 actions = 4 rows
phase 2: 4 prefixes * 2 actions = 8 rows
phase 3: 8 prefixes * 2 actions = 16 rows
total: 30 rows
```

The complete projection therefore contains exactly 122,880 rows: 57,344
nonterminal rows and 65,536 terminal rows, over 61,440 predecessor nodes.
Every row commits all of:

- split tag, regime code, episode, block, cell, and target slot;
- exact typed predecessor key and predecessor observation bytes;
- action key, scalar `int8` action bytes, and action-code lineage;
- for a nonterminal row, exact typed successor key, successor bytes, dtype,
  shape, strides, C-contiguity, immutability, legality, reward `0.0`, and
  `done=false`;
- for a terminal row, absent successor, evaluator terminal reward, reward
  dtype, and `done=true`;
- canonical donor identity `none` and canonical reward-origin identity equal
  to the source episode.

Keys are unique, typed, contiguous within each episode, and include the full
prefix identity. `bool` must never compare equal to an integer identity. The
projection is serialized canonically and committed by SHA-256.

A separately implemented replay function, sharing no row constructor with the
primary generator, must reproduce all 122,880 rows and the same digest. It must
also independently prove formula conformance, row counts, unique keys, legal
successors, exact terminal outcomes, target/action balance, and no exact public
observation or realized-path digest shared between train, validation, and test.
The 4,096 canonical behavior paths produce exactly 16,384 realized transition
rows; their keyed path commitments must be pairwise unique and split-disjoint.

The disjointness digests are computed from public observation payloads,
dtype/layout, actions, successor payloads or terminal sentinel, and `done` only.
They exclude split, regime, episode, every key, and all evaluator-only identity,
target, and reward fields, so identity metadata cannot make the proof pass.

The learner-factory and collector-factory counters begin at zero. Any family,
replay, target-swap, type, count, legality, uniqueness, or split-disjointness
failure must raise the frozen contract error while both counters remain zero.
Tests must corrupt every committed row class in turn and prove fail-closed
behavior before learner construction.

The complete family SHA-256 is not a tunable choice. After implementing only
the exact formulas above, it must be computed by the generator-only audit,
committed in the registry and focused tests, and then independently replayed
before any learner is allowed to execute. A mismatch quarantines v3.

## Genuine target-swap twins

For every canonical episode, the target-swap test first freezes and hashes all
public observation, action, transition, key, link, dtype/layout, and `done`
bytes. It then constructs a twin by copying those already frozen bytes and
flipping only the evaluator target from zero to one or one to zero. It must not
call the canonical observation or transition generator after the flip.

The canonical `target_slot` and its derived public signed signal remain frozen
in the twin; the flipped value is a separate hidden evaluator-target field.
This distinction is mandatory: regenerating the sign from the flipped target,
or treating `target_slot` itself as the flipped field, invalidates the twin.

Across all legal action rows, the twin must preserve every public byte, key,
link, action, successor, donor, origin slot, and `done` value. Only the hidden
target commitment and terminal rewards may differ, and the reward difference
must exactly match the closed terminal formula for the original and flipped
targets. Per episode exactly the all-zero and all-one terminal action rows swap
their reward values; every other terminal row remains zero. A self-comparison,
copy-only comparison without a target flip, regenerated public row, or broader
field difference is a hard failure before learner construction.

## Typed runtime contracts and information boundary

The exact key schemas are:

- `EpisodeKey(split_enum:uint8, regime_code:int32, episode:int16)`;
- `ObservationKey(episode_key, phase:int8, prefix_code:int8)`;
- `ActionKey(observation_key, action_ordinal:int8)`;
- `TransitionKey(action_key, transition_ordinal:int8)`; and
- `FeedbackKey(transition_key, feedback_ordinal:int8)`.

The episode key intentionally does not contain target, target slot, action
code, block, or cell. Those remain separately authenticated evaluator fields,
so a target-swap twin can preserve every key. Prefix codes use only the low
`phase` bits and their unused high bits must be zero. All integer fields use
the exact declared NumPy width and little-endian byte representation.

The exact scalar and record contracts are:

- observation: immutable finite C-contiguous little-endian `numpy.dtype('<f8')`
  array, shape `(6,)`, exact strides `(8,)`, exact legal phase/prefix form;
- action: zero-dimensional scalar `numpy.dtype('i1')`, value zero or one; Python
  integers, `bool`, `numpy.bool_`, floats, one-element arrays, wrong dtypes,
  and out-of-range values are rejected before any lazy value resolves;
- nonterminal reward: scalar little-endian `numpy.dtype('<f8')` value `0.0`
  with `done` exactly Python
  `False`;
- terminal reward: scalar little-endian `numpy.dtype('<f8')` in `{0.0, 1.0}`
  with `done` exactly Python `True`;
- donor and origin: exact typed episode-key objects or the exact `None`
  sentinel, never integers or Booleans;
- every metadata integer: exact declared NumPy integer dtype; no coercion.

At action selection, the policy receives an immutable object with exactly one
field, `observation`. It never receives target, preferred action, reward,
counterfactual reward, split, regime, episode/block/cell/key, phase outside the
observation, action code, done, successor, future observation, transition,
trajectory, log, RNG, generator, evaluator, environment handle, control mode,
donor, origin, or lazy value.

At train collection, the authenticated transition sink receives only the
validated predecessor observation, exact action, then the resolved successor
and scalar feedback in canonical order. The complete keyed validator may inspect
evaluator-only commitments, but after validation it emits a sealed `TDInputRow`
projection containing exactly the public predecessor observation, exact action,
public successor observation or terminal sentinel, bare scalar `update_reward`,
and exact Boolean `done`. The TD fitter receives only those projections. It
cannot access canonical evaluator reward separately, target, target slot,
origin, donor, hidden key, split/regime/episode identity, action code, block,
cell, counter, order-derived metadata, generator, evaluator, or environment.
Held-out evaluation gives a frozen policy only the current observation and
performs no update.

## Physical event order and exact attack matrix

The canonical event order for every episode is:

```text
observe0, select0, validate0, resolve_successor0, resolve_zero0, append0,
observe1, select1, validate1, resolve_successor1, resolve_zero1, append1,
observe2, select2, validate2, resolve_successor2, resolve_zero2, append2,
observe3, select3, validate3, resolve_terminal3, append3, close_episode
```

Each real environment step owns operation-counting lazy successor and reward
objects. They are physically connected to the same path used by canonical
collection. Every lazy object records installation count, attempted-access
count, permitted-materialization count, first/last attempted stage, and exact
value digest. Selection runs under a reentrancy guard. A valid action must cause
exactly the declared resolution events after validation; an invalid action or
reentrant call must cause none. A named lazy-boundary check with zero canonical
permitted materializations is vacuous and fails. Origin keys for the reward
control may be prepared from indices, but their scalar values may materialize
only in the `resolve_terminal3` slot.

The exact timing/protocol attack set is:

1. `successor_during_select`
2. `reward_during_select`
3. `invalid_action_successor_resolution`
4. `invalid_action_reward_resolution`
5. `successor_before_action_validation`
6. `nonterminal_terminal_scalar`
7. `missing_nonterminal_zero`
8. `duplicate_nonterminal_feedback`
9. `next_transition_before_pending_append`
10. `terminal_scalar_before_phase_three_action`
11. `terminal_scalar_after_extra_transition`
12. `duplicate_terminal_scalar`
13. `origin_resolution_before_terminal`
14. `origin_bearing_policy_feedback`
15. `reentrant_select_environment_step`
16. `reentrant_update_callback`
17. `duplicate_episode_close`
18. `nonempty_pending_transition_at_split`
19. `nonempty_origin_queue_at_split`
20. `heldout_source_during_fit`

Every attack must traverse the real source/environment/fitter boundary it
claims to test, assert exact lazy operation counts before and after rejection,
raise the declared contract error, leave the learner/table unchanged, and
leave no pending transition, episode, reward, or origin. A protocol-stage
substitute that cannot reach the named lazy boundary does not satisfy the case.

## Exact pending-transition and keyed-trace authentication

The transition sink stores the exact pending predecessor key, predecessor
typed-byte commitment, action key/value, expected successor key and typed-byte
commitment (or exact terminal sentinel), expected feedback slot, source episode
identity, and control-specific donor/origin expectation. Append is legal once
and only once when every component matches that pending identity exactly.
Matching only phase, prefix, shape, or value class is insufficient.

The sealed trace contains separately keyed observation, action, transition,
feedback, and episode-close components. The scorer and TD fitter must join each
component by exact typed keys, independently of list order. A second replay
independently reorders each component with a different fixed nonidentity
permutation and must reproduce the exact score and TD input commitments.

The exact malformed/recombination attack classes are:

- missing, duplicate, unknown, malformed, and cross-episode component keys;
- wrong split, regime, episode, phase, prefix, predecessor, successor, action,
  source, donor, origin, reward slot, or close link;
- wrong observation/action/reward dtype, shape, strides, byte order,
  C-contiguity, immutability, scalar rank, or digest;
- Python/NumPy Boolean substituted for integer identity or action;
- integer substituted for `done`, donor, origin, or exact Boolean;
- wrong action value, canonical reward, update reward, `done`, donor, origin,
  terminal-success formula, or nonterminal zero;
- independently swapped observations, actions, transitions, feedback, donor
  records, origin records, or episode closes;
- a valid component from the paired target twin or another generator regime.

Every attack must reject deterministically through the same validator used by
the TD fitter. Broad exception swallowing, aggregate-only comparison, or a
shared permutation across all component lists is not admissible.

## Train-only and untouched-held-out boundary

The real orchestration function accepts a train source plus optional validation
and test source handles, but the fitter API accepts only an authenticated train
iterator. Every source records factory, iterator, row-materializer, close, and
post-close operation counts. The same fitter path must run in three fresh
constructions:

1. validation and test handles absent;
2. validation and test handles present as operation-counting exploding sources;
3. validation and test handles present but unopened until after fit seals.

All three runs must have byte-identical sealed train-trace and complete learner
state commitments. Exploding-source operation counts must remain exactly zero.
The sources are not decorative: they are the actual handles later used by the
same orchestration path for held-out evaluation after the fit seal.

The inverse held-out check installs an exploding train source after the fit
seal and then runs the real validation/test path. Its train factory, iterator,
materializer, fitting, and update counts must remain zero while every expected
held-out row is evaluated exactly once.

Validation and test are entire untouched regimes. They cannot drive fitting,
normalization, thresholding, early stopping, checkpoint choice, tie behavior,
abstraction choice, sweep count, baseline choice, or policy selection. The
complete learner state is hashed before validation, after validation, and after
test and must remain byte-identical. Held-out update count is exactly zero.

## Frozen synchronous fitted-TD learner

The learner starts with all 60 `float64` Q cells exactly zero. The train trace
contains every abstract state-action cell with balanced coverage. There are
exactly four synchronous fitted-TD sweeps with discount `gamma = 1.0`.

Across the four train regimes, every abstract state-action cell occurs exactly
512 times at phase zero, 256 times at phase one, 128 times at phase two, and 64
times at phase three. The independent family replay authenticates these exact
per-cell counts before constructing the learner.

Before sweep one, one typed aggregation pass groups the terminal `TDInputRow`s
by abstract phase-three state/action cell. It requires exactly 64 authenticated
records per cell, reads every bare terminal `update_reward` exactly once (2,048
raw scalar reads total), and writes an immutable 32-cell `TerminalCellMean`
table. No raw terminal update scalar remains reachable by the sweep kernel.

For sweep `k in 1..4`, every target is computed only from the immutable table
`Q[k-1]` and written into a separate zero-initialized shadow table `Q_next`:

```text
phase 3 target = authenticated TerminalCellMean[cell]
phase 0..2 target = mean(0.0 + max_a Q[k-1][successor_state, a])
Q_next[state, action] = target
```

Targets are computed in lexicographic `(sign, phase, prefix, action)` order for
logging, but no write to `Q_next` may be read during the same sweep. After all
60 targets are authenticated, one atomic swap makes `Q[k] = Q_next`. There is
no eligibility trace, Monte Carlo return, backward replay, in-place update,
terminal-reward parameter on a nonterminal target function, optimizer, learned
representation, stochastic exploration, or hidden-state key.

The exact write count is 60 per sweep and 240 total. Under the canonical train
trace, the numbers of strictly positive cells after sweeps one through four are
exactly `[2, 4, 6, 8]`. The two positive terminal leaves are the all-zero leaf
for the negative signal and all-one leaf for the positive signal. Each later
sweep adds exactly the two next ancestors; no earlier boundary may change.

The greedy policy uses the greater Q value and exact action-zero tie break.
After sweep four it chooses four zeros for a negative signal and four ones for
a positive signal, so canonical post-fit train, validation, and test return are
analytically `1.0`.

## All-boundary terminal-dependency proof

Canonical traces are sealed and must reject any changed update reward. After a
canonical trace passes complete authentication, a separate preflight-only
`DependencyProbeManifest` binds the family digest, canonical trace digest,
selected leaf, the exact 64 repeated train records for that abstract leaf cell,
base terminal update reward `1.0`, probe terminal update reward `0.0`, the sole
allowed difference `terminal_update_reward` on all and only those 64 records,
the base/probe digests, and the exact expected changed cells after every sweep.
A dedicated
TD-input projection validator accepts only that declared scalar substitution
and rejects every other mutation. Neither projection may enter the canonical
scorer, positive metrics, collector, comparator, or production learner path.

For each of the two successful terminal leaves independently, the pure
dependency validator starts from the immutable canonical aggregate table and
builds base and probe replacements for the selected cell. For each replacement
it reads all and only the 64 selected-leaf raw scalars exactly once, reads zero
raw scalars from every other terminal cell, and authenticates the resulting
mean. The pure four-sweep kernel then runs on the base and probe aggregate
tables. The dependency case compares every Q cell after every sweep and proves:

- sweep one: only the selected phase-three leaf cell changes;
- sweep two: exactly that leaf and its phase-two ancestor change;
- sweep three: exactly those cells and the phase-one ancestor change;
- sweep four: exactly those cells and the phase-zero ancestor change;
- the opposite-sign chain and every off-chain cell remain byte-identical;
- each selected-leaf projection performs exactly 64 authenticated raw-scalar
  reads and zero raw-scalar reads from other terminal cells;
- the sweep kernel performs exactly one aggregate lookup per phase-three cell
  per sweep and exposes the aggregate table only to the phase-three target
  branch;
- no nonterminal target function receives, reads, or can materialize a
  terminal scalar directly.

The proof runs for both signs and therefore authenticates every one of the
three bootstrap boundaries twice. Testing only successor-Q sensitivity at one
boundary is insufficient.

## Metrics and complete positive gate

Primary performance is mean terminal return, averaged within each complete
regime and then macro-averaged with equal regime weight. Minimum held-out
regime return is retained. Regret is episode count minus reward sum. Every arm
and comparator is evaluated on identical keyed canonical episodes.

The complete positive gate requires all of:

- canonical behavior train mean exactly `0.0625`, reward sum exactly `128`,
  and regret exactly `1920`;
- exactly 8,192 train transition records and zero held-out updates;
- exactly four synchronous sweeps, 60 writes per sweep, 240 writes total, and
  positive-cell counts exactly `[2, 4, 6, 8]`;
- post-fit train, validation, and test macro return at least `0.99`;
- minimum validation/test regime return at least `0.98`;
- validation and test gain over the better constant at least `0.30`;
- validation and test gain over the feedback-only myopic comparator at least
  `0.30`;
- validation and test gain over the no-bootstrap comparator at least `0.30`;
- validation and test gain over seeded random at least `0.30`;
- byte-identical learner state before validation, after validation, and after
  test;
- every family, type, source, timing, pending, trace, scoring, attribution,
  difference-whitelist, sanitizer, and process-isolation invariant passing.

Each negative control is evaluated by this entire gate, not only its held-out
score clauses, and must reject it.

## Frozen feedback-only comparators

All comparators use the same authenticated behavior trace or the same keyed
evaluation episodes. None receives target, signal formula, evaluator reward
formula, counterfactual reward, hidden key, or control metadata.

- `constant_zero`: actions `(0,0,0,0)`, exact return `0.5` per regime.
- `constant_one`: actions `(1,1,1,1)`, exact return `0.5` per regime.
- `feedback_only_myopic`: independently fits mean immediate reward for each
  abstract state/action cell and never bootstraps. Nonterminal cells tie at
  zero, so phase zero chooses action zero and exact return is `0.5`.
- `no_bootstrap`: independently performs exactly the first synchronous sweep
  of the TD contract and then freezes. It has only the two terminal positive
  cells, phase zero ties to action zero, and exact return is `0.5`.
- `seeded_random`: one fresh
  `numpy.random.Generator(numpy.random.PCG64(314159265))` makes exactly one call
  per episode:
  `rng.integers(0, 2, size=(4,), dtype=numpy.int8)`. Calls consume episodes in
  canonical train, then validation, then test order. Scalar calls, four separate
  calls, default `int64`, or any other RNG call are forbidden. The dependency
  lock pins the NumPy version and the complete action-stream digest. No exact
  random return is a gate; only the frozen gain applies.

Each comparator is implemented independently of the learner's policy method.
The fitted feedback-only and no-bootstrap comparators use a genuinely separate
materializer that revalidates the sealed public observation/action components
and bare update scalars and shares no projection helper with `TDInputRow`.
Constants and random receive public evaluation observations only. No comparator
receives the complete keyed trace object. Each complete train, validation, and
test action stream is generated twice from fresh state and must be
byte-identical. Every stream is independently rescored through the authenticated
keyed scorer. Aggregate metric equality without action-stream equality is
insufficient.

## Transition-target negative control

This control uses a fresh learner and canonical source episodes, behavior
actions, targets, rewards, `done` values, counts, sweep contract, thresholds,
and held-out evaluator. It changes train public transition assignment only.

Within each regime/block/action-code pair, the two target slots form a fixed
pair. For every nonterminal source transition, the successor observation
payload, layout, and typed-byte digest are taken from the already committed
paired episode with opposite target slot and the same regime, block, action
code, phase, prefix, and chosen action. The control retains the source
`TransitionKey`, source successor
`ObservationKey`, predecessor/action/feedback links, and every source action,
target, reward, and `done` field. An evaluator-only typed `DonorRef` binds the
paired episode key and digest; neither reaches the learner. Paired episodes
have identical nuisance and action prefix; their signed-signal bytes differ.
The pairing is constructed from episode indices before any reward is
materialized and is independent of outcomes.

Downstream source rows retain source keys and links while their predecessor
observation payloads carry the substituted donor bytes. Terminal reward remains
the source episode's canonical reward. The learner sees no source, donor,
target-slot, or mode flag. The pending-transition validator authenticates the
exact source identity, paired payload, and donor reference; an arbitrary
same-phase or same-prefix row is rejected.

The mapping is a no-fixed-point involution, changes every nonterminal successor
assignment, preserves the complete successor-row multiset, and preserves all
source actions, source targets, canonical rewards, update rewards, `done`
values, episode counts, and terminal timing. Analytically it moves the positive
ancestry across the signed-signal handoff. The negative-sign policy succeeds by
its action-zero tie path while the positive-sign policy fails, so balanced
canonical held-out execution returns exactly `0.5`.

Frozen control requirements are validation and test macro return at most
`0.55`, minimum true-minus-control test gap `0.40`, exact canonical behavior
return/regret and update counts, and rejection of the complete positive gate.

## Outcome-blind reward-origin negative control

This control uses fresh canonical public observations, actions, transitions,
targets, canonical evaluator rewards, `done` values, keys, counts, TD sweeps,
thresholds, and held-out evaluation. It changes only the scalar delivered to
the train updater at the terminal boundary and its authenticated origin key.

For destination `(block=b, action_code=c, target_slot=t)`, the origin is:

```text
origin_block = (b + 1) mod 16
origin_action_code = (c + b) mod 16
origin_target_slot = t
```

The mapping is a fixed bijection over the 512 episode positions in each regime
and has no fixed point because the block always changes. It is constructed
only from indices; it never inspects or sorts targets, actions, rewards,
success, policy output, or Q values. The origin scalar is not precomputed. Its
key is queued evaluator-side and the scalar is materialized exactly once in the
destination's terminal resolution slot, then passed to the learner as a bare
`float64` with no origin metadata.

The mapping preserves the exact canonical reward multiset. Across the sixteen
blocks, every frozen `(target_slot, action_code)` destination cell receives
exactly one assigned reward one and fifteen assigned reward zeros, for exact
count `16` and mean `0.0625`. These counts and means are proven independently
after materialization, both per regime and globally. Consequently every
terminal state-action mean is `0.0625`; four synchronous sweeps make all 60 Q
cells exactly `0.0625`, with strictly positive cell counts exactly
`[32, 48, 56, 60]` after sweeps one through four; the tie policy chooses four
zeros and canonical held-out return is exactly `0.5`.

Frozen requirements are validation and test macro return at most `0.55`,
minimum true-minus-control test gap `0.40`, exact cell counts/means and reward
multiset, zero early origin materializations, exact update counts, and rejection
of the complete positive gate.

## Complete signal attribution control

The ablation makes a fresh copy of every legal and realized train, validation,
and test public observation and changes only the signed-signal coordinate to
exact `float64(0.0)`. It does not regenerate a row. Phase, prefix, nuisance,
actions, targets, keys, predecessor/successor links, transition assignments,
canonical/update rewards, donor/origin identities, `done`, and ordering remain
byte-identical.

The all-row integrity proof compares the complete 122,880-row canonical and
ablated projections, including canonical, transition-control, and reward-origin
variants, under the exact field whitelist. A fresh learner refit on ablated
train rows and the canonical true policy evaluated on ablated held-out rows
must each have validation and test macro return at most `0.55` and must reject
the complete positive gate. The expected return is `0.5`; no nuisance-only or
path-only signal may exceed the ceiling.

## Complete control-difference whitelists

Every canonical/control comparison covers all legal rows and every observation,
action, transition, feedback, target, key, link, donor, origin, dtype/layout,
and Boolean field. The only permitted differences are:

- target-swap twin: separate hidden evaluator target plus the exactly
  formula-authorized terminal canonical-reward and, if a twin trace is
  materialized, bare update-reward values; canonical `target_slot`, public sign,
  keys, and every other field remain fixed;
- transition-target control: paired successor signed-signal bytes and digest,
  the corresponding downstream predecessor signed-signal bytes/digest, and
  exact donor identity on nonterminal rows;
- reward-origin control: terminal update-reward scalar/digest and exact origin
  identity;
- signal ablation: signed-signal bytes/digest only.

All other fields must be byte-identical. Each control has a mandatory mutation
test for every protected field class; a comparison that omits a category,
checks only reduced digests, or accepts an undeclared difference fails before
the TD learner executes.

## Sanitized result boundary

The controller result has exactly the standard top-level fields:

```text
action, cases, environment, fixture, schema_version, status, study_id
```

The registry must declare the exact required fields for every case. Retained
values are bounded scalar aggregates, counts, Booleans, version strings, exact
case contracts, and lowercase SHA-256 commitments only. The sanitizer and
focused tests must reject raw or row-level values under any key containing or
representing observations, states, actions, rewards, returns, targets,
successors, transitions, trajectories, logs, Q tables, policy state, parameter
values, gradients, topology, paths, credentials, secrets, donor arrays, origin
arrays, or private evidence.

Fresh network-disabled worker projections must match the complete non-process
sanitized projection byte-for-byte. No private path, vector, raw trace, or
unbounded diagnostic may appear in stdout or the committed result summary.

## Complete case set

The registry/result case set is exactly:

1. `typed_episodic_contract`
2. `complete_family_replay`
3. `target_swap_twin`
4. `generator_partition`
5. `realized_path_disjointness`
6. `train_only_source_boundary`
7. `lazy_information_boundary`
8. `pending_transition_authentication`
9. `keyed_trace_authentication`
10. `synchronous_td_order`
11. `all_boundary_terminal_dependency`
12. `baseline_replay`
13. `multistep_td_recovery`
14. `transition_target_control`
15. `reward_origin_control`
16. `signal_attribution_control`
17. `control_difference_whitelists`
18. `sanitized_result_contract`
19. `process_isolation`

No case may be removed, merged, renamed, weakened, or substituted after this
commit. Focused implementation tests may be added but cannot redefine a case.

## Stopping rule and actions

This plan commit is the entire current checkpoint. It does not approve learner
execution. A later heartbeat may implement only this contract on a separate
v3 fixture and dedicated network-disabled worker under `experiments/local_lab`.
Before any result-bearing invocation, it must:

1. pass focused generator-only, family-replay, source-boundary, attack,
   sanitizer, validator, TD-order, control, and fresh-worker tests;
2. receive independent read-only fixture, leakage, and repository audits;
3. add one exact `studies.json` result contract with committed source hashes;
4. update only the controller's v3 worker allowlist and normalized pinned
   registry digest;
5. preserve every earlier repository snapshot, the complete submission tree,
   protected ZIP, and protected manifest;
6. obtain a second clean pre-result commit; and
7. run at most one full repository verification pass, building any submission
   artifact only to a fresh scratch path.

Only then may `tools/run_local_lab.py` invoke v3 exactly once through the
credential-scrubbed, network-disabled, leased, immutable-result controller on
local CPU. The controller must never run v1, v2, or any terminal study again.

All nineteen cases and the complete positive/control gates must pass. Success
uses action:

`synthetic_four_step_synchronous_td_propagation_confirmed_for_harness`

Any failed invariant or threshold, comparator/control recovery, malformed or
nondeterministic result, timeout, source drift, process mismatch, or substantive
pre-result confound uses action:

`park_multistep_td_research`

If the substantive confound is found before controller execution, do not add a
terminal result: quarantine v3, record a sanitized preflight rejection, keep
the controller `awaiting_study`, and require a fresh versioned plan.

## Claim boundary

A pass may say only that this fixed deterministic local-CPU harness collected
the frozen synthetic behavior trace, applied four synchronous one-step
bootstrapped TD sweeps, propagated two terminal values backward by exactly one
boundary per sweep, selected the optimal toy action prefix on untouched
generator regimes, beat the frozen feedback-only toy comparators, and lost the
positive gate under the exact transition-target, balanced reward-origin, and
signal interventions.

It cannot support a claim about online adaptation, sample efficiency, general
RL, meta-RL, delayed-credit necessity, absence of a public target-correlated
shortcut, target-independent public signal, partial observability, official
data, private or hidden topology, UIFO, the submitted optimizer, candidate
selection, a native rewrite, GPU or accelerator speed, leaderboard rank,
competition score, or authorization to change or upload the protected
submission.
