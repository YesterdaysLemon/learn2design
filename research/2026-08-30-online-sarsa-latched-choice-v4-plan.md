# Frozen plan: online SARSA paired-fork control mechanics

Date: 2026-08-30

Study ID: `online-sarsa-latched-choice-v4`

Status: draft until the commit containing this file; plan-only, with no fixture,
worker, registry entry, learner execution, or result

## Scope and independence

This checkpoint freezes one fresh deterministic, topology-independent online
control question before implementation. It does not use official data, private
topologies, leaderboard information, candidate trajectories, or an earlier
rejected fixture or development metric as a design input. The three rejected
online-SARSA IDs and both rejected multi-step IDs remain quarantined and are not
imported, repaired, executed, or used as parameter sources.

This plan deliberately keeps the evidence claim smaller than the runtime
mechanism. The approved study source is first-party, hash-pinned Python. The
runtime boundary is a reproducible restricted-name-map and process-isolation
contract, not a hostile-code sandbox and not an operating-system ACL claim.
No Windows system file, owner, DACL, package rule, parent directory, loader
rule, or registry value may be changed.

The plan commit authorizes documentation only. Exact implementation and
hostile pre-result audit are a separate checkpoint. One result-bearing guarded
invocation, if later authorized, is a third checkpoint.

## Narrow falsifiable question

Can a blank deterministic tabular SARSA(0) learner, updating online from its
own frozen forced/greedy behavior on three train regimes, acquire a public-cue
dependent two-action fork, retain exact return `1.0` on four untouched held-out
regimes without held-out updates, beat constant, myopic, no-bootstrap, and
seeded-balanced-random comparators under precommitted margins, and lose the
association under identically evaluated zero-bootstrap, phase-zero-assignment,
zero-terminal-origin, and complete-marker-ablation controls?

A pass could validate only this exact synthetic online-control harness and its
deliberately exposed toy association. It could not show that learning is
necessary, that the public cue lacks a programmed shortcut, or anything about
general RL, hidden topology, UIFO, the submitted optimizer, candidate value,
accelerators, competition performance, or score.

## Frozen generator regimes and splits

There are exactly seven immutable regimes:

| name | split | seed | marker scale | nuisance offset |
| --- | --- | ---: | ---: | ---: |
| `train-copper` | train | 17011 | 0.75 | 1 |
| `train-indigo` | train | 17027 | 1.25 | 3 |
| `train-sage` | train | 17041 | 1.75 | 5 |
| `validation-amber` | validation | 27011 | 0.625 | 7 |
| `validation-violet` | validation | 27031 | 1.375 | 9 |
| `test-silver` | test | 37003 | 0.875 | 11 |
| `test-gold` | test | 37019 | 1.625 | 13 |

Seeds and nuisance offsets are identity commitments; the generator makes zero
RNG calls. The cue is exact integer `0` or `1`. Its evaluator-only target code
is `1` (`01`) for cue 0 and `2` (`10`) for cue 1. The public signed marker is
negative for cue 0 and positive for cue 1, so it deliberately reveals the toy
target to a hand-programmed policy. Evaluator truth is nevertheless absent
from learner capabilities.

Train, validation, and test episode identities are pairwise disjoint. A train
identity is `(regime, split, seed, round, cue, block_position)`. A held-out
identity is `(regime, split, seed, evaluation_index, cue)`. Identity fields do
not enter the policy key.

## Complete two-step family

Every episode has exactly two decisions. At phase 0 the public branch sentinel
is `-1`. Action `a0` creates a target-independent phase-1 public branch equal
to `a0`. Action `a1` terminates the episode with completed code
`2*a0 + a1`.

Both environment steps consume only the authenticated predecessor and chosen
action. Phase 0 returns reward `0.0` and `done=false`. Phase 1 returns reward
`1.0` exactly when the completed code equals the evaluator target, otherwise
`0.0`, with `done=true`. Terminal reward, success, and evaluator target are
materialized only after the terminal action is consumed.

For one cue-specific episode there are two legal phase-0 rows and four legal
phase-1 rows. The frozen counts are:

| quantity | seven-regime template | all 1,376 episode instances |
| --- | ---: | ---: |
| predecessor identities | 42 | 4,128 |
| legal rows | 84 | 8,256 |
| nonterminal rows | 28 | 2,752 |
| terminal rows | 56 | 5,504 |
| successful terminal rows | 14 | 1,376 |
| realized rows | not applicable | 2,752 |

Constructor A uses nested loops in regime-table order, cue 0 then 1, and local
row order. Constructor B is a closed-form decoder. For template row index
`j in [0,83]`, it computes `(regime_index, cue, k)` by successive `divmod(j,12)`
and `divmod(remainder,6)`. Local rows `k=0,1` are phase 0 with action `k`;
rows `k=2..5` are phase 1 with branch `(k-2)//2` and action `(k-2)%2`.
Target is `1+cue`. The two constructors share only the frozen regime tuple and
canonical record encoder; A may not call B or a helper used by B.

Expanded episode order is exact: the three train regimes, then 16 rounds, cue
0 then 1, and block positions 0 through 8; then the four held-out regimes in
table order and evaluation indices 0 through 127, with cue `index%2` and block
sentinel 255. Expanded row index is `episode_index*6+k`. Constructor B decodes
that integer directly; constructor A expands its independent nested iterator.
Every key must be unique. The controller oracle implements decoder B without
importing or calling fixture code. `constructor_a_root_sha256` and
`constructor_b_root_sha256` cover all encoded rows in row-index order and must
be equal.

Before learner construction, A and B replay all train rows and authenticate
predecessor bytes, action, successor bytes or terminal sentinel, reward, done,
and evaluator-origin identity. Held-out rows do not exist until the fitted
table and train capabilities are destroyed; A and B then independently build
and replay them before evaluation.

An evaluator twin freezes every episode identity, public predecessor, action,
successor, and done byte, then swaps target codes 1 and 2. Exactly 28 template
terminal rewards and 2,752 expanded terminal rewards change. No public or
nonterminal field changes. The twin never enters learner or evaluator input.

## Public observation and policy state

The public observation is an immutable contiguous little-endian binary64
vector of length four:

`(phase, signed_marker, branch, nuisance)`.

At phase 0, `branch=-1.0`; at phase 1 it is exact `0.0` or `1.0` from `a0`.
`signed_marker` is `(-1 or +1) * marker_scale`. `nuisance` is exactly
`marker_scale / 16 + phase / 8 + (branch + 1) / 64 + nuisance_offset / 1024`.
It is public, deterministic, and excluded from the policy key.

The exact two-byte policy key is `(cue_slot, state_slot)`. Negative marker maps
to cue slot 0, positive marker to cue slot 1, and zero is illegal except in the
declared marker-ablation control, where zero maps to cue slot 0. State slot 0
is the root, 1 is phase-1 branch 0, and 2 is phase-1 branch 1. The table has six
states and twelve float64 cells, all initialized to exact positive zero.

The learner-facing ABI consists of exactly two pure primitive functions. The
selection function receives only copied primitives
`(q0:float, q1:float, mode:int, forced_action:int)`. Forced mode is 1 with
action 0 or 1. Greedy mode is 0 with sentinel 255 and returns the smaller
argmax, so ties select 0. It returns one exact built-in integer. The harness,
not selection code, authenticates observations and tokens and builds policy
keys. The update function receives only copied primitives
`(old_value:float, reward:float, next_value:float, done:bool)` and returns
`reward` when done, otherwise `reward+next_value`; the sealed harness writes
that result to the authenticated table address. Neither primitive receives an
observation, key, target code, episode identity, evaluator, table, callback, or
mutable object.

The mutable twelve-cell table is harness-owned learner state. The train-source
ABI yields only authenticated public observations and behavior tokens. Its
sealed environment closure owns evaluator target codes and accepts only an
authenticated predecessor/action pair; it exposes only successor/reward/done
after the action is consumed. Complete-family evaluator rows are audit inputs
to the harness and controller oracle, never values in the learner-facing ABI.
The fitter input is exact tuple `(three_train_source_capabilities,
token_factory_capability, blank_table)` and has no held-out or evaluator-truth
member.

The learner's only persistent mutable state is the twelve-cell table. It has no
counter, cache, history, RNG, callback, episode field, module/class write,
closure write, filesystem, network, process, native-loader, evaluator, source,
or future-token capability. Evaluation receives only an immutable table
snapshot.

## Frozen training and evaluation schedules

Training is regime-major in table order. Each train regime has 16 rounds. In
each round cue 0 is completed before cue 1. A cue block contains eight forced
episodes followed by one greedy episode. The forced code order is the literal
`[0, 3, 1, 2]`, with each code repeated for two consecutive episodes. A forced
code supplies its two actions in most-significant-bit order.

The behavior token is exact three bytes `(mode, forced_action, phase)`. Greedy
tokens contain forced-action sentinel 255. Token construction depends only on
regime, round, cue-block position, and phase; it cannot read marker sign,
evaluator target, reward, Q values, selected action, successor, or outcome.
Corresponding cue-0 and cue-1 blocks receive byte-identical three-byte
`token_bytes`; their enclosing records and digests differ because episode
identities differ.

The exact train counts are:

| quantity | per train regime | all train regimes |
| --- | ---: | ---: |
| rounds | 16 | 48 |
| episodes | 288 | 864 |
| forced episodes | 256 | 768 |
| greedy episodes | 32 | 96 |
| tokens/action decisions/updates | 576 | 1,728 |
| paired cross-cue tokens | 288 | 864 |

Each validation and test regime contains 128 episodes, alternating cue 0 then
cue 1. There are 256 validation and 256 test episodes. Evaluation is greedy,
read-only, token-free, and update-free. The fitted table and all closed learner,
module, class, global, closure, source, operation-ledger, and effect-counter
state are hashed before and after validation and independently before and after
test. The paired hashes must be equal. Each held-out source is operation
counted and must open once, yield exactly 128 identities, complete, and perform
zero other operations.

## Exact online SARSA chronology

The learner uses float64 `alpha=1.0` and `gamma=1.0`. Each episode executes:

1. `select_initial` authenticates phase-0 observation and behavior token,
   spends one exact selection permit, and returns `a0`.
2. `environment_step` spends a step permit and returns the phase-1 successor,
   zero reward, and false done.
3. `select_next` authenticates the successor and next token, spends a
   next-action permit, and latches `a1` before the predecessor update.
4. `sarsa_update` spends a feedback permit and writes
   `Q(root,a0) = 0 + Q(branch(a0),a1)`.
5. `append_trace` spends an append permit and seals phase 0.
6. `environment_step` consumes the latched `a1`, spends the next step permit,
   and returns terminal reward and true done.
7. `sarsa_update` spends the next feedback permit and writes
   `Q(branch(a0),a1) = reward`.
8. `append_trace` spends the final append permit, seals phase 1, and only then
   permits terminal return exposure.

The sealed harness consumer signatures are exact:
`select_initial(Observation,BehaviorToken,Permit)->Action`,
`environment_step(Observation,Action,claimed_action_digest,Permit)->Step`,
`select_next(Observation,BehaviorToken,Permit)->Action`,
`sarsa_update(Step,O(Action),predecessor_key,update_address_action,
feedback_origin,Permit)->Update`,
`append_trace(Step,Update,Permit)->D`, and
`expose_return(Episode)->F64`. Optional next action is present only for phase 0;
the environment supplies its authenticated reward/done and the feedback gateway
supplies update reward/origin. No consumer accepts a detached Boolean claim.

The exact train operation counts are 864 initial selections, 864 next-action
selections, 1,728 action decisions, 1,728 environment steps, 1,728 updates,
1,728 appends, 864 terminal returns, and 6,912 spent operation permits. No
action may be selected twice, changed after latching, updated before its step,
or appended before its update.

An independent sequential oracle starts from its own blank twelve-cell table,
reconstructs the schedule and environment without calling learner code, and
must match every selected action, target, old/new value, tie, and final cell
bit for bit.

The frozen algebra is: every cue block visits all four codes twice. The first
rewarding-code visit writes its terminal cell and the second propagates that
value to the distinct cue root. The final greedy episode succeeds. Canonical
training therefore has exactly 192 forced successes, 96 greedy successes,
reward `288.0`, and regret 576. The final table has exactly four positive cells
and canonical greedy evaluation returns `1.0` in every held-out regime.

## Train-only sources and action-stream replay

The fitter accepts only a complete authenticated iterator over the three train
regime factories and the frozen token factory. It has no validation/test
argument, global, closure, callback, path, or late lookup. Exactly seven fit
paths are exercised: `canonical`, `myopic`, `no-bootstrap`,
`zero-bootstrap`, `phase-zero-assignment`, `zero-terminal-origin`, and
`marker-ablation`.

The `myopic` source-audit fit path is deliberately distinct from a learned
policy. Forced episodes obey the exact frozen behavior tokens; greedy episodes
select action 0 for a negative marker and action 1 for a positive marker at
both phases. All 864 episodes still execute the complete eight-operation
chronology, including both authenticated SARSA updates into a fresh audit table,
so its event composition equals case 5. That table is destroyed and never
enters the returned comparator. The returned held-out `myopic-marker` policy is
the marker-only rule and holds no table or learner capability. Thus the source
audit exercises updates without changing comparator semantics.

Each path runs with held-out sources in modes `absent`, `exploding`, and `spy`,
for 21 checks. Validation/test factory and evaluation-snapshot spy counts must
remain zero. An inverse train iterator containing one held-out factory must
reject before a token, selection, environment step, or update.

Each path also runs once with train factories delivered in table order and once
in reverse. Before behavior starts, the fitter authenticates the complete
three-regime set and canonical-sorts it into table order. Raw delivery roots
must differ; normalized episode, token, semantic action/step/update/append/
trajectory, semantic-terminal, and final-table roots must match. Seven closed source-order
receipts bind the path ID and both raw/normalized roots.

For each SourceOrderReceipt, `raw_a` and `raw_b` use shared generic root name
`source_raw_delivery` over the three complete Regime encodings in delivered
order. `normalized_a` and `normalized_b` use shared root name
`source_normalized_run` over, in category order, three normalized Regime
records, 864 Episode records, 1,728 BehaviorToken records, 1,728 each of
SemanticAction, SemanticStep, and SemanticUpdate, 1,728 semantic append tuples
`(episode_digest:D,phase:U8,semantic_step_digest:D,semantic_update_digest:D)`,
864 SemanticTrajectory records, and final 96 table bytes;
within each category order is normalized regime, episode, phase. `terminal_a`
and `terminal_b` use shared root name `source_semantic_terminal` over the 864
complete SemanticTrajectory encodings followed by final table bytes.
SemanticAction is projected from an authenticated Action by dropping only
selection receipt; SemanticStep joins that action to replace receipt-bearing
Action digest with environment action; SemanticUpdate replaces receipt-bearing
Step digest with the complete SemanticStep digest and drops feedback receipt;
SemanticTrajectory uses those semantic digests and drops append receipts. These
semantic roots exclude LedgerId, all event/selection/step/feedback/append
receipts, receipt-bearing Action/Step/Update digests, and child roots, which intentionally differ
between table/reverse execution ledgers; the corresponding ChildSummary records
bind those physical executions separately. Raw roots must differ, while both
normalized and both semantic-terminal roots must be byte-identical.

Canonical validation and test action streams are independently replayed from
the immutable snapshot, joined by episode identity, and compared byte for byte.
Every comparator and control receives the identical held-out episode identities,
evaluator contract, tie rule, read-only boundary, metric, and source receipts.

## Frozen metrics and positive gate

Episode return is the terminal reward. Mean return is success count divided by
episode count; regret is episode count minus successes. There is no smoothing,
rounding, early stopping, checkpoint selection, omitted episode, or confidence
interval in a gate.

The canonical positive gate is the conjunction of:

1. complete family, twin, split, and source commitments replay exactly;
2. the schedule and all 1,728 selected actions replay exactly;
3. every operation follows the exact chronology and permit identity;
4. every update matches the canonical SARSA equation and independent table;
5. train successes/reward/regret equal 192/96/288.0/576;
6. every held-out regime, validation aggregate, and test aggregate returns 1.0;
7. constant and myopic margins are at least 0.90;
8. the no-bootstrap margin is at least 0.45;
9. validation and test seeded-balanced-random margins are each at least 0.70;
10. held-out updates and construction-time held-out operations are zero;
11. source, trace, boundary, malformed, and action-stream replay all pass; and
12. every negative control completes, preserves the evaluator contract, passes
    its intervention whitelist, and rejects the common performance gate.

The common control-performance vector has twelve clauses: completed,
identical evaluator contract, every held-out regime return 1.0, validation
return 1.0, test return 1.0, constant/myopic margin at least 0.90,
no-bootstrap margin at least 0.45, random margin at least 0.70, zero held-out
updates, action replay, source isolation, and trace authentication. Each
control below must have false indices exactly `3,4,5,6,7,8`; every other index
must be true. A crash, omitted row, changed evaluator, missing source, or
malformed projection is a failed control, not a successful negative result.

## Frozen comparators

All comparators use the exact held-out identities, public observation rule,
environment, tie rule, source receipts, and metric. Their action-conditioned
successor observations are independently reconstructed.

- `constant-0` completes code 0 and returns exact 0.0.
- `constant-1` completes code 3 and returns exact 0.0.
- `myopic-marker` repeats action 0 on a negative marker and action 1 on a
  positive marker, completing code 0 or 3 and returning exact 0.0. Its train
  constructor sees authenticated public markers only.
- `no-bootstrap` fits the canonical behavior online but replaces the
  nonterminal target with immediate reward. Root cells remain tied; cue 0 then
  succeeds through its learned branch-0 terminal cell and cue 1 fails, so every
  balanced held-out return is exact 0.5.
- `seeded-balanced-random` uses seed 104729. Within each regime and cue,
  consecutive quartets receive a SHA-256-determined permutation of all codes
  0, 1, 2, 3. The ordering key is SHA-256 of
  `b"online-sarsa-v4-random\0"`, little-endian uint64 seed,
  U32-little-endian UTF-8 regime-name length and bytes, cue U8,
  little-endian uint16 quartet index, and code U8. Codes are sorted by their raw
  32-byte digest, then ascending code as the impossible-collision tie rule. It
  reads no target or reward. Each
  cue quartet therefore contains one target code and exact return 0.25.
  Across held-out evaluation it makes 512 code decisions and 1,024 scalar
  actions.

Every margin is signed `candidate_mean_return - comparator_mean_return`, first
per regime and then per validation/test aggregate. The constant/myopic gate uses
the minimum over both split aggregates and all three 0.0 comparators; the
no-bootstrap and random gates use the minimum over their two split aggregates.
`minimum_margin=0.5` in case 10 is the minimum of every canonical comparator
margin just defined. A control substitutes its own stream for candidate in the
same equations; marker ablation must fail them independently for both fresh-fit
and frozen-table streams.

The outer validator independently constructs every comparator action stream
and environment result. Input, action, and output roots are all required; an
aggregate return alone is insufficient.

## Negative controls

Every control begins from a fresh blank table, uses the canonical train and
held-out identities, schedule, environment actions, evaluator contract,
comparators, sources, tie rule, metric, trace authentication, and process
boundary, and changes only its declared intervention.

### Zero bootstrap

Only the nonterminal lookup is replaced by an immutable disjoint zero row. The
latched action, environment action, terminal update, and update address remain
canonical. Root cells remain tied. Cue 0 succeeds and cue 1 fails, giving exact
train forced/greedy successes 192/48, reward 240.0, regret 624, and every
held-out return 0.5.

### Phase-zero assignment

The environment executes the authenticated selected action unchanged. At phase
0 only, the update-address action is replaced with exact 0; phase-1 terminal
updates retain the executed address. The intervention occurs after step and
next-action authentication. Cue 0 succeeds and cue 1 fails under greedy
evaluation, giving exact train 192/48/240.0/624 and held-out return 0.5.

### Zero terminal origin

The environment computes and records its canonical terminal outcome. The
feedback gateway supplies exact 0.0 only to terminal updates and binds a typed
`null-terminal-control` origin. All Q cells remain zero; greedy code 0 fails
for both cues. Exact train values are 192 forced successes, zero greedy
successes, reward 192.0, regret 672, and held-out return 0.0.

### Complete marker ablation

Every public predecessor and successor marker becomes exact positive zero
before key construction. All other public bytes, evaluator truth, environment,
schedule, reward, done, and identities remain paired. A fresh ablated fit ends
with the cue-1 code for both cues; the frozen canonical table read through
ablated keys uses the cue-0 code for both. Both streams return exact 0.5 on
every balanced held-out regime. Fresh ablated training retains canonical
192/96/288.0/576 totals.

## Static and dynamic intervention whitelists

Each of the 8,256 expanded legal rows has this exact ordered static projection:
row key, episode identity, evaluator target, predecessor public bytes,
predecessor key, action, successor public bytes or terminal, successor key or
terminal, environment reward, update reward, feedback-origin digest,
bootstrap key or terminal, update-address action, done, and control tag.

The four controls produce 33,024 paired rows. Allowed static differences are:

| control | field/predicate | exact count |
| --- | --- | ---: |
| zero bootstrap | bootstrap key on every nonterminal row | 2,752 |
| zero bootstrap | control tag | 8,256 |
| phase-zero assignment | update address on phase-0 action-1 rows | 1,376 |
| phase-zero assignment | control tag | 8,256 |
| zero terminal origin | update reward on successful terminal rows | 1,376 |
| zero terminal origin | feedback origin on terminal rows | 5,504 |
| zero terminal origin | control tag | 8,256 |
| marker ablation | predecessor public bytes | 8,256 |
| marker ablation | predecessor key on cue-1 rows | 4,128 |
| marker ablation | successor public bytes on nonterminal rows | 2,752 |
| marker ablation | successor key on nonterminal cue-1 rows | 1,376 |
| marker ablation | bootstrap key on nonterminal cue-1 rows | 1,376 |
| marker ablation | control tag | 8,256 |

Static equality applies only to the ordered static projection above; record
digests and receipts are deliberately not fields of that projection. Every
unlisted static-projection field is byte-identical and every listed difference
is required on exactly its predicate rows. Exact control tags are ASCII
`zero-bootstrap`, `phase-zero-assignment`, `zero-terminal-origin`, and
`marker-ablation`; canonical tag is `canonical`.

The dynamic projection contains realized predecessor bytes/key/digest, token
bytes/digest, selected and environment actions/action digests, successor
bytes/key/digest, reward, done, latched next action, bootstrap key, update
address, update reward, origin, target, old/new value, every step/update/append
receipt, table root after each update, trajectory digest, terminal ledger, and
return. The closed dependency edges are: predecessor public bytes to
observation digest and policy key; policy key to Q-row lookup and both lookup
values; token bytes, mode, forced action, and those values to selected action;
observation/token digests and selected action to selection receipt/action
digest; selected action and action digest to the unchanged environment action
and step receipt; successor public bytes to successor
digest/key; successor key plus latched action to bootstrap key/value;
bootstrap key/value, update reward/origin, or update address to target/new value
and table root; those fields to update/append receipts, trajectory digest,
ledger, later table lookups/actions, and return. Every semantic digest edge also
propagates through its complete-record digest, event payload, event receipt,
channel root, and full ledger root. Control tag has no semantic causal edge; it
is included only in StaticProjection, DynamicProjection, their record/event
digests, and per-control roots, whose deterministic hash-only differences are
required separately and never authorize a policy or reward difference.
An independent replay rejects any dynamic difference without a path from that
control's listed static roots, or any missing difference on a reached node.

Static events commit canonical then zero-bootstrap, phase-zero-assignment,
zero-terminal-origin, and marker-ablation projections, each in expanded row
order: `5*8256=41280`. Dynamic events commit canonical 2,752 realized rows,
then zero-bootstrap, assignment, and origin at 2,752 each, then marker fresh at
2,752 and marker-frozen held-out at 1,024: 14,784 total. Their case-18 roots are
controller-owned per-channel ledger roots over these exact partitions.
For each row, `DynamicProjection.ledger_root` is exactly the active child-ledger
root immediately before that row's dynamic event; the event receipt is not part
of its own payload. Rows emit in the partition and expanded-row orders just
stated, so the next row observes the preceding dynamic event's receipt.

## Typed records, canonical encoding, and trace closure

Internal values are immutable fixed-length tuple records. Numeric record tags
are exactly the one-based list positions below, `1..26`; tag 0 and tags above
26 are invalid. Type codes are `U8`,
`U16`, `U32`, and `U64` unsigned little-endian integers of the stated width;
`I64` signed little-endian int64; `B` byte 0 or 1; `F64` finite little-endian
binary64 with negative zero forbidden; `D` raw 32-byte SHA-256; `Y[n]` exact
bytes; `S[n]` UTF-8 with at most `n` bytes; `O(T)` a one-byte 0/1 tag followed
by `T` only for 1; `Q(T,n)` an exact-length tuple; and `L(T,n)` a tuple with
at most `n` children. Field order and record tags are exact:

1. `Regime` `(name:S[32], split:U8, seed:U64, marker_scale:F64,
   nuisance_offset:I64)`, with split enum train/validation/test = 0/1/2.
2. `Episode` `(regime_digest:D, major_index:U16, cue:U8,
   block_position:U8)`; train uses round and position 0..8, held-out uses
   evaluation index and sentinel 255.
3. `Observation` `(episode_digest:D, phase:U8, marker:F64, branch:I64,
   nuisance:F64, public_bytes:Y[32], observation_digest:D)`.
4. `BehaviorToken` `(episode_digest:D, phase:U8, mode:U8,
   forced_action:U8, token_bytes:Y[3], token_digest:D)`.
5. `Permit` `(permit_class:U8, episode_digest:D, phase:U8, ordinal:U8,
   nonce:Y[16])`; class enum selection/step/next-action/feedback/append is
   0/1/2/3/4 and ordinal is the chronology position 0..7. Operation-tag order
   is select-initial/step-0/select-next/update-0/append-0/step-1/update-1/
   append-1 = 0..7.
6. `Action` `(episode_digest:D, phase:U8, value:U8,
   observation_digest:D, token_digest:O(D), selection_receipt:D)`; evaluation
   uses absent token digest.
7. `Step` `(episode_digest:D, phase:U8, predecessor_digest:D,
   action_digest:D, successor_digest:O(D), reward:F64, done:B,
   environment_origin:D, step_receipt:D)`.
8. `Update` `(episode_digest:D, phase:U8, predecessor_key:Y[2], action:U8,
   reward:F64, feedback_origin:D, successor_key:O(Y[2]),
   next_action:O(U8), target:F64, old_value:F64, new_value:F64,
   step_digest:D, feedback_receipt:D)`.
9. `Trajectory` `(episode_digest:D, step_digests:Q(D,2),
   update_digests:Q(D,2), append_receipts:Q(D,2), return:F64,
   terminal_ledger:D)`.
10. `SourceReceipt` `(path_id:U8, mode:U8, train_episodes:I64,
    validation_accesses:I64, test_accesses:I64, snapshot_accesses:I64,
    terminal_ledger:D, completed:B)`; path order is the seven fit paths and
    mode absent/exploding/spy = 0/1/2.
11. `SourceOrderReceipt` `(path_id:U8, raw_a:D, raw_b:D, normalized_a:D,
    normalized_b:D, terminal_a:D, terminal_b:D, completed:B)`.
12. `EvaluationSourceReceipt` `(policy_id:U8, regime_digest:D, opens:I64,
    yielded:I64, other_operations:I64, completed:B)`. Policy enum order is
    canonical, constant-0, constant-1, myopic-marker, no-bootstrap,
    seeded-balanced-random, zero-bootstrap, phase-zero-assignment,
    zero-terminal-origin, marker-ablation-fresh, marker-ablation-frozen = 0..10.
13. `BoundarySnapshot` `(table_bytes:Y[96], active_episode:O(D), stage:U8,
    latched_action:O(U8), spent_permits:U8,
    operation_ledger:L((ordinal:U8,op_tag:U8,episode:D,phase:U8,receipt:D),8),
    trace_root:D,
    trace_count:I64,
    source_counters:Q((path_id:U8,mode:U8,train:I64,validation:I64,
    test:I64,snapshot:I64,completed:B),21),
    output_count:I64, file_calls:I64, network_calls:I64,
    process_calls:I64, native_calls:I64)`. Stage 0..8 is the chronology
    boundary immediately before each numbered operation and after operation 8.
14. `RejectionWitness` `(witness_id:U8, kind:U8, consumer:S[32],
    expected_exception:S[32], observed_exception:S[32], context_sha256:D,
    input_sha256:D, mutation_sha256:D, before_state_sha256:D,
    after_state_sha256:D, before_effect_sha256:D, after_effect_sha256:D,
    rejected:B)`; kind live/malformed = 0/1.
15. `StaticProjection` `(row_key:D, episode_digest:D, evaluator_target:U8,
    predecessor_public:Y[32], predecessor_key:Y[2], action:U8,
    successor_public:O(Y[32]), successor_key:O(Y[2]),
    environment_reward:F64, update_reward:F64, feedback_origin:D,
    bootstrap_key:O(Y[2]), update_address_action:U8, done:B,
    control_tag:S[32])`.
16. `DynamicProjection` `(episode_digest:D, phase:U8,
    predecessor_public:Y[32], predecessor_key:Y[2], observation_digest:D,
    token_digest:O(D), selected_action:U8, action_digest:D,
    environment_action:U8, step_digest:D, successor_public:O(Y[32]),
    successor_key:O(Y[2]), successor_digest:O(D), reward:F64,
    done:B, latched_next_action:O(U8), bootstrap_key:O(Y[2]),
    update_address_action:U8, update_reward:F64, feedback_origin:D,
    target:F64, old_value:F64, new_value:F64, update_digest:D,
    append_receipt:D, table_root:D, trajectory_digest:O(D), ledger_root:D,
    return:O(F64), control_tag:S[32])`.
17. `LedgerId` `(case_id:U8, kind:U8, path_or_policy:U8, variant:U8,
    ordinal:U16)`.
18. `ChildSummary` `(ledger_id:LedgerId, event_count:U64,
    channel_counts:Q(U64,19), terminal_root:D)`.
19. `ProcessReceipt` `(launch_index:U8, executable_sha256:D,
    fixture_sha256:D, worker_sha256:D, bootstrap_sha256:D,
    packet_abi_sha256:D, capability_map_sha256:D, host_receipt_sha256:D,
    controller_sha256:D, registry_sha256:D, job_created:B, suspended:B,
    assigned_before_resume:B, resume_succeeded:B, ready_seen:B, go_sent:B,
    exit_code:I64,
    maximum_active_processes:U64, active_processes_after:U64,
    unexpected_inherited_handles:U64, stdout_bytes:U64, stderr_bytes:U64,
    scratch_created:B, scratch_contained:B, scratch_removed:B)`.
20. `ManifestReceipt` `(launch_index:U8, module_rows:U64,
    module_set_root:D, os_dependency_manifest_root:D,
    consecutive_sets_equal:B, all_files_read_only:B)`.
21. `FramingReceipt` `(launch_index:U8, packet_bytes:U64,
    packet_sha256:D, ready_frame:Y[13], go_byte:U8,
    projection_bytes:U64, projection_sha256:D, final_lf:B,
    framing_exact:B)`.
22. `FamilyRow` `(domain:U8, regime:Regime, episode:O(Episode), cue:U8,
    phase:U8, branch:I64, action:U8, predecessor_public:Y[32],
    predecessor_key:Y[2], successor_public:O(Y[32]),
    successor_key:O(Y[2]), reward:F64, done:B, environment_origin:D)`.
23. `SemanticAction` `(episode_digest:D, phase:U8, value:U8,
    observation_digest:D, token_digest:O(D))`.
24. `SemanticStep` `(episode_digest:D, phase:U8, predecessor_digest:D,
    environment_action:U8, successor_digest:O(D), reward:F64, done:B,
    environment_origin:D)`.
25. `SemanticUpdate` `(episode_digest:D, phase:U8,
    predecessor_key:Y[2], action:U8, reward:F64, feedback_origin:D,
    successor_key:O(Y[2]), next_action:O(U8), target:F64, old_value:F64,
    new_value:F64, semantic_step_digest:D)`.
26. `SemanticTrajectory` `(episode_digest:D,
    semantic_step_digests:Q(D,2), semantic_update_digests:Q(D,2),
    return:F64)`.

`ManifestReceipt.module_set_root` is generic root `loaded_module_set` over one
64-byte item per loaded module: path digest D followed by file digest D, sorted
by path digest then file digest. `FramingReceipt.packet_sha256` hashes the exact
eight-byte outer length followed by the exact payload, excluding READY, GO, and
EOF. `projection_sha256` hashes only the exact declared ASCII JSON bytes,
including their one final LF and excluding READY and the U64 JSON length.
`unexpected_inherited_handles` counts child-inherited handles other than the
three controller-created standard pipes; it must be zero on every launch.

`FamilyRow.domain` is template/expanded = 0/1. Template rows have absent
Episode; expanded rows have the exact Episode. Template environment origin is
SHA-256 of ASCII `L2D-SARSA-V4-TEMPLATE-ENV\0`, canonical Regime digest, cue,
phase, branch, action, optional successor public/key encodings, reward, and done
in that order. Expanded origin is the already frozen environment-origin
preimage. Constructor A and B row bytes are the complete canonical FamilyRow
encoding. Both reported constructor fields deliberately use the same generic
root name `family_rows` over 84 template encodings followed by 8,256 expanded
encodings, so independent equal rows produce equal roots. Twin canonical/twin
payload bytes are likewise complete FamilyRow
encodings; only terminal reward and its derived environment origin may change.

Every record is encoded as fixed 13 bytes `L2D-SARSA-V4\0`, record tag U8,
schema U8=1, field count U16, then declared-order fields. `S` and variable `Y`
use U32 byte length then payload; fixed `Y[n]` and `D` have no length. A nested
record is U32 encoded length then its complete bytes. `Q` has no count and each
variable child has U32 length; `L` starts U32 count and each variable child has
U32 length. Composite tuple notation is encoded field-by-field without an
extra tag. Unknown fields, alternate ordering, mutable aliases,
Boolean-as-integer, out-of-range enum/integer, NaN, infinity, and cross-episode
identity reject.

The record digest is SHA-256 of its complete encoding. Every reported root
named `x_sha256` is SHA-256 of ASCII `L2D-SARSA-V4-ROOT\0`, UTF-8 `x`, one zero
byte, U32 item count, then each item as U32 length plus canonical bytes in the
declared order for that field. A `BoundarySnapshot` state hash is its record
digest. Its effect hash uses root name `boundary_effect` over exactly four
I64 encodings in file/network/process/native order. These rules are the sole
root/hash ABI unless a later section explicitly names a different complete
preimage.

Every embedded digest is derived, never opaque:

- observation digest is SHA-256 of ASCII `L2D-SARSA-V4-OBS\0`, episode digest,
  phase U8, marker F64, branch I64, nuisance F64, and 32 public bytes;
- token digest is SHA-256 of `L2D-SARSA-V4-TOKEN\0`, episode digest, phase,
  mode, forced action, and three token bytes;
- permit nonce is the first 16 bytes of SHA-256 of
  `L2D-SARSA-V4-PERMIT\0`, episode digest, permit class, phase, and ordinal;
- static row key is SHA-256 of `L2D-SARSA-V4-ROW\0`, episode digest, phase,
  branch I64, and action U8;
- environment origin is SHA-256 of `L2D-SARSA-V4-ENV\0`, episode digest,
  phase, action, optional successor tag/digest, reward F64, and done B;
- canonical feedback origin equals environment origin; zero-terminal-origin is
  SHA-256 of `L2D-SARSA-V4-NULL-ORIGIN\0`, episode digest, and phase;
- Action, Step, Update, Trajectory, source-receipt, snapshot, witness, static,
  and dynamic digests are their complete record digests. Observation and token
  complete-record digests are distinct from their semantic digest fields.

The bootstrap owns a write-only `commit_event` capability. Evidence is split
into independent, case-owned ledgers; there is deliberately no global event
count or cross-case chain. `LedgerId.kind` is case/family/twin/split/schedule/
train/table/metric/evaluation/comparator/source/trace/attack/control/
intervention/sanitizer = 0..15. The remaining identity fields use the exact
policy/path enum, variant enum, and local ordinal prescribed by the case table
below; unused fields are zero. A main case ledger has kind/path/variant/ordinal
all zero. A child ledger has nonzero kind and a unique complete identity.

The challenge for one ledger is SHA-256 of ASCII
`L2D-SARSA-V4-LEDGER-CHALLENGE\0`, revision bytes, fixture and worker digests,
packet-ABI digest, U32 length, and the complete canonical `LedgerId` bytes. The
fixture cannot choose or read the active ledger identity, challenge, roots, or
prior receipts. The bootstrap activates exactly one frozen ledger around the
named real-consumer call. Event tags are observation/token/permit/selection/
step/update/append/trajectory/source/comparator/static/dynamic/attack/
sanitizer/family/twin/split/table/metric/case-start/child-summary/case-end =
`0..21`. Tags 19..21 are bootstrap-only and cannot occur in approved fixture or
worker source.

For local ordinal `n`, payload is the exact canonical record or tuple declared
for that tag. Receipt is SHA-256 of ASCII `L2D-SARSA-V4-EVENT\0`, challenge,
U32 length plus complete `LedgerId` bytes, U64 `n`, event tag U8, U32 payload
length, payload, and previous 32-byte chain root; each ledger starts from 32
zero bytes. The receipt becomes that ledger's next root. Canonical event bytes
are the same fields from `LedgerId` through receipt. A per-channel root uses
generic root name `event_<tag-name>` over those bytes in local ordinal order.
No count, receipt, channel root, or terminal root is supplied by fixture code.

Every case main ledger contains, in order, bootstrap-owned `case-start`
payload `(case_id:U8,expected_children:U16)`, one `ChildSummary` event for each
child in ascending complete `LedgerId` byte order, then `case-end` payload
`(case_id:U8,observed_children:U16,payload_events:U64)`. A `ChildSummary` carries
all nineteen operational channel counts for tags 0..18, including zeroes, and
the child's exact terminal root. The case evidence root is the final main-ledger
receipt, so it is never empty; its main-ledger count is exactly children plus
two. The independent oracle reconstructs every child, summary, boundary, count,
and receipt. Any missing, duplicate, extra, cross-case, reordered, empty, or
miscounted child rejects. Cases execute in numeric order only for output; no
case root depends on another case's ledger.

Family payload is `(constructor:U8,domain:U8,row_index:U64,U32 row_length,
row_bytes)`, with constructor A/B = 0/1, domain template/expanded = 0/1, and
the row bytes exactly the independently specified row encoding. Twin payload is
`(domain:U8,row_index:U64,U32 canonical_length,canonical_bytes,U32
twin_length,twin_bytes)`, with domain template/expanded = 0/1; case 2 emits all
84 template pairs before all 8,256 expanded pairs, so its 28 template reward
changes are direct events rather than an aggregate inference. Split payload is
`(domain:U8,index:U16,U32
record_length,record_bytes)`, where domain regime/episode = 0/1 and record bytes
are the canonical `Regime` or `Episode`. Table payload is
`(predecessor_key:Y[2],action:U8,implementation:F64,oracle:F64)`. Metric payload
is `(policy_id:U8,split:U8,episode_digest:D,success:B,return:F64,regret:U8)`.
Comparator payload is `(policy_id:U8,regime_digest:D,episode_digest:D,
action_0:U8,action_1:U8,success:B,return:F64)`. `score_comparator` commits exactly
one comparator event after authenticating the terminal environment result and
before the matching metric event, for each of the 512 held-out episodes of each
case-10 policy, in policy, regime-table, then episode order. In cases 14..17,
the control candidate completes its candidate evaluation first; the same five
comparator streams then emit 2,560 comparator events in policy, regime-table,
episode order inside that control's evaluation child. Case 10
`comparator_event_root_sha256` is generic root `case10_comparator_channels` over
the five exact 32-byte comparator channel roots in policy order.
Permit payload is the complete `Permit`. The other operational payloads retain
the exact definitions below.

The complete case-to-consumer and event-count partition is frozen here. A
notation such as `observation=1728` is a required channel count, not an
informal estimate. Every unlisted operational channel count is zero.

| case | exact child ledgers and required operational counts |
| ---: | --- |
| 1 | one family child: `family=16680` (A template 84, B template 84, A expanded 8256, B expanded 8256) |
| 2 | one twin child: `twin=8340`, template 84 then expanded 8256 in row order |
| 3 | one split child: `split=1383` (7 regimes then 1376 episodes) |
| 4 | one schedule child: `token=1728` |
| 5 | one canonical-train child: `observation=1728, token=1728, permit=6912, selection=1728, step=1728, update=1728, append=1728, trajectory=864`, total 18144 |
| 6 | one table child: `table=12` in predecessor-key then action order |
| 7 | one train-metric child: `metric=864` |
| 8 | two regime evaluation children, amber then violet; each has `observation=256, permit=512, selection=256, step=256, metric=128, source=1`, total 1409 |
| 9 | two regime evaluation children, silver then gold, with the same 1409-event composition each |
| 10 | five four-regime evaluation children in constant-0, constant-1, myopic, no-bootstrap, random order; each has `observation=1024, permit=2048, selection=1024, step=1024, comparator=512, metric=512, source=4`, total 6148; one no-bootstrap train child has the exact case-5 18144-event composition |
| 11 | 21 path/mode children in path then absent/exploding/spy order, each case-5 composition plus `source=1`, total 18145; 14 path/order children in path then table/reverse order, each case-5 composition, total 18144; seven one-event `source` order-receipt children; one one-event `attack` inverse-injection child |
| 12 | one trace child: `observation=1728, selection=1728, step=1728, update=1728, append=1728, trajectory=864`, total 9504 |
| 13 | one attack child: `attack=24`, live IDs 0..11 then malformed IDs 0..11 |
| 14 | one zero-bootstrap train child with the case-5 18144-event composition and one four-regime evaluation child with `observation=1024, permit=2048, selection=1024, step=1024, comparator=2560, metric=512, source=4`, total 8196 |
| 15 | one phase-zero train child and one evaluation child with the same 18144/8196 compositions |
| 16 | one zero-origin train child and one evaluation child with the same 18144/8196 compositions |
| 17 | one marker-fresh train child with the case-5 composition, then marker-fresh and marker-frozen four-regime evaluation children with 8196 events each |
| 18 | three children in static/dynamic/sanitizer order: `static=41280`, `dynamic=14784`, and `sanitizer=48` |

The path, policy, regime, mode, order, witness, and control enum values in the
governing tables populate `LedgerId`. The exact namespaces are: ordinary
single-child cases use path/variant/ordinal zero; regime evaluation children
use kind evaluation, canonical policy 0, table-index regime as variant, ordinal
zero; case-10 children use kind comparator, policy enum as path, variant zero,
ordinal zero, while its training child uses kind train and no-bootstrap policy
4. Control children use kind control, control enum zero-bootstrap/phase-zero/
zero-origin/marker-fresh/marker-frozen = 0..4 as path, train/evaluation = 0/1 as
variant, ordinal zero. Case 11 alone uses these disjoint exact identities:
path/mode children `(kind=source,path=0..6,variant=mode 0..2,ordinal=0)`;
path/order execution children `(kind=train,path=0..6,variant=3+order 0..1,
ordinal=0)`; order-receipt children `(kind=source,path=0..6,variant=5,
ordinal=0)`; and inverse injection `(kind=attack,path=0,variant=6,ordinal=0)`.
Case 18 uses exactly `(case=18,kind=intervention,path=0,variant=0,ordinal=0)`
for static, the same identity with variant 1 for dynamic, and
`(case=18,kind=sanitizer,path=0,variant=0,ordinal=0)` for sanitizer.
Thus no mode, order, receipt, or attack child can collide. The table and this
mapping fix every child identity, membership, main-ledger child count,
main-ledger event count, and payload-event total. The bootstrap rejects an
activation not listed here before approved code runs.

Selection payload is Action fields through token digest, excluding selection
receipt; the returned receipt fills that field. Step payload is Step fields
through environment origin, excluding step receipt; update payload is Update
fields through step digest, excluding feedback receipt. Append payload is
episode digest, phase, Step digest, and Update digest; its returned receipt is
stored in Trajectory. Terminal ledger is the active child root after the second
append and before the trajectory event. `SourceReceipt.terminal_ledger` is the
active child root immediately before the closing source event; that event is
excluded from its own payload and becomes the child's final receipt.
Static/dynamic/attack/sanitizer payloads
are their exact records or the exact sanitizer `(source_bytes,rule_bytes)` tuple.

`commit_event` validates active ledger, event type, schema, identity, expected
next ordinal, consumer stage, uniqueness, and relation to already committed
records before advancing. Failed validation leaves state/effects unchanged. The
bootstrap, not fixture or worker, owns counts, per-channel roots, every terminal
root, and receipt values. `emit_projection` overwrites every count/root/receipt
leaf designated by cases 1..18 from those sealed ledgers and rejects disagreement
with the fixture tuple. Thus a literal worker count/root cannot pass without the
corresponding committed typed events. The controller oracle independently
reconstructs the identical event streams and case boundaries.

The source AST must contain `commit_event` calls only in exact functions
`commit_family_row`, `commit_twin_pair`, `commit_split_member`,
`make_observation`, `make_token`, `issue_permit`, `select_initial`, `select_next`,
`environment_step`, `sarsa_update`, `append_trace`, `seal_trajectory`,
`close_source`, `score_comparator`, `commit_static_projection`,
`commit_dynamic_projection`, `commit_table_cell`, `commit_metric`,
`run_rejection_attack`, and `audit_source_sample`.
Each call uses its literal matching event tag and occurs after all real-consumer
checks but before the function returns the record/receipt. No helper, alias,
default, closure, loop-supplied callable, or alternate call site may reference
the sink. The controller AST audit freezes and verifies this call-site table.

The training trace has five lists: 1,728 `Observation`, `Action`, `Step`, and
`Update` records plus 864 `Trajectory` records, totaling 7,776 components and
1,728 append receipts. Each list is independently reordered by key
`SHA256(U8 record_tag || complete_record_bytes)`, then complete bytes. Join key
for the first four types is `(episode_digest,phase)` and for trajectory is
`episode_digest`. Each action must name its observation; each step its action
and predecessor; each update its step, predecessor key, and applicable
successor/latched action; each trajectory its two steps, updates, and append
receipts. Canonical component root is tag order then episode order then phase;
the reordered join must reconstruct the same root. Duplicate, missing, extra,
stale, future, cross-episode, or digest-substituted components reject before a
trajectory is exposed.

The component root contains only realized canonical training records in exact
record-tag order 3, 6, 7, 8, 9, then train episode order and phase. Append
receipts are not a sixth component list: their exact payload/position is the
append event rule above and the two returned digests occur in each Trajectory.
`trace_event_root_sha256` is the controller-owned event-channel root over all
7,776 component events plus 1,728 append events in execution order. This root,
the component root, and reordered root must all match the independent oracle.

## Closed aggregate-root preimages

Unless a prior paragraph gives a stricter preimage, every result field ending
`_sha256` uses the generic-root ABI with root name equal to the exact field name
after removing `_sha256`. A listed typed record means its complete canonical
encoding; a listed event means its complete canonical event commitment bytes;
a listed child root means raw D bytes. These are the complete populations:

| result field(s) | exact items and order |
| --- | --- |
| `constructor_a_root_sha256`, `constructor_b_root_sha256` | the same 8,340 FamilyRow encodings and shared root name `family_rows` already frozen above |
| `chronology_event_root_sha256` | all 18,144 case-5 canonical-train child events in local ordinal order |
| `implementation_table_sha256`, `oracle_table_sha256` | shared root name `sarsa_table`; 12 items in predecessor-key then action order, each item predecessor key, action, then respectively implementation or oracle F64 |
| case-8/9 `boundary_before_sha256`, `boundary_after_sha256` | shared root name `evaluation_boundary`; one complete BoundarySnapshot encoding captured respectively before/after the split, required byte-identical |
| case-8/9 `source_root_sha256` | the two EvaluationSourceReceipt records in regime-table order |
| case-10 `input_root_sha256` | 2,560 items in policy, regime-table, episode order: policy U8, U32 phase-0 Observation length/bytes, U32 realized phase-1 Observation length/bytes |
| case-10 `action_root_sha256` | 2,560 items in the same order: policy U8, Episode digest D, action-0 U8, action-1 U8 |
| case-10 `output_root_sha256` | 2,560 items in the same order: policy U8, Episode digest D, success B, return F64 |
| `comparator_event_root_sha256` | the five comparator channel roots exactly as frozen above |
| case-11 `source_root_sha256` | 21 SourceReceipt records in path then absent/exploding/spy order |
| case-11 `order_root_sha256` | seven SourceOrderReceipt records in path order |
| case-11 `source_event_root_sha256` | the 28 source-event commitments from the 21 path/mode and seven order-receipt children, sorted by complete LedgerId |
| case-12 `component_root_sha256`, `reordered_root_sha256` | shared root name `trace_components`; the same 7,776 complete canonical component records in frozen tag, episode, phase order, with the latter reconstructed by independent reordering/joining |
| case-12 `trace_event_root_sha256` | all 9,504 trace-child event commitments in local ordinal order |
| case-13 `live_root_sha256`, `malformed_root_sha256` | respectively the twelve live and twelve malformed RejectionWitness records in witness-ID order |
| case-13 `attack_event_root_sha256` | all 24 attack-child event commitments, live then malformed |
| cases 14-16 `action_root_sha256` | 2,752 Action records: 1,728 train then 1,024 held-out, each in regime, episode, phase order |
| cases 14-16 `source_root_sha256` | four EvaluationSourceReceipt records in held-out regime-table order |
| cases 14-16 `trace_root_sha256` | train Observation/Action/Step/Update/Trajectory records in numeric record-tag then episode/phase order, followed by held-out Observation/Action/Step records in numeric record-tag then regime/episode/phase order |
| cases 14-16 `comparator_event_root_sha256` | one raw D item: the evaluation child's comparator channel root covering the five identical comparator streams |
| cases 14-16 `control_event_root_sha256` | the complete train and evaluation ChildSummary encodings in LedgerId order |
| case-17 `fresh_action_root_sha256`, `fresh_source_root_sha256`, `fresh_trace_root_sha256` | the same 2,752-action, four-receipt, and train-plus-held-out constructions as cases 14-16, using the fresh marker-ablation children |
| case-17 `frozen_action_root_sha256` | the 1,024 frozen-table held-out Action records in regime/episode/phase order |
| case-17 `frozen_source_root_sha256` | four frozen-table EvaluationSourceReceipt records in regime order |
| case-17 `frozen_trace_root_sha256` | frozen-table held-out Observation/Action/Step records in numeric record-tag then regime/episode/phase order |
| case-17 `fresh_comparator_event_root_sha256`, `frozen_comparator_event_root_sha256` | respectively one raw D item containing the fresh or frozen evaluation child's comparator channel root |
| case-17 `fresh_control_event_root_sha256` | complete fresh-train and fresh-evaluation ChildSummary encodings in LedgerId order |
| case-17 `frozen_control_event_root_sha256` | the complete frozen-evaluation ChildSummary encoding |
| case-18 static roots | each named root covers its exact contiguous 8,256 static-event commitments: canonical, zero-bootstrap, phase-zero, zero-origin, then marker, in that partition order |
| case-18 dynamic roots | canonical, zero-bootstrap, phase-zero, and zero-origin each cover their contiguous 2,752 dynamic-event commitments; marker covers its 2,752 fresh then 1,024 frozen commitments |
| `sanitizer_root_sha256`, `sanitizer_rule_root_sha256` | the 48 source and 48 rule byte items already frozen in the sanitizer section |

An aggregate field cannot substitute a child terminal root, projection leaf, or
record digest for the item type listed here. Item counts, order, exact root
name, and every nested encoding are independently replayed by the controller.

## Exact rejection contexts

The threat model is the approved first-party source and declared API. These
checks prove its real consumer/dataflow behavior; they do not prove containment
of malicious Python or native code.

Live witness `i=0..11` uses a fresh blank table and canonical episode
`train-copper, round=i//9, cue=0, block_position=i%9`. Malformed witness `i`
uses `train-indigo` with the same round/position and cue 1. These are valid,
distinct frozen train identities; there is no appended identity field. The
canonical source, token, actions, successors, permits, and records come from
constructor B and the sequential oracle. Exact live mutations, real consumers,
and exceptions in witness order are:

| i | exact preparation and mutated input | consumer | exception |
| ---: | --- | --- | --- |
| 0 | canonical selection permit with byte 0 of nonce XOR 1 | `select_initial` | `PermitIdentityError` |
| 1 | complete one valid initial selection, then submit the identical spent permit again | `select_initial` | `PermitSpentError` |
| 2 | replace only the token with the same-round/position cue-1 token | `select_initial` | `IdentityMismatchError` |
| 3 | replace token bytes with exact `(0,0,0)` and recompute its digest | `select_initial` | `TokenContractError` |
| 4 | with no selection performed, submit the oracle's canonical phase-0 action | `environment_step` | `ChronologyError` |
| 5 | select validly, then XOR selected action value with 1 while retaining its selection receipt | `environment_step` | `IdentityMismatchError` |
| 6 | before any step, submit the oracle phase-1 successor and next token | `select_next` | `ChronologyError` |
| 7 | select and step validly, then submit the oracle next action without calling `select_next` | `sarsa_update` | `ChronologyError` |
| 8 | complete phase 0 and the terminal step; in the raw terminal Step envelope replace absent `successor_digest` with the exact phase-1 predecessor Observation semantic digest, recompute `environment_origin` and the complete Step digest from those mutated fields, and retain `done=true` | `sarsa_update` | `TerminalContractError` |
| 9 | select, step, and next-select validly, then submit the oracle phase-0 update before updating | `append_trace` | `ChronologyError` |
| 10 | run through terminal update but omit terminal append | `expose_return` | `ReturnBoundaryError` |
| 11 | complete the episode and submit the identical terminal update to `append_trace` again | `append_trace` | `DuplicateIdentityError` |

Every live invocation supplies the complete argument tuple from the consumer
signature. Arguments not named as mutated in the table are the oracle's exact
canonical records/scalars for that context; the invoked operation receives its
exact unspent permit, except rows 0/1/11 where the table explicitly changes or
reuses permit/identity state. All preparation calls likewise use the oracle's
canonical full arguments and permits. Consumer-specific validation precedence
is the exception order frozen by this table; in row 11 duplicate trace identity
is checked before the already-spent append permit.

Malformed inputs start from the named canonical tuple constructor's ordered
`(field_name,value)` pairs. The exact mutation is applied before that real
constructor/consumer; no detached validator substitutes for it:

| i | exact mutation | consumer | exception |
| ---: | --- | --- | --- |
| 0 | remove Regime pair `name` | `construct_regime` | `MissingFieldError` |
| 1 | append Episode pair `unknown=0` | `construct_episode` | `UnknownFieldError` |
| 2 | replace Episode `cue` integer 1 with Boolean true | `construct_episode` | `ExactTypeError` |
| 3 | replace Observation marker with float bits `0x7ff8000000000000` | `authenticate_observation` | `FiniteFloatError` |
| 4 | remove the final byte from Observation `public_bytes` | `authenticate_observation` | `LayoutError` |
| 5 | remove the final byte from BehaviorToken `token_bytes` | `select_initial` | `LayoutError` |
| 6 | replace Observation `observation_digest` with the cue-0 donor digest while retaining cue-1 fields | `select_initial` | `CrossEpisodeError` |
| 7 | XOR byte 0 of the claimed Action digest accompanying the canonical Action | `environment_step` | `DigestMismatchError` |
| 8 | replace Step reward with float bits `0x7ff0000000000000` | `sarsa_update` | `FiniteFloatError` |
| 9 | replace Step `done` Boolean with exact integer 1 | `sarsa_update` | `ExactTypeError` |
| 10 | replace successor Observation phase 1 with integer 0 and recompute only its record digest | `select_next` | `ChronologyError` |
| 11 | after valid phase-0 append and terminal update, submit the phase-0 Step again in the terminal append slot | `append_trace` | `DuplicateIdentityError` |

Every harness consumer begins by decoding each record argument from a raw
envelope `(record_tag:U8, ordered_pairs)`, so malformed rows traverse the named
real consumer rather than a detached pre-validator. An envelope is encoded as
ASCII `L2D-SARSA-V4-RAW\0`, record tag, U16 pair count, then each field name as
U32 UTF-8 length/bytes and a value tag. Value tags none/Boolean/integer/float/
bytes/string/record/tuple are 0..7; integer is I64, float is raw U64 bits, bytes
and string are U32-length-prefixed, record uses U32 canonical-record length,
and tuple uses U32 count plus U32-length children. This permits NaN, infinity,
Boolean-as-integer attacks, missing, extra, and wrong-length values to reach the
consumer's own constructor. Canonical execution uses envelopes too, then seals
decoded immutable records before any stateful check.

For each row `context_sha256` is the canonical Episode digest and
`input_sha256` is the generic root named `attack_input`. Its first item is the
exact UTF-8 consumer-name bytes, its second item is the U32 argument count, and
each remaining item is one complete pre-mutation argument envelope or canonical
scalar in signature order, including the permit. The generic-root U32 item
length prefixes therefore frame the consumer, count, and every argument
independently; no item contains an additional implicit boundary.
`mutation_sha256` is SHA-256 of fixed ASCII
`L2D-SARSA-V4-MUTATION\0`, U8 kind, U8 witness index, U32 UTF-8 recipe length,
and the exact mutation-cell text above after replacing Markdown code marks with
their enclosed bytes. The outer validator owns the same literal recipe tuple
and reconstructs all three values. `BoundarySnapshot` is the complete mutable
state enumeration; its state/effect hashes use the frozen ABI above. The real
consumer must emit the exact observed exception name, equal before/after state
and effect hashes, and true rejection. A detached rejection or literal count
cannot satisfy a row.

## Host-feasible runtime and process boundary

Read-only probes were completed before this plan was frozen. On CPython
3.12.13, Windows build 10.0.26200.0, little endian, repository venv redirector
with SHA-256
`b800c702033743118942d2635c03a9858aea7731798694ba1b7d8f7393eb991b`,
`-I -S` successfully executed two byte-identical restricted-name-map probes.
The first produced 134 stdout bytes with SHA-256
`056d0ccecb6c0c7c7237089333d021a40a87fe834c4ef9e609a5b20a91520d21`
and zero stderr. The second exercised length-prefixed stdin source transport,
separate fixture and worker maps, single-use output, and two identical
reproductions. Its 195-byte packet SHA-256 was
`014fde0c3315eb237fef405c37fdaa7cd226b56bc735c8f094cb7104c1bc2237`;
its bootstrap SHA-256 was
`4914555606c4b3bd88711df2e05097360c75e9b23f2a5af5f4123240abe6c757`;
and its 149-byte stdout SHA-256 was
`0a9172bb726a8a8e7e39b5b9dc042353aceb402dd62050af3d81c87969c40c81`,
with zero stderr. These are feasibility receipts, not study evidence and not
future source hashes.

The venv executable is a Windows redirector, so it is not the future launch
target. A separate read-only Job probe resolved its base interpreter (the
absolute path remains local and only its normalized-path digest may be
committed) with SHA-256
`0dfbe445df3da9f94456d4e91d8feda5af587fe3cfb748e4c9fb62544909d385`.
That exact base executable ran a stdin-gated `-I -S -c` probe created suspended,
assigned to a Job before `NtResumeProcess`, and resumed with status zero. Under
active-process limit 1 it produced ASCII `OK`, zero stderr, exit zero, lifetime
process count 2 with no concurrent second process, and active-count samples
`[1,0]`. Earlier attempts against the venv redirector demonstrated why it is
excluded. No probe loaded study code or changed an ACL, owner, registry value,
or repository/private artifact.

A later implementation may add exactly two study modules:

- `experiments/local_lab/online_sarsa_latched_choice_v4.py`;
- `experiments/local_lab/online_sarsa_latched_choice_v4_worker.py`.

The approved source set is exactly those two modules, this frozen plan, and
`docs/AUTONOMOUS_LAB.md`. The controller and registry are pinned separately by
their complete committed bytes. No dependency package is required: the worker
runs pure stdlib Python under `-I -S`.

The controller launches exactly three sequential processes: one primary and
two reproduction processes. Each command uses the exact absolute base Python
path and SHA-256 above,
`-I -S -c <fixed-bootstrap>`, no shell, an empty controller-created scratch
cwd, a 300-second timeout, and a scrubbed environment containing only
`SYSTEMROOT`, `WINDIR`, `TEMP`, and `TMP`. Isolated mode ignores Python-specific
environment controls, so stdin/stdout are binary, every protocol byte is ASCII
or explicitly binary, all JSON uses `ensure_ascii=True`, and Python `hash()` or
unordered mapping/set iteration is forbidden from every preimage.

Each launch uses `subprocess.Popen` with exact argument vector, `shell=False`,
`close_fds=True`, no inherited handle except the three new pipes, and Windows
`CREATE_NO_WINDOW|CREATE_SUSPENDED`. Before any child instruction executes, the
controller assigns the process to a fresh unnamed Job Object with
`LimitFlags=0x00002008` (`KILL_ON_JOB_CLOSE|ACTIVE_PROCESS`) and
`ActiveProcessLimit=1`; process creation flags are `0x08000004`. It then calls exact
`ntdll!NtResumeProcess` and requires status zero. Failure to create, configure,
assign, or resume rejects before stdin is written. The controller writes one
packet and flushes but keeps stdin open. The bootstrap verifies source,
executes only the fixture's definition-only module body, constructs the closed
maps, writes exact 13 bytes `L2D-V4-READY\0`, and blocks before worker execution.
The controller verifies the sole active Job PID and read-only module manifest,
writes exact gate byte `0xA5`, flushes, closes stdin, and drains both output
pipes concurrently. Any alternate/trailing gate byte or missing EOF rejects.
Timeout or output overflow calls
`TerminateJobObject`, waits, and verifies active-process count zero. Normal exit
also requires active-process count zero, exit code zero, zero stderr, exact
stdout framing, and no surviving child. The Job and process handles are then
closed. Each launch permits at most 262,144 stdout bytes; the enclosing cycle
deadline remains 3,600 seconds.

The scratch parent is the exact sibling private lab root already approved by
the controller. Each scratch child is a newly generated UUID name; resolved
parent equality, nonexistence-before-create, current-user ownership, absence of
reparse points, and empty contents are checked before launch. `TEMP`, `TMP`, and
cwd all name that same child. Removal is allowed only after the Job reports zero
active processes and immutable output is sealed; any containment or cleanup
mismatch parks the controller.

The packet is little-endian uint64 outer length followed by a payload with
13-byte magic `L2D-SARSA-V4\0`, version byte 1, mode byte 1, fixture and worker
SHA-256 bytes, their two uint64 lengths, then fixture and worker blobs. The
payload header is exactly 95 bytes. Fixture, worker, and payload caps are
786,432, 131,072, and 1,048,576 bytes. Lengths, caps, hashes, and the exact
payload boundary are checked before allocation or compilation; only the later
one-byte execution gate and EOF may follow.

Each source blob is the complete raw committed file byte string, with no newline
or text normalization. Both files must be ASCII, use LF only, contain no CR or
NUL, and end in exactly one LF. The packet length is the raw-byte length, the
packet SHA-256 is over those identical raw bytes, and `compile` receives the
ASCII decoding of those same bytes without transformation. Approved-source
hashes also cover the same raw committed bytes. A checkout whose bytes differ
in hash, length, encoding, or line ending rejects before compilation.

`packet_abi_sha256` hashes this exact binary preimage: fixed bytes
`L2D-SARSA-V4-PACKET-ABI\0`, U8 schema 1, U8 outer-length width 8, exact
13-byte magic, U8 version 1, U8 mode 1, U8 digest width 32 twice, U8 inner-
length width 8 twice, then U64 fixture/worker/payload caps in that order.
It then appends U8 READY length 13, exact READY bytes, and U8 gate `0xA5`.
`bootstrap_sha256` hashes the exact ASCII bytes supplied as the `-c` argument,
with LF line endings and no normalization. The argument must be ASCII, contain
at most 16,384 bytes/code units, and its identical bytes must be committed as a
controller constant and implementation-manifest value before registration.

The trusted fixed bootstrap alone has stdin/stdout and compile/exec authority.
It verifies the packet and executes the fixture and worker in separate explicit
maps. The fixture's initial `__builtins__` mapping contains exactly
`AssertionError`, `Exception`, `RuntimeError`, `StopIteration`, `TypeError`,
`ValueError`, `abs`, `all`, `any`, `bool`, `bytearray`, `bytes`,
`enumerate`, `float`, `int`, `isinstance`, `iter`, `len`, `list`,
`max`, `min`, `next`, `range`, `reversed`, `round`,
`slice`, `sorted`, `str`, `sum`, `tuple`, and `zip`. Its initial globals are
exactly that builtins mapping, `__name__="l2d_sarsa_v4_fixture"`,
`__package__=None`, `__spec__=None`, `hashlib`, `math`, `struct`, the write-only
`commit_event` callable, and
the fifteen exception classes named in the rejection tables. Those exception
classes are controller-bootstrap-created direct subclasses of `Exception` and
have no added attributes. Fixture execution may add only top-level functions
and constants declared by its audited AST; after execution, their names must
equal the AST-derived definition/assignment set and the sole exported
capability is exact callable `run_study`.

The worker's `__builtins__` is an empty mapping. Its initial globals are exactly
that mapping, `__name__="l2d_sarsa_v4_worker"`, `__package__=None`,
`__spec__=None`, one-element tuple `fixture_api=(run_study,)`, immutable tuple
`approved_context`, and single-use bounded callable `emit_projection`. The
worker source is exact ASCII
`emit_projection(fixture_api[0](approved_context))\n`: one expression statement
and no added global. The returned object must be a newly allocated closed tuple
tree; the bootstrap rejects mutable children and aliases to fixture containers
before encoding.

`approved_context` is an exact fourteen-element tuple in this order: ASCII
study ID, 40-byte lowercase revision, ASCII claim boundary, then raw 32-byte
fixture, worker, frozen-plan, `AUTONOMOUS_LAB`, controller, normalized-registry,
Python-executable, bootstrap, packet-ABI, capability-map, and host-receipt
digests. It contains only bytes. `run_study` accepts that one tuple positional
argument, checks every field, and returns exact tuple cases 1..18. The emitter
maps that tuple to the frozen ordered JSON schema, rejects all aliases/types/
relations, and after the READY prefix writes U64 JSON byte length, ASCII JSON
bytes, then EOF exactly once. It rejects a claimed JSON length above 262,123
before allocation and rejects any byte after the declared JSON. It has no
return channel into the fixture.

For the fixture AST, `Import`, `ImportFrom`, `ClassDef`, `AsyncFunctionDef`,
`Await`, `Yield`, `YieldFrom`, `Global`, `Nonlocal`, `With`, `AsyncWith`,
`Delete`, `Match`, `Dict`, `Set`, `DictComp`, `SetComp`, and `GeneratorExp`
nodes are forbidden. Calls whose resolved name is
`__import__`, `open`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`,
`dir`, `getattr`, `setattr`, `delattr`, or `hasattr` are forbidden. Every
attribute segment or loaded source name beginning and ending with `__` is
forbidden. The identifiers
`os`, `sys`, `pathlib`, `io`, `socket`, `ssl`, `urllib`, `http`, `subprocess`,
`multiprocessing`, `ctypes`, `cffi`, `mmap`, `marshal`, `inspect`, `builtins`,
`pkgutil`, `importlib`, `runpy`, and `site` are forbidden in every name and
attribute position. Every loaded name must resolve to the exact initial map or
an AST-declared local, argument, function, or constant; stores may target only
AST-declared locals or top-level definitions. The fixture module body may
contain only function definitions and constant assignments whose values are
deeply immutable scalar/bytes/tuple trees; mutable top-level constants and every
mutable function default are forbidden. The capability-map preimage starts
fixed ASCII `L2D-SARSA-V4-CAPABILITY\0`, U8 schema 1, then seven U32-counted
lists. They are, in order: builtins, initial globals, exception classes,
AST-derived top-level definitions, loaded names, forbidden node/call names, and
forbidden identifiers. Each list is sorted by U8 namespace, raw ASCII name, U8
kind, then raw identity. Every item is U8 namespace fixture/worker = 0/1, U32 name
length/name, U8 kind, U32 identity length/identity; there is no delimiter or
escaping. The empty worker-builtins list is committed as an empty namespace-1
subsequence. Kinds builtin/global/exception/
definition/loaded/forbidden-node/forbidden-call/forbidden-identifier are 0..7.
Builtin identity is ASCII `builtins.`, exact object name, zero byte, and exact
type name. Module-global identity is U8 source/PE kind 0/1, U32 ASCII module-
name length/name, then raw SHA-256 of complete `.py` bytes after CRLF-to-LF
only or complete raw `.pyd` bytes; a missing/alternate suffix rejects.
Scalar/tuple constant identity uses a closed recursive encoding: None/Boolean/
integer/float/bytes/ASCII-string/tuple have type tags 0..6; None has no payload,
Boolean is `B`, integer is `I64`, float is canonical `F64`, bytes and string are
U32 length plus raw bytes, and tuple is U32 child count followed by each child as
U32 encoded length plus complete recursively encoded bytes. Integers outside
I64, nonfinite or negative-zero floats, non-ASCII strings, lists, mappings,
sets, aliases, and every other constant type reject. A constant-definition
identity is U32 ASCII name length/name followed by U32 encoded-value length and
that exact value. Thus scalar and tuple constants have no implementation-chosen
or repr-derived bytes;
callable identity is U32 controller-constant-name length/name plus its complete
controller-source digest. Exception identity is U32 exception-name length/name,
U32 base-name length plus `Exception`, and U16 zero namespace count. Simple
definition/loaded/forbidden items use zero identity length. This complete binary
string is the sole `capability_map_sha256` preimage and is committed before
registration.

The AST audit order is parse, forbidden node, forbidden call, dunder
attribute, forbidden identifier, loaded-name closure, then module-body shape;
the first matching rule is the expected rejection. The sanitizer matrix has
exactly these 48 ASCII samples in row-major order:

| rule family | sample 1 | sample 2 | sample 3 | sample 4 |
| --- | --- | --- | --- | --- |
| import node | `import os` | `import socket as s` | `from pathlib import Path` | `from subprocess import Popen` |
| dynamic import | `__import__("os")` | `__import__("sys")` | `__import__("socket")` | `__import__("ctypes")` |
| direct execution or I/O | `open("x")` | `eval("1")` | `exec("x=1")` | `compile("1","x","eval")` |
| namespace reflection | `globals()` | `locals()` | `vars()` | `dir()` |
| dynamic attribute | `getattr(x,"y")` | `setattr(x,"y",1)` | `delattr(x,"y")` | `hasattr(x,"y")` |
| dunder attribute | `x.__class__` | `x.__dict__` | `x.__globals__` | `x.__code__` |
| network identifier | `socket.socket()` | `ssl.SSLContext()` | `urllib.request.urlopen("x")` | `http.client.HTTPConnection("x")` |
| process identifier | `subprocess.Popen(())` | `multiprocessing.Process()` | `os.system("x")` | `runpy.run_path("x")` |
| native identifier | `ctypes.CDLL("x")` | `cffi.FFI()` | `mmap.mmap(0,1)` | `marshal.loads(b"x")` |
| introspection identifier | `inspect.getsource(x)` | `builtins.open("x")` | `sys.modules` | `pkgutil.iter_modules()` |
| loader identifier | `importlib.import_module("os")` | `runpy.run_module("x")` | `site.main()` | `io.open("x")` |
| forbidden node | `class X:\n pass` | `async def f():\n pass` | `global x` | `with x:\n pass` |

Table `\n`, `\"`, and `\\` use C-style decoding and one final LF is appended.
The 48 exact expected rule strings, in the same twelve rows, are:

1. `forbidden-node:Import`, `forbidden-node:Import`,
   `forbidden-node:ImportFrom`, `forbidden-node:ImportFrom`;
2. `forbidden-call:__import__` four times;
3. `forbidden-call:open`, `forbidden-call:eval`, `forbidden-call:exec`,
   `forbidden-call:compile`;
4. `forbidden-call:globals`, `forbidden-call:locals`, `forbidden-call:vars`,
   `forbidden-call:dir`;
5. `forbidden-call:getattr`, `forbidden-call:setattr`,
   `forbidden-call:delattr`, `forbidden-call:hasattr`;
6. `dunder-attribute:__class__`, `dunder-attribute:__dict__`,
   `dunder-attribute:__globals__`, `dunder-attribute:__code__`;
7. `forbidden-identifier:socket`, `forbidden-identifier:ssl`,
   `forbidden-identifier:urllib`, `forbidden-identifier:http`;
8. `forbidden-identifier:subprocess`,
   `forbidden-identifier:multiprocessing`, `forbidden-identifier:os`,
   `forbidden-identifier:runpy`;
9. `forbidden-identifier:ctypes`, `forbidden-identifier:cffi`,
   `forbidden-identifier:mmap`, `forbidden-identifier:marshal`;
10. `forbidden-identifier:inspect`, `forbidden-identifier:builtins`,
    `forbidden-identifier:sys`, `forbidden-identifier:pkgutil`;
11. `forbidden-identifier:importlib`, `forbidden-identifier:runpy`,
    `forbidden-identifier:site`, `forbidden-identifier:io`;
12. `forbidden-node:ClassDef`, `forbidden-node:AsyncFunctionDef`,
    `forbidden-node:Global`, `forbidden-node:With`.

`sanitizer_root_sha256` uses root name `sanitizer_source` over those 48 source
byte strings; `sanitizer_rule_root_sha256` uses root name `sanitizer_rule` over
the 48 expected first-rule ASCII names in the same order. The controller builds
both independently and requires the real source audit to produce every named
rejection.

Neither executed map contains import authority, a path, source text, repository
or private roots, credentials, or a callback other than the write-only bounded
event sink and one-way bounded emitter. After both compilations the bootstrap overwrites the mutable packet and
source bytearrays, deletes their references and decoded-source references, and
does not place any source object in an executed map; transient immutable Python
compiler objects are not claimed to be securely erased. This is a first-party
source-policy and capability-map claim, not containment of malicious Python or
native code and not an OS sandbox claim.

The controller never changes an ACL. The executable, Python base runtime, and
Windows dependencies are existing read-only host dependencies; implementation
preflight records their identities without taking ownership or altering them.
The controller resolves `sys._base_executable`, opens it with
`FILE_READ_ATTRIBUTES`, calls `GetFinalPathNameByHandleW` with
`FILE_NAME_NORMALIZED|VOLUME_NAME_DOS` for the normalized DOS
name, strips exact `\\?\` prefix, replaces backslash with slash, rejects
non-ASCII, and lowercases ASCII A-Z only. The host receipt preimage is U32
normalized-path length/bytes, raw executable SHA-256, U32 ASCII CPython-version
length/bytes, U32 cache-tag length/bytes, U32 Windows-build length/bytes, U32
`AMD64` length/bytes, and U32 `little` length/bytes in that order.

After READY and before GO, `QueryInformationJobObject` must return one active
PID. `K32EnumProcessModulesEx(LIST_MODULES_ALL)` and
`GetModuleFileNameExW` repeat until two consecutive sorted module sets match,
with at most three attempts. Every executable/DLL/PYD is opened read-only and
normalized by the same rule. Path digest is SHA-256 of ASCII
`L2D-SARSA-V4-PATH\0`, U32 path length, and normalized path. File digest covers
complete bytes. File version is four U16 values from `VS_FIXEDFILEINFO` in
major/minor/build/revision order. Owner SID is exact ASCII from
`ConvertSidToStringSidW`. Canonical SDDL is exact ASCII returned by
`ConvertSecurityDescriptorToStringSecurityDescriptorW` revision 1 with OWNER,
GROUP, and DACL security-information flags on the descriptor returned by
`GetNamedSecurityInfoW`; its raw bytes are hashed.

A manifest row is path digest D, file digest D, four version U16s, owner SID as
U32 length/bytes, and SDDL digest D. Rows sort by path digest then file digest;
duplicates reject. Generic root `os_dependency_manifest` over those row bytes
is case 19 `os_dependency_manifest_sha256`. Committed rows contain no raw path
or SDDL. The invocation must match the committed manifest, but no owner or
descriptor value is prescribed or changed. Its rows and expected digest are closed values
inside the V4 registry entry, not an approved worker source or extra file. If
the restricted maps, packet, bootstrap size,
environment, Job mechanics, three launches, scratch containment, or read-only
dependency treatment cannot be implemented, V4 is quarantined before
registration.

The controller's V4 oracle imports and calls neither approved study module. It
implements constructor B, schedule recurrence, primitive selection/update
equations, comparator/control transforms, trace joins, attacks, sanitizer
matrix, canonical encoding, and every expected scalar/root directly from this
plan. Before accepting a child, it independently reconstructs cases 1..18 and
compares every leaf and relation. Source review plus this independent replay is
the entire first-party execution claim; it is not hostile-code attestation.

The worker inner projection has exact ordered top-level fields
`schema_version`, `study_id`, `revision`, `claim_boundary`, and `cases`, with
schema `online-sarsa-v4-worker-v1`, exact ID and revision, claim
`synthetic-online-sarsa-harness-only`, and cases 1 through 18 below. The three
inner UTF-8 byte strings must match exactly. The controller then appends case
19 and creates the final ordered result fields `schema_version`, `study_id`,
`revision`, `status`, `action`, `claim_boundary`, and `cases`, with schema
`online-sarsa-v4-result-v1`. Workers never emit status, action, or case 19.

The normalized registry digest is SHA-256 of complete registry bytes after
replacing CRLF with LF and making no other transformation. The controller hash
is SHA-256 of the complete committed `tools/run_local_lab.py` bytes. Both are
verified before lease acquisition and repeated in case 19.

## Closed sanitized result schema

Type tags are: B exact Boolean; I exact signed JSON integer with Boolean
rejected; F exact finite JSON float with integer/Boolean and negative zero
rejected; S bounded exact string; H lowercase 64-hex; R lowercase 40-hex; A
closed array with fixed child schema and order. Duplicate, missing, extra, or
reordered keys reject under an object-pairs-preserving parser.

Nested JSON children are closed ordered objects:

- evaluation source receipt: `policy_id:S[32]`, `regime_id:S[32]`,
  `opens:I`, `yielded:I`, `other_operations:I`, `completed:B`,
  `receipt_sha256:H`;
- fit source receipt: `path_id:S[32]`, `mode:S[16]`, `train_episodes:I`,
  `validation_accesses:I`, `test_accesses:I`, `snapshot_accesses:I`,
  `terminal_ledger_sha256:H`, `completed:B`;
- source-order receipt: `path_id:S[32]`, `raw_a:H`, `raw_b:H`,
  `normalized_a:H`, `normalized_b:H`, `terminal_a:H`, `terminal_b:H`,
  `completed:B`;
- rejection witness: `witness_id:I`, `kind:S[9]`, `consumer:S[32]`,
  `expected_exception:S[32]`, `observed_exception:S[32]`,
  `context_sha256:H`, `input_sha256:H`, `mutation_sha256:H`,
  `before_state_sha256:H`, `after_state_sha256:H`,
  `before_effect_sha256:H`, `after_effect_sha256:H`, `rejected:B`.

Evaluation receipts are ordered by declared policy then regime table order; fit
receipts by path table order then absent/exploding/spy; order receipts by path;
witnesses by kind then witness ID. Child hashes use the internal record ABI and
the JSON digest field must equal the corresponding record digest. Arrays may
contain no alternate child shape.

JSON policy strings map to U8 in the exact policy-enum order already frozen;
regime strings are the seven exact table names and map to the digest of their
canonical Regime record; path strings are the seven exact fit-path names in
declared order; mode strings absent/exploding/spy map to 0/1/2. A child receipt
digest is recomputed only after these mappings, so a string cannot name a
different internal identity. Every `I` count is in `[0,2^63-1]`; witness IDs are
0..11, launches and process/scratch counts use their exact case-19 gates, and
no negative or Boolean integer is accepted.

Worker top-level scalar types are `schema_version:S` exact worker schema,
`study_id:S` exact ID, `revision:R`, `claim_boundary:S` exact claim, and
`cases:A` exactly 18 ordered case objects. Final top-level types are the exact
result schema/ID/revision, `status:S="passed"`,
`action:S="synthetic_online_sarsa_paired_fork_confirmed_for_harness"`, exact
claim, and exactly 19 ordered cases. Any inner failure prevents creation of a
passed final result and parks with the failure action; no failed projection is
rewritten into the passed schema.

Every case begins `case_id:S` equal to its numbered name, `passed:B=true`, and
`evidence_root_sha256:H`. For cases 1..18 that root is exactly the final receipt
of the case's bounded main ledger defined above; it includes its explicit start,
ordered child summaries, and end event, and an empty child or main ledger
rejects. Case 19 uses generic root name `case_19_evidence` over exactly nine
items in launch order: launch-0 ProcessReceipt, ManifestReceipt,
FramingReceipt, then the same three records for launches 1 and 2. Each item is
the complete canonical typed-record encoding, including its numeric record tag.
The independent oracle recomputes every root. Then each case has
exactly:

The complete ordered key vector of every case is the three-key prefix
`[case_id,passed,evidence_root_sha256]` followed by the field names in the exact
left-to-right order of that case's numbered declaration below. A field name is
the bytes before the first colon or equality/inequality marker inside each
backtick-delimited field token; explanatory backticks without a marker are not
fields. Cases 15 and 16 expand the complete case-14 vector unchanged and
substitute only their declared values; no key is inherited from any other prose.
This mechanical rule freezes complete vectors for cases 1..19. The
pairs-preserving parser independently derives the same vector from a committed
controller constant and rejects any missing, extra, duplicate, or reordered
key before checking values.

1. `complete_family_replay`: `template_rows:I=84`, `expanded_rows:I=8256`,
   `nonterminal_rows:I=2752`, `terminal_rows:I=5504`,
   `predecessor_identities:I=4128`, `realized_rows:I=2752`,
   `constructor_a_root_sha256:H`, `constructor_b_root_sha256:H` equal,
   `unique_keys:B=true`, `replay_exact:B=true`.
2. `evaluator_twin_boundary`: `changed_template_rewards:I=28`,
   `changed_expanded_rewards:I=2752`, `changed_public_fields:I=0`,
   `changed_nonterminal_fields:I=0`, `twin_exact:B=true`.
3. `split_source_commitment`: `regimes:I=7`, `train_regimes:I=3`,
   `validation_regimes:I=2`, `test_regimes:I=2`, `train_episodes:I=864`,
   `validation_episodes:I=256`, `test_episodes:I=256`,
   `heldout_in_fit_api:I=0`, `identities_disjoint:B=true`.
4. `behavior_schedule`: `forced_episodes:I=768`, `greedy_episodes:I=96`,
   `tokens:I=1728`, `paired_tokens:I=864`, `token_mismatches:I=0`,
   `schedule_exact:B=true`.
5. `online_chronology`: `initial_selections:I=864`,
   `next_selections:I=864`, `action_decisions:I=1728`, `steps:I=1728`,
   `updates:I=1728`, `appends:I=1728`, `terminal_returns:I=864`,
   `permits_spent:I=6912`, `order_mismatches:I=0`,
   `chronology_event_root_sha256:H`.
6. `sarsa_table`: `q_cells:I=12`, `positive_cells:I=4`,
   `target_mismatches:I=0`, `cell_mismatches:I=0`,
   `implementation_table_sha256:H`, `oracle_table_sha256:H` equal,
   `bit_exact:B=true`.
7. `train_metrics`: `forced_successes:I=192`, `greedy_successes:I=96`,
   `reward:F=288.0`, `regret:I=576`, `analytic_match:B=true`.
8. `validation_metrics`: `episodes:I=256`, `successes:I=256`,
   `amber_return:F=1.0`, `violet_return:F=1.0`, `mean_return:F=1.0`,
   `regret:I=0`, `updates:I=0`, `boundary_before_sha256:H`,
   `boundary_after_sha256:H` equal, `source_receipts:A` exactly two typed
   receipts in regime order, `source_root_sha256:H`, `action_replay:B=true`.
9. `test_metrics`: `episodes:I=256`, `successes:I=256`,
   `silver_return:F=1.0`, `gold_return:F=1.0`, `mean_return:F=1.0`,
   `regret:I=0`, `updates:I=0`, `boundary_before_sha256:H`,
   `boundary_after_sha256:H` equal, `source_receipts:A` exactly two receipts,
   `source_root_sha256:H`, `action_replay:B=true`.
10. `comparators`: `constant_zero_return:F=0.0`,
    `constant_one_return:F=0.0`, `myopic_return:F=0.0`,
    `no_bootstrap_return:F=0.5`, `random_validation_return:F=0.25`,
    `random_test_return:F=0.25`, `random_code_decisions:I=512`,
    `random_scalar_actions:I=1024`, `minimum_margin:F=0.5`,
     `input_root_sha256:H`, `action_root_sha256:H`, `output_root_sha256:H`,
     `comparator_event_root_sha256:H`, `replay_exact:B=true`.
11. `train_only_sources`: `source_mode_checks:I=21`, `spy_ledgers:I=14`,
    `heldout_fit_operations:I=0`, `source_receipts:A` exactly 21 path/mode
     receipts, `source_root_sha256:H`, `order_receipts:A` exactly seven path
     receipts, `order_root_sha256:H`, `raw_orders_distinct:B=true`,
     `normalized_orders_equal:B=true`, `inverse_injection_rejected:B=true`,
     `source_event_root_sha256:H`.
12. `trace_authentication`: `components:I=7776`, `joined_steps:I=1728`,
    `joined_trajectories:I=864`, `append_receipts:I=1728`,
     `component_root_sha256:H`, `reordered_root_sha256:H`,
     `trace_event_root_sha256:H`, `duplicates:I=0`, `unmatched:I=0`,
     `replay_exact:B=true`.
13. `boundary_and_malformed_rejection`: `live_attacks:I=12`,
    `live_rejected:I=12`, `live_witnesses:A` exactly twelve ordered closed
    witnesses, `live_root_sha256:H`, `malformed_attacks:I=12`,
     `malformed_rejected:I=12`, `malformed_witnesses:A` exactly twelve,
     `malformed_root_sha256:H`, `attack_event_root_sha256:H`,
     `state_mutations:I=0`, `all_rejected:B=true`.
Cases 14 through 16 use exactly the flat key order written for case 14; cases
15 and 16 substitute only the explicitly stated scalar values and their own
roots. Case 17's gate vector is the elementwise conjunction of the fresh and
frozen common vectors; both individual vectors must equal it.

| case | forced | greedy | reward | regret | each regime successes | each regime return | validation | test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| zero-bootstrap | 192 | 48 | 240.0 | 624 | 64 | 0.5 | 0.5 | 0.5 |
| phase-zero-assignment | 192 | 48 | 240.0 | 624 | 64 | 0.5 | 0.5 | 0.5 |
| zero-terminal-origin | 192 | 0 | 192.0 | 672 | 0 | 0.0 | 0.0 | 0.0 |

For every row the fixed chronology/count/replay/isolation fields are exactly
those shown in case 14, each hash field is that case's controller-owned event
commitment, and all booleans have the case-14 values. No key or relation is
inherited by informal convention beyond this explicit template/table.

14. `zero_bootstrap_control`: `train_forced_successes:I=192`,
    `train_greedy_successes:I=48`, `train_reward:F=240.0`,
    `train_regret:I=624`, `amber_successes:I=64`, `amber_return:F=0.5`,
    `violet_successes:I=64`, `violet_return:F=0.5`,
    `validation_return:F=0.5`, `silver_successes:I=64`,
    `silver_return:F=0.5`, `gold_successes:I=64`, `gold_return:F=0.5`,
    `test_return:F=0.5`, `train_actions:I=1728`, `train_steps:I=1728`,
    `train_updates:I=1728`, `train_appends:I=1728`,
    `train_trajectories:I=864`, `heldout_episodes:I=512`,
    `heldout_actions:I=1024`, `heldout_steps:I=1024`,
    `heldout_updates:I=0`,
    `action_root_sha256:H`, `source_root_sha256:H`, `trace_root_sha256:H`,
    `comparator_event_root_sha256:H`, `control_event_root_sha256:H`,
    `action_replay:B=true`, `source_isolation:B=true`,
    `trace_authenticated:B=true`, `comparator_replay:B=true`,
    `gate_vector:A` exactly
    `[true,true,false,false,false,false,false,false,true,true,true,true]`,
    `failed_indices:S="3,4,5,6,7,8"`,
    `failed_count:I=6`, `completed:B=true`,
    `identical_evaluator:B=true`, `positive_gate_recovered:B=false`.
15. `phase_zero_assignment_control`: the same fields and exact gates as case
    14, including 192/48/240.0/624 and 0.5/0.5.
16. `zero_terminal_origin_control`: the same control fields with exact
    train values 192/0/192.0/672, every per-regime success/return 0/0.0, and
    validation/test 0.0/0.0; the same failed indices, completion, evaluator,
    replay/isolation/authentication, and recovery gates.
17. `marker_ablation_control`: `train_forced_successes:I=192`,
    `train_greedy_successes:I=96`, `train_reward:F=288.0`,
    `train_regret:I=576`, `fresh_amber_successes:I=64`,
    `fresh_amber_return:F=0.5`, `fresh_violet_successes:I=64`,
    `fresh_violet_return:F=0.5`, `fresh_validation_return:F=0.5`,
    `fresh_silver_successes:I=64`, `fresh_silver_return:F=0.5`,
    `fresh_gold_successes:I=64`, `fresh_gold_return:F=0.5`,
    `fresh_test_return:F=0.5`, `frozen_amber_successes:I=64`,
    `frozen_amber_return:F=0.5`, `frozen_violet_successes:I=64`,
    `frozen_violet_return:F=0.5`, `frozen_validation_return:F=0.5`,
    `frozen_silver_successes:I=64`, `frozen_silver_return:F=0.5`,
    `frozen_gold_successes:I=64`, `frozen_gold_return:F=0.5`,
    `frozen_test_return:F=0.5`, `train_actions:I=1728`,
    `train_steps:I=1728`, `train_updates:I=1728`,
    `train_appends:I=1728`, `train_trajectories:I=864`,
    `fresh_heldout_episodes:I=512`, `fresh_heldout_actions:I=1024`,
    `fresh_heldout_steps:I=1024`, `fresh_heldout_updates:I=0`,
    `frozen_heldout_episodes:I=512`, `frozen_heldout_actions:I=1024`,
    `frozen_heldout_steps:I=1024`, `frozen_heldout_updates:I=0`,
    `fresh_action_root_sha256:H`, `fresh_source_root_sha256:H`,
    `fresh_trace_root_sha256:H`, `fresh_comparator_event_root_sha256:H`,
    `fresh_control_event_root_sha256:H`,
    `frozen_action_root_sha256:H`,
    `frozen_source_root_sha256:H`, `frozen_trace_root_sha256:H`,
    `frozen_comparator_event_root_sha256:H`,
    `frozen_control_event_root_sha256:H`,
    `fresh_action_replay:B=true`, `fresh_source_isolation:B=true`,
    `fresh_trace_authenticated:B=true`, `frozen_action_replay:B=true`,
    `frozen_source_isolation:B=true`, `frozen_trace_authenticated:B=true`,
    `fresh_comparator_replay:B=true`, `frozen_comparator_replay:B=true`;
    `gate_vector:A=[true,true,false,false,false,false,false,false,true,true,true,true]`,
    `failed_indices:S="3,4,5,6,7,8"`, `failed_count:I=6`,
    `ablated_rows:I=8256`, `unablated_marker_rows:I=0`, `completed:B=true`,
    `identical_evaluator:B=true`, `positive_gate_recovered:B=false`.
18. `intervention_and_sanitizer`: `paired_static_rows:I=33024`,
    `zero_bootstrap_bootstrap_key_differences:I=2752`,
    `zero_bootstrap_control_tag_differences:I=8256`,
    `phase_zero_update_address_differences:I=1376`,
    `phase_zero_control_tag_differences:I=8256`,
    `zero_origin_update_reward_differences:I=1376`,
    `zero_origin_feedback_origin_differences:I=5504`,
    `zero_origin_control_tag_differences:I=8256`,
    `marker_predecessor_public_differences:I=8256`,
    `marker_predecessor_key_differences:I=4128`,
    `marker_successor_public_differences:I=2752`,
    `marker_successor_key_differences:I=1376`,
    `marker_bootstrap_key_differences:I=1376`,
    `marker_control_tag_differences:I=8256`,
    `unexpected_static_differences:I=0`, `missing_required_differences:I=0`,
    `unexplained_dynamic_differences:I=0`, `causal_replay:B=true`,
    `static_events:I=41280`, `dynamic_events:I=14784`,
    `canonical_static_root_sha256:H`, `zero_bootstrap_static_root_sha256:H`,
    `phase_zero_static_root_sha256:H`, `zero_origin_static_root_sha256:H`,
    `marker_static_root_sha256:H`, `canonical_dynamic_root_sha256:H`,
    `zero_bootstrap_dynamic_root_sha256:H`,
    `phase_zero_dynamic_root_sha256:H`, `zero_origin_dynamic_root_sha256:H`,
    `marker_dynamic_root_sha256:H`,
    `sanitizer_samples:I=48`, `sanitizer_rejected:I=48`,
    `sanitizer_root_sha256:H`, `sanitizer_rule_root_sha256:H`,
    `inner_schema_exact:B=true`.
19. `runtime_reproduction`: controller-only fields `launches:I=3`,
    `reproduction_children:I=2`, `surviving_children:I=0`, `stderr_bytes:I=0`,
    `maximum_stdout_bytes:I<=262144`, `projections_equal:B=true`,
    `approved_sources:I=4`,
    `python_executable_sha256:H="0dfbe445df3da9f94456d4e91d8feda5af587fe3cfb748e4c9fb62544909d385"`,
    `bootstrap_sha256:H`, `packet_abi_sha256:H`,
    `capability_map_sha256:H`, `host_receipt_sha256:H`,
    `os_dependency_manifest_sha256:H`, `module_rows:I` equal the committed
    manifest row count, `module_sets_stable:B=true`, `case_ledgers:I=18`,
    `child_ledgers:I=74`, `empty_ledgers:I=0`, `process_receipts:I=3`,
    `manifest_receipts:I=3`, `framing_receipts:I=3`, `controller_sha256:H`,
    `registry_sha256:H`, `bootstrap_ascii_bytes:I<=16384`,
    `jobs_created:I=3`, `jobs_assigned:I=3`, `suspended_launches:I=3`,
    `assignments_before_resume:I=3`, `resume_failures:I=0`,
    `maximum_concurrent_job_processes:I=1`, `active_processes_after:I=0`,
    `unexpected_inherited_handles:I=0`, `ready_frames:I=3`, `go_bytes:I=3`,
    `scratch_directories:I=3`,
    `scratch_containment:B=true`, `scratch_removed:B=true`,
    `stdout_framing_exact:B=true`,
    `unexpected_projection_fields:I=0`, `isolated_mode:B=true`,
    `site_disabled:B=true`, `cpu_only:B=true`.

Case 19 is a relation gate over the nine typed receipts, not a second asserted
summary. Launch indices must be exactly 0, 1, 2 across all three receipt types.
Every ProcessReceipt source/runtime digest must be byte-identical across the
three launches and equal the same-named case field or the registered fixture/
worker digest; every ManifestReceipt dependency root and row count must equal
the committed manifest field/count; all three module-set roots must be equal.
`jobs_created`, `jobs_assigned`, `suspended_launches`,
`assignments_before_resume`,
`ready_frames`, `go_bytes`, and `scratch_directories` are the sums of their
corresponding ProcessReceipt Booleans. `resume_failures` is three minus the sum
of `resume_succeeded`. `surviving_children` is exactly the sum of
`ProcessReceipt.active_processes_after`; `unexpected_inherited_handles` is the
sum of the same-named receipt field. Every ProcessReceipt `exit_code` must equal
zero. `maximum_concurrent_job_processes` is exactly the maximum of
`ProcessReceipt.maximum_active_processes`; `maximum_stdout_bytes` is exactly
the maximum of ProcessReceipt stdout bytes. `active_processes_after`, `stderr_bytes`,
and the three scratch failure counts are sums. `scratch_containment` and
`scratch_removed` are conjunctions. `stdout_framing_exact` is the conjunction
of all FramingReceipt flags, every READY frame and GO byte must equal its fixed
value, each framing packet hash must match the independently reconstructed
packet, and each projection hash must be identical. `projections_equal` is
exactly that three-way projection-hash and byte-length equality. No aggregate
may pass when its receipt-level equality fails.
Every ManifestReceipt `all_files_read_only` must be true; their conjunction is
part of `module_sets_stable`, and a false value fails case 19 even when all
module hashes match.

The source/evaluation receipt child objects use the exact record fields above.
Witness strings are fixed by the attack tables and at most 64 UTF-8 bytes.
No path, exception message, repr, traceback, source text, raw observation,
token, table, target, topology, credential, or private identifier may enter a
result. The final JSON encoder is `ensure_ascii=True`, `allow_nan=False`,
compact separators, declared insertion order, and one final LF. The outer
validator recomputes every equality, count, metric, case pass, status, and
action; worker Booleans are not accepted without their closed evidence.

## Stopping rule and actions

This plan and the minimal truthful handoff update are the entire checkpoint.
The commit containing them is the immutable freeze boundary. No fixture,
worker, registry, controller, learner, result, or submission file is changed in
this checkpoint.

A later heartbeat may implement only this contract, run focused structural
tests and at most one full repository pass, obtain three hostile read-only
audits, commit exact source approvals and registry/controller identities, and
leave the private controller `awaiting_study`. A still later heartbeat may
invoke V4 exactly once through the guarded controller only after clean revision,
green CI, absent stop and lease, source/protected hash agreement, and proof that
the ID has never run.

Success requires all nineteen cases and uses exact action
`synthetic_online_sarsa_paired_fork_confirmed_for_harness`. Any failed gate,
recovered control, malformed or nondeterministic projection, runtime mismatch,
timeout, source drift, or process survivor uses
`park_online_control_v4_research`. A substantive pre-result confound
quarantines V4 without registration or invocation and requires a fresh ID.
There is no retry, top-up, alternate seed, threshold change, case removal,
same-ID repair, or direct fixture/worker execution.

The protected submission tree, ZIP SHA-256, and manifest SHA-256 remain,
respectively, `e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`,
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`, and
`99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`.

## Claim boundary

A future pass may say only that this fixed local-CPU synthetic harness selected
actions online under the frozen forced/greedy schedule, applied the declared
SARSA next-action update, acquired the deliberately public two-step cue/fork
association, froze its table on untouched held-out sources, beat the frozen toy
comparators, and lost the common performance gate under the exact bootstrap,
assignment, terminal-origin, and marker interventions.

It cannot support a claim that learning was necessary, that the task lacked a
programmed shortcut, that the exploration policy is optimal, or that the
result generalizes beyond this family. It is not evidence for production RL,
meta-RL, hidden topology, official data, UIFO, optimizer quality, candidate
selection, a native rewrite, accelerator value, leaderboard rank, competition
score, or permission to change or upload the submission or spend money.
