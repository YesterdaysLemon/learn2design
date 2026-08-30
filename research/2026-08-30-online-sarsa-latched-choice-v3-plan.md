# Frozen plan: online SARSA repeated-code control mechanics

Date: 2026-08-30
Study ID: `online-sarsa-latched-choice-v3`
Status: frozen plan only; no fixture, worker, registry entry, execution, or result

## Scope and exclusions

This checkpoint freezes one new deterministic, topology-independent online
control study before any implementation or result observation. The study asks
whether a tabular on-policy learner can acquire a deliberately exposed binary
cue-to-three-bit-action-code association from its own precommitted exploratory
behavior and propagate terminal feedback through three online SARSA updates.

`multistep-td-propagation-v1`, `multistep-td-action-prefix-v2`,
`online-sarsa-latched-choice-v1`, and `online-sarsa-latched-choice-v2` remain
quarantined. Their families, schedules, regimes, seeds, thresholds, controls,
code, and development diagnostics are not inputs to this plan. The earlier
terminal studies, official data, private evidence, leaderboard information,
and protected candidate are likewise not selection inputs. The numeric choices
below are frozen from the algebra of this self-contained family.

This plan commit authorizes documentation only. Exact implementation plus
hostile pre-result audit is a separate later checkpoint. Result-bearing
execution, if ever authorized, is a third checkpoint and may happen at most
once through the guarded local controller.

## Narrow falsifiable question

Can a blank deterministic tabular SARSA(0) learner, updating only from its own
frozen forced/greedy behavior on three train generator regimes, learn the
cue-dependent three-action sequence, score exact `1.0` mean return on all four
untouched held-out regimes without held-out updates, beat constant, myopic,
no-bootstrap, and seeded-random comparators by the frozen margins, and lose the
association under identically evaluated disjoint-zero bootstrap,
behavior-assignment, terminal-reward-origin, and public-marker-ablation
controls?

A pass can validate only this exact synthetic online-control harness and toy
signal. It cannot validate that learning is necessary, that the public signal
has no shortcut, or anything about general RL, hidden topology, the submitted
optimizer, candidate quality, accelerator value, or competition score.

## Frozen generator regimes and splits

There are exactly seven regimes. A regime is the immutable tuple
`(name, split, seed, marker_scale, nuisance_offset)`:

| name | split | seed | marker scale | nuisance offset |
| --- | --- | ---: | ---: | ---: |
| `train-a` | train | 1201 | 1.00 | 0 |
| `train-b` | train | 1213 | 1.00 | 0 |
| `train-c` | train | 1237 | 1.00 | 0 |
| `validation-a` | validation | 2401 | 1.00 | 0 |
| `validation-b` | validation | 2417 | 1.00 | 0 |
| `test-a` | test | 3607 | 1.00 | 0 |
| `test-b` | test | 3613 | 1.00 | 0 |

Regime seeds are identity commitments, not learner randomness. Generator
construction sorts by the table order and must reproduce these rows byte for
byte. Train, validation, and test serialized internal identities are pairwise
disjoint. Public observation bytes are deliberately regime-invariant for a
given cue, phase, and prefix; no marker magnitude, nuisance value, or other raw
field reveals split, regime, or seed. Validation and test sources are absent
from the train-only fitting API.

The public cue is `cue_bin in {0, 1}`. Its evaluator target action code is
fixed as:

| cue | target code | phase actions |
| ---: | ---: | --- |
| 0 | 3 | `(0, 1, 1)` |
| 1 | 4 | `(1, 0, 0)` |

The cue marker deliberately reveals this toy target: `signed_marker` is
exact `-1.0` for cue 0 and `+1.0` for cue 1. Evaluator truth is nevertheless
excluded from learner capabilities; only the public observation is available.

## Complete episode family

Every episode has exactly three decisions at phases 0, 1, and 2. Each action
is an exact integer in `{0, 1}`. Before phase `p`, `prefix_code` contains the
base-two code of the prior `p` actions and `prefix_width == p`. Appending action
`a` gives `successor_code = 2 * prefix_code + a` and
`successor_width = prefix_width + 1`. The successor is terminal exactly when
`successor_width == 3`.

The environment returns reward `0.0` on both nonterminal transitions. At the
terminal transition it returns `1.0` exactly when the completed code equals
the evaluator target for the episode cue, otherwise `0.0`. `done` is false on
the first two transitions and true on the third. Reward and success are
materialized only after the environment consumes the chosen action. The
learner never receives an evaluator target, target code, success bit, future
reward, or terminal outcome before that boundary.

For each regime, the committed complete family contains both cues and every
legal prefix/action transition, including all terminal successes and failures.
Each cue has `1 + 2 + 4 = 7` predecessor states and
`2 + 4 + 8 = 14` legal transition rows, so the frozen complete-family totals
are:

| quantity | per regime | all seven regimes |
| --- | ---: | ---: |
| cue roots | 2 | 14 |
| predecessor states | 14 | 98 |
| legal transition rows | 28 | 196 |
| nonterminal rows | 12 | 84 |
| terminal rows | 16 | 112 |
| successful terminal rows | 2 | 14 |

Every row key is
`(regime_name, split, regime_seed, cue_bin, phase, prefix_width, prefix_code, action)`.
Keys must be unique and sorted canonically. Train rows are independently
replayed through the real environment consumer before learner construction;
held-out rows are independently replayed only after fitting and table freeze as
specified by the physical boundary below. Replay authenticates
the exact observation, successor observation or terminal sentinel, reward,
`done`, predecessor identity, action, and evaluator-origin identity. Any
missing, extra, duplicate, malformed, illegal, or mismatched row fails closed.

The 196 regime-template rows are also instantiated for every scheduled train
or evaluation episode identity, using only the episode's cue-specific 14-row
subfamily. There are 1,800 training episodes and 1,024 held-out evaluation
episodes, hence exactly 2,824 episode instances. The expanded commitment has
39,536 unique legal rows, 16,944 nonterminal rows, 22,592 terminal rows, and
19,768 predecessor identities. The exercised action streams select exactly
8,472 of those rows. Expanded keys prefix the template key with the full
`EpisodeIdentity`; realized rows must be a subset with no duplicate and every
unrealized legal row must remain independently replayable.

The evaluator-twin replay freezes the episode identity, serialized public
predecessor, and action bytes, swaps only the private evaluator target code 3
and 4, and regenerates the outcome. Exactly
four of the sixteen terminal rows per regime change reward: two within each
cue subfamily. Every public byte, nonterminal reward, successor, and done bit
remains equal. Thus exactly 28 of the 112 all-regime terminal template rows and
exactly 5,648 expanded terminal rows differ in reward. This twin is a boundary
proof only and is never exposed to the learner.

## Training and evaluation schedules

Training runs regime-major in the table order. Each train regime has exactly
12 rounds. Each round visits cue 0 then cue 1. A cue block contains 25 episodes:
first exactly 24 forced episodes, formed by taking the frozen code order
`[0, 1, 2, 5, 6, 7, 3, 4]` and repeating each listed code for three consecutive
episodes, and only then exactly one greedy episode. A forced
code contributes its three phase actions in most-significant-bit order.

The schedule is constructed solely from `(regime_name, round, cue_bin,
block_position)` and the literal code order. It cannot read target code,
reward, Q values, evaluator state, terminal outcome, or a prior action. Paired
cue blocks receive byte-identical behavior tokens. There are exactly:

| training quantity | per regime | all train regimes |
| --- | ---: | ---: |
| rounds | 12 | 36 |
| cue blocks | 24 | 72 |
| episodes | 600 | 1,800 |
| forced episodes | 576 | 1,728 |
| greedy episodes | 24 | 72 |
| action decisions and updates | 1,800 | 5,400 |

The exact behavior token is three unsigned bytes
`(mode, forced_action, phase)`, where mode is 1 for forced and 0 for greedy;
greedy tokens set `forced_action` to 255. There are 5,400 tokens and exactly
2,700 cross-cue token pairs. The sealed adapter's three-byte public token input
must be byte-identical within each pair; only its copied `mode` and
`forced_action` integers enter the learner. The private
`BehaviorTokenEnvelope` binds each token copy to its distinct episode identity
without entering those public bytes.
Token construction, envelope binding, and decoding are independently replayed.

Evaluation freezes the fitted table. Each validation and test regime contains
exactly 256 episodes: 128 cue-0 and 128 cue-1 episodes in strict alternation,
starting with cue 0. Evaluation is greedy only, performs no update, consumes no
behavior token, and accepts a read-only table snapshot. The validation and test
action streams are independently replayed from the snapshot and must match the
original byte for byte.

The sealed `evaluate_frozen` consumer hashes the immutable 28-cell snapshot and
all module/class/global state before entry. For each held-out identity in the
frozen order, it constructs the phase-0 private observation envelope and then
performs exactly three repetitions of: derive the policy frame from the public
view, select the smallest argmax from the read-only snapshot, pass only that
action and the private evaluator envelope to `evaluation_step`, and append the
authenticated evaluation record. It accepts no behavior token, learner object,
RNG, callback, update permit, or mutable table. Done must be false, false, true;
all rewards before the third step must be zero. At exit every before/after state
hash must match.

The evaluation read-only check is authenticated rather than reported as a
scalar assertion. Immediately before the first held-out source is opened and
immediately after the last output record is sealed, an independent harness
constructs the typed `EvaluationBoundarySnapshot`. Its digest is
`SHA256(b"v3-evaluation-boundary\x00" ||
EvaluationBoundarySnapshot_record_bytes)`. The before and after digests must be
byte-identical for validation and independently for test. The snapshot covers
the fitted table, complete learner/module/class/global/closure state, permit and
operation ledgers, and all file/network/process/native effect ledgers. Expected
evaluation cursor and source-reader progress are deliberately excluded and are
committed instead by the source receipt and output roots below.

Each held-out regime is wrapped by a sealed operation-counting source and emits
one typed `EvaluationSourceReceipt`: the exact policy ID, stream tag
`policy_id + ":" + regime_name`, regime digest, one factory open, 256 yielded
episodes, zero other operations, and true completion. Canonical validation and
test each use policy ID `canonical` and hash their two receipts in regime-table
order as `SHA256(b"v3-evaluation-source\x00" ||
canonical_sequence_of_complete_EvaluationSourceReceipt_records)`. The pinned
outer validator reconstructs
both receipts from the frozen regime records, independently derives both roots,
and requires the receipt identities to join one-to-one to the 512 evaluator
contract and output records. A producer count or `state_mutations=0` without
equal boundary digests and the exact source root fails closed.

`evaluator_contract_sha256` commits the ordered episode identities, evaluator
origin digests, episode counts, tie rule, metric definition, read-only boundary,
zero-update rule, and the exact evaluator configuration before a policy acts.
The configuration digest is
`SHA256(b"v3-evaluator-config\x00" || canonical_primitive_sequence)` where the
sequence is exactly: config version `online-sarsa-v3-evaluator-config-v1`;
stream order `validation-a`, `validation-b`, `test-a`, `test-b`; 256 episodes
per regime; cue alternation starting at zero; tie literal
`smallest-argmax-action-0-before-1`; metric literal
`mean-terminal-return-and-episode-regret-v1`; built-in true read-only and false
updates-allowed; these five comparator-definition tuples in order:
`(constant-0,constant-action,0)`, `(constant-1,constant-action,1)`,
`(myopic-marker,marker-sign,negative=0,positive=1)`,
`(no-bootstrap,online-alpha-1,nonterminal-target=reward,tie=0)`, and
`(seeded-random,sha256-lowbit,seed=141421356,domain=online-sarsa-v3-random)`;
positive-gate version `online-sarsa-v3-positive-gate-22-v1` and clause count
22; required held-out mean 1.0; constant/myopic/no-bootstrap margin 0.90;
random margin 0.55; and random maximum 0.35. Every
`EvaluatorContractRecord.evaluator_config_sha256` carries that same digest.
The outer validator independently reconstructs it from these literals and the
frozen comparator formulas. It contains no action-dependent public observation.
`policy_observation_stream_sha256` separately commits the
three public observations actually presented along each policy trajectory;
`policy_snapshot_sha256` commits the read-only table; and
`output_trace_sha256` commits the resulting ordered action, reward, done, and
return records. Independent replay reconstructs all four. Controls preserve
the evaluator contract exactly. Their observation, policy, and output hashes
may differ only through the declared intervention and its authenticated causal
closure.

Train `EpisodeIdentity` values set `round_or_index` to the zero-based round and
`episode_in_block` to the zero-based position 0 through 24. Held-out identities
set `round_or_index` to the zero-based evaluation index 0 through 255 and
`episode_in_block` to the reserved exact integer 255. The split and regime name
make these layouts disjoint. The private `EvaluatorEnvelope.target_code` is 3
or 4 from the frozen cue table; the evaluator twin swaps only that field and
its origin digest.

## Typed public and private contracts

The immutable public observation is a contiguous little-endian float64 vector
of length five:

`(phase, signed_marker, prefix_code, prefix_width, nuisance)`.

`public_bytes` is exactly the 40-byte concatenation of those five little-endian
binary64 values in that order. The marker occupies bytes 8 through 15 inclusive.
The exact terminal policy-key sentinel is four bytes `FF FF FF FF`; an absent
terminal public observation uses optional-tag `0x00` and carries no payload.

`phase` and `prefix_width` are equal exact integers represented as floats.
`prefix_code` is legal for the width. `nuisance` is
`phase / 8 + prefix_code / 64`, which is target- and regime-independent.
The learner's four-byte policy-frame key is the exact unsigned-byte tuple
`(phase, cue_slot, prefix_width, prefix_code)` reconstructed only from the
authenticated public vector. Canonically, a negative marker maps to cue slot 0
and a positive marker maps to cue slot 1; zero is illegal. In the declared
marker-ablation control only, exact positive zero maps both cues to slot 0. No
scale, nuisance, regime name, seed, split, episode index, evaluator cue field,
reward, target, success, done, or future field enters the key.

The blank Q table has exactly 14 reachable policy-frame states and 28 float64
cells, all initialized to exact positive zero. Its only mutable operation is
the sealed SARSA update consumer. Snapshot serialization sorts keys and action
indices and rejects nonfinite values, negative zero, missing/extra cells, and
non-float64 storage.

The learner's complete persistent mutable state is exactly those 28 Q cells.
The learner object is slot-closed to that table; it has no visit counter,
history, cache, RNG, hidden state, episode field, last cue/reward/action,
module/class mutable, closure write, callback, property side effect, or ambient
registry. Selection is pure with respect to persistent state; one authenticated
update may change exactly one addressed cell. Reflection and before/after
module, class, instance, closure, and global snapshots enforce this closure on
canonical and control paths. Evaluation receives only an immutable 28-cell
snapshot.

Typed records are immutable, slot-backed dataclasses with strict construction:

| record | exact fields |
| --- | --- |
| `RegimeSpec` | `name:str`, `split:str`, `seed:int`, `marker_scale:float64`, `nuisance_offset:int` |
| `EpisodeIdentity` | `regime_name:str`, `split:str`, `seed:int`, `round_or_index:int`, `cue_bin:int`, `episode_in_block:int` |
| `EvaluatorEnvelope` | `episode_digest:bytes32`, `target_code:int`, `origin_id:bytes32` |
| `PublicObservation` | `phase:int`, `signed_marker:float64`, `prefix_code:int`, `prefix_width:int`, `nuisance:float64`, `public_bytes:bytes` |
| `ObservationEnvelope` | `episode:EpisodeIdentity`, `public:PublicObservation`, `observation_digest:bytes32` |
| `PublicBehaviorToken` | `mode:int`, `forced_action:int`, `phase:int`, `token_bytes:bytes` |
| `BehaviorTokenEnvelope` | `episode:EpisodeIdentity`, `public:PublicBehaviorToken`, `token_digest:bytes32` |
| `PermitIdentity` | `permit_class:str`, `episode_digest:bytes32`, `phase:int`, `sequence:int`, `nonce:bytes32` |
| `FeedbackOrigin` | `kind:str`, `episode_digest:bytes32`, `environment_origin_digest:bytes32`, `scalar:float64`, `origin_digest:bytes32` |
| `OperationReceipt` | `episode_digest:bytes32`, `operation:str`, `phase:int`, `permit_class:str`, `spent_identity_digest:bytes32`, `consumer_digest:bytes32`, `prior_ledger_digest:bytes32`, `receipt_digest:bytes32` |
| `ActionSelection` | `episode:EpisodeIdentity`, `phase:int`, `value:int`, `observation_digest:bytes32`, `token_digest:bytes32`, `selection_receipt_digest:bytes32` |
| `StepRecord` | `episode:EpisodeIdentity`, `predecessor:ObservationEnvelope`, `action:ActionSelection`, `successor:ObservationEnvelope-or-None`, `reward:float64`, `done:bool`, `origin_id:bytes32`, `step_receipt_digest:bytes32` |
| `UpdateRecord` | `episode:EpisodeIdentity`, `predecessor_key:bytes4`, `action:int`, `reward:float64`, `feedback_origin_digest:bytes32`, `successor_key:bytes4-or-None`, `next_action:int-or-None`, `next_token_digest:bytes32-or-None`, `next_selection_receipt_digest:bytes32-or-None`, `target:float64`, `old_value:float64`, `new_value:float64`, `done:bool`, `step_digest:bytes32`, `feedback_receipt_digest:bytes32`, `prior_ledger_digest:bytes32` |
| `Trajectory` | `episode:EpisodeIdentity`, `steps:sequence[3,StepRecord]`, `updates:sequence[3,UpdateRecord]`, `append_receipt_digests:sequence[3,bytes32]`, `episode_return:float64`, `terminal_origin_digest:bytes32`, `terminal_ledger_digest:bytes32` |
| `StaticProjection` | `row_key:bytes32`, `episode:EpisodeIdentity`, `regime:RegimeSpec`, `evaluator_target_code:int`, `predecessor_public_bytes:bytes`, `predecessor_policy_key:bytes4`, `action:int`, `successor_public_bytes:bytes-or-None`, `successor_policy_key_or_terminal:bytes4`, `environment_reward:float64`, `update_reward:float64`, `done:bool`, `feedback_origin_digest:bytes32`, `bootstrap_lookup_key_or_terminal:bytes4`, `update_address_action:int`, `control_tag:str` |
| `DynamicProjection` | `episode:EpisodeIdentity`, `phase:int`, `predecessor_observation_digest:bytes32`, `predecessor_public_bytes:bytes`, `predecessor_policy_key:bytes4`, `behavior_token_digest:bytes32`, `selected_action:int`, `selection_receipt_digest:bytes32`, `environment_action:int`, `successor_observation_digest:bytes32-or-None`, `successor_public_bytes:bytes-or-None`, `successor_policy_key_or_terminal:bytes4`, `environment_reward:float64`, `environment_origin_digest:bytes32`, `done:bool`, `next_behavior_token_digest:bytes32-or-None`, `latched_next_action:int-or-None`, `next_selection_receipt_digest:bytes32-or-None`, `bootstrap_lookup_key_or_terminal:bytes4`, `update_address_action:int`, `update_reward:float64`, `feedback_origin_digest:bytes32`, `target:float64`, `old_value:float64`, `new_value:float64`, `step_receipt_digest:bytes32`, `feedback_receipt_digest:bytes32`, `append_receipt_digest:bytes32`, `prior_ledger_digest:bytes32`, `terminal_ledger_digest:bytes32-or-None`, `trajectory_return:float64-or-None`, `control_tag:str` |
| `ControlTransform` | `control_id:str`, `version:str`, `root_field_indices:sequence[int]`, `applicability_predicates:sequence[str]`, `required_counts:sequence[int]` |
| `SourceDeliveryEvent` | `fit_path_id:str`, `delivery_position:int`, `regime_digest:bytes32` |
| `EvaluatorContractRecord` | `stream_tag:str`, `episode:EpisodeIdentity`, `evaluator_origin_digest:bytes32`, `evaluator_config_sha256:bytes32`, `tie_rule:str`, `metric:str`, `read_only:bool`, `updates_allowed:bool` |
| `PolicyObservationRecord` | `stream_tag:str`, `episode:EpisodeIdentity`, `public_observations:sequence[3,bytes]` |
| `PolicySnapshotRecord` | `stream_tag:str`, `table_sha256:bytes32` |
| `EvaluationOutputRecord` | `stream_tag:str`, `episode:EpisodeIdentity`, `actions:sequence[3,int]`, `rewards:sequence[3,float64]`, `dones:sequence[3,bool]`, `episode_return:float64` |
| `ComparatorTrace` | `comparator_id:str`, `episode:EpisodeIdentity`, `public_observations:sequence[3,bytes]`, `actions:sequence[3,int]`, `rewards:sequence[3,float64]`, `dones:sequence[3,bool]`, `episode_return:float64` |
| `BoundarySnapshot` | `q_table_sha256:bytes32`, `learner_instance_sha256:bytes32`, `module_state_sha256:bytes32`, `class_state_sha256:bytes32`, `global_state_sha256:bytes32`, `closure_state_sha256:bytes32`, `permit_ledger_sha256:bytes32`, `operation_ledger_sha256:bytes32`, `environment_cursor_sha256:bytes32`, `token_schedule_sha256:bytes32`, `rng_state_sha256:bytes32`, `hook_state_sha256:bytes32`, `source_spy_sha256:bytes32`, `filesystem_effect_sha256:bytes32`, `network_effect_sha256:bytes32`, `process_effect_sha256:bytes32`, `native_effect_sha256:bytes32` |
| `AstPolicy` | `version:str`, `allowed_node_kinds:sequence[str]`, `allowed_identifiers:sequence[str]`, `forbidden_identifiers:sequence[str]`, `allowed_literals:sequence[str]` |
| `RuntimeManifest` | `revision:str`, `python_executable_sha256:bytes32`, `python_dll_set_sha256:bytes32`, `stdlib_tree_sha256:bytes32`, `os_build:str`, `appcontainer_sid:str`, `capabilities:sequence[str]`, `acl_profile_sha256:bytes32`, `job_policy_sha256:bytes32`, `network_policy_sha256:bytes32`, `native_policy_sha256:bytes32`, `environment_sha256:bytes32`, `bootstrap_sha256:bytes32`, `packet_abi_sha256:bytes32`, `capability_map_sha256:bytes32`, `worker_interface_sha256:bytes32`, `learner_adapter_sha256:bytes32`, `ast_policy_sha256:bytes32`, `controller_sha256:bytes32`, `registry_sha256:bytes32`, `approved_source_hashes:sequence[5,bytes32]`, `worker_blob_limit:int`, `fixture_blob_limit:int`, `manifest_blob_limit:int`, `packet_payload_limit:int`, `stdout_cap:int`, `timeout_seconds:int`, `cpu_only:bool` |
| `LearnerAdapterSpec` | `version:str`, `function_signature:str`, `marker_source:str`, `q_row_order:sequence[2,int]`, `mode_source:str`, `forced_action_source:str`, `greedy_forced_sentinel:int`, `forced_mode:int`, `greedy_mode:int`, `return_type:str` |
| `EvaluationBoundarySnapshot` | `q_table_sha256:bytes32`, `learner_instance_sha256:bytes32`, `module_state_sha256:bytes32`, `class_state_sha256:bytes32`, `global_state_sha256:bytes32`, `closure_state_sha256:bytes32`, `permit_ledger_sha256:bytes32`, `operation_ledger_sha256:bytes32`, `filesystem_effect_sha256:bytes32`, `network_effect_sha256:bytes32`, `process_effect_sha256:bytes32`, `native_effect_sha256:bytes32` |
| `SourceBoundaryReceipt` | `fit_path_id:str`, `source_mode:str`, `train_episodes:int`, `validation_factory_accesses:int`, `test_factory_accesses:int`, `policy_snapshot_accesses:int`, `terminal_train_ledger_sha256:bytes32`, `completed:bool` |
| `EvaluationSourceReceipt` | `policy_id:str`, `stream_tag:str`, `regime_digest:bytes32`, `factory_opens:int`, `episodes_yielded:int`, `other_operations:int`, `completed:bool` |
| `SourceOrderReceipt` | `fit_path_id:str`, `raw_order_a_sha256:bytes32`, `raw_order_b_sha256:bytes32`, `normalized_order_a_sha256:bytes32`, `normalized_order_b_sha256:bytes32`, `terminal_order_a_sha256:bytes32`, `terminal_order_b_sha256:bytes32`, `completed:bool` |

Exact built-in `bool` is required for Boolean fields; `bool` is forbidden for
integers; NaN, infinity, negative zero, subclass instances, coercion, unknown
fields, mutable aliases, and cross-episode identities are rejected. Every
`successor_policy_key_or_terminal` and `bootstrap_lookup_key_or_terminal` field
is always four bytes: legal nonterminal key or exact `FF FF FF FF` terminal
sentinel. `UpdateRecord.successor_key` is a distinct optional field and is
exactly `None` on terminal rows; it is never the four-byte sentinel. Only absent
successor observations, terminal `UpdateRecord.successor_key`, terminal
`UpdateRecord.next_action`, terminal token/selection fields, terminal
ledger values before completion, and pending returns use `None`.
Every serialized structure uses a versioned length-delimited canonical encoding
and SHA-256 digest. The learner receives `PublicObservation` and
`PublicBehaviorToken` only through a sealed primitive adapter; envelopes,
`EpisodeIdentity`, and `EvaluatorEnvelope` never cross its capability boundary.
The adapter authenticates both immutable views, then calls exactly
`select_policy_action(marker, q_row, mode, forced_action)`, where `marker` is
the copied float64 `PublicObservation.signed_marker`, `q_row` is the immutable
two-float tuple `(Q(policy_key, 0), Q(policy_key, 1))`, `mode` is the exact int
`PublicBehaviorToken.mode`, and `forced_action` is its exact int field. Forced
mode is `1`; greedy mode is `0`; greedy uses sentinel `255`. No phase, prefix,
nuisance, bytes object, dataclass, envelope, key, table, digest, or
attribute-capable object enters the learner call. The exact return is built-in
int `0` or `1`.

The exact `LearnerAdapterSpec` is version `learner-adapter-v1`, function
signature
`select_policy_action(marker:float,q_row:tuple2-float,mode:int,forced_action:int)->int`,
marker source `PublicObservation.signed_marker`, Q order `(0, 1)`, mode source
`PublicBehaviorToken.mode`, forced-action source
`PublicBehaviorToken.forced_action`, sentinel `255`, forced mode `1`, greedy
mode `0`, and return type `exact-builtin-int-0-or-1`. Its typed record digest is
`learner_adapter_sha256 = SHA256(b"v3-learner-adapter\x00" ||
LearnerAdapterSpec_record_bytes)`. The sealed adapter, worker-interface
commitment, AST commitment, manifest, and outer validator all bind that same
digest. The learner definition has exactly four unannotated positional
arguments, no defaults, varargs, kwargs, decorators, or second function, and
its frozen body is semantically and AST-exactly:

```python
def select_policy_action(marker, q_row, mode, forced_action):
    if mode == 1:
        return int(forced_action)
    q0 = q_row[0]
    q1 = q_row[1]
    if q1 > q0:
        return 1
    return 0
```

`marker` is intentionally an authenticated but unread primitive in this fixed
body; the sealed adapter uses the authenticated public observation to construct
the policy key before extracting `q_row`. This makes the public-signal path
explicit without permitting dataclass attribute access in learner code. JSON is
used only for the final sanitized projection.

The learner returns only an exact action integer. The sealed selection wrapper
constructs `ActionSelection` from that value and the private envelopes and
receipt; study code cannot construct or mutate the authenticated record.

### Canonical binary ABI

All commitment, identity, receipt, trace, ordering, comparator, and origin
hashes use this exact ABI. A record begins with ASCII
`L2D-SARSA-V3\x00`, a one-byte record-type tag fixed by the typed-record table
order, schema byte `0x01`, and one-byte field count, then fields in declared
order. Exact encodings are: signed integers as little-endian two's-complement
int64; Booleans as one byte `0x00` or `0x01`; floats as raw little-endian
IEEE-754 binary64 after the finite/positive-zero checks; strings as UTF-8 with
little-endian uint32 byte length; fixed `bytes4` and `bytes32` as raw bytes;
other bytes as little-endian uint32 length plus bytes; optional values as tag
`0x00` for absent or `0x01` followed by the value; nested records as uint32
length plus their complete record bytes; and sequences as uint32 item count
followed by uint32-length-delimited item encodings. No Unicode normalization,
coercion, padding, platform-native width, map ordering, or alternate encoding
is permitted.

Record tags are exact: `RegimeSpec=0x01`, `EpisodeIdentity=0x02`,
`EvaluatorEnvelope=0x03`, `PublicObservation=0x04`,
`ObservationEnvelope=0x05`, `PublicBehaviorToken=0x06`,
`BehaviorTokenEnvelope=0x07`, `PermitIdentity=0x08`,
`FeedbackOrigin=0x09`, `OperationReceipt=0x0A`,
`ActionSelection=0x0B`, `StepRecord=0x0C`, `UpdateRecord=0x0D`,
`Trajectory=0x0E`, `StaticProjection=0x0F`, `DynamicProjection=0x10`, and
`ControlTransform=0x11`, `SourceDeliveryEvent=0x12`,
`EvaluatorContractRecord=0x13`, `PolicyObservationRecord=0x14`,
`PolicySnapshotRecord=0x15`, `EvaluationOutputRecord=0x16`,
`ComparatorTrace=0x17`, `BoundarySnapshot=0x18`, `AstPolicy=0x19`,
`RuntimeManifest=0x1A`, `LearnerAdapterSpec=0x1B`,
`EvaluationBoundarySnapshot=0x1C`, `SourceBoundaryReceipt=0x1D`, and
`EvaluationSourceReceipt=0x1E`, and `SourceOrderReceipt=0x1F`.

Every digest is SHA-256 over an exact domain prefix followed by these bytes.
The exact remaining prefixes are `v3-regime\x00`, `v3-episode\x00`,
`v3-evaluator-origin\x00`, `v3-observation\x00`, `v3-token\x00`,
`v3-permit-nonce\x00`, `v3-permit-identity\x00`,
`v3-feedback-origin\x00`, `v3-operation-receipt\x00`, `v3-step\x00`,
`v3-action\x00`, `v3-update\x00`, `v3-trajectory\x00`, `v3-table\x00`,
`v3-evaluator-contract\x00`, `v3-policy-observation\x00`, `v3-policy-snapshot\x00`,
`v3-evaluation-output\x00`, `v3-witness-root\x00`,
`v3-control-root\x00`, `v3-component-root\x00`,
`v3-reordered-root\x00`, `v3-append-root\x00`, `v3-ledger-root\x00`, `v3-dynamic-root\x00`, and
`v3-projection\x00`, plus `v3-component-order\x00`,
`v3-static-applicability\x00`, `v3-consumer-receipt-root\x00`,
`v3-control-transform\x00`, `v3-bootstrap-capability-map\x00`, and
`v3-row-key\x00`, `v3-ledger-genesis\x00`, `v3-ledger\x00`,
`v3-consumer\x00`, `v3-source-delivery\x00`, `v3-source-normalized\x00`,
`v3-source-delivery-path\x00`, `v3-source-normalized-path\x00`,
`v3-source-order\x00`, `v3-stateless-ledger\x00`,
`v3-comparator-input\x00`, `v3-comparator-action\x00`,
`v3-comparator-output\x00`, `v3-boundary-snapshot\x00`,
`v3-boundary-effect\x00`, `v3-dynamic-aggregate\x00`,
`v3-ast-policy\x00`, `v3-runtime-manifest\x00`, `v3-packet-abi\x00`,
`v3-worker-interface\x00`, `v3-learner-adapter\x00`,
`v3-evaluation-boundary\x00`, `v3-evaluation-boundary-receipt\x00`,
`v3-source-boundary\x00`,
`v3-evaluation-source\x00`, `v3-evaluator-config\x00`, and
`v3-learner-ast\x00`, `v3-python-dll-set\x00`, `v3-stdlib-tree\x00`,
`v3-acl-profile\x00`, `v3-job-policy\x00`, `v3-network-policy\x00`,
`v3-native-policy\x00`, `v3-environment\x00`, and
`online-sarsa-v3-random\x00`, each used only for the matching named object or
ordering rule.

Self-containing fields have one closed preimage rule. The field is replaced by
exactly 32 zero bytes while its record bytes are encoded, then the named domain
hash is written into the field. This applies to
`EvaluatorEnvelope.origin_id` with `v3-evaluator-origin`,
`ObservationEnvelope.observation_digest` with `v3-observation`,
`BehaviorTokenEnvelope.token_digest` with `v3-token`,
`PermitIdentity.nonce` with `v3-permit-nonce`,
`FeedbackOrigin.origin_digest` with `v3-feedback-origin`, and
`OperationReceipt.receipt_digest` with `v3-operation-receipt`. The identity
digest placed in `OperationReceipt.spent_identity_digest` is separately
`SHA256(b"v3-permit-identity\x00" || complete_PermitIdentity_record_bytes)`.
Canonical feedback has `kind="environment"`, the matching episode digest,
the evaluator envelope origin ID, the exact environment scalar, and its derived
origin digest. The terminal-origin control instead has `kind="null-control"`,
the same episode digest, an all-zero environment-origin digest, scalar positive
zero, and its independently derived origin digest. An update must bind the
exact supplied origin through `UpdateRecord.feedback_origin_digest`.
Permit issuance is deterministic and contains no random source: within each of
the five classes (`selection`, `step`, `next-action`, `feedback`, `append`),
`sequence` is the zero-based ordinal in the frozen regime/round/cue/episode/
phase chronology, `phase` is the exact consuming phase, and `nonce` follows the
zeroed-field rule above. A permit is issued only after its exact predecessor
receipt is authenticated. Any skipped, duplicate, out-of-order, or cross-class
ordinal rejects before the permit becomes visible.

All root preimages are closed as follows. A canonical record sequence is a
little-endian uint32 count followed by each item as a uint32 byte length and
the item's complete ABI bytes or, for result-only evidence objects, the exact
compact ordered JSON bytes frozen below. `table_sha256` sorts all 28 triples by
the four-byte key and action, encodes each as `bytes4 || uint8 action ||
binary64 value`, and hashes the resulting sequence with `v3-table`.
`evaluator_contract_sha256` hashes the canonical sequence of complete
`EvaluatorContractRecord`s. `policy_observation_stream_sha256` hashes the
canonical sequence of complete `PolicyObservationRecord`s.
`policy_snapshot_sha256` hashes the canonical sequence of complete
`PolicySnapshotRecord`s. `output_trace_sha256` hashes the canonical sequence of
complete `EvaluationOutputRecord`s. Witness and control roots hash their exact ordered closed evidence
objects. Component, append-receipt, and terminal-ledger roots hash the ordered
component record digests, append receipt digests, and terminal ledger digests
respectively; the reordered root hashes the independently joined canonical
component sequence. The dynamic root hashes the complete ordered dynamic
projections defined below. `projection_sha256` hashes the exact compact JSON
bytes of cases 1 through 20 only. A root field is never included in its own
preimage. Empty sequences use count zero and no item bytes; no producer-chosen
alternate preimage is accepted.

A `canonical_primitive_sequence` is likewise exact: little-endian uint32 item
count, then for each item one type byte, little-endian uint32 payload length,
and payload. Type bytes and payloads are `0x01` UTF-8 string bytes, `0x02`
signed little-endian int64, `0x03` one built-in Boolean byte, `0x04` checked
little-endian float64, `0x05` raw bytes32, `0x06` arbitrary raw bytes, and
`0x07` a recursively complete canonical primitive sequence. A tuple or list is
type `0x07`; no map, null, implicit coercion, omitted empty item, alternate
width, or platform-native encoding exists. Every use of “canonical primitive
sequence” below uses only this framing.

The 32-byte expanded `StaticProjection.row_key` is
`SHA256(b"v3-row-key\x00" || EpisodeIdentity_record_bytes ||
int64_le(phase) || int64_le(prefix_width) || int64_le(prefix_code) ||
int64_le(action))`. The episode record already binds regime, split, seed, cue,
round/index, and block position. Template-key ordering uses the same suffix
after `RegimeSpec_record_bytes || int64_le(cue_bin)` and is never substituted
for the expanded key.

Every control's `transform_sha256` is
`SHA256(b"v3-control-transform\x00" || ControlTransform_record_bytes)` with
`version="control-transform-v1"` and the exact root indices, predicate strings, and counts frozen in the static
difference table. `applicability_root_sha256` hashes, under
`v3-static-applicability`, one control-framed sequence. Its preimage is the
exact control ID, then uint32 group count, then groups in the
`ControlTransform.root_field_indices` order. Each group is
`int64_le(root_field_index) || uint32_len(predicate_utf8) || predicate_utf8 ||
uint32_applicable_row_count`, followed by one item per applicable row sorted by
`row_key`: `raw_32_byte_row_key || uint32_len(StaticProjection_record_bytes) ||
StaticProjection_record_bytes`. A row that satisfies two predicates appears
once in each indexed group; groups are never deduplicated or flattened. Group
counts must equal the paired required counts in the transform record. This
field-indexed preimage makes every overlap and repeated row unambiguous.
`consumer_receipt_root_sha256`
hashes, under `v3-consumer-receipt-root`, the canonical chronology-ordered
sequence of the exact real-consumer `OperationReceipt` bytes reached by that
transform. `dynamic_projection_root_sha256` hashes, under `v3-dynamic-root`,
the canonical chronology-ordered sequence of complete `DynamicProjection`
records. A control receipt is valid only when these four independently replayed
commitments agree with its control ID, transform, static predicates, dynamic
causal closure, operation ledgers, and case metrics. No scalar count or receipt
root is accepted without those real-consumer relations.

Serialized tag literals are closed by this table. A shorter spelling elsewhere
in prose is only a human label and is never accepted on the ABI:

| role | exact serialized literal |
| --- | --- |
| canonical control/policy tag | `canonical` |
| disjoint control ID/tag | `disjoint-zero-bootstrap` |
| assignment control ID/tag | `zero-assignment` |
| terminal-origin control ID/tag | `zero-terminal-origin` |
| marker control ID/tag | `marker-ablation` |
| tie rule | `smallest-argmax-action-0-before-1` |
| metric | `mean-terminal-return-and-episode-regret-v1` |

`StaticProjection.control_tag`, `DynamicProjection.control_tag`,
`ControlTransform.control_id`, and `ControlReceipt.control_id` use only those
five control/tag literals. Evaluator contract records use only the regime name
(`validation-a`, `validation-b`, `test-a`, or `test-b`) as `stream_tag`.
Policy-observation, snapshot, and output records use exact
`policy_id + ":" + regime_name`, where `policy_id` is one of the five literals
above or exact `marker-ablation-fresh` or `marker-ablation-frozen`. No
underscored case ID, shortened label, slash, alias, or free-form tag is legal.

The four exact `ControlTransform` root arrays are:

- `disjoint-zero-bootstrap`: indices `[14,16]`, predicates `["nonterminal-row",
  "every-row"]`, counts `[16944,39536]`;
- `zero-assignment`: `[15,16]`, `["canonical-action-is-one","every-row"]`,
  `[19768,39536]`;
- `zero-terminal-origin`: `[11,13,16]`, `["successful-terminal-row",
  "terminal-row","every-row"]`, `[2824,22592,39536]`;
- `marker-ablation`: `[5,6,8,9,14,16]`, `["every-row","cue-one-row",
  "nonterminal-row","nonterminal-cue-one-row",
  "nonterminal-cue-one-row","every-row"]`,
  `[39536,19768,16944,8472,8472,39536]`.

Thus overlapping predicates remain field-indexed. The real-consumer receipt
subsets, always in episode chronology, are exactly 3,600 nonterminal feedback
receipts for disjoint-zero, all 5,400 feedback receipts for zero-assignment,
1,800 terminal feedback receipts for zero-terminal-origin, and all 21,600
selection/step/next-action/feedback/append receipts for marker ablation.
Evaluation records are committed separately and never substituted for an
operation receipt. Case 19's aggregate `dynamic_projection_root_sha256` is
`SHA256(b"v3-dynamic-aggregate\x00" || canonical_sequence)` where the four
items, in the control order above, are `uint32-length(control_id) || control_id
UTF-8 || raw_32_byte_per-control_dynamic_root`. It must bind exactly the four
nested receipt roots and no fifth or reordered item.

Component reordering sorts by the tuple
`(SHA256(b"v3-component-order\x00" || component_type_u8 || record_bytes),
record_bytes)`; byte-identical duplicates reject before sorting, so the second
term is a collision tie-break rather than an acceptance path. Ledger digests
use `b"v3-ledger\x00" || prior_digest || operation_receipt_bytes`.

Identity and component equations are exact. `episode_digest` is
`SHA256(b"v3-episode\x00" || EpisodeIdentity_record_bytes)`.
`regime_digest` is
`SHA256(b"v3-regime\x00" || RegimeSpec_record_bytes)`; every source-delivery
and evaluation-source receipt uses only this exact digest.
`consumer_digest` is `SHA256(b"v3-consumer\x00" || uint32_length ||
UTF8_consumer_name)` where the only operation mappings are
`selection -> select_action`, `step -> environment_step`,
`next-action -> select_next_action`, `feedback -> sarsa_update`, and
`append -> append_trace`. Action, step, update, and trajectory component
digests are the corresponding `v3-action`, `v3-step`, `v3-update`, or
`v3-trajectory` prefix plus complete record bytes. Observation, token, permit,
feedback-origin, and receipt digests use their self/preimage rules above.

Each episode owns one ledger. Its genesis prior value is
`SHA256(b"v3-ledger-genesis\x00" || episode_digest)`. A successful operation
receipt must name the current value as `prior_ledger_digest`; the next value is
`SHA256(b"v3-ledger\x00" || prior_ledger_digest ||
OperationReceipt_record_bytes)`. The first phase-0 selection permit is issued
against genesis rather than a predecessor operation receipt. Every later permit
is issued only after the exact preceding operation updates the ledger. The
terminal ledger is the value after the third append, exactly 12 operations per
episode in the order selection, then for each phase step, optional next-action,
feedback, append. `terminal_ledger_root_sha256` is the `v3-ledger-root` hash of
the 1,800 terminal ledger digests in frozen episode order. There is no global
mutable ledger, cross-episode chain, omitted genesis, or alternate terminal
scope.

The seeded-random comparator uses no observation or private identity field. In
the frozen all-episode order (train regimes, then validation, then test), let
`draw_index = 3 * global_episode_index + phase`. It hashes exactly
`b"online-sarsa-v3-random\x00" || little_endian_uint64(141421356) ||
little_endian_uint64(draw_index)`. Its action is `digest[0] & 0x01`; no other
digest byte, bit order, observation, split/regime field, PRNG state, or platform
encoding is allowed.

## Exact online SARSA chronology

The learner uses exact float64 `alpha = 1.0` and `gamma = 1.0`. At episode
start and then across each transition, the real consumers run in this order:

1. At phase 0 only, `select_action` consumes the authenticated observation, one
   fresh selection permit, and the exact behavior token. Forced mode returns
   the token action; greedy mode returns the smallest action attaining the
   maximum Q value. At phases 1 and 2, this slot uses the action already latched
   by step 3 of the preceding transition and performs no new selection.
2. `environment_step` consumes the same observation/action identities and one
   fresh step permit, then returns the authenticated successor, reward, and
   done value. It alone may materialize terminal reward.
3. If nonterminal, `select_next_action` consumes the successor, the next frozen
   token, and one fresh next-action permit before the predecessor update. That
   returned action is latched and must be the action used at the next phase.
4. `sarsa_update` consumes the authenticated step, latched next action or
   terminal sentinel, and one fresh feedback permit. Its target is
   `reward + Q(successor_key, next_action)` when nonterminal and exact `reward`
   when terminal. With alpha one, the addressed cell becomes that target.
5. `append_trace` consumes the exact step/update pair and one fresh append
   permit. Only after a successful append may the episode advance.

No update may occur before its action and environment step. No predecessor
update may occur before the actual next action is selected. No next action may
be selected twice, substituted after the update, or differ from the following
step's action. A terminal step must have no successor and no next action. An
episode exposes return only after all three authenticated appends.

The implementation must prove the chronology with an operation ledger whose
exact canonical train counts are: 1,800 initial selections, 3,600 nonterminal
next-action selections, 5,400 total action decisions, 5,400 environment steps,
5,400 updates, 5,400 appends, 1,800 terminal returns, and no other mutation.
Every ledger row binds the exact
episode, predecessor key and bytes, selected action, successor identity,
reward, done bit, next action, target, addressed Q-cell, old/new value, token
digest, permit identity, and prior ledger digest.

The algebraic training oracle is frozen before implementation. In each cue
block, the eight threefold forced code runs visit every code. A rewarding code
therefore writes terminal value on its first visit, propagates across the
second visit's final bootstrap boundary, and propagates to the root on the
third visit. The disjoint first action of code 3 versus code 4 prevents the
later cue block from erasing the earlier cue root. Each cue block's final
greedy episode must succeed. Per train regime the exact forced successes are
72, greedy successes are 24, total reward is 96, and regret is 504. Across all
train regimes they are 216, 72, 288, and 1,512 respectively.

The final canonical table must match an independent sequential oracle that
starts from its own blank 28-cell table and receives only the frozen family,
behavior tokens, and evaluator transition rule. It derives every forced action
from the token, derives every greedy and next action from its own current Q
table before the corresponding update, steps its own environment replay, and
applies the frozen SARSA equations without calling learner code or consuming a
learner action, target, or Q value. Its complete action stream must then equal
the learner's authenticated stream. All 28 final cells, every intermediate
addressed-cell value, every target, and every greedy tie decision must match bit
for bit.

## Capabilities and physical information boundary

The threat model is explicit. The learner is fixed, committed, first-party
study code, not an untrusted Python plugin, and this study does not claim a
security sandbox against malicious bytecode or native code. The boundary claim
is limited to the exercised dataflow of that approved implementation. Before
execution, a closed AST audit rejects imports, globals, nonlocal writes,
closures, callbacks, dynamic attribute access, reflection names (`gc`, `sys`,
`inspect`, frames, modules, builtins), `eval`/`exec`, file/network/process/native
names, and any identifier outside the exact learner functions, parameters, and
numeric/local temporaries. The learner call receives only the four copied
primitive adapter values frozen above: one float64 marker, one immutable
two-float Q row, and two exact token integers. It returns one exact integer.
The authenticated dataclasses remain in the sealed adapter and never enter
learner code. The sealed harness, not learner code, performs key construction,
environment steps, updates, receipts, and trace joins. This supports a physical
API/dataflow claim for the approved source only, not hostile-code containment.

Train template/family replay occurs before learner construction. Validation and
test identities are committed by the frozen plan but their factories and raw
objects do not exist during fitting. After the fitted Q table is frozen and the
learner object and train-only capabilities are destroyed, the harness creates
and independently replays the held-out family, then calls only the sealed
read-only evaluator on the table snapshot. The absent/exploding/spy campaigns
below enforce that ordering.

Five unforgeable, one-use permit classes exist only inside the fixture:
`SelectionPermit`, `StepPermit`, `NextActionPermit`, `FeedbackPermit`, and
`AppendPermit`. A central identity ledger creates exactly the permitted count,
records object identity rather than equality, names the sole consumer method,
and marks a permit spent atomically before the protected effect. Constructors,
copying, pickling, subclassing, attribute mutation, and equality-based aliases
are forbidden. No permit, evaluator object, mutable table, behavior schedule,
or terminal-origin handle appears in an observation, token, trace field,
callback, exception, closure, or learner-visible namespace. After a permit is
spent, the private ledger emits only a non-invertible `OperationReceipt` digest;
no raw permit, nonce, object reference, or equality token enters a trace.

The learning system's capabilities are partitioned and closed. The pure
selection function receives only:

- the copied float64 marker extracted after public-observation authentication;
- the immutable two-float Q row for the exact policy frame;
- copied exact `mode` and `forced_action` integers extracted from one
  authenticated behavior token at the selection boundary.

It never receives reward, successor, permit, origin, evaluator, or update
state. Separately, the sealed harness-owned update consumer receives the scalar
feedback origin, authenticated successor, and latched next action only after
their declared boundaries; it cannot call selection, inspect a future token, or
return any value to the selection function. The only shared mutable object is
the exact addressed cell in the closed 28-cell table.

Evaluator target code, twin outcomes, future tokens, future actions,
future rewards, regime tables, held-out factories, comparator actions,
intervention masks, and complete-family commitments remain outside that set.
Reflection over the adapter-facing objects and learner-facing primitive values
must show only the declared surfaces.

Exactly these 45 live attacks must be exercised against their real consumer,
each in a fresh episode/ledger context, and must reject before any table,
environment, trace, permit, token, or schedule mutation:

| attack | real consumer | exact exception category |
| --- | --- | --- |
| forged selection permit | `select_action` | `PermitIdentityError` |
| reused selection permit | `select_action` | `PermitSpentError` |
| reentrant initial selection | `select_action` | `ReentrancyError` |
| future behavior token | `select_next_action` | `ChronologyError` |
| cross-episode behavior token | `select_action` | `IdentityMismatchError` |
| forced action outside `{0,1}` | `select_next_action` | `TokenContractError` |
| greedy token carrying a forced action | `select_action` | `TokenContractError` |
| forged step permit | `environment_step` | `PermitIdentityError` |
| reused step permit | `environment_step` | `PermitSpentError` |
| action before selection | `environment_step` | `ChronologyError` |
| substituted selected or latched action | `environment_step` | `IdentityMismatchError` |
| cross-episode action | `environment_step` | `IdentityMismatchError` |
| reentrant environment step | `environment_step` | `ReentrancyError` |
| forged next-action permit | `select_next_action` | `PermitIdentityError` |
| reused next-action permit | `select_next_action` | `PermitSpentError` |
| next action before successor | `select_next_action` | `ChronologyError` |
| reentrant next-action selection | `select_next_action` | `ReentrancyError` |
| forged feedback permit | `sarsa_update` | `PermitIdentityError` |
| reused feedback permit | `sarsa_update` | `PermitSpentError` |
| update before step | `sarsa_update` | `ChronologyError` |
| update before nonterminal next-action latch | `sarsa_update` | `ChronologyError` |
| substituted latched next action | `sarsa_update` | `IdentityMismatchError` |
| terminal update with successor or next action | `sarsa_update` | `TerminalContractError` |
| reentrant update | `sarsa_update` | `ReentrancyError` |
| forged append permit | `append_trace` | `PermitIdentityError` |
| reused append permit | `append_trace` | `PermitSpentError` |
| append before update | `append_trace` | `ChronologyError` |
| reentrant append | `append_trace` | `ReentrancyError` |
| fresh-valid duplicate initial selection | `select_action` | `ChronologyError` |
| fresh-valid duplicate step | `environment_step` | `ChronologyError` |
| fresh-valid duplicate next-action selection | `select_next_action` | `ChronologyError` |
| fresh-valid duplicate update | `sarsa_update` | `ChronologyError` |
| fresh-valid duplicate append | `append_trace` | `ChronologyError` |
| cross-episode selection permit | `select_action` | `IdentityMismatchError` |
| cross-episode step permit | `environment_step` | `IdentityMismatchError` |
| cross-episode next-action permit | `select_next_action` | `IdentityMismatchError` |
| cross-episode feedback permit | `sarsa_update` | `IdentityMismatchError` |
| cross-episode append permit | `append_trace` | `IdentityMismatchError` |
| forged public observation view | `select_action` | `ViewAuthenticationError` |
| mutated public observation alias | `select_action` | `ImmutableViewError` |
| forged Q snapshot | `select_action` | `ViewAuthenticationError` |
| mutated Q snapshot alias | `select_action` | `ImmutableViewError` |
| update event released before step-complete barrier | `sarsa_update` | `ChronologyError` |
| return exposed after only two authenticated appends | `expose_return` | `ReturnBoundaryError` |
| undeclared evaluator capability bundle containing token, permit, mutable table, and callback | `evaluate_frozen` | `EvaluationCapabilityError` |

An independent attack oracle snapshots all mutable state and one-use ledgers,
invokes the real consumer, checks the exact exception category, and compares
the complete before/after snapshots. A literal pass field or asserted rejected
count is not evidence. The result count is derived only from all 45 witnessed
consumer rejections.
For the forged or mutated public/Q-view rows, `select_action` denotes the sealed
real consumer wrapper including its mandatory first authentication step. A
rejection in that first step is consumer evidence; rejection by a detached
constructor, test helper, or pre-call assertion is not.

Reentrancy and physical timing are exercised rather than asserted. Every real
consumer contains one private `BoundaryAttackHook` barrier immediately after it
marks the operation in progress and before its first effect. In canonical and
control execution the sealed null hook is installed and its invocation ledger
must remain empty. In a named attack context only, the independent oracle
installs a single-use hook that synchronously calls the same real consumer with
the otherwise-valid pending capability while the original call remains in
progress; the nested call must raise `ReentrancyError`, after which the outer
attack context aborts and the full state/effect snapshot remains equal. The
event-order attack uses two bounded `threading.Event` barriers: the update call
is released while the authenticated step-complete event is still unset and
must raise `ChronologyError` before any Q or ledger effect. The early-return
attack calls the real `expose_return` consumer after exactly two authenticated
append receipts and before the third append; it must raise
`ReturnBoundaryError`. The test harness times out and fails closed if either
barrier or hook is not traversed. No attack hook, event handle, callback, or
attack context is learner-visible or present in a result-bearing canonical
path.

One separate atomic one-use race is exercised at a valid feedback boundary.
Two threads receive the same otherwise-valid `FeedbackPermit`, synchronize on a
barrier, and call the real `sarsa_update` consumer. Exactly one atomically marks
the identity spent and applies the one canonical cell update; the other raises
exact `PermitSpentError`. The final table and ledger must equal an independent
single sequential update bit for bit, with one winner, one rejection, and one
effect. This race is not counted among the 45 all-rejected attacks because its
single authorized winner is required.

A separate structural campaign applies six attacks to each of the five permit
classes: direct construction, `copy.copy`, pickle round-trip, subclass
construction, post-construction attribute mutation, and an equality-forged
alias. These are exactly 30 structural attacks. Each must either reject at
construction or fail identity-ledger authentication at the real consumer with
no effect. Direct/subclass/attribute attacks require
`PermitConstructionError`; copy/pickle attacks require
`PermitSerializationError`; equality-forged aliases require
`PermitIdentityError`. Case 14 reports both exact structural counts in addition
to the 45 live chronology attacks.

## Trace authentication and malformed records

The real training path emits five separately ordered component lists: public
observation envelopes, action selections, environment steps, updates, and
episode-final trajectories. There are exactly 5,400 items in each of the first
four lists and 1,800 trajectory items in the fifth, for 23,400 authenticated
components. Append authorization remains in the operation ledger rather than
being counted as a sixth component list. Each list is independently reordered
by a frozen SHA-256 ordering key before joining.
`component_root_sha256` is SHA-256 of `v3-component-root` plus five
length-delimited lists in exact type order observation, action, step, update,
trajectory; within each list it uses original emitter order and items
`component_type_u8 || raw_32_byte_component_digest`. For hostile reordering,
each list is separately sorted by the component-order tuple frozen in the ABI,
then the joiner authenticates identities without using positions. It emits one
chronological sequence in frozen episode order: for phases 0, 1, 2,
observation, action, step, update, followed after phase 2 by the trajectory.
Thus there are 13 typed digests per episode and 23,400 total.
`reordered_root_sha256` is SHA-256 of `v3-reordered-root` plus that exact
length-delimited chronological sequence. No global pre-join sort, grouped-list
preimage, append receipt, or alternate type order is accepted for the reordered
root.
The observation component list contains exactly one predecessor envelope for
each of the 5,400 steps. A nonterminal `StepRecord.successor` must be byte-for-
byte and digest-identical to the next phase's already committed predecessor
envelope in the same episode; it is a nested authenticated reference and is not
counted as a second observation component. Terminal successor is exactly
absent. Any copied-but-different, duplicate, missing, or cross-episode successor
rejects.
The joiner reconstructs chronology solely from strict identities and digests;
it must not rely on shared list position or object identity.

Every joined transition authenticates episode identity, predecessor public
bytes and key, phase, action identity/value, successor identity/bytes/key,
reward type/value, exact built-in done, origin, token digest, operation-receipt
digests authenticated against the private identity ledger, latched next action,
update target, old/new cell values, and the prior ledger digest. It rejects
missing, extra, duplicate, cross-episode, cross-phase,
out-of-order, reused, future, or mismatched components before constructing a
trajectory.

The malformed-record campaign is the closed matrix of exactly 45 selected
mutations listed below. It spans these ten classes and the named real consumers
but makes no claim that every class is crossed with every record field or
component type; no unlisted mutation is counted as covered:

1. missing required field;
2. unknown extra field;
3. wrong exact scalar type, including Boolean-as-integer;
4. NaN, infinity, or negative zero float;
5. invalid enum/range/layout;
6. duplicate identity or replayed digest;
7. cross-episode identity substitution;
8. predecessor/successor/action digest substitution;
9. reward, done, origin, or next-action substitution;
10. stale/future phase or chronology substitution.

The exact exception mapping is class 1 `MissingFieldError`, class 2
`UnknownFieldError`, class 3 `ExactTypeError`, class 4 `FiniteFloatError`,
class 5 `LayoutError`, class 6 `DuplicateIdentityError`, class 7
`CrossEpisodeError`, class 8 `DigestMismatchError`, class 9
`ValueSubstitutionError`, and class 10 `ChronologyError`.

The frozen applicability matrix is exactly these 45 component/class/consumer
rows; implementation may not substitute a different mutation:

| # | component and mutation | class | real consumer |
| ---: | --- | ---: | --- |
| 1 | `RegimeSpec` missing `name` | 1 | regime decoder |
| 2 | `RegimeSpec.seed` exact Boolean | 3 | regime decoder |
| 3 | `RegimeSpec.split = "holdout"` | 5 | source router |
| 4 | `EpisodeIdentity` unknown extra field | 2 | identity decoder |
| 5 | `EpisodeIdentity.cue_bin` exact Boolean | 3 | identity decoder |
| 6 | duplicate `EpisodeIdentity` digest | 6 | identity registry |
| 7 | cross-regime episode substitution | 7 | identity registry |
| 8 | `EvaluatorEnvelope` missing target | 1 | evaluator gateway |
| 9 | `EvaluatorEnvelope.target_code = 2` | 5 | evaluator gateway |
| 10 | evaluator origin wrong byte length | 5 | evaluator gateway |
| 11 | `PublicObservation.phase` exact Boolean | 3 | observation decoder |
| 12 | marker exact NaN | 4 | observation decoder |
| 13 | nuisance exact negative zero | 4 | observation decoder |
| 14 | prefix code illegal for width | 5 | policy-key builder |
| 15 | `ObservationEnvelope` missing public view | 1 | observation joiner |
| 16 | cross-episode public view | 7 | observation joiner |
| 17 | substituted public digest | 8 | observation joiner |
| 18 | stale envelope phase | 10 | observation joiner |
| 19 | `PublicBehaviorToken.mode` exact Boolean | 3 | token decoder |
| 20 | forced action outside `{0,1}` | 5 | token decoder |
| 21 | future public token phase | 10 | token decoder |
| 22 | `BehaviorTokenEnvelope` unknown extra field | 2 | token joiner |
| 23 | duplicate token digest | 6 | token registry |
| 24 | cross-episode token envelope | 7 | token joiner |
| 25 | `ActionSelection` missing value | 1 | environment step |
| 26 | action value exact Boolean | 3 | environment step |
| 27 | substituted observation digest | 8 | environment step |
| 28 | nonterminal `StepRecord` missing successor | 1 | step joiner |
| 29 | step reward exact infinity | 4 | step joiner |
| 30 | step done exact integer one | 3 | step joiner |
| 31 | cross-episode step action | 7 | step joiner |
| 32 | nonterminal `UpdateRecord` missing next action | 1 | update consumer |
| 33 | update target exact NaN | 4 | update consumer |
| 34 | substituted update-address action | 9 | update consumer |
| 35 | future successor phase | 10 | update consumer |
| 36 | duplicate trajectory step digest | 6 | trajectory joiner |
| 37 | substituted terminal origin | 9 | trajectory joiner |
| 38 | public vector byte length 39 | 5 | observation decoder |
| 39 | public token byte length 2 | 5 | token decoder |
| 40 | observation digest byte length 31 | 5 | observation joiner |
| 41 | operation receipt digest byte length 31 | 5 | receipt joiner |
| 42 | substituted prior-ledger digest in receipt | 8 | receipt joiner |
| 43 | substituted update old value | 9 | update consumer |
| 44 | substituted update new value | 9 | update consumer |
| 45 | missing append receipt digest | 1 | trajectory joiner |

Each row traverses its named real consumer, rejects, and leaves a complete
before/after state snapshot equal. The implementation record must map each row
to its exact test before execution. The sanitized result contains the derived
integer count 45, exact expected exception from the class map, and exact
all-rejected Boolean.

## Train-only fitting and held-out isolation

The canonical fitter accepts only an iterator of authenticated train episode
factories plus the frozen train behavior-token factory. It has no validation or
test argument, global, import, closure, callback, path, or late-bound lookup.
Seven independently fitted paths are required: canonical, disjoint-zero
bootstrap, zero behavior assignment, terminal-origin zero, marker ablation,
myopic comparator, and no-bootstrap comparator.

Each construction path is run under each of three held-out source modes:

- `absent`: validation/test constructors do not exist;
- `exploding`: any validation/test construction, iteration, length, indexing,
  serialization, or attribute access raises;
- `spy`: operation-counting wrappers remain uncalled during fitting.

This yields 21 train-only source checks. The myopic construction path consumes
only authenticated public train observations to validate the total marker-sign
rule over both cue signs, then emits a stateless policy. It receives no action,
token, Q value, reward, target, successor, outcome, or canonical action stream;
the validation oracle derives the expected action directly from each public
marker without exposing that action to the constructor. For each of the
seven paths, one aggregate validation/test source factory and one
evaluation-only policy snapshot have independent spy ledgers, for 14 exact-zero
ledgers during construction. For each of the seven independently fitted paths,
source order A delivers the three train-regime factories in the frozen table
order and order B delivers those same three authenticated factories in exact
reverse. Before any selection, token consumption, or update, the fitter
authenticates exactly that complete three-factory train source set and
canonical-sorts it by the frozen regime table; thereafter both runs execute the
same regime-major chronology. The test proves delivery-order invariance while
preserving the frozen online order.

The normalized A/B commitment contains exactly seven path-framed records, not
one unlabelled aggregate. Each path frame starts with its length-delimited exact
`fit_path_id`, then seven separately count-and-length-framed sections in this
order and exact item types:

1. complete `EpisodeIdentity` records, episode order regime, round, cue block,
   block position;
2. complete `ObservationEnvelope` records, the same episode order then phase
   0, 1, 2;
3. complete `BehaviorTokenEnvelope` records in that same episode/phase order;
4. complete `ActionSelection` records in that same episode/phase order;
5. items in that same episode/phase order, each exactly
   `uint32_len(StepRecord_bytes) || StepRecord_bytes ||
   uint32_len(UpdateRecord_bytes) || UpdateRecord_bytes`;
6. complete append-class `OperationReceipt` records in that same
   episode/phase order; and
7. raw 32-byte terminal ledger digests in episode order.

The seven sections are followed by one raw 32-byte
`terminal_train_ledger_sha256` and one table section encoded exactly as uint32
cell count followed by the sorted key/action/value items from the table-root
rule. The six stateful paths have 1,800 identities, 5,400 observation
envelopes, 5,400 token envelopes, 5,400 actions, 5,400 step/update items, 5,400
append receipts, 1,800 terminal ledger digests, and 28 table cells. The
stateless `myopic` frame has 1,800 identities and exactly the 1,800 phase-0
`ObservationEnvelope` records in the same episode order. Its token, action,
step/update, append, and terminal-ledger sections are exact count-zero
sequences, and its table section is exact count zero. Empty sections carry
their explicit zero count and are never omitted. A public view, token bytes,
record digest, or append digest cannot substitute for the named complete ABI
record.

For a stateful path, `terminal_train_ledger_sha256` is
`SHA256(b"v3-ledger-root\x00" ||
canonical_sequence_of_1800_terminal_ledger_digests)`. For `myopic` it is the
stateless commitment
`SHA256(b"v3-stateless-ledger\x00" || length_delimited_fit_path_id ||
canonical_sequence_of_1800_phase0_ObservationEnvelope_records ||
length_delimited_UTF8("negative=0,positive=1"))`. Thus the stateless path has an
explicit receipt value without inventing tokens, actions, updates, or a train
ledger. The outer validator reconstructs every section and digest from the
declared path semantics.

`normalized_order_a_sha256` and `normalized_order_b_sha256` are each
`SHA256(b"v3-source-normalized\x00" || uint32_le(7) ||
seven_length_delimited_path_frames)` in the exact path order below. They and
each stateful table, plus the stateless empty-table encoding, must be
byte-identical between A and B. Raw pre-sort delivery logs must differ and
match their respective frozen A/B commitments.
The seven `fit_path_id` strings, in hash order, are exact `canonical`,
`disjoint-zero-bootstrap`, `zero-assignment`, `zero-terminal-origin`,
`marker-ablation`, `myopic`, and `no-bootstrap`. For each path, raw order A has
three `SourceDeliveryEvent` records at positions 0, 1, 2 with the train regime
digests in table order; raw order B has the same records with those regime
digests reversed. `raw_delivery_a_sha256` and `raw_delivery_b_sha256` are
SHA-256 of `v3-source-delivery` plus the path-major, delivery-position-major
canonical sequence of those 21 complete records. Their expected values are
independently derived from the frozen `RegimeSpec` records before a source is
opened; they must be unequal. The two explicitly path-framed normalized roots
defined above must be equal.

The A/B campaign is fourteen additional executions, exact order A then B for
each of the seven paths, with held-out sources in `absent` mode. It is distinct
from the 21 path/source-mode executions below. Each path emits one typed
`SourceOrderReceipt`. Its per-path raw A and B roots are
`SHA256(b"v3-source-delivery-path\x00" ||
canonical_sequence_of_that_path's_three_SourceDeliveryEvent_records)`; its
per-path normalized A and B roots are
`SHA256(b"v3-source-normalized-path\x00" ||
length_delimited_complete_path_frame)`. Its terminal A/B fields are the exact
stateful terminal-ledger aggregate or stateless myopic digest defined above.
The seven receipts are in exact path order and their aggregate root is
`source_order_root_sha256 = SHA256(b"v3-source-order\x00" ||
canonical_sequence_of_complete_SourceOrderReceipt_records)`. The global raw
roots are independently reconstructed from the same 21 delivery events; the
global normalized roots are independently reconstructed from the same seven
path frames. Every receipt must complete, raw A/B must differ, normalized A/B
and terminal A/B must match, and every per-path root must join the corresponding
global preimage. The strict validator reconstructs all delivery, normalized,
terminal, and receipt preimages; a worker-supplied hash or Boolean is not
evidence.

Each of the 21 path/mode executions traverses the real source router and emits
one typed `SourceBoundaryReceipt` only after its complete stateful train ledger
or the exact stateless myopic commitment is sealed. Receipt order is path order
above, then source modes exact `absent`,
`exploding`, `spy`. Every receipt has `train_episodes=1800`, exact zero
validation-factory, test-factory, and evaluation-policy-snapshot accesses, the
outer-oracle-derived terminal train ledger digest for that path, and true
completion. The 21 receipts are hashed as
`source_boundary_root_sha256 = SHA256(b"v3-source-boundary\x00" ||
canonical_sequence_of_complete_SourceBoundaryReceipt_records)`. The source router owns the operation
counters; a fitter cannot write a receipt or reset a counter. The pinned outer
validator reconstructs every receipt identity and terminal ledger, rejects a
missing/duplicate/reordered path or mode, and verifies the exact root. Thus the
21 completed paths and 14 zero held-out-source/policy ledgers are not accepted
as producer counts or detached assertions.

An inverse test that injects one held-out factory into the train iterator must
reject before selection or update. After fitting, held-out evaluation must use
fresh generators and a read-only snapshot; table and train-ledger hashes must
remain unchanged, as proven by the evaluation boundary and source roots above.

## Metrics and positive gate

Episode return is the exact terminal reward in `{0.0, 1.0}`. Mean return is
the exact integer success count divided by episode count; regret is episode
count minus success count. No smoothing, confidence interval, rounding,
weighting, early stopping, best-checkpoint selection, or omitted episode is
allowed.

The canonical positive gate is the conjunction of all of these clauses:

1. regime templates and all 39,536 expanded legal rows replay exactly;
2. all 8,472 realized rows are legal, unique, and authenticated;
3. evaluator twins change exactly the frozen 28 template terminal rewards and
   5,648 expanded terminal outcomes, and no public field;
4. the 5,400 behavior tokens and 2,700 cross-cue pairs replay exactly;
5. train schedule and ledger counts equal the frozen counts;
6. every update obeys the exact canonical reward, feedback-origin, address,
   SARSA target, and chronology;
7. the independent action/reward replay produces an identical 28-cell table;
8. train forced successes equal 216;
9. train greedy successes equal 72;
10. train total reward equals 288 and regret equals 1,512;
11. each held-out regime has exact mean return `1.0` and regret zero;
12. aggregate validation and test mean return are each exact `1.0`;
13. no held-out update occurs, every held-out action stream replays exactly,
    every evaluation boundary snapshot pair is equal, and every held-out source
    receipt/root matches the frozen source sequence;
14. canonical held-out return beats each constant, myopic, and no-bootstrap
    comparator by at least `0.90`;
15. canonical held-out return beats the seeded-random comparator by at least
    `0.55` on validation and independently on test;
16. all 21 train-only source receipts and their root match, and all 14
    source/policy spy ledgers remain exact zero;
17. source order A and reverse order B produce byte-identical normalized train
    projections and fitted tables, while their raw delivery logs match their
    distinct frozen A/B commitments and all seven source-order receipts/root;
18. all 45 live boundary attacks, all 30 structural permit attacks, the exact
    one-winner/one-rejection permit race, and all 45
    applicable malformed mutations
    reject through their named real consumers without mutation;
19. every permit count, identity, one-use, timing, and reentrancy invariant
    passes;
20. component reordering reconstructs the exact 23,400-component trace;
21. every negative control satisfies its full frozen failure gate;
22. static/dynamic difference whitelists, scalar schema, sanitizer,
    fresh-process reproduction, and file/process/network isolation all pass.

Failure of any conjunct fails the study. A control cannot pass by crashing,
omitting a row, changing evaluation episodes, skipping a comparator, producing
a malformed projection, or disabling its exercised path.

## Frozen comparators

All comparators use the identical 2,824 episode identities, public-observation
generation rule and phase-0 sources, environment transition rule, evaluation
order, and metric. After phase 0, each comparator's public prefix is generated
from that comparator's own prior actions, so realized observation streams may
and generally do differ. Every action-conditioned observation stream is
committed inside its complete per-comparator trace and independently replayed
through the real environment. Comparators receive no target, terminal outcome,
canonical action, learner table, or intervention state.

- `constant-0` selects action 0 at all phases and completes code 0. Its exact
  return is zero on every regime.
- `constant-1` selects action 1 at all phases and completes code 7. Its exact
  return is zero on every regime.
- `myopic-marker` selects action 0 for a negative marker and action 1 for a
  positive marker at every phase, completing code 0 or 7. Its exact return is
  zero on every regime.
- `no-bootstrap` is fitted online with the canonical behavior stream and
  alpha one, but every nonterminal target is the immediate reward only. Its
  root cells remain tied at zero, the frozen tie rule completes code 0, and its
  exact greedy held-out return is zero.
- `seeded-random` uses seed `141421356` and the exact canonical-ABI hash and
  first-byte low-bit extraction frozen above. It makes exactly 2,824 episode
  calls and 8,472 scalar action draws. Its independently replayed validation
  return and test return must each be at most `0.35`.

Comparator construction consumes authenticated behavior feedback only when a
construction pass is declared. The no-bootstrap fitter and the myopic
rule-validation pass receive the same train-only source isolation campaign as
the canonical fitter. Constants and random are stateless; the emitted myopic
policy is also stateless after its authenticated rule-validation pass. No
comparator may derive or copy canonical actions.

Comparator commitment order is exact comparator IDs `constant-0`,
`constant-1`, `myopic-marker`, `no-bootstrap`, `seeded-random`, then all 2,824
episodes in frozen regime/episode order. For each complete `ComparatorTrace`,
the input item is the ABI primitive encoding of comparator ID, complete episode
record, and three public observation byte strings; the action item is comparator
ID, episode record, and three int64 actions; the output item is the complete
`ComparatorTrace` record. The three canonical sequences are hashed with
`v3-comparator-input`, `v3-comparator-action`, and `v3-comparator-output`.
The pinned outer validator independently constructs all five action streams
from the frozen comparator definitions, replays all 14,120 traces through its
own environment oracle, and compares all three roots and aggregates. It does
not accept `all_streams_replayed` or aggregate returns without that replay.

## Negative controls

Every control starts from a fresh blank table, uses the same training episode
identities, behavior-token schedule, environment steps, train-only boundary,
evaluation episode identities, greedy tie rule, metric, comparators, trace
authentication, process boundary, and positive-gate evaluator. Only the named
intervention is permitted to differ. Each control must complete normally and
then fail the full canonical positive gate for the intended causal reason.

For every control, `identical_evaluator_contract` means exact equality of the
ordered held-out episode identities, evaluator target/origin inputs, episode
counts, greedy tie rule, read-only/no-update boundary, metric, comparator
definitions, and threshold evaluator before policy action. It requires the
exact canonical `evaluator_config_sha256` in every contract record and receipt,
then byte-identical evaluator-contract roots. It does not mean the
policy-dependent observation stream, fitted policy snapshot, selected actions,
rewards, returns, or output digest equals canonical values; those are causal
outputs committed separately. Marker ablation preserves the same evaluator
contract but has the independently constructed exact marker-zero observation
stream transformation.

### Disjoint-zero bootstrap target

For each nonterminal update only, the bootstrap lookup replaces the successor
policy state key with a typed disjoint sentinel state key in a separate
immutable zero-only table. For a nonterminal public successor, the exact
four-byte key is `(0x80 + successor_phase, 0xFF, successor_width,
successor_code)`. The separately latched `next_action` remains the state-action
index, and both sentinel action values are exact `0.0`. The first two bytes make
the sentinel namespace disjoint from every canonical policy state. Keys contain
no cue, target, reward, origin, outcome, or selected action and can never be
selected or addressed by an update.
At the intervention point, the current selection, environment action,
successor public bytes, environment reward, terminal target, and update address
remain equal to the paired canonical row. Later greedy selections, actions,
successors, and evaluation outputs may differ only when a prior transformed Q
value reaches them through the closed DAG; the evaluator contract stays equal. The transformation is
outcome-blind and independently replayed for all 16,944 nonterminal expanded
legal rows, realized and unrealized. Terminal rows retain the exact terminal
sentinel in the static projection. The intervention
prevents any terminal value from crossing a bootstrap boundary. Each held-out
regime, validation aggregate, and test aggregate must have exact mean return
`0.0`; its exact false positive-gate indices are frozen below.
The forced environment stream still contains exactly 216 successes, but no
terminal value reaches a root; all 72 greedy train episodes therefore fail,
giving train reward 216 and regret 1,584.

### Zero behavior-assignment update

The behavior policy and environment execute the control policy's authenticated
selected action unchanged at the step boundary, but
the update-address action is replaced with exact integer 0 at every phase.
The intervention occurs only after step authentication and is recorded as a
separate update-address field; environment action and trace action must equal
that same selected action. They may differ from the canonical-reference branch
later only through a prior DAG-reachable Q difference. The mapping is
outcome-blind and applied to every realized train row.
Greedy held-out mean return must be exact zero on every regime, validation, and
test; its exact false positive-gate indices are frozen below.
The forced environment stream still contains 216 successes. Because action-1
cells remain zero while every behavior update is redirected to action 0, each
cue block's terminal assignment yields a wrong root greedy choice; all 72 train
greedy episodes fail, so reward is 216 and regret is 1,584.

### Zero terminal-reward origin

The environment computes and authenticates its canonical terminal outcome, but
the feedback gateway supplies exact `0.0` as update reward on every terminal
row and constructs the exact typed `FeedbackOrigin(kind="null-control")`
record frozen above. Its derived `origin_digest` is placed in
`UpdateRecord.feedback_origin_digest` and in the static and dynamic origin
fields; the canonical record instead binds the environment-origin digest and
environment scalar. On the directly intervened terminal row, nonterminal
rewards, actions, successors, done bits, update addresses, and public bytes
remain equal to the paired canonical row. Later greedy actions and outputs may
differ only through the closed Q-value DAG.
The intervention is applied to all expanded terminal rows before trajectory
selection. Greedy held-out mean return must be exact zero everywhere and the
control's exact false positive-gate indices are frozen below.
Forced environment success remains 216, but every terminal learning scalar is
zero, leaving all root cells tied at zero; all 72 greedy train episodes fail,
so reward is 216 and regret is 1,584.

### Complete marker ablation

Every public predecessor and successor observation in the 39,536-row expanded
family replaces `signed_marker` with exact positive zero before learner access.
The policy-frame cue byte collapses to 0 for both evaluator cues. Nuisance,
phase, prefix, evaluator truth, environment transition rule, reward rule, done
rule, schedule, and episode identities remain equal in the paired static
family. Realized actions, successor trajectories, and evaluation output may
differ only through the marker/key roots and closed DAG. The ablated
family and realized subset are independently replayed, and no unablated marker
or cue byte may reach selection, update, trace join, or evaluation.

Both a freshly fitted ablated table and the frozen canonical table evaluated
through ablated observations must have mean return at most `0.50` on every
held-out regime and on validation/test aggregates. Both must fail every
frozen held-out-performance and comparator-margin clause listed below.
Within training, each cue block rewrites the shared cue-0 cells before its final
greedy episode, so the analytic 216 forced successes, 72 greedy successes,
reward 288, and regret 1,512 remain. After the final cue-1 block the freshly
ablated table chooses code 4 for both cues, while the frozen canonical table on
ablated cue-0 keys chooses code 3 for both cues; each succeeds on exactly one
balanced cue and therefore has exact held-out mean 0.50.

The ablation byte mask is exact: every 40-byte predecessor and nonterminal
successor public vector must equal canonical at bytes 0 through 7 and 16 through
39, while bytes 8 through 15 change from the canonical `-1.0` or `+1.0`
binary64 encoding to eight zero bytes. For cue 1 only, byte 1 of the four-byte
policy key changes from `0x01` to `0x00`; the other three bytes remain equal.
Cue-0 keys remain byte-identical. Terminal sentinels remain `FF FF FF FF`.

## Static and dynamic intervention whitelists

For each of the 39,536 expanded legal rows, canonical and control construction
emits this exact 16-field static projection:

1. `row_key`;
2. `episode_identity`;
3. `regime_identity`;
4. `evaluator_target_code`;
5. `predecessor_public_bytes`;
6. `predecessor_policy_key`;
7. `action`;
8. `successor_public_bytes_or_terminal`;
9. `successor_policy_key_or_terminal`;
10. `environment_reward`;
11. `update_reward`;
12. `done`;
13. `feedback_origin_digest`;
14. `bootstrap_lookup_key_or_terminal`;
15. `update_address_action`;
16. `control_tag`.

The complete canonical/control comparison covers 158,144 paired rows. Allowed
static differences are exactly:

| control | allowed differing fields |
| --- | --- |
| disjoint-zero bootstrap | 14 and 16 |
| zero behavior assignment | 15 and 16 |
| zero terminal origin | 11, 13, and 16 |
| complete marker ablation | 5, 6, 8, 9, 14, and 16 |

Every listed field must differ exactly on its predicate rows and remain equal
elsewhere; every unlisted field must be byte-identical. The exact field-level
predicates and counts are:

| control / field | exact difference predicate | required count |
| --- | --- | ---: |
| disjoint zero / bootstrap lookup | nonterminal row | 16,944 |
| disjoint zero / control tag | every row | 39,536 |
| zero assignment / update address | canonical row action is 1 | 19,768 |
| zero assignment / control tag | every row | 39,536 |
| zero terminal / update reward | successful terminal row | 2,824 |
| zero terminal / origin | terminal row | 22,592 |
| zero terminal / control tag | every row | 39,536 |
| marker ablation / predecessor public bytes | every row | 39,536 |
| marker ablation / predecessor key | episode cue is 1 | 19,768 |
| marker ablation / successor public bytes | nonterminal row | 16,944 |
| marker ablation / successor key | nonterminal row and episode cue is 1 | 8,472 |
| marker ablation / bootstrap lookup | nonterminal row and episode cue is 1 | 8,472 |
| marker ablation / control tag | every row | 39,536 |

Applicability counts come from the committed family, cue balance, and action
rows, not from producer declarations. Comparator definition/input projections
and evaluator-contract input projections have separate all-field equality
checks across controls. Policy-observation, policy-snapshot, comparator-output,
and evaluation-output projections are not required equal; their differences
must instead lie in the declared intervention's complete authenticated dynamic
causal closure.

The static projection is defined even for unrealized legal rows without
executing a learner. Canonical `update_reward` equals the row's environment
reward, canonical `feedback_origin_digest` is the typed deterministic evaluator-feedback
origin, canonical `bootstrap_lookup_key_or_terminal` is the public successor
policy key or exact terminal sentinel, and canonical `update_address_action` is
the row action. Controls apply their closed transformations to those values.
No Q value, behavior token, selected next action, or observed trajectory is
invented for an unrealized row.

Each of the 5,400 realized transitions emits this complete 32-field dynamic
projection in exact order:

1. `episode_identity`;
2. `phase`;
3. `predecessor_observation_digest`;
4. `predecessor_public_bytes`;
5. `predecessor_policy_key`;
6. `behavior_token_digest`;
7. `selected_action`;
8. `selection_receipt_digest`;
9. `environment_action`;
10. `successor_observation_digest_or_terminal`;
11. `successor_public_bytes_or_terminal`;
12. `successor_policy_key_or_terminal`;
13. `environment_reward`;
14. `environment_origin_digest`;
15. `done`;
16. `next_behavior_token_digest_or_terminal`;
17. `latched_next_action_or_terminal`;
18. `next_selection_receipt_digest_or_terminal`;
19. `bootstrap_lookup_key_or_terminal`;
20. `update_address_action`;
21. `update_reward`;
22. `feedback_origin_digest`;
23. `target`;
24. `old_value`;
25. `new_value`;
26. `step_receipt_digest`;
27. `feedback_receipt_digest`;
28. `append_receipt_digest`;
29. `prior_ledger_digest`;
30. `terminal_ledger_digest_or_pending`;
31. `trajectory_return_or_pending`;
32. `control_tag`.

The comparison is complete-record equality outside one closed causal DAG. Its
only intervention roots are: disjoint-zero field 19; assignment field 20;
terminal-origin fields 21 and 22; marker-ablation fields 4, 5, 11, 12, and 19;
and field 32 for the matching control tag. No other root difference is legal.
The frozen directed edges are: public bytes to observation digest and policy
key; observation digest, policy key, token digest, and current addressed Q row
to selected action and selection receipt; selection to environment action;
environment action plus predecessor envelope to successor envelope, reward,
origin, and done; successor public/key, next token, and current Q row to the
latched next action and its receipt; bootstrap key, latched next action, reward,
feedback origin, and prior Q cell to target/new value; update address to the
mutated cell; step/update fields to their receipts; receipts plus prior ledger
to later ledger digests; and action/reward/done records to trajectory return and
evaluation output. A later selected action may differ only if a preceding
new-value difference reaches its exact policy row. The control tag has no edge
to a learner, environment, target, reward, action, or value; it can affect only
the tagged projection/root metadata.

The independent oracle replays every edge using authenticated records, derives
the complete allowed transitive closure for every canonical/control pair, and
requires every byte outside that closure to be equal. It also requires every
different field to be reachable from exactly one declared root. Its ordered
32-field projections are committed by `dynamic_projection_root_sha256`; the
controller recomputes that root and rejects a count, tag, receipt, or claimed
zero difference without the underlying projections. This prevents a control
from silently changing behavior, reward, receipt, ledger, or evaluation to
manufacture failure.

## Closed sanitized result contract

The worker emits one UTF-8 JSON object decoded with an object-pairs-preserving
parser. Duplicate keys, duplicate case IDs, noncanonical order, unknown keys,
wrong exact types, NaN/infinity, negative zero, integers outside signed 64-bit,
non-lowercase hashes, and extra stdout or any stderr byte fail closed. The
entire stdout limit is 524,288 bytes.

After relation validation, JSON bytes are produced once with the pinned Python
runtime's standard encoder using `ensure_ascii=True`, `allow_nan=False`,
`separators=(",", ":")`, and `sort_keys=False`, followed by one LF byte.
Objects are constructed as ordered pairs in the exact schemas below. No pretty
printing, alternate float formatter, carriage return, log prefix, or second
JSON value is permitted. Fresh-child byte comparison uses this encoding.

The exact top-level ordered fields and types are:

| field | type | gate |
| --- | --- | --- |
| `schema_version` | S | exact `online-sarsa-v3-result-v1` |
| `study_id` | S | exact `online-sarsa-latched-choice-v3` |
| `revision` | R | exact controller-approved revision |
| `status` | S | exact `passed` only when all cases pass, otherwise `failed` |
| `action` | S | exact success or failure action frozen below |
| `claim_boundary` | S | exact `synthetic-online-sarsa-harness-only` |
| `cases` | A | exactly the 21 closed case objects below, in order |

Type tags are explicit and never inferred from a field name or suffix:

- B: exact JSON Boolean;
- I: exact JSON integer, with Boolean rejected;
- F: exact finite JSON number decoded to float64, with Boolean, NaN, infinity,
  and negative zero rejected;
- S: exact JSON string;
- H: exact lowercase 64-character hexadecimal string.
- R: exact lowercase 40-character Git revision hexadecimal string.
- A: exact JSON array whose length, element schemas, and order are fixed by the
  table below.

Nested rejection evidence uses one closed `RejectionWitness` object with exact
ordered fields: `witness_id:S`, `consumer:S`, `expected_exception:S`,
`observed_exception:S`, `input_sha256:H`, `before_state_sha256:H`,
`after_state_sha256:H`, `before_effect_sha256:H`, `after_effect_sha256:H`,
`prior_ledger_sha256:H`, `terminal_ledger_sha256:H`, and `rejected:B`.
Witness IDs, consumers, and expected exceptions are fixed by the attack,
malformed, structural, or isolation tables. A passing witness requires matching
exception strings, equal before/after state and effect hashes, equal prior and
terminal ledger hashes, and true rejection. A structural attack rejected before
a ledger exists uses the exact all-zero 64-hex digest in both ledger fields.
`observed_exception` is a closed enum: it must be byte-identical to that row's
`expected_exception`, which must itself be one of the exact live-attack,
structural, malformed-class, or `IsolationDeniedError` literals frozen above.
It carries no exception message, repr, traceback, path, argument, or target text.
Witness arrays are ordered by their frozen table order; their SHA-256 root is
the domain-prefixed canonical sequence encoding, not a producer-chosen string.

The sanitized `SourceBoundaryReceipt` projection is a closed ordered object
with `fit_path_id:S`, `source_mode:S`, `train_episodes:I`,
`validation_factory_accesses:I`, `test_factory_accesses:I`,
`policy_snapshot_accesses:I`, `terminal_train_ledger_sha256:H`, and
`completed:B`. Case 11 contains exactly 21 in frozen path/mode order. The
sanitized `SourceOrderReceipt` is a closed ordered object with
`fit_path_id:S`, `raw_order_a_sha256:H`, `raw_order_b_sha256:H`,
`normalized_order_a_sha256:H`, `normalized_order_b_sha256:H`,
`terminal_order_a_sha256:H`, `terminal_order_b_sha256:H`, and `completed:B`.
Case 11 contains exactly seven in frozen path order and their exact aggregate
root. The
sanitized `EvaluationSourceReceipt` projection is a closed ordered object with
`policy_id:S`, `stream_tag:S`, `regime_digest:H`, `factory_opens:I`,
`episodes_yielded:I`, `other_operations:I`, and `completed:B`. Cases 8 and 9
each contain exactly two in frozen regime order; the control cardinalities are
frozen below. Their JSON
projections are validated field by field,
then re-encoded through their typed ABI records before either receipt root is
derived; unknown fields, reordered receipts, or a JSON/typed-record mismatch
reject.

For combined control evaluation, the sanitized
`EvaluationBoundaryReceipt` is the closed ordered object `stream_tag:S`,
`before_sha256:H`, `after_sha256:H`, and `unchanged:B`. Its hashes must decode
to the exact typed `EvaluationBoundarySnapshot` digests and be equal. The
receipt-sequence root is
`SHA256(b"v3-evaluation-boundary-receipt\x00" ||
length_delimited_ordered_receipts)`. Controls 15 through 17 contain exactly two
boundary receipts with exact tags `control_id + ":validation"` then
`control_id + ":test"`, and four source receipts with `policy_id=control_id`
and exact tags `control_id + ":" + regime_name` in frozen four-regime order.
Marker ablation contains exactly four boundary receipts tagged
`marker-ablation-fresh:validation`, `marker-ablation-fresh:test`,
`marker-ablation-frozen:validation`, and `marker-ablation-frozen:test`, in that
order. Its eight source receipts use the corresponding exact fresh or frozen
policy ID plus each regime name, grouped fresh-validation, fresh-test,
frozen-validation, frozen-test with regime order inside each group.
Each control's source root is exactly
`SHA256(b"v3-evaluation-source\x00" ||
canonical_sequence_of_its_complete_EvaluationSourceReceipt_records)` in that
stated order; no canonical or other control receipt enters the preimage.
A count, equality Boolean, or root without the exact underlying receipts
rejects.

Every witness `before_state_sha256` or `after_state_sha256` is
`SHA256(b"v3-boundary-snapshot\x00" || complete_BoundarySnapshot_record_bytes)`.
That closed record covers all Q cells; learner instance, module, class, global,
and closure state; permit and operation ledgers; environment cursor; token
schedule; RNG; attack hooks/events; source spies; and filesystem, network,
process, and native effect ledgers. The corresponding effect hash is
`SHA256(b"v3-boundary-effect\x00" || source_spy_sha256 ||
filesystem_effect_sha256 || network_effect_sha256 || process_effect_sha256 ||
native_effect_sha256)`. The pinned validator independently constructs both
records from the real consumer context. No omitted mutable namespace, cursor,
schedule, hook, or external-effect ledger may be represented by an empty
constant.

Each negative control also emits one closed `ControlReceipt` with ordered fields
`control_id:S`, `transform_sha256:H`, `applicability_root_sha256:H`,
`consumer_receipt_root_sha256:H`, `evaluator_contract_sha256:H`,
`evaluator_config_sha256:H`,
`policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`,
`output_trace_sha256:H`, `evaluation_boundary_root_sha256:H`,
`evaluation_source_root_sha256:H`,
`dynamic_projection_root_sha256:H`, `reached:B`, and `completed:B`. The
controller binds these four receipts to the static predicates, complete dynamic
causal roots, exact train metrics, and control case fields before deriving any
control pass. The receipt root is SHA-256 of the `v3-control-root` prefix plus
the canonical sequence of these four compact ordered JSON objects; the root
field itself lives only in case 19 and is not in that preimage.

Every case object starts with `case_id:S` and `passed:B`, followed by exactly
the fields in this field-by-field table. A `passed` field is never accepted as
a substitute for the listed real-consumer evidence.

| # / case ID | exact remaining fields and types | exact gates |
| --- | --- | --- |
| 1 `complete_family_replay` | `template_rows:I`, `expanded_rows:I`, `nonterminal_rows:I`, `terminal_rows:I`, `predecessor_identities:I`, `realized_rows:I`, `unique_keys:B`, `all_rows_replayed:B` | 196, 39,536, 16,944, 22,592, 19,768, 8,472, true, true |
| 2 `evaluator_twin_boundary` | `changed_template_rewards:I`, `changed_expanded_rewards:I`, `changed_public_fields:I`, `changed_nonterminal_fields:I`, `twin_replay:B` | 28, 5,648, 0, 0, true |
| 3 `split_and_source_commitment` | `regimes:I`, `train_regimes:I`, `validation_regimes:I`, `test_regimes:I`, `unexpected_public_variants:I`, `heldout_in_train_api:I`, `public_regime_invariance:B`, `commitment_replay:B` | 7, 3, 2, 2, 0, 0, true, true |
| 4 `behavior_schedule_replay` | `train_episodes:I`, `forced_episodes:I`, `greedy_episodes:I`, `tokens:I`, `paired_tokens:I`, `token_mismatches:I`, `schedule_replay:B` | 1,800, 1,728, 72, 5,400, 2,700, 0, true |
| 5 `online_chronology` | `initial_selections:I`, `next_action_selections:I`, `action_decisions:I`, `steps:I`, `updates:I`, `appends:I`, `terminal_returns:I`, `order_mismatches:I` | 1,800, 3,600, 5,400, 5,400, 5,400, 5,400, 1,800, 0 |
| 6 `sarsa_target_and_table` | `q_cells:I`, `unexpected_mutable_state:I`, `target_mismatches:I`, `cell_mismatches:I`, `implementation_table_sha256:H`, `oracle_table_sha256:H`, `state_closure:B`, `bit_exact:B` | 28, 0, 0, 0, equal hashes, true, true |
| 7 `train_metrics` | `forced_successes:I`, `greedy_successes:I`, `reward:F`, `regret:I`, `analytic_match:B` | 216, 72, 288.0, 1,512, true |
| 8 `validation_metrics` | `episodes:I`, `successes:I`, `validation_a_return:F`, `validation_b_return:F`, `mean_return:F`, `regret:I`, `updates:I`, `evaluator_config_sha256:H`, `evaluator_contract_sha256:H`, `policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`, `output_trace_sha256:H`, `evaluation_boundary_before_sha256:H`, `evaluation_boundary_after_sha256:H`, `evaluation_source_receipts:A`, `evaluation_source_root_sha256:H`, `state_mutations:I`, `action_replay:B` | 512, 512, 1.0, 1.0, 1.0, 0, 0; exact outer-derived config H; four valid stream H; equal exact boundary H; two exact typed source receipt projections and exact root; 0; true |
| 9 `test_metrics` | `episodes:I`, `successes:I`, `test_a_return:F`, `test_b_return:F`, `mean_return:F`, `regret:I`, `updates:I`, `evaluator_config_sha256:H`, `evaluator_contract_sha256:H`, `policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`, `output_trace_sha256:H`, `evaluation_boundary_before_sha256:H`, `evaluation_boundary_after_sha256:H`, `evaluation_source_receipts:A`, `evaluation_source_root_sha256:H`, `state_mutations:I`, `action_replay:B` | 512, 512, 1.0, 1.0, 1.0, 0, 0; same exact outer-derived config H; four valid stream H; equal exact boundary H; two exact typed source receipt projections and exact root; 0; true |
| 10 `comparators` | `constant_zero_return:F`, `constant_one_return:F`, `myopic_return:F`, `no_bootstrap_return:F`, `random_validation_return:F`, `random_test_return:F`, `random_draws:I`, `minimum_random_margin:F`, `comparator_input_sha256:H`, `comparator_action_sha256:H`, `comparator_output_sha256:H`, `all_streams_replayed:B` | first four exact 0.0; random values each <=0.35; 8,472; >=0.55; three exact validator-reconstructed roots; true |
| 11 `train_only_isolation` | `source_mode_checks:I`, `spy_ledgers:I`, `heldout_fit_operations:I`, `source_boundary_receipts:A`, `source_boundary_root_sha256:H`, `source_order_receipts:A`, `source_order_root_sha256:H`, `normalized_order_a_sha256:H`, `normalized_order_b_sha256:H`, `raw_delivery_a_sha256:H`, `raw_delivery_b_sha256:H`, `raw_delivery_distinct:B`, `inverse_injection_rejected:B` | 21, 14, 0; 21 exact source-boundary receipts and root; 7 exact A/B source-order receipts and root; equal normalized hashes; two valid unequal raw hashes matching the joined per-path commitments; true; true |
| 12 `trace_authentication` | `components:I`, `joined_steps:I`, `joined_trajectories:I`, `append_receipts:I`, `component_root_sha256:H`, `append_receipt_root_sha256:H`, `terminal_ledger_root_sha256:H`, `reordered_root_sha256:H`, `duplicates:I`, `unmatched:I`, `reordered_replay:B` | 23,400, 5,400, 1,800, 5,400, valid bound roots, 0, 0, true |
| 13 `malformed_record_rejection` | `applicable_mutations:I`, `rejected_mutations:I`, `witnesses:A`, `witness_root_sha256:H`, `state_mutations:I`, `all_rejected:B` | 45, 45, 45 fixed witness objects, valid root, 0, true |
| 14 `capability_and_timing_rejection` | `live_attacks:I`, `rejected_attacks:I`, `live_witnesses:A`, `live_witness_root_sha256:H`, `structural_attacks:I`, `structural_rejected:I`, `structural_witnesses:A`, `structural_witness_root_sha256:H`, `concurrent_races:I`, `race_winners:I`, `race_rejections:I`, `race_effects:I`, `canonical_permits_issued:I`, `canonical_permits_spent:I`, `attack_state_mutations:I`, `all_rejected:B` | 45, 45, 45 witnesses and valid root, 30, 30, 30 witnesses and valid root, 1, 1, 1, 1, 21,600, 21,600, 0, true |
| 15 `disjoint_zero_bootstrap_control` | `train_forced_successes:I`, `train_greedy_successes:I`, `train_reward:F`, `train_regret:I`, `canonical_update_gate:B`, `oracle_table_match:B`, `validation_a_return:F`, `validation_b_return:F`, `test_a_return:F`, `test_b_return:F`, `validation_return:F`, `test_return:F`, `evaluator_config_sha256:H`, `evaluator_contract_sha256:H`, `policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`, `output_trace_sha256:H`, `evaluation_boundary_receipts:A`, `evaluation_boundary_root_sha256:H`, `evaluation_source_receipts:A`, `evaluation_source_root_sha256:H`, `canonical_gate_vector:A`, `failed_clause_indices:S`, `failed_canonical_clauses:I`, `completed:B`, `identical_evaluator_contract:B`, `positive_gate_recovered:B` | 216, 0, 216.0, 1,584, false, true; all six returns exact 0.0; exact config H and four valid stream H; 2 exact boundary receipts and root; 4 exact source receipts and root; 22 exact Booleans; exact `6,9,10,11,12,14,15`; 7; true; true; false |
| 16 `zero_assignment_control` | `train_forced_successes:I`, `train_greedy_successes:I`, `train_reward:F`, `train_regret:I`, `canonical_update_gate:B`, `oracle_table_match:B`, `validation_a_return:F`, `validation_b_return:F`, `test_a_return:F`, `test_b_return:F`, `validation_return:F`, `test_return:F`, `evaluator_config_sha256:H`, `evaluator_contract_sha256:H`, `policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`, `output_trace_sha256:H`, `evaluation_boundary_receipts:A`, `evaluation_boundary_root_sha256:H`, `evaluation_source_receipts:A`, `evaluation_source_root_sha256:H`, `canonical_gate_vector:A`, `failed_clause_indices:S`, `failed_canonical_clauses:I`, `completed:B`, `identical_evaluator_contract:B`, `positive_gate_recovered:B` | 216, 0, 216.0, 1,584, false, true; all six returns exact 0.0; exact config H and four valid stream H; 2 exact boundary receipts and root; 4 exact source receipts and root; 22 exact Booleans; exact `6,9,10,11,12,14,15`; 7; true; true; false |
| 17 `zero_terminal_origin_control` | `train_forced_successes:I`, `train_greedy_successes:I`, `train_reward:F`, `train_regret:I`, `canonical_update_gate:B`, `oracle_table_match:B`, `validation_a_return:F`, `validation_b_return:F`, `test_a_return:F`, `test_b_return:F`, `validation_return:F`, `test_return:F`, `evaluator_config_sha256:H`, `evaluator_contract_sha256:H`, `policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`, `output_trace_sha256:H`, `evaluation_boundary_receipts:A`, `evaluation_boundary_root_sha256:H`, `evaluation_source_receipts:A`, `evaluation_source_root_sha256:H`, `canonical_gate_vector:A`, `failed_clause_indices:S`, `failed_canonical_clauses:I`, `completed:B`, `identical_evaluator_contract:B`, `positive_gate_recovered:B` | 216, 0, 216.0, 1,584, false, true; all six returns exact 0.0; exact config H and four valid stream H; 2 exact boundary receipts and root; 4 exact source receipts and root; 22 exact Booleans; exact `6,9,10,11,12,14,15`; 7; true; true; false |
| 18 `marker_ablation_control` | `train_forced_successes:I`, `train_greedy_successes:I`, `train_reward:F`, `train_regret:I`, `canonical_update_gate:B`, `oracle_table_match:B`, `fresh_validation_a_return:F`, `fresh_validation_b_return:F`, `fresh_test_a_return:F`, `fresh_test_b_return:F`, `fresh_validation_return:F`, `fresh_test_return:F`, `frozen_validation_a_return:F`, `frozen_validation_b_return:F`, `frozen_test_a_return:F`, `frozen_test_b_return:F`, `frozen_validation_return:F`, `frozen_test_return:F`, `evaluator_config_sha256:H`, `evaluator_contract_sha256:H`, `policy_observation_stream_sha256:H`, `policy_snapshot_sha256:H`, `output_trace_sha256:H`, `evaluation_boundary_receipts:A`, `evaluation_boundary_root_sha256:H`, `evaluation_source_receipts:A`, `evaluation_source_root_sha256:H`, `canonical_gate_vector:A`, `failed_clause_indices:S`, `failed_canonical_clauses:I`, `ablated_rows:I`, `unablated_signal_rows:I`, `completed:B`, `identical_evaluator_contract:B`, `positive_gate_recovered:B` | train 216, 72, 288.0, 1,512, true, true; every held-out return <=0.50; exact config H and four valid tagged-combined stream H; 4 exact boundary receipts and root; 8 exact source receipts and root; 22 exact Booleans; `failed_clause_indices="11,12,14,15"`; `failed_canonical_clauses=4`; `ablated_rows=39,536`; `unablated_signal_rows=0`; true; true; false |
| 19 `intervention_difference_whitelist` | `paired_static_rows:I`, `disjoint_bootstrap_differences:I`, `disjoint_tag_differences:I`, `assignment_address_differences:I`, `assignment_tag_differences:I`, `terminal_reward_differences:I`, `terminal_origin_differences:I`, `terminal_tag_differences:I`, `ablation_predecessor_public_differences:I`, `ablation_predecessor_key_differences:I`, `ablation_successor_public_differences:I`, `ablation_successor_key_differences:I`, `ablation_bootstrap_differences:I`, `ablation_tag_differences:I`, `control_receipts:A`, `control_receipt_root_sha256:H`, `dynamic_projection_root_sha256:H`, `unexpected_static_differences:I`, `missing_required_differences:I`, `unexplained_dynamic_differences:I`, `causal_replay:B` | 158,144; then 16,944, 39,536, 19,768, 39,536, 2,824, 22,592, 39,536, 39,536, 19,768, 16,944, 8,472, 8,472, 39,536; four fixed receipts and two valid roots; then 0, 0, 0, true |
| 20 `isolation_probes` | `network_probes:I`, `network_rejected:I`, `file_probes:I`, `file_rejected:I`, `process_probes:I`, `process_rejected:I`, `native_probes:I`, `native_rejected:I`, `probe_witnesses:A`, `probe_witness_root_sha256:H`, `unauthorized_effects:I` | 8, 8, 8, 8, 8, 8, 6, 6, 30 witnesses and valid root, 0 |
| 21 `fresh_reproduction_and_sanitizer` | `fresh_children:I`, `surviving_children:I`, `stderr_bytes:I`, `maximum_child_stdout_bytes:I`, `serializer_worst_case_bytes:I`, `serializer_budget_ok:B`, `projections_equal:B`, `projection_sha256:H`, `approved_sources:I`, `bootstrap_sha256:H`, `packet_abi_sha256:H`, `capability_map_sha256:H`, `worker_interface_sha256:H`, `learner_adapter_sha256:H`, `ast_policy_sha256:H`, `runtime_manifest_sha256:H`, `controller_validator_sha256:H`, `registry_digest_sha256:H`, `unexpected_projection_fields:I`, `cpu_only:B` | 2, 0, 0, <=524,288, controller-derived <=524,288, true, true, valid H, 5, exact embedded bootstrap, packet, capability-map, worker-interface, learner-adapter, AST-policy, runtime-manifest, committed-controller, and normalized-registry hashes, 0, true |

For case 2, neither changed-reward count is accepted as a producer-chosen
literal: the strict validator independently derives both by joining the frozen
regime/episode cue counts to the two-terminal-row-per-cue twin mapping. For case 10,
`minimum_random_margin` is `min(1-random_validation_return,
1-random_test_return)`. The controller validator recomputes every relation,
aggregate, bound, equality, hash relation, derived count, case pass, top-level
status, and action from the pairs-preserving projection. It rejects a producer
`passed` value that disagrees with any scalar or relation.

For cases 15 through 18, `canonical_gate_vector` is the exact 22-element
Boolean vector obtained by substituting that control's authenticated train and
held-out metrics, hashes, records, and intervention evidence into the positive
gate in listed order. Clauses 1 through 20 and 22 are computed first. Clause 21
is then true exactly when this control completed, its declared intervention was
reached through every applicable real consumer, every other control receipt is
present and schema-valid, and the false indices among the other 21 entries are
the exact frozen set for this control. Thus clause 21 is never self-referential
and is never permitted to hide an additional failed clause.

The three zero controls must have false indices exactly
`6,9,10,11,12,14,15`: their transformed target/update attribution fails the
canonical target clause, their greedy-train and held-out performance fails, and
their independently replayed transformed table must still match the worker
table. Marker ablation must have false indices exactly `11,12,14,15`; its forced
and per-cue greedy train blocks retain the analytic train totals, but both
held-out evaluation streams lose the cue association. Every unlisted clause is
required true. The ID-specific controller recomputes every element, the exact
comma-separated string `failed_clause_indices`, its integer count
`failed_canonical_clauses`, and `positive_gate_recovered`; any producer mismatch
rejects.

Canonical case 8 has 512 evaluator-contract, observation, and output records in
`validation-a`, then `validation-b`, episode-index order, plus exactly two
snapshot records tagged `canonical:validation-a` and
`canonical:validation-b`. Case 9 analogously has 512 records and two snapshots
for `test-a`, then `test-b`. Every `EvaluatorContractRecord` uses exact tie-rule
literal `smallest-argmax-action-0-before-1`, metric literal
`mean-terminal-return-and-episode-regret-v1`, `read_only=true`, and
`updates_allowed=false`.

Controls 15 through 17 each have one 1,024-record contract sequence ordered
validation A, validation B, test A, test B, then episode index 0 through 255.
That sequence is exactly the canonical case-8 contract records followed by the
case-9 records and therefore has the same combined contract root for all three
controls. Each has 1,024 observation and output records and exactly four
snapshot records, tagged with its exact control policy ID plus each regime name
under the colon rule above.

Marker ablation uses the exact policy IDs `marker-ablation-fresh` then
`marker-ablation-frozen`; within each it orders validation A, validation B,
test A, test B and episode indices. Its contract hash contains the one shared
1,024-record canonical contract sequence once. Its observation and output roots
each contain 2,048 records, and its policy root contains exactly eight snapshot
records: four fresh-table then four frozen-table tags. No contract, policy,
regime, fresh/frozen stream, or snapshot can be duplicated, omitted, or
exchanged.

For marker ablation, positive clauses 11, 12, 13, 14, and 15 are reduced across
both tags: a clause is true only if its predicate is true independently for
both streams. Clause 13 remains true only when both have zero updates and exact
action replay; clauses 11, 12, 14, and 15 are therefore false because each
stream has exact balanced return 0.50. All other evaluation relations are also
checked per tag before any combined aggregate is accepted.

The registry entry must repeat this exact 21-case order, all explicit type tags,
all literal and relational gates, the 524,288-byte limit, zero-stderr rule,
sanitizer allowlist, success/failure actions, worker identity, source approvals,
and claim boundary. The controller must have an ID-specific strict validator;
the generic scalar projection is insufficient.

After each cases-1-through-20 projection returns, the pinned outer validator
uses a separately implemented, separately hashed controller oracle and never
imports or calls the fixture, worker, or learner. From the frozen regime,
schedule, evaluator, comparator, intervention, ABI, and permit contracts it
independently reconstructs every family row; component/append/terminal-ledger
preimage; static and complete dynamic control projection; operation/control
receipt; comparator stream; evaluator contract, observation stream, policy
snapshot relation, and output aggregate needed by cases 1 through 20. It
compares every worker root, witness relation, count, scalar, and Boolean against
those reconstructions before deriving case 21. The bounded reconstructed rows
exist only in validator memory and are discarded after the immutable result and
sidecar are sealed. A root the outer validator cannot reconstruct is a malformed
result and parks the controller; fresh-worker agreement alone is insufficient.

Before any worker launch, the registry validator computes the exact maximum
UTF-8 serialization size from the closed schemas, 45 live, 30 structural, 45
malformed, and 30 isolation witness counts; 21 train-source boundary receipts;
seven A/B source-order receipts; 24 evaluation-source receipts; 10 control
evaluation-boundary receipts; four control receipts; fixed maximum of 80 UTF-8 bytes for every result ID, consumer,
exception, tag, action, and claim string; fixed hash/revision widths;
signed-int64 maximum decimal width; float encoder bounds; punctuation; and one
final LF. It writes that integer only into case 21 and requires it at most
524,288. The actual child output must independently be no larger. Unbounded
messages, paths, tracebacks, reprs, or producer-selected strings are forbidden.

## Fresh-process and infrastructure boundary

A later implementation may add exactly these two new study modules and no
helper module:

- `experiments/local_lab/online_sarsa_latched_choice_v3.py`;
- `experiments/local_lab/online_sarsa_latched_choice_v3_worker.py`.

The exact five worker-source approvals are:

1. `uv.lock`;
2. `experiments/local_lab/online_sarsa_latched_choice_v3.py`;
3. `docs/AUTONOMOUS_LAB.md`;
4. `research/2026-08-30-online-sarsa-latched-choice-v3-plan.md`;
5. `experiments/local_lab/online_sarsa_latched_choice_v3_worker.py`.

Those five paths are the closed worker import/input surface. Separately, the
infrastructure manifest pins the committed SHA-256 of
`tools/run_local_lab.py`, the exact V3 validator function identity, and the
normalized registry digest at the approved revision. The controller verifies
its own hash before acquiring a lease or launching the worker; it is not a
sixth worker source and cannot be imported by study code.

The implementation checkpoint must commit hashes for those exact paths, add
only the dedicated V3 worker allowlist entry, add the exact registry contract
above, and refresh the controller's normalized pinned registry digest. Its
allowed repository diff is closed to the two modules, focused V3 tests,
`experiments/local_lab/studies.json`, the minimal V3 allowlist, source-approval,
and strict-validator changes in `tools/run_local_lab.py`, one dated pre-result
audit record, and a truthful `docs/CURRENT_HANDOFF.md` gate transition. It may
not touch `submission/`, packaged defaults, generated artifacts, earlier
fixtures, earlier plans/results, or earlier registry contracts.

Before launch, the outer controller verifies the clean revision and source
hashes, creates a fresh empty directory beneath the sibling private lab root,
copies only the five approved inputs into a controller-readable but worker-denied
read-only ACL subtree, and places a
separate denied canary plus controller-owned output endpoint outside the
worker's access token. It pins the absolute Python executable hash, stdlib tree
digest, OS build, controller hash, and ACL/profile digest in a runtime manifest.
No repository root, package directory, private control file, result directory,
credential, inherited file handle, or writable source path is visible to the
worker.

The controller launches exactly
`<absolute-python> -I -S -c <fixed-bootstrap-text>` from the empty scratch root.
The bootstrap text is embedded as a fixed controller literal, committed by
`bootstrap_sha256` in the runtime manifest and case 21, and independently
byte-compared before launch. It is infrastructure, not a sixth worker source.
The controller sends one outer-length-prefixed packet over stdin containing the
approved worker bytes, fixture bytes, runtime manifest bytes, exact mode
`study-parent`, and their already verified hashes. The bootstrap establishes
the audit/import/file/network/process/native-loader policy, verifies the packet,
then compiles and executes the worker and fixture in separate explicit name
maps. It never adds the copied source tree to `sys.path`, imports a package, or
executes a source file by pathname. Therefore neither the worker, fixture,
`experiments/local_lab/__init__.py`, nor another study-source byte is compiled
before the policy is active. The
environment allowlist is exactly `SYSTEMROOT`, `WINDIR`, `TEMP`, `TMP`,
`PYTHONHASHSEED=0`, `PYTHONUTF8=1`, and `PYTHONIOENCODING=utf-8`; TEMP/TMP point
inside the isolated scratch. `PATH`, `PYTHONPATH`, `PYTHONHOME`,
`VIRTUAL_ENV`, dynamic-loader variables, proxy variables, cloud/provider keys,
and every GPU/device variable are absent. The absolute executable removes any
need for PATH lookup.

The stdin packet ABI is exact: one little-endian uint64 total-payload length,
then a payload beginning with ASCII `L2D-SARSA-V3-PACKET\x00`, version byte
`0x01`, mode byte `0x01` for `study-parent` or `0x02` for
`projection-child`, one Boolean nonrecursive byte, then the 32 raw bytes of the
bootstrap, worker, fixture, and runtime-manifest SHA-256 values in that order,
followed by three little-endian uint64 lengths for worker, fixture, and manifest
in that order, then the three corresponding blobs in that same order. The
fixed payload header is therefore exactly 175 bytes: 20-byte magic, three
control bytes, 128 hash bytes, and 24 length bytes. Parent mode requires
nonrecursive false; child mode requires true. EOF must follow the manifest
blob. Exact byte caps are `worker_blob_limit=262144`,
`fixture_blob_limit=2097152`, `manifest_blob_limit=262144`, and
`packet_payload_limit=3145728`. These four exact integers live in
`RuntimeManifest`, are repeated as typed integers in the packet-ABI commitment,
and are independently pinned in the registry/controller validator. The outer
length must equal the exact remaining byte count, be at most 3,145,728, and EOF
must immediately follow it.

Before allocation, hashing, or compilation, both the controller and bootstrap
read the outer length into one exact 8-byte fixed buffer, then the complete
payload header into one exact 175-byte fixed buffer. They reject an outer
length above the total cap, decode all three uint64 blob lengths, reject each
against its individual cap, and require with unsigned overflow-checked
arithmetic that `outer_length == 175 + worker_length + fixture_length +
manifest_length`. Blob buffers are allocated only after all four cap/equality
checks pass. A committed source exceeding its cap quarantines
V3 during preflight rather than changing a limit. Any overflow, premature EOF,
trailing byte, alternate order, wrong hash, wrong mode, cap mismatch, or blob
above its exact limit rejects before `compile`.

The capability map is also closed. Its preimage is the ASCII section name
`fixture-builtins`, then its lexicographically sorted names, followed by
`fixture-globals`, `worker-builtins`, and `worker-globals` and their sorted
names; each section and name is uint32-length-prefixed and each list begins with
its uint32 count. `capability_map_sha256` is SHA-256 of
`b"v3-bootstrap-capability-map\x00"` plus that exact sequence. Fixture builtins are exactly
`AssertionError`, `Exception`, `False`, `None`, `RuntimeError`, `StopIteration`,
`True`, `TypeError`, `ValueError`, `__build_class__`, `abs`, `all`, `any`,
`bool`, `bytearray`, `bytes`, `dict`, `enumerate`, `float`, `frozenset`, `hash`,
`int`, `isinstance`, `issubclass`, `iter`, `len`, `list`, `map`, `max`, `min`,
`next`, `object`, `open`, `range`, `repr`, `reversed`, `set`, `slice`, `sorted`, `str`,
`sum`, `super`, `tuple`, and `zip`. Exact fixture globals are `__builtins__`,
`__name__`, `__package__`, `__spec__`, `asyncio`, `concurrent_futures`, `copy`,
`ctypes`, `dataclass`, `dataclass_fields`, `dataclass_replace`, `hashlib`,
`http_client`, `io`, `json`, `math`, `mmap`, `multiprocessing`, `os`, `pathlib`,
`pickle`, `socket`, `struct`, `subprocess`, `tempfile`, `threading`, `types`,
`typing`, `urllib_request`, plus the sealed audit, permit, output, and
effect-ledger gateways. `open` exists only as the sealed audited built-in
probe gateway. `__import__`, `compile`, `eval`, `exec`, `globals`, `locals`,
`vars`, `dir`, `getattr`, `setattr`, and `delattr` are absent from executed
study namespaces.

Worker builtins are exactly `AssertionError`, `Exception`, `False`, `None`,
`True`, `TypeError`, `ValueError`, `bool`, `bytes`, `dict`, `int`, `len`, `list`,
`str`, and `tuple`. Worker globals are exactly `__builtins__`, `__name__`,
`__package__`, `__spec__`, `approved_context`, `emit_projection`, and
`fixture_api`. The four fixture gateway global names encoded in the map are
exactly `audit_gateway`, `permit_gateway`, `output_gateway`, and
`effect_ledger_gateway`; “plus” above does not permit an unnamed global.

The sealed worker interface is exact. `approved_context` is a slot-only frozen
record with fields `revision:str40`, `mode:str` (only `study-parent` or
`projection-child`), `approved_source_hashes:tuple[5,bytes32]`,
`runtime_manifest_sha256:bytes32`, and `claim_boundary:str` exact
`synthetic-online-sarsa-harness-only`. `fixture_api` is a slot-only frozen
object exposing only `run_projection(approved_context) -> bytes`; the returned
bytes are one compact ordered cases-1-through-20 JSON object and contain no raw
record, evaluator target, ledger, permit, source path, manifest, or callback.
`emit_projection(payload:bytes) -> None` is single-use, accepts only those
already validated bytes, enforces the 524,288-byte cap, writes them to stdout,
and returns `None`. Attribute enumeration, mutation, subclassing, serialization,
copying, alternate call signatures, and a second call reject. The ABI signature
strings, closed field lists, and exact `learner_adapter_sha256` are encoded as a
canonical length-prefixed sequence and committed by
`worker_interface_sha256 = SHA256(b"v3-worker-interface\x00" ||
sequence_bytes)` in the manifest and case 21. The interface has no member
through which a target, trace, table, packet, or source byte can be requested.

The bootstrap alone sees packet buffers. It compiles fixture and worker code
objects only after policy and hash verification, zeroes and deletes every raw
packet/source buffer, executes the fixture in the exact map above, then executes
the worker in a separate map containing only the frozen fixture API, immutable
approved revision/mode/hash context, and sealed output gateway. Neither map
contains bootstrap globals, stdin, a manifest blob, source bytes, code-object
buffers, repository paths, private paths, or the bootstrap/worker/fixture source
module dictionary. The fixture map does contain only the named read-only stdlib
module objects above; that is the complete qualified module surface. The
learner AST subset is stricter than this fixture map and retains the closed
selection/update capabilities above. The implementation audit independently
hashes the actual name sets before any study call; a missing or extra name
quarantines V3.

The learner AST policy is the exact `AstPolicy(version="learner-ast-v1")`
record. Its sorted allowed node-kind strings are `And`, `Assign`, `BoolOp`,
`Call`, `Compare`, `Constant`, `Eq`, `Expr`, `FunctionDef`, `Gt`, `GtE`, `If`,
`IfExp`, `List`, `Load`, `Lt`, `LtE`, `Module`, `Name`, `Not`, `NotEq`, `Or`,
`Return`, `Store`, `Subscript`, `Tuple`, `USub`, `UnaryOp`, `arg`, and
`arguments`. Allowed identifiers are exactly `action`, `forced_action`, `int`,
`marker`, `mode`, `q0`, `q1`, `q_row`, and `select_policy_action`; any other
identifier rejects. Allowed
literal spellings are exactly `-1.0`, `0`, `0.0`, `1`, `1.0`, and `255`.
Forbidden identifiers explicitly include `__builtins__`, `__import__`,
`compile`, `eval`, `exec`, `gc`, `globals`, `inspect`, `locals`, `open`,
`sys`, and `vars`; the closed allowed set rejects every unlisted alias too.
Calls are permitted only when the callee is the exact name `int`. The parsed
learner must define exactly the adapter-signature function and no second
function or top-level executable statement. Let `normalized_ast_sha256` be
`SHA256(b"v3-learner-ast\x00" || UTF8(ast.dump(tree,
annotate_fields=True, include_attributes=False)))`. The policy digest is
`SHA256(b"v3-ast-policy\x00" || AstPolicy_record_bytes ||
learner_adapter_sha256 || normalized_ast_sha256)`. The parsed AST from the
approved fixture source must reproduce that digest exactly. No Attribute,
Lambda, comprehension, import,
class, decorator, async, yield, exception, context-manager, or dynamic-call
node is allowed.

`packet_abi_sha256` is SHA-256 of `v3-packet-abi` plus the canonical primitive
sequence of the packet magic, version, mode enum, nonrecursive rule, four hash
fields, three blob fields, length widths, the exact three per-blob caps, total
payload cap, outer-length rule, checked-arithmetic rule, and EOF rule frozen
above. `RuntimeManifest` is encoded with the typed ABI and its digest is
`SHA256(b"v3-runtime-manifest\x00" || RuntimeManifest_record_bytes)`. Its five
source hashes use the numbered approval order. Runtime descriptor preimages are
closed as follows:

- `python_executable_sha256` is ordinary SHA-256 of the exact executable file
  bytes;
- `python_dll_set_sha256` hashes `v3-python-dll-set` plus the canonical
  primitive sequence of the complete loaded-module snapshot defined below,
  ordinal-sorted by canonical final path, where an entry is
  `(canonical_final_path, SHA256(file_bytes))`;
- `stdlib_tree_sha256` hashes `v3-stdlib-tree` plus the ordinal-sorted sequence
  of every regular file under the exact stdlib root defined below as
  `(forward_slash_relative_path, SHA256(file_bytes))`; directories, timestamps,
  ACLs, and alternate streams are not entries;
- `acl_profile_sha256` hashes `v3-acl-profile` plus the object-label-ordered
  sequence `(object_label, SHA256(UTF8(canonical_final_path)),
  SHA256(self_relative_security_descriptor_bytes), required_access_mask,
  forbidden_access_mask)` for executable, DLL set, stdlib tree, mapping probe,
  cwd, TEMP, approved-source subtree, repository, private lab root, canary, and
  output endpoint;
- `job_policy_sha256` hashes `v3-job-policy` plus the exact primitive tuple
  `(limit_flags=0x00002408, active_process_limit=1,
  inherited_handle_count=3, inherited_handles=(stdin,stdout,stderr),
  breakaway=false, silent_breakaway=false)`, where the flags are exactly
  `JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
  | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and neither breakaway flag is set;
- `network_policy_sha256` hashes `v3-network-policy` plus
  `(appcontainer_sid, integrity=low, capability_sid_count=0,
  firewall_exception_count=0)`;
- `native_policy_sha256` hashes `v3-native-policy` plus the canonical final path
  and file digest of `denied-native-probe.dll`, its self-relative security
  descriptor digest, and exact `allowed_native_loads=0`;
- `environment_sha256` hashes `v3-environment` plus the exact allowed
  name/value pairs ordinal-sorted by name. No absent variable is serialized;
  the controller separately rejects any name outside the seven-name allowlist.

The DLL closure is operationally exact. The fixed bootstrap imports, in this
order, `asyncio`, `concurrent.futures`, `copy`, `ctypes`, `dataclasses`,
`hashlib`, `http.client`, `io`, `json`, `math`, `mmap`, `multiprocessing`, `os`,
`pathlib`, `pickle`, `socket`, `struct`, `subprocess`, `tempfile`, `threading`,
`types`, `typing`, and `urllib.request`. Immediately after the final import and
before either approved source is compiled, it calls
`K32EnumProcessModulesEx(LIST_MODULES_ALL)`. Every returned PE module except the
pinned executable is one DLL-set entry; duplicate canonical paths reject. A
pre-result infrastructure probe and each later study process must produce the
same ordered path/content sequence. `LdrRegisterDllNotification` is installed
at that boundary, and any later module load not already in the committed set
rejects before fixture execution. Thus “DLL set” means this observed complete
loaded-module closure, not an implementation-selected dependency subset.

The stdlib root is exactly `canonical_final_path(sys.base_prefix + "/Lib")`
from the same isolated executable. It must be a directory, remain beneath the
canonical `sys.base_prefix`, and contain no reparse-point directory. Recursive
enumeration is depth-first only for discovery; the hash input is every regular
file, including any in-tree cache file, ordinal-sorted by its case-preserved
forward-slash relative UTF-8 path. A missing, extra, duplicate, hard-linked
alias, or reparse-point file rejects.

ACL labels and masks are frozen. Fixed labels occur in this order:
`python-executable`, `python-parent`, then `python-dll:` plus the zero-padded
eight-digit DLL ordinal and each distinct `python-dll-parent:` ordinal, then
`stdlib-root`, all `stdlib-dir:` relative paths, all `stdlib-file:` relative
paths, `mapping-probe`, `cwd`, `temp`, `approved-sources`, `repository`,
`private-lab`, `denied-canary`, and `output-endpoint`. Ordinal collections use
the same canonical path ordering as their content commitments. Required and
forbidden worker access masks are exactly:

| ACL object class | required mask | forbidden mask |
| --- | ---: | ---: |
| executable, parents, DLLs, stdlib root/dirs/files | `0x001200A9` | `0x000D0156` |
| mapping probe | `0x00120089` | `0x000D0176` |
| cwd and temp | `0x001200A0` | `0x000D015F` |
| approved sources, repository, private lab, canary, output | `0x00000000` | `0x001F01FF` |

Every object has a protected self-relative descriptor with owner equal to the
controller SID, null group, no SACL, ACL revision 2, control bits exact
`0x9004`, and no inherited ACE. Its DACL is canonical and contains, in order:
worker deny ACE type `0x01`, flags `0x00`, forbidden mask; worker allow ACE type
`0x00`, flags `0x00`, required mask when nonzero; controller-SID allow ACE type
`0x00`, flags `0x00`, mask `0x001F01FF`; and LocalSystem SID `S-1-5-18` allow
ACE type `0x00`, flags `0x00`, mask `0x001F01FF`. A zero-mask ACE is omitted;
no other ACE, SID, inheritance flag, audit entry, owner, group, or control bit
is permitted. The descriptor bytes, numeric masks, labels, and actual access
probes must all match this table.

`canonical_final_path` is UTF-8 from `GetFinalPathNameByHandleW`, with the
`\\?\` prefix removed, drive letter uppercased, backslashes changed to forward
slashes, and all other code points preserved without case folding or Unicode
normalization. `self_relative_security_descriptor_bytes` is the complete
self-relative binary security descriptor returned by `GetSecurityInfo` after
the controller's canonical ACE order is installed. `os_build` is the exact
ASCII `major.minor.build.revision` returned by `RtlGetVersion`.
Within every descriptor tuple above, labels, enum literals, and paths are type
`0x01`; counts and access masks are `0x02`; Booleans are `0x03`; digests are
`0x05`; and tuples are `0x07` under the canonical primitive framing. There is no
implicit textual rendering of a Boolean, mask, count, or digest.
The outer controller reconstructs every descriptor from the actual files,
handles, token, ACLs, Job, and environment before launch. Any missing/extra
manifest field, capability, DLL, stdlib file, environment name, source hash,
ACE, or policy bit rejects before lease acquisition.

The fixture executes with an exact restricted builtins/name map and cannot
import a module. The bootstrap rejects any extra packet field, byte, mode,
source, argument, or second stdin message before compilation.

The sole OS profile is frozen to Windows AppContainer name
`Learn2Design.LocalLab.OnlineSarsaV3`, whose SID is independently reproduced by
`DeriveAppContainerSidFromAppContainerName`. The lowbox token has low integrity
and an exact empty capability SID list: no internetClient, private-network,
child-process, broad-file, device, package, or GPU capability. Its ACL grants
the exact table's read/execute mask only to the pinned Python executable, its
canonical parent, pinned DLL set and parents, and pinned stdlib tree; read only
to a controller-created one-byte
`mapping-probe.bin`; traverse only, but no list/read/create/write/delete, to the
empty cwd and TEMP/TMP directory; and no access to the approved-source copies,
repository, sibling private lab controls/results, denied canary, output path,
credentials, or any other filesystem object. Worker/fixture bytes arrive only
through stdin and result bytes leave only through stdout.

Each invocation is created suspended, assigned before first instruction to a
new Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, active-process limit
one, no breakaway/silent-breakaway, die-on-unhandled-exception, and no inherited
handles beyond stdin/stdout/stderr, then resumed. There is no worker-side child
broker or authorized process capability. The outer trusted controller alone
launches the primary and two later reproduction invocations, each in a distinct
one-process Job. No network firewall exception or native-loader exception is
installed after bootstrap sealing.

The sealed audit gateway normalizes a denied probe to exact
`IsolationDeniedError` only after matching the exact manifest call/arguments
and proving the AppContainer/ACL/Job/native policy would deny the effect, and
before the underlying OS call. Connection refused, DNS error, file-not-found,
permission error, successful DLL reuse/load, child startup, or any other target
error is a probe failure, not a normalized pass. Effect ledgers independently
confirm zero OS effect. If the host cannot establish and probe this exact
profile, implementation preflight quarantines V3; Python monkeypatches alone
are insufficient.

The controller additionally enforces local CPU only, no shell, one global
3,600-second cycle/process-tree deadline measured from lease acquisition
through immutable result sealing, V3-specific 524,288-byte stdout cap before
parsing, zero-stderr, global lease, immutable result, and SHA-256 sidecar. The fixed
approved-source threat model remains narrower than arbitrary malicious native
code, but source/ACL/network/process controls must all pass.

The real sealed consumers must exercise exactly these isolation probes:

| class | exact probes |
| --- | --- |
| network (8) | `socket.socket` connect to `127.0.0.1:9`, `socket.create_connection` to `192.0.2.1:9`, `socket.getaddrinfo("example.invalid",443)`, `http.client.HTTPConnection("192.0.2.1",9).connect`, `http.client.HTTPSConnection("192.0.2.1",9).connect`, `urllib.request.urlopen("http://192.0.2.1:9/")`, `asyncio.open_connection("192.0.2.1",9)`, `asyncio.start_server` on loopback port zero |
| file (8) | built-in `open` read denied canary, built-in `open` write TEMP/TMP, `io.open` denied canary, `pathlib.Path.open` write TEMP/TMP, `os.open` denied canary, `os.listdir` empty cwd, file-backed `mmap.mmap` with `ACCESS_WRITE` over an `rb` handle to read-only `mapping-probe.bin`, `tempfile.NamedTemporaryFile` in TEMP/TMP |
| unpermitted process (8) | `subprocess.Popen` absolute Python `-c pass`, `subprocess.run` same, `subprocess.check_output` same, `os.system("exit 0")`, `os.spawnv` absolute Python, `multiprocessing.Process.start`, `concurrent.futures.ProcessPoolExecutor.submit`, `ctypes.windll.kernel32.CreateProcessW` with a fixed inert command line |
| native loader (6) | `ctypes.CDLL`, `ctypes.PyDLL`, `ctypes.WinDLL`, `ctypes.OleDLL`, `ctypes.cdll.LoadLibrary`, `ctypes.windll.LoadLibrary`, each targeting the exact absolute `denied-native-probe.dll`, a controller-created byte-identical copy of the runtime-manifest-pinned Windows `version.dll` placed under an ACL the AppContainer cannot read or execute |

Each probe is independently invoked after sealing and must reject at its real
consumer before a network, filesystem, child-process, or native-load effect.
Before/after effect ledgers must be equal. Constructing a literal rejected flag
does not count. Every probe requires exact `IsolationDeniedError`; its witness
binds the listed call and arguments, OS policy/profile digest, before/after
effect hashes, and zero unauthorized effects.

The outer trusted controller, never worker or fixture code, authorizes exactly
three sequential launches: one primary packet with mode `study-parent` and two
fresh packets with mode `projection-child`. All bind the same pinned executable,
bootstrap, worker/fixture/runtime hashes, scrubbed environment, AppContainer,
ACL, and one-process Job policy. Each has a 600-second timeout and 524,288-byte
stdout cap. The 600-second limit applies independently to each invocation and
does not reset or extend the enclosing 3,600-second cycle deadline. Either
deadline expiring fails closed. `RuntimeManifest.timeout_seconds` is exact 600;
the 3,600-second outer deadline is committed by the controller validator and
registry contract because it encloses all three manifests. Timeout, nonzero
exit, any stderr byte, malformed/extra stdout,
surviving process, wrong policy/revision, or projection mismatch fails closed;
the controller closes the Job and verifies no survivor.

Each invocation independently reconstructs the full bounded study and returns
only sanitized ordered cases 1 through 20; none may spawn. The two reproduction
projections and primary projection must be byte-identical. The pinned outer
validator separately reconstructs the commitments described below, derives and
appends case 21, validates the complete top-level projection, and writes the
single immutable result. There is no authorized process activity inside a
fixture or worker.

No fixture, worker, learner, registry path, controller path, or private result
is executed in this plan checkpoint. The later implementation checkpoint also
must not execute the fixture or worker for result-bearing metrics or invoke
`tools/run_local_lab.py`. It may run focused structural tests and at most one
full repository verification pass without observing a study result. If exact
implementation requires relaxing this contract, V3 is quarantined rather than
amended or repaired.

## Stopping rule and actions

This plan plus the minimal truthful handoff update is the entire current
checkpoint. The commit is the immutable freeze boundary. It is created on a
focused branch stacked exactly on the clean V2-rejection PR head, pushed, and
opened as a draft successor PR. It is not amended in this checkpoint and no lab
PR is merged.

A later heartbeat may implement only this contract, obtain a separate clean
pre-result commit, pass focused tests, at most one full repository verification
pass, three independent hostile audits, source/hash/registry/controller checks,
and green CI. That heartbeat leaves the private controller
`awaiting_study`. Only a still later heartbeat, after rechecking the exact clean
revision, green CI, approved source hashes, protected hashes, absent stop
marker, absent or validly absent lease, no concurrent cycle, and confirmation
that V3 has never run, may invoke the guarded controller once on local CPU.

Success requires every one of the 21 cases and all 22 positive-gate clauses and
uses exactly:

`synthetic_online_sarsa_repeated_code_control_confirmed_for_harness`

Any failed threshold or invariant, recovered negative control, malformed or
nondeterministic projection, timeout, source or revision drift, process
survivor, or controller error uses exactly:

`park_online_control_v3_research`

A substantive pre-result confound produces no terminal result: quarantine the
ID, keep the controller `awaiting_study`, record a sanitized rejection, and
require a fresh versioned plan. There is no retry, top-up, alternate seed,
threshold change, case removal, relaxed rule, same-ID repair, or direct worker
execution. Every terminal study remains excluded from rerun.

The protected submission tree must remain
`e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`; the protected ZIP and manifest
must remain, respectively,
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`
and `99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`.

## Claim boundary

A future pass may say only that this fixed local-CPU synthetic harness
reconstructed the declared family, selected actions online under the frozen
forced and greedy repeated-code schedule, applied exact SARSA(0) next-action
updates, acquired the deliberately public binary cue mapping, froze its table
on untouched generator regimes, beat the frozen toy comparators, and lost the
positive association under the exact bootstrap-key, update-assignment,
terminal-origin, and marker interventions.

It cannot support a claim that learning was necessary, that the task lacked a
public shortcut, that the exploration policy is optimal, or that results
generalize beyond this schedule. It is not evidence for general or production
RL, meta-RL, partial observability, official data, hidden or private topology,
UIFO, optimizer quality, candidate selection, a native rewrite, accelerator
value, leaderboard rank, competition score, or permission to edit or upload
the protected submission or spend money.
