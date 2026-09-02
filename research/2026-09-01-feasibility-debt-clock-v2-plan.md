# Feasibility-debt restart clock v2 - frozen pre-result design

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v2`

Status: plan frozen before V2 implementation or result observation

The commit containing this file and its matching handoff update is the freeze
boundary. After that commit exists, the family, cases, seeds, thresholds,
schemas, source boundary, and claim boundary below cannot be changed for this
ID.

## Prior-evidence boundary

`feasibility-debt-clock-v1` is terminal and parked. Its sole authenticated
terminal fact is that its parent emitted no sanitized projection after the
first frozen child exited `1`. No child output, case outcome, trajectory, or
metric was inspected. V2 must not import V1, run either V1 entry point, reuse a
V1 terminal row script or seed, infer a failing V1 case, or select any V2 value
against V1 development or terminal diagnostics.

V2 is a fresh experiment-owned study. It is motivated by the public UIFO
contract and the independently established mechanism question: a finite but
infeasible candidate can improve total loss without reducing its public
constraint penalty, so a total-loss patience clock can postpone exploration.
V2 also closes a generic result-transport requirement: a terminal parent must
emit a sanitized receipt even when a child or scientific case fails.

The protected `submission/` tree, every terminal and rejected local-lab
fixture, the closed H100 panels, official data, private outcome evidence, and
the competition portal remain outside this study.

## Falsifiable question

Can an experiment-owned copy of `BatchedRestartAdam`, with no extra objective
call and no additional learner input, make pre-feasibility patience depend only
on the public nonnegative `aux["penalty"]`, then permanently hand the same lane
back to the protected total-loss progress rule after its first finite feasible
observation, while preserving the protected implementation exactly in an
explicit compatibility mode and remaining deterministic across chunking and
fresh processes?

A pass can validate only the ordinary-runtime transition mechanics and source
isolation. It cannot establish that the treatment improves a UIFO topology,
an H100 run, the owner-reported Round-1 `0.444293`, or a competition score.

## Candidate contract

The only implementation under test is a new self-contained module:

```text
experiments/candidates/feasibility_debt_clock_v2.py
```

It must not import V1. It begins from the complete protected
`submission.submission.BatchedRestartAdam` source and adds one required
keyword-only `progress_mode` with exactly two values:

- `total_loss`: the complete optimizer projection must be byte-identical to
  the protected class for the same objective, seed, settings, callbacks, and
  budgets.
- `feasibility_debt`: each population lane has exactly three additional state
  values: `ever_feasible`, `best_infeasible_penalty`, and
  `best_post_feasibility_total_loss`.

For each full-population update, with `epsilon = minimum_improvement`:

1. If `ever_feasible` is false and the current observation is not feasible,
   progress occurs exactly when total loss is finite, penalty is finite and
   nonnegative, and
   `penalty < best_infeasible_penalty - epsilon`.
2. If the current observation is the lane's first finite feasible observation,
   it is progress, `ever_feasible` becomes true, and
   `best_post_feasibility_total_loss` becomes its total loss.
3. Once `ever_feasible` is true, feasibility no longer selects the progress
   scalar. Progress follows the protected finite total-loss rule exactly:
   `loss < best_post_feasibility_total_loss - epsilon`, including if a later
   observation is infeasible.
4. A progress observation resets that lane's stall clock; any other admitted
   full-population observation increments it.
5. A restart resets all three treatment values together with the protected
   per-lane Adam, step, best, and stall state. Non-restarted lanes are bitwise
   unchanged.
6. A partial budget tail may be evaluated and logged under the protected
   contract but cannot change treatment state, Adam state, stall state, or
   restart state.

The treatment adds no objective evaluation, random draw, topology input,
parameter dimension, prior, model, clock read, history read, private attribute,
or provider interaction. It retains the protected initialization, gradient
sanitization, clipping, Adam update, conditional restart RNG, exploit/explore
choice, callback timing, chunk order, and return behavior.

## Public auxiliary schema

Before any treatment transition, every full batch must authenticate the public
auxiliary tree already returned by `value_and_grad_aux`:

| Path | Exact requirement |
| --- | --- |
| `is_feasible` | Boolean array, leading dimension exactly population size |
| `penalty` | Real numeric array, rank 1, leading dimension exactly population size |
| `sensitivity_loss` | Real numeric array, leading dimension exactly population size |
| `violations` | Real numeric array, rank at least 2, leading dimension exactly population size |
| `power_values` | Mapping with exactly the required public members below |
| `power_values.hard` | Real numeric array, leading dimension exactly population size |
| `power_values.soft` | Real numeric array, leading dimension exactly population size |
| `power_values.detector` | Real numeric array, leading dimension exactly population size |

Finite negative penalties are malformed. `NaN`, `+inf`, and `-inf` penalty
values are authenticated as numeric but never count as progress. Boolean,
integer, scalar, complex, object, wrong-leading-dimension, missing-key, and
unexpected-container attacks fail before any treatment, optimizer, restart, or
callback transition. Validation performs no device-dependent coercion that
changes accepted types.

## Independent deterministic family

The fixture is a fresh four-lane, three-parameter scripted public Objective.
It never imports DFBench, constructs a topology, or reads a generated artifact.
Population size is `4`; `minimum_improvement` is `1e-7`; safety time is zero.
Each objective row is an exact tuple:

```text
(logical_batch, lane, total_loss, penalty, is_feasible, gradient_code)
```

`logical_batch` and `lane` are independently committed by the fixture.
Chunked calls consume consecutive lane rows and must reconstruct exactly one
logical population batch. The fixed gradient-code table is:

| Code | Gradient |
| --- | --- |
| `g0` | `[0.50, -0.50, 0.25]` |
| `g1` | `[-0.40, 0.30, -0.20]` |
| `g2` | `[0.10, 0.20, -0.30]` |
| `g3` | `[-0.25, -0.25, 0.50]` |

Lane `m` always uses code `gm`. The auxiliary non-penalty arrays are complete,
finite deterministic functions of the row identity and cannot carry a hidden
case label. Initial and restart parameters come only from the optimizer's
public random-parameter call. The objective commits every input fragment,
random draw, admitted history row, callback event excluding wall-clock fields,
and evaluation count.

The frozen seeds are unique to V2:

| Purpose | Seed |
| --- | ---: |
| compatibility, no restart | `92617` |
| compatibility, mixed restart | `92639` |
| pre-feasibility lane routing | `92657` |
| post-feasibility handoff | `92669` |
| restart-state isolation | `92681` |
| partial tail and chunking | `92707` |
| auxiliary attacks | `92723` |
| nonfinite semantics | `92737` |

These values were fixed before any V2 implementation or execution and were not
derived from V1 or any topology result.

## Complete cases

All ten cases below are mandatory. No case may be removed, weakened, repeated,
or replaced after this plan commits.

### 1. `total_loss_no_restart_identity`

Run protected code and V2 `total_loss` mode for six full batches with patience
`9`. For batch `b` and lane `m`:

```text
loss = 20.0 - 1.25*m - 0.50*b
penalty = 4.0 + 0.10*m - 0.20*b
is_feasible = b >= 3 + (m % 2)
```

The complete projections, callback fields, inputs, RNG draws, summaries, and
return values must be byte-identical, with no restart.

### 2. `total_loss_mixed_restart_identity`

Run protected code and V2 `total_loss` mode for seven batches with patience
`2`. The exact loss matrix, rows by batch and columns by lane, is:

```text
[
  [9.0, 9.0, 9.0, 9.0],
  [9.0, 8.5, 9.0, 8.0],
  [9.0, 8.0, 8.8, 8.0],
  [9.0, 7.5, 8.8, 7.8],
  [8.5, 7.0, 8.6, 7.8],
  [8.5, 6.5, 8.6, 7.6],
  [8.5, 6.0, 8.4, 7.6]
]
```

All rows are infeasible; `penalty = 3.0 + 0.1*lane`. The two complete
projections, including conditional random draws and lane-selective restarts,
must be byte-identical.

### 3. `pre_feasibility_lane_routing`

Run treatment mode for five all-infeasible batches with patience `3`:

```text
loss = [
  [12, 8, 11, 10], [11, 9, 10, 9], [10, 10, 9, 8],
  [9, 11, 8, 7], [8, 12, 7, 6]
]
penalty = [
  [3, 5, 6, 4], [3, 4, 5, 4], [3, 3, 4, 4],
  [3, 2, 3, 4], [3, 1, 2, 4]
]
```

Lanes `0` and `3` must restart first at batch `3`; lanes `1` and `2` must not
restart. The protected control has a different restart mask, while all inputs,
updates, and RNG draws before the first treatment/control divergence remain
identical.

### 4. `post_feasibility_total_loss_handoff`

Run six batches with patience `3`. The exact `(loss, penalty, feasible)` lane
sequences are below, where `T` and `F` are the exact Boolean values true and
false:

```text
lane 0: (10,4,F), (9,3,F), (8,0,T), (7,2,F), (6,3,F), (6,1,T)
lane 1: (12,5,F), (10,0,T), (10,1,F), (10,1,F), (9,2,F), (8,0,T)
lane 2: (14,6,F), (13,5,F), (12,4,F), (11,0,T), (10,4,F), (9,4,F)
lane 3: (16,8,F), (15,7,F), (14,6,F), (13,5,F), (12,4,F), (11,3,F)
```

After each first `T`, subsequent progress for that lane is decided solely by
finite total loss. In particular, lane `0` batch `3` and lane `2` batch `4`
must count as improvements despite being infeasible and having worse penalty.
The exact improvement flags, handoff flags, best values, and stall counts are
precommitted consequences of this table.

### 5. `restart_state_isolation`

Use eight batches and patience `2`. Lanes `0` and `2` have flat penalty runs
that force synchronized lane-selective restarts; lane `1` has strictly decreasing
penalty `8-b`; lane `3` becomes feasible at batch `2` and then has strictly
decreasing total loss. The full row generator is:

```text
loss[b,m] = 30 - 2*m - b
penalty[b,0] = 5 - floor(b/3)
penalty[b,1] = 8 - b
penalty[b,2] = 7 - floor(b/3)
penalty[b,3] = max(0, 4 - b)
feasible[b,3] = b >= 2; all other feasible values are false
```

Only the restart-mask lanes may reset treatment or Adam state. A paired
counterfactual replay changes only lanes `0` and `2` to strictly decreasing
penalties, suppressing their restarts while leaving lanes `1` and `3` byte-for-
byte unchanged. Lanes `1` and `3` must have identical input, treatment-state,
update, and callback commitments in both runs. For each restarted lane, a
fresh-lane oracle begins at its committed restart parameter with zero moments,
zero member step, false `ever_feasible`, and infinite best values; the lane's
next proposal and state must match that oracle exactly.

### 6. `partial_tail_no_transition`

Use four scripted batches, patience `2`, and `max_evals = 14`, yielding three
full four-lane updates plus a two-lane tail. The full rows have flat penalty and
decreasing total loss; the tail is finite and feasible. The tail must be logged
for exactly two lanes but must report `update_applied=false`, no treatment-state
change, no stall change, no restart, and no conditional restart random draw.

### 7. `chunk_projection_equivalence`

Run the case-4 script and the case-6 script at chunk sizes `None`, `1`, `2`,
and `3`, using seed `92707`. After excluding only wall-clock duration fields,
logical inputs, histories, callbacks, treatment states, RNG draws, evaluation
counts, and return commitments must have one identical root per script.

### 8. `auxiliary_schema_rejection`

Independently inject each of these attacks on the first full batch:

```text
missing each required path (8 attacks)
wrong leading dimension on each array path (7 attacks)
integer is_feasible
scalar penalty
complex penalty
list-valued power_values
finite negative penalty
```

Every attack must be rejected after exactly one objective batch and before any
optimizer update, treatment transition, restart RNG call, or callback. The
expected rejection count is exactly `20`.

### 9. `nonfinite_progress_semantics`

Exercise `NaN`, `+inf`, and `-inf` penalties before feasibility and `NaN`,
`+inf`, and `-inf` total losses after feasibility. Authenticated nonfinite
scalars never count as progress and never replace a finite best. A later finite
value must still count according to the frozen comparison rule. This case must
complete without an exception and with exact improvement and state flags.

### 10. `source_delta_boundary`

A dedicated verifier pins the complete protected and V2 source texts, every
unified-diff hunk payload, method sets, signatures, imports, and AST equality
for every helper except the intentionally changed `optimize` method and the new
auxiliary validator. V2 may add no dependency or import V1. The protected file
must match its committed hash at the V2 implementation revision.

## Result transport and process isolation

The terminal entry point is:

```text
python -m experiments.candidates.feasibility_debt_clock_v2_fixture --run
```

It requires a clean Git worktree, records the exact 40-character invocation
revision, authenticates the frozen plan hash, forces CPU, clears CUDA
visibility and proxy variables, scrubs credential-like environment keys, uses
closed stdin, captures stdout/stderr, caps each child at 524,288 bytes, and
enforces a 180-second child timeout.

Exactly two fresh child processes are precommitted. This is one terminal study,
not a retry. Each child emits exactly one canonical JSON object and exits zero
whenever it successfully evaluated the complete case vector, regardless of
whether a scientific case passed. Case failure is data, not a process error.

Each valid child object has exactly these fields:

| Field | Type |
| --- | --- |
| `study_id` | String constant `feasibility-debt-clock-v2` |
| `invocation_revision` | 40-character lowercase hex string |
| `plan_revision` | 40-character lowercase hex string |
| `plan_sha256` | 64-character lowercase hex string |
| `candidate_source_sha256` | 64-character lowercase hex string |
| `fixture_source_sha256` | 64-character lowercase hex string |
| `protected_source_sha256` | 64-character lowercase hex string |
| `case_count` | Exact integer `10` |
| `case_outcomes` | Exact ten-key Boolean mapping listed below |
| `case_roots` | Exact ten-key mapping of 64-character lowercase hex strings |
| `all_cases_passed` | Boolean equal to `all(case_outcomes.values())` |
| `source_boundary_root_sha256` | 64-character lowercase hex string |
| `core_root_sha256` | SHA-256 of the canonical object excluding this field |

The child schema rejects missing or extra keys, duplicate JSON keys,
noncanonical serialization, incorrect field types, an inconsistent aggregate,
or a root mismatch. A child must write this object before choosing its process
exit code.

The parent must never discard a child outcome before producing its own receipt.
For each launch it records only closed Boolean transport fields: process
started, exit code zero, stderr empty, stdout within cap, JSON parsed, schema
valid, and study identity valid. Raw invalid stdout or stderr is never surfaced.
The parent compares valid child bytes for exact equality and emits one canonical
sanitized JSON object on every handled pass or failure path. It writes that
object before returning exit code zero on pass or one on failure. A top-level
exception boundary emits the same closed failure schema with no exception text.
The second child is attempted even if the first child's transport fails.

The two children must be byte-identical for a pass. No child or parent may open
a socket, write a file, launch a grandchild, read credentials, inspect private
lab state, or use an accelerator. Focused process tests must exercise these
boundaries before the terminal invocation.

## Sanitized result schema

The parent result contains exactly these fields:

| Field | Type |
| --- | --- |
| `study_id` | String constant `feasibility-debt-clock-v2` |
| `invocation_revision` | 40-character lowercase hex string |
| `plan_revision` | 40-character lowercase hex string |
| `plan_sha256` | 64-character lowercase hex string |
| `candidate_source_sha256` | 64-character lowercase hex string or `null` |
| `fixture_source_sha256` | 64-character lowercase hex string or `null` |
| `protected_source_sha256` | 64-character lowercase hex string or `null` |
| `case_count` | Exact integer `10` |
| `case_outcomes` | Mapping with exactly the ten Boolean case keys |
| `transport_outcomes` | Mapping with exactly fourteen Booleans, seven per child |
| `all_cases_passed` | Boolean |
| `runs_equal` | Boolean |
| `source_boundary_root_sha256` | 64-character lowercase hex string or `null` |
| `process_replay_root_sha256` | 64-character lowercase hex string or `null` |
| `action` | One of the two exact action strings below |

`case_outcomes` has exactly these keys:

```text
total_loss_no_restart_identity
total_loss_mixed_restart_identity
pre_feasibility_lane_routing
post_feasibility_total_loss_handoff
restart_state_isolation
partial_tail_no_transition
chunk_projection_equivalence
auxiliary_schema_rejection
nonfinite_progress_semantics
source_delta_boundary
```

`transport_outcomes` has exactly these keys:

```text
child_1_process_started
child_1_exit_code_zero
child_1_stderr_empty
child_1_stdout_within_cap
child_1_json_parsed
child_1_schema_valid
child_1_study_identity_valid
child_2_process_started
child_2_exit_code_zero
child_2_stderr_empty
child_2_stdout_within_cap
child_2_json_parsed
child_2_schema_valid
child_2_study_identity_valid
```

If neither child supplies a valid case mapping, the parent emits all ten case
keys as `false`. If exactly one child is valid, the parent may project that
child's ten Booleans but `runs_equal` remains false. No missing or additional
key is permitted.

No topology, parameter vector, row value, raw output, exception text,
trajectory, timing sample, credential, provider value, or score is permitted.
Strict validation rejects extra keys, non-Boolean outcome values, noncanonical
JSON, or an action inconsistent with the outcomes.

## Stopping rule

V2 passes only if all ten cases pass, all fourteen transport outcomes are true,
the two valid child outputs are byte-identical, source verification passes, and
the complete schema is valid. The success action is:

```text
approve_feasibility_debt_v2_for_fresh_panel_planning
```

Every other handled outcome has action:

```text
park_feasibility_debt_v2
```

There is no threshold relaxation, case removal, seed swap, row change, child
inspection, repair against an observed result, or second terminal invocation.
Conformance fixes are allowed only before a separately committed clean
pre-result boundary.

## Verification sequence

1. Commit this plan and the minimal handoff update. That commit freezes V2.
2. Independently implement the V2 candidate, fixture, source verifier, and
   focused tests without running the ten result-bearing cases.
3. Commit a clean pre-result implementation boundary with exact file hashes.
4. Run focused static, source, transport, and contract tests, then at most one
   full repository verification pass.
5. Require a clean revision and green draft-PR CI.
6. Invoke `--run` exactly once on local CPU.
7. Record only its sanitized parent projection and never rerun V2.

No private-controller resume is required. No GPU, paid provider resource,
official dataset, private panel, submission change, portal action, or merge is
authorized by this plan.

## Route to a new number

Only a complete V2 pass permits a later plan-only gate for a newly generated,
archive-disjoint paired screen. That later plan must freeze the protected
`no_prior` control, the experiment-owned V2 treatment, untouched topology and
optimizer seeds, 600-second budgets, exact H100 runtime, paired action, score
projection, stopping rule, and all-in dollar ceiling. It must obtain fresh
explicit owner approval before provisioning or spending.

A generated-panel score would be a real development number but not directly
comparable to the hidden Round-1 `0.444293`. A genuinely comparable number can
come only from an admissible public competition evaluation under the organizer's
rules and a separate portal-action decision.
