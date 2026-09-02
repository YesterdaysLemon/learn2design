# Feasibility-debt restart clock v3 - frozen pre-result plan

Date: 2026-09-01

Study ID: `feasibility-debt-clock-v3`

Status: plan only; no candidate, fixture, worker, case, or terminal result exists

## Question

On a fresh deterministic five-lane synthetic optimizer family, can an
experiment-owned restart clock:

1. remain exactly identical to the protected `BatchedRestartAdam` transition
   when explicitly placed in `total_loss` mode;
2. use only the public nonnegative constraint penalty before a lane first
   becomes finitely feasible;
3. latch permanently to the protected finite total-loss rule for that
   generation after the first finite feasible observation, including later
   infeasible observations;
4. reset the latch, progress incumbent, stall count, Adam state, and generation
   together only on an authenticated restart mask; and
5. reproduce one canonical result projection across two fresh processes when
   incidental Python and file-descriptor stdout are physically separated from
   the result channel?

The answer is `passed` only if all nine frozen cases and every transport check
pass twice and the two canonical child byte streams are identical.

## Evidence boundary and independence

V1 and V2 are terminal parked studies and must never be imported, invoked,
repaired, or used as mechanics evidence. V3 may inherit only the public
mechanism question and V2's sanitized parent-level fact that two bounded,
stderr-free, zero-exit child streams were not whole-stream JSON. No V1 or V2
child output, case metric, raw stream, seed, row script, family, threshold, or
terminal diagnostic may be inspected or reused.

V3 uses a new candidate source copied independently from the protected
`submission/submission.py`, a new five-lane family, nine new cases, nine new
seeds, a new fixture, and a new stdlib-only bootstrap worker. It does not read
topologies, the official dataset, private outcomes, prior generated histories,
or any H100 result.

## Candidate contract

The new class is
`experiments.candidates.feasibility_debt_clock_v3.FeasibilityDebtBatchedRestartAdamV3`.
It is not imported by `submission/` and cannot change a packaged default.

Its `optimize` signature is the protected signature plus one required
keyword-only `progress_mode` whose exact values are `total_loss` and
`feasibility_debt`. `total_loss` must preserve the protected implementation
exactly. Treatment mode adds per-lane `generation_has_been_feasible` and
`best_progress` state:

- before the first finite feasible observation of a generation, progress is
  the finite nonnegative public `aux["penalty"]` scalar;
- the first finite feasible observation is always a progress event, resets the
  stall count, sets the latch, and initializes the post-feasibility incumbent
  from its finite total loss without comparing unlike penalty/loss units;
- while latched, every later observation uses finite total loss even if the
  observation is infeasible;
- every later improvement is strictly
  `progress < best_progress - minimum_improvement`;
- a nonfinite selected progress value is never an improvement;
- only a completed population transition may update progress state;
- an authenticated restart resets the latch to false, `best_progress` to
  positive infinity, the stall count and Adam member state to their protected
  initial values, and increments the protected generation exactly once;
- a partial tail, rejected auxiliary payload, callback, logging projection, or
  read-only inspection cannot mutate treatment, Adam, RNG, budget, or clock
  state.

The treatment adds no objective call, random draw, topology input, learned
parameter, semantic prior, or private source. Auxiliary validation accepts
only the exact public mapping required by the protected optimizer plus
`penalty`; `penalty` must be a floating array with the population leading
dimension and may not contain negative finite values. Unexpected keys,
containers, shapes, ranks, or dtypes reject before optimizer state changes.

## Independent five-lane family

Every terminal case uses population lanes `0..4`. Each admitted logical batch
contains exact arrays for total loss, penalty, feasibility, gradient, and
parameter update. Gradients and deterministic parameter updates are generated
from the batch index, lane index, and the case seed; they never depend on a
case outcome. The fixture records a canonical transition row containing:

```text
case, seed, batch, admitted_count, complete_population,
loss, penalty, feasible, selected_progress, improvement_mask,
stall_before, stall_after, restart_mask, generation_before,
generation_after, latch_before, latch_after, best_before, best_after,
adam_age_before, adam_age_after, rng_before, rng_after,
callback_count, objective_eval_count, update_applied
```

All arrays have leading size five unless the row is the frozen partial tail.
Every case starts from one scalar parameter per lane with
`param[lane] = ((seed % 64) + lane) / 128` and supplies
`gradient[batch,lane] = (1 + batch + 2*lane) / 64`; these are exact binary
fractions. The protected Adam equations, learning-rate vector, and restart
noise scale are copied from the protected source and are never replaced by a
fixture update. Unless a case states otherwise, loss is
`32 - 2*batch - lane/2`, penalty is `16 - batch - lane/4`, feasibility is
false, `minimum_improvement=0`, and every batch is complete. Float comparisons
therefore use exact binary integers, halves, and quarters. The source-boundary
comparison uses exact bytes and normalized AST shapes. No threshold or
expected mask is inferred from execution.

## Frozen cases and seeds

The complete ordered case set is:

| Case | Seed | Frozen construction and invariant |
|---|---:|---|
| `protected_composite_trace_identity` | 93503 | Eight complete batches, patience 2. Loss lanes are `[8,7,6,5,4,3,2,1]`, `[4,4,4,4,3,3,3,3]`, `[6,5,5,5,4,4,4,3]`, `[3,4,3,4,3,4,3,4]`, and `[9,8,8,7,7,6,6,5]`; feasibility is `(batch+lane)%3==0`; penalty is `32+4*batch+lane`. Run protected code and V3 `total_loss` from identical initial bytes. Histories, params, Adam state, stalls, restart masks, generation, RNG calls/state, callbacks, budgets, and final roots must be byte-identical. Penalty bytes cannot affect either projection. |
| `pre_feasible_penalty_routing` | 93521 | Six all-infeasible complete batches, patience 2. Total loss decreases strictly in every lane. Penalty rows are lane 0 `[5,4,4,4,3,3]`, lane 1 `[2,2,2,2,2,2]`, lane 2 `[6,5,4,3,2,1]`, lane 3 `[3,3,2,2,2,1]`, lane 4 `[1,1,1,0.5,0.5,0.5]`. Treatment restart batches must be `{0:[3],1:[2,5],2:[],3:[4],4:[2,5]}` while protected total-loss mode has no restart. |
| `first_feasible_irreversible_handoff` | 93529 | Seven complete batches, patience 2. Lane `i` first becomes and remains finitely feasible at batch `i` for lanes 0..4. Before that boundary its loss is `24-batch-lane/2`; at and after it, loss is the constant `10+lane`. Penalty is always `20+lane-batch`. The first feasible row is an unconditional progress event, the latch rises there, and the first restart occurs exactly at batch `i+2`. Penalty changes after the latch cannot postpone it. |
| `post_handoff_infeasible_reentry` | 93553 | Six complete batches, patience 2. Feasibility by batch is `[false,true,false,false,true,true]` in every lane. Loss is `20+lane` at batch 0, `10+lane` at batches 1..3, `9+lane` at batch 4, and `8+lane` at batch 5. Penalty is `6+lane-batch`. Every lane must remain on total loss after batch 1 and restart at batch 3; batch 4 begins a fresh generation, immediately relatches on its finite feasible row, and records `9+lane` as its new post-feasibility incumbent. |
| `masked_restart_state_rng_alignment` | 93559 | Five complete all-infeasible batches, patience 2. Loss decreases strictly. Penalty is `[4,4,4,3,3]` in lanes 0, 2, and 4 and `[5,4,3,2,1]` in lanes 1 and 3, making only lanes 0, 2, and 4 restart at batch 2. Exactly one protected restart-noise draw is allowed for that full mask. Those lanes alone reset latch/progress/Adam state and increment generation. Lanes 1 and 3 and the post-call RNG state must match an independent mask replay. |
| `chunk_partition_trace_equivalence` | 93581 | Three logical batches use loss `8-batch+lane/2`, penalty `5-batch+lane/4`, and feasibility `(batch+lane)%3==0`. Supply the same fifteen logical observations once as three full five-lane calls and once through physical chunks `(2,1,2)` per logical batch. Patience 3. Only the complete logical population may transition. Canonical logical histories, params, state, restart masks, RNG, callbacks, and roots must be identical. |
| `partial_tail_transactionality` | 93607 | `max_evals=17` uses the chunk-equivalence formulas for three complete five-lane batches followed by a two-lane tail with loss `[1,1.5]`, penalty `[0.5,0.75]`, feasibility `[true,false]`, and the common gradient formula at batch 3. The tail is logged and charged exactly once but has `complete_population=false` and `update_applied=false`; it cannot change params, treatment state, Adam state, restart generation, callback count, or RNG. |
| `auxiliary_nonfinite_fail_closed` | 93629 | Fresh instances exercise missing penalty, extra keys, wrong container, bool/integer/string dtype, scalar, wrong leading dimension, negative finite penalty, NaN penalty, positive-infinity penalty, NaN selected total loss, and positive-infinity selected total loss. Malformed schema and negative finite penalty reject before any effect. Allowed nonfinite selected progress values produce `improvement=false`, preserve the incumbent, and follow the frozen stall rule without introducing NaN into state. |
| `source_delta_and_process_seal` | 93637 | Independently normalize protected and V3 ASTs and require the exact allowlisted class name, algorithm string, auxiliary validator, treatment state, `progress_mode`, and progress-transition delta only. Hash the protected source, candidate source, fixture, bootstrap worker, plan, and normalized delta into one source root. Both terminal workers must report that root, the stdout-seal flag, the same ordered case roots, and the same core root. |

The case order, seeds, row generators, patience values, masks, allowed source
delta, and expected actions freeze with this plan. A conformance fix before
terminal execution may make implementation match these bytes but may not alter
them.

## Sealed result transport

The dedicated module
`experiments.candidates.feasibility_debt_clock_v3_worker` may import only the
Python standard library before sealing stdout. On process entry it must:

1. duplicate the inherited stdout file descriptor as the private result
   descriptor;
2. redirect file descriptor 1 to `os.devnull` with `dup2`;
3. rebind Python `sys.stdout` to the redirected descriptor;
4. only then import JAX, the candidate, or the fixture;
5. run the requested projection;
6. encode one canonical UTF-8 JSON object plus one newline; and
7. write that envelope only with bounded `os.write` calls to the duplicated
   result descriptor, then close it.

The worker cannot restore public stdout. A pre-result `--transport-probe`
mode, which does not import or execute a terminal case, must deliberately call
both `print("python-noise")` and `os.write(1,b"fd-noise")` after the seal. The
parent must receive only a fixed canonical probe envelope. Tests also inject a
module whose import writes through both paths; neither byte may reach the
result stream.

The terminal parent invokes exactly two fresh worker processes with a
credential-scrubbed environment, `JAX_PLATFORMS=cpu`, bytecode writing
disabled, no stdin, repository cwd, a 180-second child timeout, zero allowed
stderr, and a 262,144-byte stdout cap. It validates duplicate-key-free JSON,
exact keys and scalar types, study/plan/revision/source identities, ordered
case keys, Boolean outcomes, case roots, source root, core root, and the
stdout-seal flag. It never exposes or writes a rejected child stream.

## Closed projections

Each valid worker envelope contains exactly:

```text
study_id, invocation_revision, plan_revision, plan_sha256,
protected_source_sha256, candidate_source_sha256,
fixture_source_sha256, worker_source_sha256,
case_count, case_outcomes, case_roots, all_cases_passed,
stdout_sealed, source_boundary_root_sha256, core_root_sha256
```

The parent projection contains those identities plus exactly fourteen Boolean
transport fields per two workers: process started, zero exit, empty stderr,
stdout within cap, JSON parsed, schema valid, and study identity valid. It also
contains `runs_equal`, `process_replay_root_sha256`, `all_cases_passed`, and
`action`. Any exception, timeout, malformed envelope, mismatch, process
effect, nonempty stderr, unequal replay, false case, or missing hash yields a
valid closed parent failure projection with action
`park_feasibility_debt_v3`.

## Pre-result implementation gate

Implementation is a separate checkpoint. It may add only the new V3 candidate,
fixture, bootstrap worker, source verifier, focused tests, a dated pre-result
record, and the truthful handoff update. Before any terminal case executes:

- importing every module is side-effect free;
- the transport probe and import-noise probe pass in fresh processes;
- tests inspect case keys, seeds, frozen tables, AST/source boundaries, parent
  failure projection, environment scrub, timeouts, caps, and process cleanup;
- tests and collection contain no call to the case projector, terminal worker
  mode, or parent `--run` entry point;
- syntax checks, focused tests, the affected candidate boundary, and at most
  one full repository pass complete;
- exact source hashes and the plan hash are recorded at a clean pre-result
  commit; and
- the successor draft PR is green at that exact revision.

The plan commit is immutable. A substantive confound quarantines V3 without
execution. No pre-result fix may change a case, seed, row, patience, mask,
threshold, output field, success action, failure action, or claim boundary.

## One terminal attempt and stopping rule

After the clean green pre-result gate, invoke only:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --frozen --group dev --group integration python -m experiments.candidates.feasibility_debt_clock_v3_fixture --run
```

Exactly one terminal parent invocation is allowed. Never call the worker's
terminal mode or the case projector directly. Never retry, change output
locations, replace a seed, inspect a rejected stream, relax a rule, or repair
against the terminal result.

If every case and transport invariant passes twice with identical bytes, the
action is:

```text
approve_feasibility_debt_v3_for_fresh_candidate_screen_planning
```

Otherwise the action is:

```text
park_feasibility_debt_v3
```

## Claim boundary and later route

A pass can validate only the exact synthetic five-lane transition and sealed
transport. It cannot establish UIFO performance, H100 speed, topology
generalization, an optimal patience, a candidate score, or improvement over
the owner-reported Round-1 `0.444293`.

Only a pass opens a separate plan for a newly generated, archive-disjoint
candidate screen. That later plan may compare the exact Round-1 lifecycle,
public pre-clock warmup, V3 with an active pre-feasibility patience, and a
coverage-balanced V3 variant, but must independently freeze its arms, panel,
pairing, run order, score projection, selection rule, H100 environment, and
dollar ceiling. Provisioning, paid compute, submission edits, portal actions,
official/private outcome access, and merges continue to require their own
explicit owner decisions.
