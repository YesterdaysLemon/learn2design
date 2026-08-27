# Finite-infeasible restart-clock mechanics plan — 2026-08-26

## Frozen decision

Study ID: `feasible-progress-clock-v1`

Status: frozen before retained execution.

The only question is whether the unchanged `BatchedRestartAdam` member clock
treats a finite raw-loss improvement as progress even while that observation is
infeasible. The submitted patience-600/no-prior candidate remains protected.
This study does not change `submission/`, choose a patience value, compare
competition candidates, use a topology, or estimate competition performance.

Run exactly the seven deterministic cases below on the local JAX CPU backend.
All objective values, feasibility flags, gradients, initial populations, seeds,
budgets, and decision rules are fixed in the clean pre-result revision. The
worker runs with network access disabled and writes its immutable result only
beneath the sibling private local-lab root through `tools/run_local_lab.py`.

## Fixture identity

- population: 8 members;
- parameter width: 3 synthetic coordinates;
- optimizer seed: `20260826`;
- budget: four complete batches and 32 evaluations, except the declared
  improve-then-plateau case, which uses five complete batches and 40
  evaluations to observe one post-restart batch;
- patience: 2 completed non-improving evaluations per member;
- minimum improvement: `1e-7`;
- learning rate: `0.05` for every member;
- gradients: exactly zero;
- member base losses: `100 + member_index / 8` in float32;
- descending loss: base loss minus the zero-based batch index;
- restart source: the unchanged optimizer's seeded
  `random_params_unbounded` path;
- feasibility is false unless a case explicitly declares otherwise.

The scripted loss is intentionally independent of the parameter vector. This
isolates the clock predicate from gradient motion and restart quality. Loss and
parameter vectors are retained only as typed SHA-256 hashes in the sanitized
trace; no raw vector is emitted.

## Complete case set and invariants

### 1. `finite_infeasible_descent`

Every member receives the descending loss on all four batches and remains
infeasible. The case passes only if all 32 observations are finite and
infeasible, all 32 rows report member improvement, every post-observation stall
count is zero, no restart occurs, and no global feasible improvement occurs.

### 2. `finite_infeasible_plateau_control`

Every member receives its unchanged base loss on all four batches and remains
infeasible. The first batch improves from infinity. Batches one and two do not
improve, so the exact post-observation stall path must be `[0, 1, 2, 0]` for
every member. Batch two must restart all eight members, in member order, using
fresh restart kind 0 and restart round 0. The next evaluation must observe a
changed parameter hash for every restarted member and must improve from the
reset member best. Restarted members must expose Adam age 0 after the restart,
generation 0-to-1, and stall/Adam age 0 before the post-restart evaluation. No
global feasible improvement may occur.

### 3. `finite_infeasible_improve_then_plateau`

Every member receives its base loss on batch zero, a one-unit improvement on
batch one, then that improved loss thereafter. All rows remain infeasible. The
exact post-observation stall path must be `[0, 0, 1, 2, 0]` for every member.
Batch three must restart all members with fresh kind 0 and restart round 0; the
batch-four parameter hashes must differ from batch three for every member and
the reset member best must make the batch-four row an improvement. Adam age,
generation, and next-batch stall state must satisfy the same exact reset
invariants as the plateau control. This case pins the restart boundary relative
to the most recent infeasible improvement,
not merely the initial observation.

### 4. `late_feasibility_crossing`

Every member receives descending loss. All rows are infeasible for batches
zero through two; only member 0 is feasible on batch three. The case passes
only if every stall count remains zero, no restart occurs, and the sole global
feasible-improvement event is member 0 on batch three. This control separates
raw member progress from the global feasible incumbent.

### 5. `mixed_member_clock`

Even-indexed members receive descending loss and odd-indexed members plateau.
All remain infeasible. Even member stall paths must be `[0, 0, 0, 0]`; odd
member paths must be `[0, 1, 2, 0]`. Exactly members 1, 3, 5, and 7 must restart
on batch two with kind 0 and round 0. Between the batch-two evaluation and the
batch-three evaluation, parameter hashes must change for exactly those four
members. Exactly those four members must also expose the declared Adam-age,
generation, and next-batch stall resets. No global feasible improvement may
occur.

### 6. `diagnostics_disabled_control`

Run the mixed-member script twice from the identical initial state, once with
optimizer telemetry disabled and once enabled. Evaluation counts, batch sizes,
feasibility flags, typed loss hashes, typed parameter hashes, and initial
population hashes must be identical. This prevents the diagnostic callback
from creating the mechanism it purports to observe.

### 7. `process_isolation`

Run the complete sanitized mixed-member trace in two fresh, credential-scrubbed,
network-disabled CPU worker processes. Their timing-free JSON projections and
SHA-256 digests must be byte-for-byte identical.

## Stopping and decision rule

The study is terminal after one controller invocation. Do not repeat, top up,
drop a case, change a loss, change feasibility, change a seed, or relax an
invariant after any result is observed.

Pass only if all seven cases complete and every declared invariant passes. The
frozen success action is:

```text
finite_infeasible_progress_resets_clock_confirmed
```

Any failed case has the frozen action:

```text
park_feasible_progress_clock_research
```

A timeout, malformed result, nondeterminism, source drift, dirty worktree,
lease collision, or controller error also parks the laboratory. It is not a
study failure that can be repaired by rerunning the terminal fixture.

## Success and failure actions

A pass confirms only this code-level mechanism: the current member clock is
reset by finite raw-loss improvement without requiring feasibility, and such
improvement can suppress a restart through the bounded synthetic window. It
may justify separately planning an unpaid, synthetic reference-policy study.
It does not show that feasibility-aware progress is better, that any restart
policy improves optimization, or that the submitted candidate should change.

A failure or park ends mutation and requires owner review. No candidate,
submission artifact, portal, leaderboard, official dataset, private topology
panel, GPU, cloud resource, or paid endpoint is authorized by either outcome.
