# Full-surface prefix study plan

Status: frozen before execution
Study ID: `full-surface-prefix-indistinguishability-v1`
Execution class: unpaid, deterministic local CPU mechanics only

## Question

Under an explicit adapter limited to the information available on the current
optimizer's normal logged evaluation path, can two deterministic synthetic
paths remain identical through a finite bound across candidate parameters,
loss, total gradient, every public UIFO-shaped auxiliary leaf, deterministic
optimizer-transition commitments, RNG transcript, budget counters, and
incumbent state, while differing at the next feasibility observation?

This is an information-boundary question. It does not ask whether the visible
signals have distributional predictive value, whether any learned or
programmed policy improves the competition objective, or whether the submitted
candidate should change.

## Frozen adapter and fixture

- Study bound `B = 8`; the late crossing occurs at observation `B + 1 = 9`.
- The synthetic logged batch has population `P = 4` and parameter dimension
  `N = 3`; every returned or optimizer-owned vector is represented for all
  four members.
- Both worlds execute the protected checked-in `BatchedRestartAdam.optimize`
  method against a public-shape scripted objective. The objective implements
  only the preparation, RNG, budget, logging, and batched aux-evaluation
  methods used by that normal path. It rejects scalar evaluation and has no
  partial tail.
- Decision timing is observe-then-decide. A deterministic binary policy chooses
  `keep` or `restart` from the accumulated allowed snapshots.
- Exact top-level snapshot fields, in order:
  `candidate`, `loss`, `total_gradient`, `aux`, `optimizer_state`,
  `rng_transcript`, `budget_counters`, and `incumbent_state`.
- Exact auxiliary leaves:
  `is_feasible`, `penalty`, `power_values.detector`,
  `power_values.hard`, `power_values.soft`, `sensitivity_loss`, and
  `violations`.
- `optimizer_state` is an adapter-owned transition commitment, not a private
  optimizer-state dump. It contains the complete public telemetry event, the
  completed-batch count, deterministic recent-timing window, a digest of all
  fixed hyperparameters, the prior transition commitment, and a commitment to
  the current allowed inputs. Identical initial population, fixed code and
  hyperparameters, current allowed observations, RNG transcript, budget and
  timing inputs induce the same deterministic protected transition through the
  bound. The adapter neither reads nor monkeypatches private Adam attributes,
  and the study does not claim to expose arbitrary hidden implementation state.
- The RNG transcript contains only the optimizer-owned draw count and typed
  digest. Raw keys and samples never enter the retained result.
- Budget counters include batch index, evaluation count/limit/remaining,
  budget-exceeded state, deterministic elapsed/remaining time, and progress
  fraction. A deterministic clock proxy supplies identical synthetic batch
  durations to the otherwise unchanged protected timing path; no measured
  wall-clock value enters identity.
- Incumbent state is updated online from the current public evaluation only and
  includes presence, best feasible loss, and a candidate digest.
- Both worlds use one exact strictly improving, finite, infeasible executed
  prefix. The forever-infeasible world remains infeasible at step 9; the
  late-crossing world becomes feasible at step 9. Candidate population, loss,
  total gradient, and the six non-Boolean aux leaves remain identical at step
  9, so `aux.is_feasible` is the only primary evaluation difference. Normal
  downstream telemetry and incumbent state may then differ.
- Signal-class negative controls perturb candidate, loss, total gradient,
  optimizer state, RNG transcript, budget counters, and incumbent state one at
  a time. Auxiliary controls perturb each of the seven auxiliary leaves one at
  a time. Canonical recursive comparison must report exactly that one declared
  difference path for every control.
- Typed-array controls hold raw bytes equal while changing dtype or shape and
  require the typed projection identity to change.
- Extension sentinels try seven prohibited input classes: unlogged callable,
  extra evaluation, Hessian, manual log, private attribute, saved record, and
  structural metadata. The exact-schema adapter must reject every sentinel.
  This tests only schema closure; it does not claim to exercise or disable
  those forbidden APIs.
- CPU only, Python hash seed zero, two fresh credential-scrubbed,
  network-disabled worker projections for process isolation.

No official problem, dataset, topology, generated panel, private trajectory,
saved Objective history, provider, submission treatment, or outcome-bearing
artifact is an input. The fixture retains typed hashes only long enough to
compare its two synthetic executions; these audit projections are never policy
inputs and are not retained in the result. The scripted objective does not
instantiate UIFO or `Objective`, manually log an evaluation, request a Hessian,
invoke an unlogged callable, or inspect a private optimizer attribute.

## Complete frozen cases

1. `normal_path_execution`
   - Require two executions of nine complete population-four batches through
     `BatchedRestartAdam.optimize`, 36 evaluations per world, one logged start,
     one optimizer RNG draw, nine chained transition commitments, nine complete
     telemetry events, zero scalar calls, and zero restart events.
   - Require identical typed initial-population projections across worlds.
2. `adapter_schema`
   - Require the exact eight top-level fields and exact seven auxiliary leaves.
   - Require every array projection to contain exact dtype, shape, and SHA-256.
   - Require a valid baseline snapshot, rejection of a missing auxiliary leaf,
     and rejection of an extra top-level field.
3. `shared_full_surface_prefix`
   - Require both complete prefixes through `B = 8` to be finite, strictly
     improving member-wise, infeasible, valid under the exact typed schema, and
     byte-identical.
   - Require identical prefix SHA-256 values, differing complete step-9
     snapshots, and exactly `aux.is_feasible` as the primary step-9 evaluation
     difference.
4. `signal_class_negative_controls`
   - Perturb one representative leaf in each non-auxiliary signal class.
   - Require each perturbation's recursive diff set to equal its one declared
     path.
5. `aux_leaf_negative_controls`
   - Perturb each of the seven auxiliary leaves individually.
   - Require each perturbation's recursive diff set to equal its one declared
     leaf path.
6. `typed_array_metadata_boundary`
   - Hold raw bytes equal while changing dtype and then shape.
   - Require both typed projection identities to change.
7. `forbidden_extension_rejection`
   - Add each of the seven prohibited extension sentinels separately.
   - Require the exact-schema adapter to reject every extended snapshot.
8. `action_vector_exhaustion`
   - Exhaust all `2^8 = 256` binary action transcripts on the shared prefix.
   - `restart` at least once by `B` satisfies the bounded-restart obligation;
     all-`keep` preserves the late crossing. Require zero joint satisfiers,
     255 bounded-only transcripts, and one preserve-only transcript.
   - Treat this only as an abstract corollary of the executed shared prefix,
     not as an execution of a replacement optimizer policy.
9. `process_isolation`
   - Require two fresh worker projections to equal the parent projection and
     each other byte-for-byte.

## Invariants

- The adapter accepts exactly the frozen snapshot schema; no implicit metadata
  channel or extra callable is available.
- Every array field has its exact frozen dtype and shape, each transition
  commitment recomputes from the visible record, and adjacent commitments form
  one exact chain.
- Each negative control changes exactly one declared leaf or adds exactly one
  prohibited extension.
- Prefix comparison uses canonical JSON and exact SHA-256 identity, with no
  tolerance or floating-point timing input.
- The fixture emits only schema names, counts, Booleans, and hashes. Candidate
  vectors, gradients, RNG words, incumbent values, and complete snapshots are
  not retained in the result.
- The protected submission tree, submitted ZIP, and protected manifest remain
  unchanged.
- The controller's pinned registry digest, approved source hashes, global
  lease, private state/events, output cap, timeout, CPU pin, credential scrub,
  network block, immutable result, and SHA sidecar remain mandatory.

## Stopping and decision rule

Run the complete nine-case contract exactly once. There is no seed selection,
bound change, schema relaxation, case removal, retry, or top-up.

- If every case passes, record
  `synthetic_full_surface_prefix_twin_confirmed` and advance only to a new
  mechanism question. The next rung may map pure JAX/host execution boundaries
  or define a learned-controller contract, but it may not change the candidate.
- If any case fails, is malformed, times out, is nondeterministic, or encounters
  source drift, record or preserve the failure, park this research lane, and
  request owner review. Do not alter the fixture and rerun it.

## Claim boundary

The strongest permitted positive claim is:

> For the frozen deterministic adapter and finite bound, two synthetic paths
> can have byte-identical complete allowed snapshots through the bound and
> different next feasibility observations; therefore those current snapshots
> are not a universal certificate of the next feasibility result without
> additional assumptions.

This cannot establish that any public diagnostic is useless, that real UIFO
trajectories contain such twins, that a restart rule is harmful, that learning
will or will not help, or that any programmed, learned, native, or fused
optimizer improves runtime or score. It does not authorize official-data use,
candidate integration, accelerator work, paid training, or submission changes.
