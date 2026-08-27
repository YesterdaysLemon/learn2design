# Normal-path JAX boundary study plan

Status: frozen before execution
Study ID: `normal-path-jax-boundary-v1`
Execution class: unpaid, deterministic local CPU systems mechanics only

## Question

For one complete deterministic synthetic batch through the checked-in optimizer
and the public `dfbench.Objective` aux-evaluation path, where are the
source-visible compilation, dispatch, host conversion, synchronization,
callback, RNG, budget, and timing boundaries? Can an experiment-only pure-JAX
no-restart transition reproduce the same typed public observations and Adam
state exactly on the locked CPU runtime?

This is a mechanics and equivalence question. It does not ask whether fusion is
faster, whether the compiled call becomes one accelerator kernel, whether the
synthetic path represents UIFO performance, or whether the submitted candidate
should change.

## Frozen environment and fixture

- CPU only, Python hash seed zero, `dfbench==0.3.3`, JAX/JAXlib `0.9.0.1`, and
  the locked dependency wheel. The fixture authenticates the installed
  `Objective` source, dependency wheel, protected submission source, and JAX
  configuration before interpreting a trace.
- Population `P = 4`, parameter dimension `N = 3`, optimizer seed `20260827`,
  one complete full-vmap batch, and `max_evals = 4`.
- No supplied initial parameters, semantic prior, coverage balancing, chunking,
  preclock warmup, partial tail, or restart. Patience is `64`, safety seconds
  are zero, and the frozen Adam settings are learning-rate endpoints `0.05`
  and `0.11`, beta values `0.9` and `0.999`, epsilon `1e-8`, gradient clip norm
  `1.0`, and minimum improvement `1e-7`.
- The synthetic `ContinuousProblem` has float32 bounds `[-2, 2]^3`, but the
  exact runtime dtype remains part of the typed identity because the locked
  `dfbench` import enables JAX x64 and its random sampler does not pass an
  explicit dtype. Its differentiable public aux result has exactly the seven
  current UIFO-shaped leaves: `is_feasible`, `penalty`,
  `power_values.detector`, `power_values.hard`, `power_values.soft`,
  `sensitivity_loss`, and `violations`. Exactly the feasibility-anchor member
  is feasible; all losses and derivatives are finite.
- The public `Objective` has evaluation logging enabled but saves no parameter,
  derivative, aux, or timing histories and writes no checkpoint. Its normal
  explicit `vmap_value_and_grad_aux` path is used once.
- A deterministic wall clock supplies three ordered readings for
  `start_logging`, Objective evaluation logging, and telemetry. A separate
  deterministic performance clock supplies `4,000` microseconds for the
  evaluated batch. These tokens test ordering and state propagation only; no
  measured duration enters the result.
- Runtime hooks are limited to a tracing `Objective` subclass, the two public
  initial-population callbacks, the public optimizer telemetry callback, and a
  wrapper around public `jax.block_until_ready`. Hooks append labels or retain
  device references and return the original values unchanged. No hook performs
  `device_get`, array conversion, or scalar conversion until after
  `optimize()` returns.
- The pure transition receives the already-evaluated candidate, loss, total
  gradient, aux feasibility leaf, zeroed Adam/member state, geometric learning
  rates, explicit evaluation/budget/timing scalars, and the current incumbent.
  It implements the protected finite/feasible selection, member progress,
  derivative sanitation and clipping, Adam equations, no-restart mask, global
  incumbent update, and full telemetry event entirely with JAX arrays. It
  performs no callback, clock, Python or NumPy scalarization, device transfer,
  hidden objective access, or RNG operation.
- Array identity is the exact ordered triple `{dtype, shape, sha256}`. Recursive
  comparison is exact canonical JSON identity; no floating tolerance is used.
  Raw candidates, gradients, moments, keys, aux values, telemetry arrays, and
  complete snapshots are not retained in the study result.
- Two fresh credential-scrubbed, network-disabled CPU workers reproduce the
  timing-free non-process projection.

No official problem, dataset, topology, generated panel, private trajectory,
saved history, provider, candidate comparison, or outcome-bearing artifact is
an input.

## Complete frozen cases

1. `dependency_source_identity`
   - Require `dfbench==0.3.3`, its locked wheel SHA-256, the already frozen
     normalized `Objective` source SHA-256, the protected submission source
     SHA-256, JAX/JAXlib `0.9.0.1`, CPU backend, and x64 enabled.
2. `source_boundary_inventory`
   - Authenticate source before inventorying it.
   - In `BatchedRestartAdam.optimize`, require two syntactic
     `jax.block_until_ready` sites, two `time.perf_counter` sites, and the three
     exact device-dependent scalar conversions
     `int(jnp.argmin(feasible_losses))`,
     `float(feasible_losses[feasible_index])`, and
     `bool(jnp.any(restart_mask))`.
   - Require zero explicit `jax.jit` sites across `optimize` and `_adam_step`,
     and four callback invocation sites: raw initial population, final initial
     population, partial-tail telemetry, and full-batch telemetry.
   - Require the public Objective aux path to be bound as
     `vmap(jit(value_and_grad(..., has_aux=True)))` on this configuration.
     Inventory the five finite-batch device-dependent host decisions in
     Objective logging: the all-NaN branch and nanargmin conversion in
     `_nanargmin_or_none`, the repeated all-NaN branch and nanargmin conversion
     in `_log_evals`, and its best-loss comparison branch.
   - These are source-visible sites, not a claim about backend compilation or
     kernel counts.
3. `normal_path_boundary_trace`
   - Require one initial Objective RNG draw, one raw-initial callback, one
     final-initial callback, one `start_logging`, one public transformed
     evaluation call with shape `[4, 3]`, one explicit runtime ready barrier,
     two performance-clock reads, three Objective wall-clock reads, one
     full-batch telemetry callback, zero scalar objective calls, zero warmups,
     zero restarts, and evaluation count `0 -> 4` without overshoot.
   - Require exact ordering: raw-initial callback before final-initial callback;
     both before logging; performance-clock start before public evaluation;
     public evaluation return before the explicit ready barrier returns;
     performance-clock stop after that barrier; and telemetry after the stop.
   - Require exactly two reads of `budget_exceeded`, one of `time_left`, two of
     `evals_left`, one of `eval_count`, one of `time_elapsed` from the optimizer
     telemetry path, and one of `budget_progress_fraction`. Objective logging's
     separate `time_elapsed` read remains separately labeled.
   - Require the retained runtime trace to contain labels, integer counters,
     shapes, and typed hashes only.
4. `pure_jax_transition_equivalence`
   - Compare the experiment-only transition against the protected batch on the
     same device-resident inputs after the protected ready barrier.
   - Require exact typed equality for candidate, loss, total gradient, all
     seven aux leaves, the four protected `_adam_step` outputs, member best and
     stall state, incumbent selection, no-restart mask, and all 27 full-batch
     telemetry leaves.
   - Require the pure transition's source/JAXPR to contain no callback,
     `device_get`, NumPy, clock, Python scalar-conversion, or RNG operation.
5. `explicit_jit_lowering`
   - Materialize the pure transition eagerly, through `jax.jit`, and through
     one explicit `lower().compile()` call on the same frozen inputs. Require
     exact typed equality across all three projections.
   - Require a StableHLO digest and recursive JAXPR digest to reproduce across
     the two isolated workers. Require no `pure_callback`, `io_callback`, or
     host-callback primitive in the clean transition.
   - This records one explicit experiment-owned lowering/compile/call boundary;
     it does not infer the protected path's cache state, actual compile count,
     fusion, kernel count, or accelerator behavior.
6. `boundary_negative_controls`
   - Perturb telemetry feasibility, next-candidate state, evaluation count, and
     deterministic batch-microsecond state one at a time. Require canonical
     recursive comparison to report exactly the one declared path for each.
   - Hold bytes fixed while changing dtype and then shape; require typed
     identity to change in both controls.
   - Trace, but do not execute, a tiny `jax.pure_callback` sentinel and require
     the callback-primitive scanner to reject it while accepting the clean
     transition.
7. `process_isolation`
   - Require two fresh network-disabled worker projections to equal the parent
     projection and each other byte-for-byte.

## Invariants

- The positive fixture has static full-batch shapes and exercises exactly one
  no-restart transition. Restart control flow, partial tails, chunked shapes,
  warmup, and extra RNG draws are outside its claim.
- The protected public Objective may synchronize inside its logger before the
  optimizer's explicit ready barrier. The source inventory reports those
  sites; the runtime trace never calls the outer barrier the sole device
  boundary.
- Callback-enabled execution is an instrumented path. It cannot support a
  packaged-default timing comparison, even though its callback retains device
  references until the explicit post-run projection.
- Weak-scalar promotion, dtype, shape, NaN bit patterns, and conditional RNG
  advancement are not normalized away. Any typed mismatch fails the frozen
  rule.
- The fixture emits only labels, counts, Booleans, versions, and hashes. It
  does not retain vectors, gradients, aux values, moments, raw RNG material,
  paths, private state, or measured wall time.
- The protected submission tree, submitted ZIP, and protected manifest remain
  unchanged.
- The controller's pinned registry digest, five approved source hashes, global
  lease, private state/events, output cap, timeout, CPU pin, credential scrub,
  network block, immutable result, and SHA sidecar remain mandatory.

## Stopping and decision rule

Run the complete seven-case contract exactly once. There is no seed selection,
dtype or shape change, timing-token change, tolerance relaxation, case removal,
retry, or top-up.

- If every case passes, record
  `synthetic_normal_path_jax_boundary_equivalent` and advance only to a new
  learning-contract question. Do not integrate the transition into the
  submission.
- If any case fails, is malformed, times out, is nondeterministic, or encounters
  source drift, record or preserve the failure, park this research lane, and
  request owner review. Do not alter the fixture and rerun it.

## Claim boundary

The strongest permitted positive claim is:

> On the frozen JAX/JAXlib CPU runtime, the experiment-only pure-JAX
> no-restart transition exactly reproduced the frozen synthetic batch's typed
> public observations, protected Adam outputs, progress state, incumbent
> selection, and telemetry while making its explicit lowering boundary
> inspectable.

This cannot establish lower runtime, fewer backend compilations or kernels,
accelerator equivalence, real UIFO equivalence, restart or partial-tail
equivalence, a useful learned controller, predictive feasibility, improved
competition score, or a reason to change the candidate. It does not authorize
submission edits, native integration, official-data training, accelerator
benchmarking, paid compute, or portal action.
