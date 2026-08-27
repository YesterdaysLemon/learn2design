# Public signal-surface study plan

Status: frozen before execution
Study ID: `public-signal-surface-v1`
Execution class: unpaid, deterministic local CPU mechanics only

## Question

At the frozen `dfbench==0.3.3` public evaluation boundary used by the current
optimizer, does UIFO expose deterministic current-state constraint diagnostics
beyond the Boolean `is_feasible` flag, and does the checked-in population
evaluation helper preserve that complete auxiliary pytree in its full-batch and
chunked modes?

This is an interface and transport question. It does **not** ask whether any
diagnostic predicts a future feasibility crossing, whether it is a valid
restart certificate, or whether consuming it would improve competition
performance.

The observation scope is exactly the current `BatchedRestartAdam` evaluation
path: candidate parameters already owned by the optimizer, returned loss,
returned gradient, returned aux pytree, deterministic optimizer-owned state,
and normal logged budget counters. It is not a general adapter or allowlist for
future restart rules. Public unlogged callables, extra evaluations, Hessians,
manual logging, metadata, problem specifications, structural identifiers,
saved histories, and private attributes are outside this study and must not be
treated as admissible signals on the strength of its result.

## Frozen inputs and identity

- Dependency lock: checked-in `uv.lock`, including `dfbench==0.3.3`.
- Locked `dfbench` wheel SHA-256:
  `1f96d75b813ea42f93992da5c1f50d6a4f59dd7a507bcf561676b0e416378c43`.
- Public dependency source SHA-256 values:
  - `dfbench/core/objective.py`:
    `9e2c2bb54517f59efacf4c2a59908ffd55c7fb2e15089d53263ece796e71daa2`
  - `dfbench/problems/base_problem.py`:
    `e7768eb3afd061b2684dbdb761e4211c9a9709852a54b45fff17d21a851ee95d`
  - `dfbench/problems/uifo/uifo_problem.py`:
    `a6e3e95275987799761831a64a2c7b0aa793656d741df4d5d5e78b64c13f7d08`
- Protected optimizer source SHA-256:
  `34ba5a1403d22a8f9861851c2ddfb77a6ed57cc33554249f38bb9bf7b6bc1176`.
- A three-parameter deterministic synthetic `ContinuousProblem` using the
  actual public `Objective` wrapper. It emits the UIFO-shaped auxiliary fields
  without importing or instantiating any official problem instance.
- Frozen bounded points, in order:
  `[0, 0, 0]`, `[1, 0, 0]`, `[0, -1, 1]`, and `[-0.5, 0.1, -0.25]`.
- Candidate-helper modes: full batch and chunk sizes 1, 2, and 4.
- CPU only, Python hash seed zero, two fresh isolated worker projections for
  the determinism case.

No official dataset, generated panel, private evidence, outcome-bearing
artifact, provider, network service, or submitted package is an input.

## Complete frozen cases

1. `dependency_source_identity`
   - Require version `0.3.3`, the exact locked wheel digest, and exact hashes
     for the three public source files above.
2. `uifo_aux_schema`
   - Parse the authenticated public sources, without creating UIFO, and require
     exact top-level fields `is_feasible`, `penalty`, `power_values`,
     `sensitivity_loss`, and `violations`.
   - Require exact nested power fields `detector`, `hard`, and `soft`.
   - Require the UIFO auxiliary objective to use the shared `_build_aux`
     builder and the scalar total to be `sensitivity_loss + penalty`.
3. `scalar_batch_roundtrip`
   - Evaluate the four frozen points with fresh public `Objective` instances
     through scalar and batched aux APIs.
   - Require zero loss, gradient, and auxiliary-leaf discrepancy.
4. `candidate_passthrough_modes`
   - Evaluate the same points with the protected
     `BatchedRestartAdam._evaluate_population` in every frozen mode.
   - Require identical loss, gradient, and complete auxiliary projections in
     all modes.
5. `infeasible_magnitude_control`
   - Require the two frozen infeasible points to share `is_feasible=False` but
     differ in violations, penalty, and raw power leaves; require the feasible
     control to be present.
6. `no_aux_negative_control`
   - Wrap a deterministic synthetic problem that exposes no aux objective.
   - Require explicit aux evaluation to raise the public `RuntimeError`,
     demonstrating that rich aux is a problem-specific contract rather than a
     universal `Objective` guarantee.
7. `consumer_boundary`
   - Parse the protected optimizer source and require the current optimizer
     decision path to subscript only `aux["is_feasible"]`, while the population
     helper remains a pass-through for the complete pytree.
8. `process_isolation`
   - Require two fresh credential-scrubbed, network-disabled CPU worker
     projections to be byte-identical to each other and the parent projection.

## Invariants

- The installed public sources must match every frozen source digest.
- Runtime probes use only the synthetic problem and the checked-in candidate
  helper; UIFO is never instantiated.
- The actual public `Objective` is created with parameter/time-history saving
  disabled. No checkpoint, output, unlogged evaluation callable, metadata,
  Hessian, manual logging, or extra probe API is invoked.
- Scalar, batched, and chunked paths use the same ordered points and no
  tolerance: discrepancies must be exactly zero after host conversion.
- Every result is aggregate schema metadata, Boolean state, counts, bounded
  discrepancies, or SHA-256. No parameter vectors, raw derivatives, paths,
  private identifiers, or timing evidence may be retained.
- The protected submission tree and submitted ZIP remain byte-identical.
- The controller's source manifest, registry digest, global lease, private
  state/events, output cap, timeout, CPU pin, environment scrub, network block,
  immutable result, and SHA sidecar remain mandatory.

## Stopping and decision rule

Run the complete eight-case contract exactly once. There is no seed choice,
tolerance tuning, case removal, source substitution, retry, or top-up.

- If every case passes, record
  `public_current_constraint_signals_confirmed` and advance only to a new
  mechanism question about whether any such current-state signal can support a
  separately frozen progress invariant.
- If any case fails, is malformed, times out, is nondeterministic, or encounters
  drift, record or preserve the failure, park this research lane, and request
  owner review. Do not relax the rule or rerun the study.

## Claim boundary

The strongest permitted positive claim is:

> Under the authenticated `dfbench==0.3.3` public source and the protected
> checked-in population helper, UIFO's public per-evaluation aux contract
> contains deterministic non-Boolean current constraint diagnostics beyond
> `is_feasible`, and the helper preserves that pytree in the frozen synthetic
> transport probes.

This study cannot establish predictive sufficiency, monotonicity, a safe
restart rule, candidate quality, official-data behavior, runtime quality, or
leaderboard performance. It does not authorize an optimizer change or any paid
study.
