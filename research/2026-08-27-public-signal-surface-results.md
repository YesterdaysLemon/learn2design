# Public signal surface - validated result

Date: 2026-08-27

Study ID: `public-signal-surface-v1`

Study revision: `30db48ff717e2435ed9cb567c13119bb8139fa5f`

Private immutable result SHA-256:
`1548c94a5b46a0fca3054d252f8a96d38717c881b22f05036c868d6409d905cc`

## Decision

All eight frozen deterministic CPU cases passed. The authenticated action is:

```text
public_current_constraint_signals_confirmed
```

The frozen `dfbench==0.3.3` UIFO auxiliary contract exposes non-Boolean
current constraint diagnostics beyond `is_feasible`, and the protected
population helper preserved the complete synthetic auxiliary pytree in every
tested evaluation mode. This does not make any field a certificate of future
feasibility and does not justify changing the retained submission.

## Frozen-case results

- `dependency_source_identity`: the installed public dependency reported
  version `0.3.3`; the checked-in lock retained wheel SHA-256
  `1f96d75b813ea42f93992da5c1f50d6a4f59dd7a507bcf561676b0e416378c43`;
  and the exact `Objective`, optical-base, and UIFO public source hashes all
  matched the frozen plan.
- `uifo_aux_schema`: authenticated static source analysis found exact top-level
  fields `is_feasible`, `penalty`, `power_values`, `sensitivity_loss`, and
  `violations`, with nested power fields `detector`, `hard`, and `soft`. The
  UIFO aux objective uses the shared `_build_aux` builder, and its total loss
  is `sensitivity_loss + penalty`.
- `scalar_batch_roundtrip`: the actual public `Objective` wrapper returned all
  seven auxiliary leaves for the four frozen synthetic points. Scalar versus
  batched maximum absolute discrepancies were exactly zero for loss, gradient,
  and every auxiliary leaf.
- `candidate_passthrough_modes`: the protected
  `BatchedRestartAdam._evaluate_population` helper returned identical full
  projections and leaf paths in full-batch mode and chunk sizes 1, 2, and 4.
  The mode-projection SHA-256 was
  `bf5eed27688316f7ce5512105f9de4e8e684680ae49dced04300faa1a9712b82`.
- `infeasible_magnitude_control`: both declared points returned
  `is_feasible=False`, while their penalty, violation arrays, and raw power
  leaves differed. The declared feasible control returned `True`. This proves
  only that richer current diagnostics are visible.
- `no_aux_negative_control`: an otherwise valid generic synthetic
  `ContinuousProblem` without `objective_function_aux` raised the expected
  public `RuntimeError`. Rich auxiliary data is a UIFO/problem-specific
  contract, not a universal `Objective` guarantee.
- `consumer_boundary`: static analysis confirmed that the current optimizer
  receives the full aux pytree through `value_and_grad_aux` and
  `vmap_value_and_grad_aux`, but directly subscripts only
  `aux["is_feasible"]`. The other six visible leaves are currently unused by
  its decision path.
- `process_isolation`: two fresh credential-scrubbed, network-disabled CPU
  projections were byte-identical to each other and the parent projection.
  Their timing-free trace SHA-256 was
  `e4dbabf8b5fe408b85e1e7643ab116046bbe61b75ce479bf8552f2ec0d3ccce3`.

The worker used JAX `0.9.0.1` on the CPU backend under Python `3.12.13`, wrote
no stderr, and emitted only the frozen schema metadata, Booleans, bounded
discrepancies, counts, and hashes. The result JSON matched its SHA-256 sidecar.
The controller returned to `awaiting_study` with failure streak zero and
released its lease.

## Interpretation

The earlier scalar/Boolean obstruction did not characterize the actual public
evaluation path. UIFO supplies current sensitivity loss, the active penalty,
per-constraint penalty values, and raw power groups in addition to the exact
feasibility predicate. The population helper transports all of them, even
though the current optimizer uses only the Boolean.

The distinction between a diagnostic and a certificate is essential. Penalty
and `violations` depend on the active penalty function. Raw power values and
public thresholds identify the current constraint state, but this study proves
no monotonic relationship to a future optimizer step and no guarantee that an
improving or worsening value will precede a feasible crossing. The result
therefore closes an interface-inventory question, not a restart-policy design.

## Claim boundary

The observation scope is the current protected optimizer's normal logged
evaluation path: its candidate parameters, returned loss, returned total-loss
gradient, returned aux pytree, deterministic optimizer-owned state, RNG,
budget counters, and incumbent state. This checkpoint directly tested the
public aux schema and scalar/batched/chunk transport only.

It does not authorize unlogged objective callables, extra evaluations,
Hessians, manual logging, private attributes, saved histories, problem
metadata, structural identifiers, or a broader observation adapter. It does
not establish predictive sufficiency, monotonicity, a safe restart criterion,
candidate quality, official-data behavior, or leaderboard performance.

No UIFO instance, topology, official data, private trajectory, generated
panel, candidate comparison, GPU, network service, or paid endpoint was used.
The protected submission tree, owner-uploaded ZIP, and protected manifest
remained unchanged.

## Repository verification

The three focused signal-surface checks and both affected controller
queue-transition checks passed before the retained terminal controller
invocation. `uv sync --frozen --group dev --group integration` completed, and
the single full `dev` plus `integration` repository test pass then succeeded
with two expected skips.

A fresh private scratch build produced current instrumented-source archive
SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`
and manifest SHA-256
`1f4f7e14c77b35e8dde4d68895289e5bf1f0c478ea297bd34a201ae36be97b97`.
The archive is distinct from the protected owner-uploaded ZIP SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
The protected manifest remained
`99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`,
and the protected submission tree remained
`e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`. Neither protected artifact was
overwritten.

## Next analytic question

The richer diagnostics invite one narrow follow-up, not a treatment. Freeze an
explicit adapter limited to the current normal evaluation path, then ask
whether two deterministic synthetic paths can remain identical through a
finite bound across the **complete allowed snapshot**: candidate parameters,
loss, total gradient, every aux leaf, deterministic optimizer state, RNG
transcript, budget counters, and incumbent state, while differing at the next
feasibility observation.

The fixture must include negative controls in which perturbing the gradient,
one aux leaf, metadata boundary, budget/RNG state, or optimizer state breaks
snapshot identity. Any unlogged probe, extra evaluation, private attribute,
structural metadata, official/private evidence, or candidate implementation is
out of model. If a full-surface twin exists, it would show only that current
diagnostics are not a universal future-feasibility certificate without added
structural assumptions; it would still say nothing about their distributional
usefulness or performance.
