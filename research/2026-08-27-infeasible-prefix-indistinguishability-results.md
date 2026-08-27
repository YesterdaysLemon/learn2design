# Infeasible-prefix indistinguishability - validated result

Date: 2026-08-27

Study ID: `infeasible-prefix-indistinguishability-v1`

Study revision: `b20b6f80ef6825093f657bae43fd180cc656370a`

Private immutable result SHA-256:
`8aaf61bbcf21ea14e938f99f63f1c6e93f31b8d44307c79c9215ef84208b4ee5`

## Decision

All six frozen deterministic CPU cases passed. The authenticated action is:

```text
synthetic_identical_prefix_obstruction_confirmed
```

This confirms a narrow analytic obstruction for an abstract deterministic
one-lane restart rule. It does not establish that all restart policies are
impossible, evaluate the submitted optimizer, or justify changing the retained
patience-600/no-prior random-start package.

## Frozen-case results

- `shared_prefix_identity`: the forever-infeasible path and the path first
  becoming feasible at observation 14 were exactly identical through the
  frozen bound `B = 13`. Their shared prefix SHA-256 was
  `50d9f3ce3637e0c6b10b27ae5c2851269d2e0c4fc853c527b59fbcd9b5219dc1`;
  every integer loss strictly improved, the deterministic rule inputs matched,
  and the next observations differed.
- `action_vector_exhaustion`: all `2^13 = 8192` restart/no-restart action
  vectors on the shared prefix were enumerated. Zero simultaneously restarted
  the forever path by `B` and preserved the late-crossing path through `B`.
- `witness_partition`: exactly 8,191 vectors satisfied bounded restart but not
  late-crossing preservation; the one all-no-restart vector preserved the
  crossing but failed the bound. There were zero joint or unclassified vectors.
- `boundary_sweep`: the same exact partition held at bounds
  `[1, 2, 3, 5, 8, 13]`, with action-vector counts
  `[2, 4, 8, 32, 256, 8192]` and zero joint satisfiers at every bound.
- `extra_signal_positive_control`: after a Boolean certificate deliberately
  broke prefix identity at `B`, the declared deterministic rule restarted the
  forever path by the bound and preserved the late crossing. This confirms
  that the negative result depends on identical policy-visible information.
- `process_isolation`: two fresh credential-scrubbed, network-disabled CPU
  projections were byte-identical. Their timing-free trace SHA-256 was
  `7e46ddab5d83e51761503c162c8aa68762e8354bbe08353c848f4766dc3e8383`.

The worker used JAX `0.9.0.1` on the CPU backend under Python `3.12.13`, wrote
no stderr, and emitted only the frozen scalar, Boolean, list, and hash fields.
The result JSON matched its SHA-256 sidecar. The controller returned to
`awaiting_study` with failure streak zero and released its lease.

## Interpretation

The analytic statement is the non-anticipating prefix lemma. Begin two copies
of a deterministic rule in the identical internal state. If they receive the
same observations through `B`, their state and actions through `B` are also
identical. Therefore, any target-lane restart forced by `B` on the
forever-infeasible path also occurs before the first feasible observation on
the indistinguishable path that crosses at `B + 1`.

The finite action-vector tables are hostile reference checks of this lemma,
not empirical proof by sampling programs. The positive control shows why an
additional observable certificate would change the conclusion: it makes the
two inputs distinguishable before the forced action.

## Claim boundary

The rule here observes exact `(loss, is_feasible)` pairs, acts after each
observation, has identical deterministic initial state on both paths, and
controls one target lane. "Preserve" means no restart or equivalent state
change touches that lane before its first feasible observation. The result
does not cover randomized probability guarantees, expected restart time,
distributional assumptions, a finite promised crossing horizon, reversible
restarts, or rules with richer pre-feasibility signals.

In particular, the submitted optimizer also has access to gradients,
parameters, optimizer state, RNG state, budget state, and a global incumbent.
This study does not prove that trajectories can remain indistinguishable under
that full signal surface and makes no claim about competition performance.

No topology, official data, private trajectory, candidate comparison, GPU,
network service, or paid endpoint was used. The protected submission tree and
owner-uploaded artifact remained unchanged.

## Repository verification

The five focused local-lab checks for the new fixture, sanitizer, exact result
contract, and second/third-study state transitions passed before the retained
controller invocation. `uv sync --frozen --group dev --group integration`
completed, and the single full `dev` plus `integration` repository test pass
then succeeded with two expected skips.

A clean scratch build produced current instrumented-source archive SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`
and manifest SHA-256
`8dbc2d81bb0aa378788c3b3fb1a7120cee2de92c4dd39a9466d945fb0615e8b3`.
This is distinct from the protected owner-uploaded ZIP SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
The protected manifest remained
`99e18299ba6dd232f203cc2e59131cea61cd83ff8b0a9bceab6c56c764ddd86a`;
neither protected artifact nor the `submission/` tree was edited or
overwritten.

## Next analytic question

Before considering another restart mechanism, inventory the complete public,
checked-in signal surface available before feasibility: objective loss,
gradient, auxiliary fields, parameters, deterministic optimizer state, RNG,
budget, and global incumbent. Determine whether the public objective contract
offers a pre-feasibility certificate beyond the Boolean flag, using only static
source analysis and synthetic fixtures.

That audit must not use the official archive or private trajectories, implement
a treatment, choose a tenure threshold, or infer candidate quality. If a
genuinely richer signal exists, it requires a separately frozen analytic study;
if it does not, record the narrower interface boundary and leave the candidate
unchanged.
