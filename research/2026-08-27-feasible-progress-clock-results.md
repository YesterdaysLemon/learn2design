# Finite-infeasible restart-clock mechanics — validated result

Date: 2026-08-27

Study ID: `feasible-progress-clock-v1`

Study revision: `e750feed4b94b76e066c0eef446d38e6aaec007d`

Private immutable result SHA-256:
`d256572c0b67ff107ffca253f25e720175806912b3fe94c87f54478a6c98956e`

## Decision

All seven frozen deterministic CPU cases passed. The authenticated action is:

```text
finite_infeasible_progress_resets_clock_confirmed
```

This confirms a software mechanism in the unchanged submitted optimizer. It
does not establish an optimizer improvement, a competition-performance effect,
or a reason to alter the submitted patience-600/no-prior random-start package.

## Frozen-case results

- `finite_infeasible_descent`: all 32 observations were finite and infeasible;
  all 32 reset their member progress clock, every member's stall path was
  `[0, 0, 0, 0]`, and no restart or global feasible improvement occurred.
- `finite_infeasible_plateau_control`: every member followed
  `[0, 1, 2, 0]`, and all eight restarted exactly on batch 2 with fresh kind 0
  and restart round 0. Parameter hashes changed for all restarted members; Adam
  age, generation, and next-batch stall state all satisfied the frozen reset
  invariants.
- `finite_infeasible_improve_then_plateau`: a new infeasible improvement on
  batch 1 reset the boundary. Every member followed `[0, 0, 1, 2, 0]` and
  restarted exactly on batch 3, again satisfying every frozen state-reset
  invariant.
- `late_feasibility_crossing`: descending finite loss held every clock at zero
  through batch 3. No restart occurred, and the sole global feasible
  improvement was the declared member-0 crossing on batch 3.
- `mixed_member_clock`: even members continued descending with zero stalls;
  exactly plateau members 1, 3, 5, and 7 restarted on batch 2. Exactly those
  four parameter hashes and restart states changed.
- `diagnostics_disabled_control`: telemetry-off and telemetry-on objective
  projections were identical.
- `process_isolation`: the two fresh credential-scrubbed, network-disabled CPU
  traces were byte-identical after timing fields were excluded.

The worker reported zero nonfinite derivative values and zero sanitizer
violations. The result JSON and its SHA-256 sidecar matched. The controller
returned to `awaiting_study` with failure streak zero and released its lease.

## Repository verification

The focused local-lab suite passed after correcting one test's expected
sanitizer error message; no frozen fixture, invariant, threshold, source pin,
or decision action changed. The single full `dev` plus `integration` repository
test pass then completed successfully with two expected skips.

A clean scratch build produced current instrumented-source archive SHA-256
`4b7384dd6d401918b9b46ace4e65c3f116c34feeeca825ae25745dfb9e7908bd`,
matching the repository's pinned builder test. This is distinct from the
protected owner-uploaded ZIP SHA-256
`4cc0dbc65a3e61ca5358c18655c432caf478fbdfc07f10512553781f8822924b`.
The protected artifact and `submission/` tree were not edited or overwritten.

## Interpretation

The current optimizer has two distinct progress predicates. A global incumbent
requires a finite feasible observation, while a member clock resets on any
finite raw-loss decrease. The study directly confirms that a member can
therefore avoid restarting across a bounded sequence with no feasible progress.
The plateau and delayed-plateau controls show that the restart machinery itself
still fires at the exact configured boundary when raw improvements stop.

The late-crossing control is equally important: finite-infeasible descent can
precede a feasible observation. Replacing the current predicate with a naive
"feasible only" reset would therefore encode a real exploration tradeoff, not
a free correctness repair.

## Claim boundary and next analytic question

No topology, official data, private trajectory, candidate comparison, GPU,
network service, or paid endpoint was used. Raw loss and parameter vectors were
not retained in the result; only typed hashes and scalar/list mechanics
summaries crossed the worker boundary.

The next useful unpaid question is an online-information boundary: from a
finite prefix of strictly improving but infeasible observations, can any
deterministic restart rule both guarantee a bounded restart for a path that
stays infeasible forever and preserve every path that becomes feasible one
batch later? The two paths share the same observable prefix, suggesting an
indistinguishability obstruction. Freeze that statement, finite cases, and a
hostile reference check before drawing any policy conclusion. Do not implement
a submission treatment or select a tenure threshold from this result.
