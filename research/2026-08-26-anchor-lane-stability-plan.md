# Frozen local study: anchor-lane stability

Date: 2026-08-26

Status: frozen before execution. This is an unpaid CPU software-mechanics
study. It does not authorize a submission change, topology evaluation, or paid
compute.

## Question

When lane zero starts from identical parameter bytes, is it exactly repeatable
and invariant to unrelated suffix-population construction until the optimizer
records an explicit shared-state boundary?

The completed H100 postmortem found identical anchor hashes but later anchor
loss divergence in every treatment pair. That outcome-open observation only
motivates this synthetic study; it cannot validate the answer.

## Frozen implementation and fixtures

Use the unchanged `BatchedRestartAdam` on the local CPU backend with the
submitted population of eight, a fixed seed, and deterministic analytic
objectives. Retained artifacts contain only hashes and scalar event metadata.

1. **Exact twin:** run the same smooth separable fixture twice in one process.
   Initial populations, every parameter/loss hash, feasibility bit, and stable
   telemetry field must match exactly.
2. **Suffix invariance:** hold lane-zero bytes, seed, and random-stream
   consumption fixed while changing the other seven initial lanes. The raw
   pre-transform random draw must also match. With patience beyond the fixture
   horizon, every lane-zero parameter and loss hash must remain bitwise
   identical.
3. **Process isolation:** run the exact twin in two fresh Python processes on
   CPU. Their sanitized traces must match exactly.
4. **Forced shared-state boundary:** use a stop-gradient objective whose best
   feasible suffix member differs between variants. Force the first restart at
   the predeclared patience boundary. All eight members must restart exactly
   once there, with the predeclared alternating exploit/fresh kinds and no
   additional event. Lane zero must match through the batch that records its
   exploit restart and may first differ only on the following evaluation, with
   telemetry identifying the suffix incumbent and complete restart set.
5. **Exceptional arithmetic and partial tail:** exercise finite-infeasible
   losses, finite losses with NaN gradients, and a final partial population.
   Sanitization must keep lane zero invariant, global improvements must always
   be finite-feasible, sanitizer scalar outputs must be finite and bounded, no
   restart may occur, the tail must preserve Adam age, stall count, generation,
   and restart state, and accounting must stop at the exact evaluation cap.
6. **Diagnostics-disabled control:** enabling scalar/hash telemetry must not
   change the parameter or loss path when time is not a stopping variable.

No UIFO topology, official archive record, prior generated panel, private
history, or wall-clock performance quantity is permitted.

## Frozen decision rule

Status is `passed` only if all six fixtures pass exactly. There is no tolerance
relaxation, case removal, seed replacement, or fixture alteration after a
failure. Any unexplained divergence yields `failed / park_initializer_and_restart_research`.

Passing yields only `anchor_lane_mechanics_confirmed`. It permits the
laboratory to examine the already-parked feasible-progress restart hypothesis
on new analytic counterexamples. It does not justify changing the candidate or
renting a GPU.

## Evidence handling

Run from a clean committed revision with `JAX_PLATFORMS=cpu`. Write the
machine-readable result beneath the sibling `learn2design-local-lab` private
root, retain its immutable `.sha256` sidecar, record that SHA-256 in the
sanitized result report, and keep the private root outside Git.
