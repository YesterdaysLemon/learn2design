# Anchor-lane stability mechanics result

Date: 2026-08-26

Status: `passed / anchor_lane_mechanics_confirmed`.

This is deterministic CPU software-mechanics evidence. It does not measure
UIFO performance, authorize a candidate change, reopen a closed panel, or
justify paid compute.

## Provenance

- Frozen code revision:
  `202747dc8fc17319df0b4ac70ef57174fdb4bad3`
- Protected submission tree:
  `e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`
- Private result SHA-256:
  `95c1b979e70c6bc29c25901beb528a4ce184e06b4e6add842c7285f47bbab68e`
- The immutable `.sha256` sidecar, result JSON, controller state, and event
  ledger agree on that digest and terminal status.
- The worker authenticated the CPU backend. Its network and credential policy
  probes passed before the retained cycle.
- The controller observed no repository or protected-artifact drift. Its lease
  was released and its worker-temporary directory was empty at reconciliation.

The machine-readable artifact remains outside Git beneath the sibling private
lab root. It contains hashes and scalar mechanics metadata, not parameter
vectors, gradients, topology strings, credentials, or user paths.

## Frozen case results

| Case | Result | Predeclared observation |
|---|---:|---|
| Exact twin | pass | Three `8`-member batches matched exactly; trace `cb9abc0d...` |
| Suffix invariance | pass | Lane-zero parameter/loss hashes and raw random draws matched while all seven supplied suffixes differed |
| Process isolation | pass | Two fresh-process traces matched the exact-twin digest `cb9abc0d...` |
| Forced shared-state boundary | pass | Different incumbents (members 1 and 2), exactly eight boundary restarts, first lane-zero difference only at batch 3 |
| Exceptional arithmetic and partial tail | pass | `8,8,2` accounting; 22/27 nonfinite gradient values sanitized; 7/7 finite-infeasible observations; zero sanitizer or partial-state violations |
| Diagnostics-disabled control | pass | Objective trace matched with scalar/hash telemetry enabled or disabled |

All six cases passed without tolerance relaxation, case removal, seed change,
or rerun. The in-process and fresh-process exact-twin digests were identical.

## Decision

The narrow invariant is confirmed: before an explicit shared incumbent/restart
boundary, an identical lane-zero start follows the same deterministic path and
is unaffected by unrelated suffix-population values. At the forced boundary,
the lane can diverge exactly when the selected shared incumbent differs.

This supports moving to one new analytic question: whether the current restart
clock can be kept alive by improving finite-but-infeasible loss even though no
feasible candidate improves. That question must receive a new versioned study
ID, frozen counterexamples, complete result contract, and failure action before
execution. The submitted patience-600/no-prior random-start candidate remains
unchanged.

The private controller is now `awaiting_study` with failure streak zero. It
will not rerun this terminal study.
