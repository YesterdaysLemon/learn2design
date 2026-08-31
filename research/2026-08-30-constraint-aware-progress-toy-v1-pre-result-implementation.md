# Constraint-aware progress toy v1 pre-result implementation

Date: 2026-08-30

Status: exact implementation and hostile pre-result audit complete; no study
result exists.

## Frozen boundary

The study contract remains the plan frozen at commit
`02c3e2329b4906aa49d80ea0256a7db9774d491c`. The implementation checkpoint
did not change a family, world, seed, transcript, arm, order, patience value,
metric, threshold, case, stopping action, or claim boundary. The protected
submission tree remains
`e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`.

The implementation was committed separately at
`269698a3974cc12f9871e0e8a3580fbc7230cce9`. It adds only the deterministic
synthetic fixture, its dedicated isolated worker, the exact twelve-case
registry contract, ID-specific controller validation, and focused regression
tests. The five approved source digests and normalized registry digest are
pinned before any result-bearing execution.

## Implemented mechanics

- The complete 48-world synthetic family, four committed transcripts, five
  arms, forward/reverse reproduction twins, exact reference calculation, and
  bounded world aggregates are independently replayed.
- The optimizer receives only the frozen packet. Its progress comparator,
  decision-tuple adapter, and restart enablement are the only arm differences.
- Restart masks are prepared before transcript access. The provider exposes
  only committed hashes, seals the exact batch receipt, and releases current
  draw rows only after sealing.
- The worker oracle independently reconstructs evaluations, progress,
  synchronous Adam transitions, incumbent selection, restarts, trajectories,
  phase receipts, and roots.
- Development and held-out phases use separate source objects and fresh child
  processes with connected exploding sentinels. The controller independently
  reconstructs family, intervention, source, attack, and numeric gate
  relations from the sanitized projection.
- Every malformed attack now begins with an independently reconstructed,
  semantically valid canonical/world-0 batch-0 envelope. The harness owns the
  injected object, instruments entry into the named production consumer, and
  hashes the same typed internal state before and after consumption.
- Nested children block before runtime loading until the parent assigns them
  to a kill-on-close private Windows Job and verifies membership. Watchdogs
  cover spawn and bounded input transport; stdout, stderr, and survivor
  receipts include both projections and all phase children.

The guarded runtime is the separately probed system CPython `3.13.14` x64 with
NumPy `2.5.1`; its interpreter, package, `PCG64`, and `SeedSequence` identities
are hash-pinned and fail closed. The repository's uv environment uses NumPy
`2.5.2` for development tests and intentionally fails the guarded runtime
identity check. A non-result nested runtime probe returned 706 stdout bytes,
zero stderr bytes, and zero surviving children.

## Hostile audit disposition

Three independent Luna/max read-only audits covered:

1. family, chronology, restart sealing, and attack-envelope semantics;
2. schema, evidence roots, source isolation, malformed inputs, and controller
   recomputation; and
3. runtime identity, nested Windows Job containment, watchdogs, output caps,
   child receipts, and the five-source boundary.

The audits initially blocked on real issues: draw release occurred too early;
malformed-input reach and mutation receipts were asserted rather than
observed; impossible-family means did not authenticate every seed gap; child
input could precede timeout accounting; nested receipts were incomplete; and
the first attack envelope was schema-valid but not mathematically valid. Each
issue was repaired before the source hashes were refreshed. All three final
audits passed with no remaining substantive blocker.

## Verification

- `python -m py_compile` passed for the fixture, worker, controller, and focused
  tests.
- `uv sync --frozen --group dev --group integration` passed.
- `uv run --frozen --group dev --group integration pytest -q tests/test_constraint_aware_progress_toy.py`
  passed all nine focused tests.
- `uv run --frozen --group dev --group integration pytest -q tests/test_local_lab.py`
  passed the complete controller/local-lab suite.
- The single allowed full repository run,
  `uv run --frozen --group dev --group integration pytest -q`, completed at
  100 percent with no failures and two existing skips.
- `git diff --check` reported only the repository's expected working-tree
  line-ending warnings. No submission build was run because this checkpoint
  neither changes nor authorizes rebuilding the protected artifact.

## Controller state and next gate

The private controller remains `awaiting_study`, with no active cycle, no
owner stop marker, no lease, failure streak zero, and exactly the ten earlier
terminal studies in its completed ledger. `constraint-aware-progress-toy-v1`
has never been invoked and has no private result, sidecar, state transition,
terminal event, or evidentiary claim.

After this record and the handoff are committed, pushed, and green in the
stacked draft PR, the only live next action is at most one guarded local-CPU
controller invocation of this exact study. Never run the fixture or worker
directly and never retry the ID. A pass may support only the exact synthetic
constraint-aware progress/restart harness and a later experiment-owned
candidate audit. It cannot establish hidden-topology generalization, candidate
value, competition improvement, accelerator value, or score.
