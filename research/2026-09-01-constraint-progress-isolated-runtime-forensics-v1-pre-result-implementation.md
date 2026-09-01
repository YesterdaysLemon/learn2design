# Constraint-progress isolated runtime forensics v1 pre-result implementation

Date: 2026-09-01

Status: committed pre-result implementation; diagnostic not invoked

Checkpoint ID: `constraint-progress-isolated-runtime-forensics-v1`

## Frozen lineage

The standalone non-scientific plan was frozen first at
`b6efe5cfaca849fdab4531fb4dcdea04823f0a2a`. The one-source probe and focused
tests were then committed at `766a4eb`, and the controller/test quarantine for
the retired `constraint-aware-progress-toy-v2` path was closed at `7026c80`.
The plan file is byte-identical between its freeze revision and this surface.

The committed SHA-256 identities at `7026c80` are:

- frozen plan:
  `39c0d4ae185fd94d30d7baa3a4239d193b5ea8bccfd1226857a16c99d5fa2e33`;
- probe source:
  `d1de704f195b87d2dcad58bd407ae3af848518def85a3b142f552199c313c057`;
- probe tests:
  `190c89409648f9b2ca60055248756f8c654774bb4b4df0092f2172d97b56d24b`;
- controller:
  `954116c8a84ef8122adf77e510cd200eb892bebb1aebcd86ecd81831b6c4b028`;
- retired-V2 controller tests:
  `6c87f3b27c279e50bb9f076ab58e7113a607cd6e2673ef1884e0e9c72bace697`;
  and
- generic controller tests:
  `8b6e5dd54ceb31a9ca9d1f71817291aa50c5b5990700d60068a05d473706bcac`.

## Implemented boundary

The new import-inert source has only `os` and `sys` as top-level runtime
imports. Its parent launches exactly the nine frozen cumulative cases twice
through the committed source under `sys.executable -S -P`, file-backed output,
the scrubbed environment, a fresh process group, the 60-second timeout, and the
16,384-byte cap. It closes network entry points before NumPy import, validates
the frozen runtime identities, canonicalizes closed receipts, and removes its
fresh temporary result and sidecar after independent verification.

The verifier now recomputes observation byte counts, environment-key, site,
runtime-identity, run-body, receipt-root, plan, source, and sidecar commitments.
It requires the frozen plan revision to be an ancestor of the probe revision
and requires the plan blob at `HEAD` to equal the blob at the plan revision.
Neither a dirty worktree nor a silently amended plan can produce an accepted
diagnostic result.

The public controller now rejects both failed constraint-progress study IDs
before output validation or worker dispatch. Its historical V2 resume CLI is
also unreachable. Registry history remains intact; no result or completion
entry was fabricated.

## Verification receipt and contained deviation

From clean committed surfaces:

- Python compilation of the probe and controller passed;
- 26 focused probe/quarantine/controller checks passed before the repository
  pass; and
- after the repository pass, the four corrected quarantine cases plus the
  complete focused set passed, 28 checks total.

The one permitted full repository pass reached 100 percent and reported three
failures. Each was a stale test expecting retired V2 to remain invocable; each
instead stopped at `QuarantinedStudyError`. No scientific or unrelated test
failed. Those three tests were converted to assert refusal and non-spawn
behavior, and only the affected focused set was rerun. There was no second full
repository pass; the successor draft PR CI is the independent complete check.

That repository pass also contained one pre-existing test which directly
launched V2's runtime-only worker mode. It exited successfully, its payload was
not printed or inspected, and it did not enter scientific/full mode or the
controller. This was contrary to the retired runtime-probe boundary even
though it produced no model, metric, result, sidecar, state transition, or
score. The test was removed and replaced by a fail-closed assertion that
proves no process launch is reachable. This record preserves the deviation and
does not treat that boolean test outcome as diagnostic evidence.

## Live gate and claim boundary

No `--child` or `--run` mode of the new diagnostic has executed. It has no
result, sidecar, identified stage, action, scientific claim, candidate, or
score. After this branch is pushed and its complete draft-PR CI is green, the
only admissible next action is one invocation of the committed probe's
`--run` mode from a clean revision. It is never retried. A passing result may
authorize only a fresh V3 plan under the plan's frozen action; any malformed,
nondeterministic, timed-out, over-cap, stderr, survivor, provenance, or cleanup
condition parks this research line.

Nothing here estimates movement from `0.444293` toward `0.14`, changes the
submission, uses official or private outcomes, invokes a GPU, spends money,
packages a candidate, merges a PR, or interacts with the portal.
