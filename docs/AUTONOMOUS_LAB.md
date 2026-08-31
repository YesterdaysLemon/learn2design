# Autonomous local research laboratory

Status: active protocol for unpaid Round-2 mechanism research.

The laboratory advances one small falsifiable checkpoint per cycle. It is a
research workflow around the immutable Round-1 baseline, not permission to
revisit closed evidence, spend money, or upload a candidate.

## Round-2 owner decision

On 2026-08-30 the owner supplied the sanitized Round-1 aggregate result,
authorized a local Round-2 research pivot, and changed the heartbeat cadence
from four hours to two. On 2026-08-31 the first constraint-progress launch
parked on a pre-result infrastructure exception and the owner authorized
forensic recovery. The failed V1 ID is policy-quarantined and must never be
retried. Its one-shot standard-library-only
[`startup-forensics diagnostic`](../research/2026-08-31-constraint-progress-startup-forensics-v1-results.md)
passed, and the controller now refuses V1 before output validation, repository
inspection, lease acquisition, private state access, or worker launch. The
active gate is a fresh V2 plan preserving the scientific contract and changing
only the process boundary justified by that diagnostic. The private controller
remains parked until that plan and an owner-authorized atomic resume are ready.
The frozen online-SARSA V4 plan is paused intact and must not be implemented or
invoked while this gate is active.

The decision permits outcome-blind synthetic work and, only after its frozen
gate passes, an experiment-owned candidate adapter. It does not permit edits to
`submission/`, use of official data or outcome-open private panels, paid or GPU
work, opening the untouched promotion panel, building over the retained ZIP,
merging a PR, or interacting with the competition portal.

## Constitution

Every cycle must preserve all of these boundaries:

- use local CPU work only; do not provision or invoke paid compute, cloud GPU
  APIs, endpoints, pods, volumes, or rentals;
- do not upload, replace, or modify the official submission or interact with
  the submission portal;
- do not change the packaged optimizer defaults without a separate owner
  decision;
- do not rerun or top up `coverage-triage-screen-v1`, launch either closed
  coverage follow-up, or select a rescue rule against the observed Stage-A
  topologies;
- do not use the official archive, private topology strings, raw candidate
  arrays, or outcome-open panels to choose a mechanism;
- keep generated results outside Git and commit only code, frozen plans,
  tests, and sanitized aggregate conclusions;
- never merge its own pull request.

Synthetic analytic fixtures, checked-in public code, public literature, and
sanitized aggregate records are in scope. Public research happens outside the
fixture worker; the worker itself has network access disabled. CPU/JAX
diagnostics are software mechanics evidence only and cannot support a
competition-performance claim.

## Controller and private state

A Codex desktop heartbeat supplies the recurring cadence. Each heartbeat may
advance at most one bounded question; the repository executor is deliberately
single-shot so a bug cannot create an unbounded local loop. There is no catch-up
after a missed heartbeat and no overlapping run.

The sibling `learn2design-local-lab` directory is the private control plane:

- `lab-state.json` records the active cycle, completed study IDs, failure
  streak, and parked/idle state;
- `lab-events.jsonl` is the append-only start/heartbeat/terminal event ledger;
- `stop.request.json`, when owner-created, prevents every new cycle;
- `lab.lock/lease.json` is the identity-bound heartbeat lease and is never
  recovered automatically; and
- `cycles/<cycle-id>/` holds immutable result JSON and its `.sha256` sidecar.

[`experiments/local_lab/studies.json`](../experiments/local_lab/studies.json) is
the checked-in continuation queue. Each versioned entry freezes the expected
source hashes, fixture identity, complete case set, required result fields, and
success/failure actions. The controller accepts only a registry whose digest is
pinned in its source, then authenticates the full result before recording a
terminal decision. When the queue is exhausted, state becomes
`awaiting_study`: a later heartbeat may research, implement, test, and commit a
new outcome-blind analytic entry before running it. If no admissible question
exists, it performs a read-only status check and waits.

A failed rule, exception, timeout, source drift, malformed result, or stale
lease parks the laboratory for owner review. A terminal study ID cannot be run
again. Scheduled work never clears a stop, park, or lock.

## One-cycle protocol

1. Read `AGENTS.md`, `README.md`, `docs/CURRENT_HANDOFF.md`, this protocol, and
   the latest dated record for the active mechanism.
2. Inspect Git status, the current branch, open laboratory pull requests, and
   only the private lab state/event ledgers described above. Do not inspect raw
   private histories or topology evidence. If the worktree contains
   unrecognized changes or another cycle is active, make no edits and report
   the collision.
3. Select exactly one bounded question from the current handoff. State the
   fixture, invariant, stopping rule, and failure action before observing its
   result. Never relax a failed rule in the same evidence lane.
4. Work on `codex/autonomous-local-lab` or a focused `codex/lab-*` branch.
   Preserve unrelated user work and the submission payload.
5. Run only the focused CPU fixture first. The executor kills the worker tree at
   60 minutes and permits one full repository verification pass per cycle.
6. Write machine output beneath the sibling `learn2design-local-lab` private
   root, never under the checkout. Persist its immutable hash sidecar and keep
   vectors, gradients, paths, credentials, and private topology identities out
   of committed summaries.
7. If the frozen rule passes, advance only to the next unpaid mechanics
   question. If it fails or produces unexplained nondeterminism, park that lane
   and investigate the failure without tuning around it.
8. Commit and push reviewable work to the laboratory branch. Create or update
   a draft pull request, but do not merge it. Update `docs/CURRENT_HANDOFF.md`
   only when the live next decision actually changes.

## Mandatory verification

For code changes, run the repository minimum checks. Build the submission only
to a fresh scratch path so the locally retained uploaded ZIP is not
overwritten:

```powershell
uv sync --frozen --group dev --group integration
uv run --frozen --group dev --group integration pytest -q
$labScratch = Join-Path $env:TEMP ('learn2design-lab-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $labScratch | Out-Null
python tools/build_submission.py `
  --output (Join-Path $labScratch 'submission.zip') `
  --manifest (Join-Path $labScratch 'submission.manifest.json')
```

Also run the focused fixture, `git diff --check`, and any artifact-integrity
test for the changed path. Verify that `submission/` is unchanged unless the
owner explicitly opened a candidate-change gate.

## Stop and ask the owner

Pause autonomous mutation and request a decision if progress would require:

- any paid service or long-running accelerator work;
- a submission, portal, deadline, or leaderboard action;
- a change to the submitted algorithm or package defaults;
- use of an outcome-open/private topology for selection;
- merging a pull request, deleting evidence, or resolving user-owned changes;
- a materially new live study, panel, cost envelope, or competition claim.

Public-leaderboard feedback is a new external observation. Record only a
sanitized owner-reported aggregate intake, then reassess the laboratory queue
without retroactively changing any frozen rule. The Round-1 intake is recorded
in the active Round-2 plan linked above.
