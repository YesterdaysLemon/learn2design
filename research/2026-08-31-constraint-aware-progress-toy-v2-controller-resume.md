# Constraint-aware progress toy v2 controller resume

Date: 2026-08-31

Status: atomic controller recovery completed; no V2 study invocation exists.

## Authorized action

After the exact V2 implementation and hostile pre-result audit were committed
and green in draft PR #40, the owner explicitly authorized only the separate
controller resume requested by the live handoff. The authorized operation was
the dedicated controller transition from the exact parked V1 infrastructure
failure to `awaiting_study`. It did not authorize a V2 worker launch, result,
candidate change, GPU, paid compute, submission build, portal action, or merge.

## Precondition receipt

Immediately before the transition:

- the checkout was clean on
  `codex/lab-constraint-aware-progress-toy-v2-plan` at
  `c99c58dfce89fec22b4d3b0aeed1fad5b1ba3c58`;
- draft PR #40 was open and all three CI jobs were green;
- the normalized registry SHA-256 was
  `addde4630f53d0aa1e3b2bc3a7132978d54fe505e1f94b4f931795fd25983a2b`;
- all five committed V2 source hashes matched their frozen approvals;
- the protected submission tree was
  `e2b495b6f1bf9f5c8f0b36ae5bc095f6df8e7588`, and the retained ZIP and
  manifest hashes remained unchanged;
- the canonical controller state was `parked`, with no active cycle, exact
  stop reason `RuntimeError: local-lab worker emitted forbidden stderr`, and
  failure streak one;
- the completed ledger contained exactly the ten approved terminal studies,
  with neither V1 nor V2 present; and
- the owner stop marker, lease file, and lock directory were absent.

The event ledger contained 86 canonical events and had SHA-256
`a196ae00de430ac3fb9f70b1c980ab6d7c3b1110498a5bb9a4a28a634324d00c`.

## Transition and verification

The single controller operation was:

```powershell
uv run --frozen --group dev --group integration python tools/run_local_lab.py --resume-constraint-progress-v2
```

It exited zero and reported that the controller resumed without launching V2.
The post-transition readback established:

- controller status `awaiting_study`, null active cycle, and null stop reason;
- the same ten completed-study receipts and failure streak one;
- the original 86-event byte stream unchanged as the complete ledger prefix;
- exactly one appended canonical `controller_resumed` event at
  `2026-09-01T00:34:15Z`, from `parked` to `awaiting_study`, with reason
  `owner_authorized_v1_quarantine_recovery` and retired study
  `constraint-aware-progress-toy-v1`;
- no owner stop marker, lease file, or lock directory after verification; and
- no worker launch, active cycle, scientific result, result sidecar, or score.

The completed-ledger and failure-streak projection retained SHA-256
`b62eee43a62fc9767d661fd6d613d95e39c7a552da8364cd7cc7b145c13e62ed`
before and after the transition.

## Next gate and claim boundary

This recovery proves only that the frozen event-before-state controller resume
completed atomically at the approved boundary. It is not scientific evidence
for the constraint-aware progress rule.

The next checkpoint may perform at most one separately owner-authorized guarded
local-CPU invocation of `constraint-aware-progress-toy-v2`, after fresh clean
revision, green CI, source, registry, state, stop, and lease checks. The fixture
and worker must never be run directly, and V1 must never be retried. A terminal
V2 failure or malformed result parks the controller without retry.

No current evidence estimates movement from the owner-reported Round-1 score
`0.444293` toward `0.14`. Candidate integration, official or private topology
evaluation, accelerator work, paid compute, packaging, upload, portal action,
and merge remain separate owner gates.
