# Local mechanics laboratory

This package contains deterministic synthetic fixtures for questions about the
optimizer implementation. It must not import the official dataset, generated
competition panels, or private result archives.

The first frozen study is `anchor-lane-stability-v1`. It asks whether an
identical population lane is repeatable and independent of unrelated suffix
members until an explicit shared-incumbent restart boundary. Run it on CPU and
write its JSON outside the repository:

```powershell
$privateRoot = Join-Path (Split-Path (Get-Location) -Parent) `
  'learn2design-local-lab'
$cycleId = 'anchor-lane-stability-v1-20260826'
$output = Join-Path $privateRoot "cycles\$cycleId\result.json"
$env:JAX_PLATFORMS = 'cpu'
uv run --frozen --no-sync `
  python tools/run_local_lab.py `
  --study anchor-lane-stability-v1 `
  --output $output
```

The terminal follow-on studies are `feasible-progress-clock-v1`,
`infeasible-prefix-indistinguishability-v1`, and
`public-signal-surface-v1`, followed by
`full-surface-prefix-indistinguishability-v1` and
`normal-path-jax-boundary-v1`, `supervised-toy-signal-v1`, and
`contextual-bandit-toy-signal-v1`, then `two-step-delayed-credit-v1`. Their
sanitized conclusions live under `research/`; terminal study IDs must never be
reused even though the controller remains available for a newly frozen
registry entry.

The JAX-boundary checkpoint passed its one-batch CPU systems contract. It is
not a candidate treatment or timing benchmark. The supervised toy-signal
checkpoint then passed its frozen held-out, baseline, label-shuffle,
attribution, leakage, and process-isolation controls. It validates only the
synthetic supervised harness. The contextual-bandit checkpoint then passed its
frozen online-update, held-out, baseline, context-shuffle, attribution,
leakage, and process-isolation controls. It validates only the synthetic
immediate-reward bandit harness. The delayed-credit checkpoint then passed its
frozen terminal-return, transition-assignment, reward-origin, held-out,
attribution, leakage, and process-isolation controls. It validates only the
fixed synthetic two-step harness; production RL and candidate integration
remain outside this laboratory's authority.

The first two proposed multi-step fixtures were rejected before guarded
execution and remain quarantined; the fresh
[`multistep-td-action-prefix-v3` plan](../../research/2026-08-28-multistep-td-action-prefix-v3-plan.md)
was implemented, invoked exactly once, and passed all nineteen frozen cases.
Its narrow terminal conclusion is recorded in the
[`V3 result`](../../research/2026-08-29-multistep-td-action-prefix-v3-results.md).
The later online-SARSA plans are historical: V1 through V3 were rejected before
execution and V4 is paused intact after public Round-1 feedback changed the
priority.

The first `constraint-aware-progress-toy-v1` launch parked before any
authenticated result and is permanently quarantined. The controller refuses
that ID before private-state or worker mutation. Its one-shot
[`startup-forensics diagnostic`](../../research/2026-08-31-constraint-progress-startup-forensics-v1-results.md)
passed only the reproduced process boundary; it did not evaluate the optimizer.
The live contract is the fresh
[`constraint-aware-progress-toy-v2` plan](../../research/2026-08-31-constraint-aware-progress-toy-v2-plan.md).
V2 preserves the complete V1 scientific contract and changes only a framed,
exception-sealed bootstrap boundary. Its implementation, registration,
controller resume, and one possible guarded invocation are separate later
gates. No V2 study is currently approved to run.

The controller refuses a dirty or unapproved branch, an output outside the
private root, duplicate study identity, owner stop marker, parked state,
existing output, concurrent lease, protected artifact drift, or any change to
the full submission tree. It runs a credential-scrubbed, network-disabled CPU
worker with a hard one-hour deadline, rechecks repository integrity afterward,
and persists private state, events, result JSON, and a result-hash sidecar. A
foreign or stale lease requires human review and is never recovered
automatically. Retained JSON contains hashes and scalar mechanics only, never
parameter vectors, gradients, topology strings, credentials, or user paths.

`studies.json` is the approved continuation queue and result contract. A new
study must receive a unique versioned ID, frozen plan, source hashes, complete
case schema, and decision actions in a clean commit before it can run. An empty
queue leaves the controller in `awaiting_study`; it never repeats a terminal
study merely to stay busy.

Passing a local fixture shows a software invariant, not optimizer quality. A
failed frozen rule parks the affected research lane; it is not permission to
tune the fixture or tolerance.
