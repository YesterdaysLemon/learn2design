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
`normal-path-jax-boundary-v1`. Their sanitized conclusions
live under `research/`; terminal study IDs must never be reused even though the
controller remains available for a newly frozen registry entry.

The JAX-boundary checkpoint passed its one-batch CPU systems contract. It is
not a candidate treatment or timing benchmark. The next registry entry must
freeze a learning contract before any model or policy is executed.

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
