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

No approved study is currently pending. The next admissible learning rung is a
newly frozen bootstrapped multi-step value-propagation question with exact TD
targets and negative controls. The first proposed fixture,
`multistep-td-propagation-v1`, was rejected during pre-result audit and never
entered this registry: its public successor leaked prior target agreement, its
family commitment omitted successor/outcome rows, and several frozen sentinels
were incomplete. A successor must use a new ID, target-independent public
state, a complete legal-family commitment, stronger held-out and keyed-scoring
sentinels, and an outcome-blind cell-balanced reward-origin control. See the
sanitized
[`v1 preflight rejection`](../../research/2026-08-28-multistep-td-propagation-preflight-rejection.md).
Its fresh target-independent successor, `multistep-td-action-prefix-v2`, was
also rejected before controller execution. Although its development projection
passed, independent audits found that several target-swap, feedback-only
baseline, held-out, keyed-trace, timing, control-difference, and all-bootstrap
dependency checks could pass without establishing their frozen claims. V2 is
absent from `studies.json` and the controller allowlist and must not be repaired,
registered, executed, or reused. See the sanitized
[`v2 preflight rejection`](../../research/2026-08-28-multistep-td-action-prefix-v2-preflight-rejection.md).
A successor requires a fresh `multistep-td-action-prefix-v3` ID and frozen plan
before any learner executes. It is not meta-RL, production RL, an optimizer
treatment, or candidate evidence.

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
